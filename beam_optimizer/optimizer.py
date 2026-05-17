from __future__ import annotations

import heapq

from beam_optimizer.models import (
    AnalysisResult,
    BeamRequest,
    CandidateResult,
    Constraints,
    OptimizationGoal,
    OptimizationSummary,
    SectionType,
    SectionVariant,
)
from beam_optimizer.sections import generate_section_variants
from beam_optimizer.solver import analyze_section


PRIMARY_DIMENSION_KEYS = {
    SectionType.RECTANGULAR: "width",
    SectionType.SQUARE: "side",
    SectionType.SOLID_ROUND: "diameter",
    SectionType.ROUND_TUBE: "outer_diameter",
}

SECONDARY_DIMENSION_KEYS = {
    SectionType.RECTANGULAR: "height",
    SectionType.ROUND_TUBE: "thickness",
}


def run_optimization(request: BeamRequest) -> OptimizationSummary:
    """Run the full beam optimization pipeline.

    Steps:
    1. Generate all cross-section variants within the search space.
    2. Analyse each variant with the 1-D FEM solver.
    3. Separate feasible (all constraints satisfied) from infeasible candidates.
    4. Rank feasible candidates by the requested optimisation goal (min mass or
       max safety factor), using secondary keys for tie-breaking.
    5. Return the top *top_n* feasible candidates as *selected*.
       If no feasible candidate exists, return the top *top_n* infeasible
       candidates as *alternatives* so the user can see which constraints fail.

    Complexity: O(N · M) where N = number of section variants and M = number of
    FEM elements (DEFAULT_ELEMENTS = 60).  Ranking uses heapq.nsmallest which
    runs in O(N log top_n).
    """
    sections = generate_section_variants(request.section_type, request.search_space)
    if not sections:
        raise ValueError("По заданному диапазону не удалось сгенерировать ни одного варианта сечения.")
    candidates: list[CandidateResult] = []

    for section in sections:
        analysis = analyze_section(request, section)
        is_feasible, violations = _check_constraints(section, analysis, request.constraints)
        candidates.append(
            CandidateResult(
                section=section,
                analysis=analysis,
                is_feasible=is_feasible,
                violations=violations,
            )
        )

    feasible_candidates = [candidate for candidate in candidates if candidate.is_feasible]
    infeasible_candidates = [candidate for candidate in candidates if not candidate.is_feasible]

    sort_key = lambda candidate: _sort_key(candidate, request.optimization_goal)  # noqa: E731
    take_count = max(1, request.top_n)

    selected = heapq.nsmallest(take_count, feasible_candidates, key=sort_key)
    # When no feasible solution exists, populate alternatives with the closest infeasible
    # candidates so the user can see which constraints are violated and by how much.
    alternatives = (
        heapq.nsmallest(take_count, infeasible_candidates, key=sort_key)
        if not selected
        else []
    )

    return OptimizationSummary(
        total_generated=len(candidates),
        feasible_count=len(feasible_candidates),
        selected=selected,
        alternatives=alternatives,
    )


def _check_constraints(
    section: SectionVariant,
    analysis: AnalysisResult,
    constraints: Constraints,
) -> tuple[bool, list[str]]:
    """Return (is_feasible, violation_messages) for a single candidate."""
    violations: list[str] = []

    if constraints.max_stress_pa is not None and analysis.max_stress_pa > constraints.max_stress_pa:
        violations.append("Превышено ограничение по напряжению.")
    if constraints.max_deflection_m is not None and analysis.max_deflection_m > constraints.max_deflection_m:
        violations.append("Превышено ограничение по прогибу.")
    if constraints.min_safety_factor is not None and analysis.safety_factor < constraints.min_safety_factor:
        violations.append("Недостаточный коэффициент запаса.")
    if constraints.max_primary_dimension_mm is not None:
        primary_key = PRIMARY_DIMENSION_KEYS[section.section_type]
        primary_value = section.dimensions_mm[primary_key]
        if primary_value > constraints.max_primary_dimension_mm:
            violations.append("Превышен предельный основной размер.")
    if constraints.max_secondary_dimension_mm is not None and section.section_type in SECONDARY_DIMENSION_KEYS:
        secondary_key = SECONDARY_DIMENSION_KEYS[section.section_type]
        secondary_value = section.dimensions_mm[secondary_key]
        if secondary_value > constraints.max_secondary_dimension_mm:
            violations.append("Превышен предельный вторичный размер.")

    return len(violations) == 0, violations


def _sort_key(candidate: CandidateResult, goal: OptimizationGoal) -> tuple[float, ...]:
    """Return a tuple used as a min-heap key for ranking candidates.

    For MIN_WEIGHT: primary key = mass; tie-broken by −safety_factor (prefer
    safer), then by deflection.
    For MAX_SAFETY_FACTOR: primary key = −safety_factor; tie-broken by mass,
    then by deflection.
    """
    if goal == OptimizationGoal.MAX_SAFETY_FACTOR:
        return (-candidate.analysis.safety_factor, candidate.analysis.mass_kg, candidate.analysis.max_deflection_m)
    return (
        candidate.analysis.mass_kg,
        -candidate.analysis.safety_factor,
        candidate.analysis.max_deflection_m,
    )
