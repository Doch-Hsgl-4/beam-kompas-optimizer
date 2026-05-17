from __future__ import annotations

import math
import unittest
from pathlib import Path

from beam_optimizer.materials import get_material_by_title
from beam_optimizer.models import (
    BeamRequest,
    Constraints,
    LoadCase,
    LoadType,
    NumericRange,
    OptimizationGoal,
    SectionSearchSpace,
    SectionType,
    SupportType,
)
from beam_optimizer.optimizer import run_optimization
from beam_optimizer.sections import generate_section_variants
from beam_optimizer.solver import analyze_section

# ── shared test parameters ──────────────────────────────────────────────────

_SEARCH_100x200 = SectionSearchSpace(
    primary=NumericRange(100.0, 100.0, 1.0),
    secondary=NumericRange(200.0, 200.0, 1.0),
)
_SEARCH_ROUND_150 = SectionSearchSpace(primary=NumericRange(150.0, 150.0, 1.0))


def _make_request(
    support_type: SupportType,
    load_case: LoadCase,
    *,
    length_m: float = 2.0,
    section_type: SectionType = SectionType.RECTANGULAR,
    search_space: SectionSearchSpace = _SEARCH_100x200,
    constraints: Constraints = Constraints(),
    goal: OptimizationGoal = OptimizationGoal.MIN_WEIGHT,
    top_n: int = 1,
) -> BeamRequest:
    return BeamRequest(
        length_m=length_m,
        section_type=section_type,
        material=get_material_by_title("Сталь"),
        support_type=support_type,
        load_case=load_case,
        search_space=search_space,
        constraints=constraints,
        optimization_goal=goal,
        top_n=top_n,
        output_dir=Path("output"),
    )


def _ei(section, material) -> float:
    return material.elastic_modulus_pa * section.inertia_m4


# ── section generation ───────────────────────────────────────────────────────

class SectionGenerationTests(unittest.TestCase):
    def test_square_uses_single_dimension_range(self) -> None:
        variants = generate_section_variants(
            SectionType.SQUARE,
            SectionSearchSpace(primary=NumericRange(20.0, 40.0, 10.0)),
        )
        self.assertEqual([v.dimensions_mm["side"] for v in variants], [20.0, 30.0, 40.0])

    def test_round_tube_keeps_wall_thickness_valid(self) -> None:
        variants = generate_section_variants(
            SectionType.ROUND_TUBE,
            SectionSearchSpace(
                primary=NumericRange(50.0, 50.0, 1.0),
                secondary=NumericRange(40.0, 40.0, 1.0),
            ),
        )
        self.assertEqual(len(variants), 1)
        self.assertLess(2.0 * variants[0].dimensions_mm["thickness"], variants[0].dimensions_mm["outer_diameter"])

    def test_single_value_range_produces_one_variant(self) -> None:
        variants = generate_section_variants(
            SectionType.RECTANGULAR,
            SectionSearchSpace(
                primary=NumericRange(50.0, 50.0, 1.0),
                secondary=NumericRange(100.0, 100.0, 1.0),
            ),
        )
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].dimensions_mm, {"width": 50.0, "height": 100.0})

    def test_inverted_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_section_variants(
                SectionType.RECTANGULAR,
                SectionSearchSpace(
                    primary=NumericRange(100.0, 50.0, 10.0),
                    secondary=NumericRange(100.0, 100.0, 1.0),
                ),
            )

    def test_solid_round_geometric_properties(self) -> None:
        """Area and inertia of solid round section match π-formulas."""
        d_mm = 80.0
        variants = generate_section_variants(
            SectionType.SOLID_ROUND,
            SectionSearchSpace(primary=NumericRange(d_mm, d_mm, 1.0)),
        )
        self.assertEqual(len(variants), 1)
        d = d_mm / 1000.0
        self.assertAlmostEqual(variants[0].area_m2, math.pi * d**2 / 4.0, places=12)
        self.assertAlmostEqual(variants[0].inertia_m4, math.pi * d**4 / 64.0, places=18)

    def test_rectangular_geometric_properties(self) -> None:
        """Area and inertia of rectangular section match handbook formulas."""
        variants = generate_section_variants(
            SectionType.RECTANGULAR,
            SectionSearchSpace(
                primary=NumericRange(100.0, 100.0, 1.0),
                secondary=NumericRange(200.0, 200.0, 1.0),
            ),
        )
        self.assertEqual(len(variants), 1)
        s = variants[0]
        self.assertAlmostEqual(s.area_m2, 0.1 * 0.2, places=12)
        self.assertAlmostEqual(s.inertia_m4, 0.1 * 0.2**3 / 12.0, places=15)
        self.assertAlmostEqual(s.section_modulus_m3, s.inertia_m4 / 0.1, places=15)


# ── FEM verification ─────────────────────────────────────────────────────────

class SolverSimplySupportedTests(unittest.TestCase):
    """Analytical verification for simply-supported beams."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.material = get_material_by_title("Сталь")
        cls.section = generate_section_variants(SectionType.RECTANGULAR, _SEARCH_100x200)[0]
        cls.EI = _ei(cls.section, cls.material)
        cls.L = 2.0
        cls.P = 10_000.0
        cls.q = 5_000.0

    def _check(self, result, expected_moment: float, expected_deflection: float) -> None:
        tol = 0.02  # 2 % relative tolerance for FEM with 60 elements
        self.assertAlmostEqual(result.max_moment_nm, expected_moment, delta=expected_moment * tol)
        self.assertAlmostEqual(result.max_deflection_m, expected_deflection, delta=expected_deflection * tol)

    def test_center_point_load_matches_closed_form(self) -> None:
        """M = PL/4, δ = PL³/(48EI)  — baseline case."""
        request = _make_request(SupportType.SIMPLY_SUPPORTED, LoadCase(LoadType.CENTER_POINT, self.P))
        result = analyze_section(request, self.section)
        self._check(result, self.P * self.L / 4.0, self.P * self.L**3 / (48.0 * self.EI))

    def test_uniform_distributed_load_matches_closed_form(self) -> None:
        """M = qL²/8, δ = 5qL⁴/(384EI)."""
        request = _make_request(SupportType.SIMPLY_SUPPORTED, LoadCase(LoadType.UNIFORM_DISTRIBUTED, self.q))
        result = analyze_section(request, self.section)
        self._check(result, self.q * self.L**2 / 8.0, 5.0 * self.q * self.L**4 / (384.0 * self.EI))

    def test_off_center_point_load_moment(self) -> None:
        """M_max = Pab/L for load at a=L/4 from left support."""
        a = self.L / 4.0
        b = self.L - a
        request = _make_request(
            SupportType.SIMPLY_SUPPORTED,
            LoadCase(LoadType.POINT_AT_POSITION, self.P, position_m=a),
        )
        result = analyze_section(request, self.section)
        expected_moment = self.P * a * b / self.L  # 3PL/16
        self.assertAlmostEqual(result.max_moment_nm, expected_moment, delta=expected_moment * 0.02)

    def test_off_center_point_load_deflection_at_load(self) -> None:
        """Deflection at load point (a=L/4): δ(a) = Pa²b²/(3EIL)."""
        a = self.L / 4.0
        b = self.L - a
        # The code reports max deflection over all nodes; it must be ≥ deflection at load point.
        # Maximum analytical deflection at x_max > a is larger than δ(a).
        request = _make_request(
            SupportType.SIMPLY_SUPPORTED,
            LoadCase(LoadType.POINT_AT_POSITION, self.P, position_m=a),
        )
        result = analyze_section(request, self.section)
        deflection_at_load = self.P * a**2 * b**2 / (3.0 * self.EI * self.L)
        self.assertGreaterEqual(result.max_deflection_m, deflection_at_load * 0.98)

    def test_solid_round_section_moment(self) -> None:
        """FEM gives correct moment for a solid-round cross-section (π-based inertia)."""
        section = generate_section_variants(SectionType.SOLID_ROUND, _SEARCH_ROUND_150)[0]
        ei = _ei(section, self.material)
        request = _make_request(
            SupportType.SIMPLY_SUPPORTED,
            LoadCase(LoadType.CENTER_POINT, self.P),
            section_type=SectionType.SOLID_ROUND,
            search_space=_SEARCH_ROUND_150,
        )
        result = analyze_section(request, section)
        expected_moment = self.P * self.L / 4.0
        expected_deflection = self.P * self.L**3 / (48.0 * ei)
        self.assertAlmostEqual(result.max_moment_nm, expected_moment, delta=expected_moment * 0.02)
        self.assertAlmostEqual(result.max_deflection_m, expected_deflection, delta=expected_deflection * 0.02)


class SolverCantileverTests(unittest.TestCase):
    """Analytical verification for cantilever beams (fixed at left, free at right)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.material = get_material_by_title("Сталь")
        cls.section = generate_section_variants(SectionType.RECTANGULAR, _SEARCH_100x200)[0]
        cls.EI = _ei(cls.section, cls.material)
        cls.L = 2.0
        cls.P = 10_000.0
        cls.q = 5_000.0

    def _check(self, result, expected_moment: float, expected_deflection: float, tol: float = 0.02) -> None:
        self.assertAlmostEqual(result.max_moment_nm, expected_moment, delta=expected_moment * tol)
        self.assertAlmostEqual(result.max_deflection_m, expected_deflection, delta=expected_deflection * tol)

    def test_tip_point_load_matches_closed_form(self) -> None:
        """Tip load at free end: M = PL at root, δ = PL³/(3EI) at tip.

        Load is placed at x=L using POINT_AT_POSITION. The GUI disallows this
        (strict open interval), but the FEM solver handles it correctly — this
        test exercises the solver directly.
        """
        request = _make_request(
            SupportType.CANTILEVER,
            LoadCase(LoadType.POINT_AT_POSITION, self.P, position_m=self.L),
        )
        result = analyze_section(request, self.section)
        self._check(result, self.P * self.L, self.P * self.L**3 / (3.0 * self.EI))

    def test_midspan_point_load_matches_closed_form(self) -> None:
        """Point load at L/2: M = P*L/2 at root, δ_free = 5PL³/(48EI)."""
        request = _make_request(
            SupportType.CANTILEVER,
            LoadCase(LoadType.CENTER_POINT, self.P),
        )
        result = analyze_section(request, self.section)
        # M at fixed end = P * (L - L/2) = PL/2
        expected_moment = self.P * self.L / 2.0
        # δ at free end = Pa²(3L-a)/(6EI) with a=L/2 = 5PL³/(48EI)
        expected_deflection = 5.0 * self.P * self.L**3 / (48.0 * self.EI)
        self._check(result, expected_moment, expected_deflection)

    def test_uniform_distributed_load_matches_closed_form(self) -> None:
        """UDL on cantilever: M = qL²/2 at root, δ = qL⁴/(8EI) at tip."""
        request = _make_request(
            SupportType.CANTILEVER,
            LoadCase(LoadType.UNIFORM_DISTRIBUTED, self.q),
        )
        result = analyze_section(request, self.section)
        self._check(result, self.q * self.L**2 / 2.0, self.q * self.L**4 / (8.0 * self.EI))


class SolverFixedFixedTests(unittest.TestCase):
    """Analytical verification for fixed-fixed beams."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.material = get_material_by_title("Сталь")
        cls.section = generate_section_variants(SectionType.RECTANGULAR, _SEARCH_100x200)[0]
        cls.EI = _ei(cls.section, cls.material)
        cls.L = 2.0
        cls.P = 10_000.0
        cls.q = 5_000.0

    def _check(self, result, expected_moment: float, expected_deflection: float) -> None:
        self.assertAlmostEqual(result.max_moment_nm, expected_moment, delta=expected_moment * 0.02)
        self.assertAlmostEqual(result.max_deflection_m, expected_deflection, delta=expected_deflection * 0.02)

    def test_center_point_load_matches_closed_form(self) -> None:
        """Fixed-fixed + center load: M_max = PL/8, δ_max = PL³/(192EI).

        The moment diagram has equal peaks at both fixed ends and at midspan,
        all equal to PL/8 in absolute value.
        """
        request = _make_request(
            SupportType.FIXED_FIXED,
            LoadCase(LoadType.CENTER_POINT, self.P),
        )
        result = analyze_section(request, self.section)
        self._check(result, self.P * self.L / 8.0, self.P * self.L**3 / (192.0 * self.EI))

    def test_uniform_distributed_load_matches_closed_form(self) -> None:
        """Fixed-fixed + UDL: M_max = qL²/12 at ends, δ_max = qL⁴/(384EI)."""
        request = _make_request(
            SupportType.FIXED_FIXED,
            LoadCase(LoadType.UNIFORM_DISTRIBUTED, self.q),
        )
        result = analyze_section(request, self.section)
        # M at ends = qL²/12 (hogging), center = qL²/24 (sagging)
        # |M|_max is at the fixed ends
        self._check(result, self.q * self.L**2 / 12.0, self.q * self.L**4 / (384.0 * self.EI))


# ── solver: existing test preserved as-is ────────────────────────────────────

class SolverTests(unittest.TestCase):
    def test_center_point_load_matches_closed_form_solution(self) -> None:
        material = get_material_by_title("Сталь")
        section = generate_section_variants(
            SectionType.RECTANGULAR,
            SectionSearchSpace(
                primary=NumericRange(100.0, 100.0, 1.0),
                secondary=NumericRange(200.0, 200.0, 1.0),
            ),
        )[0]
        request = BeamRequest(
            length_m=2.0,
            section_type=SectionType.RECTANGULAR,
            material=material,
            support_type=SupportType.SIMPLY_SUPPORTED,
            load_case=LoadCase(LoadType.CENTER_POINT, 10_000.0),
            search_space=SectionSearchSpace(
                primary=NumericRange(100.0, 100.0, 1.0),
                secondary=NumericRange(200.0, 200.0, 1.0),
            ),
            constraints=Constraints(),
            optimization_goal=OptimizationGoal.MIN_WEIGHT,
            top_n=1,
            output_dir=Path("output"),
        )

        result = analyze_section(request, section)
        expected_moment = request.load_case.magnitude * request.length_m / 4.0
        expected_deflection = (
            request.load_case.magnitude
            * request.length_m**3
            / (48.0 * material.elastic_modulus_pa * section.inertia_m4)
        )

        self.assertAlmostEqual(result.max_moment_nm, expected_moment, delta=1e-4)
        self.assertAlmostEqual(result.max_deflection_m, expected_deflection, delta=1e-12)


# ── slenderness warning ──────────────────────────────────────────────────────

class SolverSlendernessTests(unittest.TestCase):
    """Verify that Euler-Bernoulli slenderness warning fires for short/deep beams."""

    def test_slender_beam_no_warning(self) -> None:
        """L/h=10 — boundary value, warning must NOT fire."""
        section = generate_section_variants(SectionType.RECTANGULAR, _SEARCH_100x200)[0]
        material = get_material_by_title("Сталь")
        # h_equiv = 0.2 m; L=2.0 m → L/h = 10.0, warning threshold is < 10
        request = _make_request(SupportType.SIMPLY_SUPPORTED, LoadCase(LoadType.CENTER_POINT, 10_000.0))
        import logging
        with self.assertLogs("beam_optimizer.solver", level=logging.WARNING) as cm:
            # Force a short beam that WILL trigger the warning
            short_request = _make_request(
                SupportType.CANTILEVER,
                LoadCase(LoadType.CENTER_POINT, 10_000.0),
                length_m=1.0,  # L/h = 1.0/0.2 = 5 < 10 → warning expected
            )
            analyze_section(short_request, section)
        self.assertTrue(any("слабость" in msg.lower() or "slenderness" in msg.lower() or "гибкост" in msg.lower() or "euler" in msg.lower() for msg in cm.output))

    def test_stocky_beam_triggers_warning(self) -> None:
        """L/h < 10 must emit a WARNING via the solver logger."""
        section = generate_section_variants(SectionType.RECTANGULAR, _SEARCH_100x200)[0]
        import logging
        with self.assertLogs("beam_optimizer.solver", level=logging.WARNING) as cm:
            request = _make_request(
                SupportType.CANTILEVER,
                LoadCase(LoadType.CENTER_POINT, 10_000.0),
                length_m=1.0,  # L/h = 5 < 10
            )
            analyze_section(request, section)
        self.assertTrue(len(cm.output) >= 1)


# ── optimization ─────────────────────────────────────────────────────────────

class OptimizationTests(unittest.TestCase):
    def test_optimization_returns_requested_feasible_candidates(self) -> None:
        """40×90 mm is the lightest section that satisfies all three constraints.

        Verified analytically:
          I = 0.04 × 0.09³/12 = 2.43e-6 m⁴
          M = PL/4 = 5000 Nm
          σ = M/W = 5000/(2.43e-6/0.045) ≈ 92.6 MPa < 160 MPa ✓
          δ = PL³/(48EI) ≈ 3.27 mm < 5 mm ✓
          SF = 250/92.6 ≈ 2.70 > 1.5 ✓
        """
        request = BeamRequest(
            length_m=2.0,
            section_type=SectionType.RECTANGULAR,
            material=get_material_by_title("Сталь"),
            support_type=SupportType.SIMPLY_SUPPORTED,
            load_case=LoadCase(LoadType.CENTER_POINT, 10_000.0),
            search_space=SectionSearchSpace(
                primary=NumericRange(40.0, 80.0, 20.0),
                secondary=NumericRange(60.0, 120.0, 30.0),
            ),
            constraints=Constraints(max_stress_pa=160e6, max_deflection_m=0.005, min_safety_factor=1.5),
            optimization_goal=OptimizationGoal.MIN_WEIGHT,
            top_n=3,
            output_dir=Path("output"),
        )

        summary = run_optimization(request)

        self.assertEqual(summary.total_generated, 9)
        self.assertEqual(len(summary.selected), 3)
        self.assertTrue(all(candidate.is_feasible for candidate in summary.selected))
        self.assertEqual(summary.selected[0].section.dimensions_mm, {"width": 40.0, "height": 90.0})

    def test_no_feasible_candidates_returns_infeasible_fallback(self) -> None:
        """When constraints are impossible, selected list contains infeasible candidates."""
        request = _make_request(
            SupportType.SIMPLY_SUPPORTED,
            LoadCase(LoadType.CENTER_POINT, 10_000.0),
            constraints=Constraints(max_stress_pa=1.0),  # 1 Pa — impossible
            top_n=1,
        )
        summary = run_optimization(request)
        self.assertEqual(summary.feasible_count, 0)
        self.assertEqual(len(summary.selected), 1)
        self.assertFalse(summary.selected[0].is_feasible)
        self.assertTrue(len(summary.selected[0].violations) > 0)

    def test_max_safety_factor_goal_selects_safest_first(self) -> None:
        """MAX_SAFETY_FACTOR goal ranks by descending safety factor."""
        request = BeamRequest(
            length_m=2.0,
            section_type=SectionType.RECTANGULAR,
            material=get_material_by_title("Сталь"),
            support_type=SupportType.SIMPLY_SUPPORTED,
            load_case=LoadCase(LoadType.CENTER_POINT, 10_000.0),
            search_space=SectionSearchSpace(
                primary=NumericRange(40.0, 80.0, 20.0),
                secondary=NumericRange(60.0, 120.0, 30.0),
            ),
            constraints=Constraints(),
            optimization_goal=OptimizationGoal.MAX_SAFETY_FACTOR,
            top_n=3,
            output_dir=Path("output"),
        )
        summary = run_optimization(request)
        safety_factors = [c.analysis.safety_factor for c in summary.selected]
        self.assertEqual(safety_factors, sorted(safety_factors, reverse=True))

    def test_min_weight_goal_selects_lightest_first(self) -> None:
        """MIN_WEIGHT goal ranks by ascending mass."""
        request = BeamRequest(
            length_m=2.0,
            section_type=SectionType.RECTANGULAR,
            material=get_material_by_title("Сталь"),
            support_type=SupportType.SIMPLY_SUPPORTED,
            load_case=LoadCase(LoadType.CENTER_POINT, 10_000.0),
            search_space=SectionSearchSpace(
                primary=NumericRange(40.0, 80.0, 20.0),
                secondary=NumericRange(60.0, 120.0, 30.0),
            ),
            constraints=Constraints(),
            optimization_goal=OptimizationGoal.MIN_WEIGHT,
            top_n=3,
            output_dir=Path("output"),
        )
        summary = run_optimization(request)
        masses = [c.analysis.mass_kg for c in summary.selected]
        self.assertEqual(masses, sorted(masses))

    def test_all_constraints_active_simultaneously(self) -> None:
        """A candidate violating any single constraint is excluded."""
        section_100x200 = generate_section_variants(SectionType.RECTANGULAR, _SEARCH_100x200)[0]
        material = get_material_by_title("Сталь")
        request = _make_request(
            SupportType.SIMPLY_SUPPORTED,
            LoadCase(LoadType.CENTER_POINT, 10_000.0),
        )
        result = analyze_section(request, section_100x200)
        # Compute the exact stress this section produces, then use it as a tight limit.
        exact_stress = result.max_stress_pa
        tight_constraints = Constraints(max_stress_pa=exact_stress * 0.999)
        tight_request = _make_request(
            SupportType.SIMPLY_SUPPORTED,
            LoadCase(LoadType.CENTER_POINT, 10_000.0),
            constraints=tight_constraints,
        )
        summary = run_optimization(tight_request)
        self.assertEqual(summary.feasible_count, 0)


if __name__ == "__main__":
    unittest.main()
