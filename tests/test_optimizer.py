from __future__ import annotations

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


class SectionGenerationTests(unittest.TestCase):
    def test_square_uses_single_dimension_range(self) -> None:
        variants = generate_section_variants(
            SectionType.SQUARE,
            SectionSearchSpace(primary=NumericRange(20.0, 40.0, 10.0)),
        )

        self.assertEqual([variant.dimensions_mm["side"] for variant in variants], [20.0, 30.0, 40.0])

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


class OptimizationTests(unittest.TestCase):
    def test_optimization_returns_requested_feasible_candidates(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
