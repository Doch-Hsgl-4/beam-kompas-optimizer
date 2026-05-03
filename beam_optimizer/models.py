from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SectionType(str, Enum):
    RECTANGULAR = "rectangular"
    SQUARE = "square"
    SOLID_ROUND = "solid_round"
    ROUND_TUBE = "round_tube"


class SupportType(str, Enum):
    SIMPLY_SUPPORTED = "simply_supported"
    CANTILEVER = "cantilever"
    FIXED_FIXED = "fixed_fixed"


class LoadType(str, Enum):
    CENTER_POINT = "center_point"
    POINT_AT_POSITION = "point_at_position"
    UNIFORM_DISTRIBUTED = "uniform_distributed"


class OptimizationGoal(str, Enum):
    MIN_WEIGHT = "min_weight"
    MAX_SAFETY_FACTOR = "max_safety_factor"


@dataclass(frozen=True)
class Material:
    key: str
    title: str
    elastic_modulus_pa: float
    density_kg_m3: float
    yield_strength_pa: float


@dataclass(frozen=True)
class NumericRange:
    min_value: float
    max_value: float
    step: float


@dataclass(frozen=True)
class SectionSearchSpace:
    primary: NumericRange
    secondary: NumericRange | None = None


@dataclass(frozen=True)
class LoadCase:
    load_type: LoadType
    magnitude: float
    position_m: float | None = None


@dataclass(frozen=True)
class Constraints:
    max_stress_pa: float | None = None
    max_deflection_m: float | None = None
    min_safety_factor: float | None = None
    max_primary_dimension_mm: float | None = None
    max_secondary_dimension_mm: float | None = None


@dataclass(frozen=True)
class BeamRequest:
    length_m: float
    section_type: SectionType
    material: Material
    support_type: SupportType
    load_case: LoadCase
    search_space: SectionSearchSpace
    constraints: Constraints
    optimization_goal: OptimizationGoal
    top_n: int
    output_dir: Path


@dataclass(frozen=True)
class SectionVariant:
    section_type: SectionType
    dimensions_mm: dict[str, float]
    area_m2: float
    inertia_m4: float
    section_modulus_m3: float
    title: str


@dataclass(frozen=True)
class AnalysisResult:
    max_moment_nm: float
    max_stress_pa: float
    max_deflection_m: float
    mass_kg: float
    safety_factor: float


@dataclass
class CandidateResult:
    section: SectionVariant
    analysis: AnalysisResult
    is_feasible: bool
    violations: list[str] = field(default_factory=list)
    model_path: Path | None = None


@dataclass(frozen=True)
class OptimizationSummary:
    total_generated: int
    feasible_count: int
    selected: list[CandidateResult]
    alternatives: list[CandidateResult]
