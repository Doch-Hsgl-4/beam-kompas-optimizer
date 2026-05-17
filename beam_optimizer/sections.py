"""Cross-section geometry generation and property calculation.

All second moments of area (I) are computed for bending about the horizontal
centroidal axis.  Section modulus W = I / c where c is the distance from the
neutral axis to the extreme fibre.

Formulas are taken from:
    Чигарев А.В., Кравчук А.С., Смалюк А.Ф. (2004). ANSYS для инженеров. М.: Машиностроение.
    Beer F.P., Johnston E.R. (2012). *Mechanics of Materials*, 6th ed. McGraw-Hill.
"""
from __future__ import annotations

import math

from beam_optimizer.models import NumericRange, SectionSearchSpace, SectionType, SectionVariant


def _iterate_range(value_range: NumericRange) -> list[float]:
    if value_range.step <= 0:
        raise ValueError("Step must be positive.")
    if value_range.max_value < value_range.min_value:
        raise ValueError("Range max must be greater than or equal to min.")

    count = int(round((value_range.max_value - value_range.min_value) / value_range.step)) + 1
    values: list[float] = []
    for index in range(count):
        value = value_range.min_value + index * value_range.step
        if value <= value_range.max_value + 1e-9:
            values.append(round(value, 6))
    return values


def generate_section_variants(
    section_type: SectionType,
    search_space: SectionSearchSpace,
) -> list[SectionVariant]:
    builders = {
        SectionType.RECTANGULAR: _generate_rectangular,
        SectionType.SQUARE: _generate_square,
        SectionType.SOLID_ROUND: _generate_solid_round,
        SectionType.ROUND_TUBE: _generate_round_tube,
    }
    return builders[section_type](search_space)


def _mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0


def _generate_rectangular(search_space: SectionSearchSpace) -> list[SectionVariant]:
    """I = b·h³/12,  W = b·h²/6,  A = b·h  (bending about horizontal axis)."""
    if search_space.secondary is None:
        raise ValueError("Rectangular profile requires two ranges.")

    variants: list[SectionVariant] = []
    primary_values = _iterate_range(search_space.primary)
    secondary_values = _iterate_range(search_space.secondary)
    for width_mm in primary_values:
        for height_mm in secondary_values:
            width_m = _mm_to_m(width_mm)
            height_m = _mm_to_m(height_mm)
            area = width_m * height_m
            inertia = width_m * height_m**3 / 12.0
            c = height_m / 2.0
            variants.append(
                SectionVariant(
                    section_type=SectionType.RECTANGULAR,
                    dimensions_mm={"width": width_mm, "height": height_mm},
                    area_m2=area,
                    inertia_m4=inertia,
                    section_modulus_m3=inertia / c,
                    title=f"Прямоугольник {width_mm:.1f}x{height_mm:.1f} мм",
                )
            )
    return variants


def _generate_square(search_space: SectionSearchSpace) -> list[SectionVariant]:
    """I = a⁴/12,  W = a³/6,  A = a²."""
    secondary_values = _iterate_range(search_space.secondary) if search_space.secondary is not None else _iterate_range(search_space.primary)
    side_values = sorted(
        {
            round(min(width_mm, height_mm), 6)
            for width_mm in _iterate_range(search_space.primary)
            for height_mm in secondary_values
        }
    )
    variants: list[SectionVariant] = []
    for side_mm in side_values:
        side_m = _mm_to_m(side_mm)
        area = side_m**2
        inertia = side_m**4 / 12.0
        c = side_m / 2.0
        variants.append(
            SectionVariant(
                section_type=SectionType.SQUARE,
                dimensions_mm={"side": side_mm},
                area_m2=area,
                inertia_m4=inertia,
                section_modulus_m3=inertia / c,
                title=f"Квадрат {side_mm:.1f} мм",
            )
        )
    return variants


def _generate_solid_round(search_space: SectionSearchSpace) -> list[SectionVariant]:
    """I = π·d⁴/64,  W = π·d³/32,  A = π·d²/4."""
    secondary_values = _iterate_range(search_space.secondary) if search_space.secondary is not None else _iterate_range(search_space.primary)
    diameter_values = sorted(
        {
            round(min(width_mm, height_mm), 6)
            for width_mm in _iterate_range(search_space.primary)
            for height_mm in secondary_values
        }
    )
    variants: list[SectionVariant] = []
    for diameter_mm in diameter_values:
        diameter_m = _mm_to_m(diameter_mm)
        area = math.pi * diameter_m**2 / 4.0
        inertia = math.pi * diameter_m**4 / 64.0
        c = diameter_m / 2.0
        variants.append(
            SectionVariant(
                section_type=SectionType.SOLID_ROUND,
                dimensions_mm={"diameter": diameter_mm},
                area_m2=area,
                inertia_m4=inertia,
                section_modulus_m3=inertia / c,
                title=f"Круг сплошной d={diameter_mm:.1f} мм",
            )
        )
    return variants


def _generate_round_tube(search_space: SectionSearchSpace) -> list[SectionVariant]:
    """I = π(D⁴ − d⁴)/64,  W = I / (D/2),  A = π(D² − d²)/4,  d = D − 2t.

    Wall thickness t is clamped to 0.45·D to prevent geometrically invalid tubes
    (t must satisfy 2t < D, i.e. inner diameter d > 0).
    """
    if search_space.secondary is None:
        raise ValueError("Round tube requires two ranges.")

    variants: list[SectionVariant] = []
    outer_values = _iterate_range(search_space.primary)
    thickness_values = _iterate_range(search_space.secondary)
    seen_pairs: set[tuple[float, float]] = set()

    for outer_mm in outer_values:
        for thickness_mm in thickness_values:
            # Generic width/height UI can provide values invalid for wall thickness.
            # We normalize to a valid thickness instead of dropping almost all options.
            normalized_thickness_mm = min(thickness_mm, round(outer_mm * 0.45, 6))
            if normalized_thickness_mm <= 0.0:
                continue
            if 2.0 * normalized_thickness_mm >= outer_mm:
                normalized_thickness_mm = round(outer_mm * 0.45, 6)

            pair_key = (round(outer_mm, 6), round(normalized_thickness_mm, 6))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            outer_m = _mm_to_m(outer_mm)
            inner_m = _mm_to_m(outer_mm - 2.0 * normalized_thickness_mm)
            area = math.pi * (outer_m**2 - inner_m**2) / 4.0
            inertia = math.pi * (outer_m**4 - inner_m**4) / 64.0
            c = outer_m / 2.0
            variants.append(
                SectionVariant(
                    section_type=SectionType.ROUND_TUBE,
                    dimensions_mm={"outer_diameter": outer_mm, "thickness": normalized_thickness_mm},
                    area_m2=area,
                    inertia_m4=inertia,
                    section_modulus_m3=inertia / c,
                    title=f"Труба D={outer_mm:.1f} мм, t={normalized_thickness_mm:.1f} мм",
                )
            )
    return variants
