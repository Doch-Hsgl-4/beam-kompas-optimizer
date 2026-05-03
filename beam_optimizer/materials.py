from __future__ import annotations

from beam_optimizer.models import Material


MATERIALS: dict[str, Material] = {
    "steel": Material(
        key="steel",
        title="Сталь",
        elastic_modulus_pa=210e9,
        density_kg_m3=7850.0,
        yield_strength_pa=250e6,
    ),
    "aluminum": Material(
        key="aluminum",
        title="Алюминий",
        elastic_modulus_pa=69e9,
        density_kg_m3=2700.0,
        yield_strength_pa=150e6,
    ),
    "cast_iron": Material(
        key="cast_iron",
        title="Чугун",
        elastic_modulus_pa=110e9,
        density_kg_m3=7200.0,
        yield_strength_pa=130e6,
    ),
}


def material_titles() -> list[str]:
    return [material.title for material in MATERIALS.values()]


def get_material_by_title(title: str) -> Material:
    for material in MATERIALS.values():
        if material.title == title:
            return material
    raise KeyError(f"Unknown material title: {title}")
