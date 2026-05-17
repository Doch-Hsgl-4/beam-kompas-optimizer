"""Material property database.

Properties
----------
All values are characteristic (nominal) values suitable for preliminary
structural sizing.  For final design, material certificates or national
standards should be consulted.

Sources
-------
Steel (Ст3 / S235 equivalent):
    E = 210 GPa, ρ = 7850 kg/m³, σ_y = 250 MPa (tensile yield strength).
    ГОСТ 380-2005; EN 10025-2 (S235JR).

Aluminium alloy (АД31 / 6060-T5 equivalent):
    E = 69 GPa, ρ = 2700 kg/m³, σ_y = 150 MPa (0.2 % proof stress).
    ГОСТ 22233-2018; EN 755-2 (EN AW-6060 T5).

Cast iron (СЧ20 / EN-GJL-200 equivalent):
    E = 110 GPa, ρ = 7200 kg/m³, σ_y = 130 MPa.
    Note: cast iron is brittle — σ_y here represents compressive yield
    strength, which governs for typical beam loading.
    ГОСТ 1412-85; EN 1561 (EN-GJL-200).
"""
from __future__ import annotations

from beam_optimizer.models import Material


MATERIALS: dict[str, Material] = {
    "steel": Material(
        key="steel",
        title="Сталь",
        elastic_modulus_pa=210e9,    # E = 210 GPa  (ГОСТ 380-2005 / EN 10025-2 S235)
        density_kg_m3=7850.0,        # ρ = 7850 kg/m³
        yield_strength_pa=250e6,     # σ_y = 250 MPa
    ),
    "aluminum": Material(
        key="aluminum",
        title="Алюминий",
        elastic_modulus_pa=69e9,     # E = 69 GPa   (ГОСТ 22233-2018 / EN AW-6060)
        density_kg_m3=2700.0,        # ρ = 2700 kg/m³
        yield_strength_pa=150e6,     # σ_y = 150 MPa (0.2 % proof stress)
    ),
    "cast_iron": Material(
        key="cast_iron",
        title="Чугун",
        elastic_modulus_pa=110e9,    # E = 110 GPa  (ГОСТ 1412-85 / EN-GJL-200)
        density_kg_m3=7200.0,        # ρ = 7200 kg/m³
        yield_strength_pa=130e6,     # σ_y = 130 MPa (компрессионный предел текучести)
    ),
}


def material_titles() -> list[str]:
    return [material.title for material in MATERIALS.values()]


def get_material_by_title(title: str) -> Material:
    for material in MATERIALS.values():
        if material.title == title:
            return material
    raise KeyError(f"Материал не найден: '{title}'. Доступны: {material_titles()}")
