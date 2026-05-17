from __future__ import annotations

import logging
import math

from beam_optimizer.models import AnalysisResult, BeamRequest, LoadType, SectionVariant, SupportType

logger = logging.getLogger(__name__)

DEFAULT_ELEMENTS = 60

_SLENDERNESS_THRESHOLD = 10.0


def analyze_section(request: BeamRequest, section: SectionVariant) -> AnalysisResult:
    # h_equiv = height of a rectangle with the same I/A ratio; equals actual height for rect sections.
    h_equiv = 2.0 * math.sqrt(3.0 * section.inertia_m4 / section.area_m2)
    slenderness = request.length_m / h_equiv
    if slenderness < _SLENDERNESS_THRESHOLD:
        logger.warning(
            "Коэффициент гибкости L/h = %.1f < 10 для сечения '%s'. "
            "Теория Эйлера-Бернулли даёт заниженные напряжения для коротких толстых балок.",
            slenderness,
            section.title,
        )
    nodes = _build_nodes(request.length_m, request.load_case)
    stiffness = _zeros_matrix(2 * len(nodes), 2 * len(nodes))
    load_vector = [0.0 for _ in range(2 * len(nodes))]

    for element_index in range(len(nodes) - 1):
        x1 = nodes[element_index]
        x2 = nodes[element_index + 1]
        length = x2 - x1
        k_local = _beam_element_stiffness(
            elastic_modulus=request.material.elastic_modulus_pa,
            inertia=section.inertia_m4,
            length=length,
        )
        dofs = [2 * element_index, 2 * element_index + 1, 2 * (element_index + 1), 2 * (element_index + 1) + 1]
        _assemble(stiffness, k_local, dofs)

        if request.load_case.load_type == LoadType.UNIFORM_DISTRIBUTED:
            q_downward = request.load_case.magnitude
            distributed_load = _distributed_load_vector(length, q_downward)
            _assemble_vector(load_vector, distributed_load, dofs)

    if request.load_case.load_type in {LoadType.CENTER_POINT, LoadType.POINT_AT_POSITION}:
        point_position = (
            request.length_m / 2.0
            if request.load_case.load_type == LoadType.CENTER_POINT
            else float(request.load_case.position_m)
        )
        node_index = _find_node_index(nodes, point_position)
        load_vector[2 * node_index] -= request.load_case.magnitude

    constrained_dofs = _constrained_dofs(request.support_type, len(nodes))
    displacements = _solve_system(stiffness, load_vector, constrained_dofs)
    max_moment = _estimate_max_moment(
        nodes=nodes,
        displacements=displacements,
        elastic_modulus=request.material.elastic_modulus_pa,
        inertia=section.inertia_m4,
    )
    max_stress = max_moment / section.section_modulus_m3
    max_deflection = max(abs(displacements[index]) for index in range(0, len(displacements), 2))
    mass = section.area_m2 * request.length_m * request.material.density_kg_m3
    safety_factor = float("inf") if max_stress <= 1e-12 else request.material.yield_strength_pa / max_stress

    return AnalysisResult(
        max_moment_nm=max_moment,
        max_stress_pa=max_stress,
        max_deflection_m=max_deflection,
        mass_kg=mass,
        safety_factor=safety_factor,
    )


def _build_nodes(length_m: float, load_case) -> list[float]:
    nodes = {0.0, length_m}
    for index in range(DEFAULT_ELEMENTS + 1):
        nodes.add(length_m * index / DEFAULT_ELEMENTS)
    if load_case.load_type == LoadType.CENTER_POINT:
        nodes.add(length_m / 2.0)
    elif load_case.load_type == LoadType.POINT_AT_POSITION and load_case.position_m is not None:
        nodes.add(load_case.position_m)
    return sorted(nodes)


def _beam_element_stiffness(elastic_modulus: float, inertia: float, length: float) -> list[list[float]]:
    factor = elastic_modulus * inertia / length**3
    l = length
    return [
        [12.0 * factor, 6.0 * l * factor, -12.0 * factor, 6.0 * l * factor],
        [6.0 * l * factor, 4.0 * l * l * factor, -6.0 * l * factor, 2.0 * l * l * factor],
        [-12.0 * factor, -6.0 * l * factor, 12.0 * factor, -6.0 * l * factor],
        [6.0 * l * factor, 2.0 * l * l * factor, -6.0 * l * factor, 4.0 * l * l * factor],
    ]


def _distributed_load_vector(length: float, downward_load: float) -> list[float]:
    upward_equivalent = -downward_load
    return [
        upward_equivalent * length / 2.0,
        upward_equivalent * length**2 / 12.0,
        upward_equivalent * length / 2.0,
        -upward_equivalent * length**2 / 12.0,
    ]


def _constrained_dofs(support_type: SupportType, node_count: int) -> list[int]:
    last_vertical = 2 * (node_count - 1)
    last_rotation = last_vertical + 1
    if support_type == SupportType.SIMPLY_SUPPORTED:
        return [0, last_vertical]
    if support_type == SupportType.CANTILEVER:
        return [0, 1]
    if support_type == SupportType.FIXED_FIXED:
        return [0, 1, last_vertical, last_rotation]
    raise ValueError(f"Unsupported support type: {support_type}")


def _assemble(global_matrix: list[list[float]], local_matrix: list[list[float]], dofs: list[int]) -> None:
    for local_i, global_i in enumerate(dofs):
        for local_j, global_j in enumerate(dofs):
            global_matrix[global_i][global_j] += local_matrix[local_i][local_j]


def _assemble_vector(global_vector: list[float], local_vector: list[float], dofs: list[int]) -> None:
    for local_i, global_i in enumerate(dofs):
        global_vector[global_i] += local_vector[local_i]


def _zeros_matrix(rows: int, columns: int) -> list[list[float]]:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def _find_node_index(nodes: list[float], point: float) -> int:
    for index, value in enumerate(nodes):
        if abs(value - point) <= 1e-9:
            return index
    raise ValueError(f"Point {point} is not part of the mesh.")


def _solve_system(
    stiffness: list[list[float]],
    load_vector: list[float],
    constrained_dofs: list[int],
) -> list[float]:
    dof_count = len(load_vector)
    constrained = set(constrained_dofs)
    free_dofs = [index for index in range(dof_count) if index not in constrained]
    reduced_matrix = [[stiffness[row][col] for col in free_dofs] for row in free_dofs]
    reduced_load = [load_vector[row] for row in free_dofs]
    reduced_displacements = _gaussian_elimination(reduced_matrix, reduced_load)

    displacements = [0.0 for _ in range(dof_count)]
    for local_index, dof in enumerate(free_dofs):
        displacements[dof] = reduced_displacements[local_index]
    return displacements


def _gaussian_elimination(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    a = [row[:] for row in matrix]
    b = vector[:]

    for pivot in range(size):
        max_row = max(range(pivot, size), key=lambda row: abs(a[row][pivot]))
        if abs(a[max_row][pivot]) <= 1e-14:
            raise ValueError("Singular stiffness matrix.")
        if max_row != pivot:
            a[pivot], a[max_row] = a[max_row], a[pivot]
            b[pivot], b[max_row] = b[max_row], b[pivot]

        pivot_value = a[pivot][pivot]
        for row in range(pivot + 1, size):
            factor = a[row][pivot] / pivot_value
            if factor == 0.0:
                continue
            for column in range(pivot, size):
                a[row][column] -= factor * a[pivot][column]
            b[row] -= factor * b[pivot]

    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        tail_sum = sum(a[row][column] * solution[column] for column in range(row + 1, size))
        solution[row] = (b[row] - tail_sum) / a[row][row]
    return solution


def _estimate_max_moment(
    nodes: list[float],
    displacements: list[float],
    elastic_modulus: float,
    inertia: float,
) -> float:
    max_moment = 0.0
    sample_points = 10

    for element_index in range(len(nodes) - 1):
        length = nodes[element_index + 1] - nodes[element_index]
        local_dofs = [
            displacements[2 * element_index],
            displacements[2 * element_index + 1],
            displacements[2 * (element_index + 1)],
            displacements[2 * (element_index + 1) + 1],
        ]

        for sample in range(sample_points + 1):
            xi = sample / sample_points
            curvature = _curvature(xi, length, local_dofs)
            moment = abs(elastic_modulus * inertia * curvature)
            max_moment = max(max_moment, moment)

    return max_moment


def _curvature(xi: float, length: float, local_dofs: list[float]) -> float:
    coefficients = [
        (-6.0 + 12.0 * xi) / length**2,
        (-4.0 + 6.0 * xi) / length,
        (6.0 - 12.0 * xi) / length**2,
        (-2.0 + 6.0 * xi) / length,
    ]
    return sum(coefficients[index] * local_dofs[index] for index in range(4))
