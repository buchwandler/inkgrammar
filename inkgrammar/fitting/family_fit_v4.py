from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import sqrt
from typing import Any, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from skeletonpaths import (
    SkeletonGraph,
    SkeletonPath,
    TopologyPathInventory,
    build_skeleton_graph,
    prepare_topology_paths,
    rank_topology_paths,
    select_topology_path,
)

from .annealing import AnnealingConfig, simulated_annealing
from .models import (
    FamilyTopology,
    OccurrenceFitState,
    PrimitiveTemplate,
    SplineWidthModel,
    TopologyTransform,
    compose_occurrence_transform,
)
from .objective import SplineCorrespondenceConfig, SplineCurveConfig
from .seed_migration import SeedMigrationResult, migrate_legacy_seed
from .spline import evaluate_primitive_curve, sample_curve_by_arclength
from .spline_fit import SplineObservation, solve_shared_spline_coefficients


@dataclass(frozen=True)
class ObservedCurve:
    u: np.ndarray
    xy: np.ndarray
    tangent: np.ndarray
    radius: np.ndarray
    confidence: np.ndarray
    source_indices: np.ndarray
    ambiguous: np.ndarray
    supervise_endpoints: bool = True

@dataclass(frozen=True)
class PreparedTopologyTarget:
    mask: np.ndarray
    graph: SkeletonGraph
    path_inventory: TopologyPathInventory
    radius_map: np.ndarray
    confidence_map: np.ndarray
    min_spur_length_px: float
@dataclass(frozen=True)
class TopologyPathResult:
    path: SkeletonPath | None
    exact_match: bool
    topology_penalty: float
    diagnostics: dict[str, Any]

class NoTopologyPathError(ValueError):
    """Raised when no observed skeleton path can support a local fit."""


@dataclass(frozen=True)
class FamilySearchEnergy:
    total: float
    geometry_mean: float
    geometry_worst: float
    geometry_spread: float
    endpoint: float
    tangent: float
    topology: float
    saturation: float
    assignment: float
    roughness: float
    seed_distance: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "geometry_mean": self.geometry_mean,
            "geometry_worst": self.geometry_worst,
            "geometry_spread": self.geometry_spread,
            "endpoint": self.endpoint,
            "tangent": self.tangent,
            "topology": self.topology,
            "saturation": self.saturation,
            "assignment": self.assignment,
            "roughness": self.roughness,
            "seed_distance": self.seed_distance,
        }


def evaluate_family_search_energy(
    geometry_values: Sequence[float],
    *,
    endpoint_values: Sequence[float] = (),
    tangent_values: Sequence[float] = (),
    topology_values: Sequence[float] = (),
    saturation_values: Sequence[float] = (),
    assignment_instability: float = 0.0,
    roughness_penalty: float = 0.0,
    seed_distance_penalty: float = 0.0,
    family_worst_weight: float = 0.25,
    family_spread_weight: float = 0.1,
    topology_weight: float = 0.4,
    saturation_weight: float = 0.2,
    invalid_candidate: bool = False,
) -> FamilySearchEnergy:
    def finite_values(name: str, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError(f"search energy {name} values must be finite and nonnegative")
        return array

    geometry = finite_values("geometry", geometry_values)
    if geometry.size == 0:
        raise ValueError("search energy requires at least one geometry value")
    endpoint = finite_values("endpoint", endpoint_values)
    tangent = finite_values("tangent", tangent_values)
    topology = finite_values("topology", topology_values)
    saturation = finite_values("saturation", saturation_values)
    scalar_values = np.asarray(
        [
            assignment_instability,
            roughness_penalty,
            seed_distance_penalty,
            family_worst_weight,
            family_spread_weight,
            topology_weight,
            saturation_weight,
        ],
        dtype=float,
    )
    if not np.isfinite(scalar_values).all() or np.any(scalar_values < 0.0):
        raise ValueError("search energy scalar terms and weights must be finite and nonnegative")

    geometry_mean = float(np.mean(geometry))
    geometry_worst = float(np.max(geometry))
    geometry_spread = float(np.std(geometry))
    endpoint_mean = float(np.mean(endpoint)) if endpoint.size else 0.0
    tangent_mean = float(np.mean(tangent)) if tangent.size else 0.0
    topology_mean = float(np.mean(topology)) if topology.size else 0.0
    saturation_mean = float(np.mean(saturation)) if saturation.size else 0.0
    total = (
        geometry_mean
        + (0.35 + family_worst_weight) * geometry_worst
        + family_spread_weight * geometry_spread
        + 0.50 * endpoint_mean
        + 0.25 * tangent_mean
        + topology_weight * topology_mean
        + saturation_weight * saturation_mean
        + 0.10 * assignment_instability
        + 0.05 * roughness_penalty
        + 0.05 * seed_distance_penalty
    )
    if invalid_candidate:
        total = 1e6
    return FamilySearchEnergy(
        total=float(total),
        geometry_mean=geometry_mean,
        geometry_worst=geometry_worst,
        geometry_spread=geometry_spread,
        endpoint=endpoint_mean,
        tangent=tangent_mean,
        topology=topology_mean,
        saturation=saturation_mean,
        assignment=float(assignment_instability),
        roughness=float(roughness_penalty),
        seed_distance=float(seed_distance_penalty),
    )






@dataclass(frozen=True)
class MonotoneCurveCorrespondence:
    candidate_indices: np.ndarray
    target_indices: np.ndarray
    cost: float
    mean_entropy: float = 0.0


@dataclass(frozen=True)
class SplineOccurrence:
    key: str
    target: ObservedCurve
    state: OccurrenceFitState
    origin: tuple[float, float]


@dataclass(frozen=True)
class SplineFamilyFit:
    geometry: PrimitiveTemplate
    states: dict[str, OccurrenceFitState]
    correspondences: dict[str, MonotoneCurveCorrespondence]
    cycles: int
    normalized_change: float
    width: SplineWidthModel | None
    assignment_cycles: int
    assignment_change: float
    targets: dict[str, ObservedCurve]


def _sample_polyline(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("observed curve sample count must be at least two")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("observed path must contain finite xy points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    points = points[keep]
    if len(points) < 2 or cumulative[-1] <= 1e-12:
        raise ValueError("observed path must have positive length")
    parameters = np.linspace(0.0, 1.0, count)
    distances = parameters * cumulative[-1]
    sampled = np.column_stack(
        [np.interp(distances, cumulative, points[:, axis]) for axis in range(2)]
    )
    return parameters, sampled, distances


def prepare_topology_target(
    target_mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    radius: np.ndarray | None = None,
    min_spur_length_px: float = 3.0,
) -> PreparedTopologyTarget:
    """Build the candidate-independent graph, path inventory, and target maps once."""
    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("target mask must be a nonempty two-dimensional image")
    confidence_map = (
        np.ones(mask.shape, dtype=float)
        if confidence is None
        else np.asarray(confidence, dtype=float)
    )
    if (
        confidence_map.shape != mask.shape
        or not np.isfinite(confidence_map).all()
        or np.any((confidence_map < 0.0) | (confidence_map > 1.0))
    ):
        raise ValueError(
            "target confidence must be finite, between zero and one, and match the mask"
        )
    radius_map = distance_transform_edt(mask) if radius is None else np.asarray(radius, dtype=float)
    if (
        radius_map.shape != mask.shape
        or not np.isfinite(radius_map).all()
        or np.any(radius_map < 0.0)
    ):
        raise ValueError("target radius must be finite, nonnegative, and match the mask")
    graph = build_skeleton_graph(
        mask,
        min_spur_length_px=min_spur_length_px,
        radius=radius_map,
        confidence=confidence_map,
    )
    return PreparedTopologyTarget(
        mask=mask,
        graph=graph,
        path_inventory=prepare_topology_paths(graph),
        radius_map=radius_map,
        confidence_map=confidence_map,
        min_spur_length_px=min_spur_length_px,
    )


def extract_observed_curve(
    target_mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    radius: np.ndarray | None = None,
    min_spur_length_px: float = 3.0,
    path_kind: Literal["open", "loop", "open-with-loop"] = "open",
    expected_cycles: int | None = None,
    expected_endpoints: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None,
    candidate_xy: np.ndarray | None = None,
    prepared_target: PreparedTopologyTarget | None = None,
    preselected_path: SkeletonPath | None = None,
    endpoint_tolerance_px: float = 4.0,
    sample_count: int = 128,
) -> tuple[ObservedCurve, SkeletonGraph, SkeletonPath]:
    """Build an ordered, confidence-weighted curve from a topology-selected skeleton path."""
    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("target mask must be a nonempty two-dimensional image")
    if prepared_target is None:
        prepared_target = prepare_topology_target(
            mask,
            confidence=confidence,
            radius=radius,
            min_spur_length_px=min_spur_length_px,
        )
    elif (
        confidence is not None
        or radius is not None
        or not np.array_equal(mask, prepared_target.mask)
        or min_spur_length_px != prepared_target.min_spur_length_px
    ):
        raise ValueError("prepared topology target does not match the requested observation")
    confidence_map = prepared_target.confidence_map
    radius_map = prepared_target.radius_map
    graph = prepared_target.graph
    path = preselected_path or select_topology_path(
        graph,
        path_kind=path_kind,
        expected_cycles=expected_cycles,
        expected_endpoints=expected_endpoints,
        endpoint_tolerance_px=endpoint_tolerance_px,
        candidate_xy=candidate_xy,
        path_inventory=prepared_target.path_inventory,
    )
    u, xy, _ = _sample_polyline(path.xy, sample_count)
    tangents = np.gradient(xy, axis=0)
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    tangents = np.divide(
        tangents,
        tangent_lengths[:, None],
        out=np.zeros_like(tangents),
        where=tangent_lengths[:, None] > 1e-12,
    )
    columns = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    rows = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    sample_confidence = confidence_map[rows, columns] * path.confidence
    sample_radius = radius_map[rows, columns]
    ambiguous = sample_confidence <= 0.0
    endpoints_on_border = any(
        x <= 0.5 or y <= 0.5 or x >= mask.shape[1] - 1.5 or y >= mask.shape[0] - 1.5
        for x, y in (xy[0], xy[-1])
    )
    supervise_endpoints = bool(np.all(sample_confidence[[0, -1]] > 0.0) and not endpoints_on_border)
    return (
        ObservedCurve(
            u=u,
            xy=xy,
            tangent=tangents,
            radius=sample_radius,
            confidence=sample_confidence,
            source_indices=rows * mask.shape[1] + columns,
            ambiguous=ambiguous,
            supervise_endpoints=supervise_endpoints,
        ),
        graph,
        path,
    )


def _unit_tangents(points: np.ndarray) -> np.ndarray:
    tangent = np.gradient(points, axis=0)
    length = np.linalg.norm(tangent, axis=1)
    return np.divide(
        tangent,
        length[:, None],
        out=np.zeros_like(tangent),
        where=length[:, None] > 1e-12,
    )


def monotone_curve_correspondence(
    candidate_xy: np.ndarray,
    target: ObservedCurve,
    *,
    candidate_arc_u: np.ndarray | None = None,
    sigma_position_px: float = 2.0,
    sigma_tangent_rad: float = 0.35,
    tangent_weight: float = 0.2,
    warp_weight: float = 0.10,
    gap_penalty: float = 0.20,
    open_target_ends: bool = True,
) -> MonotoneCurveCorrespondence:
    candidate = np.asarray(candidate_xy, dtype=float)
    target_xy = np.asarray(target.xy, dtype=float)
    target_tangent = np.asarray(target.tangent, dtype=float)
    target_confidence = np.asarray(target.confidence, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) < 2:
        raise ValueError("candidate curve must have shape (N, 2) with N >= 2")
    if target_xy.ndim != 2 or target_xy.shape[1] != 2 or len(target_xy) < 2:
        raise ValueError("target curve must have shape (M, 2) with M >= 2")
    if target_tangent.shape != target_xy.shape or target_confidence.shape != (len(target_xy),):
        raise ValueError("target curve arrays have inconsistent shapes")
    if not np.isfinite(candidate).all() or not np.isfinite(target_xy).all():
        raise ValueError("curve coordinates must be finite")
    if not np.isfinite(target_confidence).all() or np.any(target_confidence < 0.0):
        raise ValueError("target confidence must be finite and nonnegative")
    target_u = np.asarray(target.u, dtype=float)
    if (
        target_u.shape != (len(target_xy),)
        or not np.isfinite(target_u).all()
        or np.any(np.diff(target_u) < 0.0)
    ):
        raise ValueError(
            "target arclength coordinates must be finite, ordered, and match the curve"
        )
    if (
        not np.isfinite(
            (sigma_position_px, sigma_tangent_rad, tangent_weight, warp_weight, gap_penalty)
        ).all()
        or min(sigma_position_px, sigma_tangent_rad) <= 0.0
    ):
        raise ValueError("correspondence sigmas must be finite and positive")
    if min(tangent_weight, warp_weight, gap_penalty) < 0.0:
        raise ValueError("correspondence weights must be nonnegative")

    candidate_arc = (
        np.linspace(0.0, 1.0, len(candidate))
        if candidate_arc_u is None
        else np.asarray(candidate_arc_u, dtype=float)
    )
    if (
        candidate_arc.shape != (len(candidate),)
        or not np.isfinite(candidate_arc).all()
        or np.any(np.diff(candidate_arc) < 0.0)
    ):
        raise ValueError(
            "candidate arclength coordinates must be finite, ordered, and match the curve"
        )
    candidate_tangent = _unit_tangents(candidate)
    target_norm = np.linalg.norm(target_tangent, axis=1)
    target_tangent = np.divide(
        target_tangent,
        target_norm[:, None],
        out=np.zeros_like(target_tangent),
        where=target_norm[:, None] > 1e-12,
    )
    displacement = candidate[:, None, :] - target_xy[None, :, :]
    position = np.sum(displacement**2, axis=2) / sigma_position_px**2
    dot = np.clip(candidate_tangent @ target_tangent.T, -1.0, 1.0)
    angle = np.arccos(dot)
    tangent = (angle / sigma_tangent_rad) ** 2
    warp = (candidate_arc[:, None] - target_u[None, :]) ** 2
    local = target_confidence[None, :] * (position + tangent_weight * tangent)
    local += warp_weight * warp

    candidate_count, target_count = local.shape
    dynamic = np.full(local.shape, np.inf, dtype=float)
    shifted = local - np.min(local, axis=1, keepdims=True)
    probabilities = np.exp(-0.5 * shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    mean_entropy = float(np.mean(entropy) / np.log(target_count)) if target_count > 1 else 0.0
    predecessor = np.full(local.shape, -1, dtype=np.int8)
    if open_target_ends:
        dynamic[0] = local[0]
    else:
        dynamic[0, 0] = local[0, 0]
    for i in range(1, candidate_count):
        for j in range(target_count):
            options = []
            moves = []
            if j > 0:
                options.append(dynamic[i - 1, j - 1])
                moves.append(0)
            options.append(dynamic[i - 1, j] + gap_penalty)
            moves.append(1)
            if j > 0:
                options.append(dynamic[i, j - 1] + gap_penalty)
                moves.append(2)
            selected = int(np.argmin(options))
            dynamic[i, j] = options[selected] + local[i, j]
            predecessor[i, j] = moves[selected]

    final_j = int(np.argmin(dynamic[-1])) if open_target_ends else target_count - 1
    i, j = candidate_count - 1, final_j
    path = []
    while True:
        path.append((i, j))
        move = predecessor[i, j]
        if move < 0:
            break
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path = np.asarray(path[::-1], dtype=int)
    return MonotoneCurveCorrespondence(
        candidate_indices=path[:, 0],
        target_indices=path[:, 1],
        cost=float(dynamic[-1, final_j]),
        mean_entropy=mean_entropy,
    )


def _model_to_image(points: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    result = np.asarray(points, dtype=float).copy()
    result[:, 0] += origin[0]
    result[:, 1] = origin[1] - result[:, 1]
    return result


def _image_to_canonical(
    points: np.ndarray, origin: tuple[float, float], transform: TopologyTransform
) -> np.ndarray:
    values = np.asarray(points, dtype=float).copy()
    values[:, 0] -= origin[0]
    values[:, 1] = origin[1] - values[:, 1]
    values[:, 1] = (values[:, 1] - transform.dy) / transform.scale_y
    values[:, 0] = (values[:, 0] - transform.dx) / transform.scale_x - transform.shear_x * values[
        :, 1
    ]
    return values


def fit_occurrence_transform(
    template: PrimitiveTemplate,
    target: ObservedCurve,
    state: OccurrenceFitState,
    origin: tuple[float, float],
    candidate_parameters: np.ndarray,
    correspondence: MonotoneCurveCorrespondence,
    *,
    active_parameters: tuple[str, ...] | None = None,
    position_sigma_px: float = 1.5,
    robust_loss: Literal["huber", "soft_l1"] = "huber",
    robust_scale_px: float = 1.5,
    max_nfev: int = 120,
) -> OccurrenceFitState:
    parameter_names = state.free_parameters if active_parameters is None else active_parameters
    names = tuple(name for name in parameter_names if name in state.free_parameters)
    if not names:
        return state
    parameters = np.asarray(candidate_parameters, dtype=float)
    candidate_u = parameters[correspondence.candidate_indices]
    target_xy = target.xy[correspondence.target_indices]
    weights = np.sqrt(np.maximum(target.confidence[correspondence.target_indices], 0.0))
    bounds = [getattr(state.bounds, name) for name in names]
    initial = np.asarray([getattr(state.delta, name) for name in names], dtype=float)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    translation_sigma = 12.0
    scale_sigma = 0.05
    shear_sigma = 0.03

    def residual(vector: np.ndarray) -> np.ndarray:
        delta = state.delta.model_copy(update=dict(zip(names, vector, strict=True)))
        transform = compose_occurrence_transform(state.nominal_transform, delta)
        candidate = evaluate_primitive_curve(template, transform, candidate_u).xy
        position_residual = (
            (_model_to_image(candidate, origin) - target_xy) * weights[:, None] / position_sigma_px
        ).ravel()
        priors = []
        for name, value in zip(names, vector, strict=True):
            sigma = (
                translation_sigma
                if name in {"dx", "dy"}
                else scale_sigma
                if name in {"log_scale_x", "log_scale_y"}
                else shear_sigma
            )
            priors.append(value / sigma)
        return np.concatenate((position_residual, np.asarray(priors)))

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss=robust_loss,
        f_scale=robust_scale_px / position_sigma_px,
        max_nfev=max_nfev,
    )
    delta = state.delta.model_copy(update=dict(zip(names, result.x, strict=True)))
    return state.model_copy(update={"delta": delta})


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    midpoint = cumulative[-1] / 2.0
    index = int(np.searchsorted(cumulative, midpoint))
    if cumulative[index] == midpoint and index + 1 < len(order):
        return float((values[order[index]] + values[order[index + 1]]) / 2.0)
    return float(values[order[index]])


def estimate_constant_width(
    radii_px: np.ndarray,
    confidence: np.ndarray,
    transform: TopologyTransform,
) -> float | None:
    radii = np.asarray(radii_px, dtype=float)
    weights = np.asarray(confidence, dtype=float)
    if (
        radii.shape != weights.shape
        or not np.isfinite(radii).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("width observations must be finite matching arrays")
    valid = (radii > 0.0) & (weights > 0.0)
    if not np.any(valid):
        return None
    scale = sqrt(transform.scale_x * transform.scale_y)
    widths = 2.0 * radii[valid] / scale
    return _weighted_median(widths, weights[valid])


def fit_spline_family(
    template: PrimitiveTemplate,
    occurrences: list[SplineOccurrence],
    *,
    curve_config: SplineCurveConfig,
    correspondence_config: SplineCorrespondenceConfig,
    max_cycles: int = 12,
    transform_max_nfev: int = 120,
    irls_iterations: int = 3,
    tolerance: float = 5e-4,
    target_updater: Callable[
        [PrimitiveTemplate, dict[str, OccurrenceFitState]], dict[str, ObservedCurve]
    ]
    | None = None,
    assignment_update_interval: int = 1,
    seed_control_points: np.ndarray | None = None,
) -> SplineFamilyFit:
    if template.kind != "bspline-centerline" or template.spline is None:
        raise ValueError("v4 family fitting requires a bspline-centerline primitive")
    if not occurrences:
        raise ValueError("v4 family fitting requires at least one occurrence")
    if len({item.key for item in occurrences}) != len(occurrences):
        raise ValueError("v4 occurrence keys must be unique")
    if max_cycles < 1 or irls_iterations < 1 or assignment_update_interval < 1:
        raise ValueError("v4 cycle, IRLS, and assignment update counts must be positive")

    starting_control_points = np.asarray(template.spline.control_points, dtype=float)
    seed_controls = np.asarray(
        starting_control_points if seed_control_points is None else seed_control_points,
        dtype=float,
    )
    if seed_controls.shape != starting_control_points.shape or not np.isfinite(seed_controls).all():
        raise ValueError("seed control points must be finite and match the spline basis")
    geometry = template.spline
    states = {item.key: item.state for item in occurrences}
    correspondences: dict[str, MonotoneCurveCorrespondence] = {}
    normalized_change = float("inf")
    completed_cycles = 0
    targeted_occurrences = list(occurrences)
    assignment_cycles = 0
    assignment_change = 0.0
    for cycle in range(max_cycles):
        if target_updater is not None and cycle % assignment_update_interval == 0:
            previous_targets = {item.key: item.target for item in targeted_occurrences}
            updated_targets = target_updater(
                template.model_copy(update={"spline": geometry}), states
            )
            if set(updated_targets) != set(states):
                raise ValueError("target updater must return one target for every occurrence")
            changes = []
            for key, updated_target in updated_targets.items():
                old_pixels = set(previous_targets[key].source_indices.tolist())
                new_pixels = set(updated_target.source_indices.tolist())
                union = old_pixels | new_pixels
                changes.append(1.0 - len(old_pixels & new_pixels) / len(union) if union else 0.0)
            assignment_change = float(np.mean(changes)) if assignment_cycles else 1.0
            assignment_cycles += 1
            targeted_occurrences = [
                replace(item, target=updated_targets[item.key]) for item in occurrences
            ]
        cycle_correspondences = {}
        for occurrence in targeted_occurrences:
            state = states[occurrence.key]
            transform = compose_occurrence_transform(state.nominal_transform, state.delta)
            candidate = sample_curve_by_arclength(
                template.model_copy(update={"spline": geometry}),
                transform,
                max(32, min(256, len(occurrence.target.xy))),
            )
            candidate_xy = _model_to_image(candidate.xy, occurrence.origin)
            curve_match = monotone_curve_correspondence(
                candidate_xy,
                occurrence.target,
                candidate_arc_u=np.linspace(0.0, 1.0, len(candidate.xy)),
                sigma_position_px=correspondence_config.position_sigma_px,
                sigma_tangent_rad=correspondence_config.tangent_sigma_rad,
                tangent_weight=correspondence_config.tangent_weight,
                warp_weight=correspondence_config.warp_weight,
                gap_penalty=correspondence_config.gap_penalty,
                open_target_ends=not occurrence.target.supervise_endpoints,
            )
            cycle_correspondences[occurrence.key] = (curve_match, candidate.u)

        transformed_states = {}
        for occurrence in targeted_occurrences:
            match, candidate_parameters = cycle_correspondences[occurrence.key]
            if cycle == 0:
                active = ("dx", "dy")
            elif cycle == 1:
                active = ("dx", "dy", "log_scale_x", "log_scale_y")
            else:
                active = ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x")
            transformed_states[occurrence.key] = fit_occurrence_transform(
                template.model_copy(update={"spline": geometry}),
                occurrence.target,
                states[occurrence.key],
                occurrence.origin,
                candidate_parameters,
                match,
                active_parameters=active,
                position_sigma_px=correspondence_config.position_sigma_px,
                robust_loss=curve_config.robust_loss,
                robust_scale_px=curve_config.robust_scale_px,
                max_nfev=transform_max_nfev,
            )

        observations = []
        correspondences = {}
        for occurrence in targeted_occurrences:
            state = transformed_states[occurrence.key]
            transform = compose_occurrence_transform(state.nominal_transform, state.delta)
            candidate = sample_curve_by_arclength(
                template.model_copy(update={"spline": geometry}),
                transform,
                max(32, min(256, len(occurrence.target.xy))),
            )
            candidate_xy = _model_to_image(candidate.xy, occurrence.origin)
            match = monotone_curve_correspondence(
                candidate_xy,
                occurrence.target,
                candidate_arc_u=np.linspace(0.0, 1.0, len(candidate.xy)),
                sigma_position_px=correspondence_config.position_sigma_px,
                sigma_tangent_rad=correspondence_config.tangent_sigma_rad,
                tangent_weight=correspondence_config.tangent_weight,
                warp_weight=correspondence_config.warp_weight,
                gap_penalty=correspondence_config.gap_penalty,
                open_target_ends=not occurrence.target.supervise_endpoints,
            )
            correspondences[occurrence.key] = match
            candidate_u = candidate.u[match.candidate_indices]
            target_xy = occurrence.target.xy[match.target_indices]
            canonical = _image_to_canonical(target_xy, occurrence.origin, transform)
            weights = occurrence.target.confidence[match.target_indices].copy()
            weights[occurrence.target.ambiguous[match.target_indices]] = 0.0
            observations.append(SplineObservation(candidate_u, canonical, weights))

        occurrence_transforms = [
            compose_occurrence_transform(state.nominal_transform, state.delta)
            for state in transformed_states.values()
        ]
        typical_scale = float(
            np.median(
                [sqrt(transform.scale_x * transform.scale_y) for transform in occurrence_transforms]
            )
        )
        robust_scale_model = curve_config.robust_scale_px / typical_scale
        updated_controls = solve_shared_spline_coefficients(
            geometry,
            observations,
            seed_control_points=seed_controls,
            smoothness_lambda=curve_config.smoothness_lambda,
            seed_lambda=curve_config.seed_lambda,
            robust_scale=robust_scale_model,
            irls_iterations=irls_iterations,
        )
        scale = max(float(np.linalg.norm(starting_control_points)), 1.0)
        normalized_change = float(
            np.linalg.norm(updated_controls - np.asarray(geometry.control_points)) / scale
        )
        geometry = geometry.model_copy(
            update={"control_points": [tuple(point) for point in updated_controls]}
        )
        states = transformed_states
        completed_cycles = cycle + 1
        if normalized_change < tolerance:
            break

    widths = []
    width_weights = []
    for occurrence in targeted_occurrences:
        match = correspondences.get(occurrence.key)
        if match is None:
            continue
        state = states[occurrence.key]
        transform = compose_occurrence_transform(state.nominal_transform, state.delta)
        selected_indices = match.target_indices
        confidence = occurrence.target.confidence[selected_indices].copy()
        confidence[occurrence.target.ambiguous[selected_indices]] = 0.0
        scale = sqrt(transform.scale_x * transform.scale_y)
        widths.extend((2.0 * occurrence.target.radius[selected_indices] / scale).tolist())
        width_weights.extend(confidence.tolist())
    valid_widths = np.asarray(width_weights) > 0.0
    width_value = (
        _weighted_median(np.asarray(widths)[valid_widths], np.asarray(width_weights)[valid_widths])
        if np.any(valid_widths)
        else None
    )
    if width_value is not None:
        geometry = geometry.model_copy(update={"width": SplineWidthModel(value=width_value)})
    fitted_template = template.model_copy(update={"spline": geometry})
    return SplineFamilyFit(
        geometry=fitted_template,
        states=states,
        correspondences=correspondences,
        cycles=completed_cycles,
        normalized_change=normalized_change,
        width=geometry.width,
        assignment_cycles=assignment_cycles,
        assignment_change=assignment_change,
        targets={item.key: item.target for item in targeted_occurrences},
    )


def _spline_seed_from_template(
    template: PrimitiveTemplate, control_point_count: int
) -> SeedMigrationResult:

    if template.spline is not None:
        spline = template.spline
        report = {
            "format": "inkgrammar-spline-seed-migration-report",
            "format_version": 1,
            "source_free": True,
            "source_id": template.id,
            "source_kind": "bspline-centerline",
            "target_kind": "bspline-centerline",
            "degree": spline.degree,
            "control_point_count": len(spline.control_points),
            "interior_knot_count": len(spline.knots[4:-4]),
            "sample_count": 0,
            "closed_source": False,
            "maximum_deviation_model_units": 0.0,
            "rms_deviation_model_units": 0.0,
            "maximum_deviation_limit_model_units": 0.05,
            "within_tolerance": True,
            "endpoints_locked": spline.endpoint_policy.start == "locked"
            and spline.endpoint_policy.end == "locked",
            "canonical_seed_modified": False,
            "approved_for_canonical_use": False,
            "visual_review_required": True,
            "closed_loop_review_required": False,
        }
        return SeedMigrationResult(template=template, report=report)
    return migrate_legacy_seed(template, control_point_count=control_point_count)


def _has_polyline_self_intersection(points: np.ndarray) -> bool:
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    for first in range(len(points) - 1):
        a, b = points[first], points[first + 1]
        for second in range(first + 2, len(points) - 1):
            c, d = points[second], points[second + 1]
            if (
                orientation(a, b, c) * orientation(a, b, d) < 0.0
                and orientation(c, d, a) * orientation(c, d, b) < 0.0
            ):
                return True
    return False


def evaluate_topology_path(
    prepared_target: PreparedTopologyTarget,
    expected_topology: FamilyTopology,
    candidate_xy: np.ndarray,
    *,
    endpoint_tolerance_px: float = 3.0,
    cycle_penalty: float = 20.0,
    no_path_penalty: float = 50.0,
) -> TopologyPathResult:
    """Rank candidate paths against declared topology without rejecting mismatches."""
    candidate = np.asarray(candidate_xy, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) < 2:
        raise ValueError("candidate curve must have shape (N, 2) with N >= 2")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate curve must contain finite xy points")

    fallback_kinds = {
        "open": ("open", "open-with-loop", "loop"),
        "loop": ("loop", "open-with-loop", "open"),
        "open-with-loop": ("open-with-loop", "open", "loop"),
    }
    candidate_intersections = int(_has_polyline_self_intersection(candidate))
    ranked = []
    for path_kind in fallback_kinds[expected_topology.path_kind]:
        paths = rank_topology_paths(
            prepared_target.graph,
            path_kind=path_kind,
            endpoint_tolerance_px=endpoint_tolerance_px,
            candidate_xy=candidate,
            path_inventory=prepared_target.path_inventory,
        )
        for path in paths:
            expected_endpoints = 0 if expected_topology.path_kind == "loop" else 2
            cycle_mismatch = (
                0
                if expected_topology.expected_cycles is None
                else abs(path.cycle_count - expected_topology.expected_cycles)
            )
            endpoint_mismatch = abs(prepared_target.graph.endpoint_count - expected_endpoints)
            intersection_mismatch = (
                0
                if expected_topology.expected_self_intersections is None
                else max(
                    abs(candidate_intersections - expected_topology.expected_self_intersections),
                    abs(
                        int(_has_polyline_self_intersection(path.xy))
                        - expected_topology.expected_self_intersections
                    ),
                )
            )
            path_kind_mismatch = int(path_kind != expected_topology.path_kind)
            penalty_terms = {
                "cycle_count": cycle_penalty * cycle_mismatch,
                "endpoint_count": 10.0 * endpoint_mismatch,
                "self_intersection": 10.0 * intersection_mismatch,
                "path_kind": 20.0 * path_kind_mismatch,
                "ambiguous_path": 5.0 * int(path.ambiguous),
            }
            penalty = float(sum(penalty_terms.values()))
            path_score = path.score
            diagnostics = {
                "expected_path_kind": expected_topology.path_kind,
                "selected_path_kind": path_kind,
                "observed_graph_cycles": prepared_target.graph.cycle_count,
                "selected_path_cycles": path.cycle_count,
                "cycle_mismatch": cycle_mismatch,
                "expected_endpoint_count": expected_endpoints,
                "observed_endpoint_count": prepared_target.graph.endpoint_count,
                "endpoint_mismatch": endpoint_mismatch,
                "expected_self_intersections": expected_topology.expected_self_intersections,
                "candidate_self_intersections": candidate_intersections,
                "selected_path_self_intersections": int(_has_polyline_self_intersection(path.xy)),
                "self_intersection_mismatch": intersection_mismatch,
                "path_kind_mismatch": bool(path_kind_mismatch),
                "ambiguous": path.ambiguous,
                "orientation_ambiguous": path.orientation_ambiguous,
                "path_score": {
                    "guide_distance_px": path_score.guide_distance_px,
                    "endpoint_orientation_error_px": path_score.endpoint_orientation_error_px,
                    "measured_confidence": path_score.measured_confidence,
                    "length_error_px": path_score.length_error_px,
                    "turning_cost": path_score.turning_cost,
                },
                "no_path": False,
                "penalty_terms": penalty_terms,
                "penalty": penalty,
            }
            exact_match = (
                penalty == 0.0
                and path_kind == expected_topology.path_kind
                and not path.ambiguous
            )
            ranked.append(
                (
                    penalty,
                    round(path_score.guide_distance_px, 9)
                    if path_score.guide_distance_px is not None
                    else 0.0,
                    path_score.endpoint_orientation_error_px
                    if path_score.endpoint_orientation_error_px is not None
                    else 0.0,
                    -path_score.measured_confidence,
                    path_score.length_error_px if path_score.length_error_px is not None else 0.0,
                    path_score.turning_cost if path_score.turning_cost is not None else 0.0,
                    path.edge_ids,
                    path,
                    exact_match,
                    diagnostics,
                )
            )

    if not ranked:
        diagnostics = {
            "expected_path_kind": expected_topology.path_kind,
            "selected_path_kind": None,
            "observed_graph_cycles": prepared_target.graph.cycle_count,
            "selected_path_cycles": None,
            "cycle_mismatch": None,
            "expected_endpoint_count": 0 if expected_topology.path_kind == "loop" else 2,
            "observed_endpoint_count": prepared_target.graph.endpoint_count,
            "endpoint_mismatch": None,
            "expected_self_intersections": expected_topology.expected_self_intersections,
            "candidate_self_intersections": candidate_intersections,
            "selected_path_self_intersections": None,
            "self_intersection_mismatch": None,
            "path_kind_mismatch": False,
            "ambiguous": False,
            "orientation_ambiguous": False,
            "path_score": None,
            "no_path": True,
            "penalty_terms": {"no_path": no_path_penalty},
            "penalty": no_path_penalty,
        }
        return TopologyPathResult(None, False, no_path_penalty, diagnostics)

    ranked.sort(key=lambda item: item[:7])
    best = ranked[0]
    return TopologyPathResult(best[7], best[8], best[0], best[9])


def optimize_primitive_family_v4(
    bundle: Any,
    primitive_id: str,
    *,
    resume: bool = False,
):
    import json

    from scipy.optimize import differential_evolution, dual_annealing
    from scipy.spatial import cKDTree

    from .. import __version__
    from .checkpoint import (
        CheckpointCreator,
        CheckpointOptimizer,
        CheckpointProgress,
        CheckpointTraceSummary,
        FitCheckpoint,
        load_checkpoint,
        save_checkpoint,
    )
    from .family_fit import (
        FitRunArtifacts,
        _input_hashes,
        _prepare_sample,
        _project_output_directory,
        _safe_fit_id,
        _write_private_overlays,
    )
    from .family_fit_v3 import _prepare_occurrences
    from .models import SplineWidthModel
    from .spline import control_roughness_matrix, evaluate_primitive_curve

    _safe_fit_id(primitive_id)
    family = next(
        (item for item in bundle.families.families if item.primitive == primitive_id),
        None,
    )
    if family is None:
        raise ValueError(f"unknown fitting family for primitive: {primitive_id}")
    fit_ids = set(family.fit)
    validation_ids = set(family.validation)
    holdout_ids = set(family.hold_out)
    training = [sample for sample in bundle.corpus.samples if sample.id in fit_ids]
    validation = [sample for sample in bundle.corpus.samples if sample.id in validation_ids]
    hold_out = [sample for sample in bundle.corpus.samples if sample.id in holdout_ids]
    if not training:
        raise ValueError(f"primitive family has no fit samples: {primitive_id}")

    objective = bundle.fit_config.objective
    optimizer = bundle.fit_config.optimizer
    fit_backend = {
        "spline-alt-lsq": "inkgrammar-spline-alt-lsq",
        "spline-annealed-alt-lsq": "inkgrammar-annealed-spline-alt-lsq",
        "dual-annealing+spline-alt-lsq": "scipy-dual-annealing-seed",
        "spline-random-restarts": "inkgrammar-random-restart-spline-seed",
        "differential-evolution+spline-alt-lsq": "scipy-differential-evolution-spline-seed",
    }[optimizer.method]
    family_topology = family.topology or FamilyTopology(
        path_kind="open",
        expected_cycles=0,
        expected_self_intersections=0,
    )
    source_template = bundle.grammar.primitives[primitive_id]
    seed_migration = _spline_seed_from_template(
        source_template, objective.curve.default_control_points
    )
    base_template = seed_migration.template
    occurrences = _prepare_occurrences(bundle, family, training)
    base_states = {item.key: item.state for item in occurrences}
    initial_parameters = None

    def parameter_map(
        template: PrimitiveTemplate,
        states: dict[str, OccurrenceFitState],
    ) -> dict[str, float]:
        assert template.spline is not None
        result = {"shared.degree": float(template.spline.degree)}
        result.update(
            {
                f"shared.knots.{index}": float(value)
                for index, value in enumerate(template.spline.knots)
            }
        )
        for index, point in enumerate(template.spline.control_points):
            result[f"shared.control_points.{index}.x"] = float(point[0])
            result[f"shared.control_points.{index}.y"] = float(point[1])
        result["shared.width.value"] = (
            float(template.spline.width.value) if template.spline.width is not None else 0.0
        )
        for key, state in sorted(states.items()):
            for name in ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x"):
                result[f"occurrence.{key}.delta.{name}"] = float(getattr(state.delta, name))
        return result

    initial_parameters = parameter_map(base_template, base_states)
    input_hashes = _input_hashes(bundle, training)
    stage = f"primitive:{primitive_id}"
    checkpoint_path = (
        _project_output_directory(bundle.project_dir, "checkpoints") / f"{primitive_id}.json"
    )
    reports_dir = _project_output_directory(bundle.project_dir, "reports")
    json_report_path = reports_dir / f"{primitive_id}.json"
    markdown_report_path = reports_dir / f"{primitive_id}.md"
    if any(path.is_symlink() for path in (checkpoint_path, json_report_path, markdown_report_path)):
        raise ValueError("fitting output files must not be symlinks")

    template = base_template
    states = dict(base_states)
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if (
            checkpoint.project_id != bundle.project.id
            or checkpoint.stage != stage
            or checkpoint.inputs != input_hashes
            or checkpoint.optimizer.backend != fit_backend
            or checkpoint.optimizer.seed != optimizer.seed
        ):
            raise ValueError("checkpoint inputs or optimizer configuration do not match")
        stored = checkpoint.parameters
        if set(stored) != set(initial_parameters):
            raise ValueError("v4 checkpoint parameters do not match the spline and occurrences")
        if any(
            not np.isclose(stored[name], initial_parameters[name], atol=1e-12)
            for name in initial_parameters
            if name.startswith("shared.knots.") or name == "shared.degree"
        ):
            raise ValueError("v4 checkpoint spline basis does not match the configured seed")
        controls = [
            (
                stored[f"shared.control_points.{index}.x"],
                stored[f"shared.control_points.{index}.y"],
            )
            for index in range(len(template.spline.control_points))
        ]
        width_value = stored["shared.width.value"]
        spline = template.spline.model_copy(
            update={
                "control_points": controls,
                "width": SplineWidthModel(value=width_value) if width_value > 0.0 else None,
            }
        )
        template = template.model_copy(update={"spline": spline})
        for key, state in states.items():
            delta = state.delta.model_copy(
                update={
                    name: stored[f"occurrence.{key}.delta.{name}"]
                    for name in ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x")
                }
            )
            states[key] = state.model_copy(update={"delta": delta})

    def target_builder(raw_occurrences):
        masks = {}
        for item in raw_occurrences:
            region = np.zeros(item.prepared.reference.shape, dtype=bool)
            for _, region_mask, instance_ids in item.prepared.regions:
                if item.part.instance in instance_ids:
                    region |= region_mask > 0.0
            target_mask = (item.prepared.reference > 0.05) & region
            if not np.any(target_mask):
                raise ValueError(f"no target ink for occurrence {item.key}")
            masks[item.key] = target_mask
        prepared_targets = {
            item.key: prepare_topology_target(
                masks[item.key],
                min_spur_length_px=objective.centerline.min_spur_length_px,
            )
            for item in raw_occurrences
        }
        topology_diagnostics = {}
        sample_count = [objective.centerline.sample_schedule[-1]]
        reference_selected_edges = {}
        fallback_paths = {}

        def update(curve_template, current_states):
            targets = {}
            for item in raw_occurrences:
                state = current_states[item.key]
                transform = compose_occurrence_transform(item.part.transform, state.delta)
                candidate = sample_curve_by_arclength(
                    curve_template,
                    transform,
                    max(32, min(256, sample_count[0])),
                )
                candidate_xy = _model_to_image(candidate.xy, item.prepared.anchor)
                topology_result = evaluate_topology_path(
                    prepared_targets[item.key],
                    family_topology,
                    candidate_xy,
                    endpoint_tolerance_px=max(4.0, 2.0 * objective.assignment.sigma_px),
                    cycle_penalty=optimizer.anneal_topology_penalty,
                    no_path_penalty=optimizer.anneal_no_path_penalty,
                )
                selected_path = topology_result.path
                if selected_path is None:
                    selected_path = fallback_paths.get(item.key)
                if selected_path is None:
                    raise NoTopologyPathError(
                        f"no plausible topology path for occurrence {item.key}"
                    )
                fallback_paths.setdefault(item.key, selected_path)
                path_kind = family_topology.path_kind
                expected_endpoints = None
                target, graph, path = extract_observed_curve(
                    masks[item.key],
                    min_spur_length_px=objective.centerline.min_spur_length_px,
                    prepared_target=prepared_targets[item.key],
                    path_kind=path_kind,
                    preselected_path=selected_path,
                    expected_endpoints=expected_endpoints,
                    candidate_xy=candidate_xy,
                    endpoint_tolerance_px=max(4.0, 2.0 * objective.assignment.sigma_px),
                    sample_count=max(32, min(256, sample_count[0])),
                )
                targets[item.key] = target
                selected_edges = tuple(path.edge_ids)
                reference_edges = reference_selected_edges.get(item.key)
                if reference_edges is None:
                    reference_selected_edges[item.key] = selected_edges
                    path_changed = False
                else:
                    path_changed = reference_edges != selected_edges
                candidate_self_intersections = int(_has_polyline_self_intersection(candidate_xy))
                observed_self_intersections = int(_has_polyline_self_intersection(path.xy))
                topology_diagnostics[item.key] = {
                    **topology_result.diagnostics,
                    "topology_penalty": topology_result.topology_penalty,
                    "topology_exact": topology_result.exact_match,
                    "path_kind": family_topology.path_kind,
                    "expected_component_count": 1,
                    "expected_endpoint_count": topology_result.diagnostics[
                        "expected_endpoint_count"
                    ],
                    "selected_endpoint_count": 0 if path.start_node == path.end_node else 2,
                    "expected_cycles": family_topology.expected_cycles,
                    "observed_cycles": graph.cycle_count,
                    "expected_self_intersections": family_topology.expected_self_intersections,
                    "candidate_self_intersections": candidate_self_intersections,
                    "observed_self_intersections": observed_self_intersections,
                    "target_path_changed": path_changed,
                    "selected_cycles": path.cycle_count,
                    "component_count": graph.component_count,
                    "endpoint_count": graph.endpoint_count,
                    "junction_count": graph.junction_count,
                    "selected_edges": list(selected_edges),
                    "ambiguous": path.ambiguous,
                    "alternatives": path.alternatives,
                    "confidence": float(path.confidence),
                }
            return targets

        def set_sample_count(value: int) -> None:
            sample_count[0] = value

        return update, topology_diagnostics, set_sample_count

    target_update, topology_diagnostics, set_target_sample_count = target_builder(occurrences)
    if optimizer.method in {
        "spline-annealed-alt-lsq",
        "dual-annealing+spline-alt-lsq",
        "spline-random-restarts",
        "differential-evolution+spline-alt-lsq",
    }:
        initial_targets = {}
        spline_occurrences = []
    else:
        initial_targets = target_update(base_template, base_states)
        spline_occurrences = [
            SplineOccurrence(
                key=item.key,
                target=initial_targets[item.key],
                state=states[item.key],
                origin=item.prepared.anchor,
            )
            for item in occurrences
        ]
    annealing_metadata = None
    search_energy_summary = None
    search_evaluation_count = 0
    if optimizer.method in {
        "spline-annealed-alt-lsq",
        "dual-annealing+spline-alt-lsq",
        "spline-random-restarts",
        "differential-evolution+spline-alt-lsq",
    }:
        base_controls = np.asarray(base_template.spline.control_points, dtype=float)
        if len(base_controls) != 6:
            raise ValueError("annealed v4 fitting currently requires six spline control points")
        if (
            base_template.spline.endpoint_policy.start != "locked"
            or base_template.spline.endpoint_policy.end != "locked"
        ):
            raise ValueError("annealed v4 fitting requires locked spline endpoints")
        seed_width = max(float(np.ptp(base_controls[:, 0])), 1e-6)
        seed_height = max(float(np.ptp(base_controls[:, 1])), 1e-6)
        displacement_bounds = np.tile([0.20 * seed_width, 0.25 * seed_height], 4)
        curve_scale = max(float(np.linalg.norm(base_controls[-1] - base_controls[0])), 1.0)

        def template_from_normalized(parameters: np.ndarray) -> PrimitiveTemplate:
            normalized = np.asarray(parameters, dtype=float)
            if normalized.shape != (8,) or not np.isfinite(normalized).all():
                raise ValueError("annealed spline state must be a finite eight-dimensional vector")
            if np.any(np.abs(normalized) > 1.0):
                raise ValueError("annealed spline state must stay inside [-1, 1]")
            controls = base_controls.copy()
            controls[1:5] += (normalized * displacement_bounds).reshape(4, 2)
            spline = base_template.spline.model_copy(
                update={"control_points": [tuple(point) for point in controls]}
            )
            return base_template.model_copy(update={"spline": spline})

        def refine_local(
            start_template: PrimitiveTemplate,
            start_states: dict[str, OccurrenceFitState],
            *,
            cycles: int,
            sample_count: int,
            transform_nfev: int,
            irls_iterations: int,
        ) -> tuple[SplineFamilyFit, dict[str, ObservedCurve], dict[str, dict[str, Any]]]:
            set_target_sample_count(sample_count)
            try:
                targets = target_update(start_template, start_states)
                local_occurrences = [
                    SplineOccurrence(
                        key=item.key,
                        target=targets[item.key],
                        state=start_states[item.key],
                        origin=item.prepared.anchor,
                    )
                    for item in occurrences
                ]
                local_fit = fit_spline_family(
                    start_template,
                    local_occurrences,
                    curve_config=objective.curve,
                    correspondence_config=objective.correspondence,
                    max_cycles=cycles,
                    transform_max_nfev=transform_nfev,
                    irls_iterations=irls_iterations,
                    tolerance=optimizer.convergence_relative,
                    target_updater=target_update,
                    assignment_update_interval=1,
                    seed_control_points=base_controls,
                )
                final_local_targets = target_update(local_fit.geometry, local_fit.states)
                local_topology = dict(topology_diagnostics)
                return local_fit, final_local_targets, local_topology
            finally:
                set_target_sample_count(objective.centerline.sample_schedule[-1])

        def score_local_fit(
            local_fit: SplineFamilyFit,
            targets: dict[str, ObservedCurve],
            local_topology: dict[str, dict[str, Any]],
        ) -> FamilySearchEnergy:
            geometry_values = []
            endpoint_values = []
            tangent_values = []
            topology_values = []
            saturation_values = []
            for item in occurrences:
                target = targets[item.key]
                state = local_fit.states[item.key]
                transform = compose_occurrence_transform(item.part.transform, state.delta)
                candidate = sample_curve_by_arclength(
                    local_fit.geometry,
                    transform,
                    max(128, len(target.xy)),
                )
                candidate_xy = _model_to_image(candidate.xy, item.prepared.anchor)
                candidate_to_target, nearest = cKDTree(target.xy).query(candidate_xy)
                target_to_candidate = cKDTree(candidate_xy).query(target.xy)[0]
                geometry_values.append(
                    0.5
                    * (
                        float(np.percentile(candidate_to_target, 95))
                        + float(np.percentile(target_to_candidate, 95))
                    )
                )
                if target.supervise_endpoints:
                    endpoint_values.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    [
                                        np.linalg.norm(candidate_xy[0] - target.xy[0]) ** 2,
                                        np.linalg.norm(candidate_xy[-1] - target.xy[-1]) ** 2,
                                    ]
                                )
                            )
                        )
                    )
                candidate_tangent = _unit_tangents(candidate_xy)
                tangent_dot = np.sum(candidate_tangent * target.tangent[nearest], axis=1)
                tangent_angles = np.arccos(np.clip(tangent_dot, -1.0, 1.0))
                tangent_values.append(float(np.sqrt(np.mean(tangent_angles**2))))
                topology_values.append(float(local_topology[item.key]["topology_penalty"]))
                for name in state.free_parameters:
                    lower, upper = getattr(state.bounds, name)
                    if upper > lower:
                        normalized = (
                            2.0 * (getattr(state.delta, name) - lower) / (upper - lower) - 1.0
                        )
                        saturation_values.append(max(0.0, abs(normalized) - 0.85) ** 2 / 0.15**2)

            control_points = np.asarray(local_fit.geometry.spline.control_points, dtype=float)
            roughness = control_roughness_matrix(len(control_points), order=2) @ control_points
            roughness_penalty = float(np.linalg.norm(roughness) ** 2 / curve_scale**2)
            seed_distance_penalty = float(
                np.sum((control_points - base_controls) ** 2) / curve_scale**2
            )
            path_change = float(
                np.mean([item["target_path_changed"] for item in local_topology.values()])
            )
            assignment_entropy = float(
                np.mean([match.mean_entropy for match in local_fit.correspondences.values()])
            )
            assignment_instability = float(
                np.mean([local_fit.assignment_change, path_change, assignment_entropy])
            )
            no_path = any(item["no_path"] for item in local_topology.values())
            return evaluate_family_search_energy(
                geometry_values,
                endpoint_values=endpoint_values,
                tangent_values=tangent_values,
                topology_values=topology_values,
                saturation_values=saturation_values,
                assignment_instability=assignment_instability,
                roughness_penalty=roughness_penalty,
                seed_distance_penalty=seed_distance_penalty,
                family_worst_weight=objective.family.worst,
                family_spread_weight=objective.family.spread,
                saturation_weight=optimizer.anneal_transform_saturation_penalty,
                invalid_candidate=no_path,
            )

        def refine_parameters(
            parameters: np.ndarray,
            *,
            cycles: int,
            sample_count: int,
            transform_nfev: int,
            irls_iterations: int,
            start_fit: SplineFamilyFit | None = None,
        ) -> dict[str, Any]:
            start_template = (
                template_from_normalized(parameters) if start_fit is None else start_fit.geometry
            )
            start_states = base_states if start_fit is None else start_fit.states
            local_fit, local_targets, local_topology = refine_local(
                start_template,
                start_states,
                cycles=cycles,
                sample_count=sample_count,
                transform_nfev=transform_nfev,
                irls_iterations=irls_iterations,
            )
            return {
                "parameters": np.asarray(parameters, dtype=float).copy(),
                "fit": local_fit,
                "targets": local_targets,
                "topology": local_topology,
                "energy": score_local_fit(local_fit, local_targets, local_topology),
            }

        coarse_records = []
        coarse_cache = {}
        coarse_fit_evaluations = 0

        def coarse_objective(parameters: np.ndarray) -> float:
            nonlocal coarse_fit_evaluations
            key = tuple(float(value) for value in parameters)
            record = coarse_cache.get(key)
            if record is None:
                try:
                    record = refine_parameters(
                        parameters,
                        cycles=optimizer.anneal_coarse_cycles,
                        sample_count=32,
                        transform_nfev=min(30, optimizer.transform_max_nfev),
                        irls_iterations=1,
                    )
                except NoTopologyPathError:
                    record = {
                        "parameters": np.asarray(parameters, dtype=float).copy(),
                        "fit": None,
                        "targets": {},
                        "topology": {},
                        "energy": evaluate_family_search_energy(
                            [0.0],
                            topology_values=[optimizer.anneal_no_path_penalty],
                            invalid_candidate=True,
                        ),
                    }
                coarse_cache[key] = record
                coarse_fit_evaluations += 1
            coarse_records.append(record)
            return record["energy"].total
        annealing_config = AnnealingConfig(
            steps=optimizer.anneal_steps,
            initial_samples=optimizer.anneal_initial_samples,
            cooling=optimizer.anneal_cooling,
            sigma_max=optimizer.anneal_sigma_max,
            sigma_min=optimizer.anneal_sigma_min,
            reheat_patience=optimizer.anneal_reheat_patience,
            restart_patience=optimizer.anneal_restart_patience,
            target_uphill_acceptance=optimizer.anneal_target_uphill_acceptance,
        )
        evaluation_budget = optimizer.anneal_steps + optimizer.anneal_initial_samples + 1
        if optimizer.method == "spline-annealed-alt-lsq":
            annealing_result = simulated_annealing(
                np.zeros(8),
                [(-1.0, 1.0)] * 8,
                coarse_objective,
                config=annealing_config,
                seed=optimizer.seed,
            )
            search_metadata = {
                "method": "inkgrammar-simulated-annealing",
                **annealing_result.trace_summary,
            }
        elif optimizer.method == "dual-annealing+spline-alt-lsq":
            scipy_result = dual_annealing(
                coarse_objective,
                bounds=[(-1.0, 1.0)] * 8,
                maxiter=optimizer.anneal_steps,
                maxfun=evaluation_budget,
                seed=optimizer.seed,
                x0=np.zeros(8),
                no_local_search=True,
            )
            search_metadata = {
                "method": "scipy-dual-annealing",
                "objective_evaluations": float(scipy_result.nfev),
                "best_energy": float(scipy_result.fun),
                "status": float(scipy_result.status),
                "success": bool(scipy_result.success),
                "message": str(scipy_result.message),
            }
        elif optimizer.method == "differential-evolution+spline-alt-lsq":
            scipy_result = differential_evolution(
                coarse_objective,
                bounds=[(-1.0, 1.0)] * 8,
                maxiter=max(0, evaluation_budget // 8 - 1),
                popsize=1,
                tol=0.0,
                atol=0.0,
                polish=False,
                seed=optimizer.seed,
                updating="immediate",
                workers=1,
                x0=np.zeros(8),
            )
            search_metadata = {
                "method": "scipy-differential-evolution",
                "objective_evaluations": float(scipy_result.nfev),
                "best_energy": float(scipy_result.fun),
                "success": bool(scipy_result.success),
                "message": str(scipy_result.message),
            }
        else:
            rng = np.random.default_rng(optimizer.seed)
            random_parameters = [np.zeros(8)]
            random_parameters.extend(rng.uniform(-1.0, 1.0, size=(evaluation_budget - 1, 8)))
            for parameters in random_parameters:
                coarse_objective(parameters)
            search_metadata = {
                "method": "inkgrammar-random-restarts",
                "evaluations": float(len(random_parameters)),
                "best_energy": float(
                    min(record["energy"].total for record in coarse_records)
                ),
            }
        search_metadata["coarse_fit_evaluations"] = float(coarse_fit_evaluations)
        ranked_coarse = sorted(
            (record for record in coarse_records if record["fit"] is not None),
            key=lambda record: record["energy"].total,
        )
        if not ranked_coarse:
            raise NoTopologyPathError("no topology path supports any annealed spline candidate")
        unique_coarse = []
        for record in ranked_coarse:
            if not any(
                np.linalg.norm(record["parameters"] - existing["parameters"]) < 1e-6
                for existing in unique_coarse
            ):
                unique_coarse.append(record)
            if len(unique_coarse) == optimizer.anneal_top_k:
                break

        medium_records = []
        for record in unique_coarse:
            medium_records.append(
                refine_parameters(
                    record["parameters"],
                    cycles=optimizer.anneal_medium_cycles,
                    sample_count=64,
                    transform_nfev=min(60, optimizer.transform_max_nfev),
                    irls_iterations=min(2, optimizer.irls_iterations),
                    start_fit=record["fit"],
                )
            )
        best_medium = min(medium_records, key=lambda record: record["energy"].total)
        final_record = refine_parameters(
            best_medium["parameters"],
            cycles=optimizer.anneal_final_cycles,
            sample_count=objective.centerline.sample_schedule[-1],
            transform_nfev=optimizer.transform_max_nfev,
            irls_iterations=optimizer.irls_iterations,
            start_fit=best_medium["fit"],
        )
        fit = final_record["fit"]
        template = fit.geometry
        states = fit.states
        final_targets = final_record["targets"]
        topology_diagnostics = final_record["topology"]
        best_coarse = ranked_coarse[0]
        annealing_metadata = {
            **search_metadata,
            "top_k_refined": float(len(medium_records)),
        }
        search_evaluation_count = int(search_metadata["coarse_fit_evaluations"])
        search_evaluation_count += len(medium_records) + 1
        search_energy_summary = {
            "initial": coarse_records[0]["energy"].as_dict(),
            "best_coarse": best_coarse["energy"].as_dict(),
            "final": final_record["energy"].as_dict(),
        }
    else:
        fit = fit_spline_family(
            template,
            spline_occurrences,
            curve_config=objective.curve,
            correspondence_config=objective.correspondence,
            max_cycles=optimizer.max_cycles,
            transform_max_nfev=optimizer.transform_max_nfev,
            irls_iterations=optimizer.irls_iterations,
            tolerance=optimizer.convergence_relative,
            target_updater=target_update,
            assignment_update_interval=optimizer.assignment_update_interval,
            seed_control_points=np.asarray(base_template.spline.control_points),
        )
        template = fit.geometry
        states = fit.states
        final_targets = target_update(template, states)
    

    if not initial_targets:
        initial_targets = target_update(base_template, base_states)

    review_curves = {
        item.key: (final_targets[item.key].xy, final_targets[item.key].tangent)
        for item in occurrences
    }
    assignment_entropies = {}
    assigned_coverage = {}
    for item in occurrences:
        transform = compose_occurrence_transform(item.part.transform, states[item.key].delta)
        candidate = sample_curve_by_arclength(
            template, transform, max(32, min(256, len(final_targets[item.key].xy)))
        )
        correspondence = monotone_curve_correspondence(
            _model_to_image(candidate.xy, item.prepared.anchor),
            final_targets[item.key],
            candidate_arc_u=np.linspace(0.0, 1.0, len(candidate.xy)),
            sigma_position_px=objective.correspondence.position_sigma_px,
            sigma_tangent_rad=objective.correspondence.tangent_sigma_rad,
            tangent_weight=objective.correspondence.tangent_weight,
            warp_weight=objective.correspondence.warp_weight,
            gap_penalty=objective.correspondence.gap_penalty,
            open_target_ends=not final_targets[item.key].supervise_endpoints,
        )
        assignment_entropies[item.key] = correspondence.mean_entropy
        assigned_coverage[item.key] = len(set(correspondence.target_indices)) / len(
            final_targets[item.key].xy
        )

    def geometry_metrics(curve_template, item, state, target):
        transform = compose_occurrence_transform(item.part.transform, state.delta)
        candidate = sample_curve_by_arclength(
            curve_template,
            transform,
            max(256, len(target.xy)),
        )
        candidate_xy = _model_to_image(candidate.xy, item.prepared.anchor)
        candidate_to_target = cKDTree(target.xy).query(candidate_xy)[0]
        endpoint_rms = None
        if target.supervise_endpoints:
            endpoint_errors = np.asarray(
                [
                    np.linalg.norm(candidate_xy[0] - target.xy[0]),
                    np.linalg.norm(candidate_xy[-1] - target.xy[-1]),
                ]
            )
            endpoint_rms = float(np.sqrt(np.mean(endpoint_errors**2)))
        target_to_candidate = cKDTree(candidate_xy).query(target.xy)[0]
        return {
            "candidate_to_target_p95_px": float(np.percentile(candidate_to_target, 95)),
            "target_to_candidate_p95_px": float(np.percentile(target_to_candidate, 95)),
            "mean_nearest_distance_px": float(
                (np.mean(candidate_to_target) + np.mean(target_to_candidate)) / 2.0
            ),
            "supervised_endpoint_rms_px": endpoint_rms,
        }

    initial_metrics = {
        item.key: geometry_metrics(
            base_template, item, base_states[item.key], initial_targets[item.key]
        )
        for item in occurrences
    }
    final_metrics = {
        item.key: geometry_metrics(template, item, states[item.key], final_targets[item.key])
        for item in occurrences
    }

    def training_summary(metrics_by_key, states_by_key):
        grouped = {}
        for item in occurrences:
            metrics = metrics_by_key[item.key]
            instance_score = (
                metrics["candidate_to_target_p95_px"] + metrics["target_to_candidate_p95_px"]
            ) / 2.0
            grouped.setdefault(item.prepared.sample.id, []).append(
                {
                    "instance_id": item.part.instance,
                    "score": instance_score,
                    "geometry": metrics,
                    "transform": states_by_key[item.key].delta.model_dump(mode="json"),
                    "topology": topology_diagnostics[item.key],
                }
            )
        rows = []
        for sample in training:
            instances = grouped.get(sample.id, [])
            rows.append(
                {
                    "sample_id": sample.id,
                    "initial_objective": float(np.mean([row["score"] for row in instances])),
                    "final_objective": float(np.mean([row["score"] for row in instances])),
                    "objective": float(np.mean([row["score"] for row in instances])),
                    "instances": [
                        {key: value for key, value in row.items() if key != "score"}
                        for row in instances
                    ],
                }
            )
        scores = [row["objective"] for row in rows]
        return {
            "selection_split": "fit",
            "initial_objective": float(np.mean(scores)),
            "final_objective": float(np.mean(scores)),
            "mean_objective": float(np.mean(scores)),
            "worst_member_objective": float(np.max(scores)),
            "std_objective": float(np.std(scores)),
            "samples": rows,
        }

    training_report = training_summary(final_metrics, states)
    initial_report = training_summary(initial_metrics, base_states)
    for row, initial_row in zip(training_report["samples"], initial_report["samples"], strict=True):
        row["initial_objective"] = initial_row["final_objective"]
    training_report["initial_objective"] = initial_report["final_objective"]

    def diagnostic_split(samples):
        if not samples:
            return {
                "samples": [],
                "mean_objective": None,
                "policy": "post-fit diagnostic only; parameters are unchanged",
            }
        raw = _prepare_occurrences(bundle, family, samples)
        split_states = {item.key: item.state for item in raw}
        update, split_topology, _set_split_sample_count = target_builder(raw)
        targets = update(template, split_states)
        review_curves.update(
            {item.key: (targets[item.key].xy, targets[item.key].tangent) for item in raw}
        )
        grouped = {}
        scores = []
        for item in raw:
            metrics = geometry_metrics(template, item, split_states[item.key], targets[item.key])
            score = (
                metrics["candidate_to_target_p95_px"] + metrics["target_to_candidate_p95_px"]
            ) / 2.0
            grouped.setdefault(item.prepared.sample.id, []).append(
                {
                    "instance_id": item.part.instance,
                    "objective": score,
                    "geometry": metrics,
                    "topology": split_topology[item.key],
                }
            )
            scores.append(score)
        return {
            "policy": "post-fit diagnostic only; parameters are unchanged",
            "samples": [
                {
                    "sample_id": sample.id,
                    "objective": float(
                        np.mean([row["objective"] for row in grouped.get(sample.id, [])])
                    ),
                    "instances": grouped.get(sample.id, []),
                }
                for sample in samples
            ],
            "mean_objective": float(np.mean(scores)) if scores else None,
            "worst_member_objective": float(np.max(scores)) if scores else None,
        }

    validation_report = diagnostic_split(validation)
    holdout_report = diagnostic_split(hold_out)
    spline = template.spline
    control_points = np.asarray(spline.control_points)
    roughness = control_roughness_matrix(len(control_points), order=2) @ control_points
    dense_evaluation = sample_curve_by_arclength(template, TopologyTransform(), 512)
    curvature = np.abs(dense_evaluation.curvature)
    curve_length = float(np.sum(np.linalg.norm(np.diff(dense_evaluation.xy, axis=0), axis=1)))
    curvature_spacing = curve_length / (len(curvature) - 1)
    curvature_changes = np.diff(curvature)
    max_curvature_derivative = float(np.max(np.abs(curvature_changes)) / curvature_spacing)
    max_curvature_jump = float(np.max(np.abs(curvature_changes)))
    tangent_jumps = []
    for knot in spline.knots[4:-4]:
        epsilon = min(1e-6, float(knot) / 4.0, (1.0 - float(knot)) / 4.0)
        sides = evaluate_primitive_curve(
            template,
            TopologyTransform(),
            np.asarray([knot - epsilon, knot + epsilon]),
        ).tangent
        tangent_jumps.append(
            float(np.degrees(np.arccos(np.clip(np.dot(sides[0], sides[1]), -1.0, 1.0))))
        )
    saturation = []
    for key, state in sorted(states.items()):
        for name in state.free_parameters:
            lower, upper = getattr(state.bounds, name)
            value = float(getattr(state.delta, name))
            margin = 0.05 * (upper - lower)
            if value - lower <= margin or upper - value <= margin:
                saturation.append(
                    {
                        "parameter": f"occurrence.{key}.delta.{name}",
                        "value": value,
                        "bounds": [lower, upper],
                        "fraction_through_range": (value - lower) / (upper - lower),
                    }
                )
    topology_values = list(topology_diagnostics.values())
    target_path_change_fraction = float(
        np.mean([item["target_path_changed"] for item in topology_values])
    )
    topology_match = all(
        item["component_count"] == item["expected_component_count"]
        and item["selected_endpoint_count"] == item["expected_endpoint_count"]
        and (
            item["expected_cycles"] is None
            or item["selected_cycles"] == item["expected_cycles"]
        )
        and (
            item["expected_self_intersections"] is None
            or (
                item["candidate_self_intersections"] == item["expected_self_intersections"]
                and item["observed_self_intersections"] == item["expected_self_intersections"]
            )
        )
        and item["topology_penalty"] == 0.0
        for item in topology_values
    )
    diagnostics = {
        "curve_model": {
            "kind": "bspline-centerline",
            "degree": spline.degree,
            "control_point_count": len(spline.control_points),
            "interior_knot_count": len(spline.knots[4:-4]),
            "knots": list(spline.knots),
        },
        "smoothness": {
            "roughness_cost": float(np.sum(roughness**2)),
            "max_internal_tangent_jump_deg": max(tangent_jumps, default=0.0),
            "max_curvature": float(np.max(curvature)),
            "p95_curvature": float(np.percentile(curvature, 95)),
            "max_curvature_derivative": max_curvature_derivative,
            "max_curvature_jump": max_curvature_jump,
        },
        "assignment": {
            "cycles": fit.assignment_cycles,
            "final_change_fraction": fit.assignment_change,
            "mean_entropy": float(np.mean(list(assignment_entropies.values()))),
            "target_path_change_fraction": target_path_change_fraction,
            "mean_assigned_coverage": float(np.mean(list(assigned_coverage.values()))),
            "stable": (
                fit.assignment_change <= optimizer.convergence_relative
                and target_path_change_fraction <= optimizer.convergence_relative
            ),
        },
        "topology": {
            "expected_cycles": [item["expected_cycles"] for item in topology_values],
            "observed_cycles": [item["observed_cycles"] for item in topology_values],
            "expected_components": [item["expected_component_count"] for item in topology_values],
            "observed_components": [item["component_count"] for item in topology_values],
            "expected_endpoints": [item["expected_endpoint_count"] for item in topology_values],
            "observed_endpoints": [item["endpoint_count"] for item in topology_values],
            "expected_self_intersections": [
                item["expected_self_intersections"] for item in topology_values
            ],
            "observed_self_intersections": [
                item["observed_self_intersections"] for item in topology_values
            ],
            "matches": topology_match,
            "topology_exact_evaluations": sum(
                int(item["topology_exact"]) for item in topology_values
            ),
            "topology_penalized_evaluations": sum(
                int(item["topology_penalty"] > 0.0) for item in topology_values
            ),
            "no_path_evaluations": sum(int(item["no_path"]) for item in topology_values),
            "occurrences": topology_diagnostics,
        },
        "dense_geometry": {
            "max_candidate_to_target_p95_px": max(
                item["candidate_to_target_p95_px"] for item in final_metrics.values()
            ),
            "max_target_to_candidate_p95_px": max(
                item["target_to_candidate_p95_px"] for item in final_metrics.values()
            ),
            "max_supervised_endpoint_rms_px": max(
                (
                    item["supervised_endpoint_rms_px"]
                    for item in final_metrics.values()
                    if item["supervised_endpoint_rms_px"] is not None
                ),
                default=None,
            ),
            "minimum_assigned_coverage": min(assigned_coverage.values(), default=0.0),
        },
        "saturation": {"parameters": saturation},
        "complexity": {
            "degree": spline.degree,
            "control_point_count": len(spline.control_points),
            "interior_knot_count": len(spline.knots[4:-4]),
        },
        "seed_conversion": seed_migration.report,
        "annealing": (
            annealing_metadata
            if optimizer.method in {"spline-annealed-alt-lsq", "dual-annealing+spline-alt-lsq"}
            else None
        ),
        "seed_search": annealing_metadata,
        "search_energy": search_energy_summary,
    }

    occurrence_deltas = {}
    for item in occurrences:
        occurrence_deltas.setdefault(item.prepared.sample.id, {})[item.part.instance] = states[
            item.key
        ].delta
    review_samples = [
        _prepare_sample(bundle, sample, primitive_id)
        for sample in [*training, *validation, *hold_out]
    ]
    review_sheet = _write_private_overlays(
        bundle,
        primitive_id,
        template,
        review_samples,
        occurrence_deltas=occurrence_deltas,
        include_nominal=True,
        diagnostic_curves=review_curves,
    )
    final_geometry_values = list(final_metrics.values())
    max_candidate_p95 = max(item["candidate_to_target_p95_px"] for item in final_geometry_values)
    max_target_p95 = max(item["target_to_candidate_p95_px"] for item in final_geometry_values)
    endpoint_values = [
        item["supervised_endpoint_rms_px"]
        for item in final_geometry_values
        if item["supervised_endpoint_rms_px"] is not None
    ]
    max_endpoint_rms = max(endpoint_values, default=None)
    min_assigned_coverage = min(assigned_coverage.values(), default=0.0)
    all_saturation_clear = not saturation
    assignment_stable = diagnostics["assignment"]["stable"]
    all_saturation_clear = not saturation
    assignment_stable = diagnostics["assignment"]["stable"]
    promotion_gates = {
        "candidate_to_target_p95": max_candidate_p95 <= 2.0,
        "target_to_candidate_p95": max_target_p95 <= 2.0,
        "supervised_endpoint_rms": max_endpoint_rms is None or max_endpoint_rms <= 1.5,
        "assigned_coverage": min_assigned_coverage >= 0.90,
        "topology_match": topology_match,
        "assignment_stable": assignment_stable,
        "no_transform_saturation": all_saturation_clear,
        "visual_review": "pending",
        "automatic_promotion": False,
    }

    final_parameters = parameter_map(template, states)
    checkpoint_evaluations = (
        search_evaluation_count + fit.assignment_cycles * len(occurrences)
    )
    if checkpoint is None:
        checkpoint = FitCheckpoint(
            project_id=bundle.project.id,
            stage=stage,
            created_by=CheckpointCreator(inkgrammar_version=__version__),
            inputs=input_hashes,
            optimizer=CheckpointOptimizer(
                backend=fit_backend,
                seed=optimizer.seed,
                objective_version=objective.version,
            ),
            parameters=final_parameters,
        )
    prior_summary = checkpoint.trace_summary
    evaluation_count = prior_summary.evaluations + checkpoint_evaluations
    initial_score = training_report["initial_objective"]
    final_score = training_report["final_objective"]
    trace_summary = CheckpointTraceSummary(
        evaluations=evaluation_count,
        initial_objective=(
            prior_summary.initial_objective
            if prior_summary.initial_objective is not None
            else initial_score
        ),
        best_objective=min(
            value for value in (prior_summary.best_objective, final_score) if value is not None
        ),
        last_objective=final_score,
    )
    checkpoint = checkpoint.model_copy(
        update={
            "inputs": input_hashes,
            "optimizer": checkpoint.optimizer.model_copy(
                update={
                    "backend": fit_backend,
                    "seed": optimizer.seed,
                    "objective_version": objective.version,
                }
            ),
            "parameters": final_parameters,
            "training_metrics": {
                "initial_objective": float(initial_score),
                "final_objective": float(final_score),
                "roughness_cost": diagnostics["smoothness"]["roughness_cost"],
                "assignment_change": float(fit.assignment_change),
                "assignment_entropy": diagnostics["assignment"]["mean_entropy"],
                "target_path_change_fraction": diagnostics["assignment"][
                    "target_path_change_fraction"
                ],
                "assigned_coverage": diagnostics["assignment"]["mean_assigned_coverage"],
                "topology_match": float(topology_match),
                "topology_exact_evaluations": float(
                    diagnostics["topology"]["topology_exact_evaluations"]
                ),
                "topology_penalized_evaluations": float(
                    diagnostics["topology"]["topology_penalized_evaluations"]
                ),
                "topology_no_path_evaluations": float(
                    diagnostics["topology"]["no_path_evaluations"]
                ),
                "seed_search_coarse_fit_evaluations": float(
                    annealing_metadata.get("coarse_fit_evaluations", 0.0)
                    if annealing_metadata is not None
                    else 0.0
                ),
                "seed_search_initial_energy": float(
                    search_energy_summary["initial"]["total"]
                    if search_energy_summary is not None
                    else 0.0
                ),
                "seed_search_best_coarse_energy": float(
                    search_energy_summary["best_coarse"]["total"]
                    if search_energy_summary is not None
                    else 0.0
                ),
                "seed_search_final_energy": float(
                    search_energy_summary["final"]["total"]
                    if search_energy_summary is not None
                    else 0.0
                ),
                "seed_search_steps": float(
                    annealing_metadata.get("steps", 0.0)
                    if annealing_metadata is not None
                    else 0.0
                ),
                "seed_search_accepted_moves": float(
                    annealing_metadata.get("accepted_moves", 0.0)
                    if annealing_metadata is not None
                    else 0.0
                ),
            },
            "holdout_metrics": (
                {"mean_objective": float(holdout_report["mean_objective"])}
                if holdout_report["mean_objective"] is not None
                else {}
            ),
            "progress": CheckpointProgress(
                restart_index=checkpoint.progress.restart_index,
                cycle_index=checkpoint.progress.cycle_index + fit.cycles,
                evaluations=evaluation_count,
            ),
            "trace_summary": trace_summary,
        }
    )
    save_checkpoint(checkpoint, checkpoint_path)

    report = {
        "format": "inkgrammar-fit-report",
        "format_version": 4,
        "project_id": bundle.project.id,
        "stage": stage,
        "primitive_id": primitive_id,
        "objective_version": objective.version,
        "selection_objective": {
            "kind": "symmetric dense centerline distance p95",
            "total": float(final_score),
        },
        "optimizer": {
            "backend": fit_backend,
            "strategy": {
                "spline-alt-lsq": "topology-aware alternating sparse spline fit",
                "spline-annealed-alt-lsq": (
                    "topology-aware annealed basin search with alternating sparse spline refinement"
                ),
                "dual-annealing+spline-alt-lsq": (
                    "SciPy dual-annealing seed search with alternating sparse spline refinement"
                ),
                "spline-random-restarts": "uniform random spline seed restarts",
                "differential-evolution+spline-alt-lsq": (
                    "SciPy differential-evolution seed search with spline refinement"
                ),
            }[optimizer.method],
            "success": True,
            "cycles": fit.cycles,
            "evaluations": checkpoint_evaluations,
            "assignment_cycles": fit.assignment_cycles,
            "final_assignment_change": float(fit.assignment_change),
            "config": optimizer.model_dump(mode="json"),
        },
        "input_hashes": input_hashes.model_dump(mode="json"),
        "initial_parameters": initial_parameters,
        "parameters": final_parameters,
        "occurrences": {
            item.key: {
                "delta": states[item.key].delta.model_dump(mode="json"),
                "geometry": final_metrics[item.key],
                "topology": topology_diagnostics[item.key],
            }
            for item in occurrences
        },
        "primitive": template.model_dump(mode="json"),
        "training": training_report,
        "validation": validation_report,
        "hold_out": holdout_report,
        "generalization_gap": (
            holdout_report["mean_objective"] - training_report["mean_objective"]
            if holdout_report["mean_objective"] is not None
            else None
        ),
        "diagnostics": diagnostics,
        "promotion": {
            "status": "promotion_pending",
            "automatic_promotion": False,
            "gates": promotion_gates,
        },
        "visual_review": {
            "status": "promotion_pending",
            "scope": "private work directory",
            "variants": ["nominal", "aligned"],
            "sheet": review_sheet.relative_to(bundle.private_work_dir).as_posix(),
            "region_overlay_count": sum(len(sample.regions) for sample in review_samples),
        },
        "evaluations": checkpoint_evaluations,
    }
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        f"# Fit report: {primitive_id}",
        "",
        "Objective version: 4",
        "",
        f"Training objective: {initial_score:.6f} to {final_score:.6f}",
        f"Assignment changes: {fit.assignment_cycles}, final fraction {fit.assignment_change:.6f}",
        f"Topology match: {topology_match}",
        f"Automatic promotion: {promotion_gates['automatic_promotion']}",
        "",
        "## QC diagnostics",
        "",
        f"Candidate-to-target p95 maximum: {max_candidate_p95:.3f} px",
        f"Target-to-candidate p95 maximum: {max_target_p95:.3f} px",
        (
            f"Maximum supervised endpoint RMS: {max_endpoint_rms:.3f} px"
            if max_endpoint_rms is not None
            else "Maximum supervised endpoint RMS: not supervised"
        ),
        f"Minimum assigned coverage: {min_assigned_coverage:.3f}",
        f"Mean assignment entropy: {diagnostics['assignment']['mean_entropy']:.3f}",
        f"Target path change fraction: {target_path_change_fraction:.3f}",
        f"Maximum curvature: {diagnostics['smoothness']['max_curvature']:.6f}",
        f"p95 curvature: {diagnostics['smoothness']['p95_curvature']:.6f}",
        "",
        "## Promotion gates",
        *[f"- {name}: {value}" for name, value in promotion_gates.items()],
        "",
        "## Dense geometry diagnostics",
        "",
        "| Sample | Mean objective |",
        "| --- | ---: |",
        *[
            f"| `{row['sample_id']}` | {row['objective']:.6f} |"
            for row in training_report["samples"]
        ],
        "",
        "Validation and hold-out samples are post-fit diagnostics only.",
        "Visual review remains pending in the private work directory.",
        "",
    ]
    if annealing_metadata is not None:
        markdown.extend(
            [
                "## Global seed search",
                "",
                f"Method: {annealing_metadata['method']}",
                f"Coarse fit evaluations: {annealing_metadata['coarse_fit_evaluations']}",
                f"Initial energy: {search_energy_summary['initial']['total']:.6f}",
                f"Final energy: {search_energy_summary['final']['total']:.6f}",
                "",
            ]
        )
    markdown_report_path.write_text("\n".join(markdown), encoding="utf-8")
    return FitRunArtifacts(
        checkpoint_path=checkpoint_path,
        json_report_path=json_report_path,
        markdown_report_path=markdown_report_path,
        report=report,
        private_review_sheet_path=review_sheet,
    )
