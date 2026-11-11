from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import sqrt
from typing import Any, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares

from .models import (
    OccurrenceFitState,
    PrimitiveTemplate,
    SplineWidthModel,
    TopologyTransform,
    compose_occurrence_transform,
)
from .objective import SplineCorrespondenceConfig, SplineCurveConfig
from .seed_migration import SeedMigrationResult, migrate_legacy_seed
from .skeleton_graph import SkeletonGraph, SkeletonPath, build_skeleton_graph, select_topology_path
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
    endpoint_tolerance_px: float = 4.0,
    sample_count: int = 128,
) -> tuple[ObservedCurve, SkeletonGraph, SkeletonPath]:
    """Build an ordered, confidence-weighted curve from a topology-selected skeleton path."""
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
    path = select_topology_path(
        graph,
        path_kind=path_kind,
        expected_cycles=expected_cycles,
        expected_endpoints=expected_endpoints,
        endpoint_tolerance_px=endpoint_tolerance_px,
        candidate_xy=candidate_xy,
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


def optimize_primitive_family_v4(
    bundle: Any,
    primitive_id: str,
    *,
    resume: bool = False,
):
    import json

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
            or checkpoint.optimizer.backend != "inkgrammar-spline-alt-lsq"
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
        topology_diagnostics = {}
        sample_count = objective.centerline.sample_schedule[-1]
        previous_selected_edges = {}

        def update(curve_template, current_states):
            targets = {}
            for item in raw_occurrences:
                state = current_states[item.key]
                transform = compose_occurrence_transform(item.part.transform, state.delta)
                candidate = sample_curve_by_arclength(
                    curve_template,
                    transform,
                    max(32, min(256, sample_count)),
                )
                candidate_xy = _model_to_image(candidate.xy, item.prepared.anchor)
                closed = np.linalg.norm(candidate_xy[0] - candidate_xy[-1]) <= max(
                    1.0, 2.0 * bundle.fit_config.render.stroke_width_px
                )
                expected_endpoints = None
                if closed:
                    path_kind = "loop"
                    expected_cycles = 1
                else:
                    path_kind = (
                        "open-with-loop"
                        if _has_polyline_self_intersection(candidate_xy)
                        else "open"
                    )
                    expected_cycles = 1 if path_kind == "open-with-loop" else 0
                target, graph, path = extract_observed_curve(
                    masks[item.key],
                    min_spur_length_px=objective.centerline.min_spur_length_px,
                    path_kind=path_kind,
                    expected_cycles=expected_cycles,
                    expected_endpoints=expected_endpoints,
                    candidate_xy=candidate_xy,
                    endpoint_tolerance_px=max(4.0, 2.0 * objective.assignment.sigma_px),
                    sample_count=max(32, min(256, sample_count)),
                )
                targets[item.key] = target
                selected_edges = tuple(path.edge_ids)
                previous_edges = previous_selected_edges.get(item.key)
                path_changed = previous_edges is not None and previous_edges != selected_edges
                previous_selected_edges[item.key] = selected_edges
                expected_self_intersections = int(_has_polyline_self_intersection(candidate_xy))
                observed_self_intersections = int(_has_polyline_self_intersection(path.xy))
                topology_diagnostics[item.key] = {
                    "path_kind": path_kind,
                    "expected_component_count": 1,
                    "expected_endpoint_count": 0 if path_kind == "loop" else 2,
                    "selected_endpoint_count": 0 if path.start_node == path.end_node else 2,
                    "expected_cycles": expected_cycles,
                    "observed_cycles": graph.cycle_count,
                    "expected_self_intersections": expected_self_intersections,
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

        return update, topology_diagnostics

    target_update, topology_diagnostics = target_builder(occurrences)
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
        update, split_topology = target_builder(raw)
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
        and item["selected_cycles"] == item["expected_cycles"]
        and item["selected_endpoint_count"] == item["expected_endpoint_count"]
        and item["observed_self_intersections"] == item["expected_self_intersections"]
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
    checkpoint_evaluations = fit.assignment_cycles * len(occurrences)
    if checkpoint is None:
        checkpoint = FitCheckpoint(
            project_id=bundle.project.id,
            stage=stage,
            created_by=CheckpointCreator(inkgrammar_version=__version__),
            inputs=input_hashes,
            optimizer=CheckpointOptimizer(
                backend="inkgrammar-spline-alt-lsq",
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
                    "backend": "inkgrammar-spline-alt-lsq",
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
            "backend": "inkgrammar-spline-alt-lsq",
            "strategy": "topology-aware alternating sparse spline fit",
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
    markdown_report_path.write_text("\n".join(markdown), encoding="utf-8")
    return FitRunArtifacts(
        checkpoint_path=checkpoint_path,
        json_report_path=json_report_path,
        markdown_report_path=markdown_report_path,
        report=report,
        private_review_sheet_path=review_sheet,
    )
