from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .alignment import bilinear_sample, normal_track_ink
from .centerline import TargetTrace
from .correspondence import MonotoneCorrespondence, monotone_correspondence
from .models import OccurrenceDelta


@dataclass(frozen=True)
class GeometricResiduals:
    residuals: np.ndarray
    path_to_ink: np.ndarray
    centerline: np.ndarray
    endpoint: np.ndarray
    tangent: np.ndarray
    diagnostics: dict[str, float]


def occurrence_geometric_residuals(
    path_points_xy: np.ndarray,
    reference_mask: np.ndarray,
    target_distance_field: np.ndarray,
    *,
    expected_stroke_width: float,
    sigma_distance: float = 2.0,
    sigma_center: float = 1.5,
    sigma_tangent: float = 0.25,
    path_weight: float = 1.0,
    center_weight: float = 1.5,
    endpoint_weight: float = 0.0,
    tangent_weight: float = 0.0,
) -> GeometricResiduals:
    points = np.asarray(path_points_xy, dtype=float)
    mask = np.asarray(reference_mask, dtype=float)
    distance_field = np.asarray(target_distance_field, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("path points must be a finite array with shape (N, 2)")
    if mask.ndim != 2 or distance_field.shape != mask.shape:
        raise ValueError("reference mask and distance field must be matching 2D arrays")
    if min(sigma_distance, sigma_center, sigma_tangent) <= 0:
        raise ValueError("residual sigmas must be positive")
    if min(path_weight, center_weight, endpoint_weight, tangent_weight) < 0:
        raise ValueError("residual weights must be nonnegative")

    outside_distance = float(np.hypot(*distance_field.shape))
    distances = bilinear_sample(distance_field, points, outside_value=outside_distance)
    path_block = np.sqrt(path_weight) * distances / sigma_distance
    track = normal_track_ink(
        mask,
        points,
        expected_stroke_width=expected_stroke_width,
    )
    centerline_errors = points - track.centerline_points
    center_block = np.zeros_like(points)
    interior = track.valid.copy()
    if len(points):
        interior[[0, -1]] = False
    center_block[interior] = (
        np.sqrt(center_weight * track.confidence[interior, None])
        * centerline_errors[interior]
        / sigma_center
    )

    endpoint_block = np.zeros((2, 2), dtype=float)
    if len(points):
        for block_index, point_index in enumerate((0, len(points) - 1)):
            if track.valid[point_index]:
                endpoint_block[block_index] = (
                    np.sqrt(endpoint_weight * track.confidence[point_index])
                    * centerline_errors[point_index]
                    / sigma_center
                )

    tangent_errors = np.zeros_like(points)
    tangent_block = np.zeros_like(points)
    if len(points) > 2:
        candidate_vectors = points[2:] - points[:-2]
        observed_vectors = track.centerline_points[2:] - track.centerline_points[:-2]
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        observed_norms = np.linalg.norm(observed_vectors, axis=1)
        tangent_valid = (
            track.valid[:-2]
            & track.valid[1:-1]
            & track.valid[2:]
            & (candidate_norms > 1e-9)
            & (observed_norms > 1e-9)
        )
        candidate_tangents = np.zeros_like(candidate_vectors)
        observed_tangents = np.zeros_like(observed_vectors)
        candidate_tangents[tangent_valid] = (
            candidate_vectors[tangent_valid] / candidate_norms[tangent_valid, None]
        )
        observed_tangents[tangent_valid] = (
            observed_vectors[tangent_valid] / observed_norms[tangent_valid, None]
        )
        confidence = np.minimum.reduce(
            (track.confidence[:-2], track.confidence[1:-1], track.confidence[2:])
        )
        tangent_errors[1:-1][tangent_valid] = (
            candidate_tangents[tangent_valid] - observed_tangents[tangent_valid]
        )
        tangent_block[1:-1][tangent_valid] = (
            np.sqrt(tangent_weight * confidence[tangent_valid, None])
            * tangent_errors[1:-1][tangent_valid]
            / sigma_tangent
        )

    residuals = np.concatenate(
        (path_block, center_block.ravel(), endpoint_block.ravel(), tangent_block.ravel())
    )
    valid_count = int(np.count_nonzero(track.valid))
    endpoint_valid = track.valid[[0, -1]] if len(points) else np.zeros(2, dtype=bool)
    endpoint_errors = (
        centerline_errors[[0, -1]][endpoint_valid] if len(points) else np.empty((0, 2))
    )
    diagnostics = {
        "path_to_ink_rms_px": float(np.sqrt(np.mean(distances**2))) if len(distances) else 0.0,
        "centerline_rms_px": (
            float(np.sqrt(np.mean(np.sum(centerline_errors[track.valid] ** 2, axis=1))))
            if valid_count
            else 0.0
        ),
        "centerline_valid_fraction": float(valid_count / len(points)) if len(points) else 0.0,
        "estimated_stroke_width_px": (
            float(np.mean(track.width_px[track.valid])) if valid_count else 0.0
        ),
        "endpoint_rms_px": (
            float(np.sqrt(np.mean(np.sum(endpoint_errors**2, axis=1))))
            if len(endpoint_errors)
            else 0.0
        ),
        "tangent_rms": (
            float(np.sqrt(np.mean(np.sum(tangent_errors**2, axis=1))))
            if len(tangent_errors)
            else 0.0
        ),
        "weighted_squared_cost": float(0.5 * np.dot(residuals, residuals)),
    }
    return GeometricResiduals(
        residuals, path_block, center_block, endpoint_block, tangent_block, diagnostics
    )


def transform_prior_residuals(
    delta: OccurrenceDelta,
    sigmas: Mapping[str, float] | None = None,
) -> np.ndarray:
    prior_sigmas = sigmas or {
        "dx": 2.0,
        "dy": 2.0,
        "log_scale_x": 0.06,
        "log_scale_y": 0.06,
        "shear_x": 0.04,
    }
    names = ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x")
    residuals = []
    for name in names:
        sigma = float(prior_sigmas[name])
        if sigma <= 0:
            raise ValueError(f"transform prior sigma must be positive: {name}")
        residuals.append(float(getattr(delta, name)) / sigma)
    return np.asarray(residuals, dtype=float)


@dataclass(frozen=True)
class OrderedCenterlineResiduals:
    residuals: np.ndarray
    correspondence: MonotoneCorrespondence
    candidate_to_target: np.ndarray
    target_to_candidate: np.ndarray
    endpoint: np.ndarray
    tangent: np.ndarray
    radius: np.ndarray
    diagnostics: dict[str, float]


def _unit_tangents(points: np.ndarray) -> np.ndarray:
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    return np.divide(
        tangents,
        norms[:, None],
        out=np.zeros_like(tangents),
        where=norms[:, None] > 1e-9,
    )


def _distances_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    segments = polyline[1:] - polyline[:-1]
    squared_lengths = np.sum(segments**2, axis=1)
    offsets = points[:, None, :] - polyline[None, :-1, :]
    fractions = np.divide(
        np.sum(offsets * segments[None, :, :], axis=2),
        squared_lengths[None, :],
        out=np.zeros((len(points), len(segments)), dtype=float),
        where=squared_lengths[None, :] > 1e-12,
    )
    projections = (
        polyline[None, :-1, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * segments[None, :, :]
    )
    return np.min(np.linalg.norm(points[:, None, :] - projections, axis=2), axis=1)


def ordered_centerline_residuals(
    candidate_xy: np.ndarray,
    target: TargetTrace,
    *,
    candidate_radius: np.ndarray | None = None,
    candidate_tangent: np.ndarray | None = None,
    sigma_position_px: float = 1.5,
    sigma_coverage_px: float = 1.5,
    sigma_endpoint_px: float = 1.0,
    sigma_tangent_rad: float = 0.2,
    sigma_radius_px: float = 1.5,
    candidate_to_target_weight: float = 1.0,
    target_to_candidate_weight: float = 1.0,
    endpoint_weight: float = 0.8,
    tangent_weight: float = 0.3,
    radius_weight: float = 0.1,
    coverage_tolerance_px: float = 2.0,
    supervise_endpoints: bool = True,
    open_target_ends: bool = True,
) -> OrderedCenterlineResiduals:
    candidate = np.asarray(candidate_xy, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) < 2:
        raise ValueError("candidate curve must have shape (N, 2) with N >= 2")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate curve coordinates must be finite")
    sigmas = (
        sigma_position_px,
        sigma_coverage_px,
        sigma_endpoint_px,
        sigma_tangent_rad,
        sigma_radius_px,
    )
    weights = (
        candidate_to_target_weight,
        target_to_candidate_weight,
        endpoint_weight,
        tangent_weight,
        radius_weight,
    )
    if not np.isfinite(sigmas).all() or min(sigmas) <= 0:
        raise ValueError("residual sigmas must be finite and positive")
    if not np.isfinite(weights).all() or min(weights) < 0:
        raise ValueError("residual weights must be finite and nonnegative")
    if not np.isfinite(coverage_tolerance_px) or coverage_tolerance_px <= 0:
        raise ValueError("coverage tolerance must be positive")

    correspondence = monotone_correspondence(
        candidate,
        target,
        candidate_tangent=candidate_tangent,
        candidate_radius=candidate_radius,
        sigma_position_px=sigma_position_px,
        sigma_tangent_rad=sigma_tangent_rad,
        sigma_radius_px=sigma_radius_px,
        tangent_weight=tangent_weight,
        radius_weight=radius_weight,
        open_target_ends=open_target_ends,
    )
    matched_target = target.xy[correspondence.target_index]
    matched_confidence = correspondence.confidence
    candidate_errors = candidate - matched_target
    candidate_block = (
        np.sqrt(candidate_to_target_weight * matched_confidence[:, None])
        * candidate_errors
        / sigma_position_px
    )

    valid_target = (target.confidence > 0) & np.isfinite(target.xy).all(axis=1)
    target_indices = np.flatnonzero(valid_target)
    target_points = target.xy[target_indices]
    target_confidence = target.confidence[target_indices]
    target_distances = _distances_to_polyline(target_points, candidate)
    coverage_block = (
        np.sqrt(target_to_candidate_weight * target_confidence)
        * target_distances
        / sigma_coverage_px
    )

    endpoint_block = np.zeros((2, 2), dtype=float)
    endpoint_errors = np.empty((0, 2), dtype=float)
    if supervise_endpoints and endpoint_weight > 0:
        target_ends = target_points[[0, -1]]
        endpoint_confidence = target_confidence[[0, -1]]
        endpoint_errors = candidate[[0, -1]] - target_ends
        endpoint_block = (
            np.sqrt(endpoint_weight * endpoint_confidence[:, None])
            * endpoint_errors
            / sigma_endpoint_px
        )

    candidate_tangents = (
        _unit_tangents(candidate)
        if candidate_tangent is None
        else np.asarray(candidate_tangent, dtype=float)
    )
    candidate_norms = np.linalg.norm(candidate_tangents, axis=1)
    candidate_tangents = np.divide(
        candidate_tangents,
        candidate_norms[:, None],
        out=np.zeros_like(candidate_tangents),
        where=candidate_norms[:, None] > 1e-9,
    )
    target_tangents = target.tangent[correspondence.target_index]
    target_norms = np.linalg.norm(target_tangents, axis=1)
    target_tangents = np.divide(
        target_tangents,
        target_norms[:, None],
        out=np.zeros_like(target_tangents),
        where=target_norms[:, None] > 1e-9,
    )
    tangent_valid = (candidate_norms > 1e-9) & (target_norms > 1e-9)
    tangent_angles = np.zeros(len(candidate), dtype=float)
    tangent_angles[tangent_valid] = np.arccos(
        np.clip(
            np.sum(candidate_tangents[tangent_valid] * target_tangents[tangent_valid], axis=1),
            -1.0,
            1.0,
        )
    )
    tangent_block = np.zeros(len(candidate), dtype=float)
    tangent_block[tangent_valid] = (
        np.sqrt(tangent_weight * matched_confidence[tangent_valid])
        * tangent_angles[tangent_valid]
        / sigma_tangent_rad
    )

    radius_block = np.zeros(len(candidate), dtype=float)
    radius_errors = np.empty(0, dtype=float)
    if candidate_radius is not None:
        candidate_radius = np.asarray(candidate_radius, dtype=float)
        radius_errors = candidate_radius - target.radius[correspondence.target_index]
        radius_block = np.sqrt(radius_weight * matched_confidence) * radius_errors / sigma_radius_px

    residuals = np.concatenate(
        (
            candidate_block.ravel(),
            coverage_block,
            endpoint_block.ravel(),
            tangent_block,
            radius_block,
        )
    )
    coverage_total = float(np.sum(target_confidence))
    assigned_coverage = (
        float(np.sum(target_confidence[target_distances <= coverage_tolerance_px]) / coverage_total)
        if coverage_total > 0
        else 0.0
    )
    candidate_distances = np.linalg.norm(candidate_errors, axis=1)
    diagnostics = {
        "candidate_to_target_rms_px": float(np.sqrt(np.mean(candidate_distances**2))),
        "candidate_to_target_p95_px": float(np.percentile(candidate_distances, 95)),
        "target_to_candidate_rms_px": float(
            np.sqrt(np.average(target_distances**2, weights=target_confidence))
        ),
        "target_to_candidate_p95_px": float(np.percentile(target_distances, 95)),
        "endpoint_rms_px": (
            float(np.sqrt(np.mean(np.sum(endpoint_errors**2, axis=1))))
            if len(endpoint_errors)
            else 0.0
        ),
        "endpoint_supervised": float(bool(supervise_endpoints and endpoint_weight > 0)),
        "tangent_rms_rad": (
            float(np.sqrt(np.mean(tangent_angles[tangent_valid] ** 2)))
            if np.any(tangent_valid)
            else 0.0
        ),
        "tangent_valid_fraction": float(np.mean(tangent_valid)),
        "radius_rms_px": (float(np.sqrt(np.mean(radius_errors**2))) if len(radius_errors) else 0.0),
        "assigned_coverage": assigned_coverage,
        "correspondence_cost": correspondence.cost,
    }
    return OrderedCenterlineResiduals(
        residuals=residuals,
        correspondence=correspondence,
        candidate_to_target=candidate_block,
        target_to_candidate=coverage_block,
        endpoint=endpoint_block,
        tangent=tangent_block,
        radius=radius_block,
        diagnostics=diagnostics,
    )
