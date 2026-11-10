from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .alignment import bilinear_sample, normal_track_ink
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
