from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from planarcurve.polyline import unit_tangents


@dataclass(frozen=True)
class DifferentialGeometry:
    smoothed_xy: np.ndarray
    heading_rad: np.ndarray
    turn_rate_rad_per_progress: np.ndarray
    cumulative_turn_rad: np.ndarray


def differential_geometry(
    points_xy: np.ndarray,
    *,
    window_length: int = 11,
    polynomial_order: int = 2,
) -> DifferentialGeometry:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("path must have shape (N, 2) with N >= 3")
    if not np.isfinite(points).all():
        raise ValueError("path coordinates must be finite")
    if window_length < 3 or window_length % 2 == 0 or window_length > len(points):
        raise ValueError(
            "smoothing window must be odd, at least three, and no longer than the path"
        )
    if polynomial_order < 1 or polynomial_order >= window_length:
        raise ValueError("polynomial order must be positive and less than the smoothing window")

    from scipy.signal import savgol_filter
    smoothed = savgol_filter(points, window_length, polynomial_order, axis=0, mode="interp")
    tangents = unit_tangents(smoothed)
    if not np.isfinite(tangents).all() or np.any(np.linalg.norm(tangents, axis=1) <= 1e-12):
        raise ValueError("smoothed path contains an undefined tangent")
    heading = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    progress = np.linspace(0.0, 1.0, len(points))
    turn_rate = np.gradient(heading, progress)
    cumulative_turn = heading - heading[0]
    return DifferentialGeometry(smoothed, heading, turn_rate, cumulative_turn)
