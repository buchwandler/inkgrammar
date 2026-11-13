"""InkGrammar adapters for planarcurve trajectory geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from planarcurve.bezier import cubic_length, split_cubic
from planarcurve.trajectory import (
    evaluate_heading as _evaluate_heading,
)
from planarcurve.trajectory import (
    evaluate_heading_rate as _evaluate_heading_rate,
)
from planarcurve.trajectory import (
    evaluate_trajectory as _evaluate_trajectory,
)
from planarcurve.trajectory import (
    evaluate_trajectory_at as _evaluate_trajectory_at,
)
from planarcurve.trajectory import (
    sample_trajectory_by_arclength,
    trajectory_to_cubic_path,
)
from planarcurve.trajectory import (
    trajectory_total_turn as _trajectory_total_turn,
)
from planarcurve.types import CubicBezier

from .models import CubicSegment, CurvatureProfile, CurvatureStrokeGeometry
from .planarcurve_adapters import to_planar_heading_rate, to_planar_trajectory


@dataclass(frozen=True)
class StrokePose:
    position: np.ndarray
    heading_deg: float
    heading_rate_deg: float


@dataclass(frozen=True)
class CurvatureStrokeSample:
    u: np.ndarray
    xy: np.ndarray
    heading_deg: np.ndarray
    heading_rate_deg: np.ndarray
    tangent: np.ndarray
    arc: np.ndarray
    end_pose: StrokePose


def evaluate_heading_rate(curvature: CurvatureProfile, u: np.ndarray) -> np.ndarray:
    """Evaluate normalized heading rate through the planarcurve kernel."""
    return _evaluate_heading_rate(to_planar_heading_rate(curvature), u)


def evaluate_heading(geometry: CurvatureStrokeGeometry, u: np.ndarray) -> np.ndarray:
    """Compatibility wrapper requiring the historical ascending grid from zero."""
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("u must be a non-empty 1-D array")
    if values[0] != 0.0:
        raise ValueError("u must start at 0.0")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("u must lie in [0, 1]")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("u must be ascending")
    return _evaluate_heading(to_planar_trajectory(geometry), values)


def evaluate_curvature_stroke(
    geometry: CurvatureStrokeGeometry, *, count: int = 256
) -> CurvatureStrokeSample:
    sample = _evaluate_trajectory(to_planar_trajectory(geometry), count=count)
    end_pose = StrokePose(
        position=sample.end_pose.position,
        heading_deg=sample.end_pose.heading_deg,
        heading_rate_deg=sample.end_pose.heading_rate_deg_per_u,
    )
    return CurvatureStrokeSample(
        u=sample.u,
        xy=sample.xy,
        heading_deg=sample.heading_deg,
        heading_rate_deg=sample.heading_rate_deg_per_u,
        tangent=sample.tangent,
        arc=sample.arc_length,
        end_pose=end_pose,
    )


def trajectory_end_pose(
    geometry: CurvatureStrokeGeometry, *, count: int = 256
) -> StrokePose:
    return evaluate_curvature_stroke(geometry, count=count).end_pose


def trajectory_total_turn(
    geometry: CurvatureStrokeGeometry, *, count: int = 256
) -> float:
    if isinstance(count, bool) or count < 2:
        raise ValueError("count must be at least 2")
    return _trajectory_total_turn(to_planar_trajectory(geometry))


def sample_curvature_stroke(
    geometry: CurvatureStrokeGeometry, *, count: int = 256, samples: int = 64
) -> np.ndarray:
    return sample_trajectory_by_arclength(
        to_planar_trajectory(geometry), samples, integration_count=count
    )


def evaluate_stroke_at(
    geometry: CurvatureStrokeGeometry, u: np.ndarray, *, base_count: int = 257
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("u must be a non-empty 1-D array")
    if (
        not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or np.any(np.diff(values) < 0.0)
    ):
        raise ValueError("u must be ascending inside [0, 1]")
    return _evaluate_trajectory_at(
        to_planar_trajectory(geometry), values, base_count=base_count
    )


def curvature_to_cubic_segments(
    geometry: CurvatureStrokeGeometry,
    *,
    max_u_step: float = 0.05,
    tolerance: float = 1e-4,
) -> list[CubicSegment]:
    """Adapt planarcurve's tolerance-driven path to InkGrammar segment models."""
    if not np.isfinite(max_u_step) or max_u_step <= 0.0:
        raise ValueError("max_u_step must be positive and finite")
    path = trajectory_to_cubic_path(
        to_planar_trajectory(geometry), max_error=tolerance
    )
    segments = []
    start = np.asarray(path.start, dtype=float)

    def append_with_step_limit(curve: CubicBezier, depth: int) -> None:
        length = cubic_length(curve, tolerance=max(tolerance * 0.1, 1e-12))
        if length / geometry.length <= max_u_step:
            segments.append(CubicSegment(c1=curve.p1, c2=curve.p2, end=curve.p3))
            return
        if depth == 0:
            raise ValueError("cubic conversion could not meet max_u_step")
        left, right = split_cubic(curve)
        append_with_step_limit(left, depth - 1)
        append_with_step_limit(right, depth - 1)

    for segment in path.segments:
        append_with_step_limit(
            CubicBezier(start, segment.c1, segment.c2, segment.end), 16
        )
        start = np.asarray(segment.end, dtype=float)
    return segments
