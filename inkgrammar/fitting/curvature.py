"""Pure curvature-trajectory kernel (the brief's sections 2 and 7).

A pen-down stroke is described by a start angle, a normalized length, and a
heading-rate profile ``q(u) = d(theta_deg)/du`` over normalized stroke
progress ``u in [0, 1]``::

    theta(u) = theta0 + integral_0^u q(v) dv
    dx/du = L cos(theta(u)),  dy/du = L sin(theta(u))

``q(u)`` is scale-invariant under uniform scaling, so a small and a large
letter share one profile and differ only in length. This module contains no
optimizer and performs no file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CubicSegment, CurvatureProfile, CurvatureStrokeGeometry


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


def _profile_knots(profile: CurvatureProfile) -> tuple[np.ndarray, np.ndarray]:
    positions = [0.0, *(change.at for change in profile.changes), 1.0]
    rates = [profile.start_rate_deg, *(change.rate_deg for change in profile.changes)]
    rates.append(profile.end_rate_deg)
    return np.asarray(positions, dtype=float), np.asarray(rates, dtype=float)


def _hermite_slopes(knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Shape-preserving (PCHIP) slopes: no overshoot between keyframes."""
    deltas = np.diff(values) / np.diff(knots)
    slopes = np.zeros(knots.size)
    if knots.size == 2:
        slopes[:] = deltas[0]
        return slopes
    for index in range(1, knots.size - 1):
        if deltas[index - 1] * deltas[index] <= 0:
            slopes[index] = 0.0
        else:
            weight_left = 2 * (knots[index + 1] - knots[index]) + (
                knots[index] - knots[index - 1]
            )
            weight_right = (knots[index + 1] - knots[index]) + 2 * (
                knots[index] - knots[index - 1]
            )
            slopes[index] = (weight_left + weight_right) / (
                weight_left / deltas[index - 1] + weight_right / deltas[index]
            )
    slopes[0] = _endpoint_slope(knots, deltas, 0)
    slopes[-1] = _endpoint_slope(knots, deltas, knots.size - 1)
    return slopes


def _endpoint_slope(knots: np.ndarray, deltas: np.ndarray, index: int) -> float:
    if knots.size == 2:
        return float(deltas[0])
    if index == 0:
        h0, h1 = knots[1] - knots[0], knots[2] - knots[1]
        d0, d1 = deltas[0], deltas[1]
    else:
        h0, h1 = knots[-1] - knots[-2], knots[-2] - knots[-3]
        d0, d1 = deltas[-1], deltas[-2]
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0:
        return 0.0
    if d0 * d1 <= 0 and abs(slope) > abs(3 * d0):
        return float(3 * d0)
    return float(slope)


def evaluate_heading_rate(curvature: CurvatureProfile, u: np.ndarray) -> np.ndarray:
    """Heading rate q(u) in degrees per unit normalized progress."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("u must be a 1-D array")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("u must lie in [0, 1]")
    knots, values = _profile_knots(curvature)
    if curvature.interpolation == "linear":
        return np.interp(u, knots, values)
    slopes = _hermite_slopes(knots, values)
    indices = np.clip(np.searchsorted(knots, u, side="right") - 1, 0, knots.size - 2)
    h = knots[indices + 1] - knots[indices]
    t = (u - knots[indices]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (
        h00 * values[indices]
        + h10 * h * slopes[indices]
        + h01 * values[indices + 1]
        + h11 * h * slopes[indices + 1]
    )


def evaluate_heading(geometry: CurvatureStrokeGeometry, u: np.ndarray) -> np.ndarray:
    """Heading theta(u) in degrees on an ascending grid with ``u[0] == 0``."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size == 0:
        raise ValueError("u must be a non-empty 1-D array")
    if u[0] != 0.0:
        raise ValueError("u must start at 0.0")
    if np.any(np.diff(u) < 0):
        raise ValueError("u must be ascending")
    rates = evaluate_heading_rate(geometry.curvature, u)
    heading = np.empty_like(u)
    heading[0] = geometry.start_angle_deg
    if u.size > 1:
        increments = 0.5 * (rates[:-1] + rates[1:]) * np.diff(u)
        heading[1:] = geometry.start_angle_deg + np.cumsum(increments)
    return heading


def evaluate_curvature_stroke(
    geometry: CurvatureStrokeGeometry, *, count: int = 256
) -> CurvatureStrokeSample:
    """Integrate the trajectory on a uniform ``u`` grid (trapezoid rule)."""
    if count < 2:
        raise ValueError("count must be at least 2")
    u = np.linspace(0.0, 1.0, count)
    rates = evaluate_heading_rate(geometry.curvature, u)
    heading_deg = evaluate_heading(geometry, u)
    heading = np.deg2rad(heading_deg)
    tangent = np.column_stack((np.cos(heading), np.sin(heading)))
    steps = np.diff(u)
    xy = np.zeros((count, 2), dtype=float)
    xy[1:] = geometry.length * np.cumsum(
        0.5 * (tangent[:-1] + tangent[1:]) * steps[:, None],
        axis=0,
    )
    arc = np.zeros(count, dtype=float)
    arc[1:] = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    end_pose = StrokePose(
        position=xy[-1].copy(),
        heading_deg=float(heading_deg[-1]),
        heading_rate_deg=float(rates[-1]),
    )
    return CurvatureStrokeSample(
        u=u,
        xy=xy,
        heading_deg=heading_deg,
        heading_rate_deg=rates,
        tangent=tangent,
        arc=arc,
        end_pose=end_pose,
    )


def trajectory_end_pose(
    geometry: CurvatureStrokeGeometry, *, count: int = 256
) -> StrokePose:
    """End position, heading, and heading rate of the integrated stroke."""
    return evaluate_curvature_stroke(geometry, count=count).end_pose


def trajectory_total_turn(
    geometry: CurvatureStrokeGeometry, *, count: int = 256
) -> float:
    """Total heading change in degrees: integral of q over [0, 1]."""
    u = np.linspace(0.0, 1.0, count)
    rates = evaluate_heading_rate(geometry.curvature, u)
    return float(np.sum(0.5 * (rates[:-1] + rates[1:]) * np.diff(u)))


def sample_curvature_stroke(
    geometry: CurvatureStrokeGeometry, *, count: int = 256, samples: int = 64
) -> np.ndarray:
    """Stroke points resampled uniformly by arc length (shape ``samples x 2``)."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    sample = evaluate_curvature_stroke(geometry, count=count)
    targets = np.linspace(0.0, sample.arc[-1], samples)
    return np.column_stack(
        (np.interp(targets, sample.arc, sample.xy[:, 0]),
         np.interp(targets, sample.arc, sample.xy[:, 1]))
    )

def evaluate_stroke_at(
    geometry: CurvatureStrokeGeometry, u: np.ndarray, *, base_count: int = 257
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and headings (deg) at requested ``u`` values.

    Integration runs on the union of ``u`` and a fine uniform quadrature
    grid, so the returned states match :func:`evaluate_curvature_stroke` up
    to trapezoid error.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size == 0:
        raise ValueError("u must be a non-empty 1-D array")
    if u[0] < 0.0 or u[-1] > 1.0 or np.any(np.diff(u) < 0):
        raise ValueError("u must be ascending inside [0, 1]")
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, base_count), u]))
    rates = evaluate_heading_rate(geometry.curvature, grid)
    heading = np.empty_like(grid)
    heading[0] = geometry.start_angle_deg
    heading[1:] = geometry.start_angle_deg + np.cumsum(
        0.5 * (rates[:-1] + rates[1:]) * np.diff(grid)
    )
    tangent = np.column_stack(
        (np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading)))
    )
    xy = np.zeros((grid.size, 2), dtype=float)
    xy[1:] = geometry.length * np.cumsum(
        0.5 * (tangent[:-1] + tangent[1:]) * np.diff(grid)[:, None],
        axis=0,
    )
    indices = np.searchsorted(grid, u)
    return xy[indices], heading[indices]


def curvature_to_cubic_segments(
    geometry: CurvatureStrokeGeometry, *, max_u_step: float = 0.05
) -> list[CubicSegment]:
    """Approximate the integrated trajectory with cubic Bezier segments.

    Segments break at curvature-change locations and at most ``max_u_step``
    apart. Endpoints and tangents match the integrated stroke (Hermite to
    Bezier), so the compiled cubic path agrees with the direct integration
    within a strict tolerance.
    """
    if max_u_step <= 0:
        raise ValueError("max_u_step must be positive")
    breaks = [0.0, *(change.at for change in geometry.curvature.changes), 1.0]
    nodes = [0.0]
    for low, high in zip(breaks, breaks[1:], strict=False):
        steps = max(1, int(np.ceil((high - low) / max_u_step)))
        nodes.extend(low + (high - low) * step / steps for step in range(1, steps + 1))
    u_nodes = np.asarray(nodes)
    xy, heading_deg = evaluate_stroke_at(geometry, u_nodes)
    velocity = (
        np.column_stack(
            (np.cos(np.deg2rad(heading_deg)), np.sin(np.deg2rad(heading_deg)))
        )
        * geometry.length
    )
    segments = []
    for index in range(u_nodes.size - 1):
        step = u_nodes[index + 1] - u_nodes[index]
        start_point = xy[index]
        end_point = xy[index + 1]
        start_tangent = velocity[index] * step
        end_tangent = velocity[index + 1] * step
        segments.append(
            CubicSegment(
                c1=(
                    float(start_point[0] + start_tangent[0] / 3),
                    float(start_point[1] + start_tangent[1] / 3),
                ),
                c2=(
                    float(end_point[0] - end_tangent[0] / 3),
                    float(end_point[1] - end_tangent[1] / 3),
                ),
                end=(float(end_point[0]), float(end_point[1])),
            )
        )
    return segments
