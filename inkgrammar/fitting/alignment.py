from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import sample_path
from .models import OccurrenceDelta, TopologyTransform, compose_occurrence_transform


@dataclass(frozen=True)
class NormalTrack:
    path_points: np.ndarray
    centerline_points: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    width_px: np.ndarray


def sample_path_by_arclength(
    path: str,
    sample_count: int = 32,
    *,
    samples_per_curve: int = 96,
) -> np.ndarray:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")
    points = np.asarray(sample_path(path, samples_per_curve=samples_per_curve), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must contain at least two two-dimensional samples")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 0:
        raise ValueError("path must have positive arc length")
    targets = np.linspace(0.0, cumulative[-1], sample_count)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, dimension]) for dimension in range(2)]
    )


def path_points_in_image(
    path: str,
    origin: tuple[float, float],
    sample_count: int = 32,
) -> np.ndarray:
    points = sample_path_by_arclength(path, sample_count)
    points[:, 0] += origin[0]
    points[:, 1] = origin[1] - points[:, 1]
    return points


def bilinear_sample(
    field: np.ndarray,
    points_xy: np.ndarray,
    *,
    outside_value: float = 0.0,
) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    points = np.asarray(points_xy, dtype=float)
    if values.ndim != 2:
        raise ValueError("sampled field must be two-dimensional")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("sample coordinates must be a finite array with shape (N, 2)")
    if not np.isfinite(values).all():
        raise ValueError("sampled field must contain finite values")
    height, width = values.shape
    x, y = points[:, 0], points[:, 1]
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    clipped_x = np.clip(x, 0, width - 1)
    clipped_y = np.clip(y, 0, height - 1)
    x0, y0 = np.floor(clipped_x).astype(int), np.floor(clipped_y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = clipped_x - x0, clipped_y - y0
    sampled = (
        values[y0, x0] * (1 - fx) * (1 - fy)
        + values[y0, x1] * fx * (1 - fy)
        + values[y1, x0] * (1 - fx) * fy
        + values[y1, x1] * fx * fy
    )
    return np.where(inside, sampled, outside_value)


def normal_track_ink(
    reference_mask: np.ndarray,
    path_points_xy: np.ndarray,
    *,
    expected_stroke_width: float,
    search_scale: float = 1.5,
    sample_step: float = 0.5,
    ink_threshold: float = 0.25,
    ambiguity_tolerance: float = 1.0,
) -> NormalTrack:
    mask = np.asarray(reference_mask, dtype=float)
    points = np.asarray(path_points_xy, dtype=float)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("reference mask must be a finite two-dimensional array")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("path points must be a finite array with shape (N, 2)")
    if expected_stroke_width <= 0 or search_scale <= 0 or sample_step <= 0:
        raise ValueError("stroke width, search scale, and sample step must be positive")
    if not 0 <= ink_threshold <= 1 or ambiguity_tolerance < 0:
        raise ValueError("ink threshold and ambiguity tolerance are invalid")

    centers = np.zeros_like(points)
    valid = np.zeros(len(points), dtype=bool)
    confidence = np.zeros(len(points), dtype=float)
    widths = np.zeros(len(points), dtype=float)
    radius = expected_stroke_width * search_scale
    offsets = np.arange(-radius, radius + sample_step * 0.5, sample_step)
    tangents = np.gradient(points, axis=0)

    for index, (point, tangent) in enumerate(zip(points, tangents, strict=True)):
        length = float(np.linalg.norm(tangent))
        if length <= 1e-12:
            continue
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        probes = point + offsets[:, None] * normal
        coverage = bilinear_sample(mask, probes)
        ink_indices = np.flatnonzero(coverage >= ink_threshold)
        if not len(ink_indices):
            continue
        runs = np.split(ink_indices, np.flatnonzero(np.diff(ink_indices) > 1) + 1)
        candidates = []
        for run in runs:
            start, end = offsets[run[0]], offsets[run[-1]]
            distance = 0.0 if start <= 0 <= end else min(abs(start), abs(end))
            candidates.append((distance, run))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= ambiguity_tolerance:
            continue
        distance, run = candidates[0]
        weights = coverage[run]
        center_offset = float(np.average(offsets[run], weights=weights))
        centers[index] = point + center_offset * normal
        valid[index] = True
        confidence[index] = float(np.max(weights) * max(0.0, 1.0 - distance / radius))
        widths[index] = float(offsets[run[-1]] - offsets[run[0]] + sample_step)

    return NormalTrack(points.copy(), centers, valid, confidence, widths)


def effective_occurrence_transform(
    nominal: TopologyTransform, delta: OccurrenceDelta
) -> TopologyTransform:
    return compose_occurrence_transform(nominal, delta)
