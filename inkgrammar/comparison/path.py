from __future__ import annotations

import numpy as np
from planarcurve.polyline import polyline_length, resample_polyline_by_arclength
from planarcurve.transforms import Similarity2D

from ..skeleton_walk import _euler_path, _pixel_neighbors
from .models import AlignmentResult, OrderedPath, SkeletonDiagnostics


class PathExtractionError(ValueError):
    """Raised when the skeleton cannot provide a complete ordered path."""


def recover_ordered_path(diagnostics: SkeletonDiagnostics) -> OrderedPath:
    skeleton = diagnostics.skeleton
    pixels = {tuple(int(value) for value in point) for point in np.argwhere(skeleton)}
    adjacency = {point: _pixel_neighbors(pixels, point) for point in pixels}
    edges = {
        frozenset((point, neighbor))
        for point, neighbors in adjacency.items()
        for neighbor in neighbors
        if point < neighbor
    }
    if not edges:
        raise PathExtractionError("skeleton has no traversable edges")

    walk_yx = _euler_path(pixels, adjacency)
    if walk_yx is None or len(walk_yx) < 2:
        raise PathExtractionError("skeleton did not produce an ordered path")
    traversed = {
        frozenset((first, second))
        for first, second in zip(walk_yx, walk_yx[1:], strict=False)
    }
    covered = len(edges & traversed)
    if covered != len(edges):
        raise PathExtractionError(
            f"ordered walk covers {covered} of {len(edges)} skeleton edges"
        )

    xy = np.asarray([(x, y) for y, x in walk_yx], dtype=float)
    if not np.isfinite(xy).all() or polyline_length(xy) <= 0.0:
        raise PathExtractionError("ordered skeleton path has no finite positive arc length")
    return OrderedPath(xy, covered, len(edges))


def resample_path(points_xy: np.ndarray, sample_count: int = 256) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must have shape (N, 2) with N >= 2")
    if not np.isfinite(points).all():
        raise ValueError("path coordinates must be finite")
    if sample_count < 2:
        raise ValueError("sample count must be at least two")
    if polyline_length(points) <= 0.0:
        raise ValueError("path must have positive arc length")
    return resample_polyline_by_arclength(points, sample_count)


def fit_similarity_transform(reference_xy: np.ndarray, sample_xy: np.ndarray) -> Similarity2D:
    reference = np.asarray(reference_xy, dtype=float)
    sample = np.asarray(sample_xy, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1] != 2
        or sample.shape != reference.shape
        or len(reference) < 2
    ):
        raise ValueError("reference and sample paths must share shape (N, 2), with N >= 2")
    if not np.isfinite(reference).all() or not np.isfinite(sample).all():
        raise ValueError("alignment paths must contain finite coordinates")

    reference_mean = reference.mean(axis=0)
    sample_mean = sample.mean(axis=0)
    reference_centered = reference - reference_mean
    sample_centered = sample - sample_mean
    sample_energy = float(np.sum(sample_centered**2))
    if sample_energy <= 1e-12 or float(np.sum(reference_centered**2)) <= 1e-12:
        raise ValueError("alignment paths must have nonzero spatial extent")

    left, _, right_t = np.linalg.svd(sample_centered.T @ reference_centered)
    correction = np.eye(2)
    correction[-1, -1] = 1.0 if np.linalg.det(left @ right_t) >= 0.0 else -1.0
    row_rotation = left @ correction @ right_t
    scale = float(np.sum((sample_centered @ row_rotation) * reference_centered) / sample_energy)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("paths do not define a positive-scale similarity transform")
    translation = reference_mean - scale * (sample_mean @ row_rotation)
    rotation_deg = float(np.degrees(np.arctan2(row_rotation[0, 1], row_rotation[0, 0])))
    return Similarity2D(
        scale=scale,
        rotation_deg=rotation_deg,
        dx=float(translation[0]),
        dy=float(translation[1]),
    )


def align_ordered_paths(reference_xy: np.ndarray, sample_xy: np.ndarray) -> AlignmentResult:
    reference = np.asarray(reference_xy, dtype=float)
    sample = np.asarray(sample_xy, dtype=float)
    if reference.shape != sample.shape:
        raise ValueError("reference and sample paths must have matching sample counts")

    candidates = []
    for direction, oriented_sample in (
        ("forward", sample),
        ("reversed-for-best-match", sample[::-1]),
    ):
        transform = fit_similarity_transform(reference, oriented_sample)
        aligned = transform.apply(oriented_sample)
        error = float(np.mean(np.sum((reference - aligned) ** 2, axis=1)))
        candidates.append((error, direction, transform, aligned))
    error, direction, transform, aligned = min(candidates, key=lambda candidate: candidate[0])
    return AlignmentResult(transform, direction, aligned, error)
