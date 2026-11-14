from __future__ import annotations

from planarcurve.svg import PathInfo, analyze_path, cubic_bridge, path_without_initial_move
from planarcurve.svg import sample_path as _sample_path
from planarcurve.svg import transform_path as _transform_path
from planarcurve.transforms import Affine2D

from .models import Transform

__all__ = [
    "PathInfo",
    "Point",
    "analyze_path",
    "cubic_bridge",
    "path_without_initial_move",
    "sample_path",
    "transform_path",
    "translate_path",
]

Point = tuple[float, float]


def sample_path(d: str, samples_per_curve: int = 24) -> list[Point]:
    return [tuple(map(float, point)) for point in _sample_path(d, samples_per_curve)]


def _as_affine(transform: Transform) -> Affine2D:
    return Affine2D(
        transform.scale_x,
        transform.shear_x * transform.scale_y,
        0.0,
        transform.scale_y,
        transform.dx,
        transform.dy,
    )


def transform_path(d: str, transform: Transform) -> str:
    return _transform_path(d, _as_affine(transform))


def translate_path(d: str, dx: float, dy: float) -> str:
    return _transform_path(d, Affine2D.translation_only(dx, dy))
