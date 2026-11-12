"""Explicit coordinate frames between source pixels and glyph units.

Glyph units are x-height normalized: baseline ``y = 0``, x-height ``y = 1``,
positive y upward. Source pixels have positive y downward. The frame makes the
conversion explicit so pixel and font units can never mix implicitly.
"""

from __future__ import annotations

import numpy as np
from pydantic import model_validator

from ..models import StrictModel


class ObservationFrame(StrictModel):
    """Affine frame: glyph (0, 0) at ``origin_source_px``; one glyph unit is
    ``x_height_px`` source pixels along the axis directions.

    ``x_axis_source_px`` and ``y_axis_source_px`` are direction vectors; they
    must not be collinear. For the standard image convention use
    :func:`standard_frame` (x right, y up against image y down).
    """

    origin_source_px: tuple[float, float]
    x_axis_source_px: tuple[float, float]
    y_axis_source_px: tuple[float, float]
    x_height_px: float

    @model_validator(mode="after")
    def validate_frame(self) -> ObservationFrame:
        if self.x_height_px <= 0:
            raise ValueError("x_height_px must be positive")
        x_axis = np.asarray(self.x_axis_source_px)
        y_axis = np.asarray(self.y_axis_source_px)
        if not x_axis.any() or not y_axis.any():
            raise ValueError("frame axes must be non-zero vectors")
        cross = float(x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0])
        if abs(cross) < 1e-9:
            raise ValueError("frame axes must not be collinear")
        return self


def standard_frame(origin_source_px: tuple[float, float], x_height_px: float) -> ObservationFrame:
    """Frame with glyph x rightward and glyph y upward (image y down)."""
    return ObservationFrame(
        origin_source_px=origin_source_px,
        x_axis_source_px=(1.0, 0.0),
        y_axis_source_px=(0.0, -1.0),
        x_height_px=x_height_px,
    )


def _frame_matrix(frame: ObservationFrame) -> np.ndarray:
    return np.column_stack(
        (
            np.asarray(frame.x_axis_source_px) * frame.x_height_px,
            np.asarray(frame.y_axis_source_px) * frame.x_height_px,
        )
    )


def source_px_to_glyph_units(points: np.ndarray, frame: ObservationFrame) -> np.ndarray:
    """Map source pixel points (shape ``(..., 2)``) to x-height glyph units."""
    points = np.asarray(points, dtype=float)
    inverse = np.linalg.inv(_frame_matrix(frame))
    return (points - np.asarray(frame.origin_source_px)) @ inverse.T


def glyph_units_to_source_px(points: np.ndarray, frame: ObservationFrame) -> np.ndarray:
    """Map x-height glyph unit points (shape ``(..., 2)``) to source pixels."""
    points = np.asarray(points, dtype=float)
    return points @ _frame_matrix(frame).T + np.asarray(frame.origin_source_px)
