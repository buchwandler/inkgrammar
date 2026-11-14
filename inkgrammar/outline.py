from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import strokeoutline
from planarcurve import CubicPath
from strokeoutline import combine_outlines, outline, outline_disc, signed_area
from strokeoutline.svg import outline_svg_path

from .geometry import Point
from .shaping import ShapedText

WidthProfile = strokeoutline.WidthProfile
Contour = strokeoutline.Contour
ExpandedOutline = strokeoutline.OutlineResult


@dataclass(frozen=True)
class OutlineConfig:
    max_centerline_error: float = 0.25
    max_width_error: float = 0.25
    max_normal_angle_degrees: float = 3.0
    round_caps: bool = True
    max_subdivision_depth: int = 18

    def __post_init__(self) -> None:
        tolerances = (
            self.max_centerline_error,
            self.max_width_error,
            self.max_normal_angle_degrees,
        )
        if any(not isfinite(value) or value <= 0 for value in tolerances):
            raise ValueError("outline tolerances must be positive")
        if self.max_normal_angle_degrees >= 180:
            raise ValueError("normal angle tolerance must be less than 180 degrees")
        if self.max_subdivision_depth < 1:
            raise ValueError("maximum subdivision depth must be positive")

    def to_strokeoutline(self) -> strokeoutline.OutlineConfig:
        return strokeoutline.OutlineConfig(
            max_centerline_error=self.max_centerline_error,
            max_width_error=self.max_width_error,
            max_normal_angle_degrees=self.max_normal_angle_degrees,
            cap="round" if self.round_caps else "square",
            max_subdivision_depth=self.max_subdivision_depth,
        )


DEFAULT_OUTLINE_CONFIG = OutlineConfig()
_signed_area = signed_area


def expand_path(
    d: str | CubicPath,
    width: WidthProfile,
    *,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> ExpandedOutline:
    """Expand an SVG path or typed PlanarCurve path through StrokeOutline."""
    if isinstance(d, CubicPath):
        return outline(d, width, config=config.to_strokeoutline())
    return outline_svg_path(d, width, config=config.to_strokeoutline())


def expand_tap(
    point: Point,
    width: WidthProfile,
    *,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> ExpandedOutline:
    """Expand a point stroke through StrokeOutline's disc implementation."""
    return outline_disc(point, width, config=config.to_strokeoutline())


def expand_shaped_text(
    shaped: ShapedText,
    default_width: float,
    *,
    width_profiles: dict[str, WidthProfile] | None = None,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> ExpandedOutline:
    """Apply InkGrammar's per-stroke width policy using StrokeOutline geometry."""
    profiles = width_profiles or {}
    outlines = []
    for stroke in shaped.strokes:
        width = profiles.get(stroke.id, default_width)
        if stroke.d is not None:
            outlines.append(expand_path(stroke.d, width, config=config))
        elif stroke.tap is not None:
            outlines.append(expand_tap(stroke.tap, width, config=config))
    return combine_outlines(outlines)
