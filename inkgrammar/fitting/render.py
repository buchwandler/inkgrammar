from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from ..geometry import sample_path
from .trace import CompiledPartTrace


@dataclass(frozen=True)
class FittingRender:
    coverage: np.ndarray
    part_layers: dict[str, np.ndarray]
    stroke_layers: dict[str, np.ndarray]


def render_paths(
    paths: Mapping[str, str],
    *,
    canvas_size: tuple[int, int],
    origin: tuple[float, float],
    pixels_per_unit: float,
    stroke_width: float,
    supersampling: int = 4,
    stroke_ids: Mapping[str, str] | None = None,
) -> FittingRender:
    """Render model paths into fixed-frame grayscale ink coverage layers."""
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("canvas dimensions must be positive")
    if pixels_per_unit <= 0 or stroke_width <= 0:
        raise ValueError("scale and stroke width must be positive")
    if supersampling < 1:
        raise ValueError("supersampling must be at least one")

    scale = supersampling
    image_size = (width * scale, height * scale)
    line_width = max(1, round(stroke_width * pixels_per_unit * scale))
    part_layers: dict[str, np.ndarray] = {}
    stroke_layers: dict[str, np.ndarray] = {}
    coverage = np.zeros((height, width), dtype=np.float32)

    for part_id, path in paths.items():
        layer = Image.new("L", image_size, 0)
        points = sample_path(path)
        pixels = [
            (
                (origin[0] + x * pixels_per_unit) * scale,
                (origin[1] - y * pixels_per_unit) * scale,
            )
            for x, y in points
        ]
        draw = ImageDraw.Draw(layer)
        if len(pixels) >= 2:
            draw.line(pixels, fill=255, width=line_width, joint="curve")
        elif pixels:
            x, y = pixels[0]
            radius = line_width / 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        resized = layer.resize((width, height), Image.Resampling.LANCZOS)
        part = np.asarray(resized, dtype=np.float32) / 255.0
        part_layers[part_id] = part
        stroke_id = stroke_ids.get(part_id, part_id) if stroke_ids else part_id
        if stroke_id in stroke_layers:
            stroke_layers[stroke_id] = np.maximum(stroke_layers[stroke_id], part)
        else:
            stroke_layers[stroke_id] = part.copy()
        coverage = np.maximum(coverage, part)

    return FittingRender(coverage, part_layers, stroke_layers)


def render_traces(
    traces: Sequence[CompiledPartTrace],
    *,
    canvas_size: tuple[int, int],
    origin: tuple[float, float],
    pixels_per_unit: float,
    stroke_width: float,
    supersampling: int = 4,
) -> FittingRender:
    paths = {trace.instance_id: trace.d for trace in traces}
    stroke_ids = {trace.instance_id: trace.stroke_id for trace in traces}
    return render_paths(
        paths,
        canvas_size=canvas_size,
        origin=origin,
        pixels_per_unit=pixels_per_unit,
        stroke_width=stroke_width,
        supersampling=supersampling,
        stroke_ids=stroke_ids,
    )
