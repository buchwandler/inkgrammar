from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import sample_path
from .model import compile_text
from .models import WritingModel


def threshold_image(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), histogram)
    sum_background = 0.0
    weight_background = 0.0
    best_variance = -1.0
    threshold = 127

    for level in range(256):
        weight_background += histogram[level]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += level * histogram[level]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = level

    return gray <= threshold


def crop_to_ink(mask: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((1, 1), dtype=bool)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    return mask[y0:y1, x0:x1]


def normalize_mask(mask: np.ndarray, size: int = 256, margin: int = 16) -> np.ndarray:
    mask = crop_to_ink(mask)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    inner = size - margin * 2
    scale = min(inner / image.width, inner / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(
        resized,
        ((size - width) // 2, (size - height) // 2),
    )
    return np.asarray(canvas) > 80


def compare_masks(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    reference = normalize_mask(reference)
    generated = normalize_mask(generated)
    intersection = np.logical_and(reference, generated).sum()
    union = np.logical_or(reference, generated).sum()
    ref_count = reference.sum()
    gen_count = generated.sum()

    return {
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(intersection / gen_count) if gen_count else 0.0,
        "recall": float(intersection / ref_count) if ref_count else 0.0,
    }


def compare_anchored_masks(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    reference_anchor: tuple[float, float],
    generated_anchor: tuple[float, float],
    generated_pixels_per_reference_pixel: float = 1.0,
) -> dict[str, float]:
    """Compare in the reference frame without independently normalizing ink.

    Anchors are explicit pixel coordinates, typically the writing origin on
    the baseline. Scale is supplied from writing metrics, never inferred from
    the ink bounding box. Positive scale preserves the source aspect ratio.
    """
    if generated_pixels_per_reference_pixel <= 0:
        raise ValueError("pixel scale must be positive")
    if reference.ndim != 2 or generated.ndim != 2:
        raise ValueError("masks must be two-dimensional")
    scale = generated_pixels_per_reference_pixel
    image = Image.fromarray(generated.astype(np.uint8) * 255)
    aligned = (
        np.asarray(
            image.transform(
                (reference.shape[1], reference.shape[0]),
                Image.Transform.AFFINE,
                (
                    scale,
                    0,
                    generated_anchor[0] - reference_anchor[0] * scale,
                    0,
                    scale,
                    generated_anchor[1] - reference_anchor[1] * scale,
                ),
                resample=Image.Resampling.NEAREST,
            )
        )
        > 0
    )
    intersection = np.logical_and(reference, aligned).sum()
    union = np.logical_or(reference, aligned).sum()
    result = compare_masks(reference, generated)
    result["baseline_aligned_iou"] = float(intersection / union) if union else 1.0

    def bounds_and_center(mask):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return 0.0, 0.0, (0.0, 0.0)
        return (
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
            (float(xs.mean()), float(ys.mean())),
        )

    rw, rh, rc = bounds_and_center(reference)
    gw, gh, gc = bounds_and_center(generated)
    result["reference_ink_bbox_ratio"] = rw / rh if rh else 0.0
    result["generated_ink_bbox_ratio"] = gw / gh if gh else 0.0
    result["ink_bbox_width_ratio"] = gw / scale / rw if rw else 0.0
    result["ink_bbox_height_ratio"] = gh / scale / rh if rh else 0.0
    dx = (gc[0] - generated_anchor[0]) / scale + reference_anchor[0] - rc[0]
    dy = (gc[1] - generated_anchor[1]) / scale + reference_anchor[1] - rc[1]
    result["center_of_mass_offset"] = float(np.hypot(dx, dy))
    return result


@dataclass(frozen=True)
class RasterFrame:
    image: Image.Image
    origin: tuple[float, float]
    scale: float


def render_text_frame(
    model: WritingModel,
    text: str,
    *,
    gap_override: float | None = None,
    tension_override: float | None = None,
    scale: float = 0.7,
) -> RasterFrame:
    plan = compile_text(
        model,
        text,
        gap_override=gap_override,
        tension_override=tension_override,
    )
    metrics = model.metrics
    if scale <= 0:
        raise ValueError("render scale must be positive")
    paths = {
        stroke.id: sample_path(stroke.d, 18) for stroke in plan.strokes if stroke.d is not None
    }
    ink_points = [point for points in paths.values() for point in points]
    ink_points.extend(stroke.tap for stroke in plan.strokes if stroke.tap is not None)
    padding = metrics.default_stroke_width / 2 + 1 / scale
    left = min([0.0] + [point[0] - padding for point in ink_points])
    right = max([plan.width] + [point[0] + padding for point in ink_points])
    top = max([metrics.ascender] + [point[1] + padding for point in ink_points])
    bottom = min([metrics.descender] + [point[1] - padding for point in ink_points])
    width = max(32, math.ceil((right - left) * scale))
    height = max(32, math.ceil((top - bottom) * scale))
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    stroke_width = max(1, int(metrics.default_stroke_width * scale))

    def to_pixel(point):
        x = (point[0] - left) * scale
        y = (top - point[1]) * scale
        return (x, y)

    for stroke in plan.strokes:
        if stroke.d is not None:
            points = [to_pixel(point) for point in paths[stroke.id]]
            if len(points) >= 2:
                draw.line(
                    points,
                    fill=0,
                    width=stroke_width,
                    joint="curve",
                )
        elif stroke.tap is not None:
            x, y = to_pixel(stroke.tap)
            radius = stroke_width / 2
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=0,
            )
    return RasterFrame(image, (-left * scale, (top - metrics.baseline) * scale), scale)


def render_text_mask(
    model: WritingModel,
    text: str,
    *,
    gap_override: float | None = None,
    tension_override: float | None = None,
    scale: float = 0.7,
) -> Image.Image:
    return render_text_frame(
        model, text, gap_override=gap_override, tension_override=tension_override, scale=scale
    ).image


def compare_text_to_image(
    model: WritingModel,
    text: str,
    reference_path: str | Path,
    *,
    gap_override: float | None = None,
    tension_override: float | None = None,
) -> dict[str, float]:
    reference = threshold_image(Image.open(reference_path))
    generated_image = render_text_mask(
        model,
        text,
        gap_override=gap_override,
        tension_override=tension_override,
    )
    generated = threshold_image(generated_image)
    return compare_masks(reference, generated)
