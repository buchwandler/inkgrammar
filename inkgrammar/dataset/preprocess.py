"""Deterministic, parameterized, versioned image preprocessing.

Every operation records its policy in :class:`PreprocessParams`; behavior
changes bump :data:`PREPROCESS_VERSION`. Raw binary interpretations are never
overwritten: :func:`clean_mask` returns the raw mask untouched plus a cleanup
record of removed speck candidates. Automatic outputs are candidates for
review, never truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field

from ..models import StrictModel
from ..raster import otsu_mask
from .models import ObservationMetrics

PREPROCESS_VERSION = 1

GlyphClass = Literal["lowercase", "uppercase", "mixed"]


class PreprocessParams(StrictModel):
    """Parameters for all preprocessing operations (versioned as a set)."""

    version: Literal[1] = 1
    threshold_policy: Literal["otsu-256-v1"] = "otsu-256-v1"
    background_percentile: float | None = Field(default=None, gt=0.0, le=100.0)
    speck_min_area: int = Field(default=4, ge=1)
    crop_pad: int = Field(default=2, ge=0)


class RemovedComponent(StrictModel):
    bbox: tuple[int, int, int, int]
    area: int
    reason: str = "automatic-speck-candidate"
    reviewed: bool = False


class CleanupRecord(StrictModel):
    removed_components: list[RemovedComponent] = Field(default_factory=list)


class IsolationComponent(StrictModel):
    bbox: tuple[int, int, int, int]
    area: int
    kept: bool
    reason: str


class IsolationProposal(StrictModel):
    """Automatic target-ink isolation candidate; review decides acceptance."""

    center: tuple[float, float]
    components: list[IsolationComponent] = Field(default_factory=list)


def isolate_target_ink(
    mask: np.ndarray,
    *,
    center: tuple[float, float],
    neighbor_centers: tuple[tuple[float, float], ...] = (),
) -> tuple[np.ndarray, IsolationProposal]:
    """Propose the target ink as the connected component nearest ``center``.

    The center is the reviewed layout center; the proposal is only a
    candidate. Components that sit closer to a neighboring label center
    likely belong to the neighbor (cursive letters touch), and all rejected
    components are recorded with their reason so review can accept or
    override the isolation.
    """
    components = label_components(mask)
    if not components:
        return mask.copy(), IsolationProposal(center=center)

    def min_distance_squared(component: tuple[np.ndarray, np.ndarray], point) -> float:
        ys, xs = component
        return float(((ys - point[1]) ** 2 + (xs - point[0]) ** 2).min())

    kept_component = min(components, key=lambda c: min_distance_squared(c, center))
    candidate = np.zeros(mask.shape, dtype=bool)
    records = []
    for ys, xs in components:
        is_kept = ys is kept_component[0] and xs is kept_component[1]
        reason = "nearest-to-reviewed-center"
        if is_kept:
            candidate[ys, xs] = True
        elif neighbor_centers and min(
            min_distance_squared((ys, xs), neighbor) for neighbor in neighbor_centers
        ) < min_distance_squared((ys, xs), center):
            reason = "nearest-to-neighbor-center"
        else:
            reason = "not-nearest-component"
        records.append(
            IsolationComponent(
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                area=int(ys.size),
                kept=is_kept,
                reason=reason,
            )
        )
    return candidate, IsolationProposal(center=center, components=records)


@dataclass(frozen=True)
class PreprocessResult:
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    cleanup: CleanupRecord
    metrics: ObservationMetrics
    params: PreprocessParams


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert to 8-bit grayscale intensity."""
    return np.asarray(image.convert("L"), dtype=np.uint8)


def normalize_background(gray: np.ndarray, *, percentile: float = 95.0) -> np.ndarray:
    """Scale intensities so the background percentile maps to white."""
    background = float(np.percentile(gray, percentile))
    if background <= 0:
        return gray
    scaled = gray.astype(float) * (255.0 / background)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def threshold_mask(gray: np.ndarray) -> np.ndarray:
    """Binarize with the recorded threshold policy (``otsu-256-v1``)."""
    return otsu_mask(gray)


def label_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Label 8-connected ink components as (ys, xs) index arrays."""
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(start_y, start_x)]
        ys = []
        xs = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append((np.asarray(ys), np.asarray(xs)))
    return components


def detect_speck_candidates(mask: np.ndarray, *, min_area: int) -> list[RemovedComponent]:
    """Propose small components as dust/speck candidates (review decides)."""
    candidates = []
    for ys, xs in label_components(mask):
        area = int(ys.size)
        if area < min_area:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            candidates.append(RemovedComponent(bbox=bbox, area=area))
    return candidates


def clean_mask(mask: np.ndarray, params: PreprocessParams) -> tuple[np.ndarray, CleanupRecord]:
    """Remove speck candidates from a copy of the raw mask; never mutate raw."""
    cleaned = mask.copy()
    removed = []
    for ys, xs in label_components(mask):
        area = int(ys.size)
        if area < params.speck_min_area:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            removed.append(RemovedComponent(bbox=bbox, area=area))
            cleaned[ys, xs] = False
    return cleaned, CleanupRecord(removed_components=removed)


def pad_bbox(
    bbox: tuple[int, int, int, int],
    *,
    pad: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Pad a pixel-space bbox and clamp it to the image shape (h, w)."""
    x0, y0, x1, y1 = bbox
    height, width = shape
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1]))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def estimate_metric_guides(
    mask: np.ndarray, *, glyph_class: GlyphClass = "mixed"
) -> ObservationMetrics:
    """Propose metric guides from trimmed ink row extents.

    All values are crop-normalized (y downward) ``automatic-candidate``
    estimates. For lowercase crops the trimmed top edge is the x-height line
    candidate; for uppercase crops it is the cap height candidate. Mixed
    content only yields baseline and ascender-edge candidates.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("cannot estimate metric guides from an empty mask")
    weights = mask.sum(axis=1)[rows].astype(float)
    top = _weighted_quantile(rows, weights, 0.02) / mask.shape[0]
    bottom = _weighted_quantile(rows, weights, 0.98) / mask.shape[0]
    if glyph_class == "lowercase":
        return ObservationMetrics(
            baseline_y=bottom, x_height_y=top, metric_status="automatic-candidate"
        )
    if glyph_class == "uppercase":
        return ObservationMetrics(
            baseline_y=bottom, cap_height_y=top, metric_status="automatic-candidate"
        )
    return ObservationMetrics(
        baseline_y=bottom, ascender_y=top, metric_status="automatic-candidate"
    )


def preprocess_crop(
    image: Image.Image,
    params: PreprocessParams,
    *,
    glyph_class: GlyphClass = "mixed",
) -> PreprocessResult:
    """Run the full pipeline: gray, background, threshold, clean, metrics."""
    gray = to_grayscale(image)
    if params.background_percentile is not None:
        gray = normalize_background(gray, percentile=params.background_percentile)
    raw = threshold_mask(gray)
    cleaned, cleanup = clean_mask(raw, params)
    metrics = estimate_metric_guides(cleaned, glyph_class=glyph_class)
    return PreprocessResult(
        raw_mask=raw,
        clean_mask=cleaned,
        cleanup=cleanup,
        metrics=metrics,
        params=params,
    )
