from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..dataset.preprocess import label_components
from .segmentation import SegmentedGlyph


@dataclass(frozen=True)
class GlyphDescriptor:
    glyph_id: str
    normalized_mask: np.ndarray
    aspect_ratio: float
    ink_fill: float
    component_count: int
    endpoint_count: int | None = None
    junction_count: int | None = None
    cycle_count: int | None = None
    path_available: bool | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.normalized_mask, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("normalized glyph mask must be a nonempty 2D array")
        if self.aspect_ratio <= 0.0 or not 0.0 <= self.ink_fill <= 1.0:
            raise ValueError("glyph descriptor geometry is invalid")
        if self.component_count < 1:
            raise ValueError("glyph component count must be positive")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "normalized_mask", mask)


def canonicalize_mask(
    mask: np.ndarray,
    *,
    size: int = 96,
    padding: int = 8,
) -> np.ndarray:
    """Fit a tight binary glyph mask into a centered, aspect-preserving square."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("glyph mask must be two-dimensional")
    if size < 1 or padding < 0 or padding * 2 >= size:
        raise ValueError("size must be positive and padding must leave a nonempty canvas")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("cannot normalize an empty glyph mask")
    tight = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target = size - padding * 2
    scale = min(target / tight.shape[1], target / tight.shape[0])
    width = max(1, round(tight.shape[1] * scale))
    height = max(1, round(tight.shape[0] * scale))
    resized = Image.fromarray(tight.astype(np.uint8) * 255).resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    raster = np.asarray(resized, dtype=np.uint8) > 0
    canvas = np.zeros((size, size), dtype=bool)
    x0 = (size - width) // 2
    y0 = (size - height) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = raster
    return canvas


def glyph_descriptor(candidate: SegmentedGlyph, *, size: int = 96) -> GlyphDescriptor:
    """Create the coarse raster and topology descriptor for retrieval ranking."""
    ys, xs = np.nonzero(candidate.mask)
    if not len(xs):
        raise ValueError(f"glyph candidate {candidate.glyph_id} has an empty mask")
    tight = candidate.mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return GlyphDescriptor(
        glyph_id=candidate.glyph_id,
        normalized_mask=canonicalize_mask(tight, size=size),
        aspect_ratio=float(tight.shape[1] / tight.shape[0]),
        ink_fill=float(tight.mean()),
        component_count=len(label_components(tight)),
    )
