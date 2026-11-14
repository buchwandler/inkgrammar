from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from inkgrammar.comparison.descriptors import canonicalize_mask, glyph_descriptor
from inkgrammar.comparison.matching import (
    AMBIGUITY_ABSOLUTE_MARGIN,
    ASPECT_WEIGHT,
    FILL_WEIGHT,
    RASTER_WEIGHT,
    TOPOLOGY_WEIGHT,
    MatchScore,
    build_reference_index,
    match_status,
    rank_reference_candidates,
    refine_match,
    refine_top_k_matches,
)
from inkgrammar.comparison.models import CropRect, InkMaskResult
from inkgrammar.comparison.segmentation import PageSegmentation, ReferenceGallery, SegmentedGlyph


def _shape(kind: str, *, size: int = 64, offset: tuple[int, int] = (0, 0)) -> SegmentedGlyph:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    x, y = offset
    if kind == "e":
        draw.line(
            [(x + 12, y + 31), (x + 33, y + 31), (x + 36, y + 24), (x + 32, y + 17),
             (x + 24, y + 15), (x + 16, y + 19), (x + 12, y + 28), (x + 14, y + 38),
             (x + 21, y + 43), (x + 32, y + 42)],
            fill="black",
            width=4,
        )
    elif kind == "l":
        draw.line((x + 25, y + 12, x + 25, y + 45), fill="black", width=4)
    elif kind == "n":
        draw.line(
            [(x + 13, y + 45), (x + 13, y + 18), (x + 20, y + 15),
             (x + 31, y + 18), (x + 35, y + 25), (x + 35, y + 45)],
            fill="black",
            width=4,
        )
    elif kind == "two-strokes":
        draw.line((x + 16, y + 15, x + 16, y + 45), fill="black", width=4)
        draw.line((x + 36, y + 15, x + 36, y + 45), fill="black", width=4)
    mask = np.asarray(image.convert("L")) < 128
    return SegmentedGlyph(
        glyph_id=kind,
        bbox=CropRect(x=0, y=0, width=size, height=size),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=image,
        mask=mask,
    )


def _gallery(*candidates: SegmentedGlyph) -> ReferenceGallery:
    page_mask = np.zeros((64, 64), dtype=bool)
    segmentation = PageSegmentation(Path("reference.png"), candidates)
    return ReferenceGallery(
        source_path=Path("reference.png"),
        source_sha256="a" * 64,
        candidates=tuple(candidates),
        mask_result=InkMaskResult(page_mask, 1.0, "grayscale", []),
        segmentation=segmentation,
    )


def _translated_scaled_e() -> SegmentedGlyph:
    source = _shape("e").image
    rotated = source.rotate(4, resample=Image.Resampling.NEAREST, expand=False)
    resized = rotated.resize((48, 48), resample=Image.Resampling.NEAREST)
    image = Image.new("RGB", (80, 80), "white")
    image.paste(resized, (20, 16))
    mask = np.asarray(image.convert("L")) < 128
    return SegmentedGlyph(
        glyph_id="sample",
        bbox=CropRect(x=0, y=0, width=80, height=80),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=image,
        mask=mask,
    )


def test_canonical_mask_preserves_aspect_ratio_and_centers_shape():
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 4:36] = True

    normalized = canonicalize_mask(mask, size=96, padding=8)
    ys, xs = np.nonzero(normalized)

    assert normalized.shape == (96, 96)
    assert abs((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) - 3.2) < 0.15
    assert abs(float(xs.mean()) - float(ys.mean())) < 2.0


def test_transformed_query_ranks_its_reference_first_without_unique_assignment():
    reference_e = _shape("e")
    gallery = _gallery(reference_e, _shape("n"), _shape("l"))
    index = build_reference_index(gallery)
    sample = _translated_scaled_e()

    ranked = rank_reference_candidates(sample, index, top_k=3)
    repeated = rank_reference_candidates(sample, index, top_k=3)

    assert ranked[0].reference_id == "e"
    assert [score.rank for score in ranked] == [1, 2, 3]
    assert repeated[0].reference_id == "e"
    assert RASTER_WEIGHT == 1.0
    assert ASPECT_WEIGHT == 0.15
    assert FILL_WEIGHT == 0.10
    assert TOPOLOGY_WEIGHT == 0.15



def test_refinement_is_limited_to_the_coarse_top_k_candidates():
    gallery = _gallery(_shape("e"), _shape("n"), _shape("l"))
    coarse = rank_reference_candidates(_shape("e"), gallery, top_k=2)

    refined = refine_top_k_matches(_shape("e"), gallery, coarse)

    assert len(refined) == 2
    assert [score.rank for score in refined] == [1, 2]
    assert {score.reference_id for score in refined} <= {
        score.reference_id for score in coarse
    }


def test_ambiguity_uses_a_deterministic_best_second_margin():
    first = MatchScore("ref-a", 1, 0.1, 0.1, 0, 0, 0)
    second = MatchScore("ref-b", 2, 0.1 + AMBIGUITY_ABSOLUTE_MARGIN / 2, 0.1, 0, 0, 0)

    assert match_status([first, second]) == "ambiguous"
    assert match_status([first]) == "clear"


def test_refinement_uses_paths_when_available_and_falls_back_when_not():
    reference = _shape("l")
    descriptor = glyph_descriptor(reference)
    coarse = rank_reference_candidates(reference, _gallery(reference))[0]

    refined = refine_match(reference, reference, coarse)
    raster_only = refine_match(_shape("two-strokes"), _shape("l"), coarse)

    assert refined.centerline_cost is not None
    assert refined.heading_cost is not None
    assert refined.turn_rate_cost is not None
    assert refined.refined_cost <= refined.coarse_cost + 1e-8
    assert descriptor.path_available is None
    assert raster_only.refined_cost == raster_only.coarse_cost
    assert raster_only.centerline_cost is None
