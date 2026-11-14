from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

from inkgrammar.comparison.segmentation import (
    SegmentationError,
    detect_ink_rows,
    segment_reference_gallery,
    segment_sample_glyphs,
)


def _draw_shape(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "e":
        draw.line([(x, y + 14), (x + 18, y + 14), (x + 20, y + 9), (x + 17, y + 4),
                   (x + 10, y + 2), (x + 4, y + 5), (x + 1, y + 12), (x + 3, y + 20),
                   (x + 9, y + 25), (x + 17, y + 24)], fill="black", width=4)
    elif kind == "l":
        draw.line((x + 8, y, x + 8, y + 26), fill="black", width=4)
    elif kind == "i":
        draw.line((x + 8, y + 9, x + 8, y + 26), fill="black", width=4)
        draw.ellipse((x + 6, y, x + 10, y + 4), fill="black")
    elif kind == "n":
        draw.line([(x, y + 26), (x, y + 5), (x + 6, y + 2), (x + 15, y + 4),
                   (x + 19, y + 10), (x + 19, y + 26)], fill="black", width=4)


def _reference_page(path):
    image = Image.new("RGB", (260, 170), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 259, 169), outline="black", width=2)
    for x, kind in ((20, "e"), (75, "l"), (125, "i"), (185, "n")):
        _draw_shape(draw, x, 20, kind)
    for x, kind in ((20, "n"), (80, "e"), (145, "l"), (200, "i")):
        _draw_shape(draw, x, 105, kind)
    image.save(path)
    return path


def _sample_page(path):
    image = Image.new("RGB", (260, 135), "white")
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, 15, 35, "e")
    _draw_shape(draw, 105, 35, "i")
    _draw_shape(draw, 195, 35, "n")
    draw.line((4, 100, 255, 100), fill=(180, 180, 180), width=1)
    draw.point((250, 8), fill="black")
    image.save(path)
    return path


def test_reference_gallery_segments_rows_and_attaches_detached_marks(tmp_path):
    path = _reference_page(tmp_path / "alphabet.png")
    output = tmp_path / "reference-diagnostics"

    gallery = segment_reference_gallery(path, output_dir=output)

    assert len(gallery.candidates) == 8
    assert [candidate.glyph_id for candidate in gallery.candidates] == [
        f"ref-{index:03d}" for index in range(8)
    ]
    assert [candidate.row_index for candidate in gallery.candidates] == [0] * 4 + [1] * 4
    assert [candidate.column_index for candidate in gallery.candidates] == [0, 1, 2, 3] * 2
    dotted = gallery.candidates[2]
    assert dotted.bbox.y <= 20
    assert len(detect_ink_rows(gallery.mask_result.mask)) == 2
    assert (output / "reference-mask.png").is_file()
    assert (output / "reference-candidates.png").is_file()
    manifest = json.loads((output / "reference-candidates.json").read_text())
    assert manifest["candidate_count"] == 8
    assert manifest["candidates"][2]["id"] == "ref-002"


def test_sample_segmentation_recovers_exact_count_and_writes_review_artifacts(tmp_path):
    path = _sample_page(tmp_path / "sample.png")
    output = tmp_path / "sample-diagnostics"

    samples = segment_sample_glyphs(path, expected_count=3, ink_mode="dark", output_dir=output)

    assert [candidate.glyph_id for candidate in samples] == [
        "sample-000",
        "sample-001",
        "sample-002",
    ]
    assert samples[1].bbox.y == 35
    assert all(candidate.segmentation_confidence >= 0.3 for candidate in samples)
    for filename in (
        "sample-mask.png",
        "sample-components.png",
        "sample-boundaries.png",
        "sample-candidates.png",
        "sample-segmentation.json",
    ):
        assert (output / filename).is_file()
    manifest = json.loads((output / "sample-segmentation.json").read_text())
    assert manifest["expected_count"] == manifest["detected_count"] == 3
    assert manifest["status"] == "ok"
    assert any("rule" in warning for warning in manifest["warnings"])
    with Image.open(output / "sample-mask.png") as mask_image:
        assert mask_image.crop((0, 100, 260, 101)).getbbox() is None


def test_wrong_sample_count_fails_with_segmentation_diagnostics(tmp_path):
    path = _sample_page(tmp_path / "sample.png")
    output = tmp_path / "failed-segmentation"

    with pytest.raises(SegmentationError, match="Unable to recover 4 credible") as error:
        segment_sample_glyphs(path, expected_count=4, ink_mode="dark", output_dir=output)

    assert error.value.diagnostic_path == output / "sample-boundaries.png"
    assert (output / "sample-mask.png").is_file()
    assert (output / "sample-components.png").is_file()
    assert (output / "sample-boundaries.png").is_file()
    manifest = json.loads((output / "sample-segmentation.json").read_text())
    assert manifest["expected_count"] == 4
    assert manifest["status"] == "failed"
