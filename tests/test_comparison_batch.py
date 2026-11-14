from __future__ import annotations

import json

from PIL import Image, ImageDraw

import inkgrammar.comparison.api as comparison_api
from inkgrammar.comparison.batch import compare_letter_page


def _draw_glyph(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "e":
        draw.line(
            [(x, y + 15), (x + 19, y + 15), (x + 21, y + 10), (x + 18, y + 4),
             (x + 10, y + 2), (x + 3, y + 6), (x, y + 14), (x + 2, y + 22),
             (x + 9, y + 26), (x + 18, y + 25)],
            fill="black",
            width=4,
        )
    elif kind == "n":
        draw.line(
            [(x, y + 26), (x, y + 5), (x + 6, y + 2), (x + 15, y + 4),
             (x + 20, y + 11), (x + 20, y + 26)],
            fill="black",
            width=4,
        )
    elif kind == "l":
        draw.line((x + 8, y, x + 8, y + 26), fill="black", width=4)


def _page(path, glyphs: tuple[str, ...], *, y: int = 24):
    image = Image.new("RGB", (250, 88), "white")
    draw = ImageDraw.Draw(image)
    for x, kind in zip((15, 95, 175), glyphs, strict=True):
        _draw_glyph(draw, x, y, kind)
    image.save(path)
    return path


def test_batch_writes_pair_reports_summaries_and_reuses_hash_keyed_reference_cache(
    tmp_path, monkeypatch
):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "automatic-run"

    first = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
    )

    assert first.payload["sample"]["detected_count"] == 3
    assert first.payload["reference_cache"]["hit"] is False
    assert first.json_path.is_file()
    assert first.markdown_path.is_file()
    assert first.html_path.is_file()
    for filename in (
        "reference-mask.png",
        "reference-candidates.png",
        "reference-candidates.json",
        "sample-mask.png",
        "sample-boundaries.png",
        "sample-candidates.png",
        "sample-candidates.json",
    ):
        assert (output / "segmentation" / filename).is_file()
    assert (output / "reference-index" / "gallery.json").is_file()
    assert all(
        (output / "reference-index" / f"ref-{index:03d}.png").is_file()
        for index in range(3)
    )

    summary = json.loads(first.json_path.read_text())
    assert len(summary["matches"]) == 3
    assert "not a probability" in summary["matching"]["cost_interpretation"]
    assert "<img" in first.html_path.read_text()
    for match in summary["matches"]:
        sample_dir = output / match["sample_id"]
        assert (sample_dir / "sample.png").is_file()
        assert (sample_dir / "matched-reference.png").is_file()
        assert (sample_dir / "alternatives.png").is_file()
        assert (sample_dir / "match.json").is_file()
        assert (sample_dir / "report.json").is_file()
        assert (sample_dir / "report.html").is_file()
        assert len(match["alternatives"]) == 2

    def no_rebuild(*args, **kwargs):
        raise AssertionError("reference segmentation should be loaded from the content-hash cache")

    monkeypatch.setattr("inkgrammar.comparison.batch.segment_reference_gallery", no_rebuild)
    analysis_calls = 0
    original_analyze = comparison_api.analyze_skeleton

    def count_sample_analysis(mask):
        nonlocal analysis_calls
        analysis_calls += 1
        return original_analyze(mask)

    monkeypatch.setattr(comparison_api, "analyze_skeleton", count_sample_analysis)
    second = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
    )

    assert second.payload["reference_cache"]["hit"] is True
    public_gallery = json.loads((output / "reference-index" / "gallery.json").read_text())
    assert public_gallery["cache_hit"] is True
    assert analysis_calls == 3
