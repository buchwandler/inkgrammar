from __future__ import annotations

import json

from PIL import Image, ImageDraw
from typer.testing import CliRunner

from inkgrammar.cli import app


def _draw_glyph(draw: ImageDraw.ImageDraw, x: int, kind: str) -> None:
    y = 25
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


def _page(path, glyphs: tuple[str, ...]):
    image = Image.new("RGB", (250, 88), "white")
    draw = ImageDraw.Draw(image)
    for x, kind in zip((15, 95, 175), glyphs, strict=True):
        _draw_glyph(draw, x, kind)
    image.save(path)
    return path


def test_compare_letter_cli_automatic_mode_writes_batch_summary(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "automatic"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "3",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "HTML summary:" in result.output
    assert "JSON summary:" in result.output
    summary = json.loads((output / "summary.json").read_text())
    assert summary["sample"]["detected_count"] == 3


def test_compare_letter_cli_reports_exact_count_diagnostics(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "wrong-count"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "4",
            "--output",
            str(output),
        ],
    )

    diagnostic = output / "segmentation" / "sample-boundaries.png"
    assert result.exit_code == 2
    assert "Unable to recover 4 credible" in result.output
    assert "Diagnostics:" in result.output
    assert diagnostic.name in result.output
    assert diagnostic.is_file()


def test_compare_letter_cli_rejects_missing_mixed_and_incomplete_modes(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    base = ["compare-letter", str(reference), str(sample), "--output", str(tmp_path / "out")]
    runner = CliRunner()

    missing = runner.invoke(app, base)
    mixed = runner.invoke(
        app,
        base + ["--letters", "3", "--letter", "e", "--reference-crop", "1,1,20,20"],
    )
    incomplete = runner.invoke(
        app,
        base + ["--letter", "e", "--reference-crop", "1,1,20,20"],
    )

    assert missing.exit_code == 2
    assert "choose --letters N" in missing.output
    assert mixed.exit_code == 2
    assert "cannot be combined" in mixed.output
    assert incomplete.exit_code == 2
    assert "provide --letter" in incomplete.output
