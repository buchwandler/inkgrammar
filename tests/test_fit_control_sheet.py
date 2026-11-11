import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from inkgrammar.cli import app
from inkgrammar.fitting.control_sheet import (
    DEFAULT_CHARACTERS,
    _fit_templates,
    write_control_sheet,
)
from inkgrammar.fitting.workspace import load_fit_project

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "data" / "training" / "sas-1968" / "project.yaml"


def test_control_sheet_renders_all_requested_character_cells_and_marks_missing(tmp_path):
    bundle = load_fit_project(PROJECT)
    output = tmp_path / "alphabet.png"

    report = write_control_sheet(bundle, output)

    assert report["target_count"] == 53
    assert len(report["glyphs"]) == len(DEFAULT_CHARACTERS)
    assert report["rendered_count"] == 12
    assert "A" in report["missing"]
    assert set(report["provisional"]) == set(bundle.topology.glyphs)
    assert report["scale_policy"] == "shared"
    assert report["pixels_per_unit"] == 1.0
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    with Image.open(output) as image:
        assert image.size == (8 * 112, 7 * 140 + 28)


def test_control_sheet_is_deterministic(tmp_path):
    bundle = load_fit_project(PROJECT)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    write_control_sheet(bundle, first)
    write_control_sheet(bundle, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.with_suffix(".json").read_bytes() == second.with_suffix(".json").read_bytes()


def test_control_sheet_can_include_digits(tmp_path):
    report = write_control_sheet(
        load_fit_project(PROJECT),
        tmp_path / "digits.png",
        include_digits=True,
    )

    assert report["target_count"] == 63
    assert len(report["glyphs"]) == 63
    assert all(digit in report["missing"] for digit in "0123456789")


def test_control_sheet_does_not_read_private_source_images(monkeypatch, tmp_path):
    bundle = load_fit_project(PROJECT)

    def reject_image_open(*args, **kwargs):
        raise AssertionError("control-sheet generation must not open source images")

    monkeypatch.setattr(Image, "open", reject_image_open)
    report = write_control_sheet(bundle, tmp_path / "source-free.png")

    assert report["rendered_count"] == 12


def test_control_sheet_uses_model_state_by_default_and_occurrence_deltas_only_on_request(tmp_path):
    bundle = load_fit_project(PROJECT)
    default_path = tmp_path / "default.png"
    occurrence_path = tmp_path / "occurrences.png"

    default = write_control_sheet(bundle, default_path)
    occurrences = write_control_sheet(
        bundle,
        occurrence_path,
        occurrence_deltas=True,
    )

    assert default["occurrence_deltas"] is False
    assert occurrences["occurrence_deltas"] is True
    assert default_path.read_bytes() != occurrence_path.read_bytes()


@pytest.mark.parametrize(
    ("status", "use_candidate"),
    [("promotion_pending", False), ("not_promoted", False), ("promoted", True)],
)
def test_control_sheet_uses_only_promoted_report_templates(monkeypatch, status, use_candidate):
    bundle = load_fit_project(PROJECT)
    primitive_id = "sas-open-oval-01"
    canonical = bundle.grammar.primitives[primitive_id]
    candidate = canonical.model_dump(mode="json")
    candidate["segments"][0]["c1"][1] += 0.05
    report = {
        "primitive_id": primitive_id,
        "primitive": candidate,
        "visual_review": {"status": status},
    }
    monkeypatch.setattr(
        "inkgrammar.fitting.control_sheet.load_fit_reports",
        lambda _: [report],
    )

    templates, _ = _fit_templates(bundle)

    if use_candidate:
        assert templates[primitive_id].model_dump(mode="json") == candidate
    else:
        assert templates[primitive_id] == canonical


def test_control_sheet_cli_writes_png_and_json_with_optional_digits(tmp_path):
    output = tmp_path / "cli-alphabet.png"
    result = CliRunner().invoke(
        app,
        ["fit", "control-sheet", str(PROJECT), "--output", str(output), "--include-digits"],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["target_count"] == 63
    assert "Rendered 12 of 63 target glyphs" in result.output
