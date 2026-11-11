from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageOps

import inkgrammar.compiler as compiler
import inkgrammar.fitting.preview as preview
from examples.sas1968 import render_validation as examples
from inkgrammar.fitting.family_fit import optimizer_backend_for_method
from inkgrammar.fitting.models import PrimitiveTemplate
from inkgrammar.fitting.workspace import load_fit_project
from inkgrammar.io import load_model

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "data" / "training" / "sas-1968" / "project.yaml"
MODEL_PATH = ROOT / "data" / "models" / "sas-1968.yaml"
CANDIDATE_PRIMITIVES = {
    "sas-underturn-01",
    "sas-open-oval-01",
    "sas-hump-01",
    "sas-ascender-loop-01",
}


def _candidate_reports(bundle):
    reports = []
    for primitive_id in sorted(CANDIDATE_PRIMITIVES):
        candidate_data = bundle.grammar.primitives[primitive_id].model_dump(mode="python")
        candidate_data["segments"][0]["c1"] = (
            candidate_data["segments"][0]["c1"][0] + 0.01,
            candidate_data["segments"][0]["c1"][1],
        )
        candidate = PrimitiveTemplate.model_validate(candidate_data)
        reports.append(
            {
                "format": "inkgrammar-fit-report",
                "format_version": 4,
                "project_id": bundle.project.id,
                "primitive_id": primitive_id,
                "objective_version": bundle.fit_config.objective.version,
                "optimizer": {
                    "backend": optimizer_backend_for_method(bundle.fit_config.optimizer.method),
                    "success": True,
                },
                "input_hashes": {},
                "primitive": candidate.model_dump(mode="json"),
                "visual_review": {"status": "promotion_pending"},
            }
        )
    return reports


def _render_both_with_candidates(monkeypatch, bundle, output):
    reports = _candidate_reports(bundle)
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "validate_fit_report_inputs", lambda *_: None)
    return examples.render_validation(bundle, geometry="both", output=output)


def _file_bytes(directory: Path):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_example_outputs_show_full_coverage_words_and_reloadable_preview(tmp_path):
    bundle = load_fit_project(PROJECT)
    original_model = MODEL_PATH.read_bytes()
    output = tmp_path / "canonical"

    result = examples.render_validation(bundle, geometry="canonical", output=output)
    manifest = result["runs"]["canonical"]

    assert manifest["glyph_target_count"] == 53
    assert manifest["glyph_rendered_count"] == 12
    assert len(manifest["missing_glyphs"]) == 41
    assert len(manifest["supported_glyphs"]) == 12
    assert manifest["connection_rule_count"] == 0
    assert manifest["connected_writing"] is False
    assert manifest["source_free"] is True
    assert manifest["canonical_model_modified"] is False
    assert all((output / name).is_file() for name in (
        "manifest.json",
        "01-alphabet.png",
        "01-alphabet.json",
        "02-supported-glyphs.png",
        "03-words.png",
        "model-preview.yaml",
    ))
    assert len(list((output / "debug").glob("*.svg"))) == 6
    preview_model = load_model(output / "model-preview.yaml")
    assert preview_model.provenance["preview_geometry"] == "canonical"
    assert "data/observations" not in (output / "manifest.json").read_text()
    assert MODEL_PATH.read_bytes() == original_model

    words = manifest["words"]
    assert all(row["status"] == "supported" and row["rendered"] for row in words[:6])
    assert all(row["status"] == "blocked" and not row["rendered"] for row in words[6:])
    assert all(row["missing"] for row in words[6:])
    with Image.open(output / "03-words.png") as image:
        assert ImageOps.invert(image.convert("L")).getbbox() is not None




def test_word_preflight_blocks_unsupported_text_without_partial_image():
    model = load_model(MODEL_PATH)

    blocked = examples._word_case(model, "schule", "future coverage")
    supported = examples._word_case(model, "minimum", "repeated structures")

    assert blocked["status"] == "blocked"
    assert blocked["missing"] == ["s", "e"]
    assert blocked["rendered"] is False
    assert "image" not in blocked
    assert supported["status"] == "supported"
    assert supported["rendered"] is True
    assert ImageOps.invert(supported["image"]).getbbox() is not None


def test_digit_option_adds_all_ten_explicit_cells(tmp_path):
    bundle = load_fit_project(PROJECT)

    result = examples.render_validation(
        bundle,
        geometry="canonical",
        output=tmp_path / "digits",
        include_digits=True,
    )

    manifest = result["runs"]["canonical"]
    assert manifest["glyph_target_count"] == 63
    assert all(digit in manifest["missing_glyphs"] for digit in "0123456789")


def test_identical_runs_are_byte_deterministic(tmp_path):
    bundle = load_fit_project(PROJECT)
    first = tmp_path / "first"
    second = tmp_path / "second"

    examples.render_validation(bundle, geometry="canonical", output=first)
    examples.render_validation(bundle, geometry="canonical", output=second)

    assert _file_bytes(first) == _file_bytes(second)


def test_example_rendering_does_not_open_private_work_files(monkeypatch, tmp_path):
    bundle = load_fit_project(PROJECT)
    original_open = Path.open
    protected = tuple(
        (bundle.private_work_dir / name).resolve()
        for name in ("sources", "observations", "materialized")
    )

    def reject_private_open(path, *args, **kwargs):
        resolved = path.resolve(strict=False)
        if any(resolved == root or root in resolved.parents for root in protected):
            raise AssertionError(f"example renderer opened private work file: {resolved}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_private_open)
    examples.render_validation(bundle, geometry="canonical", output=tmp_path / "source-free")


def test_both_mode_comparisons_use_common_dimensions_and_report_candidate_state(
    monkeypatch, tmp_path
):
    bundle = load_fit_project(PROJECT)
    original_model = MODEL_PATH.read_bytes()
    output = tmp_path / "both"

    result = _render_both_with_candidates(monkeypatch, bundle, output)

    assert result["comparison"]["common_scale"] is True
    assert result["comparison"]["candidate_unpromoted"] is True
    assert result["comparison"]["modeled_glyph_count"] == 12
    for filename in (
        "01-alphabet-side-by-side.png",
        "02-modeled-glyph-difference.png",
        "03-words-side-by-side.png",
        "comparison.json",
    ):
        assert (output / "comparison" / filename).is_file()
    canonical_manifest = json.loads((output / "canonical" / "manifest.json").read_text())
    candidate_manifest = json.loads((output / "latest-candidate" / "manifest.json").read_text())
    assert canonical_manifest["geometry"] == "canonical"
    assert candidate_manifest["geometry"] == "latest-candidate"
    assert all(
        detail["visual_review_status"] == "promotion_pending"
        for detail in candidate_manifest["fit_reports"].values()
    )
    assert yaml.safe_load(
        (output / "latest-candidate" / "model-preview.yaml").read_text()
    )["provenance"]["unpromoted"] is True
    for left_name, right_name in (
        ("01-alphabet.png", "01-alphabet.png"),
        ("02-supported-glyphs.png", "02-supported-glyphs.png"),
        ("03-words.png", "03-words.png"),
    ):
        with (
            Image.open(output / "canonical" / left_name) as left,
            Image.open(output / "latest-candidate" / right_name) as right,
        ):
            assert left.size == right.size
    assert MODEL_PATH.read_bytes() == original_model
