from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import inkgrammar.compiler as compiler
import inkgrammar.fitting.family_fit as family_fit
import inkgrammar.fitting.preview as preview
from inkgrammar.fitting.checkpoint import CheckpointInputs
from inkgrammar.fitting.models import PrimitiveTemplate
from inkgrammar.fitting.preview import build_fit_preview_model
from inkgrammar.fitting.workspace import load_fit_project
from inkgrammar.io import load_model

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "data" / "training" / "sas-1968" / "project.yaml"
CANDIDATE_PRIMITIVES = {
    "sas-underturn-01",
    "sas-open-oval-01",
    "sas-hump-01",
    "sas-ascender-loop-01",
}


def _reports(bundle):
    reports = []
    for primitive_id in sorted(CANDIDATE_PRIMITIVES):
        seed = bundle.grammar.primitives[primitive_id]
        candidate_data = seed.model_dump(mode="python")
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
                    "backend": family_fit.optimizer_backend_for_method(
                        bundle.fit_config.optimizer.method
                    ),
                    "success": True,
                },
                "input_hashes": {},
                "primitive": candidate.model_dump(mode="json"),
                "visual_review": {"status": "promotion_pending"},
            }
        )
    return reports


def _compile_bundle(bundle, output_dir):
    project = bundle.project.model_copy(
        update={"compile": bundle.project.compile.model_copy(update={"output": "canonical.yaml"})}
    )
    return replace(bundle, project_dir=output_dir, project=project)


def _without_preview_provenance(model):
    provenance = dict(model.provenance)
    for key in ("preview_geometry", "canonical", "unpromoted", "primitive_geometry"):
        provenance.pop(key)
    return model.model_copy(update={"provenance": provenance})


def test_canonical_preview_matches_compiler_policy_without_writing_model(monkeypatch, tmp_path):
    bundle = load_fit_project(PROJECT)
    reports = _reports(bundle)
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "validate_fit_report_inputs", lambda *_: None)
    model_path = ROOT / "data" / "models" / "sas-1968.yaml"
    original_model = model_path.read_bytes()

    built = build_fit_preview_model(bundle, geometry="canonical")
    compiled = compiler.compile_fit_project(
        _compile_bundle(bundle, tmp_path), require_promoted=False
    )

    assert _without_preview_provenance(built.model) == compiled.model
    assert built.model.provenance["canonical"] is True
    assert built.model.provenance["unpromoted"] is False
    assert all(
        source == "seed-fallback" for source in built.diagnostics["primitive_geometry"].values()
    )
    preview_path = tmp_path / "canonical-preview.yaml"
    preview_path.write_text(
        yaml.safe_dump(
            built.model.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    assert load_model(preview_path) == built.model
    assert model_path.read_bytes() == original_model


def test_latest_candidate_preview_uses_candidates_without_promoting_reports(
    monkeypatch
):
    bundle = load_fit_project(PROJECT)
    reports = _reports(bundle)
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "validate_fit_report_inputs", lambda *_: None)
    model_path = ROOT / "data" / "models" / "sas-1968.yaml"
    original_model = model_path.read_bytes()

    canonical = build_fit_preview_model(bundle, geometry="canonical")
    candidate = build_fit_preview_model(bundle, geometry="latest-candidate")

    assert candidate.model.glyphs != canonical.model.glyphs
    assert candidate.model.provenance["canonical"] is False
    assert candidate.model.provenance["unpromoted"] is True
    assert candidate.model.provenance["primitive_geometry"] == {
        **{
            primitive_id: "latest-candidate"
            for primitive_id in CANDIDATE_PRIMITIVES
        },
        **{
            primitive_id: "seed-fallback"
            for primitive_id in set(candidate.diagnostics["primitive_geometry"])
            - CANDIDATE_PRIMITIVES
        },
    }
    assert all(
        report["visual_review"]["status"] == "promotion_pending" for report in reports
    )
    assert all(
        detail["visual_review_status"] == "promotion_pending"
        for detail in candidate.diagnostics["fit_reports"].values()
    )
    assert model_path.read_bytes() == original_model


def test_latest_candidate_rejects_stale_objective_report(monkeypatch):
    bundle = load_fit_project(PROJECT)
    reports = _reports(bundle)
    reports[0]["format_version"] = 3
    reports[0]["objective_version"] = 3
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _: reports)
    monkeypatch.setattr(preview, "load_fit_reports", lambda _: reports)

    with pytest.raises(ValueError, match="latest candidate.*stale: report format_version=3"):
        build_fit_preview_model(bundle, geometry="latest-candidate")


def test_latest_candidate_requires_reports_for_all_active_families(monkeypatch):
    bundle = load_fit_project(PROJECT)
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _: [])
    monkeypatch.setattr(preview, "load_fit_reports", lambda _: [])

    with pytest.raises(ValueError, match="latest candidate.*missing a current fit report"):
        build_fit_preview_model(bundle, geometry="latest-candidate")


def test_fit_report_input_freshness_is_reusable_and_rejects_stale_hashes(monkeypatch):
    bundle = load_fit_project(PROJECT)
    expected = CheckpointInputs(
        project_sha256="0" * 64,
        corpus_sha256="1" * 64,
        source_lock_sha256="2" * 64,
        support_sha256={},
    )
    monkeypatch.setattr(family_fit, "_input_hashes", lambda *_: expected)
    report = {"input_hashes": expected.model_dump(mode="json")}

    family_fit.validate_fit_report_inputs(bundle, "sas-underturn-01", report)

    stale = expected.model_copy(update={"corpus_sha256": "3" * 64})
    with pytest.raises(ValueError, match="stale: input hashes do not match"):
        family_fit.validate_fit_report_inputs(
            bundle,
            "sas-underturn-01",
            {"input_hashes": stale.model_dump(mode="json")},
        )
