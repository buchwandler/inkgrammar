from __future__ import annotations

import copy

from test_dataset_materialize import _materialize_fixture

from inkgrammar.dataset.materialize import materialize_dataset
from inkgrammar.dataset.project import load_dataset_project, verify_sources
from inkgrammar.dataset.status import render_report, summarize_status, validate_dataset


def _materialized(tmp_path):
    project = _materialize_fixture(tmp_path / "run")
    bundle = load_dataset_project(project)
    root = tmp_path / "run" / "out"
    index = materialize_dataset(bundle, root)
    return bundle, root, index


def _validation(bundle, root, index, tamper=None):
    if tamper is not None:
        index = tamper(copy.deepcopy(index))
    return validate_dataset(bundle, index, dataset_root=root, verification=verify_sources(bundle))


def test_status_counters_are_reported_independently(tmp_path):
    bundle, root, index = _materialized(tmp_path)
    validation = _validation(bundle, root, index)
    status = summarize_status(bundle, index, validation, verify_sources(bundle), dataset_root=root)

    assert status.source_pages_verified == 1
    assert status.source_pages_total == 1
    assert status.observations_declared == 2
    assert status.observations_materialized == 2
    assert status.label_reviewed == 1
    assert status.by_kind == {"glyph-isolated": 1, "connection-pair": 1}
    assert status.connection_examples_transcribed == 1
    assert status.ready_for_fitting == 0  # everything is candidate until review


def test_validate_clean_materialization_has_no_errors(tmp_path):
    bundle, root, index = _materialized(tmp_path)

    validation = _validation(bundle, root, index)

    assert validation.ok, validation.errors


def test_validate_reports_stale_patch_and_missing_artifact(tmp_path):
    bundle, root, index = _materialized(tmp_path)

    def tamper(data):
        record = data["observations"]["sas1968-alpha-01"]
        record["review_patch"] = {"status": "stale"}
        (root / record["artifacts"]["clean_mask"]).unlink()
        return data

    validation = _validation(bundle, root, index, tamper)

    rules = {issue.rule for issue in validation.errors}
    assert "stale-review-patch" in rules
    assert "accepted-without-materialized-artifact" in rules


def test_validate_reports_frame_and_sequence_problems(tmp_path):
    bundle, root, index = _materialized(tmp_path)

    def tamper(data):
        alpha = data["observations"]["sas1968-alpha-01"]
        alpha["fitting_status"] = "usable-reviewed"
        alpha["metrics"]["x_height_y"] = None
        pair = data["observations"]["synthetic-pair-01"]
        pair["relations"]["left_label"] = "x"
        pair["relations"]["sequence_text"] = "xy"
        return data

    validation = _validation(bundle, root, index, tamper)

    rules = {issue.rule for issue in validation.errors}
    assert "fitting-usable-without-coordinate-frame" in rules
    assert "sequence-relation-disagreement" in rules


def test_validate_reports_source_and_layout_problems(tmp_path):
    bundle, root, index = _materialized(tmp_path)

    def tamper(data):
        record = data["observations"]["synthetic-pair-01"]
        record["source_id"] = "src-unknown"
        record["extraction"]["layout_region_id"] = "missing-region"
        alpha = data["observations"]["sas1968-alpha-01"]
        alpha["source_bbox"] = [0, 0, 1000, 1000]
        return data

    validation = _validation(bundle, root, index, tamper)

    rules = {issue.rule for issue in validation.errors}
    assert "unknown-source-id" in rules
    assert "missing-layout-region" in rules
    assert "region-outside-source-bounds" in rules


def test_validate_reports_source_explicit_strokes_without_annotation_region(tmp_path):
    bundle, root, index = _materialized(tmp_path)

    def tamper(data):
        data["observations"]["synthetic-pair-01"]["annotation"] = {
            "strokes": [
                {
                    "id": "body",
                    "order": 0,
                    "pen_state": "down",
                    "direction_status": "source-explicit",
                }
            ]
        }
        return data

    validation = _validation(bundle, root, index, tamper)

    assert any(
        issue.rule == "source-explicit-stroke-order-without-annotation-region"
        for issue in validation.errors
    )


def test_validate_reports_warnings_for_masks_and_metrics(tmp_path):
    bundle, root, index = _materialized(tmp_path)

    def tamper(data):
        record = data["observations"]["synthetic-pair-01"]
        record["metrics"] = {"metric_status": "missing"}
        return data

    validation = _validation(bundle, root, index, tamper)

    rules = {issue.rule for issue in validation.warnings}
    assert "metric-confidence-low" in rules
    assert "mask-touching-crop-edge" in rules or "neighboring-ink-likely-present" in rules


def test_render_report_contains_coverage_tables(tmp_path):
    bundle, root, index = _materialized(tmp_path)
    validation = _validation(bundle, root, index)
    status = summarize_status(bundle, index, validation, verify_sources(bundle), dataset_root=root)

    report = render_report(status, validation)

    assert "source pages verified | 1/1" in report
    assert "ready for fitting | 0" in report
    assert "| connection-pair | 1 |" in report
