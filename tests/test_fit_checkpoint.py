import json

import pytest
from pydantic import ValidationError

from inkgrammar.fitting.checkpoint import (
    CheckpointIteration,
    CheckpointProgress,
    CheckpointTraceSummary,
    FitCheckpoint,
    load_checkpoint,
    save_checkpoint,
    summarize_trace,
    validate_checkpoint_objective_version,
)


def checkpoint():
    digest = "a" * 64
    return FitCheckpoint(
        project_id="sas-1968",
        stage="primitive:sas-open-oval-01",
        created_by={"inkgrammar_version": "0.1.0", "git_commit": "abc123"},
        inputs={
            "project_sha256": digest,
            "corpus_sha256": "b" * 64,
            "source_lock_sha256": "c" * 64,
        },
        optimizer={"backend": "scipy-powell", "seed": 1968, "objective_version": 1},
        parameters={"segments.0.c1.x": 0.2},
        training_metrics={"loss": 0.15},
        holdout_metrics={"loss": 0.2},
        progress={"restart_index": 2, "cycle_index": 1, "evaluations": 8},
        trace_summary={
            "evaluations": 8,
            "initial_objective": 0.5,
            "best_objective": 0.15,
            "last_objective": 0.2,
        },
    )


def legacy_payload():
    payload = checkpoint().model_dump(mode="json")
    payload.pop("progress")
    payload.pop("trace_summary")
    payload["format_version"] = 1
    payload["history"] = [
        {"iteration": 0, "objective": 0.5, "parameters": {"segments.0.c1.x": 0.1}},
        {"iteration": 1, "objective": 0.15, "parameters": {"segments.0.c1.x": 0.2}},
        {"iteration": 2, "objective": 0.2, "parameters": {"segments.0.c1.x": 0.2}},
    ]
    return payload


def test_checkpoint_v2_roundtrip_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "nested" / "second.json"

    save_checkpoint(checkpoint(), first)
    save_checkpoint(checkpoint(), second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["format_version"] == 2
    assert load_checkpoint(first) == checkpoint()


def test_checkpoint_rejects_invalid_source_hashes():
    payload = checkpoint().model_dump(mode="json")
    payload["inputs"]["project_sha256"] = "not-a-hash"

    with pytest.raises(ValidationError, match="project_sha256"):
        FitCheckpoint.model_validate(payload)


def test_checkpoint_objective_versions_cannot_be_mixed():
    checkpoint_value = checkpoint()
    validate_checkpoint_objective_version(checkpoint_value, expected_version=1)

    with pytest.raises(ValueError, match="incompatible with requested objective version 2"):
        validate_checkpoint_objective_version(checkpoint_value, expected_version=2)


def test_v1_checkpoint_migrates_without_changing_accepted_parameters(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_payload()), encoding="utf-8")

    migrated = load_checkpoint(path)

    assert migrated.format_version == 2
    assert migrated.parameters == {"segments.0.c1.x": 0.2}
    assert migrated.progress.evaluations == 3
    assert migrated.trace_summary == CheckpointTraceSummary(
        evaluations=3,
        initial_objective=0.5,
        best_objective=0.15,
        last_objective=0.2,
    )
    assert not hasattr(migrated, "history")


def test_checkpoint_does_not_serialize_per_evaluation_parameter_history(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(checkpoint(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "history" not in payload
    assert payload["parameters"] == checkpoint().parameters


def test_checkpoint_size_is_bounded_for_synthetic_5000_evaluation_trace(tmp_path):
    history = [
        CheckpointIteration(
            iteration=index,
            objective=float(5000 - index),
            parameters={
                f"parameter.{parameter}": float(index + parameter) for parameter in range(8)
            },
        )
        for index in range(5000)
    ]
    trace_summary = summarize_trace(history)
    value = checkpoint().model_copy(
        update={
            "progress": CheckpointProgress(restart_index=3, cycle_index=2, evaluations=5000),
            "trace_summary": trace_summary,
        }
    )
    path = tmp_path / "checkpoint.json"
    save_checkpoint(value, path)

    assert path.stat().st_size < 20 * 1024
    assert json.loads(path.read_text(encoding="utf-8"))["trace_summary"]["evaluations"] == 5000
