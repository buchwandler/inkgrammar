import json

import pytest
from pydantic import ValidationError

from inkgrammar.fitting.checkpoint import (
    FitCheckpoint,
    load_checkpoint,
    save_checkpoint,
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
        history=[
            {"iteration": 0, "objective": 0.5, "parameters": {"segments.0.c1.x": 0.1}},
            {"iteration": 1, "objective": 0.15, "parameters": {"segments.0.c1.x": 0.2}},
        ],
    )


def test_checkpoint_round_trips_as_portable_json_and_is_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "nested" / "second.json"

    save_checkpoint(checkpoint(), first)
    save_checkpoint(checkpoint(), second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["format"] == "inkgrammar-fit-checkpoint"
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
