from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from inkgrammar.dataset.review import (
    ReviewPatch,
    apply_review_patch,
    extraction_sha256,
    load_review_patches,
)

SOURCE_SHA = "a" * 64


def _automatic_record():
    return {
        "id": "sas1968-alphabet-U+0065",
        "kind": "glyph-isolated",
        "isolation_status": None,
        "fitting_status": "pending",
        "metrics": {"baseline_y": 0.9, "x_height_y": 0.4, "metric_status": "automatic-candidate"},
        "artifact_hashes": {"crop": "b" * 64},
    }


def _patch(record, *, extraction_sha=None, **set_fields):
    return ReviewPatch.model_validate(
        {
            "observation_id": record["id"],
            "input_hash": {
                "source_sha256": SOURCE_SHA,
                "extraction_sha256": extraction_sha or extraction_sha256(record),
            },
            "set": set_fields,
        }
    )


def test_extraction_sha256_ignores_review_state_but_not_artifacts():
    record = _automatic_record()
    digest = extraction_sha256(record)

    assert extraction_sha256({**record, "review_patch": {"status": "applied"}}) == digest
    assert extraction_sha256({**record, "isolation_status": "usable"}) == digest
    changed = {**record, "artifact_hashes": {"crop": "c" * 64}}
    assert extraction_sha256(changed) != digest


def test_apply_review_patch_composes_effective_record():
    record = _automatic_record()
    patch = _patch(
        record,
        isolation_status="usable-reviewed",
        baseline_y=0.95,
        stroke_count={"value": 1, "evidence": "inferred-reviewed"},
        reviewer_notes="checked against scan",
    )

    effective, status = apply_review_patch(record, patch, source_sha256=SOURCE_SHA)

    assert status == "applied"
    assert effective["isolation_status"] == "usable-reviewed"
    assert effective["metrics"]["baseline_y"] == 0.95
    assert effective["metrics"]["metric_status"] == "reviewed"
    assert effective["annotation"]["stroke_count"] == {
        "value": 1,
        "evidence": "inferred-reviewed",
    }
    assert effective["annotation"]["reviewer_notes"] == "checked against scan"
    assert record["isolation_status"] is None  # automatic record untouched


def test_apply_review_patch_rejects_stale_input_hash():
    record = _automatic_record()
    stale_extraction = _patch(record, extraction_sha="f" * 64, isolation_status="usable-reviewed")
    stale_source = _patch(record, isolation_status="usable-reviewed")

    effective, status = apply_review_patch(record, stale_extraction, source_sha256=SOURCE_SHA)
    assert status == "stale"
    assert effective["review_patch"]["status"] == "stale"
    assert effective["isolation_status"] is None

    effective, status = apply_review_patch(record, stale_source, source_sha256="b" * 64)
    assert status == "stale"
    assert effective["isolation_status"] is None


def test_load_review_patches_rejects_duplicates(tmp_path):
    patch = _patch(_automatic_record(), isolation_status="usable-reviewed")
    (tmp_path / "one.yaml").write_text(patch.model_dump_json(), encoding="utf-8")
    assert load_review_patches(tmp_path)["sas1968-alphabet-U+0065"].set.isolation_status == (
        "usable-reviewed"
    )
    (tmp_path / "two.yaml").write_text(patch.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate review patch"):
        load_review_patches(tmp_path)


def test_review_patch_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ReviewPatch.model_validate(
            {
                "observation_id": "x",
                "input_hash": {
                    "source_sha256": SOURCE_SHA,
                    "extraction_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "set": {"arbitrary_geometry": [1, 2, 3]},
            }
        )
