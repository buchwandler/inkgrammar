from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from inkgrammar.dataset.models import (
    ConnectionAnnotation,
    FamilyAssignment,
    Observation,
    ObservationAnnotation,
    ObservationExtraction,
    ObservationManifest,
    ObservationMetrics,
    ReviewedInt,
    ReviewedPoint,
    ReviewedRegion,
    ReviewedScalar,
    SourceImage,
    StrokeAnnotation,
    migrate_manifest_kinds,
)
from inkgrammar.dataset.transforms import (
    ObservationFrame,
    glyph_units_to_source_px,
    source_px_to_glyph_units,
    standard_frame,
)


def _observation(**overrides):
    data = {
        "id": "sas1968-alphabet-U+0065",
        "kind": "glyph-isolated",
        "label": "e",
        "unicode": ["U+0065"],
        "source_id": "sas-1968-alphabet",
        "source_bbox": (10, 20, 30, 40),
        "label_source": "declared-layout-sequence",
        "review_status": "accepted",
    }
    data.update(overrides)
    return Observation.model_validate(data)


def test_observation_extended_blocks_parse():
    obs = _observation(
        isolation_status="usable-reviewed",
        extraction={
            "extractor": "labeled-rows-v1",
            "extractor_version": 1,
            "source_bbox_policy": "declared-row-sequence",
            "layout_id": "sas-1968-alphabet-page",
            "layout_region_id": "lowercase-1",
            "detection_confidence": 0.9,
            "artifact_hashes": {"crop": "abc"},
        },
        metrics={
            "baseline_y": 0.9,
            "x_height_y": 0.4,
            "normalized_bbox": (0.1, 0.2, 0.3, 0.4),
            "metric_status": "automatic-candidate",
        },
        relations={
            "member_labels": ["c", "e"],
            "connection_category": "baseline-origin",
            "left_label": "b",
            "right_label": "a",
            "sequence_text": "ba",
        },
    )

    assert obs.extraction.extractor == "labeled-rows-v1"
    assert obs.metrics.metric_status == "automatic-candidate"
    assert obs.relations.sequence_text == "ba"


def test_observation_legacy_kind_aliases_are_accepted():
    obs = _observation(kind="movement")
    assert obs.kind == "movement-demonstration"

    obs = _observation(kind="connection")
    assert obs.kind == "connection-pair"


def test_observation_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        _observation(kind="glyph-isolated-v2")


def test_observation_bbox_validator_unchanged():
    with pytest.raises(ValidationError):
        _observation(source_bbox=(30, 20, 10, 40))


def test_manifest_validates_references():
    source = SourceImage(
        source_id="sas-1968-alphabet",
        filename="a.png",
        sha256="0" * 64,
        width_px=100,
        height_px=100,
        rights_status="unknown-needs-review",
    )
    item = _observation().model_dump(mode="json")

    with pytest.raises(ValidationError, match="duplicate source id"):
        ObservationManifest.model_validate(
            {"model_id": "sas-1968", "sources": [source, source], "items": [item]}
        )
    with pytest.raises(ValidationError, match="outside source"):
        ObservationManifest.model_validate(
            {
                "model_id": "sas-1968",
                "sources": [source],
                "items": [{**item, "source_bbox": [10, 20, 300, 40]}],
            }
        )


def test_annotation_schema_with_point_and_region_evidence():
    annotation = ObservationAnnotation.model_validate(
        {
            "observation_id": "sas1968-alphabet-U+0065",
            "annotation_version": 1,
            "review_status": "candidate",
            "baseline": {"value": 0.9, "evidence": "automatic-candidate"},
            "x_height": {"value": 0.45, "evidence": "inferred-reviewed"},
            "stroke_count": {"value": 1, "evidence": "inferred-reviewed"},
            "strokes": [
                {
                    "id": "body",
                    "order": 0,
                    "pen_state": "down",
                    "start": {
                        "point": {"x": 0.11, "y": 0.04},
                        "evidence": "inferred-reviewed",
                        "confidence": "medium",
                    },
                    "end": {"bbox": [0.6, 0.0, 0.75, 0.1], "evidence": "inferred-reviewed"},
                    "direction_status": "inferred-reviewed",
                    "family_hint": "underturn",
                }
            ],
            "family_assignments": [
                {"family_id": "sas1968-family-01", "evidence": "source-family-example"}
            ],
            "connection_annotation": {
                "left_label": "b",
                "right_label": "e",
                "category_id": "baseline-origin",
                "source_label": "Die einzügige Verbindung von der Grundlinie aus",
            },
        }
    )

    stroke = annotation.strokes[0]
    assert isinstance(stroke.start, ReviewedPoint)
    assert isinstance(stroke.end, ReviewedRegion)
    assert annotation.stroke_count == ReviewedInt(value=1, evidence="inferred-reviewed")
    assert annotation.baseline == ReviewedScalar(value=0.9, evidence="automatic-candidate")
    assert annotation.family_assignments[0] == FamilyAssignment(
        family_id="sas1968-family-01", evidence="source-family-example", notes=None
    )
    assert isinstance(annotation.connection_annotation, ConnectionAnnotation)


def test_annotation_rejects_unknown_evidence():
    with pytest.raises(ValidationError):
        StrokeAnnotation.model_validate(
            {
                "id": "body",
                "order": 0,
                "pen_state": "down",
                "direction_status": "source-verified",
            }
        )


def test_observation_metrics_require_declared_status():
    metrics = ObservationMetrics()
    assert metrics.metric_status == "missing"
    with pytest.raises(ValidationError):
        ObservationMetrics(metric_status="reviewed-by-guess")


def test_observation_extraction_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        ObservationExtraction(
            extractor="region-layout-v1",
            extractor_version=1,
            source_bbox_policy="declared",
            detection_confidence=1.5,
        )


def test_migrate_manifest_kinds_rewrites_only_legacy_kinds():
    data = {
        "model_id": "sas-1968",
        "items": [
            {"id": "a", "kind": "movement"},
            {"id": "b", "kind": "connection"},
            {"id": "c", "kind": "glyph-isolated"},
        ],
    }

    migrated = migrate_manifest_kinds(data)

    assert [item["kind"] for item in migrated["items"]] == [
        "movement-demonstration",
        "connection-pair",
        "glyph-isolated",
    ]
    assert [item["kind"] for item in data["items"]] == ["movement", "connection", "glyph-isolated"]


def test_observation_frame_round_trip():
    frame = standard_frame(origin_source_px=(100.0, 200.0), x_height_px=50.0)

    points = np.array([[100.0, 200.0], [150.0, 150.0], [80.0, 210.0]])
    glyph = source_px_to_glyph_units(points, frame)
    round_trip = glyph_units_to_source_px(glyph, frame)

    np.testing.assert_allclose(round_trip, points)
    np.testing.assert_allclose(glyph[0], [0.0, 0.0])
    np.testing.assert_allclose(glyph[1], [1.0, 1.0])


def test_standard_frame_maps_baseline_and_x_height():
    frame = standard_frame(origin_source_px=(0.0, 200.0), x_height_px=100.0)

    baseline = glyph_units_to_source_px(np.array([0.5, 0.0]), frame)
    x_height = glyph_units_to_source_px(np.array([0.5, 1.0]), frame)

    assert baseline[1] == pytest.approx(200.0)
    assert x_height[1] == pytest.approx(100.0)  # y upward against image y down


def test_observation_frame_rejects_degenerate_axes():
    with pytest.raises(ValidationError):
        ObservationFrame(
            origin_source_px=(0.0, 0.0),
            x_axis_source_px=(1.0, 0.0),
            y_axis_source_px=(2.0, 0.0),
            x_height_px=10.0,
        )
    with pytest.raises(ValidationError):
        ObservationFrame(
            origin_source_px=(0.0, 0.0),
            x_axis_source_px=(1.0, 0.0),
            y_axis_source_px=(0.0, -1.0),
            x_height_px=0.0,
        )


def test_observation_frame_rotated_axes_round_trip():
    frame = ObservationFrame(
        origin_source_px=(10.0, 10.0),
        x_axis_source_px=(0.6, 0.8),
        y_axis_source_px=(-0.8, 0.6),
        x_height_px=25.0,
    )
    points = np.array([[0.0, 0.0], [1.5, -0.5], [-2.0, 3.0]])

    glyph = source_px_to_glyph_units(glyph_units_to_source_px(points, frame), frame)

    np.testing.assert_allclose(glyph, points, atol=1e-9)
