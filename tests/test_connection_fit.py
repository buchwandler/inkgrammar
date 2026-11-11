from __future__ import annotations

import pytest

from inkgrammar.fitting.connection_fit import (
    ConnectionFitWeights,
    connection_observation_coverage,
    diagnose_connection,
    fit_connection_observation,
)
from inkgrammar.fitting.models import (
    ConnectionGeometryTarget,
    ConnectionObservation,
    FitSample,
    GrammarConnectionRule,
)
from inkgrammar.fitting.workspace import _validate_connection_observations
from inkgrammar.models import RuntimeAnchor
from inkgrammar.observations import Observation


def observation(split="fit"):
    return ConnectionObservation(
        id=f"join-{split}",
        text="ab",
        boundary_index=1,
        observation_id=f"source-{split}",
        split=split,
        from_character="a",
        to_character="b",
        target_geometry=ConnectionGeometryTarget(
            anchor_delta=(1.5, -2.0),
            exit_tangent=(1.0, 0.0),
            entry_tangent=(1.0, 0.0),
            exit_width=4.0,
            entry_width=4.0,
            centerline=[(11.5, 0.0), (12.5, 0.0)],
        ),
    )


def endpoints():
    return (
        RuntimeAnchor(
            x=10,
            y=2,
            tangent_x=1,
            tangent_y=0,
            class_="baseline",
            width=4,
        ),
        RuntimeAnchor(
            x=0,
            y=0,
            tangent_x=1,
            tangent_y=0,
            class_="baseline",
            width=4,
        ),
    )


def test_connection_observation_fit_recovers_synthetic_anchor_adjustment():
    left, right = endpoints()
    record = observation()

    fitted = fit_connection_observation(
        record,
        left,
        right,
        generated_centerline=[(0.0, 0.0), (1.0, 0.0)],
        weights=ConnectionFitWeights(rule_prior=0),
    )

    assert fitted.x_adjust == pytest.approx(11.5, abs=1e-5)
    assert fitted.y_adjust == pytest.approx(0, abs=1e-5)
    assert fitted.objective == pytest.approx(0, abs=1e-8)
    assert fitted.diagnostics["anchor_target_error"] == pytest.approx(0, abs=1e-5)
    assert fitted.diagnostics["tangent_target_error_deg"] == pytest.approx(0)
    assert fitted.diagnostics["width_target_error"] == pytest.approx(0)
    assert fitted.to_dict()["observation_id"] == record.id
    assert fitted.rule_proposal["from_class"] == "baseline"
    assert fitted.rule_proposal["x_adjust"] == pytest.approx(11.5)
    assert GrammarConnectionRule.model_validate(fitted.rule_proposal).mode == "cursive-anchor"


def test_connection_diagnostics_report_gap_tangent_width_and_font_gate():
    left, right = endpoints()

    result = diagnose_connection(
        left,
        right,
        x_adjust=10,
        y_adjust=2,
        mode="synthetic-bridge",
        rule_id="bridge",
        target=observation().target_geometry,
    )

    assert result["rule_id"] == "bridge"
    assert result["anchor_gap"] == pytest.approx(0)
    assert result["tangent_error_deg"] == pytest.approx(0)
    assert result["width_mismatch"] == pytest.approx(0)
    assert result["gap_error"] == pytest.approx(2.5)
    assert result["font_compatible"] is False


def test_connection_fit_is_fit_split_only_and_coverage_keeps_splits_separate():
    left, right = endpoints()
    records = [observation(split) for split in ("fit", "validation", "hold-out")]

    with pytest.raises(ValueError, match="fit-split"):
        fit_connection_observation(records[1], left, right)

    report = connection_observation_coverage(records)
    assert report["split_counts"] == {"fit": 1, "validation": 1, "hold-out": 1}
    assert report["pair_splits"]["ab"] == {"fit": 1, "validation": 1, "hold-out": 1}
    assert report["source_free"] is True
    assert all("source_id" not in row for row in report.values() if isinstance(row, dict))


def test_workspace_rejects_connection_split_mismatch():
    connection = observation("fit")
    source_observation = Observation(
        id=connection.observation_id,
        kind="connection",
        label="ab",
        source_id="source",
        source_bbox=(0, 0, 20, 20),
        label_source="reviewed-caption",
        review_status="accepted",
    )
    sample = FitSample(
        id="connection-sample",
        kind="word",
        text="ab",
        observation_id=connection.observation_id,
        split="hold-out",
    )

    with pytest.raises(ValueError, match="split mismatch"):
        _validate_connection_observations(
            [connection], {source_observation.id: source_observation}, [sample]
        )
