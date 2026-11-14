import json

import numpy as np
import pytest
import typer
import yaml

from inkgrammar.fitting.curvature_fit import CurvatureFitResult
from inkgrammar.fitting.models import CurvatureStrokeGeometry, TrajectoryTransform
from inkgrammar.trajectory_cli import (
    FitContext,
    _reviewed_trajectory_points,
    infer_candidate_route_from_topology,
    write_fit_outputs,
)


def test_shape_only_topology_keeps_routes_as_unreviewed_candidates():
    mask = np.zeros((15, 15), dtype=bool)
    mask[2:13, 7] = True
    mask[7, 2:13] = True

    candidate = infer_candidate_route_from_topology(mask)

    assert candidate.feasible_path_count > 1
    assert candidate.route_ambiguous is True
    assert candidate.orientation_ambiguous is True
    assert candidate.evidence_status == "inferred"
    assert candidate.physical_ductus_authoritative is False
    assert candidate.xy.ndim == 2 and candidate.xy.shape[1] == 2
    assert np.isfinite(candidate.xy).all()


def test_only_reviewed_annotations_supply_physical_trajectory_points():
    points = [[1.0, 2.0], [3.0, 4.0]]

    assert (
        _reviewed_trajectory_points({"review_status": "candidate", "trajectory_xy": points}) is None
    )
    np.testing.assert_array_equal(
        _reviewed_trajectory_points({"review_status": "reviewed", "trajectory_xy": points}),
        points,
    )
    with pytest.raises(typer.BadParameter, match="at least two xy points"):
        _reviewed_trajectory_points({"review_status": "reviewed", "trajectory_xy": [[1, 2]]})


def test_fit_outputs_record_reviewed_trajectory_evidence(tmp_path):
    context = FitContext(
        observation_id="synthetic-observation",
        record={"label": "synthetic", "source_id": "synthetic-source"},
        target_xy=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        stroke_annotation={"review_status": "reviewed"},
        trajectory_evidence_status="reviewed",
        input_hashes={"annotation_sha256": "reviewed-annotation"},
    )
    geometry = CurvatureStrokeGeometry.model_validate(
        {
            "length": 1.0,
            "start_angle_deg": 0.0,
            "curvature": {
                "interpolation": "linear",
                "start_rate_deg": 0.0,
                "changes": [],
                "end_rate_deg": 0.0,
            },
        }
    )
    result = CurvatureFitResult(
        geometry=geometry,
        transform=TrajectoryTransform(),
        success=True,
        cost=0.0,
        metrics={
            "median_position_error": 0.0,
            "p95_position_error": 0.0,
            "endpoint_error": 0.0,
            "endpoint_tangent_error_deg": 0.0,
        },
    )

    outputs = write_fit_outputs(context, result, tmp_path)

    observation = json.loads(outputs["observation"].read_text(encoding="utf-8"))
    parameters = yaml.safe_load(outputs["parameters"].read_text(encoding="utf-8"))
    assert observation["trajectory_evidence_status"] == "reviewed"
    assert parameters["trajectory_evidence_status"] == "reviewed"
