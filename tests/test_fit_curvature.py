from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from inkgrammar.fitting.models import (
    MIN_CHANGE_SPACING,
    CurvatureChange,
    CurvatureProfile,
    CurvatureStrokeGeometry,
    PrimitiveTemplate,
    TrajectoryTransform,
)


def _profile(**overrides):
    data = {
        "interpolation": "cubic-hermite",
        "start_rate_deg": 50.0,
        "changes": [
            {"at": 0.18, "rate_deg": 300.0},
            {"at": 0.38, "rate_deg": 520.0},
            {"at": 0.62, "rate_deg": 500.0},
            {"at": 0.82, "rate_deg": 240.0},
        ],
        "end_rate_deg": 40.0,
    }
    data.update(overrides)
    return data


def test_curvature_profile_accepts_ordered_changes():
    profile = CurvatureProfile.model_validate(_profile())
    assert [change.at for change in profile.changes] == [0.18, 0.38, 0.62, 0.82]
    assert profile.interpolation == "cubic-hermite"


@pytest.mark.parametrize(
    "changes",
    [
        [{"at": 0.5, "rate_deg": 1.0}, {"at": 0.5, "rate_deg": 2.0}],
        [{"at": 0.6, "rate_deg": 1.0}, {"at": 0.4, "rate_deg": 2.0}],
        [{"at": 0.4, "rate_deg": 1.0}, {"at": 0.4 + MIN_CHANGE_SPACING / 2, "rate_deg": 2.0}],
    ],
)
def test_curvature_profile_rejects_unordered_or_tight_changes(changes):
    with pytest.raises(ValidationError, match="strictly ordered"):
        CurvatureProfile.model_validate(_profile(changes=changes))


@pytest.mark.parametrize("at", [0.0, 1.0, -0.1, 1.5])
def test_curvature_change_requires_interior_position(at):
    with pytest.raises(ValidationError):
        CurvatureChange(at=at, rate_deg=10.0)


def test_curvature_rates_must_be_finite_and_bounded():
    with pytest.raises(ValidationError):
        CurvatureChange(at=0.5, rate_deg=float("inf"))
    with pytest.raises(ValidationError):
        CurvatureProfile.model_validate(_profile(start_rate_deg=float("nan")))
    with pytest.raises(ValidationError):
        CurvatureProfile.model_validate(_profile(end_rate_deg=1e9))


def test_curvature_stroke_geometry_requires_positive_length():
    stroke = CurvatureStrokeGeometry.model_validate(
        {"length": 1.4, "start_angle_deg": 55.0, "curvature": _profile()}
    )
    assert stroke.length == 1.4
    with pytest.raises(ValidationError):
        CurvatureStrokeGeometry.model_validate(
            {"length": 0.0, "start_angle_deg": 55.0, "curvature": _profile()}
        )


def test_trajectory_transform_is_similarity_only():
    transform = TrajectoryTransform()
    assert transform.scale == 1.0
    assert transform.rotation_deg == 0.0
    with pytest.raises(ValidationError):
        TrajectoryTransform(scale=0.0)
    with pytest.raises(ValidationError):
        TrajectoryTransform(scale=float("inf"))


def test_primitive_template_curvature_profile_rules():
    stroke = {"length": 1.0, "start_angle_deg": 0.0, "curvature": _profile()}
    template = PrimitiveTemplate.model_validate(
        {
            "id": "sas-e-trajectory-01",
            "kind": "curvature-profile",
            "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
            "curvature_stroke": stroke,
        }
    )
    assert template.kind == "curvature-profile"

    with pytest.raises(ValidationError, match="require curvature_stroke"):
        PrimitiveTemplate.model_validate(
            {
                "id": "x",
                "kind": "curvature-profile",
                "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
            }
        )
    with pytest.raises(ValidationError, match="must not contain segments or spline"):
        PrimitiveTemplate.model_validate(
            {
                "id": "x",
                "kind": "curvature-profile",
                "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
                "curvature_stroke": stroke,
                "segments": [
                    {
                        "c1": [0.1, 0.2],
                        "c2": [0.2, 0.2],
                        "end": [0.3, 0.0],
                    }
                ],
            }
        )


def _geometry(**overrides):
    data = {
        "length": 1.0,
        "start_angle_deg": 0.0,
        "curvature": {
            "interpolation": "linear",
            "start_rate_deg": 0.0,
            "changes": [],
            "end_rate_deg": 0.0,
        },
    }
    data.update(overrides)
    return CurvatureStrokeGeometry.model_validate(data)


def test_zero_heading_rate_produces_straight_line():
    from inkgrammar.fitting.curvature import evaluate_curvature_stroke

    sample = evaluate_curvature_stroke(_geometry(), count=64)

    np.testing.assert_allclose(sample.xy[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(sample.xy[:, 0], sample.u, atol=1e-12)
    assert sample.end_pose.heading_deg == pytest.approx(0.0)
    assert sample.end_pose.position[0] == pytest.approx(1.0)


def test_constant_heading_rate_produces_circular_arc():
    from inkgrammar.fitting.curvature import evaluate_curvature_stroke

    geometry = _geometry(
        curvature={
            "interpolation": "linear",
            "start_rate_deg": 90.0,
            "changes": [],
            "end_rate_deg": 90.0,
        }
    )
    sample = evaluate_curvature_stroke(geometry, count=257)

    # analytic circular arc: x = L/w sin(w u), y = L/w (1 - cos(w u)) in radians
    w = np.deg2rad(90.0)
    expected_x = np.sin(w * sample.u) / w
    expected_y = (1 - np.cos(w * sample.u)) / w
    np.testing.assert_allclose(sample.xy[:, 0], expected_x, atol=1e-4)
    np.testing.assert_allclose(sample.xy[:, 1], expected_y, atol=1e-4)
    assert sample.end_pose.heading_deg == pytest.approx(90.0, abs=1e-9)


def test_end_heading_equals_start_plus_total_turn():
    from inkgrammar.fitting.curvature import trajectory_end_pose, trajectory_total_turn

    geometry = _geometry(
        start_angle_deg=55.0,
        curvature={
            "interpolation": "cubic-hermite",
            "start_rate_deg": 50.0,
            "changes": [{"at": 0.4, "rate_deg": 300.0}, {"at": 0.7, "rate_deg": -100.0}],
            "end_rate_deg": 20.0,
        },
    )
    pose = trajectory_end_pose(geometry)
    total_turn = trajectory_total_turn(geometry)

    assert pose.heading_deg == pytest.approx(55.0 + total_turn, abs=1e-9)


def test_hermite_interpolates_keyframes_without_overshoot():
    from inkgrammar.fitting.curvature import evaluate_heading_rate

    profile = CurvatureProfile.model_validate(
        {
            "interpolation": "cubic-hermite",
            "start_rate_deg": 0.0,
            "changes": [{"at": 0.3, "rate_deg": 100.0}, {"at": 0.7, "rate_deg": 0.0}],
            "end_rate_deg": 0.0,
        }
    )
    rates = evaluate_heading_rate(profile, np.array([0.0, 0.3, 0.7, 1.0]))
    np.testing.assert_allclose(rates, [0.0, 100.0, 0.0, 0.0], atol=1e-9)

    dense = evaluate_heading_rate(profile, np.linspace(0.0, 1.0, 201))
    assert dense.max() <= 100.0 + 1e-9
    assert dense.min() >= -1e-9


def test_sample_curvature_stroke_resamples_by_arclength():
    from inkgrammar.fitting.curvature import sample_curvature_stroke

    points = sample_curvature_stroke(_geometry(length=2.0), samples=16)

    assert points.shape == (16, 2)
    assert points[0] == pytest.approx((0.0, 0.0))
    assert points[-1][0] == pytest.approx(2.0, abs=1e-9)


def test_evaluate_heading_requires_grid_starting_at_zero():
    from inkgrammar.fitting.curvature import evaluate_heading

    with pytest.raises(ValueError, match="start at 0.0"):
        evaluate_heading(_geometry(), np.array([0.1, 0.5]))
    with pytest.raises(ValueError, match="ascending"):
        evaluate_heading(_geometry(), np.array([0.0, 0.5, 0.2]))


def test_change_position_round_trip_preserves_order_and_spacing():
    from inkgrammar.fitting.trajectory_parameterization import (
        decode_change_positions,
        encode_change_positions,
    )

    logits = np.array([0.3, -1.2, 2.0, 0.7])
    positions = decode_change_positions(logits)

    assert positions.size == 3
    assert positions[0] > 0.0 and positions[-1] < 1.0
    assert np.all(np.diff(positions) >= MIN_CHANGE_SPACING - 1e-12)
    np.testing.assert_allclose(
        decode_change_positions(encode_change_positions(positions)),
        positions,
        atol=1e-9,
    )


@pytest.mark.parametrize("logits", [[-100.0, 100.0], [5.0, -5.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
def test_change_positions_are_ordered_by_construction(logits):
    from inkgrammar.fitting.trajectory_parameterization import decode_change_positions

    positions = decode_change_positions(np.asarray(logits))
    assert np.all(np.diff(positions) > 0)


def test_encode_rejects_positions_below_min_spacing():
    from inkgrammar.fitting.trajectory_parameterization import encode_change_positions

    with pytest.raises(ValueError, match="minimum spacing"):
        encode_change_positions(np.array([0.5, 0.51]))


def test_rate_angle_length_round_trips():
    from inkgrammar.fitting.trajectory_parameterization import (
        RATE_LIMIT_DEG,
        decode_angle,
        decode_length,
        decode_rate,
        encode_angle,
        encode_length,
        encode_rate,
    )

    assert decode_rate(encode_rate(123.0)) == pytest.approx(123.0, abs=1e-6)
    assert decode_angle(encode_angle(-215.0)) == pytest.approx(-215.0, abs=1e-6)
    assert decode_length(encode_length(2.5)) == pytest.approx(2.5, abs=1e-9)
    assert abs(decode_rate(1e6)) <= RATE_LIMIT_DEG
    assert decode_length(-1e6) > 0.0
    with pytest.raises(ValueError):
        encode_length(0.0)
