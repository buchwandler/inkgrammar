from __future__ import annotations

import numpy as np
import pytest

from inkgrammar.fitting.curvature import evaluate_curvature_stroke
from inkgrammar.fitting.models import CurvatureStrokeGeometry

PROFILE = {
    "interpolation": "cubic-hermite",
    "start_rate_deg": 40.0,
    "changes": [
        {"at": 0.25, "rate_deg": 260.0},
        {"at": 0.55, "rate_deg": 200.0},
        {"at": 0.8, "rate_deg": -80.0},
    ],
    "end_rate_deg": 10.0,
}


def _geometry(length: float, start_angle_deg: float = 25.0) -> CurvatureStrokeGeometry:
    return CurvatureStrokeGeometry.model_validate(
        {"length": length, "start_angle_deg": start_angle_deg, "curvature": PROFILE}
    )


@pytest.mark.parametrize("length", [0.5, 1.0, 2.0, 4.0])
def test_scale_changes_size_but_not_normalized_shape(length):
    reference = evaluate_curvature_stroke(_geometry(1.0), count=257)
    scaled = evaluate_curvature_stroke(_geometry(length), count=257)

    np.testing.assert_allclose(scaled.xy / length, reference.xy, atol=1e-9)
    np.testing.assert_allclose(scaled.heading_deg, reference.heading_deg, atol=1e-9)


def test_rotation_rotates_the_full_trajectory():
    base = evaluate_curvature_stroke(_geometry(1.0, start_angle_deg=0.0), count=129)
    rotated = evaluate_curvature_stroke(_geometry(1.0, start_angle_deg=30.0), count=129)

    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(rotated.xy, base.xy @ rotation.T, atol=1e-9)
    np.testing.assert_allclose(rotated.heading_deg, base.heading_deg + 30.0, atol=1e-9)


def test_end_pose_scales_linearly():
    small = evaluate_curvature_stroke(_geometry(0.5), count=257).end_pose
    large = evaluate_curvature_stroke(_geometry(2.0), count=257).end_pose

    np.testing.assert_allclose(large.position, small.position * 4.0, atol=1e-9)
    assert large.heading_deg == pytest.approx(small.heading_deg)
