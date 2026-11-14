from __future__ import annotations

import numpy as np

from inkgrammar.fitting.curvature import (
    evaluate_curvature_stroke,
    evaluate_stroke_at,
    trajectory_total_turn,
)
from inkgrammar.fitting.curvature_fit import fit_shared_profile
from inkgrammar.fitting.models import CurvatureStrokeGeometry

TRUE_GEOMETRY = CurvatureStrokeGeometry.model_validate(
    {
        "length": 1.0,
        "start_angle_deg": 20.0,
        "curvature": {
            "interpolation": "cubic-hermite",
            "start_rate_deg": 40.0,
            "changes": [
                {"at": 0.3, "rate_deg": 240.0},
                {"at": 0.6, "rate_deg": 180.0},
            ],
            "end_rate_deg": 10.0,
        },
    }
)


def _transformed_occurrence(scale: float, rotation_deg: float, dx: float, dy: float, *, seed: int):
    points = evaluate_curvature_stroke(TRUE_GEOMETRY, count=257).xy
    angle = np.deg2rad(rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rng = np.random.default_rng(seed)
    return (
        scale * (points @ rotation.T)
        + np.array([dx, dy])
        + rng.normal(0.0, 0.0015, size=points.shape)
    )


def _mapped_error(fit, transform, target: np.ndarray) -> float:
    u = np.linspace(0.0, 1.0, target.shape[0])
    fit_xy, _ = evaluate_stroke_at(fit, u)
    angle = np.deg2rad(transform.rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mapped = transform.scale * (fit_xy @ rotation.T) + np.array([transform.dx, transform.dy])
    return float(np.median(np.linalg.norm(mapped - target, axis=1)))


def test_shared_profile_recovers_across_transformed_occurrences():
    occurrences = [
        _transformed_occurrence(0.7, -20.0, 3.0, -1.5, seed=1),
        _transformed_occurrence(1.3, 35.0, -2.0, 4.0, seed=2),
        _transformed_occurrence(1.0, 5.0, 0.0, 0.0, seed=3),
    ]

    result = fit_shared_profile(occurrences, change_count=2, restarts=1, seed=0)

    assert result.success
    assert len(result.transforms) == 3

    # one shared turning program must explain every occurrence geometrically;
    # exact control-point recovery is not identifiable because rates compensate
    total_turn = trajectory_total_turn(result.geometry)
    true_turn = trajectory_total_turn(TRUE_GEOMETRY)
    assert abs(total_turn - true_turn) <= 5.0

    for transform, target in zip(result.transforms, occurrences, strict=True):
        assert _mapped_error(result.geometry, transform, target) <= 0.01


def test_shared_profile_scale_ratios_match_occurrences():
    occurrences = [
        _transformed_occurrence(0.5, 0.0, 0.0, 0.0, seed=4),
        _transformed_occurrence(2.0, 0.0, 0.0, 0.0, seed=5),
    ]

    result = fit_shared_profile(occurrences, change_count=2, restarts=0, seed=0)

    scales = [transform.scale for transform in result.transforms]
    assert abs(scales[1] / scales[0] - 4.0) / 4.0 <= 0.1
