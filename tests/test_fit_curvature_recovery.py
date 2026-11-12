from __future__ import annotations

import numpy as np

from inkgrammar.fitting.curvature import evaluate_curvature_stroke
from inkgrammar.fitting.curvature_fit import fit_curvature_stroke
from inkgrammar.fitting.models import CurvatureStrokeGeometry

TRUE_GEOMETRY = CurvatureStrokeGeometry.model_validate(
    {
        "length": 1.0,
        "start_angle_deg": 40.0,
        "curvature": {
            "interpolation": "cubic-hermite",
            "start_rate_deg": 30.0,
            "changes": [
                {"at": 0.25, "rate_deg": 260.0},
                {"at": 0.55, "rate_deg": 210.0},
                {"at": 0.8, "rate_deg": -40.0},
            ],
            "end_rate_deg": 15.0,
        },
    }
)


def _noisy_target(*, noise: float, seed: int) -> np.ndarray:
    truth = evaluate_curvature_stroke(TRUE_GEOMETRY, count=257).xy
    rng = np.random.default_rng(seed)
    return truth + rng.normal(0.0, noise, size=truth.shape)


def test_noisy_synthetic_recovery():
    target = _noisy_target(noise=0.002, seed=7)

    result = fit_curvature_stroke(target, change_count=3, restarts=2, seed=0)

    assert result.success
    assert result.metrics["median_position_error"] <= 0.008
    assert result.metrics["p95_position_error"] <= 0.02
    assert result.metrics["endpoint_tangent_error_deg"] <= 5.0
    assert result.metrics["total_turn_error_deg"] <= 3.0

    assert abs(result.geometry.start_angle_deg - 40.0) <= 6.0
    # control points are not uniquely identifiable (rates compensate locally);
    # the identifiable parameters are the start angle and the total turn
    fitted_positions = [change.at for change in result.geometry.curvature.changes]
    assert fitted_positions == sorted(fitted_positions)
    assert all(0.0 < position < 1.0 for position in fitted_positions)


def test_recovery_without_noise_matches_geometry():
    target = _noisy_target(noise=0.0, seed=0)

    result = fit_curvature_stroke(target, change_count=3, restarts=1, seed=0)

    assert result.metrics["median_position_error"] <= 0.003
    assert result.metrics["endpoint_error"] <= 0.005
    assert result.metrics["total_turn_error_deg"] <= 1.0
    assert abs(result.geometry.start_angle_deg - 40.0) <= 3.0
