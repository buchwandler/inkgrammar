import numpy as np
import pytest

from inkgrammar.fitting.models import BSplineGeometry, SplineEndpointPolicy
from inkgrammar.fitting.spline import spline_design_matrix
from inkgrammar.fitting.spline_fit import SplineObservation, solve_shared_spline_coefficients


def geometry_with_six_controls():
    return BSplineGeometry(
        knots=[0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0],
        control_points=[
            (0.0, 0.0),
            (1.0, 1.8),
            (2.3, 2.2),
            (3.8, 1.1),
            (5.0, 0.4),
            (6.0, 0.7),
        ],
        endpoint_policy=SplineEndpointPolicy(start="locked", end="locked"),
    )


def canonical_observations(geometry):
    rng = np.random.default_rng(73)
    controls = np.asarray(geometry.control_points)
    observations = []
    for count, matrix, offset in (
        (51, np.array([[1.2, 0.15], [0.0, 0.9]]), np.array([12.0, 5.0])),
        (43, np.array([[0.85, -0.1], [0.0, 1.1]]), np.array([-4.0, 8.0])),
    ):
        u = np.linspace(0.0, 1.0, count)
        xy = spline_design_matrix(u, geometry) @ controls
        observed = xy @ matrix.T + offset + rng.normal(0.0, 0.005, xy.shape)
        if count == 51:
            observed[count // 2] += [8.0, -6.0]
        canonical = (observed - offset) @ np.linalg.inv(matrix).T
        observations.append(SplineObservation(u=u, xy=canonical, weight=np.ones(count)))
    return observations


def test_sparse_irls_solver_recovers_shared_spline_with_fixed_transforms():
    geometry = geometry_with_six_controls()
    truth = np.asarray(geometry.control_points)
    seed = truth.copy()
    seed[1, 1] += 0.35
    seed[2, 0] -= 0.25
    seed[3, 1] += 0.2

    fitted = solve_shared_spline_coefficients(
        geometry,
        canonical_observations(geometry),
        seed_control_points=seed,
        smoothness_lambda=1e-5,
        seed_lambda=1e-6,
        robust_scale=0.1,
        irls_iterations=4,
    )
    fitted_again = solve_shared_spline_coefficients(
        geometry,
        canonical_observations(geometry),
        seed_control_points=seed,
        smoothness_lambda=1e-5,
        seed_lambda=1e-6,
        robust_scale=0.1,
        irls_iterations=4,
    )

    np.testing.assert_array_equal(fitted[[0, -1]], truth[[0, -1]])
    np.testing.assert_allclose(fitted, fitted_again, atol=0.0)
    assert np.max(np.linalg.norm(fitted[1:-1] - truth[1:-1], axis=1)) < 0.1
    u = np.linspace(0.0, 1.0, 201)
    design = spline_design_matrix(u, geometry)
    curve_error = np.linalg.norm(design @ fitted - design @ truth, axis=1)
    assert np.percentile(curve_error, 95) < 0.05


def test_sparse_solver_rejects_nonfinite_or_mismatched_observations():
    geometry = geometry_with_six_controls()
    bad = SplineObservation(
        u=np.array([0.0, 1.0]),
        xy=np.array([[0.0, 0.0], [np.nan, 1.0]]),
        weight=np.ones(2),
    )
    with pytest.raises(ValueError, match="finite"):
        solve_shared_spline_coefficients(geometry, [bad])
