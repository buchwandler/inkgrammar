from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from planarcurve.fitting.policy import SplineEndpointPolicy as PlanarSplineEndpointPolicy
from planarcurve.fitting.spline import (
    SplineFitOptions,
    fit_spline_control_points,
)
from planarcurve.fitting.spline import (
    SplineObservation as PlanarSplineObservation,
)

from .models import BSplineGeometry
from .spline import to_planar_bspline


@dataclass(frozen=True)
class SplineObservation:
    u: np.ndarray
    xy: np.ndarray
    weight: np.ndarray | None = None


def solve_shared_spline_coefficients(
    geometry: BSplineGeometry,
    observations: Sequence[SplineObservation],
    *,
    seed_control_points: np.ndarray | None = None,
    smoothness_lambda: float = 0.02,
    seed_lambda: float = 0.01,
    endpoint_sigma: float = 1.0,
    robust_scale: float = 1.5,
    irls_iterations: int = 3,
    options: SplineFitOptions | None = None,
) -> np.ndarray:
    """Adapt InkGrammar observations to planarcurve's generic sparse fitter."""
    endpoint_policy = PlanarSplineEndpointPolicy(
        start=geometry.endpoint_policy.start,
        end=geometry.endpoint_policy.end,
    )
    fit_options = options or SplineFitOptions(
        smoothness=smoothness_lambda,
        seed_weight=seed_lambda,
        endpoint_sigma=endpoint_sigma,
        robust_scale=robust_scale,
        irls_iterations=irls_iterations,
        endpoint_policy=endpoint_policy,
    )
    planar_observations = [
        PlanarSplineObservation(u=item.u, xy=item.xy, weight=item.weight)
        for item in observations
    ]
    result = fit_spline_control_points(
        to_planar_bspline(geometry),
        planar_observations,
        seed_control_points=seed_control_points,
        options=fit_options,
    )
    return np.asarray(result.curve.control_points, dtype=float)
