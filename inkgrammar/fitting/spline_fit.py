from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array, eye, vstack
from scipy.sparse.linalg import lsqr

from .models import BSplineGeometry
from .spline import control_roughness_matrix, spline_design_matrix


@dataclass(frozen=True)
class SplineObservation:
    u: np.ndarray
    xy: np.ndarray
    weight: np.ndarray


def _validated_observations(
    observations: Sequence[SplineObservation],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    matrices = []
    values = []
    weights = []
    for observation in observations:
        u = np.asarray(observation.u, dtype=float)
        xy = np.asarray(observation.xy, dtype=float)
        weight = np.asarray(observation.weight, dtype=float)
        if u.ndim != 1 or xy.shape != (len(u), 2) or weight.shape != (len(u),):
            raise ValueError("spline observation arrays have inconsistent shapes")
        if (
            not np.isfinite(u).all()
            or not np.isfinite(xy).all()
            or not np.isfinite(weight).all()
            or np.any(weight < 0.0)
        ):
            raise ValueError("spline observations must be finite with nonnegative weights")
        matrices.append(u)
        values.append(xy)
        weights.append(weight)
    if not matrices or sum(len(u) for u in matrices) == 0:
        raise ValueError("at least one spline observation is required")
    return matrices, np.vstack(values), np.concatenate(weights)


def _weighted_rows(matrix: csr_array, weights: np.ndarray) -> csr_array:
    return csr_array(matrix.multiply(np.sqrt(weights)[:, None]))


def _solve_coordinate(
    matrix: csr_array,
    target: np.ndarray,
    seed: np.ndarray,
    fixed_indices: np.ndarray,
) -> np.ndarray:
    control_count = len(seed)
    free_indices = np.setdiff1d(np.arange(control_count), fixed_indices, assume_unique=True)
    result = np.empty(control_count, dtype=float)
    result[fixed_indices] = seed[fixed_indices]
    right_hand_side = target - matrix[:, fixed_indices] @ result[fixed_indices]
    solution = lsqr(
        matrix[:, free_indices],
        np.asarray(right_hand_side).ravel(),
        atol=1e-12,
        btol=1e-12,
        iter_lim=max(100, control_count * 20),
    )[0]
    result[free_indices] = solution
    return result


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
) -> np.ndarray:
    """Fit fixed-knot shared coefficients with sparse weighted least squares and IRLS."""
    values = [smoothness_lambda, seed_lambda, endpoint_sigma, robust_scale]
    if not np.isfinite(values).all() or smoothness_lambda < 0.0 or seed_lambda < 0.0:
        raise ValueError("spline regularization values must be finite and nonnegative")
    if endpoint_sigma <= 0.0 or robust_scale <= 0.0:
        raise ValueError("endpoint sigma and robust scale must be positive")
    if isinstance(irls_iterations, bool) or irls_iterations < 1:
        raise ValueError("IRLS iteration count must be at least one")

    parameter_arrays, observed_xy, base_weights = _validated_observations(observations)
    if not np.any(base_weights > 0.0):
        raise ValueError("spline observations must have positive total weight")
    control_count = len(geometry.control_points)
    seed = np.asarray(
        geometry.control_points if seed_control_points is None else seed_control_points,
        dtype=float,
    )
    if seed.shape != (control_count, 2) or not np.isfinite(seed).all():
        raise ValueError("seed control points must be finite and match the spline basis")

    data_matrix = vstack(
        [spline_design_matrix(u, geometry) for u in parameter_arrays], format="csr"
    )
    roughness = control_roughness_matrix(control_count, order=2)
    roughness_rows = np.sqrt(smoothness_lambda) * roughness
    seed_rows = np.sqrt(seed_lambda) * eye(control_count, format="csr")
    endpoint_rows = []
    endpoint_values = []
    endpoint_policy = geometry.endpoint_policy
    for index, policy in ((0, endpoint_policy.start), (-1, endpoint_policy.end)):
        if policy == "anchored":
            endpoint_rows.append(eye(control_count, format="csr")[index] / endpoint_sigma)
            endpoint_values.append(seed[index] / endpoint_sigma)
    fixed_indices = np.asarray(
        [
            index % control_count
            for index, policy in ((0, endpoint_policy.start), (-1, endpoint_policy.end))
            if policy == "locked"
        ],
        dtype=int,
    )

    robust_weights = np.ones_like(base_weights)
    control_points = seed.copy()
    for _ in range(irls_iterations):
        combined_weights = base_weights * robust_weights
        augmented_matrix = vstack(
            [
                _weighted_rows(data_matrix, combined_weights),
                roughness_rows,
                seed_rows,
                *endpoint_rows,
            ],
            format="csr",
        )
        weighted_observations = observed_xy * np.sqrt(combined_weights)[:, None]
        regularized_values = [np.zeros((roughness.shape[0], 2)), np.sqrt(seed_lambda) * seed]
        regularized_values.extend(endpoint_values)
        augmented_values = np.vstack([weighted_observations, *regularized_values])
        control_points = np.column_stack(
            [
                _solve_coordinate(
                    augmented_matrix,
                    augmented_values[:, axis],
                    seed[:, axis],
                    fixed_indices,
                )
                for axis in range(2)
            ]
        )
        residuals = np.linalg.norm(data_matrix @ control_points - observed_xy, axis=1)
        robust_weights = np.minimum(1.0, robust_scale / np.maximum(residuals, 1e-12))
    return control_points
