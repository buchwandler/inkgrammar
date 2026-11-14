"""InkGrammar fit-result adapters around planarcurve's generic trajectory fitter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from planarcurve.fitting.trajectory import (
    SharedTrajectoryFitResult as PlanarSharedTrajectoryFitResult,
)
from planarcurve.fitting.trajectory import (
    TrajectoryFitOptions,
    estimate_heading_rate,
    estimate_polyline_heading,
    fit_shared_trajectory_profile,
    fit_trajectory,
    initialize_change_positions,
    matched_trajectory_state,
)
from planarcurve.fitting.trajectory import (
    resample_by_arclength as _resample_by_arclength,
)

from .models import (
    MIN_CHANGE_SPACING,
    CurvatureStrokeGeometry,
    TrajectoryTransform,
)
from .planarcurve_adapters import (
    from_planar_similarity,
    from_planar_trajectory,
    to_planar_trajectory,
)

DEFAULT_SAMPLES = 64


@dataclass(frozen=True)
class CurvatureFitResult:
    geometry: CurvatureStrokeGeometry
    transform: TrajectoryTransform
    success: bool
    cost: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedProfileFitResult:
    geometry: CurvatureStrokeGeometry
    transforms: list[TrajectoryTransform]
    success: bool
    cost: float
    metrics: dict[str, float] = field(default_factory=dict)


def resample_by_arclength(xy: np.ndarray, *, samples: int = DEFAULT_SAMPLES) -> np.ndarray:
    return _resample_by_arclength(xy, samples=samples)


def estimate_observed_heading(xy: np.ndarray, *, smooth_window: int = 5) -> np.ndarray:
    points = resample_by_arclength(xy, samples=max(len(xy), DEFAULT_SAMPLES))
    return estimate_polyline_heading(points, smooth_window=smooth_window)


def estimate_observed_heading_rate(u: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
    return estimate_heading_rate(u, heading_deg)


def initialize_event_positions(q_obs: np.ndarray, change_count: int) -> np.ndarray:
    return initialize_change_positions(q_obs, change_count, min_spacing=MIN_CHANGE_SPACING)


def matched_fit_state(
    geometry: CurvatureStrokeGeometry, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    return matched_trajectory_state(to_planar_trajectory(geometry), samples)


def fit_curvature_stroke(
    target_xy: np.ndarray,
    *,
    change_count: int,
    samples: int = DEFAULT_SAMPLES,
    restarts: int = 2,
    seed: int = 0,
) -> CurvatureFitResult:
    result = fit_trajectory(
        target_xy,
        TrajectoryFitOptions(
            change_count=change_count,
            samples=samples,
            restarts=restarts,
            seed=seed,
        ),
    )
    return CurvatureFitResult(
        geometry=from_planar_trajectory(result.trajectory),
        transform=from_planar_similarity(result.transform),
        success=result.success,
        cost=result.cost,
        metrics=dict(result.metrics),
    )


def fit_shared_profile(
    occurrences: list[np.ndarray],
    *,
    change_count: int,
    samples: int = DEFAULT_SAMPLES,
    restarts: int = 1,
    seed: int = 0,
) -> SharedProfileFitResult:
    result: PlanarSharedTrajectoryFitResult = fit_shared_trajectory_profile(
        occurrences,
        TrajectoryFitOptions(
            change_count=change_count,
            samples=samples,
            restarts=restarts,
            seed=seed,
            endpoint_weight=0.0,
            endpoint_heading_weight=0.0,
            turn_weight=0.3,
        ),
    )
    return SharedProfileFitResult(
        geometry=from_planar_trajectory(result.trajectory),
        transforms=[from_planar_similarity(item) for item in result.transforms],
        success=result.success,
        cost=result.cost,
        metrics=dict(result.metrics),
    )
