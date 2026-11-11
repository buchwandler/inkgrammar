from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ComponentAssignment:
    target_component: str
    target_weights: np.ndarray
    competing_weights: dict[str, np.ndarray]
    outlier_weights: np.ndarray

    def hard_target_mask(self, min_probability: float = 0.55) -> np.ndarray:
        if not 0.0 <= min_probability <= 1.0:
            raise ValueError("minimum target probability must be between 0 and 1")
        return self.target_weights >= min_probability


def assign_ink_to_components(
    observed_ink: np.ndarray,
    predicted_centerlines: Mapping[str, np.ndarray],
    target_component: str,
    *,
    sigma_assign: float = 4.0,
    outlier_mass: float = 0.15,
) -> ComponentAssignment:
    ink = np.asarray(observed_ink, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("observed ink must be a 2D mask")
    if target_component not in predicted_centerlines:
        raise ValueError(f"target component is not predicted: {target_component}")
    if not predicted_centerlines:
        raise ValueError("at least one predicted centerline is required")
    if sigma_assign <= 0:
        raise ValueError("assignment sigma must be positive")
    if outlier_mass <= 0:
        raise ValueError("outlier mass must be positive")

    scores = {}
    for component, centerline in predicted_centerlines.items():
        centerline_mask = np.asarray(centerline, dtype=bool)
        if centerline_mask.shape != ink.shape or centerline_mask.ndim != 2:
            raise ValueError(f"predicted centerline shape does not match observed ink: {component}")
        if not np.any(centerline_mask):
            raise ValueError(f"predicted centerline is empty: {component}")
        distance = distance_transform_edt(~centerline_mask)
        scores[component] = np.exp(-(distance**2) / (2.0 * sigma_assign**2))

    total = np.full(ink.shape, outlier_mass, dtype=np.float64)
    for score in scores.values():
        total += score
    weights = {component: np.where(ink, score / total, 0.0) for component, score in scores.items()}
    outlier_weights = np.where(ink, outlier_mass / total, 0.0)
    return ComponentAssignment(
        target_component=target_component,
        target_weights=weights[target_component],
        competing_weights={
            component: weight
            for component, weight in weights.items()
            if component != target_component
        },
        outlier_weights=outlier_weights,
    )
