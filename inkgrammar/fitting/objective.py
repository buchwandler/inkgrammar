from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import log, pi
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ..models import StrictModel
from .models import DuctusModel, PrimitiveInstance


class ObjectiveWeights(StrictModel):
    soft_mask: float = Field(default=1.0, ge=0)
    distance: float = Field(default=0.75, ge=0)
    anchor: float = Field(default=0.40, ge=0)
    bbox: float = Field(default=0.10, ge=0)
    endpoint: float = Field(default=0.50, ge=0)
    tangent: float = Field(default=0.20, ge=0)
    instance_regularization: float = Field(default=0.15, ge=0)
    curvature_regularization: float = Field(default=0.05, ge=0)
    complexity: float = Field(default=0.02, ge=0)
    local_region: float = Field(default=0.50, ge=0)


class GeometryResidualWeights(StrictModel):
    path_to_ink: float = Field(default=1.0, ge=0)
    normal_center: float = Field(default=1.5, ge=0)
    endpoint: float = Field(default=0.6, ge=0)
    tangent: float = Field(default=0.25, ge=0)
    assigned_coverage: float = Field(default=0.2, ge=0)
    position_sigma_px: float = Field(default=1.5, gt=0)
    coverage_sigma_px: float = Field(default=1.5, gt=0)
    endpoint_sigma_px: float = Field(default=1.0, gt=0)
    tangent_sigma_rad: float = Field(default=0.20, gt=0)
    radius_sigma_px: float = Field(default=1.5, gt=0)
    candidate_to_target: float = Field(default=1.0, ge=0)
    target_to_candidate: float = Field(default=1.0, ge=0)
    radius: float = Field(default=0.1, ge=0)
    coverage_tolerance_px: float = Field(default=2.0, gt=0)


class AssignmentConfig(StrictModel):
    sigma_px: float = Field(default=4.0, gt=0)
    min_target_probability: float = Field(default=0.55, ge=0, le=1)
    outlier_mass: float = Field(default=0.15, gt=0)
    recompute_each_cycle: bool = True


class CenterlineConfig(StrictModel):
    method: Literal["skeletonize"] = "skeletonize"
    sample_schedule: list[int] = Field(default_factory=lambda: [24, 48, 96], min_length=1)
    min_spur_length_px: float = Field(default=3.0, ge=0)
    min_confidence: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> CenterlineConfig:
        if any(count < 2 for count in self.sample_schedule):
            raise ValueError("centerline sample counts must be at least two")
        if self.sample_schedule != sorted(set(self.sample_schedule)):
            raise ValueError("centerline sample schedule must be strictly increasing")
        return self


class PriorWeights(StrictModel):
    transform: float = Field(default=0.2, ge=0)
    shape: float = Field(default=0.05, ge=0)
    curvature: float = Field(default=0.02, ge=0)


class FamilyBalanceWeights(StrictModel):
    worst: float = Field(default=0.25, ge=0)
    spread: float = Field(default=0.1, ge=0)


class ObjectiveConfig(StrictModel):
    version: int = Field(default=1, ge=1, le=3)
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    geometry: GeometryResidualWeights = Field(default_factory=GeometryResidualWeights)
    priors: PriorWeights = Field(default_factory=PriorWeights)
    family: FamilyBalanceWeights = Field(default_factory=FamilyBalanceWeights)
    assignment: AssignmentConfig = Field(default_factory=AssignmentConfig)
    centerline: CenterlineConfig = Field(default_factory=CenterlineConfig)
    residual_sigma: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_version_settings(self) -> ObjectiveConfig:
        legacy_geometry = {"path_to_ink", "normal_center", "assigned_coverage"}
        v3_geometry = {
            "position_sigma_px",
            "coverage_sigma_px",
            "endpoint_sigma_px",
            "tangent_sigma_rad",
            "radius_sigma_px",
            "candidate_to_target",
            "target_to_candidate",
            "radius",
            "coverage_tolerance_px",
        }
        if self.version == 3:
            if "weights" in self.model_fields_set or "residual_sigma" in self.model_fields_set:
                raise ValueError(
                    "objective v3 does not support legacy raster weights or residual_sigma"
                )
            unsupported = legacy_geometry & self.geometry.model_fields_set
            if unsupported:
                raise ValueError(
                    f"objective v3 does not support legacy geometry settings: {sorted(unsupported)}"
                )
            if not self.assignment.recompute_each_cycle:
                raise ValueError("objective v3 requires assignment recomputation each EM cycle")
        else:
            if {"assignment", "centerline"} & self.model_fields_set:
                raise ValueError("assignment and centerline settings require objective v3")
            unsupported = v3_geometry & self.geometry.model_fields_set
            if unsupported:
                raise ValueError(
                    f"objective v3 geometry settings require version 3: {sorted(unsupported)}"
                )
        return self


@dataclass(frozen=True)
class LocalRegion:
    mask: np.ndarray
    generated_part: np.ndarray


@dataclass(frozen=True)
class ObjectiveResult:
    total: float
    terms: dict[str, float]


def soft_mask_loss(reference: np.ndarray, generated: np.ndarray) -> float:
    denominator = float(reference.sum() + generated.sum())
    if denominator == 0:
        return 0.0
    dice = 2.0 * float(np.sum(reference * generated)) / denominator
    return 1.0 - dice


def _distance_field(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~(mask > 0.01))


def symmetric_distance_loss(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    target_distance_field: np.ndarray | None = None,
) -> float:
    reference_ink = reference > 0.01
    generated_ink = generated > 0.01
    if not reference_ink.any() and not generated_ink.any():
        return 0.0
    if not reference_ink.any() or not generated_ink.any():
        return 1.0

    to_reference = target_distance_field
    if to_reference is None:
        to_reference = _distance_field(reference)
    to_generated = _distance_field(generated)
    generated_weights = generated[generated_ink]
    reference_weights = reference[reference_ink]
    generated_to_reference = float(
        np.average(to_reference[generated_ink], weights=generated_weights)
    )
    reference_to_generated = float(
        np.average(to_generated[reference_ink], weights=reference_weights)
    )
    diagonal = max(float(np.hypot(*reference.shape)), 1.0)
    return (generated_to_reference + reference_to_generated) / (2.0 * diagonal)


def _ink_bounds(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > 0.01)
    if not len(xs):
        return None
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def _bbox_loss(reference: np.ndarray, generated: np.ndarray) -> float:
    reference_bounds = _ink_bounds(reference)
    generated_bounds = _ink_bounds(generated)
    if reference_bounds is None and generated_bounds is None:
        return 0.0
    if reference_bounds is None or generated_bounds is None:
        return 1.0
    return sum(
        log(max(generated_value, 1.0) / max(reference_value, 1.0)) ** 2
        for reference_value, generated_value in zip(reference_bounds, generated_bounds, strict=True)
    )


def _angle_difference(first: float, second: float) -> float:
    return (second - first + pi) % (2 * pi) - pi


def _curvature_loss(paths: Sequence[str]) -> float:
    from ..geometry import sample_path

    terms = []
    for path in paths:
        points = np.asarray(sample_path(path), dtype=float)
        if len(points) < 3:
            continue
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-12
        angles = np.arctan2(vectors[valid, 1], vectors[valid, 0])
        if len(angles) > 1:
            changes = np.asarray(
                [_angle_difference(a, b) for a, b in zip(angles[:-1], angles[1:], strict=True)]
            )
            terms.append(float(np.mean((changes / pi) ** 2)))
    return float(np.mean(terms)) if terms else 0.0


def _instance_loss(
    instances: Sequence[PrimitiveInstance],
    ductus: DuctusModel | None,
    residual_sigma: float,
) -> float:
    if not instances:
        return 0.0
    scale_sigma = ductus.instance_scale_sigma if ductus else 0.05
    shear_sigma = ductus.instance_shear_sigma if ductus else 0.03
    translation_sigma = ductus.instance_translation_sigma if ductus else 12.0
    losses = []
    for instance in instances:
        values = [
            (log(instance.scale_x) / scale_sigma) ** 2,
            (log(instance.scale_y) / scale_sigma) ** 2,
            (instance.shear_x / shear_sigma) ** 2,
            (instance.dx / translation_sigma) ** 2,
            (instance.dy / translation_sigma) ** 2,
        ]
        values.extend((value / residual_sigma) ** 2 for value in instance.shape.values())
        values.extend((value / residual_sigma) ** 2 for value in instance.residual.values())
        losses.append(sum(values))
    return float(np.mean(losses))


def _anchor_loss(
    reference_anchor: tuple[float, float] | None,
    generated_anchor: tuple[float, float] | None,
    x_height: float,
) -> float:
    if reference_anchor is None and generated_anchor is None:
        return 0.0
    if reference_anchor is None or generated_anchor is None:
        raise ValueError("both reference and generated anchors are required")
    return sum(
        ((generated - reference) / x_height) ** 2
        for reference, generated in zip(reference_anchor, generated_anchor, strict=True)
    )


def _endpoint_loss(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    x_height: float,
) -> float:
    if not pairs:
        return 0.0
    return float(
        np.mean(
            [
                sum(
                    (generated - reference) ** 2
                    for reference, generated in zip(target, actual, strict=True)
                )
                / x_height**2
                for target, actual in pairs
            ]
        )
    )


def _tangent_loss(pairs: Sequence[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return float(
        np.mean([(_angle_difference(target, actual) / pi) ** 2 for target, actual in pairs])
    )


def evaluate_objective(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    config: ObjectiveConfig | None = None,
    x_height: float,
    reference_anchor: tuple[float, float] | None = None,
    generated_anchor: tuple[float, float] | None = None,
    target_distance_field: np.ndarray | None = None,
    endpoint_pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    tangent_pairs: Sequence[tuple[float, float]] = (),
    instances: Sequence[PrimitiveInstance] = (),
    ductus: DuctusModel | None = None,
    paths: Sequence[str] = (),
    local_regions: Sequence[LocalRegion] = (),
    complexity_count: float = 0.0,
) -> ObjectiveResult:
    """Evaluate one training sample; held-out split membership is not accepted here."""
    config = config or ObjectiveConfig()
    reference = np.asarray(reference, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    if reference.ndim != 2 or generated.ndim != 2 or reference.shape != generated.shape:
        raise ValueError("reference and generated masks must be matching two-dimensional arrays")
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError("masks must contain finite coverage values")
    if (
        (reference < 0).any()
        or (generated < 0).any()
        or (reference > 1).any()
        or (generated > 1).any()
    ):
        raise ValueError("mask coverage must be between zero and one")
    if x_height <= 0:
        raise ValueError("x_height must be positive")
    if complexity_count < 0:
        raise ValueError("complexity_count must be nonnegative")

    terms = {
        "soft_mask": soft_mask_loss(reference, generated),
        "distance": symmetric_distance_loss(
            reference,
            generated,
            target_distance_field=target_distance_field,
        ),
        "anchor": _anchor_loss(reference_anchor, generated_anchor, x_height),
        "bbox": _bbox_loss(reference, generated),
        "endpoint": _endpoint_loss(endpoint_pairs, x_height),
        "tangent": _tangent_loss(tangent_pairs),
        "instance_regularization": _instance_loss(instances, ductus, config.residual_sigma),
        "curvature_regularization": _curvature_loss(paths),
        "complexity": float(complexity_count),
        "local_region": (
            float(
                np.mean(
                    [
                        soft_mask_loss(reference * region.mask, region.generated_part * region.mask)
                        for region in local_regions
                    ]
                )
            )
            if local_regions
            else 0.0
        ),
    }
    total = sum(getattr(config.weights, name) * value for name, value in terms.items())
    return ObjectiveResult(total=float(total), terms=terms)
