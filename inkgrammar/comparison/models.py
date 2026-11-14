from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from planarcurve.transforms import Similarity2D
from pydantic import Field
from skeletonpaths import SkeletonGraph

from ..models import StrictModel

InkMode = Literal["auto", "grayscale", "dark", "color"]


class CropRect(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LetterImageInput(StrictModel):
    path: Path
    crop: CropRect
    ink_mode: InkMode = "auto"


class TopologySummary(StrictModel):
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    endpoint_count: int = Field(ge=0)
    endpoints: list[tuple[float, float]]
    junction_count: int = Field(ge=0)
    junctions: list[tuple[float, float]]
    cycle_count: int = Field(ge=0)
    component_count: int = Field(ge=0)


@dataclass(frozen=True)
class OrderedPath:
    xy: np.ndarray
    covered_edge_count: int
    total_edge_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("ordered path must have shape (N, 2) with N >= 2")
        if not np.isfinite(points).all():
            raise ValueError("ordered path coordinates must be finite")
        if self.total_edge_count < 1 or self.covered_edge_count < 0:
            raise ValueError("ordered path edge counts are invalid")
        if self.covered_edge_count > self.total_edge_count:
            raise ValueError("covered edge count cannot exceed total edge count")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "xy", points)

    @property
    def coverage(self) -> float:
        return self.covered_edge_count / self.total_edge_count


@dataclass(frozen=True)
class AlignmentResult:
    transform: Similarity2D
    direction: Literal["forward", "reversed-for-best-match"]
    aligned_sample_xy: np.ndarray
    mean_squared_error: float

    def __post_init__(self) -> None:
        points = np.asarray(self.aligned_sample_xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("aligned path must have shape (N, 2) with N >= 2")
        if not np.isfinite(points).all() or not np.isfinite(self.mean_squared_error):
            raise ValueError("alignment result must be finite")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "aligned_sample_xy", points)

@dataclass(frozen=True)
class InkMaskResult:
    mask: np.ndarray
    confidence: float
    method: str
    warnings: list[str]

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("ink mask must be a 2D array")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("mask confidence must be between zero and one")
        if not self.method:
            raise ValueError("mask method must not be empty")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "warnings", list(self.warnings))


@dataclass(frozen=True)
class SkeletonDiagnostics:
    skeleton: np.ndarray
    radius: np.ndarray
    graph: SkeletonGraph
    topology: TopologySummary

    def __post_init__(self) -> None:
        skeleton = np.asarray(self.skeleton, dtype=bool)
        radius = np.asarray(self.radius, dtype=float)
        if skeleton.ndim != 2 or radius.shape != skeleton.shape:
            raise ValueError("skeleton and radius must be matching 2D arrays")
        if not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius map must be finite and nonnegative")
        skeleton = np.array(skeleton, copy=True)
        radius = np.array(radius, copy=True)
        skeleton.setflags(write=False)
        radius.setflags(write=False)
        object.__setattr__(self, "skeleton", skeleton)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class ExtractedLetter:
    source_path: Path
    crop: CropRect
    image: Image.Image
    mask_result: InkMaskResult
    diagnostics: SkeletonDiagnostics
    ordered_path: OrderedPath
    resampled_xy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        points = np.asarray(self.resampled_xy, dtype=float)
        expected_shape = (self.crop.height, self.crop.width)
        if self.image.size != (self.crop.width, self.crop.height):
            raise ValueError("extracted image dimensions must match its crop rectangle")
        if self.mask_result.mask.shape != expected_shape:
            raise ValueError("ink mask dimensions must match the extracted image")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("resampled path must have shape (N, 2) with N >= 2")
        if not np.isfinite(points).all():
            raise ValueError("resampled path coordinates must be finite")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "resampled_xy", points)


@dataclass(frozen=True)
class ComparisonResult:
    letter: str
    reference: ExtractedLetter
    sample: ExtractedLetter
    alignment: AlignmentResult
    smoothing_window: int = 11
    smoothing_polynomial_order: int = 2
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.letter) != 1:
            raise ValueError("letter label must contain exactly one character")
        if len(self.reference.resampled_xy) != len(self.sample.resampled_xy):
            raise ValueError("reference and sample paths must have the same sample count")
        if len(self.alignment.aligned_sample_xy) != len(self.reference.resampled_xy):
            raise ValueError("aligned sample path must match the reference sample count")
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing window must be odd and at least three")
        if not 1 <= self.smoothing_polynomial_order < self.smoothing_window:
            raise ValueError("smoothing polynomial order is invalid")
        object.__setattr__(self, "warnings", list(self.warnings))
