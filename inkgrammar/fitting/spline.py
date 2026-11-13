from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from planarcurve.bezier import cubic_curvature, cubic_tangent, evaluate_cubic
from planarcurve.polyline import cumulative_arclength
from planarcurve.spline import (
    BSplineCurve,
    evaluate_bspline,
)
from planarcurve.spline import (
    bspline_from_geometry as _bspline_from_geometry,
)
from planarcurve.spline import (
    control_roughness_matrix as _control_roughness_matrix,
)
from planarcurve.spline import (
    sample_curve_by_arclength as _sample_bspline_by_arclength,
)
from planarcurve.spline import (
    spline_design_matrix as _spline_design_matrix,
)
from planarcurve.spline import (
    spline_to_cubic_segments as _spline_to_cubic_segments,
)
from planarcurve.transforms import Affine2D
from planarcurve.types import CubicBezier
from scipy.interpolate import BSpline
from scipy.sparse import csr_array

from .models import BSplineGeometry, CubicSegment, PrimitiveTemplate, TopologyTransform


@dataclass(frozen=True)
class CurveEvaluation:
    u: np.ndarray
    xy: np.ndarray
    tangent: np.ndarray
    curvature: np.ndarray | None


def _validate_parameters(u: np.ndarray) -> np.ndarray:
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("curve parameters must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("curve parameters must lie in [0, 1]")
    return values


def to_planar_bspline(geometry: BSplineGeometry) -> BSplineCurve:
    validate_bspline_geometry(geometry)
    return BSplineCurve(
        knots=tuple(geometry.knots),
        control_points=tuple(geometry.control_points),
        degree=geometry.degree,
    )


def _to_affine(transform: TopologyTransform) -> Affine2D:
    return Affine2D.from_scale_shear_translation(
        scale_x=transform.scale_x,
        scale_y=transform.scale_y,
        shear_x=transform.shear_x,
        dx=transform.dx,
        dy=transform.dy,
    )


def validate_bspline_geometry(geometry: BSplineGeometry) -> None:
    BSplineGeometry.model_validate(geometry.model_dump(mode="python"))
    _to_planar_bspline_unvalidated(geometry)


def _to_planar_bspline_unvalidated(geometry: BSplineGeometry) -> BSplineCurve:
    return BSplineCurve(
        knots=tuple(geometry.knots),
        control_points=tuple(geometry.control_points),
        degree=geometry.degree,
    )


def bspline_from_geometry(geometry: BSplineGeometry) -> tuple[BSpline, BSpline]:
    return _bspline_from_geometry(to_planar_bspline(geometry))


def spline_design_matrix(u: np.ndarray, geometry: BSplineGeometry) -> csr_array:
    return _spline_design_matrix(_validate_parameters(u), to_planar_bspline(geometry))


def control_roughness_matrix(control_count: int, order: int = 2) -> csr_array:
    return _control_roughness_matrix(control_count, order=order)


def _cubic_chain_values(
    primitive: PrimitiveTemplate,
    parameters: np.ndarray,
    transform: TopologyTransform,
    *,
    derivatives: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    assert primitive.segments is not None
    segments = primitive.segments
    segment_count = len(segments)
    scaled = parameters * segment_count
    indices = np.minimum(scaled.astype(int), segment_count - 1)
    local = scaled - indices
    local[parameters == 1.0] = 1.0
    starts = [primitive.frame.start, *(segment.end for segment in segments[:-1])]
    affine = _to_affine(transform)
    xy = np.empty((len(parameters), 2), dtype=float)
    tangent = np.empty_like(xy) if derivatives else np.zeros_like(xy)
    curvature = np.empty(len(parameters), dtype=float) if derivatives else None
    for index, segment in enumerate(segments):
        selected = np.flatnonzero(indices == index)
        if selected.size == 0:
            continue
        controls = affine.apply(
            np.asarray(
                [starts[index], segment.c1, segment.c2, segment.end], dtype=float
            )
        )
        curve = CubicBezier(*(tuple(point) for point in controls))
        sample_parameters = local[selected]
        xy[selected] = evaluate_cubic(curve, sample_parameters)
        if derivatives:
            tangent[selected] = cubic_tangent(curve, sample_parameters)
            curvature[selected] = cubic_curvature(curve, sample_parameters)
    return xy, tangent, curvature


def evaluate_primitive_curve(
    primitive: PrimitiveTemplate,
    transform: TopologyTransform,
    u: np.ndarray,
    *,
    derivatives: bool = True,
) -> CurveEvaluation:
    parameters = _validate_parameters(u)
    if primitive.kind == "cubic-chain":
        xy, tangent, curvature = _cubic_chain_values(
            primitive, parameters, transform, derivatives=derivatives
        )
        return CurveEvaluation(parameters, xy, tangent, curvature)
    if primitive.spline is None:
        raise ValueError("bspline-centerline primitive has no spline geometry")
    evaluated = evaluate_bspline(
        to_planar_bspline(primitive.spline),
        parameters,
        transform=_to_affine(transform),
        derivatives=derivatives,
    )
    return CurveEvaluation(
        evaluated.u, evaluated.xy, evaluated.tangent, evaluated.curvature
    )


def sample_curve_by_arclength(
    primitive: PrimitiveTemplate,
    transform: TopologyTransform,
    count: int,
) -> CurveEvaluation:
    if isinstance(count, bool) or count < 2:
        raise ValueError("arclength sample count must be at least two")
    if primitive.kind == "bspline-centerline":
        if primitive.spline is None:
            raise ValueError("bspline-centerline primitive has no spline geometry")
        evaluated = _sample_bspline_by_arclength(
            to_planar_bspline(primitive.spline), count, transform=_to_affine(transform)
        )
        return CurveEvaluation(
            evaluated.u, evaluated.xy, evaluated.tangent, evaluated.curvature
        )
    dense_u = np.linspace(0.0, 1.0, max(1024, count * 32))
    dense = evaluate_primitive_curve(primitive, transform, dense_u, derivatives=False).xy
    arc = cumulative_arclength(dense)
    keep = np.concatenate(([True], np.diff(arc) > 1e-12))
    arc = arc[keep]
    dense_u = dense_u[keep]
    if len(arc) < 2 or arc[-1] <= 1e-12:
        raise ValueError("cannot resample a zero-length curve")
    target_lengths = np.linspace(0.0, arc[-1], count)
    parameters = np.interp(target_lengths, arc, dense_u)
    return evaluate_primitive_curve(primitive, transform, parameters)


def spline_to_cubic_segments(
    geometry: BSplineGeometry,
    *,
    tolerance: float = 1e-8,
) -> list[CubicSegment]:
    segments = _spline_to_cubic_segments(
        to_planar_bspline(geometry), tolerance=tolerance
    )
    return [
        CubicSegment(c1=segment.c1, c2=segment.c2, end=segment.end)
        for segment in segments
    ]
