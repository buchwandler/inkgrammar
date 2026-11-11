from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_array

from .models import (
    BSplineGeometry,
    CubicSegment,
    PrimitiveTemplate,
    TopologyTransform,
)


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


def validate_bspline_geometry(geometry: BSplineGeometry) -> None:
    BSplineGeometry.model_validate(geometry.model_dump(mode="python"))


def bspline_from_geometry(geometry: BSplineGeometry) -> tuple[BSpline, BSpline]:
    validate_bspline_geometry(geometry)
    knots = np.asarray(geometry.knots, dtype=float)
    control_points = np.asarray(geometry.control_points, dtype=float)
    return (
        BSpline(knots, control_points[:, 0], geometry.degree, extrapolate=False),
        BSpline(knots, control_points[:, 1], geometry.degree, extrapolate=False),
    )


def spline_design_matrix(u: np.ndarray, geometry: BSplineGeometry) -> csr_array:
    parameters = _validate_parameters(u)
    validate_bspline_geometry(geometry)
    return BSpline.design_matrix(
        parameters,
        np.asarray(geometry.knots, dtype=float),
        geometry.degree,
        extrapolate=False,
    )


def control_roughness_matrix(control_count: int, order: int = 2) -> csr_array:
    if isinstance(control_count, bool) or control_count < 2:
        raise ValueError("control_count must be at least two")
    if isinstance(order, bool) or order < 1 or order >= control_count:
        raise ValueError("difference order must be between one and control_count - 1")
    return csr_array(np.diff(np.eye(control_count), n=order, axis=0))


def _cubic_chain_values(
    primitive: PrimitiveTemplate, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert primitive.segments is not None
    segment_count = len(primitive.segments)
    scaled = parameters * segment_count
    indices = np.minimum(scaled.astype(int), segment_count - 1)
    local = scaled - indices
    local[parameters == 1.0] = 1.0
    starts = [primitive.frame.start, *(segment.end for segment in primitive.segments[:-1])]
    points = np.asarray(starts, dtype=float)
    c1 = np.asarray([segment.c1 for segment in primitive.segments], dtype=float)
    c2 = np.asarray([segment.c2 for segment in primitive.segments], dtype=float)
    ends = np.asarray([segment.end for segment in primitive.segments], dtype=float)
    p0, p1, p2, p3 = points[indices], c1[indices], c2[indices], ends[indices]
    t = local[:, None]
    one_minus_t = 1.0 - t
    xy = (
        one_minus_t**3 * p0
        + 3.0 * one_minus_t**2 * t * p1
        + 3.0 * one_minus_t * t**2 * p2
        + t**3 * p3
    )
    first = (
        3.0
        * (one_minus_t**2 * (p1 - p0) + 2.0 * one_minus_t * t * (p2 - p1) + t**2 * (p3 - p2))
        * segment_count
    )
    second = (
        6.0 * (one_minus_t * (p2 - 2.0 * p1 + p0) + t * (p3 - 2.0 * p2 + p1)) * segment_count**2
    )
    return xy, first, second


def _base_values(
    primitive: PrimitiveTemplate, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if primitive.kind == "cubic-chain":
        return _cubic_chain_values(primitive, parameters)
    if primitive.spline is None:
        raise ValueError("bspline-centerline primitive has no spline geometry")
    x_spline, y_spline = bspline_from_geometry(primitive.spline)
    xy = np.column_stack((x_spline(parameters), y_spline(parameters)))
    first = np.column_stack((x_spline.derivative()(parameters), y_spline.derivative()(parameters)))
    second = np.column_stack(
        (x_spline.derivative(2)(parameters), y_spline.derivative(2)(parameters))
    )
    return xy, first, second


def evaluate_primitive_curve(
    primitive: PrimitiveTemplate,
    transform: TopologyTransform,
    u: np.ndarray,
    *,
    derivatives: bool = True,
) -> CurveEvaluation:
    parameters = _validate_parameters(u)
    xy, first, second = _base_values(primitive, parameters)
    linear = np.array(
        [
            [transform.scale_x, transform.scale_x * transform.shear_x],
            [0.0, transform.scale_y],
        ],
        dtype=float,
    )
    xy = xy @ linear.T + np.array([transform.dx, transform.dy])
    if not derivatives:
        return CurveEvaluation(parameters, xy, np.zeros_like(xy), None)
    first = first @ linear.T
    second = second @ linear.T
    speed = np.linalg.norm(first, axis=1)
    tangent = np.divide(
        first,
        speed[:, None],
        out=np.zeros_like(first),
        where=speed[:, None] > 1e-12,
    )
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    curvature = np.divide(
        cross,
        speed**3,
        out=np.zeros_like(cross),
        where=speed > 1e-12,
    )
    return CurveEvaluation(parameters, xy, tangent, curvature)


def sample_curve_by_arclength(
    primitive: PrimitiveTemplate,
    transform: TopologyTransform,
    count: int,
) -> CurveEvaluation:
    if isinstance(count, bool) or count < 2:
        raise ValueError("arclength sample count must be at least two")
    dense_u = np.linspace(0.0, 1.0, max(1024, count * 32))
    dense = evaluate_primitive_curve(primitive, transform, dense_u, derivatives=False).xy
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    dense_u = dense_u[keep]
    if len(cumulative) < 2 or cumulative[-1] <= 1e-12:
        raise ValueError("cannot resample a zero-length curve")
    target_lengths = np.linspace(0.0, cumulative[-1], count)
    parameters = np.interp(target_lengths, cumulative, dense_u)
    return evaluate_primitive_curve(primitive, transform, parameters)


def _bezier_values(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)[:, None]
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * control_points[0]
        + 3.0 * one_minus_t**2 * t * control_points[1]
        + 3.0 * one_minus_t * t**2 * control_points[2]
        + t**3 * control_points[3]
    )


def spline_to_cubic_segments(
    geometry: BSplineGeometry,
    *,
    tolerance: float = 1e-8,
) -> list[CubicSegment]:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("conversion tolerance must be positive and finite")
    validate_bspline_geometry(geometry)
    refined = BSpline(
        np.asarray(geometry.knots, dtype=float),
        np.asarray(geometry.control_points, dtype=float),
        geometry.degree,
        axis=0,
        extrapolate=False,
    )
    for knot in sorted(set(geometry.knots[4:-4])):
        multiplicity = int(np.count_nonzero(np.asarray(refined.t) == knot))
        for _ in range(3 - multiplicity):
            refined = refined.insert_knot(knot)

    refined_knots = np.asarray(refined.t)
    coefficients = np.asarray(refined.c)[: len(refined_knots) - geometry.degree - 1]
    spans = [
        (left, right)
        for left, right in zip(refined_knots[:-1], refined_knots[1:], strict=True)
        if right > left
    ]
    segments: list[CubicSegment] = []
    x_spline, y_spline = bspline_from_geometry(geometry)
    for index, (left, right) in enumerate(spans):
        controls = coefficients[index * 3 : index * 3 + 4]
        segments.append(
            CubicSegment(c1=tuple(controls[1]), c2=tuple(controls[2]), end=tuple(controls[3]))
        )
        local = np.linspace(0.0, 1.0, 33)
        parameters = left + (right - left) * local
        expected = np.column_stack((x_spline(parameters), y_spline(parameters)))
        if np.max(np.linalg.norm(_bezier_values(controls, local) - expected, axis=1)) > tolerance:
            raise ValueError("cubic conversion exceeded the requested tolerance")
    return segments
