from __future__ import annotations

import numpy as np
from planarcurve.spline import (
    evaluate_bspline,
)
from planarcurve.spline import (
    spline_design_matrix as planar_spline_design_matrix,
)
from planarcurve.transforms import Affine2D

from inkgrammar.fitting.curvature import evaluate_curvature_stroke
from inkgrammar.fitting.models import (
    BSplineGeometry,
    CurvatureStrokeGeometry,
    PrimitiveFrame,
    PrimitiveTemplate,
    TopologyTransform,
)
from inkgrammar.fitting.planarcurve_adapters import (
    from_planar_trajectory,
    to_planar_trajectory,
)
from inkgrammar.fitting.spline import (
    evaluate_primitive_curve,
    spline_design_matrix,
    to_planar_bspline,
)


def test_curvature_model_adapter_preserves_schema_and_delegates_evaluation():
    geometry = CurvatureStrokeGeometry.model_validate(
        {
            "length": 1.4,
            "start_angle_deg": 55.0,
            "curvature": {
                "interpolation": "cubic-hermite",
                "start_rate_deg": 60.0,
                "changes": [
                    {"at": 0.3, "rate_deg": 320.0},
                    {"at": 0.6, "rate_deg": 180.0},
                ],
                "end_rate_deg": 20.0,
            },
        }
    )
    trajectory = to_planar_trajectory(geometry)
    round_trip = from_planar_trajectory(trajectory)
    np.testing.assert_allclose(
        evaluate_curvature_stroke(geometry, count=129).xy,
        trajectory.evaluate(count=129).xy,
        atol=0.0,
    )
    assert round_trip == geometry


def test_spline_adapter_uses_planarcurve_design_and_evaluation():
    spline = BSplineGeometry(
        knots=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        control_points=[(0.0, 0.0), (1.0, 1.8), (2.3, 2.2), (4.0, 0.0)],
    )
    primitive = PrimitiveTemplate(
        id="adapter-test",
        kind="bspline-centerline",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(4.0, 0.0)),
        spline=spline,
    )
    transform = TopologyTransform(scale_x=1.2, scale_y=0.8, shear_x=0.15, dx=3.0, dy=-2.0)
    affine = Affine2D.from_scale_shear_translation(
        scale_x=transform.scale_x,
        scale_y=transform.scale_y,
        shear_x=transform.shear_x,
        dx=transform.dx,
        dy=transform.dy,
    )
    u = np.linspace(0.0, 1.0, 17)
    adapted = evaluate_primitive_curve(primitive, transform, u)
    direct = evaluate_bspline(to_planar_bspline(spline), u, transform=affine)
    np.testing.assert_allclose(adapted.xy, direct.xy)
    np.testing.assert_allclose(adapted.tangent, direct.tangent)
    np.testing.assert_allclose(adapted.curvature, direct.curvature)
    np.testing.assert_allclose(
        spline_design_matrix(u, spline).toarray(),
        planar_spline_design_matrix(u, to_planar_bspline(spline)).toarray(),
    )


def test_svg_geometry_adapter_preserves_subpaths_curves_and_closure():
    import pytest
    from planarcurve.svg import analyze_path as planar_analyze_path
    from planarcurve.svg import sample_path as planar_sample_path

    from inkgrammar.geometry import analyze_path, sample_path

    path = "M 0 0 Q 1 2 2 0 C 2 0 3 0 3 1 Z M 5 5 L 6 5"
    info = analyze_path(path, samples_per_curve=8)
    direct = planar_analyze_path(path, samples_per_curve=8)

    assert info == direct
    assert info.subpath_closed == (True, False)
    assert info.closed is False
    with pytest.raises(ValueError, match="multiple subpaths"):
        sample_path(path, samples_per_curve=8)

    single_path = "M 0 0 Q 1 2 2 0 C 2 0 3 0 3 1 Z"
    np.testing.assert_allclose(
        sample_path(single_path, samples_per_curve=8),
        planar_sample_path(single_path, samples_per_curve=8),
    )


def test_svg_geometry_adapter_delegates_transforms_and_path_helpers():
    from planarcurve.svg import cubic_bridge as planar_cubic_bridge
    from planarcurve.svg import path_without_initial_move as planar_path_without_initial_move
    from planarcurve.svg import transform_path as planar_transform_path

    from inkgrammar.geometry import (
        cubic_bridge,
        path_without_initial_move,
        transform_path,
        translate_path,
    )
    from inkgrammar.models import Transform

    path = "M 1 2 Q 2 3 4 5 Z"
    transform = Transform(scale_x=1.2, scale_y=0.8, shear_x=0.15, dx=3.0, dy=-2.0)
    affine = Affine2D(1.2, 0.15 * 0.8, 0.0, 0.8, 3.0, -2.0)

    assert transform_path(path, transform) == planar_transform_path(path, affine)
    assert transform_path(path, transform).endswith("Z")
    assert translate_path(path, 2.0, -4.0) == planar_transform_path(
        path, Affine2D.translation_only(2.0, -4.0)
    )
    assert path_without_initial_move(path) == planar_path_without_initial_move(path)
    assert cubic_bridge((0, 0), (1, 0), (3, 2), (0, 1), 0.5) == planar_cubic_bridge(
        (0, 0), (1, 0), (3, 2), (0, 1), 0.5
    )


def test_fitting_polyline_helpers_delegate_to_planarcurve():
    from planarcurve.polyline import (
        distance_to_polyline,
        resample_polyline_by_arclength,
        unit_tangents,
    )
    from planarcurve.svg import sample_path as planar_sample_path

    from inkgrammar.fitting.alignment import sample_path_by_arclength
    from inkgrammar.fitting.correspondence import _unit_tangents as correspondence_tangents
    from inkgrammar.fitting.family_fit_v4 import _sample_polyline, _unit_tangents
    from inkgrammar.fitting.residuals import _distances_to_polyline
    from inkgrammar.fitting.residuals import _unit_tangents as residual_tangents

    path = "M 0 0 C 0 0 10 0 10 0"
    sampled = np.asarray(sample_path_by_arclength(path, sample_count=7))
    curve_points = np.asarray(planar_sample_path(path, samples_per_curve=96))
    np.testing.assert_allclose(sampled, resample_polyline_by_arclength(curve_points, 7))

    points = np.asarray([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    parameters, resampled, distances = _sample_polyline(points, 5)
    expected_distances = np.linspace(0.0, 7.0, 5)
    np.testing.assert_allclose(parameters, np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_allclose(resampled, resample_polyline_by_arclength(points, 5))
    np.testing.assert_allclose(_unit_tangents(points), unit_tangents(points))
    np.testing.assert_allclose(correspondence_tangents(points), unit_tangents(points))

    query = np.asarray([[1.0, 1.0]])
    line = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(
        _distances_to_polyline(query, line), distance_to_polyline(query, line)
    )
    np.testing.assert_allclose(_distances_to_polyline(query, line), [1.0])

    near_zero = np.asarray([[0.0, 0.0], [2e-10, 0.0], [4e-10, 0.0]])
    np.testing.assert_array_equal(residual_tangents(near_zero), np.zeros_like(near_zero))
    np.testing.assert_array_equal(correspondence_tangents(near_zero), np.zeros_like(near_zero))
