import numpy as np
import pytest
from pydantic import ValidationError

from inkgrammar.fitting.models import (
    BSplineGeometry,
    PrimitiveFrame,
    PrimitiveTemplate,
    TopologyTransform,
)
from inkgrammar.fitting.spline import (
    bspline_from_geometry,
    control_roughness_matrix,
    evaluate_primitive_curve,
    sample_curve_by_arclength,
    spline_design_matrix,
    spline_to_cubic_segments,
)


def geometry_payload():
    return {
        "degree": 3,
        "knots": [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        "control_points": [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 0.0],
        ],
    }


def primitive_payload(geometry=None):
    return {
        "id": "spline-test",
        "kind": "bspline-centerline",
        "frame": {"start": [0.0, 0.0], "nominal_end": [4.0, 0.0]},
        "spline": geometry or geometry_payload(),
    }


def test_width_model_supports_constant_and_separate_quadratic_profile():
    from inkgrammar.fitting.models import SplineWidthModel

    constant = SplineWidthModel(value=0.055)
    profile = SplineWidthModel(
        kind="bspline",
        degree=2,
        knots=[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        coefficients=[0.04, 0.05, 0.06, 0.055],
    )

    assert constant.kind == "constant"
    assert profile.kind == "bspline"
    with pytest.raises(ValidationError, match="greater than 0"):
        SplineWidthModel(value=0.0)


def test_bspline_geometry_requires_matching_knot_and_control_counts():
    payload = geometry_payload()
    payload["knots"].pop()
    with pytest.raises(ValidationError, match="knot count"):
        BSplineGeometry.model_validate(payload)


def test_bspline_geometry_rejects_unsorted_knots_and_repeated_interior_knots():
    payload = geometry_payload()
    payload["knots"] = [0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 1.0, 1.0, 1.0]
    with pytest.raises(ValidationError, match="nondecreasing"):
        BSplineGeometry.model_validate(payload)

    payload = geometry_payload()
    payload["knots"] = [0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 1.0, 1.0, 1.0]
    with pytest.raises(ValidationError, match="nondecreasing"):
        BSplineGeometry.model_validate(payload)

    payload = geometry_payload()
    payload["knots"] = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    payload["control_points"].append([5.0, 0.0])
    payload["knots"].insert(5, 0.5)
    with pytest.raises(ValidationError, match="interior knots must be simple"):
        BSplineGeometry.model_validate(payload)


def test_bspline_geometry_requires_clamped_normalized_finite_data():
    payload = geometry_payload()
    payload["knots"][0] = -0.1
    with pytest.raises(ValidationError, match="clamped"):
        BSplineGeometry.model_validate(payload)

    payload = geometry_payload()
    payload["control_points"][1][0] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        BSplineGeometry.model_validate(payload)


def test_primitive_geometry_payloads_are_mutually_exclusive():
    payload = primitive_payload()
    payload["segments"] = [{"c1": [1, 1], "c2": [2, 1], "end": [3, 0]}]
    with pytest.raises(ValidationError, match="must not contain cubic segments"):
        PrimitiveTemplate.model_validate(payload)

    payload = primitive_payload()
    payload["free_parameters"] = ["spline.control_points.1.0"]
    payload["bounds"] = {"spline.control_points.1.0": [0.0, 2.0]}
    with pytest.raises(ValidationError, match="must not address spline geometry"):
        PrimitiveTemplate.model_validate(payload)


def test_spline_evaluation_returns_exact_endpoints_and_derivatives():
    primitive = PrimitiveTemplate.model_validate(primitive_payload())
    result = evaluate_primitive_curve(primitive, TopologyTransform(), np.array([0.0, 1.0]))

    np.testing.assert_allclose(result.xy, [[0.0, 0.0], [4.0, 0.0]], atol=1e-12)
    assert np.linalg.norm(result.tangent, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert result.curvature is not None
    assert np.isfinite(result.curvature).all()


def test_cubic_simple_knot_has_c2_continuity():
    geometry = BSplineGeometry.model_validate(geometry_payload())
    x_spline, y_spline = bspline_from_geometry(geometry)
    left = np.array(
        [
            [spline.derivative(order)(0.5 - 1e-8) for spline in (x_spline, y_spline)]
            for order in range(3)
        ]
    )
    right = np.array(
        [
            [spline.derivative(order)(0.5 + 1e-8) for spline in (x_spline, y_spline)]
            for order in range(3)
        ]
    )
    np.testing.assert_allclose(left, right, atol=1e-5)


def test_spline_design_and_roughness_matrices_have_expected_shape():
    geometry = BSplineGeometry.model_validate(geometry_payload())
    design = spline_design_matrix(np.array([0.0, 0.25, 0.5, 1.0]), geometry)
    roughness = control_roughness_matrix(5)

    assert design.shape == (4, 5)
    np.testing.assert_allclose(np.asarray(design.sum(axis=1)).ravel(), 1.0)
    assert roughness.shape == (3, 5)
    np.testing.assert_array_equal(roughness.toarray()[0], [1, -2, 1, 0, 0])
    with pytest.raises(ValueError, match="difference order"):
        control_roughness_matrix(3, order=3)


def test_arclength_sampling_is_uniform_and_deterministic_for_a_line():
    geometry = {
        "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "control_points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    }
    primitive = PrimitiveTemplate.model_validate(primitive_payload(geometry))
    first = sample_curve_by_arclength(primitive, TopologyTransform(), 7)
    second = sample_curve_by_arclength(primitive, TopologyTransform(), 7)

    np.testing.assert_allclose(first.xy[:, 0], np.linspace(0.0, 3.0, 7), atol=1e-8)
    np.testing.assert_array_equal(first.u, second.u)
    assert np.all(np.diff(first.u) > 0.0)


def test_curve_affine_transform_applies_scale_shear_and_translation():
    primitive = PrimitiveTemplate.model_validate(primitive_payload())
    transform = TopologyTransform(scale_x=2.0, scale_y=3.0, shear_x=0.25, dx=5.0, dy=-2.0)
    base = evaluate_primitive_curve(primitive, TopologyTransform(), np.array([0.3]))
    transformed = evaluate_primitive_curve(primitive, transform, np.array([0.3]))
    expected = np.array(
        [[5.0 + 2.0 * (base.xy[0, 0] + 0.25 * base.xy[0, 1]), -2.0 + 3.0 * base.xy[0, 1]]]
    )

    np.testing.assert_allclose(transformed.xy, expected)


def test_spline_to_cubic_segments_reproduces_each_span():
    geometry = BSplineGeometry.model_validate(geometry_payload())
    segments = spline_to_cubic_segments(geometry, tolerance=1e-10)
    primitive = PrimitiveTemplate(
        id="converted",
        frame=PrimitiveFrame(
            start=geometry.control_points[0],
            nominal_end=geometry.control_points[-1],
        ),
        segments=segments,
    )
    parameters = np.linspace(0.0, 1.0, 101)
    original = evaluate_primitive_curve(
        PrimitiveTemplate.model_validate(primitive_payload()), TopologyTransform(), parameters
    )
    converted = evaluate_primitive_curve(primitive, TopologyTransform(), parameters)

    assert len(segments) == 2
    np.testing.assert_allclose(converted.xy, original.xy, atol=1e-10)


def test_spline_evaluation_is_deterministic_and_validates_parameter_domain():
    primitive = PrimitiveTemplate.model_validate(primitive_payload())
    parameters = np.linspace(0.0, 1.0, 31)
    first = evaluate_primitive_curve(primitive, TopologyTransform(), parameters)
    second = evaluate_primitive_curve(primitive, TopologyTransform(), parameters)
    np.testing.assert_array_equal(first.xy, second.xy)
    np.testing.assert_array_equal(first.tangent, second.tangent)

    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        evaluate_primitive_curve(primitive, TopologyTransform(), np.array([-0.1, 1.1]))
