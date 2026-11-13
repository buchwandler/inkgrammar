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
    transform = TopologyTransform(
        scale_x=1.2, scale_y=0.8, shear_x=0.15, dx=3.0, dy=-2.0
    )
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
