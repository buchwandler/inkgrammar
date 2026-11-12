from __future__ import annotations

import numpy as np
import pytest

from inkgrammar.compiler import (
    fit_units_to_glyph,
    fit_units_to_runtime,
    glyph_units_to_fit,
    glyph_units_to_runtime,
    runtime_units_to_fit,
    runtime_units_to_glyph,
)
from inkgrammar.fitting.control_sheet import render_trajectory_diagnostic
from inkgrammar.fitting.curvature import (
    curvature_to_cubic_segments,
    evaluate_curvature_stroke,
)
from inkgrammar.fitting.family_fit import primitive_path
from inkgrammar.fitting.models import CurvatureStrokeGeometry, PrimitiveTemplate, TopologyTransform

PROFILE = {
    "interpolation": "cubic-hermite",
    "start_rate_deg": 60.0,
    "changes": [
        {"at": 0.3, "rate_deg": 320.0},
        {"at": 0.6, "rate_deg": 180.0},
        {"at": 0.85, "rate_deg": -60.0},
    ],
    "end_rate_deg": 20.0,
}


def _geometry() -> CurvatureStrokeGeometry:
    return CurvatureStrokeGeometry.model_validate(
        {"length": 1.4, "start_angle_deg": 55.0, "curvature": PROFILE}
    )


def _bezier_points(segments, *, samples_per_segment: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples_per_segment)
    points = [np.zeros(2)]
    start = np.zeros(2)
    for segment in segments:
        c1 = np.asarray(segment.c1)
        c2 = np.asarray(segment.c2)
        end = np.asarray(segment.end)
        curve = (
            ((1 - t) ** 3)[:, None] * start
            + (3 * (1 - t) ** 2 * t)[:, None] * c1
            + (3 * (1 - t) * t**2)[:, None] * c2
            + (t**3)[:, None] * end
        )
        points.append(curve)
        start = end
    return np.vstack(points)


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    starts = polyline[:-1]
    ends = polyline[1:]
    segments = ends - starts
    lengths_squared = (segments**2).sum(axis=1)
    lengths_squared[lengths_squared == 0] = 1e-18
    distances = []
    for point in points:
        projection = ((point - starts) * segments).sum(axis=1) / lengths_squared
        projection = np.clip(projection, 0.0, 1.0)
        closest = starts + projection[:, None] * segments
        distances.append(np.linalg.norm(point - closest, axis=1).min())
    return np.asarray(distances)


def test_trajectory_to_cubic_compilation_parity():
    geometry = _geometry()
    segments = curvature_to_cubic_segments(geometry, max_u_step=0.05)
    compiled = _bezier_points(segments)
    integrated = evaluate_curvature_stroke(geometry, count=2001).xy

    forward = _point_to_polyline_distances(integrated, compiled)
    backward = _point_to_polyline_distances(compiled, integrated)

    assert forward.max() <= 1e-3  # x-height-normalized units
    assert backward.max() <= 1e-3


def test_primitive_path_dispatches_curvature_profile():
    from inkgrammar.geometry import analyze_path

    template = PrimitiveTemplate.model_validate(
        {
            "id": "sas-e-trajectory-01",
            "kind": "curvature-profile",
            "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
            "curvature_stroke": {
                "length": 1.4,
                "start_angle_deg": 55.0,
                "curvature": PROFILE,
            },
        }
    )
    path = primitive_path(template, TopologyTransform())

    assert path.startswith("M 0 0")
    assert " C " in path
    geometry = CurvatureStrokeGeometry.model_validate(
        {"length": 1.4, "start_angle_deg": 55.0, "curvature": PROFILE}
    )
    end = evaluate_curvature_stroke(geometry, count=257).end_pose.position
    info = analyze_path(path)
    assert info.end[0] == pytest.approx(end[0], abs=1e-3)
    assert info.end[1] == pytest.approx(end[1], abs=1e-3)


def test_unit_transforms_round_trip_and_scale():
    points = np.array([[0.0, 0.0], [1.0, 0.5], [-0.25, 1.0]])
    runtime = glyph_units_to_runtime(points, x_height_units=362.5)

    np.testing.assert_allclose(runtime, points * 362.5)
    np.testing.assert_allclose(runtime_units_to_glyph(runtime, x_height_units=362.5), points)
    with pytest.raises(ValueError):
        glyph_units_to_runtime(points, x_height_units=0.0)
    with pytest.raises(ValueError):
        runtime_units_to_glyph(points, x_height_units=-1.0)



def test_fit_glyph_and_runtime_coordinate_boundaries_round_trip():
    points = np.array([[31.0, 0.0], [62.0, 15.5], [-7.75, 31.0]])
    glyph = fit_units_to_glyph(points, fit_x_height_units=31.0)
    runtime = fit_units_to_runtime(
        points, fit_x_height_units=31.0, x_height_units=362.5
    )

    np.testing.assert_allclose(glyph, points / 31.0)
    np.testing.assert_allclose(glyph_units_to_fit(glyph, fit_x_height_units=31.0), points)
    np.testing.assert_allclose(runtime, glyph * 362.5)
    np.testing.assert_allclose(
        runtime_units_to_fit(runtime, fit_x_height_units=31.0, x_height_units=362.5),
        points,
    )

    with pytest.raises(ValueError, match="fit_x_height_units"):
        fit_units_to_glyph(points, fit_x_height_units=0.0)
    with pytest.raises(ValueError, match="coordinates must be finite"):
        glyph_units_to_runtime(np.array([[np.nan, 0.0]]), x_height_units=362.5)


def test_render_trajectory_diagnostic_draws_the_model():
    panel = render_trajectory_diagnostic(_geometry(), size=128)

    assert panel.size == (128, 128)
    colors = {pixel for pixel in panel.getdata()}
    assert (20, 20, 20) in colors  # centerline
    assert (220, 0, 0) in colors  # curvature-change markers
    assert (0, 160, 0) in colors  # start marker/arrow
    assert (0, 0, 220) in colors  # end marker/arrow
