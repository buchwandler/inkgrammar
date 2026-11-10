import numpy as np
import pytest

from inkgrammar.fitting.family_fit import primitive_path
from inkgrammar.fitting.models import (
    CubicSegment,
    PrimitiveFrame,
    PrimitiveTemplate,
    TopologyTransform,
)
from inkgrammar.fitting.objective import soft_mask_loss
from inkgrammar.fitting.optimizer import OptimizerConfig, optimize_parameters
from inkgrammar.fitting.parameterization import ParameterSpace
from inkgrammar.fitting.render import render_paths


@pytest.mark.parametrize(
    ("segments", "free_parameters", "initial", "known"),
    [
        (
            [
                CubicSegment(
                    c1=(0.25, 0.32),
                    c2=(0.75, 0.68),
                    end=(1.0, 0.0),
                )
            ],
            ["segments.0.c1.y"],
            [0.32],
            [0.74],
        ),
        (
            [
                CubicSegment(
                    c1=(0.12, 0.22),
                    c2=(0.30, 0.38),
                    end=(0.5, 0.8),
                ),
                CubicSegment(
                    c1=(0.70, 0.38),
                    c2=(0.88, 0.22),
                    end=(1.0, 0.0),
                ),
            ],
            ["segments.0.c2.y", "segments.1.c1.y"],
            [0.38, 0.38],
            [0.68, 0.68],
        ),
    ],
)
def test_bounded_optimizer_recovers_known_cubic_template(segments, free_parameters, initial, known):
    template = PrimitiveTemplate(
        id="synthetic-oval",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(1.0, 0.0)),
        segments=segments,
        free_parameters=free_parameters,
        bounds={name: (0.1, 0.9) for name in free_parameters},
    )
    space = ParameterSpace.from_primitive(template)
    start = space.encode(template)
    assert np.allclose(start, initial)
    target = space.decode(np.asarray(known), template)
    transform = TopologyTransform(scale_x=46.0, scale_y=42.0)

    reference = render_paths(
        {"oval": primitive_path(target, transform)},
        canvas_size=(64, 64),
        origin=(8.0, 55.0),
        pixels_per_unit=1.0,
        stroke_width=3.0,
        supersampling=4,
    ).coverage

    def objective(vector):
        candidate = space.decode(vector, template)
        generated = render_paths(
            {"oval": primitive_path(candidate, transform)},
            canvas_size=(64, 64),
            origin=(8.0, 55.0),
            pixels_per_unit=1.0,
            stroke_width=3.0,
            supersampling=4,
        ).coverage
        return soft_mask_loss(reference, generated)

    initial_loss = objective(start)
    result = optimize_parameters(
        space,
        objective,
        config=OptimizerConfig(max_iterations=250, tolerance=1e-7),
    )
    assert result.objective < initial_loss * 0.25
    assert np.allclose(result.parameters, known, atol=0.12)
    assert all(
        lower <= value <= upper
        for value, (lower, upper) in zip(result.parameters, space.bounds(), strict=True)
    )


def test_geometric_residuals_normal_track_primitive_without_contaminated_region_bias():
    from scipy.ndimage import distance_transform_edt

    from inkgrammar.fitting.alignment import (
        bilinear_sample,
        normal_track_ink,
        sample_path_by_arclength,
    )
    from inkgrammar.fitting.residuals import occurrence_geometric_residuals

    path_points = sample_path_by_arclength("M 0 0 C 0 0 10 0 10 0", sample_count=6)
    assert np.allclose(np.diff(path_points[:, 0]), 2.0)
    field = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert bilinear_sample(field, np.array([[0.5, 0.5]]))[0] == pytest.approx(1.5)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:56, 19:22] = 1
    mask[8:56, 29:32] = 1
    distance = distance_transform_edt(~mask.astype(bool)).astype(np.float32)
    candidate_points = np.column_stack((np.full(24, 21.0), np.linspace(10.0, 53.0, 24)))
    track = normal_track_ink(mask, candidate_points, expected_stroke_width=3.0)
    result = occurrence_geometric_residuals(
        candidate_points,
        mask,
        distance,
        expected_stroke_width=3.0,
    )

    assert track.valid.all()
    np.testing.assert_allclose(track.centerline_points[:, 0], 20.0, atol=0.1)
    assert result.diagnostics["path_to_ink_rms_px"] == 0.0
    assert result.diagnostics["centerline_rms_px"] == pytest.approx(1.0, abs=0.1)
    assert result.diagnostics["centerline_valid_fraction"] == 1.0
    assert np.isfinite(result.residuals).all()
    weighted_result = occurrence_geometric_residuals(
        candidate_points,
        mask,
        distance,
        expected_stroke_width=3.0,
        endpoint_weight=1.0,
        tangent_weight=1.0,
    )
    assert np.linalg.norm(weighted_result.endpoint) > 0.0

    diagonal_points = np.column_stack((np.linspace(19.0, 23.0, 24), np.linspace(10.0, 53.0, 24)))
    diagonal = occurrence_geometric_residuals(
        diagonal_points,
        mask,
        distance,
        expected_stroke_width=3.0,
        tangent_weight=1.0,
    )
    assert np.linalg.norm(diagonal.tangent) > 0.0


def test_normal_tracking_gates_ambiguous_strokes_and_regularizes_transform_deltas():
    from inkgrammar.fitting.alignment import normal_track_ink
    from inkgrammar.fitting.models import OccurrenceDelta
    from inkgrammar.fitting.residuals import transform_prior_residuals

    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[6:42, 19:22] = 1
    mask[6:42, 27:30] = 1
    points = np.column_stack((np.full(12, 24.5), np.linspace(8.0, 40.0, 12)))
    track = normal_track_ink(mask, points, expected_stroke_width=3.0)

    assert not track.valid.any()
    prior = transform_prior_residuals(OccurrenceDelta(dx=2.0, log_scale_y=-0.06))
    np.testing.assert_allclose(prior, [1.0, 0.0, 0.0, -1.0, 0.0])
