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
