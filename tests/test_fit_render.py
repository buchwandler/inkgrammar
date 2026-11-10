import numpy as np

from inkgrammar.fitting.render import render_paths


def test_fitting_renderer_returns_supersampled_coverage_and_component_layers():
    rendered = render_paths(
        {"part-a": "M 0 0 C 10 0 20 0 30 0", "part-b": "M 0 0 C 0 10 0 20 0 30"},
        canvas_size=(64, 64),
        origin=(8, 48),
        pixels_per_unit=1.0,
        stroke_width=2.0,
        supersampling=4,
        stroke_ids={"part-a": "stroke-1", "part-b": "stroke-1"},
    )

    assert rendered.coverage.shape == (64, 64)
    assert set(rendered.part_layers) == {"part-a", "part-b"}
    assert set(rendered.stroke_layers) == {"stroke-1"}
    assert rendered.coverage.dtype == np.float32
    assert rendered.coverage.min() == 0
    assert rendered.coverage.max() == 1
    assert np.any((rendered.coverage > 0) & (rendered.coverage < 1))
    np.testing.assert_array_equal(
        rendered.coverage,
        np.maximum(rendered.part_layers["part-a"], rendered.part_layers["part-b"]),
    )


def test_renderer_keeps_the_requested_source_frame_without_cropping():
    rendered = render_paths(
        {"part": "M 0 0 C 5 0 10 0 15 0"},
        canvas_size=(80, 40),
        origin=(20, 20),
        pixels_per_unit=1,
        stroke_width=2,
        supersampling=2,
    )

    ys, xs = np.nonzero(rendered.coverage)
    assert xs.min() >= 17
    assert xs.max() <= 38
    assert ys.min() < 20 < ys.max()
