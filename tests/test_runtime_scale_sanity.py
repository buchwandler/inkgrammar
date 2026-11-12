from __future__ import annotations

import pytest

from inkgrammar.raster import scale_sanity


def test_scale_sanity_reports_documented_ratios():
    report = scale_sanity(ink_bbox_height=2.0, x_height=1.0, stroke_width=0.2)

    assert report.ink_bbox_height_over_x_height == pytest.approx(2.0)
    assert report.stroke_width_over_x_height == pytest.approx(0.2)
    assert report.stroke_width_over_ink_bbox_height == pytest.approx(0.1)
    assert report.flags == ()
    assert report.pathological is False


def test_scale_sanity_flags_stroke_wider_than_ink_bbox():
    report = scale_sanity(ink_bbox_height=0.5, x_height=1.0, stroke_width=0.5)

    assert report.flags == ("stroke_width >= ink_bbox_height",)
    assert report.pathological is True


def test_scale_sanity_allow_stroke_wider_than_ink_suppresses_flag():
    report = scale_sanity(
        ink_bbox_height=0.5,
        x_height=1.0,
        stroke_width=0.8,
        allow_stroke_wider_than_ink=True,
    )

    assert report.flags == ()
    assert report.pathological is False


@pytest.mark.parametrize(
    ("ink_bbox_height", "x_height", "stroke_width"),
    [
        (0.0, 1.0, 0.1),
        (1.0, 0.0, 0.1),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.1),
    ],
)
def test_scale_sanity_rejects_non_positive_inputs(ink_bbox_height, x_height, stroke_width):
    with pytest.raises(ValueError):
        scale_sanity(
            ink_bbox_height=ink_bbox_height,
            x_height=x_height,
            stroke_width=stroke_width,
        )
