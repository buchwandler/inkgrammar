from __future__ import annotations

import pytest

from inkgrammar.geometry import analyze_path
from inkgrammar.io import load_model
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


def _runtime_ink_height(variant):
    boxes = [
        analyze_path(part.d).bbox
        for stroke in variant.strokes
        for part in getattr(stroke, "parts", [])
        if getattr(part, "d", None)
    ]
    return max(box[3] for box in boxes) - min(box[1] for box in boxes)


def test_canonical_sas_model_has_plausible_runtime_scale():
    model = load_model("data/models/sas-1968.yaml")
    pathological = []
    for character, glyph in model.glyphs.items():
        variant = glyph.variants[glyph.default_variant]
        report = scale_sanity(
            ink_bbox_height=_runtime_ink_height(variant),
            x_height=model.metrics.x_height,
            stroke_width=model.metrics.default_stroke_width,
        )
        if report.pathological:
            pathological.append(character)

    assert pathological == []
    a_height = _runtime_ink_height(model.glyphs["a"].variants["a.default"])
    assert 0.5 <= a_height / model.metrics.x_height <= 1.0
    assert 0.05 <= model.metrics.default_stroke_width / a_height <= 0.25
