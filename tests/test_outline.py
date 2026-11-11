from __future__ import annotations

import math

import pytest

from inkgrammar.outline import (
    OutlineConfig,
    _signed_area,
    expand_path,
    expand_shaped_text,
    expand_tap,
)
from inkgrammar.shaping import ShapedGlyph, ShapedStroke, ShapedText


def assert_closed_and_finite(result):
    assert result.contours
    for contour in result.contours:
        assert contour[0] == contour[-1]
        assert all(math.isfinite(value) for point in contour for value in point)
    assert result.bounds is not None
    assert all(math.isfinite(value) for value in result.bounds)


def test_open_line_has_deterministic_round_caps_and_expected_bounds():
    first = expand_path("M 0 0 L 10 0", 2.0)
    second = expand_path("M 0 0 L 10 0", 2.0)

    assert first == second
    assert_closed_and_finite(first)
    assert first.bounds == (-1.0, -1.0, 11.0, 1.0)


def test_curve_is_adaptively_sampled_and_attached_segments_share_one_outline():
    curve = expand_path("M 0 0 C 0 100 100 100 100 0", 4.0)
    composed = expand_path("M 0 0 L 10 0 L 20 0", 4.0)

    assert len(curve.contours[0]) > len(composed.contours[0])
    assert len(composed.contours) == 1
    assert_closed_and_finite(curve)
    assert_closed_and_finite(composed)


def test_variable_width_changes_bounds_and_is_adaptively_subdivided():
    result = expand_path(
        "M 0 0 L 10 0",
        [(0.0, 2.0), (0.25, 3.0), (1.0, 10.0)],
        config=OutlineConfig(max_width_error=0.05),
    )

    assert_closed_and_finite(result)
    assert result.bounds == (-1.0, -5.0, 15.0, 5.0)
    assert len(result.contours[0]) > 8


def test_closed_loop_has_oppositely_wound_inner_and_outer_contours():
    result = expand_path(
        "M 0 0 C 0 100 100 100 100 0 C 100 -100 0 -100 0 0 Z",
        8.0,
    )

    assert len(result.contours) == 2
    assert_closed_and_finite(result)
    assert _signed_area(result.contours[0]) * _signed_area(result.contours[1]) < 0


def test_tap_becomes_a_closed_repeatable_circle():
    first = expand_tap((4, -3), 10)
    second = expand_tap((4, -3), 10)

    assert first == second
    assert_closed_and_finite(first)
    assert first.bounds == (-1.0, -8.0, 9.0, 2.0)


def test_shaped_text_expands_paths_and_taps_with_per_stroke_widths():
    shaped = ShapedText(
        glyphs=(
            ShapedGlyph(
                character="a",
                variant_id="a.default",
                x=0,
                y=0,
                advance=20,
                entry=None,
                exit=None,
                strokes=(
                    ShapedStroke("body", "body", d="M 0 0 L 10 0"),
                    ShapedStroke("dot", "dot", tap=(15, 0)),
                ),
            ),
        ),
        joins=(),
        width=20,
    )

    result = expand_shaped_text(
        shaped,
        4,
        width_profiles={"body": [(0.0, 2.0), (1.0, 4.0)]},
    )

    assert len(result.contours) == 2
    assert_closed_and_finite(result)
    assert result.bounds == (-1.0, -2.0, 17.0, 2.0)


def test_width_profile_rejects_invalid_knots():
    with pytest.raises(ValueError, match="positive widths"):
        expand_path("M 0 0 L 10 0", [(0.0, 2.0), (0.5, -1.0), (1.0, 4.0)])
