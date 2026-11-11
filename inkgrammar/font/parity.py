from __future__ import annotations

from math import isclose
from pathlib import Path

from fontTools.ttLib import TTFont

from ..models import WritingModelV3
from ..shaping import shape_text
from .names import variant_glyph_names


def compare_runtime_to_font(
    model: WritingModelV3,
    font_path: Path,
    text: str,
    *,
    tolerance: float = 1.0,
) -> dict[str, object]:
    try:
        import uharfbuzz as hb
    except ImportError:
        return {
            "available": False,
            "reason": "uharfbuzz is not installed",
            "sequence_matches": None,
            "positions_match": None,
        }

    runtime = shape_text(model, text)
    variant_names = variant_glyph_names(model)
    expected_names = [
        variant_names[(glyph.character, glyph.variant_id)] for glyph in runtime.glyphs
    ]
    font_bytes = Path(font_path).read_bytes()
    ttfont = TTFont(font_path, lazy=False)
    glyph_order = ttfont.getGlyphOrder()
    units_per_em = ttfont["head"].unitsPerEm
    ttfont.close()

    face = hb.Face(font_bytes)
    hb_font = hb.Font(face)
    hb_font.scale = (units_per_em, units_per_em)
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    hb.shape(hb_font, buffer, {"rclt": True, "curs": True, "rlig": True, "kern": True})

    actual_names = [glyph_order[info.codepoint] for info in buffer.glyph_infos]
    origins: list[tuple[float, float]] = []
    pen_x = 0
    pen_y = 0
    for position in buffer.glyph_positions:
        origins.append((pen_x + position.x_offset, pen_y + position.y_offset))
        pen_x += position.x_advance
        pen_y += position.y_advance
    expected_origins = [(glyph.x, glyph.y) for glyph in runtime.glyphs]
    sequence_matches = actual_names == expected_names
    positions_match = len(origins) == len(expected_origins) and all(
        isclose(actual_x, expected_x, abs_tol=tolerance, rel_tol=0)
        and isclose(actual_y, expected_y, abs_tol=tolerance, rel_tol=0)
        for (actual_x, actual_y), (expected_x, expected_y) in zip(
            origins, expected_origins, strict=True
        )
    )
    return {
        "available": True,
        "text": text,
        "sequence_matches": sequence_matches,
        "positions_match": positions_match,
        "expected_glyphs": expected_names,
        "actual_glyphs": actual_names,
        "expected_origins": expected_origins,
        "actual_origins": origins,
        "tolerance_font_units": tolerance,
    }
