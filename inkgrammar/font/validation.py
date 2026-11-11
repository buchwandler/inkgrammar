from __future__ import annotations

from pathlib import Path

from fontTools.ttLib import TTFont

from ..models import WritingModelV3
from .models import FontPlan, FontValidationResult
from .names import glyph_unicode_values, variant_glyph_names

_REQUIRED_TABLES = ("OS/2", "cmap", "glyf", "head", "hhea", "hmtx", "maxp", "name", "post")


def _feature_tags(font: TTFont, table_tag: str) -> set[str]:
    if table_tag not in font or font[table_tag].table.FeatureList is None:
        return set()
    return {record.FeatureTag for record in font[table_tag].table.FeatureList.FeatureRecord}


def validate_font(
    path: Path,
    *,
    model: WritingModelV3 | None = None,
    plan: FontPlan | None = None,
) -> FontValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        font = TTFont(path, lazy=False)
    except Exception as error:
        return FontValidationResult(False, 0, (), (f"could not reopen font: {error}",), ())

    tables = tuple(sorted(font.keys()))
    missing_tables = sorted(set(_REQUIRED_TABLES) - set(tables))
    if missing_tables:
        errors.append("missing required tables: " + ", ".join(missing_tables))
    glyph_order = font.getGlyphOrder()
    if not glyph_order or glyph_order[0] != ".notdef":
        errors.append("glyph order must begin with .notdef")
    best_cmap = {} if "cmap" not in font else font.getBestCmap() or {}

    if model is not None:
        if "head" in font and font["head"].unitsPerEm != model.metrics.units_per_em:
            errors.append("unitsPerEm does not match the writing model")
        if plan is None:
            names = variant_glyph_names(model)
            expected_cmap = {
                codepoint: names[(character, glyph.default_variant)]
                for character, glyph in model.glyphs.items()
                for codepoint in glyph_unicode_values(character, glyph.unicode)
            }
        else:
            expected_cmap = plan.cmap
        for codepoint, glyph_name in expected_cmap.items():
            if best_cmap.get(codepoint) != glyph_name:
                errors.append(f"cmap U+{codepoint:04X} does not map to {glyph_name}")
        if "hhea" in font:
            if font["hhea"].ascent != round(model.metrics.ascender):
                errors.append("hhea ascender does not match the writing model")
            if font["hhea"].descent != round(model.metrics.descender):
                errors.append("hhea descender does not match the writing model")
        if plan is not None:
            if "hmtx" in font:
                for name, expected in plan.horizontal_metrics.items():
                    if name not in font["hmtx"].metrics or font["hmtx"].metrics[name] != expected:
                        errors.append(f"horizontal metrics do not match the plan for {name}")
            if "rclt" in plan.feature_source and "rclt" not in _feature_tags(font, "GSUB"):
                errors.append("required rclt feature is missing")
            if "curs" in plan.feature_source and "curs" not in _feature_tags(font, "GPOS"):
                errors.append("curs feature is missing")
            if "rlig" in plan.feature_source and "rlig" not in _feature_tags(font, "GSUB"):
                errors.append("rlig feature is missing")
        else:
            if model.context_rules and "rclt" not in _feature_tags(font, "GSUB"):
                warnings.append("model has contextual rules but the font has no rclt feature")
            if any(
                variant.entry is not None or variant.exit is not None
                for glyph in model.glyphs.values()
                for variant in glyph.variants.values()
            ) and "curs" not in _feature_tags(font, "GPOS"):
                warnings.append("model has anchors but the font has no curs feature")

    if "glyf" in font:
        glyf = font["glyf"]
        for name in glyph_order:
            if name not in glyf:
                errors.append(f"glyph outline is missing for {name}")
                continue
            glyph = glyf[name]
            if glyph.isComposite():
                continue
            glyph.recalcBounds(glyf)
            if glyph.numberOfContours < 0:
                errors.append(f"invalid simple glyph contour count for {name}")

    font.close()
    return FontValidationResult(
        valid=not errors,
        glyph_count=len(glyph_order),
        tables=tables,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
