from __future__ import annotations

import json
import re
from pathlib import Path

from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from ..models import WritingModelV3
from ..outline import OutlineConfig, expand_shaped_text
from ..shaping import ShapedText, shape_glyph
from .features import generate_feature_source
from .models import (
    FontBuildConfig,
    FontBuildResult,
    FontGlyphOutline,
    FontPlan,
    FontValidationResult,
)
from .names import glyph_unicode_values, variant_glyph_names
from .validation import validate_font


class FontReadinessError(ValueError):
    def __init__(self, blockers: tuple[str, ...]):
        self.blockers = blockers
        super().__init__("font build is blocked: " + "; ".join(blockers))


def _model_blockers(model: WritingModelV3) -> tuple[str, ...]:
    blockers: list[str] = []
    if model.license.strip().upper() in {"NOASSERTION", "UNKNOWN", "REPLACE-AFTER-RIGHTS-REVIEW"}:
        blockers.append("font license or rights review is unresolved")
    promotions = model.provenance.get("primitive_promotion")
    if isinstance(promotions, dict):
        fallback_ids = sorted(key for key, value in promotions.items() if value != "promoted")
        if fallback_ids:
            blockers.append(
                "unpromoted primitive geometry: " + ", ".join(str(value) for value in fallback_ids)
            )
    if any(rule.mode == "synthetic-bridge" for rule in model.connection_rules):
        blockers.append("synthetic bridge connection rules cannot be exported as release geometry")
    return tuple(blockers)


def create_font_plan(
    model: WritingModelV3,
    config: FontBuildConfig | None = None,
) -> FontPlan:
    config = config or FontBuildConfig()
    names = variant_glyph_names(model)
    glyph_order = [".notdef"]
    if " " not in model.glyphs:
        glyph_order.append("space")
    outlines: dict[str, FontGlyphOutline] = {".notdef": FontGlyphOutline((), None)}
    horizontal_metrics: dict[str, tuple[int, int]] = {
        ".notdef": (model.metrics.units_per_em // 2, 0)
    }
    cmap: dict[int, str] = {}
    outline_config = OutlineConfig(
        max_centerline_error=config.quadratic_error,
        max_width_error=config.quadratic_error,
    )

    characters = sorted(model.glyphs, key=lambda character: ord(character[0]))
    for character in characters:
        glyph = model.glyphs[character]
        default_name = names[(character, glyph.default_variant)]
        for codepoint in glyph_unicode_values(character, glyph.unicode):
            previous = cmap.get(codepoint)
            if previous is not None and previous != default_name:
                raise ValueError(f"Unicode U+{codepoint:04X} maps to multiple glyphs")
            cmap[codepoint] = default_name
        variant_ids = [glyph.default_variant]
        variant_ids.extend(sorted(set(glyph.variants) - {glyph.default_variant}))
        for variant_id in variant_ids:
            variant = glyph.variants[variant_id]
            name = names[(character, variant_id)]
            glyph_order.append(name)
            shaped_glyph = shape_glyph(model, character, variant_id)
            shaped = ShapedText(
                glyphs=(shaped_glyph,),
                joins=(),
                width=variant.advance_width,
            )
            expanded = expand_shaped_text(
                shaped,
                model.metrics.default_stroke_width,
                config=outline_config,
            )
            outlines[name] = FontGlyphOutline(expanded.contours, expanded.bounds)
            advance = round(variant.advance_width)
            left_side_bearing = round(expanded.bounds[0]) if expanded.bounds else 0
            horizontal_metrics[name] = (advance, left_side_bearing)

    if " " not in model.glyphs:
        horizontal_metrics["space"] = (model.metrics.units_per_em // 2, 0)
        outlines["space"] = FontGlyphOutline((), None)
        cmap.setdefault(0x20, "space")

    provisional = FontPlan(
        glyph_order=tuple(glyph_order),
        variant_names=names,
        cmap=dict(sorted(cmap.items())),
        horizontal_metrics=horizontal_metrics,
        outlines=outlines,
        feature_source="",
        issues=(),
        blockers=(),
    )
    feature_source, issues = generate_feature_source(model, provisional)
    blockers = tuple(sorted({*_model_blockers(model), *issues}))
    return FontPlan(
        glyph_order=provisional.glyph_order,
        variant_names=provisional.variant_names,
        cmap=provisional.cmap,
        horizontal_metrics=provisional.horizontal_metrics,
        outlines=provisional.outlines,
        feature_source=feature_source,
        issues=issues,
        blockers=blockers,
    )


def _tt_glyph(outline: FontGlyphOutline):
    pen = TTGlyphPen(None)
    for contour in outline.contours:
        points: list[tuple[int, int]] = []
        for x, y in contour:
            point = (round(x), round(y))
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[-1] == points[0]:
            points.pop()
        if len(points) < 3:
            continue
        pen.moveTo(points[0])
        for index, point in enumerate(points):
            endpoint = points[(index + 1) % len(points)]
            control = (
                round((point[0] + endpoint[0]) / 2),
                round((point[1] + endpoint[1]) / 2),
            )
            pen.qCurveTo(control, endpoint)
        pen.closePath()
    return pen.glyph()


def _safe_postscript_name(family: str, style: str) -> str:
    value = re.sub(r"[^A-Za-z0-9-]+", "", family.replace(" ", ""))
    style_value = re.sub(r"[^A-Za-z0-9-]+", "", style.replace(" ", ""))
    return f"{value or 'InkGrammar'}-{style_value or 'Regular'}"


def _build_font(model: WritingModelV3, plan: FontPlan, config: FontBuildConfig) -> TTFont:
    units_per_em = model.metrics.units_per_em
    font_builder = FontBuilder(units_per_em, isTTF=True)
    font_builder.setupGlyphOrder(list(plan.glyph_order))
    font_builder.setupCharacterMap(plan.cmap)
    font_builder.setupGlyf({name: _tt_glyph(plan.outlines[name]) for name in plan.glyph_order})
    font_builder.setupHorizontalMetrics(plan.horizontal_metrics)
    ascender = round(model.metrics.ascender)
    descender = round(model.metrics.descender)
    font_builder.setupHorizontalHeader(ascent=ascender, descent=descender, lineGap=0)
    font_builder.setupOS2(
        sTypoAscender=ascender,
        sTypoDescender=descender,
        sTypoLineGap=0,
        usWinAscent=max(0, ascender),
        usWinDescent=max(0, -descender),
        sxHeight=round(model.metrics.x_height),
        sCapHeight=round(model.metrics.cap_height or model.metrics.x_height),
    )
    family = config.family_name or model.name.get("en") or model.id
    full_name = f"{family} {config.style_name}".strip()
    font_builder.setupNameTable(
        {
            "familyName": family,
            "styleName": config.style_name,
            "uniqueFontIdentifier": f"{family};{config.style_name};InkGrammar",
            "fullName": full_name,
            "psName": _safe_postscript_name(family, config.style_name),
            "version": "Version 1.000",
        }
    )
    font_builder.setupPost(keepGlyphNames=True)
    font_builder.setupMaxp()
    font_builder.setupHead()
    font = font_builder.font
    addOpenTypeFeaturesFromString(font, plan.feature_source)
    if config.reproducible:
        font["head"].created = 2_082_844_800
        font["head"].modified = 2_082_844_800
    return font


def build_opentype_font(
    model: WritingModelV3,
    output: Path,
    config: FontBuildConfig | None = None,
    *,
    allow_development: bool = False,
) -> FontBuildResult:
    config = config or FontBuildConfig()
    plan = create_font_plan(model, config)
    if plan.blockers and not allow_development:
        raise FontReadinessError(plan.blockers)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_path = output.with_suffix(".fea")
    report_path = output.with_suffix(".report.json")
    font = _build_font(model, plan, config)
    font.save(output)
    feature_path.write_text(plan.feature_source, encoding="utf-8", newline="\n")
    validation = validate_font(output, model=model, plan=plan)
    parity_blocker = "OpenType shaping parity has not been verified"
    blockers = tuple(sorted({*plan.blockers, parity_blocker, *validation.errors}))
    report = {
        "schema_version": 1,
        "font": output.name,
        "valid": validation.valid,
        "font_ready": not blockers,
        "glyph_count": validation.glyph_count,
        "tables": list(validation.tables),
        "blockers": list(blockers),
        "issues": list(plan.issues),
        "quadratic_error_font_units": config.quadratic_error,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return FontBuildResult(
        path=output,
        glyph_count=validation.glyph_count,
        tables=validation.tables,
        feature_source_path=feature_path,
        report_path=report_path,
        font_ready=not blockers,
        blockers=blockers,
    )


def validate_opentype_font(
    path: Path,
    model: WritingModelV3 | None = None,
) -> FontValidationResult:
    return validate_font(Path(path), model=model)
