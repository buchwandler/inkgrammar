from __future__ import annotations

from html import escape
from pathlib import Path

from .geometry import sample_path
from .model import compile_glyph, compile_text
from .models import RuntimeWritingModel, WritingModelV3
from .outline import expand_shaped_text
from .shaping import ShapedGlyph, ShapedStroke, ShapedText, shape_text


def _legacy_shaped_text(model, text: str) -> ShapedText:
    compiled = compile_text(model, text)
    groups: list[list[ShapedStroke]] = [[] for _ in compiled.characters]
    for stroke in compiled.strokes:
        prefix, separator, _ = stroke.id.partition(":")
        if separator and prefix.isdigit():
            index = int(prefix)
        elif stroke.id.startswith("join:"):
            index = int(stroke.id.rsplit(":", 1)[-1])
        else:
            index = 0
        if groups:
            groups[min(index, len(groups) - 1)].append(
                ShapedStroke(stroke.id, stroke.role, d=stroke.d, tap=stroke.tap)
            )
    glyphs = []
    x = 0.0
    count = len(compiled.characters)
    for index, character in enumerate(compiled.characters):
        if count == 1:
            form = "isolated"
        elif index == 0:
            form = "initial"
        elif index == count - 1:
            form = "final"
        else:
            form = "medial"
        glyph = compile_glyph(model, character, form)
        glyphs.append(
            ShapedGlyph(
                character=character,
                variant_id=glyph.form_name,
                x=x,
                y=0.0,
                advance=glyph.advance_width,
                entry=None,
                exit=None,
                strokes=tuple(groups[index]),
            )
        )
        x += glyph.advance_width
    return ShapedText(glyphs=tuple(glyphs), joins=(), width=compiled.width)


def _shaped_text(model: RuntimeWritingModel, text: str) -> ShapedText:
    if isinstance(model, WritingModelV3):
        return shape_text(model, text, allow_synthetic_bridges=True)
    return _legacy_shaped_text(model, text)


def render_text_svg(
    model: RuntimeWritingModel,
    text: str,
    *,
    show_centerline: bool = True,
    show_outline: bool = False,
    show_anchors: bool = True,
    show_tangents: bool = True,
    show_origins: bool = True,
    show_guides: bool = True,
    show_labels: bool = True,
) -> str:
    """Render text and optional shaping diagnostics from the shared shaped plan."""
    plan = _shaped_text(model, text)
    metrics = model.metrics
    points = [
        point
        for stroke in plan.strokes
        if stroke.d is not None
        for point in sample_path(stroke.d, 32)
    ]
    points.extend(stroke.tap for stroke in plan.strokes if stroke.tap is not None)
    padding = metrics.default_stroke_width
    left = min([0.0, *(point[0] for point in points)]) - padding
    right = max([plan.width, *(point[0] for point in points)]) + padding
    bottom = min([metrics.descender, *(point[1] for point in points)]) - padding
    top = max([metrics.ascender, *(point[1] for point in points)]) + padding
    width = max(1.0, right - left)
    height = max(1.0, top - bottom)

    def n(value: float) -> str:
        return format(value, ".6g")

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n(width)} {n(height)}">',
        f'<g transform="translate({n(-left)} {n(top)}) scale(1 -1)">',
    ]
    if show_guides:
        guides = (
            ("baseline", metrics.baseline),
            ("x-height", metrics.x_height),
            ("ascender", metrics.ascender),
            ("descender", metrics.descender),
        )
        for label, y in guides:
            lines.append(
                f'<path data-guide="{label}" d="M {n(left)} {n(y)} L {n(right)} {n(y)}" '
                'fill="none" stroke="#b8b8b8" stroke-width="1" stroke-dasharray="4 4"/>'
            )
    if show_origins:
        for glyph in plan.glyphs:
            lines.append(
                f'<path data-origin="{escape(glyph.variant_id, quote=True)}" '
                f'd="M {n(glyph.x)} {n(metrics.descender)} L {n(glyph.x)} {n(metrics.ascender)}" '
                'fill="none" stroke="#89a" stroke-width="1" stroke-dasharray="2 3"/>'
            )
    if show_outline:
        outline = expand_shaped_text(plan, metrics.default_stroke_width)
        for index, contour in enumerate(outline.contours):
            commands = [f"M {n(contour[0][0])} {n(contour[0][1])}"]
            commands.extend(f"L {n(x)} {n(y)}" for x, y in contour[1:-1])
            lines.append(
                f'<path data-outline="{index}" d="{" ".join(commands)} Z" '
                'fill="#3572a5" fill-opacity="0.3" fill-rule="nonzero" stroke="none"/>'
            )
    for stroke in plan.strokes:
        if stroke.d is not None and show_centerline:
            lines.append(
                f'<path data-stroke="{escape(stroke.id, quote=True)}" '
                f'd="{escape(stroke.d, quote=True)}" fill="none" stroke="#111" '
                f'stroke-width="{n(metrics.default_stroke_width)}" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        elif stroke.tap is not None and show_centerline:
            lines.append(
                f'<circle data-stroke="{escape(stroke.id, quote=True)}" '
                f'cx="{n(stroke.tap[0])}" cy="{n(stroke.tap[1])}" '
                f'r="{n(metrics.default_stroke_width / 2)}" fill="#111"/>'
            )
    for glyph in plan.glyphs:
        for name, anchor in (("entry", glyph.entry), ("exit", glyph.exit)):
            if anchor is None:
                continue
            if show_anchors:
                lines.append(
                    f'<circle data-anchor="{name}" '
                    f'data-variant="{escape(glyph.variant_id, quote=True)}" '
                    f'cx="{n(anchor.x)}" cy="{n(anchor.y)}" '
                    f'r="{n(metrics.default_stroke_width / 7)}" fill="#e44"/>'
                )
            if show_tangents:
                length = metrics.default_stroke_width * 2
                lines.append(
                    f'<path data-tangent="{name}" d="M {n(anchor.x)} {n(anchor.y)} '
                    f"L {n(anchor.x + anchor.tangent_x * length)} "
                    f'{n(anchor.y + anchor.tangent_y * length)}" '
                    'fill="none" stroke="#e44" stroke-width="2"/>'
                )
        if show_labels:
            label = escape(glyph.variant_id)
            lines.append(
                f'<text data-variant="{label}" '
                f'transform="translate({n(glyph.x)} {n(top)}) scale(1 -1)" '
                'font-size="24" fill="#174a7e">'
                f"{label}</text>"
            )
    if show_labels:
        for join in plan.joins:
            if join.rule_id:
                lines.append(
                    f'<text data-join-rule="{escape(join.rule_id, quote=True)}" '
                    f'transform="translate(0 {n(top - 28)}) scale(1 -1)" '
                    'font-size="18" fill="#7a3e00">'
                    f"{escape(join.rule_id)}</text>"
                )
    lines.extend(("</g>", "</svg>", ""))
    return "\n".join(lines)


def write_text_svg(
    model: RuntimeWritingModel,
    text: str,
    output: str | Path,
    **options,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_text_svg(model, text, **options), encoding="utf-8")
    return path
