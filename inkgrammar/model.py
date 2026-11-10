from __future__ import annotations

from dataclasses import dataclass

from .geometry import (
    analyze_path,
    cubic_bridge,
    path_without_initial_move,
    transform_path,
    translate_path,
)
from .models import (
    ComponentPart,
    GlyphForm,
    PathStrokeRecipe,
    RawPathPart,
    TapStrokeRecipe,
    WritingModel,
)


@dataclass(frozen=True)
class CompiledStroke:
    id: str
    role: str
    d: str | None = None
    tap: tuple[float, float] | None = None


@dataclass(frozen=True)
class CompiledGlyph:
    character: str
    form_name: str
    advance_width: float
    strokes: tuple[CompiledStroke, ...]
    entry: tuple[str, str, str] | None
    exit: tuple[str, str, str] | None


@dataclass(frozen=True)
class CompiledText:
    strokes: tuple[CompiledStroke, ...]
    width: float
    characters: tuple[str, ...]


def _form_name(has_prev: bool, has_next: bool) -> str:
    if has_prev and has_next:
        return "medial"
    if has_prev:
        return "final"
    if has_next:
        return "initial"
    return "isolated"


def _select_form(model: WritingModel, character: str, form_name: str) -> tuple[str, GlyphForm]:
    glyph = model.glyphs[character]
    for candidate in (form_name, "default", "isolated"):
        if candidate in glyph.forms:
            return candidate, glyph.forms[candidate]
    raise ValueError(f"glyph {character!r} has no usable form")


def _part_path(model: WritingModel, part: ComponentPart | RawPathPart) -> str:
    if isinstance(part, ComponentPart):
        if part.component not in model.components:
            raise ValueError(f"unknown component: {part.component}")
        d = model.components[part.component].d
    else:
        d = part.d
    return transform_path(d, part.transform)


def _compiled_part_paths(model: WritingModel, recipe: PathStrokeRecipe) -> tuple[str, ...]:
    paths: list[str] = []
    current_end = None

    for index, part in enumerate(recipe.parts):
        part_d = _part_path(model, part)
        info = analyze_path(part_d)
        if index == 0:
            compiled_d = part_d
        elif part.attach:
            assert current_end is not None
            dx = current_end[0] - info.start[0]
            dy = current_end[1] - info.start[1]
            compiled_d = translate_path(part_d, dx, dy)
        else:
            raise ValueError(
                "a single pen-down stroke cannot contain a detached component; "
                "use another stroke recipe"
            )
        paths.append(compiled_d)
        current_end = analyze_path(compiled_d).end
    return tuple(paths)


def _compile_path_recipe(model: WritingModel, recipe: PathStrokeRecipe) -> str:
    paths = _compiled_part_paths(model, recipe)
    return paths[0] + "".join(" " + path_without_initial_move(path) for path in paths[1:])


def compile_glyph(
    model: WritingModel,
    character: str,
    form_name: str = "isolated",
) -> CompiledGlyph:
    if character not in model.glyphs:
        raise KeyError(f"unknown glyph: {character!r}")
    selected_name, form = _select_form(model, character, form_name)

    strokes: list[CompiledStroke] = []
    for recipe in form.strokes:
        if isinstance(recipe, TapStrokeRecipe):
            strokes.append(
                CompiledStroke(
                    id=recipe.id,
                    role=recipe.role,
                    tap=recipe.at,
                )
            )
        else:
            strokes.append(
                CompiledStroke(
                    id=recipe.id,
                    role=recipe.role,
                    d=_compile_path_recipe(model, recipe),
                )
            )

    def endpoint_tuple(endpoint):
        if endpoint is None:
            return None
        return (endpoint.stroke, endpoint.endpoint, endpoint.class_)

    return CompiledGlyph(
        character=character,
        form_name=selected_name,
        advance_width=form.advance_width,
        strokes=tuple(strokes),
        entry=endpoint_tuple(form.connections.entry),
        exit=endpoint_tuple(form.connections.exit),
    )


def _connection_rule(model: WritingModel, from_class: str, to_class: str):
    for rule in model.connection_rules:
        if rule.from_class == from_class and rule.to_class == to_class:
            return rule
    return None


def _stroke_by_id(glyph: CompiledGlyph, stroke_id: str) -> CompiledStroke:
    for stroke in glyph.strokes:
        if stroke.id == stroke_id:
            return stroke
    raise ValueError(f"connection references missing stroke {stroke_id!r}")


def compile_text(
    model: WritingModel,
    text: str,
    *,
    gap_override: float | None = None,
    tension_override: float | None = None,
) -> CompiledText:
    characters = tuple(character for character in text if character in model.glyphs)
    if not characters:
        return CompiledText(strokes=(), width=0, characters=())

    compiled: list[CompiledGlyph] = []
    for index, character in enumerate(characters):
        has_prev = index > 0
        has_next = index < len(characters) - 1
        compiled.append(compile_glyph(model, character, _form_name(has_prev, has_next)))

    output: list[CompiledStroke] = []
    x_origins = [0.0]
    connectors: list[tuple[int, str, float, float]] = []

    for index in range(len(compiled) - 1):
        current = compiled[index]
        nxt = compiled[index + 1]
        gap = 0.0
        tension = 60.0
        rule = None

        if current.exit and nxt.entry:
            rule = _connection_rule(model, current.exit[2], nxt.entry[2])
        if rule:
            gap = rule.gap
            tension = rule.tension

        pair = current.character + nxt.character
        override = model.pair_overrides.get(pair)
        if override:
            gap += override.gap_adjust
            if override.tension is not None:
                tension = override.tension

        if gap_override is not None:
            gap += gap_override
        if tension_override is not None:
            tension = tension_override

        next_origin = x_origins[-1] + current.advance_width + gap
        x_origins.append(next_origin)
        if rule and current.exit and nxt.entry:
            if rule.effective_mode == "continuous-bridge":
                connectors.append((index, pair, next_origin, tension))
            elif rule.effective_mode == "direct":
                left_stroke = _stroke_by_id(current, current.exit[0])
                right_stroke = _stroke_by_id(nxt, nxt.entry[0])
                if left_stroke.d is None or right_stroke.d is None:
                    raise ValueError(f"direct connection {pair!r} needs path endpoints")
                left_info = analyze_path(left_stroke.d)
                right_info = analyze_path(right_stroke.d)
                left_point = left_info.start if current.exit[1] == "start" else left_info.end
                right_point = right_info.start if nxt.entry[1] == "start" else right_info.end
                dx = left_point[0] + x_origins[-2] - right_point[0] - next_origin
                dy = left_point[1] - right_point[1]
                if dx * dx + dy * dy > 1e-12:
                    raise ValueError(f"direct connection {pair!r} endpoints do not meet")

    for index, (glyph, x_origin) in enumerate(zip(compiled, x_origins, strict=False)):
        for stroke in glyph.strokes:
            if stroke.d is not None:
                output.append(
                    CompiledStroke(
                        id=f"{index}:{stroke.id}",
                        role=stroke.role,
                        d=translate_path(stroke.d, x_origin, 0),
                    )
                )
            elif stroke.tap is not None:
                output.append(
                    CompiledStroke(
                        id=f"{index}:{stroke.id}",
                        role=stroke.role,
                        tap=(stroke.tap[0] + x_origin, stroke.tap[1]),
                    )
                )

    for index, pair, next_origin, tension in connectors:
        left = compiled[index]
        right = compiled[index + 1]
        if not left.exit or not right.entry:
            continue

        left_stroke = _stroke_by_id(left, left.exit[0])
        right_stroke = _stroke_by_id(right, right.entry[0])
        if left_stroke.d is None or right_stroke.d is None:
            continue

        left_info = analyze_path(translate_path(left_stroke.d, x_origins[index], 0))
        right_info = analyze_path(translate_path(right_stroke.d, next_origin, 0))

        p0 = left_info.start if left.exit[1] == "start" else left_info.end
        t0 = left_info.tangent_start if left.exit[1] == "start" else left_info.tangent_end
        p3 = right_info.start if right.entry[1] == "start" else right_info.end
        t3 = right_info.tangent_start if right.entry[1] == "start" else right_info.tangent_end

        output.append(
            CompiledStroke(
                id=f"join:{pair}:{index}",
                role="connector",
                d=cubic_bridge(p0, t0, p3, t3, tension),
            )
        )

    width = x_origins[-1] + compiled[-1].advance_width
    return CompiledText(
        strokes=tuple(output),
        width=width,
        characters=characters,
    )
