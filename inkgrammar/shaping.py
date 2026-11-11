from __future__ import annotations

from dataclasses import dataclass

from .geometry import analyze_path, path_without_initial_move, transform_path, translate_path
from .models import (
    GlyphContext,
    PathStrokeRecipe,
    RawPathPart,
    RuntimeAnchor,
    RuntimeConnectionRule,
    RuntimeContextRule,
    RuntimeGlyphVariant,
    TapStrokeRecipe,
    WritingModelV3,
)

MAX_CONTEXT_ITERATIONS = 8


class ContextCycleError(ValueError):
    pass


class RuleConflictError(ValueError):
    pass


@dataclass(frozen=True)
class ShapedStroke:
    id: str
    role: str
    d: str | None = None
    tap: tuple[float, float] | None = None


@dataclass(frozen=True)
class ShapedGlyph:
    character: str
    variant_id: str
    x: float
    y: float
    advance: float
    entry: RuntimeAnchor | None
    exit: RuntimeAnchor | None
    strokes: tuple[ShapedStroke, ...]


@dataclass(frozen=True)
class ShapedJoin:
    index: int
    left_variant: str
    right_variant: str
    rule_id: str | None
    mode: str
    dx: float
    dy: float


@dataclass(frozen=True)
class ShapedText:
    glyphs: tuple[ShapedGlyph, ...]
    joins: tuple[ShapedJoin, ...]
    width: float

    @property
    def strokes(self) -> tuple[ShapedStroke, ...]:
        return tuple(stroke for glyph in self.glyphs for stroke in glyph.strokes)


def _position(index: int, count: int) -> str:
    if count == 1:
        return "isolated"
    if index == 0:
        return "initial"
    if index == count - 1:
        return "final"
    return "medial"


def _matches(rule: RuntimeContextRule, context: GlyphContext) -> bool:
    conditions = (
        (rule.prev_characters, context.prev_character),
        (rule.next_characters, context.next_character),
        (rule.prev_exit_classes, context.prev_exit_class),
        (rule.next_entry_classes, context.next_entry_class),
        (rule.positions, context.position),
    )
    return all(not allowed or value in allowed for allowed, value in conditions)


def _specificity(rule: RuntimeContextRule) -> int:
    if rule.prev_characters and rule.next_characters:
        return 4
    if rule.prev_characters or rule.next_characters:
        return 3
    if rule.prev_exit_classes or rule.next_entry_classes:
        return 2
    if rule.positions:
        return 1
    return 0


def _select_variant(
    model: WritingModelV3,
    character: str,
    context: GlyphContext,
    optional_features: set[str] | None,
) -> str:
    glyph = model.glyphs[character]
    matching = [
        rule
        for rule in model.context_rules
        if rule.character == character
        and _matches(rule, context)
        and (rule.required or (optional_features is not None and "calt" in optional_features))
    ]
    if not matching:
        return glyph.default_variant
    matching.sort(key=lambda rule: (-_specificity(rule), -rule.priority, rule.id))
    best = matching[0]
    best_rank = (_specificity(best), best.priority)
    tied = [rule for rule in matching if (_specificity(rule), rule.priority) == best_rank]
    if len(tied) > 1:
        raise RuleConflictError(
            f"context rules tie for {character!r}: {', '.join(sorted(rule.id for rule in tied))}"
        )
    return best.variant


def _variant_context(
    model: WritingModelV3,
    characters: tuple[str, ...],
    variants: tuple[str, ...],
    index: int,
) -> GlyphContext:
    previous = model.glyphs[characters[index - 1]].variants[variants[index - 1]] if index else None
    following = (
        model.glyphs[characters[index + 1]].variants[variants[index + 1]]
        if index + 1 < len(characters)
        else None
    )
    return GlyphContext(
        prev_character=characters[index - 1] if index else None,
        next_character=characters[index + 1] if index + 1 < len(characters) else None,
        prev_exit_class=previous.exit.class_ if previous and previous.exit else None,
        next_entry_class=following.entry.class_ if following and following.entry else None,
        position=_position(index, len(characters)),
    )


def _select_variants(
    model: WritingModelV3,
    characters: tuple[str, ...],
    optional_features: set[str] | None,
) -> tuple[str, ...]:
    variants = tuple(model.glyphs[character].default_variant for character in characters)
    for _ in range(MAX_CONTEXT_ITERATIONS):
        selected = tuple(
            _select_variant(
                model,
                character,
                _variant_context(model, characters, variants, index),
                optional_features,
            )
            for index, character in enumerate(characters)
        )
        if selected == variants:
            return selected
        variants = selected
    raise ContextCycleError(f"context rules did not converge for {''.join(characters)!r}")


def _stroke_paths(stroke: PathStrokeRecipe) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    parts: list[str] = []
    current_end: tuple[float, float] | None = None
    part_index = 0

    def finish() -> None:
        nonlocal parts, part_index
        if parts:
            path = parts[0] + "".join(f" {part}" for part in parts[1:])
            stroke_id = stroke.id if part_index == 0 else f"{stroke.id}.{part_index}"
            paths.append((stroke_id, path))
            parts = []
            part_index += 1

    for part in stroke.parts:
        if not isinstance(part, RawPathPart):
            raise ValueError("runtime schema v3 glyph strokes must contain flattened paths")
        path = transform_path(part.d, part.transform)
        info = analyze_path(path)
        if parts and not part.attach:
            finish()
            current_end = None
        if parts:
            assert current_end is not None
            path = translate_path(
                path,
                current_end[0] - info.start[0],
                current_end[1] - info.start[1],
            )
            parts.append(path_without_initial_move(path))
        else:
            parts.append(path)
        current_end = analyze_path(path).end
    finish()
    return paths


def _place_glyph(
    character: str,
    variant_id: str,
    variant: RuntimeGlyphVariant,
    index: int,
    x: float,
    y: float,
) -> ShapedGlyph:
    strokes: list[ShapedStroke] = []
    for stroke in variant.strokes:
        if isinstance(stroke, TapStrokeRecipe):
            strokes.append(
                ShapedStroke(
                    id=f"{index}:{stroke.id}",
                    role=stroke.role,
                    tap=(stroke.at[0] + x, stroke.at[1] + y),
                )
            )
        else:
            for stroke_id, path in _stroke_paths(stroke):
                strokes.append(
                    ShapedStroke(
                        id=f"{index}:{stroke_id}",
                        role=stroke.role,
                        d=translate_path(path, x, y),
                    )
                )
    entry = (
        variant.entry.model_copy(update={"x": variant.entry.x + x, "y": variant.entry.y + y})
        if variant.entry
        else None
    )
    exit = (
        variant.exit.model_copy(update={"x": variant.exit.x + x, "y": variant.exit.y + y})
        if variant.exit
        else None
    )
    return ShapedGlyph(
        character=character,
        variant_id=variant_id,
        x=x,
        y=y,
        advance=variant.advance_width,
        entry=entry,
        exit=exit,
        strokes=tuple(strokes),
    )


def _connection_rule(
    model: WritingModelV3,
    left: RuntimeGlyphVariant,
    right: RuntimeGlyphVariant,
) -> RuntimeConnectionRule | None:
    if left.exit is None or right.entry is None:
        return None
    matches = [
        rule
        for rule in model.connection_rules
        if rule.from_class == left.exit.class_
        and rule.to_class == right.entry.class_
        and (rule.required_variant_left is None or rule.required_variant_left == left.id)
        and (rule.required_variant_right is None or rule.required_variant_right == right.id)
    ]
    if len(matches) > 1:
        raise RuleConflictError(
            f"connection rules tie for {left.exit.class_!r} to {right.entry.class_!r}"
        )
    return matches[0] if matches else None


def shape_glyph(
    model: WritingModelV3,
    character: str,
    variant_id: str | None = None,
) -> ShapedGlyph:
    glyph = model.glyphs[character]
    selected = variant_id or glyph.default_variant
    if selected not in glyph.variants:
        raise KeyError(f"unknown variant {selected!r} for glyph {character!r}")
    return _place_glyph(character, selected, glyph.variants[selected], 0, 0.0, 0.0)


def shape_text(
    model: WritingModelV3,
    text: str,
    *,
    optional_features: set[str] | None = None,
    allow_synthetic_bridges: bool = False,
) -> ShapedText:
    characters = tuple(character for character in text if character in model.glyphs)
    if not characters:
        return ShapedText(glyphs=(), joins=(), width=0.0)
    variant_ids = _select_variants(model, characters, optional_features)
    variants = tuple(
        model.glyphs[character].variants[variant_id]
        for character, variant_id in zip(characters, variant_ids, strict=True)
    )
    shaped: list[ShapedGlyph] = []
    joins: list[ShapedJoin] = []
    for index, (character, variant_id, variant) in enumerate(
        zip(characters, variant_ids, variants, strict=True)
    ):
        if index == 0:
            x = 0.0
            y = 0.0
        else:
            left = variants[index - 1]
            left_shaped = shaped[-1]
            rule = _connection_rule(model, left, variant)
            spacing = rule.pair_spacing if rule else 0.0
            base_x = left_shaped.x + left_shaped.advance + spacing
            base_y = left_shaped.y
            dx = rule.x_adjust if rule else 0.0
            dy = rule.y_adjust if rule else 0.0
            mode = rule.mode if rule else "none"
            if mode in {"cursive-anchor", "direct"}:
                assert left_shaped.exit is not None and variant.entry is not None
                x = left_shaped.exit.x - variant.entry.x + dx
                y = left_shaped.exit.y - variant.entry.y + dy
                dx = x - base_x
                dy = y - base_y
            elif mode in {"attached", "none", "synthetic-bridge"}:
                if mode == "synthetic-bridge" and not allow_synthetic_bridges:
                    raise ValueError("synthetic bridge requires compatibility rendering mode")
                x = base_x + dx
                y = base_y + dy
            elif mode == "ligature":
                raise ValueError(
                    "ligature connection must be materialized during variant selection"
                )
            else:
                raise ValueError(f"unsupported connection mode: {mode}")
            joins.append(
                ShapedJoin(
                    index=index - 1,
                    left_variant=left.id,
                    right_variant=variant.id,
                    rule_id=rule.id if rule else None,
                    mode=mode,
                    dx=dx,
                    dy=dy,
                )
            )
        shaped.append(_place_glyph(character, variant_id, variant, index, x, y))
    width = shaped[-1].x + shaped[-1].advance
    return ShapedText(glyphs=tuple(shaped), joins=tuple(joins), width=width)
