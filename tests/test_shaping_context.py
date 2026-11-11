from __future__ import annotations

import pytest

from inkgrammar.model import compile_glyph, compile_text
from inkgrammar.models import (
    ConnectionClass,
    Metrics,
    PathStrokeRecipe,
    RuntimeAnchor,
    RuntimeConnectionRule,
    RuntimeContextRule,
    RuntimeGlyph,
    RuntimeGlyphVariant,
    WritingModelV3,
)
from inkgrammar.shaping import ContextCycleError, RuleConflictError, shape_text


def variant(
    variant_id: str,
    character: str,
    *,
    entry_class: str = "baseline",
    exit_class: str = "baseline",
) -> RuntimeGlyphVariant:
    return RuntimeGlyphVariant(
        id=variant_id,
        character=character,
        advance_width=20,
        strokes=[PathStrokeRecipe(id="body", parts=[{"d": "M 0 0 L 10 0"}])],
        entry=RuntimeAnchor(x=0, y=0, tangent_x=1, tangent_y=0, class_=entry_class),
        exit=RuntimeAnchor(x=10, y=0, tangent_x=1, tangent_y=0, class_=exit_class),
    )


def model(*, context_rules=(), connection_rules=(), variants=None) -> WritingModelV3:
    variants = variants or {
        "a": {"a.default": variant("a.default", "a")},
        "b": {"b.default": variant("b.default", "b")},
    }
    glyphs = {
        character: RuntimeGlyph(
            character=character,
            default_variant=next(iter(character_variants)),
            variants=character_variants,
        )
        for character, character_variants in variants.items()
    }
    return WritingModelV3(
        schema_version=3,
        id="toy-script",
        name={"en": "Toy Script"},
        script="Latn",
        status="experimental",
        license="CC0-1.0",
        metrics=Metrics(
            units_per_em=1000,
            x_height=400,
            ascender=700,
            descender=-200,
            default_stroke_width=50,
        ),
        glyphs=glyphs,
        connection_classes={
            name: ConnectionClass(id=name, nominal_height="baseline", pen_state="continuous")
            for name in {"baseline", "high"}
        },
        connection_rules=list(connection_rules),
        context_rules=list(context_rules),
    )


def test_context_rules_select_variants_and_anchor_connections():
    rules = [
        RuntimeContextRule(
            id="a-before-b",
            character="a",
            variant="a.initial",
            next_characters=["b"],
        ),
        RuntimeContextRule(
            id="b-after-a",
            character="b",
            variant="b.final",
            prev_characters=["a"],
        ),
    ]
    connections = [
        RuntimeConnectionRule(
            id="baseline-cursive",
            from_class="baseline",
            to_class="baseline",
            mode="cursive-anchor",
        )
    ]
    grammar = model(
        context_rules=rules,
        connection_rules=connections,
        variants={
            "a": {
                "a.default": variant("a.default", "a"),
                "a.initial": variant("a.initial", "a"),
            },
            "b": {
                "b.default": variant("b.default", "b"),
                "b.final": variant("b.final", "b"),
            },
        },
    )

    shaped = shape_text(grammar, "ab")

    assert [glyph.variant_id for glyph in shaped.glyphs] == ["a.initial", "b.final"]
    assert shaped.joins[0].mode == "cursive-anchor"
    assert shaped.glyphs[0].exit.x == pytest.approx(shaped.glyphs[1].entry.x)
    assert shaped.glyphs[1].x == pytest.approx(10)
    assert [stroke.id for stroke in compile_text(grammar, "ab").strokes] == ["0:body", "1:body"]
    assert compile_glyph(grammar, "a", "a.initial").form_name == "a.initial"


def test_equal_specificity_and_priority_context_rules_are_rejected():
    grammar = model(
        context_rules=[
            RuntimeContextRule(
                id="rule-one",
                character="b",
                variant="b.default",
                prev_characters=["a"],
                priority=2,
            ),
            RuntimeContextRule(
                id="rule-two",
                character="b",
                variant="b.default",
                prev_characters=["a"],
                priority=2,
            ),
        ]
    )

    with pytest.raises(RuleConflictError, match="context rules tie"):
        shape_text(grammar, "ab")


def test_context_specificity_and_priority_precedence():
    variants = {
        "a": {"a.default": variant("a.default", "a")},
        "b": {
            "b.default": variant("b.default", "b"),
            "b.by-character": variant("b.by-character", "b"),
            "b.by-class": variant("b.by-class", "b"),
            "b.by-position": variant("b.by-position", "b"),
        },
    }
    rules = [
        RuntimeContextRule(
            id="by-character",
            character="b",
            variant="b.by-character",
            prev_characters=["a"],
            priority=0,
        ),
        RuntimeContextRule(
            id="by-class",
            character="b",
            variant="b.by-class",
            prev_exit_classes=["baseline"],
            priority=100,
        ),
        RuntimeContextRule(
            id="by-position",
            character="b",
            variant="b.by-position",
            positions=["final"],
            priority=200,
        ),
    ]
    grammar = model(variants=variants, context_rules=rules)

    assert shape_text(grammar, "a").glyphs[0].variant_id == "a.default"
    assert shape_text(grammar, "ab").glyphs[1].variant_id == "b.by-character"

    priority_rules = [
        RuntimeContextRule(
            id="lower-priority",
            character="b",
            variant="b.by-character",
            prev_characters=["a"],
            priority=1,
        ),
        RuntimeContextRule(
            id="higher-priority",
            character="b",
            variant="b.by-class",
            prev_characters=["a"],
            priority=9,
        ),
    ]
    prioritized = model(variants=variants, context_rules=priority_rules)
    assert shape_text(prioritized, "ab").glyphs[1].variant_id == "b.by-class"


def test_context_fixed_point_cycle_is_reported():
    low = variant("a.default", "a", entry_class="baseline", exit_class="baseline")
    high = variant("a.high", "a", entry_class="high", exit_class="high")
    grammar = model(
        variants={"a": {"a.default": low, "a.high": high}},
        context_rules=[
            RuntimeContextRule(
                id="initial-from-next-low",
                character="a",
                variant="a.high",
                next_entry_classes=["baseline"],
                positions=["initial"],
            ),
            RuntimeContextRule(
                id="initial-from-next-high",
                character="a",
                variant="a.default",
                next_entry_classes=["high"],
                positions=["initial"],
            ),
            RuntimeContextRule(
                id="final-from-prev-low",
                character="a",
                variant="a.high",
                prev_exit_classes=["baseline"],
                positions=["final"],
            ),
            RuntimeContextRule(
                id="final-from-prev-high",
                character="a",
                variant="a.default",
                prev_exit_classes=["high"],
                positions=["final"],
            ),
        ],
    )

    with pytest.raises(ContextCycleError, match="did not converge"):
        shape_text(grammar, "aa")


def test_synthetic_bridge_requires_compatibility_mode():
    grammar = model(
        connection_rules=[
            RuntimeConnectionRule(
                id="bridge",
                from_class="baseline",
                to_class="baseline",
                mode="synthetic-bridge",
            )
        ]
    )

    with pytest.raises(ValueError, match="compatibility rendering mode"):
        shape_text(grammar, "ab")
    assert (
        shape_text(grammar, "ab", allow_synthetic_bridges=True).joins[0].mode == "synthetic-bridge"
    )
