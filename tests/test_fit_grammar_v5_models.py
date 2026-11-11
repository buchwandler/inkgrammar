from __future__ import annotations

import pytest

from inkgrammar.fitting.models import (
    ConnectionObservation,
    FitCompileConfig,
    GlyphRecipe,
    GlyphStroke,
    GlyphVariantRecipe,
    GrammarConnectionRule,
    PrimitiveOccurrence,
)


def glyph_recipe() -> GlyphRecipe:
    variant = GlyphVariantRecipe(
        id="a.default",
        character="a",
        advance_width=500,
        strokes=[
            GlyphStroke(
                id="body",
                parts=[PrimitiveOccurrence(instance="a-body", primitive="a-loop")],
            )
        ],
    )
    return GlyphRecipe(
        character="a",
        default_variant="a.default",
        variants={"a.default": variant},
    )


def test_fit_grammar_models_contextual_variants_and_v3_compile_config():
    recipe = glyph_recipe()

    assert recipe.default_variant == "a.default"
    config = FitCompileConfig(output="model.yaml", runtime_schema_version=3)
    assert config.runtime_schema_version == 3
    with pytest.raises(ValueError, match="default variant is missing"):
        GlyphRecipe(character="a", default_variant="a.missing", variants=recipe.variants)


def test_connection_rule_requires_explicit_ligature_variant():
    with pytest.raises(ValueError, match="requires ligature_variant"):
        GrammarConnectionRule(
            id="exception",
            from_class="baseline",
            to_class="upper",
            mode="ligature",
        )


def test_connection_observation_checks_boundary_and_split():
    observation = ConnectionObservation(
        id="ab-1",
        text="abc",
        boundary_index=1,
        observation_id="scan-1",
        split="hold-out",
        from_character="a",
        to_character="b",
    )

    assert observation.split == "hold-out"
    with pytest.raises(ValueError, match="does not match boundary"):
        ConnectionObservation(
            id="ab-bad",
            text="abc",
            boundary_index=1,
            observation_id="scan-1",
            split="fit",
            from_character="b",
            to_character="c",
        )
