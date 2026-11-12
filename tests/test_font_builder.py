from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fontTools.ttLib import TTFont
from typer.testing import CliRunner

from inkgrammar.cli import app
from inkgrammar.font import (
    FontBuildConfig,
    FontReadinessError,
    build_opentype_font,
    create_font_plan,
    cubic_to_quadratic,
    validate_opentype_font,
)
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


def variant(
    variant_id: str,
    character: str,
    *,
    advance: float = 450,
) -> RuntimeGlyphVariant:
    return RuntimeGlyphVariant(
        id=variant_id,
        character=character,
        advance_width=advance,
        strokes=[PathStrokeRecipe(id="body", parts=[{"d": "M 0 0 C 80 0 160 80 240 80"}])],
        entry=RuntimeAnchor(x=0, y=0, tangent_x=1, tangent_y=0, class_="baseline"),
        exit=RuntimeAnchor(x=240, y=80, tangent_x=1, tangent_y=0, class_="baseline"),
    )


def model(
    license: str = "CC0-1.0", *, provenance: dict[str, object] | None = None
) -> WritingModelV3:
    return WritingModelV3(
        schema_version=3,
        id="toy-script",
        name={"en": "Toy Script"},
        script="Latn",
        status="experimental",
        license=license,
        provenance=provenance or {},
        metrics=Metrics(
            units_per_em=1000,
            x_height=400,
            ascender=700,
            descender=-200,
            default_stroke_width=50,
        ),
        connection_classes={
            "baseline": ConnectionClass(
                id="baseline", nominal_height="baseline", pen_state="continuous"
            )
        },
        glyphs={
            "a": RuntimeGlyph(
                character="a",
                unicode=["U+0061"],
                default_variant="a.default",
                variants={
                    "a.default": variant("a.default", "a"),
                    "a.join": variant("a.join", "a"),
                },
            ),
            "b": RuntimeGlyph(
                character="b",
                unicode=["U+0062"],
                default_variant="b.default",
                variants={"b.default": variant("b.default", "b")},
            ),
        },
        context_rules=[
            RuntimeContextRule(
                id="a-before-b",
                character="a",
                variant="a.join",
                next_characters=["b"],
            )
        ],
        connection_rules=[
            RuntimeConnectionRule(
                id="baseline-cursive",
                from_class="baseline",
                to_class="baseline",
                mode="cursive-anchor",
            )
        ],
    )


def test_font_plan_names_cmap_metrics_and_features_are_stable():
    plan = create_font_plan(model())

    assert plan.glyph_order == (".notdef", "space", "a", "a.join", "b")
    assert plan.cmap == {0x61: "a", 0x62: "b", 0x20: "space"}
    assert plan.horizontal_metrics["a"] == (450, -25)
    assert "feature rclt" in plan.feature_source
    assert "sub a' b by a.join;" in plan.feature_source
    assert "feature curs" in plan.feature_source
    assert "lookup IG_Cursive;" in plan.feature_source
    assert not plan.blockers


def test_font_plan_blocks_pair_specific_cursive_rules_without_static_glyph_classes():
    grammar = model().model_copy(
        update={
            "connection_rules": [
                RuntimeConnectionRule(
                    id="a-join-only",
                    from_class="baseline",
                    to_class="baseline",
                    mode="cursive-anchor",
                    required_variant_left="a.join",
                )
            ]
        }
    )

    plan = create_font_plan(grammar)

    assert any(
        "cannot represent pair-specific connection rules" in blocker for blocker in plan.blockers
    )
    assert "feature curs" not in plan.feature_source


def test_font_builder_writes_reproducible_ttf_and_structural_tables(tmp_path: Path):
    grammar = model()
    config = FontBuildConfig(quadratic_error=0.5)
    first = build_opentype_font(grammar, tmp_path / "one.ttf", config)
    second = build_opentype_font(grammar, tmp_path / "two.ttf", config)

    assert first.path.read_bytes() == second.path.read_bytes()
    first_features = first.feature_source_path.read_text(encoding="utf-8")
    second_features = second.feature_source_path.read_text(encoding="utf-8")
    assert first_features == second_features
    font = TTFont(first.path)
    assert font.getGlyphOrder() == [".notdef", "space", "a", "a.join", "b"]
    assert font.getBestCmap()[0x61] == "a"
    gsub_features = {record.FeatureTag for record in font["GSUB"].table.FeatureList.FeatureRecord}
    gpos_features = {record.FeatureTag for record in font["GPOS"].table.FeatureList.FeatureRecord}
    assert gsub_features >= {"rclt"}
    assert gpos_features >= {"curs"}
    assert font["head"].unitsPerEm == 1000
    assert first.font_ready is False
    coordinates, _, flags = font["glyf"]["a"].getCoordinates(font["glyf"])
    assert coordinates
    assert any(not flag & 1 for flag in flags)
    assert "OpenType shaping parity has not been verified" in first.blockers
    assert validate_opentype_font(first.path, model=grammar).valid


def test_font_export_blocks_unresolved_rights_and_seed_fallback_geometry(tmp_path: Path):
    output = tmp_path / "blocked.ttf"

    with pytest.raises(FontReadinessError, match="rights review is unresolved"):
        build_opentype_font(model("NOASSERTION"), output)

    assert not output.exists()

    fallback_model = model(provenance={"primitive_promotion": {"seed-glyph": "seed-fallback"}})
    with pytest.raises(FontReadinessError, match="unpromoted primitive geometry"):
        build_opentype_font(fallback_model, output)


def test_cubic_to_quadratic_conversion_preserves_endpoints_and_validates_tolerance():
    curve = ((0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0))

    converted = cubic_to_quadratic(curve, 0.5)

    assert converted[0] == curve[0]
    assert converted[-1] == curve[-1]
    assert (len(converted) - 1) % 2 == 0
    with pytest.raises(ValueError, match="max_error"):
        cubic_to_quadratic(curve, 0)


def test_font_cli_plan_build_validate_and_runtime_comparison(tmp_path: Path):
    model_path = tmp_path / "toy.yaml"
    model_path.write_text(
        yaml.safe_dump(model().model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    runner = CliRunner()
    plan_path = tmp_path / "plan.json"
    plan_result = runner.invoke(app, ["font", "plan", str(model_path), "--output", str(plan_path)])
    assert plan_result.exit_code == 0, plan_result.output
    assert json.loads(plan_path.read_text(encoding="utf-8"))["glyph_order"][0] == ".notdef"

    font_path = tmp_path / "toy.ttf"
    build_result = runner.invoke(
        app, ["font", "build", str(model_path), "--output", str(font_path)]
    )
    assert build_result.exit_code == 0, build_result.output
    assert font_path.exists()

    validate_result = runner.invoke(
        app,
        ["font", "validate", str(font_path), "--model", str(model_path)],
    )
    assert validate_result.exit_code == 0, validate_result.output
    assert json.loads(validate_result.output)["valid"] is True

    coverage_result = runner.invoke(
        app,
        ["model", "coverage", str(model_path), "--font", str(font_path)],
    )
    assert coverage_result.exit_code == 0, coverage_result.output
    coverage_report = json.loads(coverage_result.output)
    assert coverage_report["font_artifact"]["valid"] is True
    assert coverage_report["font_ready"] is False

    comparison = runner.invoke(
        app, ["font", "compare-runtime", str(model_path), str(font_path), "ab"]
    )
    assert comparison.exit_code == 0, comparison.output
    comparison_report = json.loads(comparison.output)
    if comparison_report["available"]:
        assert comparison_report["sequence_matches"] is True
        assert comparison_report["positions_match"] is True
    else:
        assert comparison_report["sequence_matches"] is None
        assert comparison_report["positions_match"] is None


def test_build_fit_cli_keeps_promotion_gate_enabled_by_default(tmp_path: Path):
    project = Path(__file__).parents[1] / "data/training/sas-1968/project.yaml"
    output = tmp_path / "sas.ttf"

    result = CliRunner().invoke(
        app,
        ["font", "build-fit", str(project), "--output", str(output)],
    )

    assert result.exit_code != 0
    assert not output.exists()
