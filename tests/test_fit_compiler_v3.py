from __future__ import annotations

from types import SimpleNamespace

import pytest

import inkgrammar.compiler as compiler
from inkgrammar.fitting.models import (
    BSplineGeometry,
    ConnectionClass,
    CubicSegment,
    DuctusModel,
    FitCompileConfig,
    GlyphEndpoint,
    GlyphRecipe,
    GlyphStroke,
    GlyphVariantRecipe,
    GrammarConnectionRule,
    PrimitiveFrame,
    PrimitiveInstance,
    PrimitiveOccurrence,
    PrimitiveTemplate,
)
from inkgrammar.io import load_model


def bundle_for(tmp_path):
    primitive = PrimitiveTemplate(
        id="line",
        kind="cubic-chain",
        frame=PrimitiveFrame(start=(0, 0), nominal_end=(100, 0)),
        segments=[
            CubicSegment(c1=(30, 0), c2=(30, 100), end=(50, 100)),
            CubicSegment(c1=(70, 100), c2=(70, 0), end=(100, 0)),
        ],
        free_parameters=[],
        locked_parameters=[],
        bounds={},
    )
    instance = PrimitiveInstance(id="a-body", primitive="line")
    variant = GlyphVariantRecipe(
        id="a.default",
        character="a",
        advance_width=500,
        strokes=[
            GlyphStroke(
                id="body",
                parts=[PrimitiveOccurrence(instance="a-body", primitive="line")],
            )
        ],
        entry=GlyphEndpoint(instance="a-body", endpoint="start", class_="baseline"),
        exit=GlyphEndpoint(instance="a-body", endpoint="end", class_="baseline"),
    )
    grammar = SimpleNamespace(
        ductus=DuctusModel(
            units_per_em=1000,
            baseline=0,
            x_height=400,
            cap_height=700,
            ascender=700,
            descender=-200,
            slant_degrees=0,
            stroke_width=50,
        ),
        primitives={"line": primitive},
        instances={"a-body": instance},
        glyphs={
            "a": GlyphRecipe(
                character="a",
                default_variant="a.default",
                variants={"a.default": variant},
            )
        },
        connection_classes={
            "baseline": ConnectionClass(
                id="baseline", nominal_height="baseline", pen_state="continuous"
            )
        },
        connection_rules=[],
        context_rules=[],
    )
    return SimpleNamespace(
        project=SimpleNamespace(
            id="toy-project",
            runtime_model_id="toy-model",
            compile=FitCompileConfig(
                output="compiled.yaml", runtime_schema_version=3, fit_x_height_units=400
            ),
        ),
        project_dir=tmp_path,
        grammar=grammar,
        topology=SimpleNamespace(glyphs={}),
    )


def test_compiler_requires_promoted_primitive_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _bundle: [])

    with pytest.raises(ValueError, match="missing fit report: line"):
        compiler.compile_fit_project(bundle_for(tmp_path))


def test_compiler_derives_anchors_and_writes_deterministically(monkeypatch, tmp_path):
    bundle = bundle_for(tmp_path)
    report = {
        "primitive_id": "line",
        "primitive": bundle.grammar.primitives["line"].model_dump(mode="json"),
        "visual_review": {"status": "promoted"},
    }
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _bundle: [report])

    result = compiler.compile_fit_project(bundle)
    output = tmp_path / "compiled.yaml"
    first_bytes = output.read_bytes()
    loaded = load_model(output)
    variant = loaded.glyphs["a"].variants["a.default"]

    assert variant.entry.x == pytest.approx(0)
    assert variant.entry.y == pytest.approx(0)
    assert variant.entry.tangent_x == pytest.approx(1)
    assert variant.exit.x == pytest.approx(100)
    assert variant.exit.tangent_x == pytest.approx(1)
    assert variant.entry.width == pytest.approx(50)
    assert result.diagnostics["maximum_centerline_deviation"] == 0

    compiler.compile_fit_project(bundle)
    assert output.read_bytes() == first_bytes


def test_compiler_flattens_bspline_primitives_to_cubic_paths(monkeypatch, tmp_path):
    bundle = bundle_for(tmp_path)
    primitive = PrimitiveTemplate(
        id="line",
        kind="bspline-centerline",
        frame=PrimitiveFrame(start=(0, 0), nominal_end=(100, 0)),
        spline=BSplineGeometry(
            knots=[0, 0, 0, 0, 1, 1, 1, 1],
            control_points=[(0, 0), (33, 100), (67, 100), (100, 0)],
        ),
        free_parameters=[],
        locked_parameters=[],
        bounds={},
    )
    bundle.grammar.primitives["line"] = primitive
    report = {
        "primitive_id": "line",
        "primitive": primitive.model_dump(mode="json"),
        "visual_review": {"status": "promoted"},
    }
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _bundle: [report])

    result = compiler.compile_fit_project(bundle)
    variant = result.model.glyphs["a"].variants["a.default"]
    path = variant.strokes[0].parts[0].d

    assert "C " in path
    assert result.diagnostics["maximum_centerline_deviation"] == 0


def test_compiler_supports_explicit_schema_v2_output(monkeypatch, tmp_path):
    bundle = bundle_for(tmp_path)
    bundle.project.compile = FitCompileConfig(
        output="compiled-v2.yaml", runtime_schema_version=2, fit_x_height_units=400
    )
    report = {
        "primitive_id": "line",
        "primitive": bundle.grammar.primitives["line"].model_dump(mode="json"),
        "visual_review": {"status": "promoted"},
    }
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _bundle: [report])

    result = compiler.compile_fit_project(bundle)
    loaded = load_model(tmp_path / "compiled-v2.yaml")

    assert result.model.schema_version == 2
    assert loaded.schema_version == 2
    assert loaded.glyphs["a"].forms["default"].advance_width == 500


def test_compiler_converts_fit_units_to_runtime_and_scales_advances(monkeypatch, tmp_path):
    bundle = bundle_for(tmp_path)
    bundle.project.compile = FitCompileConfig(
        output="compiled-scaled.yaml", runtime_schema_version=3, fit_x_height_units=200
    )
    primitive = bundle.grammar.primitives["line"]
    report = {
        "primitive_id": "line",
        "primitive": primitive.model_dump(mode="json"),
        "visual_review": {"status": "promoted"},
    }
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _bundle: [report])
    bundle.grammar.connection_rules = [
        GrammarConnectionRule(
            id="baseline-direct",
            from_class="baseline",
            to_class="baseline",
            mode="direct",
            x_adjust=2,
            y_adjust=-3,
            pair_spacing=4,
        )
    ]

    result = compiler.compile_fit_project(bundle)
    variant = result.model.glyphs["a"].variants["a.default"]

    assert variant.advance_width == pytest.approx(1000)
    assert variant.exit.x == pytest.approx(200)
    assert variant.entry.width == pytest.approx(50)
    rule = result.model.connection_rules[0]
    assert (rule.x_adjust, rule.y_adjust, rule.pair_spacing) == pytest.approx((4, -6, 8))
    assert (
        result.model.provenance["coordinate_conversion"]
        == result.diagnostics["coordinate_conversion"]
    )
    assert result.diagnostics["coordinate_conversion"]["runtime_units_per_fit_unit"] == 2
    assert result.diagnostics["scale_sanity"]["a"]["a.default"]["ink_bbox_height"] == pytest.approx(
        200
    )


def test_compiler_rejects_pathological_runtime_scale(monkeypatch, tmp_path):
    bundle = bundle_for(tmp_path)
    bundle.project.compile = FitCompileConfig(
        output="compiled-small.yaml", runtime_schema_version=3, fit_x_height_units=2000
    )
    primitive = bundle.grammar.primitives["line"]
    report = {
        "primitive_id": "line",
        "primitive": primitive.model_dump(mode="json"),
        "visual_review": {"status": "promoted"},
    }
    monkeypatch.setattr(compiler, "load_fit_reports", lambda _bundle: [report])

    with pytest.raises(ValueError, match="runtime scale sanity failed for glyph 'a'"):
        compiler.compile_fit_project(bundle)
