from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from typer.testing import CliRunner

from inkgrammar.cli import app
from inkgrammar.io import load_model
from inkgrammar.models import (
    ConnectionClass,
    Metrics,
    PathStrokeRecipe,
    RuntimeAnchor,
    RuntimeGlyph,
    RuntimeGlyphVariant,
    WritingModelV3,
)
from inkgrammar.raster import render_text_frame
from inkgrammar.svg import render_text_svg


def model() -> WritingModelV3:
    variant = RuntimeGlyphVariant(
        id="a.default",
        character="a",
        advance_width=120,
        strokes=[PathStrokeRecipe(id="body", parts=[{"d": "M 0 0 C 30 80 70 80 100 0"}])],
        entry=RuntimeAnchor(x=0, y=0, tangent_x=1, tangent_y=0, class_="baseline"),
        exit=RuntimeAnchor(x=100, y=0, tangent_x=1, tangent_y=0, class_="baseline"),
    )
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
            default_stroke_width=40,
        ),
        glyphs={
            "a": RuntimeGlyph(
                character="a",
                unicode=["U+0061"],
                default_variant="a.default",
                variants={"a.default": variant},
            )
        },
        connection_classes={
            "baseline": ConnectionClass(
                id="baseline", nominal_height="baseline", pen_state="continuous"
            )
        },
    )


def test_svg_is_deterministic_and_includes_requested_diagnostics():
    grammar = model()

    first = render_text_svg(grammar, "a")
    second = render_text_svg(grammar, "a")

    assert first == second
    root = ET.fromstring(first)
    assert root.tag.endswith("svg")
    assert 'data-stroke="0:body"' in first
    assert 'data-anchor="entry"' in first
    assert 'data-tangent="exit"' in first
    assert 'data-variant="a.default"' in first
    assert 'data-guide="baseline"' in first

    outlined = render_text_svg(grammar, "a", show_outline=True)
    ET.fromstring(outlined)
    assert 'data-outline="0"' in outlined

    without_debug = render_text_svg(
        grammar,
        "a",
        show_anchors=False,
        show_tangents=False,
        show_origins=False,
        show_guides=False,
        show_labels=False,
    )
    assert "data-anchor" not in without_debug
    assert "data-guide" not in without_debug
    assert "data-variant" not in without_debug
    assert "data-stroke" in without_debug


def test_v3_raster_renderer_consumes_shaped_output():
    frame = render_text_frame(model(), "a", scale=0.5)

    assert frame.image.width > 0
    assert frame.image.height > 0


def test_model_render_svg_cli_writes_svg(tmp_path):
    model_path = tmp_path / "toy.yaml"
    output_path = tmp_path / "render.svg"
    model_path.write_text(
        yaml.safe_dump(model().model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    result = CliRunner().invoke(
        app,
        [
            "model",
            "render-svg",
            str(model_path),
            "a",
            "--output",
            str(output_path),
            "--no-guides",
            "--outline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.read_text(encoding="utf-8").startswith(
        '<svg xmlns="http://www.w3.org/2000/svg"'
    )
    assert 'data-outline="0"' in output_path.read_text(encoding="utf-8")


def test_v2_svg_keeps_legacy_compiled_paths():
    root = Path(__file__).resolve().parents[1]
    grammar = load_model(root / "data" / "models" / "demo-looped.yaml")

    content = render_text_svg(grammar, "oi")

    ET.fromstring(content)
    assert 'data-stroke="0:' in content
    assert 'data-stroke="join:' in content
