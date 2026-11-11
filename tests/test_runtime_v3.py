from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inkgrammar.io import load_model
from inkgrammar.models import WritingModelV3


def model_payload() -> dict:
    variant = {
        "id": "a.default",
        "character": "a",
        "advance_width": 500,
        "strokes": [{"id": "body", "parts": [{"d": "M 0 0 L 100 0"}]}],
        "entry": {"x": 0, "y": 0, "tangent_x": 1, "tangent_y": 0, "class": "baseline"},
        "exit": {"x": 100, "y": 0, "tangent_x": 1, "tangent_y": 0, "class": "baseline"},
    }
    return {
        "schema_version": 3,
        "id": "toy-script",
        "name": {"en": "Toy Script"},
        "script": "Latn",
        "status": "experimental",
        "license": "CC0-1.0",
        "metrics": {
            "units_per_em": 1000,
            "x_height": 400,
            "ascender": 700,
            "descender": -200,
            "default_stroke_width": 50,
        },
        "connection_classes": {
            "baseline": {
                "id": "baseline",
                "nominal_height": "baseline",
                "pen_state": "continuous",
            }
        },
        "glyphs": {
            "a": {
                "character": "a",
                "unicode": ["U+0061"],
                "default_variant": "a.default",
                "variants": {"a.default": variant},
            }
        },
    }


def test_v3_runtime_model_validates_variant_and_context_references():
    payload = model_payload()
    payload["context_rules"] = [
        {"id": "a-after-b", "character": "a", "variant": "a.default", "prev_characters": ["b"]}
    ]

    model = WritingModelV3.model_validate(payload)

    assert model.schema_version == 3
    assert model.glyphs["a"].variants["a.default"].exit.class_ == "baseline"

    payload["context_rules"][0]["variant"] = "a.missing"
    with pytest.raises(ValueError, match="unknown variant"):
        WritingModelV3.model_validate(payload)


def test_load_model_dispatches_v2_and_v3_without_rewriting(tmp_path):
    source = model_payload()
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    loaded = load_model(path)

    assert isinstance(loaded, WritingModelV3)
    assert loaded.model_dump(mode="json") == WritingModelV3.model_validate(source).model_dump(
        mode="json"
    )
    v2_path = tmp_path / "v2.yaml"
    v2_path.write_text(
        (Path(__file__).parents[1] / "data/models/demo-looped.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    v2 = load_model(v2_path)
    assert v2.schema_version == 2
    assert hasattr(v2, "components")


def test_load_model_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "unsupported.yaml"
    path.write_text("schema_version: 9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported writing model schema version"):
        load_model(path)
