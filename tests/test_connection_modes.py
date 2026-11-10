import pytest
from pydantic import ValidationError

from scriptstrokes.model import compile_text
from scriptstrokes.models import ConnectionRule, WritingModel
from scriptstrokes.qc import validate_model


def model_for(mode="continuous-bridge", height=0):
    glyph = {
        "character": "a", "unicode": ["U+0061"],
        "provenance": {"status": "inferred"},
        "forms": {"default": {
            "advance_width": 100,
            "strokes": [{"id": "main", "parts": [{"d": f"M 0 0 L 100 {height}"}]}],
            "connections": {
                "entry": {"stroke": "main", "endpoint": "start", "class": "baseline"},
                "exit": {"stroke": "main", "endpoint": "end", "class": "baseline"},
            },
        }},
    }
    return WritingModel.model_validate({
        "schema_version": 2, "id": "test", "name": {"en": "Test"},
        "script": "Latn", "status": "experimental", "license": "test-only",
        "metrics": {"units_per_em": 1000, "x_height": 400, "ascender": 700,
                    "descender": -200, "default_stroke_width": 40},
        "glyphs": {"a": glyph},
        "connection_rules": [{"id": "rule", "from_class": "baseline",
                              "to_class": "baseline", "mode": mode}],
    })


@pytest.mark.parametrize("mode, bridges", [
    ("continuous-bridge", 1), ("direct", 0), ("attached", 0), ("none", 0),
])
def test_modes(mode, bridges):
    compiled = compile_text(model_for(mode), "aa")
    assert sum(stroke.role == "connector" for stroke in compiled.strokes) == bridges
    assert len([stroke for stroke in compiled.strokes if stroke.role == "body"]) == 2


def test_midline_bridge():
    model = model_for(height=400)
    form = model.glyphs["a"].forms["default"]
    form.connections.exit.class_ = "midline"
    model.connection_rules[0].from_class = "midline"
    assert any(stroke.role == "connector" for stroke in compile_text(model, "aa").strokes)


def test_direct_requires_matching_endpoints():
    model = model_for("direct", height=10)
    with pytest.raises(ValueError, match="endpoints do not meet"):
        compile_text(model, "aa")
    assert any(issue.code == "connection.direct-endpoints"
               for issue in validate_model(model).issues)


def test_pair_override_applies_without_attached_bridge():
    model = model_for("attached")
    from scriptstrokes.models import PairOverride
    model.pair_overrides["aa"] = PairOverride(gap_adjust=12)
    compiled = compile_text(model, "aa")
    assert compiled.width == 212
    assert not any(stroke.role == "connector" for stroke in compiled.strokes)


def test_invalid_mode():
    with pytest.raises(ValidationError):
        ConnectionRule(id="bad", from_class="a", to_class="b", mode="invalid")


def test_legacy_continuous_false_is_attached():
    rule = ConnectionRule(id="old", from_class="a", to_class="b", continuous=False)
    assert rule.effective_mode == "attached"
    model = model_for()
    model.connection_rules[0] = rule.model_copy(update={"from_class": "baseline",
                                                       "to_class": "baseline"})
    assert not any(s.role == "connector" for s in compile_text(model, "aa").strokes)


def test_explicit_mode_takes_precedence():
    rule = ConnectionRule(id="rule", from_class="a", to_class="b",
                          continuous=False, mode="continuous-bridge")
    assert rule.effective_mode == "continuous-bridge"


def test_invalid_pair_and_duplicate_rules():
    from scriptstrokes.models import PairOverride
    model = model_for()
    model.pair_overrides["ax"] = PairOverride()
    model.connection_rules.append(model.connection_rules[0])
    codes = {issue.code for issue in validate_model(model).issues}
    assert "connection.invalid-pair" in codes
    assert "connection.duplicate-rule" in codes
