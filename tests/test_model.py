from pathlib import Path

from scriptstrokes.io import load_model
from scriptstrokes.model import compile_glyph, compile_text
from scriptstrokes.qc import validate_model

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "models" / "demo-looped.yaml"


def test_model_is_clean():
    model = load_model(MODEL)
    report = validate_model(model)
    assert report.errors == 0
    assert report.warnings == 0


def test_components_are_reused():
    model = load_model(MODEL)
    glyph_u = compile_glyph(model, "u")
    glyph_m = compile_glyph(model, "m")
    assert glyph_u.strokes[0].d is not None
    assert glyph_m.strokes[0].d is not None


def test_context_form_and_join():
    model = load_model(MODEL)
    plan = compile_text(model, "oi")
    assert plan.characters == ("o", "i")
    assert any(stroke.role == "connector" for stroke in plan.strokes)
