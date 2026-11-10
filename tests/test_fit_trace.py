from scriptstrokes.fitting.trace import compile_glyph_trace
from scriptstrokes.geometry import analyze_path
from scriptstrokes.models import WritingModel


def model_with_attached_parts():
    provenance = {"status": "inferred"}
    return WritingModel.model_validate(
        {
            "schema_version": 2,
            "id": "trace-test",
            "name": {"en": "Trace test"},
            "script": "Latn",
            "status": "experimental",
            "license": "test-only",
            "metrics": {
                "units_per_em": 1000,
                "baseline": 0,
                "x_height": 360,
                "cap_height": 650,
                "ascender": 650,
                "descender": -300,
                "default_stroke_width": 55,
            },
            "components": {
                "first": {"d": "M 0 0 C 2 0 4 0 10 0", "provenance": provenance},
                "second": {"d": "M 0 0 C 1 1 2 1 4 0", "provenance": provenance},
            },
            "glyphs": {
                "g": {
                    "character": "g",
                    "unicode": ["U+0067"],
                    "provenance": provenance,
                    "forms": {
                        "isolated": {
                            "advance_width": 100,
                            "strokes": [
                                {
                                    "id": "main",
                                    "parts": [
                                        {"component": "first"},
                                        {
                                            "component": "second",
                                            "transform": {
                                                "scale_x": 2,
                                                "scale_y": 2,
                                                "dx": 5,
                                                "dy": 3,
                                            },
                                        },
                                    ],
                                }
                            ],
                        }
                    },
                }
            },
        }
    )


def test_trace_tracks_transformed_attached_component_instances():
    model = model_with_attached_parts()
    compiled, traces = compile_glyph_trace(model, "g")

    assert len(traces) == 2
    first, second = traces
    assert compiled.form_name == "isolated"
    assert (first.glyph, first.form, first.stroke_id) == ("g", "isolated", "main")
    assert (first.instance_id, first.primitive_id) == ("main:part-0", "first")
    assert (second.instance_id, second.primitive_id) == ("main:part-1", "second")
    assert first.end == second.start == (10.0, 0.0)
    assert second.end == (18.0, 0.0)
    assert second.bbox[0] == 10.0
    assert second.bbox[2] == 18.0

    compiled_path = analyze_path(compiled.strokes[0].d)
    assert compiled_path.start == first.start
    assert compiled_path.end == second.end
