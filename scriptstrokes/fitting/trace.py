from __future__ import annotations

from dataclasses import dataclass

from ..geometry import analyze_path
from ..model import (
    CompiledGlyph,
    _compiled_part_paths,
    _select_form,
    compile_glyph,
)
from ..models import ComponentPart, PathStrokeRecipe, WritingModel


@dataclass(frozen=True)
class CompiledPartTrace:
    glyph: str
    form: str
    stroke_id: str
    instance_id: str
    primitive_id: str | None
    d: str
    start: tuple[float, float]
    end: tuple[float, float]
    bbox: tuple[float, float, float, float]


def compile_glyph_trace(
    model: WritingModel,
    character: str,
    form_name: str = "isolated",
) -> tuple[CompiledGlyph, tuple[CompiledPartTrace, ...]]:
    """Compile a runtime glyph and retain fitting-only per-part ownership and geometry."""
    compiled = compile_glyph(model, character, form_name)
    _, form = _select_form(model, character, form_name)
    traces: list[CompiledPartTrace] = []

    for stroke in form.strokes:
        if not isinstance(stroke, PathStrokeRecipe):
            continue
        paths = _compiled_part_paths(model, stroke)
        for index, (part, path) in enumerate(zip(stroke.parts, paths, strict=True)):
            info = analyze_path(path)
            traces.append(
                CompiledPartTrace(
                    glyph=character,
                    form=compiled.form_name,
                    stroke_id=stroke.id,
                    instance_id=f"{stroke.id}:part-{index}",
                    primitive_id=part.component if isinstance(part, ComponentPart) else None,
                    d=path,
                    start=info.start,
                    end=info.end,
                    bbox=info.bbox,
                )
            )
    return compiled, tuple(traces)
