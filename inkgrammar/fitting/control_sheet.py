from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geometry import analyze_path, translate_path
from .family_fit import load_fit_reports, primitive_path
from .models import (
    FitSample,
    OccurrenceDelta,
    PrimitiveTemplate,
    TopologyGlyph,
    compose_occurrence_transform,
)
from .render import render_paths
from .workspace import FitProjectBundle

DEFAULT_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzß"
DIGITS = "0123456789"
CELL_WIDTH = 112
CELL_HEIGHT = 140
GRID_COLUMNS = 8
PIXELS_PER_UNIT = 1.0
BASELINE_Y = 92
ORIGIN_X = 20

Status = Literal["modeled", "provisional", "missing", "fit-failed"]


def _fit_templates(
    bundle: FitProjectBundle,
) -> tuple[dict[str, PrimitiveTemplate], dict[str, dict]]:
    templates = dict(bundle.grammar.primitives)
    reports = {report["primitive_id"]: report for report in load_fit_reports(bundle)}
    for primitive_id, report in reports.items():
        candidate = report.get("primitive")
        status = report.get("visual_review", {}).get("status")
        if status == "promoted" and isinstance(candidate, dict):
            templates[primitive_id] = PrimitiveTemplate.model_validate(candidate)
    return templates, reports


def _glyph_paths(
    glyph: TopologyGlyph,
    templates: dict[str, PrimitiveTemplate],
    deltas: dict[str, OccurrenceDelta] | None = None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    deltas = deltas or {}
    for stroke in glyph.strokes:
        current_end = None
        for index, part in enumerate(stroke.parts):
            template = templates[part.primitive]
            transform = compose_occurrence_transform(
                part.transform, deltas.get(part.instance, OccurrenceDelta())
            )
            path = primitive_path(template, transform)
            if index and part.attach:
                if current_end is None:
                    raise ValueError(f"attached part has no preceding endpoint: {part.instance}")
                start = analyze_path(path).start
                path = translate_path(path, current_end[0] - start[0], current_end[1] - start[1])
            paths[f"{stroke.id}:{part.instance}"] = path
            current_end = analyze_path(path).end
    return paths


def _status(
    glyph: TopologyGlyph,
    templates: dict[str, PrimitiveTemplate],
    reports: dict[str, dict],
) -> Status:
    parts = [part for stroke in glyph.strokes for part in stroke.parts]
    if any(part.primitive not in templates for part in parts):
        return "fit-failed"
    if any(
        reports.get(part.primitive, {}).get("optimizer", {}).get("success") is False
        for part in parts
    ):
        return "fit-failed"
    return "modeled" if glyph.review_status == "reviewed" else "provisional"


def _sample_deltas(
    sample: FitSample,
    glyph: TopologyGlyph,
    reports: dict[str, dict],
) -> dict[str, OccurrenceDelta]:
    deltas = {}
    for stroke in glyph.strokes:
        for part in stroke.parts:
            occurrence = (
                reports.get(part.primitive, {})
                .get("occurrences", {})
                .get(f"{sample.id}--{part.instance}")
            )
            if occurrence and isinstance(occurrence.get("delta"), dict):
                deltas[part.instance] = OccurrenceDelta.model_validate(occurrence["delta"])
    return deltas


def _coverage(paths: dict[str, str], bundle: FitProjectBundle) -> np.ndarray:
    return render_paths(
        paths,
        canvas_size=(CELL_WIDTH, CELL_HEIGHT),
        origin=(ORIGIN_X, BASELINE_Y),
        pixels_per_unit=PIXELS_PER_UNIT,
        stroke_width=bundle.fit_config.render.stroke_width_px,
        supersampling=bundle.fit_config.render.supersampling,
    ).coverage


def _paste_coverage(panel: Image.Image, coverage: np.ndarray, color: tuple[int, int, int]) -> None:
    mask = Image.fromarray(np.clip(coverage * 255, 0, 255).astype(np.uint8), mode="L")
    panel.paste(color, (0, 0), mask)


def write_control_sheet(
    bundle: FitProjectBundle,
    output_path: str | Path,
    *,
    include_digits: bool = False,
    occurrence_deltas: bool = False,
) -> dict:
    characters = DEFAULT_CHARACTERS + (DIGITS if include_digits else "")
    templates, reports = _fit_templates(bundle)
    font = ImageFont.load_default()
    cell_count = len(characters)
    rows = (cell_count + GRID_COLUMNS - 1) // GRID_COLUMNS
    sheet = Image.new(
        "RGB",
        (GRID_COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT + 28),
        "white",
    )
    glyph_rows = []

    for index, character in enumerate(characters):
        column, row = index % GRID_COLUMNS, index // GRID_COLUMNS
        left, top = column * CELL_WIDTH, row * CELL_HEIGHT
        panel = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.line((0, BASELINE_Y, CELL_WIDTH - 1, BASELINE_Y), fill=(190, 190, 190), width=1)
        panel_draw.text((4, 3), character, font=font, fill=(20, 20, 20))

        glyph = bundle.topology.glyphs.get(character)
        status: Status = "missing" if glyph is None else _status(glyph, templates, reports)
        rendered = False
        if glyph is not None:
            try:
                if occurrence_deltas:
                    samples = [
                        sample
                        for sample in bundle.corpus.samples
                        if sample.text == character and sample.split == "fit"
                    ]
                    for sample in samples:
                        deltas = _sample_deltas(sample, glyph, reports)
                        if deltas:
                            _paste_coverage(
                                panel,
                                _coverage(_glyph_paths(glyph, templates, deltas), bundle),
                                (130, 185, 210),
                            )
                _paste_coverage(
                    panel,
                    _coverage(_glyph_paths(glyph, templates), bundle),
                    (20, 20, 20),
                )
                rendered = True
            except KeyError:
                status = "fit-failed"
        if status == "missing":
            panel_draw.text((4, 51), "MISSING", font=font, fill=(120, 120, 120))
            panel_draw.text((4, 64), "not modeled", font=font, fill=(120, 120, 120))
        elif status == "fit-failed":
            panel_draw.text((4, 64), "fit-failed", font=font, fill=(175, 30, 30))
        else:
            color = (25, 110, 55) if status == "modeled" else (150, 95, 20)
            panel_draw.text((4, CELL_HEIGHT - 14), status, font=font, fill=color)
        sheet.paste(panel, (left, top))
        glyph_rows.append(
            {
                "character": character,
                "status": status,
                "rendered": rendered,
                "instances": (
                    [part.instance for stroke in glyph.strokes for part in stroke.parts]
                    if glyph is not None
                    else []
                ),
            }
        )

    legend_y = rows * CELL_HEIGHT + 8
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (6, legend_y),
        "Gray line: baseline   Green: modeled   Amber: provisional   "
        "Red: fit-failed   MISSING: not modeled",
        font=font,
        fill=(35, 35, 35),
    )
    image_path = Path(output_path)
    json_path = image_path.with_suffix(".json")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(image_path, format="PNG", optimize=False, compress_level=9)

    payload = {
        "format": "inkgrammar-fit-control-sheet",
        "format_version": 1,
        "project_id": bundle.project.id,
        "target_count": len(characters),
        "rendered_count": sum(row["rendered"] for row in glyph_rows),
        "missing": [row["character"] for row in glyph_rows if row["status"] == "missing"],
        "provisional": [row["character"] for row in glyph_rows if row["status"] == "provisional"],
        "modeled": [row["character"] for row in glyph_rows if row["status"] == "modeled"],
        "fit_failed": [row["character"] for row in glyph_rows if row["status"] == "fit-failed"],
        "occurrence_deltas": occurrence_deltas,
        "scale_policy": "shared",
        "pixels_per_unit": PIXELS_PER_UNIT,
        "baseline_px": BASELINE_Y,
        "cell_size": [CELL_WIDTH, CELL_HEIGHT],
        "grid_columns": GRID_COLUMNS,
        "glyphs": glyph_rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
