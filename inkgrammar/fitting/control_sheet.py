from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..compiler import compile_fit_project
from ..geometry import analyze_path, translate_path
from ..models import RuntimeContextRule
from ..raster import render_text_frame
from .connection_fit import connection_observation_coverage
from .family_fit import load_fit_reports, primitive_path
from .models import (
    CurvatureStrokeGeometry,
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


def render_glyph_grid(
    bundle: FitProjectBundle,
    *,
    templates: Mapping[str, PrimitiveTemplate],
    template_sources: Mapping[str, str],
    characters: str,
    output_path: str | Path,
    geometry_label: str | None = None,
    reports: Mapping[str, dict] | None = None,
    occurrence_deltas: bool = False,
) -> dict:
    fit_reports = dict(reports or {})
    template_map = dict(templates)
    font = ImageFont.load_default()
    cell_count = len(characters)
    rows = (cell_count + GRID_COLUMNS - 1) // GRID_COLUMNS
    sheet = Image.new(
        "RGB",
        (GRID_COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT + 28 + (24 if geometry_label else 0)),
        "white",
    )
    glyph_rows = []

    for index, character in enumerate(characters):
        column, row = index % GRID_COLUMNS, index // GRID_COLUMNS
        left, top = column * CELL_WIDTH, row * CELL_HEIGHT + (24 if geometry_label else 0)
        panel = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.line((0, BASELINE_Y, CELL_WIDTH - 1, BASELINE_Y), fill=(190, 190, 190), width=1)
        panel_draw.text((4, 3), character, font=font, fill=(20, 20, 20))

        glyph = bundle.topology.glyphs.get(character)
        status: Status = "missing" if glyph is None else _status(glyph, template_map, fit_reports)
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
                        deltas = _sample_deltas(sample, glyph, fit_reports)
                        if deltas:
                            _paste_coverage(
                                panel,
                                _coverage(_glyph_paths(glyph, template_map, deltas), bundle),
                                (130, 185, 210),
                            )
                _paste_coverage(
                    panel,
                    _coverage(_glyph_paths(glyph, template_map), bundle),
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
        glyph_sources = (
            sorted(
                {
                    template_sources.get(part.primitive, "unknown")
                    for stroke in glyph.strokes
                    for part in stroke.parts
                }
            )
            if glyph is not None
            else []
        )
        glyph_rows.append(
            {
                "character": character,
                "status": status,
                "rendered": rendered,
                "geometry_sources": glyph_sources,
                "instances": (
                    [part.instance for stroke in glyph.strokes for part in stroke.parts]
                    if glyph is not None
                    else []
                ),
            }
        )

    legend_y = rows * CELL_HEIGHT + (24 if geometry_label else 0) + 8
    draw = ImageDraw.Draw(sheet)
    if geometry_label:
        draw.text((6, 6), geometry_label, font=font, fill=(20, 20, 20))
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
        "geometry_label": geometry_label,
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
        "template_sources": dict(sorted(template_sources.items())),
        "glyphs": glyph_rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_control_sheet(
    bundle: FitProjectBundle,
    output_path: str | Path,
    *,
    include_digits: bool = False,
    occurrence_deltas: bool = False,
) -> dict:
    characters = DEFAULT_CHARACTERS + (DIGITS if include_digits else "")
    templates, reports = _fit_templates(bundle)
    template_sources = {primitive_id: "seed-fallback" for primitive_id in templates}
    for primitive_id, report in reports.items():
        if report.get("visual_review", {}).get("status") == "promoted":
            template_sources[primitive_id] = "promoted"
    return render_glyph_grid(
        bundle,
        templates=templates,
        template_sources=template_sources,
        characters=characters,
        output_path=output_path,
        reports=reports,
        occurrence_deltas=occurrence_deltas,
    )


def _preview_model(bundle: FitProjectBundle):
    with tempfile.TemporaryDirectory(prefix=".inkgrammar-preview-", dir=bundle.project_dir) as temp:
        output = (Path(temp) / "preview.yaml").relative_to(bundle.project_dir)
        compile_config = bundle.project.compile.model_copy(update={"output": str(output)})
        project = bundle.project.model_copy(update={"compile": compile_config})
        preview_bundle = replace(bundle, project=project)
        return compile_fit_project(preview_bundle, require_promoted=False)


def _write_debug_grid(rows: list[dict], output_path: str | Path, title: str) -> dict:
    columns = 4
    cell_width, cell_height = 240, 156
    visible_rows = rows or [
        {"title": "No configured examples", "message": "No evidence synthesized."}
    ]
    row_count = (len(visible_rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, row_count * cell_height + 30), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(visible_rows):
        column, y_index = index % columns, index // columns
        left, top = column * cell_width, y_index * cell_height
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline=(190, 190, 190),
        )
        draw.text((left + 6, top + 5), str(row.get("title", ""))[:34], font=font, fill=(20, 20, 20))
        draw.text(
            (left + 6, top + 19),
            str(row.get("detail", ""))[:38],
            font=font,
            fill=(55, 55, 55),
        )
        preview = row.get("preview")
        if isinstance(preview, Image.Image):
            image = preview.convert("RGB")
            image.thumbnail((cell_width - 16, 108), Image.Resampling.LANCZOS)
            sheet.paste(image, (left + (cell_width - image.width) // 2, top + 40))
        elif row.get("message"):
            draw.text((left + 6, top + 56), str(row["message"])[:38], font=font, fill=(140, 70, 30))
    draw.text((6, row_count * cell_height + 8), title, font=font, fill=(35, 35, 35))
    image_path = Path(output_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(image_path, format="PNG", optimize=False, compress_level=9)
    payload = {
        "format": "inkgrammar-fit-debug-control-sheet",
        "format_version": 1,
        "source_free": True,
        "case_count": len(rows),
        "cases": [
            {key: value for key, value in row.items() if key not in {"preview", "message"}}
            for row in rows
        ],
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _render_pair_preview(
    model, left_character: str, left_variant: str, right_character: str, right_variant: str, rule
):
    text = left_character + right_character
    preview_model = model.model_copy(
        update={
            "context_rules": [
                RuntimeContextRule(
                    id="control-left",
                    character=left_character,
                    variant=left_variant,
                    next_characters=[right_character],
                    positions=["initial"],
                    priority=100000,
                ),
                RuntimeContextRule(
                    id="control-right",
                    character=right_character,
                    variant=right_variant,
                    prev_characters=[left_character],
                    positions=["final"],
                    priority=100000,
                ),
            ],
            "connection_rules": [rule],
        }
    )
    return render_text_frame(preview_model, text, scale=0.35).image


def write_connection_control_sheet(
    bundle: FitProjectBundle,
    output_path: str | Path,
) -> dict:
    compiled = _preview_model(bundle)
    model = compiled.model
    observations = bundle.grammar.connection_observations
    rows = []
    for rule in model.connection_rules:
        left_candidates = [
            (character, variant.id)
            for character, glyph in sorted(model.glyphs.items())
            for variant in sorted(glyph.variants.values(), key=lambda item: item.id)
            if variant.exit is not None
            and variant.exit.class_ == rule.from_class
            and (rule.required_variant_left is None or variant.id == rule.required_variant_left)
        ]
        right_candidates = [
            (character, variant.id)
            for character, glyph in sorted(model.glyphs.items())
            for variant in sorted(glyph.variants.values(), key=lambda item: item.id)
            if variant.entry is not None
            and variant.entry.class_ == rule.to_class
            and (rule.required_variant_right is None or variant.id == rule.required_variant_right)
        ]
        pair = (
            (left_candidates[0], right_candidates[0])
            if left_candidates and right_candidates
            else None
        )
        matched = [
            item
            for item in observations
            if pair is not None
            and (item.from_character, item.to_character) == (pair[0][0], pair[1][0])
        ]
        preview = None
        message = None
        if pair is None:
            message = "No matching endpoint variants"
        else:
            try:
                preview = _render_pair_preview(
                    model, pair[0][0], pair[0][1], pair[1][0], pair[1][1], rule
                )
            except ValueError as error:
                message = str(error)
        rows.append(
            {
                "title": rule.id,
                "detail": f"{rule.from_class} -> {rule.to_class} | {rule.mode}",
                "rule_id": rule.id,
                "from_class": rule.from_class,
                "to_class": rule.to_class,
                "mode": rule.mode,
                "status": rule.status,
                "left_variant": pair[0][1] if pair else None,
                "right_variant": pair[1][1] if pair else None,
                "observation_splits": {
                    split: sum(item.split == split for item in matched)
                    for split in ("fit", "validation", "hold-out")
                },
                "preview": preview,
                "message": message,
            }
        )
    if not rows:
        for observation in observations:
            missing = [
                character
                for character in observation.text
                if character not in model.glyphs
            ]
            preview = None
            message = None
            if missing:
                message = f"Missing runtime glyphs: {''.join(dict.fromkeys(missing))}"
            else:
                preview = render_text_frame(model, observation.text, scale=0.35).image
            rows.append(
                {
                    "title": observation.text,
                    "detail": f"No rule | {observation.split} | mapping pending",
                    "observation_id": observation.observation_id,
                    "split": observation.split,
                    "from_character": observation.from_character,
                    "to_character": observation.to_character,
                    "expected_class": observation.expected_class,
                    "connection_rules_used": 0,
                    "preview": preview,
                    "message": message,
                }
            )
    coverage = connection_observation_coverage(observations)
    report = _write_debug_grid(
        rows,
        output_path,
        f"Connection control cases: {len(rows)} | "
        f"observations by split: {coverage['split_counts']}",
    )
    report["observation_coverage"] = coverage
    report["runtime_rule_count"] = len(model.connection_rules)
    report["seed_fallback_primitives"] = compiled.diagnostics["seed_fallback_primitives"]
    report["promoted_primitives"] = compiled.diagnostics["promoted_primitives"]
    image_path = Path(output_path)
    image_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_context_control_sheet(
    bundle: FitProjectBundle,
    output_path: str | Path,
) -> dict:
    compiled = _preview_model(bundle)
    model = compiled.model
    rows = []
    for character, glyph in sorted(model.glyphs.items()):
        for index, variant in enumerate(sorted(glyph.variants.values(), key=lambda item: item.id)):
            preview_model = model.model_copy(
                update={
                    "context_rules": [
                        RuntimeContextRule(
                            id=f"control-variant-{index}",
                            character=character,
                            variant=variant.id,
                            positions=["isolated"],
                            priority=100000,
                        )
                    ],
                    "connection_rules": [],
                }
            )
            rows.append(
                {
                    "title": f"{character} | {variant.id}",
                    "detail": f"default: {glyph.default_variant == variant.id}",
                    "character": character,
                    "variant_id": variant.id,
                    "is_default": glyph.default_variant == variant.id,
                    "entry_class": variant.entry.class_ if variant.entry else None,
                    "exit_class": variant.exit.class_ if variant.exit else None,
                    "preview": render_text_frame(preview_model, character, scale=0.35).image,
                }
            )
    report = _write_debug_grid(
        rows,
        output_path,
        f"Context variants: {len(rows)} | stage: development geometry",
    )
    report["seed_fallback_primitives"] = compiled.diagnostics["seed_fallback_primitives"]
    report["promoted_primitives"] = compiled.diagnostics["promoted_primitives"]
    image_path = Path(output_path)
    image_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report



def render_trajectory_diagnostic(
    geometry: CurvatureStrokeGeometry,
    *,
    size: int = 256,
    direction_arrow_fraction: float = 0.12,
) -> Image.Image:
    """Trajectory diagnostic panel (the brief's section 20.7).

    Shows the integrated centerline with direction arrows, start/end markers
    with tangent arrows, and numbered curvature-change markers, so the writing
    direction and the turning program are directly inspectable.
    """
    from .curvature import evaluate_curvature_stroke, evaluate_stroke_at

    sample = evaluate_curvature_stroke(geometry, count=257)
    points = sample.xy
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    scale = (size * 0.72) / float(span.max())
    center = (low + high) / 2.0

    def to_pixels(values: np.ndarray) -> list[tuple[float, float]]:
        values = np.asarray(values, dtype=float)
        mapped = (values - center) * scale
        return [(size / 2 + float(x), size / 2 - float(y)) for x, y in mapped]

    panel = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    polyline = to_pixels(points)
    draw.line(polyline, fill=(20, 20, 20), width=2)

    def draw_arrow(origin: tuple[float, float], direction: np.ndarray, color) -> None:
        direction = direction / np.linalg.norm(direction)
        tip = (origin[0] + float(direction[0]) * 18, origin[1] - float(direction[1]) * 18)
        draw.line([origin, tip], fill=color, width=2)
        back = (tip[0] - float(direction[0]) * 6, tip[1] + float(direction[1]) * 6)
        perpendicular = (-direction[1], direction[0])
        draw.line(
            [tip, (back[0] + float(perpendicular[0]) * 4, back[1] - float(perpendicular[1]) * 4)],
            fill=color,
            width=2,
        )
        draw.line(
            [tip, (back[0] - float(perpendicular[0]) * 4, back[1] + float(perpendicular[1]) * 4)],
            fill=color,
            width=2,
        )

    start_px, end_px = polyline[0], polyline[-1]
    draw.rectangle(
        (start_px[0] - 4, start_px[1] - 4, start_px[0] + 4, start_px[1] + 4),
        outline=(0, 160, 0),
        width=2,
    )
    draw.rectangle(
        (end_px[0] - 4, end_px[1] - 4, end_px[0] + 4, end_px[1] + 4),
        outline=(0, 0, 220),
        width=2,
    )
    draw_arrow(start_px, sample.tangent[0], (0, 160, 0))
    draw_arrow(end_px, sample.tangent[-1], (0, 0, 220))

    change_positions = [change.at for change in geometry.curvature.changes]
    if change_positions:
        change_xy, _ = evaluate_stroke_at(geometry, np.asarray(change_positions))
        for number, point in enumerate(to_pixels(change_xy), start=1):
            draw.ellipse(
                (point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6),
                outline=(220, 0, 0),
                width=2,
            )
            draw.text((point[0] + 7, point[1] - 6), str(number), fill=(220, 0, 0))

    step = max(1, int(round(1.0 / max(direction_arrow_fraction, 0.01))))
    direction_u = np.linspace(0.0, 1.0, step + 1)[1:-1]
    if direction_u.size:
        arrow_xy, _ = evaluate_stroke_at(geometry, direction_u)
        for index, point in enumerate(to_pixels(arrow_xy)):
            tangent = sample.tangent[
                int(np.argmin(np.abs(sample.u - direction_u[index])))
            ]
            draw_arrow(point, tangent, (120, 120, 120))

    return panel
