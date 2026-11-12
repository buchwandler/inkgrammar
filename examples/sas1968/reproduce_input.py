from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from inkgrammar.geometry import analyze_path, sample_path  # noqa: E402
from inkgrammar.io import load_model  # noqa: E402
from inkgrammar.models import WritingModelV3  # noqa: E402
from inkgrammar.raster import render_text_frame, scale_sanity  # noqa: E402
from inkgrammar.shaping import shape_text  # noqa: E402

MODEL_DEFAULT = ROOT / "data/models/sas-1968.yaml"
OUTPUT_DEFAULT = ROOT / "work/examples/sas-1968/reproduce-input"
TARGETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzß0123456789"
WORD_TARGETS = (
    ("Schulausgangsschrift", "reference title"),
    ("Schule", "reference word"),
    ("schreiben", "reference word"),
    ("Ausgangsschrift", "reference word"),
    ("minimum", "fully covered negative control"),
    ("banana", "fully covered negative control"),
    ("Cabin", "fully covered negative control"),
    ("buch", "fully covered negative control"),
    ("whim", "fully covered negative control"),
    ("calcium", "fully covered negative control"),
)
CONNECTION_GROUPS = (
    ("nu na ne nl", ("nu", "na", "ne", "nl")),
    ("Gu Ga Ge Gl", ("Gu", "Ga", "Ge", "Gl")),
    ("Ku Ka Ke Kl", ("Ku", "Ka", "Ke", "Kl")),
    ("Lu La Le", ("Lu", "La", "Le")),
    ("ru ra re", ("ru", "ra", "re")),
    ("bu bra be bl", ("bu", "bra", "be", "bl")),
    ("Au Aa Ae Al", ("Au", "Aa", "Ae", "Al")),
    ("Pu Pa Pe Pl", ("Pu", "Pa", "Pe", "Pl")),
    ("Wu Wa We Wl", ("Wu", "Wa", "We", "Wl")),
    ("Su Sa Se Sl", ("Su", "Sa", "Se", "Sl")),
    ("ba be He", ("ba", "be", "He")),
    ("Do Ja Pe", ("Do", "Ja", "Pe")),
)
OUTPUT_NAMES = (
    "01-alphabet-centerline.png",
    "02-alphabet-runtime-ink.png",
    "03-words-centerline.png",
    "04-words-runtime-ink.png",
    "05-connections-centerline.png",
    "06-scale-diagnostics.png",
    "manifest.json",
)
CENTERLINE_ALPHABET_CELL = (126, 106)
RUNTIME_ALPHABET_CELL = (220, 340)
RUNTIME_GLYPH_SCALE = 0.24
RUNTIME_WORD_SCALE = 0.16


def _missing_characters(model: WritingModelV3, text: str) -> list[str]:
    return list(
        dict.fromkeys(
            character
            for character in text
            if not character.isspace() and character not in model.glyphs
        )
    )


def _target_status(model: WritingModelV3, text: str, purpose: str) -> dict[str, Any]:
    missing = _missing_characters(model, text)
    if missing:
        return {
            "text": text,
            "purpose": purpose,
            "status": "blocked",
            "missing": missing,
            "rendered": False,
            "connection_rules_used": 0,
        }
    shaped = shape_text(model, text, allow_synthetic_bridges=True)
    return {
        "text": text,
        "purpose": purpose,
        "status": "supported",
        "missing": [],
        "rendered": True,
        "connection_rules_used": sum(join.rule_id is not None for join in shaped.joins),
    }


def _centerline_paths(
    model: WritingModelV3, text: str
) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float]]]:
    shaped = shape_text(model, text, allow_synthetic_bridges=True)
    paths = []
    taps = []
    for stroke in shaped.strokes:
        if stroke.d is not None:
            points = sample_path(stroke.d, 64)
            if points:
                paths.append(points)
        elif stroke.tap is not None:
            taps.append(stroke.tap)
    return paths, taps


def _centerline_preview(
    model: WritingModelV3,
    text: str,
    size: tuple[int, int],
    *,
    line_width: int = 2,
) -> Image.Image:
    paths, taps = _centerline_paths(model, text)
    points = [point for path in paths for point in path] + taps
    image = Image.new("L", size, 255)
    if not points:
        return image

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    margin = 10
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((size[0] - 2 * margin) / span_x, (size[1] - 2 * margin) / span_y)
    offset_x = (size[0] - span_x * scale) / 2
    offset_y = (size[1] - span_y * scale) / 2

    def pixel(point: tuple[float, float]) -> tuple[float, float]:
        return (
            offset_x + (point[0] - min_x) * scale,
            size[1] - offset_y - (point[1] - min_y) * scale,
        )

    draw = ImageDraw.Draw(image)
    for path in paths:
        if len(path) >= 2:
            draw.line([pixel(point) for point in path], fill=0, width=line_width, joint="curve")
    for tap in taps:
        x, y = pixel(tap)
        radius = max(1, line_width)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
    return image


def _draw_centerline_grid(
    model: WritingModelV3,
    targets: Sequence[str],
    output: Path,
    *,
    title: str,
    columns: int,
    cell_size: tuple[int, int],
) -> None:
    rows = math.ceil(len(targets) / columns)
    label_height = 20
    cell_width, cell_height = cell_size
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + label_height) + 34), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(20, 20, 20), font=font)

    for index, target in enumerate(targets):
        column, row = index % columns, index // columns
        left = column * cell_width
        top = row * (cell_height + label_height) + 28
        draw.text((left + 4, top + 2), target, fill=(20, 20, 20), font=font)
        missing = _missing_characters(model, target)
        if missing:
            draw.text((left + 4, top + 25), "MISSING", fill=(130, 130, 130), font=font)
            draw.text(
                (left + 4, top + 41),
                f"missing: {''.join(missing)}",
                fill=(130, 130, 130),
                font=font,
            )
            continue
        preview = _centerline_preview(model, target, cell_size)
        sheet.paste(preview.convert("RGB"), (left, top + label_height))

    _save_png(sheet, output)


def _draw_runtime_alphabet(model: WritingModelV3, output: Path) -> None:
    columns = 9
    cell_width, cell_height = RUNTIME_ALPHABET_CELL
    rows = math.ceil(len(TARGETS) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 36), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (6, 8),
        f"Canonical runtime ink at {RUNTIME_GLYPH_SCALE:g} px/font-unit",
        fill=(20, 20, 20),
        font=font,
    )
    baseline = cell_height - 88

    for index, character in enumerate(TARGETS):
        column, row = index % columns, index // columns
        left = column * cell_width
        top = row * cell_height + 30
        draw.text((left + 5, top + 3), character, fill=(20, 20, 20), font=font)
        missing = _missing_characters(model, character)
        if missing:
            draw.text((left + 5, top + 24), "MISSING", fill=(130, 130, 130), font=font)
            continue
        frame = render_text_frame(model, character, scale=RUNTIME_GLYPH_SCALE)
        x = left + (cell_width - frame.image.width) // 2 - round(frame.origin[0])
        y = top + baseline - round(frame.origin[1])
        sheet.paste(frame.image.convert("RGB"), (x, y), ImageOps.invert(frame.image))

    _save_png(sheet, output)


def _runtime_ink_crop(model: WritingModelV3, text: str) -> Image.Image:
    frame = render_text_frame(model, text, scale=RUNTIME_WORD_SCALE)
    ink = ImageOps.invert(frame.image)
    bbox = ink.getbbox()
    if bbox is None:
        raise ValueError(f"supported target rendered no ink: {text!r}")
    padding = 6
    return frame.image.crop(
        (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(frame.image.width, bbox[2] + padding),
            min(frame.image.height, bbox[3] + padding),
        )
    )


def _draw_word_sheet(
    model: WritingModelV3,
    output: Path,
    *,
    runtime_ink: bool,
) -> list[dict[str, Any]]:
    rows = [_target_status(model, text, purpose) for text, purpose in WORD_TARGETS]
    header_height = 38
    row_height = 160 if runtime_ink else 104
    sheet = Image.new("RGB", (1500, header_height + len(rows) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    title = "Words, canonical runtime ink" if runtime_ink else "Words, normalized centerlines"
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)

    for index, row in enumerate(rows):
        top = header_height + index * row_height
        draw.line((0, top, sheet.width, top), fill=(220, 220, 220), width=1)
        draw.text((8, top + 8), row["text"], fill=(20, 20, 20), font=font)
        draw.text((8, top + 25), row["purpose"], fill=(75, 75, 75), font=font)
        if row["status"] == "blocked":
            missing = "".join(row["missing"])
            draw.text((8, top + 46), f"BLOCKED, missing: {missing}", fill=(150, 40, 40), font=font)
            continue
        draw.text(
            (8, top + 45),
            f"SUPPORTED, connection rules used: {row['connection_rules_used']}",
            fill=(30, 100, 30),
            font=font,
        )
        if runtime_ink:
            preview = _runtime_ink_crop(model, row["text"])
            sheet.paste(preview.convert("RGB"), (360, top + 4), ImageOps.invert(preview))
        else:
            preview = _centerline_preview(model, row["text"], (1090, 62))
            sheet.paste(preview.convert("RGB"), (360, top + 35))

    _save_png(sheet, output)
    return rows


def _draw_connection_sheet(model: WritingModelV3, output: Path) -> list[dict[str, Any]]:
    targets = [
        {"group": group, "text": text}
        for group, examples in CONNECTION_GROUPS
        for text in examples
    ]
    columns = 5
    cell_width, cell_height = 206, 96
    rows = math.ceil(len(targets) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 36), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (6, 8),
        f"Reference connection targets, runtime rules: {len(model.connection_rules)}",
        fill=(20, 20, 20),
        font=font,
    )
    for index, row in enumerate(targets):
        status = _target_status(model, row["text"], row["group"])
        row.update(
            {
                key: status[key]
                for key in ("status", "missing", "rendered", "connection_rules_used")
            }
        )
        column, grid_row = index % columns, index // columns
        left = column * cell_width
        top = grid_row * cell_height + 30
        draw.text((left + 4, top + 1), row["text"], fill=(20, 20, 20), font=font)
        draw.text((left + 4, top + 15), row["group"], fill=(85, 85, 85), font=font)
        if not row["rendered"]:
            draw.text(
                (left + 4, top + 31),
                f"MISSING: {''.join(row['missing'])}",
                fill=(140, 40, 40),
                font=font,
            )
            continue
        draw.text(
            (left + 4, top + 31),
            f"rules used: {row['connection_rules_used']}",
            fill=(30, 100, 30),
            font=font,
        )
        preview = _centerline_preview(model, row["text"], (cell_width - 12, 48))
        sheet.paste(preview.convert("RGB"), (left + 6, top + 45))

    _save_png(sheet, output)
    return targets


def _variant_scale_report(model: WritingModelV3, character: str, variant_id: str) -> dict[str, Any]:
    variant = model.glyphs[character].variants[variant_id]
    boxes = [
        analyze_path(part.d).bbox
        for stroke in variant.strokes
        if getattr(stroke, "role", "body") not in {"dot", "diacritic"}
        for part in getattr(stroke, "parts", [])
        if getattr(part, "d", None)
    ]
    if not boxes:
        return {"status": "mark-only", "flags": []}
    height = max(box[3] for box in boxes) - min(box[1] for box in boxes)
    report = scale_sanity(
        ink_bbox_height=height,
        x_height=model.metrics.x_height,
        stroke_width=model.metrics.default_stroke_width,
    )
    return {
        "status": "pathological" if report.pathological else "ok",
        "ink_bbox_height": round(report.ink_bbox_height, 6),
        "ink_bbox_height_over_x_height": round(report.ink_bbox_height_over_x_height, 6),
        "stroke_width_over_x_height": round(report.stroke_width_over_x_height, 6),
        "stroke_width_over_ink_bbox_height": round(report.stroke_width_over_ink_bbox_height, 6),
        "flags": list(report.flags),
    }


def _scale_reports(model: WritingModelV3) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        character: {
            variant_id: _variant_scale_report(model, character, variant_id)
            for variant_id in glyph.variants
        }
        for character, glyph in model.glyphs.items()
    }


def _draw_scale_sheet(
    model: WritingModelV3,
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output: Path,
) -> None:
    width = 1040
    row_height = 42
    header_height = 68
    sheet = Image.new("RGB", (width, header_height + row_height * len(reports)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), "Runtime scale diagnostics", fill=(20, 20, 20), font=font)
    draw.text(
        (8, 27),
        f"x-height={model.metrics.x_height:g}  stroke-width={model.metrics.default_stroke_width:g}",
        fill=(50, 50, 50),
        font=font,
    )
    draw.text(
        (8, 47),
        "Glyph / variant          ink height / x-height        stroke / ink height       status",
        fill=(40, 40, 40),
        font=font,
    )

    for row_index, (character, variants) in enumerate(reports.items()):
        top = header_height + row_index * row_height
        draw.line((0, top, width, top), fill=(225, 225, 225), width=1)
        for variant_index, (variant_id, report) in enumerate(variants.items()):
            y = top + 4 + variant_index * 16
            draw.text((8, y), f"{character} / {variant_id}", fill=(20, 20, 20), font=font)
            if report["status"] == "mark-only":
                draw.text((250, y), "mark only", fill=(70, 70, 70), font=font)
                continue
            color = (160, 35, 35) if report["status"] == "pathological" else (30, 100, 30)
            draw.text(
                (250, y),
                f"{report['ink_bbox_height_over_x_height']:.3f}",
                fill=(20, 20, 20),
                font=font,
            )
            draw.text(
                (480, y),
                f"{report['stroke_width_over_ink_bbox_height']:.3f}",
                fill=(20, 20, 20),
                font=font,
            )
            draw.text((750, y), report["status"], fill=color, font=font)

    _save_png(sheet, output)


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_input(model_path: Path, output: Path) -> dict[str, Any]:
    model = load_model(model_path)
    if not isinstance(model, WritingModelV3):
        raise ValueError("SAS-1968 reproduction requires a schema-v3 runtime model")
    output.mkdir(parents=True, exist_ok=True)

    _draw_centerline_grid(
        model,
        TARGETS,
        output / OUTPUT_NAMES[0],
        title="Alphabet and digits, geometry-normalized centerlines",
        columns=9,
        cell_size=CENTERLINE_ALPHABET_CELL,
    )
    _draw_runtime_alphabet(model, output / OUTPUT_NAMES[1])
    word_centerlines = _draw_word_sheet(model, output / OUTPUT_NAMES[2], runtime_ink=False)
    word_runtime_ink = _draw_word_sheet(model, output / OUTPUT_NAMES[3], runtime_ink=True)
    connection_targets = _draw_connection_sheet(model, output / OUTPUT_NAMES[4])
    scale_reports = _scale_reports(model)
    _draw_scale_sheet(model, scale_reports, output / OUTPUT_NAMES[5])

    modeled = [character for character in TARGETS if character in model.glyphs]
    missing = [character for character in TARGETS if character not in model.glyphs]
    words = word_centerlines
    connection_supported = [row["text"] for row in connection_targets if row["rendered"]]
    connection_blocked = [row["text"] for row in connection_targets if not row["rendered"]]
    coordinate_conversion = model.provenance.get("coordinate_conversion", {})
    manifest = {
        "format": "inkgrammar-sas1968-input-reproduction",
        "format_version": 1,
        "model_id": model.id,
        "model_schema_version": model.schema_version,
        "model_sha256": _sha256(model_path),
        "reproduction_claim": "not_claimed",
        "coverage": {
            "target_count": len(TARGETS),
            "modeled_count": len(modeled),
            "missing_count": len(missing),
            "modeled_targets": modeled,
            "missing_targets": missing,
        },
        "connection_rule_count": len(model.connection_rules),
        "context_rule_count": len(model.context_rules),
        "coordinate_conversion": coordinate_conversion,
        "metrics": {
            "units_per_em": model.metrics.units_per_em,
            "x_height": model.metrics.x_height,
            "default_stroke_width": model.metrics.default_stroke_width,
        },
        "per_glyph": scale_reports,
        "words": {
            "supported": [row["text"] for row in words if row["status"] == "supported"],
            "blocked": [row["text"] for row in words if row["status"] == "blocked"],
            "targets": words,
            "runtime_ink_status": [row["status"] for row in word_runtime_ink],
        },
        "connection_targets": {
            "supported": connection_supported,
            "blocked": connection_blocked,
            "targets": connection_targets,
        },
        "outputs": list(OUTPUT_NAMES[:-1]),
    }
    (output / OUTPUT_NAMES[-1]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render source-free SAS-1968 runtime review sheets."
    )
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"runtime model does not exist: {model_path}")
    manifest = render_input(model_path, output)
    print(f"Rendered {len(manifest['outputs'])} review sheets and manifest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
