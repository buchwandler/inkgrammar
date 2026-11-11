from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from inkgrammar.fitting.control_sheet import DIGITS, render_glyph_grid
from inkgrammar.fitting.preview import PreviewBuild, PreviewGeometry, build_fit_preview_model
from inkgrammar.fitting.workspace import FitProjectBundle, load_fit_project
from inkgrammar.models import WritingModelV3
from inkgrammar.raster import RasterFrame, render_text_frame
from inkgrammar.shaping import shape_text
from inkgrammar.svg import write_text_svg

DEFAULT_PROJECT = Path("data/training/sas-1968/project.yaml")
DEFAULT_OUTPUT = Path("work/examples/sas-1968")
DEFAULT_CASES = Path(__file__).with_name("cases.yaml")
GLYPH_SCALE = 1.0
WORD_SCALE = 0.5
WORD_SHEET_WIDTH = 1200
WORD_ROW_HEIGHT = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_hashes(bundle: FitProjectBundle) -> dict[str, str]:
    tracked = {
        "runtime_model": (bundle.project_dir / bundle.project.compile.output).resolve(strict=True),
        "seed_grammar": (bundle.project_dir / bundle.project.seed_grammar).resolve(strict=True),
        "topology": (bundle.project_dir / bundle.project.topology).resolve(strict=True),
        "families": (bundle.project_dir / bundle.project.families).resolve(strict=True),
        "fit_config": (bundle.project_dir / bundle.project.fit_config).resolve(strict=True),
    }
    return {name: _sha256(path) for name, path in tracked.items()}


def _load_cases(path: Path) -> dict[str, Any]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or cases.get("schema_version") != 1:
        raise ValueError(f"unsupported SAS-1968 cases file: {path}")
    return cases


def _target_characters(cases: Mapping[str, Any], include_digits: bool) -> str:
    alphabet = cases["alphabet"]
    characters = alphabet["uppercase"] + alphabet["lowercase"]
    if include_digits:
        characters += alphabet.get("digits", DIGITS)
    return characters


def _align_glyph_frames(
    builds: Sequence[PreviewBuild], characters: Sequence[str], scale: float
) -> tuple[dict[tuple[int, str], Image.Image], tuple[int, int], int]:
    raw_frames: dict[tuple[int, str], RasterFrame] = {}
    for build_index, build in enumerate(builds):
        for character in characters:
            if character in build.model.glyphs:
                raw_frames[(build_index, character)] = render_text_frame(
                    build.model, character, scale=scale
                )
    if not raw_frames:
        return {}, (1, 1), 0

    left = max(round(frame.origin[0]) for frame in raw_frames.values())
    right = max(frame.image.width - round(frame.origin[0]) for frame in raw_frames.values())
    above = max(round(frame.origin[1]) for frame in raw_frames.values())
    below = max(frame.image.height - round(frame.origin[1]) for frame in raw_frames.values())
    margin = 12
    origin_x = margin + left
    baseline_y = margin + above
    size = (origin_x + right + margin, baseline_y + below + margin)
    aligned: dict[tuple[int, str], Image.Image] = {}
    for key, frame in raw_frames.items():
        image = Image.new("L", size, 255)
        image.paste(
            frame.image,
            (origin_x - round(frame.origin[0]), baseline_y - round(frame.origin[1])),
        )
        aligned[key] = image
    return aligned, size, baseline_y


def _draw_metric_guides(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    width: int,
    baseline_y: int,
    model: WritingModelV3,
    scale: float,
) -> None:
    baseline = top + baseline_y
    x_height = round(baseline_y - (model.metrics.x_height - model.metrics.baseline) * scale) + top
    draw.line((left, baseline, left + width - 1, baseline), fill=(125, 125, 125), width=1)
    draw.line((left, x_height, left + width - 1, x_height), fill=(185, 185, 185), width=1)


def _write_supported_glyph_sheet(
    build: PreviewBuild,
    characters: Sequence[str],
    output: Path,
    *,
    frames: Mapping[tuple[int, str], Image.Image],
    frame_size: tuple[int, int],
    baseline_y: int,
    build_index: int,
    geometry_label: str,
    scale: float,
) -> dict[str, Any]:
    columns = 6
    cell_width = frame_size[0] + 16
    cell_height = frame_size[1] + 34
    rows = (len(characters) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 38), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((6, 5), geometry_label, fill=(25, 25, 25), font=font)

    for index, character in enumerate(characters):
        column, row = index % columns, index // columns
        left, top = column * cell_width, row * cell_height + 24
        frame = frames[(build_index, character)]
        frame_left, frame_top = left + 8, top + 10
        _draw_metric_guides(
            draw,
            left=frame_left,
            top=frame_top,
            width=frame_size[0],
            baseline_y=baseline_y,
            model=build.model,
            scale=scale,
        )
        mask = ImageOps.invert(frame)
        sheet.paste(frame.convert("RGB"), (frame_left, frame_top), mask)
        draw.text((left + 5, top + 1), character, fill=(20, 20, 20), font=font)

    draw.text(
        (6, rows * cell_height + 27),
        f"Scale: {scale:g} px/unit   Gray: x-height   Dark gray: baseline",
        fill=(35, 35, 35),
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False, compress_level=9)
    return {
        "glyph_count": len(characters),
        "scale": scale,
        "cell_size": [cell_width, cell_height],
        "baseline_px": baseline_y,
        "x_height_px": round(
            baseline_y - (build.model.metrics.x_height - build.model.metrics.baseline) * scale
        ),
        "geometry": build.diagnostics["preview_geometry"],
    }


def _word_case(model: WritingModelV3, text: str, purpose: str) -> dict[str, Any]:
    missing = list(
        dict.fromkeys(
            character
            for character in text
            if not character.isspace() and character not in model.glyphs
        )
    )
    if missing:
        return {
            "text": text,
            "purpose": purpose,
            "status": "blocked",
            "missing": missing,
            "rendered": False,
            "width": None,
            "connection_rules_used": 0,
        }

    shaped = shape_text(model, text)
    frame = render_text_frame(model, text, scale=WORD_SCALE)
    ink = ImageOps.invert(frame.image)
    bbox = ink.getbbox()
    if bbox is None:
        raise ValueError(f"supported word rendered no ink: {text}")
    padding = 6
    crop = frame.image.crop(
        (
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(frame.image.width, bbox[2] + padding),
            min(frame.image.height, bbox[3] + padding),
        )
    )
    return {
        "text": text,
        "purpose": purpose,
        "status": "supported",
        "missing": [],
        "rendered": True,
        "width": round(shaped.width, 3),
        "connection_rules_used": sum(join.rule_id is not None for join in shaped.joins),
        "image": crop,
    }


def _write_word_sheet(
    build: PreviewBuild,
    cases: Mapping[str, Any],
    output: Path,
    debug_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    word_cases = list(cases.get("words", []))
    word_cases.extend(
        {"text": word, "purpose": "future coverage, intentionally not modeled"}
        for word in cases.get("future_words", [])
    )
    rows = [
        _word_case(build.model, case["text"], case.get("purpose", "")) for case in word_cases
    ]

    font = ImageFont.load_default()
    header_height = 42
    sheet = Image.new(
        "RGB", (WORD_SHEET_WIDTH, header_height + len(rows) * WORD_ROW_HEIGHT), "white"
    )
    draw = ImageDraw.Draw(sheet)
    connection_count = len(build.model.connection_rules)
    joining = "not modeled" if connection_count == 0 else "rules present, visual joins unvalidated"
    draw.text((8, 5), "SAS-1968 compositional word preview", fill=(20, 20, 20), font=font)
    draw.text(
        (8, 20),
        f"Connection rules: {connection_count} | Glyph-boundary joining: {joining}",
        fill=(60, 60, 60),
        font=font,
    )

    debug_outputs: dict[str, str] = {}
    public_rows = []
    for index, row in enumerate(rows):
        top = header_height + index * WORD_ROW_HEIGHT
        draw.text((8, top + 6), row["text"], fill=(20, 20, 20), font=font)
        status_color = (30, 90, 45) if row["rendered"] else (175, 35, 35)
        draw.text((95, top + 6), row["status"], fill=status_color, font=font)
        if row["status"] == "blocked":
            draw.text(
                (95, top + 22),
                f"missing: {', '.join(row['missing'])}",
                fill=(175, 35, 35),
                font=font,
            )
        else:
            draw.text(
                (95, top + 22),
                f"width: {row['width']:.1f} | "
                f"connection rules used: {row['connection_rules_used']}",
                fill=(45, 45, 45),
                font=font,
            )
            draw.text((95, top + 38), row["purpose"][:48], fill=(65, 65, 65), font=font)
            image = row["image"]
            sheet.paste(image, (430, top + (WORD_ROW_HEIGHT - image.height) // 2))
            filename = re.sub(r"[^A-Za-z0-9_-]+", "-", row["text"]).strip("-") or "word"
            svg_path = debug_dir / f"{filename}.svg"
            write_text_svg(
                build.model,
                row["text"],
                svg_path,
                show_centerline=True,
                show_origins=True,
                show_anchors=True,
                show_tangents=True,
                show_guides=True,
                show_labels=True,
            )
            debug_outputs[row["text"]] = svg_path.relative_to(output.parent).as_posix()
        draw.line(
            (0, top + WORD_ROW_HEIGHT - 1, WORD_SHEET_WIDTH - 1, top + WORD_ROW_HEIGHT - 1),
            fill=(220, 220, 220),
        )
        public_rows.append({key: value for key, value in row.items() if key != "image"})

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False, compress_level=9)
    return public_rows, debug_outputs


def _save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_mode(
    bundle: FitProjectBundle,
    build: PreviewBuild,
    output: Path,
    characters: str,
    cases: Mapping[str, Any],
    *,
    frames: Mapping[tuple[int, str], Image.Image],
    frame_size: tuple[int, int],
    baseline_y: int,
    build_index: int,
) -> dict[str, Any]:
    geometry = build.diagnostics["preview_geometry"]
    geometry_label = (
        "CANONICAL RUNTIME GEOMETRY"
        if geometry == "canonical"
        else "LATEST CANDIDATE, UNPROMOTED PREVIEW"
    )
    alphabet = render_glyph_grid(
        bundle,
        templates=build.templates,
        template_sources=build.template_sources,
        characters=characters,
        output_path=output / "01-alphabet.png",
        geometry_label=geometry_label,
    )
    supported_glyphs = sorted(build.model.glyphs)
    glyph_sheet = _write_supported_glyph_sheet(
        build,
        supported_glyphs,
        output / "02-supported-glyphs.png",
        frames=frames,
        frame_size=frame_size,
        baseline_y=baseline_y,
        build_index=build_index,
        geometry_label=geometry_label,
        scale=GLYPH_SCALE,
    )
    word_rows, debug_outputs = _write_word_sheet(
        build,
        cases,
        output / "03-words.png",
        output / "debug",
    )
    _save_yaml(
        output / "model-preview.yaml",
        build.model.model_dump(mode="json", by_alias=True),
    )
    output_files = {
        "alphabet_png": "01-alphabet.png",
        "alphabet_json": "01-alphabet.json",
        "supported_glyphs_png": "02-supported-glyphs.png",
        "words_png": "03-words.png",
        "model_preview_yaml": "model-preview.yaml",
        "debug_svgs": debug_outputs,
    }
    manifest = {
        "format": "inkgrammar-example-validation",
        "format_version": 1,
        "project_id": bundle.project.id,
        "runtime_model_id": build.model.id,
        "geometry": geometry,
        "source_free": True,
        "canonical_model_modified": False,
        "glyph_target_count": alphabet["target_count"],
        "glyph_rendered_count": alphabet["rendered_count"],
        "missing_glyphs": alphabet["missing"],
        "supported_glyphs": supported_glyphs,
        "connection_rule_count": len(build.model.connection_rules),
        "connected_writing": bool(build.model.connection_rules),
        "primitive_geometry": build.template_sources,
        "fit_reports": build.diagnostics["fit_reports"],
        "words": word_rows,
        "supported_glyph_sheet": glyph_sheet,
        "outputs": output_files,
    }
    _save_json(output / "manifest.json", manifest)
    return manifest


def _save_comparison(
    left_path: Path,
    right_path: Path,
    output: Path,
    *,
    left_label: str,
    right_label: str,
) -> None:
    with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
        left = left_image.convert("RGB")
        right = right_image.convert("RGB")
    if left.size != right.size:
        raise ValueError(f"comparison panels must share dimensions: {left.size} != {right.size}")
    heading = 28
    canvas = Image.new("RGB", (left.width * 2, left.height + heading), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 5), left_label, fill=(20, 20, 20), font=font)
    draw.text((left.width + 6, 5), right_label, fill=(20, 20, 20), font=font)
    canvas.paste(left, (0, heading))
    canvas.paste(right, (left.width, heading))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False, compress_level=9)


def _write_glyph_difference(
    builds: Sequence[PreviewBuild],
    characters: Sequence[str],
    frames: Mapping[tuple[int, str], Image.Image],
    frame_size: tuple[int, int],
    baseline_y: int,
    output: Path,
) -> None:
    columns = 6
    cell_width, cell_height = frame_size[0] + 16, frame_size[1] + 34
    rows = (len(characters) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 42), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (6, 5),
        "Canonical vs latest candidate, same scale; blue=canonical red=candidate",
        font=font,
        fill=(20, 20, 20),
    )

    for index, character in enumerate(characters):
        left_frame = frames.get((0, character), Image.new("L", frame_size, 255))
        right_frame = frames.get((1, character), Image.new("L", frame_size, 255))
        left_ink = np.asarray(ImageOps.invert(left_frame)) > 10
        right_ink = np.asarray(ImageOps.invert(right_frame)) > 10
        overlay = np.full((frame_size[1], frame_size[0], 3), 255, dtype=np.uint8)
        overlay[left_ink & ~right_ink] = (45, 95, 190)
        overlay[right_ink & ~left_ink] = (215, 55, 45)
        overlay[left_ink & right_ink] = (25, 25, 25)
        cell = Image.fromarray(overlay)
        column, row = index % columns, index // columns
        cell_left, cell_top = column * cell_width, row * cell_height + 24
        frame_left, frame_top = cell_left + 8, cell_top + 8
        sheet.paste(cell, (frame_left, frame_top))
        _draw_metric_guides(
            draw,
            left=frame_left,
            top=frame_top,
            width=frame_size[0],
            baseline_y=baseline_y,
            model=builds[0].model,
            scale=GLYPH_SCALE,
        )
        draw.text((cell_left + 5, cell_top + 1), character, font=font, fill=(20, 20, 20))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False, compress_level=9)


def _comparison(
    output: Path,
    canonical_dir: Path,
    candidate_dir: Path,
    builds: Sequence[PreviewBuild],
    glyph_characters: Sequence[str],
    frames: Mapping[tuple[int, str], Image.Image],
    frame_size: tuple[int, int],
    baseline_y: int,
) -> dict[str, Any]:
    comparison = output / "comparison"
    _save_comparison(
        canonical_dir / "01-alphabet.png",
        candidate_dir / "01-alphabet.png",
        comparison / "01-alphabet-side-by-side.png",
        left_label="Canonical",
        right_label="Latest candidate, unpromoted",
    )
    _write_glyph_difference(
        builds,
        glyph_characters,
        frames,
        frame_size,
        baseline_y,
        comparison / "02-modeled-glyph-difference.png",
    )
    _save_comparison(
        canonical_dir / "03-words.png",
        candidate_dir / "03-words.png",
        comparison / "03-words-side-by-side.png",
        left_label="Canonical",
        right_label="Latest candidate, unpromoted",
    )
    payload = {
        "format": "inkgrammar-example-comparison",
        "format_version": 1,
        "project_id": builds[0].model.id,
        "canonical_geometry": "canonical",
        "candidate_geometry": "latest-candidate",
        "candidate_unpromoted": True,
        "common_scale": True,
        "alphabet_cell_size": [112, 140],
        "glyph_scale": GLYPH_SCALE,
        "modeled_glyph_count": len(glyph_characters),
        "word_scale": WORD_SCALE,
        "outputs": {
            "alphabet": "01-alphabet-side-by-side.png",
            "modeled_glyph_difference": "02-modeled-glyph-difference.png",
            "words": "03-words-side-by-side.png",
        },
    }
    _save_json(comparison / "comparison.json", payload)
    return payload


def render_validation(
    bundle: FitProjectBundle,
    *,
    geometry: Literal["canonical", "latest-candidate", "both"],
    output: str | Path = DEFAULT_OUTPUT,
    include_digits: bool = False,
    cases_path: str | Path = DEFAULT_CASES,
) -> dict[str, Any]:
    output_root = Path(output).expanduser()
    if output_root.is_symlink():
        raise ValueError(f"example output directory must not be a symlink: {output_root}")
    cases = _load_cases(Path(cases_path).expanduser())
    characters = _target_characters(cases, include_digits)
    before_hashes = _tracked_hashes(bundle)

    geometries: tuple[PreviewGeometry, ...] = (
        ("canonical", "latest-candidate")
        if geometry == "both"
        else (geometry,)
    )
    builds = [build_fit_preview_model(bundle, geometry=item) for item in geometries]
    glyph_characters = sorted(set().union(*(set(build.model.glyphs) for build in builds)))
    frames, frame_size, baseline_y = _align_glyph_frames(builds, glyph_characters, GLYPH_SCALE)

    if geometry == "both":
        mode_directories = {
            "canonical": output_root / "canonical",
            "latest-candidate": output_root / "latest-candidate",
        }
    else:
        mode_directories = {geometries[0]: output_root}

    manifests = {}
    for index, build in enumerate(builds):
        mode = geometries[index]
        manifests[mode] = _write_mode(
            bundle,
            build,
            mode_directories[mode],
            characters,
            cases,
            frames=frames,
            frame_size=frame_size,
            baseline_y=baseline_y,
            build_index=index,
        )

    comparison_payload = None
    if geometry == "both":
        comparison_payload = _comparison(
            output_root,
            mode_directories["canonical"],
            mode_directories["latest-candidate"],
            builds,
            glyph_characters,
            frames,
            frame_size,
            baseline_y,
        )

    after_hashes = _tracked_hashes(bundle)
    if before_hashes != after_hashes:
        raise RuntimeError("example rendering changed tracked model or training inputs")
    return {"runs": manifests, "comparison": comparison_payload}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render source-free SAS-1968 model validation examples"
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--geometry",
        choices=("canonical", "latest-candidate", "both"),
        default="canonical",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--include-digits", action="store_true")
    args = parser.parse_args()

    bundle = load_fit_project(args.project)
    render_validation(
        bundle,
        geometry=args.geometry,
        output=args.output,
        include_digits=args.include_digits,
        cases_path=args.cases,
    )


if __name__ == "__main__":
    main()
