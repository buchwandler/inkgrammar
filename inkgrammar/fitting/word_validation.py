from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..geometry import analyze_path, transform_path
from ..models import Transform
from .glyph_fit import _composed_paths, _primitive_templates
from .models import OccurrenceDelta
from .objective import soft_mask_loss, symmetric_distance_loss
from .render import render_paths
from .workspace import FitProjectBundle


@dataclass(frozen=True)
class WordValidationArtifacts:
    report_path: Path
    overlay_path: Path
    report: dict[str, Any]


def _glyph_fit_sources(bundle: FitProjectBundle) -> dict[str, tuple[str, dict[str, Any]]]:
    training_samples = {
        sample.id: sample
        for sample in bundle.corpus.samples
        if sample.kind == "glyph" and sample.split == "fit"
    }
    reports_dir = bundle.private_work_dir / "glyph-fits"
    if reports_dir.is_symlink():
        raise ValueError("private glyph-fit reports directory must not be a symlink")
    if not reports_dir.is_dir():
        return {}
    sources: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        if path.is_symlink():
            raise ValueError(f"private glyph-fit report must not be a symlink: {path.name}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("format") != "inkgrammar-compositional-glyph-fit-report"
            or report.get("format_version") != 1
            or report.get("split") != "fit"
            or report.get("selection_split") != "fit"
        ):
            continue
        character = report.get("character")
        sample_id = report.get("sample_id")
        training_sample = training_samples.get(sample_id)
        if (
            isinstance(character, str)
            and len(character) == 1
            and training_sample is not None
            and training_sample.text == character
        ):
            sources.setdefault(character, (sample_id, report["occurrences"]))
    return sources


def _active_word_instances(sample) -> dict[int, set[str]]:
    active: dict[int, set[str]] = {}
    for value in sample.active_instances:
        try:
            index_text, instance = value.split(":", 1)
            index = int(index_text)
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"word active instances must use 'character-index:instance-id': {sample.id}"
            ) from exc
        if index < 0 or index >= len(sample.text) or not instance:
            raise ValueError(f"invalid word active instance {value!r}: {sample.id}")
        active.setdefault(index, set()).add(instance)
    if not active:
        raise ValueError(f"word sample requires active instances: {sample.id}")
    return active


def _attachment_checks(topology, paths: dict[str, str], glyph_index: int) -> list[dict[str, Any]]:
    checks = []
    for stroke in topology.strokes:
        for index, part in enumerate(stroke.parts):
            if not index or not part.attach:
                continue
            previous = stroke.parts[index - 1]
            previous_end = analyze_path(paths[previous.instance]).end
            current_start = analyze_path(paths[part.instance]).start
            gap = hypot(previous_end[0] - current_start[0], previous_end[1] - current_start[1])
            checks.append(
                {
                    "kind": "within-glyph-attachment",
                    "glyph_index": glyph_index,
                    "from_instance": previous.instance,
                    "to_instance": part.instance,
                    "gap": gap,
                    "status": "pass" if gap <= 1e-6 else "fail",
                }
            )
    return checks


def _word_paths(bundle: FitProjectBundle, sample, templates, glyph_sources):
    active = _active_word_instances(sample)
    paths: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    source_ids: dict[str, str] = {}
    pen_x = 0.0
    for index, character in enumerate(sample.text):
        topology = bundle.topology.glyphs.get(character)
        if topology is None:
            raise ValueError(f"no topology recipe for word character {character!r}: {sample.id}")
        all_parts = tuple(part for stroke in topology.strokes for part in stroke.parts)
        instance_ids = {part.instance for part in all_parts}
        if len(instance_ids) != len(all_parts):
            raise ValueError(f"duplicate topology instance in word character {character!r}")
        active_ids = active.get(index, set())
        if not active_ids <= instance_ids:
            raise ValueError(f"unknown active word instances at character {index}: {sample.id}")
        fitted = glyph_sources.get(character)
        deltas = {}
        if fitted is not None:
            source_ids[character] = fitted[0]
            deltas = {
                instance_id: OccurrenceDelta.model_validate(item["delta"])
                for instance_id, item in fitted[1].items()
            }
        else:
            source_ids[character] = "nominal-topology"
        local_paths = _composed_paths(topology, templates, deltas)
        checks.extend(_attachment_checks(topology, local_paths, index))
        for instance_id in sorted(active_ids):
            paths[f"{index}:{instance_id}"] = transform_path(
                local_paths[instance_id], Transform(dx=pen_x)
            )
        pen_x += topology.advance_width
    for index in range(len(sample.text) - 1):
        checks.append(
            {
                "kind": "glyph-boundary",
                "left_glyph_index": index,
                "right_glyph_index": index + 1,
                "status": "not-specified",
                "reason": "topology has no contextual glyph-boundary connection rule",
            }
        )
    if not paths:
        raise ValueError(f"word sample has no composed active paths: {sample.id}")
    return paths, checks, source_ids


def _word_coverage(sample, paths: dict[str, str], connection_checks: list[dict[str, Any]]) -> dict:
    """Coverage fields per the brief's section 66.

    A word composed from available glyphs without fitted boundary rules is
    ``renderable_unjoined``; it is never reported as connected-writing
    validation until every required boundary has a fitted/reviewed rule.
    """
    glyph_coverage = all(
        any(key.startswith(f"{index}:") for key in paths)
        for index in range(len(sample.text))
    )
    connection_coverage = bool(connection_checks) and all(
        check["status"] == "pass" for check in connection_checks
    )
    connected_writing = glyph_coverage and connection_coverage
    return {
        "glyph_coverage": glyph_coverage,
        "connection_coverage": connection_coverage,
        "connected_writing": connected_writing,
        "status": "connected-writing-validation" if connected_writing else "renderable_unjoined",
    }


def validate_composed_words(
    bundle: FitProjectBundle,
    *,
    sample_ids: set[str] | None = None,
) -> list[WordValidationArtifacts]:
    selected = [
        sample
        for sample in bundle.corpus.samples
        if sample.kind == "word" and (sample_ids is None or sample.id in sample_ids)
    ]
    if sample_ids is not None:
        known = {sample.id: sample for sample in bundle.corpus.samples}
        missing = sample_ids - {sample.id for sample in selected}
        if missing:
            wrong_kind = [sample_id for sample_id in missing if sample_id in known]
            if wrong_kind:
                raise ValueError(f"selected samples are not word samples: {sorted(wrong_kind)}")
            raise ValueError(f"unknown word samples: {sorted(missing)}")

    templates, template_sources = _primitive_templates(bundle)
    glyph_sources = _glyph_fit_sources(bundle)
    output_dir = bundle.private_work_dir / "word-validation"
    if output_dir.is_symlink():
        raise ValueError("private word-validation output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for sample in selected:
        mask_path = bundle.private_work_dir / "materialized" / "masks" / f"{sample.id}.png"
        distance_path = (
            bundle.private_work_dir / "materialized" / "distance-fields" / f"{sample.id}.npy"
        )
        if not mask_path.is_file() or not distance_path.is_file():
            raise ValueError(f"missing private materialization for word sample {sample.id}")
        with Image.open(mask_path) as image:
            reference = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        target_distance = np.load(distance_path, allow_pickle=False).astype(np.float32)
        if reference.ndim != 2 or reference.shape != target_distance.shape:
            raise ValueError(f"invalid materialized mask pair: {sample.id}")
        anchor_record = bundle.anchors.samples.get(sample.id)
        if anchor_record is None:
            raise ValueError(f"missing reviewed anchor for word sample {sample.id}")
        paths, connection_checks, occurrence_sources = _word_paths(
            bundle, sample, templates, glyph_sources
        )
        rendered = render_paths(
            paths,
            canvas_size=(reference.shape[1], reference.shape[0]),
            origin=(anchor_record.origin_x, anchor_record.baseline_y),
            pixels_per_unit=1.0,
            stroke_width=bundle.fit_config.render.stroke_width_px,
            supersampling=bundle.fit_config.render.supersampling,
        )
        overlay_path = output_dir / f"{sample.id}--composition.png"
        report_path = output_dir / f"{sample.id}.json"
        if overlay_path.is_symlink() or report_path.is_symlink():
            raise ValueError("private word-validation outputs must not be symlinks")
        overlay = np.zeros((*reference.shape, 3), dtype=np.uint8)
        overlay[..., 0] = np.clip(reference * 255, 0, 255).astype(np.uint8)
        overlay[..., 1] = np.clip(rendered.coverage * 255, 0, 255).astype(np.uint8)
        Image.fromarray(overlay, mode="RGB").save(
            overlay_path, format="PNG", optimize=False, compress_level=9
        )
        report = {
            "format": "inkgrammar-word-validation-report",
            "format_version": 1,
            "sample_id": sample.id,
            "text": sample.text,
            "split": sample.split,
            "selection_split": None,
            "parameter_updates": False,
            "active_instances": sorted(sample.active_instances),
            "occurrence_fit_sources": occurrence_sources,
            "shared_primitive_sources": template_sources,
            "objective_version": bundle.fit_config.objective.version,
            "metrics": {
                "soft_mask_loss": soft_mask_loss(reference, rendered.coverage),
                "symmetric_distance_loss": symmetric_distance_loss(
                    reference, rendered.coverage, target_distance_field=target_distance
                ),
            },
            **_word_coverage(sample, paths, connection_checks),
            "connection_checks": connection_checks,
            "overlay": overlay_path.name,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(WordValidationArtifacts(report_path, overlay_path, report))
    return artifacts
