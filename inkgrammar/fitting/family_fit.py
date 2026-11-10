from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .. import __version__
from .checkpoint import (
    CheckpointCreator,
    CheckpointInputs,
    CheckpointOptimizer,
    FitCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from .models import FitSample, PrimitiveTemplate, TopologyPart, TopologyTransform
from .objective import LocalRegion, ObjectiveConfig, ObjectiveResult, evaluate_objective
from .optimizer import (
    checkpoint_from_result,
    optimize_parameters,
    resume_optimization,
)
from .parameterization import ParameterSpace
from .render import FittingRender, render_paths
from .workspace import FitProjectBundle


@dataclass(frozen=True)
class _PreparedSample:
    sample: FitSample
    reference: np.ndarray
    target_distance: np.ndarray
    anchor: tuple[float, float]
    canvas_size: tuple[int, int]
    parts: tuple[TopologyPart, ...]
    regions: tuple[tuple[str, np.ndarray, tuple[str, ...]], ...]


@dataclass(frozen=True)
class FitRunArtifacts:
    checkpoint_path: Path
    json_report_path: Path
    markdown_report_path: Path
    report: dict[str, Any]
    private_review_sheet_path: Path


def _number(value: float) -> str:
    rendered = f"{value:.7f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def primitive_path(template: PrimitiveTemplate, transform: TopologyTransform) -> str:
    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            transform.dx + transform.scale_x * (x + transform.shear_x * y),
            transform.dy + transform.scale_y * y,
        )

    start = map_point(template.frame.start)
    commands = [f"M {_number(start[0])} {_number(start[1])}"]
    for segment in template.segments:
        c1, c2, end = map(map_point, (segment.c1, segment.c2, segment.end))
        commands.append(
            "C "
            f"{_number(c1[0])} {_number(c1[1])} "
            f"{_number(c2[0])} {_number(c2[1])} "
            f"{_number(end[0])} {_number(end[1])}"
        )
    return " ".join(commands)


def _prepare_sample(
    bundle: FitProjectBundle,
    sample: FitSample,
    primitive_id: str,
) -> _PreparedSample:
    materialized = bundle.private_work_dir / "materialized"
    mask_path = materialized / "masks" / f"{sample.id}.png"
    distance_path = materialized / "distance-fields" / f"{sample.id}.npy"
    if not mask_path.is_file() or not distance_path.is_file():
        raise ValueError(
            f"missing private materialization for {sample.id}; run fit materialize first"
        )
    with Image.open(mask_path) as image:
        reference = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    target_distance = np.load(distance_path, allow_pickle=False).astype(np.float32)
    if reference.ndim != 2 or reference.shape != target_distance.shape:
        raise ValueError(f"invalid materialized mask pair: {sample.id}")

    glyph = bundle.topology.glyphs.get(sample.text)
    if glyph is None:
        raise ValueError(f"no topology recipe for fit sample {sample.id}")
    parts = tuple(
        part for stroke in glyph.strokes for part in stroke.parts if part.primitive == primitive_id
    )
    if not parts:
        raise ValueError(f"glyph {sample.text!r} has no instance of {primitive_id}")
    if len({part.instance for part in parts}) != len(parts):
        raise ValueError(f"duplicate primitive instance in glyph {sample.text!r}")

    anchor_record = bundle.anchors.samples.get(sample.id)
    if anchor_record is None:
        raise ValueError(f"missing reviewed anchor for fit sample {sample.id}")
    anchor = (anchor_record.origin_x, anchor_record.baseline_y)
    width, height = reference.shape[1], reference.shape[0]
    instances = {part.instance for part in parts}
    regions = []
    for region in sample.loss_regions:
        selected_instances = tuple(
            instance for instance in region.instances if instance in instances
        )
        if not selected_instances:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in region.bbox_source)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x0 >= x1 or y0 >= y1:
            raise ValueError(f"loss region is outside materialized crop: {sample.id}")
        mask = np.zeros(reference.shape, dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0
        if not np.any(reference * mask):
            raise ValueError(f"loss region contains no target ink: {sample.id}")
        regions.append((region.id, mask, selected_instances))
    if not regions:
        raise ValueError(f"missing component-local loss region for {sample.id}")
    return _PreparedSample(
        sample=sample,
        reference=reference,
        target_distance=target_distance,
        anchor=anchor,
        canvas_size=(width, height),
        parts=parts,
        regions=tuple(regions),
    )


def _render_sample(
    template: PrimitiveTemplate,
    prepared: _PreparedSample,
    bundle: FitProjectBundle,
) -> tuple[FittingRender, tuple[str, ...]]:
    paths = {part.instance: primitive_path(template, part.transform) for part in prepared.parts}
    rendered = render_paths(
        paths,
        canvas_size=prepared.canvas_size,
        origin=prepared.anchor,
        pixels_per_unit=1.0,
        stroke_width=bundle.fit_config.render.stroke_width_px,
        supersampling=bundle.fit_config.render.supersampling,
    )
    return rendered, tuple(paths.values())


def _primitive_objective_config(bundle: FitProjectBundle) -> ObjectiveConfig:
    weights = bundle.fit_config.objective.weights.model_copy(
        update={
            "soft_mask": 0.0,
            "distance": 0.0,
            "anchor": 0.0,
            "bbox": 0.0,
            "endpoint": 0.0,
            "tangent": 0.0,
            "instance_regularization": 0.0,
            "curvature_regularization": 0.01,
            "complexity": 0.0,
            "local_region": 1.0,
        }
    )
    return ObjectiveConfig(
        version=bundle.fit_config.objective.version,
        weights=weights,
        residual_sigma=bundle.fit_config.objective.residual_sigma,
    )


def _evaluate_sample(
    template: PrimitiveTemplate,
    prepared: _PreparedSample,
    bundle: FitProjectBundle,
    config: ObjectiveConfig,
) -> ObjectiveResult:
    rendered, paths = _render_sample(template, prepared, bundle)
    local_regions = []
    for _, mask, instance_ids in prepared.regions:
        component = np.zeros(prepared.reference.shape, dtype=np.float32)
        for instance_id in instance_ids:
            component = np.maximum(component, rendered.part_layers[instance_id])
        local_regions.append(LocalRegion(mask=mask, generated_part=component))
    return evaluate_objective(
        prepared.reference,
        rendered.coverage,
        config=config,
        x_height=max(bundle.grammar.ductus.x_height, 1.0),
        reference_anchor=prepared.anchor,
        generated_anchor=prepared.anchor,
        target_distance_field=prepared.target_distance,
        local_regions=local_regions,
        paths=paths,
        complexity_count=float(len(template.segments)),
    )


def _write_private_overlays(
    bundle: FitProjectBundle,
    primitive_id: str,
    template: PrimitiveTemplate,
    samples: list[_PreparedSample],
) -> Path:
    overlays_root = bundle.private_work_dir / "overlays"
    if overlays_root.is_symlink():
        raise ValueError("private overlay directory must not be a symlink")
    overlays_root.mkdir(parents=True, exist_ok=True)
    output_dir = overlays_root / primitive_id
    if output_dir.is_symlink():
        raise ValueError("private primitive overlay directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, Image.Image]] = []
    for prepared in samples:
        rendered, _ = _render_sample(template, prepared, bundle)
        for region_id, mask, instance_ids in prepared.regions:
            _safe_fit_id(prepared.sample.id)
            _safe_fit_id(region_id)
            ys, xs = np.nonzero(mask)
            top, bottom = int(ys.min()), int(ys.max()) + 1
            left, right = int(xs.min()), int(xs.max()) + 1
            component = np.zeros(prepared.reference.shape, dtype=np.float32)
            for instance_id in instance_ids:
                component = np.maximum(component, rendered.part_layers[instance_id])
            region_reference = prepared.reference[top:bottom, left:right]
            region_generated = component[top:bottom, left:right]
            region_mask = mask[top:bottom, left:right] > 0
            reference_ink = (region_reference > 0.05) & region_mask
            generated_ink = (region_generated > 0.05) & region_mask
            overlay = np.full((*region_reference.shape, 3), 255, dtype=np.uint8)
            overlay[reference_ink] = (220, 40, 40)
            overlay[generated_ink] = (40, 70, 220)
            overlay[reference_ink & generated_ink] = (30, 30, 30)
            label = f"{prepared.sample.id} | {region_id}"
            image = Image.fromarray(overlay)
            scale = 4
            enlarged = image.resize(
                (image.width * scale, image.height * scale), Image.Resampling.NEAREST
            )
            labeled = Image.new("RGB", (enlarged.width, enlarged.height + 20), "white")
            ImageDraw.Draw(labeled).text((2, 2), label, fill="black")
            labeled.paste(enlarged, (0, 20))
            output_path = output_dir / f"{prepared.sample.id}--{region_id}.png"
            if output_path.is_symlink():
                raise ValueError("private overlay output must not be a symlink")
            labeled.save(output_path, format="PNG", optimize=False, compress_level=9)
            images.append((label, labeled))

    columns, cell_width, cell_height = 3, 180, 120
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        column, row = index % columns, index // columns
        left, top = column * cell_width, row * cell_height
        draw.text((left + 4, top + 3), label, fill="black")
        thumbnail = image.copy()
        thumbnail.thumbnail((cell_width - 8, cell_height - 24), Image.Resampling.NEAREST)
        sheet.paste(
            thumbnail,
            (left + (cell_width - thumbnail.width) // 2, top + 20),
        )
    sheet_path = output_dir / "review-sheet.png"
    if sheet_path.is_symlink():
        raise ValueError("private overlay sheet must not be a symlink")
    sheet.save(sheet_path, format="PNG", optimize=False, compress_level=9)
    return sheet_path


def _aggregate(results: list[ObjectiveResult], samples: list[_PreparedSample]) -> float:
    total_weight = sum(sample.sample.weight for sample in samples)
    return (
        sum(
            result.total * sample.sample.weight
            for result, sample in zip(results, samples, strict=True)
        )
        / total_weight
    )


def _sample_diagnostic(
    prepared: _PreparedSample,
    result: ObjectiveResult,
    initial: ObjectiveResult | None = None,
) -> dict[str, Any]:
    diagnostic = {
        "sample_id": prepared.sample.id,
        "objective": result.total,
        "terms": result.terms,
        "instances": [
            {
                "instance_id": part.instance,
                "primitive_id": part.primitive,
                "transform": part.transform.model_dump(mode="json"),
                "residuals": {},
            }
            for part in prepared.parts
        ],
    }
    if initial is not None:
        diagnostic["initial_objective"] = initial.total
        diagnostic["final_objective"] = result.total
    return diagnostic


def _group_summary(results: list[ObjectiveResult]) -> dict[str, float | None]:
    if not results:
        return {"mean_objective": None, "worst_member_objective": None}
    values = [result.total for result in results]
    return {
        "mean_objective": float(np.mean(values)),
        "worst_member_objective": max(values),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_hashes(bundle: FitProjectBundle, training: list[FitSample]) -> CheckpointInputs:
    support_files = {
        "observation_manifest": bundle.project.observation_manifest,
        "families": bundle.project.families,
        "topology": bundle.project.topology,
        "anchors": bundle.project.anchors,
        "fit_config": bundle.project.fit_config,
        "seed_grammar": bundle.project.seed_grammar,
    }
    support = {
        name: _file_sha256((bundle.project_dir / relative).resolve(strict=True))
        for name, relative in support_files.items()
    }
    materialized = bundle.private_work_dir / "materialized"
    training_artifacts = []
    for sample in sorted(training, key=lambda item: item.id):
        for path in (
            materialized / "masks" / f"{sample.id}.png",
            materialized / "distance-fields" / f"{sample.id}.npy",
        ):
            if not path.is_file():
                raise ValueError(f"missing training materialization input: {sample.id}")
            training_artifacts.append(f"{sample.id}:{_file_sha256(path)}")
    support["training_materialization"] = hashlib.sha256(
        "\n".join(training_artifacts).encode("utf-8")
    ).hexdigest()
    return CheckpointInputs(
        project_sha256=_file_sha256(bundle.project_path),
        corpus_sha256=_file_sha256(
            (bundle.project_dir / bundle.project.corpus).resolve(strict=True)
        ),
        source_lock_sha256=_file_sha256(
            (bundle.project_dir / bundle.project.source_lock).resolve(strict=True)
        ),
        support_sha256=support,
    )


def refresh_primitive_artifact_hashes(
    bundle: FitProjectBundle, primitive_id: str
) -> CheckpointInputs:
    _safe_fit_id(primitive_id)
    family = next(
        (family for family in bundle.families.families if family.primitive == primitive_id),
        None,
    )
    if family is None:
        raise ValueError(f"unknown fitting family for primitive: {primitive_id}")
    training_ids = set(family.fit)
    training = [sample for sample in bundle.corpus.samples if sample.id in training_ids]
    if not training:
        raise ValueError(f"primitive family has no fit samples: {primitive_id}")

    inputs = _input_hashes(bundle, training)
    stage = f"primitive:{primitive_id}"
    checkpoint_path = (
        _project_output_directory(bundle.project_dir, "checkpoints") / f"{primitive_id}.json"
    )
    report_path = (
        _project_output_directory(bundle.project_dir, "reports") / f"{primitive_id}.json"
    )
    if checkpoint_path.is_symlink() or report_path.is_symlink():
        raise ValueError("fitting output files must not be symlinks")

    checkpoint = load_checkpoint(checkpoint_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if checkpoint.project_id != bundle.project.id or checkpoint.stage != stage:
        raise ValueError("checkpoint does not match fitting project and primitive")
    if report.get("project_id") != bundle.project.id or report.get("stage") != stage:
        raise ValueError("fit report does not match fitting project and primitive")

    save_checkpoint(checkpoint.model_copy(update={"inputs": inputs}), checkpoint_path)
    report["input_hashes"] = inputs.model_dump(mode="json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inputs


def _optimizer_backend(method: str) -> str:
    return "scipy-powell" if method == "powell" else "scipy-differential-evolution+powell"


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Fit report: {report['primitive_id']}",
        "",
        f"Project: `{report['project_id']}`",
        "",
        "## Training members",
        "",
        "| Sample | Initial objective | Final objective |",
        "| --- | ---: | ---: |",
    ]
    for sample in report["training"]["samples"]:
        lines.append(
            f"| `{sample['sample_id']}` | {sample['initial_objective']:.6f} "
            f"| {sample['final_objective']:.6f} |"
        )
    training = report["training"]
    lines.extend(
        [
            "",
            f"Training aggregate: {training['initial_objective']:.6f} → "
            f"{training['final_objective']:.6f}",
            f"Training mean: {training['mean_objective']:.6f}; "
            f"worst member: {training['worst_member_objective']:.6f}",
            "",
            "## Validation diagnostics",
            "",
            "Validation members are evaluated after fitting and do not update parameters.",
            "",
            "| Sample | Objective |",
            "| --- | ---: |",
        ]
    )
    for sample in report["validation"]["samples"]:
        lines.append(f"| `{sample['sample_id']}` | {sample['objective']:.6f} |")
    lines.extend(
        [
            "",
            "## Hold-out diagnostics",
            "",
            "Hold-out samples are reported after selection and do not affect parameters.",
            "",
            "| Sample | Objective |",
            "| --- | ---: |",
        ]
    )
    for sample in report["hold_out"]["samples"]:
        lines.append(f"| `{sample['sample_id']}` | {sample['objective']:.6f} |")
    gap = report["generalization_gap"]
    lines.extend(
        [
            "",
            f"Training to hold-out mean gap: {'n/a' if gap is None else f'{gap:.6f}'}",
            "Model promotion remains pending; inspect overlays in the private work directory.",
            "",
            "## Selected parameters",
            "",
        ]
    )
    lines.extend(f"- `{name}`: {value:.8f}" for name, value in sorted(report["parameters"].items()))
    return "\n".join(lines) + "\n"


def _safe_fit_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise ValueError(f"unsafe fitting target identifier: {value!r}")


def _project_output_directory(project_dir: Path, name: str) -> Path:
    if name not in {"checkpoints", "reports"}:
        raise ValueError(f"invalid fitting output directory: {name}")
    root = project_dir.resolve(strict=True)
    path = root / name
    if path.is_symlink():
        raise ValueError(f"fitting output directory must not be a symlink: {name}")
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    if resolved.parent != root:
        raise ValueError(f"fitting output directory must stay inside {root}")
    return resolved


def optimize_primitive_family(
    bundle: FitProjectBundle,
    primitive_id: str,
    *,
    resume: bool = False,
) -> FitRunArtifacts:
    _safe_fit_id(primitive_id)
    family = next(
        (family for family in bundle.families.families if family.primitive == primitive_id),
        None,
    )
    if family is None:
        raise ValueError(f"unknown fitting family for primitive: {primitive_id}")
    training = [sample for sample in bundle.corpus.samples if sample.id in set(family.fit)]
    hold_out = [sample for sample in bundle.corpus.samples if sample.id in set(family.hold_out)]
    validation = [sample for sample in bundle.corpus.samples if sample.id in set(family.validation)]
    if not training:
        raise ValueError(f"primitive family has no fit samples: {primitive_id}")

    template = bundle.grammar.primitives[primitive_id]
    parameter_space = ParameterSpace.from_primitive(template)
    training_prepared = [_prepare_sample(bundle, sample, primitive_id) for sample in training]
    objective_config = _primitive_objective_config(bundle)

    def evaluate(vector: np.ndarray) -> float:
        candidate = parameter_space.decode(vector, template)
        assert isinstance(candidate, PrimitiveTemplate)
        results = [
            _evaluate_sample(candidate, prepared, bundle, objective_config)
            for prepared in training_prepared
        ]
        return _aggregate(results, training_prepared)

    initial = parameter_space.encode(template)
    initial_results = [
        _evaluate_sample(template, prepared, bundle, objective_config)
        for prepared in training_prepared
    ]
    initial_objective = _aggregate(initial_results, training_prepared)
    inputs = _input_hashes(bundle, training)
    stage = f"primitive:{primitive_id}"
    checkpoint_dir = _project_output_directory(bundle.project_dir, "checkpoints")
    reports_dir = _project_output_directory(bundle.project_dir, "reports")
    checkpoint_path = checkpoint_dir / f"{primitive_id}.json"
    json_report_path = reports_dir / f"{primitive_id}.json"
    markdown_report_path = reports_dir / f"{primitive_id}.md"
    if any(path.is_symlink() for path in (checkpoint_path, json_report_path, markdown_report_path)):
        raise ValueError("fitting output files must not be symlinks")

    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if (
            checkpoint.project_id != bundle.project.id
            or checkpoint.stage != stage
            or checkpoint.inputs != inputs
            or checkpoint.optimizer.objective_version != objective_config.version
            or checkpoint.optimizer.backend
            != _optimizer_backend(bundle.fit_config.optimizer.method)
            or checkpoint.optimizer.seed != bundle.fit_config.optimizer.seed
        ):
            raise ValueError("checkpoint inputs or optimizer configuration do not match")
        result = resume_optimization(
            checkpoint,
            parameter_space,
            evaluate,
            config=bundle.fit_config.optimizer,
        )
    else:
        checkpoint = FitCheckpoint(
            project_id=bundle.project.id,
            stage=stage,
            created_by=CheckpointCreator(inkgrammar_version=__version__),
            inputs=inputs,
            optimizer=CheckpointOptimizer(
                backend=_optimizer_backend(bundle.fit_config.optimizer.method),
                seed=bundle.fit_config.optimizer.seed,
                objective_version=objective_config.version,
            ),
            parameters=dict(
                zip(
                    (spec.name for spec in parameter_space.specs),
                    initial.tolist(),
                    strict=True,
                )
            ),
        )
        result = optimize_parameters(
            parameter_space,
            evaluate,
            config=bundle.fit_config.optimizer,
        )

    selected = parameter_space.decode(result.parameters, template)
    assert isinstance(selected, PrimitiveTemplate)
    training_results = [
        _evaluate_sample(selected, prepared, bundle, objective_config)
        for prepared in training_prepared
    ]
    final_objective = _aggregate(training_results, training_prepared)
    validation_prepared = [_prepare_sample(bundle, sample, primitive_id) for sample in validation]
    validation_results = [
        _evaluate_sample(selected, prepared, bundle, objective_config)
        for prepared in validation_prepared
    ]

    # Hold-outs are loaded and evaluated only after the training vector is fixed.
    holdout_prepared = [_prepare_sample(bundle, sample, primitive_id) for sample in hold_out]
    holdout_results = [
        _evaluate_sample(selected, prepared, bundle, objective_config)
        for prepared in holdout_prepared
    ]
    holdout_metrics = {
        prepared.sample.id: result.total
        for prepared, result in zip(holdout_prepared, holdout_results, strict=True)
    }
    private_review_sheet_path = _write_private_overlays(
        bundle,
        primitive_id,
        selected,
        [*training_prepared, *validation_prepared, *holdout_prepared],
    )
    training_summary = _group_summary(training_results)
    validation_summary = _group_summary(validation_results)
    holdout_summary = _group_summary(holdout_results)
    initial_parameters = dict(
        zip(
            (spec.name for spec in parameter_space.specs),
            initial.tolist(),
            strict=True,
        )
    )
    training_metrics = {
        "initial_objective": initial_objective,
        "objective": final_objective,
        "mean_objective": training_summary["mean_objective"],
        "worst_member_objective": training_summary["worst_member_objective"],
        **{
            f"{prepared.sample.id}.initial": result.total
            for prepared, result in zip(training_prepared, initial_results, strict=True)
        },
        **{
            f"{prepared.sample.id}.final": result.total
            for prepared, result in zip(training_prepared, training_results, strict=True)
        },
    }
    training_mean = training_summary["mean_objective"]
    holdout_mean = holdout_summary["mean_objective"]
    generalization_gap = (
        holdout_mean - training_mean
        if holdout_mean is not None and training_mean is not None
        else None
    )
    checkpoint = checkpoint_from_result(
        checkpoint,
        result,
        config=bundle.fit_config.optimizer,
    ).model_copy(
        update={
            "parameters": dict(
                zip(result.parameter_names, result.parameters.tolist(), strict=True)
            ),
            "training_metrics": training_metrics,
            "holdout_metrics": holdout_metrics,
        }
    )
    save_checkpoint(checkpoint, checkpoint_path)

    report = {
        "format": "inkgrammar-fit-report",
        "format_version": 1,
        "project_id": bundle.project.id,
        "stage": stage,
        "primitive_id": primitive_id,
        "objective_version": objective_config.version,
        "optimizer": bundle.fit_config.optimizer.model_dump(mode="json"),
        "input_hashes": checkpoint.inputs.model_dump(mode="json"),
        "initial_parameters": initial_parameters,
        "parameters": checkpoint.parameters,
        "parameter_bounds": template.bounds,
        "primitive": selected.model_dump(mode="json"),
        "training": {
            "selection_split": "fit",
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            **training_summary,
            "samples": [
                _sample_diagnostic(prepared, final, initial)
                for prepared, initial, final in zip(
                    training_prepared, initial_results, training_results, strict=True
                )
            ],
        },
        "validation": {
            "policy": "evaluated after fitting; not used for parameter optimization",
            **validation_summary,
            "samples": [
                _sample_diagnostic(prepared, result)
                for prepared, result in zip(validation_prepared, validation_results, strict=True)
            ],
        },
        "hold_out": {
            "policy": "diagnostic-only; never used for parameter selection",
            **holdout_summary,
            "samples": [
                _sample_diagnostic(prepared, result)
                for prepared, result in zip(holdout_prepared, holdout_results, strict=True)
            ],
        },
        "generalization_gap": generalization_gap,
        "visual_review": {
            "status": "promotion_pending",
            "scope": "private work directory",
            "sheet": private_review_sheet_path.relative_to(bundle.private_work_dir).as_posix(),
            "region_overlay_count": sum(
                len(prepared.regions)
                for prepared in [
                    *training_prepared,
                    *validation_prepared,
                    *holdout_prepared,
                ]
            ),
        },
        "evaluations": result.evaluations,
    }
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_report_path.write_text(_report_markdown(report), encoding="utf-8")
    return FitRunArtifacts(
        checkpoint_path=checkpoint_path,
        json_report_path=json_report_path,
        markdown_report_path=markdown_report_path,
        report=report,
        private_review_sheet_path=private_review_sheet_path,
    )


def load_fit_reports(bundle: FitProjectBundle) -> list[dict[str, Any]]:
    reports_dir = bundle.project_dir / "reports"
    if reports_dir.is_symlink():
        raise ValueError("fit reports directory must not be a symlink")
    if not reports_dir.is_dir():
        return []
    reports = []
    for path in sorted(reports_dir.glob("*.json")):
        if path.is_symlink():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "inkgrammar-fit-report":
            continue
        primitive_id = payload.get("primitive_id")
        if not isinstance(primitive_id, str):
            continue
        try:
            _safe_fit_id(primitive_id)
        except ValueError:
            continue
        if path.stem != primitive_id:
            continue
        reports.append(payload)
    return reports
