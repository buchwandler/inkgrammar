from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ..geometry import analyze_path, translate_path
from .alignment import path_points_in_image
from .family_fit import _prepare_sample, _PreparedSample, primitive_path
from .models import (
    FamilyAlignmentConfig,
    OccurrenceDelta,
    OccurrenceFitState,
    PrimitiveTemplate,
    TopologyGlyph,
    TopologyPart,
    compose_occurrence_transform,
)
from .objective import ObjectiveConfig
from .optimizer import _least_squares_cost, optimize_parameters
from .parameterization import ParameterSpace, ParameterSpec
from .render import FittingRender, render_paths
from .residuals import occurrence_geometric_residuals, transform_prior_residuals
from .workspace import FitProjectBundle


@dataclass(frozen=True)
class GlyphFitArtifacts:
    report_path: Path
    overlay_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class _GlyphOccurrence:
    key: str
    prepared: _PreparedSample
    part: TopologyPart
    state: OccurrenceFitState
    reference_mask: np.ndarray
    distance_field: np.ndarray


def _primitive_templates(
    bundle: FitProjectBundle,
) -> tuple[dict[str, PrimitiveTemplate], dict[str, str]]:
    templates = dict(bundle.grammar.primitives)
    sources = {primitive_id: "seed-grammar" for primitive_id in templates}
    reports_dir = bundle.project_dir / "reports"
    for primitive_id in templates:
        path = reports_dir / f"{primitive_id}.json"
        if path.is_symlink():
            raise ValueError(f"primitive fit report must not be a symlink: {primitive_id}")
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("format_version") != 2 or report.get("objective_version") != 2:
            continue
        templates[primitive_id] = PrimitiveTemplate.model_validate(report["primitive"])
        sources[primitive_id] = "objective-v2-family-report"
    return templates, sources


def _family_alignment(bundle: FitProjectBundle, primitive_id: str) -> FamilyAlignmentConfig:
    family = next(
        (item for item in bundle.families.families if item.primitive == primitive_id),
        None,
    )
    return family.alignment if family else FamilyAlignmentConfig()


def _family_delta(
    bundle: FitProjectBundle,
    sample_id: str,
    part: TopologyPart,
) -> OccurrenceDelta:
    report_path = bundle.project_dir / "reports" / f"{part.primitive}.json"
    if report_path.is_symlink() or not report_path.is_file():
        return OccurrenceDelta()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format_version") != 2 or report.get("objective_version") != 2:
        return OccurrenceDelta()
    occurrence = report.get("occurrences", {}).get(f"{sample_id}--{part.instance}")
    if occurrence is None:
        return OccurrenceDelta()
    return OccurrenceDelta.model_validate(occurrence["delta"])


def _occurrence_state(
    bundle: FitProjectBundle,
    sample_id: str,
    part: TopologyPart,
) -> OccurrenceFitState:
    family_alignment = _family_alignment(bundle, part.primitive)
    override = family_alignment.overrides.get(part.instance)
    free = override.free if override and override.free is not None else family_alignment.free
    bounds = (
        override.bounds if override and override.bounds is not None else family_alignment.bounds
    )
    delta = _family_delta(bundle, sample_id, part)
    if part.attach:
        free = [name for name in free if name not in {"dx", "dy"}]
        delta = delta.model_copy(update={"dx": 0.0, "dy": 0.0})
    return OccurrenceFitState(
        instance_id=f"{sample_id}--{part.instance}",
        nominal_transform=part.transform,
        bounds=bounds,
        free_parameters=free,
        delta=delta,
    )


def _composed_paths(
    topology: TopologyGlyph,
    templates: dict[str, PrimitiveTemplate],
    deltas: dict[str, OccurrenceDelta],
    active_instances: set[str] | None = None,
) -> dict[str, str]:
    paths = {}
    for stroke in topology.strokes:
        current_end = None
        for index, part in enumerate(stroke.parts):
            try:
                template = templates[part.primitive]
            except KeyError as exc:
                raise ValueError(f"missing shared primitive geometry: {part.primitive}") from exc
            transform = compose_occurrence_transform(
                part.transform, deltas.get(part.instance, OccurrenceDelta())
            )
            path = primitive_path(template, transform)
            if index and part.attach:
                if current_end is None:
                    raise ValueError(
                        f"attached topology part has no preceding endpoint: {part.instance}"
                    )
                start = analyze_path(path).start
                path = translate_path(path, current_end[0] - start[0], current_end[1] - start[1])
            if active_instances is None or part.instance in active_instances:
                paths[part.instance] = path
            current_end = analyze_path(path).end
    return paths


def _prepare_occurrences(
    bundle: FitProjectBundle,
    sample_id: str,
    parts: tuple[TopologyPart, ...],
    prepared: _PreparedSample,
) -> list[_GlyphOccurrence]:
    occurrences = []
    for part in parts:
        region = next(
            (mask for _, mask, instance_ids in prepared.regions if part.instance in instance_ids),
            None,
        )
        if region is None:
            raise ValueError(
                f"active instance {part.instance} has no assigned loss region in {sample_id}"
            )
        reference_mask = ((prepared.reference > 0.01) & (region > 0)).astype(np.float32)
        if not reference_mask.any():
            raise ValueError(f"assigned loss region contains no ink for {sample_id}")
        distance_field = distance_transform_edt(~reference_mask.astype(bool)).astype(np.float32)
        state = _occurrence_state(bundle, sample_id, part)
        occurrences.append(
            _GlyphOccurrence(
                key=state.instance_id,
                prepared=prepared,
                part=part,
                state=state,
                reference_mask=reference_mask,
                distance_field=distance_field,
            )
        )
    return occurrences


def _states_from_vector(
    vector: np.ndarray,
    occurrences: list[_GlyphOccurrence],
    parameter_names: dict[str, tuple[str, str]],
) -> dict[str, OccurrenceDelta]:
    values = dict(zip(parameter_names, vector.tolist(), strict=True))
    deltas = {}
    for occurrence in occurrences:
        delta_data = occurrence.state.delta.model_dump(mode="python")
        for name in occurrence.state.free_parameters:
            delta_data[name] = values[f"{occurrence.key}.{name}"]
        deltas[occurrence.part.instance] = OccurrenceDelta.model_validate(delta_data)
    return deltas


def _all_family_deltas(
    bundle: FitProjectBundle,
    sample_id: str,
    topology: TopologyGlyph,
) -> dict[str, OccurrenceDelta]:
    return {
        part.instance: _family_delta(bundle, sample_id, part)
        for stroke in topology.strokes
        for part in stroke.parts
    }


def _glyph_residuals(
    vector: np.ndarray,
    occurrences: list[_GlyphOccurrence],
    topology: TopologyGlyph,
    templates: dict[str, PrimitiveTemplate],
    bundle: FitProjectBundle,
    config: ObjectiveConfig,
    parameter_names: dict[str, tuple[str, str]],
    fixed_deltas: dict[str, OccurrenceDelta],
) -> np.ndarray:
    deltas = dict(fixed_deltas)
    deltas.update(_states_from_vector(vector, occurrences, parameter_names))
    active_instances = {item.part.instance for item in occurrences}
    paths = _composed_paths(topology, templates, deltas, active_instances)
    blocks = []
    free_count = sum(len(item.state.free_parameters) for item in occurrences)
    for occurrence in occurrences:
        points = path_points_in_image(
            paths[occurrence.part.instance], occurrence.prepared.anchor, sample_count=32
        )
        geometry = occurrence_geometric_residuals(
            points,
            occurrence.reference_mask,
            occurrence.distance_field,
            expected_stroke_width=bundle.fit_config.render.stroke_width_px,
            path_weight=config.geometry.path_to_ink,
            center_weight=config.geometry.normal_center,
            endpoint_weight=config.geometry.endpoint,
            tangent_weight=config.geometry.tangent,
        )
        blocks.append(geometry.residuals / sqrt(len(occurrences) * len(geometry.residuals)))
        if config.priors.transform > 0 and occurrence.state.free_parameters:
            all_priors = transform_prior_residuals(deltas[occurrence.part.instance])
            prior_by_name = dict(
                zip(
                    ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x"),
                    all_priors.tolist(),
                    strict=True,
                )
            )
            selected = np.asarray(
                [prior_by_name[name] for name in occurrence.state.free_parameters], dtype=float
            )
            blocks.append(sqrt(2.0 * config.priors.transform / free_count) * selected)
    return np.concatenate(blocks) if blocks else np.empty(0)


def _overlay(
    prepared,
    rendered: FittingRender,
    output_path: Path,
) -> None:
    reference_ink = prepared.reference > 0.05
    generated_ink = rendered.coverage > 0.05
    overlay = np.full((*prepared.reference.shape, 3), 255, dtype=np.uint8)
    overlay[reference_ink] = (220, 40, 40)
    overlay[generated_ink] = (40, 70, 220)
    overlay[reference_ink & generated_ink] = (30, 30, 30)
    Image.fromarray(overlay).save(output_path, format="PNG", optimize=False, compress_level=9)


def fit_compositional_glyphs(
    bundle: FitProjectBundle,
    *,
    sample_ids: set[str] | None = None,
) -> list[GlyphFitArtifacts]:
    if bundle.fit_config.objective.version != 2:
        raise ValueError("compositional glyph fitting requires objective version 2")
    templates, template_sources = _primitive_templates(bundle)
    known_samples = {sample.id: sample for sample in bundle.corpus.samples}
    selected = [
        sample
        for sample in bundle.corpus.samples
        if sample.split == "fit"
        and sample.kind == "glyph"
        and sample.active_instances
        and (sample_ids is None or sample.id in sample_ids)
    ]
    if sample_ids is not None:
        unknown = sample_ids - known_samples.keys()
        if unknown:
            raise ValueError(f"unknown compositional glyph samples: {sorted(unknown)}")
        invalid = [sample_id for sample_id in sample_ids if known_samples[sample_id].split != "fit"]
        if invalid:
            raise ValueError(
                f"only fit-split glyphs may update occurrence transforms: {sorted(invalid)}"
            )
        missing = sample_ids - {sample.id for sample in selected}
        if missing:
            raise ValueError(
                f"selected glyphs require active instances and glyph topology: {sorted(missing)}"
            )

    output_dir = bundle.private_work_dir / "glyph-fits"
    if output_dir.is_symlink():
        raise ValueError("private glyph-fit directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    objective = bundle.fit_config.objective
    optimizer = bundle.fit_config.optimizer.model_copy(update={"method": "least_squares"})

    for sample in selected:
        topology = bundle.topology.glyphs.get(sample.text)
        if topology is None:
            raise ValueError(f"no topology recipe for compositional glyph {sample.id}")
        all_parts = tuple(part for stroke in topology.strokes for part in stroke.parts)
        fixed_deltas = _all_family_deltas(bundle, sample.id, topology)
        all_instance_ids = {part.instance for part in all_parts}
        active_ids = set(sample.active_instances)
        if not active_ids <= all_instance_ids:
            raise ValueError(f"active instances are not in glyph topology: {sample.id}")
        active_parts = tuple(part for part in all_parts if part.instance in active_ids)
        if len(active_parts) != len(active_ids):
            raise ValueError(f"active instance IDs must be unique in topology: {sample.id}")
        prepared = _prepare_sample(
            bundle,
            sample,
            None,
            active_instances=active_ids,
        )
        occurrences = _prepare_occurrences(bundle, sample.id, active_parts, prepared)
        specs = []
        parameter_names: dict[str, tuple[str, str]] = {}
        for occurrence in occurrences:
            for name in occurrence.state.free_parameters:
                lower, upper = getattr(occurrence.state.bounds, name)
                parameter_name = f"{occurrence.key}.{name}"
                parameter_names[parameter_name] = (occurrence.key, name)
                specs.append(
                    ParameterSpec(
                        name=parameter_name,
                        lower=lower,
                        upper=upper,
                        initial=getattr(occurrence.state.delta, name),
                        scale=upper - lower,
                    )
                )
        space = ParameterSpace(specs)
        initial_vector = np.asarray([spec.initial for spec in space.specs], dtype=float)
        initial_residuals = _glyph_residuals(
            initial_vector,
            occurrences,
            topology,
            templates,
            bundle,
            objective,
            parameter_names,
            fixed_deltas,
        )
        initial_score = _least_squares_cost(initial_residuals, optimizer.loss, optimizer.f_scale)

        def residuals(
            vector: np.ndarray,
            occurrences: list[_GlyphOccurrence] = occurrences,
            topology: TopologyGlyph = topology,
            templates: dict[str, PrimitiveTemplate] = templates,
            bundle: FitProjectBundle = bundle,
            objective: ObjectiveConfig = objective,
            parameter_names: dict[str, tuple[str, str]] = parameter_names,
            fixed_deltas: dict[str, OccurrenceDelta] = fixed_deltas,
        ) -> np.ndarray:
            return _glyph_residuals(
                vector,
                occurrences,
                topology,
                templates,
                bundle,
                objective,
                parameter_names,
                fixed_deltas,
            )

        result = optimize_parameters(
            space,
            residuals,
            initial=initial_vector,
            config=optimizer,
        )
        final_deltas = dict(fixed_deltas)
        final_deltas.update(_states_from_vector(result.parameters, occurrences, parameter_names))
        composed_paths = _composed_paths(topology, templates, final_deltas, active_ids)
        rendered = render_paths(
            composed_paths,
            canvas_size=prepared.canvas_size,
            origin=prepared.anchor,
            pixels_per_unit=1.0,
            stroke_width=bundle.fit_config.render.stroke_width_px,
            supersampling=bundle.fit_config.render.supersampling,
        )
        overlay_path = output_dir / f"{sample.id}--composition.png"
        report_path = output_dir / f"{sample.id}.json"
        if overlay_path.is_symlink() or report_path.is_symlink():
            raise ValueError("private glyph-fit outputs must not be symlinks")
        _overlay(prepared, rendered, overlay_path)
        occurrence_report = {}
        for occurrence in occurrences:
            delta = final_deltas[occurrence.part.instance]
            occurrence_report[occurrence.part.instance] = {
                "primitive_id": occurrence.part.primitive,
                "free_parameters": occurrence.state.free_parameters,
                "nominal_transform": occurrence.part.transform.model_dump(mode="json"),
                "delta": delta.model_dump(mode="json"),
                "effective_transform": compose_occurrence_transform(
                    occurrence.part.transform, delta
                ).model_dump(mode="json"),
            }
        report = {
            "format": "inkgrammar-compositional-glyph-fit-report",
            "format_version": 1,
            "objective_version": objective.version,
            "sample_id": sample.id,
            "character": sample.text,
            "split": sample.split,
            "selection_split": "fit",
            "active_instances": sorted(active_ids),
            "composed_instances": list(composed_paths),
            "shared_primitives": {
                primitive_id: {
                    "source": template_sources[primitive_id],
                    "parameters": templates[primitive_id].model_dump(mode="json"),
                }
                for primitive_id in sorted({part.primitive for part in all_parts})
            },
            "shared_geometry_updated": False,
            "training_objective": {
                "initial": initial_score,
                "final": result.objective,
            },
            "optimizer": {
                "backend": result.backend,
                "success": result.success,
                "status": result.status,
                "message": result.message,
                "evaluations": result.evaluations,
                "accepted_history_index": result.accepted_history_index,
            },
            "occurrences": occurrence_report,
            "overlay": overlay_path.name,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(GlyphFitArtifacts(report_path, overlay_path, report))
    return artifacts
