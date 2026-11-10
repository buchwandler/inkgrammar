from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import ceil, log2, sqrt
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import qmc

from .. import __version__
from .alignment import path_points_in_image
from .checkpoint import (
    CheckpointCreator,
    CheckpointIteration,
    CheckpointOptimizer,
    FitCheckpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_objective_version,
)
from .family_fit import (
    FitRunArtifacts,
    _input_hashes,
    _prepare_sample,
    _PreparedSample,
    _project_output_directory,
    _report_markdown,
    _safe_fit_id,
    _write_private_overlays,
    primitive_path,
)
from .models import (
    FamilyDefinition,
    OccurrenceDelta,
    OccurrenceFitState,
    PrimitiveTemplate,
    TopologyPart,
    compose_occurrence_transform,
)
from .objective import ObjectiveConfig
from .optimizer import (
    OptimizationResult,
    OptimizerConfig,
    _least_squares_cost,
    optimize_parameters,
)
from .parameterization import CompositeParameterSpace, ParameterSpace, ParameterSpec
from .residuals import (
    GeometricResiduals,
    occurrence_geometric_residuals,
    transform_prior_residuals,
)
from .workspace import FitProjectBundle


@dataclass(frozen=True)
class _PreparedOccurrence:
    key: str
    prepared: _PreparedSample
    part: TopologyPart
    state: OccurrenceFitState
    reference_mask: np.ndarray
    distance_field: np.ndarray


def _alignment_state(
    family: FamilyDefinition,
    prepared: _PreparedSample,
    part: TopologyPart,
) -> OccurrenceFitState:
    override = family.alignment.overrides.get(part.instance)
    free = override.free if override and override.free is not None else family.alignment.free
    bounds = (
        override.bounds if override and override.bounds is not None else family.alignment.bounds
    )
    key = f"{prepared.sample.id}--{part.instance}"
    return OccurrenceFitState(
        instance_id=key,
        nominal_transform=part.transform,
        bounds=bounds,
        free_parameters=free,
    )


def _prepare_occurrences(
    bundle: FitProjectBundle,
    family: FamilyDefinition,
    samples: list[_PreparedSample],
) -> list[_PreparedOccurrence]:
    occurrences = []
    for prepared in samples:
        for part in prepared.parts:
            region = next(
                (
                    mask
                    for _, mask, instance_ids in prepared.regions
                    if part.instance in instance_ids
                ),
                None,
            )
            if region is None:
                raise ValueError(
                    f"missing supervision gate for instance {part.instance} in {prepared.sample.id}"
                )
            reference_mask = ((prepared.reference > 0.01) & (region > 0)).astype(np.float32)
            if not reference_mask.any():
                raise ValueError(f"supervision gate contains no ink for {prepared.sample.id}")
            distance_field = distance_transform_edt(~reference_mask.astype(bool)).astype(np.float32)
            state = _alignment_state(family, prepared, part)
            _safe_fit_id(state.instance_id)
            occurrences.append(
                _PreparedOccurrence(
                    key=state.instance_id,
                    prepared=prepared,
                    part=part,
                    state=state,
                    reference_mask=reference_mask,
                    distance_field=distance_field,
                )
            )
    if len({item.key for item in occurrences}) != len(occurrences):
        raise ValueError("occurrence IDs must be unique across fitting samples")
    return occurrences


def _geometry(
    template: PrimitiveTemplate,
    occurrence: _PreparedOccurrence,
    bundle: FitProjectBundle,
    config: ObjectiveConfig,
    delta: OccurrenceDelta,
) -> GeometricResiduals:
    transform = compose_occurrence_transform(occurrence.part.transform, delta)
    path = primitive_path(template, transform)
    points = path_points_in_image(path, occurrence.prepared.anchor, sample_count=32)
    return occurrence_geometric_residuals(
        points,
        occurrence.reference_mask,
        occurrence.distance_field,
        expected_stroke_width=bundle.fit_config.render.stroke_width_px,
        path_weight=config.geometry.path_to_ink,
        center_weight=config.geometry.normal_center,
        endpoint_weight=config.geometry.endpoint,
        tangent_weight=config.geometry.tangent,
    )


def _group_occurrences(
    occurrences: list[_PreparedOccurrence],
) -> list[tuple[str, list[_PreparedOccurrence]]]:
    groups: dict[str, list[_PreparedOccurrence]] = {}
    for occurrence in occurrences:
        groups.setdefault(occurrence.prepared.sample.id, []).append(occurrence)
    return list(groups.items())


def _arc_length_curvature(paths: list[str]) -> float:
    from .alignment import sample_path_by_arclength

    values = []
    for path in paths:
        points = sample_path_by_arclength(path, sample_count=64)
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 1e-9
        vectors = vectors[valid] / lengths[valid, None]
        if len(vectors) < 2:
            continue
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        changes = (np.diff(angles) + np.pi) % (2 * np.pi) - np.pi
        values.append(float(np.mean((changes / np.pi) ** 2)))
    return float(np.mean(values)) if values else 0.0


def _family_residual_vector(
    template: PrimitiveTemplate,
    occurrences: list[_PreparedOccurrence],
    bundle: FitProjectBundle,
    config: ObjectiveConfig,
    shape_space: ParameterSpace,
    shape_reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    grouped = _group_occurrences(occurrences)
    total_weight = sum(items[0].prepared.sample.weight for _, items in grouped)
    blocks: list[np.ndarray] = []
    sample_losses: dict[str, float] = {}
    geometries: dict[str, GeometricResiduals] = {}
    paths = []
    for sample_id, items in grouped:
        sample_weight = items[0].prepared.sample.weight / total_weight
        losses = []
        for occurrence in items:
            geometry = _geometry(template, occurrence, bundle, config, occurrence.state.delta)
            geometries[occurrence.key] = geometry
            paths.append(
                primitive_path(
                    template,
                    compose_occurrence_transform(occurrence.part.transform, occurrence.state.delta),
                )
            )
            values = geometry.residuals
            losses.append(0.5 * float(np.mean(values**2)))
            scale = sqrt(sample_weight / (len(items) * len(values)))
            blocks.append(scale * values)
        sample_losses[sample_id] = float(np.mean(losses))

    transform_prior_cost = 0.0
    transform_values = []
    free_count = sum(len(item.state.free_parameters) for item in occurrences)
    if config.priors.transform > 0 and free_count:
        for occurrence in occurrences:
            all_priors = transform_prior_residuals(occurrence.state.delta)
            prior_by_name = dict(
                zip(
                    ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x"),
                    all_priors.tolist(),
                    strict=True,
                )
            )
            values = np.asarray(
                [prior_by_name[name] for name in occurrence.state.free_parameters], dtype=float
            )
            transform_values.extend(values.tolist())
        scale = sqrt(2.0 * config.priors.transform / free_count)
        prior_block = scale * np.asarray(transform_values, dtype=float)
        blocks.append(prior_block)
        transform_prior_cost = 0.5 * float(np.dot(prior_block, prior_block))

    shape_prior_cost = 0.0
    shape_values = shape_space.encode(template)
    if config.priors.shape > 0 and len(shape_values):
        scales = np.asarray([spec.upper - spec.lower for spec in shape_space.specs])
        shape_delta = (shape_values - shape_reference) / scales
        shape_block = sqrt(2.0 * config.priors.shape / len(shape_values)) * shape_delta
        blocks.append(shape_block)
        shape_prior_cost = 0.5 * float(np.dot(shape_block, shape_block))

    curvature_cost = 0.0
    if config.priors.curvature > 0 and paths:
        curvature = _arc_length_curvature(paths)
        curvature_block = np.asarray([sqrt(2.0 * config.priors.curvature * curvature)])
        blocks.append(curvature_block)
        curvature_cost = 0.5 * float(np.dot(curvature_block, curvature_block))

    losses = np.asarray(list(sample_losses.values()), dtype=float)
    weights = [items[0].prepared.sample.weight for _, items in grouped]
    mean_loss = float(np.average(losses, weights=weights))
    temperature = 0.1
    maximum = float(np.max(losses))
    exponentials = np.exp((losses - maximum) / temperature)
    smooth_worst = maximum + temperature * float(np.log(exponentials.sum()))
    spread = float(np.std(losses))
    worst_cost = config.family.worst * smooth_worst
    spread_cost = config.family.spread * spread
    if worst_cost > 0:
        blocks.append(np.asarray([sqrt(2.0 * worst_cost)]))
    if spread_cost > 0:
        blocks.append(np.asarray([sqrt(2.0 * spread_cost)]))
    total = (
        mean_loss
        + transform_prior_cost
        + shape_prior_cost
        + curvature_cost
        + worst_cost
        + spread_cost
    )
    diagnostics = {
        "mean": mean_loss,
        "median": float(np.median(losses)),
        "worst": maximum,
        "std": spread,
        "smooth_worst": smooth_worst,
        "transform_prior": transform_prior_cost,
        "shape_prior": shape_prior_cost,
        "curvature": curvature_cost,
        "family_worst": worst_cost,
        "family_spread": spread_cost,
        "total": total,
        "sample_losses": sample_losses,
        "geometries": geometries,
    }
    return np.concatenate(blocks) if blocks else np.empty(0), diagnostics


def _fit_occurrence(
    template: PrimitiveTemplate,
    occurrence: _PreparedOccurrence,
    bundle: FitProjectBundle,
    objective: ObjectiveConfig,
    optimizer: OptimizerConfig,
) -> tuple[_PreparedOccurrence, OptimizationResult | None]:
    if not occurrence.state.free_parameters:
        return occurrence, None
    specs = []
    for name in occurrence.state.free_parameters:
        lower, upper = getattr(occurrence.state.bounds, name)
        specs.append(
            ParameterSpec(
                name=name,
                lower=lower,
                upper=upper,
                initial=getattr(occurrence.state.delta, name),
                scale=upper - lower,
            )
        )
    space = ParameterSpace(specs)

    def residuals(vector: np.ndarray) -> np.ndarray:
        delta_data = occurrence.state.delta.model_dump(mode="python")
        delta_data.update(dict(zip(occurrence.state.free_parameters, vector.tolist(), strict=True)))
        delta = OccurrenceDelta.model_validate(delta_data)
        geometry = _geometry(template, occurrence, bundle, objective, delta)
        priors = transform_prior_residuals(delta)
        prior_by_name = dict(
            zip(
                ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x"),
                priors.tolist(),
                strict=True,
            )
        )
        prior_block = np.asarray(
            [prior_by_name[name] for name in occurrence.state.free_parameters], dtype=float
        )
        prior_scale = sqrt(2.0 * objective.priors.transform / max(len(prior_block), 1))
        return np.concatenate((geometry.residuals, prior_scale * prior_block))

    result = optimize_parameters(
        space,
        residuals,
        config=optimizer.model_copy(update={"method": "least_squares"}),
    )
    delta_data = occurrence.state.delta.model_dump(mode="python")
    delta_data.update(
        dict(
            zip(
                occurrence.state.free_parameters,
                result.parameters.tolist(),
                strict=True,
            )
        )
    )
    fitted_state = OccurrenceFitState.model_validate(
        occurrence.state.model_copy(
            update={"delta": OccurrenceDelta.model_validate(delta_data)}
        ).model_dump()
    )
    return replace(occurrence, state=fitted_state), result


def _fit_shared_shape(
    template: PrimitiveTemplate,
    occurrences: list[_PreparedOccurrence],
    bundle: FitProjectBundle,
    objective: ObjectiveConfig,
    optimizer: OptimizerConfig,
    shape_reference: np.ndarray,
) -> tuple[PrimitiveTemplate, OptimizationResult | None]:
    space = ParameterSpace.from_primitive(template)
    if not space.specs:
        return template, None

    def residuals(vector: np.ndarray) -> np.ndarray:
        candidate = space.decode(vector, template)
        assert isinstance(candidate, PrimitiveTemplate)
        return _family_residual_vector(
            candidate, occurrences, bundle, objective, space, shape_reference
        )[0]

    result = optimize_parameters(
        space,
        residuals,
        config=optimizer.model_copy(update={"method": "least_squares"}),
    )
    fitted = space.decode(result.parameters, template)
    assert isinstance(fitted, PrimitiveTemplate)
    return fitted, result


def _restart_templates(
    seed: PrimitiveTemplate,
    space: ParameterSpace,
    restarts: int,
    optimizer: OptimizerConfig,
) -> list[PrimitiveTemplate]:
    if restarts <= 1 or not space.specs:
        return [seed]
    power = ceil(log2(restarts - 1))
    sampler = qmc.Sobol(d=len(space.specs), scramble=True, seed=optimizer.seed)
    points = sampler.random_base2(power)[: restarts - 1]
    lower = np.asarray([spec.lower for spec in space.specs])
    upper = np.asarray([spec.upper for spec in space.specs])
    starts = [seed]
    for point in points:
        candidate = space.decode(lower + point * (upper - lower), seed)
        assert isinstance(candidate, PrimitiveTemplate)
        starts.append(candidate)
    return starts


def _saturation(value: float, bounds: tuple[float, float], threshold: float) -> bool:
    lower, upper = bounds
    margin_fraction = min(value - lower, upper - value) / (upper - lower)
    return margin_fraction <= threshold


def _sample_diagnostics(
    template: PrimitiveTemplate,
    occurrences: list[_PreparedOccurrence],
    bundle: FitProjectBundle,
    objective: ObjectiveConfig,
    *,
    initial_objectives: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped = _group_occurrences(occurrences)
    samples = []
    scores = {}
    for sample_id, items in grouped:
        instance_reports = []
        instance_scores = []
        term_totals = {
            "path_to_ink": 0.0,
            "normal_center": 0.0,
            "endpoint": 0.0,
            "tangent": 0.0,
        }
        for occurrence in items:
            delta = occurrence.state.delta
            geometry = _geometry(template, occurrence, bundle, objective, delta)
            instance_loss = 0.5 * float(np.mean(geometry.residuals**2))
            instance_scores.append(instance_loss)
            term_totals["path_to_ink"] += 0.5 * float(np.mean(geometry.path_to_ink**2))
            term_totals["normal_center"] += 0.5 * float(np.mean(geometry.centerline**2))
            term_totals["endpoint"] += 0.5 * float(np.mean(geometry.endpoint**2))
            term_totals["tangent"] += 0.5 * float(np.mean(geometry.tangent**2))
            nominal = occurrence.part.transform
            effective = compose_occurrence_transform(nominal, delta)
            saturated = {
                name: _saturation(
                    getattr(delta, name),
                    getattr(occurrence.state.bounds, name),
                    0.05,
                )
                for name in occurrence.state.free_parameters
            }
            instance_reports.append(
                {
                    "instance_id": occurrence.part.instance,
                    "primitive_id": occurrence.part.primitive,
                    "nominal_transform": nominal.model_dump(mode="json"),
                    "delta": delta.model_dump(mode="json"),
                    "effective_transform": effective.model_dump(mode="json"),
                    "bound_saturation": saturated,
                    "residuals": geometry.diagnostics,
                }
            )
        score = float(np.mean(instance_scores))
        scores[sample_id] = score
        sample_report = {
            "sample_id": sample_id,
            "objective": score,
            "raw_terms": {
                "path_to_ink_rms_px": float(
                    np.mean([item["residuals"]["path_to_ink_rms_px"] for item in instance_reports])
                ),
                "normal_center_rms_px": float(
                    np.mean([item["residuals"]["centerline_rms_px"] for item in instance_reports])
                ),
                "endpoint_rms_px": float(
                    np.mean([item["residuals"]["endpoint_rms_px"] for item in instance_reports])
                ),
                "tangent_rms": float(
                    np.mean([item["residuals"]["tangent_rms"] for item in instance_reports])
                ),
            },
            "weighted_terms": {
                name: term_totals[name] / len(items)
                for name in ("path_to_ink", "normal_center", "endpoint", "tangent")
            },
            "instances": instance_reports,
        }
        if initial_objectives and sample_id in initial_objectives:
            sample_report["initial_objective"] = initial_objectives[sample_id]
            sample_report["final_objective"] = score
        samples.append(sample_report)
    return samples, scores


def _summary(scores: dict[str, float]) -> dict[str, float | None]:
    values = np.asarray(list(scores.values()), dtype=float)
    if not len(values):
        return {
            "mean_objective": None,
            "median_objective": None,
            "worst_member_objective": None,
            "std_objective": None,
        }
    return {
        "mean_objective": float(np.mean(values)),
        "median_objective": float(np.median(values)),
        "worst_member_objective": float(np.max(values)),
        "std_objective": float(np.std(values)),
    }


def _fit_diagnostic_alignment(
    template: PrimitiveTemplate,
    occurrences: list[_PreparedOccurrence],
    bundle: FitProjectBundle,
    objective: ObjectiveConfig,
    optimizer: OptimizerConfig,
) -> list[_PreparedOccurrence]:
    aligned = []
    for occurrence in occurrences:
        fitted, _ = _fit_occurrence(template, occurrence, bundle, objective, optimizer)
        aligned.append(fitted)
    return aligned


def _report_split(
    template: PrimitiveTemplate,
    occurrences: list[_PreparedOccurrence],
    bundle: FitProjectBundle,
    objective: ObjectiveConfig,
    optimizer: OptimizerConfig,
) -> tuple[dict[str, Any], list[_PreparedOccurrence]]:
    if not occurrences:
        return (
            {
                "policy": "no samples in this split",
                **_summary({}),
                "raw_nominal_mean_objective": None,
                "samples": [],
            },
            [],
        )
    nominal_samples, nominal_scores = _sample_diagnostics(template, occurrences, bundle, objective)
    aligned_occurrences = _fit_diagnostic_alignment(
        template, occurrences, bundle, objective, optimizer
    )
    aligned_samples, aligned_scores = _sample_diagnostics(
        template, aligned_occurrences, bundle, objective
    )
    aligned_by_id = {item["sample_id"]: item for item in aligned_samples}
    for item in nominal_samples:
        aligned = aligned_by_id[item["sample_id"]]
        item["raw_nominal_objective"] = item["objective"]
        item["aligned_diagnostic_objective"] = aligned["objective"]
        item["objective"] = aligned["objective"]
        item["aligned_instances"] = aligned["instances"]
    report = {
        "policy": (
            "Evaluated after fitting. Alignment deltas are diagnostic only and do not "
            "update shared geometry."
        ),
        **_summary(aligned_scores),
        "raw_nominal_mean_objective": _summary(nominal_scores)["mean_objective"],
        "samples": nominal_samples,
    }
    return report, aligned_occurrences


def _candidate_score(
    template: PrimitiveTemplate,
    occurrences: list[_PreparedOccurrence],
    bundle: FitProjectBundle,
    objective: ObjectiveConfig,
    shape_space: ParameterSpace,
    shape_reference: np.ndarray,
    optimizer: OptimizerConfig,
) -> tuple[float, dict[str, Any]]:
    residuals, diagnostics = _family_residual_vector(
        template, occurrences, bundle, objective, shape_space, shape_reference
    )
    return _least_squares_cost(residuals, optimizer.loss, optimizer.f_scale), diagnostics


def optimize_primitive_family_v2(
    bundle: FitProjectBundle,
    primitive_id: str,
    *,
    resume: bool = False,
) -> FitRunArtifacts:
    _safe_fit_id(primitive_id)
    family = next(
        (item for item in bundle.families.families if item.primitive == primitive_id),
        None,
    )
    if family is None:
        raise ValueError(f"unknown fitting family for primitive: {primitive_id}")
    training = [sample for sample in bundle.corpus.samples if sample.id in set(family.fit)]
    validation = [sample for sample in bundle.corpus.samples if sample.id in set(family.validation)]
    hold_out = [sample for sample in bundle.corpus.samples if sample.id in set(family.hold_out)]
    if not training:
        raise ValueError(f"primitive family has no fit samples: {primitive_id}")

    template = bundle.grammar.primitives[primitive_id]
    objective = bundle.fit_config.objective
    optimizer = bundle.fit_config.optimizer
    atomic_optimizer = optimizer.model_copy(update={"method": "least_squares"})
    training_prepared = [_prepare_sample(bundle, sample, primitive_id) for sample in training]
    occurrences = _prepare_occurrences(bundle, family, training_prepared)
    shape_space = ParameterSpace.from_primitive(template)
    shape_reference = shape_space.encode(template)
    initial_occurrences = occurrences
    initial_vector, _ = _family_residual_vector(
        template, initial_occurrences, bundle, objective, shape_space, shape_reference
    )
    initial_score = _least_squares_cost(initial_vector, optimizer.loss, optimizer.f_scale)
    initial_samples, _ = _sample_diagnostics(template, initial_occurrences, bundle, objective)
    initial_objectives = {item["sample_id"]: item["objective"] for item in initial_samples}

    stage = f"primitive:{primitive_id}"
    checkpoint_path = (
        _project_output_directory(bundle.project_dir, "checkpoints") / f"{primitive_id}.json"
    )
    reports_dir = _project_output_directory(bundle.project_dir, "reports")
    json_report_path = reports_dir / f"{primitive_id}.json"
    markdown_report_path = reports_dir / f"{primitive_id}.md"
    if any(path.is_symlink() for path in (checkpoint_path, json_report_path, markdown_report_path)):
        raise ValueError("fitting output files must not be symlinks")
    inputs = _input_hashes(bundle, training)
    occurrence_space = CompositeParameterSpace.from_primitive(
        template, [item.state for item in occurrences]
    )
    start_template = template
    start_occurrences = occurrences
    prior_history: list[CheckpointIteration] = []
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_objective_version(checkpoint, objective.version)
        if (
            checkpoint.project_id != bundle.project.id
            or checkpoint.stage != stage
            or checkpoint.inputs != inputs
            or checkpoint.optimizer.backend != "inkgrammar-alternating-least-squares"
            or checkpoint.optimizer.seed != optimizer.seed
        ):
            raise ValueError("checkpoint inputs or optimizer configuration do not match")
        if set(checkpoint.parameters) != {spec.name for spec in occurrence_space.specs}:
            raise ValueError("v2 checkpoint parameters do not match composite parameter space")
        start_vector = np.asarray(
            [checkpoint.parameters[spec.name] for spec in occurrence_space.specs], dtype=float
        )
        start_template, states = occurrence_space.decode(start_vector, template)
        state_by_id = {state.instance_id: state for state in states}
        start_occurrences = [replace(item, state=state_by_id[item.key]) for item in occurrences]
        prior_history = list(checkpoint.history)

    selected_template = start_template
    selected_occurrences = start_occurrences
    selected_score, selected_details = _candidate_score(
        selected_template,
        selected_occurrences,
        bundle,
        objective,
        shape_space,
        shape_reference,
        optimizer,
    )
    candidate_count = 1
    solver_history: list[CheckpointIteration] = list(prior_history)
    joint_result: OptimizationResult | None = None
    joint_evaluations = 0
    restart_starts = _restart_templates(start_template, shape_space, optimizer.restarts, optimizer)

    for restart_template in restart_starts:
        working_template = restart_template
        working_occurrences = start_occurrences
        previous_score = float("inf")
        for _ in range(optimizer.max_cycles):
            for occurrence in working_occurrences:
                fitted, _ = _fit_occurrence(
                    working_template, occurrence, bundle, objective, atomic_optimizer
                )
                working_occurrences = [
                    fitted if item.key == occurrence.key else item for item in working_occurrences
                ]
            working_template, _ = _fit_shared_shape(
                working_template,
                working_occurrences,
                bundle,
                objective,
                atomic_optimizer,
                shape_reference,
            )
            candidate_score, candidate_details = _candidate_score(
                working_template,
                working_occurrences,
                bundle,
                objective,
                shape_space,
                shape_reference,
                optimizer,
            )
            candidate_count += 1
            if candidate_score < selected_score:
                selected_template = working_template
                selected_occurrences = working_occurrences
                selected_score = candidate_score
                selected_details = candidate_details
            relative_change = abs(previous_score - candidate_score) / max(
                abs(previous_score), 1e-12
            )
            if relative_change <= optimizer.convergence_relative:
                break
            previous_score = candidate_score

        if optimizer.joint_refine:
            composite = CompositeParameterSpace.from_primitive(
                working_template, [item.state for item in working_occurrences]
            )
            joint_start = composite.encode(
                working_template, [item.state for item in working_occurrences]
            )

            states_snapshot = tuple(item.state for item in working_occurrences)
            occurrences_snapshot = tuple(working_occurrences)
            template_snapshot = working_template

            def joint_residuals(
                vector: np.ndarray,
                composite: CompositeParameterSpace = composite,
                template: PrimitiveTemplate = template_snapshot,
                occurrences: tuple[_PreparedOccurrence, ...] = occurrences_snapshot,
                states: tuple[OccurrenceFitState, ...] = states_snapshot,
            ) -> np.ndarray:
                candidate_template, candidate_states = composite.decode(vector, template, states)
                state_by_id = {state.instance_id: state for state in candidate_states}
                candidate_occurrences = [
                    replace(item, state=state_by_id[item.key]) for item in occurrences
                ]
                return _family_residual_vector(
                    candidate_template,
                    candidate_occurrences,
                    bundle,
                    objective,
                    shape_space,
                    shape_reference,
                )[0]

            joint_result = optimize_parameters(
                composite,
                joint_residuals,
                initial=joint_start,
                config=atomic_optimizer,
                prior_history=solver_history,
            )
            solver_history = list(joint_result.history)
            joint_evaluations += joint_result.evaluations
            joint_template, joint_states = composite.decode(
                joint_result.parameters,
                working_template,
                [item.state for item in working_occurrences],
            )
            joint_state_by_id = {state.instance_id: state for state in joint_states}
            joint_occurrences = [
                replace(item, state=joint_state_by_id[item.key]) for item in working_occurrences
            ]
            joint_score, joint_details = _candidate_score(
                joint_template,
                joint_occurrences,
                bundle,
                objective,
                shape_space,
                shape_reference,
                optimizer,
            )
            candidate_count += 1
            if joint_score < selected_score:
                selected_template = joint_template
                selected_occurrences = joint_occurrences
                selected_score = joint_score
                selected_details = joint_details

    final_samples, final_member_scores = _sample_diagnostics(
        selected_template,
        selected_occurrences,
        bundle,
        objective,
        initial_objectives=initial_objectives,
    )
    final_residuals, final_details = _family_residual_vector(
        selected_template,
        selected_occurrences,
        bundle,
        objective,
        shape_space,
        shape_reference,
    )
    selected_score = _least_squares_cost(final_residuals, optimizer.loss, optimizer.f_scale)
    selected_details = dict(final_details)
    selected_details["squared_residual_total"] = selected_details["total"]
    selected_details["total"] = selected_score
    validation_prepared = [_prepare_sample(bundle, sample, primitive_id) for sample in validation]
    validation_occurrences = _prepare_occurrences(bundle, family, validation_prepared)
    holdout_prepared = [_prepare_sample(bundle, sample, primitive_id) for sample in hold_out]
    holdout_occurrences = _prepare_occurrences(bundle, family, holdout_prepared)
    validation_report, validation_aligned = _report_split(
        selected_template,
        validation_occurrences,
        bundle,
        objective,
        atomic_optimizer,
    )
    holdout_report, holdout_aligned = _report_split(
        selected_template,
        holdout_occurrences,
        bundle,
        objective,
        atomic_optimizer,
    )
    deltas_by_sample: dict[str, dict[str, OccurrenceDelta]] = {}
    for occurrence in [*selected_occurrences, *validation_aligned, *holdout_aligned]:
        deltas_by_sample.setdefault(occurrence.prepared.sample.id, {})[occurrence.part.instance] = (
            occurrence.state.delta
        )
    private_review_sheet_path = _write_private_overlays(
        bundle,
        primitive_id,
        selected_template,
        [*training_prepared, *validation_prepared, *holdout_prepared],
        occurrence_deltas=deltas_by_sample,
        include_nominal=True,
    )

    input_parameters = occurrence_space.encode(
        start_template, [item.state for item in start_occurrences]
    )
    initial_parameters = dict(
        zip((spec.name for spec in occurrence_space.specs), input_parameters.tolist(), strict=True)
    )
    selected_parameters = occurrence_space.encode(
        selected_template, [item.state for item in selected_occurrences]
    )
    selected_parameter_map = dict(
        zip(
            (spec.name for spec in occurrence_space.specs),
            selected_parameters.tolist(),
            strict=True,
        )
    )
    member_summary = _summary(final_member_scores)
    training_summary = {
        "selection_split": "fit",
        "initial_objective": initial_score,
        "final_objective": selected_score,
        "mean_objective": member_summary["mean_objective"],
        "median_objective": member_summary["median_objective"],
        "worst_member_objective": member_summary["worst_member_objective"],
        "std_objective": member_summary["std_objective"],
        "samples": final_samples,
    }
    previous_report = json_report_path
    comparison_v1 = None
    if previous_report.is_file():
        old_report = json.loads(previous_report.read_text(encoding="utf-8"))
        if old_report.get("format_version") == 1:
            comparison_v1 = {
                "training": old_report.get("training", {}),
                "hold_out": old_report.get("hold_out", {}),
            }
        else:
            comparison_v1 = old_report.get("comparison_v1")

    saturation_warnings = []
    for occurrence in selected_occurrences:
        for name in occurrence.state.free_parameters:
            if _saturation(
                getattr(occurrence.state.delta, name),
                getattr(occurrence.state.bounds, name),
                0.05,
            ):
                saturation_warnings.append(
                    f"{occurrence.key}.{name} is within 5% of an occurrence bound"
                )
    for spec, value in zip(shape_space.specs, shape_space.encode(selected_template), strict=True):
        if _saturation(value, (spec.lower, spec.upper), 0.02):
            saturation_warnings.append(f"{spec.name} is within 2% of a shared-shape bound")

    if checkpoint is None:
        checkpoint = FitCheckpoint(
            project_id=bundle.project.id,
            stage=stage,
            created_by=CheckpointCreator(inkgrammar_version=__version__),
            inputs=inputs,
            optimizer=CheckpointOptimizer(
                backend="inkgrammar-alternating-least-squares",
                seed=optimizer.seed,
                objective_version=objective.version,
            ),
            parameters=initial_parameters,
        )
    checkpoint_update = checkpoint.model_copy(
        update={
            "inputs": inputs,
            "optimizer": checkpoint.optimizer.model_copy(
                update={
                    "backend": "inkgrammar-alternating-least-squares",
                    "seed": optimizer.seed,
                    "objective_version": objective.version,
                }
            ),
            "parameters": selected_parameter_map,
            "training_metrics": {
                "objective": selected_score,
                "initial_objective": initial_score,
                "mean_objective": float(member_summary["mean_objective"] or 0.0),
            },
            "history": solver_history,
        }
    )
    save_checkpoint(checkpoint_update, checkpoint_path)

    report = {
        "format": "inkgrammar-fit-report",
        "format_version": 2,
        "project_id": bundle.project.id,
        "stage": stage,
        "primitive_id": primitive_id,
        "objective_version": objective.version,
        "selection_objective": {
            "total": selected_details["total"],
            "blocks": {
                key: value
                for key, value in selected_details.items()
                if key not in {"sample_losses", "geometries"}
            },
        },
        "optimizer": {
            "backend": "scipy-least-squares-trf",
            "strategy": "alternating-least-squares",
            "success": joint_result.success if joint_result else True,
            "status": joint_result.status if joint_result else 0,
            "message": joint_result.message if joint_result else "No joint refinement requested.",
            "evaluations": joint_evaluations,
            "accepted_history_index": joint_result.accepted_history_index if joint_result else None,
            "solver_candidate": (
                dict(
                    zip(
                        joint_result.parameter_names,
                        joint_result.solver_parameters.tolist(),
                        strict=True,
                    )
                )
                if joint_result
                else selected_parameter_map
            ),
            "accepted_candidate": selected_parameter_map,
            "candidate_count": candidate_count,
            "restarts": len(restart_starts),
            "config": optimizer.model_dump(mode="json"),
        },
        "input_hashes": inputs.model_dump(mode="json"),
        "initial_parameters": initial_parameters,
        "parameters": selected_parameter_map,
        "shared_parameters": {
            spec.name: float(value)
            for spec, value in zip(
                shape_space.specs,
                shape_space.encode(selected_template),
                strict=True,
            )
        },
        "parameter_bounds": selected_template.bounds,
        "primitive": selected_template.model_dump(mode="json"),
        "occurrences": {
            occurrence.key: {
                "sample_id": occurrence.prepared.sample.id,
                "instance_id": occurrence.part.instance,
                "nominal_transform": occurrence.part.transform.model_dump(mode="json"),
                "delta": occurrence.state.delta.model_dump(mode="json"),
                "effective_transform": compose_occurrence_transform(
                    occurrence.part.transform, occurrence.state.delta
                ).model_dump(mode="json"),
            }
            for occurrence in selected_occurrences
        },
        "training": training_summary,
        "validation": validation_report,
        "hold_out": holdout_report,
        "generalization_gap": (
            holdout_report["mean_objective"] - member_summary["mean_objective"]
            if holdout_report["mean_objective"] is not None
            and member_summary["mean_objective"] is not None
            else None
        ),
        "bound_saturation_warnings": saturation_warnings,
        "comparison_v1": comparison_v1,
        "visual_review": {
            "status": "promotion_pending",
            "scope": "private work directory",
            "variants": ["nominal", "aligned"],
            "sheet": private_review_sheet_path.relative_to(bundle.private_work_dir).as_posix(),
            "region_overlay_count": sum(
                len(sample.regions)
                for sample in training_prepared + validation_prepared + holdout_prepared
            ),
        },
        "evaluations": joint_evaluations,
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
