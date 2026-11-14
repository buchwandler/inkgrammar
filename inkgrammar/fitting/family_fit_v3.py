"""Frozen objective-v3 centerline pipeline retained for reproducibility.

Do not add features or new consumers, and do not migrate its skeleton behavior
to SkeletonPaths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import ceil, log2, sqrt
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import qmc

from .. import __version__
from .alignment import bilinear_sample, path_points_in_image
from .assignment import ComponentAssignment, assign_ink_to_components
from .centerline import TargetTrace, prune_skeleton, skeletonize_component, trace_ordered_centerline
from .checkpoint import (
    CheckpointCreator,
    CheckpointIteration,
    CheckpointOptimizer,
    FitCheckpoint,
    load_checkpoint,
    save_checkpoint,
    summarize_trace,
    validate_checkpoint_objective_version,
)
from .family_fit import (
    FitRunArtifacts,
    _input_hashes,
    _prepare_sample,
    _project_output_directory,
    _report_markdown,
    _safe_fit_id,
    _write_private_overlays,
    primitive_path,
)
from .models import (
    FamilyDefinition,
    FitSample,
    OccurrenceDelta,
    OccurrenceFitState,
    PrimitiveTemplate,
    TopologyPart,
    compose_occurrence_transform,
)
from .objective import ObjectiveConfig
from .optimizer import OptimizationResult, OptimizerConfig, _least_squares_cost, optimize_parameters
from .parameterization import CompositeParameterSpace, ParameterSpace
from .render import render_paths
from .residuals import (
    OrderedCenterlineResiduals,
    ordered_centerline_residuals,
    transform_prior_residuals,
)
from .workspace import FitProjectBundle


@dataclass(frozen=True)
class _FitOccurrence:
    key: str
    prepared: Any
    part: TopologyPart
    all_parts: tuple[TopologyPart, ...]
    state: OccurrenceFitState


@dataclass(frozen=True)
class _AssignedOccurrence:
    occurrence: _FitOccurrence
    assignment: ComponentAssignment | None
    target_mask: np.ndarray
    target_trace: TargetTrace | None
    assignment_entropy: float | None


@dataclass(frozen=True)
class _StageResult:
    template: PrimitiveTemplate
    states: dict[str, OccurrenceFitState]
    score: float
    residuals: np.ndarray
    geometry: dict[str, OrderedCenterlineResiduals]
    sample_objectives: dict[str, float]
    jacobian_nnz: int
    optimization: OptimizationResult | None


def _all_parts(bundle: FitProjectBundle, sample: FitSample) -> tuple[TopologyPart, ...]:
    glyph = bundle.topology.glyphs.get(sample.text)
    if glyph is None:
        raise ValueError(f"no topology recipe for fit sample {sample.id}")
    active = set(sample.active_instances) if sample.active_instances else None
    return tuple(
        part
        for stroke in glyph.strokes
        for part in stroke.parts
        if active is None or part.instance in active
    )


def _fit_state(
    family: FamilyDefinition,
    sample: FitSample,
    part: TopologyPart,
) -> OccurrenceFitState:
    override = family.alignment.overrides.get(part.instance)
    free = override.free if override and override.free is not None else family.alignment.free
    bounds = (
        override.bounds if override and override.bounds is not None else family.alignment.bounds
    )
    return OccurrenceFitState(
        instance_id=f"{sample.id}--{part.instance}",
        nominal_transform=part.transform,
        bounds=bounds,
        free_parameters=free,
    )


def _prepare_occurrences(
    bundle: FitProjectBundle,
    family: FamilyDefinition,
    samples: list[FitSample],
) -> list[_FitOccurrence]:
    occurrences = []
    for sample in samples:
        prepared = _prepare_sample(bundle, sample, family.primitive)
        all_parts = _all_parts(bundle, sample)
        for part in prepared.parts:
            occurrences.append(
                _FitOccurrence(
                    key=f"{sample.id}--{part.instance}",
                    prepared=prepared,
                    part=part,
                    all_parts=all_parts,
                    state=_fit_state(family, sample, part),
                )
            )
    if len({item.key for item in occurrences}) != len(occurrences):
        raise ValueError("occurrence IDs must be unique across fitting samples")
    return occurrences


def _render_components(
    bundle: FitProjectBundle,
    primitive_id: str,
    template: PrimitiveTemplate,
    occurrences: list[_FitOccurrence],
    states: dict[str, OccurrenceFitState],
) -> dict[str, dict[str, np.ndarray]]:
    by_sample: dict[str, list[_FitOccurrence]] = {}
    for occurrence in occurrences:
        by_sample.setdefault(occurrence.prepared.sample.id, []).append(occurrence)
    output = {}
    for sample_id, items in by_sample.items():
        prepared = items[0].prepared
        all_parts = items[0].all_parts
        paths = {}
        for part in all_parts:
            if part.primitive == primitive_id:
                key = f"{sample_id}--{part.instance}"
                state = states.get(key)
                delta = state.delta if state is not None else OccurrenceDelta()
                part_template = template
            else:
                delta = OccurrenceDelta()
                part_template = bundle.grammar.primitives[part.primitive]
            transform = compose_occurrence_transform(part.transform, delta)
            paths[part.instance] = primitive_path(part_template, transform)
        rendered = render_paths(
            paths,
            canvas_size=prepared.canvas_size,
            origin=prepared.anchor,
            pixels_per_unit=1.0,
            stroke_width=bundle.fit_config.render.stroke_width_px,
            supersampling=bundle.fit_config.render.supersampling,
        )
        component_centerlines = {}
        for part in all_parts:
            layer = rendered.part_layers[part.instance] > 0.05
            skeleton, radius = skeletonize_component(layer)
            skeleton = prune_skeleton(
                skeleton,
                radius,
                min_spur_length_px=bundle.fit_config.objective.centerline.min_spur_length_px,
                min_confidence=bundle.fit_config.objective.centerline.min_confidence,
            )
            if np.any(skeleton):
                component_centerlines[part.instance] = skeleton
        output[sample_id] = component_centerlines
    return output


def _assignment_entropy(assignment: ComponentAssignment, observed: np.ndarray) -> float:
    probabilities = [
        assignment.target_weights,
        *assignment.competing_weights.values(),
        assignment.outlier_weights,
    ]
    values = np.stack(probabilities, axis=0)[:, observed]
    entropy = -np.sum(
        np.where(values > 0, values * np.log(np.maximum(values, 1e-12)), 0.0),
        axis=0,
    )
    return float(np.mean(entropy)) if entropy.size else 0.0


def _assign_targets(
    bundle: FitProjectBundle,
    primitive_id: str,
    template: PrimitiveTemplate,
    occurrences: list[_FitOccurrence],
    states: dict[str, OccurrenceFitState],
    sample_count: int,
) -> list[_AssignedOccurrence]:
    predicted = _render_components(bundle, primitive_id, template, occurrences, states)
    objective = bundle.fit_config.objective
    assignment_config = objective.assignment
    centerline_config = objective.centerline
    assigned = []
    for occurrence in occurrences:
        prepared = occurrence.prepared
        observed = prepared.reference > 0.05
        centerlines = predicted[prepared.sample.id]
        empty_mask = np.zeros(prepared.reference.shape, dtype=bool)
        state = states.get(occurrence.key, occurrence.state)
        occurrence = replace(occurrence, state=state)
        if occurrence.part.instance not in centerlines:
            assigned.append(_AssignedOccurrence(occurrence, None, empty_mask, None, None))
            continue
        assignment = assign_ink_to_components(
            observed,
            centerlines,
            occurrence.part.instance,
            sigma_assign=assignment_config.sigma_px,
            outlier_mass=assignment_config.outlier_mass,
        )
        target_mask = observed & assignment.hard_target_mask(
            assignment_config.min_target_probability
        )
        skeleton, radius = skeletonize_component(target_mask)
        skeleton = prune_skeleton(
            skeleton,
            radius,
            min_spur_length_px=centerline_config.min_spur_length_px,
            min_confidence=centerline_config.min_confidence,
        )
        transform = compose_occurrence_transform(occurrence.part.transform, state.delta)
        nominal_path = primitive_path(template, transform)
        nominal_xy = path_points_in_image(
            nominal_path,
            prepared.anchor,
            sample_count=sample_count,
        )
        target_trace = trace_ordered_centerline(
            skeleton,
            radius,
            nominal_xy,
            search_radius_px=max(2.0 * assignment_config.sigma_px, 4.0),
            sigma_position_px=assignment_config.sigma_px,
        )
        valid = (target_trace.confidence > 0) & np.isfinite(target_trace.xy).all(axis=1)
        confidence = np.zeros(sample_count, dtype=float)
        if np.any(valid):
            posterior = bilinear_sample(
                assignment.target_weights,
                target_trace.xy[valid],
                outside_value=0.0,
            )
            confidence[valid] = target_trace.confidence[valid] * posterior
        target_trace = TargetTrace(
            xy=target_trace.xy,
            tangent=target_trace.tangent,
            radius=target_trace.radius,
            confidence=confidence,
        )
        if np.count_nonzero(target_trace.confidence > 0) < 2:
            target_trace = None
        assigned.append(
            _AssignedOccurrence(
                occurrence=occurrence,
                assignment=assignment,
                target_mask=target_mask,
                target_trace=target_trace,
                assignment_entropy=_assignment_entropy(assignment, observed),
            )
        )
    return assigned


def _geometry_result(
    bundle: FitProjectBundle,
    assigned: _AssignedOccurrence,
    template: PrimitiveTemplate,
    state: OccurrenceFitState,
) -> tuple[OrderedCenterlineResiduals | None, np.ndarray | None]:
    target = assigned.target_trace
    if target is None:
        return None, None
    occurrence = assigned.occurrence
    transform = compose_occurrence_transform(occurrence.part.transform, state.delta)
    path = primitive_path(template, transform)
    candidate_xy = path_points_in_image(
        path,
        occurrence.prepared.anchor,
        sample_count=len(target.xy),
    )
    geometry = bundle.fit_config.objective.geometry
    result = ordered_centerline_residuals(
        candidate_xy,
        target,
        candidate_radius=np.full(len(candidate_xy), bundle.fit_config.render.stroke_width_px / 2.0),
        sigma_position_px=geometry.position_sigma_px,
        sigma_coverage_px=geometry.coverage_sigma_px,
        sigma_endpoint_px=geometry.endpoint_sigma_px,
        sigma_tangent_rad=geometry.tangent_sigma_rad,
        sigma_radius_px=geometry.radius_sigma_px,
        candidate_to_target_weight=geometry.candidate_to_target,
        target_to_candidate_weight=geometry.target_to_candidate,
        endpoint_weight=geometry.endpoint,
        tangent_weight=geometry.tangent,
        radius_weight=geometry.radius,
        coverage_tolerance_px=geometry.coverage_tolerance_px,
        supervise_endpoints=bool(target.confidence[0] > 0 and target.confidence[-1] > 0),
        open_target_ends=True,
    )
    return result, candidate_xy


def _phase_parameters(
    occurrence: _FitOccurrence,
    phase: str,
) -> list[str]:
    if phase == "translation":
        allowed = {"dx", "dy"}
    elif phase == "scale":
        allowed = {"dx", "dy", "log_scale_x", "log_scale_y"}
    elif phase in {"shape", "joint", "polish"}:
        allowed = set(occurrence.state.free_parameters)
    else:
        raise ValueError(f"unknown v3 fit phase: {phase}")
    return [name for name in occurrence.state.free_parameters if name in allowed]


def _curvature_cost(points: np.ndarray) -> float:
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-9
    unit = vectors[valid] / lengths[valid, None]
    if len(unit) < 2:
        return 0.0
    angles = np.arctan2(unit[:, 1], unit[:, 0])
    changes = (np.diff(angles) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.mean((changes / np.pi) ** 2))


def _fit_stage(
    bundle: FitProjectBundle,
    primitive_id: str,
    template: PrimitiveTemplate,
    states: dict[str, OccurrenceFitState],
    assigned: list[_AssignedOccurrence],
    phase: str,
    shape_reference: np.ndarray,
    optimizer: OptimizerConfig,
    *,
    optimize: bool = True,
) -> _StageResult:
    active = [item for item in assigned if item.target_trace is not None]
    if not active:
        return _StageResult(
            template,
            states,
            1e6,
            np.asarray([sqrt(2e6)]),
            {},
            {},
            0,
            None,
        )
    all_shape_space = ParameterSpace.from_primitive(template)
    shared_space = all_shape_space if phase in {"shape", "joint", "polish"} else ParameterSpace(())
    stage_states = [
        states[item.occurrence.key].model_copy(
            update={"free_parameters": _phase_parameters(item.occurrence, phase)}
        )
        for item in active
    ]
    space = CompositeParameterSpace(shared_space, stage_states)
    total_sample_weight = sum(
        item.occurrence.prepared.sample.weight
        for item in {entry.occurrence.prepared.sample.id: entry for entry in active}.values()
    )
    occurrences_per_sample: dict[str, int] = {}
    for item in active:
        sample_id = item.occurrence.prepared.sample.id
        occurrences_per_sample[sample_id] = occurrences_per_sample.get(sample_id, 0) + 1
    shared_columns = [
        index for index, spec in enumerate(space.specs) if spec.name.startswith("shared.")
    ]
    occurrence_columns = {
        item.occurrence.key: [
            index
            for index, spec in enumerate(space.specs)
            if spec.name.startswith(f"occurrence.{item.occurrence.key}.delta.")
        ]
        for item in active
    }
    layout: list[tuple[int, int, list[int]]] = []
    last_geometry: dict[str, OrderedCenterlineResiduals] = {}
    last_sample_objectives: dict[str, float] = {}

    def evaluate(vector: np.ndarray) -> np.ndarray:
        nonlocal layout, last_geometry, last_sample_objectives
        candidate_template, decoded_states = space.decode(vector, template, stage_states)
        assert isinstance(candidate_template, PrimitiveTemplate)
        decoded_by_key = {state.instance_id: state for state in decoded_states}
        blocks = []
        layout = []
        last_geometry = {}
        sample_losses: dict[str, list[float]] = {}
        row = 0

        def add_block(block: np.ndarray, columns: list[int]) -> None:
            nonlocal row
            block = np.asarray(block, dtype=float).ravel()
            blocks.append(block)
            layout.append((row, row + len(block), columns))
            row += len(block)

        for item in active:
            key = item.occurrence.key
            state = decoded_by_key[key]
            geometry, candidate_xy = _geometry_result(bundle, item, candidate_template, state)
            if geometry is None or candidate_xy is None:
                continue
            last_geometry[key] = geometry
            sample_id = item.occurrence.prepared.sample.id
            sample_weight = item.occurrence.prepared.sample.weight
            occurrence_scale = sqrt(
                sample_weight / total_sample_weight / occurrences_per_sample[sample_id]
            )
            geometry_columns = sorted({*shared_columns, *occurrence_columns[key]})
            add_block(occurrence_scale * geometry.residuals, geometry_columns)
            sample_losses.setdefault(sample_id, []).append(
                0.5 * float(np.mean(geometry.residuals**2))
            )
            if bundle.fit_config.objective.priors.curvature > 0:
                curvature_residual = np.asarray(
                    [
                        sqrt(
                            2.0
                            * bundle.fit_config.objective.priors.curvature
                            * _curvature_cost(candidate_xy)
                        )
                    ]
                )
                add_block(curvature_residual, geometry_columns)
            if state.free_parameters and bundle.fit_config.objective.priors.transform > 0:
                raw = transform_prior_residuals(state.delta)
                prior_names = ("dx", "dy", "log_scale_x", "log_scale_y", "shear_x")
                prior_by_name = dict(zip(prior_names, raw.tolist(), strict=True))
                values = np.asarray([prior_by_name[name] for name in state.free_parameters])
                prior_scale = sqrt(2.0 * bundle.fit_config.objective.priors.transform / len(values))
                add_block(prior_scale * values, occurrence_columns[key])

        last_sample_objectives = {
            sample_id: float(np.mean(values)) for sample_id, values in sample_losses.items()
        }
        family = bundle.fit_config.objective.family
        score_values = np.asarray(list(last_sample_objectives.values()), dtype=float)
        all_columns = list(range(len(space.specs)))
        if len(score_values) and family.worst > 0:
            maximum = float(np.max(score_values))
            smooth_worst = maximum + 0.1 * float(
                np.log(np.exp((score_values - maximum) / 0.1).sum())
            )
            add_block(
                np.asarray([sqrt(2.0 * family.worst * max(smooth_worst, 1e-12))]),
                all_columns,
            )
        if len(score_values) and family.spread > 0:
            spread = float(np.std(score_values))
            add_block(np.asarray([sqrt(2.0 * family.spread * max(spread, 1e-12))]), all_columns)

        shape_weight = bundle.fit_config.objective.priors.shape
        if shared_space.specs and shape_weight > 0:
            values = shared_space.encode(candidate_template)
            reference = np.asarray(
                [shape_reference[all_shape_space.specs.index(spec)] for spec in shared_space.specs]
            )
            widths = np.asarray([spec.upper - spec.lower for spec in shared_space.specs])
            shape_block = sqrt(2.0 * shape_weight / len(values)) * (values - reference) / widths
            add_block(shape_block, shared_columns)
        return np.concatenate(blocks) if blocks else np.zeros(1, dtype=float)

    start = space.encode(template, stage_states)
    initial_residuals = evaluate(start)
    matrix = sparse.lil_matrix((len(initial_residuals), len(space.specs)), dtype=np.int8)
    for row_start, row_end, columns in layout:
        if row_end > row_start and columns:
            matrix[row_start:row_end, columns] = 1
    jacobian_nnz = int(matrix.nnz)
    if optimize:
        result = optimize_parameters(
            space,
            evaluate,
            initial=start,
            config=optimizer,
            jac_sparsity=matrix.tocsr() if len(space.specs) > 1 else None,
        )
        candidate_template, decoded_states = space.decode(result.parameters, template, stage_states)
        assert isinstance(candidate_template, PrimitiveTemplate)
        final_states = dict(states)
        final_states.update(
            {
                state.instance_id: states[state.instance_id].model_copy(
                    update={"delta": state.delta}
                )
                for state in decoded_states
            }
        )
        residuals = evaluate(result.parameters)
        score = _least_squares_cost(residuals, optimizer.loss, optimizer.f_scale)
        return _StageResult(
            candidate_template,
            final_states,
            score,
            residuals,
            dict(last_geometry),
            dict(last_sample_objectives),
            jacobian_nnz,
            result,
        )

    residuals = initial_residuals
    score = _least_squares_cost(residuals, optimizer.loss, optimizer.f_scale)
    return _StageResult(
        template,
        states,
        score,
        residuals,
        dict(last_geometry),
        dict(last_sample_objectives),
        jacobian_nnz,
        None,
    )


def _assignment_change(
    previous: dict[str, np.ndarray] | None,
    assigned: list[_AssignedOccurrence],
) -> float:
    if previous is None:
        return 1.0
    changes = []
    for item in assigned:
        prior = previous.get(item.occurrence.key)
        if prior is None:
            changes.append(1.0)
            continue
        denominator = max(int(np.count_nonzero(prior | item.target_mask)), 1)
        changes.append(float(np.count_nonzero(prior ^ item.target_mask) / denominator))
    return float(np.mean(changes)) if changes else 0.0


def _assignment_masks(assigned: list[_AssignedOccurrence]) -> dict[str, np.ndarray]:
    return {item.occurrence.key: item.target_mask.copy() for item in assigned}


def _normalized_parameter_change(
    before: np.ndarray,
    after: np.ndarray,
    space: CompositeParameterSpace,
) -> float:
    if not len(before):
        return 0.0
    widths = np.asarray([spec.upper - spec.lower for spec in space.specs])
    return float(np.max(np.abs(after - before) / widths))


def _restart_templates(
    template: PrimitiveTemplate,
    space: ParameterSpace,
    optimizer: OptimizerConfig,
) -> list[PrimitiveTemplate]:
    if optimizer.restarts <= 1 or not space.specs:
        return [template]
    power = ceil(log2(optimizer.restarts - 1))
    sampler = qmc.Sobol(d=len(space.specs), scramble=True, seed=optimizer.seed)
    points = sampler.random_base2(power)[: optimizer.restarts - 1]
    lower = np.asarray([spec.lower for spec in space.specs])
    upper = np.asarray([spec.upper for spec in space.specs])
    starts = [template]
    for point in points:
        candidate = space.decode(lower + point * (upper - lower), template)
        assert isinstance(candidate, PrimitiveTemplate)
        starts.append(candidate)
    return starts


def _phase_plan(config: ObjectiveConfig) -> list[tuple[str, int]]:
    schedule = config.centerline.sample_schedule
    low = schedule[0]
    middle = schedule[min(1, len(schedule) - 1)]
    high = schedule[-1]
    return [
        ("translation", low),
        ("scale", low),
        ("shape", middle),
        ("joint", middle),
        ("polish", high),
    ]


def _run_restart(
    bundle: FitProjectBundle,
    primitive_id: str,
    seed_template: PrimitiveTemplate,
    occurrences: list[_FitOccurrence],
    initial_states: dict[str, OccurrenceFitState],
    optimizer: OptimizerConfig,
    shape_reference: np.ndarray,
) -> tuple[
    PrimitiveTemplate,
    dict[str, OccurrenceFitState],
    float,
    dict[str, Any],
    list[CheckpointIteration],
]:
    template = seed_template
    states = dict(initial_states)
    history: list[CheckpointIteration] = []
    previous_score: float | None = None
    stage_successes = []
    previous_masks = None
    final_stage = _StageResult(template, states, float("inf"), np.empty(0), {}, {}, 0, None)
    cycles = 0
    for cycle in range(optimizer.max_cycles):
        cycles = cycle + 1
        full_cycle_states = [
            initial_states[item.key].model_copy(update={"delta": states[item.key].delta})
            for item in occurrences
        ]
        full_space = CompositeParameterSpace.from_primitive(template, full_cycle_states)
        cycle_start = full_space.encode(template, full_cycle_states)
        for phase, sample_count in _phase_plan(bundle.fit_config.objective):
            assigned = _assign_targets(
                bundle,
                primitive_id,
                template,
                occurrences,
                states,
                sample_count,
            )
            stage = _fit_stage(
                bundle,
                primitive_id,
                template,
                states,
                assigned,
                phase,
                shape_reference,
                optimizer,
            )
            template, states = stage.template, stage.states
            final_stage = stage
            if stage.optimization is not None:
                stage_successes.append(stage.optimization.success)
                for item in stage.optimization.history:
                    history.append(item.model_copy(update={"iteration": len(history)}))
        final_assigned = _assign_targets(
            bundle,
            primitive_id,
            template,
            occurrences,
            states,
            bundle.fit_config.objective.centerline.sample_schedule[-1],
        )
        final_stage = _fit_stage(
            bundle,
            primitive_id,
            template,
            states,
            final_assigned,
            "joint",
            shape_reference,
            optimizer,
            optimize=False,
        )
        updated_cycle_states = [
            initial_states[item.key].model_copy(update={"delta": states[item.key].delta})
            for item in occurrences
        ]
        updated_space = CompositeParameterSpace.from_primitive(template, updated_cycle_states)
        after = updated_space.encode(template, updated_cycle_states)
        update = _normalized_parameter_change(cycle_start, after, updated_space)
        assignment_change = _assignment_change(previous_masks, final_assigned)
        relative = (
            abs(previous_score - final_stage.score) / max(abs(previous_score), 1e-12)
            if previous_score is not None
            else float("inf")
        )
        previous_masks = _assignment_masks(final_assigned)
        if (
            previous_score is not None
            and relative <= optimizer.convergence_relative
            and assignment_change <= optimizer.convergence_relative
            and update <= optimizer.convergence_relative
        ):
            break
        previous_score = final_stage.score

    details = {
        "cycles": cycles,
        "success": all(stage_successes) if stage_successes else True,
        "assignment_change": assignment_change,
        "geometry": final_stage.geometry,
        "sample_objectives": final_stage.sample_objectives,
        "jacobian_nnz": final_stage.jacobian_nnz,
        "residuals": final_stage.residuals,
    }
    return template, states, final_stage.score, details, history


def _sample_report(
    sample_id: str,
    assigned: list[_AssignedOccurrence],
    geometry: dict[str, OrderedCenterlineResiduals],
) -> dict[str, Any]:
    instances = []
    values = []
    for item in assigned:
        key = item.occurrence.key
        result = geometry.get(key)
        if result is not None:
            loss = 0.5 * float(np.mean(result.residuals**2))
            values.append(loss)
            metrics = result.diagnostics
            status = "assigned"
        else:
            loss = 0.0
            metrics = None
            status = "unassigned"
        assignment = item.assignment
        instances.append(
            {
                "instance_id": item.occurrence.part.instance,
                "primitive_id": item.occurrence.part.primitive,
                "status": status,
                "delta": item.occurrence.state.delta.model_dump(mode="json"),
                "assignment": (
                    {
                        "target_pixels": int(item.target_mask.sum()),
                        "mean_target_probability": (
                            float(
                                np.mean(
                                    assignment.target_weights[
                                        item.occurrence.prepared.reference > 0.05
                                    ]
                                )
                            )
                            if assignment is not None
                            else 0.0
                        ),
                        "mean_outlier_probability": (
                            float(
                                np.mean(
                                    assignment.outlier_weights[
                                        item.occurrence.prepared.reference > 0.05
                                    ]
                                )
                            )
                            if assignment is not None
                            else 0.0
                        ),
                        "entropy": item.assignment_entropy,
                    }
                ),
                "geometry": metrics,
            }
        )
    return {
        "sample_id": sample_id,
        "objective": float(np.mean(values)) if values else 0.0,
        "instances": instances,
    }


def _diagnostic_split(
    bundle: FitProjectBundle,
    family: FamilyDefinition,
    samples: list[FitSample],
    template: PrimitiveTemplate,
    optimizer: OptimizerConfig,
    shape_reference: np.ndarray,
    state_by_sample: dict[str, dict[str, OccurrenceFitState]] | None = None,
) -> tuple[dict[str, Any], list[_AssignedOccurrence]]:
    if not samples:
        return {"samples": [], "mean_objective": None, "policy": "no samples in this split"}, []
    occurrences = _prepare_occurrences(bundle, family, samples)
    states = {
        item.key: (
            state_by_sample.get(item.prepared.sample.id, {}).get(item.key, item.state)
            if state_by_sample is not None
            else item.state
        )
        for item in occurrences
    }
    count = bundle.fit_config.objective.centerline.sample_schedule[-1]
    assigned = _assign_targets(bundle, family.primitive, template, occurrences, states, count)
    stage = _fit_stage(
        bundle,
        family.primitive,
        template,
        states,
        assigned,
        "joint",
        shape_reference,
        optimizer,
        optimize=False,
    )
    grouped: dict[str, list[_AssignedOccurrence]] = {}
    for item in assigned:
        grouped.setdefault(item.occurrence.prepared.sample.id, []).append(item)
    report_samples = [
        _sample_report(sample.id, grouped.get(sample.id, []), stage.geometry) for sample in samples
    ]
    objectives = [item["objective"] for item in report_samples]
    report = {
        "policy": "evaluated after fitting; diagnostic only; parameters are unchanged",
        "samples": report_samples,
        "mean_objective": float(np.mean(objectives)) if objectives else None,
        "worst_member_objective": float(np.max(objectives)) if objectives else None,
        "std_objective": float(np.std(objectives)) if objectives else None,
    }
    return report, assigned


def optimize_primitive_family_v3(
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
    training_ids = set(family.fit)
    validation_ids = set(family.validation)
    holdout_ids = set(family.hold_out)
    training = [sample for sample in bundle.corpus.samples if sample.id in training_ids]
    validation = [sample for sample in bundle.corpus.samples if sample.id in validation_ids]
    hold_out = [sample for sample in bundle.corpus.samples if sample.id in holdout_ids]
    if not training:
        raise ValueError(f"primitive family has no fit samples: {primitive_id}")

    objective: ObjectiveConfig = bundle.fit_config.objective
    optimizer: OptimizerConfig = bundle.fit_config.optimizer
    initial_template = bundle.grammar.primitives[primitive_id]
    shape_space = ParameterSpace.from_primitive(initial_template)
    shape_reference = shape_space.encode(initial_template)
    occurrences = _prepare_occurrences(bundle, family, training)
    base_states = {item.key: item.state for item in occurrences}
    full_space = CompositeParameterSpace.from_primitive(
        initial_template, [base_states[item.key] for item in occurrences]
    )
    inputs = _input_hashes(bundle, training)
    stage_name = f"primitive:{primitive_id}"
    checkpoint_path = (
        _project_output_directory(bundle.project_dir, "checkpoints") / f"{primitive_id}.json"
    )
    reports_dir = _project_output_directory(bundle.project_dir, "reports")
    json_report_path = reports_dir / f"{primitive_id}.json"
    markdown_report_path = reports_dir / f"{primitive_id}.md"
    if any(path.is_symlink() for path in (checkpoint_path, json_report_path, markdown_report_path)):
        raise ValueError("fitting output files must not be symlinks")

    start_template = initial_template
    start_states = base_states
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_objective_version(checkpoint, objective.version)
        if (
            checkpoint.project_id != bundle.project.id
            or checkpoint.stage != stage_name
            or checkpoint.inputs != inputs
            or checkpoint.optimizer.backend != "inkgrammar-centerline-em-trf"
            or checkpoint.optimizer.seed != optimizer.seed
        ):
            raise ValueError("checkpoint inputs or optimizer configuration do not match")
        if set(checkpoint.parameters) != {spec.name for spec in full_space.specs}:
            raise ValueError("v3 checkpoint parameters do not match composite parameter space")
        start_vector = np.asarray(
            [checkpoint.parameters[spec.name] for spec in full_space.specs], dtype=float
        )
        start_template, decoded = full_space.decode(
            start_vector,
            initial_template,
            [base_states[item.key] for item in occurrences],
        )
        start_states = {state.instance_id: state for state in decoded}

    initial_assigned = _assign_targets(
        bundle,
        primitive_id,
        start_template,
        occurrences,
        start_states,
        objective.centerline.sample_schedule[-1],
    )
    initial_stage = _fit_stage(
        bundle,
        primitive_id,
        start_template,
        start_states,
        initial_assigned,
        "joint",
        shape_reference,
        optimizer,
        optimize=False,
    )
    restart_templates = _restart_templates(start_template, shape_space, optimizer)
    selected_template = start_template
    selected_states = dict(start_states)
    selected_score = float("inf")
    selected_details: dict[str, Any] = {}
    history: list[CheckpointIteration] = []
    restart_scores = []
    for restart_template in restart_templates:
        candidate_template, candidate_states, score, details, restart_history = _run_restart(
            bundle,
            primitive_id,
            restart_template,
            occurrences,
            start_states,
            optimizer,
            shape_reference,
        )
        history.extend(restart_history)
        restart_scores.append(score)
        if score < selected_score:
            selected_template = candidate_template
            selected_states = candidate_states
            selected_score = score
            selected_details = details

    final_state_by_sample = {}
    for occurrence in occurrences:
        final_state_by_sample.setdefault(occurrence.prepared.sample.id, {})[occurrence.key] = (
            selected_states[occurrence.key]
        )
    final_assigned = _assign_targets(
        bundle,
        primitive_id,
        selected_template,
        occurrences,
        selected_states,
        objective.centerline.sample_schedule[-1],
    )
    final_stage = _fit_stage(
        bundle,
        primitive_id,
        selected_template,
        selected_states,
        final_assigned,
        "joint",
        shape_reference,
        optimizer,
        optimize=False,
    )
    initial_grouped: dict[str, list[_AssignedOccurrence]] = {}
    for item in initial_assigned:
        initial_grouped.setdefault(item.occurrence.prepared.sample.id, []).append(item)
    final_grouped: dict[str, list[_AssignedOccurrence]] = {}
    for item in final_assigned:
        final_grouped.setdefault(item.occurrence.prepared.sample.id, []).append(item)
    initial_reports = {
        sample.id: _sample_report(
            sample.id,
            initial_grouped.get(sample.id, []),
            initial_stage.geometry,
        )
        for sample in training
    }
    final_reports = {
        sample.id: _sample_report(
            sample.id,
            final_grouped.get(sample.id, []),
            final_stage.geometry,
        )
        for sample in training
    }
    training_samples = []
    for sample in training:
        initial_report = initial_reports[sample.id]
        final_report = final_reports[sample.id]
        training_samples.append(
            {
                **final_report,
                "initial_objective": initial_report["objective"],
                "final_objective": final_report["objective"],
            }
        )
    training_objectives = [item["final_objective"] for item in training_samples]
    training_summary = {
        "selection_split": "fit",
        "initial_objective": initial_stage.score,
        "final_objective": final_stage.score,
        "mean_objective": float(np.mean(training_objectives)) if training_objectives else 0.0,
        "median_objective": float(np.median(training_objectives)) if training_objectives else 0.0,
        "worst_member_objective": (
            float(np.max(training_objectives)) if training_objectives else 0.0
        ),
        "std_objective": float(np.std(training_objectives)) if training_objectives else 0.0,
        "samples": training_samples,
    }

    validation_report, validation_assigned = _diagnostic_split(
        bundle, family, validation, selected_template, optimizer, shape_reference
    )
    holdout_report, holdout_assigned = _diagnostic_split(
        bundle, family, hold_out, selected_template, optimizer, shape_reference
    )
    validation_state_by_sample = {
        item.occurrence.prepared.sample.id: {
            item.occurrence.key: item.occurrence.state for item in validation_assigned
        }
        for item in validation_assigned
    }
    holdout_state_by_sample = {
        item.occurrence.prepared.sample.id: {
            item.occurrence.key: item.occurrence.state for item in holdout_assigned
        }
        for item in holdout_assigned
    }
    review_samples = [
        _prepare_sample(bundle, sample, primitive_id)
        for sample in [*training, *validation, *hold_out]
    ]
    occurrence_deltas = {}
    for sample_id, states_for_sample in final_state_by_sample.items():
        occurrence_deltas[sample_id] = {
            key.split("--", 1)[1]: state.delta for key, state in states_for_sample.items()
        }
    for sample_id, states_for_sample in {
        **validation_state_by_sample,
        **holdout_state_by_sample,
    }.items():
        occurrence_deltas[sample_id] = {
            key.split("--", 1)[1]: state.delta for key, state in states_for_sample.items()
        }
    private_review_sheet_path = _write_private_overlays(
        bundle,
        primitive_id,
        selected_template,
        review_samples,
        occurrence_deltas=occurrence_deltas,
        include_nominal=True,
    )

    final_parameters = full_space.encode(
        selected_template,
        [selected_states[item.key] for item in occurrences],
    )
    parameter_map = dict(
        zip(
            (spec.name for spec in full_space.specs),
            final_parameters.tolist(),
            strict=True,
        )
    )
    trace_summary = summarize_trace(history, checkpoint.trace_summary if checkpoint else None)
    if checkpoint is None:
        checkpoint = FitCheckpoint(
            project_id=bundle.project.id,
            stage=stage_name,
            created_by=CheckpointCreator(inkgrammar_version=__version__),
            inputs=inputs,
            optimizer=CheckpointOptimizer(
                backend="inkgrammar-centerline-em-trf",
                seed=optimizer.seed,
                objective_version=objective.version,
            ),
            parameters=parameter_map,
        )
    checkpoint_update = checkpoint.model_copy(
        update={
            "inputs": inputs,
            "optimizer": checkpoint.optimizer.model_copy(
                update={
                    "backend": "inkgrammar-centerline-em-trf",
                    "seed": optimizer.seed,
                    "objective_version": objective.version,
                }
            ),
            "parameters": parameter_map,
            "training_metrics": {
                "objective": final_stage.score,
                "initial_objective": initial_stage.score,
                "mean_objective": training_summary["mean_objective"],
            },
            "progress": checkpoint.progress.model_copy(
                update={"evaluations": trace_summary.evaluations}
            ),
            "trace_summary": trace_summary,
        }
    )
    save_checkpoint(checkpoint_update, checkpoint_path)

    member_mean = training_summary["mean_objective"]
    holdout_mean = holdout_report["mean_objective"]
    report = {
        "format": "inkgrammar-fit-report",
        "format_version": 3,
        "project_id": bundle.project.id,
        "stage": stage_name,
        "primitive_id": primitive_id,
        "objective_version": objective.version,
        "selection_objective": {
            "total": final_stage.score,
            "blocks": {
                "candidate_to_target_weight": objective.geometry.candidate_to_target,
                "target_to_candidate_weight": objective.geometry.target_to_candidate,
                "endpoint_weight": objective.geometry.endpoint,
                "tangent_weight": objective.geometry.tangent,
                "radius_weight": objective.geometry.radius,
                "prior_transform": objective.priors.transform,
                "prior_shape": objective.priors.shape,
                "prior_curvature": objective.priors.curvature,
                "assignment_sigma_px": objective.assignment.sigma_px,
                "assignment_outlier_mass": objective.assignment.outlier_mass,
            },
        },
        "optimizer": {
            "backend": "inkgrammar-centerline-em-trf",
            "strategy": "staged-coarse-to-fine-em",
            "success": selected_details.get("success", False),
            "evaluations": trace_summary.evaluations,
            "restarts": len(restart_templates),
            "restart_scores": restart_scores,
            "assignment_cycles": selected_details.get("cycles", 0),
            "final_assignment_change": selected_details.get("assignment_change", 0.0),
            "jacobian_sparsity_nnz": selected_details.get("jacobian_nnz", 0),
            "config": optimizer.model_dump(mode="json"),
        },
        "input_hashes": inputs.model_dump(mode="json"),
        "initial_parameters": dict(
            zip(
                (spec.name for spec in full_space.specs),
                full_space.encode(
                    initial_template,
                    [base_states[item.key] for item in occurrences],
                ).tolist(),
                strict=True,
            )
        ),
        "parameters": parameter_map,
        "occurrences": {
            f"{sample['sample_id']}--{instance['instance_id']}": {"delta": instance["delta"]}
            for sample in training_samples
            for instance in sample["instances"]
        },
        "primitive": selected_template.model_dump(mode="json"),
        "training": training_summary,
        "validation": validation_report,
        "hold_out": holdout_report,
        "generalization_gap": holdout_mean - member_mean if holdout_mean is not None else None,
        "visual_review": {
            "status": "promotion_pending",
            "scope": "private work directory",
            "variants": ["nominal", "aligned"],
            "sheet": private_review_sheet_path.relative_to(bundle.private_work_dir).as_posix(),
            "region_overlay_count": sum(len(sample.regions) for sample in review_samples),
        },
        "evaluations": trace_summary.evaluations,
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
