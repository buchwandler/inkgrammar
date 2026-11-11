from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from ..models import StrictModel
from .checkpoint import CheckpointIteration, FitCheckpoint, summarize_trace
from .parameterization import ParameterSpace


class OptimizerConfig(StrictModel):
    method: Literal[
        "powell",
        "differential_evolution+powell",
        "least_squares",
        "alternating-least-squares",
        "centerline-em-trf",
        "spline-alt-lsq",
        "spline-annealed-alt-lsq",
        "dual-annealing+spline-alt-lsq",
        "spline-random-restarts",
        "differential-evolution+spline-alt-lsq",
    ] = "powell"
    seed: int = Field(default=1968, ge=0)
    max_iterations: int = Field(default=200, gt=0)
    tolerance: float = Field(default=1e-5, gt=0)
    de_max_iterations: int = Field(default=20, gt=0)
    de_population_size: int = Field(default=8, gt=0)
    de_dimension_limit: int = Field(default=12, gt=0)
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    f_scale: float = Field(default=1.0, gt=0)
    max_nfev: int = Field(default=250, gt=0)
    max_cycles: int = Field(default=6, gt=0)
    transform_max_nfev: int = Field(default=120, gt=0)
    irls_iterations: int = Field(default=3, ge=1)
    assignment_update_interval: int = Field(default=1, gt=0)
    convergence_relative: float = Field(default=0.0005, gt=0)
    joint_refine: bool = True
    restarts: int = Field(default=8, ge=1)

    anneal_steps: int = Field(default=240, ge=1)
    anneal_initial_samples: int = Field(default=32, ge=1)
    anneal_target_uphill_acceptance: float = Field(default=0.80, gt=0, lt=1)
    anneal_cooling: float = Field(default=0.975, gt=0, lt=1)
    anneal_sigma_max: float = Field(default=0.30, gt=0)
    anneal_sigma_min: float = Field(default=0.025, gt=0)
    anneal_reheat_patience: int = Field(default=40, ge=1)
    anneal_restart_patience: int = Field(default=100, ge=1)
    anneal_top_k: int = Field(default=4, ge=1)
    anneal_coarse_cycles: int = Field(default=2, ge=1)
    anneal_medium_cycles: int = Field(default=4, ge=1)
    anneal_final_cycles: int = Field(default=10, ge=1)
    anneal_topology_penalty: float = Field(default=20.0, ge=0)
    anneal_no_path_penalty: float = Field(default=50.0, ge=0)
    anneal_transform_saturation_penalty: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def validate_annealing_ranges(self) -> OptimizerConfig:
        if self.anneal_sigma_min > self.anneal_sigma_max:
            raise ValueError("annealing sigma_min must not exceed sigma_max")
        if not self.anneal_coarse_cycles <= self.anneal_medium_cycles <= self.anneal_final_cycles:
            raise ValueError("annealing refinement cycles must be nondecreasing")
        return self

@dataclass(frozen=True)
class OptimizationResult:
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    objective: float
    history: tuple[CheckpointIteration, ...]
    evaluations: int
    backend: str
    success: bool
    status: int | None
    message: str
    accepted_history_index: int
    solver_parameters: np.ndarray
    solver_objective: float


def _backend(config: OptimizerConfig) -> str:
    if config.method == "powell":
        return "scipy-powell"
    if config.method == "least_squares":
        return "scipy-least-squares-trf"
    if config.method == "centerline-em-trf":
        return "inkgrammar-centerline-em-trf"
    if config.method == "alternating-least-squares":
        return "inkgrammar-alternating-least-squares"
    if config.method == "spline-alt-lsq":
        return "inkgrammar-spline-alt-lsq"
    if config.method == "spline-annealed-alt-lsq":
        return "inkgrammar-annealed-spline-alt-lsq"
    if config.method == "dual-annealing+spline-alt-lsq":
        return "scipy-dual-annealing-seed"
    if config.method == "spline-random-restarts":
        return "inkgrammar-random-restart-spline-seed"
    if config.method == "differential-evolution+spline-alt-lsq":
        return "scipy-differential-evolution-spline-seed"
    return "scipy-differential-evolution+powell"


def _least_squares_cost(residuals: np.ndarray, loss: str, f_scale: float) -> float:
    scaled_squared = (residuals / f_scale) ** 2
    if loss == "linear":
        return float(0.5 * np.sum(residuals**2))
    if loss == "soft_l1":
        return float(f_scale**2 * np.sum(np.sqrt(1.0 + scaled_squared) - 1.0))
    if loss == "huber":
        rho = np.where(scaled_squared <= 1.0, scaled_squared, 2.0 * np.sqrt(scaled_squared) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(scaled_squared)
    elif loss == "arctan":
        rho = np.arctan(scaled_squared)
    else:
        raise ValueError(f"unsupported least-squares loss: {loss}")
    return float(0.5 * f_scale**2 * np.sum(rho))


def _best_history_candidate(
    history: Sequence[CheckpointIteration], parameter_space: ParameterSpace
) -> tuple[int, CheckpointIteration]:
    feasible = []
    names = tuple(spec.name for spec in parameter_space.specs)
    for index, item in enumerate(history):
        if set(item.parameters) != set(names) or not np.isfinite(item.objective):
            continue
        vector = np.asarray([item.parameters[name] for name in names], dtype=float)
        try:
            parameter_space._validate_vector(vector)
        except ValueError:
            continue
        feasible.append((index, item))
    if not feasible:
        raise ValueError("optimizer did not evaluate a feasible candidate")
    return min(feasible, key=lambda pair: pair[1].objective)


def optimize_parameters(
    parameter_space: ParameterSpace,
    objective: Callable[[np.ndarray], float | Sequence[float] | np.ndarray],
    *,
    initial: Sequence[float] | np.ndarray | None = None,
    config: OptimizerConfig | None = None,
    prior_history: Sequence[CheckpointIteration] = (),
    jac_sparsity: Any | None = None,
) -> OptimizationResult:
    """Minimize a bounded scalar objective or a residual vector."""
    config = config or OptimizerConfig()
    specs = parameter_space.specs
    names = tuple(spec.name for spec in specs)
    start = np.asarray(
        initial if initial is not None else [spec.initial for spec in specs],
        dtype=float,
    )
    parameter_space._validate_vector(start)
    optimizer_start = parameter_space.encode_optimizer_space(start)
    history = list(prior_history)
    initial_history_size = len(history)

    def raw_candidate(optimizer_vector: np.ndarray) -> np.ndarray:
        return parameter_space.decode_optimizer_space(np.asarray(optimizer_vector, dtype=float))

    def record(candidate: np.ndarray, value: float) -> None:
        history.append(
            CheckpointIteration(
                iteration=len(history),
                objective=value,
                parameters=dict(zip(names, candidate.tolist(), strict=True)),
            )
        )

    def evaluate_scalar(optimizer_vector: np.ndarray) -> float:
        candidate = raw_candidate(optimizer_vector)
        value = float(objective(candidate.copy()))
        if not np.isfinite(value):
            raise ValueError("optimizer objective must return a finite value")
        record(candidate, value)
        return value

    def evaluate_residuals(optimizer_vector: np.ndarray) -> np.ndarray:
        candidate = raw_candidate(optimizer_vector)
        residuals = np.asarray(objective(candidate.copy()), dtype=float)
        if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all():
            raise ValueError("least-squares objective must return a finite non-empty vector")
        record(candidate, _least_squares_cost(residuals, config.loss, config.f_scale))
        return residuals

    def result_from_history(
        *,
        success: bool,
        status: int | None,
        message: str,
        solver_parameters: np.ndarray,
        solver_objective: float,
    ) -> OptimizationResult:
        accepted_index, accepted = _best_history_candidate(history, parameter_space)
        parameters = np.asarray([accepted.parameters[name] for name in names], dtype=float)
        return OptimizationResult(
            parameter_names=names,
            parameters=parameters,
            objective=float(accepted.objective),
            history=tuple(history),
            evaluations=len(history) - initial_history_size,
            backend=_backend(config),
            success=success,
            status=status,
            message=message,
            accepted_history_index=accepted_index,
            solver_parameters=solver_parameters.copy(),
            solver_objective=float(solver_objective),
        )

    if not names:
        if config.method in {"least_squares", "alternating-least-squares", "centerline-em-trf"}:
            residuals = evaluate_residuals(optimizer_start)
            value = _least_squares_cost(residuals, config.loss, config.f_scale)
        else:
            value = evaluate_scalar(optimizer_start)
        return result_from_history(
            success=True,
            status=0,
            message="No free parameters.",
            solver_parameters=start,
            solver_objective=value,
        )

    from scipy.optimize import differential_evolution, least_squares, minimize

    if config.method == "differential_evolution+powell":
        if len(names) > config.de_dimension_limit:
            raise ValueError(
                f"differential evolution is limited to {config.de_dimension_limit} parameters"
            )
        coarse = differential_evolution(
            evaluate_scalar,
            bounds=parameter_space.normalized_bounds(),
            seed=config.seed,
            workers=1,
            updating="immediate",
            popsize=config.de_population_size,
            maxiter=config.de_max_iterations,
            tol=config.tolerance,
            polish=False,
        )
        optimizer_start = np.asarray(coarse.x, dtype=float)

    if config.method in {"least_squares", "alternating-least-squares", "centerline-em-trf"}:
        refined = least_squares(
            evaluate_residuals,
            optimizer_start,
            bounds=(-np.ones(len(names)), np.ones(len(names))),
            method="trf",
            loss=config.loss,
            f_scale=config.f_scale,
            x_scale="jac",
            max_nfev=config.max_nfev,
            ftol=config.tolerance,
            xtol=config.tolerance,
            gtol=config.tolerance,
            jac_sparsity=jac_sparsity,
        )
        solver_parameters = raw_candidate(refined.x)
        residuals = evaluate_residuals(refined.x)
        solver_objective = _least_squares_cost(residuals, config.loss, config.f_scale)
    else:
        refined = minimize(
            evaluate_scalar,
            optimizer_start,
            method="Powell",
            bounds=parameter_space.normalized_bounds(),
            options={
                "maxiter": config.max_iterations,
                "xtol": config.tolerance,
                "ftol": config.tolerance,
                "disp": False,
            },
        )
        solver_parameters = raw_candidate(refined.x)
        solver_objective = evaluate_scalar(refined.x)

    return result_from_history(
        success=bool(refined.success),
        status=int(refined.status) if hasattr(refined, "status") else None,
        message=str(refined.message),
        solver_parameters=solver_parameters,
        solver_objective=solver_objective,
    )


def resume_optimization(
    checkpoint: FitCheckpoint,
    parameter_space: ParameterSpace,
    objective: Callable[[np.ndarray], float | Sequence[float] | np.ndarray],
    *,
    config: OptimizerConfig | None = None,
) -> OptimizationResult:
    """Resume from the accepted checkpoint vector, not optimizer internals."""
    names = tuple(spec.name for spec in parameter_space.specs)
    if set(checkpoint.parameters) != set(names):
        raise ValueError("checkpoint parameter names do not match parameter space")
    initial = [checkpoint.parameters[name] for name in names]
    if config is None:
        methods = {
            "scipy-powell": "powell",
            "scipy-differential-evolution+powell": "differential_evolution+powell",
            "scipy-least-squares-trf": "least_squares",
            "inkgrammar-centerline-em-trf": "centerline-em-trf",
        }
        try:
            method = methods[checkpoint.optimizer.backend]
        except KeyError as exc:
            raise ValueError("unsupported checkpoint optimizer backend") from exc
        config = OptimizerConfig(method=method, seed=checkpoint.optimizer.seed)
    return optimize_parameters(
        parameter_space,
        objective,
        initial=initial,
        config=config,
    )


def checkpoint_from_result(
    checkpoint: FitCheckpoint,
    result: OptimizationResult,
    *,
    config: OptimizerConfig | None = None,
) -> FitCheckpoint:
    """Build a portable checkpoint from accepted numeric optimization state."""
    if set(checkpoint.parameters) != set(result.parameter_names):
        raise ValueError("checkpoint parameter names do not match optimization result")
    trace_summary = summarize_trace(result.history, checkpoint.trace_summary)
    updates = {
        "parameters": dict(zip(result.parameter_names, result.parameters.tolist(), strict=True)),
        "training_metrics": {
            **checkpoint.training_metrics,
            "objective": result.objective,
        },
        "progress": checkpoint.progress.model_copy(
            update={"evaluations": trace_summary.evaluations}
        ),
        "trace_summary": trace_summary,
    }
    if config is not None:
        updates["optimizer"] = checkpoint.optimizer.model_copy(
            update={"backend": _backend(config), "seed": config.seed}
        )
    return checkpoint.model_copy(update=updates)


def save_optimization_trace(result: OptimizationResult, path: str | Path) -> None:
    """Write an optional debug trace to a private path under ignored work/."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    history = result.history
    parameters = np.asarray(
        [[item.parameters[name] for name in result.parameter_names] for item in history],
        dtype=np.float32,
    ).reshape(len(history), len(result.parameter_names))
    accepted = np.zeros(len(history), dtype=bool)
    if history:
        accepted[result.accepted_history_index] = True
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            objective=np.asarray([item.objective for item in history], dtype=np.float64),
            parameters=parameters,
            accepted=accepted,
        )
