from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from ..models import StrictModel
from .checkpoint import CheckpointIteration, FitCheckpoint
from .parameterization import ParameterSpace


class OptimizerConfig(StrictModel):
    method: Literal["powell", "differential_evolution+powell"] = "powell"
    seed: int = Field(default=1968, ge=0)
    max_iterations: int = Field(default=200, gt=0)
    tolerance: float = Field(default=1e-5, gt=0)
    de_max_iterations: int = Field(default=20, gt=0)
    de_population_size: int = Field(default=8, gt=0)
    de_dimension_limit: int = Field(default=12, gt=0)


@dataclass(frozen=True)
class OptimizationResult:
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    objective: float
    history: tuple[CheckpointIteration, ...]
    evaluations: int


def _backend(config: OptimizerConfig) -> str:
    if config.method == "powell":
        return "scipy-powell"
    return "scipy-differential-evolution+powell"


def optimize_parameters(
    parameter_space: ParameterSpace,
    objective: Callable[[np.ndarray], float],
    *,
    initial: Sequence[float] | np.ndarray | None = None,
    config: OptimizerConfig | None = None,
    prior_history: Sequence[CheckpointIteration] = (),
) -> OptimizationResult:
    """Minimize an objective over the explicit bounded parameter vector."""
    config = config or OptimizerConfig()
    specs = parameter_space.specs
    names = tuple(spec.name for spec in specs)
    start = np.asarray(
        initial if initial is not None else [spec.initial for spec in specs],
        dtype=float,
    )
    parameter_space._validate_vector(start)
    history = list(prior_history)

    def evaluate(vector: np.ndarray) -> float:
        candidate = np.asarray(vector, dtype=float)
        parameter_space._validate_vector(candidate)
        value = float(objective(candidate.copy()))
        if not np.isfinite(value):
            raise ValueError("optimizer objective must return a finite value")
        history.append(
            CheckpointIteration(
                iteration=len(history),
                objective=value,
                parameters=dict(zip(names, candidate.tolist(), strict=True)),
            )
        )
        return value

    if not names:
        value = evaluate(start)
        return OptimizationResult(names, start.copy(), value, tuple(history), 1)

    from scipy.optimize import differential_evolution, minimize

    if config.method == "differential_evolution+powell":
        if len(names) > config.de_dimension_limit:
            raise ValueError(
                f"differential evolution is limited to {config.de_dimension_limit} parameters"
            )
        coarse = differential_evolution(
            evaluate,
            bounds=parameter_space.bounds(),
            seed=config.seed,
            workers=1,
            updating="immediate",
            popsize=config.de_population_size,
            maxiter=config.de_max_iterations,
            tol=config.tolerance,
            polish=False,
        )
        start = np.asarray(coarse.x, dtype=float)

    refined = minimize(
        evaluate,
        start,
        method="Powell",
        bounds=parameter_space.bounds(),
        options={
            "maxiter": config.max_iterations,
            "xtol": config.tolerance,
            "ftol": config.tolerance,
            "disp": False,
        },
    )
    parameters = np.asarray(refined.x, dtype=float)
    value = evaluate(parameters)
    return OptimizationResult(
        names,
        parameters,
        value,
        tuple(history),
        len(history) - len(prior_history),
    )


def resume_optimization(
    checkpoint: FitCheckpoint,
    parameter_space: ParameterSpace,
    objective: Callable[[np.ndarray], float],
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
        prior_history=checkpoint.history,
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
    updates = {
        "parameters": dict(zip(result.parameter_names, result.parameters.tolist(), strict=True)),
        "training_metrics": {
            **checkpoint.training_metrics,
            "objective": result.objective,
        },
        "history": list(result.history),
    }
    if config is not None:
        updates["optimizer"] = checkpoint.optimizer.model_copy(
            update={"backend": _backend(config), "seed": config.seed}
        )
    return checkpoint.model_copy(update=updates)
