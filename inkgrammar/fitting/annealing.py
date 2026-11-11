from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, default_rng


@dataclass(frozen=True)
class AnnealingConfig:
    steps: int = 240
    initial_samples: int = 32
    cooling: float = 0.975
    sigma_max: float = 0.30
    sigma_min: float = 0.025
    reheat_patience: int = 40
    restart_patience: int = 100
    target_uphill_acceptance: float = 0.80

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.initial_samples <= 0:
            raise ValueError("annealing steps and initial samples must be positive")
        if not np.isfinite(
            [self.cooling, self.sigma_max, self.sigma_min, self.target_uphill_acceptance]
        ).all():
            raise ValueError("annealing configuration values must be finite")
        if not 0.0 < self.cooling < 1.0:
            raise ValueError("annealing cooling must be between zero and one")
        if self.sigma_min <= 0.0 or self.sigma_max < self.sigma_min:
            raise ValueError("annealing proposal scales must satisfy 0 < sigma_min <= sigma_max")
        if self.reheat_patience <= 0 or self.restart_patience <= 0:
            raise ValueError("annealing patience values must be positive")
        if not 0.0 < self.target_uphill_acceptance < 1.0:
            raise ValueError("target uphill acceptance must be between zero and one")


@dataclass(frozen=True)
class AnnealingIteration:
    step: int
    temperature: float
    current_energy: float
    proposal_energy: float
    accepted: bool
    best_energy: float


@dataclass(frozen=True)
class AnnealingResult:
    best_parameters: np.ndarray
    best_energy: float
    current_parameters: np.ndarray
    current_energy: float
    evaluations: int
    accepted_moves: int
    uphill_accepted_moves: int
    trace_summary: dict[str, float]
    trace: tuple[AnnealingIteration, ...] = ()


def _finite_energy(objective: Callable[[np.ndarray], float], parameters: np.ndarray) -> float:
    energy = float(objective(parameters.copy()))
    if not np.isfinite(energy):
        raise ValueError("annealing objective must return a finite value")
    return energy


def _validated_bounds(
    initial: np.ndarray, bounds: Sequence[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (len(initial), 2) or not np.isfinite(limits).all():
        raise ValueError("annealing bounds must contain one finite pair per parameter")
    lower = limits[:, 0]
    upper = limits[:, 1]
    if np.any(lower > upper):
        raise ValueError("annealing lower bounds must not exceed upper bounds")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial annealing parameters must be inside the bounds")
    return lower, upper


def _bounded(parameters: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(parameters, dtype=float), lower, upper)


def _default_proposal(
    current: np.ndarray,
    sigma: float,
    rng: Generator,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    dimensions = len(current)
    scale = (upper - lower) * sigma
    delta = np.zeros(dimensions, dtype=float)
    if dimensions == 1 or rng.random() < 0.7:
        count = int(rng.integers(1, min(2, dimensions) + 1))
        selected = rng.choice(dimensions, size=count, replace=False)
        delta[selected] = rng.normal(0.0, scale[selected])
    else:
        delta = rng.normal(0.0, scale)
    return _bounded(current + delta, lower, upper)


def _heavy_tailed_proposal(
    current: np.ndarray, sigma: float, rng: Generator, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    scale = (upper - lower) * max(sigma, 0.25)
    return _bounded(current + rng.standard_cauchy(len(current)) * scale, lower, upper)


def metropolis_accept(delta: float, temperature: float, rng: Generator) -> bool:
    if np.isnan(delta):
        raise ValueError("annealing energy difference must not be NaN")
    if delta <= 0.0:
        return True
    if temperature <= 0.0:
        return False
    return bool(rng.random() < np.exp(-delta / temperature))


def calibrate_temperature(
    initial: np.ndarray,
    objective: Callable[[np.ndarray], float],
    proposer: Callable[[np.ndarray, float, Generator], np.ndarray],
    *,
    samples: int,
    target_uphill_acceptance: float,
    rng: Generator,
    initial_energy: float | None = None,
) -> float:
    if samples <= 0:
        raise ValueError("temperature calibration samples must be positive")
    if not 0.0 < target_uphill_acceptance < 1.0:
        raise ValueError("target uphill acceptance must be between zero and one")
    start_energy = (
        _finite_energy(objective, initial) if initial_energy is None else float(initial_energy)
    )
    if not np.isfinite(start_energy):
        raise ValueError("initial annealing energy must be finite")

    energies = [start_energy]
    positive_deltas = []
    for _ in range(samples):
        candidate = np.asarray(proposer(initial.copy(), 1.0, rng), dtype=float)
        if candidate.shape != initial.shape or not np.isfinite(candidate).all():
            raise ValueError("temperature calibration proposer returned an invalid vector")
        energy = _finite_energy(objective, candidate)
        energies.append(energy)
        delta = energy - start_energy
        if delta > 0.0 and np.isfinite(delta):
            positive_deltas.append(delta)

    if len(positive_deltas) >= max(2, samples // 8):
        typical_uphill = float(np.median(positive_deltas))
        return max(typical_uphill / -np.log(target_uphill_acceptance), 1e-9)

    interquartile_range = float(np.percentile(energies, 75) - np.percentile(energies, 25))
    robust_scale = max(interquartile_range, float(np.median(np.abs(energies))) * 0.01, 1e-9)
    return robust_scale


def simulated_annealing(
    initial: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    objective: Callable[[np.ndarray], float],
    *,
    config: AnnealingConfig,
    seed: int,
    propose: Callable[[np.ndarray, float, Generator], np.ndarray] | None = None,
    include_trace: bool = False,
) -> AnnealingResult:
    current = np.asarray(initial, dtype=float).copy()
    if current.ndim != 1 or current.size == 0 or not np.isfinite(current).all():
        raise ValueError("initial annealing parameters must be a finite non-empty vector")
    lower, upper = _validated_bounds(current, bounds)
    rng = default_rng(seed)
    current_energy = _finite_energy(objective, current)
    initial_energy = current_energy
    evaluations = 1

    def propose_bounded(parameters: np.ndarray, sigma: float, generator: Generator) -> np.ndarray:
        proposal = (
            _default_proposal(parameters, sigma, generator, lower, upper)
            if propose is None
            else np.asarray(propose(parameters.copy(), sigma, generator), dtype=float)
        )
        if proposal.shape != current.shape or not np.isfinite(proposal).all():
            raise ValueError("annealing proposer returned an invalid vector")
        return _bounded(proposal, lower, upper)

    initial_temperature = calibrate_temperature(
        current,
        objective,
        propose_bounded,
        samples=config.initial_samples,
        target_uphill_acceptance=config.target_uphill_acceptance,
        rng=rng,
        initial_energy=current_energy,
    )
    evaluations += config.initial_samples
    temperature = initial_temperature
    best = current.copy()
    best_energy = current_energy
    best_step = 0
    accepted_moves = 0
    uphill_accepted_moves = 0
    accepted_since_best = 0
    proposals_since_best = 0
    reheats = 0
    restarts = 0
    trace = []

    for step in range(1, config.steps + 1):
        restart = proposals_since_best >= config.restart_patience
        if restart:
            current = best.copy()
            current_energy = best_energy
            proposals_since_best = 0
            accepted_since_best = 0
            restarts += 1
            temperature = max(temperature, 0.25 * initial_temperature)
            proposal = _heavy_tailed_proposal(current, config.sigma_max, rng, lower, upper)
        else:
            sigma = config.sigma_min + (config.sigma_max - config.sigma_min) * np.sqrt(
                max(0.0, min(1.0, temperature / initial_temperature))
            )
            proposal = propose_bounded(current, float(sigma), rng)

        proposal_energy = _finite_energy(objective, proposal)
        evaluations += 1
        delta = proposal_energy - current_energy
        accepted = metropolis_accept(delta, temperature, rng)
        if accepted:
            current = proposal.copy()
            current_energy = proposal_energy
            accepted_moves += 1
            if delta > 0.0:
                uphill_accepted_moves += 1

        if proposal_energy < best_energy:
            best = proposal.copy()
            best_energy = proposal_energy
            best_step = step
            accepted_since_best = 0
            proposals_since_best = 0
        else:
            proposals_since_best += 1
            if accepted:
                accepted_since_best += 1

        if accepted_since_best >= config.reheat_patience:
            temperature = max(temperature, 0.25 * initial_temperature)
            accepted_since_best = 0
            reheats += 1

        if include_trace:
            trace.append(
                AnnealingIteration(
                    step=step,
                    temperature=float(temperature),
                    current_energy=float(current_energy),
                    proposal_energy=float(proposal_energy),
                    accepted=accepted,
                    best_energy=float(best_energy),
                )
            )
        temperature = max(initial_temperature * 1e-12, temperature * config.cooling)

    trace_summary = {
        "steps": float(config.steps),
        "evaluations": float(evaluations),
        "accepted_moves": float(accepted_moves),
        "uphill_accepted_moves": float(uphill_accepted_moves),
        "acceptance_rate": float(accepted_moves / config.steps),
        "initial_temperature": float(initial_temperature),
        "final_temperature": float(temperature),
        "reheats": float(reheats),
        "restarts": float(restarts),
        "best_step": float(best_step),
        "initial_energy": float(initial_energy),
        "best_energy": float(best_energy),
    }
    return AnnealingResult(
        best_parameters=best.copy(),
        best_energy=float(best_energy),
        current_parameters=current.copy(),
        current_energy=float(current_energy),
        evaluations=evaluations,
        accepted_moves=accepted_moves,
        uphill_accepted_moves=uphill_accepted_moves,
        trace_summary=trace_summary,
        trace=tuple(trace),
    )
