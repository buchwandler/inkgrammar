import numpy as np
import pytest
from scipy.optimize import minimize

from inkgrammar.fitting.annealing import (
    AnnealingConfig,
    calibrate_temperature,
    metropolis_accept,
    simulated_annealing,
)


def test_annealing_is_deterministic_and_keeps_compact_summary():
    def objective(parameters):
        return float(np.sum((parameters - 0.25) ** 2))

    config = AnnealingConfig(steps=30, initial_samples=12, cooling=0.95)
    arguments = (np.zeros(3), [(-1.0, 1.0)] * 3, objective)

    first = simulated_annealing(*arguments, config=config, seed=41, include_trace=True)
    second = simulated_annealing(*arguments, config=config, seed=41, include_trace=True)

    np.testing.assert_array_equal(first.best_parameters, second.best_parameters)
    np.testing.assert_array_equal(first.current_parameters, second.current_parameters)
    assert first.best_energy == second.best_energy
    assert first.trace_summary == second.trace_summary
    assert first.trace == second.trace
    assert first.evaluations == 1 + config.initial_samples + config.steps
    assert first.trace_summary["evaluations"] == first.evaluations
    assert len(first.trace) == config.steps


def test_temperature_calibration_uses_median_uphill_energy():
    temperature = calibrate_temperature(
        np.array([0.0]),
        lambda parameters: float(parameters[0] ** 2),
        lambda initial, _scale, _rng: np.array([0.5]),
        samples=8,
        target_uphill_acceptance=0.8,
        rng=np.random.default_rng(4),
    )

    assert temperature == pytest.approx(0.25 / -np.log(0.8))


def test_annealing_accepts_an_uphill_barrier_move_and_finds_lower_basin():
    def objective(parameters):
        return float((parameters[0] ** 2 - 1.0) ** 2 - 0.2 * parameters[0])

    samples = 8
    calls = 0

    def proposal(current, _sigma, _rng):
        nonlocal calls
        calls += 1
        if calls <= samples:
            return current + 0.1
        return np.array([0.0]) if calls == samples + 1 else np.array([1.0])

    local = minimize(objective, np.array([-1.0]), method="L-BFGS-B", bounds=[(-2.0, 2.0)])
    result = simulated_annealing(
        np.array([-1.0]),
        [(-2.0, 2.0)],
        objective,
        config=AnnealingConfig(
            steps=2,
            initial_samples=samples,
            target_uphill_acceptance=0.999,
            reheat_patience=20,
            restart_patience=40,
        ),
        seed=7,
        propose=proposal,
        include_trace=True,
    )

    assert local.x[0] < 0.0
    assert result.trace[0].proposal_energy > result.trace_summary["initial_energy"]
    assert result.trace[0].accepted is True
    assert result.best_parameters[0] > 0.0
    assert result.best_energy < local.fun
    assert result.uphill_accepted_moves >= 1


def test_all_evaluated_custom_proposals_are_clipped_to_bounds():
    evaluated = []

    def objective(parameters):
        evaluated.append(parameters.copy())
        return float(np.sum(parameters**2))

    result = simulated_annealing(
        np.array([0.0, 0.0]),
        [(-1.0, 1.0), (-2.0, 2.0)],
        objective,
        config=AnnealingConfig(steps=5, initial_samples=4),
        seed=2,
        propose=lambda current, _sigma, _rng: current + 100.0,
    )

    assert all(-1.0 <= item[0] <= 1.0 and -2.0 <= item[1] <= 2.0 for item in evaluated)
    assert np.all(result.best_parameters >= [-1.0, -2.0])
    assert np.all(result.best_parameters <= [1.0, 2.0])


def test_best_ever_state_is_retained_across_worse_accepted_moves():
    calls = 0

    def proposal(current, _sigma, _rng):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return current.copy()
        return np.array([0.0]) if calls == 5 else np.array([0.9])

    result = simulated_annealing(
        np.array([0.5]),
        [(-1.0, 1.0)],
        lambda parameters: float(parameters[0] ** 2),
        config=AnnealingConfig(
            steps=2,
            initial_samples=4,
            target_uphill_acceptance=0.999,
            reheat_patience=10,
            restart_patience=20,
        ),
        seed=5,
        propose=proposal,
    )

    assert result.best_parameters[0] == 0.0
    assert result.best_energy == 0.0
    assert result.current_energy >= result.best_energy


def test_reheating_and_heavy_tailed_restarts_are_counted():
    result = simulated_annealing(
        np.array([0.0]),
        [(-1.0, 1.0)],
        lambda _parameters: 0.0,
        config=AnnealingConfig(
            steps=6,
            initial_samples=4,
            reheat_patience=1,
            restart_patience=2,
        ),
        seed=9,
        propose=lambda current, _sigma, _rng: current,
    )

    assert result.trace_summary["reheats"] >= 1.0
    assert result.trace_summary["restarts"] >= 1.0
    assert result.trace_summary["evaluations"] == result.evaluations


def test_non_finite_objective_and_invalid_config_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        simulated_annealing(
            np.array([0.0]),
            [(-1.0, 1.0)],
            lambda _parameters: np.inf,
            config=AnnealingConfig(steps=1, initial_samples=1),
            seed=1,
        )
    with pytest.raises(ValueError, match="cooling"):
        AnnealingConfig(cooling=1.0)


def test_metropolis_accepts_improvements_and_probabilistic_uphill_moves():
    rng = np.random.default_rng(0)

    assert metropolis_accept(-1.0, 0.1, rng) is True
    assert metropolis_accept(0.1, 100.0, rng) is True
    assert metropolis_accept(1.0, 0.0, rng) is False
