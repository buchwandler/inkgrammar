import numpy as np
import pytest

from inkgrammar.fitting.checkpoint import (
    FitCheckpoint,
    FitCheckpointV1,
    migrate_checkpoint_v1,
)
from inkgrammar.fitting.models import PrimitiveTemplate
from inkgrammar.fitting.optimizer import (
    OptimizerConfig,
    checkpoint_from_result,
    optimize_parameters,
    resume_optimization,
    save_optimization_trace,
)
from inkgrammar.fitting.parameterization import ParameterSpace


def primitive():
    return PrimitiveTemplate.model_validate(
        {
            "id": "synthetic-cubic",
            "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
            "segments": [{"c1": [0.1, 0.4], "c2": [0.3, 0.8], "end": [0.5, 1.0]}],
            "free_parameters": ["segments.0.c1.x", "segments.0.c1.y"],
            "bounds": {
                "segments.0.c1.x": [-0.5, 0.5],
                "segments.0.c1.y": [0.0, 1.0],
            },
        }
    )


def checkpoint(parameters, backend="scipy-powell"):
    digest = "a" * 64
    return FitCheckpoint(
        project_id="synthetic",
        stage="primitive:synthetic-cubic",
        created_by={"inkgrammar_version": "0.1.0"},
        inputs={
            "project_sha256": digest,
            "corpus_sha256": "b" * 64,
            "source_lock_sha256": "c" * 64,
        },
        optimizer={"backend": backend, "seed": 42, "objective_version": 1},
        parameters=parameters,
    )


def quadratic(vector):
    return float(np.sum((vector - [0.25, 0.7]) ** 2))


def test_powell_finds_bounded_quadratic_minimum_deterministically():
    space = ParameterSpace.from_primitive(primitive())
    objective = quadratic
    config = OptimizerConfig(seed=42, max_iterations=100)

    first = optimize_parameters(space, objective, config=config)
    second = optimize_parameters(space, objective, config=config)

    np.testing.assert_allclose(first.parameters, [0.25, 0.7], atol=1e-4)
    np.testing.assert_allclose(first.parameters, second.parameters, atol=0)
    assert first.objective < 1e-8
    assert all(-0.5 <= row.parameters["segments.0.c1.x"] <= 0.5 for row in first.history)


def test_differential_evolution_initialization_is_seeded_and_dimension_bounded():
    space = ParameterSpace.from_primitive(primitive())
    objective = quadratic
    config = OptimizerConfig(
        method="differential_evolution+powell",
        seed=1968,
        max_iterations=100,
        de_max_iterations=5,
        de_population_size=5,
    )

    first = optimize_parameters(space, objective, config=config)
    second = optimize_parameters(space, objective, config=config)

    np.testing.assert_allclose(first.parameters, second.parameters, atol=0)
    assert first.objective < 1e-8
    metadata = checkpoint({"segments.0.c1.x": 0.1, "segments.0.c1.y": 0.4})
    fitted = checkpoint_from_result(metadata, first, config=config)
    assert fitted.optimizer.backend == "scipy-differential-evolution+powell"
    with pytest.raises(ValueError, match="limited to"):
        optimize_parameters(
            space,
            objective,
            config=config.model_copy(update={"de_dimension_limit": 1}),
        )


def test_checkpoint_resume_uses_accepted_parameters_without_persisting_history():
    space = ParameterSpace.from_primitive(primitive())
    objective = quadratic
    start = checkpoint({"segments.0.c1.x": 0.1, "segments.0.c1.y": 0.4})
    config = OptimizerConfig(max_iterations=100)
    first = optimize_parameters(space, objective, config=config)
    stored = checkpoint_from_result(start, first, config=config)
    resumed = resume_optimization(stored, space, objective)

    np.testing.assert_allclose(resumed.parameters, first.parameters, atol=1e-4)
    assert stored.trace_summary.evaluations == first.evaluations
    assert resumed.evaluations == len(resumed.history)
    assert resumed.objective < 1e-8


def test_resume_after_v1_migration_starts_from_accepted_parameters():
    space = ParameterSpace.from_primitive(primitive())
    accepted = {"segments.0.c1.x": 0.1, "segments.0.c1.y": 0.4}
    payload = checkpoint(accepted).model_dump(mode="json")
    payload.pop("progress")
    payload.pop("trace_summary")
    payload["format_version"] = 1
    payload["history"] = [
        {
            "iteration": 0,
            "objective": 0.5,
            "parameters": {"segments.0.c1.x": -0.4, "segments.0.c1.y": 0.2},
        }
    ]
    migrated = migrate_checkpoint_v1(FitCheckpointV1.model_validate(payload))
    first_candidates = []

    def residuals(vector):
        first_candidates.append(vector.copy())
        return vector - np.array([0.25, 0.7])

    result = resume_optimization(
        migrated,
        space,
        residuals,
        config=OptimizerConfig(method="least_squares", loss="linear"),
    )

    np.testing.assert_allclose(
        first_candidates[0],
        [accepted["segments.0.c1.x"], accepted["segments.0.c1.y"]],
    )
    np.testing.assert_allclose(result.parameters, [0.25, 0.7], atol=1e-7)


def test_optional_optimizer_trace_is_written_as_compressed_arrays(tmp_path):
    result = optimize_parameters(
        ParameterSpace.from_primitive(primitive()),
        quadratic,
        config=OptimizerConfig(max_iterations=20),
    )
    path = tmp_path / "work" / "traces" / "synthetic.npz"

    save_optimization_trace(result, path)

    with np.load(path) as trace:
        assert trace["objective"].shape == (len(result.history),)
        assert trace["parameters"].shape == (len(result.history), len(result.parameter_names))
        assert trace["accepted"].sum() == 1


def test_resume_rejects_checkpoint_with_different_parameter_names():
    space = ParameterSpace.from_primitive(primitive())
    source = checkpoint({"different.parameter": 0.2})

    unsupported = checkpoint(
        {"segments.0.c1.x": 0.1, "segments.0.c1.y": 0.4},
        backend="unknown-backend",
    )
    with pytest.raises(ValueError, match="unsupported checkpoint optimizer backend"):
        resume_optimization(unsupported, space, lambda vector: float(vector.sum()))
    with pytest.raises(ValueError, match="parameter names do not match"):
        resume_optimization(source, space, lambda vector: float(vector.sum()))


def test_optimizer_accepts_best_evaluated_candidate_over_worse_solver_result(monkeypatch):
    from types import SimpleNamespace

    import scipy.optimize

    space = ParameterSpace.from_primitive(primitive())
    calls = []

    def fake_minimize(function, initial, **kwargs):
        better = np.array([0.5, 0.4])
        worse = np.array([-0.9, -0.8])
        calls.append(function(better))
        calls.append(function(worse))
        return SimpleNamespace(x=worse, success=True, status=0, message="finished")

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)
    result = optimize_parameters(space, quadratic)

    np.testing.assert_allclose(result.parameters, [0.25, 0.7])
    np.testing.assert_allclose(result.solver_parameters, [-0.45, 0.1])
    assert result.objective < result.solver_objective
    assert result.accepted_history_index < len(result.history) - 1
    assert result.success is True
    assert result.status == 0
    assert result.backend == "scipy-powell"


def test_least_squares_backend_fits_residual_vector():
    space = ParameterSpace.from_primitive(primitive())
    result = optimize_parameters(
        space,
        lambda vector: vector - np.array([0.25, 0.7]),
        config=OptimizerConfig(method="least_squares", loss="linear"),
    )

    np.testing.assert_allclose(result.parameters, [0.25, 0.7], atol=1e-7)
    assert result.objective < 1e-12
    assert result.backend == "scipy-least-squares-trf"
    assert result.status is not None
    assert result.success is True


def test_spline_alt_lsq_configuration_exposes_alternating_cycle_controls():
    config = OptimizerConfig(method="spline-alt-lsq")

    assert config.max_cycles == 6
    assert config.transform_max_nfev == 120
    assert config.irls_iterations == 3
    assert config.assignment_update_interval == 1
