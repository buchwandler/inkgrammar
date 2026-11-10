import numpy as np
import pytest

from scriptstrokes.fitting.checkpoint import FitCheckpoint
from scriptstrokes.fitting.models import PrimitiveTemplate
from scriptstrokes.fitting.optimizer import (
    OptimizerConfig,
    checkpoint_from_result,
    optimize_parameters,
    resume_optimization,
)
from scriptstrokes.fitting.parameterization import ParameterSpace


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
        created_by={"scriptstrokes_version": "0.1.0"},
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


def test_checkpoint_resume_uses_accepted_parameters_and_preserves_history():
    space = ParameterSpace.from_primitive(primitive())
    objective = quadratic
    start = checkpoint({"segments.0.c1.x": 0.1, "segments.0.c1.y": 0.4})
    config = OptimizerConfig(max_iterations=100)
    first = optimize_parameters(space, objective, config=config)
    stored = checkpoint_from_result(start, first, config=config)
    resumed = resume_optimization(stored, space, objective)

    np.testing.assert_allclose(resumed.parameters, first.parameters, atol=1e-4)
    assert len(resumed.history) > len(stored.history)
    assert resumed.history[: len(stored.history)] == tuple(stored.history)
    assert resumed.objective < 1e-8


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
