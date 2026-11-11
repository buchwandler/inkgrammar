from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from ..models import StrictModel


class CheckpointCreator(StrictModel):
    inkgrammar_version: str
    git_commit: str | None = None


class CheckpointInputs(StrictModel):
    project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("support_sha256")
    @classmethod
    def validate_support_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in value.values()):
            raise ValueError("support hashes must be SHA-256 values")
        return value


class CheckpointOptimizer(StrictModel):
    backend: str
    seed: int
    objective_version: int = Field(gt=0)


class CheckpointIteration(StrictModel):
    iteration: int = Field(ge=0)
    objective: float
    parameters: dict[str, float] = Field(default_factory=dict)


class CheckpointProgress(StrictModel):
    restart_index: int = Field(default=0, ge=0)
    cycle_index: int = Field(default=0, ge=0)
    evaluations: int = Field(default=0, ge=0)


class CheckpointTraceSummary(StrictModel):
    evaluations: int = Field(default=0, ge=0)
    initial_objective: float | None = None
    best_objective: float | None = None
    last_objective: float | None = None


class _CheckpointFields(StrictModel):
    format: Literal["inkgrammar-fit-checkpoint"] = "inkgrammar-fit-checkpoint"
    project_id: str
    stage: str
    created_by: CheckpointCreator
    inputs: CheckpointInputs
    optimizer: CheckpointOptimizer
    parameters: dict[str, float]
    training_metrics: dict[str, float] = Field(default_factory=dict)
    holdout_metrics: dict[str, float] = Field(default_factory=dict)


class FitCheckpointV1(_CheckpointFields):
    format_version: Literal[1] = 1
    history: list[CheckpointIteration] = Field(default_factory=list)


class FitCheckpoint(_CheckpointFields):
    format_version: Literal[2] = 2
    progress: CheckpointProgress = Field(default_factory=CheckpointProgress)
    trace_summary: CheckpointTraceSummary = Field(default_factory=CheckpointTraceSummary)


def summarize_trace(
    history: Sequence[CheckpointIteration],
    previous: CheckpointTraceSummary | None = None,
) -> CheckpointTraceSummary:
    objectives = [item.objective for item in history]
    prior = previous or CheckpointTraceSummary()
    combined_best = [value for value in (prior.best_objective, *objectives) if value is not None]
    return CheckpointTraceSummary(
        evaluations=prior.evaluations + len(objectives),
        initial_objective=(
            prior.initial_objective
            if prior.initial_objective is not None
            else objectives[0]
            if objectives
            else None
        ),
        best_objective=min(combined_best) if combined_best else None,
        last_objective=objectives[-1] if objectives else prior.last_objective,
    )


def migrate_checkpoint_v1(checkpoint: FitCheckpointV1) -> FitCheckpoint:
    objectives = [item.objective for item in checkpoint.history]
    return FitCheckpoint(
        project_id=checkpoint.project_id,
        stage=checkpoint.stage,
        created_by=checkpoint.created_by,
        inputs=checkpoint.inputs,
        optimizer=checkpoint.optimizer,
        parameters=checkpoint.parameters,
        training_metrics=checkpoint.training_metrics,
        holdout_metrics=checkpoint.holdout_metrics,
        progress=CheckpointProgress(evaluations=len(objectives)),
        trace_summary=CheckpointTraceSummary(
            evaluations=len(objectives),
            initial_objective=objectives[0] if objectives else None,
            best_objective=min(objectives) if objectives else None,
            last_objective=objectives[-1] if objectives else None,
        ),
    )


def validate_checkpoint_objective_version(checkpoint: FitCheckpoint, expected_version: int) -> None:
    actual_version = checkpoint.optimizer.objective_version
    if actual_version != expected_version:
        raise ValueError(
            f"checkpoint objective version {actual_version} is incompatible with requested "
            f"objective version {expected_version}"
        )


def save_checkpoint(checkpoint: FitCheckpoint, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        checkpoint.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    destination.write_text(text + "\n", encoding="utf-8")


def load_checkpoint(path: str | Path) -> FitCheckpoint:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") == 1:
        return migrate_checkpoint_v1(FitCheckpointV1.model_validate(payload))
    return FitCheckpoint.model_validate(payload)
