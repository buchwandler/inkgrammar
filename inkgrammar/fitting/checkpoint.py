from __future__ import annotations

import json
import re
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


class FitCheckpoint(StrictModel):
    format: Literal["inkgrammar-fit-checkpoint"] = "inkgrammar-fit-checkpoint"
    format_version: Literal[1] = 1
    project_id: str
    stage: str
    created_by: CheckpointCreator
    inputs: CheckpointInputs
    optimizer: CheckpointOptimizer
    parameters: dict[str, float]
    training_metrics: dict[str, float] = Field(default_factory=dict)
    holdout_metrics: dict[str, float] = Field(default_factory=dict)
    history: list[CheckpointIteration] = Field(default_factory=list)


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
    return FitCheckpoint.model_validate_json(Path(path).read_text(encoding="utf-8"))
