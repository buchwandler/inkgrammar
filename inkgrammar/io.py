from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import PageLayout, RuntimeWritingModel, WritingModelV2, WritingModelV3


def read_yaml(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(path: str | Path) -> RuntimeWritingModel:
    payload = read_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError("writing model must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version == 2:
        return WritingModelV2.model_validate(payload)
    if schema_version == 3:
        return WritingModelV3.model_validate(payload)
    raise ValueError(f"unsupported writing model schema version: {schema_version!r}")


def load_layout(path: str | Path) -> PageLayout:
    return PageLayout.model_validate(read_yaml(path))
