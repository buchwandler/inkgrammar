from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import PageLayout, WritingModel


def read_yaml(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(path: str | Path) -> WritingModel:
    return WritingModel.model_validate(read_yaml(path))


def load_layout(path: str | Path) -> PageLayout:
    return PageLayout.model_validate(read_yaml(path))
