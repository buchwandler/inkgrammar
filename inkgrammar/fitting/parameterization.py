from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from pydantic import BaseModel

from .models import PrimitiveTemplate, _path_value


def _coordinate_index(part: str) -> int:
    if part == "x":
        return 0
    if part == "y":
        return 1
    return int(part)


def _set_path(value: Any, parts: list[str], replacement: float) -> Any:
    if not parts:
        return replacement
    part, *rest = parts
    if isinstance(value, Mapping):
        updated = dict(value)
        updated[part] = _set_path(updated[part], rest, replacement)
        return updated
    if isinstance(value, (list, tuple)):
        index = _coordinate_index(part)
        updated = list(value)
        updated[index] = _set_path(updated[index], rest, replacement)
        return tuple(updated) if isinstance(value, tuple) else updated
    raise KeyError(".".join(parts))


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    initial: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        if not all(isfinite(value) for value in (self.lower, self.upper, self.initial, self.scale)):
            raise ValueError("parameter values must be finite")
        if self.lower >= self.upper:
            raise ValueError("parameter lower bound must be less than upper bound")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("parameter initial value must be within bounds")
        if self.scale <= 0:
            raise ValueError("parameter scale must be positive")


@dataclass(frozen=True)
class ParameterSpace:
    specs: tuple[ParameterSpec, ...]

    def __init__(self, specs: Sequence[ParameterSpec]):
        specs = tuple(specs)
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        object.__setattr__(self, "specs", specs)

    @classmethod
    def from_primitive(cls, primitive: PrimitiveTemplate) -> ParameterSpace:
        values = primitive.model_dump(mode="python")
        specs = []
        for name in primitive.free_parameters:
            lower, upper = primitive.bounds[name]
            initial = float(_path_value(values, name))
            specs.append(
                ParameterSpec(
                    name=name,
                    lower=lower,
                    upper=upper,
                    initial=initial,
                    scale=upper - lower,
                )
            )
        return cls(specs)

    def encode(self, state: BaseModel | Mapping[str, Any]) -> np.ndarray:
        values = state.model_dump(mode="python") if isinstance(state, BaseModel) else state
        vector = np.asarray([_path_value(values, spec.name) for spec in self.specs], dtype=float)
        self._validate_vector(vector)
        return vector

    def decode(
        self,
        vector: Sequence[float] | np.ndarray,
        state: BaseModel | Mapping[str, Any],
    ) -> BaseModel | dict[str, Any]:
        values = np.asarray(vector, dtype=float)
        self._validate_vector(values)
        result = state.model_dump(mode="python") if isinstance(state, BaseModel) else dict(state)
        for spec, value in zip(self.specs, values, strict=True):
            result = _set_path(result, spec.name.split("."), float(value))
        if isinstance(state, BaseModel):
            return type(state).model_validate(result)
        return result

    def bounds(self) -> list[tuple[float, float]]:
        return [(spec.lower, spec.upper) for spec in self.specs]

    def _validate_vector(self, vector: np.ndarray) -> None:
        if vector.ndim != 1 or len(vector) != len(self.specs):
            raise ValueError(f"parameter vector must have shape ({len(self.specs)},)")
        if not np.isfinite(vector).all():
            raise ValueError("parameter vector must contain only finite values")
        for spec, value in zip(self.specs, vector, strict=True):
            if not spec.lower <= value <= spec.upper:
                raise ValueError(f"parameter is outside bounds: {spec.name}")
