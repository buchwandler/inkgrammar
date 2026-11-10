from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from pydantic import BaseModel

from .models import OccurrenceFitState, PrimitiveTemplate, _path_value


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

    def encode_optimizer_space(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        """Map raw parameter coordinates to the normalized optimizer interval [-1, 1]."""
        values = np.asarray(vector, dtype=float)
        self._validate_vector(values)
        lower = np.asarray([spec.lower for spec in self.specs], dtype=float)
        upper = np.asarray([spec.upper for spec in self.specs], dtype=float)
        return 2.0 * (values - lower) / (upper - lower) - 1.0

    def decode_optimizer_space(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        """Map normalized optimizer coordinates back to raw parameter coordinates."""
        normalized = np.asarray(vector, dtype=float)
        self._validate_optimizer_vector(normalized)
        lower = np.asarray([spec.lower for spec in self.specs], dtype=float)
        upper = np.asarray([spec.upper for spec in self.specs], dtype=float)
        return lower + (normalized + 1.0) * (upper - lower) / 2.0

    def normalized_bounds(self) -> list[tuple[float, float]]:
        return [(-1.0, 1.0) for _ in self.specs]

    def _validate_optimizer_vector(self, vector: np.ndarray) -> None:
        if vector.ndim != 1 or len(vector) != len(self.specs):
            raise ValueError(f"optimizer vector must have shape ({len(self.specs)},)")
        if not np.isfinite(vector).all():
            raise ValueError("optimizer vector must contain only finite values")
        if np.any(vector < -1.0) or np.any(vector > 1.0):
            raise ValueError("optimizer vector is outside normalized bounds")

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


@dataclass(frozen=True)
class _OccurrenceParameter:
    instance_id: str
    name: str
    spec: ParameterSpec


@dataclass(frozen=True)
class CompositeParameterSpace:
    """Explicit shared-shape and occurrence-delta coordinates for family fitting."""

    shared_space: ParameterSpace
    occurrence_states: tuple[OccurrenceFitState, ...]
    occurrence_parameters: tuple[_OccurrenceParameter, ...]
    _space: ParameterSpace

    def __init__(
        self,
        shared_space: ParameterSpace,
        occurrence_states: Sequence[OccurrenceFitState],
    ) -> None:
        states = tuple(occurrence_states)
        ids = [state.instance_id for state in states]
        if len(set(ids)) != len(ids):
            raise ValueError("occurrence instance IDs must be unique")
        shared_specs = [
            ParameterSpec(
                name=f"shared.{spec.name}",
                lower=spec.lower,
                upper=spec.upper,
                initial=spec.initial,
                scale=spec.scale,
            )
            for spec in shared_space.specs
        ]
        occurrence_parameters = []
        for state in states:
            for name in state.free_parameters:
                lower, upper = getattr(state.bounds, name)
                initial = getattr(state.delta, name)
                spec = ParameterSpec(
                    name=f"occurrence.{state.instance_id}.delta.{name}",
                    lower=lower,
                    upper=upper,
                    initial=initial,
                    scale=upper - lower,
                )
                occurrence_parameters.append(_OccurrenceParameter(state.instance_id, name, spec))
        object.__setattr__(self, "shared_space", shared_space)
        object.__setattr__(self, "occurrence_states", states)
        object.__setattr__(self, "occurrence_parameters", tuple(occurrence_parameters))
        composite_specs = [*shared_specs, *(item.spec for item in occurrence_parameters)]
        object.__setattr__(self, "_space", ParameterSpace(composite_specs))

    @classmethod
    def from_primitive(
        cls,
        primitive: PrimitiveTemplate,
        occurrence_states: Sequence[OccurrenceFitState],
    ) -> CompositeParameterSpace:
        return cls(ParameterSpace.from_primitive(primitive), occurrence_states)

    @property
    def specs(self) -> tuple[ParameterSpec, ...]:
        return self._space.specs

    def encode(
        self,
        primitive: PrimitiveTemplate,
        occurrence_states: Sequence[OccurrenceFitState] | None = None,
    ) -> np.ndarray:
        states = tuple(self.occurrence_states if occurrence_states is None else occurrence_states)
        self._validate_occurrences(states)
        values = [*self.shared_space.encode(primitive).tolist()]
        states_by_id = {state.instance_id: state for state in states}
        values.extend(
            float(getattr(states_by_id[item.instance_id].delta, item.name))
            for item in self.occurrence_parameters
        )
        vector = np.asarray(values, dtype=float)
        self._space._validate_vector(vector)
        return vector

    def decode(
        self,
        vector: Sequence[float] | np.ndarray,
        primitive: PrimitiveTemplate,
        occurrence_states: Sequence[OccurrenceFitState] | None = None,
    ) -> tuple[PrimitiveTemplate, tuple[OccurrenceFitState, ...]]:
        states = tuple(self.occurrence_states if occurrence_states is None else occurrence_states)
        self._validate_occurrences(states)
        values = np.asarray(vector, dtype=float)
        self._space._validate_vector(values)
        shared_count = len(self.shared_space.specs)
        decoded_primitive = self.shared_space.decode(values[:shared_count], primitive)
        assert isinstance(decoded_primitive, PrimitiveTemplate)
        updates: dict[str, dict[str, float]] = {}
        for item, value in zip(self.occurrence_parameters, values[shared_count:], strict=True):
            updates.setdefault(item.instance_id, {})[item.name] = float(value)
        decoded_states = []
        for state in states:
            data = state.model_dump(mode="python")
            delta = dict(data["delta"])
            delta.update(updates.get(state.instance_id, {}))
            data["delta"] = delta
            decoded_states.append(OccurrenceFitState.model_validate(data))
        return decoded_primitive, tuple(decoded_states)

    def encode_optimizer_space(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        return self._space.encode_optimizer_space(vector)

    def decode_optimizer_space(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        return self._space.decode_optimizer_space(vector)

    def normalized_bounds(self) -> list[tuple[float, float]]:
        return self._space.normalized_bounds()

    def bounds(self) -> list[tuple[float, float]]:
        return self._space.bounds()

    def _validate_vector(self, vector: np.ndarray) -> None:
        self._space._validate_vector(vector)

    def _validate_occurrences(self, states: Sequence[OccurrenceFitState]) -> None:
        expected = {state.instance_id for state in self.occurrence_states}
        actual = {state.instance_id for state in states}
        if len(actual) != len(states) or actual != expected:
            raise ValueError("occurrence states do not match composite parameter space")
