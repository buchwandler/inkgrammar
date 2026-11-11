from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos, degrees, hypot

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..models import RuntimeAnchor
from .models import ConnectionGeometryTarget, ConnectionObservation

Point = tuple[float, float]


@dataclass(frozen=True)
class ConnectionFitWeights:
    anchor_position: float = 1.0
    tangent: float = 0.25
    centerline: float = 0.5
    width: float = 0.25
    gap: float = 0.5
    rule_prior: float = 0.02
    position_scale: float = 1.0
    width_scale: float = 1.0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.anchor_position,
                self.tangent,
                self.centerline,
                self.width,
                self.gap,
                self.rule_prior,
            )
        ):
            raise ValueError("connection loss weights must be nonnegative")
        if self.position_scale <= 0 or self.width_scale <= 0:
            raise ValueError("connection loss scales must be positive")


@dataclass(frozen=True)
class ConnectionFitResult:
    observation_id: str
    split: str
    x_adjust: float
    y_adjust: float
    objective: float
    diagnostics: dict[str, float | str | bool | None]
    rule_proposal: dict

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "split": self.split,
            "x_adjust": self.x_adjust,
            "y_adjust": self.y_adjust,
            "objective": self.objective,
            "diagnostics": self.diagnostics,
            "rule_proposal": self.rule_proposal,
        }


def _unit(vector: Point) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(value))
    if length == 0:
        raise ValueError("connection tangent must be nonzero")
    return value / length


def _angle(left: Point, right: Point) -> float:
    dot = float(np.clip(np.dot(_unit(left), _unit(right)), -1.0, 1.0))
    return acos(dot)


def diagnose_connection(
    left_exit: RuntimeAnchor,
    right_entry: RuntimeAnchor,
    *,
    target: ConnectionGeometryTarget | None = None,
    x_adjust: float = 0.0,
    y_adjust: float = 0.0,
    mode: str = "cursive-anchor",
    rule_id: str | None = None,
    left_variant: str | None = None,
    right_variant: str | None = None,
) -> dict[str, float | str | bool | None]:
    delta = (
        right_entry.x + x_adjust - left_exit.x,
        right_entry.y + y_adjust - left_exit.y,
    )
    anchor_gap = hypot(*delta)
    tangent_error = degrees(
        _angle(
            (left_exit.tangent_x, left_exit.tangent_y),
            (right_entry.tangent_x, right_entry.tangent_y),
        )
    )
    width_mismatch = (
        abs(left_exit.width - right_entry.width)
        if left_exit.width is not None and right_entry.width is not None
        else None
    )
    anchor_target_error = None
    gap_error = None
    tangent_target_error = None
    width_target_error = None
    if target is not None:
        anchor_target_error = hypot(
            delta[0] - target.anchor_delta[0], delta[1] - target.anchor_delta[1]
        )
        gap_error = abs(anchor_gap - hypot(*target.anchor_delta))
        if target.exit_tangent is not None and target.entry_tangent is not None:
            tangent_target_error = degrees(
                (
                    _angle((left_exit.tangent_x, left_exit.tangent_y), target.exit_tangent)
                    + _angle((right_entry.tangent_x, right_entry.tangent_y), target.entry_tangent)
                )
                / 2.0
            )
        if (
            target.exit_width is not None
            and target.entry_width is not None
            and width_mismatch is not None
        ):
            target_width_mismatch = abs(target.exit_width - target.entry_width)
            width_target_error = abs(width_mismatch - target_width_mismatch)
    return {
        "rule_id": rule_id,
        "left_variant": left_variant,
        "right_variant": right_variant,
        "mode": mode,
        "anchor_gap": anchor_gap,
        "anchor_target_error": anchor_target_error,
        "tangent_error_deg": tangent_error,
        "tangent_target_error_deg": tangent_target_error,
        "width_mismatch": width_mismatch,
        "width_error": width_target_error if width_target_error is not None else width_mismatch,
        "width_target_error": width_target_error,
        "gap_error": gap_error,
        "font_compatible": mode != "synthetic-bridge",
    }


def fit_connection_observation(
    observation: ConnectionObservation,
    left_exit: RuntimeAnchor,
    right_entry: RuntimeAnchor,
    *,
    initial_adjustment: Point = (0.0, 0.0),
    generated_centerline: Sequence[Point] = (),
    mode: str = "cursive-anchor",
    rule_id: str | None = None,
    left_variant: str | None = None,
    right_variant: str | None = None,
    weights: ConnectionFitWeights | None = None,
) -> ConnectionFitResult:
    if observation.split != "fit":
        raise ValueError("connection fitting accepts only fit-split observations")
    weights = weights or ConnectionFitWeights()
    target = observation.target_geometry
    if target is None:
        raise ValueError("connection observation has no reviewed target geometry")
    candidate_curve = np.asarray(generated_centerline, dtype=float)
    target_curve = np.asarray(target.centerline, dtype=float)
    compare_centerline = len(candidate_curve) >= 2 and len(target_curve) >= 2
    target_tree = cKDTree(target_curve) if compare_centerline else None

    def residuals(adjustment: np.ndarray) -> np.ndarray:
        dx, dy = map(float, adjustment)
        actual_delta = np.asarray(
            [right_entry.x + dx - left_exit.x, right_entry.y + dy - left_exit.y]
        )
        target_delta = np.asarray(target.anchor_delta)
        values = [
            np.sqrt(weights.anchor_position)
            * (actual_delta - target_delta)
            / weights.position_scale
        ]
        if weights.gap:
            gap_residual = np.linalg.norm(actual_delta) - np.linalg.norm(target_delta)
            values.append(
                np.asarray([np.sqrt(weights.gap) * gap_residual / weights.position_scale])
            )
        if weights.tangent and target.exit_tangent is not None and target.entry_tangent is not None:
            values.append(
                np.sqrt(weights.tangent)
                * np.asarray(
                    [
                        _angle((left_exit.tangent_x, left_exit.tangent_y), target.exit_tangent),
                        _angle(
                            (right_entry.tangent_x, right_entry.tangent_y), target.entry_tangent
                        ),
                    ]
                )
            )
        if (
            weights.width
            and left_exit.width is not None
            and right_entry.width is not None
            and target.exit_width is not None
            and target.entry_width is not None
        ):
            actual_width_delta = abs(left_exit.width - right_entry.width)
            target_width_delta = abs(target.exit_width - target.entry_width)
            values.append(
                np.asarray(
                    [
                        np.sqrt(weights.width)
                        * (actual_width_delta - target_width_delta)
                        / weights.width_scale
                    ]
                )
            )
        if weights.centerline and compare_centerline and target_tree is not None:
            shifted_curve = candidate_curve + np.asarray([dx, dy])
            candidate_to_target = target_tree.query(shifted_curve)[0]
            target_to_candidate = cKDTree(shifted_curve).query(target_curve)[0]
            values.append(
                np.sqrt(weights.centerline)
                * np.concatenate((candidate_to_target, target_to_candidate))
                / weights.position_scale
            )
        if weights.rule_prior:
            values.append(
                np.sqrt(weights.rule_prior)
                * (adjustment - np.asarray(initial_adjustment))
                / weights.position_scale
            )
        return np.concatenate(values)

    result = least_squares(residuals, np.asarray(initial_adjustment, dtype=float))
    x_adjust, y_adjust = map(float, result.x)
    diagnostics = diagnose_connection(
        left_exit,
        right_entry,
        target=target,
        x_adjust=x_adjust,
        y_adjust=y_adjust,
        mode=mode,
        rule_id=rule_id,
        left_variant=left_variant,
        right_variant=right_variant,
    )
    return ConnectionFitResult(
        observation_id=observation.id,
        split=observation.split,
        x_adjust=x_adjust,
        y_adjust=y_adjust,
        rule_proposal={
            "id": f"proposal-{observation.id}",
            "from_class": left_exit.class_,
            "to_class": right_entry.class_,
            "mode": mode,
            "x_adjust": 0.0 if mode == "attached" else x_adjust,
            "y_adjust": y_adjust,
            "pair_spacing": x_adjust if mode == "attached" else 0.0,
            "evidence": [observation.id],
            "status": "inferred",
        },
        objective=float(np.dot(result.fun, result.fun)),
        diagnostics=diagnostics,
    )


def connection_observation_coverage(
    observations: Sequence[ConnectionObservation],
) -> dict:
    splits = ("fit", "validation", "hold-out")
    split_counts = {split: 0 for split in splits}
    pairs: dict[str, dict[str, int]] = {}
    for observation in observations:
        split_counts[observation.split] += 1
        pair = observation.from_character + observation.to_character
        counts = pairs.setdefault(pair, {split: 0 for split in splits})
        counts[observation.split] += 1
    return {
        "observation_count": len(observations),
        "split_counts": split_counts,
        "pair_splits": {pair: pairs[pair] for pair in sorted(pairs)},
        "target_geometry_count": sum(
            observation.target_geometry is not None for observation in observations
        ),
        "source_free": True,
    }
