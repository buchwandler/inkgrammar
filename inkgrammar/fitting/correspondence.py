from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centerline import TargetTrace


@dataclass(frozen=True)
class MonotoneCorrespondence:
    candidate_index: np.ndarray
    target_index: np.ndarray
    confidence: np.ndarray
    path: np.ndarray
    cost: float


def _unit_tangents(points: np.ndarray) -> np.ndarray:
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    return np.divide(
        tangents,
        norms[:, None],
        out=np.zeros_like(tangents),
        where=norms[:, None] > 1e-9,
    )


def monotone_correspondence(
    candidate_xy: np.ndarray,
    target: TargetTrace,
    *,
    candidate_tangent: np.ndarray | None = None,
    candidate_radius: np.ndarray | None = None,
    sigma_position_px: float = 2.0,
    sigma_tangent_rad: float = 0.35,
    sigma_radius_px: float = 1.5,
    tangent_weight: float = 0.2,
    radius_weight: float = 0.1,
    warp_penalty: float = 0.05,
    open_target_ends: bool = True,
) -> MonotoneCorrespondence:
    candidate = np.asarray(candidate_xy, dtype=float)
    if candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) < 2:
        raise ValueError("candidate curve must have shape (N, 2) with N >= 2")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate curve coordinates must be finite")
    if (
        not np.isfinite((sigma_position_px, sigma_tangent_rad, sigma_radius_px)).all()
        or min(sigma_position_px, sigma_tangent_rad, sigma_radius_px) <= 0
    ):
        raise ValueError("correspondence sigmas must be positive")
    if (
        not np.isfinite((tangent_weight, radius_weight, warp_penalty)).all()
        or min(tangent_weight, radius_weight, warp_penalty) < 0
    ):
        raise ValueError("correspondence weights must be nonnegative")

    target_xy = np.asarray(target.xy, dtype=float)
    target_tangent = np.asarray(target.tangent, dtype=float)
    target_radius = np.asarray(target.radius, dtype=float)
    target_confidence = np.asarray(target.confidence, dtype=float)
    if (
        target_xy.ndim != 2
        or target_xy.shape[1] != 2
        or target_tangent.shape != target_xy.shape
        or target_radius.shape != (len(target_xy),)
        or target_confidence.shape != (len(target_xy),)
    ):
        raise ValueError("target trace arrays have inconsistent shapes")
    if not np.isfinite(target_confidence).all() or np.any(target_confidence < 0):
        raise ValueError("target confidence must be finite and nonnegative")
    valid_target = (target_confidence > 0) & np.isfinite(target_xy).all(axis=1)
    if (
        not np.isfinite(target_tangent[valid_target]).all()
        or not np.isfinite(target_radius[valid_target]).all()
        or np.any(target_radius[valid_target] < 0)
    ):
        raise ValueError("confident target samples must have finite tangents and nonnegative radii")
    target_indices = np.flatnonzero(valid_target)
    if len(target_indices) < 2:
        raise ValueError("target trace must contain at least two confident points")
    target_xy = target_xy[target_indices]
    target_tangent = target_tangent[target_indices]
    target_radius = target_radius[target_indices]
    target_confidence = target_confidence[target_indices]

    if candidate_tangent is None:
        candidate_tangent = _unit_tangents(candidate)
    else:
        candidate_tangent = np.asarray(candidate_tangent, dtype=float)
        if candidate_tangent.shape != candidate.shape or not np.isfinite(candidate_tangent).all():
            raise ValueError("candidate tangents must be finite and match candidate curve shape")
        norms = np.linalg.norm(candidate_tangent, axis=1)
        candidate_tangent = np.divide(
            candidate_tangent,
            norms[:, None],
            out=np.zeros_like(candidate_tangent),
            where=norms[:, None] > 1e-9,
        )
    if candidate_radius is not None:
        candidate_radius = np.asarray(candidate_radius, dtype=float)
        if (
            candidate_radius.shape != (len(candidate),)
            or not np.isfinite(candidate_radius).all()
            or np.any(candidate_radius < 0)
        ):
            raise ValueError(
                "candidate radii must be finite, nonnegative, and match candidate curve"
            )

    target_tangent_norms = np.linalg.norm(target_tangent, axis=1)
    target_tangent = np.divide(
        target_tangent,
        target_tangent_norms[:, None],
        out=np.zeros_like(target_tangent),
        where=target_tangent_norms[:, None] > 1e-9,
    )
    displacement = candidate[:, None, :] - target_xy[None, :, :]
    position_cost = np.sum(displacement**2, axis=2) / sigma_position_px**2
    tangent_dot = np.clip(candidate_tangent @ target_tangent.T, -1.0, 1.0)
    tangent_valid = (np.linalg.norm(candidate_tangent, axis=1)[:, None] > 0) & (
        target_tangent_norms[None, :] > 1e-9
    )
    angle = np.arccos(tangent_dot)
    tangent_cost = np.where(tangent_valid, (angle / sigma_tangent_rad) ** 2, 0.0)
    local_cost = target_confidence[None, :] * (position_cost + tangent_weight * tangent_cost)
    if candidate_radius is not None:
        radius_cost = (candidate_radius[:, None] - target_radius[None, :]) ** 2 / sigma_radius_px**2
        local_cost += target_confidence[None, :] * radius_weight * radius_cost

    candidate_count, target_count = local_cost.shape
    dynamic = np.full((candidate_count, target_count), np.inf, dtype=float)
    predecessor = np.full((candidate_count, target_count), -1, dtype=np.int8)
    if open_target_ends:
        dynamic[0] = local_cost[0]
        predecessor[0] = 3
    else:
        dynamic[0, 0] = local_cost[0, 0]
        predecessor[0, 0] = 3

    for candidate_index in range(1, candidate_count):
        for target_index in range(target_count):
            options = []
            moves = []
            if target_index > 0:
                options.append(dynamic[candidate_index - 1, target_index - 1])
                moves.append(0)
            options.append(dynamic[candidate_index - 1, target_index] + warp_penalty)
            moves.append(1)
            if target_index > 0:
                options.append(dynamic[candidate_index, target_index - 1] + warp_penalty)
                moves.append(2)
            selected = int(np.argmin(options))
            dynamic[candidate_index, target_index] = (
                options[selected] + local_cost[candidate_index, target_index]
            )
            predecessor[candidate_index, target_index] = moves[selected]

    final_target_index = int(np.argmin(dynamic[-1])) if open_target_ends else target_count - 1
    path = []
    candidate_index = candidate_count - 1
    target_index = final_target_index
    while True:
        path.append((candidate_index, target_index))
        move = predecessor[candidate_index, target_index]
        if move == 3:
            break
        if move == 0:
            candidate_index -= 1
            target_index -= 1
        elif move == 1:
            candidate_index -= 1
        else:
            target_index -= 1
    path = np.asarray(path[::-1], dtype=int)

    mapped_target = np.empty(candidate_count, dtype=int)
    for index in range(candidate_count):
        path_targets = path[path[:, 0] == index, 1]
        costs = local_cost[index, path_targets]
        mapped_target[index] = target_indices[path_targets[int(np.argmin(costs))]]
    confidence = np.asarray(target.confidence, dtype=float)[mapped_target]
    return MonotoneCorrespondence(
        candidate_index=np.arange(candidate_count, dtype=int),
        target_index=mapped_target,
        confidence=confidence,
        path=np.column_stack((path[:, 0], target_indices[path[:, 1]])),
        cost=float(dynamic[-1, final_target_index]),
    )
