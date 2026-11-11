from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class TargetTrace:
    xy: np.ndarray
    tangent: np.ndarray
    radius: np.ndarray
    confidence: np.ndarray


def skeletonize_component(
    mask: np.ndarray,
    *,
    min_component_area: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    component = np.asarray(mask, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component mask must be 2D")
    if min_component_area < 1:
        raise ValueError("minimum component area must be positive")
    if not np.any(component):
        return np.zeros(component.shape, dtype=bool), np.zeros(component.shape, dtype=float)
    labels, component_count = label(component)
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    keep = sizes >= min_component_area
    keep[0] = False
    cleaned = keep[labels]
    if not np.any(cleaned):
        return np.zeros(component.shape, dtype=bool), np.zeros(component.shape, dtype=float)
    return skeletonize(cleaned), distance_transform_edt(cleaned)


def _neighbors(skeleton: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    y, x = point
    top, bottom = max(0, y - 1), min(skeleton.shape[0], y + 2)
    left, right = max(0, x - 1), min(skeleton.shape[1], x + 2)
    candidates = np.argwhere(skeleton[top:bottom, left:right])
    return [
        (int(candidate[0] + top), int(candidate[1] + left))
        for candidate in candidates
        if tuple(candidate + (top, left)) != point
    ]


def prune_skeleton(
    skeleton: np.ndarray,
    radius: np.ndarray,
    *,
    min_spur_length_px: float = 3.0,
    min_confidence: float = 0.25,
) -> np.ndarray:
    pruned = np.asarray(skeleton, dtype=bool).copy()
    radius = np.asarray(radius, dtype=float)
    if pruned.ndim != 2 or radius.shape != pruned.shape:
        raise ValueError("skeleton and radius must be matching 2D arrays")
    if min_spur_length_px < 0 or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("spur length and confidence thresholds are invalid")
    radius_max = float(np.max(radius[pruned])) if np.any(pruned) else 0.0
    if radius_max <= 0:
        return pruned

    changed = True
    while changed:
        changed = False
        endpoints = [
            tuple(int(value) for value in point)
            for point in np.argwhere(pruned)
            if len(_neighbors(pruned, tuple(int(value) for value in point))) == 1
        ]
        for endpoint in endpoints:
            if not pruned[endpoint]:
                continue
            path = [endpoint]
            previous = None
            current = endpoint
            junction = None
            length = 0.0
            while True:
                onward = [point for point in _neighbors(pruned, current) if point != previous]
                if len(onward) != 1:
                    break
                next_point = onward[0]
                length += float(np.hypot(next_point[0] - current[0], next_point[1] - current[1]))
                if len(_neighbors(pruned, next_point)) >= 3:
                    junction = next_point
                    break
                path.append(next_point)
                previous, current = current, next_point
            if junction is None or length >= min_spur_length_px:
                continue
            path_array = np.asarray(path, dtype=int)
            confidence = float(np.mean(radius[path_array[:, 0], path_array[:, 1]] / radius_max))
            if confidence < min_confidence:
                pruned[path_array[:, 0], path_array[:, 1]] = False
                changed = True
    return pruned


def _local_tangent_map(skeleton: np.ndarray) -> np.ndarray:
    tangents = np.zeros((*skeleton.shape, 2), dtype=float)
    coordinates = np.argwhere(skeleton)
    for y, x in coordinates:
        local = np.argwhere(skeleton[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3])
        if len(local) < 2:
            continue
        local = local + (max(0, y - 2), max(0, x - 2))
        centered = local - local.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        tangent_yx = vectors[0]
        tangents[y, x] = (tangent_yx[1], tangent_yx[0])
    return tangents


def _trace_tangents(xy: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(xy, dtype=float)
    valid = (confidence > 0) & np.isfinite(xy).all(axis=1)
    for index in np.flatnonzero(valid):
        previous = index - 1
        while previous >= 0 and valid[previous]:
            if np.linalg.norm(xy[previous] - xy[index]) > 0:
                break
            previous -= 1
        if previous < 0 or not valid[previous]:
            previous = None

        following = index + 1
        while following < len(xy) and valid[following]:
            if np.linalg.norm(xy[following] - xy[index]) > 0:
                break
            following += 1
        if following >= len(xy) or not valid[following]:
            following = None

        if previous is not None and following is not None:
            vector = xy[following] - xy[previous]
        elif following is not None:
            vector = xy[following] - xy[index]
        elif previous is not None:
            vector = xy[index] - xy[previous]
        else:
            continue
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            tangents[index] = vector / norm
    return tangents


def _empty_trace(sample_count: int) -> TargetTrace:
    return TargetTrace(
        xy=np.full((sample_count, 2), np.nan, dtype=float),
        tangent=np.zeros((sample_count, 2), dtype=float),
        radius=np.zeros(sample_count, dtype=float),
        confidence=np.zeros(sample_count, dtype=float),
    )


def trace_ordered_centerline(
    skeleton: np.ndarray,
    radius: np.ndarray,
    nominal_xy: np.ndarray,
    *,
    candidates_per_sample: int = 8,
    search_radius_px: float = 6.0,
    sigma_position_px: float = 3.0,
    unassigned_cost: float = 2.0,
    connectivity_weight: float = 0.4,
    tangent_weight: float = 0.5,
    radius_weight: float = 0.25,
) -> TargetTrace:
    skeleton = np.asarray(skeleton, dtype=bool)
    radius = np.asarray(radius, dtype=float)
    nominal = np.asarray(nominal_xy, dtype=float)
    if skeleton.ndim != 2 or radius.shape != skeleton.shape:
        raise ValueError("skeleton and radius must be matching 2D arrays")
    if nominal.ndim != 2 or nominal.shape[1] != 2 or len(nominal) < 2:
        raise ValueError("nominal curve must have shape (N, 2) with N >= 2")
    if not np.isfinite(nominal).all():
        raise ValueError("nominal curve coordinates must be finite")
    if candidates_per_sample < 1 or search_radius_px <= 0 or sigma_position_px <= 0:
        raise ValueError("candidate count and curve distance scales must be positive")

    skeleton_yx = np.argwhere(skeleton)
    if not len(skeleton_yx):
        return _empty_trace(len(nominal))
    skeleton_xy = skeleton_yx[:, ::-1].astype(float)
    skeleton_radii = radius[skeleton_yx[:, 0], skeleton_yx[:, 1]]
    max_radius = max(float(np.max(skeleton_radii)), 1e-9)
    tangent_map = _local_tangent_map(skeleton)
    nominal_tangents = _trace_tangents(nominal, np.ones(len(nominal)))
    candidate_points: list[np.ndarray] = []
    candidate_costs: list[np.ndarray] = []
    candidate_confidence: list[np.ndarray] = []

    for index, point in enumerate(nominal):
        distances = np.linalg.norm(skeleton_xy - point, axis=1)
        order = np.argsort(distances)
        selected = [item for item in order if distances[item] <= search_radius_px][
            :candidates_per_sample
        ]
        points = skeleton_xy[selected]
        selected_distances = distances[selected]
        radii = skeleton_radii[selected]
        local_tangents = np.asarray(
            [tangent_map[int(y), int(x)] for x, y in points], dtype=float
        ).reshape(-1, 2)
        guide_tangent = nominal_tangents[index]
        tangent_alignment = np.abs(local_tangents @ guide_tangent) if len(points) else np.empty(0)
        costs = (
            (selected_distances / sigma_position_px) ** 2
            + tangent_weight * (1.0 - tangent_alignment)
            - radius_weight * np.clip(radii / max_radius, 0.0, 1.0)
        )
        confidence = np.exp(-0.5 * (selected_distances / sigma_position_px) ** 2) * np.clip(
            radii / max_radius, 0.0, 1.0
        )
        candidate_points.append(points)
        candidate_costs.append(costs)
        candidate_confidence.append(confidence)

    states = [
        np.concatenate((points, np.full((1, 2), np.nan)), axis=0) for points in candidate_points
    ]
    costs = [np.append(local, unassigned_cost) for local in candidate_costs]
    dynamic = costs[0].copy()
    parents: list[np.ndarray] = [np.full(len(dynamic), -1, dtype=int)]

    for index in range(1, len(nominal)):
        previous_states = states[index - 1]
        current_states = states[index]
        current_cost = np.empty(len(current_states), dtype=float)
        current_parent = np.empty(len(current_states), dtype=int)
        expected_step = float(np.linalg.norm(nominal[index] - nominal[index - 1]))
        for current_index, current_point in enumerate(current_states):
            transitions = np.empty(len(previous_states), dtype=float)
            for previous_index, previous_point in enumerate(previous_states):
                if not np.isfinite(current_point).all() or not np.isfinite(previous_point).all():
                    transitions[previous_index] = unassigned_cost * 0.5
                    continue
                step = current_point - previous_point
                distance = float(np.linalg.norm(step))
                guide = nominal_tangents[index]
                direction_cost = (
                    1.0 - float(np.clip(np.dot(step / distance, guide), -1.0, 1.0))
                    if distance > 1e-9
                    else 1.0
                )
                excess = max(0.0, distance - max(2.0, expected_step * 2.0))
                transitions[previous_index] = (
                    connectivity_weight * excess**2 + tangent_weight * direction_cost
                )
            options = dynamic + transitions
            parent = int(np.argmin(options))
            current_cost[current_index] = options[parent] + costs[index][current_index]
            current_parent[current_index] = parent
        dynamic = current_cost
        parents.append(current_parent)

    state_index = int(np.argmin(dynamic))
    selected_states = [None] * len(nominal)
    for index in range(len(nominal) - 1, -1, -1):
        point = states[index][state_index]
        if np.isfinite(point).all():
            selected_states[index] = state_index
        if index:
            state_index = int(parents[index][state_index])

    xy = np.full((len(nominal), 2), np.nan, dtype=float)
    trace_radius = np.zeros(len(nominal), dtype=float)
    confidence = np.zeros(len(nominal), dtype=float)
    for index, selected_index in enumerate(selected_states):
        if selected_index is None:
            continue
        point = states[index][selected_index]
        xy[index] = point
        x, y = int(point[0]), int(point[1])
        trace_radius[index] = radius[y, x]
        confidence[index] = candidate_confidence[index][selected_index]
    return TargetTrace(xy, _trace_tangents(xy, confidence), trace_radius, confidence)


def resample_trace(trace: TargetTrace, sample_count: int) -> TargetTrace:
    if sample_count < 2:
        raise ValueError("trace sample count must be at least two")
    valid = (trace.confidence > 0) & np.isfinite(trace.xy).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        return _empty_trace(sample_count)
    points = trace.xy[valid_indices]
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-9:
        return _empty_trace(sample_count)
    positions = np.linspace(0.0, cumulative[-1], sample_count)
    right = np.clip(np.searchsorted(cumulative, positions, side="right"), 1, len(points) - 1)
    left = right - 1
    span = cumulative[right] - cumulative[left]
    fraction = np.divide(
        positions - cumulative[left], span, out=np.zeros_like(positions), where=span > 0
    )
    xy = points[left] + fraction[:, None] * (points[right] - points[left])
    radii = trace.radius[valid_indices[left]] + fraction * (
        trace.radius[valid_indices[right]] - trace.radius[valid_indices[left]]
    )
    confidence = trace.confidence[valid_indices[left]] + fraction * (
        trace.confidence[valid_indices[right]] - trace.confidence[valid_indices[left]]
    )
    gaps = valid_indices[right] - valid_indices[left] > 1
    xy[gaps] = np.nan
    radii[gaps] = 0.0
    confidence[gaps] = 0.0
    tangent = _trace_tangents(xy, confidence)
    return TargetTrace(xy, tangent, radii, confidence)
