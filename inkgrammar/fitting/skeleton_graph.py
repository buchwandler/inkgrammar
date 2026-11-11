from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

Point = tuple[float, float]
Pixel = tuple[int, int]


@dataclass(frozen=True)
class SkeletonNode:
    id: int
    xy: Point
    degree: int
    radius: float
    confidence: float


@dataclass(frozen=True)
class SkeletonEdge:
    a: int
    b: int
    pixels: np.ndarray
    length_px: float
    mean_radius_px: float
    confidence: float


@dataclass(frozen=True)
class SkeletonGraph:
    nodes: tuple[SkeletonNode, ...]
    edges: tuple[SkeletonEdge, ...]

    @property
    def component_count(self) -> int:
        if not self.nodes:
            return 0
        adjacency = {node.id: set() for node in self.nodes}
        for edge in self.edges:
            adjacency[edge.a].add(edge.b)
            adjacency[edge.b].add(edge.a)
        remaining = set(adjacency)
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current] & remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        return count

    @property
    def cycle_count(self) -> int:
        return max(0, len(self.edges) - len(self.nodes) + self.component_count)

    @property
    def endpoint_count(self) -> int:
        return sum(node.degree == 1 for node in self.nodes)

    @property
    def junction_count(self) -> int:
        return sum(node.degree > 2 for node in self.nodes)


@dataclass(frozen=True)
class SkeletonPath:
    xy: np.ndarray
    edge_ids: tuple[int, ...]
    start_node: int
    end_node: int
    length_px: float
    confidence: float
    cycle_count: int
    ambiguous: bool
    alternatives: int


def _pixel_adjacency(skeleton: np.ndarray) -> dict[Pixel, tuple[Pixel, ...]]:
    pixels = {tuple(map(int, pixel)) for pixel in np.argwhere(skeleton)}
    adjacency: dict[Pixel, tuple[Pixel, ...]] = {}
    for row, column in sorted(pixels):
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (row + dy, column + dx)
                if dx and dy and ((row, column + dx) in pixels or (row + dy, column) in pixels):
                    continue
                if neighbor in pixels:
                    neighbors.append(neighbor)
        adjacency[(row, column)] = tuple(sorted(neighbors))
    return adjacency


def _components(adjacency: dict[Pixel, tuple[Pixel, ...]]) -> list[set[Pixel]]:
    remaining = set(adjacency)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def prune_spurs(skeleton: np.ndarray, min_length_px: float) -> np.ndarray:
    """Remove endpoint-to-junction branches shorter than the configured length."""
    source = np.asarray(skeleton, dtype=bool)
    if source.ndim != 2:
        raise ValueError("skeleton must be a two-dimensional mask")
    if not np.isfinite(min_length_px) or min_length_px < 0.0:
        raise ValueError("minimum spur length must be finite and nonnegative")
    result = source.copy()
    if min_length_px == 0.0:
        return result
    while True:
        adjacency = _pixel_adjacency(result)
        degrees = {pixel: len(neighbors) for pixel, neighbors in adjacency.items()}
        remove: set[Pixel] = set()
        for endpoint in sorted(pixel for pixel, degree in degrees.items() if degree == 1):
            path = [endpoint]
            previous = None
            current = endpoint
            length = 0.0
            while True:
                next_pixels = [pixel for pixel in adjacency[current] if pixel != previous]
                if len(next_pixels) != 1:
                    break
                following = next_pixels[0]
                length += hypot(following[0] - current[0], following[1] - current[1])
                path.append(following)
                previous, current = current, following
            if degrees.get(current, 0) > 2 and length < min_length_px:
                remove.update(path[:-1])
        if not remove:
            return result
        for row, column in remove:
            result[row, column] = False


def _radius_and_confidence(
    point: Pixel,
    radius_map: np.ndarray | None,
    confidence_map: np.ndarray,
) -> tuple[float, float]:
    row, column = point
    radius = float(radius_map[row, column]) if radius_map is not None else 0.0
    return radius, float(confidence_map[row, column])


def build_skeleton_graph(
    observed_ink: np.ndarray,
    *,
    min_spur_length_px: float = 3.0,
    radius: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    already_skeleton: bool = False,
) -> SkeletonGraph:
    """Skeletonize observed ink and collapse degree-two pixel chains into graph edges."""
    ink = np.asarray(observed_ink, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("observed ink must be a two-dimensional mask")
    if radius is not None:
        radius = np.asarray(radius, dtype=float)
        if radius.shape != ink.shape or not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius map must be finite, nonnegative, and match observed ink")
    if confidence is None:
        confidence_map = np.ones(ink.shape, dtype=float)
    else:
        confidence_map = np.asarray(confidence, dtype=float)
        if (
            confidence_map.shape != ink.shape
            or not np.isfinite(confidence_map).all()
            or np.any((confidence_map < 0.0) | (confidence_map > 1.0))
        ):
            raise ValueError("confidence map must be between zero and one and match observed ink")

    thin = ink.copy() if already_skeleton else skeletonize(ink)
    thin = prune_spurs(thin, min_spur_length_px)
    adjacency = _pixel_adjacency(thin)
    degrees = {pixel: len(neighbors) for pixel, neighbors in adjacency.items()}
    components = _components(adjacency)
    node_pixels = {pixel for pixel, degree in degrees.items() if degree != 2}
    for component in components:
        if component and all(degrees[pixel] == 2 for pixel in component):
            node_pixels.add(min(component))

    ordered_nodes = sorted(node_pixels)
    node_ids = {pixel: index for index, pixel in enumerate(ordered_nodes)}
    nodes = []
    for pixel in ordered_nodes:
        radius_value, confidence_value = _radius_and_confidence(pixel, radius, confidence_map)
        nodes.append(
            SkeletonNode(
                id=node_ids[pixel],
                xy=(float(pixel[1]), float(pixel[0])),
                degree=degrees[pixel],
                radius=radius_value,
                confidence=confidence_value,
            )
        )

    visited: set[frozenset[Pixel]] = set()
    edges = []
    for start in ordered_nodes:
        for neighbor in adjacency[start]:
            first_key = frozenset((start, neighbor))
            if first_key in visited:
                continue
            chain = [start, neighbor]
            visited.add(first_key)
            previous, current = start, neighbor
            while current not in node_pixels:
                following_pixels = [pixel for pixel in adjacency[current] if pixel != previous]
                if len(following_pixels) != 1:
                    break
                following = following_pixels[0]
                key = frozenset((current, following))
                if key in visited:
                    break
                visited.add(key)
                chain.append(following)
                previous, current = current, following
            if current not in node_ids:
                continue
            xy = np.asarray([(column, row) for row, column in chain], dtype=float)
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            radius_values = [
                _radius_and_confidence(pixel, radius, confidence_map)[0] for pixel in chain
            ]
            confidence_values = [
                _radius_and_confidence(pixel, radius, confidence_map)[1] for pixel in chain
            ]
            edges.append(
                SkeletonEdge(
                    a=node_ids[start],
                    b=node_ids[current],
                    pixels=xy,
                    length_px=length,
                    mean_radius_px=float(np.mean(radius_values)),
                    confidence=float(np.mean(confidence_values)),
                )
            )

    return SkeletonGraph(tuple(nodes), tuple(edges))


def _other_node(edge: SkeletonEdge, node_id: int) -> int:
    return edge.b if edge.a == node_id else edge.a


def _edge_route(
    graph: SkeletonGraph, edge_ids: tuple[int, ...], start: int
) -> tuple[np.ndarray, int]:
    current = start
    points: list[np.ndarray] = []
    for edge_id in edge_ids:
        edge = graph.edges[edge_id]
        if edge.a == current:
            edge_points = edge.pixels
        elif edge.b == current:
            edge_points = edge.pixels[::-1]
        else:
            raise ValueError("edge sequence is not connected")
        points.extend(edge_points if not points else edge_points[1:])
        current = _other_node(edge, current)
    return np.asarray(points, dtype=float), current


def _open_paths(
    graph: SkeletonGraph, path_kind: str
) -> list[tuple[tuple[int, ...], int, int, int]]:
    incident: dict[int, list[int]] = {node.id: [] for node in graph.nodes}
    for edge_id, edge in enumerate(graph.edges):
        incident[edge.a].append(edge_id)
        if edge.b != edge.a:
            incident[edge.b].append(edge_id)
    endpoints = sorted(node.id for node in graph.nodes if node.degree == 1)
    paths: dict[tuple[tuple[int, ...], int], tuple[tuple[int, ...], int, int, int]] = {}
    allow_one_cycle = path_kind == "open-with-loop"

    def walk(
        start: int,
        current: int,
        used_edges: tuple[int, ...],
        visits: dict[int, int],
        cycle_count: int,
    ) -> None:
        if current != start and current in endpoints:
            if (path_kind == "open" and cycle_count == 0) or (allow_one_cycle and cycle_count == 1):
                key = (tuple(sorted(used_edges)), start)
                paths.setdefault(key, (used_edges, start, current, cycle_count))
            return
        used = set(used_edges)
        for edge_id in sorted(incident[current]):
            if edge_id in used:
                continue
            next_node = _other_node(graph.edges[edge_id], current)
            next_visits = visits.get(next_node, 0)
            if next_visits >= 2:
                continue
            added_cycle = int(next_visits > 0)
            if cycle_count + added_cycle > 1:
                continue
            walk(
                start,
                next_node,
                (*used_edges, edge_id),
                visits | {next_node: next_visits + 1},
                cycle_count + added_cycle,
            )

    for start in endpoints:
        walk(start, start, (), {start: 1}, 0)
    return list(paths.values())


def _loop_paths(graph: SkeletonGraph) -> list[tuple[tuple[int, ...], int, int, int]]:
    incident: dict[int, list[int]] = {node.id: [] for node in graph.nodes}
    for edge_id, edge in enumerate(graph.edges):
        incident[edge.a].append(edge_id)
        if edge.b != edge.a:
            incident[edge.b].append(edge_id)
    paths: dict[tuple[int, ...], tuple[tuple[int, ...], int, int, int]] = {}

    def walk(
        start: int,
        current: int,
        used_edges: tuple[int, ...],
        visited_nodes: set[int],
    ) -> None:
        for edge_id in sorted(incident[current]):
            if edge_id in used_edges:
                continue
            next_node = _other_node(graph.edges[edge_id], current)
            if next_node == start:
                sequence = (*used_edges, edge_id)
                paths.setdefault(tuple(sorted(sequence)), (sequence, start, start, 1))
            elif next_node not in visited_nodes:
                walk(start, next_node, (*used_edges, edge_id), visited_nodes | {next_node})

    for node in graph.nodes:
        walk(node.id, node.id, (), {node.id})
    return list(paths.values())


def _path_curvature_cost(xy: np.ndarray) -> float:
    vectors = np.diff(xy, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12
    directions = np.divide(
        vectors,
        lengths[:, None],
        out=np.zeros_like(vectors),
        where=valid[:, None],
    )
    if np.count_nonzero(valid) < 2:
        return 0.0
    adjacent = valid[:-1] & valid[1:]
    turns = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0))
    return float(np.mean(turns[adjacent] ** 2)) if np.any(adjacent) else 0.0


def select_topology_path(
    graph: SkeletonGraph,
    *,
    path_kind: Literal["open", "loop", "open-with-loop"] = "open",
    expected_cycles: int | None = None,
    expected_endpoints: tuple[Point | None, Point | None] | None = None,
    endpoint_tolerance_px: float = 3.0,
    expected_length_px: float | None = None,
    candidate_xy: np.ndarray | None = None,
) -> SkeletonPath:
    """Choose a path after enforcing topology and endpoint evidence.

    Candidate proximity is only the final tie-breaker among paths that pass the
    hard topology and endpoint constraints.
    """
    if path_kind not in {"open", "loop", "open-with-loop"}:
        raise ValueError(f"unsupported path kind: {path_kind}")
    if expected_cycles is not None and graph.cycle_count != expected_cycles:
        raise ValueError(f"skeleton has {graph.cycle_count} cycles, expected {expected_cycles}")
    if not np.isfinite(endpoint_tolerance_px) or endpoint_tolerance_px < 0.0:
        raise ValueError("endpoint tolerance must be finite and nonnegative")
    if expected_length_px is not None and (
        not np.isfinite(expected_length_px) or expected_length_px <= 0.0
    ):
        raise ValueError("expected path length must be positive and finite")
    candidate = None if candidate_xy is None else np.asarray(candidate_xy, dtype=float)
    if candidate is not None and (
        candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) == 0
    ):
        raise ValueError("candidate curve must contain finite xy points")
    if candidate is not None and not np.isfinite(candidate).all():
        raise ValueError("candidate curve must contain finite xy points")

    candidates = _loop_paths(graph) if path_kind == "loop" else _open_paths(graph, path_kind)
    if not candidates:
        raise ValueError(f"no path satisfies topology kind {path_kind!r}")
    start_expected, end_expected = expected_endpoints or (None, None)
    for point in (start_expected, end_expected):
        if point is not None and (
            len(point) != 2 or not np.isfinite(np.asarray(point, dtype=float)).all()
        ):
            raise ValueError("expected endpoint coordinates must be finite xy pairs")
    start_nodes = {node.id: np.asarray(node.xy) for node in graph.nodes}
    filtered = []
    for edge_ids, start, end, cycles in candidates:
        if (
            start_expected is not None
            and np.linalg.norm(start_nodes[start] - np.asarray(start_expected, dtype=float))
            > endpoint_tolerance_px
        ):
            continue
        if (
            end_expected is not None
            and np.linalg.norm(start_nodes[end] - np.asarray(end_expected, dtype=float))
            > endpoint_tolerance_px
        ):
            continue
        xy, routed_end = _edge_route(graph, edge_ids, start)
        if routed_end != end:
            continue
        length = float(sum(graph.edges[edge_id].length_px for edge_id in edge_ids))
        confidence = float(
            np.average(
                [graph.edges[edge_id].confidence for edge_id in edge_ids],
                weights=[max(graph.edges[edge_id].length_px, 1e-12) for edge_id in edge_ids],
            )
        )
        proximity = (
            float(cKDTree(candidate).query(xy, k=1)[0].mean()) if candidate is not None else 0.0
        )
        directional_proximity = (
            float(np.linalg.norm(xy[0] - candidate[0]) + np.linalg.norm(xy[-1] - candidate[-1]))
            if candidate is not None
            else 0.0
        )
        length_error = (
            abs(length - expected_length_px) if expected_length_px is not None else length
        )
        curvature = _path_curvature_cost(xy)
        filtered.append(
            (
                edge_ids,
                start,
                end,
                cycles,
                xy,
                length,
                confidence,
                curvature,
                proximity,
                length_error,
                directional_proximity,
            )
        )
    if not filtered:
        raise ValueError("no path satisfies the expected endpoint regions")

    filtered.sort(key=lambda item: (-item[6], item[10], item[8], item[9], item[7], item[0]))
    selected = filtered[0]
    ambiguous = False
    if len(filtered) > 1:
        alternative = filtered[1]
        same_confidence = abs(selected[6] - alternative[6]) < 0.05
        similar_length = abs(selected[5] - alternative[5]) < max(2.0, 0.05 * selected[5])
        similar_proximity = candidate is None or (
            abs(selected[8] - alternative[8]) < 0.5 and abs(selected[10] - alternative[10]) < 0.5
        )
        ambiguous = same_confidence and similar_length and similar_proximity
    confidence = selected[6] * (0.5 if ambiguous else 1.0)
    return SkeletonPath(
        xy=selected[4],
        edge_ids=selected[0],
        start_node=selected[1],
        end_node=selected[2],
        length_px=selected[5],
        confidence=confidence,
        cycle_count=selected[3],
        ambiguous=ambiguous,
        alternatives=len(filtered),
    )
