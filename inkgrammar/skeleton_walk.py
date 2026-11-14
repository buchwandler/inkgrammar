"""Full-coverage ordered walks over pixel skeleton graphs."""

from __future__ import annotations


def _pixel_neighbors(
    pixel_set: set[tuple[int, int]], pixel: tuple[int, int]
) -> list[tuple[int, int]]:
    y, x = pixel
    return sorted(
        (ny, nx)
        for ny in range(y - 1, y + 2)
        for nx in range(x - 1, x + 2)
        if (ny, nx) in pixel_set and (ny, nx) != pixel
    )


def _chain_graph(
    pixel_set: set[tuple[int, int]], adjacency: dict
) -> tuple[set, list[tuple]] | None:
    """Collapse the skeleton into nodes (ends/junctions) and chains between."""
    nodes = {pixel for pixel in pixel_set if len(adjacency[pixel]) != 2}
    if not pixel_set:
        return None
    if not nodes:
        nodes = {min(pixel_set)}
    visited_edges: set[frozenset] = set()
    chains: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    for node in sorted(nodes):
        for first in adjacency[node]:
            if frozenset((node, first)) in visited_edges:
                continue
            visited_edges.add(frozenset((node, first)))
            chain = [node, first]
            previous, current = node, first
            while current not in nodes:
                onward = [other for other in adjacency[current] if other != previous]
                if not onward:
                    break
                next_pixel = onward[0]
                visited_edges.add(frozenset((current, next_pixel)))
                chain.append(next_pixel)
                previous, current = current, next_pixel
            chains.append((node, current, chain))
    return nodes, chains


def _euler_path(pixel_set: set[tuple[int, int]], adjacency: dict) -> list[tuple[int, int]] | None:
    """Chinese-Postman walk over the chain graph covering every skeleton edge.

    Nodes (ends and junctions) are chained into edges; odd-degree nodes are
    paired with minimal connecting chains (doubled for the walk), so the
    result is one ordered pixel walk covering the whole skeleton in pen order
    with only tiny junction retraces. Starts at the leftmost end node (Latin
    writing starts at the left).
    """
    graph = _chain_graph(pixel_set, adjacency)
    if graph is None:
        return None
    nodes, chains = graph

    incident: dict = {node: [] for node in nodes}
    for index, (first, second, _) in enumerate(chains):
        incident[first].append((index, second))
        if second != first:
            incident[second].append((index, first))
    ends = sorted((node for node in nodes if len(incident[node]) == 1), key=lambda p: (p[1], p[0]))
    if len(ends) >= 2:
        start, goal = ends[0], ends[-1]
    else:
        start = goal = min(nodes, key=lambda p: (p[1], p[0]))

    # node-level shortest paths over chain edges (BFS, first chain wins)
    node_paths: dict[tuple, list[int]] = {}

    def node_path(source, target):
        key = (source, target)
        if key in node_paths:
            return node_paths[key]
        previous = {source: None}
        queue = [source]
        for node in queue:
            if node == target:
                break
            for index, other in incident[node]:
                if other not in previous:
                    previous[other] = (node, index)
                    queue.append(other)
        if target not in previous:
            node_paths[key] = []
            return []
        chain_indexes = []
        current = target
        while previous[current] is not None:
            node, index = previous[current]
            chain_indexes.append(index)
            current = node
        chain_indexes.reverse()
        node_paths[key] = chain_indexes
        return chain_indexes

    odd_nodes = sorted(node for node in nodes if len(incident[node]) % 2 == 1)
    remaining_odd = [node for node in odd_nodes if node not in (start, goal)]

    def pair_up(vertices):
        remaining = list(vertices)
        pairs = []
        while len(remaining) >= 2:
            best = None
            for first_index in range(len(remaining)):
                for second_index in range(first_index + 1, len(remaining)):
                    distance = len(node_path(remaining[first_index], remaining[second_index]))
                    if best is None or distance < best[0]:
                        best = (distance, first_index, second_index)
            _, first_index, second_index = best
            pairs.append((remaining[first_index], remaining[second_index]))
            remaining = [
                vertex
                for index, vertex in enumerate(remaining)
                if index not in (first_index, second_index)
            ]
        return pairs

    multiplicity = [1] * len(chains)
    for first, second in pair_up(remaining_odd):
        for index in node_path(first, second):
            multiplicity[index] += 1

    stack_nodes = [start]
    stack_chains: list = [None]
    result_nodes: list = []
    result_chains: list = []
    while stack_nodes:
        current = stack_nodes[-1]
        step = None
        for index, other in incident[current]:
            if multiplicity[index] > 0:
                multiplicity[index] -= 1
                step = (index, other)
                break
        if step is None:
            result_nodes.append(stack_nodes.pop())
            result_chains.append(stack_chains.pop())
        else:
            stack_nodes.append(step[1])
            stack_chains.append(step)
    result_nodes.reverse()
    result_chains.reverse()

    walk: list[tuple[int, int]] = [result_nodes[0]]
    previous_node = result_nodes[0]
    for node, chain_step in zip(result_nodes[1:], result_chains[1:], strict=True):
        index, _ = chain_step
        first_node, _, pixels = chains[index]
        segment = pixels if first_node == previous_node else list(reversed(pixels))
        walk.extend(segment[1:])
        previous_node = node
    return walk or None

