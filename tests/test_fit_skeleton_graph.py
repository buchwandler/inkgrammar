import numpy as np
from skimage.draw import circle_perimeter

from inkgrammar.fitting.skeleton_graph import (
    build_skeleton_graph,
    prune_spurs,
    select_topology_path,
)


def test_short_endpoint_spur_is_pruned_and_graph_chains_are_collapsed():
    skeleton = np.zeros((31, 31), dtype=bool)
    skeleton[4:27, 15] = True
    skeleton[15, 9:16] = True

    pruned = prune_spurs(skeleton, min_length_px=7.0)
    graph = build_skeleton_graph(
        skeleton,
        min_spur_length_px=7.0,
        already_skeleton=True,
    )

    assert not pruned[15, 9:15].any()
    assert graph.endpoint_count == 2
    assert graph.junction_count == 0
    assert len(graph.edges) == 1
    assert graph.edges[0].length_px == 22.0


def test_hard_endpoint_constraints_override_a_bad_candidate_path():
    skeleton = np.zeros((31, 31), dtype=bool)
    skeleton[3:28, 15] = True
    skeleton[15, 5:16] = True
    graph = build_skeleton_graph(
        skeleton,
        min_spur_length_px=0.0,
        already_skeleton=True,
    )
    bad_candidate = np.column_stack((np.arange(5, 16), np.full(11, 15)))

    path = select_topology_path(
        graph,
        path_kind="open",
        expected_cycles=0,
        expected_endpoints=((15.0, 3.0), (15.0, 27.0)),
        endpoint_tolerance_px=1.0,
        candidate_xy=bad_candidate,
    )

    np.testing.assert_array_equal(path.xy[0], [15.0, 3.0])
    np.testing.assert_array_equal(path.xy[-1], [15.0, 27.0])
    assert np.all(path.xy[:, 0] == 15.0)
    assert path.cycle_count == 0
    assert path.confidence == 1.0


def test_candidate_orientation_breaks_reversed_path_ties_without_hard_endpoints():
    skeleton = np.zeros((9, 9), dtype=bool)
    skeleton[4, 1:8] = True
    graph = build_skeleton_graph(
        skeleton,
        min_spur_length_px=0.0,
        already_skeleton=True,
    )
    candidate = np.column_stack((np.arange(1.0, 8.0), np.full(7, 4.0)))

    path = select_topology_path(
        graph,
        path_kind="open",
        expected_cycles=0,
        candidate_xy=candidate,
    )

    np.testing.assert_array_equal(path.xy[0], candidate[0])
    np.testing.assert_array_equal(path.xy[-1], candidate[-1])
    assert path.ambiguous is False


def test_closed_skeleton_cycle_is_reported_and_selected_as_a_loop():
    skeleton = np.zeros((31, 31), dtype=bool)
    rows, columns = circle_perimeter(15, 15, 8)
    skeleton[rows, columns] = True
    graph = build_skeleton_graph(
        skeleton,
        min_spur_length_px=0.0,
        already_skeleton=True,
    )

    assert graph.cycle_count == 1
    path = select_topology_path(graph, path_kind="loop", expected_cycles=1)

    assert path.cycle_count == 1
    assert path.start_node == path.end_node
    np.testing.assert_array_equal(path.xy[0], path.xy[-1])
    assert path.length_px > 0


def test_graph_propagates_radius_and_ink_confidence():
    skeleton = np.zeros((9, 9), dtype=bool)
    skeleton[2:7, 4] = True
    radius = np.full(skeleton.shape, 2.0)
    confidence = np.ones(skeleton.shape)
    confidence[4, 4] = 0.25
    graph = build_skeleton_graph(
        skeleton,
        radius=radius,
        confidence=confidence,
        min_spur_length_px=0.0,
        already_skeleton=True,
    )

    assert all(node.radius == 2.0 for node in graph.nodes)
    assert graph.edges[0].mean_radius_px == 2.0
    assert graph.edges[0].confidence < 1.0


def test_open_with_loop_path_preserves_a_cycle_and_its_continuation():
    from inkgrammar.fitting.skeleton_graph import SkeletonEdge, SkeletonGraph, SkeletonNode

    graph = SkeletonGraph(
        nodes=(
            SkeletonNode(0, (0.0, 0.0), degree=1, radius=1.0, confidence=1.0),
            SkeletonNode(1, (1.0, 0.0), degree=3, radius=1.0, confidence=1.0),
            SkeletonNode(2, (2.0, 0.0), degree=1, radius=1.0, confidence=1.0),
        ),
        edges=(
            SkeletonEdge(0, 1, np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0, 1.0, 1.0),
            SkeletonEdge(
                1,
                1,
                np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [1.0, 0.0]]),
                4.0,
                1.0,
                1.0,
            ),
            SkeletonEdge(1, 2, np.array([[1.0, 0.0], [2.0, 0.0]]), 1.0, 1.0, 1.0),
        ),
    )

    path = select_topology_path(
        graph,
        path_kind="open-with-loop",
        expected_cycles=1,
        expected_endpoints=((0.0, 0.0), (2.0, 0.0)),
        endpoint_tolerance_px=0.0,
    )

    assert path.cycle_count == 1
    assert path.edge_ids == (0, 1, 2)
    np.testing.assert_array_equal(path.xy[0], [0.0, 0.0])
    np.testing.assert_array_equal(path.xy[-1], [2.0, 0.0])
