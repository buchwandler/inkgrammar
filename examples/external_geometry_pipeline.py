from __future__ import annotations

import json

import numpy as np
from planarcurve import CubicPath, CubicSegment
from skeletonpaths import build_skeleton_graph, prepare_topology_paths, select_path

from inkgrammar.outline import expand_path


def _cubic_path(points: np.ndarray) -> CubicPath:
    segments = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        delta = end - start
        segments.append(
            CubicSegment(
                tuple(start + delta / 3.0),
                tuple(start + 2.0 * delta / 3.0),
                tuple(end),
            )
        )
    return CubicPath(tuple(points[0]), tuple(segments))


def run_example() -> dict[str, object]:
    skeleton = np.zeros((19, 19), dtype=bool)
    skeleton[9, 2:17] = True
    skeleton[2:17, 9] = True
    graph = build_skeleton_graph(
        skeleton,
        min_spur_length_px=0.0,
        already_skeleton=True,
    )
    inventory = prepare_topology_paths(graph, path_kinds=("open",))
    selection = select_path(
        graph,
        path_kind="open",
        expected_cycles=0,
        path_inventory=inventory,
    )
    centerline = _cubic_path(selection.path.xy)
    outline = expand_path(centerline, 4.0)
    assert outline.bounds is not None

    return {
        "source_free": True,
        "evidence_status": "inferred",
        "route_role": "topology_candidate",
        "physical_direction_claim": None,
        "route_ambiguous": selection.ambiguous,
        "orientation_ambiguous": selection.orientation_ambiguous,
        "feasible_path_count": selection.feasible_path_count,
        "centerline_xy": selection.path.xy.tolist(),
        "contours": [[list(point) for point in contour] for contour in outline.contours],
        "bounds": list(outline.bounds),
    }


def main() -> None:
    print(json.dumps(run_example(), sort_keys=True))


if __name__ == "__main__":
    main()
