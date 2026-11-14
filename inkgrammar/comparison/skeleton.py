from __future__ import annotations

import numpy as np
from skeletonpaths import graph_from_skeleton

from .models import SkeletonDiagnostics, TopologySummary


def diagnostics_from_skeleton(
    skeleton: np.ndarray,
    radius: np.ndarray,
) -> SkeletonDiagnostics:
    """Build topology diagnostics from persisted skeleton and radius rasters."""
    skeleton = np.asarray(skeleton, dtype=bool)
    radius = np.asarray(radius, dtype=float)
    graph = graph_from_skeleton(skeleton, radius=radius)
    endpoints = [node.xy for node in graph.nodes if node.degree == 1]
    junctions = [node.xy for node in graph.nodes if node.degree > 2]
    topology = TopologySummary(
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        endpoint_count=len(endpoints),
        endpoints=endpoints,
        junction_count=len(junctions),
        junctions=junctions,
        cycle_count=graph.cycle_count,
        component_count=graph.component_count,
    )
    return SkeletonDiagnostics(skeleton, radius, graph, topology)


def analyze_skeleton(mask: np.ndarray) -> SkeletonDiagnostics:
    ink = np.array(mask, dtype=bool, copy=True)
    if ink.ndim != 2:
        raise ValueError("ink mask must be a 2D array")
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    radius = distance_transform_edt(ink)
    skeleton = skeletonize(ink) if np.any(ink) else np.zeros_like(ink)
    return diagnostics_from_skeleton(skeleton, radius)
