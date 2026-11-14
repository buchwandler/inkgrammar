from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from planarcurve.polyline import polyline_length

from .geometry import DifferentialGeometry, differential_geometry
from .models import ComparisonResult


@dataclass(frozen=True)
class ComparisonAnalysis:
    progress: np.ndarray
    reference_geometry: DifferentialGeometry
    sample_geometry: DifferentialGeometry
    reference_width_px: np.ndarray
    sample_width_px: np.ndarray
    series: dict[str, np.ndarray]
    metrics: dict[str, Any]


def _angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = second - first
    return np.arctan2(np.sin(difference), np.cos(difference))


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _sample_width(radius: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    height, width = radius.shape
    columns = np.clip(np.rint(points_xy[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.rint(points_xy[:, 1]).astype(int), 0, height - 1)
    return 2.0 * radius[rows, columns]


def analyze_comparison(result: ComparisonResult) -> ComparisonAnalysis:
    reference_xy = result.reference.resampled_xy
    sample_xy = result.alignment.aligned_sample_xy
    sample_source_xy = result.sample.resampled_xy
    if result.alignment.direction == "reversed-for-best-match":
        sample_source_xy = sample_source_xy[::-1]

    reference_geometry = differential_geometry(
        reference_xy,
        window_length=result.smoothing_window,
        polynomial_order=result.smoothing_polynomial_order,
    )
    sample_geometry = differential_geometry(
        sample_xy,
        window_length=result.smoothing_window,
        polynomial_order=result.smoothing_polynomial_order,
    )
    progress = np.linspace(0.0, 1.0, len(reference_xy))
    reference_length = float(polyline_length(result.reference.ordered_path.xy))
    sample_length = float(polyline_length(result.sample.ordered_path.xy))
    if reference_length <= 0.0:
        raise ValueError("reference path must have positive arc length")

    position_delta = sample_xy - reference_xy
    position_error = np.linalg.norm(position_delta, axis=1) / reference_length
    tangent_error_rad = _angle_difference(
        reference_geometry.heading_rad,
        sample_geometry.heading_rad,
    )
    tangent_error_deg = np.degrees(tangent_error_rad)
    heading_error_deg = np.abs(tangent_error_deg)
    turn_rate_error = (
        sample_geometry.turn_rate_rad_per_progress
        - reference_geometry.turn_rate_rad_per_progress
    )
    absolute_turn_rate_error = np.abs(turn_rate_error)
    normalized_origin = reference_xy[0]
    reference_xy_normalized = (reference_xy - normalized_origin) / reference_length
    sample_xy_normalized = (sample_xy - normalized_origin) / reference_length

    reference_start, reference_end = reference_xy[0], reference_xy[-1]
    sample_start, sample_end = sample_xy[0], sample_xy[-1]
    endpoint_position_error = {
        "start": float(np.linalg.norm(sample_start - reference_start) / reference_length),
        "end": float(np.linalg.norm(sample_end - reference_end) / reference_length),
    }
    endpoint_tangent_error = {
        "start": float(abs(tangent_error_deg[0])),
        "end": float(abs(tangent_error_deg[-1])),
    }
    turn_summary = {
        "mae": float(np.mean(absolute_turn_rate_error)),
        "rmse": float(np.sqrt(np.mean(turn_rate_error**2))),
        "p95_abs": float(np.percentile(absolute_turn_rate_error, 95.0)),
        "maximum_abs": float(np.max(absolute_turn_rate_error)),
    }

    reference_topology = result.reference.diagnostics.topology
    sample_topology = result.sample.diagnostics.topology
    reference_width = _sample_width(result.reference.diagnostics.radius, reference_xy)
    sample_width = _sample_width(result.sample.diagnostics.radius, sample_source_xy)
    position_summary = _summary(position_error)
    metrics = {
        "path": {
            "reference_arc_length_px": reference_length,
            "sample_arc_length_px": sample_length,
            "raw_sample_to_reference_length_ratio": sample_length / reference_length,
            "reference_arc_length_normalization_px": reference_length,
            "reference_normalized_arc_length": 1.0,
            "aligned_sample_normalized_arc_length": float(
                polyline_length(sample_xy) / reference_length
            ),
            "mean_error": position_summary["mean"],
            "median_error": position_summary["median"],
            "p90_error": position_summary["p90"],
            "p95_error": position_summary["p95"],
            "max_error": position_summary["maximum"],
            "x_rmse": float(
                np.sqrt(
                    np.mean(
                        (sample_xy_normalized[:, 0] - reference_xy_normalized[:, 0]) ** 2
                    )
                )
            ),
            "y_rmse": float(
                np.sqrt(
                    np.mean(
                        (sample_xy_normalized[:, 1] - reference_xy_normalized[:, 1]) ** 2
                    )
                )
            ),
        },
        "heading": {
            "mean_abs_error_deg": float(np.mean(heading_error_deg)),
            "median_abs_error_deg": float(np.median(heading_error_deg)),
            "p95_abs_error_deg": float(np.percentile(heading_error_deg, 95.0)),
            "maximum_abs_error_deg": float(np.max(heading_error_deg)),
        },
        "turn_rate": {
            "units": "radians per normalized stroke progress",
            **turn_summary,
        },
        "cumulative_turn": {
            "reference_total_rad": float(reference_geometry.cumulative_turn_rad[-1]),
            "sample_total_rad": float(sample_geometry.cumulative_turn_rad[-1]),
            "difference_rad": float(
                sample_geometry.cumulative_turn_rad[-1]
                - reference_geometry.cumulative_turn_rad[-1]
            ),
        },
        "endpoints": {
            "direction_source": "inferred",
            "start_error": endpoint_position_error["start"],
            "end_error": endpoint_position_error["end"],
            "start_tangent_error_deg": endpoint_tangent_error["start"],
            "end_tangent_error_deg": endpoint_tangent_error["end"],
            "reference_start_xy_px": reference_start.tolist(),
            "reference_end_xy_px": reference_end.tolist(),
            "sample_start_xy_aligned_px": sample_start.tolist(),
            "sample_end_xy_aligned_px": sample_end.tolist(),
        },
        "topology": {
            "reference": reference_topology.model_dump(mode="json"),
            "sample": sample_topology.model_dump(mode="json"),
            "reference_path_coverage": result.reference.ordered_path.coverage,
            "sample_path_coverage": result.sample.ordered_path.coverage,
        },
        "width": {
            "scored": False,
            "units": "pixels in each original crop",
            "reference_median_px": float(np.median(reference_width)),
            "sample_median_px": float(np.median(sample_width)),
        },
    }
    series = {
        "progress": progress,
        "position_error": position_error,
        "reference_xy_normalized": reference_xy_normalized,
        "sample_xy_normalized": sample_xy_normalized,
        "reference_heading_deg": np.degrees(reference_geometry.heading_rad),
        "sample_heading_deg": np.degrees(sample_geometry.heading_rad),
        "tangent_error_deg": tangent_error_deg,
        "reference_turn_rate": reference_geometry.turn_rate_rad_per_progress,
        "sample_turn_rate": sample_geometry.turn_rate_rad_per_progress,
        "turn_rate_error": turn_rate_error,
        "reference_cumulative_turn": reference_geometry.cumulative_turn_rad,
        "sample_cumulative_turn": sample_geometry.cumulative_turn_rad,
        "reference_width_px": reference_width,
        "sample_width_px": sample_width,
        "reference_raw_xy": result.reference.ordered_path.xy,
        "sample_raw_xy": result.sample.ordered_path.xy,
        "reference_resampled_xy": reference_xy,
        "sample_resampled_xy": sample_xy,
        "sample_source_resampled_xy": sample_source_xy,
    }
    return ComparisonAnalysis(
        progress,
        reference_geometry,
        sample_geometry,
        reference_width,
        sample_width,
        series,
        metrics,
    )
