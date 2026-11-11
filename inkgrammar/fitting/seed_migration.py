from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import BSplineGeometry, PrimitiveTemplate, SplineEndpointPolicy, TopologyTransform
from .spline import sample_curve_by_arclength, spline_design_matrix


@dataclass(frozen=True)
class SeedMigrationResult:
    template: PrimitiveTemplate
    report: dict[str, Any]


def migrate_legacy_seed(
    template: PrimitiveTemplate,
    *,
    control_point_count: int = 6,
    maximum_deviation_model_units: float = 0.05,
    sample_count: int = 2048,
) -> SeedMigrationResult:
    if template.kind != "cubic-chain" or template.segments is None:
        raise ValueError("seed migration requires a legacy cubic-chain primitive")
    if isinstance(control_point_count, bool) or not isinstance(control_point_count, int):
        raise ValueError("control_point_count must be an integer")
    if control_point_count < 4:
        raise ValueError("control_point_count must be at least four")
    if not np.isfinite(maximum_deviation_model_units) or maximum_deviation_model_units <= 0.0:
        raise ValueError("maximum deviation must be finite and positive")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 256:
        raise ValueError("sample_count must be an integer of at least 256")

    source = sample_curve_by_arclength(template, TopologyTransform(), sample_count)
    interior_count = control_point_count - 4
    knots = [0.0] * 4
    knots.extend(index / (interior_count + 1) for index in range(1, interior_count + 1))
    knots.extend([1.0] * 4)
    initial_points = np.linspace(source.xy[0], source.xy[-1], control_point_count)
    geometry = BSplineGeometry(
        knots=knots,
        control_points=[tuple(point) for point in initial_points],
        endpoint_policy=SplineEndpointPolicy(start="locked", end="locked"),
    )
    design = spline_design_matrix(source.u, geometry).toarray()
    fixed = np.asarray([0, control_point_count - 1])
    free = np.arange(1, control_point_count - 1)
    residual = source.xy - design[:, fixed] @ initial_points[fixed]
    fitted = np.linalg.lstsq(design[:, free], residual, rcond=None)[0]
    control_points = initial_points.copy()
    control_points[free] = fitted
    geometry = geometry.model_copy(
        update={"control_points": [tuple(point) for point in control_points]}
    )
    migrated = PrimitiveTemplate(
        id=template.id,
        kind="bspline-centerline",
        frame=template.frame,
        spline=geometry,
        provenance=template.provenance,
    )

    converted = sample_curve_by_arclength(migrated, TopologyTransform(), sample_count)
    deviations = np.linalg.norm(converted.xy - source.xy, axis=1)
    maximum_deviation = float(np.max(deviations))
    closed_source = bool(
        np.allclose(template.frame.start, template.segments[-1].end, atol=1e-10, rtol=0.0)
    )
    report = {
        "format": "inkgrammar-spline-seed-migration-report",
        "format_version": 1,
        "source_free": True,
        "source_id": template.id,
        "source_kind": "cubic-chain",
        "target_kind": "bspline-centerline",
        "degree": geometry.degree,
        "control_point_count": control_point_count,
        "interior_knot_count": interior_count,
        "sample_count": sample_count,
        "closed_source": closed_source,
        "maximum_deviation_model_units": maximum_deviation,
        "rms_deviation_model_units": float(np.sqrt(np.mean(deviations**2))),
        "maximum_deviation_limit_model_units": maximum_deviation_model_units,
        "within_tolerance": maximum_deviation < maximum_deviation_model_units,
        "endpoints_locked": True,
        "canonical_seed_modified": False,
        "approved_for_canonical_use": False,
        "visual_review_required": True,
        "closed_loop_review_required": closed_source,
    }
    return SeedMigrationResult(template=migrated, report=report)


def write_seed_migration_report(result: SeedMigrationResult, path: str | Path) -> Path:
    report_path = Path(path)
    if report_path.is_symlink():
        raise ValueError("migration report output must not be a symlink")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path
