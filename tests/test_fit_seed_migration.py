import numpy as np
import pytest

from inkgrammar.fitting.models import (
    CubicSegment,
    PrimitiveFrame,
    PrimitiveTemplate,
    TopologyTransform,
)
from inkgrammar.fitting.seed_migration import (
    migrate_legacy_seed,
    write_seed_migration_report,
)
from inkgrammar.fitting.spline import sample_curve_by_arclength


def cubic_template() -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id="migration-curve",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(1.0, 0.0)),
        segments=[CubicSegment(c1=(0.25, 0.8), c2=(0.75, 0.8), end=(1.0, 0.0))],
    )


def test_legacy_seed_migration_locks_endpoints_and_is_deterministic(tmp_path):
    source = cubic_template()
    result = migrate_legacy_seed(source, sample_count=512)
    repeated = migrate_legacy_seed(source, sample_count=512)
    report_path = tmp_path / "reports" / "migration.json"
    write_seed_migration_report(result, report_path)
    report_bytes = report_path.read_bytes()
    write_seed_migration_report(repeated, report_path)
    assert report_path.read_bytes() == report_bytes

    assert result.template.kind == "bspline-centerline"
    assert result.report["within_tolerance"] is True
    assert result.report["maximum_deviation_model_units"] < 0.05
    assert result.report["endpoints_locked"] is True
    assert result.report["canonical_seed_modified"] is False
    assert result.report["approved_for_canonical_use"] is False
    assert result.report == repeated.report
    assert result.template.model_dump(mode="json") == repeated.template.model_dump(mode="json")
    assert source.kind == "cubic-chain"
    assert source.segments == cubic_template().segments

    migrated = sample_curve_by_arclength(result.template, TopologyTransform(), 512)
    legacy = sample_curve_by_arclength(source, TopologyTransform(), 512)
    assert np.max(np.linalg.norm(migrated.xy - legacy.xy, axis=1)) < 0.05
    np.testing.assert_allclose(migrated.xy[[0, -1]], legacy.xy[[0, -1]], atol=1e-12)


def test_legacy_seed_migration_reports_rejected_approximation_and_loop_review():
    source = PrimitiveTemplate(
        id="complex-seed",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(2.0, 0.0)),
        segments=[
            CubicSegment(c1=(0.2, 1.0), c2=(0.8, 1.0), end=(1.0, 0.0)),
            CubicSegment(c1=(1.2, -1.0), c2=(1.8, -1.0), end=(2.0, 0.0)),
        ],
    )
    rejected = migrate_legacy_seed(
        source,
        control_point_count=4,
        maximum_deviation_model_units=1e-12,
        sample_count=512,
    )
    assert rejected.report["within_tolerance"] is False
    assert rejected.report["maximum_deviation_model_units"] >= 1e-12

    loop = PrimitiveTemplate(
        id="demo-looped",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(0.0, 0.0)),
        segments=[CubicSegment(c1=(0.0, 1.0), c2=(1.0, 1.0), end=(0.0, 0.0))],
    )
    loop_result = migrate_legacy_seed(loop, sample_count=512)
    assert loop_result.report["closed_source"] is True
    assert loop_result.report["closed_loop_review_required"] is True
    assert loop_result.report["visual_review_required"] is True


def test_legacy_seed_migration_rejects_invalid_controls_and_spline_input():
    source = cubic_template()
    with pytest.raises(ValueError, match="at least four"):
        migrate_legacy_seed(source, control_point_count=3)

    migrated = migrate_legacy_seed(source, sample_count=256)
    with pytest.raises(ValueError, match="legacy cubic-chain"):
        migrate_legacy_seed(migrated.template)
