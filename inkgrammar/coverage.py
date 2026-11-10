"""Coverage reporting is separate from historical verification and structural QC."""

from __future__ import annotations

from collections import Counter

from .models import WritingModel
from .observations import ObservationManifest
from .regression import RegressionManifest

SAS_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzß0123456789"


def model_coverage(
    model: WritingModel,
    *,
    observations: ObservationManifest | None = None,
    regressions: RegressionManifest | None = None,
) -> dict:
    from .qc import validate_model

    qc = validate_model(model)
    expected = set(SAS_SYMBOLS if model.id == "sas-1968" else model.glyphs)
    covered = set(model.glyphs) & expected
    forms = {
        name: [
            char
            for char, glyph in model.glyphs.items()
            if name in glyph.forms or "default" in glyph.forms
        ]
        for name in ("isolated", "initial", "medial", "final")
    }
    blockers = []
    if model.license == "REPLACE-AFTER-RIGHTS-REVIEW":
        blockers.append("Model-data rights and license are unresolved.")
    for source in model.sources:
        if source.license is None:
            blockers.append(f"{source.id}: scan redistribution is not authorized; metadata only.")
    results = regressions.results if regressions else []
    fixtures = regressions.fixtures if regressions else []
    examples = (
        [item for item in observations.items if item.kind == "connection"] if observations else []
    )
    missing = sorted(expected - covered)
    missing_results = sorted(
        {fixture.id for fixture in fixtures} - {result.fixture_id for result in results}
    )
    failed = [result.fixture_id for result in results if result.status != "pass"]
    coverage_blockers = []
    if model.id == "sas-1968":
        groups = {fixture.group for fixture in fixtures}
        required_groups = {
            "isolated-uppercase",
            "isolated-lowercase",
            "digits",
            "stroke-order-backed",
            "connection-baseline",
            "connection-midline",
            "connection-attached",
            "numbered-connection-examples",
        }
        coverage_blockers.extend(
            f"Missing regression group: {group}" for group in sorted(required_groups - groups)
        )
        isolated = {
            fixture.text
            for fixture in fixtures
            if fixture.group.startswith("isolated-") or fixture.group == "digits"
        }
        coverage_blockers.extend(
            f"Missing isolated regression: {char}" for char in sorted(expected - isolated)
        )
        for group in ("connection-baseline", "connection-midline", "connection-attached"):
            splits = {fixture.split for fixture in fixtures if fixture.group == group}
            if not {"fit", "hold-out"} <= splits:
                coverage_blockers.append(f"Missing fit/hold-out split: {group}")
        if not examples:
            coverage_blockers.append("Missing source connection observations")
        if any(result.visual_review != "accepted" for result in results):
            coverage_blockers.append("Regression visual review is incomplete")
    return {
        "model_id": model.id,
        "glyph_count": len(model.glyphs),
        "expected_glyphs": sorted(expected),
        "covered_glyphs": sorted(covered),
        "missing_glyphs": missing,
        "unexpected_glyphs": sorted(set(model.glyphs) - expected),
        "glyphs_by_provenance": dict(Counter(g.provenance.status for g in model.glyphs.values())),
        "components_by_provenance": dict(
            Counter(c.provenance.status for c in model.components.values())
        ),
        "forms": forms,
        "connection_classes": sorted(
            {value for rule in model.connection_rules for value in (rule.from_class, rule.to_class)}
        ),
        "connection_example_count": len(examples),
        "regression_fixture_count": len(fixtures),
        "regression_pass_count": sum(result.status == "pass" for result in results),
        "regression_fail_count": sum(result.status == "fail" for result in results),
        "regression_not_run_count": sum(result.status == "not-run" for result in results)
        + len(missing_results),
        "regression_missing_results": missing_results,
        "source_rights_release_blockers": blockers,
        "coverage_blockers": coverage_blockers,
        "qc_errors": qc.errors,
        "complete": (
            not missing
            and not failed
            and not missing_results
            and not coverage_blockers
            and not qc.errors
        ),
        "note": "Coverage and schema validity do not establish historical verification.",
    }


def coverage_markdown(report: dict) -> str:
    lines = [
        f"# Coverage: {report['model_id']}",
        "",
        f"Glyphs: {report['glyph_count']}/{len(report['expected_glyphs'])}",
        f"Complete: {report['complete']}",
        "",
        "Missing glyphs: " + (", ".join(report["missing_glyphs"]) or "none"),
        "Connection classes: " + ", ".join(report["connection_classes"]),
        f"Connection examples: {report['connection_example_count']}",
        f"Regression fixtures: {report['regression_fixture_count']}",
        f"Regression passes: {report['regression_pass_count']}",
        f"Regression failures: {report['regression_fail_count']}",
        f"Regression not run: {report['regression_not_run_count']}",
        "",
    ]
    for name, glyphs in report["forms"].items():
        lines.append(f"{name}: {len(glyphs)} ({' '.join(glyphs)})")
    for name in ("glyphs_by_provenance", "components_by_provenance"):
        lines.extend(["", f"## {name.replace('_', ' ').capitalize()}"])
        lines.extend(f"- {status}: {count}" for status, count in report[name].items())
    lines.extend(["", "## Coverage blockers"])
    lines.extend(f"- {blocker}" for blocker in report["coverage_blockers"])
    lines.extend(["", "## Rights and release blockers"])
    lines.extend(f"- {blocker}" for blocker in report["source_rights_release_blockers"])
    lines.extend(["", report["note"], ""])
    return "\n".join(lines)
