"""Coverage reporting is separate from historical verification and structural QC."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .models import WritingModel, WritingModelV3
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
        [item for item in observations.items if item.kind == "connection-pair"]
        if observations
        else []
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


def runtime_model_coverage(
    model: WritingModelV3,
    *,
    observations: ObservationManifest | None = None,
    regressions: RegressionManifest | None = None,
    connection_observations: list | None = None,
    font_path: str | Path | None = None,
) -> dict:
    expected = set(SAS_SYMBOLS if model.id == "sas-1968" else model.glyphs)
    modeled = set(model.glyphs) & expected
    missing = sorted(expected - modeled)
    source_glyph_items = [
        item
        for item in (observations.items if observations else [])
        if item.kind == "glyph-isolated"
    ]

    def codepoints(items, predicate) -> set[str]:
        characters = set()
        for item in items:
            if not predicate(item):
                continue
            for value in item.unicode:
                if value.startswith("U+"):
                    characters.add(chr(int(value[2:], 16)))
        return characters & expected

    observed = codepoints(source_glyph_items, lambda item: True)
    accepted_observed = codepoints(
        source_glyph_items, lambda item: item.review_status == "accepted"
    )
    fit_usable_observed = codepoints(
        source_glyph_items,
        lambda item: item.review_status == "accepted" and item.fitting_status == "usable",
    )
    source_connections = [
        item
        for item in (observations.items if observations else [])
        if item.kind == "connection-pair"
    ]
    fit_connections = connection_observations or []
    fit_splits = Counter(item.split for item in fit_connections)
    character_pairs = Counter(
        f"{item.from_character}{item.to_character}" for item in fit_connections
    )
    model_pairs = sorted(
        {f"{rule.from_class} → {rule.to_class}" for rule in model.connection_rules}
    )
    promotion = model.provenance.get("primitive_promotion", {})
    promotion_counts = Counter(promotion.values()) if isinstance(promotion, dict) else Counter()

    rights_blockers: list[str] = []
    if model.license.strip().upper() in {"NOASSERTION", "UNKNOWN", "REPLACE-AFTER-RIGHTS-REVIEW"}:
        rights_blockers.append("Model license or rights review is unresolved.")
    if observations is not None:
        unresolved = sum(
            source.rights_status.lower() not in {"approved", "cleared", "licensed", "public-domain"}
            for source in observations.sources
        )
        if unresolved:
            rights_blockers.append(f"Rights review is unresolved for {unresolved} source(s).")
    if any(source.license is None for source in model.sources):
        rights_blockers.append("At least one model source has no redistribution license.")

    coverage_blockers: list[str] = []
    if missing:
        coverage_blockers.append(f"{len(missing)} required glyph(s) have no runtime geometry.")
    if expected - accepted_observed:
        coverage_blockers.append(
            f"{len(expected - accepted_observed)} required glyph(s) lack accepted source labels."
        )
    if expected - fit_usable_observed:
        coverage_blockers.append(
            f"{len(expected - fit_usable_observed)} required glyph(s) lack usable fit observations."
        )
    if not source_connections:
        coverage_blockers.append("No source connection observations are declared.")
    if not fit_connections:
        coverage_blockers.append(
            "No fit-project connection observations are available; "
            "class-pair coverage is unassessed."
        )
    if model.id == "sas-1968" and not model.connection_rules:
        coverage_blockers.append("No runtime connection rules are compiled.")
    if any(value != "promoted" for value in promotion.values()):
        coverage_blockers.append("One or more runtime primitives use unpromoted fallback geometry.")
    if any(rule.mode == "synthetic-bridge" for rule in model.connection_rules):
        coverage_blockers.append("Synthetic bridge rules are not font-compatible.")

    font_artifact = None
    font_blockers = [*rights_blockers, *coverage_blockers]
    if font_path is None:
        font_blockers.append("No font artifact was supplied for structural validation.")
    else:
        from .font import validate_opentype_font

        path = Path(font_path)
        validation = validate_opentype_font(path, model=model)
        build_report_path = path.with_suffix(".report.json")
        build_report = (
            json.loads(build_report_path.read_text(encoding="utf-8"))
            if build_report_path.exists()
            else None
        )
        font_artifact = {
            "filename": path.name,
            "valid": validation.valid,
            "tables": validation.tables,
            "errors": validation.errors,
            "build_report_present": build_report is not None,
            "build_report_font_ready": build_report.get("font_ready") if build_report else None,
        }
        if not validation.valid:
            font_blockers.extend(validation.errors)
        if build_report is None:
            font_blockers.append("Font build readiness report is missing.")
        elif not build_report.get("font_ready", False):
            font_blockers.extend(build_report.get("blockers", []))

    font_blockers.append("OpenType shaping parity has not been verified for the required corpus.")
    regression_fixtures = regressions.fixtures if regressions else []
    regression_results = regressions.results if regressions else []
    regression_missing = sorted(
        {fixture.id for fixture in regression_fixtures}
        - {result.fixture_id for result in regression_results}
    )
    complete = (
        not missing
        and not (expected - accepted_observed)
        and not (expected - fit_usable_observed)
        and bool(source_connections)
        and bool(fit_connections)
        and bool(model.connection_rules)
        and not regression_missing
        and all(result.status == "pass" for result in regression_results)
    )
    return {
        "model_id": model.id,
        "schema_version": 3,
        "model_status": model.status,
        "glyph_coverage": {
            "expected": len(expected),
            "modeled": len(modeled),
            "missing": missing,
            "observed_in_sources": len(observed),
            "accepted_source_labels": len(accepted_observed),
            "usable_fit_observations": len(fit_usable_observed),
            "observed_characters": sorted(observed),
        },
        "variant_count": sum(len(glyph.variants) for glyph in model.glyphs.values()),
        "context_rule_count": len(model.context_rules),
        "connection_coverage": {
            "source_observation_count": len(source_connections),
            "fit_project_observation_count": len(fit_connections),
            "fit_split_counts": {
                split: fit_splits.get(split, 0) for split in ("fit", "validation", "hold-out")
            },
            "fit_project_character_pairs": dict(sorted(character_pairs.items())),
            "runtime_rule_count": len(model.connection_rules),
            "runtime_class_pairs": model_pairs,
            "class_pair_evidence_status": "unassessed; character pairs are not mapped to classes",
        },
        "primitive_promotion_counts": dict(sorted(promotion_counts.items())),
        "rights_blockers": rights_blockers,
        "coverage_blockers": coverage_blockers,
        "font_artifact": font_artifact,
        "font_ready": not font_blockers,
        "font_readiness_blockers": sorted(set(font_blockers)),
        "regression_fixture_count": len(regression_fixtures),
        "regression_missing_results": regression_missing,
        "complete": complete,
        "note": (
            "Staged coverage is not historical verification. "
            "Missing observations and class mappings are reported, not inferred."
        ),
    }


def runtime_coverage_markdown(report: dict) -> str:
    glyphs = report["glyph_coverage"]
    connections = report["connection_coverage"]
    lines = [
        f"# Runtime coverage: {report['model_id']}",
        "",
        f"Schema/status: v{report['schema_version']} / {report['model_status']}",
        f"Modeled glyphs: {glyphs['modeled']}/{glyphs['expected']}",
        f"Accepted source labels: {glyphs['accepted_source_labels']}/{glyphs['expected']}",
        f"Usable fit observations: {glyphs['usable_fit_observations']}/{glyphs['expected']}",
        f"Missing runtime glyphs: {', '.join(glyphs['missing']) or 'none'}",
        f"Variants/context rules: {report['variant_count']}/{report['context_rule_count']}",
        f"Connection source observations: {connections['source_observation_count']}",
        f"Fit-project connection observations: {connections['fit_project_observation_count']}",
        f"Runtime connection rules: {connections['runtime_rule_count']}",
        "Class-pair evidence: " + connections["class_pair_evidence_status"],
        f"Font ready: {report['font_ready']}",
        f"Coverage complete: {report['complete']}",
        "",
        "## Coverage blockers",
        *(f"- {blocker}" for blocker in report["coverage_blockers"]),
        "",
        "## Rights blockers",
        *(f"- {blocker}" for blocker in report["rights_blockers"]),
        "",
        "## Font readiness blockers",
        *(f"- {blocker}" for blocker in report["font_readiness_blockers"]),
        "",
        report["note"],
        "",
    ]
    return "\n".join(lines)


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
