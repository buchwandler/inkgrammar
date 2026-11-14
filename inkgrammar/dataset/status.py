"""Dataset coverage status and validation (the brief's sections 38-39, 97-98).

Coverage counters are reported independently; readiness is never reduced to a
single percentage. Validation distinguishes hard errors from review warnings;
automatic outputs stay candidates until review decides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import Field

from ..io import load_layout
from ..models import StrictModel
from .layouts import load_region_layout
from .preprocess import label_components
from .project import DatasetProjectBundle, SourceVerification

FITTING_USABLE = ("usable", "usable-reviewed")
ISOLATION_ACCEPTED = ("usable", "usable-reviewed")
CONNECTION_KINDS = ("connection-pair", "word", "sequence")


class ValidationIssue(StrictModel):
    rule: str
    message: str
    observation_id: str | None = None


class DatasetValidation(StrictModel):
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class DatasetStatus(StrictModel):
    source_pages_verified: int
    source_pages_total: int
    observations_declared: int
    observations_materialized: int
    label_reviewed: int
    isolation_reviewed: int
    metric_reviewed: int
    stroke_semantics_reviewed: int
    connection_examples_transcribed: int
    ready_for_fitting: int
    by_kind: dict[str, int] = Field(default_factory=dict)
    warning_counts: dict[str, int] = Field(default_factory=dict)


def _declared_count(bundle: DatasetProjectBundle) -> int:
    total = 0
    for page in bundle.project.pages:
        layout_path = bundle.layout_path(page)
        if page.extractor == "labeled-rows-v1":
            layout = load_layout(layout_path)
            total += sum(len(row.labels) for row in layout.rows)
        else:
            layout = load_region_layout(layout_path)
            total += len(layout.regions)
    return total


def _frame_available(record: dict) -> bool:
    metrics = record.get("metrics") or {}
    return metrics.get("baseline_y") is not None and metrics.get("x_height_y") is not None


def _artifact_missing(record: dict, dataset_root: Path) -> str | None:
    for name, rel_path in (record.get("artifacts") or {}).items():
        if rel_path is not None and not (dataset_root / rel_path).is_file():
            return name
    return None


def _mask_warnings(record: dict, dataset_root: Path) -> list[ValidationIssue]:
    clean_rel = (record.get("artifacts") or {}).get("clean_mask")
    if clean_rel is None or not (dataset_root / clean_rel).is_file():
        return []
    mask = np.asarray(Image.open(dataset_root / clean_rel).convert("L")) > 0
    issues = []
    obs_id = record["id"]
    if mask.size and (
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    ):
        issues.append(
            ValidationIssue(
                rule="mask-touching-crop-edge",
                observation_id=obs_id,
                message="clean mask touches the crop edge",
            )
        )
    components = label_components(mask)
    areas = [int(ys.size) for ys, _ in components]
    total = sum(areas)
    large = [area for area in areas if total and area > 0.1 * total]
    if len(large) >= 2:
        issues.append(
            ValidationIssue(
                rule="multiple-large-connected-components",
                observation_id=obs_id,
                message=f"{len(large)} large connected components in clean mask",
            )
        )
    if len(components) >= 2:
        issues.append(
            ValidationIssue(
                rule="neighboring-ink-likely-present",
                observation_id=obs_id,
                message=f"{len(components)} connected components in clean mask",
            )
        )
    return issues


def validate_dataset(
    bundle: DatasetProjectBundle,
    index: dict,
    *,
    dataset_root: Path,
    verification: SourceVerification | None = None,
) -> DatasetValidation:
    """Run the section 38 error rules and review warnings."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if verification is not None:
        for check in verification.checks:
            if check.status != "verified":
                rule = (
                    "missing-source-hash"
                    if check.status == "hash-mismatch"
                    else f"source-{check.status}"
                )
                errors.append(
                    ValidationIssue(
                        rule=rule,
                        observation_id=None,
                        message=f"{check.source_id}: {'; '.join(check.problems)}",
                    )
                )

    sources = {source.source_id: source for source in _manifest_sources(bundle)}
    layout_regions = _layout_region_ids(bundle)
    seen_ids: set[str] = set()
    for record in index.get("observations", {}).values():
        obs_id = record["id"]
        if obs_id in seen_ids:
            errors.append(
                ValidationIssue(
                    rule="duplicate-observation-id", observation_id=obs_id, message=obs_id
                )
            )
        seen_ids.add(obs_id)

        source = sources.get(record["source_id"])
        if source is None:
            errors.append(
                ValidationIssue(
                    rule="unknown-source-id",
                    observation_id=obs_id,
                    message=f"unknown source: {record['source_id']}",
                )
            )
        else:
            x0, y0, x1, y1 = record["source_bbox"]
            if x1 > source.width_px or y1 > source.height_px:
                errors.append(
                    ValidationIssue(
                        rule="region-outside-source-bounds",
                        observation_id=obs_id,
                        message=f"bbox {record['source_bbox']} outside source",
                    )
                )

        extraction = record["extraction"]
        region_key = f"{extraction['layout_id']}:{extraction['layout_region_id']}"
        if region_key not in layout_regions:
            errors.append(
                ValidationIssue(
                    rule="missing-layout-region",
                    observation_id=obs_id,
                    message=f"layout region not found: {record['extraction']['layout_region_id']}",
                )
            )

        if (record.get("review_patch") or {}).get("status") == "stale":
            errors.append(
                ValidationIssue(
                    rule="stale-review-patch",
                    observation_id=obs_id,
                    message="review patch input hashes do not match the current extraction",
                )
            )

        if record["review_status"] == "accepted":
            missing = _artifact_missing(record, dataset_root)
            if missing is not None:
                errors.append(
                    ValidationIssue(
                        rule="accepted-without-materialized-artifact",
                        observation_id=obs_id,
                        message=f"missing artifact: {missing}",
                    )
                )

        if record["fitting_status"] in FITTING_USABLE and not _frame_available(record):
            errors.append(
                ValidationIssue(
                    rule="fitting-usable-without-coordinate-frame",
                    observation_id=obs_id,
                    message="fitting-usable observation lacks baseline/x-height frame",
                )
            )

        errors.extend(_sequence_disagreement(record))
        errors.extend(_source_explicit_strokes(record))

        confidence = record["extraction"].get("detection_confidence")
        if confidence is not None and confidence < 0.5:
            warnings.append(
                ValidationIssue(
                    rule="low-segmentation-confidence",
                    observation_id=obs_id,
                    message=f"detection confidence {confidence}",
                )
            )
        metrics = record.get("metrics") or {}
        if not metrics or metrics.get("metric_status") == "missing":
            warnings.append(
                ValidationIssue(
                    rule="metric-confidence-low",
                    observation_id=obs_id,
                    message="no metric guides available",
                )
            )
        for stroke in (record.get("annotation") or {}).get("strokes", []):
            if stroke.get("start") is None or stroke.get("end") is None:
                warnings.append(
                    ValidationIssue(
                        rule="start-end-candidate-ambiguous",
                        observation_id=obs_id,
                        message=f"stroke {stroke.get('id')} lacks start/end annotation",
                    )
                )
        warnings.extend(_mask_warnings(record, dataset_root))

    return DatasetValidation(errors=errors, warnings=warnings)


def _sequence_disagreement(record: dict) -> list[ValidationIssue]:
    if record["kind"] not in CONNECTION_KINDS:
        return []
    relations = record.get("relations") or {}
    sequence = (record.get("detection") or {}).get("sequence") or []
    issues = []
    if relations.get("sequence_text") and relations["sequence_text"] != record["label"]:
        issues.append(
            ValidationIssue(
                rule="sequence-relation-disagreement",
                observation_id=record["id"],
                message="relations.sequence_text disagrees with the declared label",
            )
        )
    if len(sequence) == 2 and (
        relations.get("left_label") not in (None, sequence[0])
        or relations.get("right_label") not in (None, sequence[1])
    ):
        issues.append(
            ValidationIssue(
                rule="sequence-relation-disagreement",
                observation_id=record["id"],
                message="left/right labels disagree with the declared sequence",
            )
        )
    return issues


def _source_explicit_strokes(record: dict) -> list[ValidationIssue]:
    annotation = record.get("annotation") or {}
    strokes = annotation.get("strokes") or []
    has_explicit = any(stroke.get("direction_status") == "source-explicit" for stroke in strokes)
    if not has_explicit:
        return []
    detection = record.get("detection") or {}
    if detection.get("annotation_regions"):
        return []
    return [
        ValidationIssue(
            rule="source-explicit-stroke-order-without-annotation-region",
            observation_id=record["id"],
            message="stroke marked source-explicit but no source annotation region recorded",
        )
    ]


def _manifest_sources(bundle: DatasetProjectBundle):
    from .project import load_observation_manifest

    return load_observation_manifest(bundle.observation_manifest_path).sources


def _layout_region_ids(bundle: DatasetProjectBundle) -> set[str]:
    keys = set()
    for page in bundle.project.pages:
        if page.extractor == "labeled-rows-v1":
            layout = load_layout(bundle.layout_path(page))
            for row in layout.rows:
                for index in range(len(row.labels)):
                    keys.add(f"{layout.id}:{row.id}:{index}")
        else:
            layout = load_region_layout(bundle.layout_path(page))
            for region in layout.regions:
                keys.add(f"{layout.id}:{region.id}")
    return keys


def summarize_status(
    bundle: DatasetProjectBundle,
    index: dict,
    validation: DatasetValidation,
    verification: SourceVerification,
    *,
    dataset_root: Path,
) -> DatasetStatus:
    records = list(index.get("observations", {}).values())
    by_kind: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for record in records:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
    for issue in validation.warnings:
        warning_counts[issue.rule] = warning_counts.get(issue.rule, 0) + 1
    error_ids = {issue.observation_id for issue in validation.errors if issue.observation_id}
    return DatasetStatus(
        source_pages_verified=sum(1 for check in verification.checks if check.status == "verified"),
        source_pages_total=len(verification.checks),
        observations_declared=_declared_count(bundle),
        observations_materialized=len(records),
        label_reviewed=sum(1 for record in records if record["review_status"] == "accepted"),
        isolation_reviewed=sum(
            1 for record in records if record.get("isolation_status") in ISOLATION_ACCEPTED
        ),
        metric_reviewed=sum(
            1
            for record in records
            if (record.get("metrics") or {}).get("metric_status") == "reviewed"
        ),
        stroke_semantics_reviewed=sum(
            1 for record in records if (record.get("annotation") or {}).get("strokes")
        ),
        connection_examples_transcribed=sum(
            1 for record in records if record["kind"] in CONNECTION_KINDS
        ),
        ready_for_fitting=sum(
            1
            for record in records
            if record["review_status"] == "accepted"
            and record.get("isolation_status") in ISOLATION_ACCEPTED
            and _frame_available(record)
            and _artifact_missing(record, dataset_root) is None
            and record["id"] not in error_ids
        ),
        by_kind=by_kind,
        warning_counts=warning_counts,
    )


def render_report(status: DatasetStatus, validation: DatasetValidation) -> str:
    """Markdown report (the brief's section 98) for human inspection."""
    lines = [
        "# SAS dataset report",
        "",
        "## Coverage",
        "",
        "| counter | value |",
        "| --- | --- |",
        f"| source pages verified | {status.source_pages_verified}/{status.source_pages_total} |",
        f"| observations declared | {status.observations_declared} |",
        f"| observations materialized | {status.observations_materialized} |",
        f"| label-reviewed | {status.label_reviewed} |",
        f"| isolation-reviewed | {status.isolation_reviewed} |",
        f"| metric-reviewed | {status.metric_reviewed} |",
        f"| stroke-semantics-reviewed | {status.stroke_semantics_reviewed} |",
        f"| connection examples transcribed | {status.connection_examples_transcribed} |",
        f"| ready for fitting | {status.ready_for_fitting} |",
        "",
        "## Observations by kind",
        "",
        "| kind | count |",
        "| --- | --- |",
    ]
    lines.extend(f"| {kind} | {count} |" for kind, count in sorted(status.by_kind.items()))
    lines.extend(["", "## Warnings", "", "| rule | count |", "| --- | --- |"])
    lines.extend(f"| {rule} | {count} |" for rule, count in sorted(status.warning_counts.items()))
    lines.extend(["", "## Errors", ""])
    if validation.errors:
        lines.extend(
            f"- `{issue.rule}` {issue.observation_id or ''}: {issue.message}"
            for issue in validation.errors
        )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
