from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..compiler import _compile_fit_model, _select_fit_templates
from ..models import WritingModelV3
from .family_fit import (
    load_fit_reports,
    optimizer_backend_for_method,
    validate_fit_report_inputs,
)
from .models import PrimitiveTemplate
from .workspace import FitProjectBundle

PreviewGeometry = Literal["canonical", "latest-candidate"]


@dataclass(frozen=True)
class PreviewBuild:
    model: WritingModelV3
    diagnostics: dict[str, Any]
    templates: dict[str, PrimitiveTemplate]
    template_sources: dict[str, str]


def _fit_command(bundle: FitProjectBundle, primitive_id: str) -> str:
    try:
        project_path = bundle.project_path.relative_to(bundle.repository_root).as_posix()
    except ValueError:
        project_path = str(bundle.project_path)
    return f"inkgrammar fit optimize-primitive {project_path} {primitive_id}"


def _candidate_failure(
    bundle: FitProjectBundle, primitive_id: str, message: str
) -> ValueError:
    return ValueError(
        f"latest candidate for {primitive_id} is {message}.\n"
        f"Run:\n{_fit_command(bundle, primitive_id)}"
    )


def _candidate_templates(
    bundle: FitProjectBundle,
    seed_templates: Mapping[str, PrimitiveTemplate],
) -> tuple[dict[str, PrimitiveTemplate], dict[str, str], dict[str, dict[str, Any]]]:
    reports = {report["primitive_id"]: report for report in load_fit_reports(bundle)}
    templates = dict(seed_templates)
    sources = {primitive_id: "seed-fallback" for primitive_id in templates}
    report_details: dict[str, dict[str, Any]] = {}
    objective_version = bundle.fit_config.objective.version
    expected_backend = optimizer_backend_for_method(bundle.fit_config.optimizer.method)
    families = sorted(
        (
            family
            for family in bundle.families.families
            if family.fit and family.review_status != "uncertain"
        ),
        key=lambda family: family.primitive,
    )
    if not families:
        raise ValueError("fit project has no eligible primitive families for candidate preview")

    for family in families:
        primitive_id = family.primitive
        report = reports.get(primitive_id)
        if report is None:
            raise _candidate_failure(bundle, primitive_id, "missing a current fit report")
        if report.get("format") != "inkgrammar-fit-report":
            raise _candidate_failure(bundle, primitive_id, "has an unsupported report format")
        report_version = report.get("format_version")
        if report_version != objective_version:
            raise _candidate_failure(
                bundle,
                primitive_id,
                f"stale: report format_version={report_version}, "
                f"project objective version={objective_version}",
            )
        if report.get("project_id") != bundle.project.id:
            raise _candidate_failure(bundle, primitive_id, "for a different fit project")
        if report.get("primitive_id") != primitive_id:
            raise _candidate_failure(bundle, primitive_id, "for a different primitive")
        if report.get("objective_version") != objective_version:
            raise _candidate_failure(
                bundle,
                primitive_id,
                f"stale: report objective_version={report.get('objective_version')}, "
                f"project objective version={objective_version}",
            )

        optimizer = report.get("optimizer")
        if not isinstance(optimizer, Mapping) or optimizer.get("success") is not True:
            raise _candidate_failure(bundle, primitive_id, "not a successful optimizer result")
        if optimizer.get("backend") != expected_backend:
            raise _candidate_failure(
                bundle,
                primitive_id,
                f"stale: report optimizer.backend={optimizer.get('backend')}, "
                f"current optimizer backend={expected_backend}",
            )

        try:
            candidate = PrimitiveTemplate.model_validate(report.get("primitive"))
        except (TypeError, ValueError) as exc:
            raise _candidate_failure(
                bundle, primitive_id, "has invalid primitive geometry"
            ) from exc
        if candidate.id != primitive_id:
            raise _candidate_failure(
                bundle, primitive_id, "contains geometry for a different primitive"
            )
        try:
            validate_fit_report_inputs(bundle, primitive_id, report)
        except ValueError as exc:
            raise _candidate_failure(
                bundle, primitive_id, f"has stale or unverifiable inputs: {exc}"
            ) from exc

        templates[primitive_id] = candidate
        sources[primitive_id] = "latest-candidate"
        report_details[primitive_id] = {
            "format_version": report_version,
            "objective_version": objective_version,
            "optimizer_backend": expected_backend,
            "visual_review_status": (
                report.get("visual_review", {}).get("status")
                if isinstance(report.get("visual_review"), Mapping)
                else None
            ),
        }

    return templates, sources, report_details


def build_fit_preview_model(
    bundle: FitProjectBundle,
    *,
    geometry: PreviewGeometry,
) -> PreviewBuild:
    if geometry not in {"canonical", "latest-candidate"}:
        raise ValueError(f"unsupported preview geometry: {geometry}")

    canonical_templates, promotion = _select_fit_templates(bundle, require_promoted=False)
    if geometry == "canonical":
        templates = canonical_templates
        sources = dict(promotion)
        report_details: dict[str, dict[str, Any]] = {}
    else:
        seed_templates = {
            primitive_id: bundle.grammar.primitives[primitive_id]
            for primitive_id in canonical_templates
        }
        templates, sources, report_details = _candidate_templates(bundle, seed_templates)

    result = _compile_fit_model(bundle, templates, promotion)
    if not isinstance(result.model, WritingModelV3):
        raise ValueError("fit preview requires runtime schema version 3")

    provenance = {
        **result.model.provenance,
        "preview_geometry": geometry,
        "canonical": geometry == "canonical",
        "unpromoted": geometry == "latest-candidate",
        "primitive_geometry": sources,
    }
    model_data = result.model.model_dump(mode="python")
    model_data["provenance"] = provenance
    model = WritingModelV3.model_validate(model_data)

    diagnostics = {
        **result.diagnostics,
        "preview_geometry": geometry,
        "canonical": geometry == "canonical",
        "unpromoted": geometry == "latest-candidate",
        "primitive_geometry": sources,
        "fit_reports": report_details,
    }
    return PreviewBuild(
        model=model,
        diagnostics=diagnostics,
        templates=dict(templates),
        template_sources=dict(sources),
    )
