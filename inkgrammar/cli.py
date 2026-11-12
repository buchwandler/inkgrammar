from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from . import __version__
from .data_cli import app as data_app
from .dataset_cli import app as dataset_app
from .fit import fit_join
from .io import load_layout, load_model
from .page import extract_labeled_page
from .qc import validate_model
from .raster import compare_text_to_image, render_text_mask
from .release import build_release
from .trajectory_cli import register_trajectory_commands

app = typer.Typer(no_args_is_help=True, help="InkGrammar writing-model tools.")
page_app = typer.Typer(no_args_is_help=True, help="Scan-page extraction tools.")
model_app = typer.Typer(no_args_is_help=True, help="Writing-model tools.")
fit_app = typer.Typer(no_args_is_help=True, help="Hierarchical handwriting fitting tools.")
font_app = typer.Typer(no_args_is_help=True, help="OpenType font export tools.")
app.add_typer(page_app, name="page")
app.add_typer(model_app, name="model")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(fit_app, name="fit")
register_trajectory_commands(fit_app)
app.add_typer(font_app, name="font")
CONTROL_SHEET_OUTPUT = Path("work/sas-1968/control/alphabet.png")


@app.command()
def version() -> None:
    """Print the setuptools-scm derived package version."""
    typer.echo(__version__)


@page_app.command("extract")
def page_extract(
    image: Path = typer.Argument(..., exists=True),  # noqa: B008
    layout: Path = typer.Option(..., "--layout", exists=True),  # noqa: B008
    output: Path = typer.Option(Path("work/page"), "--output"),  # noqa: B008
) -> None:
    """Segment and label an alphabet/sample page using a declared row layout."""
    manifest = extract_labeled_page(
        image,
        load_layout(layout),
        output,
    )
    typer.echo(f"Extracted {len(manifest['detections'])} labelled crops into {output}")
    typer.echo(f"Review: {output / 'review.png'}")
    typer.echo(f"Manifest: {output / 'detections.json'}")


@model_app.command("validate")
def model_validate(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    strict: bool = typer.Option(False, help="Treat warnings as failure."),  # noqa: B008
) -> None:
    """Validate components, glyph forms, provenance, and joins."""
    loaded = load_model(model)
    report = validate_model(loaded)
    for issue in report.issues:
        location = "/".join(value for value in (issue.model, issue.glyph) if value)
        typer.echo(f"{issue.severity.upper()} {issue.code} [{location}]: {issue.message}")
    typer.echo(
        f"QC: {report.glyph_count} glyph(s), {report.component_count} component(s), "
        f"{report.errors} error(s), {report.warnings} warning(s)"
    )
    if report.errors or (strict and report.warnings):
        raise typer.Exit(1)


@model_app.command("coverage")
def model_coverage_command(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    format_: str = typer.Option("json", "--format"),  # noqa: B008
    fail_incomplete: bool = typer.Option(False, "--fail-incomplete"),  # noqa: B008
    font: Path | None = typer.Option(None, "--font", exists=True),  # noqa: B008
    fit_project: Path | None = typer.Option(None, "--fit-project", exists=True),  # noqa: B008
) -> None:
    """Report glyph, provenance, regression and rights coverage."""
    from .coverage import (
        coverage_markdown,
        model_coverage,
        runtime_coverage_markdown,
        runtime_model_coverage,
    )
    from .models import WritingModelV3
    from .observations import ObservationManifest
    from .regression import RegressionManifest

    loaded = load_model(model)
    root = model.parent.parent
    observation_path = root / "observations" / loaded.id / "manifest.json"
    regression_path = root / "regressions" / loaded.id / "manifest.json"
    observations = (
        ObservationManifest.model_validate_json(observation_path.read_text())
        if observation_path.exists()
        else None
    )
    regressions = (
        RegressionManifest.model_validate_json(regression_path.read_text())
        if regression_path.exists()
        else None
    )
    connection_observations = []
    if fit_project is not None:
        if not isinstance(loaded, WritingModelV3):
            raise typer.BadParameter("--fit-project requires a schema-v3 runtime model")
        from .fitting.workspace import load_fit_project

        bundle = load_fit_project(fit_project)
        if bundle.project.runtime_model_id != loaded.id:
            raise typer.BadParameter("fit project runtime_model_id does not match the model")
        connection_observations = bundle.grammar.connection_observations

    if isinstance(loaded, WritingModelV3):
        report = runtime_model_coverage(
            loaded,
            observations=observations,
            regressions=regressions,
            connection_observations=connection_observations,
            font_path=font,
        )
        markdown = runtime_coverage_markdown
    else:
        if font is not None:
            raise typer.BadParameter("--font coverage requires a schema-v3 runtime model")
        report = model_coverage(loaded, observations=observations, regressions=regressions)
        markdown = coverage_markdown
    if format_ == "json":
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    elif format_ == "markdown":
        typer.echo(markdown(report))
    else:
        raise typer.BadParameter("format must be json or markdown")
    if fail_incomplete and not report["complete"]:
        raise typer.Exit(1)


@model_app.command("render")
def model_render(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    text: str = typer.Argument(...),  # noqa: B008
    output: Path = typer.Option(Path("render.png"), "--output"),  # noqa: B008
) -> None:
    """Render text from the writing model."""
    image = render_text_mask(load_model(model), text)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    typer.echo(str(output))


@model_app.command("render-svg")
def model_render_svg(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    text: str = typer.Argument(...),  # noqa: B008
    output: Path = typer.Option(Path("render.svg"), "--output"),  # noqa: B008
    outline: bool = typer.Option(False, "--outline/--no-outline"),
    centerline: bool = typer.Option(True, "--centerline/--no-centerline"),
    anchors: bool = typer.Option(True, "--anchors/--no-anchors"),
    tangents: bool = typer.Option(True, "--tangents/--no-tangents"),
    origins: bool = typer.Option(True, "--origins/--no-origins"),
    guides: bool = typer.Option(True, "--guides/--no-guides"),
    labels: bool = typer.Option(True, "--labels/--no-labels"),
) -> None:
    """Render text as SVG with optional shaping diagnostics."""
    from .svg import write_text_svg

    write_text_svg(
        load_model(model),
        text,
        output,
        show_outline=outline,
        show_centerline=centerline,
        show_anchors=anchors,
        show_tangents=tangents,
        show_origins=origins,
        show_guides=guides,
        show_labels=labels,
    )
    typer.echo(str(output))


@font_app.command("build")
def font_build(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(..., "--output"),  # noqa: B008
    family_name: str | None = typer.Option(None, "--family-name"),
    style_name: str = typer.Option("Regular", "--style-name"),
    quadratic_error: float = typer.Option(0.5, "--quadratic-error"),
    allow_development: bool = typer.Option(False, "--allow-development"),
) -> None:
    """Build a deterministic TTF from a compiled schema-v3 runtime model."""
    from .font import FontBuildConfig, build_opentype_font
    from .models import WritingModelV3

    try:
        runtime_model = load_model(model)
        if not isinstance(runtime_model, WritingModelV3):
            raise ValueError("font export requires a schema-v3 runtime model")
        result = build_opentype_font(
            runtime_model,
            output,
            FontBuildConfig(
                family_name=family_name,
                style_name=style_name,
                quadratic_error=quadratic_error,
            ),
            allow_development=allow_development,
        )
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "font": str(result.path),
                "report": str(result.report_path),
                "features": str(result.feature_source_path),
                "glyph_count": result.glyph_count,
                "font_ready": result.font_ready,
                "blockers": result.blockers,
                "tables": result.tables,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@font_app.command("build-fit")
def font_build_fit(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(..., "--output"),  # noqa: B008
    family_name: str | None = typer.Option(None, "--family-name"),
    style_name: str = typer.Option("Regular", "--style-name"),
    quadratic_error: float = typer.Option(0.5, "--quadratic-error"),
    allow_unpromoted: bool = typer.Option(False, "--allow-unpromoted"),
    allow_development: bool = typer.Option(False, "--allow-development"),
) -> None:
    """Compile a fitting project using promotion gates, then build its TTF."""
    from dataclasses import replace
    from tempfile import TemporaryDirectory

    from .compiler import compile_fit_project
    from .fitting.workspace import load_fit_project
    from .font import FontBuildConfig, build_opentype_font

    try:
        bundle = load_fit_project(project)
        with TemporaryDirectory(prefix="inkgrammar-font-build-fit-") as temporary:
            compiled_path = (Path(temporary) / "runtime-model.yaml").resolve()
            compile_config = bundle.project.compile.model_copy(
                update={"output": str(compiled_path)}
            )
            temp_bundle = replace(
                bundle,
                project=bundle.project.model_copy(update={"compile": compile_config}),
            )
            compiled = compile_fit_project(
                temp_bundle,
                require_promoted=not allow_unpromoted,
            )
            result = build_opentype_font(
                compiled.model,
                output,
                FontBuildConfig(
                    family_name=family_name,
                    style_name=style_name,
                    quadratic_error=quadratic_error,
                ),
                allow_development=allow_development,
            )
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "font": str(result.path),
                "report": str(result.report_path),
                "features": str(result.feature_source_path),
                "glyph_count": result.glyph_count,
                "font_ready": result.font_ready,
                "blockers": result.blockers,
                "tables": result.tables,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@font_app.command("plan")
def font_plan(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
    quadratic_error: float = typer.Option(0.5, "--quadratic-error"),
) -> None:
    """Print deterministic glyph, cmap, metrics, feature, and readiness planning."""
    from .font import FontBuildConfig, create_font_plan
    from .models import WritingModelV3

    try:
        runtime_model = load_model(model)
        if not isinstance(runtime_model, WritingModelV3):
            raise ValueError("font planning requires a schema-v3 runtime model")
        plan = create_font_plan(runtime_model, FontBuildConfig(quadratic_error=quadratic_error))
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    blockers = sorted({*plan.blockers, "OpenType shaping parity has not been verified"})
    report = {
        "glyph_order": plan.glyph_order,
        "cmap": [
            {"codepoint": f"U+{codepoint:04X}", "glyph": glyph}
            for codepoint, glyph in plan.cmap.items()
        ],
        "horizontal_metrics": plan.horizontal_metrics,
        "feature_source": plan.feature_source,
        "issues": plan.issues,
        "font_ready": not blockers,
        "blockers": blockers,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if output is None:
        typer.echo(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
        typer.echo(str(output))


@font_app.command("validate")
def font_validate(
    font: Path = typer.Argument(..., exists=True),  # noqa: B008
    model: Path | None = typer.Option(None, "--model", exists=True),  # noqa: B008
) -> None:
    """Reopen a TTF and validate tables, outlines, metrics, cmap, and features."""
    from .font import validate_opentype_font
    from .models import WritingModelV3

    runtime_model = load_model(model) if model is not None else None
    if runtime_model is not None and not isinstance(runtime_model, WritingModelV3):
        raise typer.BadParameter("font validation requires a schema-v3 runtime model")
    report = validate_opentype_font(font, model=runtime_model)
    typer.echo(
        json.dumps(
            {
                "valid": report.valid,
                "glyph_count": report.glyph_count,
                "tables": report.tables,
                "errors": report.errors,
                "warnings": report.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report.valid:
        raise typer.Exit(1)


@font_app.command("compare-runtime")
def font_compare_runtime(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    font: Path = typer.Argument(..., exists=True),  # noqa: B008
    text: str = typer.Argument(...),  # noqa: B008
    tolerance: float = typer.Option(1.0, "--tolerance"),
    require_harfbuzz: bool = typer.Option(False, "--require-harfbuzz"),
) -> None:
    """Compare runtime shaping with HarfBuzz output when uharfbuzz is installed."""
    from .font.parity import compare_runtime_to_font
    from .models import WritingModelV3

    runtime_model = load_model(model)
    if not isinstance(runtime_model, WritingModelV3):
        raise typer.BadParameter("runtime comparison requires a schema-v3 runtime model")
    try:
        report = compare_runtime_to_font(runtime_model, font, text, tolerance=tolerance)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["available"] and require_harfbuzz:
        raise typer.Exit(2)
    if report["available"] and not (report["sequence_matches"] and report["positions_match"]):
        raise typer.Exit(1)


@model_app.command("compare")
def model_compare(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    text: str = typer.Argument(...),  # noqa: B008
    reference: Path = typer.Argument(..., exists=True),  # noqa: B008
) -> None:
    """Compare generated model output with an observed reference crop."""
    metrics = compare_text_to_image(load_model(model), text, reference)
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@model_app.command("fit-join")
def model_fit_join(
    model: Path = typer.Argument(..., exists=True),  # noqa: B008
    text: str = typer.Argument(...),  # noqa: B008
    reference: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
) -> None:
    """Grid-search pair spacing/tension against an observed two-letter crop."""
    result = fit_join(load_model(model), text, reference)
    payload = {
        "pair_overrides": {
            text: {
                "gap_adjust": result.gap_adjust,
                "tension": result.tension,
                "evidence": [str(reference)],
                "score": result.score,
            }
        },
        "metrics": {
            "iou": result.score,
            "precision": result.precision,
            "recall": result.recall,
        },
    }
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(str(output))
    else:
        typer.echo(rendered, nl=False)


@app.command("build-release")
def build_release_command(
    models: list[Path] = typer.Argument(..., exists=True),  # noqa: B008
    version: str = typer.Option(..., "--version"),  # noqa: B008
    output: Path = typer.Option(Path("dist"), "--output"),  # noqa: B008
    requires_inkgrammar: str | None = typer.Option(None, "--requires-inkgrammar"),  # noqa: B008
    git_commit: str | None = typer.Option(None, "--git-commit", envvar="GITHUB_SHA"),  # noqa: B008
) -> None:
    """Build a consumer-only JSON/schema/QC release ZIP."""
    release_root, archive = build_release(
        models,
        output,
        version,
        requires_inkgrammar=requires_inkgrammar,
        git_commit=git_commit,
    )
    typer.echo(f"Built {release_root}")
    typer.echo(f"Built {archive}")
    typer.echo(f"Built {archive.with_suffix('.zip.sha256')}")


@fit_app.command("validate-project")
def fit_validate_project(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
) -> None:
    """Validate the source-free fitting project and all referenced metadata."""
    from .fitting.workspace import load_fit_project

    try:
        bundle = load_fit_project(project)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Validated {bundle.project.id}: {len(bundle.source_lock.sources)} source locks, "
        f"{len(bundle.corpus.samples)} samples, {len(bundle.grammar.primitives)} primitives, "
        f"{len(bundle.topology.glyphs)} topology glyphs"
    )


@fit_app.command("compile")
def fit_compile(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    allow_unpromoted: bool = typer.Option(False, "--allow-unpromoted"),
) -> None:
    """Compile promoted fitting data into the configured runtime model."""
    from .compiler import compile_fit_project
    from .fitting.workspace import load_fit_project

    try:
        result = compile_fit_project(
            load_fit_project(project), require_promoted=not allow_unpromoted
        )
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result.diagnostics, ensure_ascii=False, indent=2, sort_keys=True))


@fit_app.command("materialize")
def fit_materialize(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    source_dir: Path = typer.Option(..., "--source-dir", exists=True, file_okay=False),  # noqa: B008
    work_dir: Path | None = typer.Option(None, "--work-dir"),  # noqa: B008
) -> None:
    """Verify private source locks and regenerate crops, masks, fields, and review sheet."""
    from .fitting.workspace import load_fit_project, materialize_fit_project

    try:
        bundle = load_fit_project(project)
        result = materialize_fit_project(bundle, source_dir, work_dir=work_dir)
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Materialized {result.sample_count} samples from {result.source_count} locked sources "
        f"into {result.output_dir} ({result.artifact_count} artifacts)"
    )


@fit_app.command("optimize-primitive")
def fit_optimize_primitive(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    primitive: str = typer.Argument(...),
    resume: bool = typer.Option(False, "--resume"),  # noqa: B008
) -> None:
    """Fit a shared primitive from its fit-split glyph samples."""
    from .fitting.family_fit import optimize_primitive_family
    from .fitting.workspace import load_fit_project

    try:
        bundle = load_fit_project(project)
        artifacts = optimize_primitive_family(bundle, primitive, resume=resume)
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    training = artifacts.report["training"]
    typer.echo(
        f"Fitted {primitive}: training objective "
        f"{training['initial_objective']:.6f} -> {training['final_objective']:.6f}"
    )
    typer.echo(f"Checkpoint: {artifacts.checkpoint_path}")
    typer.echo(f"Report: {artifacts.json_report_path}")
    typer.echo(f"Visual review sheet: {artifacts.private_review_sheet_path}")


@fit_app.command("control-sheet")
def fit_control_sheet(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(CONTROL_SHEET_OUTPUT, "--output"),  # noqa: B008
    include_digits: bool = typer.Option(False, "--include-digits"),  # noqa: B008
    occurrence_deltas: bool = typer.Option(False, "--occurrence-deltas"),  # noqa: B008
) -> None:
    """Render a source-free, common-scale control sheet for the target alphabet."""
    from .fitting.control_sheet import write_control_sheet
    from .fitting.workspace import load_fit_project

    try:
        result = write_control_sheet(
            load_fit_project(project),
            output,
            include_digits=include_digits,
            occurrence_deltas=occurrence_deltas,
        )
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Control sheet: {output}")
    typer.echo(f"JSON report: {output.with_suffix('.json')}")
    typer.echo(f"Rendered {result['rendered_count']} of {result['target_count']} target glyphs")


@fit_app.command("connection-control-sheet")
def fit_connection_control_sheet(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(Path("work/connections.png"), "--output"),  # noqa: B008
) -> None:
    """Render source-free connection-rule previews and split coverage."""
    from .fitting.control_sheet import write_connection_control_sheet
    from .fitting.workspace import load_fit_project

    try:
        report = write_connection_control_sheet(load_fit_project(project), output)
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Connection sheet: {output}")
    typer.echo(f"JSON report: {output.with_suffix('.json')}")
    typer.echo(f"Control cases: {report['case_count']}")


@fit_app.command("context-control-sheet")
def fit_context_control_sheet(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(Path("work/context-variants.png"), "--output"),  # noqa: B008
) -> None:
    """Render source-free glyph-variant previews and class labels."""
    from .fitting.control_sheet import write_context_control_sheet
    from .fitting.workspace import load_fit_project

    try:
        report = write_context_control_sheet(load_fit_project(project), output)
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Context sheet: {output}")
    typer.echo(f"JSON report: {output.with_suffix('.json')}")
    typer.echo(f"Variants: {report['case_count']}")


@fit_app.command("connection-coverage")
def fit_connection_coverage(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Report source-free connection observation splits and pair coverage."""
    from .fitting.connection_fit import connection_observation_coverage
    from .fitting.workspace import load_fit_project

    report = connection_observation_coverage(
        load_fit_project(project).grammar.connection_observations
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


@fit_app.command("fit-glyphs")
def fit_glyphs(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Fit active occurrences in fit-split glyphs with shared primitive geometry fixed."""
    from .fitting.glyph_fit import fit_compositional_glyphs
    from .fitting.workspace import load_fit_project

    try:
        artifacts = fit_compositional_glyphs(load_fit_project(project))
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not artifacts:
        typer.echo("No fit-split glyphs with active instances.")
        return
    for artifact in artifacts:
        typer.echo(f"Fitted {artifact.report['sample_id']}")
        typer.echo(f"Report: {artifact.report_path}")
        typer.echo(f"Composition overlay: {artifact.overlay_path}")


@fit_app.command("validate-words")
def fit_validate_words(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Evaluate fit, validation, and hold-out word compositions without fitting parameters."""
    from .fitting.word_validation import validate_composed_words
    from .fitting.workspace import load_fit_project

    try:
        artifacts = validate_composed_words(load_fit_project(project))
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not artifacts:
        typer.echo("No word samples found.")
        return
    for artifact in artifacts:
        metrics = artifact.report["metrics"]
        typer.echo(
            f"Validated {artifact.report['sample_id']} ({artifact.report['split']}): "
            f"soft-mask loss {metrics['soft_mask_loss']:.6f}, "
            f"distance loss {metrics['symmetric_distance_loss']:.6f}"
        )
        typer.echo(f"Report: {artifact.report_path}")
        typer.echo(f"Composition overlay: {artifact.overlay_path}")


@fit_app.command("report")
def fit_report(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    report_format: str = typer.Option("markdown", "--format"),  # noqa: B008
) -> None:
    """Print generated primitive-fit reports without loading private source imagery."""
    from .fitting.family_fit import load_fit_reports
    from .fitting.workspace import load_fit_project

    if report_format not in {"json", "markdown"}:
        raise typer.BadParameter("--format must be json or markdown")
    try:
        bundle = load_fit_project(project)
        reports = load_fit_reports(bundle)
        if not reports:
            typer.echo("No fit reports found.")
        elif report_format == "json":
            typer.echo(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            for index, report in enumerate(reports):
                if index:
                    typer.echo("")
                report_path = bundle.project_dir / "reports" / f"{report['primitive_id']}.md"
                if report_path.is_symlink():
                    raise ValueError("fit report must not be a symlink")
                typer.echo(report_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc
