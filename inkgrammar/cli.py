from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from . import __version__
from .data_cli import app as data_app
from .fit import fit_join
from .io import load_layout, load_model
from .page import extract_labeled_page
from .qc import validate_model
from .raster import compare_text_to_image, render_text_mask
from .release import build_release

app = typer.Typer(no_args_is_help=True, help="InkGrammar writing-model tools.")
page_app = typer.Typer(no_args_is_help=True, help="Scan-page extraction tools.")
model_app = typer.Typer(no_args_is_help=True, help="Writing-model tools.")
fit_app = typer.Typer(no_args_is_help=True, help="Hierarchical handwriting fitting tools.")
app.add_typer(page_app, name="page")
app.add_typer(model_app, name="model")
app.add_typer(data_app, name="data")
app.add_typer(fit_app, name="fit")


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
) -> None:
    """Report glyph, provenance, regression and rights coverage."""
    from .coverage import coverage_markdown, model_coverage
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
    report = model_coverage(loaded, observations=observations, regressions=regressions)
    if format_ == "json":
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    elif format_ == "markdown":
        typer.echo(coverage_markdown(report))
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
