from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from . import __version__
from .fit import fit_join
from .io import load_layout, load_model
from .page import extract_labeled_page
from .qc import validate_model
from .raster import compare_text_to_image, render_text_mask
from .release import build_release

app = typer.Typer(no_args_is_help=True, help="ScriptStrokes writing-model tools.")
page_app = typer.Typer(no_args_is_help=True, help="Scan-page extraction tools.")
model_app = typer.Typer(no_args_is_help=True, help="Writing-model tools.")
app.add_typer(page_app, name="page")
app.add_typer(model_app, name="model")


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
) -> None:
    """Build a consumer-only JSON/schema/QC release ZIP."""
    release_root, archive = build_release(models, output, version)
    typer.echo(f"Built {release_root}")
    typer.echo(f"Built {archive}")
