"""Commands for the deterministic dataset pipeline (the brief's section 33)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .dataset.materialize import materialize_dataset
from .dataset.project import load_dataset_project, verify_sources
from .dataset.review import render_review_sheets
from .dataset.status import render_report, summarize_status, validate_dataset

app = typer.Typer(no_args_is_help=True, help="Source-free dataset materialization tools.")


def _bundle(project: Path):
    return load_dataset_project(project)


def _default_dataset_root(bundle) -> Path:
    return bundle.private_root.parent / "dataset"


def _load_index(dataset_root: Path) -> dict:
    index_path = dataset_root / "index.json"
    if not index_path.is_file():
        raise typer.BadParameter(f"dataset index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


@app.command("verify-sources")
def verify_sources_cmd(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Verify private source files by SHA-256 and dimensions."""
    verification = verify_sources(_bundle(project))
    for check in verification.checks:
        typer.echo(f"{check.source_id}: {check.status}")
        for problem in check.problems:
            typer.echo(f"  {problem}")
    if not verification.ok:
        raise typer.Exit(code=1)
    typer.echo("All sources verified.")


@app.command("materialize")
def materialize_cmd(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(None, "--output"),  # noqa: B008
    review_dir: Path = typer.Option(None, "--review-dir"),  # noqa: B008
) -> None:
    """Materialize the dataset deterministically from the recipe and sources."""
    bundle = _bundle(project)
    output_dir = output or _default_dataset_root(bundle)
    index = materialize_dataset(bundle, output_dir, review_dir=review_dir)
    typer.echo(f"Materialized {len(index['observations'])} observations into {output_dir}")
    typer.echo(f"Index: {output_dir / 'index.json'}")


@app.command("review-sheet")
def review_sheet_cmd(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(..., "--output"),  # noqa: B008
    dataset_root: Path = typer.Option(None, "--dataset-root"),  # noqa: B008
) -> None:
    """Render deterministic review sheets from a materialized dataset."""
    bundle = _bundle(project)
    root = dataset_root or _default_dataset_root(bundle)
    written = render_review_sheets(root, output)
    typer.echo(f"Wrote {len(written)} review sheets to {output}")
    typer.echo(f"Index: {output / 'review-index.md'}")


@app.command("validate")
def validate_cmd(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    dataset_root: Path = typer.Option(None, "--dataset-root"),  # noqa: B008
) -> None:
    """Validate the materialized dataset against the section 38 rules."""
    bundle = _bundle(project)
    root = dataset_root or _default_dataset_root(bundle)
    index = _load_index(root)
    verification = verify_sources(bundle)
    validation = validate_dataset(bundle, index, dataset_root=root, verification=verification)
    for issue in validation.errors:
        typer.echo(f"ERROR {issue.rule} {issue.observation_id or ''}: {issue.message}")
    for issue in validation.warnings:
        typer.echo(f"WARN  {issue.rule} {issue.observation_id or ''}: {issue.message}")
    if not validation.ok:
        raise typer.Exit(code=1)
    typer.echo(f"Validation passed with {len(validation.warnings)} warnings.")


@app.command("status")
def status_cmd(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    dataset_root: Path = typer.Option(None, "--dataset-root"),  # noqa: B008
    report: Path = typer.Option(None, "--report"),  # noqa: B008
) -> None:
    """Report independent coverage counters and write the dataset report."""
    bundle = _bundle(project)
    root = dataset_root or _default_dataset_root(bundle)
    index = _load_index(root)
    verification = verify_sources(bundle)
    validation = validate_dataset(bundle, index, dataset_root=root, verification=verification)
    status = summarize_status(bundle, index, validation, verification, dataset_root=root)
    typer.echo(
        f"source pages verified             "
        f"{status.source_pages_verified}/{status.source_pages_total}"
    )
    typer.echo(f"observations declared             {status.observations_declared}")
    typer.echo(f"observations materialized         {status.observations_materialized}")
    typer.echo(f"label-reviewed                    {status.label_reviewed}")
    typer.echo(f"isolation-reviewed                {status.isolation_reviewed}")
    typer.echo(f"metric-reviewed                   {status.metric_reviewed}")
    typer.echo(f"stroke-semantics-reviewed         {status.stroke_semantics_reviewed}")
    typer.echo(f"connection examples transcribed   {status.connection_examples_transcribed}")
    typer.echo(f"ready for fitting                 {status.ready_for_fitting}")
    report_path = report or root / "report.md"
    report_path.write_text(render_report(status, validation), encoding="utf-8")
    typer.echo(f"Report: {report_path}")


@app.command("validate-sas-private")
def validate_sas_private_cmd(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    output: Path = typer.Option(None, "--output"),  # noqa: B008
    review_dir: Path = typer.Option(None, "--review-dir"),  # noqa: B008
) -> None:
    """One-shot authorized local run: verify, materialize, validate, review sheets, status."""
    bundle = _bundle(project)
    output_dir = output or _default_dataset_root(bundle)
    verification = verify_sources(bundle)
    if not verification.ok:
        for check in verification.checks:
            typer.echo(f"{check.source_id}: {check.status}")
        raise typer.Exit(code=1)
    typer.echo(f"sources verified: {len(verification.checks)}")
    index = materialize_dataset(bundle, output_dir, review_dir=review_dir)
    typer.echo(f"observations materialized: {len(index['observations'])}")
    validation = validate_dataset(bundle, index, dataset_root=output_dir, verification=verification)
    for issue in validation.errors:
        typer.echo(f"ERROR {issue.rule} {issue.observation_id or ''}: {issue.message}")
    for issue in validation.warnings:
        typer.echo(f"WARN  {issue.rule} {issue.observation_id or ''}: {issue.message}")
    sheets = render_review_sheets(output_dir, output_dir / "review")
    typer.echo(f"review sheets: {len(sheets)}")
    status = summarize_status(bundle, index, validation, verification, dataset_root=output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(render_report(status, validation), encoding="utf-8")
    typer.echo(f"report: {report_path}")
    if not validation.ok:
        raise typer.Exit(code=1)
