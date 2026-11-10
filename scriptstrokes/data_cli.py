"""Commands for explicit runtime data installation and inspection."""
import typer

from .data import (
    install_release,
    list_installed_releases,
    released_model_path,
    remove_release,
    resolve_release,
    verify_release,
)

app = typer.Typer(no_args_is_help=True, help="Versioned runtime data tools.")


@app.command("install")
def install(release: str = typer.Argument("latest")) -> None:
    """Download, verify and atomically install a data release."""
    resolved, url = resolve_release(release)
    typer.echo(f"Release URL: {url}")
    root = install_release(release)
    typer.echo(f"Version: {resolved}")
    typer.echo("Archive and internal SHA-256 checksums verified.")
    typer.echo(f"Installed: {root}")


@app.command("list")
def list_releases() -> None:
    """List locally installed versions without network access."""
    for release in list_installed_releases():
        typer.echo(release)


@app.command("path")
def path(release: str) -> None:
    """Print the verified local installation path."""
    typer.echo(str(verify_release(release)))


@app.command("verify")
def verify(release: str) -> None:
    """Verify internal checksums, compatibility and catalog integrity."""
    typer.echo(f"Verified: {verify_release(release)}")


@app.command("remove")
def remove(release: str) -> None:
    """Remove an installed release, leaving downloaded archives intact."""
    remove_release(release)
    typer.echo(f"Removed: {release}")


@app.command("model")
def model(model_id: str, release: str = typer.Option(..., "--release")) -> None:
    """Print a verified installed model path without network access."""
    typer.echo(str(released_model_path(model_id, release=release)))
