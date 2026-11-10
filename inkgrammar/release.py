from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from importlib.metadata import version as package_version
from pathlib import Path

from packaging.version import Version

from .models import PageLayout, WritingModel
from .observations import ObservationManifest
from .qc import validate_model
from .regression import RegressionManifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_release(
    model_paths: list[str | Path],
    output_root: str | Path,
    version: str,
    *,
    requires_inkgrammar: str | None = None,
    git_commit: str | None = None,
) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
        raise ValueError("invalid release version")
    installed = Version(package_version("inkgrammar"))
    if requires_inkgrammar is None:
        requires_inkgrammar = f">={installed.public},<{installed.major}.{installed.minor + 1}"
    models = [
        WritingModel.model_validate_json(Path(path).read_text())
        if str(path).endswith(".json")
        else None
        for path in model_paths
    ]
    if any(model is not None for model in models):
        raise ValueError("JSON source models are not supported in this MVP; use YAML")

    from .io import load_model

    loaded = [load_model(path) for path in model_paths]
    for model in loaded:
        report = validate_model(model)
        if report.errors:
            raise ValueError(f"{model.id}: release blocked by {report.errors} QC error(s)")

    output_root = Path(output_root)
    release_root = output_root / f"inkgrammar-data-{version}"
    archive_path = output_root / f"inkgrammar-data-{version}.zip"
    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True, exist_ok=True)

    catalog = []
    for model in loaded:
        payload = model.model_dump(mode="json", by_alias=True)
        _write_json(release_root / "models" / f"{model.id}.json", payload)
        catalog.append(
            {
                "id": model.id,
                "name": model.name,
                "script": model.script,
                "languages": model.languages,
                "regions": model.regions,
                "status": model.status,
                "license": model.license,
                "glyph_count": len(model.glyphs),
                "model": f"models/{model.id}.json",
            }
        )
        _write_json(
            release_root / "qc" / f"{model.id}.json",
            validate_model(model).to_dict(),
        )

    _write_json(
        release_root / "catalog.json",
        {
            "format": "inkgrammar-catalog",
            "format_version": 2,
            "release_version": version,
            "models": catalog,
        },
    )
    _write_json(
        release_root / "manifest.json",
        {
            "format": "inkgrammar-release",
            "format_version": 3,
            "release_version": version,
            "catalog": "catalog.json",
            "checksums": "checksums.sha256",
            "requires_inkgrammar": requires_inkgrammar,
            "created_from": {
                "git_commit": git_commit,
                "model_sources": [Path(path).as_posix() for path in model_paths],
            },
        },
    )
    _write_json(
        release_root / "schemas" / "writing-model.schema.json",
        WritingModel.model_json_schema(),
    )
    _write_json(
        release_root / "schemas" / "page-layout.schema.json",
        PageLayout.model_json_schema(),
    )

    _write_json(
        release_root / "schemas" / "observation.schema.json",
        ObservationManifest.model_json_schema(),
    )
    _write_json(
        release_root / "schemas" / "regression.schema.json", RegressionManifest.model_json_schema()
    )
    for model_path, model in zip(model_paths, loaded, strict=True):
        data_root = Path(model_path).parent.parent
        for directory, target, schema in (
            ("observations", "observations", ObservationManifest),
            ("regressions", "regression", RegressionManifest),
        ):
            path = data_root / directory / model.id / "manifest.json"
            if path.exists():
                evidence = schema.model_validate_json(path.read_text(encoding="utf-8"))
                if evidence.model_id != model.id:
                    raise ValueError("evidence manifest model ID mismatch")
                _write_json(
                    release_root / target / f"{model.id}.json", evidence.model_dump(mode="json")
                )
        license_path = release_root / "licenses" / f"{model.id}.txt"
        license_path.parent.mkdir(parents=True, exist_ok=True)
        license_path.write_text(
            f"Model {model.id}: {model.license}\n"
            + "\n".join(f"{source.id}: {source.rights}" for source in model.sources)
            + "\n",
            encoding="utf-8",
        )

    rows = []
    for path in sorted(item for item in release_root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(release_root).as_posix()}")
    (release_root / "checksums.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in release_root.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(output_root).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    archive_path.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return release_root, archive_path
