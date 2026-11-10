from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .models import PageLayout, WritingModel
from .qc import validate_model


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
) -> tuple[Path, Path]:
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
    release_root = output_root / f"scriptstrokes-data-{version}"
    archive_path = output_root / f"scriptstrokes-data-{version}.zip"
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
            "format": "scriptstrokes-catalog",
            "format_version": 2,
            "release_version": version,
            "models": catalog,
        },
    )
    _write_json(
        release_root / "manifest.json",
        {
            "format": "scriptstrokes-release",
            "format_version": 2,
            "release_version": version,
            "catalog": "catalog.json",
            "checksums": "checksums.sha256",
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

    return release_root, archive_path
