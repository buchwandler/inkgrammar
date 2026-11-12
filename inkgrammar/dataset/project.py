"""Dataset project recipe: a source-free description of dataset materialization.

The recipe contains no scan bytes. ``observation_manifest.path`` and
``pages[].layout`` resolve relative to the project file's directory;
``sources.private_root`` points at the private directory holding the authorized
source images (never committed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image
from pydantic import Field

from ..models import StrictModel
from .models import ObservationManifest, SourceImage

ExtractorName = Literal["labeled-rows-v1", "region-layout-v1", "annotated-region-v1"]
SourceStatus = Literal["verified", "missing", "hash-mismatch", "dimension-mismatch"]


class FileRef(StrictModel):
    path: str


class DatasetSources(StrictModel):
    private_root: str


class DatasetPage(StrictModel):
    source_id: str
    layout: str
    extractor: ExtractorName


class DatasetProject(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    observation_manifest: FileRef
    sources: DatasetSources
    pages: list[DatasetPage] = Field(min_length=1)


class SourceCheck(StrictModel):
    source_id: str
    filename: str
    path: str
    status: SourceStatus
    problems: list[str] = Field(default_factory=list)


class SourceVerification(StrictModel):
    checks: list[SourceCheck]

    @property
    def ok(self) -> bool:
        return all(check.status == "verified" for check in self.checks)


@dataclass(frozen=True)
class DatasetProjectBundle:
    path: Path
    project: DatasetProject

    @property
    def project_dir(self) -> Path:
        return self.path.parent

    @property
    def observation_manifest_path(self) -> Path:
        return self.project_dir / self.project.observation_manifest.path

    @property
    def private_root(self) -> Path:
        return self.project_dir / self.project.sources.private_root

    def layout_path(self, page: DatasetPage) -> Path:
        return self.project_dir / page.layout


def load_dataset_project(path: str | Path) -> DatasetProjectBundle:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetProjectBundle(path=path, project=DatasetProject.model_validate(data))


def load_observation_manifest(path: str | Path) -> ObservationManifest:

    return ObservationManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _check_source(source: SourceImage, private_root: Path) -> SourceCheck:
    source_path = private_root / source.filename
    if not source_path.is_file():
        return SourceCheck(
            source_id=source.source_id,
            filename=source.filename,
            path=str(source_path),
            status="missing",
            problems=["source file not found"],
        )
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != source.sha256:
        return SourceCheck(
            source_id=source.source_id,
            filename=source.filename,
            path=str(source_path),
            status="hash-mismatch",
            problems=[f"sha256 mismatch: expected {source.sha256}, got {digest}"],
        )
    problems = []
    with Image.open(source_path) as image:
        width, height = image.size
    if (width, height) != (source.width_px, source.height_px):
        problems.append(
            f"dimension mismatch: expected {source.width_px}x{source.height_px}, "
            f"got {width}x{height}"
        )
    status: SourceStatus = "verified" if not problems else "dimension-mismatch"
    return SourceCheck(
        source_id=source.source_id,
        filename=source.filename,
        path=str(source_path),
        status=status,
        problems=problems,
    )


def verify_sources(bundle: DatasetProjectBundle) -> SourceVerification:
    """Verify private source files against the manifest's source evidence."""
    manifest = load_observation_manifest(bundle.observation_manifest_path)
    known = {source.source_id for source in manifest.sources}
    unknown_pages = [page.source_id for page in bundle.project.pages if page.source_id not in known]
    if unknown_pages:
        raise ValueError(f"unknown source id in pages: {sorted(set(unknown_pages))}")
    checks = [_check_source(source, bundle.private_root) for source in manifest.sources]
    return SourceVerification(checks=checks)
