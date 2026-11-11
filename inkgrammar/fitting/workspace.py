from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, model_validator

from ..models import StrictModel
from ..observations import ObservationManifest
from ..raster import threshold_image
from .models import (
    FitAnchorsFile,
    FitCorpus,
    FitFamiliesFile,
    FitProject,
    FitTopologyFile,
    ScriptGrammar,
    SourceLockFile,
)
from .objective import ObjectiveConfig
from .optimizer import OptimizerConfig


class FitSplitPolicy(StrictModel):
    fit: str
    validation: str
    hold_out: str = Field(alias="hold-out")


class FitRenderConfig(StrictModel):
    stroke_width_px: float = Field(default=3.0, gt=0)
    supersampling: int = Field(default=4, ge=1)


class FitRunConfig(StrictModel):
    format: str
    format_version: int = Field(gt=0)
    objective: ObjectiveConfig
    optimizer: OptimizerConfig
    split_policy: FitSplitPolicy
    render: FitRenderConfig = Field(default_factory=FitRenderConfig)

    @model_validator(mode="after")
    def validate_format(self) -> FitRunConfig:
        if self.format != "inkgrammar-fit-config" or self.format_version != 1:
            raise ValueError("unsupported fitting configuration format")
        if self.objective.version == 3:
            if self.optimizer.method != "centerline-em-trf":
                raise ValueError("objective v3 requires optimizer method centerline-em-trf")
            unsupported = {
                "max_iterations",
                "de_max_iterations",
                "de_population_size",
                "de_dimension_limit",
                "joint_refine",
            } & self.optimizer.model_fields_set
            if unsupported:
                raise ValueError(
                    f"objective v3 does not support optimizer settings: {sorted(unsupported)}"
                )
        elif self.optimizer.method == "centerline-em-trf":
            raise ValueError("optimizer method centerline-em-trf requires objective v3")
        return self


@dataclass(frozen=True)
class FitProjectBundle:
    project_path: Path
    project_dir: Path
    repository_root: Path
    private_work_dir: Path
    project: FitProject
    source_lock: SourceLockFile
    observations: ObservationManifest
    corpus: FitCorpus
    families: FitFamiliesFile
    topology: FitTopologyFile
    anchors: FitAnchorsFile
    grammar: ScriptGrammar
    fit_config: FitRunConfig


@dataclass(frozen=True)
class MaterializationResult:
    output_dir: Path
    source_count: int
    sample_count: int
    artifact_count: int


def _read_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _repository_root(project_path: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repository root must be a directory")
        return root
    for candidate in (project_path.parent, *project_path.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    raise ValueError("fit project must be inside a Git checkout or pass repository_root")


def _require_within(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{description} must stay inside {root}") from exc
    return resolved


def _project_file(project_dir: Path, repo_root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be a relative project path")
    resolved = (project_dir / relative).resolve(strict=True)
    _require_within(resolved, repo_root, label)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {value}")
    return resolved


def _project_output(project_dir: Path, repo_root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be a relative project path")
    return _require_within(project_dir / relative, repo_root, label)


def _resolve_work_dir(
    project_dir: Path,
    repo_root: Path,
    configured_path: str,
    override: Path | None = None,
) -> Path:
    requested = override if override is not None else Path(configured_path)
    if not requested.is_absolute():
        base = repo_root if override is not None else project_dir
        requested = base / requested
    work_root = repo_root / "work"
    if work_root.is_symlink():
        raise ValueError("work directory must not be a symlink")
    root = work_root.resolve(strict=False)
    resolved = _require_within(requested, root, "private work directory")
    if resolved == root:
        raise ValueError("private work directory must be below repository work/")
    relative = resolved.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("private work directory must not traverse symlinks")
    return resolved


def _safe_artifact_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise ValueError(f"unsafe fitting artifact identifier: {value!r}")


def load_fit_project(
    project_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> FitProjectBundle:
    requested_path = Path(project_path).expanduser().resolve(strict=True)
    if not requested_path.is_file():
        raise ValueError(f"fit project is not a file: {requested_path}")
    repo_root = _repository_root(
        requested_path,
        Path(repository_root).expanduser() if repository_root is not None else None,
    )
    _require_within(requested_path, repo_root, "fit project")
    project_dir = requested_path.parent
    project = FitProject.model_validate(_read_yaml(requested_path))

    references = {
        "source lock": _project_file(project_dir, repo_root, project.source_lock, "source lock"),
        "observation manifest": _project_file(
            project_dir, repo_root, project.observation_manifest, "observation manifest"
        ),
        "corpus": _project_file(project_dir, repo_root, project.corpus, "corpus"),
        "families": _project_file(project_dir, repo_root, project.families, "families"),
        "topology": _project_file(project_dir, repo_root, project.topology, "topology"),
        "anchors": _project_file(project_dir, repo_root, project.anchors, "anchors"),
        "fit config": _project_file(project_dir, repo_root, project.fit_config, "fit config"),
        "seed grammar": _project_file(project_dir, repo_root, project.seed_grammar, "seed grammar"),
    }
    _project_output(project_dir, repo_root, project.compile.output, "compile output")
    private_work_dir = _resolve_work_dir(project_dir, repo_root, project.private_work_dir)

    source_lock = SourceLockFile.model_validate_json(
        references["source lock"].read_text(encoding="utf-8")
    )
    observations = ObservationManifest.model_validate_json(
        references["observation manifest"].read_text(encoding="utf-8")
    )
    corpus = FitCorpus.model_validate(_read_yaml(references["corpus"]))
    families = FitFamiliesFile.model_validate(_read_yaml(references["families"]))
    topology = FitTopologyFile.model_validate(_read_yaml(references["topology"]))
    anchors = FitAnchorsFile.model_validate_json(references["anchors"].read_text(encoding="utf-8"))
    grammar = ScriptGrammar.model_validate(_read_yaml(references["seed grammar"]))
    fit_config = FitRunConfig.model_validate(_read_yaml(references["fit config"]))

    sources_by_id = {source.source_id: source for source in observations.sources}
    locked_ids = {source.id for source in source_lock.sources}
    if locked_ids != set(sources_by_id):
        raise ValueError("source lock IDs do not match observation manifest sources")
    for locked in source_lock.sources:
        source = sources_by_id[locked.id]
        if (
            locked.expected_filename != source.filename
            or locked.sha256 != source.sha256
            or locked.width_px != source.width_px
            or locked.height_px != source.height_px
        ):
            raise ValueError(f"source lock metadata mismatch for {locked.id}")

    observations_by_id = {item.id: item for item in observations.items}
    samples_by_id = {sample.id: sample for sample in corpus.samples}
    for sample in corpus.samples:
        _safe_artifact_id(sample.id)
        _safe_artifact_id(sample.observation_id)
        observation = observations_by_id.get(sample.observation_id)
        if observation is None:
            raise ValueError(f"unknown observation: {sample.observation_id}")
        if sample.kind == "glyph" and observation.label != sample.text:
            raise ValueError(f"sample text does not match observation label: {sample.id}")

    for family in families.families:
        if family.primitive not in grammar.primitives and family.review_status != "uncertain":
            raise ValueError(f"family references unknown primitive: {family.primitive}")
        for split_name, sample_ids in (
            ("fit", family.fit),
            ("validation", family.validation),
            ("hold-out", family.hold_out),
        ):
            for sample_id in sample_ids:
                sample = samples_by_id.get(sample_id)
                if sample is None:
                    raise ValueError(f"family references unknown sample: {sample_id}")
                if sample.split != split_name:
                    raise ValueError(f"family split does not match corpus for {sample_id}")

    corpus_glyphs = {sample.text for sample in corpus.samples if sample.kind == "glyph"}
    for key, glyph in topology.glyphs.items():
        if key not in corpus_glyphs:
            raise ValueError(f"topology glyph is not present in the corpus: {key}")
        for stroke in glyph.strokes:
            for part in stroke.parts:
                if part.primitive not in grammar.primitives:
                    raise ValueError(f"topology references unknown primitive: {part.primitive}")

    for sample in corpus.samples:
        observation = observations_by_id[sample.observation_id]
        glyph = topology.glyphs.get(sample.text)
        instances = (
            {part.instance for stroke in glyph.strokes for part in stroke.parts} if glyph else set()
        )
        crop_width = observation.source_bbox[2] - observation.source_bbox[0]
        crop_height = observation.source_bbox[3] - observation.source_bbox[1]
        for region in sample.loss_regions:
            if not set(region.instances) <= instances:
                raise ValueError(f"loss region references unknown glyph instance: {sample.id}")
            x0, y0, x1, y1 = region.bbox_source
            if x1 > crop_width or y1 > crop_height:
                raise ValueError(f"loss region exceeds observation crop: {sample.id}")
    for sample_id, anchor in anchors.samples.items():
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"anchor references unknown sample: {sample_id}")
        observation = observations_by_id[sample.observation_id]
        if (
            anchor.source_id != observation.source_id
            or anchor.source_bbox != observation.source_bbox
        ):
            raise ValueError(f"anchor source metadata mismatch: {sample_id}")

    return FitProjectBundle(
        project_path=requested_path,
        project_dir=project_dir,
        repository_root=repo_root,
        private_work_dir=private_work_dir,
        project=project,
        source_lock=source_lock,
        observations=observations,
        corpus=corpus,
        families=families,
        topology=topology,
        anchors=anchors,
        grammar=grammar,
        fit_config=fit_config,
    )


def _verified_sources(bundle: FitProjectBundle, source_dir: Path) -> dict[str, Image.Image]:
    directory = source_dir.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"source directory is not a directory: {directory}")
    verified: dict[str, Image.Image] = {}
    for locked in bundle.source_lock.sources:
        source_path = directory / locked.expected_filename
        if source_path.is_symlink():
            raise ValueError(f"locked source must not be a symlink: {locked.expected_filename}")
        if not source_path.is_file():
            raise ValueError(f"missing locked source image: {locked.expected_filename}")
        data = source_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != locked.sha256:
            raise ValueError(f"source hash mismatch: {locked.id}")
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                if image.size != (locked.width_px, locked.height_px):
                    raise ValueError(f"source dimensions mismatch: {locked.id}")
                verified[locked.id] = image.convert("L")
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("source dimensions mismatch"):
                raise
            raise ValueError(f"invalid locked source image: {locked.id}") from exc
    return verified


def _save_png(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _build_review_sheet(samples, crops: dict[str, Image.Image], output: Path) -> None:
    columns = 4
    cell_width, cell_height = 160, 112
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("L", (columns * cell_width, rows * cell_height), 255)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, sample in enumerate(samples):
        column, row = index % columns, index // columns
        left, top = column * cell_width, row * cell_height
        draw.text((left + 5, top + 4), f"{sample.text} | {sample.split}", fill=0, font=font)
        crop = crops[sample.id].copy()
        crop.thumbnail((cell_width - 12, cell_height - 28), Image.Resampling.LANCZOS)
        sheet.paste(crop, (left + (cell_width - crop.width) // 2, top + 22))
    _save_png(sheet, output)


def materialize_fit_project(
    bundle: FitProjectBundle,
    source_dir: str | Path,
    *,
    work_dir: str | Path | None = None,
) -> MaterializationResult:
    verified_sources = _verified_sources(bundle, Path(source_dir))
    selected_work_dir = _resolve_work_dir(
        bundle.project_dir,
        bundle.repository_root,
        bundle.project.private_work_dir,
        Path(work_dir).expanduser() if work_dir is not None else None,
    )
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("materialization requires the optional fit extra (SciPy)") from exc

    work_root = bundle.repository_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    selected_work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = selected_work_dir / "materialized"
    if output_dir.is_symlink():
        raise ValueError("materialization output must not be a symlink")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("materialization output must be a directory")

    observations_by_id = {item.id: item for item in bundle.observations.items}
    staging = Path(tempfile.mkdtemp(prefix=".materialized-", dir=selected_work_dir))
    backup = selected_work_dir / ".materialized-previous"
    crops_dir = staging / "crops"
    masks_dir = staging / "masks"
    distances_dir = staging / "distance-fields"
    crops_dir.mkdir()
    masks_dir.mkdir()
    distances_dir.mkdir()
    crop_images: dict[str, Image.Image] = {}
    sample_rows = []
    output_records = []
    try:
        for sample in bundle.corpus.samples:
            observation = observations_by_id[sample.observation_id]
            source = verified_sources[observation.source_id]
            crop = source.crop(observation.source_bbox).convert("L")
            mask = threshold_image(crop)
            if not mask.any():
                raise ValueError(f"materialized crop contains no ink: {sample.id}")
            crop_name = f"{sample.id}.png"
            mask_name = f"{sample.id}.png"
            distance_name = f"{sample.id}.npy"
            crop_path = crops_dir / crop_name
            mask_path = masks_dir / mask_name
            distance_path = distances_dir / distance_name
            _save_png(crop, crop_path)
            _save_png(Image.fromarray(mask.astype(np.uint8) * 255), mask_path)
            distance = distance_transform_edt(~mask).astype(np.float32)
            with distance_path.open("wb") as stream:
                np.save(stream, distance, allow_pickle=False)
            crop_images[sample.id] = crop.copy()
            sample_rows.append(
                {
                    "id": sample.id,
                    "observation_id": sample.observation_id,
                    "source_id": observation.source_id,
                    "source_bbox": list(observation.source_bbox),
                    "split": sample.split,
                    "crop": f"crops/{crop_name}",
                    "mask": f"masks/{mask_name}",
                    "distance_field": f"distance-fields/{distance_name}",
                }
            )
            output_records.extend((crop_path, mask_path, distance_path))

        review_sheet = staging / "review-sheet.png"
        _build_review_sheet(bundle.corpus.samples, crop_images, review_sheet)
        output_records.append(review_sheet)
        outputs = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(
                output_records, key=lambda item: item.relative_to(staging).as_posix()
            )
        ]
        manifest = {
            "format": "inkgrammar-fit-materialization",
            "format_version": 1,
            "project_id": bundle.project.id,
            "sources": [
                {
                    "id": source.id,
                    "filename": source.expected_filename,
                    "sha256": source.sha256,
                    "width_px": source.width_px,
                    "height_px": source.height_px,
                }
                for source in bundle.source_lock.sources
            ],
            "samples": sample_rows,
            "outputs": outputs,
        }
        (staging / "materialization.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if backup.exists():
            raise ValueError("previous materialization backup exists; inspect it before retrying")
        moved_old = False
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_old = True
        try:
            os.replace(staging, output_dir)
        except OSError:
            if moved_old:
                os.replace(backup, output_dir)
            raise
        if moved_old:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        for image in verified_sources.values():
            image.close()

    return MaterializationResult(
        output_dir=output_dir,
        source_count=len(bundle.source_lock.sources),
        sample_count=len(bundle.corpus.samples),
        artifact_count=len(output_records) + 1,
    )
