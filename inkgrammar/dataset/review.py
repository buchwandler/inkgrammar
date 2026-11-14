"""Review patches: bounded human decisions applied onto automatic candidates.

Materialization combines ``automatic candidate + review patch = effective
reviewed dataset``. Every patch records the input hashes it was reviewed
against; when extraction changes, stale patches are reported and never
silently applied to the new crop.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image
from pydantic import Field

from ..models import StrictModel
from .models import (
    FamilyAssignment,
    FittingStatus,
    IsolationStatus,
    ObservationAnnotation,
    ReviewedInt,
    ReviewedRegion,
    StrokeAnnotation,
)

REVIEW_PATCH_VERSION = 1

#: Record keys excluded from the extraction hash: they hold review state or
#: patch-settable fields, so applying a patch must not invalidate it.
_EXTRACTION_HASH_EXCLUDE = frozenset(
    {"review_patch", "review_status", "fitting_status", "isolation_status", "annotation"}
)


class ReviewPatchInputHash(StrictModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReviewPatchSet(StrictModel):
    """The bounded set of fields a reviewer may decide (no geometry redrawing)."""

    isolation_status: IsolationStatus | None = None
    fitting_status: FittingStatus | None = None
    baseline_y: float | None = None
    x_height_y: float | None = None
    cap_height_y: float | None = None
    ascender_y: float | None = None
    descender_y: float | None = None
    stroke_count: ReviewedInt | None = None
    strokes: list[StrokeAnnotation] = Field(default_factory=list)
    family_assignments: list[FamilyAssignment] = Field(default_factory=list)
    reviewer_notes: str | None = None


class ReviewPatch(StrictModel):
    observation_id: str
    input_hash: ReviewPatchInputHash
    set: ReviewPatchSet = Field(default_factory=ReviewPatchSet)


def extraction_sha256(automatic_record: dict) -> str:
    """Hash of the automatic extraction record a patch was reviewed against."""
    payload = {
        key: value for key, value in automatic_record.items() if key not in _EXTRACTION_HASH_EXCLUDE
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_review_patch(path: str | Path) -> ReviewPatch:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ReviewPatch.model_validate(data)


def load_review_patches(directory: str | Path) -> dict[str, ReviewPatch]:
    """Load ``<observation_id>.yaml`` patches keyed by observation id."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    patches = {}
    for path in sorted(directory.glob("*.yaml")):
        patch = load_review_patch(path)
        if patch.observation_id in patches:
            raise ValueError(f"duplicate review patch for {patch.observation_id}")
        patches[patch.observation_id] = patch
    return patches


def apply_review_patch(
    automatic_record: dict,
    patch: ReviewPatch,
    *,
    source_sha256: str,
) -> tuple[dict, str]:
    """Compose candidate + patch into the effective record.

    Returns ``(effective_record, status)`` with status ``applied`` or
    ``stale``. Stale patches are reported on the record but never applied.
    """
    effective = copy.deepcopy(automatic_record)
    if (
        patch.input_hash.source_sha256 != source_sha256
        or patch.input_hash.extraction_sha256 != extraction_sha256(automatic_record)
    ):
        effective["review_patch"] = {"status": "stale", "input_hash": patch.input_hash.model_dump()}
        return effective, "stale"

    patch_set = patch.set
    if patch_set.isolation_status is not None:
        effective["isolation_status"] = patch_set.isolation_status
    if patch_set.fitting_status is not None:
        effective["fitting_status"] = patch_set.fitting_status
    metric_fields = ("baseline_y", "x_height_y", "cap_height_y", "ascender_y", "descender_y")
    metrics = {
        key: value for key in metric_fields if (value := getattr(patch_set, key)) is not None
    }
    if metrics:
        effective.setdefault("metrics", {})
        effective["metrics"].update(metrics)
        effective["metrics"]["metric_status"] = "reviewed"
    annotation = {}
    if patch_set.stroke_count is not None:
        annotation["stroke_count"] = patch_set.stroke_count.model_dump()
    if patch_set.strokes:
        annotation["strokes"] = [stroke.model_dump() for stroke in patch_set.strokes]
    if patch_set.family_assignments:
        annotation["family_assignments"] = [
            assignment.model_dump() for assignment in patch_set.family_assignments
        ]
    if patch_set.reviewer_notes is not None:
        annotation["reviewer_notes"] = patch_set.reviewer_notes
    if annotation:
        effective.setdefault("annotation", {})
        effective["annotation"].update(annotation)
    effective["review_patch"] = {"status": "applied", "input_hash": patch.input_hash.model_dump()}
    return effective, "applied"


# --- annotation candidates (layer C, section 11) ---


def centerline_end_regions(
    record: dict, dataset_root: str | Path, *, crop_size: tuple[int, int]
) -> tuple[ReviewedRegion, ReviewedRegion] | None:
    """Start/end region candidates from the first derived centerline candidate."""

    centerline_path = (record.get("derived") or {}).get("centerline", {}).get("path")
    if centerline_path is None:
        return None
    candidates = json.loads((Path(dataset_root) / centerline_path).read_text(encoding="utf-8"))[
        "candidates"
    ]
    if not candidates:
        return None
    width, height = crop_size

    def region(point: list[int]) -> ReviewedRegion:
        x, y = point
        return ReviewedRegion(
            bbox=(
                max(0.0, (x - 2) / width),
                max(0.0, (y - 2) / height),
                min(1.0, (x + 3) / width),
                min(1.0, (y + 3) / height),
            ),
            evidence="automatic-candidate",
        )

    return region(candidates[0]["start"]), region(candidates[0]["end"])


def write_annotation_candidate(annotation: ObservationAnnotation, directory: str | Path) -> Path:
    """Write a layer-C annotation candidate as human-editable YAML."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{annotation.observation_id}.yaml"
    payload = yaml.safe_dump(
        annotation.model_dump(mode="json"), allow_unicode=True, sort_keys=False
    )
    path.write_text(payload, encoding="utf-8")
    return path


# --- review sheets (section 30: bounded decisions, no GUI) ---

PANEL_SIZE = 128
TEXT_HEIGHT = 56
SHEET_BACKGROUND = (255, 255, 255)


def _fit_panel(image: Image.Image) -> tuple[Image.Image, float, int, int]:
    scale = min(PANEL_SIZE / image.width, PANEL_SIZE / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resample=Image.NEAREST,
    )
    offset_x = (PANEL_SIZE - resized.width) // 2
    offset_y = (PANEL_SIZE - resized.height) // 2
    return resized.convert("RGB"), scale, offset_x, offset_y


def _render_observation_sheet(dataset_root: Path, record: dict, output_dir: Path) -> Path:
    from PIL import ImageDraw

    from .materialize import PREVIEW_BASELINE, PREVIEW_UNIT

    artifacts = record["artifacts"]
    crop = Image.open(dataset_root / artifacts["crop"])
    clean = (
        Image.open(dataset_root / artifacts["clean_mask"])
        .convert("L")
        .point(lambda value: 255 - value)
    )
    normalized = (
        Image.open(dataset_root / artifacts["normalized"])
        .convert("L")
        .point(lambda value: 255 - value)
    )

    sheet = Image.new("RGB", (PANEL_SIZE * 3, PANEL_SIZE + TEXT_HEIGHT), SHEET_BACKGROUND)
    panels = []
    for offset_index, panel_image in enumerate((crop, clean, normalized)):
        panel, scale, offset_x, offset_y = _fit_panel(panel_image)
        sheet.paste(panel, (offset_index * PANEL_SIZE + offset_x, offset_y))
        panels.append((scale, offset_x, offset_y))

    draw = ImageDraw.Draw(sheet)
    centerline_path = (record.get("derived") or {}).get("centerline", {}).get("path")
    if centerline_path is not None:
        candidates = json.loads((dataset_root / centerline_path).read_text(encoding="utf-8"))[
            "candidates"
        ]
        if candidates:
            scale, offset_x, offset_y = panels[0]
            points = candidates[0]["points"]
            for x, y in points:
                draw.ellipse(
                    (
                        x * scale + offset_x - 1,
                        y * scale + offset_y - 1,
                        x * scale + offset_x + 1,
                        y * scale + offset_y + 1,
                    ),
                    fill=(220, 0, 0),
                )
            start_x, start_y = candidates[0]["start"]
            end_x, end_y = candidates[0]["end"]
            draw.rectangle(
                (
                    start_x * scale + offset_x - 3,
                    start_y * scale + offset_y - 3,
                    start_x * scale + offset_x + 3,
                    start_y * scale + offset_y + 3,
                ),
                outline=(0, 160, 0),
            )
            draw.rectangle(
                (
                    end_x * scale + offset_x - 3,
                    end_y * scale + offset_y - 3,
                    end_x * scale + offset_x + 3,
                    end_y * scale + offset_y + 3,
                ),
                outline=(0, 0, 220),
            )

    scale, offset_x, offset_y = panels[2]
    baseline_y = PREVIEW_BASELINE * scale + offset_y
    x_height_y = (PREVIEW_BASELINE - PREVIEW_UNIT) * scale + offset_y
    draw.line((2 * PANEL_SIZE, baseline_y, 3 * PANEL_SIZE, baseline_y), fill=(0, 160, 0), width=1)
    draw.line((2 * PANEL_SIZE, x_height_y, 3 * PANEL_SIZE, x_height_y), fill=(0, 0, 220), width=1)

    metrics = record.get("metrics") or {}
    text_lines = [
        f"id: {record['id']}",
        f"kind: {record['kind']}   review: {record['review_status']}   "
        f"isolation: {record.get('isolation_status')}",
        f"baseline_y: {metrics.get('baseline_y')}   x_height_y: {metrics.get('x_height_y')}   "
        f"metric_status: {metrics.get('metric_status')}",
        "panels: source crop | clean mask | normalized (green baseline, blue x-height; "
        "red centerline candidate, green start, blue end)",
    ]
    for line_index, text in enumerate(text_lines):
        draw.text((4, PANEL_SIZE + 4 + line_index * 12), text, fill=(0, 0, 0))

    page_dir = artifacts["crop"].split("/", 1)[0]
    sheet_path = output_dir / page_dir / f"{record['id']}.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    return sheet_path


def render_review_sheets(dataset_root: str | Path, output_dir: str | Path) -> list[Path]:
    """Render one deterministic review sheet per observation plus an index."""

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    index = json.loads((dataset_root / "index.json").read_text(encoding="utf-8"))
    written = []
    lines = [
        "# Review sheets",
        "",
        "Automatic candidates for bounded human review; context sheets are the",
        "per-page review.png files in the materialized dataset root.",
        "",
        "| observation | kind | review | isolation | sheet |",
        "| --- | --- | --- | --- | --- |",
    ]
    for obs_id, record in sorted(index["observations"].items()):
        sheet = _render_observation_sheet(dataset_root, record, output_dir)
        written.append(sheet)
        lines.append(
            f"| {obs_id} | {record['kind']} | {record['review_status']} | "
            f"{record.get('isolation_status')} | {sheet.relative_to(output_dir).as_posix()} |"
        )
    (output_dir / "review-index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written
