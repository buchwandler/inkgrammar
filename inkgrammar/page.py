"""Scan-page extraction strategies over shared low-level pieces.

Strategies:

- ``labeled-rows-v1``: segments declared label rows (:class:`PageLayout`);
  identity comes from the declared sequence, never from the image.
- ``region-layout-v1``: extracts declared region records
  (:class:`~inkgrammar.dataset.layouts.RegionLayout`) without cutting them
  into glyphs.
- ``annotated-region-v1``: like ``region-layout-v1`` but keeps declared ink
  subregions separate from printed annotation marks (numbers, arrows) so those
  marks can never become part of the target ink.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset.layouts import (
    ANNOTATION_SUBREGION_PREFIX,
    INK_SUBREGION,
    LayoutRegion,
    RegionLayout,
)
from .models import PageLayout
from .raster import threshold_image

DETECTIONS_FORMAT = "inkgrammar-page-detections"


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(float)
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values.astype(float), kernel, mode="same")


def threshold_source(image: Image.Image) -> np.ndarray:
    """Binarize a source image into a boolean ink mask."""
    return threshold_image(image)


def crop_region(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop a pixel-space bounding box from a source image."""
    return image.crop(bbox)


def tighten_to_ink(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int = 4,
) -> tuple[int, int, int, int]:
    """Shrink a pixel-space bounding box to the ink inside it."""
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return bbox
    left = max(x0, x0 + int(xs.min()) - pad)
    right = min(x1, x0 + int(xs.max()) + pad + 1)
    top = max(y0, y0 + int(ys.min()) - pad)
    bottom = min(y1, y0 + int(ys.max()) + pad + 1)
    return (left, top, right, bottom)


def segment_row(projection: np.ndarray, count: int) -> list[int]:
    """Split a row ink projection into ``count`` cells by low-ink boundaries."""
    if count <= 1:
        return [0, len(projection)]
    width = len(projection)
    smoothed = _smooth(projection, max(1, width // 600))
    boundaries = [0]

    for index in range(1, count):
        expected = width * index / count
        half_window = max(4, int(width / count * 0.38))
        lo = max(boundaries[-1] + 2, int(expected - half_window))
        hi = min(width - (count - index) * 2, int(expected + half_window))
        if hi <= lo:
            cut = int(expected)
        else:
            positions = np.arange(lo, hi)
            local = smoothed[lo:hi]
            local_scale = float(np.percentile(local, 85)) + 1e-9
            ink_cost = local / local_scale
            distance_cost = np.abs(positions - expected) / max(1.0, half_window)
            score = ink_cost + 0.45 * distance_cost
            cut = lo + int(np.argmin(score))
        boundaries.append(cut)
    boundaries.append(width)
    return boundaries


def render_review(
    image: Image.Image,
    boxes: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]],
) -> Image.Image:
    """Draw labeled outlines (bbox, text, color) over a copy of the source."""
    review = image.copy()
    draw = ImageDraw.Draw(review)
    for bbox, text, color in boxes:
        draw.rectangle(bbox, outline=color, width=1)
        draw.text((bbox[0], max(0, bbox[1] - 12)), text, fill=color)
    return review


def write_detection_manifest(output_dir: Path, manifest: dict) -> Path:
    """Write ``detections.json`` deterministically and return its path."""
    path = output_dir / "detections.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _absolute_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0 = round(bbox[0] * width)
    y0 = round(bbox[1] * height)
    x1 = round(bbox[2] * width)
    y1 = round(bbox[3] * height)
    return (x0, y0, x1, y1)


def _load_source(image_path: Path) -> tuple[Image.Image, np.ndarray, int, int]:
    image = Image.open(image_path).convert("RGB")
    mask = threshold_source(image)
    width, height = image.size
    return image, mask, width, height


def extract_labeled_page(
    image_path: str | Path,
    layout: PageLayout,
    output_dir: str | Path,
) -> dict:
    """Strategy ``labeled-rows-v1``: segment declared label rows."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crops").mkdir(exist_ok=True)

    image, mask, width, height = _load_source(image_path)

    detections = []
    review_boxes = []

    for row in layout.rows:
        row_y0 = round(row.y[0] * height)
        row_y1 = round(row.y[1] * height)
        row_x0 = round(row.x[0] * width)
        row_x1 = round(row.x[1] * width)
        row_mask = mask[row_y0:row_y1, row_x0:row_x1]
        projection = row_mask.sum(axis=0)
        if row.centers is not None:
            centers = [round(value * width) for value in row.centers]
            absolute_boundaries = [row_x0]
            absolute_boundaries.extend(
                round((left + right) / 2) for left, right in zip(centers, centers[1:], strict=False)
            )
            absolute_boundaries.append(row_x1)
            boundaries = [value - row_x0 for value in absolute_boundaries]
        else:
            boundaries = segment_row(projection, len(row.labels))

        for index, label in enumerate(row.labels):
            raw_x0 = row_x0 + boundaries[index]
            raw_x1 = row_x0 + boundaries[index + 1]
            bbox = tighten_to_ink(
                mask,
                (raw_x0, row_y0, raw_x1, row_y1),
            )
            x0, y0, x1, y1 = bbox
            crop = crop_region(image, bbox)

            codepoints = "-".join(f"U+{ord(ch):04X}" for ch in label)
            filename = f"{row.id}-{index:02d}-{codepoints}.png"
            crop.save(output_dir / "crops" / filename)

            gap_left = (
                projection[
                    max(0, boundaries[index] - 1) : min(len(projection), boundaries[index] + 2)
                ].mean()
                if index > 0
                else 0.0
            )
            gap_right = (
                projection[
                    max(0, boundaries[index + 1] - 1) : min(
                        len(projection), boundaries[index + 1] + 2
                    )
                ].mean()
                if index < len(row.labels) - 1
                else 0.0
            )
            ink_mean = float(projection.mean()) + 1e-9
            confidence = max(
                0.0,
                min(1.0, 1.0 - float((gap_left + gap_right) / 2 / ink_mean)),
            )

            detections.append(
                {
                    "row": row.id,
                    "index": index,
                    "label": label,
                    "unicode": [f"U+{ord(ch):04X}" for ch in label],
                    "bbox": [x0, y0, x1, y1],
                    "crop": f"crops/{filename}",
                    "confidence": round(confidence, 4),
                    "label_source": "layout-sequence",
                }
            )
            review_boxes.append((bbox, label, (255, 0, 0)))

    manifest = {
        "format": DETECTIONS_FORMAT,
        "format_version": 1,
        "extractor": "labeled-rows-v1",
        "source_image": image_path.name,
        "layout": layout.id,
        "width": width,
        "height": height,
        "detections": detections,
        "label_policy": (
            "Shapes are segmented from the image; character identity comes from the "
            "declared row sequence and therefore requires human review."
        ),
    }
    write_detection_manifest(output_dir, manifest)
    render_review(image, review_boxes).save(output_dir / "review.png")
    return manifest


def _region_record(
    region: LayoutRegion,
    image: Image.Image,
    output_dir: Path,
    width: int,
    height: int,
    review_boxes: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]],
) -> dict:
    record = {
        "region": region.id,
        "kind": region.kind,
        "label": region.label,
        "sequence": region.sequence,
        "members": region.members,
        "category_hint": region.category_hint,
        "notes": region.notes,
    }
    if region.bbox is not None:
        bbox = _absolute_bbox(region.bbox, width, height)
        filename = f"{region.id}.png"
        crop_region(image, bbox).save(output_dir / "crops" / filename)
        record["bbox"] = list(bbox)
        record["crop"] = f"crops/{filename}"
        review_boxes.append((bbox, region.id, (255, 0, 0)))
    subregions = {}
    for name, sub_bbox in region.subregions.items():
        bbox = _absolute_bbox(sub_bbox, width, height)
        filename = f"{region.id}__{name}.png"
        crop_region(image, bbox).save(output_dir / "crops" / filename)
        subregions[name] = {"bbox": list(bbox), "crop": f"crops/{filename}"}
        review_boxes.append((bbox, f"{region.id}:{name}", (0, 120, 255)))
    record["subregions"] = subregions
    return record


def extract_region_page(
    image_path: str | Path,
    layout: RegionLayout,
    output_dir: str | Path,
) -> dict:
    """Strategy ``region-layout-v1``: extract declared regions intact."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crops").mkdir(exist_ok=True)

    image, _, width, height = _load_source(image_path)

    review_boxes: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]] = []
    detections = [
        _region_record(region, image, output_dir, width, height, review_boxes)
        for region in layout.regions
    ]

    manifest = {
        "format": DETECTIONS_FORMAT,
        "format_version": 1,
        "extractor": "region-layout-v1",
        "source_image": image_path.name,
        "layout": layout.id,
        "width": width,
        "height": height,
        "detections": detections,
        "label_policy": (
            "Region identity comes from the declared layout; regions are extracted "
            "intact and are not cut into glyphs."
        ),
    }
    write_detection_manifest(output_dir, manifest)
    render_review(image, review_boxes).save(output_dir / "review.png")
    return manifest


def extract_annotated_region_page(
    image_path: str | Path,
    layout: RegionLayout,
    output_dir: str | Path,
) -> dict:
    """Strategy ``annotated-region-v1``: separate ink from annotation marks.

    Every region with annotation subregions must declare an ``ink`` subregion;
    annotation marks (numbers, arrows) are recorded separately and never enter
    the target ink region.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crops").mkdir(exist_ok=True)

    image, _, width, height = _load_source(image_path)

    review_boxes: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]] = []
    detections = []
    for region in layout.regions:
        annotation_names = sorted(
            name for name in region.subregions if name.startswith(ANNOTATION_SUBREGION_PREFIX)
        )
        if annotation_names and INK_SUBREGION not in region.subregions:
            raise ValueError(
                f"annotated region {region.id} declares annotation subregions "
                f"without an '{INK_SUBREGION}' subregion"
            )
        record = _region_record(region, image, output_dir, width, height, review_boxes)
        ink = record["subregions"].pop(INK_SUBREGION, None)
        annotations = [
            {"name": name, **record["subregions"].pop(name)} for name in annotation_names
        ]
        record["ink_region"] = ink
        record["annotation_regions"] = annotations
        detections.append(record)

    manifest = {
        "format": DETECTIONS_FORMAT,
        "format_version": 1,
        "extractor": "annotated-region-v1",
        "source_image": image_path.name,
        "layout": layout.id,
        "width": width,
        "height": height,
        "detections": detections,
        "label_policy": (
            "Ink regions and printed annotation marks are recorded separately; "
            "annotation marks must never become part of the target glyph mask."
        ),
    }
    write_detection_manifest(output_dir, manifest)
    render_review(image, review_boxes).save(output_dir / "review.png")
    return manifest
