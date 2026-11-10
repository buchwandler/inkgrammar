from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import PageLayout
from .raster import threshold_image


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(float)
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values.astype(float), kernel, mode="same")


def _segment_boundaries(projection: np.ndarray, count: int) -> list[int]:
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


def _tight_bbox(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int, pad: int = 4):
    crop = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        return (x0, y0, x1, y1)
    left = max(x0, x0 + int(xs.min()) - pad)
    right = min(x1, x0 + int(xs.max()) + pad + 1)
    top = max(y0, y0 + int(ys.min()) - pad)
    bottom = min(y1, y0 + int(ys.max()) + pad + 1)
    return (left, top, right, bottom)


def extract_labeled_page(
    image_path: str | Path,
    layout: PageLayout,
    output_dir: str | Path,
) -> dict:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    mask = threshold_image(image)
    width, height = image.size

    detections = []
    review = image.copy()
    draw = ImageDraw.Draw(review)

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
                round((left + right) / 2)
                for left, right in zip(centers, centers[1:])
            )
            absolute_boundaries.append(row_x1)
            boundaries = [value - row_x0 for value in absolute_boundaries]
        else:
            boundaries = _segment_boundaries(projection, len(row.labels))

        for index, label in enumerate(row.labels):
            raw_x0 = row_x0 + boundaries[index]
            raw_x1 = row_x0 + boundaries[index + 1]
            bbox = _tight_bbox(
                mask,
                raw_x0,
                raw_x1,
                row_y0,
                row_y1,
            )
            x0, y0, x1, y1 = bbox
            crop = image.crop(bbox)

            codepoints = "-".join(f"U+{ord(ch):04X}" for ch in label)
            filename = f"{row.id}-{index:02d}-{codepoints}.png"
            crop.save(output_dir / filename)

            gap_left = projection[
                max(0, boundaries[index] - 1):
                min(len(projection), boundaries[index] + 2)
            ].mean() if index > 0 else 0.0
            gap_right = projection[
                max(0, boundaries[index + 1] - 1):
                min(len(projection), boundaries[index + 1] + 2)
            ].mean() if index < len(row.labels) - 1 else 0.0
            ink_mean = float(projection.mean()) + 1e-9
            confidence = max(
                0.0,
                min(1.0, 1.0 - float((gap_left + gap_right) / 2 / ink_mean)),
            )

            detections.append({
                "row": row.id,
                "index": index,
                "label": label,
                "unicode": [f"U+{ord(ch):04X}" for ch in label],
                "bbox": [x0, y0, x1, y1],
                "crop": filename,
                "confidence": round(confidence, 4),
                "label_source": "layout-sequence",
            })
            draw.rectangle(bbox, outline=(255, 0, 0), width=1)
            draw.text((x0, max(0, y0 - 12)), label, fill=(255, 0, 0))

    manifest = {
        "format": "scriptstrokes-page-detections",
        "format_version": 1,
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
    (output_dir / "detections.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    review.save(output_dir / "review.png")
    return manifest
