from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from ..dataset.preprocess import label_components
from ..page import tighten_to_ink
from .mask import extract_ink_mask
from .models import CropRect, InkMaskResult, InkMode


@dataclass(frozen=True)
class Component:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    ys: np.ndarray
    xs: np.ndarray

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class RowBand:
    x0: int
    y0: int
    x1: int
    y1: int
    component_indices: tuple[int, ...]


@dataclass(frozen=True)
class SegmentedGlyph:
    glyph_id: str
    bbox: CropRect
    row_index: int
    column_index: int
    reading_order: int
    segmentation_confidence: float
    image: Image.Image
    mask: np.ndarray
    label: str | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != (self.bbox.height, self.bbox.width):
            raise ValueError("glyph mask dimensions must match its bounding box")
        if self.image.size != (self.bbox.width, self.bbox.height):
            raise ValueError("glyph image dimensions must match its bounding box")
        if not 0.0 <= self.segmentation_confidence <= 1.0:
            raise ValueError("segmentation confidence must be between zero and one")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class PageSegmentation:
    source_path: Path
    candidates: tuple[SegmentedGlyph, ...]
    expected_count: int | None = None
    status: Literal["ok", "ambiguous", "failed"] = "ok"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceGallery:
    source_path: Path
    source_sha256: str
    candidates: tuple[SegmentedGlyph, ...]
    mask_result: InkMaskResult
    segmentation: PageSegmentation


class SegmentationError(ValueError):
    def __init__(self, message: str, diagnostic_path: Path | None = None):
        super().__init__(message)
        self.diagnostic_path = diagnostic_path


def _load_page(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def extract_page_ink_mask(image: Image.Image, *, mode: InkMode = "auto") -> InkMaskResult:
    """Apply the pairwise comparison ink policy to a full page."""
    return extract_ink_mask(image, mode=mode)


def _components(mask: np.ndarray) -> list[Component]:
    result = []
    for index, (ys, xs) in enumerate(label_components(mask)):
        result.append(
            Component(
                index=index,
                x0=int(xs.min()),
                y0=int(ys.min()),
                x1=int(xs.max()) + 1,
                y1=int(ys.max()) + 1,
                area=int(ys.size),
                ys=ys,
                xs=xs,
            )
        )
    return result


def _size_scale(components: list[Component]) -> tuple[float, float]:
    if not components:
        return 0.0, 0.0
    areas = np.asarray([component.area for component in components], dtype=float)
    upper_half = np.sort(areas)[len(areas) // 2 :]
    primary_area = float(np.median(upper_half))
    body = [component for component in components if component.area >= primary_area * 0.2]
    median_height = float(np.median([component.height for component in body])) if body else 0.0
    return primary_area, median_height


def _is_page_artifact(component: Component, width: int, height: int) -> bool:
    box_area = component.width * component.height
    touches_border = (
        component.x0 == 0
        or component.y0 == 0
        or component.x1 == width
        or component.y1 == height
    )
    if touches_border and (box_area / (width * height) > 0.05 or component.width > width * 0.8):
        return True
    if component.width >= width * 0.2 and component.width / max(1, component.height) >= 12:
        return True
    if component.height >= height * 0.2 and component.height / max(1, component.width) >= 12:
        return True
    return box_area / (width * height) > 0.2


def _clean_page_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, list[Component], float, float, list[str]]:
    height, width = mask.shape
    components = _components(mask)
    if not components:
        return mask.copy(), [], 0.0, 0.0, ["No ink components were detected."]
    non_artifacts = [
        component
        for component in components
        if not _is_page_artifact(component, width, height)
    ]
    primary_area, median_height = _size_scale(non_artifacts)
    cleaned = np.array(mask, dtype=bool, copy=True)
    warnings = []
    for component in components:
        reason = None
        if _is_page_artifact(component, width, height):
            reason = "page border, rule, or oversized artifact"
        elif primary_area and component.area < primary_area * 0.008:
            reason = "tiny isolated noise"
        if reason:
            cleaned[component.ys, component.xs] = False
            warnings.append(f"Removed component {component.index}: {reason}.")
    kept = [
        component
        for component in components
        if np.any(cleaned[component.ys, component.xs])
    ]
    primary_area, median_height = _size_scale(kept)
    return cleaned, kept, primary_area, median_height, warnings


def _body_components(
    components: list[Component], primary_area: float, median_height: float
) -> list[Component]:
    if not components:
        return []
    minimum_area = primary_area * 0.18
    minimum_height = median_height * 0.25
    bodies = [
        component
        for component in components
        if component.area >= minimum_area
        and (component.height >= minimum_height or component.width >= minimum_height)
    ]
    return bodies or [max(components, key=lambda component: component.area)]


def detect_ink_rows(mask: np.ndarray) -> list[RowBand]:
    """Detect coherent glyph rows from page projection and component centers."""
    ink = np.asarray(mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("page ink mask must be two-dimensional")
    _, components, primary_area, median_height, _ = _clean_page_mask(ink)
    bodies = _body_components(components, primary_area, median_height)
    if not bodies:
        return []
    ordered = sorted(bodies, key=lambda component: component.center_y)
    rows: list[list[Component]] = []
    for component in ordered:
        if not rows or component.center_y - np.median(
            [member.center_y for member in rows[-1]]
        ) > max(1.0, median_height * 0.72):
            rows.append([component])
        else:
            rows[-1].append(component)
    return [
        RowBand(
            x0=min(component.x0 for component in row),
            y0=min(component.y0 for component in row),
            x1=max(component.x1 for component in row),
            y1=max(component.y1 for component in row),
            component_indices=tuple(component.index for component in row),
        )
        for row in rows
    ]


def _row_y_bounds(
    row: RowBand,
    components: list[Component],
    median_height: float,
    page_height: int,
) -> tuple[int, int]:
    body_indices = set(row.component_indices)
    bodies = [component for component in components if component.index in body_indices]
    satellites = [
        component
        for component in components
        if component.index not in body_indices
        and any(_satellite_score(component, body, median_height) is not None for body in bodies)
    ]
    top = min([row.y0, *(component.y0 for component in satellites)])
    bottom = max([row.y1, *(component.y1 for component in satellites)])
    return max(0, top), min(page_height, bottom)


def _group_nearby_bodies(
    bodies: list[Component], median_height: float
) -> list[list[Component]]:
    groups: list[list[Component]] = []
    for component in sorted(bodies, key=lambda item: item.x0):
        if not groups:
            groups.append([component])
            continue
        previous = groups[-1]
        previous_right = max(member.x1 for member in previous)
        previous_top = min(member.y0 for member in previous)
        previous_bottom = max(member.y1 for member in previous)
        gap = component.x0 - previous_right
        overlap = min(previous_bottom, component.y1) - max(previous_top, component.y0)
        if gap <= max(1.0, median_height * 0.12) and overlap > 0:
            previous.append(component)
        else:
            groups.append([component])
    return groups


def _satellite_score(
    satellite: Component, body: Component, median_height: float
) -> float | None:
    if satellite.area >= body.area * 0.6:
        return None
    horizontal_gap = max(body.x0 - satellite.x1, satellite.x0 - body.x1, 0)
    vertical_gap = max(body.y0 - satellite.y1, satellite.y0 - body.y1, 0)
    if horizontal_gap > median_height * 0.4 or vertical_gap > median_height * 1.25:
        return None
    overlaps_x = max(0, min(body.x1, satellite.x1) - max(body.x0, satellite.x0))
    return (
        horizontal_gap / max(1.0, median_height)
        + vertical_gap / max(1.0, median_height)
        - 0.35 * overlaps_x / max(1.0, median_height)
    )


def _crop_candidate(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    glyph_id: str,
    row_index: int,
    column_index: int,
    reading_order: int,
    confidence: float,
    padding: int,
    clip_bbox: tuple[int, int, int, int] | None = None,
) -> SegmentedGlyph:
    x0, y0, x1, y1 = bbox
    height, width = mask.shape
    clip_x0, clip_y0, clip_x1, clip_y1 = clip_bbox or (0, 0, width, height)
    expanded = (
        max(clip_x0, x0 - padding),
        max(clip_y0, y0 - padding),
        min(clip_x1, x1 + padding),
        min(clip_y1, y1 + padding),
    )
    x0, y0, x1, y1 = tighten_to_ink(mask, expanded, pad=padding)
    rect = CropRect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
    return SegmentedGlyph(
        glyph_id=glyph_id,
        bbox=rect,
        row_index=row_index,
        column_index=column_index,
        reading_order=reading_order,
        segmentation_confidence=float(np.clip(confidence, 0.0, 1.0)),
        image=image.crop((x0, y0, x1, y1)).copy(),
        mask=mask[y0:y1, x0:x1],
    )


def _candidate_record(candidate: SegmentedGlyph) -> dict:
    return {
        "id": candidate.glyph_id,
        "label": candidate.label,
        "row": candidate.row_index,
        "column": candidate.column_index,
        "reading_order": candidate.reading_order,
        "bbox": [
            candidate.bbox.x,
            candidate.bbox.y,
            candidate.bbox.x + candidate.bbox.width,
            candidate.bbox.y + candidate.bbox.height,
        ],
        "confidence": round(candidate.segmentation_confidence, 4),
    }


def _write_overlay(
    image: Image.Image,
    candidates: tuple[SegmentedGlyph, ...] | list[SegmentedGlyph],
    output: Path,
) -> None:
    preview = image.convert("RGB")
    draw = ImageDraw.Draw(preview)
    for candidate in candidates:
        x0, y0 = candidate.bbox.x, candidate.bbox.y
        x1, y1 = x0 + candidate.bbox.width - 1, y0 + candidate.bbox.height - 1
        draw.rectangle((x0, y0, x1, y1), outline=(230, 45, 35), width=2)
        draw.text((x0, max(0, y0 - 12)), candidate.glyph_id, fill=(25, 75, 220))
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, format="PNG")


def _write_page_mask(mask: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output, format="PNG")


def _write_reference_diagnostics(
    image: Image.Image,
    mask: np.ndarray,
    segmentation: PageSegmentation,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(mask, output_dir / "reference-mask.png")
    _write_overlay(image, segmentation.candidates, output_dir / "reference-candidates.png")
    payload = {
        "source_image": segmentation.source_path.name,
        "candidate_count": len(segmentation.candidates),
        "status": segmentation.status,
        "warnings": list(segmentation.warnings),
        "candidates": [_candidate_record(candidate) for candidate in segmentation.candidates],
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def segment_reference_gallery(
    image_path: str | Path,
    *,
    ink_mode: InkMode = "auto",
    output_dir: str | Path | None = None,
) -> ReferenceGallery:
    """Segment an unlabeled reference page into source-ordered glyph candidates."""
    source_path = Path(image_path)
    image = _load_page(source_path)
    mask_result = extract_page_ink_mask(image, mode=ink_mode)
    mask, components, primary_area, median_height, warnings = _clean_page_mask(
        mask_result.mask
    )
    components = _components(mask)
    primary_area, median_height = _size_scale(components)
    rows = detect_ink_rows(mask)
    bodies = _body_components(components, primary_area, median_height)
    body_indices = {component.index for component in bodies}
    candidates: list[SegmentedGlyph] = []
    reading_order = 0
    padding = max(1, round(median_height * 0.08))

    for row_index, row in enumerate(rows):
        row_bodies = [
            component
            for component in bodies
            if component.index in row.component_indices
        ]
        if not row_bodies:
            continue
        groups = _group_nearby_bodies(row_bodies, median_height)
        satellites = [component for component in components if component.index not in body_indices]
        attached: list[list[Component]] = [[] for _ in groups]
        for satellite in satellites:
            options = []
            for group_index, group in enumerate(groups):
                scores = [
                    score
                    for body in group
                    if (score := _satellite_score(satellite, body, median_height)) is not None
                ]
                score = min(scores, default=None)
                if score is not None:
                    options.append((score, group_index))
            if options:
                _, group_index = min(options)
                attached[group_index].append(satellite)

        for column_index, (group, marks) in enumerate(zip(groups, attached, strict=True)):
            members = [*group, *marks]
            x0 = min(component.x0 for component in members)
            y0 = min(component.y0 for component in members)
            x1 = max(component.x1 for component in members)
            y1 = max(component.y1 for component in members)
            next_group = groups[column_index + 1] if column_index + 1 < len(groups) else None
            previous_group = groups[column_index - 1] if column_index else None
            left_gap = (
                group[0].x0 - max(component.x1 for component in previous_group)
                if previous_group
                else median_height
            )
            right_gap = (
                min(component.x0 for component in next_group)
                - max(component.x1 for component in group)
                if next_group
                else median_height
            )
            nearest_gap = min(left_gap, right_gap)
            confidence = 0.92 if nearest_gap >= median_height * 0.15 else 0.58
            candidates.append(
                _crop_candidate(
                    image,
                    mask,
                    (x0, y0, x1, y1),
                    glyph_id=f"ref-{reading_order:03d}",
                    row_index=row_index,
                    column_index=column_index,
                    reading_order=reading_order,
                    confidence=confidence,
                    padding=padding,
                )
            )
            reading_order += 1

    segmentation = PageSegmentation(
        source_path=source_path,
        candidates=tuple(candidates),
        status="ok" if candidates else "failed",
        warnings=tuple(warnings),
    )
    if output_dir is not None:
        _write_reference_diagnostics(image, mask, segmentation, Path(output_dir))
    if not candidates:
        raise SegmentationError("No reference glyph candidates were detected.")
    return ReferenceGallery(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        candidates=tuple(candidates),
        mask_result=InkMaskResult(
            mask=mask,
            confidence=mask_result.confidence,
            method=mask_result.method,
            warnings=[*mask_result.warnings, *warnings],
        ),
        segmentation=segmentation,
    )


def _remove_long_rules(mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Remove only very long, thin horizontal runs, preserving short glyph strokes."""
    cleaned = np.array(mask, dtype=bool, copy=True)
    height, width = cleaned.shape
    minimum_run = round(width * 0.65)
    removed_rows = []
    for y in range(height):
        xs = np.flatnonzero(cleaned[y])
        if xs.size < minimum_run:
            continue
        starts = xs[np.r_[True, np.diff(xs) > 1]]
        ends = np.r_[xs[np.flatnonzero(np.diff(xs) > 1)], xs[-1]] + 1
        longest = max((end - start for start, end in zip(starts, ends, strict=True)), default=0)
        if longest >= minimum_run:
            cleaned[y, :] = False
            removed_rows.append(y)
    warnings = (
        [f"Removed long thin horizontal page rule across {len(removed_rows)} row(s)."]
        if removed_rows
        else []
    )
    return cleaned, warnings


def _candidate_cut_positions(projection: np.ndarray) -> list[int]:
    width = len(projection)
    if width < 2:
        return []
    positions = {0, width}
    zero = projection == 0
    edges = np.diff(np.r_[False, zero, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends, strict=True):
        if start > 0 and end < width:
            positions.add((int(start) + int(end) - 1) // 2)
    scale = float(np.percentile(projection[projection > 0], 35)) if np.any(projection) else 0.0
    for x in range(1, width - 1):
        if (
            projection[x] <= projection[x - 1]
            and projection[x] <= projection[x + 1]
            and projection[x] <= scale
        ):
            positions.add(x)
    return sorted(positions)


def _partition_row(
    row_mask: np.ndarray,
    expected_count: int,
    components: list[Component],
    *,
    x_offset: int,
) -> tuple[list[int], float] | None:
    projection = np.asarray(row_mask, dtype=np.uint8).sum(axis=0)
    occupied = np.flatnonzero(projection)
    if occupied.size == 0:
        return None
    left, right = int(occupied[0]), int(occupied[-1]) + 1
    projection = projection[left:right]
    cuts = _candidate_cut_positions(projection)
    if len(cuts) < expected_count + 1:
        return None

    nonzero_scale = max(1.0, float(np.percentile(projection[projection > 0], 80)))
    crossing = {}
    for cut in cuts[1:-1]:
        absolute_x = x_offset + left + cut
        crossing[cut] = sum(
            component.x0 < absolute_x < component.x1 for component in components
        )
    boundary_cost = {
        cut: float(projection[cut] / nonzero_scale) * 3.0 + crossing.get(cut, 0) * 5.0
        for cut in cuts[1:-1]
    }
    prefix = np.r_[0.0, np.cumsum(projection, dtype=float)]

    component_scale, median_height = _size_scale(components)
    primary_threshold = component_scale * 0.18

    def segment_cost(start: int, end: int) -> float:
        width = end - start
        ink = prefix[end] - prefix[start]
        if width <= 0 or ink <= 0:
            return float("inf")
        absolute_start = x_offset + left + start
        absolute_end = x_offset + left + end
        body_count = sum(
            component.area >= primary_threshold
            and absolute_start <= component.center_x < absolute_end
            for component in components
        )
        component_cost = 0.8 * abs(body_count - 1)
        narrow_cost = max(0.0, median_height * 0.15 - width) / max(1.0, median_height) * 4.0
        area_cost = 2.0 if ink < max(1.0, float(prefix[-1]) * 0.006) else 0.0
        return component_cost + narrow_cost + area_cost

    costs = np.full((expected_count + 1, len(cuts)), np.inf)
    previous = np.full((expected_count + 1, len(cuts)), -1, dtype=int)
    costs[0, 0] = 0.0
    for segments in range(1, expected_count + 1):
        for end_index in range(1, len(cuts)):
            end = cuts[end_index]
            for start_index in range(end_index):
                if not np.isfinite(costs[segments - 1, start_index]):
                    continue
                start = cuts[start_index]
                extra_boundary = boundary_cost[start] if start_index else 0.0
                score = (
                    costs[segments - 1, start_index]
                    + segment_cost(start, end)
                    + extra_boundary
                )
                if score < costs[segments, end_index]:
                    costs[segments, end_index] = score
                    previous[segments, end_index] = start_index
    final_index = len(cuts) - 1
    if not np.isfinite(costs[expected_count, final_index]):
        return None

    boundary_indexes = [final_index]
    cursor = final_index
    for segments in range(expected_count, 0, -1):
        cursor = int(previous[segments, cursor])
        if cursor < 0:
            return None
        boundary_indexes.append(cursor)
    boundary_indexes.reverse()
    boundaries = [left + cuts[index] for index in boundary_indexes]
    internal_confidences = []
    for boundary in boundaries[1:-1]:
        local_cut = boundary - left
        valley = float(projection[local_cut] / nonzero_scale)
        component_penalty = crossing.get(local_cut, 0)
        internal_confidences.append(
            float(np.clip(1.0 - valley - 0.55 * component_penalty, 0.0, 1.0))
        )
    confidence = min(internal_confidences, default=0.0)
    return boundaries, confidence


def _write_sample_diagnostics(
    image: Image.Image,
    source_name: str,
    mask: np.ndarray,
    components: list[Component],
    candidates: list[SegmentedGlyph],
    expected_count: int,
    status: str,
    warnings: list[str],
    boundaries: list[int] | None,
    row_y0: int,
    row_y1: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(mask, output_dir / "sample-mask.png")
    component_preview = image.convert("RGB")
    component_draw = ImageDraw.Draw(component_preview)
    for component in components:
        component_draw.rectangle(
            (component.x0, component.y0, component.x1 - 1, component.y1 - 1),
            outline=(30, 130, 210),
            width=1,
        )
    component_preview.save(output_dir / "sample-components.png", format="PNG")
    _write_overlay(image, candidates, output_dir / "sample-candidates.png")
    boundary_preview = image.convert("RGB")
    boundary_draw = ImageDraw.Draw(boundary_preview)
    if boundaries is not None:
        for boundary in boundaries[1:-1]:
            boundary_draw.line((boundary, row_y0, boundary, row_y1), fill=(240, 25, 25), width=2)
    boundary_preview.save(output_dir / "sample-boundaries.png", format="PNG")
    payload = {
        "source_image": source_name,
        "expected_count": expected_count,
        "detected_count": len(candidates),
        "status": status,
        "warnings": warnings,
        "samples": [_candidate_record(candidate) for candidate in candidates],
        "boundaries": boundaries,
    }
    for filename in ("sample-segmentation.json", "sample-candidates.json"):
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def segment_sample_glyphs(
    image_path: str | Path,
    *,
    expected_count: int,
    ink_mode: InkMode = "auto",
    output_dir: str | Path | None = None,
) -> list[SegmentedGlyph]:
    """Recover exactly ``expected_count`` separated glyphs from a sample page."""
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    source_path = Path(image_path)
    image = _load_page(source_path)
    mask_result = extract_page_ink_mask(image, mode=ink_mode)
    mask, components, primary_area, median_height, warnings = _clean_page_mask(
        mask_result.mask
    )
    mask, rule_warnings = _remove_long_rules(mask)
    warnings.extend(rule_warnings)
    components = _components(mask)
    rows = detect_ink_rows(mask)
    primary_area, median_height = _size_scale(components)
    candidates_by_row = []
    partition_by_row = []
    for row_index, row in enumerate(rows):
        row_y0, row_y1 = _row_y_bounds(row, components, median_height, mask.shape[0])
        row_mask = mask[row_y0:row_y1, :]
        partition = _partition_row(
            row_mask,
            expected_count,
            components,
            x_offset=0,
        )
        if partition is None:
            continue
        boundaries, confidence = partition
        candidates = []
        for column_index, (x0, x1) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            segment = mask[row_y0:row_y1, x0:x1]
            ys, xs = np.nonzero(segment)
            if not len(xs):
                candidates = []
                break
            bbox = (x0, row_y0 + int(ys.min()), x1, row_y0 + int(ys.max()) + 1)
            candidates.append(
                _crop_candidate(
                    image,
                    mask,
                    bbox,
                    glyph_id=f"sample-{column_index:03d}",
                    row_index=row_index,
                    column_index=column_index,
                    reading_order=column_index,
                    confidence=confidence,
                    padding=max(1, round(median_height * 0.08)),
                    clip_bbox=(x0, row_y0, x1, row_y1),
                )
            )
        if len(candidates) == expected_count:
            candidates_by_row.append(candidates)
            partition_by_row.append((boundaries, confidence, row, row_y0, row_y1))
    selected_index = max(
        range(len(candidates_by_row)),
        key=lambda index: (
            partition_by_row[index][1],
            sum(candidate.mask.sum() for candidate in candidates_by_row[index]),
        ),
        default=None,
    )
    if selected_index is None:
        selected: list[SegmentedGlyph] = []
        boundaries = None
        selected_row = rows[0] if rows else RowBand(0, 0, mask.shape[1], mask.shape[0], ())
        selected_row_y0, selected_row_y1 = selected_row.y0, selected_row.y1
        confidence = 0.0
        status = "failed"
    else:
        selected = candidates_by_row[selected_index]
        (
            boundaries,
            confidence,
            selected_row,
            selected_row_y0,
            selected_row_y1,
        ) = partition_by_row[selected_index]
        status = "ok" if confidence >= 0.3 and len(selected) == expected_count else "failed"
    if output_dir is not None:
        _write_sample_diagnostics(
            image,
            source_path.name,
            mask,
            components,
            selected,
            expected_count,
            status,
            warnings,
            boundaries,
            selected_row_y0,
            selected_row_y1,
            Path(output_dir),
        )
    if status != "ok":
        diagnostic = Path(output_dir) / "sample-boundaries.png" if output_dir is not None else None
        raise SegmentationError(
            f"Unable to recover {expected_count} credible letter regions.", diagnostic
        )
    return selected
