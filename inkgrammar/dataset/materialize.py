"""Deterministic dataset materialization.

Turns a dataset recipe plus authorized private sources into a materialized
dataset root (the brief's section 13 layout): per-page extraction output,
raw/clean masks with cleanup records, normalized previews, derived
skeleton/centerline/topology artifacts with algorithm versions, and a
canonical ``index.json``. All artifact paths are relative to the dataset root
and all JSON is written canonically (sorted keys), so two clean runs produce
byte-identical semantic artifacts.

Materialization combines automatic candidates with review patches into the
effective dataset; stale patches are reported and never applied.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import load_layout
from ..skeleton_walk import _euler_path, _pixel_neighbors
from .layouts import load_region_layout
from .models import ObservationKind, ObservationManifest
from .preprocess import (
    PreprocessParams,
    estimate_metric_guides,
    isolate_target_ink,
    preprocess_crop,
)
from .project import DatasetPage, DatasetProjectBundle, load_observation_manifest, verify_sources
from .review import apply_review_patch, load_review_patches
from .transforms import standard_frame

DATASET_INDEX_FORMAT = "inkgrammar-materialized-dataset"
EXTRACTION_VERSION = 1
SOURCE_VERIFICATION_FORMAT = "inkgrammar-source-verification"
DERIVE_ALGORITHMS = {
    "skeleton": "zhang-suen-v1+spur-prune-v1",
    "centerline": "edge-covering-walk-v1",
    "topology": "pixel-graph-v1",
}
MIN_SPUR_LENGTH_PX = 3

# Normalized preview canvas convention (preview only, not stored geometry):
# glyph x = 0 at canvas x 16, baseline (glyph y = 0) at canvas y 72, one
# glyph unit (x-height when known) is 32 canvas px, y upward against image y.
PREVIEW_SIZE = (128, 96)
PREVIEW_X_ORIGIN = 16
PREVIEW_BASELINE = 72
PREVIEW_UNIT = 32


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict) -> str:
    """Write canonical JSON and return its SHA-256."""
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return _sha256_bytes(payload.encode("utf-8"))


def _write_mask(path: Path, mask: np.ndarray) -> str:
    """Write a binary mask PNG and return its SHA-256."""
    buffer = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


# --- derived artifacts (section 28: candidates, not source truth) ---


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    count = np.zeros(mask.shape, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            count += padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return count * mask


def _thin_zhang_suen(mask: np.ndarray) -> np.ndarray:
    image = mask.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1)
            offsets = {
                "p2": (-1, 0),
                "p3": (-1, 1),
                "p4": (0, 1),
                "p5": (1, 1),
                "p6": (1, 0),
                "p7": (1, -1),
                "p8": (0, -1),
                "p9": (-1, -1),
            }
            p = {
                name: padded[1 + dy : 1 + dy + image.shape[0], 1 + dx : 1 + dx + image.shape[1]]
                for name, (dy, dx) in offsets.items()
            }
            neighbors = sum(p.values())
            transitions = sum(
                (~p[a] & p[b]).astype(int)
                for a, b in (
                    ("p2", "p3"),
                    ("p3", "p4"),
                    ("p4", "p5"),
                    ("p5", "p6"),
                    ("p6", "p7"),
                    ("p7", "p8"),
                    ("p8", "p9"),
                    ("p9", "p2"),
                )
            )
            if phase == 0:
                condition = ~(p["p2"] & p["p4"] & p["p6"]) & ~(p["p4"] & p["p6"] & p["p8"])
            else:
                condition = ~(p["p2"] & p["p4"] & p["p8"]) & ~(p["p2"] & p["p6"] & p["p8"])
            remove = image & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & condition
            if remove.any():
                image[remove] = False
                changed = True
    return image


def _spur_prune(skeleton: np.ndarray, *, min_length: int) -> np.ndarray:
    pruned = skeleton.copy()
    while True:
        endpoints = sorted(zip(*np.nonzero(_neighbor_count(pruned) == 1), strict=True))
        removed_any = False
        for start in endpoints:
            if not pruned[start]:
                continue
            branch = [start]
            previous, current = None, start
            while True:
                y, x = current
                height, width = pruned.shape
                next_pixels = [
                    (ny, nx)
                    for ny in range(max(0, y - 1), min(height, y + 2))
                    for nx in range(max(0, x - 1), min(width, x + 2))
                    if pruned[ny, nx] and (ny, nx) != current and (ny, nx) != previous
                ]
                if len(next_pixels) != 1:
                    break
                previous, current = current, next_pixels[0]
                if _neighbor_count(pruned)[current] != 2:
                    break
                branch.append(current)
            if len(branch) < min_length and _neighbor_count(pruned)[branch[-1]] >= 3:
                for pixel in branch[:-1]:
                    pruned[pixel] = False
                removed_any = True
        if not removed_any:
            return pruned


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _component_count(skeleton: np.ndarray) -> int:
    """Number of connected components in the skeleton pixel graph."""
    ys, xs = np.nonzero(skeleton)
    remaining = set(zip(ys.tolist(), xs.tolist(), strict=True))
    components = 0
    while remaining:
        components += 1
        stack = [min(remaining)]
        remaining.discard(stack[0])
        while stack:
            y, x = stack.pop()
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if (ny, nx) in remaining:
                        remaining.discard((ny, nx))
                        stack.append((ny, nx))
    return components



def _chain_cost(chain: list[tuple[int, int]]) -> float:
    """Jump energy plus direction-change energy of an ordered pixel chain."""
    cost = 0.0
    previous_direction = None
    for first, second in zip(chain, chain[1:], strict=False):
        cost += (second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2
        direction = np.asarray((second[0] - first[0], second[1] - first[1]), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction = direction / norm
        if previous_direction is not None:
            cost += 4.0 * (1.0 - float(previous_direction @ direction))
        previous_direction = direction
    return cost


def _best_chain(pieces: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Full-coverage chain with the smallest jumps and smoothest turns.

    The true pen order is the ordering of the walk pieces that minimizes
    junction jumps and direction changes, so for up to five pieces every order
    and orientation is tried. More pieces fall back to greedy chaining.
    """
    if len(pieces) <= 5:
        from itertools import permutations

        best_cost = None
        best_chain = None
        for order in permutations(range(len(pieces))):
            for mask in range(2 ** len(pieces)):
                chain: list[tuple[int, int]] = []
                for position, index in enumerate(order):
                    piece = pieces[index]
                    if mask & (1 << position):
                        piece = list(reversed(piece))
                    chain.extend(piece)
                cost = _chain_cost(chain)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_chain = chain
        return best_chain

    chain = list(pieces[0])
    remaining = list(pieces[1:])
    while remaining:
        best_cost = None
        best_index = None
        best_flipped = None
        for index, piece in enumerate(remaining):
            for flipped in (piece, list(reversed(piece))):
                jump = (chain[-1][0] - flipped[0][0]) ** 2 + (chain[-1][1] - flipped[0][1]) ** 2
                if best_cost is None or jump < best_cost:
                    best_cost = jump
                    best_index = index
                    best_flipped = flipped
        chain.extend(best_flipped)
        remaining.pop(best_index)
    return chain


def _stitch_candidates(candidates: list[dict]) -> list[list[int]] | None:
    """Chain walk pieces into one ordered polyline for the fitter.

    The chain is oriented to start at its leftmost endpoint (Latin writing
    starts at the left). Pieces shorter than the caller's threshold are
    junction clutter and must already be filtered out.
    """
    pieces = [[tuple(point) for point in candidate["points"]] for candidate in candidates]
    if not pieces:
        return None
    if len(pieces) == 1:
        return [list(point) for point in pieces[0]]
    chain = _best_chain(pieces)
    if chain[0][0] > chain[-1][0]:
        chain = list(reversed(chain))
    return [list(point) for point in chain]


def derive_artifacts(clean_mask: np.ndarray) -> dict[str, dict]:
    """Compute derived skeleton/centerline/topology artifacts with versions.

    These are derived computational candidates, never source truth; each
    artifact records the algorithm and version that produced it.
    """
    skeleton = _spur_prune(_thin_zhang_suen(clean_mask), min_length=MIN_SPUR_LENGTH_PX)

    ys, xs = np.nonzero(skeleton)
    pixels = sorted(zip(ys.tolist(), xs.tolist(), strict=True))
    pixel_set = set(pixels)
    counts = _neighbor_count(skeleton)
    endpoints = [[int(x), int(y)] for (y, x) in pixels if counts[y, x] == 1]
    junctions = [[int(x), int(y)] for (y, x) in pixels if counts[y, x] >= 3]
    edge_count = sum(
        1
        for (y, x) in pixels
        for ny in (y - 1, y, y + 1)
        for nx in (x - 1, x, x + 1)
        if (ny, nx) in pixel_set and (ny, nx) > (y, x)
    )

    traversed: set[frozenset[tuple[int, int]]] = set()

    def neighbors(pixel: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = pixel
        return [
            (ny, nx)
            for ny in range(max(0, y - 1), min(skeleton.shape[0], y + 2))
            for nx in range(max(0, x - 1), min(skeleton.shape[1], x + 2))
            if (ny, nx) in pixel_set and (ny, nx) != pixel
        ]

    starts = sorted(pixel_set, key=lambda px: (counts[px] != 1, px))
    candidates = []
    for start in starts:
        if all(frozenset((start, other)) in traversed for other in neighbors(start)):
            continue
        walk = [start]
        previous = None
        current = start
        while True:
            options = [
                other
                for other in neighbors(current)
                if frozenset((current, other)) not in traversed
            ]
            if not options:
                break
            if previous is not None:
                step_y, step_x = current[0] - previous[0], current[1] - previous[1]
                options.sort(
                    key=lambda other: (
                        -((other[0] - current[0]) * step_y + (other[1] - current[1]) * step_x)
                    )
                )
            next_pixel = options[0]
            traversed.add(frozenset((current, next_pixel)))
            walk.append(next_pixel)
            previous, current = current, next_pixel
        if len(walk) >= 2:
            length = sum(
                float(np.hypot(b[0] - a[0], b[1] - a[1]))
                for a, b in zip(walk, walk[1:], strict=False)
            )
            candidates.append(
                {
                    "points": [[int(x), int(y)] for (y, x) in walk],
                    "start": [int(walk[0][1]), int(walk[0][0])],
                    "end": [int(walk[-1][1]), int(walk[-1][0])],
                    "length_px": round(length, 4),
                }
            )
    candidates.sort(key=lambda candidate: candidate["length_px"], reverse=True)
    adjacency = {pixel: _pixel_neighbors(pixel_set, pixel) for pixel in pixel_set}
    euler_walk = _euler_path(pixel_set, adjacency)
    if euler_walk is not None:
        stitched_points = [[int(x), int(y)] for (y, x) in euler_walk]
    else:
        stitchable = [c for c in candidates if c["length_px"] >= 4.0]
        stitched_points = _stitch_candidates(stitchable) if stitchable else None
    if stitched_points is not None:
        stitched_length = sum(
            float(np.hypot(b[0] - a[0], b[1] - a[1]))
            for a, b in zip(stitched_points, stitched_points[1:], strict=False)
        )
        for candidate in candidates:
            candidate["stitched"] = False
        candidates.insert(
            0,
            {
                "points": stitched_points,
                "start": stitched_points[0],
                "end": stitched_points[-1],
                "length_px": round(stitched_length, 4),
                "stitched": True,
            },
        )
    else:
        for candidate in candidates:
            candidate["stitched"] = False

    components = _component_count(skeleton)
    return {
        "skeleton": {
            "data": _npy_bytes(skeleton),
            "format": "npy",
            "algorithm": DERIVE_ALGORITHMS["skeleton"],
        },
        "centerline": {
            "data": {"algorithm": DERIVE_ALGORITHMS["centerline"], "candidates": candidates},
            "format": "json",
            "algorithm": DERIVE_ALGORITHMS["centerline"],
        },
        "topology": {
            "data": {
                "algorithm": DERIVE_ALGORITHMS["topology"],
                "pixel_count": len(pixels),
                "components": components,
                "endpoints": endpoints,
                "junctions": junctions,
                "cycle_rank": edge_count - len(pixels) + components,
            },
            "format": "json",
            "algorithm": DERIVE_ALGORITHMS["topology"],
        },
    }


# --- normalized preview ---


def render_normalized_preview(
    clean_mask: np.ndarray, *, baseline_px: float, unit_px: float
) -> Image.Image:
    height, width = clean_mask.shape
    canvas_y, canvas_x = np.mgrid[0 : PREVIEW_SIZE[1], 0 : PREVIEW_SIZE[0]]
    source_x = (canvas_x - PREVIEW_X_ORIGIN) / PREVIEW_UNIT * unit_px
    source_y = baseline_px - (PREVIEW_BASELINE - canvas_y) / PREVIEW_UNIT * unit_px
    sample_x = np.rint(source_x).astype(int)
    sample_y = np.rint(source_y).astype(int)
    valid = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    canvas = np.zeros(PREVIEW_SIZE[::-1], dtype=bool)
    canvas[valid] = clean_mask[sample_y[valid], sample_x[valid]]
    return Image.fromarray((canvas * 255).astype(np.uint8))


def _preview_frame(mask: np.ndarray, metrics: dict | None) -> tuple[float, float]:
    """Baseline and unit (px) for the preview: metric candidates first."""
    rows = np.flatnonzero(mask.any(axis=1))
    ink_bottom = float(rows.max()) if rows.size else 0.0
    ink_height = float(rows.max() - rows.min() + 1) if rows.size else 1.0
    if metrics and metrics.get("baseline_y") is not None:
        baseline_px = float(metrics["baseline_y"]) * mask.shape[0]
        for key in ("x_height_y", "cap_height_y"):
            if metrics.get(key) is not None:
                return baseline_px, abs(baseline_px - float(metrics[key]) * mask.shape[0])
        return baseline_px, ink_height
    return ink_bottom, ink_height


# --- materialization ---


def _glyph_class(label: str) -> str:
    if len(label) == 1 and label.islower():
        return "lowercase"
    if len(label) == 1 and label.isupper():
        return "uppercase"
    return "mixed"


def _locator_basename(locator: str) -> str:
    """Basename of a crop locator such as ``local-only:alphabet/xx.png``."""
    return locator.rsplit("/", 1)[-1].split(":")[-1]


def _page_dir_name(bundle: DatasetProjectBundle, page: DatasetPage) -> str:
    stem = Path(page.layout).stem
    return stem.removeprefix(f"{bundle.project.id}-")


def _relations_from_region(record: dict) -> dict:
    relations = {}
    if record.get("members"):
        relations["member_labels"] = record["members"]
    if record.get("category_hint"):
        relations["connection_category"] = record["category_hint"]
    sequence = record.get("sequence") or []
    if record.get("kind") == "connection-pair" and len(sequence) == 2:
        relations["left_label"] = sequence[0]
        relations["right_label"] = sequence[1]
        relations["sequence_text"] = record.get("label") or "".join(sequence)
    elif record.get("kind") in ("word", "sequence") and record.get("label"):
        relations["sequence_text"] = record["label"]
    return relations


_EXTRACTORS = {
    "labeled-rows-v1": "extract_labeled_page",
    "region-layout-v1": "extract_region_page",
    "annotated-region-v1": "extract_annotated_region_page",
}


def _run_extractor(image_path: Path, layout, output_dir: Path, extractor: str) -> list[dict]:
    from .. import page as page_module

    extract = getattr(page_module, _EXTRACTORS[extractor])
    return extract(image_path, layout, output_dir)["detections"]


def _row_centers(
    layout, detection: dict, source_size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...], tuple[float, float]] | None:
    """Reviewed cell center and the row's other label centers, in crop pixels."""
    row = next((row for row in layout.rows if row.id == detection["row"]), None)
    if row is None or row.centers is None:
        return None
    width, height = source_size
    bbox = detection["bbox"]
    row_y = (row.y[0] + row.y[1]) / 2 * height
    points = [(center * width - bbox[0], row_y - bbox[1]) for center in row.centers]
    own = detection["index"]
    neighbors = tuple(point for index, point in enumerate(points) if index != own)
    left_fence = (points[own - 1][0] + points[own][0]) / 2 if own > 0 else -1.0
    right_fence = (
        (points[own][0] + points[own + 1][0]) / 2 if own + 1 < len(points) else width + 1.0
    )
    return points[own], neighbors, (left_fence, right_fence)


def _frame_from_metrics(
    source_bbox: list[int], crop_size: tuple[int, int], metrics: dict | None
) -> dict | None:
    """Candidate ObservationFrame in source pixels from metric guides."""
    baseline_y = (metrics or {}).get("baseline_y")
    x_height_y = (metrics or {}).get("x_height_y")
    if baseline_y is None or x_height_y is None:
        return None
    x_height_px = (baseline_y - x_height_y) * crop_size[1]
    if x_height_px <= 0:
        return None
    frame = standard_frame(
        origin_source_px=(
            float(source_bbox[0]),
            float(source_bbox[1]) + baseline_y * crop_size[1],
        ),
        x_height_px=float(x_height_px),
    )
    return frame.model_dump(mode="json")


def _write_overlay(
    output_dir: Path,
    rel_path: str,
    crop_image: Image.Image,
    clean_mask: np.ndarray,
    kept_mask: np.ndarray,
) -> str:
    """Isolation review overlay: kept ink green, excluded ink red, over crop."""
    gray = np.asarray(crop_image.convert("L"))
    overlay = np.stack([gray, gray, gray], axis=-1)
    overlay[kept_mask] = (60, 180, 60)
    overlay[clean_mask & ~kept_mask] = (200, 40, 40)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    payload = buffer.getvalue()
    path = output_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def materialize_dataset(
    bundle: DatasetProjectBundle,
    output_dir: str | Path,
    *,
    review_dir: str | Path | None = None,
    params: PreprocessParams | None = None,
) -> dict:
    """Materialize the dataset described by ``bundle`` into ``output_dir``.

    Returns the canonical index dict (also written to ``index.json``).
    """
    output_dir = Path(output_dir)
    params = params or PreprocessParams()
    if review_dir is None:
        review_dir = bundle.private_root.parent / "annotations" / bundle.project.id / "reviews"
    review_dir = Path(review_dir)

    verification = verify_sources(bundle)
    if not verification.ok:
        problems = {
            check.source_id: check.problems for check in verification.checks if check.problems
        }
        raise ValueError(f"source verification failed: {problems}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "sources" / "verified.json",
        {
            "format": SOURCE_VERIFICATION_FORMAT,
            "format_version": 1,
            "sources": [check.model_dump() for check in verification.checks],
        },
    )

    manifest: ObservationManifest = load_observation_manifest(bundle.observation_manifest_path)
    source_hashes = {source.source_id: source.sha256 for source in manifest.sources}
    manifest_items = {
        _locator_basename(item.crop_locator): item.model_dump(mode="json")
        for item in manifest.items
        if item.crop_locator
    }
    patches = load_review_patches(review_dir)
    prefix = bundle.project.id.replace("-", "")

    observations: dict[str, dict] = {}
    for page in bundle.project.pages:
        source = next(s for s in manifest.sources if s.source_id == page.source_id)
        image_path = bundle.private_root / source.filename
        page_dir_name = _page_dir_name(bundle, page)
        page_dir = output_dir / page_dir_name
        if page.extractor == "labeled-rows-v1":
            layout = load_layout(bundle.layout_path(page))
        else:
            layout = load_region_layout(bundle.layout_path(page))
        detections = _run_extractor(image_path, layout, page_dir, page.extractor)

        for detection in detections:
            record = _observation_record(
                prefix=prefix,
                page=page,
                layout=layout,
                detection=detection,
                manifest_items=manifest_items,
                output_dir=output_dir,
                page_dir_name=page_dir_name,
                params=params,
                source_size=(source.width_px, source.height_px),
            )
            if record["id"] in observations:
                raise ValueError(f"duplicate observation id: {record['id']}")
            patch = patches.get(record["id"])
            if patch is None:
                record["review_patch"] = {"status": "none"}
            else:
                record, _ = apply_review_patch(
                    record, patch, source_sha256=source_hashes[page.source_id]
                )
            observations[record["id"]] = record

    index = {
        "format": DATASET_INDEX_FORMAT,
        "format_version": 1,
        "model_id": bundle.project.id,
        "source_hashes": source_hashes,
        "extraction_version": EXTRACTION_VERSION,
        "preprocess": params.model_dump(mode="json"),
        "observations": observations,
    }
    _write_json(output_dir / "index.json", index)
    return index


def _observation_record(
    *,
    prefix: str,
    page: DatasetPage,
    layout,
    detection: dict,
    manifest_items: dict[str, dict],
    output_dir: Path,
    page_dir_name: str,
    params: PreprocessParams,
    source_size: tuple[int, int],
) -> dict:
    matched = manifest_items.get(_locator_basename(detection["crop"]))
    label = detection.get("label") or (matched or {}).get("label") or ""
    if "row" in detection:
        kind = (matched or {}).get("kind", "glyph-isolated")
        region_id = f"{detection['row']}:{detection['index']}"
        source_bbox = (matched or {}).get("source_bbox") or detection["bbox"]
        target_crop = detection["crop"]
        confidence = detection.get("confidence")
        relations = (matched or {}).get("relations") or {}
    else:
        kind = detection["kind"]
        region_id = detection["region"]
        ink = detection.get("ink_region")
        source_bbox = (ink or detection).get("bbox")
        target_crop = (ink or {}).get("crop") or detection["crop"]
        confidence = detection.get("confidence")
        relations = _relations_from_region(detection)
    if kind not in ObservationKind.__args__:
        raise ValueError(f"region kind is not an observation kind: {kind}")

    crop_rel = f"{page_dir_name}/{target_crop}"
    crop_image = Image.open(output_dir / crop_rel).convert("RGB")
    result = preprocess_crop(
        crop_image, params, glyph_class=_glyph_class(label) if label else "mixed"
    )
    target_mask = result.clean_mask
    metrics = result.metrics.model_dump(mode="json")
    isolation = None
    centers = _row_centers(layout, detection, source_size) if "row" in detection else None
    if centers is not None:
        center, neighbor_centers, fence = centers
        target_mask, proposal = isolate_target_ink(
            result.clean_mask, center=center, neighbor_centers=neighbor_centers
        )
        # cell ownership fence: cursive neighbors touch, so keep only ink
        # between the midpoints to the adjacent label centers
        columns = np.arange(target_mask.shape[1])
        inside = (columns >= fence[0]) & (columns < fence[1])
        target_mask = target_mask & inside[None, :]
        metrics = estimate_metric_guides(
            target_mask, glyph_class=_glyph_class(label) if label else "mixed"
        ).model_dump(mode="json")
        isolation = {"status": "pending", **proposal.model_dump(mode="json")}
    basename = Path(target_crop).stem
    masks_dir = f"{page_dir_name}/masks"
    derived_dir = f"{page_dir_name}/derived/{basename}"
    artifacts = {
        "crop": crop_rel,
        "raw_mask": f"{masks_dir}/{basename}__raw.png",
        "clean_mask": f"{masks_dir}/{basename}__clean.png",
        "cleanup": f"{masks_dir}/{basename}__cleanup.json",
        "fitting_mask": f"{masks_dir}/{basename}__clean.png",
        "normalized": f"{page_dir_name}/normalized/{basename}.png",
    }
    if isolation is not None:
        artifacts["fitting_mask"] = f"{masks_dir}/{basename}__isolation.png"
        artifacts["isolation_mask"] = f"{masks_dir}/{basename}__isolation.png"
        artifacts["isolation_overlay"] = f"{page_dir_name}/isolation/{basename}__overlay.png"
    artifact_hashes = {
        "crop": _sha256_bytes((output_dir / crop_rel).read_bytes()),
        "raw_mask": _write_mask(output_dir / artifacts["raw_mask"], result.raw_mask),
        "clean_mask": _write_mask(output_dir / artifacts["clean_mask"], result.clean_mask),
        "cleanup": _write_json(
            output_dir / artifacts["cleanup"], result.cleanup.model_dump(mode="json")
        ),
    }
    if isolation is not None:
        artifact_hashes["isolation_mask"] = _write_mask(
            output_dir / artifacts["isolation_mask"], target_mask
        )
        artifact_hashes["isolation_overlay"] = _write_overlay(
            output_dir, artifacts["isolation_overlay"], crop_image, result.clean_mask, target_mask
        )

    baseline_px, unit_px = _preview_frame(target_mask, metrics)
    preview = render_normalized_preview(target_mask, baseline_px=baseline_px, unit_px=unit_px)
    preview_path = output_dir / artifacts["normalized"]
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_bytes = io.BytesIO()
    preview.save(preview_bytes, format="PNG")
    preview_path.write_bytes(preview_bytes.getvalue())
    artifact_hashes["normalized"] = _sha256_bytes(preview_bytes.getvalue())

    derived = {}
    for name, artifact in derive_artifacts(target_mask).items():
        if artifact["format"] == "npy":
            rel_path = f"{derived_dir}/{name}.npy"
            (output_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (output_dir / rel_path).write_bytes(artifact["data"])
            digest = _sha256_bytes(artifact["data"])
        else:
            rel_path = f"{derived_dir}/{name}.json"
            digest = _write_json(output_dir / rel_path, artifact["data"])
        derived[name] = {
            "path": rel_path,
            "sha256": digest,
            "algorithm": artifact["algorithm"],
        }

    record = {
        "id": (matched or {}).get("id") or f"{prefix}-{region_id}",
        "kind": kind,
        "label": label,
        "unicode": (matched or {}).get("unicode") or _unicode_of(label, kind),
        "source_id": page.source_id,
        "source_bbox": list(source_bbox),
        "label_source": (matched or {}).get("label_source") or "declared-layout-region",
        "review_status": (matched or {}).get("review_status") or "pending",
        "fitting_status": (matched or {}).get("fitting_status") or "pending",
        "isolation_status": (matched or {}).get("isolation_status")
        or ("pending" if isolation is not None else None),
        "extraction": {
            "extractor": page.extractor,
            "extractor_version": EXTRACTION_VERSION,
            "source_bbox_policy": "declared-row-sequence" if "row" in detection else "declared",
            "layout_id": layout.id,
            "layout_region_id": region_id,
            "detection_confidence": confidence,
        },
        "metrics": metrics,
        "frame": _frame_from_metrics(source_bbox, crop_image.size, metrics),
        "isolation": isolation,
        "relations": relations,
        "detection": detection,
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "derived": derived,
    }
    if matched:
        record["crop_locator"] = matched.get("crop_locator")
        record["notes"] = matched.get("notes")
    return record


def _unicode_of(label: str, kind: str) -> list[str]:
    if kind.startswith("glyph") and label:
        return [f"U+{ord(ch):04X}" for ch in label]
    return []
