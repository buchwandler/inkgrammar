from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .api import compare_extracted_letters, extract_letter
from .batch_report import (
    BatchComparisonReport,
    BatchSampleResult,
    match_score_payload,
    write_batch_report,
)
from .descriptors import GlyphDescriptor
from .matching import (
    DEFAULT_TOP_K,
    MINIMUM_ALTERNATIVES,
    MatchScore,
    ReferenceIndex,
    build_reference_index,
    match_status,
    rank_reference_candidates,
    refine_top_k_matches,
)
from .models import (
    ComparisonIdentity,
    CropRect,
    ExtractedLetter,
    InkMaskResult,
    InkMode,
    OrderedPath,
)
from .report import write_comparison_report
from .segmentation import (
    PageSegmentation,
    ReferenceGallery,
    SegmentationError,
    SegmentedGlyph,
    segment_reference_gallery,
    segment_sample_glyphs,
)
from .skeleton import diagnostics_from_skeleton

_REFERENCE_CACHE_VERSION = 1
_DESCRIPTOR_SIZE = 96
_REFERENCE_DIAGNOSTICS = (
    "reference-mask.png",
    "reference-candidates.png",
)


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_reference_diagnostics(cache_dir: Path, output_dir: Path) -> None:
    source = cache_dir / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in _REFERENCE_DIAGNOSTICS:
        cached = source / filename
        if cached.is_file():
            shutil.copy2(cached, output_dir / filename)


def _cache_metadata(
    gallery: ReferenceGallery,
    index: ReferenceIndex,
    ink_mode: InkMode,
) -> dict[str, Any]:
    return {
        "cache_version": _REFERENCE_CACHE_VERSION,
        "descriptor_size": _DESCRIPTOR_SIZE,
        "ink_mode": ink_mode,
        "source_sha256": gallery.source_sha256,
        "mask_confidence": gallery.mask_result.confidence,
        "mask_method": gallery.mask_result.method,
        "mask_warnings": gallery.mask_result.warnings,
        "segmentation_status": gallery.segmentation.status,
        "segmentation_warnings": list(gallery.segmentation.warnings),
        "candidate_count": len(gallery.candidates),
        "candidates": [
            {
                "id": candidate.glyph_id,
                "bbox": candidate.bbox.model_dump(mode="json"),
                "row": candidate.row_index,
                "column": candidate.column_index,
                "reading_order": candidate.reading_order,
                "confidence": candidate.segmentation_confidence,
                "label": candidate.label,
                "crop": f"{candidate.glyph_id}.png",
                "mask": f"{candidate.glyph_id}-mask.png",
            }
            for candidate in gallery.candidates
        ],
        "descriptors": [
            {
                "glyph_id": descriptor.glyph_id,
                "aspect_ratio": descriptor.aspect_ratio,
                "ink_fill": descriptor.ink_fill,
                "component_count": descriptor.component_count,
                "endpoint_count": descriptor.endpoint_count,
                "junction_count": descriptor.junction_count,
                "cycle_count": descriptor.cycle_count,
                "path_available": descriptor.path_available,
            }
            for descriptor in index.descriptors
        ],
    }


def _save_reference_cache(
    cache_dir: Path,
    gallery: ReferenceGallery,
    index: ReferenceIndex,
    ink_mode: InkMode,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = cache_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for candidate in gallery.candidates:
        candidate.image.save(crops_dir / f"{candidate.glyph_id}.png", format="PNG")
        Image.fromarray(candidate.mask.astype(np.uint8) * 255).save(
            crops_dir / f"{candidate.glyph_id}-mask.png", format="PNG"
        )
    descriptor_masks = (
        np.stack([descriptor.normalized_mask for descriptor in index.descriptors])
        if index.descriptors
        else np.zeros((0, _DESCRIPTOR_SIZE, _DESCRIPTOR_SIZE), dtype=bool)
    )
    np.savez_compressed(
        cache_dir / "descriptors.npz",
        glyph_ids=np.asarray([item.glyph_id for item in index.descriptors], dtype="U"),
        normalized_masks=descriptor_masks,
        aspect_ratios=np.asarray([item.aspect_ratio for item in index.descriptors]),
        ink_fills=np.asarray([item.ink_fill for item in index.descriptors]),
        component_counts=np.asarray([item.component_count for item in index.descriptors]),
    )
    metadata = _cache_metadata(gallery, index, ink_mode)
    (cache_dir / "gallery.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _load_reference_cache(
    cache_dir: Path,
    source_path: Path,
    source_sha256: str,
    ink_mode: InkMode,
) -> tuple[ReferenceGallery, ReferenceIndex, dict[str, Any]]:
    metadata = json.loads((cache_dir / "gallery.json").read_text(encoding="utf-8"))
    if (
        metadata["cache_version"] != _REFERENCE_CACHE_VERSION
        or metadata["descriptor_size"] != _DESCRIPTOR_SIZE
        or metadata["ink_mode"] != ink_mode
        or metadata["source_sha256"] != source_sha256
    ):
        raise ValueError("reference cache metadata does not match this input")
    if not (cache_dir / "descriptors.npz").is_file():
        raise ValueError("reference descriptor cache is missing")
    if not (cache_dir / "segmentation" / "reference-mask.png").is_file():
        raise ValueError("reference mask cache is missing")

    candidates = []
    crops_dir = cache_dir / "crops"
    for item in metadata["candidates"]:
        with Image.open(crops_dir / item["crop"]) as crop_image:
            image = crop_image.convert("RGB")
        with Image.open(crops_dir / item["mask"]) as crop_mask:
            mask = np.asarray(crop_mask.convert("L"), dtype=np.uint8) > 0
        candidates.append(
            SegmentedGlyph(
                glyph_id=item["id"],
                bbox=CropRect.model_validate(item["bbox"]),
                row_index=item["row"],
                column_index=item["column"],
                reading_order=item["reading_order"],
                segmentation_confidence=item["confidence"],
                image=image,
                mask=mask,
                label=item["label"],
            )
        )

    with Image.open(cache_dir / "segmentation" / "reference-mask.png") as mask_image:
        page_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    mask_result = InkMaskResult(
        mask=page_mask,
        confidence=metadata["mask_confidence"],
        method=metadata["mask_method"],
        warnings=list(metadata["mask_warnings"]),
    )
    segmentation = PageSegmentation(
        source_path=source_path,
        candidates=tuple(candidates),
        status=metadata["segmentation_status"],
        warnings=tuple(metadata["segmentation_warnings"]),
    )
    gallery = ReferenceGallery(
        source_path=source_path,
        source_sha256=source_sha256,
        candidates=tuple(candidates),
        mask_result=mask_result,
        segmentation=segmentation,
    )

    with np.load(cache_dir / "descriptors.npz", allow_pickle=False) as stored:
        descriptors = tuple(
            GlyphDescriptor(
                glyph_id=str(glyph_id),
                normalized_mask=normalized_mask,
                aspect_ratio=float(aspect_ratio),
                ink_fill=float(ink_fill),
                component_count=int(component_count),
                endpoint_count=descriptor_data["endpoint_count"],
                junction_count=descriptor_data["junction_count"],
                cycle_count=descriptor_data["cycle_count"],
                path_available=descriptor_data["path_available"],
            )
            for (
                glyph_id,
                normalized_mask,
                aspect_ratio,
                ink_fill,
                component_count,
                descriptor_data,
            ) in zip(
                stored["glyph_ids"],
                stored["normalized_masks"],
                stored["aspect_ratios"],
                stored["ink_fills"],
                stored["component_counts"],
                metadata["descriptors"],
                strict=True,
            )
        )
    index = ReferenceIndex(gallery=gallery, descriptors=descriptors)
    return gallery, index, metadata



def _reference_extraction_paths(cache_dir: Path, glyph_id: str) -> tuple[Path, Path, Path]:
    extraction_dir = cache_dir / "extractions"
    return (
        extraction_dir / f"{glyph_id}.json",
        extraction_dir / f"{glyph_id}.npz",
        extraction_dir / f"{glyph_id}.png",
    )


def _load_reference_extraction(
    cache_dir: Path,
    source_path: Path,
    source_sha256: str,
    candidate: SegmentedGlyph,
    ink_mode: InkMode,
) -> ExtractedLetter:
    metadata_path, arrays_path, image_path = _reference_extraction_paths(
        cache_dir, candidate.glyph_id
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["source_sha256"] != source_sha256
        or metadata["reference_id"] != candidate.glyph_id
        or metadata["ink_mode"] != ink_mode
        or metadata["bbox"] != candidate.bbox.model_dump(mode="json")
    ):
        raise ValueError("cached reference extraction does not match the candidate")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        mask = arrays["mask"]
        skeleton = arrays["skeleton"]
        radius = arrays["radius"]
        path_xy = arrays["path_xy"]
        resampled_xy = arrays["resampled_xy"]
    diagnostics = diagnostics_from_skeleton(skeleton, radius)
    path_info = metadata["ordered_path"]
    ordered_path = (
        OrderedPath(
            path_xy,
            covered_edge_count=path_info["covered_edge_count"],
            total_edge_count=path_info["total_edge_count"],
        )
        if path_info is not None
        else None
    )
    with Image.open(image_path) as crop_image:
        image = crop_image.convert("RGB")
    return ExtractedLetter(
        source_path=source_path,
        crop=candidate.bbox,
        image=image,
        mask_result=InkMaskResult(
            mask=mask,
            confidence=metadata["mask_confidence"],
            method=metadata["mask_method"],
            warnings=metadata["mask_warnings"],
        ),
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=resampled_xy if ordered_path is not None else None,
    )


def _save_reference_extraction(
    cache_dir: Path,
    source_sha256: str,
    ink_mode: InkMode,
    glyph_id: str,
    extraction: ExtractedLetter,
) -> None:
    metadata_path, arrays_path, image_path = _reference_extraction_paths(
        cache_dir, glyph_id
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_path = extraction.ordered_path
    np.savez_compressed(
        arrays_path,
        mask=extraction.mask_result.mask,
        skeleton=extraction.diagnostics.skeleton,
        radius=extraction.diagnostics.radius,
        path_xy=(ordered_path.xy if ordered_path is not None else np.zeros((0, 2))),
        resampled_xy=(
            extraction.resampled_xy
            if extraction.resampled_xy is not None
            else np.zeros((0, 2))
        ),
    )
    extraction.image.save(image_path, format="PNG")
    metadata = {
        "source_sha256": source_sha256,
        "ink_mode": ink_mode,
        "reference_id": glyph_id,
        "bbox": extraction.crop.model_dump(mode="json"),
        "mask_confidence": extraction.mask_result.confidence,
        "mask_method": extraction.mask_result.method,
        "mask_warnings": extraction.mask_result.warnings,
        "topology": extraction.diagnostics.topology.model_dump(mode="json"),
        "ordered_path": (
            {
                "covered_edge_count": ordered_path.covered_edge_count,
                "total_edge_count": ordered_path.total_edge_count,
            }
            if ordered_path is not None
            else None
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _get_reference_extraction(
    cache_dir: Path,
    source_path: Path,
    source_sha256: str,
    candidate: SegmentedGlyph,
    ink_mode: InkMode,
) -> ExtractedLetter:
    try:
        return _load_reference_extraction(
            cache_dir,
            source_path,
            source_sha256,
            candidate,
            ink_mode,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        extraction = extract_letter(
            source_path,
            candidate.bbox,
            ink_mode=ink_mode,
            allow_path_failure=True,
        )
        _save_reference_extraction(
            cache_dir,
            source_sha256,
            ink_mode,
            candidate.glyph_id,
            extraction,
        )
        return extraction


def _write_reference_index(
    gallery: ReferenceGallery,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous in output_dir.glob("ref-*.png"):
        previous.unlink()
    public_metadata = {
        **metadata,
        "source_image": gallery.source_path.name,
        "cache_hit": metadata.get("cache_hit", False),
    }
    (output_dir / "gallery.json").write_text(
        json.dumps(public_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for candidate in gallery.candidates:
        candidate.image.save(output_dir / f"{candidate.glyph_id}.png", format="PNG")


def _write_reference_manifest(
    gallery: ReferenceGallery,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    payload = {
        "source_image": gallery.source_path.name,
        "source_sha256": gallery.source_sha256,
        "status": gallery.segmentation.status,
        "warnings": list(gallery.segmentation.warnings),
        "candidate_count": len(gallery.candidates),
        "candidates": metadata["candidates"],
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumbnail = image.convert("RGB").copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        thumbnail,
        ((size[0] - thumbnail.width) // 2, (size[1] - thumbnail.height) // 2),
    )
    return canvas


def _write_alternatives(
    sample: SegmentedGlyph,
    ranked: list[MatchScore],
    candidates: dict[str, SegmentedGlyph],
    output_dir: Path,
) -> None:
    cell_width, image_height, cell_height = 220, 110, 142
    sheet_width = cell_width * min(3, len(ranked))
    sheet_height = cell_height * ((len(ranked) + 2) // 3)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, score in enumerate(ranked):
        x = (index % 3) * cell_width
        y = (index // 3) * cell_height
        image = _fit_thumbnail(
            candidates[score.reference_id].image,
            (cell_width - 16, image_height),
        )
        sheet.paste(image, (x + 8, y + 4))
        draw.text(
            (x + 8, y + image_height + 8),
            f"#{score.rank} {score.reference_id} cost={score.cost:.4f}",
            fill="black",
        )
    sheet.save(output_dir / "alternatives.png", format="PNG")
    sample.image.convert("RGB").save(output_dir / "sample.png", format="PNG")


def _write_match_json(
    output_dir: Path,
    *,
    sample: SegmentedGlyph,
    best: MatchScore,
    ranked: list[MatchScore],
    candidates: dict[str, SegmentedGlyph],
    status: str,
    pair_report_dir: Path,
) -> None:
    payload = {
        "sample_id": sample.glyph_id,
        "sample_bbox": sample.bbox.model_dump(mode="json"),
        "segmentation_confidence": sample.segmentation_confidence,
        "status": status,
        "best": {
            **match_score_payload(best),
            "label": candidates[best.reference_id].label,
        },
        "alternatives": [
            {
                **match_score_payload(score),
                "label": candidates[score.reference_id].label,
            }
            for score in ranked[1:]
        ],
        "pair_report_dir": pair_report_dir.name,
    }
    (output_dir / "match.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def compare_letter_page(
    reference_path: str | Path,
    sample_path: str | Path,
    *,
    expected_letters: int,
    output_dir: str | Path,
    reference_ink_mode: InkMode = "auto",
    sample_ink_mode: InkMode = "auto",
    top_k: int = DEFAULT_TOP_K,
) -> BatchComparisonReport:
    """Segment, retrieve, compare, and report a page of standalone sample glyphs."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if expected_letters < 1:
        raise ValueError("expected_letters must be positive")
    reference = Path(reference_path)
    sample_path = Path(sample_path)
    root = Path(output_dir)
    segmentation_dir = root / "segmentation"
    reference_index_dir = root / "reference-index"
    cache_dir = root / ".reference-cache" / _source_hash(reference)
    source_sha256 = cache_dir.name
    root.mkdir(parents=True, exist_ok=True)

    cache_hit = False
    try:
        gallery, index, metadata = _load_reference_cache(
            cache_dir,
            reference,
            source_sha256,
            reference_ink_mode,
        )
        cache_hit = True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        try:
            gallery = segment_reference_gallery(
                reference,
                ink_mode=reference_ink_mode,
                output_dir=cache_dir / "segmentation",
            )
        except SegmentationError as error:
            _copy_reference_diagnostics(cache_dir, segmentation_dir)
            diagnostic = segmentation_dir / "reference-candidates.png"
            raise SegmentationError(str(error), diagnostic_path=diagnostic) from error
        index = build_reference_index(gallery)
        metadata = _save_reference_cache(cache_dir, gallery, index, reference_ink_mode)

    metadata = {**metadata, "cache_hit": cache_hit}
    _copy_reference_diagnostics(cache_dir, segmentation_dir)
    _write_reference_index(gallery, metadata, reference_index_dir)
    _write_reference_manifest(gallery, metadata, segmentation_dir)

    samples = segment_sample_glyphs(
        sample_path,
        expected_count=expected_letters,
        ink_mode=sample_ink_mode,
        output_dir=segmentation_dir,
    )
    effective_top_k = max(top_k, MINIMUM_ALTERNATIVES)
    candidates = {candidate.glyph_id: candidate for candidate in gallery.candidates}
    results = []
    reference_extractions: dict[str, ExtractedLetter] = {}

    for sample_candidate in samples:
        coarse = rank_reference_candidates(
            sample_candidate,
            index,
            top_k=min(effective_top_k, len(gallery.candidates)),
        )
        for coarse_score in coarse:
            reference_id = coarse_score.reference_id
            if reference_id not in reference_extractions:
                reference_extractions[reference_id] = _get_reference_extraction(
                    cache_dir,
                    reference,
                    source_sha256,
                    candidates[reference_id],
                    reference_ink_mode,
                )
        ranked = refine_top_k_matches(
            sample_candidate,
            gallery,
            coarse,
            reference_extractions=reference_extractions,
        )
        if not ranked:
            raise SegmentationError("No reference candidates are available for matching.")
        best = ranked[0]
        status = match_status(ranked)
        sample_dir = root / sample_candidate.glyph_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        matched_reference = candidates[best.reference_id]
        matched_reference.image.convert("RGB").save(
            sample_dir / "matched-reference.png", format="PNG"
        )
        _write_alternatives(sample_candidate, ranked, candidates, sample_dir)

        reference_extracted = reference_extractions[matched_reference.glyph_id]
        sample_extracted = extract_letter(
            sample_path,
            sample_candidate.bbox,
            ink_mode=sample_ink_mode,
            allow_path_failure=True,
        )
        pair_result = compare_extracted_letters(
            reference_extracted,
            sample_extracted,
            identity=ComparisonIdentity(
                sample_id=sample_candidate.glyph_id,
                reference_id=matched_reference.glyph_id,
                label=matched_reference.label,
            ),
        )
        pair_report = write_comparison_report(pair_result, sample_dir)
        _write_match_json(
            sample_dir,
            sample=sample_candidate,
            best=best,
            ranked=ranked,
            candidates=candidates,
            status=status,
            pair_report_dir=pair_report.output_dir,
        )
        results.append(
            BatchSampleResult(
                sample_id=sample_candidate.glyph_id,
                sample_bbox=sample_candidate.bbox,
                sample_confidence=sample_candidate.segmentation_confidence,
                reference_id=matched_reference.glyph_id,
                reference_label=matched_reference.label,
                status=status,
                score=best,
                alternatives=tuple(ranked[1:]),
                report_dir=pair_report.output_dir,
                report_json=pair_report.json_path,
            )
        )

    return write_batch_report(
        root,
        reference_path=reference,
        sample_path=sample_path,
        reference_sha256=source_sha256,
        expected_count=expected_letters,
        results=tuple(results),
        reference_cache_hit=cache_hit,
    )
