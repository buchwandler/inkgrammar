from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .crop import coerce_crop_rect, load_crop
from .mask import extract_ink_mask
from .models import (
    ComparisonIdentity,
    ComparisonResult,
    CropRect,
    ExtractedLetter,
    InkMode,
)
from .path import (
    PathExtractionError,
    align_ordered_paths,
    recover_ordered_path,
    resample_path,
)
from .report import WrittenComparisonReport, write_comparison_report
from .skeleton import analyze_skeleton


def extract_letter(
    image_path: str | Path,
    crop: CropRect | Sequence[int],
    *,
    ink_mode: InkMode = "auto",
    allow_path_failure: bool = False,
) -> ExtractedLetter:
    """Crop an image and recover its mask, topology, and optional ordered path."""
    source_path = Path(image_path)
    crop_rect = coerce_crop_rect(crop)
    image = load_crop(source_path, crop_rect)
    mask_result = extract_ink_mask(image, mode=ink_mode)
    diagnostics = analyze_skeleton(mask_result.mask)
    try:
        ordered_path = recover_ordered_path(diagnostics)
    except PathExtractionError:
        if not allow_path_failure:
            raise
        ordered_path = None
    return ExtractedLetter(
        source_path=source_path,
        crop=crop_rect,
        image=image,
        mask_result=mask_result,
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=resample_path(ordered_path.xy) if ordered_path is not None else None,
    )


def compare_extracted_letters(
    reference: ExtractedLetter,
    sample: ExtractedLetter,
    *,
    identity: ComparisonIdentity,
    smoothing_window: int = 11,
    smoothing_polynomial_order: int = 2,
) -> ComparisonResult:
    """Compare extracted glyphs, using trajectory metrics when both paths exist."""
    alignment = None
    warnings = []
    if reference.resampled_xy is not None and sample.resampled_xy is not None:
        alignment = align_ordered_paths(reference.resampled_xy, sample.resampled_xy)
        warnings.extend(
            f"{side} skeleton has {extracted.diagnostics.topology.endpoint_count} endpoints; "
            "start and end are inferred from reference matching."
            for side, extracted in (("Reference", reference), ("Sample", sample))
            if extracted.diagnostics.topology.endpoint_count != 2
        )
    else:
        unavailable = [
            side.lower()
            for side, extracted in (("Reference", reference), ("Sample", sample))
            if extracted.resampled_xy is None
        ]
        warnings.append(
            "Ordered-path trajectory metrics are unavailable for "
            + " and ".join(unavailable)
            + "; raster and topology comparison remain available."
        )
    return ComparisonResult(
        letter=identity.label,
        reference=reference,
        sample=sample,
        alignment=alignment,
        smoothing_window=smoothing_window,
        smoothing_polynomial_order=smoothing_polynomial_order,
        warnings=warnings,
        identity=identity,
    )


def compare_letter_images(
    reference_path: str | Path,
    sample_path: str | Path,
    *,
    letter: str,
    reference_crop: CropRect | Sequence[int],
    sample_crop: CropRect | Sequence[int],
    output_dir: str | Path,
    reference_ink_mode: InkMode = "auto",
    sample_ink_mode: InkMode = "auto",
    smoothing_window: int = 11,
    smoothing_polynomial_order: int = 2,
) -> WrittenComparisonReport:
    """Compare two manually cropped images of the same known letter."""
    if len(letter) != 1:
        raise ValueError("letter label must contain exactly one character")
    reference = extract_letter(reference_path, reference_crop, ink_mode=reference_ink_mode)
    sample = extract_letter(sample_path, sample_crop, ink_mode=sample_ink_mode)
    result = compare_extracted_letters(
        reference,
        sample,
        identity=ComparisonIdentity(
            sample_id=Path(sample_path).stem or "sample",
            reference_id=Path(reference_path).stem or "reference",
            label=letter,
        ),
        smoothing_window=smoothing_window,
        smoothing_polynomial_order=smoothing_polynomial_order,
    )
    return write_comparison_report(result, output_dir)
