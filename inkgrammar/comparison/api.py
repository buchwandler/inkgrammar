from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .crop import coerce_crop_rect, load_crop
from .mask import extract_ink_mask
from .models import ComparisonResult, CropRect, ExtractedLetter, InkMode
from .path import align_ordered_paths, recover_ordered_path, resample_path
from .report import WrittenComparisonReport, write_comparison_report
from .skeleton import analyze_skeleton


def extract_letter(
    image_path: str | Path,
    crop: CropRect | Sequence[int],
    *,
    ink_mode: InkMode = "auto",
) -> ExtractedLetter:
    """Crop an image and recover a topology-covering ordered letter path."""
    source_path = Path(image_path)
    crop_rect = coerce_crop_rect(crop)
    image = load_crop(source_path, crop_rect)
    mask_result = extract_ink_mask(image, mode=ink_mode)
    diagnostics = analyze_skeleton(mask_result.mask)
    ordered_path = recover_ordered_path(diagnostics)
    return ExtractedLetter(
        source_path=source_path,
        crop=crop_rect,
        image=image,
        mask_result=mask_result,
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=resample_path(ordered_path.xy),
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
    alignment = align_ordered_paths(reference.resampled_xy, sample.resampled_xy)
    warnings = [
        f"{side} skeleton has {extracted.diagnostics.topology.endpoint_count} endpoints; "
        "start and end are inferred from reference matching."
        for side, extracted in (("Reference", reference), ("Sample", sample))
        if extracted.diagnostics.topology.endpoint_count != 2
    ]
    result = ComparisonResult(
        letter=letter,
        reference=reference,
        sample=sample,
        alignment=alignment,
        smoothing_window=smoothing_window,
        smoothing_polynomial_order=smoothing_polynomial_order,
        warnings=warnings,
    )
    return write_comparison_report(result, output_dir)
