"""Tools for comparing manually selected letter crops."""
from .api import compare_letter_images, extract_letter
from .crop import coerce_crop_rect, crop_image, load_crop
from .geometry import DifferentialGeometry, differential_geometry
from .mask import extract_ink_mask
from .metrics import ComparisonAnalysis, analyze_comparison
from .models import (
    AlignmentResult,
    ComparisonResult,
    CropRect,
    ExtractedLetter,
    InkMaskResult,
    InkMode,
    LetterImageInput,
    OrderedPath,
    SkeletonDiagnostics,
    TopologySummary,
)
from .path import (
    PathExtractionError,
    align_ordered_paths,
    fit_similarity_transform,
    recover_ordered_path,
    resample_path,
)
from .report import WrittenComparisonReport, write_comparison_report
from .skeleton import analyze_skeleton

__all__ = [
    "AlignmentResult",
    "ComparisonAnalysis",
    "ComparisonResult",
    "compare_letter_images",
    "CropRect",
    "DifferentialGeometry",
    "ExtractedLetter",
    "extract_letter",
    "InkMaskResult",
    "InkMode",
    "LetterImageInput",
    "OrderedPath",
    "PathExtractionError",
    "SkeletonDiagnostics",
    "TopologySummary",
    "WrittenComparisonReport",
    "align_ordered_paths",
    "analyze_comparison",
    "analyze_skeleton",
    "coerce_crop_rect",
    "crop_image",
    "differential_geometry",
    "extract_ink_mask",
    "fit_similarity_transform",
    "load_crop",
    "recover_ordered_path",
    "resample_path",
    "write_comparison_report",
]
