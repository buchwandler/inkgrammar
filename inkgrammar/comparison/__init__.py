"""Tools for comparing manually selected letter crops."""
from .api import compare_extracted_letters, compare_letter_images, extract_letter
from .batch import compare_letter_page
from .batch_report import BatchComparisonReport, BatchSampleResult
from .crop import coerce_crop_rect, crop_image, load_crop
from .descriptors import GlyphDescriptor, canonicalize_mask, glyph_descriptor
from .geometry import DifferentialGeometry, differential_geometry
from .mask import extract_ink_mask
from .matching import (
    MatchScore,
    ReferenceIndex,
    build_reference_index,
    coarse_match_cost,
    match_status,
    rank_reference_candidates,
    refine_match,
    refine_top_k_matches,
)
from .metrics import ComparisonAnalysis, analyze_comparison
from .models import (
    AlignmentResult,
    ComparisonIdentity,
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
from .segmentation import (
    PageSegmentation,
    ReferenceGallery,
    RowBand,
    SegmentationError,
    SegmentedGlyph,
    detect_ink_rows,
    extract_page_ink_mask,
    segment_reference_gallery,
    segment_sample_glyphs,
)
from .skeleton import analyze_skeleton

__all__ = [
    "AlignmentResult",
    "ComparisonAnalysis",
    "ComparisonResult",
    "compare_extracted_letters",
    "ComparisonIdentity",
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
    "PageSegmentation",
    "ReferenceGallery",
    "RowBand",
    "SegmentationError",
    "SegmentedGlyph",
    "detect_ink_rows",
    "extract_page_ink_mask",
    "segment_reference_gallery",
    "segment_sample_glyphs",
    "GlyphDescriptor",
    "MatchScore",
    "ReferenceIndex",
    "build_reference_index",
    "canonicalize_mask",
    "coarse_match_cost",
    "glyph_descriptor",
    "match_status",
    "rank_reference_candidates",
    "refine_match",
    "refine_top_k_matches",
    "BatchComparisonReport",
    "BatchSampleResult",
    "compare_letter_page",
]
