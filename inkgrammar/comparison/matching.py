from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from PIL import Image
from planarcurve.polyline import polyline_length

from .descriptors import GlyphDescriptor, glyph_descriptor
from .geometry import differential_geometry
from .models import ExtractedLetter
from .path import (
    PathExtractionError,
    align_ordered_paths,
    recover_ordered_path,
    resample_path,
)
from .segmentation import ReferenceGallery, SegmentedGlyph
from .skeleton import analyze_skeleton

RASTER_WEIGHT = 1.0
ASPECT_WEIGHT = 0.15
FILL_WEIGHT = 0.10
TOPOLOGY_WEIGHT = 0.15
ROTATION_ANGLES_DEG = (-8, -4, 0, 4, 8)
DEFAULT_TOP_K = 5
MINIMUM_ALTERNATIVES = 3
CENTERLINE_WEIGHT = 0.45
HEADING_WEIGHT = 0.15
TURN_RATE_WEIGHT = 0.15
AMBIGUITY_ABSOLUTE_MARGIN = 0.04
AMBIGUITY_RELATIVE_MARGIN = 0.15


@dataclass(frozen=True)
class ReferenceIndex:
    gallery: ReferenceGallery
    descriptors: tuple[GlyphDescriptor, ...]

    def __post_init__(self) -> None:
        if len(self.gallery.candidates) != len(self.descriptors):
            raise ValueError("reference candidates and descriptors must have matching lengths")


@dataclass(frozen=True)
class MatchScore:
    reference_id: str
    rank: int
    coarse_cost: float
    raster_cost: float
    aspect_cost: float
    fill_cost: float
    topology_cost: float
    refined_cost: float | None = None
    centerline_cost: float | None = None
    heading_cost: float | None = None
    turn_rate_cost: float | None = None

    @property
    def cost(self) -> float:
        return self.refined_cost if self.refined_cost is not None else self.coarse_cost


def build_reference_index(gallery: ReferenceGallery) -> ReferenceIndex:
    """Compute reusable coarse descriptors for every reference candidate."""
    return ReferenceIndex(
        gallery=gallery,
        descriptors=tuple(glyph_descriptor(candidate) for candidate in gallery.candidates),
    )


def _normalized_count_difference(first: int | None, second: int | None) -> float | None:
    if first is None or second is None:
        return None
    return abs(first - second) / max(first, second, 1)


def _raster_cost(first: np.ndarray, second: np.ndarray) -> float:
    best_iou = 0.0
    for angle in ROTATION_ANGLES_DEG:
        if angle:
            rotated = Image.fromarray(second.astype(np.uint8) * 255).rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
                fillcolor=0,
            )
            compared = np.asarray(rotated, dtype=np.uint8) > 0
        else:
            compared = second
        intersection = int(np.logical_and(first, compared).sum())
        union = int(np.logical_or(first, compared).sum())
        best_iou = max(best_iou, intersection / union if union else 1.0)
    return 1.0 - best_iou


def coarse_match_cost(sample: GlyphDescriptor, reference: GlyphDescriptor) -> MatchScore:
    """Compute an internal retrieval rank cost, not a handwriting quality score."""
    raster = _raster_cost(sample.normalized_mask, reference.normalized_mask)
    aspect = min(1.0, abs(np.log(sample.aspect_ratio / reference.aspect_ratio)))
    fill = abs(sample.ink_fill - reference.ink_fill) / max(
        sample.ink_fill, reference.ink_fill, 1e-9
    )
    topology_terms = [
        _normalized_count_difference(sample.component_count, reference.component_count),
        _normalized_count_difference(sample.endpoint_count, reference.endpoint_count),
        _normalized_count_difference(sample.junction_count, reference.junction_count),
        _normalized_count_difference(sample.cycle_count, reference.cycle_count),
    ]
    available_topology = [term for term in topology_terms if term is not None]
    topology = float(np.mean(available_topology)) if available_topology else 0.0
    cost = (
        RASTER_WEIGHT * raster
        + ASPECT_WEIGHT * aspect
        + FILL_WEIGHT * fill
        + TOPOLOGY_WEIGHT * topology
    )
    return MatchScore(
        reference_id=reference.glyph_id,
        rank=0,
        coarse_cost=float(cost),
        raster_cost=float(raster),
        aspect_cost=float(aspect),
        fill_cost=float(fill),
        topology_cost=topology,
    )


def rank_reference_candidates(
    sample: SegmentedGlyph,
    gallery: ReferenceGallery | ReferenceIndex,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[MatchScore]:
    """Return the best coarse shape matches, independently for this sample."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    index = gallery if isinstance(gallery, ReferenceIndex) else build_reference_index(gallery)
    sample_descriptor = glyph_descriptor(sample)
    ranked = sorted(
        (
            coarse_match_cost(sample_descriptor, descriptor)
            for descriptor in index.descriptors
        ),
        key=lambda score: (score.coarse_cost, score.reference_id),
    )
    return [replace(score, rank=rank) for rank, score in enumerate(ranked[:top_k], start=1)]


def refine_match(
    sample: SegmentedGlyph,
    reference: SegmentedGlyph,
    coarse_score: MatchScore,
    *,
    reference_extracted: ExtractedLetter | None = None,
) -> MatchScore:
    """Add optional ordered-path geometry evidence without rejecting raster matches."""
    try:
        if reference_extracted is None:
            reference_diagnostics = analyze_skeleton(reference.mask)
            reference_path = recover_ordered_path(reference_diagnostics)
            reference_xy = resample_path(reference_path.xy)
        else:
            reference_path = reference_extracted.ordered_path
            reference_xy = reference_extracted.resampled_xy
            if reference_path is None or reference_xy is None:
                return replace(coarse_score, refined_cost=coarse_score.coarse_cost)
        sample_diagnostics = analyze_skeleton(sample.mask)
        sample_path = recover_ordered_path(sample_diagnostics)
        sample_xy = resample_path(sample_path.xy)
        alignment = align_ordered_paths(reference_xy, sample_xy)
        reference_geometry = differential_geometry(reference_xy)
        sample_geometry = differential_geometry(alignment.aligned_sample_xy)
    except (PathExtractionError, ValueError):
        return replace(coarse_score, refined_cost=coarse_score.coarse_cost)

    reference_length = float(polyline_length(reference_path.xy))
    if reference_length <= 0.0:
        return replace(coarse_score, refined_cost=coarse_score.coarse_cost)
    centerline = float(
        np.mean(
            np.linalg.norm(
                alignment.aligned_sample_xy - reference_xy,
                axis=1,
            )
        )
        / reference_length
    )
    heading_delta = sample_geometry.heading_rad - reference_geometry.heading_rad
    heading = float(
        np.mean(np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))) / np.pi
    )
    turn_rate = float(
        np.mean(
            np.abs(
                sample_geometry.turn_rate_rad_per_progress
                - reference_geometry.turn_rate_rad_per_progress
            )
        )
        / (2.0 * np.pi)
    )
    refined = (
        coarse_score.coarse_cost
        + CENTERLINE_WEIGHT * centerline
        + HEADING_WEIGHT * heading
        + TURN_RATE_WEIGHT * turn_rate
    )
    return replace(
        coarse_score,
        refined_cost=float(refined),
        centerline_cost=centerline,
        heading_cost=heading,
        turn_rate_cost=turn_rate,
    )

def refine_top_k_matches(
    sample: SegmentedGlyph,
    gallery: ReferenceGallery,
    ranked: list[MatchScore],
    *,
    reference_extractions: Mapping[str, ExtractedLetter] | None = None,
) -> list[MatchScore]:
    """Refine only the supplied coarse top-K candidates and rerank them."""
    candidates = {candidate.glyph_id: candidate for candidate in gallery.candidates}
    refined = []
    for score in ranked:
        candidate = candidates.get(score.reference_id)
        if candidate is None:
            continue
        extracted = (
            reference_extractions.get(score.reference_id)
            if reference_extractions is not None
            else None
        )
        refined.append(refine_match(sample, candidate, score, reference_extracted=extracted))
    refined.sort(key=lambda score: (score.cost, score.reference_id))
    return [replace(score, rank=rank) for rank, score in enumerate(refined, start=1)]

def match_status(scores: list[MatchScore]) -> Literal["clear", "ambiguous"]:
    """Classify best-match ambiguity from the gap to the second-ranked candidate."""
    if len(scores) < 2:
        return "clear"
    ranked = sorted(scores, key=lambda score: (score.cost, score.reference_id))
    best, second = ranked[:2]
    margin = second.cost - best.cost
    relative_margin = margin / max(abs(best.cost), 1e-9)
    if (
        margin <= AMBIGUITY_ABSOLUTE_MARGIN
        or relative_margin <= AMBIGUITY_RELATIVE_MARGIN
    ):
        return "ambiguous"
    return "clear"
