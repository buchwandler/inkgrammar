"""Evidence-first dataset package: source facts, observations, reviewed annotations.

The four evidence layers are kept separate by design:

- layer A (source evidence) and layer B (extracted observations) live in
  :mod:`inkgrammar.dataset.models`;
- layer C (reviewed interpretation) lives in :mod:`inkgrammar.dataset.models`
  as annotation schemas;
- layer D (fitted model parameters) lives in :mod:`inkgrammar.fitting.models`
  and never overwrites evidence.
"""

from .models import (
    ConnectionAnnotation,
    FamilyAssignment,
    Observation,
    ObservationAnnotation,
    ObservationExtraction,
    ObservationManifest,
    ObservationMetrics,
    ObservationRelations,
    SourceImage,
    StrokeAnnotation,
    migrate_manifest_kinds,
)

__all__ = [
    "ConnectionAnnotation",
    "FamilyAssignment",
    "Observation",
    "ObservationAnnotation",
    "ObservationExtraction",
    "ObservationManifest",
    "ObservationMetrics",
    "ObservationRelations",
    "SourceImage",
    "StrokeAnnotation",
    "migrate_manifest_kinds",
]
