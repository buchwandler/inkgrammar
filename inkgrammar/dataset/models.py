"""Dataset evidence schemas (layers A-C of the data-first pipeline).

Layer A, source evidence, records immutable facts about a source image and
never contains model estimates. Layer B, extracted observations, records
deterministic extractions from a source and optional extraction, metric, and
relation metadata. Layer C, reviewed interpretation, records human-reviewed
semantic claims with per-field provenance.

Fitted model parameters (layer D) intentionally live in
:mod:`inkgrammar.fitting.models`; evidence records must never be overwritten
by fitting results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from ..models import StrictModel

AnnotationEvidence = Literal[
    "source-explicit",
    "source-layout",
    "source-family-example",
    "source-connection-example",
    "inferred-reviewed",
    "automatic-candidate",
]

ObservationKind = Literal[
    "glyph-isolated",
    "glyph-contextual",
    "component-family",
    "movement-demonstration",
    "stroke-order-demonstration",
    "connection-pair",
    "connection-category",
    "word",
    "sequence",
]

#: Legacy kind names still accepted on load; files are only rewritten by the
#: explicit migration helper :func:`migrate_manifest_kinds`.
KIND_ALIASES: dict[str, str] = {
    "movement": "movement-demonstration",
    "connection": "connection-pair",
}

ReviewStatus = Literal["pending", "accepted", "uncertain", "rejected"]
FittingStatus = Literal["pending", "usable", "needs-isolation", "excluded", "usable-reviewed"]
IsolationStatus = Literal["pending", "usable", "needs-isolation", "excluded", "usable-reviewed"]
MetricStatus = Literal["missing", "automatic-candidate", "reviewed"]
Confidence = Literal["high", "medium", "low"]


class SourceImage(StrictModel):
    """Layer A: immutable source facts. No model estimates allowed here."""

    source_id: str
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    rights_status: str


class ObservationExtraction(StrictModel):
    """How a crop was produced. Confidence is not truth."""

    extractor: str
    extractor_version: int = Field(ge=1)
    source_bbox_policy: str
    threshold_policy: str | None = None
    cleanup_policy: str | None = None
    layout_id: str | None = None
    layout_region_id: str | None = None
    detection_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)


class ObservationMetrics(StrictModel):
    """Normalized observation metrics in a source-independent coordinate frame.

    Coordinates are crop- or source-normalized; pixel and font units must never
    share the same field names.
    """

    baseline_y: float | None = None
    x_height_y: float | None = None
    cap_height_y: float | None = None
    ascender_y: float | None = None
    descender_y: float | None = None
    normalized_bbox: tuple[float, float, float, float] | None = None
    metric_status: MetricStatus = "missing"


class ObservationRelations(StrictModel):
    """Structural relations declared or reviewed for an observation."""

    family_ids: list[str] = Field(default_factory=list)
    member_labels: list[str] = Field(default_factory=list)
    related_observation_ids: list[str] = Field(default_factory=list)
    connection_category: str | None = None
    left_label: str | None = None
    right_label: str | None = None
    sequence_text: str | None = None


class Observation(StrictModel):
    """Layer B: a deterministic extraction from a source.

    It answers "what region of the source are we talking about", never "how was
    it written"; stroke semantics live in layer C annotations.
    """

    id: str
    kind: ObservationKind
    label: str
    unicode: list[str] = Field(default_factory=list)
    source_id: str
    source_bbox: tuple[int, int, int, int]
    crop_locator: str | None = None
    label_source: str
    review_status: ReviewStatus
    fitting_status: FittingStatus = "pending"
    isolation_status: IsolationStatus | None = None
    extraction: ObservationExtraction | None = None
    metrics: ObservationMetrics | None = None
    relations: ObservationRelations | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_kind_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("kind"), str):
            kind = data["kind"]
            if kind in KIND_ALIASES:
                data = {**data, "kind": KIND_ALIASES[kind]}
        return data

    @model_validator(mode="after")
    def validate_bbox(self) -> Observation:
        x0, y0, x1, y1 = self.source_bbox
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("source_bbox must have nonnegative origin and positive dimensions")
        return self


class ObservationManifest(StrictModel):
    format: Literal["inkgrammar-observations"] = "inkgrammar-observations"
    format_version: Literal[1] = 1
    model_id: str
    sources: list[SourceImage]
    items: list[Observation]

    @model_validator(mode="after")
    def validate_references(self) -> ObservationManifest:
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("duplicate source id")
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate observation id")
        sources = {source.source_id: source for source in self.sources}
        for item in self.items:
            if item.source_id not in sources:
                raise ValueError(f"unknown source: {item.source_id}")
            source = sources[item.source_id]
            if item.source_bbox[2] > source.width_px or item.source_bbox[3] > source.height_px:
                raise ValueError(f"observation outside source: {item.id}")
        return self


class ReviewedScalar(StrictModel):
    value: float
    evidence: AnnotationEvidence


class ReviewedInt(StrictModel):
    value: int
    evidence: AnnotationEvidence


class NormalizedPoint(StrictModel):
    x: float
    y: float


class ReviewedPoint(StrictModel):
    point: NormalizedPoint
    evidence: AnnotationEvidence
    confidence: Confidence


class ReviewedRegion(StrictModel):
    """Region-based annotation for ambiguous start/end evidence."""

    bbox: tuple[float, float, float, float]
    evidence: AnnotationEvidence


class StrokeAnnotation(StrictModel):
    id: str
    order: int = Field(ge=0)
    pen_state: Literal["down", "tap"]
    start: ReviewedPoint | ReviewedRegion | None = None
    end: ReviewedPoint | ReviewedRegion | None = None
    direction_status: Literal["source-explicit", "inferred-reviewed", "unknown"]
    family_hint: str | None = None


class FamilyAssignment(StrictModel):
    family_id: str
    evidence: AnnotationEvidence
    notes: str | None = None


class ConnectionAnnotation(StrictModel):
    left_label: str | None = None
    right_label: str | None = None
    category_id: str | None = None
    source_label: str | None = None
    notes: str | None = None


class ObservationAnnotation(StrictModel):
    """Layer C: reviewed semantic claims derived from a source observation.

    Every semantic field carries provenance; inferred values must never be
    marked as source-verified merely because they fit the scan.
    """

    observation_id: str
    annotation_version: int = Field(ge=1)
    review_status: Literal["candidate", "reviewed", "uncertain", "rejected"]
    baseline: ReviewedScalar | None = None
    x_height: ReviewedScalar | None = None
    stroke_count: ReviewedInt | None = None
    strokes: list[StrokeAnnotation] = Field(default_factory=list)
    family_assignments: list[FamilyAssignment] = Field(default_factory=list)
    connection_annotation: ConnectionAnnotation | None = None
    reviewer_notes: str | None = None


def migrate_manifest_kinds(data: dict) -> dict:
    """Return a manifest dict with legacy observation kinds rewritten.

    Load-time alias acceptance never rewrites committed files; this explicit
    helper is the only place that produces migrated manifest content.
    """
    migrated = {**data}
    items = []
    for item in data.get("items", []):
        kind = item.get("kind")
        if isinstance(kind, str) and kind in KIND_ALIASES:
            item = {**item, "kind": KIND_ALIASES[kind]}
        items.append(item)
    migrated["items"] = items
    return migrated
