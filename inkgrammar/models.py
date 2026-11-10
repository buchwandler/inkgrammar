from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VerificationStatus = Literal[
    "synthetic",
    "inferred",
    "source-backed",
    "reviewed",
    "verified",
]
StrokeRole = Literal[
    "body",
    "connector",
    "crossbar",
    "dot",
    "diacritic",
    "flourish",
    "other",
]
FormName = Literal["isolated", "initial", "medial", "final", "default"]

ConnectionMode = Literal["continuous-bridge", "direct", "attached", "none"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Metrics(StrictModel):
    units_per_em: int = Field(gt=0, le=16384)
    baseline: float = 0
    x_height: float
    cap_height: float | None = None
    ascender: float
    descender: float
    slant_degrees: float = Field(default=0, ge=-45, le=45)
    default_stroke_width: float = Field(gt=0)


class SourceRecord(StrictModel):
    id: str
    title: str
    kind: Literal[
        "publication",
        "font",
        "website",
        "archive",
        "project-original",
        "scan",
        "other",
    ]
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=3000)
    url: str | None = None
    locator: str | None = None
    rights: str
    license: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("source id must be lowercase kebab-case")
        return value


class Provenance(StrictModel):
    status: VerificationStatus
    source_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class Component(StrictModel):
    d: str = Field(min_length=1)
    description: str | None = None
    provenance: Provenance
    unused_reason: str | None = None


class Transform(StrictModel):
    scale_x: float = Field(default=1.0, gt=0)
    scale_y: float = Field(default=1.0, gt=0)
    shear_x: float = Field(default=0.0, ge=-2.0, le=2.0)
    dx: float = 0.0
    dy: float = 0.0


class ComponentPart(StrictModel):
    component: str
    attach: bool = True
    transform: Transform = Field(default_factory=Transform)


class RawPathPart(StrictModel):
    d: str = Field(min_length=1)
    attach: bool = True
    transform: Transform = Field(default_factory=Transform)


StrokePart = Annotated[
    ComponentPart | RawPathPart,
    Field(discriminator=None),
]


class PathStrokeRecipe(StrictModel):
    id: str
    role: StrokeRole = "body"
    parts: list[ComponentPart | RawPathPart] = Field(min_length=1)


class TapStrokeRecipe(StrictModel):
    id: str
    role: StrokeRole = "dot"
    at: tuple[float, float]


StrokeRecipe = Annotated[
    PathStrokeRecipe | TapStrokeRecipe,
    Field(discriminator=None),
]


class ConnectionEndpoint(StrictModel):
    stroke: str
    endpoint: Literal["start", "end"]
    class_: str = Field(alias="class")


class FormConnections(StrictModel):
    entry: ConnectionEndpoint | None = None
    exit: ConnectionEndpoint | None = None


class GlyphForm(StrictModel):
    advance_width: float = Field(gt=0)
    strokes: list[PathStrokeRecipe | TapStrokeRecipe] = Field(min_length=1)
    connections: FormConnections = Field(default_factory=FormConnections)


class GlyphModel(StrictModel):
    character: str = Field(min_length=1)
    unicode: list[str] = Field(min_length=1)
    forms: dict[FormName, GlyphForm]
    provenance: Provenance
    required_forms: list[FormName] = Field(default_factory=list)

    @field_validator("unicode")
    @classmethod
    def validate_unicode(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"U\+[0-9A-F]{4,6}", value):
                raise ValueError(f"invalid Unicode code point: {value}")
        return values

    @model_validator(mode="after")
    def require_usable_form(self) -> GlyphModel:
        if "default" not in self.forms and "isolated" not in self.forms:
            raise ValueError("glyph needs at least a default or isolated form")
        return self


class ConnectionRule(StrictModel):
    id: str
    from_class: str
    to_class: str
    gap: float = 0
    tension: float = Field(default=60, ge=0, le=1000)
    continuous: bool = True
    mode: ConnectionMode | None = None

    @property
    def effective_mode(self) -> ConnectionMode:
        if self.mode is not None:
            return self.mode
        return "continuous-bridge" if self.continuous else "attached"


class PairOverride(StrictModel):
    gap_adjust: float = 0
    tension: float | None = Field(default=None, ge=0, le=1000)
    evidence: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0, le=1)


class WritingModel(StrictModel):
    schema_version: Literal[2]
    id: str
    name: dict[str, str]
    script: str
    languages: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    status: Literal["experimental", "documented", "verified"]
    license: str
    description: dict[str, str] = Field(default_factory=dict)
    metrics: Metrics
    sources: list[SourceRecord] = Field(default_factory=list)
    components: dict[str, Component] = Field(default_factory=dict)
    connection_rules: list[ConnectionRule] = Field(default_factory=list)
    pair_overrides: dict[str, PairOverride] = Field(default_factory=dict)
    glyphs: dict[str, GlyphModel]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("model id must be lowercase kebab-case")
        return value

    @field_validator("script")
    @classmethod
    def validate_script(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Z][a-z]{3}", value):
            raise ValueError("script must look like an ISO 15924 code, e.g. Latn")
        return value


class LayoutRow(StrictModel):
    id: str
    y: tuple[float, float]
    labels: list[str] = Field(min_length=1)
    x: tuple[float, float] = (0.03, 0.97)
    centers: list[float] | None = None

    @field_validator("y", "x")
    @classmethod
    def validate_fraction_pair(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not (0 <= value[0] < value[1] <= 1):
            raise ValueError("layout ranges must be normalized fractions in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_centers(self) -> LayoutRow:
        if self.centers is not None:
            if len(self.centers) != len(self.labels):
                raise ValueError("centers must contain one normalized x position per label")
            if any(not (0 <= value <= 1) for value in self.centers):
                raise ValueError("centers must be normalized fractions in [0, 1]")
            if self.centers != sorted(self.centers):
                raise ValueError("centers must be strictly ordered left-to-right")
        return self


class PageLayout(StrictModel):
    schema_version: Literal[1]
    id: str
    description: str | None = None
    rows: list[LayoutRow] = Field(min_length=1)
