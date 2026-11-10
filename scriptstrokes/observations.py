"""Release-safe source observations, independent of private scan files."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import StrictModel


class SourceImage(StrictModel):
    source_id: str
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    rights_status: str


class Observation(StrictModel):
    id: str
    kind: Literal["glyph-isolated", "component-family", "movement", "connection"]
    label: str
    unicode: list[str] = Field(default_factory=list)
    source_id: str
    source_bbox: tuple[int, int, int, int]
    crop_locator: str | None = None
    label_source: str
    review_status: Literal["pending", "accepted", "uncertain", "rejected"]
    fitting_status: Literal["pending", "usable", "needs-isolation", "excluded"] = "pending"
    notes: str | None = None

    @model_validator(mode="after")
    def validate_bbox(self) -> Observation:
        x0, y0, x1, y1 = self.source_bbox
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("source_bbox must have nonnegative origin and positive dimensions")
        return self


class ObservationManifest(StrictModel):
    format: Literal["scriptstrokes-observations"] = "scriptstrokes-observations"
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
