"""Region layout schema for dataset extraction (generalization of PageLayout v1).

``PageLayout`` v1 (declared alphabet rows) stays in :mod:`inkgrammar.models`.
This module adds the region-based layout schema used by the ``region-layout-v1``
and ``annotated-region-v1`` extractors for the component-family, stroke-order,
and connection pages. All coordinates are normalized to source image
dimensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from ..models import StrictModel

NormalizedBBox = tuple[float, float, float, float]

#: Subregion names with reserved meaning for ``annotated-region-v1``.
INK_SUBREGION = "ink"
ANNOTATION_SUBREGION_PREFIX = "annotation"


def _validate_bbox(value: NormalizedBBox) -> NormalizedBBox:
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox must be normalized fractions with positive dimensions in [0, 1]")
    return value


class LayoutRegion(StrictModel):
    """One declared region of a source page.

    ``subregions`` hold named normalized boxes inside the region. By
    convention ``annotated-region-v1`` treats the subregion named ``ink`` as
    the target ink region and subregions named ``annotation*`` as printed
    annotation marks that must never enter the target mask.
    """

    id: str
    kind: str
    bbox: NormalizedBBox | None = None
    label: str | None = None
    sequence: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    subregions: dict[str, NormalizedBBox] = Field(default_factory=dict)
    category_hint: str | None = None
    notes: str | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: NormalizedBBox | None) -> NormalizedBBox | None:
        return _validate_bbox(value) if value is not None else value

    @field_validator("subregions")
    @classmethod
    def validate_subregions(cls, value: dict[str, NormalizedBBox]) -> dict[str, NormalizedBBox]:
        for bbox in value.values():
            _validate_bbox(bbox)
        return value


class RegionLayout(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    description: str | None = None
    regions: list[LayoutRegion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_region_ids(self) -> RegionLayout:
        ids = [region.id for region in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate layout region id")
        return self


def load_region_layout(path: str | Path) -> RegionLayout:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RegionLayout.model_validate(data)
