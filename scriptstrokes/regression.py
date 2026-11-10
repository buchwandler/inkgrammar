"""Release-safe regression fixture declarations and recorded measurements."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import StrictModel

RegressionGroup = Literal[
    "isolated-uppercase",
    "isolated-lowercase",
    "digits",
    "stroke-order-backed",
    "connection-baseline",
    "connection-midline",
    "connection-attached",
    "numbered-connection-examples",
]


class RegressionFixture(StrictModel):
    id: str
    group: RegressionGroup
    text: str = Field(min_length=1)
    observation_ids: list[str] = Field(default_factory=list)
    reference_locator: str | None = None
    split: Literal["fit", "hold-out", "public-snapshot"]
    reference_anchor: tuple[float, float] | None = None
    generated_anchor: tuple[float, float] | None = None
    generated_pixels_per_reference_pixel: float = Field(default=1, gt=0)
    minimum_iou: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = None


class RegressionResult(StrictModel):
    fixture_id: str
    status: Literal["pass", "fail", "not-run"]
    metrics: dict[str, float] = Field(default_factory=dict)
    visual_review: Literal["pending", "accepted", "rejected", "not-applicable"] = "pending"
    notes: str | None = None


class RegressionManifest(StrictModel):
    format: Literal["scriptstrokes-regressions"] = "scriptstrokes-regressions"
    format_version: Literal[1] = 1
    model_id: str
    fixtures: list[RegressionFixture]
    results: list[RegressionResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids(self) -> RegressionManifest:
        ids = [fixture.id for fixture in self.fixtures]
        result_ids = [result.fixture_id for result in self.results]
        if len(set(ids)) != len(ids) or len(set(result_ids)) != len(result_ids):
            raise ValueError("duplicate regression fixture or result")
        if set(result_ids) - set(ids):
            raise ValueError("regression result references unknown fixture")
        return self
