from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import WritingModel
from .raster import compare_text_to_image


@dataclass(frozen=True)
class JoinFit:
    text: str
    gap_adjust: float
    tension: float
    score: float
    precision: float
    recall: float


def fit_join(
    model: WritingModel,
    text: str,
    reference: str | Path,
    *,
    gap_min: int = -80,
    gap_max: int = 80,
    gap_step: int = 20,
    tensions: tuple[int, ...] = (30, 60, 90, 120),
) -> JoinFit:
    if len(text) != 2:
        raise ValueError("MVP join fitting requires exactly two characters")

    best: JoinFit | None = None
    for gap in range(gap_min, gap_max + 1, gap_step):
        for tension in tensions:
            metrics = compare_text_to_image(
                model,
                text,
                reference,
                gap_override=gap,
                tension_override=tension,
            )
            candidate = JoinFit(
                text=text,
                gap_adjust=float(gap),
                tension=float(tension),
                score=metrics["iou"],
                precision=metrics["precision"],
                recall=metrics["recall"],
            )
            if best is None or candidate.score > best.score:
                best = candidate
    assert best is not None
    return best
