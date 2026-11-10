"""Joint component fitting with independently reported hold-out glyphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .models import ComponentPart, PathStrokeRecipe, WritingModel
from .raster import compare_masks, render_text_mask, threshold_image


@dataclass(frozen=True)
class ComponentFitResult:
    model: WritingModel
    training_scores: dict[str, float]
    holdout_scores: dict[str, float]
    candidate_index: int


def fit_component_candidates(
    model: WritingModel,
    component_id: str,
    candidates: list[str],
    references: dict[str, str | Path],
    *,
    training: list[str],
    holdout: list[str],
    scale_x_values: tuple[float, ...] = (0.8, 1.0, 1.1, 1.25, 1.5),
    scale_y_values: tuple[float, ...] = (0.9, 1.0, 1.1),
    shear_values: tuple[float, ...] = (-0.1, 0.0, 0.1),
) -> ComponentFitResult:
    """Choose shared geometry using aggregate training loss, never hold-out loss.

    Component-instance scale and shear are fitted jointly per training glyph;
    raw paths, taps and advance widths remain unchanged. Hold-outs retain their
    predeclared transforms. The score is normalized raster IoU; anchor-aware
    and visual review remain separate gates.
    """
    if not training or not candidates:
        raise ValueError("joint fitting requires training members and path candidates")
    if set(training) & set(holdout):
        raise ValueError("training and hold-out members must be disjoint")
    members = training + holdout
    if len(set(members)) != len(members) or set(members) != set(references):
        raise ValueError("each reference must belong to exactly one declared split")
    if any(character not in model.glyphs for character in members):
        raise ValueError("fit members must be modeled glyphs")
    for character in members:
        if not any(
            isinstance(stroke, PathStrokeRecipe)
            and any(
                isinstance(part, ComponentPart) and part.component == component_id
                for part in stroke.parts
            )
            for form in model.glyphs[character].forms.values()
            for stroke in form.strokes
        ):
            raise ValueError(f"{character!r} does not use component {component_id!r}")
    masks = {}
    for character, path in references.items():
        with Image.open(path) as image:
            masks[character] = threshold_image(image)

    def score(candidate_model, character):
        generated = threshold_image(render_text_mask(candidate_model, character, scale=0.25))
        return compare_masks(masks[character], generated)["iou"]

    best_model = None
    best_scores = {}
    best_mean = -1.0
    best_index = 0
    for index, path in enumerate(candidates):
        candidate_model = model.model_copy(deep=True)
        candidate_model.components[component_id].d = path
        scores = {}
        for character in training:
            original = candidate_model.glyphs[character].model_copy(deep=True)
            best_glyph = original
            best_score = -1.0
            for scale_x in scale_x_values:
                for scale_y in scale_y_values:
                    for shear in shear_values:
                        glyph = original.model_copy(deep=True)
                        for form in glyph.forms.values():
                            for stroke in form.strokes:
                                if isinstance(stroke, PathStrokeRecipe):
                                    for part in stroke.parts:
                                        if (
                                            isinstance(part, ComponentPart)
                                            and part.component == component_id
                                        ):
                                            transform = part.transform
                                            transform.scale_x *= scale_x
                                            transform.dx = (
                                                transform.dx * scale_x + shear * transform.dy
                                            )
                                            transform.scale_y *= scale_y
                                            transform.dy *= scale_y
                                            transform.shear_x = (
                                                transform.shear_x * scale_x + shear
                                            ) / scale_y
                        candidate_model.glyphs[character] = glyph
                        value = score(candidate_model, character)
                        if value > best_score:
                            best_glyph, best_score = glyph, value
            candidate_model.glyphs[character] = best_glyph
            scores[character] = best_score
        mean = float(np.mean(list(scores.values())))
        if mean > best_mean:
            best_model, best_scores, best_mean, best_index = candidate_model, scores, mean, index
    assert best_model is not None
    return ComponentFitResult(
        best_model,
        best_scores,
        {character: score(best_model, character) for character in holdout},
        best_index,
    )
