import pytest
from PIL import Image

from inkgrammar.component_fit import fit_component_candidates
from inkgrammar.io import load_model
from inkgrammar.raster import render_text_mask


def references_for(model, tmp_path):
    result = {}
    for character in ("i", "u"):
        path = tmp_path / f"{character}.png"
        render_text_mask(model, character, scale=0.25).save(path)
        result[character] = path
    return result


def test_joint_fit_does_not_use_holdout_to_select_or_mutate_input(tmp_path):
    model = load_model("data/models/demo-looped.yaml")
    original = model.model_dump()
    references = references_for(model, tmp_path)
    candidates = ["M 0 0 L 1 1", model.components["underturn"].d]
    kwargs = dict(
        training=["i"],
        holdout=["u"],
        scale_x_values=(1.0,),
        scale_y_values=(1.0,),
        shear_values=(0.0,),
    )
    result = fit_component_candidates(model, "underturn", candidates, references, **kwargs)
    assert result.candidate_index == 1
    assert result.training_scores["i"] == 1
    assert result.holdout_scores["u"] == 1
    Image.new("L", (20, 20), 255).save(references["u"])
    altered = fit_component_candidates(model, "underturn", candidates, references, **kwargs)
    assert altered.candidate_index == result.candidate_index
    assert altered.training_scores == result.training_scores
    assert altered.holdout_scores["u"] == 0
    assert model.model_dump() == original


def test_joint_fit_requires_disjoint_complete_membership(tmp_path):
    model = load_model("data/models/demo-looped.yaml")
    references = references_for(model, tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        fit_component_candidates(
            model,
            "underturn",
            [model.components["underturn"].d],
            references,
            training=["i"],
            holdout=["i", "u"],
        )
    with pytest.raises(ValueError, match="each reference"):
        fit_component_candidates(
            model,
            "underturn",
            [model.components["underturn"].d],
            references,
            training=["i"],
            holdout=[],
        )
