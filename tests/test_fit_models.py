import numpy as np
import pytest
from pydantic import ValidationError

from inkgrammar.fitting.models import (
    DuctusModel,
    FitCorpus,
    FitSample,
    PrimitiveTemplate,
)


def primitive_payload():
    return {
        "id": "sas-underturn-01",
        "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
        "segments": [
            {"c1": [0.1, 0.4], "c2": [0.3, 0.8], "end": [0.5, 1.0]},
            {"c1": [0.7, 1.0], "c2": [0.9, 0.2], "end": [1.0, 0.0]},
        ],
        "free_parameters": ["segments.0.c1.x", "segments.0.c1.y"],
        "locked_parameters": ["frame.start", "segments.1.end"],
        "bounds": {
            "segments.0.c1.x": [-0.1, 0.3],
            "segments.0.c1.y": [0.1, 0.8],
        },
        "provenance": {"construction": "source-backed", "trajectory": "inferred"},
    }


def test_primitive_template_validates_bounded_typed_segments():
    primitive = PrimitiveTemplate.model_validate(primitive_payload())

    assert primitive.kind == "cubic-chain"
    assert primitive.segments[0].c1 == (0.1, 0.4)
    assert primitive.bounds["segments.0.c1.x"] == (-0.1, 0.3)


def test_primitive_template_requires_a_bound_for_each_free_coordinate():
    payload = primitive_payload()
    del payload["bounds"]["segments.0.c1.y"]

    with pytest.raises(ValidationError, match="every free parameter"):
        PrimitiveTemplate.model_validate(payload)


def test_primitive_template_rejects_bad_paths_and_out_of_range_initials():
    payload = primitive_payload()
    payload["free_parameters"][0] = "segments.9.c1.x"
    del payload["bounds"]["segments.0.c1.x"]
    payload["bounds"]["segments.9.c1.x"] = [-0.1, 0.3]

    with pytest.raises(ValidationError, match="unknown primitive parameter path"):
        PrimitiveTemplate.model_validate(payload)

    payload = primitive_payload()
    payload["bounds"]["segments.0.c1.x"] = [0.2, 0.3]
    with pytest.raises(ValidationError, match="initial value is outside bounds"):
        PrimitiveTemplate.model_validate(payload)


def test_fitting_models_are_strict_and_validate_corpus_ids():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DuctusModel.model_validate(
            {
                "units_per_em": 1000,
                "baseline": 0,
                "x_height": 360,
                "cap_height": 650,
                "ascender": 650,
                "descender": -300,
                "slant_degrees": 20,
                "stroke_width": 55,
                "unmodeled": 1,
            }
        )

    sample = FitSample(
        id="sample-a",
        kind="glyph",
        text="a",
        observation_id="obs-a",
        split="fit",
    )
    with pytest.raises(ValidationError, match="duplicate fit sample id"):
        FitCorpus(samples=[sample, sample])

    with pytest.raises(ValidationError, match="active instance IDs must be unique"):
        FitSample.model_validate(sample.model_dump() | {"active_instances": ["part-a", "part-a"]})


def test_occurrence_transform_composes_nominal_values_with_bounded_deltas():
    from inkgrammar.fitting.models import (
        OccurrenceDelta,
        OccurrenceFitBounds,
        OccurrenceFitState,
        TopologyTransform,
        compose_occurrence_transform,
    )

    nominal = TopologyTransform(scale_x=30, scale_y=32, shear_x=0.1, dx=-3, dy=2)
    delta = OccurrenceDelta(dx=1.5, dy=-2.0, log_scale_x=0.05, log_scale_y=-0.05, shear_x=0.02)
    state = OccurrenceFitState(
        instance_id="loop",
        nominal_transform=nominal,
        delta=delta,
        bounds=OccurrenceFitBounds(),
        free_parameters=["dx", "dy", "log_scale_x", "log_scale_y", "shear_x"],
    )

    effective = compose_occurrence_transform(state.nominal_transform, state.delta)

    assert effective.dx == -1.5
    assert effective.dy == 0.0
    assert effective.scale_x == pytest.approx(30 * np.exp(0.05))
    assert effective.scale_y == pytest.approx(32 * np.exp(-0.05))
    assert effective.shear_x == pytest.approx(0.12)


def test_occurrence_fit_state_rejects_delta_outside_configured_bounds():
    from inkgrammar.fitting.models import OccurrenceFitState, TopologyTransform

    with pytest.raises(ValidationError, match="outside bounds"):
        OccurrenceFitState(
            instance_id="loop",
            nominal_transform=TopologyTransform(),
            delta={"dx": 5.0},
        )
