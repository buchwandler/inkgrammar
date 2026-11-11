import numpy as np
import pytest
from pydantic import ValidationError

from inkgrammar.fitting.models import PrimitiveInstance
from inkgrammar.fitting.objective import (
    LocalRegion,
    ObjectiveConfig,
    evaluate_objective,
    soft_mask_loss,
    symmetric_distance_loss,
)


def masks():
    reference = np.zeros((32, 32), dtype=np.float32)
    reference[8:24, 14:18] = 1
    generated = np.zeros_like(reference)
    generated[8:24, 14:18] = 1
    return reference, generated


def test_soft_mask_and_symmetric_distance_losses_are_zero_for_exact_masks():
    reference, generated = masks()

    assert soft_mask_loss(reference, generated) == 0
    assert symmetric_distance_loss(reference, generated) == 0


def test_anchor_and_endpoint_terms_penalize_placement_without_mask_recentering():
    reference, generated = masks()
    config = ObjectiveConfig()

    aligned = evaluate_objective(
        reference,
        generated,
        config=config,
        x_height=16,
        reference_anchor=(10, 20),
        generated_anchor=(10, 20),
        endpoint_pairs=[((0, 0), (0, 0))],
    )
    shifted = evaluate_objective(
        reference,
        generated,
        config=config,
        x_height=16,
        reference_anchor=(10, 20),
        generated_anchor=(14, 24),
        endpoint_pairs=[((0, 0), (4, 4))],
    )

    assert aligned.total == 0
    assert shifted.terms["anchor"] > 0
    assert shifted.terms["endpoint"] > 0


def test_local_region_objective_can_ignore_unrelated_glyph_parts():
    reference = np.zeros((32, 32), dtype=np.float32)
    reference[4:12, 4:10] = 1
    reference[20:28, 22:28] = 1
    generated_whole = reference.copy()
    generated_whole[20:28, 22:28] = 0
    region_mask = np.zeros_like(reference)
    region_mask[3:13, 3:11] = 1
    generated_part = np.zeros_like(reference)
    generated_part[4:12, 4:10] = 1

    result = evaluate_objective(
        reference,
        generated_whole,
        config=ObjectiveConfig(),
        x_height=16,
        local_regions=[LocalRegion(region_mask, generated_part)],
    )

    assert result.terms["soft_mask"] > 0
    assert result.terms["local_region"] == 0


def test_instance_regularization_penalizes_large_transform_and_residuals():
    reference, generated = masks()
    instance = PrimitiveInstance(
        id="a-hump",
        primitive="sas-hump-01",
        scale_x=1.2,
        scale_y=0.9,
        shear_x=0.1,
        dx=3,
        dy=-2,
        residual={"shape.y": 0.02},
    )

    result = evaluate_objective(
        reference,
        generated,
        config=ObjectiveConfig(),
        x_height=16,
        instances=[instance],
        complexity_count=2,
    )

    assert result.terms["instance_regularization"] > 0
    assert result.terms["complexity"] == 2
    assert result.total > 0


def test_objective_v3_rejects_legacy_raster_weights_and_settings():
    with pytest.raises(ValueError, match="legacy raster weights"):
        ObjectiveConfig.model_validate({"version": 3, "weights": {"distance": 1.0}})

    with pytest.raises(ValueError, match="legacy geometry settings"):
        ObjectiveConfig.model_validate({"version": 3, "geometry": {"path_to_ink": 1.0}})

    with pytest.raises(ValueError, match="require objective v3"):
        ObjectiveConfig.model_validate({"version": 2, "centerline": {"sample_schedule": [8, 16]}})


def test_objective_config_is_versioned_and_weights_are_nonnegative():
    assert ObjectiveConfig().version == 1
    with pytest.raises(ValidationError):
        ObjectiveConfig(weights={"distance": -1})


def test_objective_rejects_invalid_frame_and_coverage_values():
    reference, generated = masks()
    with pytest.raises(ValueError, match="matching two-dimensional"):
        evaluate_objective(reference, generated[:, :-1], x_height=16)
    generated[0, 0] = 2
    with pytest.raises(ValueError, match="between zero and one"):
        evaluate_objective(reference, generated, x_height=16)
