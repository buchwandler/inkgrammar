import numpy as np
import pytest
from PIL import Image

from scriptstrokes.raster import compare_anchored_masks, threshold_image


def test_binary_threshold_retains_black_ink():
    image = Image.new("L", (10, 10), 255)
    image.putpixel((4, 5), 0)
    mask = threshold_image(image)
    assert mask.sum() == 1
    assert mask[5, 4]


def test_anchor_comparison_exposes_displacement_hidden_by_normalization():
    reference = np.zeros((60, 60), dtype=bool)
    generated = reference.copy()
    reference[10:40, 10:15] = True
    generated[20:50, 20:25] = True
    metrics = compare_anchored_masks(
        reference, generated, reference_anchor=(0, 50), generated_anchor=(0, 50)
    )
    assert metrics["iou"] == 1
    assert metrics["baseline_aligned_iou"] == 0
    assert metrics["center_of_mass_offset"] == pytest.approx(10 * 2**0.5)
    assert metrics["ink_bbox_width_ratio"] == 1


def test_anchor_translation_aligns_same_shape():
    reference = np.zeros((60, 60), dtype=bool)
    generated = reference.copy()
    reference[10:40, 10:15] = True
    generated[20:50, 20:25] = True
    metrics = compare_anchored_masks(
        reference, generated, reference_anchor=(0, 40), generated_anchor=(10, 50)
    )
    assert metrics["baseline_aligned_iou"] == 1
    assert metrics["center_of_mass_offset"] == 0


def test_anchor_scale_preserves_dimensions():
    reference = np.zeros((60, 60), dtype=bool)
    reference[10:30, 10:20] = True
    generated = np.repeat(np.repeat(reference, 2, axis=0), 2, axis=1)
    metrics = compare_anchored_masks(
        reference,
        generated,
        reference_anchor=(0, 0),
        generated_anchor=(0, 0),
        generated_pixels_per_reference_pixel=2,
    )
    assert metrics["baseline_aligned_iou"] == 1
    assert metrics["ink_bbox_width_ratio"] == 1
    assert metrics["ink_bbox_height_ratio"] == 1


def test_invalid_anchor_scale():
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="positive"):
        compare_anchored_masks(
            mask,
            mask,
            reference_anchor=(0, 0),
            generated_anchor=(0, 0),
            generated_pixels_per_reference_pixel=0,
        )


def test_render_frame_preserves_negative_bearings_and_vertical_overshoot():
    from scriptstrokes.io import load_model
    from scriptstrokes.raster import render_text_frame

    model = load_model("data/models/demo-looped.yaml")
    model.components["underturn"].d = "M -100 1000 L 100 -400"
    frame = render_text_frame(model, "i", scale=0.2)
    mask = threshold_image(frame.image)
    assert frame.origin[0] > 0
    assert frame.origin[1] > model.metrics.ascender * frame.scale
    assert mask.any()
    assert not mask[0].any()
    assert not mask[-1].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()


def test_render_frame_reports_writing_origin():
    from scriptstrokes.io import load_model
    from scriptstrokes.raster import render_text_frame

    model = load_model("data/models/demo-looped.yaml")
    frame = render_text_frame(model, "i", scale=0.2)
    assert frame.origin[1] == model.metrics.ascender * frame.scale
    assert frame.origin[0] > 0
