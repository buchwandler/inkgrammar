from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from skimage.draw import circle_perimeter

from inkgrammar.comparison import (
    CropRect,
    analyze_skeleton,
    crop_image,
    extract_ink_mask,
)


def test_crop_rect_keeps_requested_pixels_without_resizing() -> None:
    image = Image.new("RGB", (40, 30), "white")
    ImageDraw.Draw(image).rectangle((7, 5, 16, 14), fill="blue")

    crop = crop_image(image, (5, 3, 20, 18))

    assert crop.size == (20, 18)
    assert crop.getpixel((1, 1)) == (255, 255, 255)
    assert crop.getpixel((3, 3)) == (0, 0, 255)


def test_crop_rect_rejects_invalid_and_out_of_bounds_rectangles() -> None:
    with pytest.raises(ValueError):
        CropRect(x=0, y=0, width=0, height=2)

    with pytest.raises(ValueError, match="bounds"):
        crop_image(Image.new("L", (8, 8)), (7, 7, 2, 2))


def test_auto_mask_uses_color_to_remove_gray_notebook_rule() -> None:
    image = Image.new("RGB", (100, 70), "white")
    draw = ImageDraw.Draw(image)
    draw.line((4, 35, 95, 35), fill=(180, 180, 180), width=1)
    draw.line((20, 50, 42, 50), fill=(20, 70, 220), width=5)
    draw.line((42, 50, 50, 43), fill=(20, 70, 220), width=5)
    draw.line((50, 43, 58, 50), fill=(20, 70, 220), width=5)

    result = extract_ink_mask(image)

    assert result.method == "color"
    assert result.mask[50, 30]
    assert not result.mask[35, 30]
    assert result.confidence > 0.5
    assert result.warnings == []


def test_auto_mask_uses_grayscale_for_black_ink() -> None:
    image = Image.new("RGB", (50, 50), "white")
    ImageDraw.Draw(image).line((25, 5, 25, 44), fill="black", width=3)

    result = extract_ink_mask(image)

    assert result.method == "grayscale"
    assert result.mask[5:45, 25].all()
    assert result.confidence > 0.9


def test_dark_mode_finds_ink_on_a_gray_paper_background() -> None:
    image = np.full((40, 40), 180, dtype=np.uint8)
    image[8:32, 19:22] = 20

    result = extract_ink_mask(image, mode="dark")

    assert result.method == "dark"
    assert result.mask[8:32, 20].all()
    assert not result.mask[0, 0]


def test_skeleton_diagnostics_report_endpoints_radius_and_components() -> None:
    mask = np.zeros((24, 24), dtype=bool)
    mask[4:20, 12] = True

    result = analyze_skeleton(mask)

    assert result.topology.endpoint_count == 2
    assert result.topology.junction_count == 0
    assert result.topology.component_count == 1
    assert result.topology.cycle_count == 0
    assert result.radius[10, 12] > 0
    assert result.skeleton.sum() >= 2


def test_skeleton_diagnostics_preserve_a_cycle() -> None:
    mask = np.zeros((31, 31), dtype=bool)
    rows, columns = circle_perimeter(15, 15, 8)
    mask[rows, columns] = True

    result = analyze_skeleton(mask)

    assert result.topology.cycle_count == 1
    assert result.topology.component_count == 1
    assert result.topology.endpoint_count == 0


def test_empty_mask_is_reported_without_fake_topology() -> None:
    result = extract_ink_mask(Image.new("RGB", (12, 12), "white"))
    diagnostics = analyze_skeleton(result.mask)

    assert result.warnings == ["No ink pixels were detected."]
    assert diagnostics.topology.component_count == 0
    assert diagnostics.topology.endpoint_count == 0
