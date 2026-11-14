from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .models import CropRect


def coerce_crop_rect(crop: CropRect | Sequence[int]) -> CropRect:
    if isinstance(crop, CropRect):
        return crop
    if len(crop) != 4:
        raise ValueError("crop must contain x, y, width, and height")
    return CropRect(x=crop[0], y=crop[1], width=crop[2], height=crop[3])


def crop_image(image: Image.Image, crop: CropRect | Sequence[int]) -> Image.Image:
    rect = coerce_crop_rect(crop)
    if rect.x + rect.width > image.width or rect.y + rect.height > image.height:
        raise ValueError("crop rectangle extends beyond the image bounds")
    return image.crop((rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)).copy()


def load_crop(path: str | Path, crop: CropRect | Sequence[int]) -> Image.Image:
    with Image.open(path) as image:
        return crop_image(image, crop)
