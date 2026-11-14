from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from .models import InkMaskResult, InkMode

MaskMethod = Literal["grayscale", "dark", "color"]


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        return np.clip(rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha), 0.0, 255.0)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be a grayscale, RGB, or RGBA image")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("image pixels must be finite")
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    if array.shape[2] == 4:
        alpha = array[:, :, 3:4] / 255.0
        array = array[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
    return np.clip(array[:, :, :3], 0.0, 255.0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _otsu_threshold(values: np.ndarray) -> float:
    histogram = np.bincount(np.clip(values, 0, 255).astype(np.uint8).ravel(), minlength=256)
    probability = histogram.astype(float) / histogram.sum()
    levels = np.arange(256, dtype=float)
    background_weight = np.cumsum(probability)
    background_mean = np.cumsum(probability * levels)
    total_mean = background_mean[-1]
    denominator = background_weight * (1.0 - background_weight)
    between_variance = np.divide(
        (total_mean * background_weight - background_mean) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.argmax(between_variance))


def _chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    colored = (saturation >= 0.20) & (maximum < 250.0)
    return saturation, colored


def _select_method(rgb: np.ndarray) -> MaskMethod:
    _, colored = _chroma(rgb)
    minimum_pixels = max(4, int(rgb.shape[0] * rgb.shape[1] * 0.00005))
    if int(colored.sum()) >= minimum_pixels:
        return "color"
    return "grayscale"


def _quality(mask: np.ndarray, luminance: np.ndarray, saturation: np.ndarray, method: str) -> float:
    if not np.any(mask):
        return 0.0
    ink_luminance = float(np.median(luminance[mask]))
    background = ~mask
    if np.any(background):
        contrast = float(np.median(luminance[background])) - ink_luminance
        contrast_confidence = np.clip(contrast / 128.0, 0.0, 1.0)
    else:
        contrast_confidence = 0.0
    if method == "color":
        chroma_confidence = np.clip(float(np.median(saturation[mask])) / 0.5, 0.0, 1.0)
        return float(0.5 * contrast_confidence + 0.5 * chroma_confidence)
    return float(contrast_confidence)


def _warnings(mask: np.ndarray, confidence: float) -> list[str]:
    warnings = []
    fraction = float(mask.mean())
    if not np.any(mask):
        warnings.append("No ink pixels were detected.")
    elif fraction > 0.5:
        warnings.append("More than half of the crop was classified as ink.")
    elif confidence < 0.25:
        warnings.append("Ink/background separation confidence is low.")
    return warnings


def extract_ink_mask(
    image: Image.Image | np.ndarray,
    mode: InkMode = "auto",
) -> InkMaskResult:
    if mode not in {"auto", "grayscale", "dark", "color"}:
        raise ValueError("ink mode must be auto, grayscale, dark, or color")
    rgb = _rgb_array(image)
    luminance = _luminance(rgb)
    saturation, colored = _chroma(rgb)
    selected: MaskMethod = _select_method(rgb) if mode == "auto" else mode

    if selected == "color":
        mask = colored
    elif selected == "dark":
        paper_level = float(np.percentile(luminance, 95.0))
        threshold = max(12.0, min(240.0, paper_level - 32.0))
        mask = luminance <= threshold
    else:
        mask = luminance <= _otsu_threshold(luminance)

    mask = np.asarray(mask, dtype=bool)
    confidence = _quality(mask, luminance, saturation, selected)
    return InkMaskResult(mask, confidence, selected, _warnings(mask, confidence))
