"""Compatibility wrappers around planarcurve's neutral fitting encoders."""

from __future__ import annotations

import numpy as np
from planarcurve.fitting.parameters import (
    decode_interval,
    decode_positive,
    encode_interval,
    encode_positive,
    ordered_positions_from_logits,
    ordered_positions_to_logits,
)

from .models import MAX_RATE_DEG, MIN_CHANGE_SPACING

RATE_LIMIT_DEG = MAX_RATE_DEG
ANGLE_LIMIT_DEG = 360.0


def decode_change_positions(
    logits: np.ndarray,
    *,
    min_spacing: float = MIN_CHANGE_SPACING,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    return ordered_positions_from_logits(
        logits, min_spacing=min_spacing, start=start, end=end
    )


def encode_change_positions(
    positions: np.ndarray,
    *,
    min_spacing: float = MIN_CHANGE_SPACING,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    return ordered_positions_to_logits(
        positions, min_spacing=min_spacing, start=start, end=end
    )


def decode_bounded(param: float, *, low: float, high: float) -> float:
    return decode_interval(param, low=low, high=high)


def encode_bounded(value: float, *, low: float, high: float) -> float:
    return encode_interval(value, low=low, high=high)


def decode_rate(param: float) -> float:
    return decode_interval(param, low=-RATE_LIMIT_DEG, high=RATE_LIMIT_DEG)


def encode_rate(rate_deg: float) -> float:
    return encode_interval(rate_deg, low=-RATE_LIMIT_DEG, high=RATE_LIMIT_DEG)


def decode_angle(param: float) -> float:
    return decode_interval(param, low=-ANGLE_LIMIT_DEG, high=ANGLE_LIMIT_DEG)


def encode_angle(angle_deg: float) -> float:
    return encode_interval(angle_deg, low=-ANGLE_LIMIT_DEG, high=ANGLE_LIMIT_DEG)


def decode_length(param: float) -> float:
    return decode_positive(param)


def encode_length(length: float) -> float:
    return encode_positive(length)
