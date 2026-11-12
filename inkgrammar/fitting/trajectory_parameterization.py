"""Order-preserving trajectory parameterization (the brief's section 10).

Change locations are represented as positive interval logits: softmax gives
interval fractions, reserved minimum spacings are added to the interior gaps,
and a cumulative sum produces strictly ordered positions. Ordering and
spacing hold by construction, so fitting never needs to sort events or repair
collapsed ones.

Heading rates, start angles, and lengths use bounded logistic/log transforms
so optimizer parameters stay unconstrained while decoded values stay inside
the schema bounds.
"""

from __future__ import annotations

import numpy as np

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
    """Decode ``K + 1`` interval logits into ``K`` ordered positions in ``(start, end)``.

    Adjacent decoded positions are separated by at least ``min_spacing`` and
    the first/last positions stay strictly inside the interval.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1:
        raise ValueError("logits must be a 1-D array")
    count = logits.size
    if count == 0:
        return np.zeros(0, dtype=float)
    span = end - start
    reserved = max(count - 2, 0) * min_spacing
    if reserved >= span:
        raise ValueError("interval too small for the requested changes and spacing")
    weights = np.exp(logits - logits.max())
    fractions = weights / weights.sum()
    available = span - reserved
    gaps = fractions * available
    gaps[1:-1] += min_spacing
    return start + np.cumsum(gaps[:-1])


def encode_change_positions(
    positions: np.ndarray,
    *,
    min_spacing: float = MIN_CHANGE_SPACING,
    start: float = 0.0,
    end: float = 1.0,
) -> np.ndarray:
    """Encode ``K`` ordered positions back into ``K + 1`` interval logits (up to a constant).

    ``decode_change_positions(encode_change_positions(p)) == p``.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 1:
        raise ValueError("positions must be a 1-D array")
    count = positions.size
    if count == 0:
        return np.zeros(0, dtype=float)
    interior = np.concatenate(([start], positions, [end]))
    gaps = np.diff(interior)
    if np.any(gaps < min_spacing - 1e-12):
        raise ValueError(f"positions must respect minimum spacing {min_spacing}")
    if gaps[0] <= 0 or gaps[-1] <= 0:
        raise ValueError("positions must lie strictly inside the interval")
    available = (end - start) - (count - 1) * min_spacing
    fractions = gaps.copy()
    fractions[1:-1] -= min_spacing
    fractions = np.clip(fractions / available, 1e-12, None)
    return np.log(fractions)


def decode_bounded(param: float, *, low: float, high: float) -> float:
    """Logistic map from unconstrained parameters into ``(low, high)``."""
    if not low < high:
        raise ValueError("low must be less than high")
    return low + (high - low) / (1.0 + np.exp(-param))


def encode_bounded(value: float, *, low: float, high: float) -> float:
    """Inverse logistic map from ``(low, high)`` to unconstrained parameters."""
    if not low < high:
        raise ValueError("low must be less than high")
    fraction = (value - low) / (high - low)
    fraction = float(np.clip(fraction, 1e-12, 1.0 - 1e-12))
    return float(np.log(fraction / (1.0 - fraction)))


def decode_rate(param: float) -> float:
    """Heading rate in deg per unit progress, bounded by ``RATE_LIMIT_DEG``."""
    return decode_bounded(param, low=-RATE_LIMIT_DEG, high=RATE_LIMIT_DEG)


def encode_rate(rate_deg: float) -> float:
    return encode_bounded(rate_deg, low=-RATE_LIMIT_DEG, high=RATE_LIMIT_DEG)


def decode_angle(param: float) -> float:
    """Start angle in degrees, bounded by ``ANGLE_LIMIT_DEG``."""
    return decode_bounded(param, low=-ANGLE_LIMIT_DEG, high=ANGLE_LIMIT_DEG)


def encode_angle(angle_deg: float) -> float:
    return encode_bounded(angle_deg, low=-ANGLE_LIMIT_DEG, high=ANGLE_LIMIT_DEG)


def decode_length(param: float) -> float:
    """Positive stroke length from an unconstrained log parameter."""
    return float(np.exp(np.clip(param, -50.0, 50.0)))


def encode_length(length: float) -> float:
    if length <= 0:
        raise ValueError("length must be positive")
    return float(np.log(length))
