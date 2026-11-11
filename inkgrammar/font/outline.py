from __future__ import annotations

from math import isfinite

from fontTools.cu2qu.cu2qu import curve_to_quadratic

CubicCurve = tuple[tuple[float, float], ...]


def cubic_to_quadratic(
    curve: CubicCurve,
    max_error: float,
) -> tuple[tuple[float, float], ...]:
    if len(curve) != 4:
        raise ValueError("a cubic Bézier curve must contain four points")
    if not isfinite(max_error) or max_error <= 0:
        raise ValueError("max_error must be positive and finite")
    if any(not isfinite(value) for point in curve for value in point):
        raise ValueError("cubic Bézier points must be finite")
    converted = curve_to_quadratic(curve, max_error)
    return tuple((float(x), float(y)) for x, y in converted)
