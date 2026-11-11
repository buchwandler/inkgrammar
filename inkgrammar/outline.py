from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import acos, atan2, ceil, cos, degrees, hypot, isfinite, pi, sin

from .geometry import Point, _record
from .shaping import ShapedText

WidthProfile = float | Sequence[tuple[float, float]] | Callable[[float], float]
Contour = tuple[Point, ...]


@dataclass(frozen=True)
class OutlineConfig:
    max_centerline_error: float = 0.25
    max_width_error: float = 0.25
    max_normal_angle_degrees: float = 3.0
    round_caps: bool = True
    max_subdivision_depth: int = 18

    def __post_init__(self) -> None:
        tolerances = (
            self.max_centerline_error,
            self.max_width_error,
            self.max_normal_angle_degrees,
        )
        if any(not isfinite(value) or value <= 0 for value in tolerances):
            raise ValueError("outline tolerances must be positive")
        if self.max_normal_angle_degrees >= 180:
            raise ValueError("normal angle tolerance must be less than 180 degrees")
        if self.max_subdivision_depth < 1:
            raise ValueError("maximum subdivision depth must be positive")


DEFAULT_OUTLINE_CONFIG = OutlineConfig()


@dataclass(frozen=True)
class ExpandedOutline:
    contours: tuple[Contour, ...]
    bounds: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class _Segment:
    kind: str
    p0: Point
    p1: Point
    p2: Point | None = None
    p3: Point | None = None

    def point(self, t: float) -> Point:
        if self.kind == "line":
            return _lerp(self.p0, self.p1, t)
        if self.kind == "quadratic":
            control = self.p1
            end = self.p2
            assert end is not None
            return _lerp(_lerp(self.p0, control, t), _lerp(control, end, t), t)
        control1, control2, end = self.p1, self.p2, self.p3
        assert control2 is not None and end is not None
        a = _lerp(self.p0, control1, t)
        b = _lerp(control1, control2, t)
        c = _lerp(control2, end, t)
        return _lerp(_lerp(a, b, t), _lerp(b, c, t), t)

    def tangent(self, t: float) -> Point:
        if self.kind == "line":
            return (self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        if self.kind == "quadratic":
            control, end = self.p1, self.p2
            assert end is not None
            return (
                2 * ((1 - t) * (control[0] - self.p0[0]) + t * (end[0] - control[0])),
                2 * ((1 - t) * (control[1] - self.p0[1]) + t * (end[1] - control[1])),
            )
        control1, control2, end = self.p1, self.p2, self.p3
        assert control2 is not None and end is not None
        u = 1 - t
        return (
            3
            * (
                u * u * (control1[0] - self.p0[0])
                + 2 * u * t * (control2[0] - control1[0])
                + t * t * (end[0] - control2[0])
            ),
            3
            * (
                u * u * (control1[1] - self.p0[1])
                + 2 * u * t * (control2[1] - control1[1])
                + t * t * (end[1] - control2[1])
            ),
        )


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _point(args) -> Point:
    return (float(args[0]), float(args[1]))


def _subpaths(d: str) -> list[tuple[list[_Segment], bool]]:
    result: list[tuple[list[_Segment], bool]] = []
    segments: list[_Segment] = []
    current: Point | None = None
    start: Point | None = None
    for operator, args in _record(d):
        if operator == "moveTo":
            if segments:
                result.append((segments, False))
            current = start = _point(args[0])
            segments = []
        elif operator == "lineTo":
            if current is None:
                raise ValueError("outline path draws before moveTo")
            end = _point(args[0])
            segments.append(_Segment("line", current, end))
            current = end
        elif operator == "curveTo":
            if current is None or len(args) != 3:
                raise ValueError("outline path has an invalid cubic segment")
            c1, c2, end = (_point(value) for value in args)
            segments.append(_Segment("cubic", current, c1, c2, end))
            current = end
        elif operator == "qCurveTo":
            if current is None or len(args) != 2 or args[-1] is None:
                raise ValueError("outline path has an invalid quadratic segment")
            control, end = (_point(value) for value in args)
            segments.append(_Segment("quadratic", current, control, end))
            current = end
        elif operator == "closePath":
            if current is None or start is None:
                raise ValueError("outline path closes before moveTo")
            if current != start:
                segments.append(_Segment("line", current, start))
            if segments:
                result.append((segments, True))
            segments = []
            current = start = None
        elif operator == "endPath":
            if segments:
                result.append((segments, False))
            segments = []
            current = start = None
        else:
            raise ValueError(f"unsupported outline path operation: {operator}")
    if segments:
        result.append((segments, False))
    if not result:
        raise ValueError("outline path has no drawable segments")
    return result


def _width_function(profile: WidthProfile) -> Callable[[float], float]:
    if callable(profile):
        function = profile
    elif isinstance(profile, (int, float)):
        constant = float(profile)

        def constant_width(_u: float) -> float:
            return constant

        function = constant_width
    else:
        knots = tuple((float(position), float(width)) for position, width in profile)
        if any(
            not isfinite(position) or not isfinite(width) or width <= 0 for position, width in knots
        ):
            raise ValueError("width profile knots must be finite with positive widths")
        if len(knots) < 2:
            raise ValueError("width profile requires at least two knots")
        if knots[0][0] != 0 or knots[-1][0] != 1:
            raise ValueError("width profile must span parameter values 0 through 1")
        if any(b[0] <= a[0] for a, b in zip(knots, knots[1:], strict=False)):
            raise ValueError("width profile parameters must increase")

        def function(u: float) -> float:
            for (u0, w0), (u1, w1) in zip(knots, knots[1:], strict=False):
                if u <= u1:
                    return w0 + (w1 - w0) * (u - u0) / (u1 - u0)
            return knots[-1][1]

    probes = (function(0.0), function(0.5), function(1.0))
    if any(not isfinite(value) or value <= 0 for value in probes):
        raise ValueError("outline widths must be finite and positive")
    return function


def _unit(vector: Point) -> Point:
    length = hypot(*vector)
    if length <= 1e-12:
        raise ValueError("outline contains a zero-length tangent")
    return (vector[0] / length, vector[1] / length)


def _shift(point: Point, direction: Point, distance: float) -> Point:
    return (point[0] + direction[0] * distance, point[1] + direction[1] * distance)


def _distance_to_chord(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    if length2 <= 1e-24:
        return hypot(point[0] - start[0], point[1] - start[1])
    u = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
    projection = (start[0] + u * dx, start[1] + u * dy)
    return hypot(point[0] - projection[0], point[1] - projection[1])


def _angle_between(left: Point, right: Point) -> float:
    a, b = _unit(left), _unit(right)
    return degrees(acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))))


def _flatten_segment(
    segment: _Segment,
    segment_index: int,
    segment_count: int,
    width_at: Callable[[float], float],
    config: OutlineConfig,
) -> list[tuple[float, Point]]:
    def subdivide(t0: float, t1: float, depth: int) -> list[tuple[float, Point]]:
        p0, p1 = segment.point(t0), segment.point(t1)
        span = t1 - t0
        probes = [segment.point(t0 + span * fraction) for fraction in (0.25, 0.5, 0.75)]
        chord_error = max(_distance_to_chord(point, p0, p1) for point in probes)
        tangent0 = segment.tangent(t0)
        tangent1 = segment.tangent(t1)
        tangent_mid = segment.tangent((t0 + t1) / 2)
        if hypot(*tangent0) <= 1e-12:
            tangent0 = (p1[0] - p0[0], p1[1] - p0[1])
        if hypot(*tangent1) <= 1e-12:
            tangent1 = (p1[0] - p0[0], p1[1] - p0[1])
        if hypot(*tangent_mid) <= 1e-12:
            tangent_mid = (p1[0] - p0[0], p1[1] - p0[1])
        normal_change = max(
            _angle_between(tangent0, tangent_mid),
            _angle_between(tangent_mid, tangent1),
        )
        u0 = (segment_index + t0) / segment_count
        u1 = (segment_index + t1) / segment_count
        fractions = (0.25, 0.5, 0.75)
        widths = (width_at(u0), width_at(u1))
        width_error = max(
            abs(
                width_at(u0 + fraction * (u1 - u0))
                - (widths[0] + fraction * (widths[1] - widths[0]))
            )
            for fraction in fractions
        )
        acceptable = (
            chord_error <= config.max_centerline_error
            and normal_change <= config.max_normal_angle_degrees
            and width_error <= config.max_width_error
        )
        if acceptable:
            return [(t0, p0), (t1, p1)]
        if depth >= config.max_subdivision_depth:
            raise ValueError("adaptive outline subdivision exceeded its depth limit")
        middle = (t0 + t1) / 2
        left = subdivide(t0, middle, depth + 1)
        right = subdivide(middle, t1, depth + 1)
        return left[:-1] + right

    return subdivide(0.0, 1.0, 0)


def _profile_points(
    segments: list[_Segment], width_at: Callable[[float], float], config: OutlineConfig
) -> tuple[list[Point], list[float]]:
    sampled: list[Point] = []
    widths: list[float] = []
    count = len(segments)
    for index, segment in enumerate(segments):
        samples = _flatten_segment(segment, index, count, width_at, config)
        for sample_index, (t, point) in enumerate(samples):
            if index and sample_index == 0:
                continue
            u = (index + t) / count
            current_width = width_at(u)
            if not isfinite(current_width) or current_width <= 0:
                raise ValueError("outline widths must be finite and positive")
            sampled.append(point)
            widths.append(current_width)
    return sampled, widths


def _normals(points: list[Point], closed: bool) -> list[Point]:
    result = []
    count = len(points)
    for index, point in enumerate(points):
        if closed:
            previous = points[(index - 1) % count]
            following = points[(index + 1) % count]
        else:
            previous = points[max(0, index - 1)]
            following = points[min(count - 1, index + 1)]
        tangent = (following[0] - previous[0], following[1] - previous[1])
        if hypot(*tangent) <= 1e-12:
            tangent = (
                points[min(index + 1, count - 1)][0] - point[0],
                points[min(index + 1, count - 1)][1] - point[1],
            )
        tx, ty = _unit(tangent)
        result.append((-ty, tx))
    return result


def _offset(points: list[Point], widths: list[float], closed: bool, side: float) -> list[Point]:
    normals = _normals(points, closed)
    offsets = []
    count = len(points)
    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        radius = widths[index] / 2
        if closed or 0 < index < count - 1:
            before = normals[(index - 1) % count]
            after = normals[(index + 1) % count]
            bisector = (before[0] + after[0], before[1] + after[1])
            if hypot(*bisector) > 1e-12:
                bisector = _unit(bisector)
                denominator = bisector[0] * normal[0] + bisector[1] * normal[1]
                scale = min(
                    widths[index] / 2 / max(denominator, 0.25),
                    widths[index] * 2,
                )
                normal = (bisector[0] * scale / radius, bisector[1] * scale / radius)
        offsets.append((point[0] + side * radius * normal[0], point[1] + side * radius * normal[1]))
    return offsets


def _arc_points(center: Point, radius: float, start_angle: float, sweep: float, tolerance: float):
    if radius <= 0:
        return [center]
    ratio = max(-1.0, min(1.0, 1 - tolerance / radius))
    max_angle = 2 * acos(ratio)
    steps = max(2, ceil(abs(sweep) / max(max_angle, 1e-6)))
    if steps % 2:
        steps += 1
    return [
        (
            center[0] + radius * cos(start_angle + sweep * index / steps),
            center[1] + radius * sin(start_angle + sweep * index / steps),
        )
        for index in range(steps + 1)
    ]


def _signed_area(points: Sequence[Point]) -> float:
    return (
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, (*points[1:], points[0]), strict=False)
        )
        / 2
    )


def _close(points: list[Point]) -> Contour:
    if not points:
        return ()
    if points[-1] != points[0]:
        points.append(points[0])
    if not all(isfinite(value) for point in points for value in point):
        raise ValueError("outline produced non-finite coordinates")
    return tuple(points)


def _expand_subpath(
    segments: list[_Segment],
    closed: bool,
    width_at,
    config: OutlineConfig,
) -> tuple[Contour, ...]:
    points, widths = _profile_points(segments, width_at, config)
    if len(points) < 2 or any(
        hypot(b[0] - a[0], b[1] - a[1]) <= 1e-12 for a, b in zip(points, points[1:], strict=False)
    ):
        raise ValueError("outline path contains a zero-length segment")
    if closed and points[-1] == points[0]:
        points.pop()
        widths.pop()
    left = _offset(points, widths, closed, 1.0)
    right = _offset(points, widths, closed, -1.0)
    if closed:
        left_contour = list(left)
        right_contour = list(right)
        if _signed_area(left_contour) * _signed_area(right_contour) > 0:
            right_contour.reverse()
        return (_close(left_contour), _close(right_contour))
    if config.round_caps:
        start_tangent = _unit((points[1][0] - points[0][0], points[1][1] - points[0][1]))
        end_tangent = _unit((points[-1][0] - points[-2][0], points[-1][1] - points[-2][1]))
        end_angle = atan2(end_tangent[1], end_tangent[0])
        start_angle = atan2(start_tangent[1], start_tangent[0])
        end_cap = _arc_points(
            points[-1],
            widths[-1] / 2,
            end_angle + pi / 2,
            -pi,
            config.max_centerline_error,
        )
        start_cap = _arc_points(
            points[0],
            widths[0] / 2,
            start_angle - pi / 2,
            -pi,
            config.max_centerline_error,
        )
        contour = left + end_cap[1:] + list(reversed(right))[1:] + start_cap[1:]
    else:
        start_tangent = _unit((points[1][0] - points[0][0], points[1][1] - points[0][1]))
        end_tangent = _unit((points[-1][0] - points[-2][0], points[-1][1] - points[-2][1]))
        start_extension, end_extension = widths[0] / 2, widths[-1] / 2
        left[0] = _shift(left[0], start_tangent, -start_extension)
        right[0] = _shift(right[0], start_tangent, -start_extension)
        left[-1] = _shift(left[-1], end_tangent, end_extension)
        right[-1] = _shift(right[-1], end_tangent, end_extension)
        contour = left + list(reversed(right))
    return (_close(contour),)


def _result(contours: Sequence[Contour]) -> ExpandedOutline:
    nonempty = tuple(contour for contour in contours if contour)
    if not nonempty:
        return ExpandedOutline((), None)
    points = [point for contour in nonempty for point in contour]
    return ExpandedOutline(
        nonempty,
        (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        ),
    )


def expand_path(
    d: str,
    width: WidthProfile,
    *,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> ExpandedOutline:
    width_at = _width_function(width)
    contours = [
        contour
        for segments, closed in _subpaths(d)
        for contour in _expand_subpath(segments, closed, width_at, config)
    ]
    return _result(contours)


def expand_tap(
    point: Point,
    width: float,
    *,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> ExpandedOutline:
    width_at = _width_function(width)
    diameter = width_at(0.5)
    radius = diameter / 2
    max_angle = 2 * acos(max(-1.0, 1 - config.max_centerline_error / radius))
    steps = max(8, ceil(2 * pi / max(max_angle, 1e-6)))
    steps = ceil(steps / 4) * 4
    contour = _close(
        [
            (
                point[0] + radius * cos(2 * pi * index / steps),
                point[1] + radius * sin(2 * pi * index / steps),
            )
            for index in range(steps)
        ]
    )
    return _result((contour,))


def expand_shaped_text(
    shaped: ShapedText,
    default_width: float,
    *,
    width_profiles: dict[str, WidthProfile] | None = None,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> ExpandedOutline:
    profiles = width_profiles or {}
    contours: list[Contour] = []
    for stroke in shaped.strokes:
        width = profiles.get(stroke.id, default_width)
        if stroke.d is not None:
            contours.extend(expand_path(stroke.d, width, config=config).contours)
        elif stroke.tap is not None:
            tap_width = _width_function(width)(0.5)
            contours.extend(expand_tap(stroke.tap, tap_width, config=config).contours)
    return _result(contours)
