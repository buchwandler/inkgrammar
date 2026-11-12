"""Trajectory fitting from ordered centerlines (the brief's sections 9-12).

The fitter receives an ordered target centerline (never raw ad-hoc pixels):
it estimates observed heading and heading rate, initializes curvature-change
events from the observed profile, and optimizes the low-dimensional trajectory
parameters with bounded robust least squares and small deterministic restarts.
Raster overlap is never the dominant loss.

Canonical stroke length is fixed at 1.0 (stroke space); physical size, rotation,
and placement are per-occurrence similarity transform parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .curvature import evaluate_stroke_at, trajectory_total_turn
from .models import (
    MIN_CHANGE_SPACING,
    CurvatureChange,
    CurvatureProfile,
    CurvatureStrokeGeometry,
    TrajectoryTransform,
)
from .trajectory_parameterization import (
    decode_angle,
    decode_change_positions,
    decode_length,
    decode_rate,
    encode_angle,
    encode_change_positions,
    encode_length,
    encode_rate,
)

DEFAULT_SAMPLES = 64


@dataclass(frozen=True)
class CurvatureFitResult:
    geometry: CurvatureStrokeGeometry
    transform: TrajectoryTransform
    success: bool
    cost: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedProfileFitResult:
    geometry: CurvatureStrokeGeometry
    transforms: list[TrajectoryTransform]
    success: bool
    cost: float
    metrics: dict[str, float] = field(default_factory=dict)


def resample_by_arclength(xy: np.ndarray, *, samples: int = DEFAULT_SAMPLES) -> np.ndarray:
    """Resample an ordered polyline uniformly by arc length (``samples x 2``)."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("xy must have shape (N, 2) with N >= 2")
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    if arc[-1] <= 0:
        raise ValueError("xy has zero length")
    targets = np.linspace(0.0, arc[-1], samples)
    return np.column_stack(
        (np.interp(targets, arc, xy[:, 0]), np.interp(targets, arc, xy[:, 1]))
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge padding (zero padding would spike at borders)."""
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def estimate_observed_heading(xy: np.ndarray, *, smooth_window: int = 5) -> np.ndarray:
    """Observed heading theta_obs(u) in degrees from an ordered polyline."""
    points = resample_by_arclength(xy, samples=max(len(xy), DEFAULT_SAMPLES))
    if smooth_window > 1:
        points = np.column_stack(
            (
                _smooth(points[:, 0], smooth_window),
                _smooth(points[:, 1], smooth_window),
            )
        )
    gradient = np.gradient(points, axis=0)
    heading = np.unwrap(np.arctan2(gradient[:, 1], gradient[:, 0]))
    heading = _smooth(heading, 7)
    return np.rad2deg(heading)


def estimate_observed_heading_rate(u: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
    """Observed heading rate q_obs(u) in deg per unit progress (smoothed)."""
    u = np.asarray(u, dtype=float)
    rates = np.gradient(heading_deg, u)
    return _smooth(rates, 7)



def initialize_event_positions(q_obs: np.ndarray, change_count: int) -> np.ndarray:
    """Initial change locations from the |dq/du| score mass.

    Event candidates sit at evenly spaced quantiles of the accumulated
    turning-change score, which is robust to noise and stays away from the
    boundaries. Falls back to evenly spaced events for flat profiles.
    """
    if change_count <= 0:
        return np.zeros(0)
    u = np.linspace(0.0, 1.0, q_obs.size)
    scores = np.abs(np.gradient(q_obs, u))
    cumulative = np.cumsum(scores)
    if cumulative[-1] <= 0:
        return np.linspace(0.5 / (change_count + 1), 1.0 - 0.5 / (change_count + 1), change_count)
    fractions = np.arange(1, change_count + 1) / (change_count + 1)
    positions = np.interp(fractions * cumulative[-1], cumulative, u)
    if positions.size > 1 and np.diff(positions).min() < MIN_CHANGE_SPACING:
        return np.linspace(0.5 / (change_count + 1), 1.0 - 0.5 / (change_count + 1), change_count)
    return positions




@dataclass
class _ProfileCodec:
    """z = [theta0, K+1 position logits, K+2 rate parameters]."""

    change_count: int

    @property
    def size(self) -> int:
        return 1 + (self.change_count + 1) + (self.change_count + 2)

    def decode(self, z: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        start_angle = decode_angle(float(z[0]))
        positions = decode_change_positions(z[1 : 1 + self.change_count + 1])
        rates = np.array([decode_rate(value) for value in z[1 + self.change_count + 1 :]])
        return start_angle, positions, rates

    def encode(
        self, start_angle: float, positions: np.ndarray, rates: np.ndarray
    ) -> np.ndarray:
        parts = [encode_angle(start_angle)]
        parts.extend(encode_change_positions(positions))
        parts.extend(encode_rate(rate) for rate in rates)
        return np.asarray(parts, dtype=float)


def _geometry_from(start_angle: float, positions: np.ndarray, rates: np.ndarray):
    return CurvatureStrokeGeometry(
        length=1.0,
        start_angle_deg=float(start_angle),
        curvature=CurvatureProfile(
            interpolation="cubic-hermite",
            start_rate_deg=float(rates[0]),
            changes=[
                CurvatureChange(at=float(at), rate_deg=float(rate))
                for at, rate in zip(positions, rates[1:-1], strict=True)
            ],
            end_rate_deg=float(rates[-1]),
        ),
    )


def _target_state(xy: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    points = resample_by_arclength(xy, samples=samples)
    heading_deg = estimate_observed_heading(points)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(steps.sum())
    return points, heading_deg, total_length


def _initial_profile(
    target_points: np.ndarray, target_heading: np.ndarray, change_count: int
) -> tuple[float, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, target_points.shape[0])
    q_obs = estimate_observed_heading_rate(u, target_heading)
    positions = initialize_event_positions(q_obs, change_count)
    knots = np.concatenate(([0.0], positions, [1.0]))
    rates = np.interp(knots, u, q_obs)
    return float(target_heading[0]), positions, rates


def matched_fit_state(geometry, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit positions/headings matched to uniform arc positions.

    Targets are resampled uniformly by arc length, so the fit side must be
    matched by arc position too; comparing uniform-u samples against
    uniform-arc samples would bias the optimization.
    """
    dense_u = np.linspace(0.0, 1.0, 257)
    dense_xy, _ = evaluate_stroke_at(geometry, dense_u)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense_xy, axis=0), axis=1))))
    if arc[-1] <= 0:
        u_matched = dense_u[:samples] if samples <= dense_u.size else dense_u
    else:
        u_matched = np.interp(np.linspace(0.0, 1.0, samples) * arc[-1], arc, dense_u)
    return evaluate_stroke_at(geometry, u_matched)


def _fit_residuals(
    z_profile: np.ndarray,
    codec: _ProfileCodec,
    *,
    target_points: np.ndarray,
    target_heading: np.ndarray,
    scale: float,
    rotation_deg: float,
    translation: np.ndarray,
    weights: dict[str, float],
    smooth_weight: float,
) -> np.ndarray:
    start_angle, positions, rates = codec.decode(z_profile)
    geometry = _geometry_from(start_angle, positions, rates)
    fit_xy, fit_heading = matched_fit_state(geometry, target_points.shape[0])
    angle = np.deg2rad(rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mapped = scale * (fit_xy @ rotation.T) + translation

    position_residuals = ((mapped - target_points) * weights["position"]).ravel()
    heading_delta = (fit_heading + rotation_deg - target_heading + 180.0) % 360.0 - 180.0
    tangent_residuals = heading_delta * weights["tangent"]
    endpoint_residuals = np.concatenate(
        (
            (mapped[0] - target_points[0]) * weights["endpoint"],
            (mapped[-1] - target_points[-1]) * weights["endpoint"],
            [heading_delta[0] * weights["endpoint"] * 0.1],
            [heading_delta[-1] * weights["endpoint"] * 0.1],
        )
    )
    target_turn = float(target_heading[-1] - target_heading[0])
    turn_residuals = np.array(
        [(trajectory_total_turn(geometry) - target_turn) * weights["turn"]]
    )
    profile_residuals = heading_delta * weights["profile"]
    smoothness = np.diff(rates, 2) * smooth_weight
    return np.concatenate(
        (
            position_residuals,
            tangent_residuals,
            endpoint_residuals,
            turn_residuals,
            profile_residuals,
            smoothness,
        )
    )


def _optimization_weights() -> dict[str, float]:
    return {
        "position": 1.0,
        "tangent": 0.15,
        "endpoint": 2.0,
        "turn": 0.3,
        "profile": 0.2,
        "smooth": 0.02,
    }


def _profile_metrics(
    geometry: CurvatureStrokeGeometry,
    transform: TrajectoryTransform,
    target_points: np.ndarray,
    target_heading: np.ndarray,
) -> dict[str, float]:
    fit_xy, fit_heading = matched_fit_state(geometry, target_points.shape[0])
    angle = np.deg2rad(transform.rotation_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mapped = transform.scale * (fit_xy @ rotation.T) + np.array([transform.dx, transform.dy])
    errors = np.linalg.norm(mapped - target_points, axis=1)
    return {
        "median_position_error": float(np.median(errors)),
        "p95_position_error": float(np.percentile(errors, 95)),
        "endpoint_error": float(
            max(
                np.linalg.norm(mapped[0] - target_points[0]),
                np.linalg.norm(mapped[-1] - target_points[-1]),
            )
        ),
        "endpoint_tangent_error_deg": float(
            max(
                abs(
                    (fit_heading[0] + transform.rotation_deg - target_heading[0] + 180)
                    % 360
                    - 180
                ),
                abs(
                    (fit_heading[-1] + transform.rotation_deg - target_heading[-1] + 180)
                    % 360
                    - 180
                ),
            )
        ),
        "total_turn_error_deg": abs(
            trajectory_total_turn(geometry) - float(target_heading[-1] - target_heading[0])
        ),
    }


def fit_curvature_stroke(
    target_xy: np.ndarray,
    *,
    change_count: int,
    samples: int = DEFAULT_SAMPLES,
    restarts: int = 2,
    seed: int = 0,
) -> CurvatureFitResult:
    """Fit one curvature stroke to an ordered target centerline.

    Deterministic initialization followed by bounded robust least squares
    (soft-l1 loss) with small seeded restarts.
    """
    if change_count < 0:
        raise ValueError("change_count must be non-negative")
    target_points, target_heading, target_length = _target_state(target_xy, samples)
    codec = _ProfileCodec(change_count)
    start_angle, positions, rates = _initial_profile(
        target_points, target_heading, change_count
    )
    z0 = codec.encode(start_angle, positions, rates)
    initial_geometry = _geometry_from(start_angle, positions, rates)
    unit_end = np.linalg.norm(
        evaluate_stroke_at(initial_geometry, np.linspace(0.0, 1.0, samples))[0][-1]
    )
    log_scale0 = encode_length(target_length / max(unit_end, 1e-9))
    translation0 = np.array([target_points[0, 0], target_points[0, 1]])
    vector0 = np.concatenate([z0, [log_scale0, translation0[0], translation0[1]]])

    def residuals(vector: np.ndarray) -> np.ndarray:
        scale = decode_length(float(vector[codec.size]))
        translation = vector[codec.size + 1 : codec.size + 3]
        return _fit_residuals(
            vector[: codec.size],
            codec,
            target_points=target_points,
            target_heading=target_heading,
            scale=scale,
            rotation_deg=0.0,
            translation=translation,
            weights=_optimization_weights(),
            smooth_weight=_optimization_weights()["smooth"],
        )

    def decode_candidate(vector):
        start_angle, positions, rates = codec.decode(vector[: codec.size])
        geometry = _geometry_from(start_angle, positions, rates)
        transform = TrajectoryTransform(
            scale=decode_length(float(vector[codec.size])),
            rotation_deg=0.0,
            dx=float(vector[codec.size + 1]),
            dy=float(vector[codec.size + 2]),
        )
        return geometry, transform

    def candidate_score(vector) -> float:
        geometry, transform = decode_candidate(vector)
        candidate_metrics = _profile_metrics(
            geometry, transform, target_points, target_heading
        )
        # prefer geometrically close fits, but reject implausible fold/spin
        # minima by penalizing total-turn error
        return candidate_metrics["median_position_error"] + 0.2 * max(
            0.0, abs(candidate_metrics["total_turn_error_deg"]) - 25.0
        )

    best_result = None
    best_vector = vector0
    best_score = candidate_score(vector0)
    rng = np.random.default_rng(seed)
    for attempt in range(restarts + 1):
        start_vector = (
            vector0
            if attempt == 0
            else vector0 + rng.normal(0.0, 0.15, size=vector0.shape)
        )
        result = least_squares(
            residuals, start_vector, loss="soft_l1", f_scale=1.0, max_nfev=400
        )
        score = candidate_score(result.x)
        if best_result is None or score < best_score:
            best_score = score
            best_result = result
            best_vector = result.x

    start_angle, positions, rates = codec.decode(best_vector[: codec.size])
    geometry = _geometry_from(start_angle, positions, rates)
    transform = TrajectoryTransform(
        scale=decode_length(float(best_vector[codec.size])),
        rotation_deg=0.0,
        dx=float(best_vector[codec.size + 1]),
        dy=float(best_vector[codec.size + 2]),
    )
    metrics = _profile_metrics(geometry, transform, target_points, target_heading)
    return CurvatureFitResult(
        geometry=geometry,
        transform=transform,
        success=bool(best_result.success),
        cost=float(best_result.cost),
        metrics=metrics,
    )


def fit_shared_profile(
    occurrences: list[np.ndarray],
    *,
    change_count: int,
    samples: int = DEFAULT_SAMPLES,
    restarts: int = 1,
    seed: int = 0,
) -> SharedProfileFitResult:
    """Fit one shared curvature profile to several transformed occurrences.

    Per-occurrence nuisance parameters are similarity-only (scale, rotation,
    dx, dy); the normalized turning program is shared.
    """
    if not occurrences:
        raise ValueError("at least one occurrence is required")
    states = [_target_state(xy, samples) for xy in occurrences]
    codec = _ProfileCodec(change_count)

    init_profiles = [
        _initial_profile(points, heading, change_count) for points, heading, _ in states
    ]
    start_angle = float(np.mean([p[0] for p in init_profiles]))
    positions = np.mean([p[1] for p in init_profiles], axis=0)
    rates = np.mean([p[2] for p in init_profiles], axis=0)
    z0 = codec.encode(start_angle, positions, rates)

    transform0 = []
    for (points, heading, length), (_, init_positions, init_rates) in zip(
        states, init_profiles, strict=True
    ):
        init_geometry = _geometry_from(heading[0], init_positions, init_rates)
        unit_length = np.linalg.norm(
            evaluate_stroke_at(init_geometry, np.linspace(0.0, 1.0, samples))[0][-1]
        )
        transform0.append(
            TrajectoryTransform(
                scale=length / max(unit_length, 1e-9),
                rotation_deg=0.0,
                dx=float(points[0, 0]),
                dy=float(points[0, 1]),
            )
        )

    def pack(transforms: list[TrajectoryTransform]) -> np.ndarray:
        values = []
        for transform in transforms:
            values.extend(
                [encode_length(transform.scale), transform.rotation_deg, transform.dx, transform.dy]
            )
        return np.asarray(values, dtype=float)

    def unpack(values: np.ndarray) -> list[TrajectoryTransform]:
        transforms = []
        for index in range(len(states)):
            base = index * 4
            transforms.append(
                TrajectoryTransform(
                    scale=decode_length(float(values[base])),
                    rotation_deg=float(values[base + 1]),
                    dx=float(values[base + 2]),
                    dy=float(values[base + 3]),
                )
            )
        return transforms

    def residuals(vector: np.ndarray) -> np.ndarray:
        z_profile = vector[: codec.size]
        transforms = unpack(vector[codec.size :])
        start, pos, rate_values = codec.decode(z_profile)
        geometry = _geometry_from(start, pos, rate_values)
        fit_xy, fit_heading = matched_fit_state(geometry, samples)
        blocks = []
        weights = _optimization_weights()
        for (points, heading, _), transform in zip(states, transforms, strict=True):
            angle = np.deg2rad(transform.rotation_deg)
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            mapped = transform.scale * (fit_xy @ rotation.T) + np.array(
                [transform.dx, transform.dy]
            )
            heading_delta = (
                fit_heading + transform.rotation_deg - heading + 180.0
            ) % 360.0 - 180.0
            blocks.append((mapped - points) * weights["position"])
            blocks.append(heading_delta[:, None] * weights["tangent"])
        smoothness = np.diff(rate_values, 2) * weights["smooth"]
        blocks.append(smoothness[:, None] if smoothness.size else np.zeros((1, 1)))
        return np.concatenate([np.asarray(block).ravel() for block in blocks])

    best = least_squares(
        residuals, np.concatenate([z0, pack(transform0)]), loss="soft_l1", max_nfev=800
    )
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        candidate = least_squares(
            residuals,
            np.concatenate([z0, pack(transform0)])
            + rng.normal(0.0, 0.1, size=z0.size + 4 * len(states)),
            loss="soft_l1",
            max_nfev=800,
        )
        if candidate.cost < best.cost:
            best = candidate

    start, pos, rate_values = codec.decode(best.x[: codec.size])
    geometry = _geometry_from(start, pos, rate_values)
    transforms = unpack(best.x[codec.size :])
    return SharedProfileFitResult(
        geometry=geometry,
        transforms=transforms,
        success=bool(best.success),
        cost=float(best.cost),
    )
