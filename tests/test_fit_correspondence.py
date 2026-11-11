import numpy as np

from inkgrammar.fitting.centerline import TargetTrace
from inkgrammar.fitting.correspondence import monotone_correspondence
from inkgrammar.fitting.residuals import ordered_centerline_residuals


def _trace(xy, *, radius=None, confidence=None):
    xy = np.asarray(xy, dtype=float)
    tangent = np.gradient(xy, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    tangent = np.divide(
        tangent,
        norms[:, None],
        out=np.zeros_like(tangent),
        where=norms[:, None] > 1e-9,
    )
    return TargetTrace(
        xy=xy,
        tangent=tangent,
        radius=np.ones(len(xy)) if radius is None else np.asarray(radius, dtype=float),
        confidence=np.ones(len(xy)) if confidence is None else np.asarray(confidence, dtype=float),
    )


def test_monotone_correspondence_does_not_jump_backward_at_repeated_crossing():
    target_xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    candidate = target_xy.copy()
    candidate[5, 1] = 0.05
    target = _trace(target_xy)

    nearest = np.argmin(np.linalg.norm(candidate[:, None] - target_xy[None, :], axis=2), axis=1)
    correspondence = monotone_correspondence(
        candidate,
        target,
        open_target_ends=False,
    )

    assert np.any(np.diff(nearest) < 0)
    assert np.all(np.diff(correspondence.target_index) >= 0)
    assert correspondence.target_index[5] >= 5
    assert correspondence.candidate_index.tolist() == list(range(len(candidate)))


def test_partial_candidate_fit_keeps_candidate_error_low_and_reports_uncovered_target():
    target_xy = np.column_stack((np.linspace(0.0, 10.0, 21), np.zeros(21)))
    candidate = np.column_stack((np.linspace(4.0, 10.0, 13), np.zeros(13)))

    result = ordered_centerline_residuals(
        candidate,
        _trace(target_xy),
        supervise_endpoints=False,
        radius_weight=0.0,
        coverage_tolerance_px=0.1,
    )

    assert result.diagnostics["candidate_to_target_p95_px"] < 0.1
    assert result.diagnostics["target_to_candidate_p95_px"] >= 3.5
    assert result.diagnostics["assigned_coverage"] == 13 / 21
    assert np.linalg.norm(result.target_to_candidate) > np.linalg.norm(result.candidate_to_target)


def test_bidirectional_position_and_endpoint_metrics_are_interpretable():
    target_xy = np.column_stack((np.linspace(8.0, 40.0, 17), np.full(17, 20.0)))
    candidate = target_xy + np.array([0.0, 1.0])

    result = ordered_centerline_residuals(candidate, _trace(target_xy))

    assert result.diagnostics["candidate_to_target_rms_px"] == 1.0
    assert result.diagnostics["target_to_candidate_rms_px"] == 1.0
    assert result.diagnostics["endpoint_rms_px"] == 1.0
    assert result.diagnostics["tangent_rms_rad"] == 0.0
    assert result.diagnostics["assigned_coverage"] == 1.0
    assert np.isfinite(result.residuals).all()


def test_centerline_fit_objective_is_invariant_to_matching_stroke_widths():
    centerline = np.column_stack((np.linspace(8.0, 40.0, 17), np.linspace(10.0, 30.0, 17)))
    candidate = centerline + np.array([0.0, 0.5])
    norms = np.linalg.norm(np.gradient(centerline, axis=0), axis=1)
    tangent = np.gradient(centerline, axis=0) / norms[:, None]
    residual_vectors = []

    for width in (2.0, 4.0, 7.0):
        radius = np.full(len(centerline), width / 2.0)
        target = TargetTrace(centerline, tangent, radius, np.ones(len(centerline)))
        result = ordered_centerline_residuals(
            candidate,
            target,
            candidate_radius=radius,
        )
        residual_vectors.append(result.residuals)
        assert result.diagnostics["radius_rms_px"] == 0.0

    np.testing.assert_allclose(residual_vectors[0], residual_vectors[1], atol=1e-12)
    np.testing.assert_allclose(residual_vectors[1], residual_vectors[2], atol=1e-12)


def test_radius_residual_reports_stroke_width_mismatch():
    centerline = np.column_stack((np.linspace(8.0, 40.0, 17), np.full(17, 20.0)))
    target = _trace(centerline, radius=np.full(len(centerline), 2.0))

    result = ordered_centerline_residuals(
        centerline,
        target,
        candidate_radius=np.full(len(centerline), 3.0),
    )

    assert result.diagnostics["radius_rms_px"] == 1.0
    assert np.linalg.norm(result.radius) > 0
