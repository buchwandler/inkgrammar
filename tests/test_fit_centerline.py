import numpy as np

from inkgrammar.fitting.centerline import (
    TargetTrace,
    prune_skeleton,
    resample_trace,
    skeletonize_component,
    trace_ordered_centerline,
)


def test_skeletonize_component_discards_tiny_islands_and_returns_radius():
    mask = np.zeros((48, 48), dtype=bool)
    mask[20:27, 5:40] = True
    mask[40, 40] = True

    skeleton, radius = skeletonize_component(mask)

    assert skeleton.any()
    assert not skeleton[40, 40]
    assert radius[23, 20] > 0
    assert skeleton.sum() < mask.sum()


def test_prune_skeleton_removes_short_thin_attached_spur():
    mask = np.zeros((60, 60), dtype=bool)
    mask[26:35, 5:55] = True
    mask[14:27, 29:30] = True
    skeleton, radius = skeletonize_component(mask)

    pruned = prune_skeleton(
        skeleton,
        radius,
        min_spur_length_px=16,
        min_confidence=0.3,
    )

    assert not pruned[14:26, 29].any()
    assert pruned[30, 10:50].any()
    assert pruned.sum() < skeleton.sum()


def test_ordered_trace_follows_guide_through_crossing():
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:33, 5:59] = True
    mask[5:59, 29:34] = True
    skeleton, radius = skeletonize_component(mask)
    nominal = np.column_stack((np.linspace(6, 57, 64), np.full(64, 30.0)))

    trace = trace_ordered_centerline(
        skeleton,
        radius,
        nominal,
        search_radius_px=4.0,
        candidates_per_sample=8,
    )

    assert np.all(trace.confidence > 0)
    assert np.median(np.abs(trace.xy[:, 1] - 30.0)) < 1.0
    assert np.max(np.linalg.norm(np.diff(trace.xy, axis=0), axis=1)) <= 2.0
    assert np.allclose(np.linalg.norm(trace.tangent[trace.confidence > 0], axis=1), 1.0)


def test_ordered_trace_marks_missing_guide_region_unassigned():
    skeleton = np.zeros((40, 48), dtype=bool)
    skeleton[20, 5:22] = True
    radius = skeleton.astype(float)
    nominal = np.column_stack((np.linspace(8, 42, 48), np.full(48, 20.0)))

    trace = trace_ordered_centerline(
        skeleton,
        radius,
        nominal,
        search_radius_px=2.0,
    )

    assert trace.confidence[:12].max() > 0
    assert not trace.confidence[-10:].any()
    assert np.isnan(trace.xy[-10:]).all()


def test_resampling_preserves_unassigned_trace_gaps():
    trace = TargetTrace(
        xy=np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [5.0, 0.0], [6.0, 0.0]]),
        tangent=np.zeros((5, 2)),
        radius=np.ones(5),
        confidence=np.array([1.0, 1.0, 0.0, 1.0, 1.0]),
    )

    sampled = resample_trace(trace, 9)

    assert len(sampled.xy) == 9
    assert (sampled.confidence == 0).any()
    assert (sampled.confidence > 0).any()
    assert np.isnan(sampled.xy[sampled.confidence == 0]).all()
