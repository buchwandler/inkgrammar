from __future__ import annotations

import numpy as np
import pytest
from planarcurve.transforms import Similarity2D
from skimage.draw import circle_perimeter

from inkgrammar.comparison.geometry import differential_geometry
from inkgrammar.comparison.path import (
    PathExtractionError,
    align_ordered_paths,
    fit_similarity_transform,
    recover_ordered_path,
    resample_path,
)
from inkgrammar.comparison.skeleton import analyze_skeleton


def _asymmetric_path() -> np.ndarray:
    progress = np.linspace(0.0, 1.0, 80)
    return np.column_stack(
        (progress * 50.0, 9.0 * np.sin(2.0 * np.pi * progress) + 4.0 * progress**2)
    )


def test_resample_path_returns_fixed_finite_arc_length_samples() -> None:
    sampled = resample_path(_asymmetric_path(), 256)
    segment_lengths = np.linalg.norm(np.diff(sampled, axis=0), axis=1)

    assert sampled.shape == (256, 2)
    assert np.isfinite(sampled).all()
    np.testing.assert_allclose(segment_lengths, segment_lengths.mean(), rtol=0.02)


@pytest.mark.parametrize(
    "known_transform",
    [
        Similarity2D(dx=13.0, dy=-8.0),
        Similarity2D(rotation_deg=37.0, dx=13.0, dy=-8.0),
        Similarity2D(scale=2.4, dx=13.0, dy=-8.0),
        Similarity2D(scale=0.65, rotation_deg=-22.0, dx=-4.0, dy=11.0),
    ],
)
def test_similarity_alignment_removes_translation_rotation_and_scale(
    known_transform: Similarity2D,
) -> None:
    reference = _asymmetric_path()
    sample = known_transform.apply(reference)

    fitted = fit_similarity_transform(reference, sample)
    aligned = fitted.apply(sample)

    np.testing.assert_allclose(aligned, reference, atol=1e-9)
    assert fitted.scale == pytest.approx(1.0 / known_transform.scale)
    assert fitted.rotation_deg == pytest.approx(-known_transform.rotation_deg)


def test_alignment_selects_reversed_sample_orientation() -> None:
    reference = _asymmetric_path()

    result = align_ordered_paths(reference, reference[::-1])

    assert result.direction == "reversed-for-best-match"
    np.testing.assert_allclose(result.aligned_sample_xy, reference, atol=1e-9)
    assert result.transform.scale == pytest.approx(1.0)
    assert result.mean_squared_error < 1e-20


def test_alignment_forbids_reflection() -> None:
    reference = _asymmetric_path()
    reflected = reference.copy()
    reflected[:, 1] *= -1.0

    fitted = fit_similarity_transform(reference, reflected)
    error = np.mean(np.sum((reference - fitted.apply(reflected)) ** 2, axis=1))

    assert fitted.scale > 0.0
    assert error > 1.0


def test_differential_geometry_is_finite_and_keeps_normalized_turn_units() -> None:
    geometry = differential_geometry(resample_path(_asymmetric_path(), 256))

    assert geometry.smoothed_xy.shape == (256, 2)
    assert geometry.heading_rad.shape == (256,)
    assert geometry.turn_rate_rad_per_progress.shape == (256,)
    assert np.isfinite(geometry.turn_rate_rad_per_progress).all()
    np.testing.assert_allclose(
        geometry.cumulative_turn_rad,
        geometry.heading_rad - geometry.heading_rad[0],
    )


def test_ordered_path_covers_loop_and_entry_branch() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    rows, columns = circle_perimeter(32, 32, 14)
    mask[rows, columns] = True
    mask[31, 3:19] = True
    diagnostics = analyze_skeleton(mask)

    path = recover_ordered_path(diagnostics)

    assert path.coverage == 1.0
    assert path.covered_edge_count == path.total_edge_count
    assert diagnostics.topology.cycle_count == 1
    assert np.isfinite(path.xy).all()
    assert np.linalg.norm(np.diff(path.xy, axis=0), axis=1).sum() > 0.0


def test_ordered_path_rejects_disconnected_skeletons() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:13, 5] = True
    mask[5:13, 30] = True

    with pytest.raises(PathExtractionError, match="covers"):
        recover_ordered_path(analyze_skeleton(mask))
