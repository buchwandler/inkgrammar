import numpy as np
import pytest

from inkgrammar.fitting.assignment import assign_ink_to_components


def test_assignment_separates_target_distractor_and_outlier_ink():
    observed = np.zeros((64, 64), dtype=bool)
    observed[8:56, 30:33] = True
    observed[30:33, 30:56] = True
    observed[59, 59] = True
    target = np.zeros_like(observed)
    target[8:56, 31] = True
    distractor = np.zeros_like(observed)
    distractor[31, 30:56] = True

    assignment = assign_ink_to_components(
        observed,
        {"target": target, "part-branch": distractor},
        "target",
    )

    target_pixel = (14, 31)
    distractor_pixel = (31, 50)
    outlier_pixel = (59, 59)
    assert assignment.target_weights[target_pixel] > 0.7
    assert assignment.competing_weights["part-branch"][distractor_pixel] > 0.7
    assert assignment.outlier_weights[outlier_pixel] > 0.99
    owned = (
        assignment.target_weights
        + assignment.competing_weights["part-branch"]
        + assignment.outlier_weights
    )
    np.testing.assert_allclose(owned[observed], 1.0)
    assert not assignment.target_weights[~observed].any()
    assert assignment.hard_target_mask()[target_pixel]
    assert not assignment.hard_target_mask()[distractor_pixel]


def test_assignment_rejects_missing_or_mismatched_components():
    observed = np.ones((8, 8), dtype=bool)
    centerline = np.zeros_like(observed)
    centerline[4, 2:6] = True

    with pytest.raises(ValueError, match="target component"):
        assign_ink_to_components(observed, {"branch": centerline}, "target")
    with pytest.raises(ValueError, match="shape"):
        assign_ink_to_components(observed, {"target": np.zeros((7, 8))}, "target")
