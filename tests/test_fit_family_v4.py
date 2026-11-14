import numpy as np
import pytest
from skeletonpaths import PathRankingPolicy, PathSearchLimits

from inkgrammar.fitting.family_fit_v4 import (
    MonotoneCurveCorrespondence,
    ObservedCurve,
    PreparedTopologyTarget,
    SplineOccurrence,
    estimate_constant_width,
    evaluate_family_search_energy,
    evaluate_topology_path,
    extract_observed_curve,
    fit_occurrence_transform,
    fit_spline_family,
    monotone_curve_correspondence,
    prepare_topology_target,
)
from inkgrammar.fitting.models import (
    BSplineGeometry,
    FamilyTopology,
    OccurrenceFitBounds,
    OccurrenceFitState,
    PrimitiveFrame,
    PrimitiveTemplate,
    SplineEndpointPolicy,
    TopologyTransform,
)
from inkgrammar.fitting.objective import SplineCorrespondenceConfig, SplineCurveConfig
from inkgrammar.fitting.skeleton_graph import (
    SkeletonEdge,
    SkeletonGraph,
    SkeletonNode,
    prepare_topology_paths,
)
from inkgrammar.fitting.spline import sample_curve_by_arclength


def spline_template(control_points=None):
    points = control_points or [
        (0.0, 0.0),
        (1.0, 1.6),
        (2.0, 2.2),
        (3.2, 1.7),
        (4.0, 0.4),
        (5.0, 0.8),
    ]
    geometry = BSplineGeometry(
        knots=[0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0],
        control_points=points,
        endpoint_policy=SplineEndpointPolicy(start="locked", end="locked"),
    )
    return PrimitiveTemplate(
        id="v4-synthetic",
        kind="bspline-centerline",
        frame=PrimitiveFrame(start=points[0], nominal_end=points[-1]),
        spline=geometry,
    )


def observed_curve(template, transform, radius_px, origin=(100.0, 100.0), count=64):
    sampled = sample_curve_by_arclength(template, transform, count)
    xy = sampled.xy.copy()
    xy[:, 0] += origin[0]
    xy[:, 1] = origin[1] - xy[:, 1]
    tangent = np.gradient(xy, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    return ObservedCurve(
        u=np.linspace(0.0, 1.0, count),
        xy=xy,
        tangent=tangent,
        radius=np.full(count, radius_px),
        confidence=np.ones(count),
        source_indices=np.arange(count),
        ambiguous=np.zeros(count, dtype=bool),
    )


def occurrence(key, target, origin=(100.0, 100.0)):
    state = OccurrenceFitState(
        instance_id=key,
        nominal_transform=TopologyTransform(),
        bounds=OccurrenceFitBounds(
            dx=(-5.0, 5.0),
            dy=(-5.0, 5.0),
            log_scale_x=(-0.1, 0.1),
            log_scale_y=(-0.1, 0.1),
            shear_x=(-0.05, 0.05),
        ),
        free_parameters=["dx", "dy", "log_scale_x", "log_scale_y"],
    )
    return SplineOccurrence(key=key, target=target, state=state, origin=origin)


def test_topology_target_extracts_an_ordered_confidence_weighted_curve():
    mask = np.zeros((41, 41), dtype=bool)
    mask[5:36, 20] = True
    confidence = np.ones(mask.shape)
    confidence[20, 20] = 0.4
    radius = np.ones(mask.shape)
    candidate = np.column_stack((np.full(31, 20.0), np.arange(5.0, 36.0)))
    prepared = prepare_topology_target(
        mask,
        confidence=confidence,
        radius=radius,
        min_spur_length_px=0.0,
    )

    selected = extract_observed_curve(
        mask,
        min_spur_length_px=0.0,
        expected_cycles=0,
        expected_endpoints=((20.0, 5.0), (20.0, 35.0)),
        candidate_xy=candidate,
        prepared_target=prepared,
        sample_count=31,
    )

    target, graph, path = selected.curve, selected.graph, selected.path
    assert graph.endpoint_count == 2
    assert graph is prepared.graph
    assert path.start_node != path.end_node
    assert selected.feasible_path_count >= 1
    assert isinstance(selected.route_ambiguous, bool)
    assert isinstance(selected.orientation_ambiguous, bool)
    assert np.all(np.diff(target.u) > 0.0)
    assert target.xy.shape == (31, 2)
    assert np.any(target.confidence < 1.0)
    assert target.source_indices.shape == (31,)


def test_monotone_arclength_correspondence_is_ordered_and_warp_regularized():
    x = np.linspace(0.0, 10.0, 51)
    candidate = np.column_stack((x, np.zeros_like(x)))
    target_u = np.linspace(0.0, 1.0, 67)
    target_xy = np.column_stack((10.0 * target_u, np.zeros_like(target_u)))
    target = ObservedCurve(
        u=target_u,
        xy=target_xy,
        tangent=np.tile([1.0, 0.0], (len(target_u), 1)),
        radius=np.ones(len(target_u)),
        confidence=np.ones(len(target_u)),
        source_indices=np.arange(len(target_u)),
        ambiguous=np.zeros(len(target_u), dtype=bool),
    )

    match = monotone_curve_correspondence(
        candidate, target, warp_weight=0.2, open_target_ends=False
    )

    assert len(match.candidate_indices) >= len(candidate)
    assert np.all(np.diff(match.candidate_indices) >= 0)
    assert np.all(np.diff(match.target_indices) >= 0)
    assert 0.0 <= match.mean_entropy <= 1.0
    np.testing.assert_allclose(
        np.linspace(0.0, 1.0, len(candidate))[match.candidate_indices],
        target.u[match.target_indices],
        atol=0.04,
    )


def test_alternating_family_fit_recovers_shape_and_occurrence_transforms():
    truth = spline_template()
    initial = spline_template(
        [
            (0.0, 0.0),
            (1.0, 1.25),
            (2.0, 2.5),
            (3.2, 1.4),
            (4.0, 0.7),
            (5.0, 0.8),
        ]
    )
    shifts = ((1.6, -1.1), (-1.3, 1.4))
    occurrences = []
    for index, shift in enumerate(shifts):
        true_transform = TopologyTransform(dx=shift[0], dy=shift[1])
        target = observed_curve(truth, true_transform, radius_px=1.0 + index)
        occurrences.append(occurrence(f"occ-{index}", target))

    result = fit_spline_family(
        initial,
        occurrences,
        curve_config=SplineCurveConfig(
            smoothness_lambda=1e-6,
            seed_lambda=1e-6,
            robust_scale_px=1.5,
        ),
        correspondence_config=SplineCorrespondenceConfig(
            position_sigma_px=1.0,
            tangent_weight=0.05,
            warp_weight=0.2,
            gap_penalty=0.5,
        ),
        max_cycles=6,
        transform_max_nfev=80,
        irls_iterations=3,
        tolerance=1e-5,
    )

    curve_errors = []
    for item in occurrences:
        delta = result.states[item.key].delta
        transform = TopologyTransform(
            dx=delta.dx,
            dy=delta.dy,
            scale_x=np.exp(delta.log_scale_x),
            scale_y=np.exp(delta.log_scale_y),
            shear_x=delta.shear_x,
        )
        predicted = sample_curve_by_arclength(result.geometry, transform, len(item.target.xy)).xy
        predicted[:, 0] += item.origin[0]
        predicted[:, 1] = item.origin[1] - predicted[:, 1]
        curve_errors.append(np.max(np.linalg.norm(predicted - item.target.xy, axis=1)))
    assert max(curve_errors) < 0.03
    for index, shift in enumerate(shifts):
        state = result.states[f"occ-{index}"]
        assert state.delta.dx == pytest.approx(shift[0], abs=0.15)
        assert state.delta.dy == pytest.approx(shift[1], abs=0.15)
    assert result.width is not None
    assert result.width.value == pytest.approx(3.0, abs=0.2)


def test_robust_occurrence_transform_fit_resists_an_isolated_outlier():
    template = spline_template()
    true_transform = TopologyTransform(dx=1.2, dy=-0.8)
    target = observed_curve(template, true_transform, radius_px=1.0)
    target.xy[25] += np.array([40.0, -35.0])
    initial_state = occurrence("robust", target).state
    candidate = sample_curve_by_arclength(template, TopologyTransform(), len(target.xy))
    correspondence = MonotoneCurveCorrespondence(
        candidate_indices=np.arange(len(target.xy)),
        target_indices=np.arange(len(target.xy)),
        cost=0.0,
    )

    fitted = fit_occurrence_transform(
        template,
        target,
        initial_state,
        (100.0, 100.0),
        candidate.u,
        correspondence,
        active_parameters=("dx", "dy"),
        position_sigma_px=1.0,
        robust_scale_px=1.0,
    )

    assert fitted.delta.dx == pytest.approx(1.2, abs=0.05)
    assert fitted.delta.dy == pytest.approx(-0.8, abs=0.05)


def test_width_profile_is_separate_from_centerline_shape():
    truth = spline_template()
    initial = spline_template(
        [
            (0.0, 0.0),
            (1.0, 1.5),
            (2.0, 2.4),
            (3.2, 1.5),
            (4.0, 0.5),
            (5.0, 0.8),
        ]
    )
    target_thin = observed_curve(truth, TopologyTransform(), radius_px=1.0)
    target_thick = observed_curve(truth, TopologyTransform(), radius_px=3.0)
    config = SplineCurveConfig(smoothness_lambda=1e-6, seed_lambda=1e-6, robust_scale_px=0.2)
    correspondence = SplineCorrespondenceConfig(position_sigma_px=1.0, warp_weight=0.2)
    thin = fit_spline_family(
        initial,
        [occurrence("thin", target_thin)],
        curve_config=config,
        correspondence_config=correspondence,
        max_cycles=4,
    )
    thick = fit_spline_family(
        initial,
        [occurrence("thick", target_thick)],
        curve_config=config,
        correspondence_config=correspondence,
        max_cycles=4,
    )

    np.testing.assert_allclose(
        thin.geometry.spline.control_points,
        thick.geometry.spline.control_points,
        atol=1e-12,
    )
    assert thin.width.value == pytest.approx(2.0, abs=1e-3)
    assert thick.width.value == pytest.approx(6.0, abs=1e-3)
    assert (
        estimate_constant_width(np.array([1.0, 2.0]), np.array([1.0, 1.0]), TopologyTransform())
        == 3.0
    )


def prepared_target_from_graph(graph):
    shape = (9, 9)
    return PreparedTopologyTarget(
        mask=np.zeros(shape, dtype=bool),
        graph=graph,
        path_inventory=prepare_topology_paths(graph),
        radius_map=np.zeros(shape),
        confidence_map=np.ones(shape),
        min_spur_length_px=0.0,
    )


def incidental_cycle_graph():
    return SkeletonGraph(
        nodes=(
            SkeletonNode(0, (1.0, 2.0), degree=1, radius_px=1.0, confidence=1.0),
            SkeletonNode(1, (4.0, 2.0), degree=3, radius_px=1.0, confidence=1.0),
            SkeletonNode(2, (7.0, 2.0), degree=1, radius_px=1.0, confidence=1.0),
        ),
        edges=(
            SkeletonEdge(0, 1, np.array([[1.0, 2.0], [4.0, 2.0]]), 3.0, 1.0, 1.0),
            SkeletonEdge(
                1,
                1,
                np.array([[4.0, 2.0], [4.0, 1.0], [5.0, 1.0], [5.0, 2.0], [4.0, 2.0]]),
                4.0,
                1.0,
                1.0,
            ),
            SkeletonEdge(1, 2, np.array([[4.0, 2.0], [7.0, 2.0]]), 3.0, 1.0, 1.0),
        ),
    )


def test_declared_open_topology_returns_finite_penalty_for_incidental_graph_cycle():
    prepared = prepared_target_from_graph(incidental_cycle_graph())
    candidate = np.array([[1.0, 2.0], [4.0, 2.0], [7.0, 2.0]])

    result = evaluate_topology_path(
        prepared,
        FamilyTopology(
            path_kind="open",
            expected_cycles=0,
            expected_self_intersections=0,
        ),
        candidate,
    )

    assert result.path is not None
    assert result.exact_match is True
    assert np.isfinite(result.topology_penalty)
    assert result.diagnostics["expected_path_kind"] == "open"
    assert result.diagnostics["selected_path_kind"] == "open"
    assert result.diagnostics["cycle_mismatch"] == 0
    assert result.diagnostics["observed_graph_cycles"] == 1
    assert result.diagnostics["orientation_ambiguous"] is False
    assert result.diagnostics["path_score"]["guide_distance_px"] >= 0.0


def test_candidate_self_intersection_does_not_change_declared_open_topology():
    prepared = prepared_target_from_graph(incidental_cycle_graph())
    candidate = np.array([[1.0, 3.0], [5.0, 3.0], [5.0, 1.0], [3.0, 5.0], [7.0, 5.0]])

    result = evaluate_topology_path(
        prepared,
        FamilyTopology(
            path_kind="open",
            expected_cycles=0,
            expected_self_intersections=0,
        ),
        candidate,
    )

    assert result.path is not None
    assert result.diagnostics["expected_path_kind"] == "open"
    assert result.diagnostics["candidate_self_intersections"] == 1
    assert result.topology_penalty >= 10.0


def test_search_topology_returns_finite_no_path_result():
    graph = SkeletonGraph(nodes=(), edges=())
    prepared = prepared_target_from_graph(graph)

    result = evaluate_topology_path(
        prepared,
        FamilyTopology(path_kind="open-with-loop", expected_cycles=1),
        np.array([[1.0, 2.0], [3.0, 2.0]]),
    )

    assert result.path is None
    assert result.exact_match is False
    assert result.topology_penalty == 50.0
    assert result.diagnostics["no_path"] is True
    assert result.diagnostics["orientation_ambiguous"] is False
    assert result.diagnostics["path_score"] is None

    configured = evaluate_topology_path(
        prepared,
        FamilyTopology(path_kind="open-with-loop", expected_cycles=1),
        np.array([[1.0, 2.0], [3.0, 2.0]]),
        no_path_penalty=17.0,
    )
    assert configured.topology_penalty == 17.0


def test_family_search_energy_reports_balanced_terms_and_invalid_penalty():
    energy = evaluate_family_search_energy(
        [1.0, 3.0],
        endpoint_values=[2.0],
        tangent_values=[0.4],
        topology_values=[5.0],
        saturation_values=[0.25],
        assignment_instability=0.3,
        roughness_penalty=0.2,
        seed_distance_penalty=0.1,
    )

    assert energy.total == pytest.approx(7.095)
    assert energy.geometry_mean == pytest.approx(2.0)
    assert energy.geometry_worst == pytest.approx(3.0)
    assert energy.geometry_spread == pytest.approx(1.0)
    assert energy.as_dict()["total"] == energy.total

    invalid = evaluate_family_search_energy([1.0], invalid_candidate=True)
    assert invalid.total == 1e6
    assert np.isfinite(invalid.total)


def test_family_search_energy_rejects_nonfinite_or_negative_terms():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_family_search_energy([float("nan")])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_family_search_energy([-0.1])


def test_topology_preparation_respects_requested_kinds_and_policy(monkeypatch):
    from inkgrammar.fitting import family_fit_v4 as family_fit_v4_module

    mask = np.zeros((15, 15), dtype=bool)
    mask[2:13, 7] = True
    candidate = np.column_stack((np.full(11, 7.0), np.arange(2.0, 13.0)))
    limits = PathSearchLimits(max_paths=32, max_states=1000)
    policy = PathRankingPolicy(guide_samples=12)
    prepared = prepare_topology_target(
        mask,
        min_spur_length_px=0.0,
        path_kinds=("open",),
        search_limits=limits,
        ranking_policy=policy,
    )

    assert tuple(prepared.path_inventory.paths_by_kind) == ("open",)
    assert prepared.search_limits == limits
    assert prepared.ranking_policy == policy
    rank_calls = []
    select_calls = []
    original_rank_paths = family_fit_v4_module.rank_paths
    original_select_path = family_fit_v4_module.select_path

    def capture_rank_paths(*args, **kwargs):
        rank_calls.append((kwargs["policy"], kwargs["limits"]))
        return original_rank_paths(*args, **kwargs)

    def capture_select_path(*args, **kwargs):
        select_calls.append((kwargs["policy"], kwargs["limits"]))
        return original_select_path(*args, **kwargs)

    monkeypatch.setattr(family_fit_v4_module, "rank_paths", capture_rank_paths)
    monkeypatch.setattr(family_fit_v4_module, "select_path", capture_select_path)
    result = evaluate_topology_path(
        prepared, FamilyTopology(path_kind="open", expected_cycles=0), candidate
    )

    assert result.path is not None
    assert rank_calls == [(policy, limits)]
    assert select_calls == [(policy, limits)]
    assert result.diagnostics["prepared_path_kinds"] == ["open"]
