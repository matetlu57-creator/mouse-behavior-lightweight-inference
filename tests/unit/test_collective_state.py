from __future__ import annotations

from dataclasses import replace

import numpy as np

from mouse_behavior.behavior import collective_state
from mouse_behavior.behavior.collective_state import (
    CollectiveRuleConfig,
    bbox_contact_features,
    classify_collective_candidate,
    collective_candidate_windows,
    graph_density,
    extract_group_window_features,
    robust_body_centers,
    score_social_clustering,
    score_huddling,
)


def _keypoints_for_centers(centers: np.ndarray) -> np.ndarray:
    points = np.full((*centers.shape[:2], 7, 2), np.nan, dtype=float)
    offsets = np.asarray(
        [
            [3.0, 0.0],
            [2.0, -0.4],
            [2.0, 0.4],
            [1.0, 0.0],
            [-1.0, -0.5],
            [-1.0, 0.5],
            [-3.0, 0.0],
        ]
    )
    for frame in range(len(centers)):
        for mouse in range(centers.shape[1]):
            points[frame, mouse] = centers[frame, mouse] + offsets
    return points


def _boxes_for_centers(centers: np.ndarray) -> np.ndarray:
    boxes = np.empty((len(centers), centers.shape[1], 4), dtype=float)
    boxes[:, :, 0] = centers[:, :, 0] - 2.0
    boxes[:, :, 1] = centers[:, :, 1] - 1.5
    boxes[:, :, 2] = centers[:, :, 0] + 2.0
    boxes[:, :, 3] = centers[:, :, 1] + 1.5
    return boxes


def _candidate(
    centers: np.ndarray,
    fps: float,
    config: CollectiveRuleConfig,
    valid: np.ndarray | None = None,
):
    frames, mice = centers.shape[:2]
    velocity = np.zeros_like(centers)
    if frames > 1:
        velocity[1:] = np.diff(centers, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=2)
    return classify_collective_candidate(
        centers_cm=centers,
        valid=(
            np.ones((frames, mice), dtype=bool)
            if valid is None
            else valid
        ),
        speed_cm_s=speed,
        velocity_cm_s=velocity,
        member_slots=tuple(range(mice)),
        fps=fps,
        body_length_ref_cm=np.full(mice, 6.0),
        body_width_ref_cm=np.full(mice, 1.0),
        config=config,
        keypoints_cm=_keypoints_for_centers(centers),
        boxes_cm=_boxes_for_centers(centers),
    )


def _window_features(
    centers: np.ndarray,
    fps: float,
    config: CollectiveRuleConfig,
    valid: np.ndarray,
):
    velocity = np.zeros_like(centers)
    if len(centers) > 1:
        velocity[1:] = np.diff(centers, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=2)
    return extract_group_window_features(
        centers_cm=centers,
        valid=valid,
        speed_cm_s=speed,
        velocity_cm_s=velocity,
        member_slots=tuple(range(centers.shape[1])),
        fps=fps,
        body_length_ref_cm=np.full(centers.shape[1], 6.0),
        body_width_ref_cm=np.full(centers.shape[1], 1.0),
        start_frame=0,
        end_frame=len(centers) - 1,
        config=config,
        keypoints_cm=_keypoints_for_centers(centers),
        boxes_cm=_boxes_for_centers(centers),
    )


def test_robust_body_center_uses_partial_body_and_nan_fallback() -> None:
    points = np.full((2, 1, 7, 2), np.nan)
    points[0, 0, 3] = [1.0, 1.0]
    points[0, 0, 4] = [3.0, 1.0]
    points[1, 0, 0] = [9.0, 4.0]
    centers, valid = robust_body_centers(points)
    assert valid[:, 0].tolist() == [True, True]
    assert np.allclose(centers[0, 0], [2.0, 1.0])
    assert np.allclose(centers[1, 0], [9.0, 4.0])


def test_bbox_contact_features_reports_overlap_and_body_width_gap() -> None:
    iou, iom, overlap_left, overlap_right, gap, gap_bw = bbox_contact_features(
        np.asarray([0.0, 0.0, 4.0, 4.0]),
        np.asarray([2.0, 0.0, 6.0, 4.0]),
        2.0,
    )
    assert np.isclose(iou, 1.0 / 3.0)
    assert np.isclose(iom, 0.5)
    assert np.isclose(overlap_left, 0.5)
    assert np.isclose(overlap_right, 0.5)
    assert gap == 0.0
    assert gap_bw == 0.0
    assert graph_density(np.asarray([[False, True, False], [True, False, True], [False, True, False]])) == 2 / 3


def test_social_clustering_requires_contraction_and_inward_motion() -> None:
    fps = 10.0
    positions = np.linspace(
        np.asarray([[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0]]),
        np.asarray([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]]),
        51,
    ) * 1.3375
    config = CollectiveRuleConfig(
        social_nnd_drop_bw_min=0.2,
        social_radius_drop_bw_min=0.2,
        social_inward_fraction_min=0.4,
        social_active_fraction_min=0.3,
        social_soft_score_min=0.6,
    )
    result = _candidate(positions, fps, config)
    assert result.social_clustering_score > 0
    assert result.social_checks["nnd_contraction"]
    assert result.social_checks["radius_contraction"]
    assert result.social_checks["inward_motion"]
    assert result.social_checks["active_inward_member"]


def test_huddling_requires_compact_low_motion_stable_contact_graph() -> None:
    fps = 10.0
    centers = np.repeat(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]]]),
        31,
        axis=0,
    )
    result = _candidate(centers, fps, CollectiveRuleConfig())
    assert result.huddling_score > 0
    assert result.huddle_checks["compact_nnd"]
    assert result.huddle_checks["low_motion"]
    assert result.huddle_checks["graph_stability"]
    assert result.huddle_checks["huddling_at_clip_onset"]


def test_candidate_windows_reuse_pair_geometry_without_changing_features(monkeypatch) -> None:
    fps = 10.0
    frames = 31
    centers = np.repeat(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]]]),
        frames,
        axis=0,
    )
    valid = np.ones((frames, 3), dtype=bool)
    velocity = np.zeros_like(centers)
    speed = np.zeros((frames, 3), dtype=float)
    config = CollectiveRuleConfig()
    arguments = {
        "centers_cm": centers,
        "valid": valid,
        "speed_cm_s": speed,
        "velocity_cm_s": velocity,
        "member_slots": (0, 1, 2),
        "fps": fps,
        "body_length_ref_cm": np.full(3, 6.0),
        "body_width_ref_cm": np.full(3, 1.0),
        "config": config,
        "keypoints_cm": _keypoints_for_centers(centers),
        "boxes_cm": _boxes_for_centers(centers),
    }
    expected = collective_candidate_windows(**arguments)

    original = collective_state.pair_contact_features
    calls: set[tuple[int, int, int]] = set()

    def count_unique_pair_calculations(**kwargs):
        calls.add((kwargs["frame"], kwargs["mouse_i"], kwargs["mouse_j"]))
        return original(**kwargs)

    monkeypatch.setattr(collective_state, "pair_contact_features", count_unique_pair_calculations)
    pair_cache = {}
    actual = collective_candidate_windows(**arguments, pair_contact_cache=pair_cache)

    assert actual == expected
    assert len(calls) == frames * 3
    assert len(pair_cache) == len(calls)


def test_formation_phase_is_not_overwritten_by_a_later_stable_huddle() -> None:
    fps = 10.0
    frames = 121
    centers = np.empty((frames, 3, 2), dtype=float)
    separated = np.asarray([[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0]])
    compact = np.asarray([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    centers[:41] = np.linspace(separated, compact, 41)
    centers[41:] = compact

    result = _candidate(centers, fps, CollectiveRuleConfig())

    assert result.social_clustering_score > 0.0
    assert result.social_checks["nnd_contraction"]
    assert result.social_checks["radius_contraction"]
    assert result.social_checks["inward_motion"]
    assert result.social_checks["active_inward_member"]
    assert not result.huddle_checks["huddling_at_clip_onset"]
    assert result.huddling_score == 0.0


def test_social_clustering_can_be_supported_by_one_inward_member_in_a_large_group() -> None:
    fps = 10.0
    compact = np.asarray([[float(index), 0.0] for index in range(10)])
    centers = np.repeat(compact[None, :, :], 51, axis=0)
    config = CollectiveRuleConfig(social_active_inward_member_fraction_min=0.20)
    base = _window_features(
        centers,
        fps,
        config,
        np.ones((len(centers), len(compact)), dtype=bool),
    )
    social = replace(
        base.social,
        nnd_drop_bw=0.5,
        nnd_slope_bw_s=-0.1,
        radius_drop_bw=0.5,
        radius_slope_bw_s=-0.1,
        inward_motion_fraction=0.8,
        active_inward_member_fraction=0.1,
        active_member_fraction=0.5,
        density_slope_s=0.1,
        largest_component_growth=1.0,
    )
    graph = replace(base.graph, edge_gain_rate=0.1)
    features = replace(
        base,
        duration_s=5.1,
        group_visible_fraction=1.0,
        minimum_member_support_fraction=1.0,
        social=social,
        graph=graph,
    )

    score, checks = score_social_clustering(features, config)

    assert checks["active_inward_member"]
    assert score > 0.0


def test_huddling_present_at_onset_remains_primary_over_later_motion() -> None:
    fps = 10.0
    compact = np.asarray([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
    centers = np.repeat(compact[None, :, :], 121, axis=0)
    # Small later movement remains below the existing anti-attack limits and
    # must not turn an already compact, stable onset into a new cluster.
    centers[61:, 2, 0] += np.linspace(0.0, 1.0, 60)

    result = _candidate(centers, fps, CollectiveRuleConfig())

    assert result.huddling_score > 0.0
    assert result.huddle_checks["huddling_at_clip_onset"]


def test_huddling_rejects_violent_motion_even_when_mice_are_compact() -> None:
    fps = 10.0
    centers = np.repeat(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]]]),
        31,
        axis=0,
    )
    centers[:, 0, 0] = np.arange(len(centers)) * 8.025
    result = _candidate(centers, fps, CollectiveRuleConfig())
    assert result.huddling_score == 0
    assert not result.huddle_checks.get("not_attack", True)


def test_huddling_bridges_same_id_dropout_only_after_near_group_recovery() -> None:
    fps = 10.0
    centers = np.repeat(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]]]),
        31,
        axis=0,
    )
    valid = np.ones((len(centers), 3), dtype=bool)
    valid[10:20, 2] = False
    config = CollectiveRuleConfig(
        huddle_min_group_visible_fraction=0.60,
        huddle_min_member_support_fraction=0.80,
    )

    near = _window_features(centers, fps, config, valid)
    near_score, near_checks = score_huddling(near, config)
    assert near_score > 0
    assert near_checks["group_visibility"]
    assert near.huddle.recovery_near_group_fraction == 1.0
    assert near.huddle_minimum_member_support_fraction == 1.0
    assert near.huddle_group_visible_fraction == 1.0

    far_centers = centers.copy()
    far_centers[20:, 2] += np.asarray([30.0, 30.0])
    far = _window_features(far_centers, fps, config, valid)
    far_score, far_checks = score_huddling(far, config)
    assert far_score == 0
    assert far.huddle.recovery_near_group_fraction == 0.0
    assert far.huddle_minimum_member_support_fraction < 0.80
    assert not far_checks["member_support"]


def test_huddling_does_not_bridge_a_dropout_without_reacquisition() -> None:
    fps = 10.0
    centers = np.repeat(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]]]),
        31,
        axis=0,
    )
    valid = np.ones((len(centers), 3), dtype=bool)
    valid[10:, 2] = False
    config = CollectiveRuleConfig(
        huddle_min_group_visible_fraction=0.60,
        huddle_min_member_support_fraction=0.40,
    )
    result = _window_features(centers, fps, config, valid)
    score, checks = score_huddling(result, config)
    assert score == 0
    assert result.huddle_minimum_member_support_fraction < 0.40
    assert not checks["member_support"]
