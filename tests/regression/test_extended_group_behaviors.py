from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mouse_behavior.behavior import ethogram
from mouse_behavior.behavior.ethogram import (
    _extended_individual_and_group_events,
    _extended_pair_events,
)


def _pair_frame(count: int = 40) -> pd.DataFrame:
    values = {
        "frame": np.arange(count),
        "pair_key": ["0_1"] * count,
        "mouse_a_id": np.zeros(count, dtype=int),
        "mouse_b_id": np.ones(count, dtype=int),
        "valid_pair": np.ones(count, dtype=bool),
        "center_distance_cm": np.full(count, 12.0),
        "selected_actor_id": np.zeros(count, dtype=int),
        "selected_target_id": np.ones(count, dtype=int),
        "selected_actor_behavior_speed_cm_s": np.full(count, 6.0),
        "selected_target_behavior_speed_cm_s": np.full(count, 5.0),
        "selected_distance_drop_cm": np.zeros(count),
        "selected_closing_speed_cm_s": np.zeros(count),
        "selected_target_escape_alignment": np.full(count, 0.05),
        "selected_actor_pursuit_alignment": np.full(count, 0.80),
        "a_to_b_actor_behavior_speed_cm_s": np.full(count, 6.0),
        "a_to_b_target_behavior_speed_cm_s": np.full(count, 5.0),
        "a_to_b_direction_similarity": np.full(count, 0.90),
        "a_to_b_pursuit_alignment": np.full(count, 0.80),
        "a_to_b_target_escape_alignment": np.full(count, 0.05),
        "a_to_b_actor_behind_target": np.ones(count, dtype=bool),
        "b_to_a_actor_behavior_speed_cm_s": np.full(count, 5.0),
        "b_to_a_target_behavior_speed_cm_s": np.full(count, 6.0),
        "b_to_a_direction_similarity": np.full(count, 0.90),
        "b_to_a_pursuit_alignment": np.full(count, 0.10),
        "b_to_a_target_escape_alignment": np.full(count, 0.05),
        "b_to_a_actor_behind_target": np.zeros(count, dtype=bool),
    }
    return pd.DataFrame(values)


def _group_kinematics(centers: np.ndarray, fps: float = 10.0) -> dict[str, np.ndarray]:
    frames, mice, _ = centers.shape
    velocity = np.zeros_like(centers, dtype=float)
    velocity[1:] = (centers[1:] - centers[:-1]) * fps
    speed = np.linalg.norm(velocity, axis=2)
    return {
        "valid": np.ones((frames, mice), dtype=bool),
        "behavior_speed": speed,
        "pose_quality": np.ones((frames, mice), dtype=float),
        "centers_cm": centers,
        "velocity": velocity,
        "body_cm": np.full((frames, mice), 8.0),
        "reference_body_cm": 8.0,
    }


def _group_events(centers: np.ndarray, config: dict) -> list[dict]:
    return _extended_individual_and_group_events(
        _group_kinematics(centers),
        pair_metrics={},
        pair_i=np.asarray([], dtype=int),
        pair_j=np.asarray([], dtype=int),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config=config,
    )


def test_following_has_follower_leader_roles_and_requires_three_seconds():
    pair_df = _pair_frame()

    events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={},
    )

    following = [event for event in events if event["behavior"] == "following"]
    assert len(following) == 1
    assert following[0]["actor_id"] == 0
    assert following[0]["target_id"] == 1
    assert following[0]["actor_role"] == "follower"
    assert following[0]["target_role"] == "leader"
    assert following[0]["core_duration_s"] >= 3.0


def test_following_rejects_existing_chase_and_short_bouts():
    pair_df = _pair_frame()
    chase = pd.DataFrame(
        {"weak_standard_final_chase": np.ones(len(pair_df), dtype=bool)},
        index=pair_df.index,
    )

    chase_events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=chase,
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={},
    )
    short_events = _extended_pair_events(
        pair_df.iloc[:29].copy(),
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index[:29]),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={},
    )

    assert not any(event["behavior"] == "following" for event in chase_events)
    assert not any(event["behavior"] == "following" for event in short_events)


def test_following_rejects_semantic_chase_spans(monkeypatch: pytest.MonkeyPatch):
    pair_df = _pair_frame()
    semantic_chase = {
        "behavior": "chase",
        "analysis_start_frame": 0,
        "analysis_end_frame": len(pair_df) - 1,
    }
    monkeypatch.setattr(
        ethogram,
        "_semantic_extended_pair_events",
        lambda *args, **kwargs: [semantic_chase],
    )

    events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "social": {"semantic_fsm": {"enabled": True}},
            }
        },
    )

    assert events == [semantic_chase]


def test_together_requires_both_pair_members_to_be_stationary():
    pair_df = _pair_frame()
    pair_df["center_distance_cm"] = 4.0
    events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={},
    )
    assert any(event["behavior"] == "together" for event in events)

    pair_df["center_distance_cm"] = 7.9
    within_together_range = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={},
    )
    assert any(event["behavior"] == "together" for event in within_together_range)

    pair_df["center_distance_cm"] = 8.0
    at_together_boundary = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={},
    )
    assert not any(event["behavior"] == "together" for event in at_together_boundary)

    pair_df["center_distance_cm"] = 4.0
    pair_df["selected_actor_behavior_speed_cm_s"] = 15.0
    pair_df["selected_target_behavior_speed_cm_s"] = 0.0
    moving_events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "social": {"together_max_individual_speed_cm_s": 10.0}
            }
        },
    )
    assert not any(event["behavior"] == "together" for event in moving_events)


def test_together_speed_limit_is_independent_of_individual_static_limit():
    pair_df = _pair_frame()
    pair_df["center_distance_cm"] = 4.0
    pair_df["selected_actor_behavior_speed_cm_s"] = 8.0
    pair_df["selected_target_behavior_speed_cm_s"] = 0.0

    events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "individual": {"stationary_max_speed_cm_s": 5.0},
                "social": {"together_max_individual_speed_cm_s": 10.0},
            }
        },
    )

    assert any(event["behavior"] == "together" for event in events)


def test_together_rejects_sustained_speed_above_low_motion_limit():
    pair_df = _pair_frame()
    pair_df["center_distance_cm"] = 4.0
    pair_df["selected_actor_behavior_speed_cm_s"] = 10.1
    pair_df["selected_target_behavior_speed_cm_s"] = 0.0

    events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={"extended_behavior": {"social": {"together_max_individual_speed_cm_s": 10.0}}},
    )

    assert not any(event["behavior"] == "together" for event in events)


def test_approach_uses_configured_terminal_distance_not_pair_candidate_radius():
    pair_df = _pair_frame()
    pair_df["center_distance_cm"] = 12.0
    pair_df["selected_distance_drop_cm"] = 2.0

    events = _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "social": {
                    "pair_max_distance_cm": 5.0,
                    "approach_terminal_distance_cm": 17.0,
                }
            }
        },
    )

    assert any(event["behavior"] == "approach" for event in events)


def test_group_locomotion_requires_three_coordinated_mice():
    frames = 40
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, :, 0] = np.arange(frames)[:, None] * 0.5
    centers[:, :, 1] = np.asarray([0.0, 6.0, 12.0])

    events = _group_events(centers, {})

    locomotion = [event for event in events if event["behavior"] == "group_locomotion"]
    assert len(locomotion) == 1
    assert locomotion[0]["member_ids"] == [0, 1, 2]
    assert locomotion[0]["core_duration_s"] >= 3.0


def test_fast_dense_group_is_locomotion_not_huddling():
    frames = 40
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, :, 0] = np.arange(frames)[:, None] * 1.2
    centers[:, :, 1] = np.asarray([0.0, 2.0, 4.0])

    events = _group_events(centers, {})

    assert any(event["behavior"] == "group_locomotion" for event in events)
    assert not any(event["behavior"] == "huddle" for event in events)


def test_stationary_dense_triple_is_huddling():
    centers = np.zeros((40, 3, 2), dtype=float)
    centers[:, :, 1] = np.asarray([0.0, 2.0, 4.0])

    events = _group_events(centers, {})

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert len(huddles) == 1
    assert huddles[0]["member_ids"] == [0, 1, 2]


def test_social_clustering_ends_when_formation_motion_stops():
    frames = 100
    # The group approaches over six seconds, then remains tightly grouped but
    # stationary. The latter phase belongs to Huddling, not Social clustering.
    spacing = np.concatenate(
        (np.full(10, 24.0), np.linspace(24.0, 2.0, 61)[1:], np.full(30, 2.0))
    )
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, 1, 0] = spacing
    centers[:, 2, 0] = spacing * 2.0

    events = _group_events(centers, {})

    clustering = [event for event in events if event["behavior"] == "social_clustering"]
    assert len(clustering) == 1
    assert clustering[0]["member_ids"] == [0, 1, 2]
    assert clustering[0]["core_duration_s"] >= 5.0
    assert clustering[0]["analysis_end_frame"] < 75
    assert any(event["behavior"] == "huddle" for event in events)


def test_slow_group_formation_does_not_count_as_social_clustering():
    frames = 100
    spacing = np.concatenate(
        (np.full(10, 24.0), np.linspace(24.0, 18.0, 61)[1:], np.full(30, 18.0))
    )
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, 1, 0] = spacing
    centers[:, 2, 0] = spacing * 2.0

    events = _group_events(centers, {})

    assert not any(event["behavior"] == "social_clustering" for event in events)


def test_dispersal_requires_prior_cluster_and_ten_second_separation():
    frames = 150
    spacing = np.concatenate(
        (
            np.full(20, 8.0),
            np.linspace(8.0, 35.0, 21)[1:],
            np.full(frames - 40, 35.0),
        )
    )
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, 1, 0] = spacing
    centers[:, 2, 0] = spacing * 2.0

    events = _group_events(centers, {})

    dispersal = [event for event in events if event["behavior"] == "dispersal"]
    assert len(dispersal) == 1
    assert dispersal[0]["member_ids"] == [0, 1, 2]
    assert dispersal[0]["core_duration_s"] >= 10.0


def test_dispersal_rejects_a_group_that_was_already_spread_out():
    frames = 150
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, 1, 0] = 35.0
    centers[:, 2, 0] = 70.0

    events = _group_events(centers, {})

    assert not any(event["behavior"] == "dispersal" for event in events)
