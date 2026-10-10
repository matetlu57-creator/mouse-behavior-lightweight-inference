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


def test_following_suspended_and_same_input_emits_no_following_events():
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复：原 follower/leader 角色断言改为暂停回归断言。
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

    assert [event for event in events if event["behavior"] == "following"] == []
    assert [event for event in events if event["behavior"] == "avoidance"] == []


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


_TOGETHER_BAND_CONFIG = {
    # [STEP6 2026-10-05] Together: 中心距离带 [3,6] cm。测试显式给出新默认值，
    # 不依赖 ethogram 内置回退值。
    "extended_behavior": {
        "social": {"together_max_distance_cm": 6.0, "together_min_distance_cm": 3.0}
    },
}


def _together_events(pair_df: pd.DataFrame, config: dict | None = None) -> list[dict]:
    return _extended_pair_events(
        pair_df,
        metrics={},
        pair_index=0,
        enriched=pd.DataFrame(index=pair_df.index),
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        config=_TOGETHER_BAND_CONFIG if config is None else config,
    )


def _has_together(events: list[dict]) -> bool:
    return any(event["behavior"] == "together" for event in events)


def test_together_uses_three_to_six_cm_distance_band():
    # [STEP6 2026-10-05] Together 由单一 8 cm 上限改为 [3,6] cm 中心距离带。
    pair_df = _pair_frame()

    pair_df["center_distance_cm"] = 4.0
    assert _has_together(_together_events(pair_df))

    # 上边界：6 cm 含，6.1/7 cm 不含（旧 8 cm cap 已删除）。
    pair_df["center_distance_cm"] = 6.0
    assert _has_together(_together_events(pair_df))
    pair_df["center_distance_cm"] = 6.1
    assert not _has_together(_together_events(pair_df))
    pair_df["center_distance_cm"] = 7.0
    assert not _has_together(_together_events(pair_df))

    # 下边界：3 cm 含；2.9 cm 落在 [2,3) 空隙里不属 Together；
    # <2 cm 的接触帧仍然容忍为 Together。
    pair_df["center_distance_cm"] = 3.0
    assert _has_together(_together_events(pair_df))
    pair_df["center_distance_cm"] = 2.9
    assert not _has_together(_together_events(pair_df))
    pair_df["center_distance_cm"] = 1.9
    assert _has_together(_together_events(pair_df))


def test_together_is_independent_of_individual_static_limit():
    # [STEP6 2026-10-05] Together 不再复用个体 Static 上限：个体层已判为
    # Locomotion 的速度（>stationary_max_speed_cm_s）也不得阻止 Together。
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
                "social": {
                    "together_max_distance_cm": 6.0,
                    "together_min_distance_cm": 3.0,
                },
            }
        },
    )

    assert _has_together(events)


def test_together_no_longer_rejects_sustained_speed():
    # [STEP6 2026-10-05] Together 速度门已删除（门不存在回归断言）：
    # 持续高速不再是 Together 的排除条件，距离带是唯一判据。
    pair_df = _pair_frame()
    pair_df["center_distance_cm"] = 4.0
    pair_df["selected_actor_behavior_speed_cm_s"] = 10.1
    pair_df["selected_target_behavior_speed_cm_s"] = 0.0
    assert _has_together(_together_events(pair_df))

    pair_df["selected_actor_behavior_speed_cm_s"] = 15.0
    pair_df["selected_target_behavior_speed_cm_s"] = 12.0
    assert _has_together(_together_events(pair_df))

    # 高速不能救回距离带：离开 [3,6] cm 仍然不成立。
    pair_df["center_distance_cm"] = 7.0
    assert not _has_together(_together_events(pair_df))


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


def test_fast_dense_group_is_huddle_and_locomotion_in_parallel_layers():
    # [STEP6 2026-10-05] Huddling 删除全部速度门：快速密集群同时构成 Huddling
    # （群体层）与 Group locomotion / 个体 Locomotion（并行层，互不删除）。
    frames = 40
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, :, 0] = np.arange(frames)[:, None] * 1.605
    centers[:, :, 1] = np.asarray([0.0, 2.675, 5.35])

    events = _group_events(centers, {})

    assert any(event["behavior"] == "huddle" for event in events)
    assert any(event["behavior"] == "group_locomotion" for event in events)
    assert any(event["behavior"] == "locomotion" for event in events)


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
    spacing = np.concatenate((np.full(10, 24.0), np.linspace(24.0, 2.0, 61)[1:], np.full(30, 2.0)))
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
