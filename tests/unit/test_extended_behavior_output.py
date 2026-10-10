from __future__ import annotations

import numpy as np
from pathlib import Path

from mouse_behavior.annotation_website_export import build_annotations
from mouse_behavior.behavior.ethogram import (
    _extended_individual_and_group_events,
    _sustained_member_ids_by_frame,
)
from mouse_behavior.visualization.overlay import build_mouse_overlays, normalize_focus_behavior


def _tracks(frames: int = 20, mice: int = 3) -> dict[str, np.ndarray]:
    centers = np.zeros((frames, mice, 2), dtype=float)
    centers[:, :, 0] = np.arange(mice, dtype=float) * 10.0
    return {
        "valid": np.ones((frames, mice), dtype=bool),
        "centers_px": centers,
    }


def test_isolation_member_duration_counts_evidence_across_missing_id_only():
    members = [(0,) for _ in range(50)]
    members[24:26] = [(), ()]
    missing = np.zeros((50, 1), dtype=bool)
    missing[24:26] = True

    sustained = _sustained_member_ids_by_frame(
        members,
        frames=50,
        mice=1,
        min_duration_frames=50,
        max_gap_frames=2,
        eligible_gap_by_frame=missing,
    )
    assert all(not frame for frame in sustained)

    extended = members + [(0,), (0,)]
    extended_missing = np.vstack((missing, np.zeros((2, 1), dtype=bool)))
    sustained = _sustained_member_ids_by_frame(
        extended,
        frames=52,
        mice=1,
        min_duration_frames=50,
        max_gap_frames=2,
        eligible_gap_by_frame=extended_missing,
    )
    assert all(frame == (0,) for frame in sustained)

    extended_missing[24] = False
    sustained = _sustained_member_ids_by_frame(
        extended,
        frames=52,
        mice=1,
        min_duration_frames=50,
        max_gap_frames=2,
        eligible_gap_by_frame=extended_missing,
    )
    assert all(not frame for frame in sustained)


def test_renderer_suspends_following_roles_and_shows_group_members():
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复：following 事件不再产生角色文案。
    following = {
        "behavior": "following",
        "actor_id": 0,
        "target_id": 1,
        "pair_key": "0_1",
        "peak_score": 0.9,
    }
    pair_overlays = build_mouse_overlays([following], "social", [0, 1])

    assert all("跟随" not in overlay.text for overlay in pair_overlays.values())

    group = {
        "behavior": "group_locomotion",
        "member_ids": [0, 1, 2],
        "peak_score": 0.9,
    }
    group_overlays = build_mouse_overlays([group], "group", [0, 1, 2])

    assert all("群体：群体同步运动" in overlay.text for overlay in group_overlays.values())
    assert normalize_focus_behavior("社会聚集") == "social_clustering"
    assert normalize_focus_behavior("群体分散") == "dispersal"


def test_website_export_includes_all_new_behavior_names_and_participants():
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复：following 事件从网站导出用例移除。
    events = [
        *[
            {
                "behavior": behavior,
                "member_ids": [0, 1, 2],
                "pair_key": "group",
                "start_frame": 0,
                "end_frame": 9,
                "candidate_level": "extended",
                "peak_score": 0.9,
                "role_ambiguous": False,
            }
            for behavior in ("group_locomotion", "social_clustering", "dispersal")
        ],
    ]
    # A lineage can accumulate a historical member in member_ids. The website
    # annotation should describe the group at the representative peak frame.
    events[1]["member_ids"] = [0, 1, 2, 3]
    events[1]["member_ids_at_peak"] = [0, 1, 2]

    annotations, skipped = build_annotations(
        events,
        [],
        _tracks(mice=4),
        fps=10.0,
        frame_count=20,
        huddle_distance_cm=5.0,
        cm_per_pixel=1.0,
    )

    assert skipped == []
    by_name = {annotation["behavior"]: annotation for annotation in annotations}
    assert "跟随" not in by_name
    assert by_name["群体同步运动"]["mouse_ids"] == [0, 1, 2]
    assert by_name["社会聚集"]["mouse_ids"] == [0, 1, 2]
    assert by_name["群体分散"]["mouse_ids"] == [0, 1, 2]


def _individual_events(speed: np.ndarray, frames: int, individual_cfg: dict) -> list[dict]:
    """Run only the individual/group ethogram over a synthetic one-mouse track."""

    kin = {
        "valid": np.ones((frames, 1), dtype=bool),
        "behavior_speed": speed,
        "pose_quality": np.ones((frames, 1), dtype=float),
        "centers_cm": np.zeros((frames, 1, 2), dtype=float),
        "body_cm": np.full((frames, 1), 8.0, dtype=float),
        "reference_body_cm": 8.0,
    }
    config = {"extended_behavior": {"individual": individual_cfg}}
    return _extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.empty(0, dtype=int),
        pair_j=np.empty(0, dtype=int),
        source_video=Path("synthetic.mp4"),
        source_fps=30.0,
        sample_stride=1,
        config=config,
    )


def test_locomotion_spans_the_whole_merged_speed_band():
    """One merged label covers the former Walking *and* Running speed ranges."""

    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    frames = 9
    speed = np.array([84.9] * 3 + [85.0] * 3 + [85.1] * 3, dtype=float)[:, None]
    events = _individual_events(speed, frames, {
        "stationary_max_speed_cm_s": 4.0,
        # running_min_speed_cm_s is retained as the locomotion score normaliser
        # and the Chasing high-speed floor; it no longer splits individual labels.
        "running_min_speed_cm_s": 85.0,
        "locomotion_min_duration_seconds": 0.05,
        "confirm_seconds": 0.05,
        "fill_gap_seconds": 0.0,
        "min_pose_quality": 0.2,
    })

    by_behavior = {event["behavior"]: event for event in events}
    # The old 85.0 cm/s Walking/Running boundary is gone: frames 0-8 are one event.
    assert by_behavior["locomotion"]["start_frame"] == 0
    assert by_behavior["locomotion"]["end_frame"] == 8
    assert sum(event["behavior"] == "locomotion" for event in events) == 1
    for deleted in ("walking", "running"):
        assert deleted not in by_behavior, f"{deleted} must stay deleted"


def test_locomotion_boundary_is_only_the_static_ceiling():
    """The single individual speed gate is `speed > stationary_max_speed_cm_s`."""

    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    frames = 5
    speed = np.array([1.0, 4.0, 4.1, 85.0, 200.0], dtype=float)[:, None]
    events = _individual_events(speed, frames, {
        "stationary_max_speed_cm_s": 4.0,
        "running_min_speed_cm_s": 85.0,
        "locomotion_min_duration_seconds": 0.05,
        "stationary_min_duration_seconds": 0.05,
        "confirm_seconds": 0.05,
        "fill_gap_seconds": 0.0,
        "min_pose_quality": 0.2,
    })

    by_behavior = {event["behavior"]: event for event in events}
    assert by_behavior["stationary"]["start_frame"] == 0
    assert by_behavior["stationary"]["end_frame"] == 1
    # Frames 2-4 stay one Locomotion event even though they cross the retained
    # 85.0 cm/s normaliser, proving no second individual boundary exists.
    assert by_behavior["locomotion"]["start_frame"] == 2
    assert by_behavior["locomotion"]["end_frame"] == 4
    assert sum(event["behavior"] == "locomotion" for event in events) == 1


def test_isolation_does_not_average_distance_to_a_far_companion():
    frames, mice = 12, 10
    centers = np.zeros((frames, mice, 2), dtype=float)
    centers[:, :8, 0] = np.arange(8, dtype=float)
    centers[:, 8, 0] = 100.0
    centers[:, 9, 0] = 110.0
    kin = {
        "valid": np.ones((frames, mice), dtype=bool),
        "behavior_speed": np.zeros((frames, mice), dtype=float),
        "pose_quality": np.ones((frames, mice), dtype=float),
        "centers_cm": centers,
        "body_cm": np.full((frames, mice), 8.0, dtype=float),
        "reference_body_cm": 8.0,
    }

    events = _extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.empty(0, dtype=int),
        pair_j=np.empty(0, dtype=int),
        source_video=Path("synthetic.mp4"),
        source_fps=1.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "individual": {"stationary_max_speed_cm_s": 10.0},
                "group": {
                    "huddle_distance_cm": 5.0,
                    "isolation_distance_cm": 20.0,
                    "isolation_min_duration_seconds": 10.0,
                    "isolation_min_cluster_size": 3,
                    "isolation_min_main_cluster_fraction": 0.60,
                    "isolation_fill_gap_seconds": 0.0,
                },
            }
        },
    )

    isolation_events = [event for event in events if event["behavior"] == "isolation"]
    # The far pair members are only 10 cm apart. Their distance to the main
    # group is large, but each one's nearest companion remains below 20 cm.
    assert not isolation_events
