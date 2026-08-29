from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import numpy as np


def load_lightweight():
    return importlib.import_module("mouse_behavior.lightweight_behavior_inference")


def contact_frame(frame: int, head: float, tail: float) -> dict[str, object]:
    return {
        "frame": frame,
        "valid_pair": True,
        "mouse_a_id": 1,
        "mouse_b_id": 2,
        "a_to_b_nose_head_distance_cm": head,
        "a_to_b_nose_tail_distance_cm": tail,
        "b_to_a_nose_head_distance_cm": float("inf"),
        "b_to_a_nose_tail_distance_cm": float("inf"),
    }


def test_nose_head_and_nose_tail_contacts_are_separate_events():
    lightweight = load_lightweight()
    pair_df = pd.DataFrame(
        [
            contact_frame(0, 2.0, 5.0),
            contact_frame(1, 2.2, 5.1),
            contact_frame(2, 5.0, 2.0),
            contact_frame(3, 5.1, 2.1),
            contact_frame(4, 8.0, 8.0),
        ]
    )
    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={
            "enabled": True,
            "nose_head_distance_cm": 3.0,
            "nose_tail_distance_cm": 3.0,
        },
    )

    assert [event["contact_type"] for event in events] == ["nose_head", "nose_tail"]
    assert [(event["start_frame"], event["end_frame"]) for event in events] == [
        (0, 1),
        (2, 3),
    ]
    assert all(event["contact_actor_id"] == 1 for event in events)
    assert all(event["contact_target_id"] == 2 for event in events)


def test_simultaneous_head_and_tail_contact_keeps_both_components():
    lightweight = load_lightweight()
    pair_df = pd.DataFrame([contact_frame(4, 2.0, 2.0)])
    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True},
    )

    assert len(events) == 1
    assert events[0]["contact_type"] == "nose_head_and_nose_tail"
    assert events[0]["contact_type_components"] == "nose_head;nose_tail"


def test_contact_direction_flip_does_not_fragment_one_nose_event():
    lightweight = load_lightweight()
    rows = [contact_frame(0, 2.0, 8.0), contact_frame(1, 2.1, 8.0)]
    rows.extend(
        [
            {
                **contact_frame(2, 8.0, 8.0),
                "b_to_a_nose_head_distance_cm": 2.0,
            },
            {
                **contact_frame(3, 8.0, 8.0),
                "b_to_a_nose_head_distance_cm": 2.1,
            },
        ]
    )

    events = lightweight._extract_contact_events(
        pd.DataFrame(rows),
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True, "nose_head_min_duration_seconds": 0.0},
    )

    assert len(events) == 1
    assert (events[0]["start_frame"], events[0]["end_frame"]) == (0, 3)
    assert events[0]["contact_direction"] == "both"
    assert events[0]["contact_actor_id"] == -1
    assert events[0]["contact_target_id"] == -1
    assert events[0]["role_ambiguous"] is True


def test_tolerant_head_contact_bridges_short_missing_observations():
    lightweight = load_lightweight()
    rows = []
    for frame in range(36):
        row = contact_frame(frame, 4.2, 8.0)
        if frame in {16, 17}:
            row["valid_pair"] = False
        rows.append(row)

    events = lightweight._extract_contact_events(
        pd.DataFrame(rows),
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={
            "enabled": True,
            "nose_head_distance_cm": 3.0,
            "nose_head_distance_multiplier": 1.5,
            "nose_head_min_duration_seconds": 1.0,
            "fill_gap_seconds": 0.10,
        },
    )

    assert len(events) == 1
    assert events[0]["contact_type"] == "nose_head"
    assert (events[0]["start_frame"], events[0]["end_frame"]) == (0, 35)
    assert events[0]["duration_s"] >= 1.0


def test_contact_extraction_avoids_full_dataframe_record_materialization(monkeypatch):
    lightweight = load_lightweight()
    pair_df = pd.DataFrame(
        [
            contact_frame(0, 8.0, 8.0),
            contact_frame(1, 2.0, 8.0),
            contact_frame(2, 8.0, 8.0),
        ]
    )

    def fail_to_dict(*args, **kwargs):
        raise AssertionError("contact extraction must not materialize all rows")

    monkeypatch.setattr(pd.DataFrame, "to_dict", fail_to_dict)
    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True},
    )

    assert [(event["start_frame"], event["end_frame"]) for event in events] == [(1, 1)]


def test_bidirectional_contact_remains_role_ambiguous():
    lightweight = load_lightweight()
    row = contact_frame(7, 2.0, 8.0)
    row["b_to_a_nose_head_distance_cm"] = 8.0
    row["b_to_a_nose_tail_distance_cm"] = 2.0

    events = lightweight._extract_contact_events(
        pd.DataFrame([row]),
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True},
    )

    assert len(events) == 1
    assert events[0]["contact_direction"] == "both"
    assert events[0]["contact_actor_id"] == -1
    assert events[0]["contact_target_id"] == -1
    assert events[0]["role_ambiguous"] is True


def test_invalid_endpoint_id_keeps_both_contact_roles_unknown():
    lightweight = load_lightweight()
    row = contact_frame(7, 2.0, 8.0)
    row["mouse_b_id"] = np.nan

    events = lightweight._extract_contact_events(
        pd.DataFrame([row]),
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True},
    )

    assert len(events) == 1
    assert events[0]["contact_actor_id"] == -1
    assert events[0]["contact_target_id"] == -1
    assert events[0]["role_ambiguous"] is True


def test_disabled_parallel_fsm_suppresses_contact_events():
    lightweight = load_lightweight()
    pair_df = pd.DataFrame([contact_frame(0, 2.0, 8.0)])

    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("contact.mp4"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={"enabled": True},
        fsm_coordinator=lightweight.ParallelBehaviorFSM({"enabled": False, "mode": "active"}),
    )

    assert events == []


def test_disabled_parallel_fsm_suppresses_individual_and_group_events():
    lightweight = load_lightweight()
    frames = 12
    kin = {
        "valid": np.ones((frames, 1), dtype=bool),
        "behavior_speed": np.zeros((frames, 1), dtype=float),
        "pose_quality": np.ones((frames, 1), dtype=float),
        "centers_cm": np.zeros((frames, 1, 2), dtype=float),
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("disabled.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {"enabled": True},
            "parallel_behavior_fsm": {"enabled": False, "mode": "active"},
        },
    )

    assert events == []


def test_extended_individual_and_group_events_keep_scopes_separate():
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 3), dtype=bool)
    centers = np.zeros((frames, 3, 2), dtype=float)
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([2.0, 0.0])
    centers[:, 2] = np.array([4.0, 0.0])
    # Keep the valid three-mouse aggregation for the full test window. A
    # short six-frame cluster would be below the configured huddle duration.
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 3), dtype=float),
        "pose_quality": np.ones((frames, 3), dtype=float),
        "centers_cm": centers,
    }
    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([0, 0, 1]),
        pair_j=np.array([1, 2, 2]),
        source_video=Path("example.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={"extended_behavior": {"enabled": True}},
    )
    assert events
    assert {event["event_scope"] for event in events} == {"individual", "group"}
    assert any(event["behavior"] == "stationary" for event in events)
    assert any(event["behavior"] == "huddle" for event in events)


def test_huddle_uses_local_cluster_in_multi_mouse_scene():
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 8), dtype=bool)
    centers = np.zeros((frames, 8, 2), dtype=float)
    # Five mice form a dense local cluster; three visible mice remain apart.
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([2.0, 0.0])
    centers[:, 2] = np.array([0.0, 2.0])
    centers[:, 3] = np.array([2.0, 2.0])
    centers[:, 4] = np.array([1.0, 1.0])
    centers[:, 5] = np.array([40.0, 0.0])
    centers[:, 6] = np.array([60.0, 0.0])
    centers[:, 7] = np.array([80.0, 0.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 8), dtype=float),
        "pose_quality": np.ones((frames, 8), dtype=float),
        "centers_cm": centers,
    }
    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("cluster.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={"extended_behavior": {"enabled": True}},
    )
    assert any(event["behavior"] == "huddle" for event in events)
    huddle = next(event for event in events if event["behavior"] == "huddle")
    assert set(huddle["member_ids"]) == {0, 1, 2, 3, 4}
    assert huddle["member_trace"] == [
        {"member_ids": [0, 1, 2, 3, 4], "start_frame": 0, "end_frame": 11}
    ]


def test_huddle_keeps_large_local_group_when_diagonal_pairs_are_far_apart() -> None:
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 8), dtype=bool)
    centers = np.zeros((frames, 8, 2), dtype=float)
    # This is a locally packed row: each mouse has at least two neighbours
    # within 11 cm, but the two end points are 35 cm apart. A global all-pairs
    # density gate would reject the group even though the local huddle is valid.
    centers[:] = np.asarray([[5.0 * index, 0.0] for index in range(8)], dtype=float)
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 8), dtype=float),
        "pose_quality": np.ones((frames, 8), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("large_local_huddle.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "group": {
                    "huddle_distance_cm": 11.0,
                    "huddle_max_pair_distance_body_lengths": None,
                    "huddle_density_mode": "local",
                    "huddle_min_duration_seconds": 0.3,
                },
            }
        },
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert huddles
    assert set(huddles[0]["member_ids"]) == set(range(8))


def test_huddle_mixed_body_sizes_do_not_shrink_calibrated_distance_gate() -> None:
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 3), dtype=bool)
    centers = np.zeros((frames, 3, 2), dtype=float)
    # The inferred body lengths are intentionally heterogeneous. The three
    # centres are within 11 cm, while a median-body-length cap would shrink the
    # effective gate and incorrectly reject this huddle.
    centers[:] = np.asarray([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]], dtype=float)
    body_cm = np.tile(np.asarray([[5.0, 5.0, 12.0]], dtype=float), (frames, 1))
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 3), dtype=float),
        "pose_quality": np.ones((frames, 3), dtype=float),
        "centers_cm": centers,
        "body_cm": body_cm,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("mixed_body_sizes.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "group": {
                    "huddle_distance_cm": 11.0,
                    "huddle_max_pair_distance_body_lengths": 1.75,
                    "huddle_body_length_cap_enabled": False,
                    "huddle_density_mode": "local",
                    "huddle_min_duration_seconds": 0.3,
                },
            }
        },
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert huddles
    assert set(huddles[0]["member_ids"]) == {0, 1, 2}


def test_huddle_rejects_a_three_mouse_connected_chain() -> None:
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 3), dtype=bool)
    centers = np.zeros((frames, 3, 2), dtype=float)
    # Adjacent links are below the five-centimetre threshold, but the two
    # endpoints are eight centimetres apart. This is a connected chain, not
    # a three-mouse local aggregation.
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([4.0, 0.0])
    centers[:, 2] = np.array([8.0, 0.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 3), dtype=float),
        "pose_quality": np.ones((frames, 3), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("chain.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={"extended_behavior": {"enabled": True}},
    )

    assert not any(event["behavior"] == "huddle" for event in events)


def test_huddle_rejects_changing_local_memberships_without_stability() -> None:
    lightweight = load_lightweight()
    frames = 30
    valid = np.ones((frames, 4), dtype=bool)
    centers = np.zeros((frames, 4, 2), dtype=float)
    for block_start in range(0, frames, 5):
        block_end = block_start + 5
        if (block_start // 5) % 2 == 0:
            centers[block_start:block_end, 0] = np.array([0.0, 0.0])
            centers[block_start:block_end, 1] = np.array([2.0, 0.0])
            centers[block_start:block_end, 2] = np.array([4.0, 0.0])
            centers[block_start:block_end, 3] = np.array([20.0, 0.0])
        else:
            centers[block_start:block_end, 0] = np.array([20.0, 0.0])
            centers[block_start:block_end, 1] = np.array([0.0, 0.0])
            centers[block_start:block_end, 2] = np.array([2.0, 0.0])
            centers[block_start:block_end, 3] = np.array([4.0, 0.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 4), dtype=float),
        "pose_quality": np.ones((frames, 4), dtype=float),
        "centers_cm": centers,
    }
    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("changing_huddle.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "fill_gap_seconds": 0.0,
                },
            }
        },
    )

    assert not any(event["behavior"] == "huddle" for event in events)


def test_huddle_survives_short_single_member_tracking_flicker() -> None:
    lightweight = load_lightweight()
    frames = 60
    valid = np.ones((frames, 4), dtype=bool)
    centers = np.zeros((frames, 4, 2), dtype=float)
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([2.0, 0.0])
    centers[:, 2] = np.array([4.0, 0.0])
    centers[:, 3] = np.array([30.0, 0.0])
    # A short detector hand-off replaces only the third member. The two-frame
    # substitution must not erase a one-second physical huddle, and the
    # transient replacement ID must not leak into the rendered member list.
    for start in range(10, frames, 15):
        centers[start : start + 2, 2] = np.array([30.0, 0.0])
        centers[start : start + 2, 3] = np.array([4.0, 0.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 4), dtype=float),
        "pose_quality": np.ones((frames, 4), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("stable_huddle_with_id_flicker.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "fill_gap_seconds": 0.20,
                },
            }
        },
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert huddles
    assert set(huddles[0]["member_ids"]) == {0, 1, 2}
    assert 3 not in huddles[0]["member_ids"]


def test_huddle_does_not_merge_unrelated_groups_across_fillable_pair_gap() -> None:
    lightweight = load_lightweight()
    frames = 66
    valid = np.ones((frames, 8), dtype=bool)
    centers = np.zeros((frames, 8, 2), dtype=float)
    centers[:] = np.asarray(
        [[30.0 * mouse_id, 20.0] for mouse_id in range(8)],
        dtype=float,
    )

    # Group 0/1/2 huddles for one second.  A six-frame two-mouse interval is
    # exactly within the configured generic FSM gap budget, then a completely
    # unrelated group 5/6/7 huddles.  These must remain two group events.
    centers[:36, 0] = np.array([0.0, 0.0])
    centers[:36, 1] = np.array([2.0, 0.0])
    centers[:30, 2] = np.array([0.0, 2.0])
    centers[36:, 5] = np.array([0.0, 0.0])
    centers[36:, 6] = np.array([2.0, 0.0])
    centers[36:, 7] = np.array([0.0, 2.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 8), dtype=float),
        "pose_quality": np.ones((frames, 8), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("two_unrelated_huddles.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "fill_gap_seconds": 0.20,
                },
            }
        },
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert len(huddles) == 2
    assert [(event["start_frame"], event["end_frame"]) for event in huddles] == [
        (0, 29),
        (36, 65),
    ]
    assert [set(event["member_ids"]) for event in huddles] == [
        {0, 1, 2},
        {5, 6, 7},
    ]
    assert all(
        len(segment["member_ids"]) >= 3 for event in huddles for segment in event["member_trace"]
    )


def test_transient_union_cannot_bridge_two_unrelated_huddle_lineages() -> None:
    lightweight = load_lightweight()
    frames = 63
    valid = np.ones((frames, 6), dtype=bool)
    centers = np.tile(
        np.asarray([[[30.0 * mouse_id, 20.0] for mouse_id in range(6)]], dtype=float),
        (frames, 1, 1),
    )
    centers[:33, 0:3] = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    centers[30:, 3:6] = np.asarray([[1.0, 0.0], [2.0, 2.0], [0.0, 1.0]])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 6), dtype=float),
        "pose_quality": np.ones((frames, 6), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("transient_union.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "fill_gap_seconds": 0.20,
                },
            }
        },
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert len(huddles) == 2
    assert [set(event["member_ids"]) for event in huddles] == [
        {0, 1, 2},
        {3, 4, 5},
    ]


def test_attack_contained_by_stable_huddle_does_not_displace_group() -> None:
    lightweight = load_lightweight()
    frames = 60
    valid = np.ones((frames, 3), dtype=bool)
    centers = np.tile(
        np.asarray([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=float),
        (frames, 1, 1),
    )
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 3), dtype=float),
        "pose_quality": np.ones((frames, 3), dtype=float),
        "centers_cm": centers,
    }
    attack = {
        "behavior": "attack",
        "candidate_level": "extended",
        "analysis_start_frame": 0,
        "analysis_end_frame": frames - 1,
        "actor_id": 0,
        "target_id": 1,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("attack_with_bystander.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "unspecified_min_duration_seconds": 1.0,
                "social": {"attack_min_duration_seconds": 1.0},
                "group": {"huddle_min_duration_seconds": 1.0},
            }
        },
        pair_behavior_events=[attack],
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert len(huddles) == 1
    assert set(huddles[0]["member_ids"]) == {0, 1, 2}
    assert attack["huddle_conflict_status"] == "contained_by_stable_huddle"


def test_transient_subthreshold_cluster_does_not_suppress_attack() -> None:
    lightweight = load_lightweight()
    frames = 60
    valid = np.ones((frames, 3), dtype=bool)
    centers = np.tile(
        np.asarray([[[0.0, 20.0], [20.0, 20.0], [40.0, 20.0]]], dtype=float),
        (frames, 1, 1),
    )
    # Twenty close frames are a raw three-mouse geometry candidate, but they
    # do not satisfy the configured one-second huddle duration at 30 FPS.
    centers[20:40] = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 3), dtype=float),
        "pose_quality": np.ones((frames, 3), dtype=float),
        "centers_cm": centers,
    }
    attack = {
        "behavior": "attack",
        "candidate_level": "extended",
        "analysis_start_frame": 0,
        "analysis_end_frame": frames - 1,
        "actor_id": 0,
        "target_id": 1,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("attack_with_transient_cluster.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "unspecified_min_duration_seconds": 1.0,
                "social": {"attack_min_duration_seconds": 1.0},
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "huddle_attack_independent_seconds": 2.0,
                },
            }
        },
        pair_behavior_events=[attack],
    )

    assert not any(event["behavior"] == "huddle" for event in events)
    assert attack.get("huddle_conflict_status") != "contained_by_stable_huddle"


def test_independent_attack_pair_is_not_promoted_to_a_three_mouse_huddle() -> None:
    lightweight = load_lightweight()
    frames = 60
    valid = np.ones((frames, 3), dtype=bool)
    centers = np.tile(
        np.asarray([[[0.0, 20.0], [20.0, 20.0], [40.0, 20.0]]], dtype=float),
        (frames, 1, 1),
    )
    centers[15:45] = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 3), dtype=float),
        "pose_quality": np.ones((frames, 3), dtype=float),
        "centers_cm": centers,
    }
    attack = {
        "behavior": "attack",
        "candidate_level": "extended",
        "analysis_start_frame": 0,
        "analysis_end_frame": frames - 1,
        "actor_id": 0,
        "target_id": 1,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("independent_attack_with_bystander.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "unspecified_min_duration_seconds": 1.0,
                "social": {"attack_min_duration_seconds": 1.0},
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "huddle_attack_independent_seconds": 0.50,
                },
            }
        },
        pair_behavior_events=[attack],
    )

    assert not any(event["behavior"] == "huddle" for event in events)
    assert attack["huddle_conflict_status"] == "independent_pair_behavior"


def test_non_attacking_members_can_still_form_their_own_huddle() -> None:
    lightweight = load_lightweight()
    frames = 60
    valid = np.ones((frames, 5), dtype=bool)
    centers = np.tile(
        np.asarray(
            [[[0.0, 20.0], [20.0, 20.0], [40.0, 20.0], [60.0, 20.0], [80.0, 20.0]]],
            dtype=float,
        ),
        (frames, 1, 1),
    )
    centers[15:45] = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 5), dtype=float),
        "pose_quality": np.ones((frames, 5), dtype=float),
        "centers_cm": centers,
    }
    attack = {
        "behavior": "attack",
        "candidate_level": "extended",
        "analysis_start_frame": 0,
        "analysis_end_frame": frames - 1,
        "actor_id": 0,
        "target_id": 1,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("attack_beside_real_huddle.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "enabled": True,
                "unspecified_min_duration_seconds": 1.0,
                "social": {"attack_min_duration_seconds": 1.0},
                "group": {
                    "huddle_min_duration_seconds": 1.0,
                    "huddle_attack_independent_seconds": 0.50,
                },
            }
        },
        pair_behavior_events=[attack],
    )

    huddles = [event for event in events if event["behavior"] == "huddle"]
    assert len(huddles) == 1
    assert set(huddles[0]["member_ids"]) == {2, 3, 4}
    assert (huddles[0]["start_frame"], huddles[0]["end_frame"]) == (15, 44)


def test_isolation_keeps_one_mouse_outside_a_three_mouse_main_cluster():
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 10), dtype=bool)
    centers = np.zeros((frames, 10, 2), dtype=float)
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([2.0, 0.0])
    centers[:, 2] = np.array([0.0, 2.0])
    centers[:, 3] = np.array([2.0, 2.0])
    centers[:, 4] = np.array([4.0, 0.0])
    centers[:, 5] = np.array([4.0, 2.0])
    centers[:, 6] = np.array([0.0, 4.0])
    centers[:, 7] = np.array([2.0, 4.0])
    centers[:, 8] = np.array([4.0, 4.0])
    centers[:, 9] = np.array([50.0, 50.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 10), dtype=float),
        "pose_quality": np.ones((frames, 10), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("isolation.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "group": {
                    "huddle_distance_cm": 9.0,
                    "isolation_distance_cm": 15.0,
                    "isolation_min_duration_seconds": 0.3,
                }
            }
        },
    )

    isolation = [event for event in events if event["behavior"] == "isolation"]
    assert isolation
    assert set(isolation[0]["member_ids"]) == {9}


def test_isolation_does_not_keep_a_short_lived_swapped_member():
    lightweight = load_lightweight()
    frames = 30
    valid = np.ones((frames, 10), dtype=bool)
    centers = np.zeros((frames, 10, 2), dtype=float)
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([2.0, 0.0])
    centers[:, 2] = np.array([0.0, 2.0])
    centers[:, 3] = np.array([2.0, 2.0])
    centers[:, 4] = np.array([4.0, 0.0])
    centers[:, 5] = np.array([4.0, 2.0])
    centers[:, 6] = np.array([0.0, 4.0])
    centers[:, 7] = np.array([2.0, 4.0])
    centers[:, 8] = np.array([4.0, 4.0])
    centers[:, 9] = np.array([50.0, 50.0])
    # The true outlier is stable. ID 8 is temporarily displaced for only two
    # frames, which is shorter than the configured three-frame rule below.
    centers[10:12, 8] = np.array([50.0, 50.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 10), dtype=float),
        "pose_quality": np.ones((frames, 10), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("isolation_swap.mov"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "group": {
                    "huddle_distance_cm": 9.0,
                    "isolation_distance_cm": 15.0,
                    "isolation_min_duration_seconds": 0.3,
                    "confirm_seconds": 0.1,
                    "fill_gap_seconds": 0.0,
                }
            }
        },
    )

    isolation = [event for event in events if event["behavior"] == "isolation"]
    assert isolation
    assert set(isolation[0]["member_ids"]) == {9}
    assert isolation[0]["actor_id"] == 9


def test_isolation_requires_a_majority_main_group() -> None:
    lightweight = load_lightweight()
    frames = 30
    valid = np.ones((frames, 10), dtype=bool)
    centers = np.zeros((frames, 10, 2), dtype=float)
    # Only three mice form a local component. The other seven are mutually
    # separated, so this is a scattered scene rather than one mouse isolated
    # from a meaningful main group.
    centers[:] = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [25.0, 0.0],
            [50.0, 0.0],
            [75.0, 0.0],
            [0.0, 25.0],
            [0.0, 50.0],
            [0.0, 75.0],
            [75.0, 75.0],
        ],
        dtype=float,
    )
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 10), dtype=float),
        "pose_quality": np.ones((frames, 10), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("scattered.mov"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "group": {
                    "huddle_distance_cm": 9.0,
                    "isolation_distance_cm": 15.0,
                    "isolation_min_duration_seconds": 0.3,
                    "isolation_min_main_cluster_fraction": 0.6,
                    "fill_gap_seconds": 0.0,
                }
            }
        },
    )

    assert not any(event["behavior"] == "isolation" for event in events)


def test_isolation_keeps_only_the_strongest_outlier_from_a_half_scene_main_group() -> None:
    lightweight = load_lightweight()
    frames = 40
    valid = np.ones((frames, 10), dtype=bool)
    centers = np.zeros((frames, 10, 2), dtype=float)
    centers[:] = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [4.0, 1.0],
            [30.0, 0.0],
            [0.0, 35.0],
            [45.0, 45.0],
            [80.0, 0.0],
            [120.0, 100.0],
        ],
        dtype=float,
    )
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 10), dtype=float),
        "pose_quality": np.ones((frames, 10), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("single_outlier.mov"),
        source_fps=10.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "group": {
                    "huddle_distance_cm": 9.0,
                    "isolation_distance_cm": 15.0,
                    "isolation_min_duration_seconds": 3.0,
                    "isolation_min_main_cluster_fraction": 0.5,
                    "fill_gap_seconds": 0.0,
                }
            }
        },
    )

    isolation = [event for event in events if event["behavior"] == "isolation"]
    assert len(isolation) == 1
    assert isolation[0]["actor_id"] == 9
    assert isolation[0]["member_ids"] == [9]


def test_two_mice_do_not_create_a_group_huddle_event():
    lightweight = load_lightweight()
    frames = 12
    valid = np.ones((frames, 2), dtype=bool)
    centers = np.zeros((frames, 2, 2), dtype=float)
    centers[:, 0] = np.array([0.0, 0.0])
    centers[:, 1] = np.array([2.0, 0.0])
    kin = {
        "valid": valid,
        "behavior_speed": np.zeros((frames, 2), dtype=float),
        "pose_quality": np.ones((frames, 2), dtype=float),
        "centers_cm": centers,
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("pair.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={"extended_behavior": {"enabled": True}},
    )

    assert not any(event["behavior"] == "huddle" for event in events)


def test_document_duration_rules_gate_individual_events_by_behavior() -> None:
    lightweight = load_lightweight()
    frames = 40
    valid = np.ones((frames, 2), dtype=bool)
    speed = np.full((frames, 2), 20.0, dtype=float)
    speed[15:, 0] = 0.0
    speed[20:, 1] = 0.0
    kin = {
        "valid": valid,
        "behavior_speed": speed,
        "pose_quality": np.ones((frames, 2), dtype=float),
        "centers_cm": np.zeros((frames, 2, 2), dtype=float),
    }

    events = lightweight._extended_individual_and_group_events(
        kin,
        pair_metrics={},
        pair_i=np.array([], dtype=int),
        pair_j=np.array([], dtype=int),
        source_video=Path("duration.mov"),
        source_fps=30.0,
        sample_stride=1,
        config={
            "extended_behavior": {
                "individual": {
                    "stationary_max_speed_cm_s": 4.0,
                    "running_min_speed_cm_s": 18.0,
                    "running_min_duration_seconds": 0.5,
                    "walking_min_duration_seconds": 1.0,
                    "stationary_min_duration_seconds": 1.0,
                    "confirm_seconds": 0.3,
                    "fill_gap_seconds": 0.0,
                    "min_pose_quality": 0.2,
                }
            }
        },
    )

    assert any(event["behavior"] == "running" for event in events)
    assert not any(event["behavior"] == "stationary" for event in events)


def test_document_duration_rules_gate_nose_tail_contact() -> None:
    lightweight = load_lightweight()
    pair_df = pd.DataFrame([contact_frame(frame, 8.0, 2.0) for frame in range(10)])

    events = lightweight._extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("duration.mov"),
        source_fps=30.0,
        sample_stride=1,
        contact_config={
            "enabled": True,
            "nose_head_distance_cm": 1.0,
            "nose_tail_distance_cm": 3.0,
            "nose_tail_min_duration_seconds": 0.5,
        },
    )

    assert events == []


def test_unspecified_nose_head_contact_requires_one_second() -> None:
    lightweight = load_lightweight()
    contact_config = {
        "enabled": True,
        "nose_head_distance_cm": 3.0,
        "nose_tail_distance_cm": 3.0,
        "nose_head_min_duration_seconds": 1.0,
    }

    short_events = lightweight._extract_contact_events(
        pd.DataFrame([contact_frame(frame, 2.0, 8.0) for frame in range(29)]),
        pair_key="1_2",
        source_video=Path("duration.mov"),
        source_fps=30.0,
        sample_stride=1,
        contact_config=contact_config,
    )
    complete_events = lightweight._extract_contact_events(
        pd.DataFrame([contact_frame(frame, 2.0, 8.0) for frame in range(30)]),
        pair_key="1_2",
        source_video=Path("duration.mov"),
        source_fps=30.0,
        sample_stride=1,
        contact_config=contact_config,
    )

    assert short_events == []
    assert len(complete_events) == 1
    assert complete_events[0]["contact_type"] == "nose_head"
    assert complete_events[0]["duration_s"] == 1.0


def test_combined_contact_uses_the_stricter_component_duration() -> None:
    lightweight = load_lightweight()
    contact_config = {
        "enabled": True,
        "nose_head_distance_cm": 3.0,
        "nose_tail_distance_cm": 3.0,
        "nose_head_min_duration_seconds": 1.0,
        "nose_tail_min_duration_seconds": 0.5,
    }

    short_events = lightweight._extract_contact_events(
        pd.DataFrame([contact_frame(frame, 2.0, 2.0) for frame in range(15)]),
        pair_key="1_2",
        source_video=Path("duration.mov"),
        source_fps=30.0,
        sample_stride=1,
        contact_config=contact_config,
    )
    complete_events = lightweight._extract_contact_events(
        pd.DataFrame([contact_frame(frame, 2.0, 2.0) for frame in range(30)]),
        pair_key="1_2",
        source_video=Path("duration.mov"),
        source_fps=30.0,
        sample_stride=1,
        contact_config=contact_config,
    )

    assert short_events == []
    assert len(complete_events) == 1
    assert complete_events[0]["contact_type"] == "nose_head_and_nose_tail"
    assert complete_events[0]["duration_s"] == 1.0
