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
    centers[6:, 2, 0] = 30.0
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
