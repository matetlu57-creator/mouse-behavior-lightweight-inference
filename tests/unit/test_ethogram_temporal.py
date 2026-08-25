from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mouse_behavior.behavior.ethogram import (
    _event_rows_from_mask,
    _extended_short_clip_pair_events,
)
from mouse_behavior.behavior.pair_analysis import _finalize_event_records_in_place


def test_short_event_keeps_core_span_and_adds_bounded_display_context() -> None:
    mask = np.zeros(10, dtype=bool)
    mask[5] = True

    events = _event_rows_from_mask(
        mask,
        behavior="attack",
        level="extended",
        fps=30.0,
        source_video=Path("example.mp4"),
        sample_stride=1,
        score=mask.astype(float),
        actor_id=np.full(10, 1),
        target_id=np.full(10, 2),
        pair_key="1_2",
        min_duration_seconds=1 / 30,
        fill_gap_seconds=0.0,
        short_event_padding_seconds=0.10,
        short_event_max_duration_seconds=0.35,
    )

    assert len(events) == 1
    event = events[0]
    assert event["analysis_start_frame"] == 5
    assert event["analysis_end_frame"] == 5
    assert event["core_start_frame"] == 5
    assert event["core_end_frame"] == 5
    assert event["start_frame"] == 2
    assert event["end_frame"] == 8
    assert event["temporal_padding_frames"] == 6
    assert event["core_duration_s"] == 1 / 30
    assert event["duration_s"] == 7 / 30
    assert event["event_recovery"] == "none"


def test_short_attack_recovery_accepts_two_temporally_supported_hits() -> None:
    pair = pd.DataFrame(
        {
            "pair_key": ["4_5"] * 4,
            "valid_pair": [False, True, True, False],
            "center_distance_cm": [20.0, 8.45, 8.20, 20.0],
            "selected_actor_id": [4, 4, 4, 4],
            "selected_target_id": [5, 5, 5, 5],
            "mouse_a_id": [4, 4, 4, 4],
            "mouse_b_id": [5, 5, 5, 5],
            "selected_actor_speed_cm_s": [0.0, 27.0, 26.0, 0.0],
            "selected_actor_pursuit_alignment": [0.0, 0.94, 0.93, 0.0],
            "selected_target_escape_alignment": [0.0, 0.99, 0.98, 0.0],
            "a_to_b_nose_head_distance_cm": [20.0, 1.9, 1.8, 20.0],
            "b_to_a_nose_head_distance_cm": [20.0, 20.0, 20.0, 20.0],
        }
    )
    enriched = pd.DataFrame(
        {
            "weak_standard_attack_score": [0.0, 0.809, 0.82, 0.0],
            "strong_standard_attack_score": [0.0, 0.809, 0.82, 0.0],
            "weak_standard_dynamic_attack_score": [0.0, 0.809, 0.82, 0.0],
            "strong_standard_dynamic_attack_score": [0.0, 0.809, 0.82, 0.0],
            "weak_standard_attack_evidence_count": [0.0, 2.0, 2.0, 0.0],
            "strong_standard_attack_evidence_count": [0.0, 2.0, 2.0, 0.0],
            "weak_standard_attack_role_confidence": [0.0, 0.261, 0.26, 0.0],
            "strong_standard_attack_role_confidence": [0.0, 0.261, 0.26, 0.0],
            "weak_standard_initiation_score": [0.0, 1.0, 0.98, 0.0],
            "strong_standard_initiation_score": [0.0, 1.0, 0.98, 0.0],
            "weak_standard_reaction_score": [0.0, 0.45, 0.45, 0.0],
            "strong_standard_reaction_score": [0.0, 0.45, 0.45, 0.0],
        }
    )

    events = _extended_short_clip_pair_events(
        pair,
        enriched,
        source_video=Path("attack_example.mp4"),
        source_fps=30.0,
        sample_stride=1,
        config={"parallel_behavior_fsm": {"enabled": True}},
    )

    assert len(events) == 1
    assert events[0]["behavior"] == "attack"
    assert events[0]["event_recovery"] == "short_high_evidence"
    assert events[0]["analysis_start_frame"] == 1
    assert events[0]["analysis_end_frame"] == 2
    assert events[0]["core_duration_s"] == 2 / 30
    assert events[0]["duration_s"] > events[0]["core_duration_s"]


def test_short_attack_recovery_rejects_one_isolated_evidence_hit() -> None:
    pair = pd.DataFrame(
        {
            "pair_key": ["4_5"] * 3,
            "valid_pair": [False, True, False],
            "center_distance_cm": [20.0, 8.45, 20.0],
            "selected_actor_id": [4, 4, 4],
            "selected_target_id": [5, 5, 5],
            "mouse_a_id": [4, 4, 4],
            "mouse_b_id": [5, 5, 5],
            "selected_actor_speed_cm_s": [0.0, 27.0, 0.0],
            "selected_actor_pursuit_alignment": [0.0, 0.94, 0.0],
            "selected_target_escape_alignment": [0.0, 0.99, 0.0],
            "a_to_b_nose_head_distance_cm": [20.0, 1.9, 20.0],
            "b_to_a_nose_head_distance_cm": [20.0, 20.0, 20.0],
        }
    )
    enriched = pd.DataFrame(
        {
            "weak_standard_attack_score": [0.0, 0.809, 0.0],
            "strong_standard_attack_score": [0.0, 0.809, 0.0],
            "weak_standard_dynamic_attack_score": [0.0, 0.809, 0.0],
            "strong_standard_dynamic_attack_score": [0.0, 0.809, 0.0],
            "weak_standard_attack_evidence_count": [0.0, 2.0, 0.0],
            "strong_standard_attack_evidence_count": [0.0, 2.0, 0.0],
            "weak_standard_attack_role_confidence": [0.0, 0.261, 0.0],
            "strong_standard_attack_role_confidence": [0.0, 0.261, 0.0],
            "weak_standard_initiation_score": [0.0, 1.0, 0.0],
            "strong_standard_initiation_score": [0.0, 1.0, 0.0],
            "weak_standard_reaction_score": [0.0, 0.45, 0.0],
            "strong_standard_reaction_score": [0.0, 0.45, 0.0],
        }
    )

    events = _extended_short_clip_pair_events(
        pair,
        enriched,
        source_video=Path("attack_example.mp4"),
        source_fps=30.0,
        sample_stride=1,
        config={"parallel_behavior_fsm": {"enabled": True}},
    )

    assert events == []


def test_finalizer_suppresses_legacy_one_frame_attack_rows() -> None:
    events = [
        {
            "behavior": "attack",
            "analysis_start_frame": 12,
            "analysis_end_frame": 12,
            "start_frame": 12,
            "end_frame": 12,
        },
        {
            "behavior": "attack",
            "analysis_start_frame": 20,
            "analysis_end_frame": 21,
            "start_frame": 20,
            "end_frame": 21,
        },
    ]

    _finalize_event_records_in_place(events, [], 30.0)

    assert len(events) == 1
    assert events[0]["start_frame"] == 20
    assert events[0]["light_event_id"] == "LWE00001"
