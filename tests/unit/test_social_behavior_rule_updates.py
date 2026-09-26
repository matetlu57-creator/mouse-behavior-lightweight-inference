from pathlib import Path

import numpy as np
import pandas as pd

from mouse_behavior.behavior.ethogram import _extract_contact_events
from mouse_behavior.behavior.social_fsm import _bridge_attack_reacquisition_gap


def _contact_pair(contact_frames: set[int], frame_count: int = 10) -> pd.DataFrame:
    head = np.full(frame_count, np.inf, dtype=float)
    head[list(contact_frames)] = 2.0
    return pd.DataFrame(
        {
            "frame": np.arange(frame_count),
            "valid_pair": np.ones(frame_count, dtype=bool),
            "mouse_a_id": np.ones(frame_count, dtype=int),
            "mouse_b_id": np.full(frame_count, 2, dtype=int),
            "a_to_b_nose_head_distance_cm": head,
            "a_to_b_nose_tail_distance_cm": np.full(frame_count, np.inf),
            "b_to_a_nose_head_distance_cm": np.full(frame_count, np.inf),
            "b_to_a_nose_tail_distance_cm": np.full(frame_count, np.inf),
        }
    )


def _extract_contacts(pair_df: pd.DataFrame) -> list[dict]:
    return _extract_contact_events(
        pair_df,
        pair_key="1_2",
        source_video=Path("synthetic.mp4"),
        source_fps=10.0,
        sample_stride=1,
        contact_config={
            "enabled": True,
            "nose_head_distance_cm": 3.0,
            "nose_tail_distance_cm": 3.0,
            "nose_head_min_cumulative_duration_seconds": 0.5,
            "nose_tail_min_cumulative_duration_seconds": 0.5,
        },
    )


def test_repeated_contact_is_suppressed_below_cumulative_half_second():
    assert _extract_contacts(_contact_pair({0, 1, 5, 6})) == []


def test_repeated_contact_bouts_are_kept_when_pair_cumulative_time_reaches_half_second():
    events = _extract_contacts(_contact_pair({0, 1, 5, 6, 7}))

    assert len(events) == 2
    assert all(event["contact_type"] == "nose_head" for event in events)
    assert sum(event["duration_s"] for event in events) == 0.5


def test_attack_seed_can_bridge_missing_pair_until_nearby_reacquisition():
    pair_df = pd.DataFrame(
        {
            "bbox_pair_valid": [True, False, False, True, True],
            "bbox_pair_observed": [True, False, False, True, True],
            "bbox_center_distance_body_lengths": [1.0, np.inf, np.inf, 2.0, 2.0],
        }
    )

    result = _bridge_attack_reacquisition_gap(
        np.asarray([True, False, False, False, False]),
        pair_df,
        fps=10.0,
        attack_config={
            "state_reacquisition_gap_seconds": 0.5,
            "state_reacquisition_max_distance_body_lengths": 2.8,
        },
    )

    assert result.tolist() == [True, True, True, True, False]


def test_attack_reacquisition_does_not_bridge_visible_or_far_pair():
    visible = pd.DataFrame(
        {
            "bbox_pair_valid": [True, True, True],
            "bbox_pair_observed": [True, True, True],
            "bbox_center_distance_body_lengths": [1.0, 1.2, 1.0],
        }
    )
    far_reacquisition = pd.DataFrame(
        {
            "bbox_pair_valid": [True, False, True],
            "bbox_pair_observed": [True, False, True],
            "bbox_center_distance_body_lengths": [1.0, np.inf, 4.0],
        }
    )
    mask = np.asarray([True, False, False])
    config = {
        "state_reacquisition_gap_seconds": 0.5,
        "state_reacquisition_max_distance_body_lengths": 2.8,
    }

    assert _bridge_attack_reacquisition_gap(
        mask, visible, fps=10.0, attack_config=config
    ).tolist() == mask.tolist()
    assert _bridge_attack_reacquisition_gap(
        mask, far_reacquisition, fps=10.0, attack_config=config
    ).tolist() == mask.tolist()
