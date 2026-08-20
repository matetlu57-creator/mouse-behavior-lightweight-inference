from __future__ import annotations

import numpy as np

from mouse_behavior import lightweight_behavior_inference as lightweight


def test_pair_workset_preserves_full_pair_to_candidate_index_mapping():
    frames, mice = 6, 3
    centers = np.zeros((frames, mice, 2), dtype=float)
    centers[:, 1, 0] = 1.0
    centers[:, 2, 0] = 50.0
    heading = np.zeros((frames, mice, 2), dtype=float)
    heading[..., 0] = 1.0
    keypoints = np.repeat(centers[:, :, None, :], lightweight.KEYPOINTS, axis=2)
    kin = {
        "centers_cm": centers,
        "head_cm": centers.copy(),
        "keypoints_cm": keypoints,
        "heading": heading,
        "velocity": np.zeros((frames, mice, 2), dtype=float),
        "valid": np.ones((frames, mice), dtype=bool),
        "speed": np.zeros((frames, mice), dtype=float),
        "nose_speed": np.zeros((frames, mice), dtype=float),
        "acceleration": np.zeros((frames, mice), dtype=float),
        "angular_speed": np.zeros((frames, mice), dtype=float),
        "pose_quality": np.ones((frames, mice), dtype=float),
        "behavior_speed": np.zeros((frames, mice), dtype=float),
        "pose_deformation": np.zeros((frames, mice), dtype=float),
    }
    config = {
        "standard_behavior_engine": {"interaction_graph": {"radius_cm": 10.0}},
        "lightweight_behavior_inference": {
            "pair_prefilter": {
                "enabled": True,
                "close_distance_cm": 3.0,
                "min_heading_cosine": 0.0,
                "window": {
                    "enabled": True,
                    "padding_seconds": 0.0,
                    "fill_gap_seconds": 0.0,
                },
            }
        },
    }

    workset = lightweight._prepare_pair_workset(kin, 30.0, config)

    assert list(zip(workset.all_pair_i, workset.all_pair_j)) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert workset.candidate_pair_indices == (0,)
    assert workset.candidate_metric_index == {0: 0}
    assert workset.candidate_frame_mask.shape == (frames, 1)
    assert workset.metrics["valid_pair"].shape == (frames, 1)


def test_event_finalization_assigns_ids_by_stable_contract_order():
    events = [
        {"start_frame": 10, "end_frame": 12, "pair_key": "2_3", "level": "weak"},
        {"start_frame": 2, "end_frame": 4, "pair_key": "1_2", "level": "strong"},
    ]
    contacts = [
        {"start_frame": 5, "pair_key": "2_3", "contact_type": "nose_tail"},
        {"start_frame": 1, "pair_key": "1_2", "contact_type": "nose_head"},
    ]

    lightweight._finalize_event_records_in_place(events, contacts, source_fps=10.0)

    assert events[1]["light_event_id"] == "LWE00001"
    assert events[0]["light_event_id"] == "LWE00002"
    assert events[1]["start_time_s"] == 0.2
    assert events[1]["duration_s"] == 0.3
    assert contacts[1]["contact_event_id"] == "LCE00001"
    assert contacts[0]["contact_event_id"] == "LCE00002"
