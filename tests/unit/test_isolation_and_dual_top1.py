from __future__ import annotations

import numpy as np

from mouse_behavior.behavior.ethogram import _sustained_member_ids_by_frame
from mouse_behavior.evaluation.exported_dataset import (
    SampleRecord,
    _isolation_cumulative_durations,
    classify_target_ids,
    evaluate_target_classifications,
)


def _record(behavior: str, mouse_ids: tuple[int, ...] = (1, 2)) -> SampleRecord:
    canonical = {
        "Together": "together",
        "Snout-head_contact": "nose_head_contact",
        "Snout-rear_contact": "nose_tail_contact",
    }[behavior]
    return SampleRecord(
        sample_id=behavior,
        behavior=behavior,
        canonical_behavior=canonical,
        sample_dir="sample",
        annotation_path="annotation.json",
        metadata_path="metadata.json",
        tracks_path="tracks.json",
        clip_path="clip.mp4",
        source_video="session.mp4",
        recording_session="session.mp4",
        fps=10.0,
        frame_count=102,
        mouse_ids=mouse_ids,
        confidence="certain",
        frame_start=0,
        frame_end=101,
        time_start=0.0,
        time_end=10.1,
    )


def test_isolation_counts_observed_frames_across_short_missing_id_gap() -> None:
    evidence = np.zeros((102, 1), dtype=bool)
    unknown = np.zeros_like(evidence)
    evidence[:50, 0] = True
    evidence[52:, 0] = True
    unknown[50:52, 0] = True

    assert _isolation_cumulative_durations(evidence, unknown, 10.0, 0.2)[0] == 10.0


def test_isolation_does_not_bridge_measured_nearby_frames() -> None:
    evidence = np.zeros((102, 1), dtype=bool)
    unknown = np.zeros_like(evidence)
    evidence[:50, 0] = True
    evidence[52:, 0] = True

    assert _isolation_cumulative_durations(evidence, unknown, 10.0, 0.2)[0] == 5.0


def test_runtime_isolation_gap_counts_only_observed_membership() -> None:
    members = [(1,)] * 50 + [(), ()] + [(1,)] * 50
    eligible = np.zeros((102, 2), dtype=bool)
    eligible[50:52, 1] = True

    result = _sustained_member_ids_by_frame(
        members,
        frames=102,
        mice=2,
        min_duration_frames=100,
        max_gap_frames=2,
        eligible_gap_by_frame=eligible,
    )

    assert all(1 in frame for frame in result)


def test_dual_top1_accepts_together_contact_but_not_head_rear_substitution() -> None:
    together_scores = {
        "approach": 0.0,
        "together": 1.0,
        "chase": 0.0,
        "avoidance": 0.0,
        "attack": 0.0,
        "nose_head_contact": 0.0,
        "nose_tail_contact": 0.0,
        "following": 0.0,
    }
    prediction = {
        "track_ids": [1, 2],
        "identity_scores": {"1": {}, "2": {}},
        "pair_scores": {"1,2": together_scores},
        "group_scores": {},
    }
    head_record = _record("Snout-head_contact")
    head_result = classify_target_ids(head_record, prediction)

    assert head_result["correct"] is False
    assert head_result["compatible_correct"] is True

    tail_prediction = {
        **prediction,
        "pair_scores": {
            "1,2": {**together_scores, "together": 0.0, "nose_tail_contact": 1.0}
        },
    }
    assert classify_target_ids(head_record, tail_prediction)["compatible_correct"] is False

    together_record = _record("Together")
    together_result = classify_target_ids(together_record, prediction)
    metrics = evaluate_target_classifications(
        [(head_record, head_result), (together_record, together_result)]
    )
    assert metrics["strict_top1_accuracy"] == 0.5
    assert metrics["compatible_top1_accuracy"] == 1.0
    assert metrics["per_behavior"]["nose_head_contact"]["compatible_top1_accuracy"] == 1.0
