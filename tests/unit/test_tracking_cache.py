from __future__ import annotations

import gzip
import json
import pickle

import numpy as np
import pytest

from mouse_behavior.tracking import cache as tracking_cache
from mouse_behavior.tracking.cache import (
    _payload_detection,
    _solve_track_assignments,
    _track_cache,
)


def _pose_payload(x: float) -> dict[str, object]:
    points = np.asarray(
        [
            [x + 20.0, 20.0],
            [x + 20.0, 34.0],
            [x + 14.0, 42.0],
            [x + 26.0, 42.0],
            [x + 20.0, 50.0],
            [x + 20.0, 80.0],
            [x + 20.0, 100.0],
        ],
        dtype=float,
    )
    return {
        "keypoints_px": points,
        "keypoint_conf": np.ones(7, dtype=float),
        "bbox_xyxy": np.asarray([x, 0.0, x + 40.0, 120.0], dtype=float),
        "box_conf": 0.90,
        "pose_quality": 0.90,
    }


def test_bbox_detection_survives_incomplete_pose_for_occlusion_tracking() -> None:
    points = np.full((7, 2), np.nan, dtype=float)
    confidence = np.zeros(7, dtype=float)
    bbox = np.asarray([100.0, 200.0, 180.0, 320.0], dtype=float)

    detection = _payload_detection(
        {
            "keypoints_px": points,
            "keypoint_conf": confidence,
            "bbox_xyxy": bbox,
            "box_conf": 0.90,
            "pose_quality": 0.90,
        }
    )

    assert detection is not None
    np.testing.assert_allclose(detection["center"], np.asarray([140.0, 260.0]))
    assert detection["body_length"] == 80.0
    assert detection["pose_quality"] == 0.0
    assert np.isnan(detection["points"]).all()


def test_short_bbox_hold_is_separate_from_pose_track_validity(tmp_path) -> None:
    status = {
        "status": "complete",
        "total_frames": 7,
        "next_frame": 7,
    }
    (tmp_path / "yolo_results_status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    records = [
        {"frame": 0, "pose_detections": [_pose_payload(100.0)]},
        {"frame": 1, "pose_detections": []},
        {"frame": 2, "pose_detections": []},
        {"frame": 3, "pose_detections": [{**_pose_payload(110.0)}]},
        {"frame": 4, "pose_detections": []},
        {"frame": 5, "pose_detections": []},
        {"frame": 6, "pose_detections": []},
    ]
    with gzip.open(tmp_path / "yolo_results.000000.000005.pkl.gz", "wb") as handle:
        pickle.dump(records, handle)

    tracks, stats = _track_cache(
        tmp_path,
        total_frames=7,
        expected_mice=1,
        bbox_occlusion_max_gap_frames=2,
    )

    assert tracks["valid"][:, 0].tolist() == [True, False, False, True, False, False, False]
    assert tracks["bbox_observed"][:, 0].tolist() == [True, False, False, True, False, False, False]
    assert tracks["bbox_imputed"][:, 0].tolist() == [False, True, True, False, True, True, False]
    assert np.isfinite(tracks["bboxes"][1:3, 0]).all()
    assert np.isnan(tracks["bboxes"][6, 0]).all()
    assert stats["bbox_imputed_count"] == 4


def test_tracker_initializes_ids_that_first_appear_after_frame_zero(tmp_path) -> None:
    """A partial first frame must not permanently cap the logical track count."""

    status = {
        "status": "complete",
        "total_frames": 3,
        "next_frame": 3,
    }
    (tmp_path / "yolo_results_status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    records = [
        {"frame": 0, "pose_detections": [_pose_payload(100.0)]},
        {
            "frame": 1,
            "pose_detections": [_pose_payload(102.0), _pose_payload(300.0)],
        },
        {
            "frame": 2,
            "pose_detections": [_pose_payload(104.0), _pose_payload(302.0)],
        },
    ]
    with gzip.open(tmp_path / "yolo_results.000000.000002.pkl.gz", "wb") as handle:
        pickle.dump(records, handle)

    tracks, stats = _track_cache(
        tmp_path,
        total_frames=3,
        expected_mice=2,
    )

    assert tracks["valid"].sum(axis=1).tolist() == [1, 2, 2]
    assert np.all(np.isfinite(tracks["centers_px"][1, 1]))
    assert float(tracks["centers_px"][1, 1, 0]) > 250.0
    assert stats["track_valid_frames_max"] == 2


def test_low_confidence_first_frame_candidate_does_not_consume_permanent_id(tmp_path) -> None:
    """A mirror/partial box must not displace a later reliable real mouse."""

    low_confidence = {**_pose_payload(300.0), "box_conf": 0.10, "pose_quality": 0.90}
    records = [
        {
            "frame": 0,
            "pose_detections": [_pose_payload(100.0), low_confidence],
        },
        {
            "frame": 1,
            "pose_detections": [_pose_payload(102.0), _pose_payload(500.0)],
        },
        {
            "frame": 2,
            "pose_detections": [_pose_payload(104.0), _pose_payload(502.0)],
        },
    ]
    with gzip.open(tmp_path / "yolo_results.000000.000002.pkl.gz", "wb") as handle:
        pickle.dump(records, handle)

    tracks, stats = _track_cache(
        tmp_path,
        total_frames=3,
        expected_mice=2,
        initial_min_detection_score=0.50,
    )

    assert tracks["valid"].sum(axis=1).tolist() == [1, 2, 2]
    assert np.isnan(tracks["centers_px"][0, 1]).all()
    assert float(tracks["centers_px"][1, 1, 0]) > 450.0
    assert stats["initial_min_detection_score"] == 0.50


def test_configured_reacquisition_gate_rejects_a_far_identity_jump(tmp_path) -> None:
    """A lost mouse must not reclaim an unrelated detection across the cage."""

    records = [
        {"frame": 0, "pose_detections": [_pose_payload(100.0)]},
        {"frame": 1, "pose_detections": []},
        {"frame": 2, "pose_detections": []},
        {"frame": 3, "pose_detections": []},
        {"frame": 4, "pose_detections": []},
        {"frame": 5, "pose_detections": [_pose_payload(420.0)]},
    ]
    status = {"status": "complete", "total_frames": 6, "next_frame": 6}
    (tmp_path / "yolo_results_status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    with gzip.open(tmp_path / "yolo_results.000000.000005.pkl.gz", "wb") as handle:
        pickle.dump(records, handle)

    tracks, stats = _track_cache(
        tmp_path,
        total_frames=6,
        expected_mice=1,
        recent_assignment_gate=3.2,
        reacquisition_assignment_gate=3.2,
        reacquisition_after_missed_frames=3,
    )

    assert tracks["valid"][:, 0].tolist() == [True, False, False, False, False, False]
    assert stats["reacquisition_assignment_gate"] == 3.2


def test_assignment_solver_does_not_let_false_detection_steal_valid_match() -> None:
    """A rejected far match must not cause a second track to steal the true box."""

    # Track 0 has an excellent match in detection 0. Track 1 has disappeared;
    # detection 1 is a false positive outside both tracks' gates. A plain 2x2
    # Hungarian solve chooses the lower global sum (0->1, 1->0), after which
    # 0->1 is rejected and detection 0 remains attached to the wrong identity.
    cost = np.asarray(
        [
            [0.10, 7.40],
            [3.10, 10.50],
        ],
        dtype=float,
    )
    center_cost = cost.copy()

    rows, columns = _solve_track_assignments(
        cost,
        center_cost,
        np.asarray([3.20, 3.20], dtype=float),
    )

    assert list(zip(rows.tolist(), columns.tolist())) == [(0, 0)]


def test_assignment_solver_does_not_favor_wider_reacquisition_gate() -> None:
    """A missed track's wider gate must not steal an active track's box."""

    # Row 0 has been missing long enough to use the wider 5.0 gate, while row
    # 1 is continuously observed and has the much better match.  Using each
    # gate as that row's dummy cost reverses this result globally: 0.60 + 3.20
    # is considered cheaper than 0.15 + 5.00, so row 0 steals the detection.
    rows, columns = _solve_track_assignments(
        np.asarray([[0.60], [0.15]], dtype=float),
        np.asarray([[0.59], [0.14]], dtype=float),
        np.asarray([5.00, 3.20], dtype=float),
    )

    assert list(zip(rows.tolist(), columns.tolist())) == [(1, 0)]


def test_assignment_solver_fails_explicitly_without_scipy(monkeypatch) -> None:
    """Do not silently reintroduce row-order-dependent greedy ID assignment."""

    monkeypatch.setattr(tracking_cache, "linear_sum_assignment", None)

    with pytest.raises(RuntimeError, match="scipy is required"):
        _solve_track_assignments(
            np.asarray([[0.1]], dtype=float),
            np.asarray([[0.1]], dtype=float),
            np.asarray([1.0], dtype=float),
        )
