from __future__ import annotations

import numpy as np
import pytest

from mouse_behavior.identity.mtrack import load_identity_map
from mouse_behavior.tracking.cache import TrajectoryCacheError, load_trajectory_cache


def test_visual_only_slots_follow_ultralytics_ids_not_spatial_order(
    tmp_path,
    write_pose_cache,
    pose_payload_factory,
):
    cache_dir = write_pose_cache(
        tmp_path / "cache",
        [
            {
                "frame": 0,
                "pose_detections": [
                    pose_payload_factory(10, 10.0),
                    pose_payload_factory(20, 90.0),
                ],
            },
            {
                "frame": 1,
                "pose_detections": [
                    pose_payload_factory(10, 90.0),
                    pose_payload_factory(20, 10.0),
                ],
            },
        ],
    )

    tracks, stats = load_trajectory_cache(cache_dir, total_frames=2, expected_mice=2)

    assert tuple(tracks["identity_labels"]) == ("visual:10", "visual:20")
    assert tuple(tracks["visual_track_ids_by_slot"]) == (10, 20)
    assert np.allclose(tracks["centers_px"][:, 0, 0], [10.0, 90.0])
    assert np.allclose(tracks["centers_px"][:, 1, 0], [90.0, 10.0])
    assert stats["identity_source"] == "visual_only"


def test_rfid_mapping_corrects_visual_id_switch_without_guessing(
    tmp_path,
    write_pose_cache,
    pose_payload_factory,
):
    cache_dir = write_pose_cache(
        tmp_path / "cache",
        [
            {
                "frame": 0,
                "pose_detections": [
                    pose_payload_factory(10, 10.0),
                    pose_payload_factory(20, 90.0),
                ],
            },
            {
                "frame": 1,
                "pose_detections": [
                    pose_payload_factory(10, 90.0),
                    pose_payload_factory(20, 10.0),
                ],
            },
        ],
    )
    mapping_path = tmp_path / "mtrack_identity_map.csv"
    mapping_path.write_text(
        "frame,visual_track_id,animal_id,identity_confidence\n"
        "0,10,mouse-A,0.99\n"
        "0,20,mouse-B,0.99\n"
        "1,10,mouse-B,0.97\n"
        "1,20,mouse-A,0.97\n",
        encoding="utf-8",
    )
    identity_map = load_identity_map(mapping_path)

    tracks, stats = load_trajectory_cache(
        cache_dir,
        total_frames=2,
        expected_mice=2,
        identity_map=identity_map,
    )

    assert tuple(tracks["identity_labels"]) == ("mouse-A", "mouse-B")
    assert np.allclose(tracks["centers_px"][:, 0, 0], [10.0, 10.0])
    assert np.allclose(tracks["centers_px"][:, 1, 0], [90.0, 90.0])
    assert np.allclose(tracks["identity_confidence"], [[0.99, 0.99], [0.97, 0.97]])
    assert stats["identity_source"] == "rfid_mtrack"
    assert stats["unresolved_detection_count"] == 0


def test_old_untracked_cache_is_rejected(tmp_path, write_pose_cache, pose_payload_factory):
    cache_dir = write_pose_cache(
        tmp_path / "cache",
        [{"frame": 0, "pose_detections": [pose_payload_factory(None, 10.0)]}],
    )

    with pytest.raises(TrajectoryCacheError, match="visual_track_id"):
        load_trajectory_cache(cache_dir, total_frames=1, expected_mice=1)
