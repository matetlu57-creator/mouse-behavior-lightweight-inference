"""Unit and resilience smoke tests for social cropping validation and review website."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mouse_behavior.behavior.segment_clipping_common import InMemoryTrackStore
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.social_cropping_validation import (
    SOCIAL_BEHAVIOR_KEYS,
    audit_social_segment,
    refine_all_social_events,
    select_social_review_clips,
    summarize_social_cropping_validation,
)
from mouse_behavior.visualization.long_video_social_review_site import (
    DEMO_FALLBACK_SOCIAL_BUNDLE,
    build_long_video_social_review_site,
    create_long_video_social_review_server,
    validate_social_review_bundle_schema,
)


def _det(track_id: int, cx: float, cy: float, dx_nose: float = 15.0) -> dict:
    return {
        "track_id": int(track_id),
        "box": [cx - 22.0, cy - 12.0, cx + 22.0, cy + 12.0],
        "keypoints": [
            [cx + dx_nose, cy, 0.9],
            [cx + 0.4 * dx_nose, cy - 6.0, 0.9],
            [cx + 0.4 * dx_nose, cy + 6.0, 0.9],
            [cx, cy - 4.0, 0.9],
            [cx, cy, 0.9],
            [cx, cy + 4.0, 0.9],
            [cx - 20.0, cy, 0.9],
        ],
    }


def _row(frame: int, fps: float, dets: list[dict]) -> dict:
    return {"frame": int(frame), "source_frame": int(frame), "time": float(frame / fps), "detections": dets}


def test_social_validation_summary_and_review_clip_selection() -> None:
    fps = 10.0
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 100}}
    params = HeuristicParameters()

    rows = []
    for f in range(100):
        c2x = 235.0 if 10 <= f < 50 else 650.0
        rows.append(_row(f, fps, [_det(1, 200.0, 200.0, 15.0), _det(2, c2x, 200.0, -15.0)]))
    store = InMemoryTrackStore(rows)

    raw_events = [
        {
            "event_id": "hc_valid",
            "behavior": "Snout-head_contact",
            "canonical_behavior": "nose_head_contact",
            "mouse_ids": [1, 2],
            "start_frame": 15,
            "end_frame": 35,
        },
        {
            "event_id": "tog_too_short",
            "behavior": "Together",
            "canonical_behavior": "together",
            "mouse_ids": [1, 2],
            "start_frame": 5,
            "end_frame": 9,  # 0.4s separated/short -> rejected
        },
    ]

    post_events, _ = refine_all_social_events(raw_events, store, binding, scale, params)
    assert len(post_events) == 1
    assert (post_events[0]["start_frame"], post_events[0]["end_frame"]) == (10, 50)

    audit = audit_social_segment(post_events[0], store, binding, fps, params)
    assert audit["duration_ok"] is True
    assert audit["endpoint_observed_ok"] is True
    assert audit["role_ok"] is True
    assert audit["role_summary_zh"]
    assert audit["full_crop_precision_ok"] is True

    # Verify >0.7s calibrated Chase annotation & package qualification
    from mouse_behavior.visualization.behavior_packages import _annotation, _require_package_event_duration

    chase_ev = {
        "event_id": "chase_07",
        "behavior": "Chasing",
        "canonical_behavior": "chase",
        "mouse_ids": [1, 2],
        "actor_id": 1,
        "target_id": 2,
        "initiator_id": 1,
        "receiver_id": 2,
        "role_status": "measured_directed_role",
        "roles": {"initiator_role": "chaser", "target_role": "chased", "summary_zh": "V1(追逐者) → V2(被追逐者)"},
        "start_frame": 10,
        "end_frame": 34,
        "chase_min_duration_policy_s": 0.7,
    }
    assert _require_package_event_duration(chase_ev, 30.0)["ok"] is True
    ann = _annotation(chase_ev, [], 24, 30.0, crop_start=10)
    assert ann["actor_id"] == 1
    assert ann["target_id"] == 2
    assert ann["initiator_id"] == 1
    assert ann["receiver_id"] == 2
    assert ann["roles"]["summary_zh"] == "V1(追逐者) → V2(被追逐者)"

    idsw_att = {
        "event_id": "att_idsw_1",
        "behavior": "Attack",
        "canonical_behavior": "attack",
        "mouse_ids": [1, 2, 3, 4],
        "actor_id": 1,
        "target_id": 2,
        "initiator_id": 1,
        "receiver_id": 2,
        "role_status": "measured_directed_role",
        "roles": {"initiator_role": "attacker", "target_role": "victim", "summary_zh": "V1(攻击发起者) → V2(被攻击者)"},
        "has_visual_id_switch": True,
        "endpoint_observed_track_ids": {"start": [1, 2], "end": [3, 4]},
        "observed_onset_pair": [1, 2],
        "id_switch_note": "raw tracker IDs preserved",
        "discovery_source": "fight_ball_id_switch_scan",
        "start_frame": 10,
        "end_frame": 35,
    }
    ann_idsw = _annotation(idsw_att, [], 25, 30.0, crop_start=10)
    assert ann_idsw["has_visual_id_switch"] is True
    assert ann_idsw["endpoint_observed_track_ids"] == {"start": [1, 2], "end": [3, 4]}
    assert ann_idsw["observed_onset_pair"] == [1, 2]
    assert ann_idsw["id_switch_note"] == "raw tracker IDs preserved"

    summary, per_event = summarize_social_cropping_validation(raw_events, post_events, store, binding, params)
    assert set(summary["by_behavior"]) == set(SOCIAL_BEHAVIOR_KEYS)
    assert summary["aggregate"]["total_raw_social_candidates"] == 2
    assert summary["aggregate"]["total_post_crop_social_segments"] == 1
    assert summary["aggregate"]["duration_pass_rate"] == 1.0
    assert summary["aggregate"]["endpoint_observed_rate"] == 1.0
    assert summary["aggregate"]["zero_shared_mouse_overlap_ok"] is True

    selected = select_social_review_clips(post_events, store, binding, per_event_audit=per_event, max_per_behavior=5)
    assert len(selected) == 1

    # Verify both review clip selectors default to max_per_behavior=None (no 15/30 output cap)
    from mouse_behavior.evaluation.group_cropping_validation import select_group_review_clips

    many_social = [
        {
            "event_id": f"hc_unlim_{i}",
            "behavior": "Snout-head_contact",
            "canonical_behavior": "nose_head_contact",
            "mouse_ids": [1, 2],
            "start_frame": i * 20,
            "end_frame": i * 20 + 10,
            "boundary": {"changed": False},
        }
        for i in range(22)
    ]
    social_store = InMemoryTrackStore(
        [_row(f, fps, [_det(1, 200.0, 200.0, 15.0), _det(2, 235.0, 200.0, -15.0)]) for f in range(500)]
    )
    selected_unlim_social = select_social_review_clips(
        many_social, social_store, {"source_video": {"fps": fps, "frame_count": 500}}
    )
    assert len(selected_unlim_social) == 22

    many_group = [
        {
            "event_id": f"sc_unlim_{i}",
            "behavior": "Social clustering",
            "canonical_behavior": "social_clustering",
            "mouse_ids": [1, 2, 3],
            "start_frame": i * 40,
            "end_frame": i * 40 + 35,
        }
        for i in range(35)
    ]
    group_store = InMemoryTrackStore(
        [
            _row(
                f,
                fps,
                [_det(1, 200.0, 200.0), _det(2, 220.0, 200.0), _det(3, 210.0, 215.0)],
            )
            for f in range(1500)
        ]
    )
    selected_unlim_group = select_group_review_clips(
        many_group, group_store, {"source_video": {"fps": fps, "frame_count": 1500}}
    )
    assert len(selected_unlim_group) == 35


def test_social_review_site_build_resilience_and_http_routes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_social"
    raw_pkg = run_dir / "clips_raw" / "Attack" / "att_1"
    ov_pkg = run_dir / "clips_overlay" / "Attack" / "att_1"
    for pkg in (raw_pkg, ov_pkg):
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
        (pkg / "annotation.json").write_text("{}", encoding="utf-8")
        (pkg / "metadata.json").write_text("{}", encoding="utf-8")
        (pkg / "tracks.json").write_text("[]", encoding="utf-8")

    (run_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "video_key": "video22",
                "title": "Video Project 22",
                "duration_s": 595.5,
                "fps": 30.0,
                "frame_count": 17864,
                "aggregate": {"total_raw_social_candidates": 2, "total_post_crop_social_segments": 1, "duration_pass_rate": 1.0},
                "by_behavior": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "review_candidates.json").write_text(
        json.dumps(
            [
                {
                    "event_id": "att_1",
                    "behavior": "Attack",
                    "canonical_behavior": "attack",
                    "mouse_ids": [1, 2],
                    "actor_id": 1,
                    "target_id": 2,
                    "initiator_id": 1,
                    "receiver_id": 2,
                    "role_status": "measured_directed_role",
                    "roles": {"initiator_role": "attacker", "target_role": "victim", "summary_zh": "V1(攻击发起者) → V2(被攻击者)"},
                    "start_frame": 10,
                    "end_frame": 30,
                    "frame_count": 20,
                    "start_time_s": 0.333,
                    "end_time_s": 1.0,
                    "duration_s": 0.667,
                    "raw_package_dir": str(raw_pkg),
                    "overlay_package_dir": str(ov_pkg),
                    "audit": {"duration_ok": True, "role_ok": True, "full_crop_precision_ok": True},
                }
            ]
        ),
        encoding="utf-8",
    )

    site_dir = tmp_path / "site"
    stale_file = site_dir / "videos" / "stale" / "clip.mp4"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_bytes(b"stale")
    built = build_long_video_social_review_site([run_dir], site_dir, port=0, clean_videos=True)
    assert not stale_file.exists()
    assert built["video_count"] == 1
    assert built["total_review_clips"] == 1
    assert "角色分工" in (site_dir / "index.html").read_text(encoding="utf-8")
    assert validate_social_review_bundle_schema(DEMO_FALLBACK_SOCIAL_BUNDLE)

    server = create_long_video_social_review_server(site_dir, port=0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["bundle_exists"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bundle") as resp:
            bundle = json.loads(resp.read())
            assert validate_social_review_bundle_schema(bundle)
            assert bundle["is_fallback_demo"] is False

        post_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/reviews",
            data=json.dumps({"clip_uid": "video22::att_1", "status": "pass", "note": "完整"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(post_req) as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["status"] == "saved"

        # Malformed bundle fallback/error handling + shell always renders HTML 200
        bundle_path = site_dir / "data" / "bundle.json"
        original_bundle = bundle_path.read_text(encoding="utf-8")
        bundle_path.write_text("{corrupt", encoding="utf-8")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            assert b"<html" in resp.read().lower()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bundle")
        assert err.value.code == 500
        bundle_path.unlink()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bundle") as resp:
            fallback = json.loads(resp.read())
            assert fallback["is_fallback_demo"] is True
        bundle_path.write_text(original_bundle, encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
