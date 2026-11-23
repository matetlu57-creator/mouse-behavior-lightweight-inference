"""Unit and HTTP resilience tests for long-video group cropping validation & review site."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
import gzip
import io
import pickle

import numpy as np
import pytest

from mouse_behavior.behavior.segment_clipping_common import InMemoryTrackStore
from mouse_behavior.canonical_video_pipeline import TrackStore
from mouse_behavior.tracking.cache import _CompatUnpickler, _iter_cache_records
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.group_cropping_validation import (
    CachedTrackStore,
    audit_group_segment_kinematics,
    load_historical_review_decisions,
    summarize_group_cropping_validation,
)
from mouse_behavior.visualization.long_video_group_review_site import (
    DEMO_FALLBACK_LONG_VIDEO_BUNDLE,
    build_long_video_group_review_site,
    create_long_video_group_review_server,
    render_long_video_group_review_page,
    validate_review_bundle_schema,
)


def _make_track_row(frame: int, fps: float, coords: dict[int, tuple[float, float]]) -> dict:
    dets = []
    for tid, (x, y) in coords.items():
        box = [x - 15.0, y - 15.0, x + 15.0, y + 15.0]
        kp = [[x + 5.0, y, 0.9]] * 7
        kp[3] = [x, y, 0.95]
        dets.append({"track_id": tid, "box": box, "keypoints": kp})
    return {"frame": frame, "source_frame": frame, "time": frame / fps, "detections": dets}


def test_group_cropping_validation_and_historical_alignment(tmp_path: Path) -> None:
    fps = 10.0
    rows = []
    # Dispersal [0, 60): tracks 1,2,3 start compact near (200,200) and move outward > 4cm
    for f in range(80):
        d = max(0.0, f - 5) * 3.0
        coords = {
            1: (200.0 - d, 200.0),
            2: (200.0 + d, 200.0),
            3: (200.0, 200.0 + d),
        }
        rows.append(_make_track_row(f, fps, coords))
    store = InMemoryTrackStore(rows)
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    binding = {"source_video": {"fps": fps, "frame_count": 80}}
    params = HeuristicParameters()

    pre_events = [
        {
            "event_id": "dispersal_raw_1",
            "behavior": "Dispersal",
            "canonical_behavior": "dispersal",
            "mouse_ids": [1, 2, 3],
            "start_frame": 0,
            "end_frame": 75,
            "observed_all_participants_frame_count": 75,
        }
    ]
    post_events = [
        {
            "event_id": "dispersal_crop_1",
            "parent_candidate_event_id": "dispersal_raw_1",
            "behavior": "Dispersal",
            "canonical_behavior": "dispersal",
            "mouse_ids": [1, 2, 3],
            "start_frame": 2,
            "end_frame": 65,
            "observed_all_participants_frame_count": 63,
            "canonical_support_frame_count": 63,
        }
    ]

    kin = audit_group_segment_kinematics(post_events[0], store, scale, fps, params)
    assert kin["duration_ok"] is True
    assert kin["endpoint_observed_ok"] is True
    assert kin["kinematic_ok"] is True
    assert kin["full_crop_precision_ok"] is True

    hist_file = tmp_path / "hist.json"
    hist_file.write_text(
        json.dumps(
            {
                "decisions": {
                    "dispersal_00000000_00000060_abc": {"behavior": "dispersal", "status": "pass"},
                    "dispersal_00005000_00005060_def": {"behavior": "dispersal", "status": "fail"},
                }
            }
        ),
        encoding="utf-8",
    )
    hist = load_historical_review_decisions(hist_file)
    summary, audits = summarize_group_cropping_validation(
        pre_events,
        post_events,
        store,
        binding,
        scale,
        params,
        historical_reviews=hist,
    )
    assert summary["aggregate"]["total_raw_group_candidates"] == 1
    assert summary["aggregate"]["total_post_crop_group_segments"] == 1
    assert summary["aggregate"]["total_boundary_adjusted"] == 1
    disp = summary["by_behavior"]["dispersal"]
    assert disp["full_crop_precision_rate"] == 1.0
    assert disp["historical_alignment"]["retained_pass_count"] == 1
    assert disp["historical_alignment"]["eliminated_or_recropped_fail_count"] == 1
    assert audits["dispersal_crop_1"]["boundary_delta"]["changed"] is True


def test_group_cropping_validation_social_clustering_uses_group_config() -> None:
    fps = 10.0
    rows = []
    # 3 mice start spread out before frame 10 and converge into a compact cluster by frame 20..30
    for f in range(40):
        spread = max(10.0, 220.0 - f * 8.0) if f < 25 else 10.0
        coords = {
            1: (200.0 - spread, 200.0),
            2: (200.0 + spread, 200.0),
            3: (200.0, 200.0 + spread),
        }
        rows.append(_make_track_row(f, fps, coords))
    store = InMemoryTrackStore(rows)
    scale = {"cm_per_pixel": 0.1, "reference_body_cm": 7.918}
    event = {
        "event_id": "sc_crop_1",
        "behavior": "Social clustering",
        "canonical_behavior": "social_clustering",
        "mouse_ids": [1, 2, 3],
        "start_frame": 10,
        "end_frame": 30,
        "observed_all_participants_frame_count": 20,
        "canonical_support_frame_count": 20,
    }
    kin = audit_group_segment_kinematics(event, store, scale, fps, HeuristicParameters())
    assert kin["duration_ok"] is True
    assert kin["endpoint_observed_ok"] is True
    assert kin["kinematic_ok"] is True
    assert kin["full_crop_precision_ok"] is True


def test_review_site_schema_and_api_resilience(tmp_path: Path) -> None:
    assert validate_review_bundle_schema(DEMO_FALLBACK_LONG_VIDEO_BUNDLE) is True
    assert validate_review_bundle_schema({"videos": "bad"}) is False

    html_fallback = render_long_video_group_review_page(embedded_bundle=None)
    assert "长视频群体行为裁剪精度验证与人工审核台" in html_fallback
    assert "is_fallback_demo" in html_fallback

    run_dir = tmp_path / "run_v22"
    raw_pkg = run_dir / "clips_raw" / "Dispersal" / "ev1"
    ov_pkg = run_dir / "clips_overlay" / "Dispersal" / "ev1"
    for pkg in (raw_pkg, ov_pkg):
        pkg.mkdir(parents=True)
        (pkg / "clip.mp4").write_bytes(b"0123456789abcdef")
        for fn in ("annotation.json", "metadata.json", "tracks.json"):
            (pkg / fn).write_text("{}", encoding="utf-8")

    (run_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "video_key": "video22",
                "title": "Video Project 22",
                "duration_s": 595.5,
                "fps": 30.0,
                "frame_count": 17864,
                "inference_source": "delivered_tracks_import",
                "aggregate": {
                    "total_raw_group_candidates": 1,
                    "total_post_crop_group_segments": 1,
                    "total_rejected_by_fine_crop": 0,
                    "total_boundary_adjusted": 1,
                    "duration_pass_rate": 1.0,
                    "endpoint_observed_rate": 1.0,
                    "kinematic_pass_rate": 1.0,
                    "full_crop_precision_rate": 1.0,
                },
                "by_behavior": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "review_candidates.json").write_text(
        json.dumps(
            [
                {
                    "event_id": "ev1",
                    "behavior": "Dispersal",
                    "canonical_behavior": "dispersal",
                    "mouse_ids": [1, 2, 3],
                    "start_frame": 10,
                    "end_frame": 170,
                    "frame_count": 160,
                    "start_time_s": 0.33,
                    "end_time_s": 5.67,
                    "duration_s": 5.33,
                    "raw_package_dir": str(raw_pkg),
                    "overlay_package_dir": str(ov_pkg),
                    "audit": {"full_crop_precision_ok": True, "boundary_delta": {"changed": True}},
                }
            ]
        ),
        encoding="utf-8",
    )

    site_dir = tmp_path / "site"
    res = build_long_video_group_review_site([run_dir], site_dir, copy_videos=True, port=0)
    assert res["total_review_clips"] == 1
    assert res["copied_videos"] == 2

    server = create_long_video_group_review_server(site_dir, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/") as r:
            assert r.status == 200
            assert "text/html" in r.headers.get("Content-Type", "")
            assert "Video Project 22" in r.read().decode("utf-8")

        with urllib.request.urlopen(f"{base}/api/bundle") as r:
            assert r.status == 200
            assert "application/json" in r.headers.get("Content-Type", "")
            bundle = json.loads(r.read().decode("utf-8"))
            assert validate_review_bundle_schema(bundle)
            assert bundle["is_fallback_demo"] is False
            assert len(bundle["clips"]) == 1

        req = urllib.request.Request(
            f"{base}/api/reviews",
            data=json.dumps({"clip_uid": "video22::ev1", "status": "pass", "note": "good"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 200

        with urllib.request.urlopen(f"{base}/api/reviews") as r:
            rev = json.loads(r.read().decode("utf-8"))
            assert rev["decisions"]["video22::ev1"]["status"] == "pass"

        # Malformed POST returns 400 JSON error without crashing server
        bad_req = urllib.request.Request(
            f"{base}/api/reviews",
            data=b"{not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_req)
        assert exc_info.value.code == 400

        # HTTP Range test on MP4 clip
        range_req = urllib.request.Request(
            f"{base}/{bundle['clips'][0]['overlay_url']}",
            headers={"Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(range_req) as r:
            assert r.status == 206
            assert r.read() == b"0123"
    finally:
        server.shutdown()
        server.server_close()


def test_compat_unpickler_remaps_numpy2_core(tmp_path: Path) -> None:
    arr = np.asarray([1.5, 2.5], dtype=np.float32)
    payload = [{"frame": 0, "timestamp_seconds": 0.0, "pose_detections": [{"keypoints_px": arr}]}]
    buf = io.BytesIO()
    pickle.dump(payload, buf, protocol=0)
    raw_bytes = buf.getvalue().replace(b"numpy.core", b"numpy._core")
    loaded = _CompatUnpickler(io.BytesIO(raw_bytes)).load()
    np.testing.assert_allclose(loaded[0]["pose_detections"][0]["keypoints_px"], arr)

    chunk = tmp_path / "yolo_results.000000.000000.pkl.gz"
    with gzip.open(chunk, "wb") as handle:
        handle.write(raw_bytes)
    reloaded = list(_iter_cache_records(tmp_path))
    assert len(reloaded) == 1
    np.testing.assert_allclose(reloaded[0]["pose_detections"][0]["keypoints_px"], arr)


def test_cached_track_store_matches_track_store(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "tracks.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    for i in range(6):
        payload = {"frame": i, "source_frame": i, "time": i / 30.0, "source_fps": 30.0, "detections": []}
        conn.execute("INSERT INTO frames VALUES(?, ?)", (i, json.dumps(payload)))
    conn.commit()
    conn.close()

    raw_store = TrackStore(db)
    for preload_limit in (100, 1):
        cached = CachedTrackStore(raw_store, preload_max_frames=preload_limit)
        try:
            assert list(cached.rows(1, 5)) == list(raw_store.rows(1, 5))
            assert list(cached.rows(2, 5, rebase=True)) == list(raw_store.rows(2, 5, rebase=True))
        finally:
            cached.close()


def test_pair_rule_masks_monkeypatch_without_cache_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    from mouse_behavior.evaluation import confirmed_rules as rules
    from mouse_behavior.evaluation import social_event_geometry as seg
    from mouse_behavior.evaluation.exported_dataset_models import ExportedFeatures

    n = 12
    f = ExportedFeatures(
        fps=30.0,
        track_ids=np.array([1, 2], dtype=int),
        valid=np.ones((n, 2), dtype=bool),
        centers_cm=np.zeros((n, 2, 2), dtype=float),
        speed_cm_s=np.zeros((n, 2), dtype=float),
        velocity_cm_s=np.zeros((n, 2, 2), dtype=float),
        pair_i=np.array([0], dtype=int),
        pair_j=np.array([1], dtype=int),
        pair_distance_cm=np.full((n, 1), 10.0, dtype=float),
        pair_closing_speed_cm_s=np.zeros((n, 1), dtype=float),
        pair_direction_similarity=np.zeros((n, 1), dtype=float),
        pair_nose_head_cm=np.full((n, 1), 10.0, dtype=float),
        pair_nose_tail_cm=np.full((n, 1), 10.0, dtype=float),
        nearest_distance_cm=np.full((n, 2), 10.0, dtype=float),
        mean_nearest_distance_cm=np.full((n, 2), 10.0, dtype=float),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
        pair_box_gap_cm=np.full((n, 1), 2.0, dtype=float),
    )

    monkeypatch.setattr(seg, "body_footprint_distances", lambda features, edge, *, max_body_lengths, frame_mask=None: np.full(n, 5.0))
    assert seg.observed_body_separation(f, 0, max_body_lengths=0.5, _geom_cache={}).all()

    monkeypatch.setattr(rules, "endpoint_distances", lambda features, edge, kind: (np.full(n, 5.0), np.full(n, 5.0)))
    monkeypatch.setattr(rules, "snout_surface_distances", lambda features, edge: (np.full(n, 5.0), np.full(n, 5.0)))
    monkeypatch.setattr(rules, "body_footprint_distances", lambda features, edge, *, max_body_lengths, frame_mask=None: np.full(n, 5.0))
    monkeypatch.setattr(rules, "observed_body_separation", lambda features, edge, *, max_body_lengths: np.ones(n, dtype=bool))
    res = rules.pair_rule_masks(f, HeuristicParameters())
    assert "head_contact" in res

    monkeypatch.setattr(
        rules,
        "_base_pair_rule_masks",
        lambda features, params, config, *, allow_observed_attack_extension=True, diagnostics=None: {
            name: np.zeros((n, 1), dtype=bool)
            for name in ("approach", "together", "attack", "chasing", "head_contact", "tail_contact", "long_contact_review", "contact_start_context")
        }
        | {"head_contact_seconds": np.zeros(1), "tail_contact_seconds": np.zeros(1)},
    )
    res2 = rules.pair_rule_masks(f, HeuristicParameters())
    assert not res2["head_contact"].any()



