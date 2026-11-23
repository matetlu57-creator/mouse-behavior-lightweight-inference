"""Unit and HTTP resilience tests for the Mouse Clip Review & Collection website."""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import pytest

from mouse_behavior.visualization.mouse_clip_cropper import (
    get_ffmpeg_executable,
    is_ffmpeg_available,
)
from mouse_behavior.visualization.mouse_clip_review_site import (
    create_mouse_clip_review_server,
)
from mouse_behavior.visualization.mouse_clip_review_store import (
    MouseClipReviewStore,
    make_clip_uid,
    validate_output_dir_separate,
)

_KEYPOINT_NAMES = [
    "nose",
    "right_ear",
    "left_ear",
    "base_of_neck",
    "left_hip",
    "right_hip",
    "base_of_tail",
]
_SKELETON_EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
    [3, 5],
    [4, 6],
    [5, 6],
]


def _write_synthetic_h264_mp4(
    dest: Path,
    *,
    num_frames: int = 12,
    width: int = 160,
    height: int = 120,
    fps: float = 30.0,
) -> None:
    """Create a tiny valid H.264 MP4 in ``tmp_path`` using imageio-ffmpeg."""
    exe = get_ffmpeg_executable()
    if not exe:
        pytest.skip("FFmpeg executable is unavailable")
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (idx * 20) % 256
            frame[:, :, 1] = 60
            frame[:, :, 2] = 120
            cv2.rectangle(
                frame,
                (10 + idx * 4, 20),
                (42 + idx * 4, 52),
                (255, 255, 255),
                -1,
            )
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        rc = proc.wait(timeout=60)
        assert rc == 0, f"ffmpeg synthetic encode failed: {proc.stderr.read().decode(errors='replace')}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def _make_detection(track_id: int, x: float, y: float) -> dict[str, Any]:
    kpts = [[x + 5.0 + i * 2.0, y + 5.0 + i * 2.0, 0.99] for i in range(7)]
    return {
        "track_id": track_id,
        "box": [x, y, x + 36.0, y + 28.0],
        "keypoints": kpts,
        "detection_confidence": 0.85,
        "class_id": 0,
    }


def _create_clip_package(
    source_dir: Path,
    *,
    behavior: str,
    clip_id: str,
    mouse_ids: list[int],
    participants: list[dict[str, Any]],
    num_frames: int = 12,
    fps: float = 30.0,
    ann_start: int = 2,
    ann_end: int = 9,
    dropout_frames_by_id: dict[int, set[int]] | None = None,
    extra_track_ids: list[int] | None = None,
    distance_scale: dict[str, Any] | None = None,
) -> Path:
    clip_dir = source_dir / behavior / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_h264_mp4(clip_dir / "clip.mp4", num_frames=num_frames, fps=fps)

    annotation = {
        "behavior": behavior,
        "mouse_ids": mouse_ids,
        "participants": participants,
        "confidence": "certain",
        "frame_range": {"start": ann_start, "end": ann_end},
        "time_range": {"start": round(ann_start / fps, 6), "end": round((ann_end + 1) / fps, 6)},
    }
    (clip_dir / "annotation.json").write_text(
        json.dumps(annotation, ensure_ascii=False), encoding="utf-8"
    )

    metadata: dict[str, Any] = {
        "schema_version": "2.0",
        "clip": {
            "filename": "clip.mp4",
            "fps": fps,
            "width": 160,
            "height": 120,
            "frame_count": num_frames,
        },
        "tracks": {
            "frame_origin": 0,
            "time_origin": 0.0,
            "coordinate_system": "clip_pixels",
        },
        "pose": {
            "keypoint_names": list(_KEYPOINT_NAMES),
            "skeleton_edges": [list(edge) for edge in _SKELETON_EDGES],
        },
        "source": {
            "video_filename": "WIN_20260920_10_31_19_Pro_part001.mp4",
            "clip": {
                "start_frame": 100,
                "end_frame": 100 + num_frames - 1,
                "start_time": round(100 / fps, 6),
                "end_time": round((100 + num_frames) / fps, 6),
            },
            "annotation": {
                "start_frame": 100 + ann_start,
                "end_frame": 100 + ann_end,
                "start_time": round((100 + ann_start) / fps, 6),
                "end_time": round((100 + ann_end + 1) / fps, 6),
            },
            "crop_region": {"x": 0, "y": 0, "w": 160, "h": 120},
        },
    }
    if distance_scale is not None:
        metadata["distance_scale"] = distance_scale
    (clip_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )

    dropouts = dropout_frames_by_id or {}
    all_ids = list(mouse_ids) + list(extra_track_ids or [])
    tracks: list[dict[str, Any]] = []
    for f_idx in range(num_frames):
        dets = []
        for offset, tid in enumerate(all_ids):
            if f_idx in dropouts.get(tid, set()):
                continue
            dets.append(_make_detection(tid, 16.0 + offset * 30.0 + f_idx * 2.0, 24.0 + offset * 10.0))
        tracks.append(
            {
                "frame": f_idx,
                "time": round(f_idx / fps, 6),
                "detections": dets,
            }
        )
    (clip_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    return clip_dir


def _snapshot_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            snapshot[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snapshot


def _http_request(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    host_header: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        req_headers = dict(headers or {})
        if host_header is not None:
            req_headers["Host"] = host_header
        conn.request(method, path, body=body, headers=req_headers)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


def _http_json(
    port: int,
    method: str,
    path: str,
    *,
    payload: Any = None,
    token: str | None = None,
    extra_headers: dict[str, str] | None = None,
    host_header: str | None = None,
) -> tuple[int, dict[str, str], Any]:
    headers = dict(extra_headers or {})
    body: bytes | None = None
    if payload is not None:
        headers.setdefault("Content-Type", "application/json")
        body = json.dumps(payload).encode("utf-8")
    if token is not None:
        headers["X-Review-Token"] = token
    status, resp_headers, data = _http_request(
        port, method, path, headers=headers, body=body, host_header=host_header
    )
    parsed = json.loads(data.decode("utf-8")) if data else None
    return status, resp_headers, parsed


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source_dir = tmp_path / "readonly_mouse"
    output_dir = tmp_path / "review_outputs"

    _create_clip_package(
        source_dir,
        behavior="Approach",
        clip_id="Approach_WIN_20260920_000102-000109",
        mouse_ids=[10, 62],
        participants=[
            {"role_key": "approaching", "role_name": "Approaching", "track_ids": [62]},
            {"role_key": "approached", "role_name": "Approached", "track_ids": [10]},
        ],
        num_frames=12,
        ann_start=2,
        ann_end=9,
        dropout_frames_by_id={10: {3, 4, 5}},
        extra_track_ids=[99],
        distance_scale={"cm_per_pixel": 0.099, "unit": "cm"},
    )
    _create_clip_package(
        source_dir,
        behavior="Group locomotion",
        clip_id="Group locomotion_WIN_20260920_000200-000208",
        mouse_ids=[9, 143, 165],
        participants=[],
        num_frames=10,
        ann_start=1,
        ann_end=7,
    )

    approach_uid = make_clip_uid("Approach", "Approach_WIN_20260920_000102-000109")
    group_uid = make_clip_uid("Group locomotion", "Group locomotion_WIN_20260920_000200-000208")
    return source_dir, output_dir, approach_uid, group_uid


@pytest.fixture
def live_server(synthetic_dataset: tuple[Path, Path, str, str]):
    source_dir, output_dir, approach_uid, group_uid = synthetic_dataset
    before_snapshot = _snapshot_tree(source_dir)
    server = create_mouse_clip_review_server(source_dir, output_dir, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "server": server,
            "port": server.server_port,
            "source_dir": source_dir,
            "output_dir": output_dir,
            "approach_uid": approach_uid,
            "group_uid": group_uid,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        # Source directory must remain bit-for-bit immutable after every test.
        assert _snapshot_tree(source_dir) == before_snapshot


def test_readonly_guard_and_loopback_host_enforcement(tmp_path: Path) -> None:
    source_dir = tmp_path / "mouse_src"
    source_dir.mkdir()

    with pytest.raises(ValueError, match="strictly outside"):
        validate_output_dir_separate(source_dir, source_dir)

    with pytest.raises(ValueError, match="strictly outside"):
        validate_output_dir_separate(source_dir, source_dir / "nested_output")

    with pytest.raises(ValueError, match="strictly outside"):
        MouseClipReviewStore(source_dir=source_dir, output_dir=source_dir / "sub")

    with pytest.raises(ValueError, match="loopback"):
        create_mouse_clip_review_server(
            source_dir=source_dir,
            output_dir=tmp_path / "out",
            host="0.0.0.0",
            port=0,
        )


def test_catalog_scan_space_behavior_and_partial_dropout_without_fabrication(
    live_server: dict[str, Any],
) -> None:
    port = live_server["port"]
    approach_uid = live_server["approach_uid"]
    group_uid = live_server["group_uid"]

    assert " " not in group_uid and "/" not in group_uid

    status, headers, catalog = _http_json(port, "GET", "/api/catalog")
    assert status == 200
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert catalog["schema"] == "mouse_clip_catalog_v1"
    assert catalog["total_clips"] == 2
    assert {b["name"]: b["count"] for b in catalog["behaviors"]} == {
        "Approach": 1,
        "Group locomotion": 1,
    }

    by_uid = {c["clip_uid"]: c for c in catalog["clips"]}
    assert by_uid[group_uid]["behavior"] == "Group locomotion"
    assert by_uid[group_uid]["target_mouse_ids"] == [9, 143, 165]
    assert by_uid[approach_uid]["annotation_range"] == [2, 9]

    # Detail for Group locomotion (behavior folder with spaces)
    status, _, group_detail = _http_json(port, "GET", f"/api/clip/{quote(group_uid, safe='')}")
    assert status == 200
    assert group_detail["schema"] == "mouse_clip_detail_v1"
    assert group_detail["clip"]["behavior"] == "Group locomotion"
    assert len(group_detail["frames"]) == 10

    # Detail for Approach with partial track dropout (mouse 10 absent in frames 3, 4, 5)
    status, _, detail = _http_json(port, "GET", f"/api/clip/{quote(approach_uid, safe='')}")
    assert status == 200
    assert detail["metadata"]["distance_scale"] == {"cm_per_pixel": 0.099, "unit": "cm"}
    vis_map = {t["track_id"]: t for t in detail["visible_tracks"]}
    assert set(vis_map) == {10, 62, 99}
    assert vis_map[10] == {"track_id": 10, "frame_count": 9, "first_frame": 0, "last_frame": 11}
    assert vis_map[62]["frame_count"] == 12

    # Never fabricate missing mouse 10 in dropout frames 3, 4, 5
    for dropout_frame in (3, 4, 5):
        tids = {d["track_id"] for d in detail["frames"][dropout_frame]["detections"]}
        assert 10 not in tids
        assert {62, 99}.issubset(tids)


def test_static_spa_shell_no_cdn_and_resilience_markers(live_server: dict[str, Any]) -> None:
    port = live_server["port"]
    assets: dict[str, str] = {}
    for route, expected_mime in (
        ("/", "text/html"),
        ("/app.css", "text/css"),
        ("/app.js", "application/javascript"),
    ):
        status, headers, raw = _http_request(port, "GET", route)
        assert status == 200
        assert expected_mime in headers["content-type"]
        assert headers["x-content-type-options"] == "nosniff"
        text = raw.decode("utf-8")
        assets[route] = text
        external_urls = re.findall(r"https?://[^\s'\"<>)]+", text)
        assert external_urls == [], f"unexpected external URL in {route}: {external_urls}"

    assert "offline-banner" in assets["/"]
    assert "演示回退数据" in assets["/"]
    assert "function escHtml" in assets["/app.js"]
    assert "fetchJson" in assets["/app.js"]
    assert "application/json" in assets["/app.js"]
    assert "X-Review-Token" in assets["/app.js"]
    assert "flushPendingReviews" in assets["/app.js"]


def test_security_headers_host_origin_token_and_port_exclusivity(
    live_server: dict[str, Any],
) -> None:
    port = live_server["port"]
    source_dir = live_server["source_dir"]
    output_dir = live_server["output_dir"]
    approach_uid = live_server["approach_uid"]

    # Loopback Host variants allowed: 127.0.0.1, localhost, bracketed IPv6 [::1]:port
    token = ""
    for host_hdr in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"):
        status, headers, health = _http_json(port, "GET", "/api/health", host_header=host_hdr)
        assert status == 200
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["cache-control"] == "no-store"
        assert health["read_only_source"] is True
        assert health["source_exists"] is True
        token = health["session_token"]
        assert len(token) >= 20

    # Non-loopback Host rejected
    status, _, _ = _http_json(port, "GET", "/api/health", host_header=f"evil.example:{port}")
    assert status == 403

    # Missing or invalid X-Review-Token rejected on POST and DELETE
    review_body = {"clip_uid": approach_uid, "status": "pass", "tags": [], "note": ""}
    status, _, _ = _http_json(port, "POST", "/api/reviews", payload=review_body)
    assert status == 403
    status, _, _ = _http_json(port, "POST", "/api/reviews", payload=review_body, token="wrong-token")
    assert status == 403
    status, _, _ = _http_json(port, "DELETE", "/api/collections/nonexistent")
    assert status == 403

    # Cross-origin mutation rejected even with valid token
    status, _, _ = _http_json(
        port,
        "POST",
        "/api/reviews",
        payload=review_body,
        token=token,
        extra_headers={"Origin": "http://evil.example"},
    )
    assert status == 403

    # Duplicate bind on the same loopback port fails with OSError
    with pytest.raises(OSError):
        create_mouse_clip_review_server(source_dir, output_dir, host="127.0.0.1", port=port)


def test_reviews_crud_export_and_corrupt_ledger_preservation(live_server: dict[str, Any]) -> None:
    port = live_server["port"]
    output_dir = live_server["output_dir"]
    approach_uid = live_server["approach_uid"]

    _, _, health = _http_json(port, "GET", "/api/health")
    token = health["session_token"]

    # Initial empty state
    status, _, initial = _http_json(port, "GET", "/api/reviews")
    assert status == 200
    assert initial["schema"] == "mouse_clip_reviews_v1"
    assert initial["reviews"] == {}

    # Malformed JSON / wrong content-type / invalid fields
    status, _, _ = _http_request(
        port,
        "POST",
        "/api/reviews",
        headers={"Content-Type": "text/plain", "X-Review-Token": token},
        body=b"{}",
    )
    assert status == 415

    status, _, _ = _http_request(
        port,
        "POST",
        "/api/reviews",
        headers={"Content-Type": "application/json", "X-Review-Token": token},
        body=b"{not-json",
    )
    assert status == 400

    status, _, _ = _http_json(
        port,
        "POST",
        "/api/reviews",
        payload={"clip_uid": approach_uid, "status": "bogus_status"},
        token=token,
    )
    assert status == 400

    status, _, _ = _http_json(
        port,
        "POST",
        "/api/reviews",
        payload={"clip_uid": approach_uid, "status": "needs_recrop", "crop_range": [5, 5]},
        token=token,
    )
    assert status == 400

    # Valid save
    status, _, saved = _http_json(
        port,
        "POST",
        "/api/reviews",
        payload={
            "clip_uid": approach_uid,
            "status": "needs_recrop",
            "tags": ["边界不准", "ID跳变/缺失"],
            "note": "trim edge frames",
            "crop_range": [2, 10],
        },
        token=token,
    )
    assert status == 200
    assert saved["review"]["status"] == "needs_recrop"
    assert saved["review"]["crop_range"] == [2, 10]

    # Export JSON and UTF-8-BOM CSV
    status, _, raw_json = _http_request(port, "GET", "/api/reviews/export?format=json")
    assert status == 200
    exported_json = json.loads(raw_json.decode("utf-8"))
    assert exported_json["reviews"][approach_uid]["note"] == "trim edge frames"

    status, headers, raw_csv = _http_request(port, "GET", "/api/reviews/export?format=csv")
    assert status == 200
    assert "text/csv" in headers["content-type"]
    assert raw_csv.startswith(b"\xef\xbb\xbf")
    csv_text = raw_csv.decode("utf-8-sig")
    assert approach_uid in csv_text and "needs_recrop" in csv_text and "trim edge frames" in csv_text

    # Corrupt reviews.json -> GET 500, POST 409, exact corrupt bytes preserved
    corrupt_bytes = b'{"reviews": broken_json_payload'
    reviews_file = output_dir / "reviews.json"
    reviews_file.write_bytes(corrupt_bytes)

    status, _, err_get = _http_json(port, "GET", "/api/reviews")
    assert status == 500
    assert err_get["corrupt_store"] is True

    status, _, err_post = _http_json(
        port,
        "POST",
        "/api/reviews",
        payload={"clip_uid": approach_uid, "status": "pass", "tags": [], "note": ""},
        token=token,
    )
    assert status == 409
    assert err_post["corrupt_store"] is True
    assert reviews_file.read_bytes() == corrupt_bytes


@pytest.mark.skipif(not is_ffmpeg_available(), reason="imageio-ffmpeg / ffmpeg not available")
def test_manual_collection_crop_range_streaming_batch_delete_and_corrupt_manifest(
    live_server: dict[str, Any],
) -> None:
    port = live_server["port"]
    output_dir = live_server["output_dir"]
    approach_uid = live_server["approach_uid"]
    group_uid = live_server["group_uid"]

    _, _, health = _http_json(port, "GET", "/api/health")
    token = health["session_token"]

    # Source video HTTP 206 Range streaming
    status, headers, chunk = _http_request(
        port,
        "GET",
        f"/api/video/{quote(approach_uid, safe='')}",
        headers={"Range": "bytes=0-255"},
    )
    assert status == 206
    assert headers["content-range"].startswith("bytes 0-255/")
    assert len(chunk) == 256
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["cache-control"] == "no-store"

    # Invalid Range -> 416
    status, _, _ = _http_request(
        port,
        "GET",
        f"/api/video/{quote(approach_uid, safe='')}",
        headers={"Range": "bytes=99999999-999999999"},
    )
    assert status == 416

    # 1) POST /api/collect with padding_frames=0 + include_overlay=True + custom fields
    collect_req = {
        "clip_uid": approach_uid,
        "start_frame": 2,
        "end_frame": 8,
        "behavior": "Attack",
        "mouse_ids": [10, 62],
        "participants": [
            {"role_key": "attacker", "role_name": "Attacker", "track_ids": [62]},
            {"role_key": "attacked", "role_name": "Attacked", "track_ids": [10]},
        ],
        "padding_frames": 0,
        "include_overlay": True,
        "confidence": "certain",
        "note": "exact 6-frame window",
        "overwrite": False,
    }
    status, _, res1 = _http_json(port, "POST", "/api/collect", payload=collect_req, token=token)
    assert status == 200
    col1 = res1["collection"]
    export_id_1 = col1["export_id"]
    out_dir_1 = Path(col1["output_dir"])
    assert col1["frame_count"] == 6
    assert col1["clip_start_frame"] == 2 and col1["clip_end_frame"] == 8
    assert sorted(col1["files"]) == [
        "annotation.json",
        "clip.mp4",
        "clip_overlay.mp4",
        "metadata.json",
        "tracks.json",
    ]
    assert (out_dir_1 / "clip.mp4").is_file()
    assert (out_dir_1 / "clip_overlay.mp4").is_file()

    # Verify exported H.264 clip.mp4 and clip_overlay.mp4 decode exact frame count
    for vname in ("clip.mp4", "clip_overlay.mp4"):
        cap = cv2.VideoCapture(str(out_dir_1 / vname))
        try:
            decoded = 0
            while cap.grab():
                decoded += 1
            assert decoded == 6
        finally:
            cap.release()

    # Verify 0-based tracks.json with source_frame and measured-only detections (no fabricated mouse 10 on source frames 3,4,5)
    rebased_tracks = json.loads((out_dir_1 / "tracks.json").read_text(encoding="utf-8"))
    assert len(rebased_tracks) == 6
    assert [row["frame"] for row in rebased_tracks] == [0, 1, 2, 3, 4, 5]
    assert [row["source_frame"] for row in rebased_tracks] == [2, 3, 4, 5, 6, 7]
    for row in rebased_tracks:
        tids = {d["track_id"] for d in row["detections"]}
        if row["source_frame"] in {3, 4, 5}:
            assert 10 not in tids
        else:
            assert 10 in tids

    # Verify metadata.json schema_version 2.0 + preserved distance_scale
    meta1 = json.loads((out_dir_1 / "metadata.json").read_text(encoding="utf-8"))
    assert meta1["schema_version"] == "2.0"
    assert meta1["clip"]["frame_count"] == 6
    assert meta1["distance_scale"] == {"cm_per_pixel": 0.099, "unit": "cm"}
    assert meta1["manual_collection"]["export_id"] == export_id_1

    # Verify 6 canonical keys in annotation.json + custom behavior/participants
    ann1 = json.loads((out_dir_1 / "annotation.json").read_text(encoding="utf-8"))
    for key in ("behavior", "mouse_ids", "participants", "confidence", "frame_range", "time_range"):
        assert key in ann1
    assert ann1["behavior"] == "Attack"
    assert ann1["mouse_ids"] == [10, 62]
    assert [p["role_name"] for p in ann1["participants"]] == ["Attacker", "Attacked"]
    assert ann1["frame_range"] == {"start": 0, "end": 5}

    # Exported video HTTP 206 Range streaming (raw & overlay)
    for variant in ("raw", "overlay"):
        status, headers, seg = _http_request(
            port,
            "GET",
            f"/api/exported-video/{quote(export_id_1, safe='')}?variant={variant}",
            headers={"Range": "bytes=0-127"},
        )
        assert status == 206
        assert headers["content-range"].startswith("bytes 0-127/")
        assert len(seg) == 128

    # 2) Duplicate export with overwrite=False -> 409 (NOT corrupt_store) vs overwrite=True -> 200
    status, _, dup_err = _http_json(port, "POST", "/api/collect", payload=collect_req, token=token)
    assert status == 409
    assert not dup_err.get("corrupt_store")
    assert re.search(r"exist|overwrite", dup_err.get("error", ""), re.IGNORECASE)

    overwrite_req = dict(collect_req, overwrite=True, note="overwritten cleanly")
    status, _, res_ow = _http_json(port, "POST", "/api/collect", payload=overwrite_req, token=token)
    assert status == 200
    assert res_ow["collection"]["export_id"] == export_id_1
    assert res_ow["collection"]["note"] == "overwritten cleanly"

    # 3) POST /api/crop alias with padding_frames=2 and include_overlay=False
    crop_req = {
        "clip_uid": group_uid,
        "start_frame": 3,
        "end_frame": 7,
        "behavior": "Group locomotion",
        "mouse_ids": [9, 143],
        "padding_frames": 2,
        "include_overlay": False,
        "overwrite": False,
    }
    status, _, res2 = _http_json(port, "POST", "/api/crop", payload=crop_req, token=token)
    assert status == 200
    col2 = res2["collection"]
    export_id_2 = col2["export_id"]
    out_dir_2 = Path(col2["output_dir"])
    assert col2["clip_start_frame"] == 1 and col2["clip_end_frame"] == 9
    assert col2["frame_count"] == 8
    assert col2["overlay_video_url"] is None
    assert not (out_dir_2 / "clip_overlay.mp4").exists()
    ann2 = json.loads((out_dir_2 / "annotation.json").read_text(encoding="utf-8"))
    # Selected [3, 7) inside clipped [1, 9) -> clip-relative inclusive [2, 5]
    assert ann2["frame_range"] == {"start": 2, "end": 5}

    # Collections listing & CSV export (UTF-8 BOM)
    status, _, listed = _http_json(port, "GET", "/api/collections")
    assert status == 200
    assert listed["total"] == 2
    status, _, raw_col_csv = _http_request(port, "GET", "/api/collections/export?format=csv")
    assert status == 200
    assert raw_col_csv.startswith(b"\xef\xbb\xbf")
    assert export_id_1 in raw_col_csv.decode("utf-8-sig")

    # 4) Batch collection POST /api/collect/batch (202) + polling
    batch_payload = {
        "items": [
            {
                "clip_uid": group_uid,
                "start_frame": 0,
                "end_frame": 4,
                "behavior": "Group locomotion",
                "mouse_ids": [9],
                "padding_frames": 0,
                "include_overlay": False,
            },
            {
                "clip_uid": "nonexistent_uid",
                "start_frame": 0,
                "end_frame": 2,
                "behavior": "Moving",
                "mouse_ids": [1],
            },
        ]
    }
    status, _, accepted = _http_json(
        port, "POST", "/api/collect/batch", payload=batch_payload, token=token
    )
    assert status == 202
    assert accepted["status"] == "accepted"
    job_id = accepted["job_id"]

    job_state: dict[str, Any] = {}
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status, _, job_state = _http_json(port, "GET", f"/api/collect/jobs/{job_id}")
        assert status == 200
        if job_state["status"] in {"completed", "completed_with_errors", "failed"}:
            break
        time.sleep(0.1)
    assert job_state["status"] == "completed_with_errors"
    assert job_state["completed"] == 1
    assert job_state["failed"] == 1
    batch_export_id = job_state["results"][0]["export_id"]

    # 5) DELETE /api/collections/<export_id>
    status, _, del_res = _http_json(
        port, "DELETE", f"/api/collections/{quote(batch_export_id, safe='')}", token=token
    )
    assert status == 200
    assert del_res == {"status": "ok", "export_id": batch_export_id}
    assert not Path(job_state["results"][0]["output_dir"]).exists()

    # 6) Corrupt collections_manifest.json -> GET 500, POST 409, DELETE 409, bytes preserved
    manifest_path = output_dir / "collections_manifest.json"
    corrupt_manifest = b'{"collections": [corrupt'
    manifest_path.write_bytes(corrupt_manifest)

    status, _, err_get = _http_json(port, "GET", "/api/collections")
    assert status == 500 and err_get["corrupt_store"] is True

    status, _, err_collect = _http_json(
        port, "POST", "/api/collect", payload=overwrite_req, token=token
    )
    assert status == 409 and err_collect["corrupt_store"] is True

    status, _, err_del = _http_json(
        port, "DELETE", f"/api/collections/{quote(export_id_2, safe='')}", token=token
    )
    assert status == 409 and err_del["corrupt_store"] is True
    assert manifest_path.read_bytes() == corrupt_manifest
