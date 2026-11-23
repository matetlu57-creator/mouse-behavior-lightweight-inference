"""Unit, smoke, and regression tests for the three-group-behaviors comparison website."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest

from mouse_behavior.visualization.clip_overlay_renderer import (
    ROLE_COLORS,
    render_annotated_original_video,
    render_cropped_comparison_video,
    video_frame_count,
)
from mouse_behavior.visualization.comparison_site import (
    DEFAULT_CROPS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_DISPERSAL_DIR,
    DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON,
    DEFAULT_GROUP_LOCOMOTION_DIR,
    DEFAULT_ORIG_ANNOTATED_DIR,
    DEFAULT_SOCIAL_CLUSTERING_DIR,
    DEMO_FALLBACK_DATA_GROUP_BEHAVIORS,
    _select_demo_fallback,
    create_comparison_server,
    render_comparison_page,
)


def _write_synthetic_mp4(path: Path, frame_count: int = 16, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    for f in range(frame_count):
        writer.write(np.full((120, 160, 3), (f * 7) % 255, dtype=np.uint8))
    writer.release()


def test_group_behaviors_demo_fallback_and_page_rendering() -> None:
    """Group-behavior demo fallback must be labeled and include all three behaviors."""
    assert DEMO_FALLBACK_DATA_GROUP_BEHAVIORS["is_fallback_demo"] is True
    assert "仅用于界面交互验证" in DEMO_FALLBACK_DATA_GROUP_BEHAVIORS["demo_warning"]
    behaviors = {
        c["ground_truth_behavior"] for c in DEMO_FALLBACK_DATA_GROUP_BEHAVIORS["per_clip_results"]
    }
    assert behaviors == {"Dispersal", "Group locomotion", "Social clustering"}
    assert (
        _select_demo_fallback(Path("outputs/group_behaviors_evaluation.json"))
        is DEMO_FALLBACK_DATA_GROUP_BEHAVIORS
    )

    html = render_comparison_page(
        initial_data=DEMO_FALLBACK_DATA_GROUP_BEHAVIORS,
        fallback_data=DEMO_FALLBACK_DATA_GROUP_BEHAVIORS,
    )
    assert "rulesDrawer" in html
    assert "toggleRulesDrawer()" in html
    assert "origOverlay" in html
    assert "origVideo.onerror" in html
    assert "badge-dispersal" in html
    assert "badge-clustering" in html


def test_overlay_renderer_supports_dispersal_and_social_clustering(tmp_path: Path) -> None:
    """Cropped and annotated MP4 rendering must support Dispersal and Social clustering."""
    assert "Dispersal" in ROLE_COLORS
    assert "Social clustering" in ROLE_COLORS
    assert "Group locomotion" in ROLE_COLORS

    src = tmp_path / "src.mp4"
    _write_synthetic_mp4(src, frame_count=12, fps=10.0)
    tracks = [
        {
            "frame": f,
            "detections": [
                {"track_id": 1, "box": [10.0, 10.0, 40.0, 40.0]},
                {"track_id": 2, "box": [60.0, 20.0, 90.0, 50.0]},
                {"track_id": 3, "box": [100.0, 40.0, 130.0, 70.0]},
            ],
        }
        for f in range(12)
    ]
    for beh in ("Dispersal", "Social clustering"):
        slug = beh.lower().replace(" ", "_")
        crop_out = tmp_path / f"{slug}_crop.mp4"
        orig_out = tmp_path / f"{slug}_orig.mp4"
        assert render_cropped_comparison_video(
            src,
            crop_out,
            2,
            10,
            10.0,
            tracks=tracks,
            target_mouse_ids=[1, 2],
            semantic_roles={beh: [1, 2]},
            pred_behavior=beh,
            gt_behavior=beh,
            iou=1.0,
        )
        assert crop_out.is_file() and video_frame_count(crop_out) == 8
        assert render_annotated_original_video(
            src,
            orig_out,
            10.0,
            tracks=tracks,
            target_mouse_ids=[1, 2],
            semantic_roles={beh: [1, 2]},
            behavior=beh,
            show_banner=True,
        )
        assert orig_out.is_file() and video_frame_count(orig_out) == 12


@pytest.fixture
def synthetic_group_server(tmp_path: Path) -> Generator[dict, None, None]:
    """Spin up an isolated test server with Dispersal, Group locomotion, and Social clustering."""
    eval_file = tmp_path / "group_behaviors_evaluation.json"
    clips = [
        ("disp_001", "Dispersal", [1, 2, 3]),
        ("gloco_001", "Group locomotion", [4, 5, 6]),
        ("sc_001", "Social clustering", [7, 8, 9]),
    ]
    payload = {
        "is_fallback_demo": False,
        "dataset_total_clips": 605,
        "total_clips": 3,
        "overall_accuracy": 1.0,
        "overall_mean_iou": 1.0,
        "per_clip_results": [
            {
                "clip_id": cid,
                "ground_truth_behavior": beh,
                "predicted_behavior": beh,
                "is_correct": True,
                "target_mouse_ids": mids,
                "semantic_roles": {beh: mids},
                "ground_truth_range_half_open": {"start": 0, "end": 30},
                "predicted_range_half_open": {"start": 0, "end": 30},
                "temporal_iou": 1.0,
                "iou_ge_90": True,
                "iou_ge_95": True,
                "iou_ge_05": True,
                "total_frames": 30,
                "fps": 30.0,
            }
            for cid, beh, mids in clips
        ],
    }
    eval_file.write_text(json.dumps(payload), encoding="utf-8")

    video_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42" + b"Z" * 1024
    disp_dir = tmp_path / "Dispersal"
    gloco_dir = tmp_path / "Group locomotion"
    sc_dir = tmp_path / "Social clustering"
    crops_dir = tmp_path / "crops"
    orig_dir = tmp_path / "orig"

    for cid, beh, _ in clips:
        base = {"Dispersal": disp_dir, "Group locomotion": gloco_dir, "Social clustering": sc_dir}[beh]
        (base / cid).mkdir(parents=True)
        (base / cid / "clip.mp4").write_bytes(video_bytes)
        (crops_dir / cid).mkdir(parents=True)
        (crops_dir / cid / "clip_cropped.mp4").write_bytes(video_bytes)
        (orig_dir / cid).mkdir(parents=True)
        (orig_dir / cid / "clip_annotated.mp4").write_bytes(video_bytes)

    server = create_comparison_server(
        evaluation_file=eval_file,
        dataset_dir=tmp_path,
        dispersal_dir=disp_dir,
        group_locomotion_dir=gloco_dir,
        social_clustering_dir=sc_dir,
        crops_dir=crops_dir,
        orig_annotated_dir=orig_dir,
        port=0,
        host="127.0.0.1",
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "eval_file": eval_file,
            "video_bytes": video_bytes,
            "clips": clips,
        }
    finally:
        server.shutdown()
        server.server_close()


def test_group_server_streaming_and_api_resilience(synthetic_group_server: dict) -> None:
    """Verify 206 Range streaming for all 3 behaviors plus malformed/missing API resilience."""
    base_url = synthetic_group_server["base_url"]
    video_bytes = synthetic_group_server["video_bytes"]

    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "disp_001" in html and "gloco_001" in html and "sc_001" in html

    for cid, beh, _ in synthetic_group_server["clips"]:
        enc_beh = urllib.parse.quote(beh)
        for url in (
            f"{base_url}/api/video/original/{enc_beh}/{cid}?annotated=1",
            f"{base_url}/api/video/original/{enc_beh}/{cid}?raw=1",
            f"{base_url}/api/video/cropped/{cid}",
        ):
            req = urllib.request.Request(url, headers={"Range": "bytes=0-127"})
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 206
                assert resp.read() == video_bytes[:128]

    # Malformed evaluation JSON -> HTTP 500 JSON error (never HTML)
    eval_file: Path = synthetic_group_server["eval_file"]
    eval_file.write_text("{bad json", encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/evaluation")
    assert exc_info.value.code == 500
    assert "application/json" in (exc_info.value.headers.get("Content-Type") or "")

    # Missing evaluation JSON -> labeled demo fallback
    eval_file.unlink()
    with urllib.request.urlopen(f"{base_url}/api/evaluation") as resp:
        assert resp.status == 200
        fallback = json.loads(resp.read().decode("utf-8"))
        assert fallback["is_fallback_demo"] is True


def test_live_group_behaviors_report_and_videos() -> None:
    """Regression check for generated 50-clip group behavior report and rendered videos."""
    if not DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON.is_file():
        pytest.skip(f"Group behavior report not generated: {DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON}")

    data = json.loads(DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON.read_text(encoding="utf-8"))
    assert data["total_clips"] == 50
    assert data["dispersal_clips"] == 35
    assert data["group_locomotion_clips"] == 8
    assert data["social_clustering_clips"] == 7
    assert data["overall_accuracy"] == 1.0
    assert data["overall_mean_iou"] == 1.0

    for clip in data["per_clip_results"]:
        cid = clip["clip_id"]
        crop_path = DEFAULT_CROPS_DIR / cid / "clip_cropped.mp4"
        orig_path = DEFAULT_ORIG_ANNOTATED_DIR / cid / "clip_annotated.mp4"
        assert crop_path.is_file() and crop_path.stat().st_size > 1000
        assert orig_path.is_file() and orig_path.stat().st_size > 1000

    server = create_comparison_server(
        evaluation_file=DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON,
        dataset_dir=DEFAULT_DATASET_DIR,
        dispersal_dir=DEFAULT_DISPERSAL_DIR,
        group_locomotion_dir=DEFAULT_GROUP_LOCOMOTION_DIR,
        social_clustering_dir=DEFAULT_SOCIAL_CLUSTERING_DIR,
        crops_dir=DEFAULT_CROPS_DIR,
        orig_annotated_dir=DEFAULT_ORIG_ANNOTATED_DIR,
        port=0,
        host="127.0.0.1",
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for beh in ("Dispersal", "Group locomotion", "Social clustering"):
            sample = next(
                c for c in data["per_clip_results"] if c["ground_truth_behavior"] == beh
            )
            cid = sample["clip_id"]
            enc_beh = urllib.parse.quote(beh)
            for url in (
                f"http://127.0.0.1:{port}/api/video/cropped/{cid}",
                f"http://127.0.0.1:{port}/api/video/original/{enc_beh}/{cid}?annotated=1",
                f"http://127.0.0.1:{port}/api/video/original/{enc_beh}/{cid}?raw=1",
            ):
                req = urllib.request.Request(url, headers={"Range": "bytes=0-255"})
                with urllib.request.urlopen(req) as resp:
                    assert resp.status == 206
                    assert len(resp.read()) == 256
    finally:
        server.shutdown()
        server.server_close()
