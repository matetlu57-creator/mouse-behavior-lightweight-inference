"""Regression tests for Static and Locomotion segment cropping and comparison website.

Verifies:
1. Evaluation metrics on the 605-segment dataset (128 clips: 30 Static, 98 Locomotion)
   achieving >= 90% temporal IoU and >= 90% classification accuracy (in fact 100%).
2. Output visualized cropped videos and annotated original videos existence and MP4 format.
3. Server routes for Static and Locomotion behaviors (case-insensitive, range streaming).
4. Partial content streaming (HTTP 206) with byte range headers.
5. Blind inference without expected_behavior hints.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator

import pytest

from mouse_behavior.visualization.comparison_site import (
    DEFAULT_CROPS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_LOCOMOTION_DIR,
    DEFAULT_ORIG_ANNOTATED_DIR,
    DEFAULT_STATIC_DIR,
    DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON,
    create_comparison_server,
)


@pytest.fixture(scope="module")
def eval_data() -> dict:
    """Load the generated Static & Locomotion evaluation report."""
    if not DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON.is_file():
        pytest.skip(f"Evaluation report not found at {DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON}")
    text = DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON.read_text(encoding="utf-8")
    return json.loads(text)


def test_evaluation_metrics_exceed_90_percent_threshold(eval_data: dict) -> None:
    """Ensure the segment cropping module achieves >= 90% temporal IoU and high accuracy."""
    assert eval_data["total_clips"] == 128
    assert eval_data["static_clips"] == 30
    assert eval_data["locomotion_clips"] == 98

    assert eval_data["static_accuracy"] >= 0.90
    assert eval_data["locomotion_accuracy"] >= 0.90
    assert eval_data["overall_accuracy"] >= 0.90

    assert eval_data["static_mean_iou"] >= 0.90
    assert eval_data["locomotion_mean_iou"] >= 0.90
    assert eval_data["overall_mean_iou"] >= 0.95
    assert eval_data["iou_ge_90_rate"] >= 0.95

    per_clip = eval_data.get("per_clip_results", [])
    assert len(per_clip) == 128

    for clip in per_clip:
        cid = clip["clip_id"]
        assert clip["is_correct"] is True, f"Clip {cid} behavior classification failed"
        assert clip["temporal_iou"] > 0.0, f"Clip {cid} produced zero temporal overlap"
        assert clip["iou_ge_90"] is (clip["temporal_iou"] >= 0.90)


def test_visualized_and_annotated_videos_rendered(eval_data: dict) -> None:
    """Verify that all cropped and annotated MP4 video files exist on disk with valid headers."""
    for clip in eval_data["per_clip_results"]:
        cid = clip["clip_id"]

        crop_path = DEFAULT_CROPS_DIR / cid / "clip_cropped.mp4"
        assert crop_path.is_file(), f"Cropped video missing: {crop_path}"
        assert crop_path.stat().st_size > 1000, f"Cropped video too small: {crop_path}"

        # Verify MP4 container header (ftyp box)
        with crop_path.open("rb") as f:
            header = f.read(12)
            assert b"ftyp" in header, f"Invalid MP4 container: {crop_path}"

        orig_path = DEFAULT_ORIG_ANNOTATED_DIR / cid / "clip_annotated.mp4"
        assert orig_path.is_file(), f"Annotated original video missing: {orig_path}"
        assert orig_path.stat().st_size > 1000, f"Annotated video too small: {orig_path}"
        with orig_path.open("rb") as f:
            header = f.read(12)
            assert b"ftyp" in header, f"Invalid MP4 container: {orig_path}"


@pytest.fixture
def static_loco_server() -> Generator[str, None, None]:
    """Spin up an ephemeral comparison server configured with the Static & Locomotion evaluation."""
    if not DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON.is_file():
        pytest.skip(f"Evaluation report not found at {DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON}")

    server = create_comparison_server(
        evaluation_file=DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON,
        dataset_dir=DEFAULT_DATASET_DIR,
        static_dir=DEFAULT_STATIC_DIR,
        locomotion_dir=DEFAULT_LOCOMOTION_DIR,
        crops_dir=DEFAULT_CROPS_DIR,
        orig_annotated_dir=DEFAULT_ORIG_ANNOTATED_DIR,
        port=0,
        host="127.0.0.1",
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


def test_comparison_server_routes_and_streaming(static_loco_server: str, eval_data: dict) -> None:
    """Verify HTTP endpoints including behavior routes and range requests."""
    base_url = static_loco_server

    # 1. Health check
    with urllib.request.urlopen(f"{base_url}/api/health") as resp:
        assert resp.status == 200
        health = json.loads(resp.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["evaluation_file_exists"] is True

    # 2. Evaluation API
    with urllib.request.urlopen(f"{base_url}/api/evaluation") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["total_clips"] == 128

    # 3. Stream Static cropped and original video
    static_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Static")
    static_id = static_clip["clip_id"]

    crop_url = f"{base_url}/api/video/cropped/{static_id}"
    req = urllib.request.Request(crop_url, headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Range", "").startswith("bytes 0-1023/")
        assert len(resp.read()) == 1024

    orig_url = f"{base_url}/api/video/original/Static/{static_id}?annotated=1"
    req = urllib.request.Request(orig_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512

    # 4. Stream Locomotion cropped and original video
    loco_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Locomotion")
    loco_id = loco_clip["clip_id"]

    loco_crop_url = f"{base_url}/api/video/cropped/{loco_id}"
    req = urllib.request.Request(loco_crop_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512

    loco_orig_url = f"{base_url}/api/video/original/Locomotion/{loco_id}?annotated=1"
    req = urllib.request.Request(loco_orig_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512


def test_comparison_server_404_resilience(static_loco_server: str) -> None:
    """Verify non-existent routes return clean 404 JSON."""
    base_url = static_loco_server

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/video/cropped/nonexistent_clip_id")
    assert exc_info.value.code == 404

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/video/original/Static/nonexistent_clip_id")
    assert exc_info.value.code == 404


def test_evaluate_single_clip_blind_inference() -> None:
    """Verify evaluate_single_clip performs data-driven inference without expected_behavior hint."""
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from evaluate_static_locomotion import evaluate_single_clip

    static_clip_dir = DEFAULT_STATIC_DIR / "3837d36fc953a94770b41ae9"
    if not static_clip_dir.is_dir():
        pytest.skip(f"Clip not found: {static_clip_dir}")

    # Pass expected_behavior=None to test blind inference
    rec_static = evaluate_single_clip(static_clip_dir, expected_behavior=None, visualize=False)
    assert rec_static["predicted_behavior"] == "Static"
    assert rec_static["ground_truth_behavior"] == "Static"
    assert rec_static["is_correct"] is True
    assert rec_static["temporal_iou"] >= 0.90

    loco_clip_dir = DEFAULT_LOCOMOTION_DIR / "007a717ba81cb573958f905c"
    if not loco_clip_dir.is_dir():
        pytest.skip(f"Clip not found: {loco_clip_dir}")

    rec_loco = evaluate_single_clip(loco_clip_dir, expected_behavior=None, visualize=False)
    assert rec_loco["predicted_behavior"] == "Locomotion"
    assert rec_loco["ground_truth_behavior"] == "Locomotion"
    assert rec_loco["is_correct"] is True
    assert rec_loco["temporal_iou"] >= 0.90


def test_comparison_server_case_insensitive_and_preloaded_html(static_loco_server: str, eval_data: dict) -> None:
    """Verify case-insensitive video routes and preloaded evaluation HTML in index.html."""
    base_url = static_loco_server

    # 1. Index HTML preloaded with evaluation data
    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "小鼠行为识别与精准边界裁剪对比验证系统" in html
        assert "3837d36fc953a94770b41ae9" in html

    # 2. Case-insensitive lowercase original video routes
    static_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Static")
    static_id = static_clip["clip_id"]
    lower_static_url = f"{base_url}/api/video/original/static/{static_id}?raw=1"
    req = urllib.request.Request(lower_static_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512

    loco_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Locomotion")
    loco_id = loco_clip["clip_id"]
    lower_loco_url = f"{base_url}/api/video/original/locomotion/{loco_id}?raw=1"
    req = urllib.request.Request(lower_loco_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512
