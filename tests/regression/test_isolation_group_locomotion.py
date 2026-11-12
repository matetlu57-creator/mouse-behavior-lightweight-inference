"""Regression tests for Isolation and Group locomotion segment cropping and comparison website.

Verifies:
1. Evaluation metrics on the 605-segment dataset (33 clips: 25 Isolation, 8 Group locomotion)
   achieving >= 95% temporal IoU and 100% classification accuracy.
2. Output visualized cropped videos and annotated original videos existence and MP4 format.
3. Server routes for Isolation and Group locomotion (including URL-encoded paths with spaces).
4. Partial content streaming (HTTP 206) with byte range headers.
5. Website resilience against missing or malformed data and 404s.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Generator

import pytest

from mouse_behavior.visualization.comparison_site import (
    DEFAULT_CROPS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_GROUP_LOCOMOTION_DIR,
    DEFAULT_ISOLATION_DIR,
    DEFAULT_ISO_GLOCO_EVALUATION_JSON,
    DEFAULT_ORIG_ANNOTATED_DIR,
    create_comparison_server,
)


@pytest.fixture(scope="module")
def eval_data() -> dict:
    """Load the generated Isolation & Group locomotion evaluation report."""
    if not DEFAULT_ISO_GLOCO_EVALUATION_JSON.is_file():
        pytest.skip(f"Evaluation report not found at {DEFAULT_ISO_GLOCO_EVALUATION_JSON}")
    text = DEFAULT_ISO_GLOCO_EVALUATION_JSON.read_text(encoding="utf-8")
    return json.loads(text)


def test_evaluation_metrics_exceed_95_percent_threshold(eval_data: dict) -> None:
    """Ensure the segment cropping module achieves >= 95% temporal IoU and high accuracy."""
    assert eval_data["total_clips"] == 33
    assert eval_data["isolation_clips"] == 25
    assert eval_data["group_locomotion_clips"] == 8

    assert eval_data["isolation_accuracy"] >= 0.95
    assert eval_data["group_locomotion_accuracy"] >= 0.95
    assert eval_data["overall_accuracy"] >= 0.95

    assert eval_data["isolation_mean_iou"] >= 0.95
    assert eval_data["group_locomotion_mean_iou"] >= 0.95
    assert eval_data["overall_mean_iou"] >= 0.95
    assert eval_data["iou_ge_95_rate"] >= 0.95

    per_clip = eval_data.get("per_clip_results", [])
    assert len(per_clip) == 33

    for clip in per_clip:
        cid = clip["clip_id"]
        assert clip["is_correct"] is True, f"Clip {cid} behavior classification failed"
        assert clip["temporal_iou"] >= 0.95, f"Clip {cid} IoU {clip['temporal_iou']} < 0.95"
        assert clip["iou_ge_95"] is True


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
def iso_gloco_server() -> Generator[str, None, None]:
    """Spin up an ephemeral comparison server configured with the new dataset evaluation."""
    if not DEFAULT_ISO_GLOCO_EVALUATION_JSON.is_file():
        pytest.skip(f"Evaluation report not found at {DEFAULT_ISO_GLOCO_EVALUATION_JSON}")

    server = create_comparison_server(
        evaluation_file=DEFAULT_ISO_GLOCO_EVALUATION_JSON,
        dataset_dir=DEFAULT_DATASET_DIR,
        isolation_dir=DEFAULT_ISOLATION_DIR,
        group_locomotion_dir=DEFAULT_GROUP_LOCOMOTION_DIR,
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


def test_comparison_server_routes_and_streaming(iso_gloco_server: str, eval_data: dict) -> None:
    """Verify HTTP endpoints including URL-encoded behavior routes and range requests."""
    base_url = iso_gloco_server

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
        assert data["total_clips"] == 33

    # 3. Stream Isolation cropped and original video
    iso_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Isolation")
    iso_id = iso_clip["clip_id"]

    crop_url = f"{base_url}/api/video/cropped/{iso_id}"
    req = urllib.request.Request(crop_url, headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Range", "").startswith("bytes 0-1023/")
        assert len(resp.read()) == 1024

    orig_url = f"{base_url}/api/video/original/Isolation/{iso_id}?annotated=1"
    req = urllib.request.Request(orig_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512

    # 4. Stream Group locomotion with space in URL
    gloco_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Group locomotion")
    gloco_id = gloco_clip["clip_id"]

    gloco_orig_url = f"{base_url}/api/video/original/Group%20locomotion/{gloco_id}?annotated=1"
    req = urllib.request.Request(gloco_orig_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512


def test_comparison_server_404_resilience(iso_gloco_server: str) -> None:
    """Verify non-existent routes return clean 404 JSON."""
    base_url = iso_gloco_server

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/video/cropped/nonexistent_clip_id")
    assert exc_info.value.code == 404

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/api/video/original/Isolation/nonexistent_clip_id")
    assert exc_info.value.code == 404


def test_evaluate_single_clip_blind_inference() -> None:
    """Verify evaluate_single_clip performs data-driven inference without expected_behavior hint."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from evaluate_isolation_group_locomotion import evaluate_single_clip

    iso_clip_dir = DEFAULT_ISOLATION_DIR / "0580160ac4fa7d99b838c7ce"
    if not iso_clip_dir.is_dir():
        pytest.skip(f"Clip not found: {iso_clip_dir}")

    # Pass expected_behavior=None to test blind inference
    rec_iso = evaluate_single_clip(iso_clip_dir, expected_behavior=None, visualize=False)
    assert rec_iso["predicted_behavior"] == "Isolation"
    assert rec_iso["ground_truth_behavior"] == "Isolation"
    assert rec_iso["is_correct"] is True
    assert rec_iso["temporal_iou"] >= 0.95

    gloco_clip_dir = DEFAULT_GROUP_LOCOMOTION_DIR / "012e147d845dba5f6acf6a01"
    if not gloco_clip_dir.is_dir():
        pytest.skip(f"Clip not found: {gloco_clip_dir}")

    rec_gloco = evaluate_single_clip(gloco_clip_dir, expected_behavior=None, visualize=False)
    assert rec_gloco["predicted_behavior"] == "Group locomotion"
    assert rec_gloco["ground_truth_behavior"] == "Group locomotion"
    assert rec_gloco["is_correct"] is True
    assert rec_gloco["temporal_iou"] >= 0.95


def test_comparison_server_case_insensitive_and_preloaded_html(iso_gloco_server: str, eval_data: dict) -> None:
    """Verify case-insensitive video routes and preloaded evaluation HTML in index.html."""
    base_url = iso_gloco_server

    # 1. Index HTML preloaded with evaluation data
    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "Isolation and Group Locomotion Boundary Cropping Evaluation" in html
        assert "0580160ac4fa7d99b838c7ce" in html

    # 2. Case-insensitive lowercase original video routes
    iso_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Isolation")
    iso_id = iso_clip["clip_id"]
    lower_iso_url = f"{base_url}/api/video/original/isolation/{iso_id}?raw=1"
    req = urllib.request.Request(lower_iso_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512

    gloco_clip = next(c for c in eval_data["per_clip_results"] if c["ground_truth_behavior"] == "Group locomotion")
    gloco_id = gloco_clip["clip_id"]
    lower_gloco_url = f"{base_url}/api/video/original/group_locomotion/{gloco_id}?raw=1"
    req = urllib.request.Request(lower_gloco_url, headers={"Range": "bytes=0-511"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert len(resp.read()) == 512

