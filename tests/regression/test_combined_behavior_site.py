"""Regression test for unified multi-behavior evaluation report and comparison site.

Verifies:
1. Unified 6-behavior evaluation report covering 268 clips:
   - Attack: 85 clips
   - Chasing: 22 clips (accuracy >= 90%)
   - Isolation: 25 clips
   - Group locomotion: 8 clips
   - Static: 30 clips (IoU >= 90%)
   - Locomotion: 98 clips (IoU >= 90%)
2. Overall accuracy >= 90% and overall mean IoU >= 95%.
3. Server routes serving video streams for all 6 behaviors.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from typing import Generator

import pytest

from mouse_behavior.visualization.comparison_site import (
    DEFAULT_ATTACK_DIR,
    DEFAULT_CHASING_DIR,
    DEFAULT_COMBINED_EVALUATION_JSON,
    DEFAULT_CROPS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_GROUP_LOCOMOTION_DIR,
    DEFAULT_ISOLATION_DIR,
    DEFAULT_LOCOMOTION_DIR,
    DEFAULT_ORIG_ANNOTATED_DIR,
    DEFAULT_STATIC_DIR,
    create_comparison_server,
)


@pytest.fixture(scope="module")
def combined_eval_data() -> dict:
    """Load the unified combined behavior evaluation report."""
    if not DEFAULT_COMBINED_EVALUATION_JSON.is_file():
        pytest.skip(f"Combined evaluation report not found at {DEFAULT_COMBINED_EVALUATION_JSON}")
    text = DEFAULT_COMBINED_EVALUATION_JSON.read_text(encoding="utf-8")
    return json.loads(text)


def test_combined_dataset_metrics(combined_eval_data: dict) -> None:
    """Verify combined evaluation dataset satisfies all behavior targets."""
    assert combined_eval_data["total_clips"] == 268
    assert combined_eval_data["attack_clips"] == 85
    assert combined_eval_data["chasing_clips"] == 22
    assert combined_eval_data["isolation_clips"] == 25
    assert combined_eval_data["group_locomotion_clips"] == 8
    assert combined_eval_data["static_clips"] == 30
    assert combined_eval_data["locomotion_clips"] == 98

    # Chasing accuracy requirement >= 90%
    assert combined_eval_data["chasing_accuracy"] >= 0.90, (
        f"Chasing accuracy {combined_eval_data['chasing_accuracy']} < 0.90"
    )

    # Static and Locomotion requirements >= 90%
    assert combined_eval_data["static_accuracy"] >= 0.90
    assert combined_eval_data["static_mean_iou"] >= 0.90
    assert combined_eval_data["locomotion_accuracy"] >= 0.90
    assert combined_eval_data["locomotion_mean_iou"] >= 0.90

    # Overall metrics
    assert combined_eval_data["overall_accuracy"] >= 0.90
    assert combined_eval_data["overall_mean_iou"] >= 0.95
    assert combined_eval_data["iou_ge_90_rate"] >= 0.95


@pytest.fixture
def combined_server() -> Generator[str, None, None]:
    """Spin up comparison server configured with the full combined evaluation."""
    if not DEFAULT_COMBINED_EVALUATION_JSON.is_file():
        pytest.skip(f"Combined evaluation report not found at {DEFAULT_COMBINED_EVALUATION_JSON}")

    server = create_comparison_server(
        evaluation_file=DEFAULT_COMBINED_EVALUATION_JSON,
        dataset_dir=DEFAULT_DATASET_DIR,
        isolation_dir=DEFAULT_ISOLATION_DIR,
        group_locomotion_dir=DEFAULT_GROUP_LOCOMOTION_DIR,
        static_dir=DEFAULT_STATIC_DIR,
        locomotion_dir=DEFAULT_LOCOMOTION_DIR,
        attack_dir=DEFAULT_ATTACK_DIR,
        chasing_dir=DEFAULT_CHASING_DIR,
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


def test_combined_server_all_behaviors_streaming(combined_server: str, combined_eval_data: dict) -> None:
    """Verify video streaming endpoints for all 6 behaviors on the combined server."""
    base_url = combined_server

    # 1. Health check
    with urllib.request.urlopen(f"{base_url}/api/health") as resp:
        assert resp.status == 200
        health = json.loads(resp.read().decode("utf-8"))
        assert health["status"] == "ok"

    # 2. Evaluation API
    with urllib.request.urlopen(f"{base_url}/api/evaluation") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["total_clips"] == 268

    # 3. Test streaming video for each behavior category
    behaviors = ["Attack", "Chasing", "Isolation", "Group locomotion", "Static", "Locomotion"]
    for b in behaviors:
        cand = next(
            (c for c in combined_eval_data["per_clip_results"] if c.get("ground_truth_behavior") == b and c.get("is_correct")),
            None,
        )
        assert cand is not None, f"No correct clip found for behavior {b}"
        cid = cand["clip_id"]

        # Stream cropped video
        crop_url = f"{base_url}/api/video/cropped/{cid}"
        req_crop = urllib.request.Request(crop_url, headers={"Range": "bytes=0-511"})
        with urllib.request.urlopen(req_crop) as c_resp:
            assert c_resp.status == 206
            assert c_resp.headers.get("Accept-Ranges") == "bytes"
            assert len(c_resp.read()) == 512

        # Stream original annotated video and raw source video
        encoded_b = urllib.parse.quote(b)
        for query in ("?annotated=1", "?raw=1"):
            orig_url = f"{base_url}/api/video/original/{encoded_b}/{cid}{query}"
            req_orig = urllib.request.Request(orig_url, headers={"Range": "bytes=0-511"})
            with urllib.request.urlopen(req_orig) as o_resp:
                assert o_resp.status == 206
                assert o_resp.headers.get("Accept-Ranges") == "bytes"
                assert len(o_resp.read()) == 512
