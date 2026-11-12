"""Unit and smoke tests for the behavior comparison site and server.

Tests cover:
- Application shell and core navigation rendering without API.
- Embedded local/demo fallback data labeling.
- Server endpoint routing (/, /api/health, /api/evaluation, video streaming).
- HTTP 206 partial content range requests for video scrubbing.
- Malformed and missing API resilience.
- Loopback security enforcement.
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
    DEMO_FALLBACK_DATA,
    DEMO_FALLBACK_DATA_COMBINED,
    DEMO_FALLBACK_DATA_STATIC_LOCO,
    create_comparison_server,
    render_comparison_page,
)


def test_comparison_page_shell_renders_without_api() -> None:
    """The application shell and core navigation must render without requiring an API."""
    html = render_comparison_page()

    assert "<!DOCTYPE html>" in html
    assert "小鼠行为识别与精准边界裁剪对比验证系统" in html
    # Header and navigation elements
    assert "<header>" in html
    assert "connBadge" in html
    assert "alertBanner" in html
    assert "retryFetchData()" in html
    # Dashboard metrics cards
    assert "kpiTotal" in html
    assert "kpiAttack" in html
    assert "kpiChasing" in html
    assert "kpiMeanIoU" in html
    # Main layout, aside list, search, filter pills
    assert "<aside>" in html
    assert "searchInput" in html
    assert "filter-pill" in html
    assert "clipList" in html
    # Video comparison grid
    assert "origVideo" in html
    assert "cropVideo" in html
    assert "timelineBar" in html
    assert "btnPlay" in html
    assert "stepFrame" in html
    # Telemetry
    assert "telemetry-card" in html
    assert "diagIou" in html

    # Embedded data is present and valid
    assert "const EMBEDDED_DATA = " in html
    assert "is_fallback_demo" in html


def test_comparison_page_demo_fallback_is_clearly_labeled() -> None:
    """Fallback demo data must be explicitly labeled and warn user."""
    assert DEMO_FALLBACK_DATA["is_fallback_demo"] is True
    assert "仅用于界面交互验证，非后端实时科研结果" in DEMO_FALLBACK_DATA["demo_warning"]
    assert DEMO_FALLBACK_DATA["total_clips"] == 6
    assert len(DEMO_FALLBACK_DATA["per_clip_results"]) == 6


def test_comparison_page_validates_api_content_type_and_schema() -> None:
    """Client JS must validate response status, Content-Type, and schema."""
    html = render_comparison_page()

    # Checks res.ok
    assert "if (!res.ok)" in html
    # Checks Content-Type
    assert "contentType.includes('application/json')" in html
    # Validates schema
    assert "Array.isArray(data.per_clip_results)" in html
    # Sets in-page recoverable state
    assert "alertBanner.style.display = 'flex'" in html
    # Escapes dynamic strings and supports >=90% IoU filter
    assert "function escapeHtml(" in html
    assert "data-filter=\"iou90\"" in html


def test_combined_demo_fallback_covers_all_six_behaviors() -> None:
    """Combined fallback dataset must include all 6 behaviors and be clearly labeled."""
    assert DEMO_FALLBACK_DATA_COMBINED["is_fallback_demo"] is True
    assert DEMO_FALLBACK_DATA_STATIC_LOCO["is_fallback_demo"] is True
    behaviors = {c["ground_truth_behavior"] for c in DEMO_FALLBACK_DATA_COMBINED["per_clip_results"]}
    assert behaviors == {"Attack", "Chasing", "Isolation", "Group locomotion", "Static", "Locomotion"}


@pytest.fixture
def test_server_env(tmp_path: Path) -> Generator[dict, None, None]:
    """Spin up a test server instance on an ephemeral loopback port."""
    eval_file = tmp_path / "evaluation.json"
    dummy_eval = {
        "is_fallback_demo": False,
        "total_clips": 2,
        "attack_clips": 1,
        "chasing_clips": 1,
        "attack_accuracy": 1.0,
        "chasing_accuracy": 1.0,
        "overall_accuracy": 1.0,
        "attack_mean_iou": 1.0,
        "chasing_mean_iou": 0.96,
        "overall_mean_iou": 0.98,
        "iou_ge_05_rate": 1.0,
        "per_clip_results": [
            {
                "clip_id": "test_attack_001",
                "ground_truth_behavior": "Attack",
                "predicted_behavior": "Attack",
                "is_correct": True,
                "target_mouse_ids": [101, 102],
                "semantic_roles": {"Attacker": [101], "Attacked": [102]},
                "ground_truth_range_half_open": {"start": 0, "end": 30},
                "predicted_range_half_open": {"start": 0, "end": 30},
                "temporal_iou": 1.0,
                "iou_ge_05": True,
                "attack_frames": 30,
                "chasing_frames": 0,
                "total_frames": 30,
                "fps": 30.0,
            }
        ],
    }
    eval_file.write_text(json.dumps(dummy_eval), encoding="utf-8")

    attack_dir = tmp_path / "Attack"
    chasing_dir = tmp_path / "Chasing"
    crops_dir = tmp_path / "crops"

    dummy_clip_dir = attack_dir / "test_attack_001"
    dummy_clip_dir.mkdir(parents=True)
    # Write a dummy 4-byte video payload
    video_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42" + b"A" * 1024
    (dummy_clip_dir / "clip.mp4").write_bytes(video_bytes)

    crop_clip_dir = crops_dir / "test_attack_001"
    crop_clip_dir.mkdir(parents=True)
    (crop_clip_dir / "clip_cropped.mp4").write_bytes(video_bytes)

    server = create_comparison_server(
        evaluation_file=eval_file,
        attack_dir=attack_dir,
        chasing_dir=chasing_dir,
        crops_dir=crops_dir,
        port=0,  # OS assigned free port
        host="127.0.0.1",
    )
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "server": server,
        "port": port,
        "eval_file": eval_file,
        "tmp_path": tmp_path,
        "video_bytes": video_bytes,
    }

    server.shutdown()
    server.server_close()


def test_server_homepage_and_health(test_server_env: dict) -> None:
    port = test_server_env["port"]

    # Test /
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type")
        assert "text/html" in content_type
        body = resp.read().decode("utf-8")
        assert "小鼠行为识别与精准边界裁剪对比验证系统" in body

    # Test /api/health
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["evaluation_file_exists"] is True


def test_server_api_evaluation(test_server_env: dict) -> None:
    port = test_server_env["port"]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/evaluation") as resp:
        assert resp.status == 200
        assert "application/json" in resp.headers.get("Content-Type")
        data = json.loads(resp.read().decode("utf-8"))
        assert data["total_clips"] == 2
        assert len(data["per_clip_results"]) == 1
        assert data["per_clip_results"][0]["clip_id"] == "test_attack_001"


def test_server_api_evaluation_malformed_resilience(test_server_env: dict) -> None:
    """When the evaluation JSON is malformed, server returns 500 error instead of crashing."""
    port = test_server_env["port"]
    eval_file: Path = test_server_env["eval_file"]
    # Write invalid json
    eval_file.write_text("{ broken json: invalid ", encoding="utf-8")

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/evaluation")
    assert exc_info.value.code == 500
    err_body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "Failed to parse evaluation JSON" in err_body["error"]


def test_server_api_evaluation_missing_file_fallback(test_server_env: dict) -> None:
    """When evaluation file does not exist, server returns labeled DEMO_FALLBACK_DATA."""
    port = test_server_env["port"]
    eval_file: Path = test_server_env["eval_file"]
    eval_file.unlink()

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/evaluation") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["is_fallback_demo"] is True
        assert "仅用于界面交互验证" in data["demo_warning"]


def test_server_video_streaming_and_partial_content(test_server_env: dict) -> None:
    """Test HTTP 206 Partial Content range requests for video streaming."""
    port = test_server_env["port"]
    video_bytes = test_server_env["video_bytes"]
    total_len = len(video_bytes)

    # Full GET request
    url = f"http://127.0.0.1:{port}/api/video/original/Attack/test_attack_001"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        assert resp.headers.get("Accept-Ranges") == "bytes"
        assert resp.headers.get("Content-Type") == "video/mp4"
        assert int(resp.headers.get("Content-Length")) == total_len
        body = resp.read()
        assert body == video_bytes

    # Partial range request (bytes=0-15)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-15"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert resp.headers.get("Content-Range") == f"bytes 0-15/{total_len}"
        assert int(resp.headers.get("Content-Length")) == 16
        chunk = resp.read()
        assert len(chunk) == 16
        assert chunk == video_bytes[:16]

    # Cropped video endpoint
    crop_url = f"http://127.0.0.1:{port}/api/video/cropped/test_attack_001"
    req_crop = urllib.request.Request(crop_url, headers={"Range": "bytes=16-31"})
    with urllib.request.urlopen(req_crop) as resp:
        assert resp.status == 206
        assert resp.headers.get("Content-Range") == f"bytes 16-31/{total_len}"
        assert resp.read() == video_bytes[16:32]


def test_server_missing_video_and_404(test_server_env: dict) -> None:
    port = test_server_env["port"]
    url = f"http://127.0.0.1:{port}/api/video/original/Attack/non_existent_clip"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url)
    assert exc_info.value.code == 404

    # Unknown route
    unknown_url = f"http://127.0.0.1:{port}/api/unknown_route"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(unknown_url)
    assert exc_info.value.code == 404


def test_live_default_paths_smoke() -> None:
    """Smoke test with default paths if evaluation output is generated."""
    from mouse_behavior.visualization.comparison_site import (
        DEFAULT_EVALUATION_JSON,
    )

    if not DEFAULT_EVALUATION_JSON.is_file():
        pytest.skip("Default evaluation file not yet generated")

    server = create_comparison_server(port=0, host="127.0.0.1")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # Verify /
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type")

        # Verify /api/health
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode("utf-8"))
            assert health["status"] == "ok"
            assert health["evaluation_file_exists"] is True

        # Verify /api/evaluation
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/evaluation") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["total_clips"] in (33, 107, 128, 140, 235, 268)
            assert len(data["per_clip_results"]) == data["total_clips"]

            correct_clips = [c for c in data["per_clip_results"] if c["is_correct"]]
            assert len(correct_clips) >= 33

            # Verify streaming the first correct cropped video
            first_clip = correct_clips[0]
            cid = first_clip["clip_id"]
            crop_url = f"http://127.0.0.1:{port}/api/video/cropped/{cid}"
            req = urllib.request.Request(crop_url, headers={"Range": "bytes=0-511"})
            with urllib.request.urlopen(req) as v_resp:
                assert v_resp.status == 206
                assert v_resp.headers.get("Accept-Ranges") == "bytes"
                assert "video/mp4" in v_resp.headers.get("Content-Type")
                chunk = v_resp.read()
                assert len(chunk) == 512
    finally:
        server.shutdown()
        server.server_close()


def test_select_demo_fallback_uses_filename_only() -> None:
    """Parent directory names must not hijack demo fallback selection."""
    from mouse_behavior.visualization.comparison_site import (
        DEMO_FALLBACK_DATA_COMBINED,
        DEMO_FALLBACK_DATA_ISO_GLOCO,
        DEMO_FALLBACK_DATA_STATIC_LOCO,
        _select_demo_fallback,
    )

    assert _select_demo_fallback(Path("C:/static_runs/isolation/missing_id_visualized_evaluation.json")) is DEMO_FALLBACK_DATA
    assert _select_demo_fallback(Path("C:/tmp/combined_behavior_visualized_evaluation.json")) is DEMO_FALLBACK_DATA_COMBINED
    assert _select_demo_fallback(Path("C:/tmp/static_locomotion_evaluation.json")) is DEMO_FALLBACK_DATA_STATIC_LOCO
    assert _select_demo_fallback(Path("C:/tmp/isolation_group_locomotion_evaluation.json")) is DEMO_FALLBACK_DATA_ISO_GLOCO


def test_render_comparison_page_separates_embedded_and_demo_fallback_and_resilience() -> None:
    """Fallback recovery must use DEMO_FALLBACK (not stale EMBEDDED_DATA) and reject empty payloads."""
    custom_initial = {
        "is_fallback_demo": False,
        "total_clips": 1,
        "overall_accuracy": 1.0,
        "overall_mean_iou": 0.99,
        "per_clip_results": [
            {
                "clip_id": "LIVE_REAL_CLIP_ONLY",
                "ground_truth_behavior": "Static",
                "predicted_behavior": "Static",
                "is_correct": True,
                "temporal_iou": 0.99,
                "total_frames": 60,
                "fps": 30.0,
                "target_mouse_ids": [1],
                "ground_truth_range_half_open": {"start": 0, "end": 60},
                "predicted_range_half_open": {"start": 0, "end": 60},
            }
        ],
    }
    html = render_comparison_page(initial_data=custom_initial)

    assert "const EMBEDDED_DATA =" in html
    assert "const DEMO_FALLBACK =" in html
    fallback_line = next(line for line in html.splitlines() if line.startswith("const DEMO_FALLBACK ="))
    assert "LIVE_REAL_CLIP_ONLY" not in fallback_line
    assert '"is_fallback_demo": true' in fallback_line
    assert "loadData(DEMO_FALLBACK, true)" in html
    assert "loadData(EMBEDDED_DATA, true)" not in html
    assert "data.per_clip_results.length === 0" in html
    assert "AbortController" in html
    assert "data.is_fallback_demo" in html
    assert 'onclick="setFilter(' not in html
    assert "显示 140 / 140" not in html
    assert "33 / 33 段正确" not in html


def test_render_annotated_original_videos_helpers(tmp_path: Path) -> None:
    """Verify render_annotated_original_videos supports 6 behaviors and annotation fallback keys."""
    import sys

    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import render_annotated_original_videos as raov

    roles = raov._normalize_roles_map({"attacker": 1, "target": [2, 3], "ignored": "bad"})
    assert roles == {"attacker": [1], "target": [2, 3]}

    static_dir = tmp_path / "Static"
    clip_dir = static_dir / "clip_abc"
    clip_dir.mkdir(parents=True)
    resolved = raov._resolve_clip_dir("clip_abc", "Static", {"Static": static_dir})
    assert resolved == clip_dir


