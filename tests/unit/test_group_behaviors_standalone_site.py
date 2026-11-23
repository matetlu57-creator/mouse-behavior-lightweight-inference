"""Unit, API-resilience, and standalone bundle tests for group_behaviors_site."""
from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator

import pytest

from mouse_behavior.visualization.group_behaviors_site import (
    DEFAULT_SITE_DIR,
    DEMO_FALLBACK_GROUP_DATA,
    build_standalone_group_website,
    create_group_behaviors_server,
    render_group_behaviors_page,
)


def test_standalone_page_shell_and_api_resilience() -> None:
    """Page shell renders without API, validates JSON/schema, and separates file:// bundle vs HTTP fallback."""
    assert DEMO_FALLBACK_GROUP_DATA["is_fallback_demo"] is True
    assert "仅用于界面交互验证" in DEMO_FALLBACK_GROUP_DATA["demo_warning"]

    bundle = {
        "is_fallback_demo": False,
        "dataset_total_clips": 605,
        "total_clips": 1,
        "per_clip_results": [
            {
                "clip_id": "BUNDLE_ONLY_ID",
                "ground_truth_behavior": "Dispersal",
                "predicted_behavior": "Dispersal",
                "is_correct": True,
                "target_mouse_ids": [1, 2, 3],
                "temporal_iou": 1.0,
                "total_frames": 30,
                "fps": 30.0,
            }
        ],
    }
    html = render_group_behaviors_page(bundle_data=bundle, fallback_data=DEMO_FALLBACK_GROUP_DATA)

    assert "<!DOCTYPE html>" in html
    assert "小鼠群体行为独立视频验证平台" in html
    assert "origVideo" in html and "cropVideo" in html
    assert "if (!res.ok)" in html
    assert "contentType.includes('application/json')" in html
    assert "Array.isArray(data.per_clip_results)" in html
    assert "AbortController" in html
    assert "loadDataset(DEMO_FALLBACK, true)" in html

    # Verify BUNDLE_ONLY_ID never leaks into DEMO_FALLBACK
    fallback_line = next(line for line in html.splitlines() if line.startswith("const DEMO_FALLBACK ="))
    assert "BUNDLE_ONLY_ID" not in fallback_line
    assert '"is_fallback_demo": true' in fallback_line


@pytest.fixture
def built_standalone_site(tmp_path: Path) -> Generator[dict, None, None]:
    """Build a standalone group-behaviors website bundle in tmp_path and serve it."""
    video_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42" + b"M" * 1024
    clips_meta = [
        ("disp_clip_1", "Dispersal", [1, 2, 3]),
        ("gloco_clip_1", "Group locomotion", [4, 5, 6]),
        ("sc_clip_1", "Social clustering", [7, 8, 9]),
    ]
    per_clip = []
    for cid, beh, mids in clips_meta:
        cdir = tmp_path / "src_ds" / beh / cid
        cdir.mkdir(parents=True)
        (cdir / "clip.mp4").write_bytes(video_bytes)
        ann_p = tmp_path / "src_ann" / cid / "clip_annotated.mp4"
        ann_p.parent.mkdir(parents=True)
        ann_p.write_bytes(video_bytes)
        crp_p = tmp_path / "src_crp" / cid / "clip_cropped.mp4"
        crp_p.parent.mkdir(parents=True)
        crp_p.write_bytes(video_bytes)
        per_clip.append({
            "clip_id": cid,
            "clip_dir": str(cdir),
            "ground_truth_behavior": beh,
            "predicted_behavior": beh,
            "is_correct": True,
            "target_mouse_ids": mids,
            "semantic_roles": {beh: mids},
            "ground_truth_range_half_open": {"start": 0, "end": 30},
            "predicted_range_half_open": {"start": 0, "end": 30},
            "temporal_iou": 1.0,
            "total_frames": 30,
            "fps": 30.0,
            "annotated_orig_path": str(ann_p),
            "cropped_video_path": str(crp_p),
        })

    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps({"total_clips": 3, "dataset_total_clips": 605, "per_clip_results": per_clip}),
        encoding="utf-8",
    )

    site_dir = tmp_path / "standalone_site"
    summary = build_standalone_group_website(evaluation_file=eval_json, site_dir=site_dir, copy_videos=True)
    assert summary["total_clips"] == 3
    assert summary["copied_videos"] == {"raw": 3, "annotated": 3, "cropped": 3}

    server = create_group_behaviors_server(site_dir=site_dir, port=0, host="127.0.0.1")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "site_dir": site_dir,
            "video_bytes": video_bytes,
            "clips_meta": clips_meta,
        }
    finally:
        server.shutdown()
        server.server_close()


def test_standalone_site_bundle_streaming_and_resilience(built_standalone_site: dict) -> None:
    """Verify bundled files, generated server.py, HTTP 206 video streaming, and API error handling."""
    site_dir: Path = built_standalone_site["site_dir"]
    base_url: str = built_standalone_site["base_url"]
    video_bytes: bytes = built_standalone_site["video_bytes"]

    assert (site_dir / "index.html").is_file()
    assert (site_dir / "data" / "evaluation.json").is_file()
    assert (site_dir / "server.py").is_file()
    assert (site_dir / "start_local_site.bat").is_file()

    with urllib.request.urlopen(f"{base_url}/") as r:
        assert r.status == 200
        assert "text/html" in r.headers.get("Content-Type", "")
        html = r.read().decode("utf-8")
        assert "disp_clip_1" in html

    with urllib.request.urlopen(f"{base_url}/api/health") as r:
        assert r.status == 200
        h = json.loads(r.read().decode("utf-8"))
        assert h["status"] == "ok" and h["videos_dir_exists"] is True

    with urllib.request.urlopen(f"{base_url}/api/evaluation") as r:
        assert r.status == 200
        ev = json.loads(r.read().decode("utf-8"))
        assert ev["total_clips"] == 3

    for cid, _, _ in built_standalone_site["clips_meta"]:
        for kind in ("raw", "annotated", "cropped"):
            req = urllib.request.Request(
                f"{base_url}/videos/{cid}/{kind}.mp4",
                headers={"Range": "bytes=0-127"},
            )
            with urllib.request.urlopen(req) as vr:
                assert vr.status == 206
                assert vr.read() == video_bytes[:128]

    # Verify generated standalone server.py compiles & serves Range 206 independently
    spec = importlib.util.spec_from_file_location("generated_standalone_server", site_dir / "server.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen_server = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.StandaloneSiteHandler)
    gen_port = gen_server.server_address[1]
    t = threading.Thread(target=gen_server.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{gen_port}/videos/disp_clip_1/cropped.mp4",
            headers={"Range": "bytes=0-63"},
        )
        with urllib.request.urlopen(req) as vr:
            assert vr.status == 206
            assert vr.read() == video_bytes[:64]
    finally:
        gen_server.shutdown()
        gen_server.server_close()

    # Malformed evaluation JSON -> HTTP 500 JSON error
    eval_path = site_dir / "data" / "evaluation.json"
    eval_path.write_text("{broken json", encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{base_url}/api/evaluation")
    assert exc.value.code == 500
    assert "application/json" in (exc.value.headers.get("Content-Type") or "")

    # Missing evaluation JSON -> labeled demo fallback
    eval_path.unlink()
    with urllib.request.urlopen(f"{base_url}/api/evaluation") as r:
        assert r.status == 200
        fb = json.loads(r.read().decode("utf-8"))
        assert fb["is_fallback_demo"] is True


def test_live_built_standalone_site_50_clips() -> None:
    """Verify the built standalone website under outputs/group_behaviors_website has all 50x3 videos."""
    if not (DEFAULT_SITE_DIR / "index.html").is_file():
        pytest.skip(f"Standalone website not yet built at {DEFAULT_SITE_DIR}")

    data = json.loads((DEFAULT_SITE_DIR / "data" / "evaluation.json").read_text(encoding="utf-8"))
    assert data["total_clips"] == 50
    for clip in data["per_clip_results"]:
        cid = clip["clip_id"]
        for kind in ("raw", "annotated", "cropped"):
            vp = DEFAULT_SITE_DIR / "videos" / cid / f"{kind}.mp4"
            assert vp.is_file() and vp.stat().st_size > 1000
