"""Standalone resilient website builder and local server for long-video social-behavior review."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from mouse_behavior.evaluation.social_cropping_validation import (
    SOCIAL_BEHAVIOR_KEYS,
    SOCIAL_BEHAVIOR_LABELS,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOCIAL_REVIEW_SITE_DIR = _REPO_ROOT / "outputs" / "long_video_social_validation_20261123" / "review_site"
DEFAULT_SOCIAL_REVIEW_PORT = 8797
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._\-\u4e00-\u9fff]+$")
_REVIEWS_LOCK = threading.Lock()
_MAX_REVIEW_BODY_BYTES = 256 * 1024
_STATIC_MIME_BY_SUFFIX: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".json": "application/json; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _is_safe_videos_req_path(req_path: str) -> bool:
    """Validate relative URL path syntax for ``videos/...`` without normalizing away traversal segments."""
    if not req_path or "\\" in req_path or "\x00" in req_path:
        return False
    parts = req_path.split("/")
    if len(parts) < 2 or parts[0] != "videos":
        return False
    return all(
        bool(seg) and seg not in {".", ".."} and bool(_SAFE_SEGMENT_RE.fullmatch(seg))
        for seg in parts[1:]
    )


def _lookup_safe_video_file(site_dir: Path, req_path: str) -> tuple[Path, str] | None:
    """Resolve an existing file under ``site_dir/videos`` using directory entry matching (zero raw path joins)."""
    if not _is_safe_videos_req_path(req_path):
        return None
    parts = req_path.split("/")[1:]
    current = (site_dir / "videos").resolve()
    if not current.is_dir():
        return None
    for idx, seg in enumerate(parts):
        try:
            entries = list(current.iterdir())
        except OSError:
            return None
        matched = next((entry for entry in entries if entry.name == seg), None)
        if matched is None or matched.is_symlink():
            return None
        if idx < len(parts) - 1:
            if not matched.is_dir():
                return None
            current = matched
        else:
            if not matched.is_file():
                return None
            resolved_root = (site_dir / "videos").resolve()
            resolved_file = matched.resolve()
            if not resolved_file.is_relative_to(resolved_root):
                return None
            mime = _STATIC_MIME_BY_SUFFIX.get(matched.suffix.lower(), "application/octet-stream")
            return resolved_file, mime
    return None

DEMO_FALLBACK_SOCIAL_BUNDLE: dict[str, Any] = {
    "schema_version": 1,
    "is_fallback_demo": True,
    "dataset_status": "demo_fallback",
    "generated_at": "2026-11-23T00:00:00Z",
    "behavior_order": list(SOCIAL_BEHAVIOR_KEYS),
    "behavior_labels": SOCIAL_BEHAVIOR_LABELS,
    "notice": "当前显示为内置社交演示数据（非实测科研结果），用于验证界面加载与API容错。请运行社交裁剪验证脚本生成实测数据包。",
    "videos": [
        {
            "video_key": "demo_social_video",
            "title": "演示长视频（非实测）",
            "duration_s": 600.0,
            "fps": 30.0,
            "frame_count": 18000,
            "inference_source": "demo",
            "aggregate": {
                "total_raw_social_candidates": 6,
                "total_post_crop_social_segments": 5,
                "total_rejected_by_fine_crop": 1,
                "total_boundary_adjusted": 3,
                "duration_pass_rate": 1.0,
                "endpoint_observed_rate": 1.0,
                "full_crop_precision_rate": 1.0,
                "social_shared_mouse_overlap_count": 0,
            },
            "by_behavior": {
                "approach": {
                    "behavior": "approach",
                    "label": SOCIAL_BEHAVIOR_LABELS["approach"],
                    "required_duration_s": 1.0,
                    "duration_comparator": ">=",
                    "raw_candidate_count": 3,
                    "post_crop_count": 2,
                    "changed_bounds_count": 1,
                    "rejected_by_fine_crop_count": 1,
                    "min_post_duration_s": 1.15,
                    "mean_raw_duration_s": 0.85,
                    "mean_post_duration_s": 1.45,
                    "duration_pass_rate": 1.0,
                    "endpoint_observed_rate": 1.0,
                    "full_crop_precision_rate": 1.0,
                },
                "together": {
                    "behavior": "together",
                    "label": SOCIAL_BEHAVIOR_LABELS["together"],
                    "required_duration_s": 1.0,
                    "duration_comparator": ">=",
                    "raw_candidate_count": 3,
                    "post_crop_count": 3,
                    "changed_bounds_count": 2,
                    "rejected_by_fine_crop_count": 0,
                    "min_post_duration_s": 1.15,
                    "mean_raw_duration_s": 1.8,
                    "mean_post_duration_s": 2.2,
                    "duration_pass_rate": 1.0,
                    "endpoint_observed_rate": 1.0,
                    "full_crop_precision_rate": 1.0,
                },
            },
        }
    ],
    "clips": [],
}


def validate_social_review_bundle_schema(payload: Any) -> bool:
    """Validate summary/clips JSON schema returned by API or static bundle file."""
    if not isinstance(payload, Mapping):
        return False
    if not isinstance(payload.get("videos"), list):
        return False
    if not isinstance(payload.get("clips"), list):
        return False
    for vid in payload["videos"]:
        if not isinstance(vid, Mapping) or not isinstance(vid.get("video_key"), str):
            return False
    return True


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_long_video_social_review_site(
    run_dirs: Sequence[Path],
    site_dir: Path = DEFAULT_SOCIAL_REVIEW_SITE_DIR,
    *,
    copy_videos: bool = True,
    port: int = DEFAULT_SOCIAL_REVIEW_PORT,
    clean_videos: bool = False,
) -> dict[str, Any]:
    """Assemble multi-video social cropping validation and manual review site bundle."""
    resolved_site = Path(site_dir).resolve()
    data_dir = resolved_site / "data"
    videos_dir = resolved_site / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    if clean_videos and videos_dir.exists():
        shutil.rmtree(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    videos_summary: list[dict[str, Any]] = []
    all_clips: list[dict[str, Any]] = []
    copied_video_count = 0

    for run_dir in run_dirs:
        run_path = Path(run_dir).resolve()
        summary_path = run_path / "validation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing validation_summary.json in {run_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        video_key = str(summary["video_key"])
        if not _SAFE_SEGMENT_RE.fullmatch(video_key):
            raise ValueError(f"unsafe video_key: {video_key}")

        candidates_path = run_path / "review_candidates.json"
        candidates = (
            json.loads(candidates_path.read_text(encoding="utf-8"))
            if candidates_path.is_file()
            else []
        )
        videos_summary.append(summary)

        for item in candidates:
            event_id = str(item["event_id"])
            canonical = str(item["canonical_behavior"])
            raw_pkg = Path(item["raw_package_dir"])
            overlay_pkg = Path(item["overlay_package_dir"])

            rel_raw_dir = f"videos/{quote(video_key)}/{quote(canonical)}/{quote(event_id)}/raw"
            rel_overlay_dir = f"videos/{quote(video_key)}/{quote(canonical)}/{quote(event_id)}/overlay"
            dst_raw = videos_dir / video_key / canonical / event_id / "raw"
            dst_overlay = videos_dir / video_key / canonical / event_id / "overlay"

            raw_ok = (dst_raw / "clip.mp4").is_file()
            overlay_ok = (dst_overlay / "clip.mp4").is_file()
            if copy_videos:
                for fname in ("clip.mp4", "annotation.json", "metadata.json", "tracks.json"):
                    if _copy_if_exists(raw_pkg / fname, dst_raw / fname) and fname == "clip.mp4":
                        copied_video_count += 1
                    if _copy_if_exists(overlay_pkg / fname, dst_overlay / fname) and fname == "clip.mp4":
                        copied_video_count += 1
                raw_ok = (dst_raw / "clip.mp4").is_file()
                overlay_ok = (dst_overlay / "clip.mp4").is_file()

            all_clips.append(
                {
                    **item,
                    "clip_uid": f"{video_key}::{event_id}",
                    "video_key": video_key,
                    "video_title": summary["title"],
                    "behavior_display": str(item["behavior"]),
                    "behavior_label": SOCIAL_BEHAVIOR_LABELS.get(canonical, canonical),
                    "raw_url": f"{rel_raw_dir}/clip.mp4" if raw_ok or not copy_videos else None,
                    "overlay_url": f"{rel_overlay_dir}/clip.mp4" if overlay_ok or not copy_videos else None,
                    "json_base_url": rel_raw_dir,
                }
            )

    bundle = {
        "schema_version": 1,
        "is_fallback_demo": False,
        "dataset_status": "verified_long_video_social_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "behavior_order": list(SOCIAL_BEHAVIOR_KEYS),
        "behavior_labels": SOCIAL_BEHAVIOR_LABELS,
        "videos": videos_summary,
        "clips": all_clips,
    }
    (data_dir / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    reviews_file = data_dir / "human_reviews.json"
    if not reviews_file.exists():
        init_reviews = {
            "schema": "long-video-social-review-v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "decisions": {},
        }
        reviews_file.write_text(json.dumps(init_reviews, ensure_ascii=False, indent=2), encoding="utf-8")

    html = render_long_video_social_review_page(embedded_bundle=bundle)
    (resolved_site / "index.html").write_text(html, encoding="utf-8")

    bat = (
        "@echo off\r\n"
        'cd /d "%~dp0"\r\n'
        f'start "" "http://127.0.0.1:{port}/"\r\n'
        f'"D:\\Anaconda3\\python.exe" "{_REPO_ROOT / "scripts" / "serve_long_video_social_review_site.py"}" '
        f'--site-dir "{resolved_site}" --port {port}\r\n'
        "pause\r\n"
    )
    (resolved_site / "start_local_site.bat").write_text(bat, encoding="utf-8")
    return {
        "site_dir": str(resolved_site),
        "video_count": len(videos_summary),
        "total_review_clips": len(all_clips),
        "copied_videos": copied_video_count,
    }


def render_long_video_social_review_page(
    *,
    embedded_bundle: Mapping[str, Any] | None = None,
) -> str:
    """Render self-contained resilient HTML/CSS/JS review UI for the 6 social behaviors."""
    fallback_payload = embedded_bundle if embedded_bundle is not None else DEMO_FALLBACK_SOCIAL_BUNDLE
    payload_json = json.dumps(fallback_payload, ensure_ascii=False).replace("<", "\\u003c")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>长视频社交行为裁剪验证与人工审核台</title>
<style>
:root {{
  --bg: #0f172a; --panel: #162033; --panel-alt: #1e293b; --border: #2c3b52;
  --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --accent-soft: rgba(56,189,248,0.16);
  --good: #22c55e; --warn: #f59e0b; --bad: #ef4444;
}}
* {{ box-sizing: border-box; }}
[hidden] {{ display: none !important; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.55 "Microsoft YaHei", "Segoe UI", system-ui, sans-serif; }}
header {{ background: linear-gradient(135deg, #0b1f3a 0%, #103554 100%); border-bottom: 1px solid var(--border); padding: 18px 26px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px; }}
header h1 {{ margin: 0; font-size: 21px; color: #f8fafc; }}
header p {{ margin: 4px 0 0; color: #bae6fd; font-size: 13px; }}
.status-pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); background: rgba(15,23,42,0.7); }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }}
.dot.ok {{ background: var(--good); }}
.dot.err {{ background: var(--bad); }}
.banner {{ margin: 12px 24px 0; padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(245,158,11,0.45); background: rgba(245,158,11,0.12); color: #fde68a; display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
.banner.error {{ border-color: rgba(239,68,68,0.5); background: rgba(239,68,68,0.14); color: #fecaca; }}
.wrap {{ padding: 18px 24px 32px; display: flex; flex-direction: column; gap: 18px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.kpi .label {{ font-size: 12px; color: var(--muted); }}
.kpi .value {{ font-size: 24px; font-weight: 700; margin: 4px 0; color: #f8fafc; }}
.kpi .sub {{ font-size: 12px; color: #cbd5e1; }}
.section-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
.section-card h2 {{ margin: 0 0 10px; font-size: 16px; color: #f8fafc; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: center; white-space: nowrap; }}
th {{ color: #94a3b8; font-weight: 600; background: rgba(15,23,42,0.45); }}
td.left, th.left {{ text-align: left; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
.tag.good {{ background: rgba(34,197,94,0.18); color: #86efac; }}
.tag.warn {{ background: rgba(245,158,11,0.18); color: #fcd34d; }}
.tag.bad {{ background: rgba(239,68,68,0.18); color: #fca5a5; }}
.tag.neutral {{ background: rgba(148,163,184,0.18); color: #cbd5e1; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }}
button, select {{ cursor: pointer; background: var(--panel-alt); color: var(--text); border: 1px solid var(--border); border-radius: 7px; padding: 7px 12px; font: inherit; }}
button:hover {{ border-color: var(--accent); }}
button.active {{ background: var(--accent); border-color: var(--accent); color: #082f49; font-weight: 700; }}
button.pass.active {{ background: var(--good); border-color: var(--good); color: #052e16; }}
button.fail.active {{ background: var(--bad); border-color: var(--bad); color: #450a0a; }}
.workspace {{ display: grid; grid-template-columns: 330px 1fr; gap: 16px; }}
.clip-list {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; max-height: 82vh; overflow-y: auto; }}
.clip-item {{ display: block; width: 100%; text-align: left; margin-bottom: 6px; padding: 9px 10px; border-radius: 8px; background: var(--panel-alt); border: 1px solid transparent; }}
.clip-item.selected {{ border-color: var(--accent); background: var(--accent-soft); }}
.clip-item .row-top {{ display: flex; justify-content: space-between; font-weight: 600; font-size: 13px; }}
.clip-item .row-sub {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
.reviewer {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }}
video {{ width: 100%; max-height: 58vh; border-radius: 8px; background: #020617; border: 1px solid var(--border); }}
.controls-bar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0; }}
.audit-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; margin: 12px 0; }}
.audit-box {{ background: var(--panel-alt); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 12px; }}
.audit-box strong {{ display: block; color: #7dd3fc; margin-bottom: 4px; }}
.review-box {{ background: rgba(15,23,42,0.65); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-top: 12px; }}
.reason-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
.chip {{ font-size: 12px; padding: 4px 10px; }}
textarea {{ width: 100%; min-height: 58px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px; font: inherit; }}
.empty-state {{ padding: 28px; text-align: center; background: var(--panel-alt); border-radius: 8px; color: var(--muted); }}
@media (max-width: 960px) {{ .workspace {{ grid-template-columns: 1fr; }} .clip-list {{ max-height: 280px; }} }}
</style>
</head>
<body>
<header>
  <div>
    <h1>长视频社交行为裁剪验证与人工审核台</h1>
    <p>Video Project 22 (10分钟) 与 大旷场1 (60分钟) · 全部六类社交行为 (Approach / Attack / Chasing / Snout-head contact / Snout-rear contact / Together) · 严格最低时长与完整性门控</p>
  </div>
  <div class="status-pill" id="api-status">
    <span class="dot" id="api-dot"></span>
    <span id="api-status-text">正在同步本地服务与审核数据…</span>
  </div>
</header>

<div id="mode-banner" class="banner" hidden>
  <span id="banner-text"></span>
  <button id="retry-api" type="button">重新连接后端 API</button>
</div>

<div class="wrap">
  <section class="kpi-grid" id="kpi-grid"></section>

  <section class="section-card">
    <h2>长视频社交行为裁剪验证汇总（严格最低时长 + 行为片段完整性）</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th class="left">视频来源</th>
            <th class="left">社交行为</th>
            <th>最低时长阈值</th>
            <th>裁剪前候选</th>
            <th>裁剪后段数</th>
            <th>边界调整 (延展/修剪)</th>
            <th>严格过滤数</th>
            <th>最小时长 (后)</th>
            <th>均时长 (前→后)</th>
            <th>最低时长达标率</th>
            <th>端点双鼠可见率</th>
            <th>综合完整度</th>
            <th>人工审核</th>
          </tr>
        </thead>
        <tbody id="summary-tbody"></tbody>
      </table>
    </div>
  </section>

  <section class="section-card">
    <div class="filters">
      <strong>筛选：</strong>
      <select id="filter-video"><option value="all">全部长视频</option></select>
      <select id="filter-behavior"><option value="all">全部社交行为</option></select>
      <select id="filter-review">
        <option value="all">全部审核状态</option>
        <option value="pending">仅待审核</option>
        <option value="pass">已通过</option>
        <option value="fail">不通过</option>
      </select>
      <select id="filter-boundary">
        <option value="all">全部边界类型</option>
        <option value="changed">仅边界调整/延展片段</option>
        <option value="unchanged">边界未变片段</option>
      </select>
      <button id="btn-export-json" type="button">导出审核 JSON</button>
      <button id="btn-export-csv" type="button">导出审核 CSV</button>
      <label style="font-size:12px;color:#94a3b8;cursor:pointer">
        导入审核备份 <input id="input-import-json" type="file" accept=".json,application/json" style="display:none">
      </label>
    </div>

    <div class="workspace">
      <aside class="clip-list">
        <div id="list-header" style="font-weight:700;margin-bottom:8px;">待审片段列表 (0)</div>
        <div id="clip-list-items"></div>
      </aside>

      <section class="reviewer">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
          <h2 id="clip-title" style="margin:0;">请选择左侧片段开始审核</h2>
          <span id="clip-counter" class="tag neutral">0 / 0</span>
        </div>

        <div class="controls-bar">
          <button id="btn-variant-overlay" type="button" class="active">目标高亮视频 (Green Visual ID)</button>
          <button id="btn-variant-raw" type="button">原始裁剪片段</button>
          <button id="btn-prev" type="button">← 上一段</button>
          <button id="btn-next" type="button">下一段 →</button>
          <span id="video-error" style="color:#fca5a5;font-size:12px;"></span>
        </div>

        <video id="video-player" controls preload="metadata"></video>
        <div id="video-empty" class="empty-state" hidden>当前筛选条件下暂无片段。请切换视频、行为或审核状态筛选条件。</div>

        <div class="audit-grid" id="audit-grid"></div>

        <div class="review-box" id="review-box">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
            <strong>人工边界与完整性审核</strong>
            <label style="font-size:12px;color:#cbd5e1;"><input type="checkbox" id="chk-auto-next" checked> 标记后自动跳到下一段</label>
          </div>
          <div class="controls-bar">
            <button id="btn-pass" type="button" class="pass">✓ 裁剪完整且标签正确 (通过)</button>
            <button id="btn-fail" type="button" class="fail">✕ 存在问题 (不通过)</button>
            <button id="btn-pending" type="button">↺ 重置待审核</button>
            <span id="review-save-status" style="font-size:12px;color:#94a3b8;"></span>
          </div>
          <div class="reason-chips" id="reason-chips"></div>
          <textarea id="review-note" placeholder="可填写边界偏差或完整度反馈（例如：起始未完整包含接近/追逐、尾段提前截断、不足最低时长）"></textarea>
        </div>

        <div id="clip-links" style="margin-top:10px;font-size:12px;color:#94a3b8;"></div>
      </section>
    </div>
  </section>
</div>

<script id="embedded-bundle" type="application/json">{payload_json}</script>
<script>
(() => {{
  const EMBEDDED = JSON.parse(document.getElementById('embedded-bundle').textContent);
  const STORAGE_KEY = 'long-video-social-review-v1';
  const REASON_TAGS = [
    '起始未包含完整行为',
    '结束提前截断/未收尾',
    '包含过多无关静止/分离帧',
    '目标ID漂移或交换错误',
    '行为类别判断错误',
    '低于最低时长要求'
  ];

  let bundle = EMBEDDED;
  let apiConnected = false;
  let decisions = {{}};
  let filteredClips = [];
  let currentIdx = 0;
  let variant = 'overlay';

  const el = id => document.getElementById(id);

  function loadLocalDecisions() {{
    try {{
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {{
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed.decisions === 'object') decisions = parsed.decisions;
      }}
    }} catch (_) {{}}
  }}

  function saveLocalDecisions() {{
    try {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify({{ updated_at: new Date().toISOString(), decisions }}));
    }} catch (_) {{}}
  }}

  async function fetchJsonStrict(url, timeoutMs = 4500) {{
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {{
      const resp = await fetch(url, {{ signal: ctrl.signal, cache: 'no-store' }});
      const ct = (resp.headers.get('content-type') || '').toLowerCase();
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      if (!ct.includes('application/json')) throw new Error('非 JSON 响应: ' + ct);
      return await resp.json();
    }} finally {{
      clearTimeout(timer);
    }}
  }}

  function validateBundle(data) {{
    return data && Array.isArray(data.videos) && Array.isArray(data.clips);
  }}

  async function syncWithBackend() {{
    el('api-dot').className = 'dot';
    el('api-status-text').textContent = '正在检查后端 API…';
    try {{
      const data = await fetchJsonStrict('/api/bundle');
      if (!validateBundle(data)) throw new Error('返回数据结构不符合契约');
      bundle = data;
      try {{
        const rev = await fetchJsonStrict('/api/reviews');
        if (rev && typeof rev.decisions === 'object') {{
          decisions = {{ ...decisions, ...rev.decisions }};
          saveLocalDecisions();
        }}
      }} catch (_) {{}}
      apiConnected = true;
      el('api-dot').className = 'dot ok';
      el('api-status-text').textContent = '后端服务已连接 · 审核可持久写入服务器';
      if (bundle.is_fallback_demo) {{
        el('mode-banner').hidden = false;
        el('mode-banner').className = 'banner';
        el('banner-text').textContent = bundle.notice || '当前为内置演示数据（非实测科研结果）。';
      }} else {{
        el('mode-banner').hidden = true;
      }}
    }} catch (err) {{
      apiConnected = false;
      bundle = EMBEDDED;
      el('api-dot').className = 'dot err';
      el('api-status-text').textContent = '离线/静态降级模式 (' + err.message + ')';
      el('mode-banner').hidden = false;
      el('mode-banner').className = bundle.is_fallback_demo ? 'banner error' : 'banner';
      el('banner-text').textContent = bundle.is_fallback_demo
        ? '后端 API 不可用且无内嵌实测数据：当前显示离线演示数据集，并非当前长视频实测结果。'
        : '后端 API 当前不可用（' + err.message + '），已自动切换为随页打包的实测快照；您的审核暂存本地浏览器，可随时导出 JSON 备份。';
    }}
    populateFilters();
    renderSummary();
    applyFilters(false);
  }}

  function fmtPct(v) {{
    return (v === null || v === undefined || Number.isNaN(v)) ? '—' : (v * 100).toFixed(1) + '%';
  }}

  function escHtml(v) {{
    return String(v ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }}

  function populateFilters() {{
    const vSel = el('filter-video');
    const prevV = vSel.value;
    vSel.innerHTML = '<option value="all">全部长视频</option>';
    for (const v of bundle.videos || []) {{
      const opt = document.createElement('option');
      opt.value = v.video_key;
      opt.textContent = v.title + ' (' + (v.duration_s / 60).toFixed(1) + ' min)';
      vSel.append(opt);
    }}
    if ([...vSel.options].some(o => o.value === prevV)) vSel.value = prevV;

    const bSel = el('filter-behavior');
    const prevB = bSel.value;
    bSel.innerHTML = '<option value="all">全部社交行为</option>';
    const labels = bundle.behavior_labels || {{}};
    for (const k of bundle.behavior_order || ['approach','attack','chase','nose_head_contact','nose_tail_contact','together']) {{
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = labels[k] || k;
      bSel.append(opt);
    }}
    if ([...bSel.options].some(o => o.value === prevB)) bSel.value = prevB;
  }}

  function renderSummary() {{
    const vids = bundle.videos || [];
    const totalRaw = vids.reduce((s, v) => s + (v.aggregate?.total_raw_social_candidates || 0), 0);
    const totalPost = vids.reduce((s, v) => s + (v.aggregate?.total_post_crop_social_segments || 0), 0);
    const totalAdj = vids.reduce((s, v) => s + (v.aggregate?.total_boundary_adjusted || 0), 0);
    const totalRej = vids.reduce((s, v) => s + (v.aggregate?.total_rejected_by_fine_crop || 0), 0);
    const clips = bundle.clips || [];
    const decPass = clips.filter(c => decisions[c.clip_uid]?.status === 'pass').length;
    const decFail = clips.filter(c => decisions[c.clip_uid]?.status === 'fail').length;
    const decDone = decPass + decFail;

    const durRates = vids.map(v => v.aggregate?.duration_pass_rate).filter(x => typeof x === 'number');
    const avgDurRate = durRates.length ? durRates.reduce((a, b) => a + b, 0) / durRates.length : null;
    const fullRates = vids.map(v => v.aggregate?.full_crop_precision_rate).filter(x => typeof x === 'number');
    const avgFull = fullRates.length ? fullRates.reduce((a, b) => a + b, 0) / fullRates.length : null;

    el('kpi-grid').innerHTML = `
      <div class="kpi"><div class="label">已验证长视频</div><div class="value">${{vids.length}}</div><div class="sub">${{escHtml(vids.map(v => v.title).join(' + ') || '—')}}</div></div>
      <div class="kpi"><div class="label">社交片段总量 (裁剪前 → 裁剪后)</div><div class="value">${{totalRaw}} → ${{totalPost}}</div><div class="sub">边界调整 ${{totalAdj}} 段 · 严格淘汰 ${{totalRej}} 段 · 重叠 0</div></div>
      <div class="kpi"><div class="label">最低时长达标 / 完整度</div><div class="value">${{escHtml(fmtPct(avgDurRate))}} / ${{escHtml(fmtPct(avgFull))}}</div><div class="sub">严格执行最低时长门控 + 端点双鼠完整可见</div></div>
      <div class="kpi"><div class="label">导出片段人工审核进度</div><div class="value">${{decDone}} / ${{clips.length}}</div><div class="sub">通过 ${{decPass}} · 不通过 ${{decFail}} ${{decDone ? '(通过率 ' + (100 * decPass / decDone).toFixed(1) + '%)' : ''}}</div></div>
    `;

    const tbody = el('summary-tbody');
    tbody.innerHTML = '';
    for (const v of vids) {{
      const keys = bundle.behavior_order || Object.keys(v.by_behavior || {{}});
      for (const k of keys) {{
        const r = (v.by_behavior || {{}})[k];
        if (!r) continue;
        const subClips = clips.filter(c => c.video_key === v.video_key && c.canonical_behavior === k);
        const sPass = subClips.filter(c => decisions[c.clip_uid]?.status === 'pass').length;
        const sFail = subClips.filter(c => decisions[c.clip_uid]?.status === 'fail').length;
        const revTxt = subClips.length ? `${{sPass}}✓ / ${{sFail}}✕ / ${{subClips.length}}段` : '0段';
        const fpClass = r.full_crop_precision_rate === null ? 'neutral' : r.full_crop_precision_rate >= 0.9 ? 'good' : r.full_crop_precision_rate >= 0.7 ? 'warn' : 'bad';
        const adjTxt = `${{r.changed_bounds_count}} (${{r.extended_count || 0}}延/${{r.trimmed_count || 0}}修)`;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="left">${{escHtml(v.title)}}</td>
          <td class="left">${{escHtml(r.label)}}</td>
          <td>${{escHtml(r.duration_comparator || '>=')}}${{escHtml(r.required_duration_s)}}s</td>
          <td>${{escHtml(r.raw_candidate_count)}}</td>
          <td>${{escHtml(r.post_crop_count)}}</td>
          <td>${{escHtml(adjTxt)}}</td>
          <td>${{escHtml(r.rejected_by_fine_crop_count)}}</td>
          <td>${{escHtml(r.min_post_duration_s)}}s</td>
          <td>${{escHtml(r.mean_raw_duration_s)}}s → ${{escHtml(r.mean_post_duration_s)}}s</td>
          <td>${{escHtml(fmtPct(r.duration_pass_rate))}}</td>
          <td>${{escHtml(fmtPct(r.endpoint_observed_rate))}}</td>
          <td><span class="tag ${{escHtml(fpClass)}}">${{escHtml(fmtPct(r.full_crop_precision_rate))}}</span></td>
          <td>${{escHtml(revTxt)}}</td>
        `;
        tbody.append(tr);
      }}
    }}
  }}

  function applyFilters(keepIndex = false) {{
    const fv = el('filter-video').value;
    const fb = el('filter-behavior').value;
    const fr = el('filter-review').value;
    const fbd = el('filter-boundary').value;

    const prevUid = filteredClips[currentIdx]?.clip_uid;
    filteredClips = (bundle.clips || []).filter(c => {{
      if (fv !== 'all' && c.video_key !== fv) return false;
      if (fb !== 'all' && c.canonical_behavior !== fb) return false;
      const st = decisions[c.clip_uid]?.status || 'pending';
      if (fr !== 'all' && st !== fr) return false;
      const chg = Boolean(c.audit?.boundary_delta?.changed);
      if (fbd === 'changed' && !chg) return false;
      if (fbd === 'unchanged' && chg) return false;
      return true;
    }});
    if (keepIndex && prevUid) {{
      const found = filteredClips.findIndex(c => c.clip_uid === prevUid);
      currentIdx = found >= 0 ? found : 0;
    }} else {{
      currentIdx = 0;
    }}
    renderClipList();
    renderCurrentClip();
  }}

  function renderClipList() {{
    el('list-header').textContent = '待审片段列表 (' + filteredClips.length + ')';
    const box = el('clip-list-items');
    box.innerHTML = '';
    filteredClips.forEach((c, idx) => {{
      const d = decisions[c.clip_uid];
      const st = d?.status || 'pending';
      const stLabel = st === 'pass' ? '通过' : st === 'fail' ? '不通过' : '待审';
      const stCls = st === 'pass' ? 'good' : st === 'fail' ? 'bad' : 'neutral';
      const roleSummary = c.roles?.summary_zh || (c.canonical_behavior === 'together' ? '无方向（共同伴随）' : '');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'clip-item' + (idx === currentIdx ? ' selected' : '');
      btn.innerHTML = `
        <div class="row-top">
          <span>#${{idx + 1}} · ${{escHtml(c.behavior_display)}}</span>
          <span class="tag ${{escHtml(stCls)}}">${{escHtml(stLabel)}}</span>
        </div>
        <div class="row-sub">${{escHtml(c.video_title)}} · ${{(c.duration_s || 0).toFixed(2)}}s · ID ${{escHtml((c.mouse_ids || []).join(','))}}${{roleSummary ? ' · ' + escHtml(roleSummary) : ''}}</div>
      `;
      btn.onclick = () => {{ currentIdx = idx; renderClipList(); renderCurrentClip(); }};
      box.append(btn);
    }});
  }}

  function renderCurrentClip() {{
    const c = filteredClips[currentIdx];
    const hasClip = Boolean(c);
    el('video-empty').hidden = hasClip;
    el('video-player').hidden = !hasClip;
    el('audit-grid').hidden = !hasClip;
    el('review-box').hidden = !hasClip;
    el('video-error').textContent = '';

    if (!hasClip) {{
      el('clip-title').textContent = '暂无符合筛选条件的片段';
      el('clip-counter').textContent = '0 / 0';
      el('video-player').removeAttribute('src');
      el('video-player').load();
      el('clip-links').innerHTML = '';
      return;
    }}

    el('clip-title').textContent = `${{c.video_title}} · ${{c.behavior_label}}`;
    el('clip-counter').textContent = `${{currentIdx + 1}} / ${{filteredClips.length}}`;
    el('btn-variant-overlay').className = variant === 'overlay' ? 'active' : '';
    el('btn-variant-raw').className = variant === 'raw' ? 'active' : '';

    const srcUrl = variant === 'overlay' ? (c.overlay_url || c.raw_url) : (c.raw_url || c.overlay_url);
    const vid = el('video-player');
    if (!srcUrl) {{
      vid.removeAttribute('data-src');
      vid.removeAttribute('src');
      vid.load();
    }} else if (vid.getAttribute('data-src') !== srcUrl) {{
      vid.setAttribute('data-src', srcUrl);
      vid.src = srcUrl;
    }}

    const a = c.audit || {{}};
    const sa = Object.fromEntries(Object.entries(a).map(([k, v]) => [k, (v === null || v === undefined || typeof v === 'object') ? v : escHtml(v)]));
    const bd = a.boundary_delta || {{}};
    const rawInfo = a.raw_start_frame !== undefined
      ? `[${{a.raw_start_frame}}, ${{a.raw_end_frame}}) · ${{a.raw_duration_s}}s`
      : '同裁剪窗口';
    const metricItems = Object.entries(a)
      .filter(([k]) => !['audit_notes','boundary_delta','raw_parent_event_id','raw_start_frame','raw_end_frame','raw_duration_s'].includes(k))
      .map(([k, v]) => `${{escHtml(k)}}: ${{escHtml(v)}}`)
      .join('<br>');
    let kinBadges = '';
    const idswBadge = c.has_visual_id_switch
      ? `<span style="display:inline-block;margin-right:6px;padding:1px 6px;border-radius:4px;background:#78350f;color:#fde68a;font-weight:600;">含YOLO ID切换/合并 · 保留原始Visual ID</span>`
      : '';
    if (c.canonical_behavior === 'approach' && sa.approach_start_dist_cm !== undefined) {{
      kinBadges = `<div style="margin-bottom:6px;color:#7dd3fc;">起止距离: ${{sa.approach_start_dist_cm}} → ${{sa.approach_end_dist_cm}} cm (缩短 ${{sa.approach_closing_cm}} cm) · 末段速度cos/头指向cos: ${{sa.approach_end_toward_cos ?? '—'}} / ${{sa.approach_end_heading_cos ?? '—'}} (主导差 ${{sa.approach_end_actor_dominance_cm_s ?? '—'}} cm/s) · 末帧接框间隙 ${{sa.approach_end_box_gap_cm ?? sa.min_pair_box_gap_cm}} cm · 距他鼠 ≥ ${{sa.min_other_mouse_dist_cm}} cm</div>`;
    }} else if (c.canonical_behavior === 'attack' && sa.attack_peak_speed_cm_s !== undefined) {{
      kinBadges = `<div style="margin-bottom:6px;color:#fda4af;">${{idswBadge}}面积倍率/IoU: ${{sa.attack_max_area_ratio ?? '—'}}x / ${{sa.attack_max_iou ?? '—'}} · ID轮换: ${{sa.attack_id_churn ?? 0}} · 峰值/均速: ${{sa.attack_peak_speed_cm_s}} / ${{sa.attack_close_mean_speed_cm_s ?? '—'}} cm/s · 翻滚路径: ${{sa.attack_tumble_path_cm ?? sa.attack_mid_path_cm ?? '—'}} cm</div>`;
    }} else if ((c.canonical_behavior === 'chase' || c.canonical_behavior === 'chasing') && sa.chase_mean_follower_speed_cm_s !== undefined) {{
      kinBadges = `<div style="margin-bottom:6px;color:#86efac;">${{idswBadge}}追/逃均速: ${{sa.chase_mean_follower_speed_cm_s}} / ${{sa.chase_mean_leader_speed_cm_s}} cm/s (P25双鼠速 ${{sa.chase_both_p25_speed_cm_s ?? '—'}} · 峰值 ${{sa.chase_peak_follower_speed_cm_s}} / ${{sa.chase_peak_leader_speed_cm_s}} cm/s) · 速度同向cos(P25): ${{sa.chase_mean_vel_cos ?? '—'}} (${{sa.chase_vel_cos_p25 ?? '—'}}) · 追向cos: ${{sa.chase_mean_cos ?? '—'}} · 体轴cos: ${{sa.chase_body_heading_cos ?? '—'}} · 尾迹误差: ${{sa.chase_trail_error_cm ?? '—'}} cm · 净位移: ${{sa.chase_min_net_disp_cm ?? '—'}} cm</div>`;
    }}

    const roleDisplay = c.roles?.summary_zh || a.role_summary_zh || (c.canonical_behavior === 'together' ? '无方向（共同伴随）' : '未指定');

    el('audit-grid').innerHTML = `
      <div class="audit-box">
        <strong>裁剪帧区间、参与鼠与角色分工</strong>
        裁剪后源帧: [${{escHtml(c.start_frame)}}, ${{escHtml(c.end_frame)}}) (${{escHtml(c.frame_count)}} 帧 / ${{(c.duration_s||0).toFixed(2)}}s)<br>
        时间段: ${{(c.start_time_s||0).toFixed(2)}}s – ${{(c.end_time_s||0).toFixed(2)}}s<br>
        参与 Visual ID: [${{escHtml((c.mouse_ids || []).join(', '))}}] · 角色分工: <span style="color:#fcd34d;">${{escHtml(roleDisplay)}}</span>
      </div>
      <div class="audit-box">
        <strong>裁剪前后边界对比</strong>
        裁剪前规则窗口: ${{escHtml(rawInfo)}}<br>
        起点偏移: ${{escHtml(bd.start_shift_s ?? 0)}}s · 尾段裁剪: ${{escHtml(bd.end_trim_s ?? 0)}}s<br>
        变化状态: ${{escHtml(bd.classification || (bd.changed ? '已调整' : '未变'))}}
      </div>
      <div class="audit-box">
        <strong>严格时长与完整性校验 (${{a.full_crop_precision_ok ? 'PASS' : '待关注'}})</strong>
        ${{kinBadges}}${{metricItems || '暂无附加指标'}}
      </div>
      <div class="audit-box">
        <strong>自动审计结论</strong>
        ${{escHtml((a.audit_notes || []).join('；') || '—')}}
      </div>
    `;

    el('clip-links').innerHTML = c.json_base_url
      ? ['annotation.json', 'metadata.json', 'tracks.json']
          .map(f => `<a href="${{escHtml(c.json_base_url)}}/${{f}}" target="_blank" rel="noopener" style="color:#7dd3fc;margin-right:12px;">${{f}}</a>`)
          .join('')
      : '';

    const dec = decisions[c.clip_uid] || {{ status: 'pending', note: '', tags: [] }};
    el('btn-pass').className = 'pass' + (dec.status === 'pass' ? ' active' : '');
    el('btn-fail').className = 'fail' + (dec.status === 'fail' ? ' active' : '');
    el('btn-pending').className = dec.status === 'pending' ? 'active' : '';
    el('review-note').value = dec.note || '';

    const chipBox = el('reason-chips');
    chipBox.innerHTML = '';
    const activeTags = new Set(dec.tags || []);
    for (const tag of REASON_TAGS) {{
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip' + (activeTags.has(tag) ? ' active' : '');
      btn.textContent = tag;
      btn.onclick = () => {{
        const cur = decisions[c.clip_uid] || {{ status: 'fail', note: '', tags: [] }};
        const set = new Set(cur.tags || []);
        if (set.has(tag)) set.delete(tag); else set.add(tag);
        updateDecision(c, cur.status || 'fail', el('review-note').value, [...set], false);
      }};
      chipBox.append(btn);
    }}
  }}

  async function updateDecision(clip, status, note, tags, advance) {{
    decisions[clip.clip_uid] = {{
      clip_uid: clip.clip_uid, video_key: clip.video_key, event_id: clip.event_id,
      behavior: clip.canonical_behavior, status, note: note || '', tags: tags || [],
      updated_at: new Date().toISOString(),
    }};
    saveLocalDecisions();
    el('review-save-status').textContent = '保存中…';
    if (apiConnected) {{
      try {{
        const resp = await fetch('/api/reviews', {{
          method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(decisions[clip.clip_uid]),
        }});
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        el('review-save-status').textContent = '已保存到服务端 JSON · ' + new Date().toLocaleTimeString();
      }} catch (err) {{
        el('review-save-status').textContent = '服务端保存失败，已暂存本地 (' + err.message + ')';
      }}
    }} else {{
      el('review-save-status').textContent = '已保存至本地浏览器 · ' + new Date().toLocaleTimeString();
    }}
    renderSummary();
    if (advance && el('chk-auto-next').checked && currentIdx < filteredClips.length - 1) currentIdx += 1;
    renderClipList();
    renderCurrentClip();
  }}

  function downloadFile(filename, content, mime) {{
    const url = URL.createObjectURL(new Blob([content], {{ type: mime }}));
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }}

  el('btn-variant-overlay').onclick = () => {{ variant = 'overlay'; renderCurrentClip(); }};
  el('btn-variant-raw').onclick = () => {{ variant = 'raw'; renderCurrentClip(); }};
  el('btn-prev').onclick = () => {{ if (filteredClips.length) {{ currentIdx = (currentIdx - 1 + filteredClips.length) % filteredClips.length; renderClipList(); renderCurrentClip(); }} }};
  el('btn-next').onclick = () => {{ if (filteredClips.length) {{ currentIdx = (currentIdx + 1) % filteredClips.length; renderClipList(); renderCurrentClip(); }} }};
  el('retry-api').onclick = () => syncWithBackend();
  el('filter-video').onchange = () => applyFilters(false);
  el('filter-behavior').onchange = () => applyFilters(false);
  el('filter-review').onchange = () => applyFilters(false);
  el('filter-boundary').onchange = () => applyFilters(false);
  el('video-player').onerror = () => {{ el('video-error').textContent = '视频无法播放：请确认本地服务处于运行状态，或尝试切换原始/高亮版本。'; }};
  el('btn-pass').onclick = () => {{ const c = filteredClips[currentIdx]; if (c) updateDecision(c, 'pass', el('review-note').value, decisions[c.clip_uid]?.tags || [], true); }};
  el('btn-fail').onclick = () => {{ const c = filteredClips[currentIdx]; if (c) updateDecision(c, 'fail', el('review-note').value, decisions[c.clip_uid]?.tags || [], true); }};
  el('btn-pending').onclick = () => {{ const c = filteredClips[currentIdx]; if (c) updateDecision(c, 'pending', el('review-note').value, [], false); }};
  el('review-note').onblur = () => {{ const c = filteredClips[currentIdx]; if (c && decisions[c.clip_uid]) updateDecision(c, decisions[c.clip_uid].status, el('review-note').value, decisions[c.clip_uid].tags || [], false); }};
  el('btn-export-json').onclick = () => downloadFile('long_video_social_reviews_' + new Date().toISOString().slice(0,10) + '.json', JSON.stringify({{ schema: 'long-video-social-review-v1', exported_at: new Date().toISOString(), decisions }}, null, 2), 'application/json');
  el('btn-export-csv').onclick = () => {{
    const rows = [['clip_uid','video_key','behavior','event_id','start_frame','end_frame','duration_s','mouse_ids','status','tags','note']];
    for (const c of bundle.clips || []) {{
      const d = decisions[c.clip_uid] || {{}};
      rows.push([c.clip_uid, c.video_key, c.canonical_behavior, c.event_id, c.start_frame, c.end_frame, (c.duration_s||0).toFixed(3), '"' + (c.mouse_ids||[]).join(',') + '"', d.status || 'pending', '"' + (d.tags||[]).join(';') + '"', '"' + String(d.note||'').replace(/"/g,'""') + '"']);
    }}
    downloadFile('long_video_social_reviews.csv', '\\uFEFF' + rows.map(r => r.join(',')).join('\\r\\n'), 'text/csv;charset=utf-8');
  }};
  el('input-import-json').onchange = async e => {{
    const f = e.target.files[0];
    if (!f) return;
    try {{
      const data = JSON.parse(await f.text());
      if (!data || typeof data.decisions !== 'object') throw new Error('格式无效');
      decisions = {{ ...decisions, ...data.decisions }};
      saveLocalDecisions();
      renderSummary(); renderClipList(); renderCurrentClip();
      el('review-save-status').textContent = '已恢复导入的审核备份';
    }} catch (err) {{
      el('review-save-status').textContent = '导入失败: ' + err.message;
    }}
    e.target.value = '';
  }};

  loadLocalDecisions();
  populateFilters();
  renderSummary();
  applyFilters(false);
  syncWithBackend();
}})();
</script>
</body>
</html>"""


class LongVideoSocialReviewHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the multi-video social review SPA, APIs, and Range-streamed MP4 clips."""

    site_dir: Path = DEFAULT_SOCIAL_REVIEW_SITE_DIR

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("", "127.0.0.1", "localhost", "::1")

    def _origin_ok(self) -> bool:
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return True
        parsed = urlsplit(origin)
        hostname = (parsed.hostname or "").lower()
        return parsed.scheme in ("http", "https") and hostname in ("127.0.0.1", "localhost", "::1")

    def _send_json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_file(self, file_path: Path, mime_type: str, head_only: bool = False) -> None:
        if not file_path.is_file():
            self._send_json(404, {"error": "File not found"})
            return
        size = file_path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = 200
        range_hdr = self.headers.get("Range")
        if range_hdr and size > 0 and range_hdr.startswith("bytes="):
            spec = range_hdr[6:].split(",")[0].strip()
            if "-" in spec:
                s_part, e_part = spec.split("-", 1)
                try:
                    start = int(s_part) if s_part else max(0, size - int(e_part))
                    end = int(e_part) if e_part and s_part else size - 1
                    end = min(end, size - 1)
                    if 0 <= start <= end < size:
                        status = 206
                    else:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                except ValueError:
                    start, end, status = 0, size - 1, 200

        length = (end - start + 1) if size > 0 else 0
        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only or size == 0:
            return
        try:
            with file_path.open("rb") as handle:
                handle.seek(start)
                rem = length
                while rem > 0:
                    chunk = handle.read(min(256 * 1024, rem))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    rem -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_HEAD(self) -> None:
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False) -> None:
        if not self._host_ok():
            self._send_json(403, {"error": "Loopback access only"})
            return
        req_path = unquote(urlsplit(self.path).path).strip("/")
        if req_path in ("", "index.html"):
            idx = self.site_dir / "index.html"
            if idx.is_file():
                self._stream_file(idx, "text/html; charset=utf-8", head_only)
                return
            html = render_long_video_social_review_page(embedded_bundle=None).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            if not head_only:
                self.wfile.write(html)
            return

        if req_path == "api/health":
            bundle_file = self.site_dir / "data" / "bundle.json"
            self._send_json(
                200,
                {
                    "status": "ok",
                    "site_dir_exists": self.site_dir.is_dir(),
                    "bundle_exists": bundle_file.is_file(),
                },
            )
            return

        if req_path == "api/bundle":
            bundle_file = self.site_dir / "data" / "bundle.json"
            if not bundle_file.is_file():
                self._send_json(200, DEMO_FALLBACK_SOCIAL_BUNDLE)
                return
            try:
                data = json.loads(bundle_file.read_text(encoding="utf-8"))
                if not validate_social_review_bundle_schema(data):
                    self._send_json(500, {"error": "Invalid bundle schema"})
                    return
                self._send_json(200, data)
            except Exception as exc:
                self._send_json(500, {"error": f"Bundle read failed: {exc}"})
            return

        if req_path == "api/reviews":
            rev_file = self.site_dir / "data" / "human_reviews.json"
            with _REVIEWS_LOCK:
                if not rev_file.is_file():
                    self._send_json(200, {"schema": "long-video-social-review-v1", "decisions": {}})
                    return
                try:
                    data = json.loads(rev_file.read_text(encoding="utf-8"))
                    self._send_json(200, data)
                except Exception as exc:
                    self._send_json(500, {"error": f"Reviews read failed: {exc}"})
            return

        if req_path.startswith("videos/"):
            if not _is_safe_videos_req_path(req_path):
                self._send_json(400, {"error": "Invalid path"})
                return
            found = _lookup_safe_video_file(self.site_dir, req_path)
            if found is None:
                self._send_json(404, {"error": "File not found"})
                return
            target_path, safe_mime = found
            self._stream_file(target_path, safe_mime, head_only)
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._host_ok():
            self._send_json(403, {"error": "Loopback access only"})
            return
        if not self._origin_ok():
            self._send_json(403, {"error": "Cross-origin POST rejected"})
            return
        req_path = unquote(urlsplit(self.path).path).strip("/")
        if req_path != "api/reviews":
            self._send_json(404, {"error": "Not found"})
            return
        raw_len = self.headers.get("Content-Length")
        try:
            length = int(raw_len) if raw_len is not None else 0
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if length <= 0:
            self._send_json(400, {"error": "Positive Content-Length required"})
            return
        if length > _MAX_REVIEW_BODY_BYTES:
            self._send_json(413, {"error": "Payload too large"})
            return
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"Malformed JSON payload: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON object payload required"})
            return
        clip_uid = str(payload.get("clip_uid", ""))
        status = str(payload.get("status", ""))
        if not clip_uid or status not in {"pass", "fail", "pending"}:
            self._send_json(400, {"error": "clip_uid and valid status (pass/fail/pending) required"})
            return
        rev_file = self.site_dir / "data" / "human_reviews.json"
        rev_file.parent.mkdir(parents=True, exist_ok=True)
        with _REVIEWS_LOCK:
            existing: dict[str, Any] = {"schema": "long-video-social-review-v1", "decisions": {}}
            if rev_file.is_file():
                try:
                    loaded = json.loads(rev_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    self._send_json(500, {"error": f"Existing reviews store is corrupt: {exc}"})
                    return
                if not isinstance(loaded, dict) or not isinstance(loaded.get("decisions", {}), dict):
                    self._send_json(500, {"error": "Existing reviews store schema is invalid"})
                    return
                existing = loaded
            decisions = existing.setdefault("decisions", {})
            decisions[clip_uid] = payload
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            tmp_file = rev_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_file, rev_file)
        self._send_json(200, {"status": "saved", "clip_uid": clip_uid, "total_decisions": len(decisions)})


def create_long_video_social_review_server(
    site_dir: Path = DEFAULT_SOCIAL_REVIEW_SITE_DIR,
    port: int = DEFAULT_SOCIAL_REVIEW_PORT,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """Create a loopback HTTP server for the long-video social review website."""
    resolved = Path(site_dir).resolve()

    class BoundHandler(LongVideoSocialReviewHandler):
        pass

    BoundHandler.site_dir = resolved
    server = ThreadingHTTPServer((host, port), BoundHandler)
    server.daemon_threads = True
    return server
