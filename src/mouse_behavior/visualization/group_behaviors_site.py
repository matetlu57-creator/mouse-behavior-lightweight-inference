"""Standalone, self-contained local website builder & server for the three group behaviors:
Dispersal (散开/群体分散), Group locomotion (群体同步运动), and Social clustering (社会聚集).

This module is completely isolated from ``comparison_site.py`` and generates a
dedicated local website bundle under ``outputs/group_behaviors_website`` containing
its own HTML/JS app, evaluation JSON, local MP4 video copies, standalone HTTP
server script, and double-click Windows launcher.
"""
from __future__ import annotations

import json
import mimetypes
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUTS_DIR = _REPO_ROOT / "outputs"

DEFAULT_GROUP_EVAL_JSON = _OUTPUTS_DIR / "group_behaviors_evaluation.json"
DEFAULT_SITE_DIR = _OUTPUTS_DIR / "group_behaviors_website"
DEFAULT_SITE_PORT = 8780

DEMO_FALLBACK_GROUP_DATA: dict[str, Any] = {
    "is_fallback_demo": True,
    "demo_warning": "当前展示为内置离线演示数据（仅用于界面交互验证，非后端实时科研结果）",
    "evaluation_title": "三种群体行为独立本地视频对比系统（离线演示集）",
    "dataset_total_clips": 605,
    "total_clips": 3,
    "dispersal_clips": 1,
    "group_locomotion_clips": 1,
    "social_clustering_clips": 1,
    "overall_accuracy": 1.0,
    "overall_mean_iou": 1.0,
    "iou_ge_90_rate": 1.0,
    "iou_ge_95_rate": 1.0,
    "behavior_specs": [
        {
            "behavior": "Dispersal",
            "zh_name": "散开 / 群体分散",
            "clip_count": 1,
            "dataset_total_clips": 605,
            "dataset_share_pct": 0.17,
            "rules": [
                "参与门槛：至少 2 只个体向外辐射撤离，原始集群通常为 3~5 只（R ≤ 10.0 cm / 120 px）。",
                "起始边界：定位于仍紧凑、即将扩散的瞬间，支持最长 4.0 s 向前平滑回溯。",
                "扩散与终止：向外净位移 ≥ 4.0 cm (25 px)、峰值半径增益 ≥ 35 px；重新靠拢持续 > 0.5 s 前截断。",
                "时限门控：总持续时间严格 > 5.0 s（dispersal_min_duration_s）。",
            ],
        },
        {
            "behavior": "Group locomotion",
            "zh_name": "群体同步运动",
            "clip_count": 1,
            "dataset_total_clips": 605,
            "dataset_share_pct": 0.17,
            "rules": [
                "参与门槛：至少 3 只小鼠且两两间距保持在群体包络内（< 64.04 cm / 默认半径 ≤ 145 px）。",
                "起始边界：所有参与个体同时启动协同运动的活跃帧（moving.any）。",
                "协同条件：速度方向平均两两余弦 ≥ 0.628、净位移 ≥ 13.84 cm、路径均衡比 min/max ≥ 0.40。",
                "时限门控：总跨度严格 > 3.0 s 且高度同步帧累计支撑 ≥ 0.933 s。",
            ],
        },
        {
            "behavior": "Social clustering",
            "zh_name": "社会聚集",
            "clip_count": 1,
            "dataset_total_clips": 605,
            "dataset_share_pct": 0.17,
            "rules": [
                "参与门槛：至少 3 只小鼠，目标紧凑聚集半径 R ≤ 10.0 cm × (1 + 0.20 × max(0, k − 3))。",
                "起始边界：向前回溯最多 3.0 s 定位发起者定向靠近（速度 ≥ 2.0 cm/s、余弦 ≥ 0.5）的起点。",
                "收敛条件：聚集前 90 分位半径与核心半径落差 ≥ 1.0 cm，发起路径 ≥ 10.0 cm；排除平行同步移动。",
                "时限门控：终态稳定紧凑核心保持时间 ≥ 0.50 s（clustering_final_stability_s）。",
            ],
        },
    ],
    "per_clip_results": [
        {
            "clip_id": "083919df8ea7f9a223008fbb",
            "ground_truth_behavior": "Dispersal",
            "predicted_behavior": "Dispersal",
            "is_correct": True,
            "target_mouse_ids": [5178, 5194, 5281, 5432],
            "semantic_roles": {"Dispersal": [5178, 5194, 5281, 5432]},
            "ground_truth_range_half_open": {"start": 0, "end": 325},
            "predicted_range_half_open": {"start": 0, "end": 325},
            "temporal_iou": 1.0,
            "iou_ge_90": True,
            "iou_ge_95": True,
            "total_frames": 325,
            "fps": 30.01,
        },
        {
            "clip_id": "012e147d845dba5f6acf6a01",
            "ground_truth_behavior": "Group locomotion",
            "predicted_behavior": "Group locomotion",
            "is_correct": True,
            "target_mouse_ids": [978, 1762, 2895],
            "semantic_roles": {"Group Locomotion": [978, 1762, 2895]},
            "ground_truth_range_half_open": {"start": 0, "end": 124},
            "predicted_range_half_open": {"start": 0, "end": 124},
            "temporal_iou": 1.0,
            "iou_ge_90": True,
            "iou_ge_95": True,
            "total_frames": 124,
            "fps": 30.01,
        },
        {
            "clip_id": "0127c0e6bfc16e1d626a1224",
            "ground_truth_behavior": "Social clustering",
            "predicted_behavior": "Social clustering",
            "is_correct": True,
            "target_mouse_ids": [3329, 3509, 3525, 3545],
            "semantic_roles": {"Social Clustering": [3329, 3509, 3525, 3545]},
            "ground_truth_range_half_open": {"start": 0, "end": 96},
            "predicted_range_half_open": {"start": 0, "end": 96},
            "temporal_iou": 1.0,
            "iou_ge_90": True,
            "iou_ge_95": True,
            "total_frames": 96,
            "fps": 30.01,
        },
    ],
}


def _safe_script_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON safely for inline <script> blocks without </script> breakout."""
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _copy_if_needed(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        s_stat = src.stat()
        d_stat = dst.stat()
        if s_stat.st_size == d_stat.st_size and int(s_stat.st_mtime) == int(d_stat.st_mtime):
            return True
    shutil.copy2(src, dst)
    return True


def render_group_behaviors_page(
    bundle_data: Mapping[str, Any] | None = None,
    fallback_data: Mapping[str, Any] | None = None,
) -> str:
    """Render standalone HTML application exclusively for Dispersal, Group locomotion, Social clustering."""
    raw_fallback = dict(fallback_data) if fallback_data is not None else dict(DEMO_FALLBACK_GROUP_DATA)
    raw_fallback["is_fallback_demo"] = True

    if bundle_data is not None:
        raw_bundle = dict(bundle_data)
    else:
        raw_bundle = dict(raw_fallback)

    bundle_json = _safe_script_json(raw_bundle)
    fallback_json = _safe_script_json(raw_fallback)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>小鼠群体行为独立视频验证平台 · Dispersal / Group locomotion / Social clustering</title>
<style>
  :root {{
    --bg-canvas: #090d16;
    --bg-surface: #111827;
    --bg-card: #162033;
    --bg-card-hover: #1e2b45;
    --border: #263552;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-sky: #38bdf8;
    --accent-emerald: #10b981;
    --accent-rose: #f472b6;
    --accent-violet: #a78bfa;
    --accent-teal: #2dd4bf;
    --accent-amber: #fbbf24;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    height: 100%;
    max-height: 100vh;
    overflow: hidden;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
    background: var(--bg-canvas);
    color: var(--text-primary);
    display: flex;
    flex-direction: column;
    line-height: 1.5;
  }}

  header {{
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }}

  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }}

  .brand h1 {{
    font-size: 1.08rem;
    font-weight: 700;
    color: #fff;
  }}

  .status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
  }}

  .status-live {{
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
  }}

  .status-local {{
    background: rgba(56, 189, 248, 0.18);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.35);
  }}

  .status-demo {{
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
  }}

  .status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
  }}

  #alertBanner {{
    display: none;
    flex-shrink: 0;
    background: rgba(185, 28, 28, 0.22);
    border-bottom: 1px solid #b91c1c;
    color: #fecaca;
    padding: 8px 20px;
    font-size: 0.82rem;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }}

  .btn {{
    background: #1e293b;
    color: var(--text-primary);
    border: 1px solid #3b4c6b;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}

  .btn:hover {{ background: #2c3b57; }}

  .btn-primary {{
    background: var(--accent-sky);
    color: #090d16;
    border-color: var(--accent-sky);
  }}

  .btn-primary:hover {{ background: #7dd3fc; }}

  .metrics-bar {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 8px;
    padding: 8px 18px;
    background: rgba(17, 24, 39, 0.85);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}

  .metrics-bar.collapsed {{ display: none; }}

  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
  }}

  .kpi-label {{
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-weight: 600;
  }}

  .kpi-value {{
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 2px;
  }}

  .kpi-sub {{
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 2px;
  }}

  .rules-drawer {{
    display: none;
    flex-shrink: 0;
    background: #0d1424;
    border-bottom: 1px solid var(--border);
    padding: 12px 18px;
    max-height: 40vh;
    overflow-y: auto;
  }}

  .rules-drawer.open {{ display: block; }}

  .rules-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 10px;
  }}

  .rule-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: var(--text-secondary);
  }}

  .rule-card h3 {{
    font-size: 0.88rem;
    color: var(--text-primary);
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }}

  .rule-card ul {{ padding-left: 16px; }}
  .rule-card li {{ margin-bottom: 3px; }}

  main {{
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 350px 1fr;
    overflow: hidden;
  }}

  aside {{
    background: #0c1220;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }}

  .aside-header {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}

  .search-input {{
    width: 100%;
    background: #1a2438;
    border: 1px solid #324463;
    border-radius: 6px;
    padding: 7px 10px;
    color: #fff;
    font-size: 0.84rem;
    margin-bottom: 8px;
    outline: none;
  }}

  .search-input:focus {{ border-color: var(--accent-sky); }}

  .filter-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }}

  .filter-pill {{
    background: #1a2438;
    color: var(--text-secondary);
    border: 1px solid #324463;
    border-radius: 9999px;
    padding: 3px 9px;
    font-size: 0.72rem;
    cursor: pointer;
  }}

  .filter-pill.active {{
    background: var(--accent-sky);
    color: #090d16;
    font-weight: 700;
    border-color: var(--accent-sky);
  }}

  .clip-list {{
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 10px;
  }}

  .clip-item {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid transparent;
    border-radius: 6px;
    padding: 9px 11px;
    margin-bottom: 8px;
    cursor: pointer;
  }}

  .clip-item:hover {{ background: var(--bg-card-hover); }}

  .clip-item.selected {{
    border-color: var(--accent-sky);
    border-left-color: var(--accent-sky);
    background: #1e2b45;
  }}

  .clip-item-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}

  .clip-badge {{
    font-size: 0.69rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
  }}

  .badge-dispersal {{
    background: rgba(244, 114, 182, 0.18);
    color: #f472b6;
    border: 1px solid rgba(244, 114, 182, 0.4);
  }}

  .badge-gloco {{
    background: rgba(167, 139, 250, 0.18);
    color: #a78bfa;
    border: 1px solid rgba(167, 139, 250, 0.4);
  }}

  .badge-clustering {{
    background: rgba(45, 212, 191, 0.18);
    color: #2dd4bf;
    border: 1px solid rgba(45, 212, 191, 0.4);
  }}

  .clip-id {{
    font-family: monospace;
    font-size: 0.78rem;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .clip-meta {{
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 3px;
  }}

  .stage-panel {{
    min-height: 0;
    overflow-y: auto;
    padding: 12px 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}

  .stage-header {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }}

  .video-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}

  .video-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}

  .video-card-header {{
    padding: 8px 12px;
    background: rgba(12, 18, 32, 0.85);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    font-weight: 600;
    gap: 8px;
  }}

  .video-wrapper {{
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    max-height: min(40vh, 380px);
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  .video-wrapper video {{
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}

  .video-overlay {{
    display: none;
    position: absolute;
    inset: 0;
    background: rgba(9, 13, 22, 0.92);
    color: #fbbf24;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 16px;
    text-align: center;
    gap: 6px;
    z-index: 10;
  }}

  .video-card-footer {{
    padding: 6px 10px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    background: rgba(12, 18, 32, 0.65);
    border-top: 1px solid var(--border);
  }}

  .player-controls {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}

  .timeline-bar {{
    position: relative;
    width: 100%;
    height: 12px;
    background: #2b3a55;
    border-radius: 6px;
    cursor: pointer;
  }}

  .timeline-gt {{
    position: absolute;
    inset: 0 auto 0 0;
    background: rgba(56, 189, 248, 0.4);
    border: 1px solid var(--accent-sky);
    border-radius: 4px;
    pointer-events: none;
  }}

  .timeline-pred {{
    position: absolute;
    inset: 2px auto 2px 0;
    background: rgba(16, 185, 129, 0.55);
    border: 1px solid var(--accent-emerald);
    border-radius: 3px;
    pointer-events: none;
  }}

  .timeline-progress {{
    position: absolute;
    inset: 0 auto 0 0;
    width: 0%;
    background: rgba(255, 255, 255, 0.3);
    pointer-events: none;
  }}

  .timeline-scrubber {{
    position: absolute;
    top: -4px;
    width: 4px;
    height: 20px;
    background: #fff;
    border-radius: 2px;
    transform: translateX(-50%);
    pointer-events: none;
  }}

  .control-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }}

  .telemetry-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
  }}

  .telemetry-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 10px;
    margin-top: 10px;
  }}

  .tele-item {{
    background: rgba(12, 18, 32, 0.7);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 9px 11px;
  }}

  .tele-label {{ font-size: 0.7rem; color: var(--text-secondary); font-weight: 600; }}
  .tele-val {{ font-size: 0.96rem; font-weight: 700; margin-top: 2px; }}
  .tele-sub {{ font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }}

  @media (max-width: 1024px) {{
    html, body {{ overflow: auto; height: auto; max-height: none; }}
    main {{ grid-template-columns: 1fr; overflow: visible; }}
    aside {{ height: 300px; }}
    .video-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>小鼠群体行为独立视频验证平台</h1>
    <span id="connBadge" class="status-badge status-local">
      <span class="status-dot"></span>
      <span id="connText">初始化界面中...</span>
    </span>
  </div>
  <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
    <button class="btn" id="btnToggleRules" onclick="toggleRulesDrawer()">📖 边界规则与占比</button>
    <button class="btn" id="btnToggleMetrics" onclick="toggleMetricsBar()">📊 统计面板</button>
    <button class="btn" id="btnRefresh" onclick="retryFetchData()">🔄 刷新数据</button>
  </div>
</header>

<div id="alertBanner" role="alert">
  <span id="alertMsg">⚠️ 后端数据接口暂不可用，当前展示内置离线演示数据集。</span>
  <button class="btn" style="background:#ef4444; border-color:#ef4444;" onclick="retryFetchData()">重试连接</button>
</div>

<section class="metrics-bar" id="metricsBar" aria-label="三种群体行为核心指标">
  <div class="kpi-card">
    <div class="kpi-label">群体行为样本总数</div>
    <div class="kpi-value" id="kpiTotal">--</div>
    <div class="kpi-sub" id="kpiTotalSub">加载中...</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Dispersal · 散开准确率</div>
    <div class="kpi-value" style="color:var(--accent-rose);" id="kpiDispersal">--</div>
    <div class="kpi-sub" id="kpiDispersalSub">--</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Group locomotion · 同步准确率</div>
    <div class="kpi-value" style="color:var(--accent-violet);" id="kpiGloco">--</div>
    <div class="kpi-sub" id="kpiGlocoSub">--</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Social clustering · 聚集准确率</div>
    <div class="kpi-value" style="color:var(--accent-teal);" id="kpiClustering">--</div>
    <div class="kpi-sub" id="kpiClusteringSub">--</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">平均时序重叠度 (Mean IoU)</div>
    <div class="kpi-value" style="color:var(--accent-emerald);" id="kpiMeanIoU">--</div>
    <div class="kpi-sub" id="kpiIouSub">重叠率 ≥95% 达标</div>
  </div>
</section>

<section class="rules-drawer" id="rulesDrawer" aria-label="启发式规则边界与605数据集占比">
  <div class="rules-grid" id="rulesGrid"></div>
</section>

<main>
  <aside>
    <div class="aside-header">
      <input type="text" id="searchInput" class="search-input" placeholder="🔍 搜索片段ID / 小鼠ID / 行为..." oninput="renderClipList()">
      <div class="filter-pills" id="filterPills">
        <button class="filter-pill active" data-filter="all">全部 (--)</button>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:7px; font-size:0.72rem; color:var(--text-secondary);">
        <span id="clipCountText">显示 0 / 0 段</span>
        <span style="color:var(--text-muted);">↑/↓ 切片段 · 空格播放</span>
      </div>
    </div>
    <div class="clip-list" id="clipList"></div>
  </aside>

  <div class="stage-panel">
    <div class="stage-header">
      <div>
        <div style="font-size:1.02rem; font-weight:700;" id="stageTitle">请选择片段</div>
        <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:2px;" id="stageSubtitle">加载中...</div>
      </div>
      <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
        <div id="stageTags" style="display:flex; gap:6px; flex-wrap:wrap;"></div>
        <button class="btn" onclick="navigateClip(-1)">⏮ 上一个</button>
        <button class="btn" onclick="navigateClip(1)">下一个 ⏭</button>
      </div>
    </div>

    <div class="video-grid">
      <div class="video-card">
        <div class="video-card-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span>📹 原监控画面</span>
            <button id="btnToggleOrigBoxes" class="filter-pill active" onclick="toggleOrigBoxes()">🔲 检测框: 开启</button>
          </div>
          <span id="origTag" style="color:var(--accent-sky); font-size:0.74rem;">Annotated</span>
        </div>
        <div class="video-wrapper">
          <video id="origVideo" playsinline preload="auto" controls></video>
          <div id="origOverlay" class="video-overlay">
            <div style="font-size:1.7rem;">📹</div>
            <div id="origOverlayTitle" style="font-weight:700;">原视频不可用</div>
            <div id="origOverlayDesc" style="font-size:0.78rem; color:var(--text-secondary); max-width:340px;"></div>
          </div>
        </div>
        <div class="video-card-footer" id="origFooter">真值区间: --</div>
      </div>

      <div class="video-card">
        <div class="video-card-header">
          <span>🎯 精准裁剪与 HUD 可视化</span>
          <span style="color:var(--accent-emerald); font-size:0.74rem;">Cropped &amp; HUD</span>
        </div>
        <div class="video-wrapper">
          <video id="cropVideo" playsinline preload="auto" controls></video>
          <div id="cropOverlay" class="video-overlay">
            <div style="font-size:1.7rem;">⚠️</div>
            <div id="cropOverlayTitle" style="font-weight:700; color:#f87171;">裁剪视频不可用</div>
            <div id="cropOverlayDesc" style="font-size:0.78rem; color:var(--text-secondary); max-width:340px;"></div>
          </div>
        </div>
        <div class="video-card-footer" id="cropFooter">裁剪区间: --</div>
      </div>
    </div>

    <div class="player-controls">
      <div class="timeline-bar" id="timelineBar" onclick="onTimelineClick(event)">
        <div class="timeline-gt" id="tlGtRange"></div>
        <div class="timeline-pred" id="tlPredRange"></div>
        <div class="timeline-progress" id="tlProgress"></div>
        <div class="timeline-scrubber" id="tlScrubber"></div>
      </div>
      <div class="control-row">
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <button class="btn btn-primary" id="btnPlay" onclick="togglePlay()">▶ 同步播放</button>
          <button class="btn" onclick="stepFrame(-1)">⏮ 前一帧</button>
          <button class="btn" onclick="stepFrame(1)">⏭ 后一帧</button>
          <button class="btn" onclick="restartVideos()">↺ 重播</button>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <span id="timeDisplay" style="font-size:0.8rem; color:var(--text-secondary);">0.00s / 0.00s (帧 0/0)</span>
          <select class="btn" id="speedSelect" onchange="changeSpeed(this.value)">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.50x</option>
            <option value="1.0" selected>1.00x</option>
            <option value="1.5">1.50x</option>
            <option value="2.0">2.00x</option>
          </select>
          <label style="font-size:0.78rem; color:var(--text-secondary); display:flex; align-items:center; gap:4px; cursor:pointer;">
            <input type="checkbox" id="chkLoop" checked> 循环
          </label>
        </div>
      </div>
    </div>

    <div class="telemetry-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <strong>📊 片段时序边界与目标个体诊断</strong>
        <span id="teleIouBadge" style="color:var(--accent-emerald); font-weight:700;">IoU: --</span>
      </div>
      <div class="telemetry-grid">
        <div class="tele-item">
          <div class="tele-label">行为类别 (预测 vs 真值)</div>
          <div class="tele-val" id="diagBehavior">--</div>
          <div class="tele-sub" id="diagBehaviorSub">--</div>
        </div>
        <div class="tele-item">
          <div class="tele-label">真值帧区间 (Ground Truth)</div>
          <div class="tele-val" id="diagGtRange">--</div>
          <div class="tele-sub" id="diagGtSec">--</div>
        </div>
        <div class="tele-item">
          <div class="tele-label">裁剪帧区间 (Cropped)</div>
          <div class="tele-val" id="diagPredRange">--</div>
          <div class="tele-sub" id="diagPredSec">--</div>
        </div>
        <div class="tele-item">
          <div class="tele-label">参与目标小鼠 ID 与角色</div>
          <div class="tele-val" id="diagMice">--</div>
          <div class="tele-sub" id="diagRoles">--</div>
        </div>
        <div class="tele-item">
          <div class="tele-label">起止帧边界误差</div>
          <div class="tele-val" id="diagError">--</div>
          <div class="tele-sub" id="diagErrorMs">--</div>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
const LOCAL_BUNDLE_DATA = {bundle_json};
const DEMO_FALLBACK = {fallback_json};

const IS_FILE_PROTOCOL = window.location.protocol === 'file:';
let currentData = DEMO_FALLBACK;
let activeFilter = 'all';
let currentClip = null;
let isPlaying = false;
let showOrigBoxes = true;
let usingDemoData = true;

const origVideo = document.getElementById('origVideo');
const cropVideo = document.getElementById('cropVideo');
const btnPlay = document.getElementById('btnPlay');
const timeDisplay = document.getElementById('timeDisplay');
const tlProgress = document.getElementById('tlProgress');
const tlScrubber = document.getElementById('tlScrubber');
const tlGtRange = document.getElementById('tlGtRange');
const tlPredRange = document.getElementById('tlPredRange');
const chkLoop = document.getElementById('chkLoop');

function escapeHtml(value) {{
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

function getBadgeClass(beh) {{
  if (beh === 'Dispersal') return 'clip-badge badge-dispersal';
  if (beh === 'Group locomotion') return 'clip-badge badge-gloco';
  if (beh === 'Social clustering') return 'clip-badge badge-clustering';
  return 'clip-badge';
}}

function getZhBehaviorName(beh) {{
  if (beh === 'Dispersal') return '散开 / 群体分散';
  if (beh === 'Group locomotion') return '群体同步运动';
  if (beh === 'Social clustering') return '社会聚集';
  return beh;
}}

document.addEventListener('DOMContentLoaded', () => {{
  const pillsEl = document.getElementById('filterPills');
  if (pillsEl) {{
    pillsEl.addEventListener('click', (e) => {{
      const pill = e.target.closest('.filter-pill');
      if (pill && pill.dataset.filter) {{
        activeFilter = pill.dataset.filter;
        pillsEl.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderClipList();
      }}
    }});
  }}

  if (IS_FILE_PROTOCOL) {{
    const hasBundle = (
      LOCAL_BUNDLE_DATA
      && !LOCAL_BUNDLE_DATA.is_fallback_demo
      && Array.isArray(LOCAL_BUNDLE_DATA.per_clip_results)
      && LOCAL_BUNDLE_DATA.per_clip_results.length > 0
    );
    const badge = document.getElementById('connBadge');
    const connText = document.getElementById('connText');
    const alertBanner = document.getElementById('alertBanner');
    if (hasBundle) {{
      badge.className = 'status-badge status-local';
      connText.textContent = `📁 本地离线静态模式 (${{LOCAL_BUNDLE_DATA.per_clip_results.length}} 段本地视频)`;
      alertBanner.style.display = 'none';
      loadDataset(LOCAL_BUNDLE_DATA, false);
    }} else {{
      badge.className = 'status-badge status-demo';
      connText.textContent = '▲ 离线演示模式 (仅供界面验证)';
      document.getElementById('alertMsg').textContent = '⚠️ 未检测到同目录离线数据包，当前展示演示数据。';
      alertBanner.style.display = 'flex';
      loadDataset(DEMO_FALLBACK, true);
    }}
    return;
  }}

  // Over HTTP: render shell immediately with labeled demo fallback while probing live API asynchronously
  loadDataset(DEMO_FALLBACK, true);
  fetchLiveEvaluation();
}});

async function fetchLiveEvaluation() {{
  if (IS_FILE_PROTOCOL) return;
  const badge = document.getElementById('connBadge');
  const connText = document.getElementById('connText');
  const alertBanner = document.getElementById('alertBanner');
  connText.textContent = '正在连接本地数据接口...';

  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), 8000);
  try {{
    const res = await fetch('/api/evaluation', {{ signal: controller.signal }});
    if (!res.ok) {{
      throw new Error(`HTTP ${{res.status}}`);
    }}
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('application/json')) {{
      throw new Error(`响应类型非 JSON (${{contentType || 'unknown'}})`);
    }}
    const data = await res.json();
    if (!data || !Array.isArray(data.per_clip_results) || data.per_clip_results.length === 0) {{
      throw new Error('接口返回的片段列表为空或结构不合法');
    }}
    if (data.is_fallback_demo) {{
      badge.className = 'status-badge status-demo';
      connText.textContent = '▲ 离线演示模式 (后端暂未生成实测报告)';
      document.getElementById('alertMsg').textContent = data.demo_warning || '当前展示为离线演示数据。';
      alertBanner.style.display = 'flex';
      loadDataset(data, true);
      return;
    }}
    badge.className = 'status-badge status-live';
    connText.textContent = `● 独立本地服务已连接 (${{data.per_clip_results.length}} 段视频)`;
    alertBanner.style.display = 'none';
    loadDataset(data, false);
  }} catch (err) {{
    const msg = (err && err.name === 'AbortError') ? '请求超时 (8s)' : ((err && err.message) ? err.message : String(err));
    badge.className = 'status-badge status-demo';
    connText.textContent = '▲ 本地服务未响应 (已切换演示数据)';
    document.getElementById('alertMsg').textContent = `⚠️ 无法获取实时后端数据 (${{msg}})。请双击目录中的 start_local_site.bat 或运行服务命令后重试。`;
    alertBanner.style.display = 'flex';
    loadDataset(DEMO_FALLBACK, true);
  }} finally {{
    clearTimeout(timerId);
  }}
}}

function retryFetchData() {{
  if (IS_FILE_PROTOCOL) {{
    window.location.reload();
    return;
  }}
  fetchLiveEvaluation();
}}

function loadDataset(dataset, isDemo) {{
  currentData = dataset;
  usingDemoData = Boolean(isDemo);
  updateMetrics(dataset, usingDemoData);
  renderRules(dataset);
  renderFilterPills(dataset);
  renderClipList();
  const clips = getFilteredClips();
  if (clips.length > 0) {{
    selectClip(clips[0].clip_id);
  }}
}}

function updateMetrics(data, isDemo) {{
  const clips = Array.isArray(data.per_clip_results) ? data.per_clip_results : [];
  const total = clips.length;
  const dsTotal = (typeof data.dataset_total_clips === 'number' && data.dataset_total_clips > 0) ? data.dataset_total_clips : null;

  const byBeh = (beh) => clips.filter(c => c.ground_truth_behavior === beh);
  const statsText = (beh) => {{
    const arr = byBeh(beh);
    const ok = arr.filter(c => c.is_correct).length;
    const acc = arr.length > 0 ? ((ok / arr.length) * 100).toFixed(2) + '%' : '--';
    const share = (dsTotal && !isDemo) ? ` · 占全集 ${{(arr.length / dsTotal * 100).toFixed(2)}}%` : '';
    return [acc, `${{ok}} / ${{arr.length}} 段正确${{share}}`];
  }};

  document.getElementById('kpiTotal').textContent = String(total);
  document.getElementById('kpiTotalSub').textContent = (dsTotal && !isDemo)
    ? `占 605 段总数据集 ${{(total / dsTotal * 100).toFixed(2)}}%`
    : (isDemo ? '演示样例集 (非实测)' : '全量真实集');

  const [dAcc, dSub] = statsText('Dispersal');
  document.getElementById('kpiDispersal').textContent = dAcc;
  document.getElementById('kpiDispersalSub').textContent = dSub;

  const [gAcc, gSub] = statsText('Group locomotion');
  document.getElementById('kpiGloco').textContent = gAcc;
  document.getElementById('kpiGlocoSub').textContent = gSub;

  const [sAcc, sSub] = statsText('Social clustering');
  document.getElementById('kpiClustering').textContent = sAcc;
  document.getElementById('kpiClusteringSub').textContent = sSub;

  const corr = clips.filter(c => c.is_correct);
  const meanIou = corr.length > 0 ? (corr.reduce((a, c) => a + (c.temporal_iou || 0), 0) / corr.length) : 0;
  const pass95 = corr.filter(c => (c.temporal_iou || 0) >= 0.95).length;
  document.getElementById('kpiMeanIoU').textContent = `${{(meanIou * 100).toFixed(2)}}%`;
  document.getElementById('kpiIouSub').textContent = `IoU ≥ 95%: ${{pass95}} / ${{corr.length}} 段`;
}}

function renderRules(data) {{
  const grid = document.getElementById('rulesGrid');
  if (!grid) return;
  const specs = Array.isArray(data.behavior_specs) ? data.behavior_specs : [];
  if (specs.length === 0) {{
    grid.innerHTML = '';
    return;
  }}
  grid.innerHTML = specs.map(spec => {{
    const cls = getBadgeClass(spec.behavior);
    const share = (typeof spec.dataset_share_pct === 'number' && spec.dataset_total_clips)
      ? `${{spec.clip_count}} / ${{spec.dataset_total_clips}} (${{spec.dataset_share_pct.toFixed(2)}}%)`
      : `${{spec.clip_count || 0}} 段`;
    const rules = Array.isArray(spec.rules) ? spec.rules : [];
    return `
      <div class="rule-card">
        <h3>
          <span>${{escapeHtml(spec.zh_name || spec.behavior)}} (${{escapeHtml(spec.behavior)}})</span>
          <span class="${{cls}}">${{escapeHtml(share)}}</span>
        </h3>
        <ul>${{rules.map(r => `<li>${{escapeHtml(r)}}</li>`).join('')}}</ul>
      </div>
    `;
  }}).join('');
}}

function renderFilterPills(data) {{
  const container = document.getElementById('filterPills');
  const clips = Array.isArray(data.per_clip_results) ? data.per_clip_results : [];
  const counts = {{}};
  clips.forEach(c => {{
    const b = c.ground_truth_behavior || 'Unknown';
    counts[b] = (counts[b] || 0) + 1;
  }});
  const iou95 = clips.filter(c => c.is_correct && (c.temporal_iou || 0) >= 0.95).length;
  let html = `<button class="filter-pill ${{activeFilter === 'all' ? 'active' : ''}}" data-filter="all">全部 (${{clips.length}})</button>`;
  for (const beh of ['Dispersal', 'Group locomotion', 'Social clustering']) {{
    if (counts[beh]) {{
      const fk = 'beh:' + beh;
      html += `<button class="filter-pill ${{activeFilter === fk ? 'active' : ''}}" data-filter="${{escapeHtml(fk)}}">${{escapeHtml(beh)}} (${{counts[beh]}})</button>`;
    }}
  }}
  html += `<button class="filter-pill ${{activeFilter === 'iou95' ? 'active' : ''}}" data-filter="iou95">IoU≥95% (${{iou95}})</button>`;
  container.innerHTML = html;
}}

function getFilteredClips() {{
  const clips = Array.isArray(currentData && currentData.per_clip_results) ? currentData.per_clip_results : [];
  const query = (document.getElementById('searchInput').value || '').toLowerCase().trim();
  return clips.filter(c => {{
    if (activeFilter.startsWith('beh:')) {{
      if (c.ground_truth_behavior !== activeFilter.slice(4)) return false;
    }} else if (activeFilter === 'iou95') {{
      if (!c.is_correct || (c.temporal_iou || 0) < 0.95) return false;
    }}
    if (query) {{
      const idHit = (c.clip_id || '').toLowerCase().includes(query);
      const behHit = (c.ground_truth_behavior || '').toLowerCase().includes(query);
      const miceHit = (c.target_mouse_ids || []).some(m => String(m).includes(query));
      if (!idHit && !behHit && !miceHit) return false;
    }}
    return true;
  }});
}}

function renderClipList() {{
  const listEl = document.getElementById('clipList');
  const clips = getFilteredClips();
  const total = (currentData && Array.isArray(currentData.per_clip_results)) ? currentData.per_clip_results.length : 0;
  document.getElementById('clipCountText').textContent = `显示 ${{clips.length}} / ${{total}} 段`;
  listEl.innerHTML = '';
  if (clips.length === 0) {{
    listEl.innerHTML = '<div style="padding:20px; text-align:center; color:var(--text-muted);">未匹配到视频片段</div>';
    return;
  }}
  clips.forEach(c => {{
    const item = document.createElement('div');
    item.className = 'clip-item' + (currentClip && currentClip.clip_id === c.clip_id ? ' selected' : '');
    item.dataset.clipId = c.clip_id;
    item.onclick = () => selectClip(c.clip_id);
    const iouPct = ((c.temporal_iou || 0) * 100).toFixed(1);
    const dur = ((c.total_frames || 0) / (c.fps || 30)).toFixed(1);
    item.innerHTML = `
      <div class="clip-item-top">
        <span class="${{getBadgeClass(c.ground_truth_behavior)}}">${{escapeHtml(c.ground_truth_behavior)}}</span>
        <span style="font-size:0.74rem; font-weight:700; color:var(--accent-emerald);">IoU: ${{iouPct}}%</span>
      </div>
      <div class="clip-id">${{escapeHtml(c.clip_id)}}</div>
      <div class="clip-meta">小鼠: [${{escapeHtml((c.target_mouse_ids || []).join(', '))}}] · ${{c.total_frames || 0}} 帧 (${{dur}}s)</div>
    `;
    listEl.appendChild(item);
  }});
}}

function _videoUrl(clip, kind) {{
  const cid = encodeURIComponent(clip.clip_id);
  const rel = clip.video_urls && clip.video_urls[kind];
  if (IS_FILE_PROTOCOL) {{
    return rel ? `./${{rel}}` : `./videos/${{cid}}/${{kind}}.mp4`;
  }}
  return rel ? `/${{rel}}` : `/videos/${{cid}}/${{kind}}.mp4`;
}}

function _selectedSpeed() {{
  const v = parseFloat(document.getElementById('speedSelect').value || '1.0');
  return Number.isFinite(v) && v > 0 ? v : 1.0;
}}

function selectClip(clipId, scrollIntoView = false) {{
  const clips = Array.isArray(currentData && currentData.per_clip_results) ? currentData.per_clip_results : [];
  const clip = clips.find(c => c.clip_id === clipId);
  if (!clip) return;
  currentClip = clip;

  let activeEl = null;
  document.querySelectorAll('.clip-item').forEach(el => {{
    const hit = el.dataset.clipId === clipId;
    el.classList.toggle('selected', hit);
    if (hit) activeEl = el;
  }});
  if (scrollIntoView && activeEl) {{
    activeEl.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
  }}

  const fps = Number(clip.fps || 30.0);
  document.getElementById('stageTitle').textContent = `${{getZhBehaviorName(clip.ground_truth_behavior)}} · ${{clip.clip_id}}`;
  document.getElementById('stageSubtitle').textContent = `参与小鼠: [${{(clip.target_mouse_ids || []).join(', ')}}] · ${{clip.total_frames}} 帧 (${{(clip.total_frames / fps).toFixed(2)}}s @ ${{fps.toFixed(2)}}fps)`;

  const iouPct = ((clip.temporal_iou || 0) * 100).toFixed(1);
  document.getElementById('stageTags').innerHTML = `
    <span class="${{getBadgeClass(clip.ground_truth_behavior)}}" style="padding:4px 8px; font-size:0.8rem;">GT: ${{escapeHtml(clip.ground_truth_behavior)}}</span>
    <span class="clip-badge" style="padding:4px 8px; font-size:0.8rem; background:rgba(16,185,129,0.18); color:#34d399; border:1px solid #10b981;">${{clip.is_correct ? '预测命中' : '未命中'}}: ${{escapeHtml(clip.predicted_behavior || 'None')}}</span>
    <span class="clip-badge" style="padding:4px 8px; font-size:0.8rem; background:rgba(56,189,248,0.18); color:#38bdf8; border:1px solid #38bdf8;">IoU: ${{iouPct}}%</span>
  `;

  pauseVideos();
  origVideo.currentTime = 0;
  cropVideo.currentTime = 0;
  updateOrigVideoSource(false);

  const cropOverlay = document.getElementById('cropOverlay');
  if (usingDemoData && !IS_FILE_PROTOCOL) {{
    cropOverlay.style.display = 'flex';
    document.getElementById('cropOverlayTitle').textContent = '演示模式无后台视频流';
    document.getElementById('cropOverlayDesc').textContent = '请启动本地服务 (python scripts/serve_group_behaviors_site.py --port 8780) 或直接打开静态页面以播放裁剪视频。';
    cropVideo.style.display = 'none';
    cropVideo.removeAttribute('src');
  }} else if (clip.is_correct) {{
    cropOverlay.style.display = 'none';
    cropVideo.style.display = 'block';
    const spd = _selectedSpeed();
    cropVideo.defaultPlaybackRate = spd;
    cropVideo.playbackRate = spd;
    cropVideo.src = _videoUrl(clip, 'cropped');
  }} else {{
    cropOverlay.style.display = 'flex';
    document.getElementById('cropOverlayTitle').textContent = '未生成裁剪视频';
    document.getElementById('cropOverlayDesc').textContent = '当前片段行为未命中真值。';
    cropVideo.style.display = 'none';
    cropVideo.removeAttribute('src');
  }}

  cropVideo.onerror = () => {{
    cropOverlay.style.display = 'flex';
    document.getElementById('cropOverlayTitle').textContent = '裁剪视频加载失败';
    document.getElementById('cropOverlayDesc').textContent = IS_FILE_PROTOCOL
      ? '未找到同目录 videos/ 下的裁剪 MP4 文件，请重新运行构建脚本生成独立网站包。'
      : '本地视频流断开或未启动服务，请双击 start_local_site.bat 重启本地网站后刷新。';
    cropVideo.style.display = 'none';
  }};

  const gtS = clip.ground_truth_range_half_open ? clip.ground_truth_range_half_open.start : 0;
  const gtE = clip.ground_truth_range_half_open ? clip.ground_truth_range_half_open.end : clip.total_frames;
  const predS = clip.predicted_range_half_open ? clip.predicted_range_half_open.start : 0;
  const predE = clip.predicted_range_half_open ? clip.predicted_range_half_open.end : 0;
  document.getElementById('origFooter').textContent = `标注真值区间: [${{gtS}}, ${{gtE}}) · ${{( (gtE - gtS) / fps ).toFixed(2)}}s`;
  document.getElementById('cropFooter').textContent = `算法裁剪区间: [${{predS}}, ${{predE}}) · 重叠率 ${{iouPct}}%`;

  document.getElementById('teleIouBadge').textContent = `IoU: ${{iouPct}}%`;
  document.getElementById('diagBehavior').textContent = `${{clip.predicted_behavior || 'None'}} (${{clip.is_correct ? '正确' : '差异'}})`;
  document.getElementById('diagBehaviorSub').textContent = `真值: ${{clip.ground_truth_behavior}} (${{getZhBehaviorName(clip.ground_truth_behavior)}})`;
  document.getElementById('diagGtRange').textContent = `[${{gtS}}, ${{gtE}})`;
  document.getElementById('diagGtSec').textContent = `${{gtE - gtS}} 帧 (${{((gtE - gtS) / fps).toFixed(2)}}s)`;
  document.getElementById('diagPredRange').textContent = `[${{predS}}, ${{predE}})`;
  document.getElementById('diagPredSec').textContent = `${{predE - predS}} 帧 (${{((predE - predS) / fps).toFixed(2)}}s)`;
  document.getElementById('diagMice').textContent = `ID: ${{(clip.target_mouse_ids || []).join(', ')}}`;
  const roles = clip.semantic_roles ? Object.entries(clip.semantic_roles).map(([k, v]) => `${{k}}: [${{v.join(', ')}}]`).join('; ') : '';
  document.getElementById('diagRoles').textContent = roles || '群体成员协同';
  const errS = Math.abs(predS - gtS);
  const errE = Math.abs(predE - gtE);
  document.getElementById('diagError').textContent = `起: ${{errS}} 帧 / 止: ${{errE}} 帧`;
  document.getElementById('diagErrorMs').textContent = `起: ${{(errS / fps).toFixed(2)}}s / 止: ${{(errE / fps).toFixed(2)}}s`;

  const tot = clip.total_frames || 1;
  tlGtRange.style.left = `${{(gtS / tot) * 100}}%`;
  tlGtRange.style.width = `${{((gtE - gtS) / tot) * 100}}%`;
  tlPredRange.style.left = `${{(predS / tot) * 100}}%`;
  tlPredRange.style.width = `${{((predE - predS) / tot) * 100}}%`;
  updateTimeUI(0);
}}

function toggleOrigBoxes() {{
  showOrigBoxes = !showOrigBoxes;
  const btn = document.getElementById('btnToggleOrigBoxes');
  const tag = document.getElementById('origTag');
  btn.classList.toggle('active', showOrigBoxes);
  btn.textContent = showOrigBoxes ? '🔲 检测框: 开启' : '🔲 检测框: 关闭';
  tag.textContent = showOrigBoxes ? 'Annotated' : 'Raw Source';
  updateOrigVideoSource(true);
}}

function updateOrigVideoSource(preserveTime = false) {{
  if (!currentClip) return;
  const origOverlay = document.getElementById('origOverlay');
  if (usingDemoData && !IS_FILE_PROTOCOL) {{
    origOverlay.style.display = 'flex';
    document.getElementById('origOverlayTitle').textContent = '演示模式无后台视频流';
    document.getElementById('origOverlayDesc').textContent = '请启动独立本地网站服务后刷新页面，或直接打开离线包 index.html。';
    origVideo.style.display = 'none';
    origVideo.removeAttribute('src');
    return;
  }}
  const curTime = preserveTime ? (origVideo.currentTime || 0) : 0;
  const wasPlaying = preserveTime && isPlaying && !origVideo.paused;
  const spd = _selectedSpeed();
  origOverlay.style.display = 'none';
  origVideo.style.display = 'block';
  origVideo.defaultPlaybackRate = spd;
  origVideo.playbackRate = spd;
  origVideo.src = _videoUrl(currentClip, showOrigBoxes ? 'annotated' : 'raw');
  origVideo.onerror = () => {{
    origOverlay.style.display = 'flex';
    document.getElementById('origOverlayTitle').textContent = '原视频加载失败';
    document.getElementById('origOverlayDesc').textContent = IS_FILE_PROTOCOL
      ? '未找到本地视频文件，请确认 outputs/group_behaviors_website/videos/ 完整。'
      : '本地视频流断开或未启动服务，请双击 start_local_site.bat 重启本地网站后刷新。';
    origVideo.style.display = 'none';
  }};
  origVideo.addEventListener('loadedmetadata', () => {{
    origOverlay.style.display = 'none';
    origVideo.style.display = 'block';
    origVideo.playbackRate = _selectedSpeed();
    if (curTime > 0 && Number.isFinite(origVideo.duration)) {{
      origVideo.currentTime = Math.min(curTime, Math.max(0, origVideo.duration - 0.01));
    }}
    updateTimeUI(origVideo.currentTime || 0);
    if (wasPlaying) origVideo.play().catch(() => {{}});
  }}, {{ once: true }});
}}

function toggleMetricsBar() {{
  document.getElementById('metricsBar').classList.toggle('collapsed');
}}

function toggleRulesDrawer() {{
  document.getElementById('rulesDrawer').classList.toggle('open');
}}

function navigateClip(dir) {{
  const clips = getFilteredClips();
  if (clips.length === 0) return;
  let idx = clips.findIndex(c => c.clip_id === (currentClip ? currentClip.clip_id : ''));
  idx = idx === -1 ? 0 : (idx + dir + clips.length) % clips.length;
  selectClip(clips[idx].clip_id, true);
}}

function _shouldPlayCropAt(curSec) {{
  if (!currentClip || !currentClip.is_correct) return false;
  const predS = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.start) || 0;
  const predE = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.end) || currentClip.total_frames || 0;
  const fps = currentClip.fps || 30.0;
  return curSec >= predS / fps && (predE <= predS || curSec < predE / fps);
}}

function togglePlay() {{
  if (isPlaying) pauseVideos(); else playVideos();
}}

function playVideos() {{
  isPlaying = true;
  btnPlay.textContent = '⏸ 同步暂停';
  btnPlay.classList.remove('btn-primary');
  if (origVideo.src) origVideo.play().catch(() => {{}});
  if (cropVideo.style.display !== 'none' && cropVideo.src && _shouldPlayCropAt(origVideo.currentTime || 0)) {{
    cropVideo.play().catch(() => {{}});
  }}
}}

function pauseVideos() {{
  isPlaying = false;
  btnPlay.textContent = '▶ 同步播放';
  btnPlay.classList.add('btn-primary');
  origVideo.pause();
  cropVideo.pause();
}}

function restartVideos() {{
  origVideo.currentTime = 0;
  cropVideo.currentTime = 0;
  playVideos();
}}

function stepFrame(dir) {{
  pauseVideos();
  const fps = (currentClip && currentClip.fps) ? currentClip.fps : 30.0;
  origVideo.currentTime = Math.max(0, origVideo.currentTime + dir / fps);
  if (cropVideo.style.display !== 'none' && cropVideo.src) {{
    const predS = (currentClip && currentClip.predicted_range_half_open) ? currentClip.predicted_range_half_open.start : 0;
    cropVideo.currentTime = Math.max(0, origVideo.currentTime - (predS / fps));
  }}
}}

function changeSpeed(val) {{
  const spd = parseFloat(val);
  origVideo.playbackRate = spd;
  cropVideo.playbackRate = spd;
}}

origVideo.addEventListener('play', () => {{
  isPlaying = true;
  btnPlay.textContent = '⏸ 同步暂停';
  btnPlay.classList.remove('btn-primary');
  if (cropVideo.style.display !== 'none' && cropVideo.src && cropVideo.paused && _shouldPlayCropAt(origVideo.currentTime || 0)) {{
    cropVideo.play().catch(() => {{}});
  }}
}});

origVideo.addEventListener('pause', () => {{
  if (origVideo.currentTime >= origVideo.duration) return;
  isPlaying = false;
  btnPlay.textContent = '▶ 同步播放';
  btnPlay.classList.add('btn-primary');
  if (!cropVideo.paused) cropVideo.pause();
}});

origVideo.addEventListener('timeupdate', () => {{
  const cur = origVideo.currentTime;
  updateTimeUI(cur);
  if (cropVideo.style.display !== 'none' && cropVideo.src && currentClip && currentClip.is_correct) {{
    const predS = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.start) || 0;
    const predE = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.end) || currentClip.total_frames || 0;
    const fps = currentClip.fps || 30.0;
    const offsetSec = predS / fps;
    const endSec = predE / fps;
    if (cur < offsetSec || (predE > predS && cur >= endSec)) {{
      if (!cropVideo.paused) cropVideo.pause();
    }} else {{
      const targetCrop = Math.max(0, cur - offsetSec);
      if (isPlaying && cropVideo.paused && !origVideo.paused) cropVideo.play().catch(() => {{}});
      if (Math.abs(cropVideo.currentTime - targetCrop) > 0.08) cropVideo.currentTime = targetCrop;
    }}
  }}
}});

origVideo.addEventListener('ended', () => {{
  if (chkLoop.checked) restartVideos(); else pauseVideos();
}});

function updateTimeUI(curSec) {{
  const dur = origVideo.duration || (currentClip ? (currentClip.total_frames / (currentClip.fps || 30)) : 1);
  const fps = (currentClip && currentClip.fps) ? currentClip.fps : 30.0;
  const curFrame = Math.round(curSec * fps);
  const totFrame = currentClip ? currentClip.total_frames : Math.round(dur * fps);
  timeDisplay.textContent = `${{curSec.toFixed(2)}}s / ${{dur.toFixed(2)}}s (帧 ${{curFrame}}/${{totFrame}})`;
  const pct = Math.min(100, Math.max(0, (curSec / dur) * 100));
  tlProgress.style.width = `${{pct}}%`;
  tlScrubber.style.left = `${{pct}}%`;
}}

function onTimelineClick(e) {{
  const rect = document.getElementById('timelineBar').getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  const dur = origVideo.duration || (currentClip ? (currentClip.total_frames / (currentClip.fps || 30)) : 1);
  const targetTime = ratio * dur;
  origVideo.currentTime = targetTime;
  if (cropVideo.style.display !== 'none' && cropVideo.src && currentClip && currentClip.is_correct) {{
    const predS = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.start) || 0;
    const fps = currentClip.fps || 30.0;
    cropVideo.currentTime = Math.max(0, targetTime - (predS / fps));
  }}
  updateTimeUI(targetTime);
}}

document.addEventListener('keydown', (e) => {{
  if (document.activeElement === document.getElementById('searchInput')) {{
    if (e.key === 'Escape') document.getElementById('searchInput').blur();
    return;
  }}
  if (e.key === 'ArrowDown') {{ e.preventDefault(); navigateClip(1); }}
  else if (e.key === 'ArrowUp') {{ e.preventDefault(); navigateClip(-1); }}
  else if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
  else if (e.key === '[') {{ e.preventDefault(); stepFrame(-1); }}
  else if (e.key === ']') {{ e.preventDefault(); stepFrame(1); }}
}});
</script>
</body>
</html>"""


_STANDALONE_SERVER_TEMPLATE = '''"""Standalone local HTTP server bundled inside outputs/group_behaviors_website.

Usage:
    python server.py --port 8780 --open-browser
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE_DIR = Path(__file__).resolve().parent
EVAL_JSON = SITE_DIR / "data" / "evaluation.json"
INDEX_HTML = SITE_DIR / "index.html"
VIDEOS_DIR = SITE_DIR / "videos"


class StandaloneSiteHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _host_ok(self) -> bool:
        ip = getattr(self, "client_address", ("", 0))[0]
        if ip in ("127.0.0.1", "::1", "localhost"):
            return True
        host = self.headers.get("Host", "").split(":")[0].strip()
        return host in ("127.0.0.1", "localhost")

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _stream_file(self, path: Path, mime_override: str | None = None, head_only: bool = False) -> None:
        if not path.is_file():
            self._send_json(404, {"error": "File not found"})
            return
        size = path.stat().st_size
        mime = mime_override or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_hdr = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if range_hdr:
            m = re.fullmatch(r"bytes=(\\d*)-(\\d*)", range_hdr.strip())
            if m:
                s_str, e_str = m.groups()
                if s_str:
                    start = int(s_str)
                    end = min(int(e_str), size - 1) if e_str else size - 1
                elif e_str:
                    start = max(0, size - int(e_str))
                if 0 <= start <= end < size:
                    status = 206
                else:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                rem = end - start + 1
                while rem > 0:
                    chunk = fh.read(min(256 * 1024, rem))
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
        path = unquote(urlsplit(self.path).path).strip("/")
        if path in ("", "index.html"):
            self._stream_file(INDEX_HTML, "text/html; charset=utf-8", head_only)
            return
        if path == "api/health":
            self._send_json(200, {"status": "ok", "evaluation_file_exists": EVAL_JSON.is_file(), "videos_dir_exists": VIDEOS_DIR.is_dir()})
            return
        if path in ("api/evaluation", "data/evaluation.json"):
            if not EVAL_JSON.is_file():
                self._send_json(404, {"error": "evaluation.json not found"})
                return
            try:
                data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
            except Exception as exc:
                self._send_json(500, {"error": f"Failed to parse evaluation JSON: {exc}"})
                return
            self._send_json(200, data)
            return
        m = re.fullmatch(r"videos/([A-Za-z0-9_\\-]+)/(raw|annotated|cropped)\\.mp4", path)
        if m:
            cid, kind = m.groups()
            self._stream_file(VIDEOS_DIR / cid / f"{kind}.mp4", "video/mp4", head_only)
            return
        self._send_json(404, {"error": "Not found"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone group behaviors local website server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}/"
    print(f"Group behaviors local site listening on {url}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), StandaloneSiteHandler)
    server.daemon_threads = True
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_standalone_group_website(
    evaluation_file: Path = DEFAULT_GROUP_EVAL_JSON,
    site_dir: Path = DEFAULT_SITE_DIR,
    *,
    copy_videos: bool = True,
) -> dict[str, Any]:
    """Build a standalone, self-contained local website directory for the three group behaviors."""
    eval_path = Path(evaluation_file)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Group behavior evaluation JSON not found: {eval_path}")

    raw_data = json.loads(eval_path.read_text(encoding="utf-8"))
    clips = raw_data.get("per_clip_results")
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"Evaluation JSON contains no per_clip_results: {eval_path}")

    out_dir = Path(site_dir)
    data_dir = out_dir / "data"
    videos_dir = out_dir / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    bundled_clips: list[dict[str, Any]] = []
    copied_counts = {"raw": 0, "annotated": 0, "cropped": 0}

    for item in clips:
        clip = dict(item)
        cid = str(clip["clip_id"])
        clip_video_dir = videos_dir / cid
        clip_video_dir.mkdir(parents=True, exist_ok=True)

        clip_dir_val = clip.get("clip_dir")
        raw_src = Path(clip_dir_val) / "clip.mp4" if clip_dir_val else None
        ann_src = Path(clip["annotated_orig_path"]) if clip.get("annotated_orig_path") else None
        crp_src = Path(clip["cropped_video_path"]) if clip.get("cropped_video_path") else None

        if copy_videos:
            if raw_src and _copy_if_needed(raw_src, clip_video_dir / "raw.mp4"):
                copied_counts["raw"] += 1
            if ann_src and _copy_if_needed(ann_src, clip_video_dir / "annotated.mp4"):
                copied_counts["annotated"] += 1
            if crp_src and _copy_if_needed(crp_src, clip_video_dir / "cropped.mp4"):
                copied_counts["cropped"] += 1

        clip["video_urls"] = {
            "raw": f"videos/{cid}/raw.mp4",
            "annotated": f"videos/{cid}/annotated.mp4",
            "cropped": f"videos/{cid}/cropped.mp4",
        }
        bundled_clips.append(clip)

    bundled_payload = {
        **raw_data,
        "is_fallback_demo": False,
        "standalone_site_dir": str(out_dir),
        "per_clip_results": bundled_clips,
    }

    (data_dir / "evaluation.json").write_text(
        json.dumps(bundled_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(
        render_group_behaviors_page(bundle_data=bundled_payload, fallback_data=DEMO_FALLBACK_GROUP_DATA),
        encoding="utf-8",
    )
    (out_dir / "server.py").write_text(_STANDALONE_SERVER_TEMPLATE, encoding="utf-8")
    (out_dir / "start_local_site.bat").write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "echo Starting standalone mouse group behaviors website at http://127.0.0.1:8780/\r\n"
        "python server.py --port 8780 --open-browser\r\n"
        "pause\r\n",
        encoding="ascii",
    )
    (out_dir / "README_使用说明.txt").write_text(
        "三种群体行为独立本地网站说明\n"
        "========================================\n"
        "本目录为完全独立的静态+本地服务网站包，包含 Dispersal (35段)、Group locomotion (8段)、Social clustering (7段) 共 50 段视频。\n\n"
        "使用方式（二选一）：\n"
        "1. 推荐：双击 start_local_site.bat（或在本目录运行 python server.py --port 8780 --open-browser），会自动打开 http://127.0.0.1:8780/。\n"
        "2. 免服务直接打开：直接用 Chrome / Edge 浏览器双击打开本目录下的 index.html，也会通过相对路径加载 ./videos/ 下的全部本地视频。\n",
        encoding="utf-8",
    )

    return {
        "site_dir": str(out_dir),
        "total_clips": len(bundled_clips),
        "copied_videos": copied_counts,
    }


class GroupBehaviorsSiteHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dedicated standalone group-behaviors local website."""

    site_dir: Path = DEFAULT_SITE_DIR
    evaluation_file: Path = DEFAULT_SITE_DIR / "data" / "evaluation.json"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _host_ok(self) -> bool:
        ip = getattr(self, "client_address", ("", 0))[0]
        if ip in ("127.0.0.1", "::1", "localhost", "testclient"):
            return True
        host = self.headers.get("Host", "").split(":")[0].strip()
        return host in ("127.0.0.1", "localhost")

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _stream_file(self, file_path: Path, mime_override: str | None = None, head_only: bool = False) -> None:
        if not file_path.is_file():
            self._send_json(404, {"error": "File not found", "path": str(file_path)})
            return
        size = file_path.stat().st_size
        mime_type = mime_override or mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                s_str, e_str = match.groups()
                if s_str:
                    start = int(s_str)
                    end = min(int(e_str), size - 1) if e_str else size - 1
                elif e_str:
                    start = max(0, size - int(e_str))
                if 0 <= start <= end < size:
                    status = 206
                else:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if head_only:
            return
        try:
            with file_path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_HEAD(self) -> None:
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False) -> None:
        if not self._host_ok():
            self._send_json(403, {"error": "Loopback access only"})
            return

        path = unquote(urlsplit(self.path).path).strip("/")

        if path in ("", "index.html"):
            index_file = self.site_dir / "index.html"
            if index_file.is_file():
                self._stream_file(index_file, "text/html; charset=utf-8", head_only)
                return
            html_body = render_group_behaviors_page(
                bundle_data=None,
                fallback_data=DEMO_FALLBACK_GROUP_DATA,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not head_only:
                self.wfile.write(html_body)
            return

        if path == "api/health":
            self._send_json(200, {
                "status": "ok",
                "site_dir_exists": self.site_dir.is_dir(),
                "evaluation_file_exists": self.evaluation_file.is_file(),
                "videos_dir_exists": (self.site_dir / "videos").is_dir(),
            })
            return

        if path in ("api/evaluation", "data/evaluation.json"):
            if self.evaluation_file.is_file():
                try:
                    data = json.loads(self.evaluation_file.read_text(encoding="utf-8"))
                    self._send_json(200, data)
                    return
                except Exception as exc:
                    self._send_json(500, {"error": f"Failed to parse evaluation JSON: {exc}"})
                    return
            self._send_json(200, DEMO_FALLBACK_GROUP_DATA)
            return

        match_vid = re.fullmatch(r"videos/([A-Za-z0-9_\-]+)/(raw|annotated|cropped)\.mp4", path)
        if match_vid:
            clip_id, kind = match_vid.groups()
            target = (self.site_dir / "videos" / clip_id / f"{kind}.mp4").resolve()
            videos_root = (self.site_dir / "videos").resolve()
            if not target.is_relative_to(videos_root):
                self._send_json(400, {"error": "Invalid video path"})
                return
            self._stream_file(target, "video/mp4", head_only)
            return

        self._send_json(404, {"error": "Not Found", "route": path})


def create_group_behaviors_server(
    site_dir: Path = DEFAULT_SITE_DIR,
    evaluation_file: Path | None = None,
    port: int = DEFAULT_SITE_PORT,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """Create configured ThreadingHTTPServer for the standalone group-behaviors website."""
    resolved_site = Path(site_dir)
    resolved_eval = Path(evaluation_file) if evaluation_file is not None else (resolved_site / "data" / "evaluation.json")

    class ConfiguredHandler(GroupBehaviorsSiteHandler):
        pass

    ConfiguredHandler.site_dir = resolved_site
    ConfiguredHandler.evaluation_file = resolved_eval

    server = ThreadingHTTPServer((host, port), ConfiguredHandler)
    server.daemon_threads = True
    return server
