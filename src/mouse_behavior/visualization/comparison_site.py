"""Comparison website generator and server for behavior recognition and cropping."""
from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

DEFAULT_ATTACK_DIR = Path(r"F:\补全行为素材429_261107\四行为训练素材429\dataset\Attack")
DEFAULT_CHASING_DIR = Path(r"F:\补全行为素材429_261107\四行为训练素材429\dataset\Chasing")
DEFAULT_DATASET_DIR = Path(r"E:\鼠行为数据集_人工清洗完成_605段_20261024\dataset")
DEFAULT_ISOLATION_DIR = DEFAULT_DATASET_DIR / "Isolation"
DEFAULT_GROUP_LOCOMOTION_DIR = DEFAULT_DATASET_DIR / "Group locomotion"
DEFAULT_STATIC_DIR = DEFAULT_DATASET_DIR / "Static"
DEFAULT_LOCOMOTION_DIR = DEFAULT_DATASET_DIR / "Locomotion"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUTPUTS_DIR = _REPO_ROOT / "outputs"
DEFAULT_CROPS_DIR = _OUTPUTS_DIR / "visualized_cropped_videos"
DEFAULT_ORIG_ANNOTATED_DIR = _OUTPUTS_DIR / "visualized_original_videos"
DEFAULT_COMBINED_EVALUATION_JSON = _OUTPUTS_DIR / "combined_behavior_visualized_evaluation.json"
DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON = _OUTPUTS_DIR / "static_locomotion_evaluation.json"
DEFAULT_ISO_GLOCO_EVALUATION_JSON = _OUTPUTS_DIR / "isolation_group_locomotion_evaluation.json"
DEFAULT_ATTACK_CHASING_EVALUATION_JSON = _OUTPUTS_DIR / "missing_id_visualized_evaluation.json"
DEFAULT_EVALUATION_JSON = (
    DEFAULT_COMBINED_EVALUATION_JSON
    if DEFAULT_COMBINED_EVALUATION_JSON.is_file()
    else (
        DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON
        if DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON.is_file()
        else (
            DEFAULT_ISO_GLOCO_EVALUATION_JSON
            if DEFAULT_ISO_GLOCO_EVALUATION_JSON.is_file()
            else DEFAULT_ATTACK_CHASING_EVALUATION_JSON
        )
    )
)

# Curated offline fallback demo data embedded directly in page
DEMO_FALLBACK_DATA = {
    "is_fallback_demo": True,
    "demo_warning": "当前展示为内置离线演示数据（仅用于界面交互验证，非后端实时科研结果）",
    "total_clips": 6,
    "attack_clips": 3,
    "chasing_clips": 3,
    "attack_accuracy": 1.0,
    "chasing_accuracy": 1.0,
    "overall_accuracy": 1.0,
    "attack_mean_iou": 1.0,
    "chasing_mean_iou": 1.0,
    "overall_mean_iou": 1.0,
    "iou_ge_05_rate": 1.0,
    "per_clip_results": [
        {
            "clip_id": "02c9f31a225e8c87678601f3",
            "ground_truth_behavior": "Attack",
            "predicted_behavior": "Attack",
            "is_correct": True,
            "target_mouse_ids": [1690, 1691],
            "semantic_roles": {"Attacker": [1691], "Attacked": [1690]},
            "ground_truth_range_half_open": {"start": 0, "end": 43},
            "predicted_range_half_open": {"start": 0, "end": 42},
            "temporal_iou": 1.0,
            "iou_ge_05": True,
            "attack_frames": 40,
            "chasing_frames": 0,
            "total_frames": 43,
            "fps": 30.0,
        },
        {
            "clip_id": "0aaaecb4454e028da01c5853",
            "ground_truth_behavior": "Attack",
            "predicted_behavior": "Attack",
            "is_correct": True,
            "target_mouse_ids": [1535, 1624],
            "semantic_roles": {"Attacker": [1624], "Attacked": [1535]},
            "ground_truth_range_half_open": {"start": 0, "end": 38},
            "predicted_range_half_open": {"start": 0, "end": 38},
            "temporal_iou": 1.0,
            "iou_ge_05": True,
            "attack_frames": 35,
            "chasing_frames": 0,
            "total_frames": 38,
            "fps": 30.0,
        },
        {
            "clip_id": "0aec86af99fa79968f7a352e",
            "ground_truth_behavior": "Attack",
            "predicted_behavior": "Attack",
            "is_correct": True,
            "target_mouse_ids": [1535, 1546],
            "semantic_roles": {"Attacker": [1546], "Attacked": [1535]},
            "ground_truth_range_half_open": {"start": 0, "end": 27},
            "predicted_range_half_open": {"start": 0, "end": 27},
            "temporal_iou": 1.0,
            "iou_ge_05": True,
            "attack_frames": 24,
            "chasing_frames": 0,
            "total_frames": 27,
            "fps": 30.0,
        },
        {
            "clip_id": "1750d23511065ca44967b61d",
            "ground_truth_behavior": "Chasing",
            "predicted_behavior": "Chasing",
            "is_correct": True,
            "target_mouse_ids": [5, 309],
            "semantic_roles": {"Chasing": [5], "Chased": [309]},
            "ground_truth_range_half_open": {"start": 0, "end": 52},
            "predicted_range_half_open": {"start": 0, "end": 52},
            "temporal_iou": 1.0,
            "iou_ge_05": True,
            "attack_frames": 0,
            "chasing_frames": 50,
            "total_frames": 52,
            "fps": 30.0,
        },
        {
            "clip_id": "1b8623b4b834a58640211406",
            "ground_truth_behavior": "Chasing",
            "predicted_behavior": "Chasing",
            "is_correct": True,
            "target_mouse_ids": [3785, 4032],
            "semantic_roles": {"Chasing": [4032], "Chased": [3785]},
            "ground_truth_range_half_open": {"start": 0, "end": 64},
            "predicted_range_half_open": {"start": 0, "end": 64},
            "temporal_iou": 1.0,
            "iou_ge_05": True,
            "attack_frames": 0,
            "chasing_frames": 62,
            "total_frames": 64,
            "fps": 30.0,
        },
        {
            "clip_id": "38c55d4fe73a46ecc22b6757",
            "ground_truth_behavior": "Chasing",
            "predicted_behavior": "Chasing",
            "is_correct": True,
            "target_mouse_ids": [384, 520],
            "semantic_roles": {"Chasing": [384], "Chased": [520]},
            "ground_truth_range_half_open": {"start": 0, "end": 45},
            "predicted_range_half_open": {"start": 0, "end": 45},
            "temporal_iou": 1.0,
            "iou_ge_95": True,
            "iou_ge_05": True,
            "attack_frames": 0,
            "chasing_frames": 43,
            "total_frames": 45,
            "fps": 30.0,
        },
    ],
}

DEMO_FALLBACK_DATA_ISO_GLOCO = {
    "is_fallback_demo": True,
    "demo_warning": "当前展示为内置离线演示数据（仅用于界面交互验证，非后端实时科研结果）",
    "total_clips": 2,
    "isolation_clips": 1,
    "group_locomotion_clips": 1,
    "isolation_accuracy": 1.0,
    "group_locomotion_accuracy": 1.0,
    "overall_accuracy": 1.0,
    "isolation_mean_iou": 1.0,
    "group_locomotion_mean_iou": 1.0,
    "overall_mean_iou": 1.0,
    "iou_ge_95_rate": 1.0,
    "iou_ge_05_rate": 1.0,
    "per_clip_results": [
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
            "iou_ge_95": True,
            "iou_ge_05": True,
            "total_frames": 124,
            "fps": 30.01,
        },
        {
            "clip_id": "0580160ac4fa7d99b838c7ce",
            "ground_truth_behavior": "Isolation",
            "predicted_behavior": "Isolation",
            "is_correct": True,
            "target_mouse_ids": [1],
            "semantic_roles": {"Isolated": [1]},
            "ground_truth_range_half_open": {"start": 0, "end": 399},
            "predicted_range_half_open": {"start": 0, "end": 399},
            "temporal_iou": 1.0,
            "iou_ge_95": True,
            "iou_ge_05": True,
            "total_frames": 399,
            "fps": 30.01,
        },
    ],
}

DEMO_FALLBACK_DATA_STATIC_LOCO = {
    "is_fallback_demo": True,
    "demo_warning": "当前展示为内置离线演示数据（仅用于界面交互验证，非后端实时科研结果）",
    "total_clips": 2,
    "static_clips": 1,
    "locomotion_clips": 1,
    "static_accuracy": 1.0,
    "locomotion_accuracy": 1.0,
    "overall_accuracy": 1.0,
    "static_mean_iou": 1.0,
    "locomotion_mean_iou": 1.0,
    "overall_mean_iou": 1.0,
    "iou_ge_90_rate": 1.0,
    "iou_ge_95_rate": 1.0,
    "iou_ge_05_rate": 1.0,
    "per_clip_results": [
        {
            "clip_id": "3837d36fc953a94770b41ae9",
            "ground_truth_behavior": "Static",
            "predicted_behavior": "Static",
            "is_correct": True,
            "target_mouse_ids": [1],
            "semantic_roles": {"Static": [1]},
            "ground_truth_range_half_open": {"start": 0, "end": 65},
            "predicted_range_half_open": {"start": 0, "end": 65},
            "temporal_iou": 1.0,
            "iou_ge_90": True,
            "iou_ge_95": True,
            "iou_ge_05": True,
            "total_frames": 65,
            "fps": 30.0,
        },
        {
            "clip_id": "007a717ba81cb573958f905c",
            "ground_truth_behavior": "Locomotion",
            "predicted_behavior": "Locomotion",
            "is_correct": True,
            "target_mouse_ids": [1],
            "semantic_roles": {"Locomotion": [1]},
            "ground_truth_range_half_open": {"start": 0, "end": 43},
            "predicted_range_half_open": {"start": 0, "end": 43},
            "temporal_iou": 1.0,
            "iou_ge_90": True,
            "iou_ge_95": True,
            "iou_ge_05": True,
            "total_frames": 43,
            "fps": 30.0,
        },
    ],
}

DEMO_FALLBACK_DATA_COMBINED = {
    "is_fallback_demo": True,
    "demo_warning": "当前展示为内置离线演示数据（仅用于界面交互验证，非后端实时科研结果）",
    "total_clips": 6,
    "attack_clips": 1,
    "chasing_clips": 1,
    "isolation_clips": 1,
    "group_locomotion_clips": 1,
    "static_clips": 1,
    "locomotion_clips": 1,
    "overall_accuracy": 1.0,
    "overall_mean_iou": 1.0,
    "iou_ge_90_rate": 1.0,
    "iou_ge_95_rate": 1.0,
    "iou_ge_05_rate": 1.0,
    "per_clip_results": [
        DEMO_FALLBACK_DATA["per_clip_results"][0],
        DEMO_FALLBACK_DATA["per_clip_results"][3],
        DEMO_FALLBACK_DATA_ISO_GLOCO["per_clip_results"][1],
        DEMO_FALLBACK_DATA_ISO_GLOCO["per_clip_results"][0],
        DEMO_FALLBACK_DATA_STATIC_LOCO["per_clip_results"][0],
        DEMO_FALLBACK_DATA_STATIC_LOCO["per_clip_results"][1],
    ],
}


def _select_demo_fallback(evaluation_file: Path) -> dict[str, Any]:
    name = evaluation_file.name.lower()
    if "combined" in name:
        return DEMO_FALLBACK_DATA_COMBINED
    if "static" in name or ("locomotion" in name and "group" not in name):
        return DEMO_FALLBACK_DATA_STATIC_LOCO
    if "isolation" in name or "group" in name:
        return DEMO_FALLBACK_DATA_ISO_GLOCO
    return DEMO_FALLBACK_DATA


def render_comparison_page(
    initial_data: Mapping[str, Any] | None = None,
    fallback_data: Mapping[str, Any] | None = None,
) -> str:
    """Generate self-contained, responsive, modern comparison HTML page."""
    raw_fallback = (
        dict(fallback_data)
        if fallback_data is not None
        else dict(DEMO_FALLBACK_DATA_COMBINED)
    )
    raw_fallback["is_fallback_demo"] = True
    resolved_initial = dict(initial_data) if initial_data is not None else raw_fallback
    embedded_data_json = json.dumps(resolved_initial, ensure_ascii=False)
    fallback_data_json = json.dumps(raw_fallback, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>小鼠行为识别与精准边界裁剪对比验证系统</title>
<style>
  :root {{
    --bg-main: #0b0f19;
    --bg-card: #151d30;
    --bg-card-hover: #1e2945;
    --border-color: #263353;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #38bdf8;
    --accent-attack: #ef4444;
    --accent-chasing: #f59e0b;
    --accent-green: #10b981;
    --badge-bg: rgba(56, 189, 248, 0.15);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    height: 100%;
    max-height: 100vh;
    overflow: hidden;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.5;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}

  /* Top Navigation */
  header {{
    background-color: #111827;
    border-bottom: 1px solid var(--border-color);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
    z-index: 100;
  }}

  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .brand h1 {{
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
  }}

  .status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
  }}

  .status-live {{
    background-color: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
  }}

  .status-demo {{
    background-color: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
  }}

  .status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: currentColor;
    animation: pulse 2s infinite;
  }}

  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
  }}

  /* Recoverable Alert Banner */
  #alertBanner {{
    display: none;
    flex-shrink: 0;
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.2));
    border-bottom: 1px solid #b91c1c;
    color: #fecaca;
    padding: 10px 24px;
    font-size: 0.85rem;
    align-items: center;
    justify-content: space-between;
  }}

  .alert-btn {{
    background: #ef4444;
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
  }}

  /* Metric KPI Cards */
  .metrics-bar {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px;
    padding: 8px 18px;
    background-color: rgba(21, 29, 48, 0.6);
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
    transition: max-height 0.25s ease, padding 0.25s ease, opacity 0.2s ease;
    max-height: 180px;
  }}

  .metrics-bar.collapsed {{
    max-height: 0;
    padding-top: 0;
    padding-bottom: 0;
    border-bottom: none;
    opacity: 0;
    overflow: hidden;
    pointer-events: none;
  }}

  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 12px;
  }}

  .kpi-label {{
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 2px;
  }}

  .kpi-value {{
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
  }}

  .kpi-sub {{
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 2px;
  }}

  /* Main Workspace */
  main {{
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 0;
    overflow: hidden;
    height: 100%;
  }}

  /* Sidebar Clip Selector (Independent Scrollable Container) */
  aside {{
    background-color: #0f172a;
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    overflow: hidden;
  }}

  .aside-header {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    background: #0f172a;
    flex-shrink: 0;
  }}

  .search-input {{
    width: 100%;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    color: white;
    font-size: 0.85rem;
    outline: none;
    margin-bottom: 8px;
  }}

  .search-input:focus {{
    border-color: var(--accent-blue);
  }}

  .filter-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }}

  .filter-pill {{
    background: #1e293b;
    color: var(--text-secondary);
    border: 1px solid #334155;
    border-radius: 9999px;
    padding: 3px 9px;
    font-size: 0.72rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .filter-pill.active {{
    background: var(--accent-blue);
    color: #0f172a;
    font-weight: 700;
    border-color: var(--accent-blue);
  }}

  .clip-list {{
    flex: 1;
    min-height: 0;
    overflow-y: scroll;
    overscroll-behavior: contain;
    padding: 10px 12px;
    scrollbar-width: thin;
    scrollbar-color: #475569 #0f172a;
  }}

  .clip-list::-webkit-scrollbar {{
    width: 6px;
  }}
  .clip-list::-webkit-scrollbar-track {{
    background: #0f172a;
  }}
  .clip-list::-webkit-scrollbar-thumb {{
    background: #334155;
    border-radius: 3px;
  }}
  .clip-list::-webkit-scrollbar-thumb:hover {{
    background: #64748b;
  }}

  .clip-item {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-left: 3px solid transparent;
    border-radius: 6px;
    padding: 9px 12px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .clip-item:hover {{
    background: var(--bg-card-hover);
    border-color: #475569;
  }}

  .clip-item.selected {{
    border-color: var(--accent-blue);
    border-left: 3px solid var(--accent-blue);
    background: #1e2945;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
  }}

  .clip-item-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}

  .clip-badge {{
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
  }}

  .badge-attack {{
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
  }}

  .badge-chasing {{
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
  }}

  .badge-isolation {{
    background: rgba(6, 182, 212, 0.2);
    color: #22d3ee;
    border: 1px solid rgba(6, 182, 212, 0.4);
  }}

  .badge-gloco {{
    background: rgba(139, 92, 246, 0.2);
    color: #a78bfa;
    border: 1px solid rgba(139, 92, 246, 0.4);
  }}

  .badge-static {{
    background: rgba(14, 165, 233, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(14, 165, 233, 0.4);
  }}

  .badge-locomotion {{
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
  }}

  .clip-iou {{
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-green);
  }}

  .clip-id {{
    font-family: monospace;
    font-size: 0.8rem;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .clip-meta {{
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 4px;
  }}

  /* Content Panel */
  .stage-panel {{
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 12px 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scrollbar-width: thin;
    scrollbar-color: #475569 transparent;
  }}

  .stage-panel::-webkit-scrollbar {{
    width: 6px;
  }}
  .stage-panel::-webkit-scrollbar-thumb {{
    background: #334155;
    border-radius: 3px;
  }}

  /* Fixed / Sticky Video Stage Zone (Pins Video & Synchronized Controls) */
  .stage-sticky-video-zone {{
    position: sticky;
    top: 0;
    z-index: 20;
    background: var(--bg-main);
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-color);
  }}

  .stage-header {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }}

  .stage-title {{
    font-size: 1.05rem;
    font-weight: 700;
  }}

  .stage-tags {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}

  /* Video Comparison Grid */
  .video-comparison-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}

  .video-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}

  .video-card-header {{
    padding: 8px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(15, 23, 42, 0.8);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .video-wrapper {{
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    max-height: min(38vh, 360px);
    background: #000000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}

  .video-wrapper video {{
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}

  .video-card-footer {{
    padding: 6px 10px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    background: rgba(15, 23, 42, 0.5);
    border-top: 1px solid var(--border-color);
  }}

  /* Synchronized Controls */
  .player-controls {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}

  .timeline-bar {{
    position: relative;
    width: 100%;
    height: 12px;
    background: #334155;
    border-radius: 6px;
    cursor: pointer;
  }}

  .timeline-gt-range {{
    position: absolute;
    top: 0;
    bottom: 0;
    background: rgba(56, 189, 248, 0.4);
    border: 1px solid var(--accent-blue);
    border-radius: 4px;
    pointer-events: none;
  }}

  .timeline-pred-range {{
    position: absolute;
    top: 2px;
    bottom: 2px;
    background: rgba(16, 185, 129, 0.55);
    border: 1px solid var(--accent-green);
    border-radius: 3px;
    pointer-events: none;
  }}

  .timeline-progress {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 0%;
    background: rgba(255, 255, 255, 0.3);
    pointer-events: none;
  }}

  .timeline-scrubber {{
    position: absolute;
    top: -4px;
    width: 4px;
    height: 20px;
    background: #ffffff;
    border-radius: 2px;
    transform: translateX(-50%);
    pointer-events: none;
  }}

  .control-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }}

  .control-btn-group {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .btn {{
    background: #1e293b;
    color: white;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }}

  .btn:hover {{
    background: #334155;
    border-color: #64748b;
  }}

  .btn-primary {{
    background: var(--accent-blue);
    color: #0f172a;
    border-color: var(--accent-blue);
  }}

  .btn-primary:hover {{
    background: #7dd3fc;
  }}

  /* Diagnostic Telemetry Table */
  .telemetry-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px 20px;
  }}

  .telemetry-title {{
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .telemetry-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
  }}

  .tele-item {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px 12px;
  }}

  .tele-item-label {{
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-weight: 600;
  }}

  .tele-item-val {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
  }}

  .progress-outer {{
    width: 100%;
    height: 8px;
    background: #334155;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
  }}

  .progress-inner {{
    height: 100%;
    background: linear-gradient(90deg, #34d399, #10b981);
    border-radius: 4px;
    transition: width 0.3s ease;
  }}

  /* Responsive */
  @media (max-width: 1024px) {{
    html, body {{
      overflow: auto;
      height: auto;
      max-height: none;
    }}
    main {{
      grid-template-columns: 1fr;
      height: auto;
      overflow: visible;
    }}
    aside {{
      height: 320px;
      max-height: 320px;
    }}
    .video-comparison-grid {{
      grid-template-columns: 1fr;
    }}
    .stage-sticky-video-zone {{
      position: relative;
      top: auto;
    }}
  }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>小鼠行为识别与精准边界裁剪对比验证系统</h1>
    <span id="connBadge" class="status-badge status-live">
      <span class="status-dot"></span>
      <span id="connText">实时后端数据连接中...</span>
    </span>
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <button class="btn" id="btnToggleMetrics" onclick="toggleMetricsBar()" title="展开/收起顶部统计面板">📊 折叠统计</button>
    <button class="btn" onclick="retryFetchData()">🔄 刷新数据</button>
  </div>
</header>

<div id="alertBanner">
  <span id="alertMsg">⚠️ 无法连接后端数据 API。当前已自动呈现离线演示数据集。</span>
  <button class="alert-btn" onclick="retryFetchData()">重试 API 连接</button>
</div>

<!-- Dashboard Metrics Bar -->
<section class="metrics-bar" id="metricsBar" aria-label="核心指标统计">
  <div class="kpi-card">
    <div class="kpi-label">视频样本总数</div>
    <div class="kpi-value" id="kpiTotal">--</div>
    <div class="kpi-sub" id="kpiTotalSub">多行为验证集加载中...</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Attack 准确率</div>
    <div class="kpi-value" style="color:#f87171;" id="kpiAttack">--</div>
    <div class="kpi-sub" id="kpiAttackSub">载入中...</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Chasing 准确率</div>
    <div class="kpi-value" style="color:#fbbf24;" id="kpiChasing">--</div>
    <div class="kpi-sub" id="kpiChasingSub">载入中...</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">整体分类准确率</div>
    <div class="kpi-value" style="color:#38bdf8;" id="kpiOverall">--</div>
    <div class="kpi-sub" id="kpiOverallSub">载入中...</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">时序重叠度 (Mean IoU)</div>
    <div class="kpi-value" style="color:#34d399;" id="kpiMeanIoU">--</div>
    <div class="kpi-sub" id="kpiIouSub">命中片段平均 (≥90% 达标)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">重叠率 ≥ 90% 比例</div>
    <div class="kpi-value" style="color:#34d399;" id="kpiPass90">--</div>
    <div class="kpi-sub" id="kpiPass90Sub">载入中...</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">重叠率 ≥ 95% 比例</div>
    <div class="kpi-value" style="color:#34d399;" id="kpiPass95">--</div>
    <div class="kpi-sub" id="kpiPass95Sub">载入中...</div>
  </div>
</section>

<!-- Main Workspace -->
<main>
  <!-- Aside List (Independently Scrollable) -->
  <aside>
    <div class="aside-header">
      <input type="text" id="searchInput" class="search-input" placeholder="🔍 搜索片段ID / 小鼠ID / 行为..." oninput="onFilterChange()">
      <div class="filter-pills" id="filterPills">
        <button class="filter-pill active" data-filter="all">全部 (--)</button>
        <button class="filter-pill" data-filter="iou90">重叠率≥90% (--)</button>
        <button class="filter-pill" data-filter="iou95">重叠率≥95% (--)</button>
        <button class="filter-pill" data-filter="correct">识别正确 (--)</button>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.75rem; color:var(--text-secondary);">
        <span id="clipCountText">显示 0 / 0 个片段</span>
        <span style="font-size:0.7rem; color:var(--text-muted);">快捷键: ↑ / ↓ 切片 · 空格播放</span>
      </div>
    </div>
    <div class="clip-list" id="clipList">
      <!-- Generated via JS -->
    </div>
  </aside>

  <!-- Stage Panel -->
  <div class="stage-panel">
    <!-- Fixed / Sticky Video Zone (Pins Video & Synchronized Controls) -->
    <div class="stage-sticky-video-zone">
      <!-- Selected Clip Header -->
      <div class="stage-header">
        <div>
          <div class="stage-title" id="stageTitle">片段载入中...</div>
          <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:3px;" id="stageSubtitle">请从左侧列表选择待检片段</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <div class="stage-tags" id="stageTags">
            <!-- Badges -->
          </div>
          <div style="display:flex; align-items:center; gap:6px; margin-left:4px;">
            <button class="btn" style="padding:4px 10px; font-size:0.75rem;" onclick="navigateClip(-1)" title="上一片段 (快捷键: ↑ 或 [)">⏮ 上一个</button>
            <button class="btn" style="padding:4px 10px; font-size:0.75rem;" onclick="navigateClip(1)" title="下一片段 (快捷键: ↓ 或 ])">下一个 ⏭</button>
          </div>
        </div>
      </div>

      <!-- Video Comparison Grid -->
      <div class="video-comparison-grid">
        <!-- Original Video Card -->
        <div class="video-card">
          <div class="video-card-header">
            <div style="display:flex; align-items:center; gap:8px;">
              <span>📹 【原视频】完整监控画面</span>
              <button id="btnToggleOrigBoxes" class="filter-pill active" onclick="toggleOrigBoxes()" style="cursor:pointer; font-size:0.75rem; padding:2px 8px;" title="切换是否在原画上叠加目标检测框与角色标注">🔲 检测框: 开启</button>
            </div>
            <span id="origTag" style="color:var(--accent-blue); font-size:0.75rem;">Annotated (BBoxes & Roles)</span>
          </div>
          <div class="video-wrapper">
            <video id="origVideo" playsinline preload="auto" controls></video>
          </div>
          <div class="video-card-footer" id="origFooter">
            标注真实时序区间: 载入中...
          </div>
        </div>

        <!-- Cropped Visualized Video Card -->
        <div class="video-card">
          <div class="video-card-header">
            <span>🎯 【裁剪视频】精准行为窗口 (HUD与目标追踪叠加)</span>
            <span id="cropTag" style="color:var(--accent-green); font-size:0.75rem;">Cropped & Visualized</span>
          </div>
          <div class="video-wrapper">
            <video id="cropVideo" playsinline preload="auto" controls></video>
            <div id="cropOverlay" style="display:none; position:absolute; top:0; left:0; right:0; bottom:0; background:rgba(15,23,42,0.92); color:#f87171; flex-direction:column; align-items:center; justify-content:center; padding:20px; text-align:center; z-index:10;">
              <div style="font-size:2rem; margin-bottom:8px;">⚠️</div>
              <div id="cropOverlayTitle" style="font-weight:700; font-size:1rem; margin-bottom:4px; color:#f87171;">未生成裁剪视频</div>
              <div id="cropOverlayDesc" style="font-size:0.8rem; color:#94a3b8; max-width:320px; line-height:1.6;">该样本行为与真值不符或未检测到目标行为。请在左侧切换到【识别正确】或【重叠率≥95%】标签查看对比。</div>
            </div>
          </div>
          <div class="video-card-footer" id="cropFooter">
            算法裁剪时序区间: 载入中...
          </div>
        </div>
      </div>

      <!-- Synchronized Player Controls -->
      <div class="player-controls">
        <div class="timeline-bar" id="timelineBar" onclick="onTimelineClick(event)">
          <div class="timeline-gt-range" id="tlGtRange" title="人工标注真值区间"></div>
          <div class="timeline-pred-range" id="tlPredRange" title="算法裁剪事件区间"></div>
          <div class="timeline-progress" id="tlProgress"></div>
          <div class="timeline-scrubber" id="tlScrubber"></div>
        </div>

        <div class="control-row">
          <div class="control-btn-group">
            <button class="btn btn-primary" id="btnPlay" onclick="togglePlay()">▶ 同步播放</button>
            <button class="btn" onclick="stepFrame(-1)">⏮ 前一帧</button>
            <button class="btn" onclick="stepFrame(1)">⏭ 后一帧</button>
            <button class="btn" onclick="restartVideos()">↺ 重新播放</button>
          </div>

          <div class="control-btn-group">
            <span style="font-size:0.85rem; color:var(--text-secondary);" id="timeDisplay">0.00s / 0.00s (帧 0/0)</span>
            <select class="btn" id="speedSelect" onchange="changeSpeed(this.value)">
              <option value="0.25">0.25x 慢放</option>
              <option value="0.5">0.50x 慢放</option>
              <option value="1.0" selected>1.00x 正常</option>
              <option value="1.5">1.50x 快放</option>
              <option value="2.0">2.00x 快速</option>
            </select>
            <label style="font-size:0.8rem; color:var(--text-secondary); display:flex; align-items:center; gap:4px; cursor:pointer;">
              <input type="checkbox" id="chkLoop" checked> 循环
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Diagnostic Telemetry Details Table -->
    <div class="telemetry-card">
      <div class="telemetry-title">
        <span>📊 裁剪精确度与行为时序诊断明细</span>
        <span id="teleIouBadge" style="font-size:0.85rem; color:var(--accent-green); font-weight:700;">IoU: 100.0%</span>
      </div>

      <div class="telemetry-grid">
        <div class="tele-item">
          <div class="tele-item-label">时序重叠率 (Temporal IoU)</div>
          <div class="tele-item-val" id="diagIou">100.0%</div>
          <div class="progress-outer">
            <div class="progress-inner" id="diagIouBar" style="width: 100%;"></div>
          </div>
        </div>

        <div class="tele-item">
          <div class="tele-item-label">行为预测 vs 人工真值</div>
          <div class="tele-item-val" id="diagBehavior">Attack (正确)</div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;" id="diagBehaviorSub">真值: Attack</div>
        </div>

        <div class="tele-item">
          <div class="tele-item-label">标注真值区间 (Ground Truth)</div>
          <div class="tele-item-val" id="diagGtRange">[0, 43)</div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;" id="diagGtSec">43 帧 (1.43s)</div>
        </div>

        <div class="tele-item">
          <div class="tele-item-label">算法裁剪区间 (Cropped)</div>
          <div class="tele-item-val" id="diagPredRange">[0, 42)</div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;" id="diagPredSec">42 帧 (1.40s)</div>
        </div>

        <div class="tele-item">
          <div class="tele-item-label">目标小鼠 ID 与角色分配</div>
          <div class="tele-item-val" id="diagMice">ID: 1690, 1691</div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;" id="diagRoles">Attacker: 1691, Attacked: 1690</div>
        </div>

        <div class="tele-item">
          <div class="tele-item-label">起止帧边界误差</div>
          <div class="tele-item-val" id="diagError">起: 0 帧 / 止: 1 帧</div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;" id="diagErrorMs">起始误差: 0.00s / 截止误差: 0.03s</div>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
// Embedded initial or fallback data
const EMBEDDED_DATA = {embedded_data_json};
const DEMO_FALLBACK = {fallback_data_json};

let currentData = EMBEDDED_DATA;
let activeFilter = 'all';
let currentClip = null;
let isPlaying = false;
let isLiveApi = false;
let showOrigBoxes = true;

const origVideo = document.getElementById('origVideo');
const cropVideo = document.getElementById('cropVideo');
const btnPlay = document.getElementById('btnPlay');
const timeDisplay = document.getElementById('timeDisplay');
const tlProgress = document.getElementById('tlProgress');
const tlScrubber = document.getElementById('tlScrubber');
const tlGtRange = document.getElementById('tlGtRange');
const tlPredRange = document.getElementById('tlPredRange');
const chkLoop = document.getElementById('chkLoop');

function toggleOrigBoxes() {{
  showOrigBoxes = !showOrigBoxes;
  const btn = document.getElementById('btnToggleOrigBoxes');
  const tag = document.getElementById('origTag');
  if (showOrigBoxes) {{
    btn.classList.add('active');
    btn.textContent = '🔲 检测框: 开启';
    tag.textContent = 'Annotated (BBoxes & Roles)';
    tag.style.color = 'var(--accent-blue)';
  }} else {{
    btn.classList.remove('active');
    btn.textContent = '🔲 检测框: 关闭 (纯原画)';
    tag.textContent = 'Raw Source (clip.mp4)';
    tag.style.color = 'var(--text-muted)';
  }}
  updateOrigVideoSource(true);
}}

function _getSelectedPlaybackRate() {{
  const sel = document.getElementById('speedSelect');
  const spd = sel ? parseFloat(sel.value) : 1.0;
  return Number.isFinite(spd) && spd > 0 ? spd : 1.0;
}}

function updateOrigVideoSource(preserveTime = false) {{
  if (!currentClip) return;
  const curTime = preserveTime ? (origVideo.currentTime || 0) : 0;
  const wasPlaying = preserveTime && isPlaying && !origVideo.paused;
  const behDir = currentClip.ground_truth_behavior;
  const param = showOrigBoxes ? '1' : '0';
  const spd = _getSelectedPlaybackRate();
  origVideo.defaultPlaybackRate = spd;
  origVideo.playbackRate = spd;
  origVideo.src = `/api/video/original/${{encodeURIComponent(behDir)}}/${{encodeURIComponent(currentClip.clip_id)}}?annotated=${{param}}`;
  origVideo.addEventListener('loadedmetadata', () => {{
    origVideo.playbackRate = _getSelectedPlaybackRate();
    if (curTime > 0 && Number.isFinite(origVideo.duration)) {{
      origVideo.currentTime = Math.min(curTime, Math.max(0, origVideo.duration - 0.01));
    }}
    updateTimeUI(origVideo.currentTime || 0);
    if (wasPlaying) {{
      origVideo.play().catch(() => {{}});
    }}
  }}, {{ once: true }});
}}

cropVideo.addEventListener('loadedmetadata', () => {{
  cropVideo.playbackRate = _getSelectedPlaybackRate();
}});

// Initial App Shell Boot
document.addEventListener('DOMContentLoaded', () => {{
  const filterPillsEl = document.getElementById('filterPills');
  if (filterPillsEl) {{
    filterPillsEl.addEventListener('click', (e) => {{
      const pill = e.target.closest('.filter-pill');
      if (pill && pill.dataset.filter) {{
        setFilter(pill.dataset.filter, pill);
      }}
    }});
  }}

  // 1. Initial shell renders immediately with valid embedded or fallback data
  const hasValidInitial = (
    EMBEDDED_DATA
    && Array.isArray(EMBEDDED_DATA.per_clip_results)
    && EMBEDDED_DATA.per_clip_results.length > 0
  );
  if (hasValidInitial) {{
    loadData(EMBEDDED_DATA, Boolean(EMBEDDED_DATA.is_fallback_demo));
  }} else {{
    loadData(DEMO_FALLBACK, true);
  }}

  // 2. Asynchronously probe live backend API
  fetchLiveEvaluation();
}});

async function fetchLiveEvaluation() {{
  const badge = document.getElementById('connBadge');
  const connText = document.getElementById('connText');
  const alertBanner = document.getElementById('alertBanner');
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), 8000);

  try {{
    const res = await fetch('/api/evaluation', {{ signal: controller.signal }});
    if (!res.ok) {{
      throw new Error(`HTTP ${{res.status}} (${{res.statusText}})`);
    }}
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('application/json')) {{
      throw new Error(`响应非合法 JSON (Content-Type: ${{contentType}})`);
    }}
    const data = await res.json();
    if (!data || !Array.isArray(data.per_clip_results)) {{
      throw new Error('API 返回的数据缺少 per_clip_results 字段');
    }}
    if (data.per_clip_results.length === 0) {{
      throw new Error('API 返回的评测结果为空 (per_clip_results 为空)');
    }}

    if (data.is_fallback_demo) {{
      isLiveApi = false;
      badge.className = 'status-badge status-demo';
      connText.textContent = '▲ 离线演示数据模式 (后端暂无实测文件)';
      document.getElementById('alertMsg').textContent = '⚠️ 后端尚未生成实测评测 JSON，当前展示内置演示数据集。';
      alertBanner.style.display = 'flex';
      loadData(data, true);
      return;
    }}

    // API Success
    isLiveApi = true;
    badge.className = 'status-badge status-live';
    connText.textContent = `● 实时后端数据已连接 (${{data.per_clip_results.length}} 个全量视频)`;
    alertBanner.style.display = 'none';
    loadData(data, false);
  }} catch (err) {{
    // Graceful in-page recoverable state using dedicated demo fallback
    isLiveApi = false;
    const errMsg = (err && err.name === 'AbortError')
      ? 'API 请求超时 (8s)'
      : ((err && err.message) ? err.message : String(err));
    badge.className = 'status-badge status-demo';
    connText.textContent = '▲ 离线演示数据模式 (仅供交互验证)';
    document.getElementById('alertMsg').textContent = `⚠️ 无法获取实时评测 API 数据 (${{errMsg}})。已自动加载内置演示数据集，页面导航与功能完全可用。`;
    alertBanner.style.display = 'flex';
    loadData(DEMO_FALLBACK, true);
  }} finally {{
    clearTimeout(timerId);
  }}
}}

function retryFetchData() {{
  document.getElementById('connText').textContent = '正在重新连接 API...';
  fetchLiveEvaluation();
}}

function escapeHtml(value) {{
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

function loadData(dataset, isFallback) {{
  currentData = dataset;
  updateDashboardMetrics(dataset, isFallback);
  renderFilterPills(dataset);
  renderClipList();
  
  // Select first available clip
  const clips = getFilteredClips();
  if (clips.length > 0) {{
    selectClip(clips[0].clip_id);
  }}
}}

function getBehaviorBadgeClass(beh) {{
  if (beh === 'Isolation') return 'clip-badge badge-isolation';
  if (beh === 'Group locomotion') return 'clip-badge badge-gloco';
  if (beh === 'Attack') return 'clip-badge badge-attack';
  if (beh === 'Chasing') return 'clip-badge badge-chasing';
  if (beh === 'Static') return 'clip-badge badge-static';
  if (beh === 'Locomotion') return 'clip-badge badge-locomotion';
  return 'clip-badge';
}}

function updateDashboardMetrics(data, isFallback) {{
  const container = document.getElementById('metricsBar');
  if (!container) return;
  const clips = Array.isArray(data.per_clip_results) ? data.per_clip_results : [];
  const total = data.total_clips !== undefined ? data.total_clips : clips.length;

  const behMap = {{}};
  clips.forEach(c => {{
    const b = c.ground_truth_behavior || 'Unknown';
    if (!behMap[b]) behMap[b] = {{ total: 0, correct: 0 }};
    behMap[b].total += 1;
    if (c.is_correct) behMap[b].correct += 1;
  }});

  const overallCorrect = clips.filter(c => c.is_correct).length;
  const hasClips = clips.length > 0;
  const ovAcc = hasClips
    ? (overallCorrect / clips.length)
    : (typeof data.overall_accuracy === 'number' ? data.overall_accuracy : 0.0);
  const correctClips = clips.filter(c => c.is_correct);
  const meanIou = correctClips.length > 0
    ? (correctClips.reduce((acc, c) => acc + (c.temporal_iou || 0), 0) / correctClips.length)
    : (hasClips ? 0.0 : (typeof data.overall_mean_iou === 'number' ? data.overall_mean_iou : 0.0));
  const pass90Clips = correctClips.filter(c => (c.temporal_iou || 0) >= 0.90);
  const pass90Rate = correctClips.length > 0 ? (pass90Clips.length / correctClips.length) : 0.0;
  const pass95Clips = correctClips.filter(c => (c.temporal_iou || 0) >= 0.95);
  const pass95Rate = correctClips.length > 0 ? (pass95Clips.length / correctClips.length) : 0.0;

  const behColors = {{
    'Attack': '#f87171',
    'Chasing': '#fbbf24',
    'Isolation': '#22d3ee',
    'Group locomotion': '#a78bfa',
    'Static': '#38bdf8',
    'Locomotion': '#34d399'
  }};

  const behSub = Object.entries(behMap).map(([k, v]) => `${{v.total}} ${{escapeHtml(k)}}`).join(' / ');

  let html = `
    <div class="kpi-card">
      <div class="kpi-label">视频样本总数</div>
      <div class="kpi-value" id="kpiTotal">${{total}}</div>
      <div class="kpi-sub" id="kpiTotalSub">${{behSub}}</div>
    </div>
  `;

  for (const [beh, stats] of Object.entries(behMap)) {{
    const acc = stats.total > 0 ? (stats.correct / stats.total) * 100 : 0;
    const col = behColors[beh] || '#38bdf8';
    const valIdAttr = beh === 'Attack' ? ' id="kpiAttack"' : (beh === 'Chasing' ? ' id="kpiChasing"' : '');
    html += `
      <div class="kpi-card">
        <div class="kpi-label">${{escapeHtml(beh)}} 准确率</div>
        <div class="kpi-value"${{valIdAttr}} style="color:${{col}};">${{acc.toFixed(2)}}%</div>
        <div class="kpi-sub">${{stats.correct}} / ${{stats.total}} 段正确</div>
      </div>
    `;
  }}

  html += `
    <div class="kpi-card">
      <div class="kpi-label">整体分类准确率</div>
      <div class="kpi-value" style="color:#38bdf8;" id="kpiOverall">${{(ovAcc * 100).toFixed(2)}}%</div>
      <div class="kpi-sub" id="kpiOverallSub">${{overallCorrect}} / ${{clips.length}} 段正确 (${{isFallback ? '离线演示' : '全量真实'}})</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">时序重叠度 (Mean IoU)</div>
      <div class="kpi-value" style="color:#34d399;" id="kpiMeanIoU">${{(meanIou * 100).toFixed(2)}}%</div>
      <div class="kpi-sub" id="kpiIouSub">命中片段平均 (≥90% 达标)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">重叠率 ≥ 90% 比例</div>
      <div class="kpi-value" style="color:#34d399;" id="kpiPass90">${{(pass90Rate * 100).toFixed(2)}}%</div>
      <div class="kpi-sub" id="kpiPass90Sub">${{pass90Clips.length}} / ${{correctClips.length}} 段达成 ≥90% 重叠</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">重叠率 ≥ 95% 比例</div>
      <div class="kpi-value" style="color:#34d399;" id="kpiPass95">${{(pass95Rate * 100).toFixed(2)}}%</div>
      <div class="kpi-sub" id="kpiPass95Sub">${{pass95Clips.length}} / ${{correctClips.length}} 段达成 ≥95% 重叠</div>
    </div>
  `;

  container.innerHTML = html;
}}

function renderFilterPills(data) {{
  const container = document.getElementById('filterPills');
  if (!container || !data || !Array.isArray(data.per_clip_results)) return;
  const clips = data.per_clip_results;
  const total = clips.length;

  const behCounts = {{}};
  clips.forEach(c => {{
    const b = c.ground_truth_behavior || 'Unknown';
    behCounts[b] = (behCounts[b] || 0) + 1;
  }});

  const iou90Count = clips.filter(c => c.is_correct && (c.temporal_iou || 0) >= 0.90).length;
  const iou95Count = clips.filter(c => c.is_correct && (c.temporal_iou || 0) >= 0.95).length;
  const correctCount = clips.filter(c => c.is_correct).length;

  let html = `<button class="filter-pill ${{activeFilter === 'all' ? 'active' : ''}}" data-filter="all">全部 (${{total}})</button>`;

  for (const [beh, count] of Object.entries(behCounts)) {{
    const filterKey = 'beh:' + beh;
    const safeKey = escapeHtml(filterKey);
    const isActive = activeFilter === filterKey ? 'active' : '';
    html += `<button class="filter-pill ${{isActive}}" data-filter="${{safeKey}}">${{escapeHtml(beh)}} (${{count}})</button>`;
  }}

  html += `<button class="filter-pill ${{activeFilter === 'iou90' ? 'active' : ''}}" data-filter="iou90">重叠率≥90% (${{iou90Count}})</button>`;
  html += `<button class="filter-pill ${{activeFilter === 'iou95' ? 'active' : ''}}" data-filter="iou95">重叠率≥95% (${{iou95Count}})</button>`;
  html += `<button class="filter-pill ${{activeFilter === 'correct' ? 'active' : ''}}" data-filter="correct">识别正确 (${{correctCount}})</button>`;

  container.innerHTML = html;
}}

function getFilteredClips() {{
  if (!currentData || !Array.isArray(currentData.per_clip_results)) return [];
  const query = (document.getElementById('searchInput').value || '').toLowerCase().trim();

  return currentData.per_clip_results.filter(clip => {{
    // Filter pill
    if (activeFilter.startsWith('beh:')) {{
      const targetBeh = activeFilter.slice(4);
      if (clip.ground_truth_behavior !== targetBeh) return false;
    }} else if (activeFilter === 'attack' && clip.ground_truth_behavior !== 'Attack') {{
      return false;
    }} else if (activeFilter === 'chasing' && clip.ground_truth_behavior !== 'Chasing') {{
      return false;
    }} else if (activeFilter === 'correct' && !clip.is_correct) {{
      return false;
    }} else if (activeFilter === 'iou90' && (!clip.is_correct || (clip.temporal_iou || 0) < 0.90)) {{
      return false;
    }} else if (activeFilter === 'iou95' && (!clip.is_correct || (clip.temporal_iou || 0) < 0.95)) {{
      return false;
    }}

    // Search query
    if (query) {{
      const matchId = (clip.clip_id || '').toLowerCase().includes(query);
      const matchBeh = (clip.ground_truth_behavior || '').toLowerCase().includes(query);
      const matchPred = (clip.predicted_behavior || '').toLowerCase().includes(query);
      const matchMice = (clip.target_mouse_ids || []).some(m => String(m).includes(query));
      if (!matchId && !matchMice && !matchBeh && !matchPred) return false;
    }}
    return true;
  }});
}}

function toggleMetricsBar() {{
  const bar = document.getElementById('metricsBar');
  const btn = document.getElementById('btnToggleMetrics');
  if (!bar || !btn) return;
  if (bar.classList.contains('collapsed')) {{
    bar.classList.remove('collapsed');
    btn.textContent = '📊 折叠统计';
  }} else {{
    bar.classList.add('collapsed');
    btn.textContent = '📊 展开统计';
  }}
}}

function navigateClip(dir) {{
  const clips = getFilteredClips();
  if (clips.length === 0) return;
  let idx = clips.findIndex(c => c.clip_id === (currentClip ? currentClip.clip_id : ''));
  if (idx === -1) {{
    idx = 0;
  }} else {{
    idx = (idx + dir + clips.length) % clips.length;
  }}
  selectClip(clips[idx].clip_id, true);
}}

function renderClipList() {{
  const listEl = document.getElementById('clipList');
  const clips = getFilteredClips();

  listEl.innerHTML = '';
  const countEl = document.getElementById('clipCountText');
  const tot = (currentData && Array.isArray(currentData.per_clip_results)) ? currentData.per_clip_results.length : clips.length;
  if (countEl) {{
    countEl.textContent = `显示 ${{clips.length}} / ${{tot}} 个片段`;
  }}

  if (clips.length === 0) {{
    listEl.innerHTML = '<div style="color:var(--text-muted); padding:20px; text-align:center;">没有找到匹配的视频片段</div>';
    return;
  }}

  clips.forEach(clip => {{
    const item = document.createElement('div');
    item.className = 'clip-item' + (currentClip && currentClip.clip_id === clip.clip_id ? ' selected' : '');
    item.dataset.clipId = clip.clip_id;
    item.onclick = () => selectClip(clip.clip_id);

    const badgeClass = getBehaviorBadgeClass(clip.ground_truth_behavior);
    const iouPct = ((clip.temporal_iou || 0) * 100).toFixed(1);
    const iouColor = clip.temporal_iou >= 0.95 ? 'var(--accent-green)' : (clip.temporal_iou >= 0.5 ? '#fbbf24' : '#ef4444');
    const safeBeh = escapeHtml(clip.ground_truth_behavior);
    const safeId = escapeHtml(clip.clip_id);
    const safeTargets = escapeHtml((clip.target_mouse_ids || []).join(', '));

    item.innerHTML = `
      <div class="clip-item-top">
        <span class="${{badgeClass}}">${{safeBeh}}</span>
        <span class="clip-iou" style="color:${{iouColor}};">IoU: ${{iouPct}}%</span>
      </div>
      <div class="clip-id" title="${{safeId}}">${{safeId}}</div>
      <div class="clip-meta">
        目标: [${{safeTargets}}] · ${{clip.total_frames || 0}} 帧 (${{((clip.total_frames || 0) / (clip.fps || 30)).toFixed(1)}}s)
        · ${{clip.is_correct ? '✅ 预测正确' : '❌ 预测差异'}}
      </div>
    `;
    listEl.appendChild(item);
  }});
}}

function setFilter(filter, el) {{
  activeFilter = filter;
  document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  renderClipList();
}}

function onFilterChange() {{
  renderClipList();
}}

function selectClip(clipId, scrollIntoView = false) {{
  if (!currentData || !Array.isArray(currentData.per_clip_results)) return;
  const clip = currentData.per_clip_results.find(c => c.clip_id === clipId);
  if (!clip) return;

  currentClip = clip;

  // Highlight list
  let activeEl = null;
  document.querySelectorAll('.clip-item').forEach(el => {{
    const isThis = el.dataset.clipId === clipId;
    el.classList.toggle('selected', isThis);
    if (isThis) activeEl = el;
  }});

  if (scrollIntoView && activeEl) {{
    activeEl.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
  }}

  // Update Headers & Details
  document.getElementById('stageTitle').textContent = `片段: ${{clip.clip_id}}`;
  document.getElementById('stageSubtitle').textContent = `目标小鼠: [${{(clip.target_mouse_ids || []).join(', ')}}] · 全长 ${{clip.total_frames}} 帧 (${{(clip.total_frames / (clip.fps || 30)).toFixed(2)}}s @ ${{clip.fps || 30}}fps)`;

  const tagsEl = document.getElementById('stageTags');
  tagsEl.innerHTML = '';
  
  const badgeClass = getBehaviorBadgeClass(clip.ground_truth_behavior);
  tagsEl.innerHTML += `<span class="${{badgeClass}}" style="font-size:0.85rem; padding:4px 8px;">GT: ${{escapeHtml(clip.ground_truth_behavior)}}</span>`;

  if (clip.is_correct) {{
    tagsEl.innerHTML += `<span class="clip-badge" style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid #10b981; font-size:0.85rem; padding:4px 8px;">预测正确: ${{escapeHtml(clip.predicted_behavior)}}</span>`;
  }} else {{
    tagsEl.innerHTML += `<span class="clip-badge" style="background:rgba(239,68,68,0.2); color:#f87171; border:1px solid #ef4444; font-size:0.85rem; padding:4px 8px;">未命中: ${{escapeHtml(clip.predicted_behavior || 'None')}}</span>`;
  }}

  const iouPct = ((clip.temporal_iou || 0) * 100).toFixed(1);
  if (clip.temporal_iou >= 0.95) {{
    tagsEl.innerHTML += `<span class="clip-badge" style="background:rgba(56,189,248,0.2); color:#38bdf8; border:1px solid #38bdf8; font-size:0.85rem; padding:4px 8px;">重叠率 ≥95% (PASS)</span>`;
  }}

  // Reset players before loading new sources
  pauseVideos();
  origVideo.currentTime = 0;
  cropVideo.currentTime = 0;

  // Update Video Sources & State
  updateOrigVideoSource(false);

  const cropOverlay = document.getElementById('cropOverlay');
  const spd = _getSelectedPlaybackRate();
  if (clip.is_correct) {{
    cropOverlay.style.display = 'none';
    cropVideo.style.display = 'block';
    cropVideo.defaultPlaybackRate = spd;
    cropVideo.playbackRate = spd;
    cropVideo.src = `/api/video/cropped/${{encodeURIComponent(clip.clip_id)}}`;
  }} else {{
    cropOverlay.style.display = 'flex';
    document.getElementById('cropOverlayTitle').textContent = '未生成裁剪视频';
    document.getElementById('cropOverlayDesc').textContent = `该样本行为预测为 [${{clip.predicted_behavior || 'None'}}]，与真实标签 [${{clip.ground_truth_behavior}}] 不符。请在左侧切换到【识别正确】或【重叠率≥95%】标签进行比对。`;
    cropVideo.style.display = 'none';
    cropVideo.removeAttribute('src');
  }}

  cropVideo.onerror = () => {{
    if (clip.is_correct) {{
      cropOverlay.style.display = 'flex';
      document.getElementById('cropOverlayTitle').textContent = '裁剪视频加载失败';
      document.getElementById('cropOverlayDesc').textContent = '可能文件正在生成或网络受限，请尝试刷新页面重试。';
      cropVideo.style.display = 'none';
    }}
  }};

  // Update Telemetry Panel
  const gtS = clip.ground_truth_range_half_open ? clip.ground_truth_range_half_open.start : 0;
  const gtE = clip.ground_truth_range_half_open ? clip.ground_truth_range_half_open.end : clip.total_frames;
  const predS = clip.predicted_range_half_open ? clip.predicted_range_half_open.start : 0;
  const predE = clip.predicted_range_half_open ? clip.predicted_range_half_open.end : 0;

  document.getElementById('origFooter').textContent = `标注真实区间: 帧 [${{gtS}} ~ ${{gtE}}) | 时长: ${{((gtE - gtS)/(clip.fps||30)).toFixed(2)}} 秒`;
  document.getElementById('cropFooter').textContent = `算法裁剪区间: 帧 [${{predS}} ~ ${{predE}}) | 重叠率: ${{iouPct}}%`;

  document.getElementById('teleIouBadge').textContent = `IoU: ${{iouPct}}%` + (clip.temporal_iou >= 0.95 ? ' (≥95% PASS)' : '');
  document.getElementById('diagIou').textContent = `${{iouPct}}%`;
  document.getElementById('diagIouBar').style.width = `${{Math.min(100, Math.max(0, clip.temporal_iou * 100))}}%`;
  document.getElementById('diagBehavior').textContent = `${{clip.predicted_behavior || 'None'}} (${{clip.is_correct ? '正确' : '差异'}})`;
  document.getElementById('diagBehaviorSub').textContent = `真值标注: ${{clip.ground_truth_behavior}}`;
  document.getElementById('diagGtRange').textContent = `[${{gtS}}, ${{gtE}})`;
  document.getElementById('diagGtSec').textContent = `${{gtE - gtS}} 帧 (${{((gtE - gtS)/(clip.fps||30)).toFixed(2)}}s)`;
  document.getElementById('diagPredRange').textContent = `[${{predS}}, ${{predE}})`;
  document.getElementById('diagPredSec').textContent = `${{predE - predS}} 帧 (${{((predE - predS)/(clip.fps||30)).toFixed(2)}}s)`;
  document.getElementById('diagMice').textContent = `ID: ${{ (clip.target_mouse_ids || []).join(', ') }}`;
  
  let rolesStr = '';
  if (clip.semantic_roles) {{
    rolesStr = Object.entries(clip.semantic_roles).map(([k, v]) => `${{k}}: [${{v.join(', ')}}]`).join('; ');
  }}
  document.getElementById('diagRoles').textContent = rolesStr || '未分配具体子角色';

  const errS = Math.abs(predS - gtS);
  const errE = Math.abs(predE - gtE);
  document.getElementById('diagError').textContent = `起: ${{errS}} 帧 / 止: ${{errE}} 帧`;
  document.getElementById('diagErrorMs').textContent = `起始误差: ${{((errS)/(clip.fps||30)).toFixed(2)}}s / 截止误差: ${{((errE)/(clip.fps||30)).toFixed(2)}}s`;

  // Update Timeline Ranges
  const tot = clip.total_frames || 1;
  tlGtRange.style.left = `${{(gtS / tot) * 100}}%`;
  tlGtRange.style.width = `${{((gtE - gtS) / tot) * 100}}%`;

  tlPredRange.style.left = `${{(predS / tot) * 100}}%`;
  tlPredRange.style.width = `${{((predE - predS) / tot) * 100}}%`;

  updateTimeUI(0);
}}

// Synchronized Video Controls
function togglePlay() {{
  if (isPlaying) {{
    pauseVideos();
  }} else {{
    playVideos();
  }}
}}

function _shouldPlayCropAt(curSec) {{
  if (!currentClip || !currentClip.is_correct) return true;
  const predS = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.start) || 0;
  const predE = (currentClip.predicted_range_half_open && currentClip.predicted_range_half_open.end) || currentClip.total_frames || 0;
  const fps = currentClip.fps || 30.0;
  const offsetSec = predS / fps;
  const endSec = predE / fps;
  return curSec >= offsetSec && (predE <= predS || curSec < endSec);
}}

function playVideos() {{
  isPlaying = true;
  btnPlay.textContent = '⏸ 同步暂停';
  btnPlay.classList.remove('btn-primary');
  origVideo.play().catch(e => console.warn('origVideo play error:', e));
  if (cropVideo.style.display !== 'none' && cropVideo.src && _shouldPlayCropAt(origVideo.currentTime || 0)) {{
    cropVideo.play().catch(e => console.warn('cropVideo play error:', e));
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
  const delta = (1.0 / fps) * dir;
  origVideo.currentTime = Math.max(0, origVideo.currentTime + delta);
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
  if (!cropVideo.paused) {{
    cropVideo.pause();
  }}
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
    if (cur < offsetSec) {{
      if (!cropVideo.paused) cropVideo.pause();
      if (Math.abs(cropVideo.currentTime) > 0.05) cropVideo.currentTime = 0;
    }} else if (predE > predS && cur >= endSec) {{
      if (!cropVideo.paused) cropVideo.pause();
    }} else {{
      const targetCrop = Math.max(0, cur - offsetSec);
      if (isPlaying && cropVideo.paused && !origVideo.paused) {{
        cropVideo.play().catch(() => {{}});
      }}
      if (Math.abs(cropVideo.currentTime - targetCrop) > 0.08) {{
        cropVideo.currentTime = targetCrop;
      }}
    }}
  }}
}});

origVideo.addEventListener('ended', () => {{
  if (chkLoop.checked) {{
    restartVideos();
  }} else {{
    pauseVideos();
  }}
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
  const clickX = e.clientX - rect.left;
  const ratio = Math.min(1, Math.max(0, clickX / rect.width));
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

// Keyboard Navigation for fast browsing across clips
document.addEventListener('keydown', (e) => {{
  if (document.activeElement === document.getElementById('searchInput')) {{
    if (e.key === 'Escape') {{
      document.getElementById('searchInput').blur();
    }}
    return;
  }}

  if (e.key === 'ArrowDown' || e.key === 'j') {{
    e.preventDefault();
    navigateClip(1);
  }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
    e.preventDefault();
    navigateClip(-1);
  }} else if (e.code === 'Space') {{
    e.preventDefault();
    togglePlay();
  }} else if (e.key === '[' || (e.ctrlKey && e.key === 'ArrowLeft')) {{
    e.preventDefault();
    stepFrame(-1);
  }} else if (e.key === ']' || (e.ctrlKey && e.key === 'ArrowRight')) {{
    e.preventDefault();
    stepFrame(1);
  }}
}});
</script>

</body>
</html>"""


class ComparisonServerHandler(BaseHTTPRequestHandler):
    """HTTP request handler supporting Range streaming, JSON API, and resilient fallbacks."""

    evaluation_file: Path = DEFAULT_EVALUATION_JSON
    dataset_dir: Path = DEFAULT_DATASET_DIR
    isolation_dir: Path = DEFAULT_ISOLATION_DIR
    group_locomotion_dir: Path = DEFAULT_GROUP_LOCOMOTION_DIR
    static_dir: Path = DEFAULT_STATIC_DIR
    locomotion_dir: Path = DEFAULT_LOCOMOTION_DIR
    attack_dir: Path = DEFAULT_ATTACK_DIR
    chasing_dir: Path = DEFAULT_CHASING_DIR
    crops_dir: Path = DEFAULT_CROPS_DIR
    orig_annotated_dir: Path = DEFAULT_ORIG_ANNOTATED_DIR

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy request logs."""
        return

    def _host_ok(self) -> bool:
        client_ip = getattr(self, "client_address", ("", 0))[0]
        if client_ip in ("127.0.0.1", "::1", "localhost", "testclient"):
            return True
        host = self.headers.get("Host", "").split(":")[0].strip()
        return host in ("127.0.0.1", "localhost")

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _stream_file(self, file_path: Path, head_only: bool = False) -> None:
        if not file_path.is_file():
            self._send_json(404, {"error": "File not found", "path": str(file_path)})
            return

        size = file_path.stat().st_size
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "video/mp4" if file_path.suffix.lower() == ".mp4" else "application/octet-stream"

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

        parsed = urlsplit(self.path)
        path = unquote(parsed.path).strip("/")

        # Route 1: Home page
        if path in ("", "index.html"):
            fallback_data = _select_demo_fallback(self.evaluation_file)
            initial_data = None
            if self.evaluation_file.is_file():
                try:
                    loaded = json.loads(self.evaluation_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and isinstance(loaded.get("per_clip_results"), list):
                        initial_data = loaded
                except Exception:
                    initial_data = None
            if initial_data is None:
                initial_data = fallback_data

            html_text = render_comparison_page(initial_data, fallback_data=fallback_data)
            body = html_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
            return

        # Route 2: Health check
        if path == "api/health":
            self._send_json(200, {
                "status": "ok",
                "evaluation_file_exists": self.evaluation_file.is_file(),
                "crops_dir_exists": self.crops_dir.is_dir(),
            })
            return

        # Route 3: Evaluation JSON dataset
        if path == "api/evaluation":
            if self.evaluation_file.is_file():
                try:
                    data = json.loads(self.evaluation_file.read_text(encoding="utf-8"))
                    self._send_json(200, data)
                    return
                except Exception as exc:
                    self._send_json(500, {"error": f"Failed to parse evaluation JSON: {exc}"})
                    return
            else:
                # Return demo fallback if file not yet generated
                self._send_json(200, _select_demo_fallback(self.evaluation_file))
                return

        # Route 4: Stream Original Video: /api/video/original/<behavior>/<clip_id>
        match_orig = re.fullmatch(r"api/video/original/([^/]+)/([A-Za-z0-9_\-]+)", path)
        if match_orig:
            behavior, clip_id = match_orig.groups()
            query = parsed.query or ""
            is_raw = "annotated=0" in query or "raw=1" in query

            annotated_file = self.orig_annotated_dir / clip_id / "clip_annotated.mp4"

            source_file: Path | None = None
            beh_clean = behavior.strip()
            beh_lower = beh_clean.lower()
            if beh_lower in ("isolation",) and (self.isolation_dir / clip_id / "clip.mp4").is_file():
                source_file = self.isolation_dir / clip_id / "clip.mp4"
            elif beh_lower in ("group locomotion", "group_locomotion") and (self.group_locomotion_dir / clip_id / "clip.mp4").is_file():
                source_file = self.group_locomotion_dir / clip_id / "clip.mp4"
            elif beh_lower in ("attack",) and (self.attack_dir / clip_id / "clip.mp4").is_file():
                source_file = self.attack_dir / clip_id / "clip.mp4"
            elif beh_lower in ("chasing",) and (self.chasing_dir / clip_id / "clip.mp4").is_file():
                source_file = self.chasing_dir / clip_id / "clip.mp4"
            elif beh_lower in ("static", "stationary") and (self.static_dir / clip_id / "clip.mp4").is_file():
                source_file = self.static_dir / clip_id / "clip.mp4"
            elif beh_lower in ("locomotion",) and (self.locomotion_dir / clip_id / "clip.mp4").is_file():
                source_file = self.locomotion_dir / clip_id / "clip.mp4"
            elif (self.dataset_dir / beh_clean / clip_id / "clip.mp4").is_file():
                source_file = self.dataset_dir / beh_clean / clip_id / "clip.mp4"
            elif self.dataset_dir.is_dir():
                for sub in self.dataset_dir.iterdir():
                    if sub.is_dir() and sub.name.lower() == beh_lower:
                        cand = sub / clip_id / "clip.mp4"
                        if cand.is_file():
                            source_file = cand
                            break

            if not is_raw and annotated_file.is_file():
                self._stream_file(annotated_file, head_only)
            elif source_file and source_file.is_file():
                self._stream_file(source_file, head_only)
            else:
                self._send_json(404, {"error": "Video file not found", "clip_id": clip_id, "behavior": behavior})
            return

        # Route 5: Stream Cropped Video: /api/video/cropped/<clip_id>
        match_crop = re.fullmatch(r"api/video/cropped/([A-Za-z0-9_\-]+)", path)
        if match_crop:
            clip_id = match_crop.group(1)
            crop_file = self.crops_dir / clip_id / "clip_cropped.mp4"
            self._stream_file(crop_file, head_only)
            return

        # 404
        self._send_json(404, {"error": "Not Found", "route": path})


def create_comparison_server(
    evaluation_file: Path = DEFAULT_EVALUATION_JSON,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    isolation_dir: Path = DEFAULT_ISOLATION_DIR,
    group_locomotion_dir: Path = DEFAULT_GROUP_LOCOMOTION_DIR,
    static_dir: Path = DEFAULT_STATIC_DIR,
    locomotion_dir: Path = DEFAULT_LOCOMOTION_DIR,
    attack_dir: Path = DEFAULT_ATTACK_DIR,
    chasing_dir: Path = DEFAULT_CHASING_DIR,
    crops_dir: Path = DEFAULT_CROPS_DIR,
    orig_annotated_dir: Path = DEFAULT_ORIG_ANNOTATED_DIR,
    port: int = 8769,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """Create and return configured ThreadingHTTPServer instance."""

    eval_path = Path(evaluation_file)

    class ConfiguredHandler(ComparisonServerHandler):
        pass

    ConfiguredHandler.evaluation_file = eval_path
    ConfiguredHandler.dataset_dir = Path(dataset_dir)
    ConfiguredHandler.isolation_dir = Path(isolation_dir)
    ConfiguredHandler.group_locomotion_dir = Path(group_locomotion_dir)
    ConfiguredHandler.static_dir = Path(static_dir)
    ConfiguredHandler.locomotion_dir = Path(locomotion_dir)
    ConfiguredHandler.attack_dir = Path(attack_dir)
    ConfiguredHandler.chasing_dir = Path(chasing_dir)
    ConfiguredHandler.crops_dir = Path(crops_dir)
    ConfiguredHandler.orig_annotated_dir = Path(orig_annotated_dir)

    server = ThreadingHTTPServer((host, port), ConfiguredHandler)
    server.daemon_threads = True
    return server
