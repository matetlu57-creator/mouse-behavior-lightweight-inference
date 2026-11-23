#!/usr/bin/env python3
"""Generate a standalone, 100% offline-functional HTML failure report.

Inlines all comparison tables, kinematic attributions, failure diagnostics,
and timeline intervals into a self-contained single-page dashboard.
Zero CDN or external network dependencies.
Video files are loaded from relative ./videos/{dataset}_{clip_id}.mp4
or ./videos/{clip_id}.mp4.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ImportError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

LOGGER = logging.getLogger("offline_report_generator")

DEFAULT_EVAL_JSON = Path("outputs/reconciled_behavior_evaluation.json")
DEFAULT_COMP_JSON = Path("outputs/dataset_comparison_table.json")
DEFAULT_OUTPUT_HTML = Path("outputs/failure_report_offline/index.html")

BEHAVIOR_CHINESE: dict[str, str] = {
    "Approach": "接近",
    "Attack": "攻击",
    "Chasing": "追逐",
    "Snout-head_contact": "嗅闻头部",
    "Snout-rear_contact": "嗅闻尾部",
    "Together": "共处",
    "Static": "静止",
    "Locomotion": "移动",
    "Grooming": "理毛",
    "Rearing": "站立",
    "Isolation": "孤立",
    "Group locomotion": "群体同步运动",
    "Dispersal": "散开",
    "Social clustering": "社会聚集",
    "Huddling": "扎堆",
}

BEHAVIOR_LAYERS: dict[str, str] = {
    "Approach": "Social",
    "Attack": "Social",
    "Chasing": "Social",
    "Snout-head_contact": "Social",
    "Snout-rear_contact": "Social",
    "Together": "Social",
    "Static": "Individual",
    "Locomotion": "Individual",
    "Grooming": "Individual",
    "Rearing": "Individual",
    "Isolation": "Group",
    "Group locomotion": "Group",
    "Dispersal": "Group",
    "Social clustering": "Group",
    "Huddling": "Group",
}


def build_report_data(
    eval_json_path: Path,
    comp_json_path: Path,
) -> dict[str, Any]:
    """Parse and combine evaluation and comparison data into a compact bundle."""
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    comp_data = json.loads(comp_json_path.read_text(encoding="utf-8"))

    raw_clips = eval_data.get("clips", [])
    raw_failures = [c for c in raw_clips if not c.get("primary_correct")]

    # Process failure records
    failures_list = []
    seen_keys: set[str] = set()

    for item in raw_failures:
        cid = item["clip_id"]
        ds = item["dataset"]
        key = f"{ds}_{cid}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        beh = item["behavior"]
        beh_zh = BEHAVIOR_CHINESE.get(beh, beh)
        layer = item.get("layer") or BEHAVIOR_LAYERS.get(beh, "Social")
        fps = float(item.get("fps") or 30.0)
        fc = int(item.get("frame_count") or 0)
        dur = round(fc / max(fps, 1e-3), 2)

        cat = item.get("failure_category", "misclassified")
        if cat == "misclassified":
            cat_zh = "识别行为出错"
            cat_type = "misclassified"
        else:
            cat_zh = "行为未识别到"
            cat_type = "missed"

        subcat = item.get("failure_subcategory") or ""
        reason_zh = item.get("failure_reason_zh") or ""
        joint_reason_zh = item.get("joint_failure_reason_zh") or ""

        # Extract timeline series
        ts = item.get("timeline_series") or {}
        gt_interval = (
            ts.get("ground_truth_interval")
            or item.get("frame_range_half_open")
            or {"start": 0, "end": fc}
        )
        pred_interval = ts.get("predicted_interval") or item.get(
            "primary_predicted_range_half_open"
        )

        # Extract kinematics summary
        kin = ts.get("kinematics_summary") or {}

        # Competing candidates
        competing = item.get("top_competing_candidates") or []

        failures_list.append(
            {
                "id": key,
                "clip_id": cid,
                "dataset": ds,
                "behavior": beh,
                "behavior_zh": beh_zh,
                "layer": layer.capitalize(),
                "target_mouse_ids": item.get("target_mouse_ids") or [],
                "fps": fps,
                "frame_count": fc,
                "duration_sec": dur,
                "failure_category": cat_type,
                "failure_category_zh": cat_zh,
                "failure_subcategory": subcat,
                "failure_reason_zh": reason_zh,
                "joint_failure_reason_zh": joint_reason_zh,
                "primary_predicted_behavior": item.get("primary_predicted_behavior"),
                "ground_truth_interval": gt_interval,
                "predicted_interval": pred_interval,
                "in_overlap": item.get("in_overlap", False),
                "in_union_b_priority": item.get("in_union_b_priority", False),
                "in_union_a_priority": item.get("in_union_a_priority", False),
                "kinematics": kin,
                "competing_candidates": competing[:6],
                "video_url_ds": f"./videos/{ds}_{cid}.mp4",
                "video_url_generic": f"./videos/{cid}.mp4",
            }
        )

    # Sort failures: Union B-priority first, then behavior, then clip_id
    failures_list.sort(key=lambda x: (not x["in_union_b_priority"], x["behavior"], x["clip_id"]))

    return {
        "metadata": comp_data.get("metadata", {}),
        "reconciliation_summary": comp_data.get("reconciliation_summary", {}),
        "overall_summary": comp_data.get("overall_summary", {}),
        "behaviors": comp_data.get("behaviors", []),
        "failures": failures_list,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>小鼠行为识别评估与未通过视频分析报告 (离线静态版)</title>
  <style>
    :root {
      --bg-primary: #0b0f19;
      --bg-secondary: #111827;
      --bg-card: #1e293b;
      --bg-card-hover: #283548;
      --border-color: #334155;
      --border-subtle: #1e293b;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-blue: #38bdf8;
      --accent-indigo: #818cf8;
      --accent-emerald: #34d399;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #c084fc;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -2px rgba(0, 0, 0, 0.5);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -4px rgba(0, 0, 0, 0.6);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    .dashboard-container {
      max-width: 1720px;
      margin: 0 auto;
      padding: 24px 20px 60px 20px;
    }

    header.dashboard-header {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 24px 28px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-md);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    .header-titles h1 {
      font-size: 24px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 12px;
      letter-spacing: -0.02em;
    }

    .header-titles p {
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 6px;
    }

    .header-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: none;
    }

    .badge-blue { background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); border: 1px solid rgba(56, 189, 248, 0.35); }
    .badge-emerald { background: rgba(52, 211, 153, 0.15); color: var(--accent-emerald); border: 1px solid rgba(52, 211, 153, 0.35); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.35); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.35); }
    .badge-purple { background: rgba(192, 132, 252, 0.15); color: var(--accent-purple); border: 1px solid rgba(192, 132, 252, 0.35); }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }

    .metric-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 20px;
      box-shadow: var(--shadow-sm);
      position: relative;
      overflow: hidden;
    }

    .metric-card::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--accent-blue);
    }

    .metric-card.card-amber::before { background: var(--accent-amber); }
    .metric-card.card-rose::before { background: var(--accent-rose); }
    .metric-card.card-emerald::before { background: var(--accent-emerald); }

    .metric-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .metric-value {
      font-size: 28px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }

    .metric-subtext {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 6px;
      line-height: 1.4;
    }

    .section-box {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      margin-bottom: 24px;
      overflow: hidden;
      box-shadow: var(--shadow-md);
    }

    .section-header {
      padding: 18px 24px;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      user-select: none;
    }

    .section-title {
      font-size: 17px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .toggle-icon {
      font-size: 14px;
      color: var(--text-muted);
      transition: transform 0.2s ease;
    }

    .section-box.collapsed .section-content {
      display: none;
    }

    .section-box.collapsed .toggle-icon {
      transform: rotate(-90deg);
    }

    .section-content {
      padding: 24px;
    }

    .table-responsive {
      overflow-x: auto;
      border-radius: var(--radius-sm);
    }

    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }

    table.data-table th, table.data-table td {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-subtle);
    }

    table.data-table th {
      background: #1e293b;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11.5px;
      letter-spacing: 0.05em;
      position: sticky;
      top: 0;
      z-index: 10;
    }

    table.data-table tbody tr:hover {
      background: var(--bg-card-hover);
    }

    table.data-table tbody tr.row-total {
      background: #172554;
      font-weight: 700;
      border-top: 2px solid var(--accent-blue);
      border-bottom: 2px solid var(--accent-blue);
    }

    .acc-high { color: #4ade80; font-weight: 700; }
    .acc-normal { color: #38bdf8; }
    .acc-low { color: #fb7185; }

    .conflict-card {
      background: #1e293b;
      border: 1px solid #475569;
      border-left: 4px solid var(--accent-amber);
      border-radius: var(--radius-md);
      padding: 20px;
      margin-bottom: 10px;
    }

    .conflict-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      flex-wrap: wrap;
      gap: 10px;
    }

    .conflict-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 14px;
    }

    .conflict-col {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 14px;
    }

    .conflict-col h4 {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 14px;
      border-radius: var(--radius-sm);
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: all 0.15s ease;
    }

    .btn-primary {
      background: var(--accent-blue);
      color: #0b0f19;
    }

    .btn-primary:hover {
      background: #7dd3fc;
    }

    .btn-outline {
      background: transparent;
      color: var(--text-main);
      border: 1px solid var(--border-color);
    }

    .btn-outline:hover {
      background: var(--bg-card-hover);
      border-color: var(--text-muted);
    }

    .browser-layout {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
      align-items: start;
    }

    @media (max-width: 1080px) {
      .browser-layout {
        grid-template-columns: 1fr;
      }
    }

    .sidebar-panel {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 18px;
      box-shadow: var(--shadow-md);
      position: sticky;
      top: 20px;
      max-height: calc(100vh - 40px);
      display: flex;
      flex-direction: column;
    }

    .filter-header {
      margin-bottom: 14px;
    }

    .filter-header h3 {
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }

    .search-input {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      color: #fff;
      padding: 9px 12px;
      font-size: 13px;
      margin-bottom: 12px;
      outline: none;
    }

    .search-input:focus {
      border-color: var(--accent-blue);
    }

    .filter-pills-row {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
    }

    .filter-pill {
      flex: 1;
      padding: 6px 8px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
      background: var(--bg-card);
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }

    .filter-pill:hover {
      color: #fff;
      background: var(--bg-card-hover);
    }

    .filter-pill.active {
      background: var(--accent-blue);
      color: #0b0f19;
      border-color: var(--accent-blue);
    }

    .filter-pill.active.pill-rose {
      background: var(--accent-rose);
      color: #fff;
      border-color: var(--accent-rose);
    }

    .filter-pill.active.pill-amber {
      background: var(--accent-amber);
      color: #0b0f19;
      border-color: var(--accent-amber);
    }

    .filter-selects-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }

    .filter-select {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      color: var(--text-main);
      padding: 7px 10px;
      font-size: 12px;
      outline: none;
      cursor: pointer;
    }

    .filter-stats-bar {
      font-size: 12px;
      color: var(--text-dim);
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .failure-items-list {
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-right: 4px;
    }

    .failure-item-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 12px;
      cursor: pointer;
      transition: all 0.15s ease;
      position: relative;
    }

    .failure-item-card:hover {
      background: var(--bg-card-hover);
      border-color: #64748b;
    }

    .failure-item-card.selected {
      border-color: var(--accent-blue);
      background: #172554;
      box-shadow: 0 0 0 1px var(--accent-blue);
    }

    .item-card-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }

    .item-card-title {
      font-size: 13.5px;
      font-weight: 700;
      color: #fff;
    }

    .item-card-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      margin-top: 4px;
    }

    .item-card-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      font-size: 11px;
      color: var(--text-dim);
    }

    .viewer-panel {
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-md);
    }

    .viewer-empty {
      padding: 80px 20px;
      text-align: center;
      color: var(--text-muted);
    }

    .clip-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 18px;
      flex-wrap: wrap;
      gap: 12px;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 16px;
    }

    .clip-title-group h2 {
      font-size: 20px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .clip-title-group .clip-sub {
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 4px;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .clip-actions {
      display: flex;
      gap: 8px;
    }

    .cause-callout {
      border-radius: var(--radius-md);
      padding: 16px 20px;
      margin-bottom: 20px;
      border: 1px solid transparent;
    }

    .cause-callout.cause-misclassified {
      background: rgba(244, 63, 94, 0.10);
      border-color: rgba(244, 63, 94, 0.40);
    }

    .cause-callout.cause-missed {
      background: rgba(245, 158, 11, 0.10);
      border-color: rgba(245, 158, 11, 0.40);
    }

    .cause-title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .cause-misclassified .cause-title { color: #fda4af; }
    .cause-missed .cause-title { color: #fde68a; }

    .cause-body {
      font-size: 13.5px;
      line-height: 1.6;
      color: var(--text-main);
    }

    .video-container {
      background: #000;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--border-color);
      margin-bottom: 16px;
      position: relative;
    }

    video.player-video {
      width: 100%;
      height: auto;
      max-height: 540px;
      display: block;
      margin: 0 auto;
      background: #000;
    }

    .video-custom-controls {
      background: #111827;
      border-top: 1px solid var(--border-color);
      padding: 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }

    .ctrl-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .ctrl-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 10px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }

    .ctrl-btn:hover {
      background: var(--bg-card-hover);
      border-color: var(--accent-blue);
    }

    .time-readout {
      font-family: monospace;
      font-size: 12.5px;
      color: var(--accent-blue);
      font-weight: 600;
    }

    .timeline-wrapper {
      background: #0f172a;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px;
      margin-bottom: 22px;
    }

    .timeline-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .timeline-lanes {
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
    }

    .timeline-lane-row {
      display: grid;
      grid-template-columns: 120px 1fr;
      align-items: center;
      gap: 12px;
    }

    .lane-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-align: right;
    }

    .lane-track {
      height: 24px;
      background: #1e293b;
      border-radius: 4px;
      position: relative;
      cursor: pointer;
      overflow: hidden;
      border: 1px solid #334155;
    }

    .lane-bar {
      position: absolute;
      top: 2px;
      bottom: 2px;
      border-radius: 3px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 700;
      color: #fff;
      text-shadow: 0 1px 2px rgba(0,0,0,0.8);
      padding: 0 6px;
      white-space: nowrap;
      overflow: hidden;
    }

    .bar-gt {
      background: linear-gradient(90deg, #0284c7, #38bdf8);
      border: 1px solid #7dd3fc;
    }

    .bar-pred {
      background: linear-gradient(90deg, #b91c1c, #f43f5e);
      border: 1px solid #fda4af;
    }

    .bar-none {
      background: #334155;
      color: var(--text-dim);
      border: 1px dashed #64748b;
    }

    .scrubber-cursor {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #fbbf24;
      pointer-events: none;
      z-index: 5;
      box-shadow: 0 0 4px #fbbf24;
    }

    .diagnostics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }

    .diag-box {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 16px;
    }

    .diag-box h4 {
      font-size: 13.5px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 8px;
    }

    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12.5px;
      padding: 5px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .metric-row:last-child {
      border-bottom: none;
    }

    .m-key { color: var(--text-muted); }
    .m-val { color: var(--text-main); font-weight: 600; font-family: monospace; }

    .hidden { display: none !important; }
    .text-right { text-align: right; }
    .text-center { text-align: center; }
    .nowrap { white-space: nowrap; }
  </style>
</head>
<body>

<div class="dashboard-container">

  <!-- Header -->
  <header class="dashboard-header">
    <div class="header-titles">
      <h1>
        <span>🐭</span> 小鼠 15 类行为识别评测与未通过归因报告
      </h1>
      <p>
        离线自包含报告包 | 严格 Target-ID 联合评测口径 (AGENTS.md) | 双数据集全面核对
      </p>
    </div>
    <div class="header-badges">
      <span class="badge badge-emerald">全 15 类行为评测</span>
      <span class="badge badge-blue">姿态筛选: 已启用</span>
      <span class="badge badge-purple" id="header-git">Git HEAD</span>
      <span class="badge badge-amber">100% 离线免联网</span>
    </div>
  </header>

  <!-- Top Metric Cards -->
  <div class="metrics-grid">
    <div class="metric-card card-emerald">
      <div class="metric-label">
        <span>数据集核对规模</span>
        <span>A / B / 并集</span>
      </div>
      <div class="metric-value">
        <span>764</span>
        <span style="font-size: 16px; color: var(--text-muted); font-weight: 500;">段去重并集</span>
      </div>
      <div class="metric-subtext">
        A (补全集 429段) + B (清洗集 605段) | 重叠 270段 (99.63% 一致)
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-label">
        <span>并集 B 优先整体准确率</span>
        <span>Primary Acc</span>
      </div>
      <div class="metric-value">
        <span class="acc-high">85.99%</span>
        <span style="font-size: 16px; color: var(--text-muted); font-weight: 500;">657 / 764</span>
      </div>
      <div class="metric-subtext">
        清洗集 B 独占达 88.10% (533/605) | Strict Joint Top-1: 181/764 (23.7%)
      </div>
    </div>

    <div class="metric-card card-rose">
      <div class="metric-label">
        <span>未通过评测片段</span>
        <span>Primary Failures</span>
      </div>
      <div class="metric-value">
        <span style="color: var(--accent-rose);">107</span>
        <span style="font-size: 16px; color: var(--text-muted); font-weight: 500;">/ 764 并集 B</span>
      </div>
      <div class="metric-subtext">
        双集覆盖去重独立视频 113 段 (总 133 次评测未通过实例)
      </div>
    </div>

    <div class="metric-card card-amber">
      <div class="metric-label">
        <span>失败根因二分类统计</span>
        <span>并集 B (107 段)</span>
      </div>
      <div class="metric-value" style="font-size: 22px;">
        <span style="color: var(--accent-amber);">未识别到: 52</span>
        <span style="color: var(--text-dim); margin: 0 4px;">|</span>
        <span style="color: var(--accent-rose);">出错: 55</span>
      </div>
      <div class="metric-subtext">
        行为未识别到 (52段, 48.6%) | 识别行为出错 (55段, 51.4%)
      </div>
    </div>
  </div>

  <!-- Section 1: 15-Behavior Reconciliation Comparison Table -->
  <div class="section-box" id="section-reconciliation">
    <div class="section-header" onclick="toggleSection('section-reconciliation')">
      <div class="section-title">
        <span>📊</span> 全 15 类行为多数据集核对与准确率对比总表
      </div>
      <div class="toggle-icon">▼</div>
    </div>
    <div class="section-content">
      <div class="table-responsive">
        <table class="data-table" id="table-reconciliation">
          <thead>
            <tr>
              <th>行为类别 (中 / 英)</th>
              <th>层级</th>
              <th class="text-right">数据集 A (429)<br><span style="font-weight:400;">命中/P/R</span></th>
              <th class="text-right">A Strict Joint<br><span style="font-weight:400;">命中/P/R</span></th>
              <th class="text-right">数据集 B (605)<br><span style="font-weight:400;">命中/P/R</span></th>
              <th class="text-right">B Strict Joint<br><span style="font-weight:400;">命中/P/R</span></th>
              <th class="text-right">重叠 A轨 (270)</th>
              <th class="text-right">重叠 B轨 (270)</th>
              <th class="text-right">并集 B优先 (764)<br><span style="font-weight:400;">命中/P/R</span></th>
              <th class="text-right">并集 B Strict<br><span style="font-weight:400;">命中/P/R</span></th>
              <th class="text-right">并集 A优先 (764)<br><span style="font-weight:400;">命中/P/R</span></th>
              <th>专项评测方法</th>
            </tr>
          </thead>
          <tbody id="reconciliation-tbody">
            <!-- Injected by JavaScript -->
          </tbody>
        </table>
      </div>
      <div style="margin-top: 18px; font-weight: 600; color: var(--accent-blue); font-size: 13px;">
        统一 15 类同台竞争模型 (pf_target，全片段无类别提示) 跨类精确率 (Precision) / 召回率 (Recall) / F1 对照表
      </div>
      <div class="table-responsive" style="margin-top: 8px;">
        <table class="data-table" id="table-unified-pr">
          <thead>
            <tr>
              <th>行为类别 (中 / 英)</th>
              <th class="text-right">数据集 A (429) — P / R / F1</th>
              <th class="text-right">数据集 B (605) — P / R / F1</th>
              <th class="text-right">并集 B优先 (764) — P / R / F1</th>
            </tr>
          </thead>
          <tbody id="unified-pr-tbody">
            <!-- Injected by JavaScript -->
          </tbody>
        </table>
      </div>
      <div style="font-size: 12px; color: var(--text-dim); margin-top: 14px; line-height: 1.5;">
        * 注：1. 分项主指标中，P = 精确率 (TP / 该行为总预测数)，R = 召回率 (TP / 该行为真值总数)；其中 9 类非双鼠配对行为仅在所属真值片段上单独评测（类内 P 恒为 100%），真实跨类误报 P 请参考下方 pf_target 统一模型表。<br>
        2. 数据集 B 上 14 类单项严格准确率均 ≥ 80.0% (站立 83.3%, 理毛 83.3%；扎堆 A1 候选 >=50% 覆盖达 89.4% [59/66]，严格完整目标 ID 匹配 A2 为 62.1% [41/66])。<br>
        3. Strict Joint Top-1：盲评排名前 1 的行为与包含的完整目标 ID 同时精确命中（移除同层双标签互换兼容）。
      </div>
    </div>
  </div>

  <!-- Section 2: Conflict Clip Deep Dive -->
  <div class="section-box" id="section-conflict">
    <div class="section-header" onclick="toggleSection('section-conflict')">
      <div class="section-title">
        <span>🔍</span> 冲突样本深度排查与归因分析：<code>52512294eac95441035cb075</code>
      </div>
      <div class="toggle-icon">▼</div>
    </div>
    <div class="section-content">
      <div class="conflict-card">
        <div class="conflict-header">
          <div>
            <h3 style="font-size: 16px; color: #fff; margin-bottom: 4px;">
              唯一标签冲突样本：嗅闻头部 (Snout-head) vs 嗅闻尾部 (Snout-rear)
            </h3>
            <p style="font-size: 13px; color: var(--text-muted);">
              源视频: <code>WIN_20260822_14_13_59_Pro_part001.mp4 (frames 3114-3153, 40 frames, 1.34s, 29.95 fps)</code> | 目标小鼠 ID: <code>[71, 230]</code>
            </p>
          </div>
          <button class="btn btn-primary" onclick="focusConflictClip()">
            <span>▶</span> 在播放器中查看该冲突样本
          </button>
        </div>

        <div class="conflict-grid">
          <div class="conflict-col">
            <h4 style="color: var(--accent-rose);">
              <span>⚠️</span> 数据集 A 表现 (误报 Attack & 标为 Snout-head)
            </h4>
            <div style="font-size: 13px; line-height: 1.6; color: var(--text-muted);">
              • <strong>真值标签</strong>: <code>Snout-head_contact</code> (嗅闻头部)<br>
              • <strong>人工修复异常</strong>: 存在 14 处轨迹修复 (6 帧线性插值 + 8 帧 SAM2 修复)<br>
              • <strong>合成速度突跳</strong>: 小鼠 71 速度突跳至 <strong>51.43 cm/s</strong><br>
              • <strong>错误判定</strong>: 触发 32 帧 Attack 误检抢占 Top-1，导致主行为未通过
            </div>
          </div>

          <div class="conflict-col">
            <h4 style="color: var(--accent-emerald);">
              <span>✅</span> 数据集 B 表现 (物理真值 Snout-rear 100% 正确)
            </h4>
            <div style="font-size: 13px; line-height: 1.6; color: var(--text-muted);">
              • <strong>真值标签</strong>: <code>Snout-rear_contact</code> (嗅闻尾部)<br>
              • <strong>干净轨迹</strong>: 0 人工插值，保持原始连续视觉追踪<br>
              • <strong>运动平稳</strong>: 速度平稳真实 (最大 <strong>8.21 cm/s</strong>)<br>
              • <strong>正确判定</strong>: 正确触发 Snout-rear_contact 30 帧 (100% 正确命中)
            </div>
          </div>

          <div class="conflict-col">
            <h4 style="color: var(--accent-blue);">
              <span>📏</span> 几何间距实测物理证据
            </h4>
            <div style="font-size: 13px; line-height: 1.6; color: var(--text-muted);">
              • <strong>鼻-尾实测间距</strong>: 最小 <strong>1.131 cm</strong> (中位数 1.455 cm) <br>
              • <strong>鼻-头实测间距</strong>: 最小 <strong>4.631 cm</strong> (中位数 7.230 cm) <br>
              • <strong>接触判定阈值</strong>: 2.23 cm - 4.38 cm<br>
              • <strong>仲裁结论</strong>: 物理接触点确为鼻-尾接触，数据集 B 标签正确。并集采用 B 优先策略。
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Section 3: Interactive Failure Review Browser -->
  <div class="section-box" id="section-browser">
    <div class="section-header" onclick="toggleSection('section-browser')">
      <div class="section-title">
        <span>🎬</span> 未通过视频片段交互审查器 (全离线视频高亮 + 交互式时间轴 + 运动学归因)
      </div>
      <div class="toggle-icon">▼</div>
    </div>
    <div class="section-content" style="padding: 18px;">

      <div class="browser-layout">

        <!-- Left Sidebar: Filters & List -->
        <aside class="sidebar-panel">
          <div class="filter-header">
            <h3>
              <span>筛选过滤</span>
              <span class="badge badge-blue" id="sidebar-count-badge">0 / 0</span>
            </h3>

            <!-- Search -->
            <input type="text" id="search-input" class="search-input" placeholder="搜索 Clip ID、行为、小鼠ID、原因..." oninput="handleFilterChange()">

            <!-- Failure Category Pills -->
            <div class="filter-pills-row">
              <div class="filter-pill active" id="pill-all" onclick="setCategoryFilter('all')">全部</div>
              <div class="filter-pill pill-rose" id="pill-misclassified" onclick="setCategoryFilter('misclassified')">识别出错</div>
              <div class="filter-pill pill-amber" id="pill-missed" onclick="setCategoryFilter('missed')">未识别到</div>
            </div>

            <!-- Dataset & Behavior Selects -->
            <div class="filter-selects-row">
              <select id="select-dataset" class="filter-select" onchange="handleFilterChange()">
                <option value="all">所有数据集</option>
                <option value="union_b">并集 B优先 (107)</option>
                <option value="A">数据集 A (61)</option>
                <option value="B">数据集 B (72)</option>
              </select>

              <select id="select-behavior" class="filter-select" onchange="handleFilterChange()">
                <option value="all">所有行为类别</option>
                <!-- Injected by JavaScript -->
              </select>
            </div>

            <div class="filter-stats-bar">
              <span id="filter-status-text">正在加载片段...</span>
              <button class="btn btn-outline" style="padding: 2px 8px; font-size: 11px;" onclick="resetFilters()">重置筛选</button>
            </div>
          </div>

          <!-- Scrollable Failure Clips List -->
          <div class="failure-items-list" id="failure-items-list">
            <!-- Injected by JavaScript -->
          </div>
        </aside>

        <!-- Right Main: Detail View -->
        <main class="viewer-panel" id="viewer-panel">
          <div class="viewer-empty" id="viewer-empty">
            <div style="font-size: 40px; margin-bottom: 12px;">👈</div>
            <h3>请在左侧列表中选择未通过视频片段</h3>
            <p style="font-size: 13px; color: var(--text-dim); margin-top: 6px;">支持按识别错误分类、数据集、行为类型即时检索</p>
          </div>

          <div id="viewer-content" class="hidden">
            <!-- Header -->
            <div class="clip-header">
              <div class="clip-title-group">
                <h2 id="view-beh-title">
                  <span id="view-beh-zh">接近</span>
                  <span id="view-beh-en" style="font-size: 14px; color: var(--text-muted); font-weight: 500;">(Approach)</span>
                  <span id="view-layer-badge" class="badge badge-purple">Social</span>
                  <span id="view-ds-badge" class="badge badge-blue">数据集 B</span>
                </h2>
                <div class="clip-sub">
                  <span>Clip ID: <code id="view-clip-id" style="color: var(--accent-blue);">--</code></span>
                  <button class="btn btn-outline" style="padding: 2px 8px; font-size: 11px;" onclick="copyClipId()">复制 ID</button>
                  <span>时长: <strong id="view-duration">0.0s</strong> (<span id="view-frames">0</span> 帧, <span id="view-fps">30</span> fps)</span>
                </div>
              </div>
              <div class="clip-actions">
                <span id="view-targets-container" style="display: flex; gap: 6px; align-items: center;">
                  <!-- Target mouse ID badges -->
                </span>
              </div>
            </div>

            <!-- Failure Categorization Banner -->
            <div class="cause-callout" id="view-cause-box">
              <div class="cause-title">
                <span id="view-cause-badge" class="badge badge-rose">【识别行为出错】</span>
                <span id="view-subcat-badge" class="badge badge-purple" style="font-size: 11px;">--</span>
                <span style="font-size: 12px; color: var(--text-dim);" id="view-union-badge">并集 B 优先样本</span>
              </div>
              <div class="cause-body" id="view-cause-text">
                --
              </div>
              <div style="margin-top: 10px; font-size: 12px; color: var(--text-dim);" id="view-joint-cause-box">
                <strong>严格盲评 Top-1 竞逐情况：</strong> <span id="view-joint-cause-text">--</span>
              </div>
            </div>

            <!-- Video Player -->
            <div class="video-container">
              <video id="player-video" class="player-video" playsinline preload="metadata">
                <source id="video-source-ds" src="" type="video/mp4">
                <source id="video-source-generic" src="" type="video/mp4">
                您的浏览器不支持 HTML5 视频播放。
              </video>
              <div class="video-custom-controls">
                <div class="ctrl-group">
                  <button class="ctrl-btn" id="btn-play-pause" onclick="togglePlay()">播放 / 暂停</button>
                  <button class="ctrl-btn" onclick="stepFrame(-1)">◀ 上一帧</button>
                  <button class="ctrl-btn" onclick="stepFrame(1)">下一帧 ▶</button>
                </div>
                <div class="time-readout" id="time-readout">帧: 0 / 0 (0.00s)</div>
                <div class="ctrl-group">
                  <span style="font-size: 11.5px; color: var(--text-muted);">倍速:</span>
                  <select id="select-speed" class="filter-select" style="width: 75px; padding: 4px;" onchange="setPlaySpeed(this.value)">
                    <option value="0.25">0.25x</option>
                    <option value="0.5">0.5x</option>
                    <option value="1.0" selected>1.0x</option>
                    <option value="2.0">2.0x</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Interactive Dual-Lane Timeline Scrubber -->
            <div class="timeline-wrapper">
              <div class="timeline-title">
                <span>⏱️ 双轨时序交互时间轴 (点击跳转指定帧)</span>
                <span style="font-size: 11px; color: var(--text-dim);">快捷键: 空格播放/暂停，左右箭头逐帧步进</span>
              </div>
              <div class="timeline-lanes">
                <!-- Lane 1: Ground Truth -->
                <div class="timeline-lane-row">
                  <div class="lane-label">真值标注区间</div>
                  <div class="lane-track" id="track-gt" onclick="seekByTrackClick(event, 'gt')">
                    <div class="lane-bar bar-gt" id="bar-gt" style="left: 0%; width: 100%;">
                      标注行为区间: 0 - 0 帧
                    </div>
                    <div class="scrubber-cursor" id="cursor-gt" style="left: 0%;"></div>
                  </div>
                </div>

                <!-- Lane 2: Prediction / Candidate -->
                <div class="timeline-lane-row">
                  <div class="lane-label">算法预测候选</div>
                  <div class="lane-track" id="track-pred" onclick="seekByTrackClick(event, 'pred')">
                    <div class="lane-bar bar-pred" id="bar-pred" style="left: 0%; width: 0%;">
                      未识别到有效候选
                    </div>
                    <div class="scrubber-cursor" id="cursor-pred" style="left: 0%;"></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Kinematics & Competing Attribution Details -->
            <div class="diagnostics-grid">
              <!-- Kinematics Box -->
              <div class="diag-box">
                <h4><span>📐</span> 关键运动学实测指标 (Kinematics)</h4>
                <div class="metric-row">
                  <span class="m-key">两鼠中心距离 (Min / Med / Max)</span>
                  <span class="m-val" id="kin-center-dist">-- / -- / -- cm</span>
                </div>
                <div class="metric-row">
                  <span class="m-key">鼻-头间距 (Min / Med)</span>
                  <span class="m-val" id="kin-snout-head">-- / -- cm</span>
                </div>
                <div class="metric-row">
                  <span class="m-key">鼻-尾间距 (Min / Med)</span>
                  <span class="m-val" id="kin-snout-rear">-- / -- cm</span>
                </div>
                <div class="metric-row">
                  <span class="m-key">目标运动速度 (Med / Max)</span>
                  <span class="m-val" id="kin-speed">-- / -- cm/s</span>
                </div>
                <div class="metric-row">
                  <span class="m-key">两鼠相对闭合速度 (Med / Max)</span>
                  <span class="m-val" id="kin-closing-speed">-- / -- cm/s</span>
                </div>
                <div class="metric-row">
                  <span class="m-key">目标观测帧率比 / 共现比</span>
                  <span class="m-val" id="kin-obs-ratio">-- / --</span>
                </div>
                <div class="metric-row">
                  <span class="m-key">关键点缺失率 (鼻部 / 尾根)</span>
                  <span class="m-val" id="kin-missing-kps">-- / --</span>
                </div>
              </div>

              <!-- Competing Candidates Box -->
              <div class="diag-box">
                <h4><span>⚔️</span> 竞争行为候选项与仲裁 (Arbitration Rivals)</h4>
                <div id="competing-candidates-list" style="display: flex; flex-direction: column; gap: 8px;">
                  <!-- Injected by JavaScript -->
                </div>
              </div>
            </div>

          </div>
        </main>

      </div>

    </div>
  </div>

</div>

<!-- Embedded JSON Data -->
<script>
const REPORT_DATA = /* __REPORT_DATA_PLACEHOLDER__ */ {};

// Global App State
let activeClip = null;
let filteredClips = [];
let currentCategoryFilter = 'all';
let currentDatasetFilter = 'all';
let currentBehaviorFilter = 'all';
let currentSearchQuery = '';

// DOM Elements
const videoEl = document.getElementById('player-video');
const timeReadout = document.getElementById('time-readout');
const cursorGt = document.getElementById('cursor-gt');
const cursorPred = document.getElementById('cursor-pred');

function initDashboard() {
  // 1. Populate Git commit
  const gitHead = (REPORT_DATA.metadata && REPORT_DATA.metadata.git_head) || 'main';
  document.getElementById('header-git').innerText = 'Git: ' + gitHead.slice(0, 8);

  // 2. Populate 15-Behavior Reconciliation Table
  renderReconciliationTable();

  // 3. Populate Behavior filter dropdown
  populateBehaviorDropdown();

  // 4. Initial filter run
  filterClips();

  // 5. Select first clip by default if available
  if (filteredClips.length > 0) {
    selectClip(filteredClips[0].id);
  }

  // 6. Bind video events
  videoEl.addEventListener('timeupdate', onVideoTimeUpdate);
  videoEl.addEventListener('play', function() { document.getElementById('btn-play-pause').innerText = '暂停'; });
  videoEl.addEventListener('pause', function() { document.getElementById('btn-play-pause').innerText = '播放'; });

  // 7. Bind Keyboard Shortcuts
  window.addEventListener('keydown', onKeyDown);
}

function toggleSection(sectionId) {
  const el = document.getElementById(sectionId);
  if (el) {
    el.classList.toggle('collapsed');
  }
}

function fmtRatePct(val, digits) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  return (val * 100).toFixed(digits === undefined ? 1 : digits) + '%';
}

function fmtPrimaryPR(d, highlightClass) {
  if (!d || !d.support) return '—';
  const accStr = d.primary_correct + '/' + d.support + ' (' + fmtRatePct(d.primary_accuracy) + ')';
  const mainSpan = highlightClass ? '<strong class="' + highlightClass + '">' + accStr + '</strong>' : accStr;
  const pStr = fmtRatePct(d.primary_precision);
  const rStr = fmtRatePct(d.primary_recall !== undefined ? d.primary_recall : d.primary_accuracy);
  const pred = d.primary_predicted_count !== undefined ? d.primary_predicted_count : d.primary_correct;
  return mainSpan + '<br><span style="font-size:11px; color: var(--text-muted);">P=' + pStr + ' (' + d.primary_correct + '/' + pred + ') · R=' + rStr + '</span>';
}

function fmtJointPR(d) {
  if (!d || !d.support) return '—';
  const accStr = d.pf_joint_correct + '/' + d.support + ' (' + fmtRatePct(d.pf_joint_accuracy) + ')';
  const pStr = fmtRatePct(d.pf_joint_precision);
  const rStr = fmtRatePct(d.pf_joint_recall !== undefined ? d.pf_joint_recall : d.pf_joint_accuracy);
  const pred = d.pf_joint_predicted_count !== undefined ? d.pf_joint_predicted_count : 0;
  return accStr + '<br><span style="font-size:11px; color: var(--text-dim);">P=' + pStr + ' (' + d.pf_joint_correct + '/' + pred + ') · R=' + rStr + '</span>';
}

function fmtTargetPR(d) {
  if (!d || !d.support) return '—';
  const tp = d.pf_target_correct !== undefined ? d.pf_target_correct : 0;
  const pred = d.pf_target_predicted_count !== undefined ? d.pf_target_predicted_count : 0;
  return 'P=<strong>' + fmtRatePct(d.pf_target_precision) + '</strong> (' + tp + '/' + pred + ') · R=<strong>' + fmtRatePct(d.pf_target_recall) + '</strong> (' + tp + '/' + d.support + ') · F1=' + fmtRatePct(d.pf_target_f1);
}

function renderReconciliationTable() {
  const tbody = document.getElementById('reconciliation-tbody');
  tbody.innerHTML = '';

  const behaviors = REPORT_DATA.behaviors || [];
  behaviors.forEach(function(b) {
    const tr = document.createElement('tr');

    const aPrimary = fmtPrimaryPR(b.dataset_a, null);
    const aStrict = fmtJointPR(b.dataset_a);

    let bPrimaryClass = 'acc-normal';
    if (b.dataset_b && b.dataset_b.primary_accuracy >= 0.80) bPrimaryClass = 'acc-high';
    else if (b.dataset_b && b.dataset_b.primary_accuracy < 0.70) bPrimaryClass = 'acc-low';

    const bPrimary = fmtPrimaryPR(b.dataset_b, bPrimaryClass);
    const bStrict = fmtJointPR(b.dataset_b);

    const ovA = b.overlap_a && b.overlap_a.support ? (b.overlap_a.primary_correct + '/' + b.overlap_a.support + ' (' + fmtRatePct(b.overlap_a.primary_accuracy) + ')') : '—';
    const ovB = b.overlap_b && b.overlap_b.support ? (b.overlap_b.primary_correct + '/' + b.overlap_b.support + ' (' + fmtRatePct(b.overlap_b.primary_accuracy) + ')') : '—';

    const uB = fmtPrimaryPR(b.union_b_priority, null);
    const uBStrict = fmtJointPR(b.union_b_priority);
    const uA = fmtPrimaryPR(b.union_a_priority, null);

    tr.innerHTML = '<td><strong>' + b.behavior_zh + '</strong> <span style="color: var(--text-muted); font-size: 11px;">(' + b.behavior + ')</span></td>' +
      '<td><span class="badge badge-purple" style="font-size: 11px;">' + b.layer + '</span></td>' +
      '<td class="text-right nowrap">' + aPrimary + '</td>' +
      '<td class="text-right nowrap" style="color: var(--text-dim);">' + aStrict + '</td>' +
      '<td class="text-right nowrap">' + bPrimary + '</td>' +
      '<td class="text-right nowrap" style="color: var(--text-dim);">' + bStrict + '</td>' +
      '<td class="text-right nowrap">' + ovA + '</td>' +
      '<td class="text-right nowrap">' + ovB + '</td>' +
      '<td class="text-right nowrap">' + uB + '</td>' +
      '<td class="text-right nowrap" style="color: var(--text-dim);">' + uBStrict + '</td>' +
      '<td class="text-right nowrap">' + uA + '</td>' +
      '<td><code style="font-size: 11px; color: var(--accent-blue);">' + b.primary_method + '</code></td>';
    tbody.appendChild(tr);
  });

  // Total Summary Row
  const ovSummary = REPORT_DATA.overall_summary || {};
  const trTotal = document.createElement('tr');
  trTotal.className = 'row-total';

  const sumA = ovSummary.dataset_a || {};
  const sumB = ovSummary.dataset_b || {};
  const sumOvA = ovSummary.overlap_a || {};
  const sumOvB = ovSummary.overlap_b || {};
  const sumUB = ovSummary.union_b_priority || {};
  const sumUA = ovSummary.union_a_priority || {};

  function fmtTot(s, prefix, highlightClass) {
    const tp = s[prefix + '_correct'];
    const n = s.total_clips;
    const acc = s[prefix + '_accuracy'];
    const mp = fmtRatePct(s[prefix + '_micro_precision'], 2);
    const mr = fmtRatePct(s[prefix + '_micro_recall'], 2);
    const macP = fmtRatePct(s[prefix + '_macro_precision'], 2);
    const macR = fmtRatePct(s[prefix + '_macro_recall'], 2);
    const clsAttr = highlightClass ? ' class="' + highlightClass + '"' : '';
    return '<strong' + clsAttr + '>' + tp + '/' + n + ' (' + fmtRatePct(acc, 2) + ')</strong>' +
      '<br><span style="font-size:11px; color: var(--text-muted);">Micro P=' + mp + ' · R=' + mr + '<br>Macro P=' + macP + ' · R=' + macR + '</span>';
  }

  trTotal.innerHTML = '<td><strong>总体合计 (TOTAL)</strong></td>' +
    '<td><span class="badge badge-blue">全部15类</span></td>' +
    '<td class="text-right nowrap">' + fmtTot(sumA, 'primary', null) + '</td>' +
    '<td class="text-right nowrap">' + fmtTot(sumA, 'pf_joint', null) + '</td>' +
    '<td class="text-right nowrap">' + fmtTot(sumB, 'primary', 'acc-high') + '</td>' +
    '<td class="text-right nowrap">' + fmtTot(sumB, 'pf_joint', null) + '</td>' +
    '<td class="text-right nowrap">' + sumOvA.primary_correct + '/' + sumOvA.total_clips + ' (' + fmtRatePct(sumOvA.primary_accuracy, 1) + ')</td>' +
    '<td class="text-right nowrap">' + sumOvB.primary_correct + '/' + sumOvB.total_clips + ' (' + fmtRatePct(sumOvB.primary_accuracy, 1) + ')</td>' +
    '<td class="text-right nowrap">' + fmtTot(sumUB, 'primary', null) + '</td>' +
    '<td class="text-right nowrap">' + fmtTot(sumUB, 'pf_joint', null) + '</td>' +
    '<td class="text-right nowrap">' + fmtTot(sumUA, 'primary', null) + '</td>' +
    '<td><strong>Micro / Macro P & R</strong></td>';
  tbody.appendChild(trTotal);

  const unifiedBody = document.getElementById('unified-pr-tbody');
  if (unifiedBody) {
    unifiedBody.innerHTML = '';
    behaviors.forEach(function(b) {
      const trU = document.createElement('tr');
      trU.innerHTML = '<td><strong>' + b.behavior_zh + '</strong> <span style="color: var(--text-muted); font-size: 11px;">(' + b.behavior + ')</span></td>' +
        '<td class="text-right nowrap">' + fmtTargetPR(b.dataset_a) + '</td>' +
        '<td class="text-right nowrap">' + fmtTargetPR(b.dataset_b) + '</td>' +
        '<td class="text-right nowrap">' + fmtTargetPR(b.union_b_priority) + '</td>';
      unifiedBody.appendChild(trU);
    });
    const trUTot = document.createElement('tr');
    trUTot.className = 'row-total';
    function fmtUTot(s) {
      return 'Micro P=<strong>' + fmtRatePct(s.pf_target_micro_precision, 2) + '</strong> · R=<strong>' + fmtRatePct(s.pf_target_micro_recall, 2) + '</strong>' +
        '<br><span style="font-size:11px; color: var(--text-muted);">Macro P=' + fmtRatePct(s.pf_target_macro_precision, 2) + ' · R=' + fmtRatePct(s.pf_target_macro_recall, 2) + '</span>';
    }
    trUTot.innerHTML = '<td><strong>总体合计 (Micro / Macro)</strong></td>' +
      '<td class="text-right nowrap">' + fmtUTot(sumA) + '</td>' +
      '<td class="text-right nowrap">' + fmtUTot(sumB) + '</td>' +
      '<td class="text-right nowrap">' + fmtUTot(sumUB) + '</td>';
    unifiedBody.appendChild(trUTot);
  }
}

function populateBehaviorDropdown() {
  const select = document.getElementById('select-behavior');
  const counts = {};
  (REPORT_DATA.failures || []).forEach(function(f) {
    counts[f.behavior] = (counts[f.behavior] || 0) + 1;
  });

  const behaviors = Object.keys(counts).sort();
  behaviors.forEach(function(b) {
    const opt = document.createElement('option');
    const zh = BEHAVIOR_CHINESE[b] || b;
    opt.value = b;
    opt.innerText = zh + ' (' + b + ') [' + counts[b] + ']';
    select.appendChild(opt);
  });
}

function setCategoryFilter(cat) {
  currentCategoryFilter = cat;
  document.getElementById('pill-all').classList.toggle('active', cat === 'all');
  document.getElementById('pill-misclassified').classList.toggle('active', cat === 'misclassified');
  document.getElementById('pill-missed').classList.toggle('active', cat === 'missed');
  handleFilterChange();
}

function handleFilterChange() {
  currentSearchQuery = document.getElementById('search-input').value.trim().toLowerCase();
  currentDatasetFilter = document.getElementById('select-dataset').value;
  currentBehaviorFilter = document.getElementById('select-behavior').value;

  filterClips();
}

function filterClips() {
  const allFailures = REPORT_DATA.failures || [];

  filteredClips = allFailures.filter(function(c) {
    // 1. Category Filter
    if (currentCategoryFilter !== 'all' && c.failure_category !== currentCategoryFilter) {
      return false;
    }

    // 2. Dataset Filter
    if (currentDatasetFilter === 'union_b' && !c.in_union_b_priority) {
      return false;
    } else if (currentDatasetFilter === 'A' && c.dataset !== 'A') {
      return false;
    } else if (currentDatasetFilter === 'B' && c.dataset !== 'B') {
      return false;
    }

    // 3. Behavior Filter
    if (currentBehaviorFilter !== 'all' && c.behavior !== currentBehaviorFilter) {
      return false;
    }

    // 4. Search Query
    if (currentSearchQuery) {
      const matchId = c.clip_id.toLowerCase().includes(currentSearchQuery);
      const matchBeh = c.behavior.toLowerCase().includes(currentSearchQuery) || c.behavior_zh.includes(currentSearchQuery);
      const matchReason = (c.failure_reason_zh || '').toLowerCase().includes(currentSearchQuery);
      const matchMouse = (c.target_mouse_ids || []).some(function(tid) { return tid.toString().includes(currentSearchQuery); });
      if (!matchId && !matchBeh && !matchReason && !matchMouse) {
        return false;
      }
    }

    return true;
  });

  // Update Stats
  document.getElementById('sidebar-count-badge').innerText = filteredClips.length + ' / ' + allFailures.length;
  document.getElementById('filter-status-text').innerText = '已筛选出 ' + filteredClips.length + ' 个未通过视频';

  // Render Sidebar List
  renderSidebarList();

  // If current active clip is not in filtered list, select the first one
  if (filteredClips.length > 0) {
    if (!activeClip || !filteredClips.some(function(c) { return c.id === activeClip.id; })) {
      selectClip(filteredClips[0].id);
    }
  } else {
    activeClip = null;
    document.getElementById('viewer-empty').classList.remove('hidden');
    document.getElementById('viewer-content').classList.add('hidden');
  }
}

function renderSidebarList() {
  const listEl = document.getElementById('failure-items-list');
  listEl.innerHTML = '';

  filteredClips.forEach(function(item) {
    const card = document.createElement('div');
    const isSelected = activeClip && activeClip.id === item.id;
    card.className = 'failure-item-card ' + (isSelected ? 'selected' : '');
    card.id = 'card-' + item.id;
    card.onclick = function() { selectClip(item.id); };

    const isMisclassified = item.failure_category === 'misclassified';
    const catBadgeClass = isMisclassified ? 'badge-rose' : 'badge-amber';
    const catZh = isMisclassified ? '识别出错' : '未识别到';
    const dsBadgeClass = item.dataset === 'A' ? 'badge-blue' : 'badge-emerald';

    card.innerHTML = '<div class="item-card-top">' +
      '<span class="item-card-title">' + item.behavior_zh + ' <span style="font-size: 11px; color: var(--text-muted);">(' + item.behavior + ')</span></span>' +
      '<span class="badge ' + catBadgeClass + '" style="font-size: 10.5px;">' + catZh + '</span>' +
      '</div>' +
      '<div class="item-card-desc">' + (item.failure_reason_zh || '') + '</div>' +
      '<div class="item-card-meta">' +
      '<span class="badge ' + dsBadgeClass + '" style="font-size: 10px; padding: 2px 6px;">集 ' + item.dataset + '</span>' +
      '<span>ID: ' + item.clip_id.slice(0, 8) + '...</span>' +
      '<span>目标: [' + item.target_mouse_ids.join(', ') + ']</span>' +
      '<span>' + item.duration_sec + 's</span>' +
      '</div>';
    listEl.appendChild(card);
  });
}

function selectClip(recordId) {
  const found = (REPORT_DATA.failures || []).find(function(c) { return c.id === recordId; });
  if (!found) return;

  activeClip = found;

  // Highlight in sidebar
  document.querySelectorAll('.failure-item-card').forEach(function(el) { el.classList.remove('selected'); });
  const activeCard = document.getElementById('card-' + recordId);
  if (activeCard) {
    activeCard.classList.add('selected');
    activeCard.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // Show Viewer Content
  document.getElementById('viewer-empty').classList.add('hidden');
  document.getElementById('viewer-content').classList.remove('hidden');

  // Populate Header
  document.getElementById('view-beh-zh').innerText = found.behavior_zh;
  document.getElementById('view-beh-en').innerText = '(' + found.behavior + ')';
  document.getElementById('view-layer-badge').innerText = found.layer;
  document.getElementById('view-ds-badge').innerText = '数据集 ' + found.dataset;
  document.getElementById('view-ds-badge').className = found.dataset === 'A' ? 'badge badge-blue' : 'badge badge-emerald';
  document.getElementById('view-clip-id').innerText = found.clip_id;
  document.getElementById('view-duration').innerText = found.duration_sec + 's';
  document.getElementById('view-frames').innerText = found.frame_count;
  document.getElementById('view-fps').innerText = found.fps.toFixed(1);

  // Target IDs badges
  const targetsCont = document.getElementById('view-targets-container');
  targetsCont.innerHTML = '<span style="font-size: 12px; color: var(--text-muted); margin-right: 4px;">目标小鼠:</span>';
  const colors = ['#f59e0b', '#06b6d4', '#10b981', '#ec4899'];
  found.target_mouse_ids.forEach(function(tid, idx) {
    const pill = document.createElement('span');
    pill.className = 'badge';
    const c = colors[idx % colors.length];
    pill.style.background = c + '26';
    pill.style.color = c;
    pill.style.border = '1px solid ' + c + '66';
    pill.innerText = 'ID: ' + tid;
    targetsCont.appendChild(pill);
  });

  // Failure Cause Callout
  const causeBox = document.getElementById('view-cause-box');
  const isMis = found.failure_category === 'misclassified';
  causeBox.className = 'cause-callout ' + (isMis ? 'cause-misclassified' : 'cause-missed');

  const causeBadge = document.getElementById('view-cause-badge');
  causeBadge.className = isMis ? 'badge badge-rose' : 'badge badge-amber';
  causeBadge.innerText = '【' + found.failure_category_zh + '】';

  document.getElementById('view-subcat-badge').innerText = found.failure_subcategory || 'heuristic_rule';
  document.getElementById('view-union-badge').innerText = found.in_union_b_priority ? '并集 B 优先样本' : '数据集独占样本';
  document.getElementById('view-cause-text').innerText = found.failure_reason_zh;

  if (found.joint_failure_reason_zh) {
    document.getElementById('view-joint-cause-box').classList.remove('hidden');
    document.getElementById('view-joint-cause-text').innerText = found.joint_failure_reason_zh;
  } else {
    document.getElementById('view-joint-cause-box').classList.add('hidden');
  }

  // Setup Video Player
  document.getElementById('video-source-ds').src = found.video_url_ds;
  document.getElementById('video-source-generic').src = found.video_url_generic;
  videoEl.load();

  // Reset Scrubber
  cursorGt.style.left = '0%';
  cursorPred.style.left = '0%';
  timeReadout.innerText = '帧: 0 / ' + found.frame_count + ' (0.00s)';

  // Setup Timeline Lanes
  const fc = Math.max(found.frame_count, 1);
  const gt = found.ground_truth_interval;
  const gtStart = gt ? (gt.start || 0) : 0;
  const gtEnd = gt ? (gt.end || fc) : fc;
  const gtLeft = (gtStart / fc) * 100;
  const gtWidth = Math.max(((gtEnd - gtStart) / fc) * 100, 3);

  const barGt = document.getElementById('bar-gt');
  barGt.style.left = gtLeft + '%';
  barGt.style.width = gtWidth + '%';
  barGt.innerText = '真值区间: ' + gtStart + ' - ' + gtEnd + ' 帧 (' + found.behavior_zh + ')';

  const barPred = document.getElementById('bar-pred');
  const pred = found.predicted_interval;
  if (pred && (pred.end > pred.start || (Array.isArray(pred) && pred[1] > pred[0]))) {
    const pStart = Array.isArray(pred) ? pred[0] : (pred.start || 0);
    const pEnd = Array.isArray(pred) ? pred[1] : (pred.end || 0);
    const pLeft = (pStart / fc) * 100;
    const pWidth = Math.max(((pEnd - pStart) / fc) * 100, 4);
    barPred.className = 'lane-bar bar-pred';
    barPred.style.left = pLeft + '%';
    barPred.style.width = pWidth + '%';
    const predBeh = found.primary_predicted_behavior || '候选区间';
    const predBehZh = BEHAVIOR_CHINESE[predBeh] || predBeh;
    barPred.innerText = '预测区间: ' + pStart + ' - ' + pEnd + ' 帧 (' + predBehZh + ')';
  } else {
    barPred.className = 'lane-bar bar-none';
    barPred.style.left = '0%';
    barPred.style.width = '100%';
    barPred.innerText = '未触发任何有效候选区间 (规则阈值未满足)';
  }

  // Populate Kinematics
  const kin = found.kinematics || {};
  document.getElementById('kin-center-dist').innerText = (kin.center_distance_min_cm != null ? kin.center_distance_min_cm : '--') + ' / ' +
    (kin.center_distance_median_cm != null ? kin.center_distance_median_cm : '--') + ' / ' +
    (kin.center_distance_max_cm != null ? kin.center_distance_max_cm : '--') + ' cm';
  document.getElementById('kin-snout-head').innerText = (kin.nose_head_min_cm != null ? kin.nose_head_min_cm : '--') + ' / ' +
    (kin.nose_head_median_cm != null ? kin.nose_head_median_cm : '--') + ' cm';
  document.getElementById('kin-snout-rear').innerText = (kin.nose_tail_min_cm != null ? kin.nose_tail_min_cm : '--') + ' / ' +
    (kin.nose_tail_median_cm != null ? kin.nose_tail_median_cm : '--') + ' cm';
  document.getElementById('kin-speed').innerText = (kin.speed_median_cm_s != null ? kin.speed_median_cm_s : '--') + ' / ' +
    (kin.speed_max_cm_s != null ? kin.speed_max_cm_s : '--') + ' cm/s';
  document.getElementById('kin-closing-speed').innerText = (kin.closing_speed_median_cm_s != null ? kin.closing_speed_median_cm_s : '--') + ' / ' +
    (kin.closing_speed_max_cm_s != null ? kin.closing_speed_max_cm_s : '--') + ' cm/s';

  const obsR = kin.observed_ratio_per_target ? Object.values(kin.observed_ratio_per_target).map(function(v) { return (v * 100).toFixed(0) + '%'; }).join(', ') : '--';
  const coObs = kin.coobserved_ratio != null ? (kin.coobserved_ratio * 100).toFixed(0) + '%' : '--';
  document.getElementById('kin-obs-ratio').innerText = obsR + ' (共现: ' + coObs + ')';

  const misNose = kin.missing_nose_fraction != null ? (kin.missing_nose_fraction * 100).toFixed(1) + '%' : '--';
  const misTail = kin.missing_tail_fraction != null ? (kin.missing_tail_fraction * 100).toFixed(1) + '%' : '--';
  document.getElementById('kin-missing-kps').innerText = '鼻: ' + misNose + ', 尾: ' + misTail;

  // Populate Competing Candidates
  const rivalsList = document.getElementById('competing-candidates-list');
  rivalsList.innerHTML = '';
  const rivals = found.competing_candidates || [];
  if (rivals.length === 0) {
    rivalsList.innerHTML = '<div style="font-size: 12.5px; color: var(--text-dim);">无竞争候选记录（未触发其他行为掩码）</div>';
  } else {
    rivals.forEach(function(r, idx) {
      const row = document.createElement('div');
      row.className = 'metric-row';
      const rZh = BEHAVIOR_CHINESE[r.behavior] || r.behavior;
      const tIds = r.target_ids ? ('[' + r.target_ids.join(', ') + ']') : '';
      const sc = r.score != null ? (r.score.toFixed(2) + 's') : '';
      row.innerHTML = '<span class="m-key">' + (idx + 1) + '. <strong style="color: var(--text-main);">' + rZh + '</strong> (' + r.behavior + ') ' + tIds + '</span>' +
        '<span class="m-val" style="color: var(--accent-amber);">' + sc + '</span>';
      rivalsList.appendChild(row);
    });
  }
}

function onVideoTimeUpdate() {
  if (!activeClip || !videoEl.duration) return;

  const curTime = videoEl.currentTime;
  const dur = videoEl.duration;
  const progress = Math.min(Math.max((curTime / dur) * 100, 0), 100);

  cursorGt.style.left = progress + '%';
  cursorPred.style.left = progress + '%';

  const curFrame = Math.floor(curTime * activeClip.fps);
  timeReadout.innerText = '帧: ' + curFrame + ' / ' + activeClip.frame_count + ' (' + curTime.toFixed(2) + 's)';
}

function seekByTrackClick(e, trackKind) {
  if (!activeClip || !videoEl.duration) return;
  const trackEl = document.getElementById('track-' + trackKind);
  const rect = trackEl.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const fraction = Math.max(0, Math.min(1, clickX / rect.width));

  videoEl.currentTime = fraction * videoEl.duration;
}

function togglePlay() {
  if (videoEl.paused) {
    videoEl.play();
  } else {
    videoEl.pause();
  }
}

function stepFrame(step) {
  if (!activeClip) return;
  videoEl.pause();
  const frameTime = 1.0 / activeClip.fps;
  videoEl.currentTime = Math.max(0, Math.min(videoEl.duration || 100, videoEl.currentTime + step * frameTime));
}

function setPlaySpeed(val) {
  videoEl.playbackRate = parseFloat(val) || 1.0;
}

function onKeyDown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

  if (e.code === 'Space') {
    e.preventDefault();
    togglePlay();
  } else if (e.code === 'ArrowLeft') {
    e.preventDefault();
    stepFrame(-1);
  } else if (e.code === 'ArrowRight') {
    e.preventDefault();
    stepFrame(1);
  } else if (e.code === 'ArrowUp') {
    e.preventDefault();
    navigateClip(-1);
  } else if (e.code === 'ArrowDown') {
    e.preventDefault();
    navigateClip(1);
  }
}

function navigateClip(offset) {
  if (!activeClip || filteredClips.length === 0) return;
  const curIdx = filteredClips.findIndex(function(c) { return c.id === activeClip.id; });
  const nextIdx = Math.max(0, Math.min(filteredClips.length - 1, curIdx + offset));
  if (nextIdx !== curIdx) {
    selectClip(filteredClips[nextIdx].id);
  }
}

function resetFilters() {
  document.getElementById('search-input').value = '';
  document.getElementById('select-dataset').value = 'all';
  document.getElementById('select-behavior').value = 'all';
  setCategoryFilter('all');
}

function focusConflictClip() {
  resetFilters();
  const conflictClipId = '52512294eac95441035cb075';
  const found = (REPORT_DATA.failures || []).find(function(c) { return c.clip_id === conflictClipId; });
  if (found) {
    selectClip(found.id);
    document.getElementById('section-browser').scrollIntoView({ behavior: 'smooth' });
  }
}

function copyClipId() {
  if (activeClip) {
    navigator.clipboard.writeText(activeClip.clip_id);
    alert('已复制 Clip ID: ' + activeClip.clip_id);
  }
}

// Initialize when DOM is ready
window.addEventListener('DOMContentLoaded', initDashboard);
</script>

</body>
</html>
"""


def generate_offline_report(
    eval_json_path: Path = DEFAULT_EVAL_JSON,
    comp_json_path: Path = DEFAULT_COMP_JSON,
    output_html_path: Path = DEFAULT_OUTPUT_HTML,
) -> Path:
    """Read data, generate standalone HTML, and write to disk."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = build_report_data(eval_json_path, comp_json_path)
    report_json_str = json.dumps(report_data, ensure_ascii=False)

    html_content = HTML_TEMPLATE.replace("/* __REPORT_DATA_PLACEHOLDER__ */ {}", report_json_str)

    output_html_path.write_text(html_content, encoding="utf-8")
    LOGGER.info(
        "Successfully wrote offline failure report: %s (size: %d bytes)",
        output_html_path,
        len(html_content.encode("utf-8")),
    )
    return output_html_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-json", type=Path, default=DEFAULT_EVAL_JSON, help="Path to evaluation JSON"
    )
    parser.add_argument(
        "--comp-json", type=Path, default=DEFAULT_COMP_JSON, help="Path to comparison JSON"
    )
    parser.add_argument(
        "--output-html", type=Path, default=DEFAULT_OUTPUT_HTML, help="Path to output HTML"
    )
    args = parser.parse_args()

    out_file = generate_offline_report(
        eval_json_path=args.eval_json,
        comp_json_path=args.comp_json,
        output_html_path=args.output_html,
    )
    print(f"Report generated: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
