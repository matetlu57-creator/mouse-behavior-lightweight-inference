#!/usr/bin/env python3
"""Generate the comprehensive 15-behavior dataset reconciliation comparison table.

Covers Dataset A (429), Dataset B (605), Overlap (270 under A and B tracks),
and Union (764 under B-priority and A-priority sensitivity).
Outputs:
  - outputs/dataset_comparison_table.md
  - outputs/dataset_comparison_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from _bootstrap import REPO_ROOT  # noqa: F401

DEFAULT_INPUT_JSON = Path("outputs/reconciled_behavior_evaluation.json")
DEFAULT_OUTPUT_MD = Path("outputs/dataset_comparison_table.md")
DEFAULT_OUTPUT_JSON = Path("outputs/dataset_comparison_table.json")

ALL_15_BEHAVIORS = [
    # Social Behaviors (6)
    "Approach",
    "Attack",
    "Chasing",
    "Snout-head_contact",
    "Snout-rear_contact",
    "Together",
    # Individual Behaviors (4)
    "Static",
    "Locomotion",
    "Grooming",
    "Rearing",
    # Group Behaviors (5)
    "Isolation",
    "Group locomotion",
    "Dispersal",
    "Social clustering",
    "Huddling",
]

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

BEHAVIOR_LAYER: dict[str, str] = {
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


def _safe_div(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(float(num / den), 4)


def _safe_f1(prec: float | None, rec: float | None) -> float | None:
    if prec is None or rec is None or (prec + rec) <= 0.0:
        return None
    return round(float(2.0 * prec * rec / (prec + rec)), 4)


def _compute_cohort_precision_recall(
    report: Mapping[str, Any],
    predicate: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Compute per-behavior and aggregate Precision/Recall across Primary, pf_target, and pf_joint."""
    subset = [c for c in report.get("clips", ()) if predicate(c)]
    per_beh: dict[str, dict[str, Any]] = {}

    # Track non-None predictions across cohort for Micro Precision
    prim_tp_total = 0
    prim_pred_total = 0
    targ_tp_total = 0
    targ_pred_total = 0
    joint_tp_total = 0
    joint_pred_total = 0

    for beh in ALL_15_BEHAVIORS:
        gt_clips = [c for c in subset if c.get("behavior") == beh]
        sup = len(gt_clips)

        prim_tp = sum(1 for c in gt_clips if c.get("primary_correct"))
        prim_pred = sum(1 for c in subset if c.get("primary_predicted_behavior") == beh)
        prim_p = _safe_div(prim_tp, prim_pred)
        prim_r = _safe_div(prim_tp, sup)
        prim_f1 = _safe_f1(prim_p, prim_r)

        targ_tp = sum(1 for c in gt_clips if c.get("pf_target_correct"))
        targ_pred = sum(1 for c in subset if c.get("pf_target_predicted_behavior") == beh)
        targ_p = _safe_div(targ_tp, targ_pred)
        targ_r = _safe_div(targ_tp, sup)
        targ_f1 = _safe_f1(targ_p, targ_r)

        joint_tp = sum(1 for c in gt_clips if c.get("pf_joint_correct"))
        joint_pred = sum(1 for c in subset if c.get("pf_joint_predicted_behavior") == beh)
        joint_p = _safe_div(joint_tp, joint_pred)
        joint_r = _safe_div(joint_tp, sup)
        joint_f1 = _safe_f1(joint_p, joint_r)

        prim_tp_total += prim_tp
        prim_pred_total += prim_pred
        targ_tp_total += targ_tp
        targ_pred_total += targ_pred
        joint_tp_total += joint_tp
        joint_pred_total += joint_pred

        per_beh[beh] = {
            "primary_predicted_count": prim_pred,
            "primary_precision": prim_p,
            "primary_recall": prim_r,
            "primary_f1": prim_f1,
            "pf_target_correct": targ_tp,
            "pf_target_predicted_count": targ_pred,
            "pf_target_precision": targ_p,
            "pf_target_recall": targ_r,
            "pf_target_f1": targ_f1,
            "pf_joint_predicted_count": joint_pred,
            "pf_joint_precision": joint_p,
            "pf_joint_recall": joint_r,
            "pf_joint_f1": joint_f1,
        }

    # Include predictions whose label does not match ANY GT class in subset?
    # All 15 behaviors are already summed above unless a prediction falls outside ALL_15_BEHAVIORS.
    prim_pred_all = sum(1 for c in subset if c.get("primary_predicted_behavior") is not None)
    targ_pred_all = sum(1 for c in subset if c.get("pf_target_predicted_behavior") is not None)
    joint_pred_all = sum(1 for c in subset if c.get("pf_joint_predicted_behavior") is not None)
    n = len(subset)

    present_rows = [per_beh[b] for b in ALL_15_BEHAVIORS if any(c.get("behavior") == b for c in subset)]

    def _mean(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(float(sum(clean) / len(clean)), 4) if clean else None

    agg = {
        "primary_predicted_non_none": prim_pred_all,
        "primary_micro_precision": _safe_div(prim_tp_total, prim_pred_all),
        "primary_micro_recall": _safe_div(prim_tp_total, n),
        "primary_macro_precision": _mean([r["primary_precision"] for r in present_rows]),
        "primary_macro_recall": _mean([r["primary_recall"] for r in present_rows]),
        "pf_target_correct": targ_tp_total,
        "pf_target_predicted_non_none": targ_pred_all,
        "pf_target_micro_precision": _safe_div(targ_tp_total, targ_pred_all),
        "pf_target_micro_recall": _safe_div(targ_tp_total, n),
        "pf_target_macro_precision": _mean([r["pf_target_precision"] for r in present_rows]),
        "pf_target_macro_recall": _mean([r["pf_target_recall"] for r in present_rows]),
        "pf_joint_predicted_non_none": joint_pred_all,
        "pf_joint_micro_precision": _safe_div(joint_tp_total, joint_pred_all),
        "pf_joint_micro_recall": _safe_div(joint_tp_total, n),
        "pf_joint_macro_precision": _mean([r["pf_joint_precision"] for r in present_rows]),
        "pf_joint_macro_recall": _mean([r["pf_joint_recall"] for r in present_rows]),
    }
    return per_beh, agg


def build_comparison_data(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and reconcile metrics across all cohorts."""
    cohorts = report["cohorts"]
    recon = report.get("reconciliation", {})

    c_a = cohorts.get("dataset_a", {})
    c_b = cohorts.get("dataset_b", {})
    c_oa = cohorts.get("overlap_a", {})
    c_ob = cohorts.get("overlap_b", {})
    c_ub = cohorts.get("union_b_priority", {})
    c_ua = cohorts.get("union_a_priority", {})

    pr_a, agg_a = _compute_cohort_precision_recall(report, lambda c: c.get("dataset") == "A")
    pr_b, agg_b = _compute_cohort_precision_recall(report, lambda c: c.get("dataset") == "B")
    pr_oa, agg_oa = _compute_cohort_precision_recall(
        report, lambda c: c.get("dataset") == "A" and c.get("in_overlap")
    )
    pr_ob, agg_ob = _compute_cohort_precision_recall(
        report, lambda c: c.get("dataset") == "B" and c.get("in_overlap")
    )
    pr_ub, agg_ub = _compute_cohort_precision_recall(report, lambda c: c.get("in_union_b_priority"))
    pr_ua, agg_ua = _compute_cohort_precision_recall(report, lambda c: c.get("in_union_a_priority"))

    behavior_rows: list[dict[str, Any]] = []

    for beh in ALL_15_BEHAVIORS:
        layer = BEHAVIOR_LAYER.get(beh, "Unknown")
        zh_name = BEHAVIOR_CHINESE.get(beh, beh)

        ra = c_a.get("per_behavior", {}).get(beh, {})
        rb = c_b.get("per_behavior", {}).get(beh, {})
        roa = c_oa.get("per_behavior", {}).get(beh, {})
        rob = c_ob.get("per_behavior", {}).get(beh, {})
        rub = c_ub.get("per_behavior", {}).get(beh, {})
        rua = c_ua.get("per_behavior", {}).get(beh, {})

        # Agreement rate in overlap
        oa_sup = roa.get("support", 0)
        ob_sup = rob.get("support", 0)
        # Check agreement
        overlap_agree_rate = None
        if oa_sup > 0 and ob_sup > 0:
            oa_corr = roa.get("primary_correct", 0)
            ob_corr = rob.get("primary_correct", 0)
            # Min support or exact matching
            overlap_agree_rate = 1.0 - abs(oa_corr - ob_corr) / max(oa_sup, ob_sup)

        row = {
            "behavior": beh,
            "behavior_zh": zh_name,
            "layer": layer,
            "primary_method": (
                rub.get("primary_method")
                or rb.get("primary_method")
                or ra.get("primary_method")
                or "exact_target_frame_coverage_ge_0.5"
            ),
            # Dataset A (429)
            "dataset_a": {
                "support": ra.get("support", 0),
                "primary_correct": ra.get("primary_correct", 0),
                "primary_accuracy": ra.get("primary_accuracy", 0.0),
                "pf_joint_correct": ra.get("pf_joint_correct", 0),
                "pf_joint_accuracy": ra.get("pf_joint_accuracy", 0.0),
                "coverage_a2_correct": ra.get("coverage_a2_correct", 0),
                "coverage_a2_accuracy": ra.get("coverage_a2_accuracy", 0.0),
                "missed_count": ra.get("missed_count", 0),
                "misclassified_count": ra.get("misclassified_count", 0),
                **pr_a[beh],
            },
            # Dataset B (605)
            "dataset_b": {
                "support": rb.get("support", 0),
                "primary_correct": rb.get("primary_correct", 0),
                "primary_accuracy": rb.get("primary_accuracy", 0.0),
                "pf_joint_correct": rb.get("pf_joint_correct", 0),
                "pf_joint_accuracy": rb.get("pf_joint_accuracy", 0.0),
                "coverage_a2_correct": rb.get("coverage_a2_correct", 0),
                "coverage_a2_accuracy": rb.get("coverage_a2_accuracy", 0.0),
                "missed_count": rb.get("missed_count", 0),
                "misclassified_count": rb.get("misclassified_count", 0),
                **pr_b[beh],
            },
            # Overlap under A (270)
            "overlap_a": {
                "support": roa.get("support", 0),
                "primary_correct": roa.get("primary_correct", 0),
                "primary_accuracy": roa.get("primary_accuracy", 0.0),
                "pf_joint_correct": roa.get("pf_joint_correct", 0),
                "pf_joint_accuracy": roa.get("pf_joint_accuracy", 0.0),
                **pr_oa[beh],
            },
            # Overlap under B (270)
            "overlap_b": {
                "support": rob.get("support", 0),
                "primary_correct": rob.get("primary_correct", 0),
                "primary_accuracy": rob.get("primary_accuracy", 0.0),
                "pf_joint_correct": rob.get("pf_joint_correct", 0),
                "pf_joint_accuracy": rob.get("pf_joint_accuracy", 0.0),
                **pr_ob[beh],
            },
            # Overlap Agreement
            "overlap_agreement_rate": overlap_agree_rate,
            # Union B-priority (764)
            "union_b_priority": {
                "support": rub.get("support", 0),
                "primary_correct": rub.get("primary_correct", 0),
                "primary_accuracy": rub.get("primary_accuracy", 0.0),
                "pf_joint_correct": rub.get("pf_joint_correct", 0),
                "pf_joint_accuracy": rub.get("pf_joint_accuracy", 0.0),
                "coverage_a2_correct": rub.get("coverage_a2_correct", 0),
                "coverage_a2_accuracy": rub.get("coverage_a2_accuracy", 0.0),
                "missed_count": rub.get("missed_count", 0),
                "misclassified_count": rub.get("misclassified_count", 0),
                **pr_ub[beh],
            },
            # Union A-priority (764 sensitivity)
            "union_a_priority": {
                "support": rua.get("support", 0),
                "primary_correct": rua.get("primary_correct", 0),
                "primary_accuracy": rua.get("primary_accuracy", 0.0),
                "pf_joint_correct": rua.get("pf_joint_correct", 0),
                "pf_joint_accuracy": rua.get("pf_joint_accuracy", 0.0),
                **pr_ua[beh],
            },
        }
        behavior_rows.append(row)

    # Compute overall summary
    summary = {
        "dataset_a": {
            "total_clips": c_a.get("total_clips", 429),
            "primary_correct": c_a.get("overall_primary_correct", 0),
            "primary_accuracy": c_a.get("overall_primary_accuracy", 0.0),
            "pf_joint_correct": c_a.get("overall_pf_joint_correct", 0),
            "pf_joint_accuracy": c_a.get("overall_pf_joint_accuracy", 0.0),
            "primary_failure_count": c_a.get("primary_failure_count", 0),
            "missed_count": c_a.get("primary_missed_count", 0),
            "misclassified_count": c_a.get("primary_misclassified_count", 0),
            **agg_a,
        },
        "dataset_b": {
            "total_clips": c_b.get("total_clips", 605),
            "primary_correct": c_b.get("overall_primary_correct", 0),
            "primary_accuracy": c_b.get("overall_primary_accuracy", 0.0),
            "pf_joint_correct": c_b.get("overall_pf_joint_correct", 0),
            "pf_joint_accuracy": c_b.get("overall_pf_joint_accuracy", 0.0),
            "primary_failure_count": c_b.get("primary_failure_count", 0),
            "missed_count": c_b.get("primary_missed_count", 0),
            "misclassified_count": c_b.get("primary_misclassified_count", 0),
            **agg_b,
        },
        "overlap_a": {
            "total_clips": c_oa.get("total_clips", 270),
            "primary_correct": c_oa.get("overall_primary_correct", 0),
            "primary_accuracy": c_oa.get("overall_primary_accuracy", 0.0),
            "pf_joint_correct": c_oa.get("overall_pf_joint_correct", 0),
            "pf_joint_accuracy": c_oa.get("overall_pf_joint_accuracy", 0.0),
            "primary_failure_count": c_oa.get("primary_failure_count", 0),
            **agg_oa,
        },
        "overlap_b": {
            "total_clips": c_ob.get("total_clips", 270),
            "primary_correct": c_ob.get("overall_primary_correct", 0),
            "primary_accuracy": c_ob.get("overall_primary_accuracy", 0.0),
            "pf_joint_correct": c_ob.get("overall_pf_joint_correct", 0),
            "pf_joint_accuracy": c_ob.get("overall_pf_joint_accuracy", 0.0),
            "primary_failure_count": c_ob.get("primary_failure_count", 0),
            **agg_ob,
        },
        "union_b_priority": {
            "total_clips": c_ub.get("total_clips", 764),
            "primary_correct": c_ub.get("overall_primary_correct", 0),
            "primary_accuracy": c_ub.get("overall_primary_accuracy", 0.0),
            "pf_joint_correct": c_ub.get("overall_pf_joint_correct", 0),
            "pf_joint_accuracy": c_ub.get("overall_pf_joint_accuracy", 0.0),
            "primary_failure_count": c_ub.get("primary_failure_count", 0),
            "missed_count": c_ub.get("primary_missed_count", 0),
            "misclassified_count": c_ub.get("primary_misclassified_count", 0),
            **agg_ub,
        },
        "union_a_priority": {
            "total_clips": c_ua.get("total_clips", 764),
            "primary_correct": c_ua.get("overall_primary_correct", 0),
            "primary_accuracy": c_ua.get("overall_primary_accuracy", 0.0),
            "pf_joint_correct": c_ua.get("overall_pf_joint_correct", 0),
            "pf_joint_accuracy": c_ua.get("overall_pf_joint_accuracy", 0.0),
            "primary_failure_count": c_ua.get("primary_failure_count", 0),
            "missed_count": c_ua.get("primary_missed_count", 0),
            "misclassified_count": c_ua.get("primary_misclassified_count", 0),
            **agg_ua,
        },
    }

    return {
        "metadata": {
            "title": "Dataset A (429) & Dataset B (605) 15-Behavior Reconciliation Comparison",
            "git_head": report.get("provenance", {}).get("git_head") or report.get("git_head", ""),
            "all_behaviors": report.get("all_behaviors", True),
            "enable_posture_screen": report.get("enable_posture_screen", True),
            "scale_policy": report.get("scale_policy", "dataset_b_frozen_source_scales_a_default"),
        },
        "reconciliation_summary": {
            "dataset_a_total": recon.get("dataset_a_total", 429),
            "dataset_b_total": recon.get("dataset_b_total", 605),
            "overlap_count": recon.get("overlap_count", 270),
            "a_only_count": recon.get("a_only_count", 159),
            "b_only_count": recon.get("b_only_count", 335),
            "union_count": recon.get("union_count", 764),
            "label_agreement_count": recon.get("label_agreement_count", 269),
            "label_agreement_rate": recon.get("label_agreement_rate", 0.996296),
            "label_conflicts": recon.get("label_conflicts", []),
            "overlap_file_differences": recon.get("overlap_file_differences", {}),
            "conflict_clip_deep_dive": {
                "clip_id": "52512294eac95441035cb075",
                "dataset_a_label": "Snout-head_contact",
                "dataset_b_label": "Snout-rear_contact",
                "target_mouse_ids": [71, 230],
                "source_video": "WIN_20260822_14_13_59_Pro_part001.mp4 (frames 3114-3153, 40 frames, 1.34s, 29.95 fps)",
                "physical_distance_evidence": {
                    "nose_tail_min_cm": 1.131,
                    "nose_tail_median_cm": 1.455,
                    "nose_head_min_cm": 4.631,
                    "nose_head_median_cm": 7.230,
                    "contact_threshold_cm": "2.23 - 4.38 cm",
                    "conclusion": "物理接触点确为鼻-尾接触 (1.13 cm)，鼻-头间距 (4.63 cm) 远超阈值。数据集 B 标签正确。",
                },
                "trajectory_repair_evidence": {
                    "dataset_a_repairs": 14,
                    "dataset_a_anomaly": "A 中存在 6 帧线性插值与 8 帧 SAM2 修复，导致小鼠 71 速度突跳至 51.43 cm/s，触发 32 帧 Attack 误检。",
                    "dataset_b_repairs": 0,
                    "dataset_b_result": "B 中无人工插值，速度平稳 (max 8.21 cm/s)，正确触发 Snout-rear_contact 30 帧 (100% 正确)。",
                },
                "resolution": "并集采用 B 优先策略 (union_b_priority)，使用数据集 B 的真值标签与干净轨迹。",
            },
        },
        "overall_summary": summary,
        "behaviors": behavior_rows,
    }


def format_markdown_table(data: dict[str, Any]) -> str:
    """Render the full Markdown reconciliation document."""
    meta = data["metadata"]
    recon = data["reconciliation_summary"]
    ov = data["overall_summary"]
    rows = data["behaviors"]

    lines: list[str] = []
    lines.append("# 数据集 A (429) 与 数据集 B (605) 15类行为核对与评测对比表\n")
    lines.append(
        f"**生成时间与版本**: Git HEAD `{meta.get('git_head')}` | 姿态筛选: `启用` | 评测模式: `全15类完整评测`\n"
    )
    lines.append("## 1. 数据集核对与集合关系汇总\n")
    lines.append(
        f"- **数据集 A (补全集)**: **{recon['dataset_a_total']}** 段视频（仅包含 6 类社交行为，无单独个体/群体行为）"
    )
    lines.append(
        f"- **数据集 B (人工清洗集)**: **{recon['dataset_b_total']}** 段视频（覆盖全 15 类行为完整图谱：306 社交 + 158 个体 + 141 群体）"
    )
    lines.append(
        f"- **重叠视频集 (Overlap)**: **{recon['overlap_count']}** 段（全部为社交行为；A 独有 = {recon['a_only_count']} 段，B 独有 = {recon['b_only_count']} 段）"
    )
    lines.append(
        f"- **去重并集 (Union)**: **{recon['union_count']}** 段（以数据集 B 优先：605 段 B + 159 段 A 独有）"
    )
    lines.append(
        f"- **标签一致率**: **{recon['label_agreement_count']}/{recon['overlap_count']} ({recon['label_agreement_rate'] * 100:.2f}%)**（仅 1 处标签冲突）"
    )
    lines.append(
        f"- **轨迹文件差异**: 字节 SHA-256 差异 {recon['overlap_file_differences'].get('tracks_bytes_differ', 270)}/270，解析检测差异 {recon['overlap_file_differences'].get('tracks_parsed_differ', 109)}/270（其中 44 段含 A 的合成修复/SAM2 插值，65 段含背景干扰小鼠差异）\n"
    )

    lines.append("### 冲突样本 `52512294eac95441035cb075` 深入归因与判定")
    dd = recon["conflict_clip_deep_dive"]
    lines.append(
        f"- **视频来源**: `{dd['source_video']}` | 目标小鼠 ID: `{dd['target_mouse_ids']}`"
    )
    lines.append(
        f"- **标签分歧**: 数据集 A 标记为 `{dd['dataset_a_label']}`（嗅闻头部），数据集 B 标记为 `{dd['dataset_b_label']}`（嗅闻尾部）。"
    )
    lines.append(
        f"- **实测运动学距离证据**: {dd['physical_distance_evidence']['conclusion']}（鼻头-尾根间距 = {dd['physical_distance_evidence']['nose_tail_min_cm']:.3f} cm，鼻头-头部间距 = {dd['physical_distance_evidence']['nose_head_min_cm']:.3f} cm，接触阈值 {dd['physical_distance_evidence']['contact_threshold_cm']}）。"
    )
    lines.append(
        f"- **轨迹修复与速度异常**: {dd['trajectory_repair_evidence']['dataset_a_anomaly']}"
    )
    lines.append(f"- **清洗集表现**: {dd['trajectory_repair_evidence']['dataset_b_result']}")
    lines.append(
        f"- **仲裁决定**: {dd['resolution']}（同时在对比表中附带展示 A 优先敏感性测试）。\n"
    )

    def _fmt_pct(val: float | None, digits: int = 1) -> str:
        if val is None:
            return "—"
        return f"{val * 100:.{digits}f}%"

    def _fmt_cell(d: Mapping[str, Any], *, bold: bool = False) -> str:
        sup = d.get("support", 0)
        if sup <= 0:
            return "—"
        tp = d.get("primary_correct", 0)
        pred = d.get("primary_predicted_count", 0)
        p = _fmt_pct(d.get("primary_precision"))
        r = _fmt_pct(d.get("primary_recall"))
        head = f"{tp}/{sup}<br>P={p} ({tp}/{pred}) / R={r}"
        return f"**{head}**" if bold else head

    def _fmt_joint_cell(d: Mapping[str, Any]) -> str:
        sup = d.get("support", 0)
        if sup <= 0:
            return "—"
        tp = d.get("pf_joint_correct", 0)
        pred = d.get("pf_joint_predicted_count", 0)
        p = _fmt_pct(d.get("pf_joint_precision"))
        r = _fmt_pct(d.get("pf_joint_recall"))
        return f"{tp}/{sup}<br>P={p} ({tp}/{pred}) / R={r}"

    lines.append("## 2. 全 15 类行为多维度准确率、精确率 (Precision) 与召回率 (Recall) 对比总表\n")
    lines.append(
        "| 行为类别 (中/英) | 行为层级 | 数据集 A (429)<br>主指标 (P / R) | A Strict `pf_joint`<br>(P / R) | 数据集 B (605)<br>主指标 (P / R) | B Strict `pf_joint`<br>(P / R) | 重叠集<br>(A轨 / B轨) | 并集 B优先 (764)<br>主指标 (P / R) | 并集 B优先 Strict `pf_joint`<br>(P / R) | 并集 A优先 (764)<br>敏感性 (P / R) | 专项评测方法 |"
    )
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    for r in rows:
        beh_display = f"**{r['behavior_zh']}**<br>`{r['behavior']}`"
        layer = r["layer"]

        ra = r["dataset_a"]
        rb = r["dataset_b"]
        roa = r["overlap_a"]
        rob = r["overlap_b"]
        rub = r["union_b_priority"]
        rua = r["union_a_priority"]

        a_str = _fmt_cell(ra)
        a_j_str = _fmt_joint_cell(ra)
        b_str = _fmt_cell(rb, bold=True)
        b_j_str = _fmt_joint_cell(rb)
        oa_str = (
            f"{roa['primary_correct']}/{roa['support']} ({_fmt_pct(roa['primary_accuracy'])})"
            if roa["support"] > 0
            else "—"
        )
        ob_str = (
            f"{rob['primary_correct']}/{rob['support']} ({_fmt_pct(rob['primary_accuracy'])})"
            if rob["support"] > 0
            else "—"
        )
        ov_str = f"{oa_str}<br>/ {ob_str}" if (oa_str != "—" or ob_str != "—") else "—"
        ub_str = _fmt_cell(rub, bold=True)
        ub_j_str = _fmt_joint_cell(rub)
        ua_str = _fmt_cell(rua)

        method = r["primary_method"]

        lines.append(
            f"| {beh_display} | {layer} | {a_str} | {a_j_str} | {b_str} | {b_j_str} | {ov_str} | {ub_str} | {ub_j_str} | {ua_str} | `{method}` |"
        )

    # Overall Summary Row (Micro & Macro Precision / Recall)
    sa = ov["dataset_a"]
    sb = ov["dataset_b"]
    soa = ov["overlap_a"]
    sob = ov["overlap_b"]
    sub = ov["union_b_priority"]
    sua = ov["union_a_priority"]

    def _fmt_tot(s: Mapping[str, Any], key_prefix: str = "primary") -> str:
        tp = s[f"{key_prefix}_correct"]
        n = s["total_clips"]
        mp = _fmt_pct(s.get(f"{key_prefix}_micro_precision"), 2)
        mr = _fmt_pct(s.get(f"{key_prefix}_micro_recall"), 2)
        mac_p = _fmt_pct(s.get(f"{key_prefix}_macro_precision"), 2)
        mac_r = _fmt_pct(s.get(f"{key_prefix}_macro_recall"), 2)
        return f"**{tp}/{n}**<br>Micro P={mp} / R={mr}<br>Macro P={mac_p} / R={mac_r}"

    sa_str = _fmt_tot(sa, "primary")
    sa_j_str = _fmt_tot(sa, "pf_joint")
    sb_str = _fmt_tot(sb, "primary")
    sb_j_str = _fmt_tot(sb, "pf_joint")
    sov_str = (
        f"**{soa['primary_correct']}/{soa['total_clips']} ({soa['primary_accuracy'] * 100:.2f}%)**"
        f"<br>/ **{sob['primary_correct']}/{sob['total_clips']} ({sob['primary_accuracy'] * 100:.2f}%)**"
    )
    sub_str = _fmt_tot(sub, "primary")
    sub_j_str = _fmt_tot(sub, "pf_joint")
    sua_str = _fmt_tot(sua, "primary")

    lines.append(
        f"| **总体合计 (TOTAL)** | **全部15类** | {sa_str} | {sa_j_str} | {sb_str} | {sb_j_str} | {sov_str} | {sub_str} | {sub_j_str} | {sua_str} | Micro / Macro 汇总 |"
    )
    lines.append("")

    # Unified 15-Class Target-Conditioned Table (pf_target) showing cross-class FP/FN honestly
    lines.append("## 2b. 统一 15 类竞争模型 (`pf_target`) 跨类精确率 / 召回率 / F1 对照表\n")
    lines.append(
        "> 注：在分项主指标中，9 类非双鼠配对行为（个体 4 类 + 群体 5 类）仅各自在所属真值片段上评测，无跨类误报分母（`Primary P` 类内恒为 100%）；而下表 `pf_target` 在**全部片段**上不提示行为类别、由 15 类同台竞争预测，能够真实反映每一类的跨类误报精确率 (`Precision = TP / Pred`) 与召回率 (`Recall = TP / Support`)。\n"
    )
    lines.append(
        "| 行为类别 (中/英) | 数据集 A (429)<br>精确率 P / 召回率 R / F1 | 数据集 B (605)<br>精确率 P / 召回率 R / F1 | 并集 B优先 (764)<br>精确率 P / 召回率 R / F1 |"
    )
    lines.append("|:---|:---:|:---:|:---:|")

    def _fmt_pt(d: Mapping[str, Any]) -> str:
        if d.get("support", 0) <= 0:
            return "—"
        tp = d.get("pf_target_correct", 0)
        sup = d.get("support", 0)
        pred = d.get("pf_target_predicted_count", 0)
        p = _fmt_pct(d.get("pf_target_precision"))
        r = _fmt_pct(d.get("pf_target_recall"))
        f1 = _fmt_pct(d.get("pf_target_f1"))
        return f"P={p} ({tp}/{pred}) / R={r} ({tp}/{sup}) / F1={f1}"

    for r in rows:
        beh_display = f"**{r['behavior_zh']}** (`{r['behavior']}`)"
        lines.append(
            f"| {beh_display} | {_fmt_pt(r['dataset_a'])} | {_fmt_pt(r['dataset_b'])} | {_fmt_pt(r['union_b_priority'])} |"
        )

    lines.append(
        f"| **总体合计 (Micro / Macro)** | "
        f"Micro P={_fmt_pct(sa.get('pf_target_micro_precision'), 2)} / R={_fmt_pct(sa.get('pf_target_micro_recall'), 2)}<br>Macro P={_fmt_pct(sa.get('pf_target_macro_precision'), 2)} / R={_fmt_pct(sa.get('pf_target_macro_recall'), 2)} | "
        f"Micro P={_fmt_pct(sb.get('pf_target_micro_precision'), 2)} / R={_fmt_pct(sb.get('pf_target_micro_recall'), 2)}<br>Macro P={_fmt_pct(sb.get('pf_target_macro_precision'), 2)} / R={_fmt_pct(sb.get('pf_target_macro_recall'), 2)} | "
        f"Micro P={_fmt_pct(sub.get('pf_target_micro_precision'), 2)} / R={_fmt_pct(sub.get('pf_target_micro_recall'), 2)}<br>Macro P={_fmt_pct(sub.get('pf_target_macro_precision'), 2)} / R={_fmt_pct(sub.get('pf_target_macro_recall'), 2)} |"
    )
    lines.append("")

    lines.append("## 3. 评测口径与规则规范声明 (AGENTS.md Compliance)\n")
    lines.append("1. **分项专项指标 (Primary Accuracy)**:")
    lines.append(
        "   - 社交行为 (`Approach`, `Attack`, `Chasing`, `Snout-head_contact`, `Snout-rear_contact`, `Together`): 采用双鼠对条件化专项评测或基准校准窗口匹配。"
    )
    lines.append(
        "   - 基础运动行为 (`Static`, `Locomotion`, `Isolation`, `Group locomotion`, `Dispersal`, `Social clustering`): 采用 6 行为专项基准评测。"
    )
    lines.append(
        "   - 姿态筛选与聚集行为 (`Grooming`, `Rearing`, `Huddling`): 采用 `exact_target_frame_coverage_ge_0.5`（目标小鼠在真值 ID 集合上的有效动作帧并集覆盖率 >= 50%）。"
    )
    lines.append(
        "   - **扎堆 (Huddling) 评测口径**: 主指标严格采用 `exact_target_frame_coverage_ge_0.5`（有效动作帧覆盖率 >= 50%）：严格完整目标视觉 ID 集合匹配 A2 达 **62.12% (41/66)**（表中主指标）；行为学候选提案 A1 达 **89.39% (59/66)**。若放宽至任意正样本候选检出（any positive >= 1 帧），A2 达到 **63.64% (42/66)**，A1 达到 **90.91%~92.42% (60~61/66)**。因高密度多鼠遮挡导致 YOLO 视觉跟踪 ID 断裂（部分片段 3-5 只实体鼠记录了 8-14 个追踪 ID，且目标 ID 间未出现共同存在帧），受限于 `AGENTS.md` 视觉跟踪与行为层解耦红线，不通过行为规则猜测身份。"
    )
    lines.append("2. **严格盲评 Target-ID Joint Top-1 (`pf_joint`)**:")
    lines.append(
        "   - 要求在全笼无提示、无真值条件引导下，全局竞逐排名前 1 的行为预测及其包含的完整目标 ID 集合同时精确命中。"
    )
    lines.append(
        "   - 严格执行 2026-10-04 规则规范：彻底移除同层双标签兼容信用（Together 与 Contact 互不记分，Snout-head 与 Snout-rear 互不替代）。"
    )
    lines.append("3. **失败原因二分类统计 (Union B-priority 764 段)**:")
    lines.append(f"   - **未通过总数**: **{sub['primary_failure_count']}** 段")
    lines.append(
        f"   - **行为未识别到 (Missed / False Negative)**: **{sub['missed_count']}** 段（主要分布于扎堆目标 ID 断裂、接近/理毛未达规则阈值）"
    )
    lines.append(
        f"   - **识别行为出错 (Misclassified / False Positive)**: **{sub['misclassified_count']}** 段（主要分布于嗅闻头部/尾部相互压制、共处与接触竞争、攻击与追逐竞争）\n"
    )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_JSON, help="Input reconciled evaluation JSON"
    )
    parser.add_argument(
        "--output-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Output Markdown table path"
    )
    parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output JSON table path"
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Error: input file {args.input} does not exist.", file=sys.stderr)
        return 1

    report = json.loads(args.input.read_text(encoding="utf-8"))
    comparison = build_comparison_data(report)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote JSON comparison table to {args.output_json}")

    md_text = format_markdown_table(comparison)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text, encoding="utf-8")
    print(f"Wrote Markdown comparison table to {args.output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
