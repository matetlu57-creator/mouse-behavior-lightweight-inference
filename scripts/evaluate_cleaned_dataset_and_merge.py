#!/usr/bin/env python3
"""Evaluate newly cleaned Dataset C (764 clips) and merge with Dataset A+B baselines.

Outputs under ``--output-dir`` (default: ``outputs/cleaned_dataset_c_20261121_eval``):
  - ``dataset_c_report.json``: Full ``all_behaviors=True`` evaluation of Dataset C.
  - ``comparison_and_merged_summary.json``: Machine-readable 3-metric comparison & A∪B∪C merge.
  - ``comparison_and_merged_summary.md``: Chinese Markdown tables for standalone & merged cohorts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from _bootstrap import REPO_ROOT  # noqa: F401

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_dataset_comparison_table import (  # noqa: E402
    ALL_15_BEHAVIORS,
    BEHAVIOR_CHINESE,
    BEHAVIOR_LAYER,
    _compute_cohort_precision_recall,
    _safe_f1,
)
from mouse_behavior.evaluation.reconciled_evaluation import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PRIMARY_METHOD_BY_BEHAVIOR,
    SUSPENDED_DISPLAY_BEHAVIORS,
    _sanitize_json_value,
    aggregate_cohort_metrics,
    evaluate_single_dataset,
)

DEFAULT_DATASET_C_ROOT = Path(r"F:\mouse_behavior_cleaned_dataset_20261121")
DEFAULT_AB_QUOTE2_REPORT = Path("outputs/reconciled_behavior_evaluation.json")
DEFAULT_OUTPUT_DIR = Path("outputs/cleaned_dataset_c_20261121_eval")


def _mean_non_none(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _round_delta(lhs: float | None, rhs: float | None, *, digits: int = 4) -> float | None:
    if lhs is None or rhs is None:
        return None
    return round(float(lhs) - float(rhs), digits)


def _read_source_window(clip_dir_str: str | None) -> tuple[str, int | None, int | None]:
    if not clip_dir_str:
        return "", None, None
    meta_path = Path(clip_dir_str) / "metadata.json"
    if not meta_path.is_file():
        return "", None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", None, None
    src = meta.get("source", {})
    if not isinstance(src, Mapping):
        return "", None, None
    src_clip = src.get("clip") if isinstance(src.get("clip"), Mapping) else src
    start_f = src_clip.get("start_frame")
    end_f = src_clip.get("end_frame")
    return (
        str(src.get("video_filename", "")),
        int(start_f) if start_f is not None else None,
        int(end_f) if end_f is not None else None,
    )


def _summarize_cohort(cohort_name: str, clips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute primary, pf_target, and strict pf_joint metrics with Quote-2 parity."""
    agg = aggregate_cohort_metrics(clips, cohort_name)
    per_beh_pr, agg_pr = _compute_cohort_precision_recall({"clips": clips}, lambda _c: True)
    agg_per_beh = agg.get("per_behavior", {})

    present_rows = [
        per_beh_pr[b] for b in ALL_15_BEHAVIORS if agg_per_beh.get(b, {}).get("support", 0) > 0
    ]

    summary: dict[str, Any] = {
        "cohort": cohort_name,
        "total_clips": agg["total_clips"],
        "primary_correct": agg["overall_primary_correct"],
        "primary_accuracy": agg["overall_primary_accuracy"],
        "overall_coverage_a2_correct": agg["overall_coverage_a2_correct"],
        "overall_coverage_a2_accuracy": agg["overall_coverage_a2_accuracy"],
        "pf_joint_correct": agg["overall_pf_joint_correct"],
        "pf_joint_accuracy": agg["overall_pf_joint_accuracy"],
        "primary_failure_count": agg["primary_failure_count"],
        "missed_count": agg["primary_missed_count"],
        "misclassified_count": agg["primary_misclassified_count"],
        "failure_subcategory_counts": agg.get("failure_subcategory_counts", {}),
        "joint_failure_subcategory_counts": agg.get("joint_failure_subcategory_counts", {}),
        **agg_pr,
        "primary_micro_f1": _safe_f1(
            agg_pr.get("primary_micro_precision"), agg_pr.get("primary_micro_recall")
        ),
        "primary_macro_f1": _mean_non_none([r.get("primary_f1") for r in present_rows]),
        "pf_target_micro_f1": _safe_f1(
            agg_pr.get("pf_target_micro_precision"), agg_pr.get("pf_target_micro_recall")
        ),
        "pf_target_macro_f1": _mean_non_none([r.get("pf_target_f1") for r in present_rows]),
        "pf_joint_micro_f1": _safe_f1(
            agg_pr.get("pf_joint_micro_precision"), agg_pr.get("pf_joint_micro_recall")
        ),
        "pf_joint_macro_f1": _mean_non_none([r.get("pf_joint_f1") for r in present_rows]),
    }

    per_behavior: dict[str, dict[str, Any]] = {}
    for beh in ALL_15_BEHAVIORS:
        r_agg = agg_per_beh.get(beh, {})
        r_pr = per_beh_pr[beh]
        sup = int(r_agg.get("support", 0))
        default_method = (
            "exact_target_frame_coverage_ge_0.5"
            if beh in {"Grooming", "Rearing", "Huddling"}
            else PRIMARY_METHOD_BY_BEHAVIOR[beh]
        )
        row: dict[str, Any] = {
            "behavior": beh,
            "behavior_zh": BEHAVIOR_CHINESE.get(beh, beh),
            "layer": BEHAVIOR_LAYER.get(beh, "Unknown"),
            "primary_method": r_agg.get("primary_method", default_method),
            "rule_enabled": bool(r_agg.get("rule_enabled", True)),
            "support": sup,
            "primary_correct": int(r_agg.get("primary_correct", 0)),
            "primary_accuracy": r_agg.get("primary_accuracy"),
            "coverage_a2_correct": int(r_agg.get("coverage_a2_correct", 0)),
            "coverage_a2_accuracy": r_agg.get("coverage_a2_accuracy"),
            "pf_joint_correct": int(r_agg.get("pf_joint_correct", 0)),
            "pf_joint_accuracy": r_agg.get("pf_joint_accuracy"),
            "pf_target_accuracy": r_agg.get("pf_target_accuracy"),
            "missed_count": int(r_agg.get("missed_count", 0)),
            "misclassified_count": int(r_agg.get("misclassified_count", 0)),
            **r_pr,
        }
        if beh == "Huddling":
            row["a1_coverage_count"] = r_agg.get("a1_coverage_count", 0)
            row["a1_coverage_rate"] = r_agg.get("a1_coverage_rate")
            row["a2_coverage_count"] = r_agg.get("a2_coverage_count", 0)
            row["a2_coverage_rate"] = r_agg.get("a2_coverage_rate")
        if "mean_temporal_iou_correct" in r_agg:
            row["mean_temporal_iou_all"] = r_agg.get("mean_temporal_iou_all")
            row["mean_temporal_iou_correct"] = r_agg.get("mean_temporal_iou_correct")
            row["iou_ge_05_count"] = r_agg.get("iou_ge_05_count")
            row["iou_ge_05_rate"] = r_agg.get("iou_ge_05_rate")
        per_behavior[beh] = row

    return {"summary": summary, "per_behavior": per_behavior}


def _reconcile_abc(
    ab_report: Mapping[str, Any],
    c_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Audit A, B, C overlaps and assemble merged deduplicated union (B > C > A) and raw ref."""
    a_clips = [dict(c) for c in ab_report.get("clips", ()) if c.get("dataset") == "A"]
    b_clips = [dict(c) for c in ab_report.get("clips", ()) if c.get("dataset") == "B"]
    c_clips = [dict(c) for c in c_report.get("clips", ())]

    by_a = {c["clip_id"]: c for c in a_clips}
    by_b = {c["clip_id"]: c for c in b_clips}
    by_c = {c["clip_id"]: c for c in c_clips}

    ids_a = set(by_a)
    ids_b = set(by_b)
    ids_c = set(by_c)

    ab_overlap_ids = sorted(ids_a & ids_b)
    ac_overlap_ids = sorted(ids_a & ids_c)
    bc_overlap_ids = sorted(ids_b & ids_c)
    abc_overlap_ids = sorted(ids_a & ids_b & ids_c)
    a_only_ab_ids = sorted(ids_a - ids_b)
    a_only_after_bc_ids = sorted(ids_a - ids_b - ids_c)

    union_ab_b_priority = [*b_clips, *[c for c in a_clips if c["clip_id"] not in ids_b]]
    merged_union_abc = [
        *b_clips,
        *[c for c in c_clips if c["clip_id"] not in ids_b],
        *[c for c in a_clips if c["clip_id"] not in ids_b and c["clip_id"] not in ids_c],
    ]
    merged_raw_union_plus_c = [*union_ab_b_priority, *c_clips]

    ac_overlap_details: list[dict[str, Any]] = []
    for cid in ac_overlap_ids:
        ra = by_a[cid]
        rc = by_c[cid]
        sv_a, s_a, e_a = _read_source_window(ra.get("clip_dir"))
        sv_c, s_c, e_c = _read_source_window(rc.get("clip_dir"))
        ac_overlap_details.append(
            {
                "clip_id": cid,
                "in_dataset_b": cid in ids_b,
                "in_union_ab_as_a_only": cid not in ids_b,
                "a_behavior": ra.get("behavior"),
                "c_behavior": rc.get("behavior"),
                "label_match": ra.get("behavior") == rc.get("behavior"),
                "a_target_mouse_ids": list(ra.get("target_mouse_ids", [])),
                "c_target_mouse_ids": list(rc.get("target_mouse_ids", [])),
                "target_ids_match": list(ra.get("target_mouse_ids", []))
                == list(rc.get("target_mouse_ids", [])),
                "tracks_sha256_differ": ra.get("tracks_sha256") != rc.get("tracks_sha256"),
                "a_primary_correct": bool(ra.get("primary_correct")),
                "c_primary_correct": bool(rc.get("primary_correct")),
                "a_primary_predicted": ra.get("primary_predicted_behavior"),
                "c_primary_predicted": rc.get("primary_predicted_behavior"),
                "a_pf_target_correct": bool(ra.get("pf_target_correct")),
                "c_pf_target_correct": bool(rc.get("pf_target_correct")),
                "a_pf_joint_correct": bool(ra.get("pf_joint_correct")),
                "c_pf_joint_correct": bool(rc.get("pf_joint_correct")),
                "a_coverage_a2_correct": bool(ra.get("coverage_a2_correct")),
                "c_coverage_a2_correct": bool(rc.get("coverage_a2_correct")),
                "source_video": sv_c or sv_a or rc.get("source_video") or ra.get("source_video") or "",
                "source_window_a": [s_a, e_a] if s_a is not None and e_a is not None else None,
                "source_window_c": [s_c, e_c] if s_c is not None and e_c is not None else None,
            }
        )

    # Auxiliary audit: source_video + overlapping source frame window + target_ids across non-identical clip_ids
    ab_windows: list[tuple[dict[str, Any], str, int, int, tuple[int, ...]]] = []
    for rab in union_ab_b_priority:
        sv, sf, ef = _read_source_window(rab.get("clip_dir"))
        if sv and sf is not None and ef is not None:
            tids = tuple(sorted(int(x) for x in rab.get("target_mouse_ids", [])))
            ab_windows.append((rab, sv, sf, ef, tids))

    window_overlaps: list[dict[str, Any]] = []
    same_id_window_overlaps: list[dict[str, Any]] = []
    for rc in c_clips:
        sv_c, sf_c, ef_c = _read_source_window(rc.get("clip_dir"))
        if not sv_c or sf_c is None or ef_c is None:
            continue
        tids_c = tuple(sorted(int(x) for x in rc.get("target_mouse_ids", [])))
        for rab, sv_ab, sf_ab, ef_ab, tids_ab in ab_windows:
            if rc["clip_id"] == rab["clip_id"] or sv_c != sv_ab:
                continue
            if max(sf_c, sf_ab) <= min(ef_c, ef_ab):
                same_ids = tids_c == tids_ab
                item = {
                    "c_clip_id": rc["clip_id"],
                    "c_behavior": rc["behavior"],
                    "c_window": [sf_c, ef_c],
                    "c_target_ids": list(tids_c),
                    "ab_dataset": rab["dataset"],
                    "ab_clip_id": rab["clip_id"],
                    "ab_behavior": rab["behavior"],
                    "ab_window": [sf_ab, ef_ab],
                    "ab_target_ids": list(tids_ab),
                    "source_video": sv_c,
                    "same_target_ids": same_ids,
                }
                window_overlaps.append(item)
                if same_ids:
                    same_id_window_overlaps.append(item)

    reconciliation = {
        "dedup_priority": "B > C > A",
        "dataset_a_total": len(a_clips),
        "dataset_b_total": len(b_clips),
        "union_ab_total": len(union_ab_b_priority),
        "dataset_c_total": len(c_clips),
        "overlap_ab_count": len(ab_overlap_ids),
        "overlap_ac_count": len(ac_overlap_ids),
        "overlap_ac_in_a_only_count": sum(1 for cid in ac_overlap_ids if cid not in ids_b),
        "overlap_bc_count": len(bc_overlap_ids),
        "overlap_abc_count": len(abc_overlap_ids),
        "a_only_before_c_count": len(a_only_ab_ids),
        "a_only_after_bc_count": len(a_only_after_bc_ids),
        "merged_union_abc_count": len(merged_union_abc),
        "merged_raw_union_plus_c_count": len(merged_raw_union_plus_c),
        "merged_union_abc_source_breakdown": dict(
            sorted(Counter(c["dataset"] for c in merged_union_abc).items())
        ),
        "ac_overlap_ids": ac_overlap_ids,
        "bc_overlap_ids": bc_overlap_ids,
        "ac_label_agreement_count": sum(1 for d in ac_overlap_details if d["label_match"]),
        "ac_overlap_details": ac_overlap_details,
        "auxiliary_source_window_audit": {
            "non_identical_clip_id_window_overlap_count": len(window_overlaps),
            "same_target_ids_window_overlap_count": len(same_id_window_overlaps),
            "same_target_ids_window_overlaps": same_id_window_overlaps,
            "all_window_overlaps": window_overlaps,
        },
    }
    cohorts_clips = {
        "dataset_a": a_clips,
        "dataset_b": b_clips,
        "union_ab_b_priority": union_ab_b_priority,
        "dataset_c": c_clips,
        "merged_union_abc": merged_union_abc,
        "merged_raw_union_plus_c": merged_raw_union_plus_c,
    }
    return reconciliation, cohorts_clips


def _compute_suite_deltas(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute overall and per-behavior metric deltas (``lhs - rhs``)."""
    ls = lhs["summary"]
    rs = rhs["summary"]
    overall = {
        "total_clips_delta": int(ls["total_clips"]) - int(rs["total_clips"]),
        "primary_accuracy_delta": _round_delta(ls.get("primary_accuracy"), rs.get("primary_accuracy")),
        "primary_micro_precision_delta": _round_delta(
            ls.get("primary_micro_precision"), rs.get("primary_micro_precision")
        ),
        "primary_micro_recall_delta": _round_delta(
            ls.get("primary_micro_recall"), rs.get("primary_micro_recall")
        ),
        "primary_macro_precision_delta": _round_delta(
            ls.get("primary_macro_precision"), rs.get("primary_macro_precision")
        ),
        "primary_macro_recall_delta": _round_delta(
            ls.get("primary_macro_recall"), rs.get("primary_macro_recall")
        ),
        "coverage_a2_accuracy_delta": _round_delta(
            ls.get("overall_coverage_a2_accuracy"), rs.get("overall_coverage_a2_accuracy")
        ),
        "pf_target_micro_precision_delta": _round_delta(
            ls.get("pf_target_micro_precision"), rs.get("pf_target_micro_precision")
        ),
        "pf_target_micro_recall_delta": _round_delta(
            ls.get("pf_target_micro_recall"), rs.get("pf_target_micro_recall")
        ),
        "pf_target_macro_precision_delta": _round_delta(
            ls.get("pf_target_macro_precision"), rs.get("pf_target_macro_precision")
        ),
        "pf_target_macro_recall_delta": _round_delta(
            ls.get("pf_target_macro_recall"), rs.get("pf_target_macro_recall")
        ),
        "pf_joint_accuracy_delta": _round_delta(
            ls.get("pf_joint_accuracy"), rs.get("pf_joint_accuracy")
        ),
        "pf_joint_micro_precision_delta": _round_delta(
            ls.get("pf_joint_micro_precision"), rs.get("pf_joint_micro_precision")
        ),
        "pf_joint_macro_precision_delta": _round_delta(
            ls.get("pf_joint_macro_precision"), rs.get("pf_joint_macro_precision")
        ),
        "pf_joint_macro_recall_delta": _round_delta(
            ls.get("pf_joint_macro_recall"), rs.get("pf_joint_macro_recall")
        ),
    }
    per_behavior: dict[str, dict[str, Any]] = {}
    for beh in ALL_15_BEHAVIORS:
        lb = lhs["per_behavior"][beh]
        rb = rhs["per_behavior"][beh]
        per_behavior[beh] = {
            "support_delta": int(lb["support"]) - int(rb["support"]),
            "primary_recall_delta": _round_delta(lb.get("primary_recall"), rb.get("primary_recall")),
            "primary_f1_delta": _round_delta(lb.get("primary_f1"), rb.get("primary_f1")),
            "pf_target_precision_delta": _round_delta(
                lb.get("pf_target_precision"), rb.get("pf_target_precision")
            ),
            "pf_target_recall_delta": _round_delta(
                lb.get("pf_target_recall"), rb.get("pf_target_recall")
            ),
            "pf_target_f1_delta": _round_delta(lb.get("pf_target_f1"), rb.get("pf_target_f1")),
            "pf_joint_precision_delta": _round_delta(
                lb.get("pf_joint_precision"), rb.get("pf_joint_precision")
            ),
            "pf_joint_recall_delta": _round_delta(
                lb.get("pf_joint_recall"), rb.get("pf_joint_recall")
            ),
            "pf_joint_f1_delta": _round_delta(lb.get("pf_joint_f1"), rb.get("pf_joint_f1")),
        }
    return {"overall": overall, "per_behavior": per_behavior}


def _evaluate_baseline_suite(
    ab_report: Mapping[str, Any],
    c_report: Mapping[str, Any],
) -> dict[str, Any]:
    reconciliation, cohorts_clips = _reconcile_abc(ab_report, c_report)
    cohorts = {
        name: _summarize_cohort(name, clips) for name, clips in cohorts_clips.items()
    }
    deltas = {
        "c_minus_union_ab": _compute_suite_deltas(
            cohorts["dataset_c"], cohorts["union_ab_b_priority"]
        ),
        "merged_abc_minus_union_ab": _compute_suite_deltas(
            cohorts["merged_union_abc"], cohorts["union_ab_b_priority"]
        ),
        "merged_abc_minus_raw_1528": _compute_suite_deltas(
            cohorts["merged_union_abc"], cohorts["merged_raw_union_plus_c"]
        ),
    }
    return {
        "reconciliation_abc": reconciliation,
        "cohorts": cohorts,
        "deltas": deltas,
    }


def _diff_ab_reports(
    quote2_report: Mapping[str, Any],
    current_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare Quote 2 baseline against current A+B report on summary, rows, and per-clip records."""
    q_suite = {
        name: _summarize_cohort(
            name,
            [c for c in quote2_report.get("clips", ()) if pred(c)],
        )
        for name, pred in (
            ("dataset_a", lambda c: c.get("dataset") == "A"),
            ("dataset_b", lambda c: c.get("dataset") == "B"),
            ("union_ab_b_priority", lambda c: bool(c.get("in_union_b_priority"))),
        )
    }
    c_suite = {
        name: _summarize_cohort(
            name,
            [c for c in current_report.get("clips", ()) if pred(c)],
        )
        for name, pred in (
            ("dataset_a", lambda c: c.get("dataset") == "A"),
            ("dataset_b", lambda c: c.get("dataset") == "B"),
            ("union_ab_b_priority", lambda c: bool(c.get("in_union_b_priority"))),
        )
    }

    summary_diffs: list[dict[str, Any]] = []
    behavior_row_diffs: list[dict[str, Any]] = []
    for cohort_name in ("dataset_a", "dataset_b", "union_ab_b_priority"):
        qs = q_suite[cohort_name]["summary"]
        cs = c_suite[cohort_name]["summary"]
        for key in sorted(set(qs) | set(cs)):
            if key in {"failure_subcategory_counts", "joint_failure_subcategory_counts"}:
                if qs.get(key) != cs.get(key):
                    summary_diffs.append(
                        {"cohort": cohort_name, "field": key, "quote2": qs.get(key), "current": cs.get(key)}
                    )
            elif qs.get(key) != cs.get(key):
                summary_diffs.append(
                    {"cohort": cohort_name, "field": key, "quote2": qs.get(key), "current": cs.get(key)}
                )
        for beh in ALL_15_BEHAVIORS:
            qb = q_suite[cohort_name]["per_behavior"][beh]
            cb = c_suite[cohort_name]["per_behavior"][beh]
            for key in sorted(set(qb) | set(cb)):
                if qb.get(key) != cb.get(key):
                    behavior_row_diffs.append(
                        {
                            "cohort": cohort_name,
                            "behavior": beh,
                            "field": key,
                            "quote2": qb.get(key),
                            "current": cb.get(key),
                        }
                    )

    q_clips = {c["record_key"]: c for c in quote2_report.get("clips", ())}
    c_clips = {c["record_key"]: c for c in current_report.get("clips", ())}
    per_clip_diffs: list[dict[str, Any]] = []
    check_fields = (
        "primary_correct",
        "primary_predicted_behavior",
        "pf_joint_correct",
        "pf_joint_predicted_behavior",
        "pf_joint_predicted_target_ids",
        "pf_target_correct",
        "pf_target_predicted_behavior",
        "coverage_a2_correct",
    )
    for rkey in sorted(set(q_clips) | set(c_clips)):
        qr = q_clips.get(rkey)
        cr = c_clips.get(rkey)
        if qr is None or cr is None:
            per_clip_diffs.append({"record_key": rkey, "reason": "missing_record"})
            continue
        for f in check_fields:
            if qr.get(f) != cr.get(f):
                per_clip_diffs.append(
                    {"record_key": rkey, "field": f, "quote2": qr.get(f), "current": cr.get(f)}
                )

    return {
        "summary_diffs_count": len(summary_diffs),
        "behavior_row_diffs_count": len(behavior_row_diffs),
        "per_clip_diffs_count": len(per_clip_diffs),
        "exact_match": not (summary_diffs or behavior_row_diffs or per_clip_diffs),
        "summary_diffs": summary_diffs[:50],
        "behavior_row_diffs": behavior_row_diffs[:50],
        "per_clip_diffs": per_clip_diffs[:50],
    }


def _audit_dataset_c(c_report: Mapping[str, Any]) -> dict[str, Any]:
    clips = list(c_report.get("clips", ()))
    err_clips = [c["clip_id"] for c in clips if c.get("benchmark_status") == "error"]
    disabled_clips = [
        c["clip_id"]
        for c in clips
        if c.get("canonical_rule_status") != "enabled"
        or c.get("failure_subcategory") == "disabled_rule"
    ]
    moving_clips = [c for c in clips if c.get("annotation_behavior") == "Moving"]
    grooming_clips = [c for c in clips if c.get("behavior") == "Grooming"]
    rearing_clips = [c for c in clips if c.get("behavior") == "Rearing"]
    return {
        "total_clips": len(clips),
        "all_behaviors": bool(c_report.get("inputs", {}).get("all_behaviors", False)),
        "enable_posture_screen": bool(
            c_report.get("inputs", {}).get("enable_posture_screen", False)
        ),
        "dual_label_credit_enabled": bool(
            c_report.get("inputs", {}).get("dual_label_credit_enabled", False)
        ),
        "benchmark_error_count": len(err_clips),
        "benchmark_error_clip_ids": err_clips,
        "disabled_rule_clip_count": len(disabled_clips),
        "moving_to_locomotion_count": len(moving_clips),
        "grooming_count": len(grooming_clips),
        "rearing_count": len(rearing_clips),
        "raw_behavior_counts": dict(c_report.get("raw_behavior_counts", {})),
        "normalized_behavior_counts": dict(c_report.get("normalized_behavior_counts", {})),
        "scale_source_counts": dict(c_report.get("scale_source_counts", {})),
    }


def _fmt_pct(val: float | None, digits: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.{digits}f}%"


def _fmt_delta_pp(val: float | None, digits: int = 2) -> str:
    if val is None:
        return "—"
    pp = val * 100.0
    sign = "+" if pp > 0 else ""
    return f"{sign}{pp:.{digits}f} pp"


def _fmt_prf_cell(
    tp: int,
    sup: int,
    pred: int,
    p: float | None,
    r: float | None,
    f1: float | None,
) -> str:
    if sup <= 0 and pred <= 0:
        return "—"
    if sup <= 0:
        return f"Pred={pred} (TP=0, P=0.0%)"
    return f"{tp}/{sup}<br>P={_fmt_pct(p, 1)} ({tp}/{pred}) / R={_fmt_pct(r, 1)} / F1={_fmt_pct(f1, 1)}"


def format_merged_markdown(summary_payload: Mapping[str, Any]) -> str:
    """Render comprehensive Chinese Markdown report for standalone & merged cohorts."""
    meta = summary_payload["metadata"]
    c_audit = summary_payload["dataset_c_audit"]
    q_suite = summary_payload["baselines"]["quote2"]
    recon = q_suite["reconciliation_abc"]
    cohorts = q_suite["cohorts"]
    deltas = q_suite["deltas"]
    aux = recon["auxiliary_source_window_audit"]
    c_cohort = cohorts["dataset_c"]
    non_empty_c = sum(1 for b in ALL_15_BEHAVIORS if c_cohort["per_behavior"][b]["support"] > 0)
    zero_c = [b for b in ALL_15_BEHAVIORS if c_cohort["per_behavior"][b]["support"] == 0]
    meta_scale_cnt = c_audit["scale_source_counts"].get("metadata_distance_scale", 0)
    default_scale_cnt = c_audit["scale_source_counts"].get("default_scale_config_body_length", 0)

    lines: list[str] = [
        f"# 人工清洗数据集 C ({c_audit['total_clips']} 段) 评测与 A+B+C 三集合并对比报告",
        "",
        f"- **数据集 C 路径**: `{meta['dataset_c_root']}`",
        f"- **Quote 2 A+B 基线**: `{meta['ab_quote2_report']}`",
        f"- **当前最新代码 A+B 基线**: `{meta.get('ab_current_report') or '未提供（仅使用 Quote 2 基线）'}`",
        f"- **评测配置**: `all_behaviors={c_audit['all_behaviors']}`, `enable_posture_screen={c_audit['enable_posture_screen']}`, "
        f"`dual_label_credit_enabled={c_audit['dual_label_credit_enabled']}` (严格禁止 Together 与接触标签互通计分)",
        f"- **标签规范化审计**: `Moving` ({c_audit['moving_to_locomotion_count']} 段) 已全部规范化映射为 `Locomotion`，"
        f"保留 `15` 类正典常量契约；`Grooming` ({c_audit['grooming_count']} 段) 与 `Rearing` ({c_audit['rearing_count']} 段) 全量参与评测",
        f"- **尺度来源统计 (C)**: {c_audit['scale_source_counts']}（其中 `{meta_scale_cnt}` 段使用 `metadata.distance_scale` 固定尺度，`{default_scale_cnt}` 段使用默认体长尺度 `7.918 cm`）",
        f"- **Benchmark 错误数**: `{c_audit['benchmark_error_count']}` | 规则停用数: `{c_audit['disabled_rule_clip_count']}`",
        "",
        f"## 1. 三集合对齐、重叠与去重审计 (A={recon['dataset_a_total']}, B={recon['dataset_b_total']}, C={recon['dataset_c_total']})",
        "",
        f"- **Dataset A**: `{recon['dataset_a_total']}` 段（补全集）",
        f"- **Dataset B**: `{recon['dataset_b_total']}` 段（人工清洗集）",
        f"- **Union(A, B) B 优先**: `{recon['union_ab_total']}` 段（`{recon['dataset_b_total']}` 段 B + `{recon['a_only_before_c_count']}` 段 A 独有；`A∩B={recon['overlap_ab_count']}`）",
        f"- **Dataset C**: `{recon['dataset_c_total']}` 段（`{non_empty_c}` 个非空规范行为，零样本行为={zero_c or '无'}，含 `Moving={c_audit['moving_to_locomotion_count']} -> Locomotion`）",
        "- **精确 `clip_id` 重叠关系**:",
        f"  - `A ∩ C` = **`{recon['overlap_ac_count']}`** 段（其中 `{recon['overlap_ac_in_a_only_count']}` 段位于 Union(A,B) 的 A-only 中，标签一致率 `{recon['ac_label_agreement_count']}/{recon['overlap_ac_count']}`）",
        f"  - `B ∩ C` = **`{recon['overlap_bc_count']}`** 段 | `A ∩ B ∩ C` = **`{recon['overlap_abc_count']}`** 段",
        f"- **三集合并去重并集 `merged_union_abc`（优先级 `B > C > A`）**: **`{recon['merged_union_abc_count']}`** 段 "
        f"（`{recon['union_ab_total']} + {recon['dataset_c_total']} - {recon['overlap_ac_in_a_only_count']} = {recon['merged_union_abc_count']}`；来源构成：`{recon['merged_union_abc_source_breakdown']}`）",
        f"- **未去重拼接参考 `merged_raw_union_plus_c`**: **`{recon['merged_raw_union_plus_c_count']}`** 段（`{recon['union_ab_total']} + {recon['dataset_c_total']} = {recon['merged_raw_union_plus_c_count']}`）",
        f"- **辅助源视频/时间窗审计**: 在非相同 `clip_id` 中，同 `source_video` 且源帧区间重叠的有 `{aux['non_identical_clip_id_window_overlap_count']}` 对，"
        f"其中目标 `target_ids` 也完全相同的有 `{aux['same_target_ids_window_overlap_count']}` 对（已保留审计记录，未从 {recon['merged_union_abc_count']} `clip_id` 去重并集中额外剔除）。",
    ]

    if summary_payload.get("ab_quote2_vs_current_diff") is not None:
        diff_info = summary_payload["ab_quote2_vs_current_diff"]
        lines.append(
            f"- **Quote 2 基线 vs 当前代码 A+B 一致性核验**: `exact_match={diff_info['exact_match']}` "
            f"(summary_diffs=`{diff_info['summary_diffs_count']}`, row_diffs=`{diff_info['behavior_row_diffs_count']}`, "
            f"per_clip_diffs=`{diff_info['per_clip_diffs_count']}`)"
        )
    lines.append("")

    # Section 2: Overall table across all cohorts
    lines.extend(
        [
            "## 2. 全体分母三大指标总体对比总表（Quote 2 基线口径）",
            "",
            "| 数据集 / 合并分母 | 片段数 | `pf_target` (15类竞争模型)<br>TP/N · Micro P/R · Macro P/R | `pf_joint` (严格盲评 Joint Top-1)<br>Acc · Micro P · Macro P/R | `primary` (分项专项主指标)<br>Acc · Micro P · Macro P/R | 统一 `exact-ID >=50%` 覆盖 |",
            "|---|---:|:---:|:---:|:---:|:---:|",
        ]
    )

    cohort_labels = (
        ("dataset_a", f"Dataset A ({recon['dataset_a_total']})"),
        ("dataset_b", f"Dataset B ({recon['dataset_b_total']})"),
        ("union_ab_b_priority", f"Union(A,B) B优先 ({recon['union_ab_total']})"),
        ("dataset_c", f"Dataset C 新清洗集 ({recon['dataset_c_total']})"),
        ("merged_union_abc", f"**合并去重并集 A∪B∪C (B>C>A, {recon['merged_union_abc_count']})**"),
        ("merged_raw_union_plus_c", f"未去重参考 Union(A,B)+C ({recon['merged_raw_union_plus_c_count']})"),
    )
    for key, label in cohort_labels:
        s = cohorts[key]["summary"]
        n = s["total_clips"]
        pt_cell = (
            f"{s['pf_target_correct']}/{n} ({_fmt_pct(s['pf_target_micro_recall'])})<br>"
            f"Micro P={_fmt_pct(s['pf_target_micro_precision'])} / R={_fmt_pct(s['pf_target_micro_recall'])}<br>"
            f"Macro P={_fmt_pct(s['pf_target_macro_precision'])} / R={_fmt_pct(s['pf_target_macro_recall'])}"
        )
        pj_cell = (
            f"{s['pf_joint_correct']}/{n} ({_fmt_pct(s['pf_joint_accuracy'])})<br>"
            f"Micro P={_fmt_pct(s['pf_joint_micro_precision'])}<br>"
            f"Macro P={_fmt_pct(s['pf_joint_macro_precision'])} / R={_fmt_pct(s['pf_joint_macro_recall'])}"
        )
        pr_cell = (
            f"{s['primary_correct']}/{n} ({_fmt_pct(s['primary_accuracy'])})<br>"
            f"Micro P={_fmt_pct(s['primary_micro_precision'])}<br>"
            f"Macro P={_fmt_pct(s['primary_macro_precision'])} / R={_fmt_pct(s['primary_macro_recall'])}"
        )
        cov_cell = f"{s['overall_coverage_a2_correct']}/{n} ({_fmt_pct(s['overall_coverage_a2_accuracy'])})"
        lines.append(f"| {label} | {n} | {pt_cell} | {pj_cell} | {pr_cell} | {cov_cell} |")

    for d_key, d_label in (
        ("c_minus_union_ab", f"**差值: Dataset C ({recon['dataset_c_total']}) − Union(A,B) ({recon['union_ab_total']})**"),
        ("merged_abc_minus_union_ab", f"**差值: 合并并集 ({recon['merged_union_abc_count']}) − Union(A,B) ({recon['union_ab_total']})**"),
        ("merged_abc_minus_raw_1528", f"敏感性差值: 去重 ({recon['merged_union_abc_count']}) − 未去重 ({recon['merged_raw_union_plus_c_count']})"),
    ):
        d = deltas[d_key]["overall"]
        lines.append(
            f"| {d_label} | {d['total_clips_delta']:+d} | "
            f"Micro P {_fmt_delta_pp(d['pf_target_micro_precision_delta'])} / R {_fmt_delta_pp(d['pf_target_micro_recall_delta'])}<br>"
            f"Macro P {_fmt_delta_pp(d['pf_target_macro_precision_delta'])} / R {_fmt_delta_pp(d['pf_target_macro_recall_delta'])} | "
            f"Acc {_fmt_delta_pp(d['pf_joint_accuracy_delta'])}<br>"
            f"Macro P {_fmt_delta_pp(d['pf_joint_macro_precision_delta'])} / R {_fmt_delta_pp(d['pf_joint_macro_recall_delta'])} | "
            f"Acc {_fmt_delta_pp(d['primary_accuracy_delta'])}<br>"
            f"Macro P {_fmt_delta_pp(d['primary_macro_precision_delta'])} / R {_fmt_delta_pp(d['primary_macro_recall_delta'])} | "
            f"{_fmt_delta_pp(d['coverage_a2_accuracy_delta'])} |"
        )
    lines.append("")

    # Section 3: Standalone Dataset C
    lines.extend(
        [
            f"## 3. Dataset C ({recon['dataset_c_total']} 段) 单独展开明细表",
            "",
            f"> 注：`Moving` ({c_audit['moving_to_locomotion_count']} 段) 规范化并入 `Locomotion`；零真值行为（{zero_c or '无'}）若存在模型误报仍计入 `Pred` 分母并压低 Micro Precision。`Grooming` ({c_audit['grooming_count']} 段)、`Rearing` ({c_audit['rearing_count']} 段) 与 `Huddling` ({c_cohort['per_behavior']['Huddling']['support']} 段) 在 `all_behaviors=True` 下主指标均采用 `exact_target_frame_coverage_ge_0.5`。",
            "",
            "| 行为类别 (中/英) | 层级 | Support | 专项主指标 `primary`<br>(TP/Sup · P/R/F1) | Exact-ID 覆盖>=50% | 统一15类 `pf_target`<br>(TP/Sup · P/R/F1) | 严格盲评 `pf_joint`<br>(TP/Sup · P/R/F1) | 失败 (Missed/Miscls) | 专项评测方法 |",
            "|:---|:---|---:|:---:|:---:|:---:|:---:|:---:|:---|",
        ]
    )
    for beh in ALL_15_BEHAVIORS:
        r = c_cohort["per_behavior"][beh]
        sup = r["support"]
        prim_cell = _fmt_prf_cell(
            r["primary_correct"], sup, r["primary_predicted_count"],
            r["primary_precision"], r["primary_recall"], r["primary_f1"],
        )
        cov_cell = (
            f"{r['coverage_a2_correct']}/{sup} ({_fmt_pct(r['coverage_a2_accuracy'], 1)})"
            if sup > 0
            else "—"
        )
        pt_cell = _fmt_prf_cell(
            r["pf_target_correct"], sup, r["pf_target_predicted_count"],
            r["pf_target_precision"], r["pf_target_recall"], r["pf_target_f1"],
        )
        pj_cell = _fmt_prf_cell(
            r["pf_joint_correct"], sup, r["pf_joint_predicted_count"],
            r["pf_joint_precision"], r["pf_joint_recall"], r["pf_joint_f1"],
        )
        fail_cell = f"{r['missed_count']} / {r['misclassified_count']}" if sup > 0 else "—"
        note = f" *(含 Moving {c_audit['moving_to_locomotion_count']})*" if beh == "Locomotion" and c_audit["moving_to_locomotion_count"] > 0 else ""
        lines.append(
            f"| **{r['behavior_zh']}**{note}<br>`{beh}` | {r['layer']} | {sup} | {prim_cell} | {cov_cell} | "
            f"{pt_cell} | {pj_cell} | {fail_cell} | `{r['primary_method']}` |"
        )
    cs = c_cohort["summary"]
    lines.append(
        f"| **Dataset C 总体合计** | **全 15 类** | **{cs['total_clips']}** | "
        f"**{cs['primary_correct']}/{cs['total_clips']} ({_fmt_pct(cs['primary_accuracy'])})**<br>Micro P={_fmt_pct(cs['primary_micro_precision'])} / Macro R={_fmt_pct(cs['primary_macro_recall'])} | "
        f"**{cs['overall_coverage_a2_correct']}/{cs['total_clips']} ({_fmt_pct(cs['overall_coverage_a2_accuracy'])})** | "
        f"**{cs['pf_target_correct']}/{cs['total_clips']}**<br>Micro P={_fmt_pct(cs['pf_target_micro_precision'])} / R={_fmt_pct(cs['pf_target_micro_recall'])}<br>Macro P={_fmt_pct(cs['pf_target_macro_precision'])} / R={_fmt_pct(cs['pf_target_macro_recall'])} | "
        f"**{cs['pf_joint_correct']}/{cs['total_clips']} ({_fmt_pct(cs['pf_joint_accuracy'])})**<br>Macro P={_fmt_pct(cs['pf_joint_macro_precision'])} / R={_fmt_pct(cs['pf_joint_macro_recall'])} | "
        f"**{cs['missed_count']} / {cs['misclassified_count']}** | — |"
    )
    lines.append("")

    # Section 4: Per-behavior cross-cohort comparison tables for each of the 3 metric systems
    u_n = recon["union_ab_total"]
    c_n = recon["dataset_c_total"]
    m_n = recon["merged_union_abc_count"]
    r_n = recon["merged_raw_union_plus_c_count"]
    lines.append(f"## 4. 分行为四分母对照表：Union(A,B) {u_n} vs C {c_n} vs 合并并集 {m_n} vs 未去重 {r_n}\n")

    def _append_metric_table(
        title: str,
        tp_key: str,
        pred_key: str,
        p_key: str,
        r_key: str,
        f1_key: str,
        summary_prefix: str,
    ) -> None:
        lines.append(f"### {title}\n")
        lines.append(
            f"| 行为类别 (中/英) | Union(A,B) B优先 ({u_n}) | Dataset C ({c_n}) | **合并去重并集 A∪B∪C ({m_n})** | 未去重参考 ({r_n}) | 合并{m_n} − Union{u_n} |"
        )
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for beh in ALL_15_BEHAVIORS:
            r_u = cohorts["union_ab_b_priority"]["per_behavior"][beh]
            r_c = cohorts["dataset_c"]["per_behavior"][beh]
            r_m = cohorts["merged_union_abc"]["per_behavior"][beh]
            r_raw = cohorts["merged_raw_union_plus_c"]["per_behavior"][beh]
            db = deltas["merged_abc_minus_union_ab"]["per_behavior"][beh]
            f1_delta = db.get(f"{summary_prefix}_f1_delta")
            r_delta = db.get(f"{summary_prefix}_recall_delta")
            d_str = (
                f"Sup {db['support_delta']:+d}<br>R {_fmt_delta_pp(r_delta, 1)} / F1 {_fmt_delta_pp(f1_delta, 1)}"
                if f1_delta is not None or r_delta is not None
                else f"Sup {db['support_delta']:+d}"
            )
            lines.append(
                f"| **{r_m['behavior_zh']}** (`{beh}`) | "
                f"{_fmt_prf_cell(r_u[tp_key], r_u['support'], r_u[pred_key], r_u[p_key], r_u[r_key], r_u[f1_key])} | "
                f"{_fmt_prf_cell(r_c[tp_key], r_c['support'], r_c[pred_key], r_c[p_key], r_c[r_key], r_c[f1_key])} | "
                f"{_fmt_prf_cell(r_m[tp_key], r_m['support'], r_m[pred_key], r_m[p_key], r_m[r_key], r_m[f1_key])} | "
                f"{_fmt_prf_cell(r_raw[tp_key], r_raw['support'], r_raw[pred_key], r_raw[p_key], r_raw[r_key], r_raw[f1_key])} | "
                f"{d_str} |"
            )

        def _tot_cell(cname: str) -> str:
            s = cohorts[cname]["summary"]
            return (
                f"Micro P={_fmt_pct(s.get(f'{summary_prefix}_micro_precision'))} / "
                f"R={_fmt_pct(s.get(f'{summary_prefix}_micro_recall'))}<br>"
                f"Macro P={_fmt_pct(s.get(f'{summary_prefix}_macro_precision'))} / "
                f"R={_fmt_pct(s.get(f'{summary_prefix}_macro_recall'))}"
            )

        lines.append(
            f"| **总体汇总 (Micro / Macro)** | {_tot_cell('union_ab_b_priority')} | {_tot_cell('dataset_c')} | "
            f"**{_tot_cell('merged_union_abc')}** | {_tot_cell('merged_raw_union_plus_c')} | — |"
        )
        lines.append("")

    _append_metric_table(
        "4a. 统一 15 类竞争模型 (`pf_target`) Precision / Recall / F1 对照",
        "pf_target_correct",
        "pf_target_predicted_count",
        "pf_target_precision",
        "pf_target_recall",
        "pf_target_f1",
        "pf_target",
    )
    _append_metric_table(
        "4b. 严格盲评 Target-ID Joint Top-1 (`pf_joint`) 对照",
        "pf_joint_correct",
        "pf_joint_predicted_count",
        "pf_joint_precision",
        "pf_joint_recall",
        "pf_joint_f1",
        "pf_joint",
    )
    _append_metric_table(
        "4c. 各行为分项专项主指标 (`primary`) 对照（含 Grooming/Rearing/Huddling >=50% 精确 ID 覆盖）",
        "primary_correct",
        "primary_predicted_count",
        "primary_precision",
        "primary_recall",
        "primary_f1",
        "primary",
    )

    # Special Huddling A1/A2 note
    hud_u = cohorts["union_ab_b_priority"]["per_behavior"]["Huddling"]
    hud_c = cohorts["dataset_c"]["per_behavior"]["Huddling"]
    hud_m = cohorts["merged_union_abc"]["per_behavior"]["Huddling"]
    lines.append(
        f"- **Huddling (扎堆) A1/A2 覆盖专项明细**: Union(A,B) `A1={hud_u.get('a1_coverage_count')}/{hud_u['support']} ({_fmt_pct(hud_u.get('a1_coverage_rate'))})`, "
        f"`A2={hud_u.get('a2_coverage_count')}/{hud_u['support']} ({_fmt_pct(hud_u.get('a2_coverage_rate'))})`; "
        f"Dataset C `A1={hud_c.get('a1_coverage_count')}/{hud_c['support']} ({_fmt_pct(hud_c.get('a1_coverage_rate'))})`, "
        f"`A2={hud_c.get('a2_coverage_count')}/{hud_c['support']} ({_fmt_pct(hud_c.get('a2_coverage_rate'))})`; "
        f"Merged {m_n} `A1={hud_m.get('a1_coverage_count')}/{hud_m['support']} ({_fmt_pct(hud_m.get('a1_coverage_rate'))})`, "
        f"`A2={hud_m.get('a2_coverage_count')}/{hud_m['support']} ({_fmt_pct(hud_m.get('a2_coverage_rate'))})`。\n"
    )

    # Section 5: Current A+B baseline comparison & A∩C clips
    lines.append(f"## 5. Quote 2 vs 当前代码 A+B 基线差异与 {recon['overlap_ac_count']} 个 `A∩C` 重叠样本核对\n")
    cur_suite = summary_payload["baselines"].get("current")
    if cur_suite is not None:
        diff_info = summary_payload.get("ab_quote2_vs_current_diff", {})
        lines.append(
            f"- **A+B 基线比对结论**: `exact_match={diff_info.get('exact_match')}` "
            f"(summary_diffs=`{diff_info.get('summary_diffs_count')}`, behavior_row_diffs=`{diff_info.get('behavior_row_diffs_count')}`, "
            f"per_clip_diffs=`{diff_info.get('per_clip_diffs_count')}`)。"
        )
        cur_merged = cur_suite["cohorts"]["merged_union_abc"]["summary"]
        q_merged = cohorts["merged_union_abc"]["summary"]
        lines.append(
            f"- **基于当前代码 A+B 的合并去重并集 ({cur_merged['total_clips']}) 汇总**: "
            f"`pf_target`={cur_merged['pf_target_correct']}/{cur_merged['total_clips']} (Micro P={_fmt_pct(cur_merged['pf_target_micro_precision'])}, R={_fmt_pct(cur_merged['pf_target_micro_recall'])}), "
            f"`pf_joint`={cur_merged['pf_joint_correct']}/{cur_merged['total_clips']} ({_fmt_pct(cur_merged['pf_joint_accuracy'])}), "
            f"`primary`={cur_merged['primary_correct']}/{cur_merged['total_clips']} ({_fmt_pct(cur_merged['primary_accuracy'])}) "
            f"[与 Quote 2 合并并集差值: `primary` {_fmt_delta_pp(_round_delta(cur_merged['primary_accuracy'], q_merged['primary_accuracy']))}, "
            f"`pf_joint` {_fmt_delta_pp(_round_delta(cur_merged['pf_joint_accuracy'], q_merged['pf_joint_accuracy']))}, "
            f"`pf_target Micro R` {_fmt_delta_pp(_round_delta(cur_merged['pf_target_micro_recall'], q_merged['pf_target_micro_recall']))}]。\n"
        )
    else:
        lines.append("- 未传入 `--ab-current-report`，仅输出基于 Quote 2 基线的合并对照。\n")

    lines.extend(
        [
            f"### {recon['overlap_ac_count']} 个 `A ∩ C` 重叠样本逐条核对（`B > C > A` 中均采用 C 版本替换 A-only）",
            "",
            "| Clip ID | 行为类别 | 轨迹字节差异 | A Primary | C Primary | A `pf_target` | C `pf_target` | A `pf_joint` | C `pf_joint` |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for d in recon["ac_overlap_details"]:
        lines.append(
            f"| `{d['clip_id']}` | `{d['c_behavior']}` | {'是' if d['tracks_sha256_differ'] else '否'} | "
            f"{'PASS' if d['a_primary_correct'] else 'FAIL'} ({d['a_primary_predicted'] or '—'}) | "
            f"{'PASS' if d['c_primary_correct'] else 'FAIL'} ({d['c_primary_predicted'] or '—'}) | "
            f"{'PASS' if d['a_pf_target_correct'] else 'FAIL'} | {'PASS' if d['c_pf_target_correct'] else 'FAIL'} | "
            f"{'PASS' if d['a_pf_joint_correct'] else 'FAIL'} | {'PASS' if d['c_pf_joint_correct'] else 'FAIL'} |"
        )
    lines.append("")

    # Section 6: AGENTS.md compliance
    lines.extend(
        [
            "## 6. AGENTS.md 红线与口径合规声明",
            "",
            "1. **严格 Target-ID Joint 口径与无双标签互通**: 严禁在 `Together` 与 `Snout-head_contact` / `Snout-rear_contact` 之间给予双标签兼容计分（`dual_label_credit_enabled=False`）。",
            "2. **视觉追踪与身份解耦**: 仅使用 `tracks.json` 中既有的 `visual_track_id` 与标注目标 ID 集合，绝不通过位置、外观、姿态或运动猜测缺失身份。",
            "3. **全行为评测完整性**: `all_behaviors=True`，`Moving` (64 段) 规范化为 `Locomotion`，`Grooming` (118 段) 与 `Rearing` (362 段) 全量参与评测并计入统一 `pf_target` / `pf_joint` / `primary` 总分。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-c",
        type=Path,
        default=DEFAULT_DATASET_C_ROOT,
        help="Path to newly cleaned Dataset C root (default: F:\\mouse_behavior_cleaned_dataset_20261121)",
    )
    parser.add_argument(
        "--ab-quote2-report",
        type=Path,
        default=DEFAULT_AB_QUOTE2_REPORT,
        help="Path to historical Quote 2 A+B report JSON",
    )
    parser.add_argument(
        "--ab-current-report",
        type=Path,
        default=None,
        help="Optional path to current-code A+B report JSON for baseline drift verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write dataset_c_report.json and comparison_and_merged_summary.{json,md}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Parallel workers for Dataset C evaluation (default: 14)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dataset-c-source-scales",
        type=Path,
        default=None,
        help="Optional explicit frozen source_scales.json for Dataset C",
    )
    parser.add_argument(
        "--no-metadata-distance-scale",
        action="store_true",
        help="Disable reading metadata.json distance_scale for Dataset C clips",
    )
    parser.add_argument(
        "--skip-eval-c",
        action="store_true",
        help="Skip re-running Dataset C evaluation and reuse <output-dir>/dataset_c_report.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c_report_path = out_dir / "dataset_c_report.json"

    if not args.ab_quote2_report.is_file():
        print(f"Error: --ab-quote2-report does not exist: {args.ab_quote2_report}", file=sys.stderr)
        return 1
    if args.ab_current_report is not None and not args.ab_current_report.is_file():
        print(f"Error: --ab-current-report does not exist: {args.ab_current_report}", file=sys.stderr)
        return 1

    if args.skip_eval_c:
        if not c_report_path.is_file():
            print(f"Error: --skip-eval-c set, but {c_report_path} does not exist.", file=sys.stderr)
            return 1
        c_report = json.loads(c_report_path.read_text(encoding="utf-8"))
        if not c_report.get("inputs", {}).get("all_behaviors"):
            print("Error: cached dataset_c_report.json was not generated with all_behaviors=True.", file=sys.stderr)
            return 1
    else:
        c_report = evaluate_single_dataset(
            args.dataset_c,
            dataset_label="C",
            config_path=args.config,
            workers=args.workers,
            all_behaviors=True,
            dataset_source_scales=args.dataset_c_source_scales,
            use_metadata_distance_scale=not args.no_metadata_distance_scale,
        )
        c_report_path.write_text(
            json.dumps(_sanitize_json_value(c_report), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote Dataset C report to {c_report_path}")

    ab_quote2_report = json.loads(args.ab_quote2_report.read_text(encoding="utf-8"))
    ab_current_report = (
        json.loads(args.ab_current_report.read_text(encoding="utf-8"))
        if args.ab_current_report is not None
        else None
    )

    c_audit = _audit_dataset_c(c_report)
    quote2_suite = _evaluate_baseline_suite(ab_quote2_report, c_report)
    baselines: dict[str, Any] = {"quote2": quote2_suite}
    ab_diff: dict[str, Any] | None = None
    quote2_vs_current_merged: dict[str, Any] | None = None

    if ab_current_report is not None:
        current_suite = _evaluate_baseline_suite(ab_current_report, c_report)
        baselines["current"] = current_suite
        ab_diff = _diff_ab_reports(ab_quote2_report, ab_current_report)
        quote2_vs_current_merged = {
            cohort_key: _compute_suite_deltas(
                current_suite["cohorts"][cohort_key], quote2_suite["cohorts"][cohort_key]
            )["overall"]
            for cohort_key in ("union_ab_b_priority", "merged_union_abc", "merged_raw_union_plus_c")
        }

    summary_payload = _sanitize_json_value(
        {
            "schema_version": "cleaned_dataset_c_and_merged_abc_v1",
            "metadata": {
                "dataset_c_root": str(args.dataset_c),
                "c_report_path": str(c_report_path),
                "ab_quote2_report": str(args.ab_quote2_report),
                "ab_current_report": str(args.ab_current_report) if args.ab_current_report else None,
                "all_behaviors": True,
                "dual_label_credit_enabled": False,
                "suspended_posture_behaviors": [] if c_audit["all_behaviors"] else sorted(SUSPENDED_DISPLAY_BEHAVIORS),
                "scale_policy_c": c_report.get("inputs", {}).get("scale_policy"),
            },
            "dataset_c_audit": c_audit,
            "ab_quote2_vs_current_diff": ab_diff,
            "standalone": {
                "dataset_c": quote2_suite["cohorts"]["dataset_c"],
                "ab_quote2": {
                    "dataset_a": quote2_suite["cohorts"]["dataset_a"],
                    "dataset_b": quote2_suite["cohorts"]["dataset_b"],
                    "union_ab_b_priority": quote2_suite["cohorts"]["union_ab_b_priority"],
                },
                "ab_current": (
                    {
                        "dataset_a": baselines["current"]["cohorts"]["dataset_a"],
                        "dataset_b": baselines["current"]["cohorts"]["dataset_b"],
                        "union_ab_b_priority": baselines["current"]["cohorts"]["union_ab_b_priority"],
                    }
                    if "current" in baselines
                    else None
                ),
            },
            "baselines": baselines,
            "quote2_vs_current_on_merged": quote2_vs_current_merged,
        }
    )

    out_json = out_dir / "comparison_and_merged_summary.json"
    out_md = out_dir / "comparison_and_merged_summary.md"
    out_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote merged JSON summary to {out_json}")

    md_text = format_merged_markdown(summary_payload)
    out_md.write_text(md_text, encoding="utf-8")
    print(f"Wrote merged Markdown summary to {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
