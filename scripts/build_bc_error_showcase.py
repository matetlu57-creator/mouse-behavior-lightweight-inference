#!/usr/bin/env python3
"""Build curated B ∪ C (excluding interpolated A) behavior error & success showcase report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT  # noqa: F401

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_cleaned_dataset_and_merge import _summarize_cohort  # noqa: E402
from generate_dataset_comparison_table import ALL_15_BEHAVIORS  # noqa: E402

DEFAULT_AB_REPORT = REPO_ROOT / "outputs" / "group_upgrade_20261122_eval" / "ab_report.json"
DEFAULT_C_REPORT = REPO_ROOT / "outputs" / "group_upgrade_20261122_eval" / "dataset_c_report.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "bc_behavior_showcase_20261123" / "showcase_report.json"


def _select_curated_clips(bc_clips: list[dict[str, Any]], *, max_per_behavior: int = 10) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for behavior in ALL_15_BEHAVIORS:
        pool = [c for c in bc_clips if c.get("behavior") == behavior]
        if not pool:
            continue

        misclassified: list[dict[str, Any]] = []
        missed: list[dict[str, Any]] = []
        correct: list[dict[str, Any]] = []

        for c in pool:
            p_ok = bool(c.get("primary_correct"))
            t_ok = bool(c.get("pf_target_correct", True))
            fcat = str(c.get("failure_category") or "")
            t_pred = c.get("pf_target_predicted_behavior")
            if (not p_ok and fcat == "misclassified") or (p_ok and not t_ok and t_pred not in (None, behavior)):
                misclassified.append(c)
            elif (not p_ok) or (p_ok and not t_ok):
                missed.append(c)
            else:
                correct.append(c)

        # Prioritize diverse wrong predictions (e.g., Rearing -> Grooming) and both B & C representation.
        misclassified.sort(
            key=lambda c: (
                0 if (not c.get("primary_correct") and c.get("primary_predicted_behavior")) else 1,
                str(c.get("pf_target_predicted_behavior") or c.get("primary_predicted_behavior") or ""),
                0 if c.get("dataset") == "C" else 1,
                str(c.get("clip_id")),
            )
        )
        missed.sort(key=lambda c: (0 if not c.get("primary_correct") else 1, 0 if c.get("dataset") == "C" else 1, str(c.get("clip_id"))))
        correct.sort(key=lambda c: (0 if c.get("dataset") == "C" else 1, -float(c.get("true_target_coverage") or 0.0), str(c.get("clip_id"))))

        picked: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def _take_diverse_mis(items: list[dict[str, Any]], limit: int) -> None:
            used_preds: set[str] = set()
            for item in items:
                if len(picked) >= limit:
                    break
                pred = str(item.get("pf_target_predicted_behavior") or item.get("primary_predicted_behavior") or "")
                if pred and pred not in used_preds:
                    used_preds.add(pred)
                    picked.append(item)
                    seen_keys.add(str(item["record_key"]))
            for item in items:
                if len(picked) >= limit:
                    break
                rk = str(item["record_key"])
                if rk not in seen_keys:
                    picked.append(item)
                    seen_keys.add(rk)

        def _take_balanced(items: list[dict[str, Any]], count: int) -> None:
            added = 0
            for ds in ("C", "B", "C", "B"):
                if added >= count or len(picked) >= max_per_behavior:
                    break
                for item in items:
                    rk = str(item["record_key"])
                    if rk not in seen_keys and item.get("dataset") == ds:
                        picked.append(item)
                        seen_keys.add(rk)
                        added += 1
                        break
            for item in items:
                if added >= count or len(picked) >= max_per_behavior:
                    break
                rk = str(item["record_key"])
                if rk not in seen_keys:
                    picked.append(item)
                    seen_keys.add(rk)
                    added += 1

        _take_diverse_mis(misclassified, min(4, max_per_behavior))
        _take_balanced(missed, 3)
        _take_balanced(correct, max_per_behavior - len(picked))
        if len(picked) < max_per_behavior:
            _take_balanced(misclassified + missed + correct, max_per_behavior - len(picked))

        selected.extend(picked[:max_per_behavior])
    return selected


def build_showcase(ab_path: Path, c_path: Path, output_path: Path, *, max_per_behavior: int = 10) -> dict[str, Any]:
    ab = json.loads(ab_path.read_text(encoding="utf-8"))
    cr = json.loads(c_path.read_text(encoding="utf-8"))
    b_clips = [dict(c) for c in ab["clips"] if c.get("dataset") == "B"]
    c_clips = [dict(c) for c in cr["clips"]]
    bc_clips = b_clips + c_clips

    full_cohort_stats = {
        "union": _summarize_cohort("merged_bc", bc_clips),
        "B": _summarize_cohort("dataset_b", b_clips),
        "C": _summarize_cohort("dataset_c", c_clips),
    }
    curated = _select_curated_clips(bc_clips, max_per_behavior=max_per_behavior)

    provenance = dict(cr.get("provenance") or {})
    payload = {
        "schema_version": "reconciled_behavior_evaluation_v1",
        "generated_at_epoch_s": cr.get("generated_at_epoch_s"),
        "provenance": provenance,
        "full_cohort_stats": full_cohort_stats,
        "clips": curated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_beh = {}
    for b in ALL_15_BEHAVIORS:
        sub = [c for c in curated if c["behavior"] == b]
        by_beh[b] = {
            "total": len(sub),
            "passed": sum(1 for c in sub if c.get("primary_correct")),
            "failed": sum(1 for c in sub if not c.get("primary_correct")),
        }
    return {
        "output": str(output_path),
        "total_curated_clips": len(curated),
        "by_behavior": by_beh,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-report", type=Path, default=DEFAULT_AB_REPORT)
    parser.add_argument("--c-report", type=Path, default=DEFAULT_C_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-behavior", type=int, default=10)
    args = parser.parse_args()
    summary = build_showcase(args.ab_report, args.c_report, args.output, max_per_behavior=args.max_per_behavior)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
