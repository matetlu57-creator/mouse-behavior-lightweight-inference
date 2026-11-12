"""Generate unified combined evaluation JSON covering all 6 behavior categories."""
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.clip_benchmark import (
    BEHAVIOR_ORDER,
    combine_evaluation_reports,
)
from mouse_behavior.visualization.comparison_site import (
    DEFAULT_ATTACK_CHASING_EVALUATION_JSON,
    DEFAULT_COMBINED_EVALUATION_JSON,
    DEFAULT_ISO_GLOCO_EVALUATION_JSON,
    DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-category evaluation JSON files into combined evaluation report.",
    )
    parser.add_argument(
        "--missing-id-json",
        type=Path,
        default=DEFAULT_ATTACK_CHASING_EVALUATION_JSON,
        help="Evaluation JSON for Attack and Chasing",
    )
    parser.add_argument(
        "--iso-gloco-json",
        type=Path,
        default=DEFAULT_ISO_GLOCO_EVALUATION_JSON,
        help="Evaluation JSON for Isolation and Group locomotion",
    )
    parser.add_argument(
        "--static-loco-json",
        type=Path,
        default=DEFAULT_STATIC_LOCOMOTION_EVALUATION_JSON,
        help="Evaluation JSON for Static and Locomotion",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_COMBINED_EVALUATION_JSON,
        help="Combined evaluation JSON output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = combine_evaluation_reports(
        report_paths=[args.missing_id_json, args.iso_gloco_json, args.static_loco_json],
        output_path=args.out,
    )
    print(f"Generated unified evaluation JSON with {summary['total_clips']} clips at {args.out}")
    for beh in BEHAVIOR_ORDER:
        b_key = beh.lower().replace(" ", "_")
        print(
            f"  * {beh}: {summary[f'{b_key}_clips']} clips, "
            f"Acc={summary[f'{b_key}_accuracy'] * 100:.2f}%, "
            f"Mean IoU={summary[f'{b_key}_mean_iou'] * 100:.2f}%"
        )
    print(
        f"Overall Acc: {summary['overall_accuracy'] * 100:.2f}%, "
        f"Overall Mean IoU: {summary['overall_mean_iou'] * 100:.2f}%, "
        f"IoU>=90%: {summary['iou_ge_90_rate'] * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
