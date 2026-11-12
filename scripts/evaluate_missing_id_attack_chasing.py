"""Evaluation runner for completed missing-ID Attack and Chasing clips.

Computes clip-level behavior classification accuracy and boundary cropping
temporal IoU against human-confirmed ground truth annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.behavior.segment_clipping_common import InMemoryTrackStore  # noqa: F401
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.clip_benchmark import (
    compute_temporal_iou,  # noqa: F401
    evaluate_attack_chasing_clip as evaluate_clip,
)
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG
from mouse_behavior.visualization.clip_overlay_renderer import (
    render_cropped_comparison_video as crop_video,  # noqa: F401
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTACK_DIR = Path(r"F:\补全行为素材429_261107\四行为训练素材429\dataset\Attack")
DEFAULT_CHASING_DIR = Path(r"F:\补全行为素材429_261107\四行为训练素材429\dataset\Chasing")


def evaluate_dataset(
    attack_dir: Path = DEFAULT_ATTACK_DIR,
    chasing_dir: Path = DEFAULT_CHASING_DIR,
    limit: int | None = None,
    config_path: Path | None = None,
    verbose: bool = False,
    crop_output_dir: Path | None = None,
    visualize: bool = True,
) -> dict[str, Any]:
    """Evaluate missing-ID Attack and Chasing clips and produce aggregate metrics."""
    attack_dir = Path(attack_dir)
    chasing_dir = Path(chasing_dir)

    cfg_file = config_path if config_path is not None else ROOT / "configs/default.yaml"
    params = config_seed_parameters(load_config(cfg_file))
    social_cfg = DEFAULT_SOCIAL_HEURISTIC_CONFIG

    attack_clips = sorted([d for d in attack_dir.iterdir() if d.is_dir()]) if attack_dir.exists() else []
    chasing_clips = sorted([d for d in chasing_dir.iterdir() if d.is_dir()]) if chasing_dir.exists() else []

    if limit is not None and limit > 0:
        attack_clips = attack_clips[:limit]
        chasing_clips = chasing_clips[:limit]

    results: list[dict[str, Any]] = []
    print(f"Starting evaluation: {len(attack_clips)} Attack clips, {len(chasing_clips)} Chasing clips")
    t0 = time.perf_counter()

    for idx, clip in enumerate(attack_clips, start=1):
        res = evaluate_clip(
            clip,
            params=params,
            social_config=social_cfg,
            crop_output_dir=crop_output_dir,
            visualize=visualize,
        )
        results.append(res)
        if verbose:
            print(
                f"[{idx:3d}/{len(attack_clips)}] Attack: {clip.name[:16]} "
                f"pred={res['predicted_behavior']} correct={res['is_correct']} IoU={res['temporal_iou']:.3f}"
            )

    for idx, clip in enumerate(chasing_clips, start=1):
        res = evaluate_clip(
            clip,
            params=params,
            social_config=social_cfg,
            crop_output_dir=crop_output_dir,
            visualize=visualize,
        )
        results.append(res)
        if verbose:
            print(
                f"[{idx:3d}/{len(chasing_clips)}] Chasing: {clip.name[:16]} "
                f"pred={res['predicted_behavior']} correct={res['is_correct']} IoU={res['temporal_iou']:.3f}"
            )

    elapsed = time.perf_counter() - t0

    attack_results = [r for r in results if r["ground_truth_behavior"] == "Attack"]
    chasing_results = [r for r in results if r["ground_truth_behavior"] == "Chasing"]

    n_attack = len(attack_results)
    n_chasing = len(chasing_results)
    n_total = len(results)

    att_correct = sum(1 for r in attack_results if r["is_correct"])
    ch_correct = sum(1 for r in chasing_results if r["is_correct"])
    total_correct = att_correct + ch_correct

    att_acc = float(att_correct / n_attack) if n_attack > 0 else 0.0
    ch_acc = float(ch_correct / n_chasing) if n_chasing > 0 else 0.0
    overall_acc = float(total_correct / n_total) if n_total > 0 else 0.0

    att_ious = [r["temporal_iou"] for r in attack_results]
    ch_ious = [r["temporal_iou"] for r in chasing_results]
    all_ious = [r["temporal_iou"] for r in results]

    att_mean_iou = float(np.mean(att_ious)) if att_ious else 0.0
    ch_mean_iou = float(np.mean(ch_ious)) if ch_ious else 0.0
    overall_mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    correct_attack_results = [r for r in attack_results if r["is_correct"]]
    correct_chasing_results = [r for r in chasing_results if r["is_correct"]]
    correct_results = [r for r in results if r["is_correct"]]

    correct_att_mean_iou = float(np.mean([r["temporal_iou"] for r in correct_attack_results])) if correct_attack_results else 0.0
    correct_ch_mean_iou = float(np.mean([r["temporal_iou"] for r in correct_chasing_results])) if correct_chasing_results else 0.0
    correct_overall_mean_iou = float(np.mean([r["temporal_iou"] for r in correct_results])) if correct_results else 0.0

    correct_iou_ge_095_count = sum(1 for r in correct_results if r["temporal_iou"] >= 0.95)
    correct_iou_ge_095_rate = float(correct_iou_ge_095_count / len(correct_results)) if correct_results else 0.0

    iou_ge_05_count = sum(1 for r in results if r["iou_ge_05"])
    iou_ge_05_rate = float(iou_ge_05_count / n_total) if n_total > 0 else 0.0

    videos_verified_count = sum(1 for r in results if r.get("video_verified", False))
    video_verification_pass_rate = float(videos_verified_count / n_total) if n_total > 0 else 0.0

    summary = {
        "attack_accuracy": att_acc,
        "chasing_accuracy": ch_acc,
        "overall_accuracy": overall_acc,
        "attack_mean_iou": att_mean_iou,
        "chasing_mean_iou": ch_mean_iou,
        "overall_mean_iou": overall_mean_iou,
        "correct_attack_mean_iou": correct_att_mean_iou,
        "correct_chasing_mean_iou": correct_ch_mean_iou,
        "correct_overall_mean_iou": correct_overall_mean_iou,
        "correct_iou_ge_095_rate": correct_iou_ge_095_rate,
        "correct_iou_ge_095_count": correct_iou_ge_095_count,
        "iou_ge_05_rate": iou_ge_05_rate,
        "video_verification_pass_rate": video_verification_pass_rate,
        "videos_verified_count": videos_verified_count,
        "total_clips": n_total,
        "attack_clips": n_attack,
        "chasing_clips": n_chasing,
        "attack_correct": att_correct,
        "chasing_correct": ch_correct,
        "overall_correct": total_correct,
        "iou_ge_05_count": iou_ge_05_count,
        "elapsed_seconds": elapsed,
        "per_clip_results": results,
    }

    print("\n" + "=" * 70)
    print("MISSING-ID BEHAVIOR EVALUATION & CROPPING SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<32} | {'Attack':<10} | {'Chasing':<10} | {'Overall':<10}")
    print("-" * 70)
    print(f"{'Sample Count':<32} | {n_attack:<10d} | {n_chasing:<10d} | {n_total:<10d}")
    print(f"{'Videos Verified (clip.mp4)':<32} | {sum(1 for r in attack_results if r.get('video_verified')):<10d} | {sum(1 for r in chasing_results if r.get('video_verified')):<10d} | {videos_verified_count:<10d}")
    print(f"{'Correct Predictions':<32} | {att_correct:<10d} | {ch_correct:<10d} | {total_correct:<10d}")
    print(f"{'Classification Accuracy':<32} | {att_acc*100:>9.2f}% | {ch_acc*100:>9.2f}% | {overall_acc*100:>9.2f}%")
    print(f"{'Mean Temporal IoU (all)':<32} | {att_mean_iou:>10.4f} | {ch_mean_iou:>10.4f} | {overall_mean_iou:>10.4f}")
    print(f"{'Mean Temporal IoU (correct)':<32} | {correct_att_mean_iou:>10.4f} | {correct_ch_mean_iou:>10.4f} | {correct_overall_mean_iou:>10.4f}")
    print(f"{'Temporal IoU >= 0.95 (correct)':<32} | {'-':<10} | {'-':<10} | {correct_iou_ge_095_rate*100:>9.2f}%")
    print(f"{'Temporal IoU >= 0.50 (all)':<32} | {'-':<10} | {'-':<10} | {iou_ge_05_rate*100:>9.2f}%")
    print("-" * 70)
    print(f"Elapsed Time: {elapsed:.2f} seconds ({n_total} clips processed)")
    print("=" * 70 + "\n")

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Attack and Chasing recognition and boundary precision on missing-ID dataset."
    )
    parser.add_argument("--attack-dir", type=Path, default=DEFAULT_ATTACK_DIR)
    parser.add_argument("--chasing-dir", type=Path, default=DEFAULT_CHASING_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--crop-output-dir",
        "--export-crops",
        type=Path,
        default=None,
        dest="crop_output_dir",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", default=True)

    args = parser.parse_args(argv)
    summary = evaluate_dataset(
        attack_dir=args.attack_dir,
        chasing_dir=args.chasing_dir,
        limit=args.limit,
        config_path=args.config,
        verbose=args.verbose,
        crop_output_dir=args.crop_output_dir,
        visualize=args.visualize,
    )

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Saved evaluation JSON report to {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
