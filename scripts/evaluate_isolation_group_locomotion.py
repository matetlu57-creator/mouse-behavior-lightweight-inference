"""Evaluation and video cropping runner for Isolation and Group locomotion behaviors."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.behavior.segment_clipping_common import InMemoryTrackStore  # noqa: F401
from mouse_behavior.evaluation.clip_benchmark import (
    compute_temporal_iou,  # noqa: F401
    evaluate_isolation_gloco_clip as evaluate_single_clip,
)
from mouse_behavior.visualization.clip_overlay_renderer import (
    render_annotated_original_video,  # noqa: F401
    render_cropped_comparison_video as render_cropped_video,  # noqa: F401
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path(r"E:\鼠行为数据集_人工清洗完成_605段_20261024\dataset")
DEFAULT_ISOLATION_DIR = DEFAULT_DATASET_DIR / "Isolation"
DEFAULT_GROUP_LOCOMOTION_DIR = DEFAULT_DATASET_DIR / "Group locomotion"
DEFAULT_CROPS_DIR = ROOT / "outputs" / "visualized_cropped_videos"
DEFAULT_ORIG_ANNOTATED_DIR = ROOT / "outputs" / "visualized_original_videos"
DEFAULT_REPORT_JSON = ROOT / "outputs" / "isolation_group_locomotion_evaluation.json"


def evaluate_isolation_and_group_locomotion(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    isolation_dir: Path = DEFAULT_ISOLATION_DIR,
    group_locomotion_dir: Path = DEFAULT_GROUP_LOCOMOTION_DIR,
    limit: int | None = None,
    crop_output_dir: Path | None = DEFAULT_CROPS_DIR,
    orig_annotated_dir: Path | None = DEFAULT_ORIG_ANNOTATED_DIR,
    visualize: bool = True,
    render_videos: bool = True,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate all Isolation and Group locomotion clips and calculate aggregate metrics."""
    iso_clips = sorted([d for d in Path(isolation_dir).iterdir() if d.is_dir()])
    gloco_clips = sorted([d for d in Path(group_locomotion_dir).iterdir() if d.is_dir()])

    if limit is not None and limit > 0:
        iso_clips = iso_clips[:limit]
        gloco_clips = gloco_clips[:limit]

    tasks: list[tuple[Path, str]] = [(c, "Isolation") for c in iso_clips] + [
        (c, "Group locomotion") for c in gloco_clips
    ]

    print(
        f"Total clips to evaluate: {len(tasks)} "
        f"(Isolation: {len(iso_clips)}, Group locomotion: {len(gloco_clips)})",
        flush=True,
    )

    t0 = time.time()
    per_clip_results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_clip,
                clip_dir,
                behavior,
                crop_output_dir=crop_output_dir,
                orig_annotated_dir=orig_annotated_dir,
                visualize=visualize,
                render_videos=render_videos,
            ): (clip_dir.name, behavior)
            for clip_dir, behavior in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            c_name, _ = futures[future]
            try:
                res = future.result()
                per_clip_results.append(res)
                p_s = (res.get("predicted_range_half_open") or {}).get("start", "-")
                p_e = (res.get("predicted_range_half_open") or {}).get("end", "-")
                print(
                    f"  [{res['ground_truth_behavior'][:5]}] {c_name}: match={res['is_correct']}, "
                    f"IoU={res['temporal_iou'] * 100:.1f}%, pred=[{p_s}:{p_e})",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [ERROR] {c_name}: {exc}", flush=True)

    per_clip_results.sort(key=lambda r: (r["ground_truth_behavior"], r["clip_id"]))

    iso_results = [r for r in per_clip_results if r["ground_truth_behavior"] == "Isolation"]
    gloco_results = [r for r in per_clip_results if r["ground_truth_behavior"] == "Group locomotion"]

    iso_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in iso_results])) if iso_results else 0.0
    gloco_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in gloco_results])) if gloco_results else 0.0
    overall_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in per_clip_results])) if per_clip_results else 0.0

    iso_iou = float(np.mean([r["temporal_iou"] for r in iso_results])) if iso_results else 0.0
    gloco_iou = float(np.mean([r["temporal_iou"] for r in gloco_results])) if gloco_results else 0.0
    overall_iou = float(np.mean([r["temporal_iou"] for r in per_clip_results])) if per_clip_results else 0.0

    iou_95_rate = float(np.mean([1.0 if r["temporal_iou"] >= 0.95 else 0.0 for r in per_clip_results])) if per_clip_results else 0.0
    iou_05_rate = float(np.mean([1.0 if r["temporal_iou"] >= 0.50 else 0.0 for r in per_clip_results])) if per_clip_results else 0.0

    return {
        "evaluation_title": "Isolation and Group Locomotion Boundary Cropping Evaluation",
        "dataset_path": str(dataset_dir),
        "total_clips": len(per_clip_results),
        "isolation_clips": len(iso_results),
        "group_locomotion_clips": len(gloco_results),
        "isolation_accuracy": iso_acc,
        "group_locomotion_accuracy": gloco_acc,
        "overall_accuracy": overall_acc,
        "isolation_mean_iou": iso_iou,
        "group_locomotion_mean_iou": gloco_iou,
        "overall_mean_iou": overall_iou,
        "iou_ge_95_rate": iou_95_rate,
        "iou_ge_05_rate": iou_05_rate,
        "elapsed_seconds": round(time.time() - t0, 2),
        "per_clip_results": per_clip_results,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Evaluate Isolation and Group locomotion segment cropping module on 605-segment dataset.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--isolation-dir", type=Path, default=DEFAULT_ISOLATION_DIR)
    parser.add_argument("--group-locomotion-dir", type=Path, default=DEFAULT_GROUP_LOCOMOTION_DIR)
    parser.add_argument("--crop-output-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--orig-annotated-output-dir", type=Path, default=DEFAULT_ORIG_ANNOTATED_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips per behavior")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel worker threads")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("  孤立 (Isolation) 与 群体同步运动 (Group locomotion) 裁剪模块评测", flush=True)
    print("=" * 72, flush=True)

    summary = evaluate_isolation_and_group_locomotion(
        dataset_dir=args.dataset_dir,
        isolation_dir=args.isolation_dir,
        group_locomotion_dir=args.group_locomotion_dir,
        limit=args.limit,
        crop_output_dir=args.crop_output_dir,
        orig_annotated_dir=args.orig_annotated_output_dir,
        render_videos=not args.no_render,
        max_workers=args.workers,
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72, flush=True)
    print("  评测结果汇总 (Evaluation Summary):", flush=True)
    print(f"  ● 总评估片段数: {summary['total_clips']} (孤立: {summary['isolation_clips']}, 群体同步: {summary['group_locomotion_clips']})")
    print(f"  ● 孤立行为分类准确率: {summary['isolation_accuracy'] * 100:.2f}%")
    print(f"  ● 群体同步分类准确率: {summary['group_locomotion_accuracy'] * 100:.2f}%")
    print(f"  ● 总体行为分类准确率: {summary['overall_accuracy'] * 100:.2f}%")
    print(f"  ● 孤立行为平均时序重叠 (Mean IoU): {summary['isolation_mean_iou'] * 100:.2f}%")
    print(f"  ● 群体同步平均时序重叠 (Mean IoU): {summary['group_locomotion_mean_iou'] * 100:.2f}%")
    print(f"  ● 总体平均时序重叠 (Mean IoU): {summary['overall_mean_iou'] * 100:.2f}%")
    print(f"  ● 时序重叠率 >= 95% 比例: {summary['iou_ge_95_rate'] * 100:.2f}%")
    print(f"  ● 评测报告输出路径: {args.report_json}")
    print("=" * 72, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
