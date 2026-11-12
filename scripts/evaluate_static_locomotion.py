r"""Evaluation and video cropping runner for Static (静止) and Locomotion (移动) behaviors."""
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
    evaluate_static_loco_clip as evaluate_single_clip,
)
from mouse_behavior.visualization.clip_overlay_renderer import (
    render_annotated_original_video,  # noqa: F401
    render_cropped_comparison_video as render_cropped_video,  # noqa: F401
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path(r"E:\鼠行为数据集_人工清洗完成_605段_20261024\dataset")
DEFAULT_STATIC_DIR = DEFAULT_DATASET_DIR / "Static"
DEFAULT_LOCOMOTION_DIR = DEFAULT_DATASET_DIR / "Locomotion"
DEFAULT_CROPS_DIR = ROOT / "outputs" / "visualized_cropped_videos"
DEFAULT_ORIG_ANNOTATED_DIR = ROOT / "outputs" / "visualized_original_videos"
DEFAULT_REPORT_JSON = ROOT / "outputs" / "static_locomotion_evaluation.json"


def evaluate_static_and_locomotion(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    static_dir: Path = DEFAULT_STATIC_DIR,
    locomotion_dir: Path = DEFAULT_LOCOMOTION_DIR,
    limit: int | None = None,
    crop_output_dir: Path | None = DEFAULT_CROPS_DIR,
    orig_annotated_dir: Path | None = DEFAULT_ORIG_ANNOTATED_DIR,
    visualize: bool = True,
    render_videos: bool = True,
    overwrite_videos: bool = False,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate all Static and Locomotion clips and calculate aggregate metrics."""
    static_clips = sorted([d for d in Path(static_dir).iterdir() if d.is_dir()])
    loco_clips = sorted([d for d in Path(locomotion_dir).iterdir() if d.is_dir()])

    if limit is not None and limit > 0:
        static_clips = static_clips[:limit]
        loco_clips = loco_clips[:limit]

    tasks: list[tuple[Path, str]] = [(c, "Static") for c in static_clips] + [
        (c, "Locomotion") for c in loco_clips
    ]

    print(
        f"Total clips to evaluate: {len(tasks)} "
        f"(Static: {len(static_clips)}, Locomotion: {len(loco_clips)})",
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
                overwrite_videos=overwrite_videos,
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

    static_results = [r for r in per_clip_results if r["ground_truth_behavior"] == "Static"]
    loco_results = [r for r in per_clip_results if r["ground_truth_behavior"] == "Locomotion"]

    static_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in static_results])) if static_results else 0.0
    loco_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in loco_results])) if loco_results else 0.0
    overall_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in per_clip_results])) if per_clip_results else 0.0

    static_iou = float(np.mean([r["temporal_iou"] for r in static_results])) if static_results else 0.0
    loco_iou = float(np.mean([r["temporal_iou"] for r in loco_results])) if loco_results else 0.0
    overall_iou = float(np.mean([r["temporal_iou"] for r in per_clip_results])) if per_clip_results else 0.0

    iou_90_rate = float(np.mean([1.0 if r["temporal_iou"] >= 0.90 else 0.0 for r in per_clip_results])) if per_clip_results else 0.0
    iou_95_rate = float(np.mean([1.0 if r["temporal_iou"] >= 0.95 else 0.0 for r in per_clip_results])) if per_clip_results else 0.0
    iou_05_rate = float(np.mean([1.0 if r["temporal_iou"] >= 0.50 else 0.0 for r in per_clip_results])) if per_clip_results else 0.0

    return {
        "evaluation_title": "Static and Locomotion Boundary Cropping Evaluation",
        "dataset_path": str(dataset_dir),
        "total_clips": len(per_clip_results),
        "static_clips": len(static_results),
        "locomotion_clips": len(loco_results),
        "static_accuracy": static_acc,
        "locomotion_accuracy": loco_acc,
        "overall_accuracy": overall_acc,
        "static_mean_iou": static_iou,
        "locomotion_mean_iou": loco_iou,
        "overall_mean_iou": overall_iou,
        "iou_ge_90_rate": iou_90_rate,
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
        description="Evaluate Static and Locomotion segment cropping module on 605-segment dataset.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--locomotion-dir", type=Path, default=DEFAULT_LOCOMOTION_DIR)
    parser.add_argument("--crop-output-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--orig-annotated-output-dir", type=Path, default=DEFAULT_ORIG_ANNOTATED_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips per behavior")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing cropped videos")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel worker threads")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("  静止 (Static) 与 移动 (Locomotion) 行为裁剪模块评测", flush=True)
    print("=" * 72, flush=True)

    summary = evaluate_static_and_locomotion(
        dataset_dir=args.dataset_dir,
        static_dir=args.static_dir,
        locomotion_dir=args.locomotion_dir,
        limit=args.limit,
        crop_output_dir=args.crop_output_dir,
        orig_annotated_dir=args.orig_annotated_output_dir,
        render_videos=not args.no_render,
        overwrite_videos=args.overwrite,
        max_workers=args.workers,
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72, flush=True)
    print("  评测结果汇总 (Evaluation Summary):", flush=True)
    print(f"  ● 总评估片段数: {summary['total_clips']} (静止: {summary['static_clips']}, 移动: {summary['locomotion_clips']})")
    print(f"  ● 静止行为分类准确率: {summary['static_accuracy'] * 100:.2f}%")
    print(f"  ● 移动行为分类准确率: {summary['locomotion_accuracy'] * 100:.2f}%")
    print(f"  ● 总体行为分类准确率: {summary['overall_accuracy'] * 100:.2f}%")
    print(f"  ● 静止行为平均时序重叠 (Mean IoU): {summary['static_mean_iou'] * 100:.2f}%")
    print(f"  ● 移动行为平均时序重叠 (Mean IoU): {summary['locomotion_mean_iou'] * 100:.2f}%")
    print(f"  ● 总体平均时序重叠 (Mean IoU): {summary['overall_mean_iou'] * 100:.2f}%")
    print(f"  ● 时序重叠率 >= 90% 比例: {summary['iou_ge_90_rate'] * 100:.2f}%")
    print(f"  ● 时序重叠率 >= 95% 比例: {summary['iou_ge_95_rate'] * 100:.2f}%")
    print(f"  ● 评测报告输出路径: {args.report_json}")
    print("=" * 72, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
