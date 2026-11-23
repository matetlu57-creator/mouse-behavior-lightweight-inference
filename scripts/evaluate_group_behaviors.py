"""Evaluation and video cropping runner for the three group behaviors:
Dispersal (散开/群体分散), Group locomotion (群体同步运动), and Social clustering (社会聚集).
"""
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
from mouse_behavior.evaluation.clip_benchmark import (
    evaluate_isolation_gloco_clip as evaluate_single_clip,
)
from mouse_behavior.visualization.comparison_site import (
    DEFAULT_CROPS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_DISPERSAL_DIR,
    DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON,
    DEFAULT_GROUP_LOCOMOTION_DIR,
    DEFAULT_ORIG_ANNOTATED_DIR,
    DEFAULT_SOCIAL_CLUSTERING_DIR,
)

GROUP_BEHAVIOR_SPECS: list[dict[str, Any]] = [
    {
        "behavior": "Dispersal",
        "zh_name": "散开 / 群体分散",
        "rules": [
            "参与门槛：至少 2 只个体向外辐射撤离，原始集群通常为 3~5 只（R ≤ 10.0 cm / 120 px）。",
            "起始边界：定位于仍紧凑、即将扩散的瞬间，支持最长 4.0 s 向前平滑回溯与 2.0 s 间隙对齐。",
            "扩散与终止：向外净位移 ≥ 4.0 cm (25 px)、峰值半径增益 ≥ 35 px；重新靠拢持续 > 0.5 s 前截断。",
            "时限门控：总持续时间严格 > 5.0 s（dispersal_min_duration_s）。",
        ],
    },
    {
        "behavior": "Group locomotion",
        "zh_name": "群体同步运动",
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
        "rules": [
            "参与门槛：至少 3 只小鼠，目标紧凑聚集半径 R ≤ 10.0 cm × (1 + 0.20 × max(0, k − 3))。",
            "起始边界：向前回溯最多 3.0 s 定位发起者定向靠近（速度 ≥ 2.0 cm/s、余弦 ≥ 0.5）的起点。",
            "收敛条件：聚集前 90 分位半径与核心半径落差 ≥ 1.0 cm，发起路径 ≥ 10.0 cm；排除平行同步移动。",
            "时限门控：终态稳定紧凑核心保持时间 ≥ 0.50 s（clustering_final_stability_s）。",
        ],
    },
]


def _count_dataset_total_clips(dataset_dir: Path) -> int | None:
    if not dataset_dir.is_dir():
        return None
    total = 0
    for beh_dir in dataset_dir.iterdir():
        if beh_dir.is_dir():
            total += sum(1 for item in beh_dir.iterdir() if item.is_dir())
    return total if total > 0 else None


def evaluate_group_behaviors(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dispersal_dir: Path = DEFAULT_DISPERSAL_DIR,
    group_locomotion_dir: Path = DEFAULT_GROUP_LOCOMOTION_DIR,
    social_clustering_dir: Path = DEFAULT_SOCIAL_CLUSTERING_DIR,
    limit: int | None = None,
    crop_output_dir: Path | None = DEFAULT_CROPS_DIR,
    orig_annotated_dir: Path | None = DEFAULT_ORIG_ANNOTATED_DIR,
    visualize: bool = True,
    render_videos: bool = True,
    overwrite_videos: bool = False,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate Dispersal, Group locomotion, and Social clustering clips and render videos."""
    beh_dirs: list[tuple[str, Path]] = [
        ("Dispersal", Path(dispersal_dir)),
        ("Group locomotion", Path(group_locomotion_dir)),
        ("Social clustering", Path(social_clustering_dir)),
    ]

    tasks: list[tuple[Path, str]] = []
    counts_by_beh: dict[str, int] = {}
    for beh, bdir in beh_dirs:
        clips = sorted([d for d in bdir.iterdir() if d.is_dir()]) if bdir.is_dir() else []
        if limit is not None and limit > 0:
            clips = clips[:limit]
        counts_by_beh[beh] = len(clips)
        tasks.extend((c, beh) for c in clips)

    print(
        f"Total group-behavior clips to evaluate: {len(tasks)} "
        f"(Dispersal: {counts_by_beh['Dispersal']}, "
        f"Group locomotion: {counts_by_beh['Group locomotion']}, "
        f"Social clustering: {counts_by_beh['Social clustering']})",
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
            c_name, beh = futures[future]
            try:
                res = future.result()
                per_clip_results.append(res)
                p_s = (res.get("predicted_range_half_open") or {}).get("start", "-")
                p_e = (res.get("predicted_range_half_open") or {}).get("end", "-")
                print(
                    f"  [{beh[:8]}] {c_name}: match={res['is_correct']}, "
                    f"IoU={res['temporal_iou'] * 100:.1f}%, pred=[{p_s}:{p_e})",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [ERROR] {c_name}: {exc}", flush=True)

    if render_videos:
        for idx, res in enumerate(per_clip_results):
            needs_crop = (
                crop_output_dir is not None
                and res.get("predicted_range_half_open") is not None
                and not (res.get("cropped_video_path") and Path(str(res["cropped_video_path"])).is_file())
            )
            needs_orig = (
                orig_annotated_dir is not None
                and not (res.get("annotated_orig_path") and Path(str(res["annotated_orig_path"])).is_file())
            )
            if needs_crop or needs_orig:
                print(f"  [RETRY] Re-rendering missing video(s) sequentially for {res['clip_id']}...", flush=True)
                per_clip_results[idx] = evaluate_single_clip(
                    Path(res["clip_dir"]),
                    res.get("ground_truth_behavior"),
                    crop_output_dir=crop_output_dir,
                    orig_annotated_dir=orig_annotated_dir,
                    visualize=visualize,
                    render_videos=True,
                    overwrite_videos=False,
                )

    per_clip_results.sort(key=lambda r: (str(r.get("ground_truth_behavior", "")), str(r["clip_id"])))

    dataset_total_clips = _count_dataset_total_clips(Path(dataset_dir))

    def _stats(beh: str) -> tuple[int, float, float]:
        subset = [r for r in per_clip_results if r.get("ground_truth_behavior") == beh]
        if not subset:
            return 0, 0.0, 0.0
        acc = float(np.mean([1.0 if r.get("is_correct") else 0.0 for r in subset]))
        corr = [r for r in subset if r.get("is_correct")]
        iou = float(np.mean([float(r.get("temporal_iou", 0.0)) for r in corr])) if corr else 0.0
        return len(subset), round(acc, 4), round(iou, 4)

    disp_n, disp_acc, disp_iou = _stats("Dispersal")
    gloco_n, gloco_acc, gloco_iou = _stats("Group locomotion")
    sc_n, sc_acc, sc_iou = _stats("Social clustering")

    overall_acc = (
        float(np.mean([1.0 if r.get("is_correct") else 0.0 for r in per_clip_results]))
        if per_clip_results
        else 0.0
    )
    corr_all = [r for r in per_clip_results if r.get("is_correct")]
    overall_iou = (
        float(np.mean([float(r.get("temporal_iou", 0.0)) for r in corr_all]))
        if corr_all
        else 0.0
    )
    iou_90_rate = (
        float(np.mean([1.0 if float(r.get("temporal_iou", 0.0)) >= 0.90 else 0.0 for r in corr_all]))
        if corr_all
        else 0.0
    )
    iou_95_rate = (
        float(np.mean([1.0 if float(r.get("temporal_iou", 0.0)) >= 0.95 else 0.0 for r in corr_all]))
        if corr_all
        else 0.0
    )
    iou_05_rate = (
        float(np.mean([1.0 if float(r.get("temporal_iou", 0.0)) >= 0.50 else 0.0 for r in corr_all]))
        if corr_all
        else 0.0
    )

    specs_with_counts: list[dict[str, Any]] = []
    for spec in GROUP_BEHAVIOR_SPECS:
        beh = spec["behavior"]
        cnt = counts_by_beh.get(beh, 0)
        share_pct = (
            round((cnt / dataset_total_clips) * 100.0, 2)
            if dataset_total_clips
            else None
        )
        specs_with_counts.append({
            **spec,
            "clip_count": cnt,
            "dataset_total_clips": dataset_total_clips,
            "dataset_share_pct": share_pct,
        })

    return {
        "evaluation_title": "三种群体行为边界裁剪与视频验证报告 (Dispersal / Group locomotion / Social clustering)",
        "dataset_path": str(dataset_dir),
        "dataset_total_clips": dataset_total_clips,
        "total_clips": len(per_clip_results),
        "dispersal_clips": disp_n,
        "group_locomotion_clips": gloco_n,
        "social_clustering_clips": sc_n,
        "dispersal_accuracy": disp_acc,
        "group_locomotion_accuracy": gloco_acc,
        "social_clustering_accuracy": sc_acc,
        "overall_accuracy": round(overall_acc, 4),
        "dispersal_mean_iou": disp_iou,
        "group_locomotion_mean_iou": gloco_iou,
        "social_clustering_mean_iou": sc_iou,
        "overall_mean_iou": round(overall_iou, 4),
        "iou_ge_90_rate": round(iou_90_rate, 4),
        "iou_ge_95_rate": round(iou_95_rate, 4),
        "iou_ge_05_rate": round(iou_05_rate, 4),
        "behavior_specs": specs_with_counts,
        "elapsed_seconds": round(time.time() - t0, 2),
        "per_clip_results": per_clip_results,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Evaluate Dispersal, Group locomotion, and Social clustering and render videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dispersal-dir", type=Path, default=DEFAULT_DISPERSAL_DIR)
    parser.add_argument("--group-locomotion-dir", type=Path, default=DEFAULT_GROUP_LOCOMOTION_DIR)
    parser.add_argument("--social-clustering-dir", type=Path, default=DEFAULT_SOCIAL_CLUSTERING_DIR)
    parser.add_argument("--crop-output-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--orig-annotated-output-dir", type=Path, default=DEFAULT_ORIG_ANNOTATED_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_GROUP_BEHAVIORS_EVALUATION_JSON)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips per behavior")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing videos")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel worker threads")
    args = parser.parse_args()

    print("=" * 76, flush=True)
    print("  三种群体行为 (Dispersal / Group locomotion / Social clustering) 视频渲染与评测", flush=True)
    print("=" * 76, flush=True)

    summary = evaluate_group_behaviors(
        dataset_dir=args.dataset_dir,
        dispersal_dir=args.dispersal_dir,
        group_locomotion_dir=args.group_locomotion_dir,
        social_clustering_dir=args.social_clustering_dir,
        limit=args.limit,
        crop_output_dir=args.crop_output_dir,
        orig_annotated_dir=args.orig_annotated_output_dir,
        render_videos=not args.no_render,
        overwrite_videos=args.overwrite,
        max_workers=args.workers,
    )

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 76, flush=True)
    print("  评测结果汇总 (Strict Top-1 / Temporal IoU Summary):", flush=True)
    print(
        f"  ● 总评估片段: {summary['total_clips']} "
        f"(Dispersal={summary['dispersal_clips']}, "
        f"Group locomotion={summary['group_locomotion_clips']}, "
        f"Social clustering={summary['social_clustering_clips']})",
        flush=True,
    )
    print(f"  ● Dispersal 严格准确率: {summary['dispersal_accuracy'] * 100:.2f}% | Mean IoU: {summary['dispersal_mean_iou'] * 100:.2f}%", flush=True)
    print(f"  ● Group locomotion 严格准确率: {summary['group_locomotion_accuracy'] * 100:.2f}% | Mean IoU: {summary['group_locomotion_mean_iou'] * 100:.2f}%", flush=True)
    print(f"  ● Social clustering 严格准确率: {summary['social_clustering_accuracy'] * 100:.2f}% | Mean IoU: {summary['social_clustering_mean_iou'] * 100:.2f}%", flush=True)
    print(f"  ● 三行为整体严格准确率: {summary['overall_accuracy'] * 100:.2f}% | Mean IoU: {summary['overall_mean_iou'] * 100:.2f}%", flush=True)
    print(f"  ● IoU >= 95% 比例: {summary['iou_ge_95_rate'] * 100:.2f}%", flush=True)
    print(f"  ● 评测报告输出路径: {args.report_json}", flush=True)
    print("=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
