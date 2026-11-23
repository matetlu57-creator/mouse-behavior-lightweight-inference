#!/usr/bin/env python3
"""Batch render annotated H.264 overlay videos for all failed clips.

Burns bounding boxes, 7-keypoint skeletons, orientation vectors, target roles,
and top failure diagnostic banners onto every frame.
Outputs browser-native H.264 (yuv420p + faststart) MP4 files to:
  outputs/failure_report_offline/videos/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import imageio_ffmpeg
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _FONT_CANDIDATES = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    )
    _VALID_FONT_PATH = next((p for p in _FONT_CANDIDATES if p.is_file()), None)
except Exception:
    _VALID_FONT_PATH = None

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ImportError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

LOGGER = logging.getLogger("failure_video_renderer")

DEFAULT_REPORT_JSON = Path("outputs/reconciled_behavior_evaluation.json")
DEFAULT_OUTPUT_DIR = Path("outputs/failure_report_offline/videos")

SKELETON_EDGES = (
    (0, 1),
    (0, 2),  # nose to ears
    (1, 3),
    (2, 3),  # ears to neck
    (3, 4),
    (3, 5),  # neck to hips
    (4, 6),
    (5, 6),  # hips to tail base
)

BEHAVIOR_CHINESE = {
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

ROLE_CHINESE = {
    "Attacker": "攻击个体",
    "Attacked": "受击个体",
    "Chaser": "追逐个体",
    "Chased": "被追个体",
    "Approacher": "接近个体",
    "Approached": "被接近个体",
    "Snout_contacting": "嗅闻个体",
    "Head_contacted": "被嗅闻头部",
    "Rear_contacted": "被嗅闻尾部",
    "Actor": "主动个体",
    "Target": "目标个体",
    "Member": "扎堆成员",
    "Together": "共处个体",
    "Isolated": "孤立个体",
}

# Color palette (BGR for OpenCV)
COLOR_AMBER = (11, 158, 245)  # Primary target: Amber/Gold
COLOR_CYAN = (212, 182, 6)  # Secondary target: Cyan
COLOR_EMERALD = (129, 182, 16)  # Tertiary target: Emerald Green
COLOR_MAGENTA = (200, 50, 255)  # Quaternary target: Magenta
COLOR_GRAY = (148, 140, 128)  # Non-target: Muted Slate Gray
COLOR_SKELETON_HEAD = (0, 230, 255)  # Yellow for head
COLOR_SKELETON_BODY = (255, 100, 0)  # Blue/Cyan for spine


def _get_target_colors(target_ids: Sequence[int]) -> dict[int, tuple[int, int, int]]:
    palette = [COLOR_AMBER, COLOR_CYAN, COLOR_EMERALD, COLOR_MAGENTA]
    colors = {}
    for i, tid in enumerate(target_ids):
        colors[tid] = palette[i % len(palette)]
    return colors


def _normalize_roles_map(raw_roles: Any) -> dict[str, list[int]]:
    norm: dict[str, list[int]] = {}
    if isinstance(raw_roles, dict):
        for k, v in raw_roles.items():
            if isinstance(v, (list, tuple)):
                norm[str(k)] = [int(x) for x in v if str(x).isdigit()]
            elif isinstance(v, (int, str)) and str(v).isdigit():
                norm[str(k)] = [int(v)]
    elif isinstance(raw_roles, (list, tuple)):
        for p in raw_roles:
            if isinstance(p, dict):
                r_name = str(p.get("role_name") or p.get("role") or p.get("name") or "")
                t_ids = p.get("track_ids") or p.get("mouse_ids") or []
                if isinstance(t_ids, (list, tuple)):
                    norm.setdefault(r_name, []).extend(int(x) for x in t_ids if str(x).isdigit())
    return norm


def _index_tracks(tracks: Sequence[Mapping[str, Any]] | None) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = {}
    if not tracks:
        return indexed
    for item in tracks:
        f = int(item.get("frame", item.get("original_clip_frame", 0)))
        dets = item.get("detections") or []
        indexed[f] = list(dets)
    return indexed


def render_failure_overlay_video(
    source_video: Path,
    output_video: Path,
    *,
    tracks_data: Sequence[Mapping[str, Any]] | None,
    clip_id: str,
    dataset: str,
    behavior: str,
    target_mouse_ids: Sequence[int],
    roles_map: Mapping[str, Sequence[int]] | None,
    failure_category: str | None,
    failure_subcategory: str | None,
    failure_reason_zh: str | None,
    ffmpeg_exe: str,
) -> bool:
    """Render a single failure clip with rich bounding boxes, skeletons, and diagnostics."""
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if not source_video.is_file():
        LOGGER.error("Source video not found: %s", source_video)
        return False

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        LOGGER.error("Cannot open video: %s", source_video)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if w <= 0 or h <= 0 or total_frames <= 0:
        cap.release()
        return False

    tracks_by_frame = _index_tracks(tracks_data)
    target_ids_set = set(target_mouse_ids)
    norm_roles = _normalize_roles_map(roles_map)
    target_colors = _get_target_colors(sorted(target_ids_set))

    # Scale factor based on resolution
    scale = max(0.65, min(w, h) / 1080.0)
    thick = max(1, int(round(2 * scale)))
    banner_h = int(round(88 * scale))

    beh_zh = BEHAVIOR_CHINESE.get(behavior, behavior)
    category_zh = "识别行为出错" if failure_category == "misclassified" else "行为未识别到"
    cat_color = (60, 60, 240) if failure_category == "misclassified" else (0, 165, 255)

    font_b1 = None
    font_b2 = None
    font_tag = None
    if _VALID_FONT_PATH is not None:
        try:
            font_b1 = ImageFont.truetype(str(_VALID_FONT_PATH), max(12, int(round(20 * scale))))
            font_b2 = ImageFont.truetype(str(_VALID_FONT_PATH), max(10, int(round(15 * scale))))
            font_tag = ImageFont.truetype(str(_VALID_FONT_PATH), max(10, int(round(14 * scale))))
        except Exception:
            font_b1 = None
            font_b2 = None
            font_tag = None

    raw_tmp = output_video.with_suffix(".tmp_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_tmp), fourcc, fps, (w, h))

    try:
        cur_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Draw top diagnostic banner (semi-transparent dark overlay)
            hud = frame.copy()
            cv2.rectangle(hud, (0, 0), (w, banner_h), (20, 18, 18), -1)
            cv2.addWeighted(hud, 0.85, frame, 0.15, 0, frame)

            # Banner border bottom
            cv2.line(
                frame, (0, banner_h), (w, banner_h), (60, 50, 50), max(1, int(round(1.5 * scale)))
            )

            # Banner line 1: Behavior, Dataset, Targets
            t_ids_str = ", ".join(map(str, sorted(target_ids_set))) if target_ids_set else "None"
            line1_left = (
                f"【{beh_zh} ({behavior})】 数据集: {dataset}  |  目标小鼠 ID: [{t_ids_str}]"
            )
            cat_badge = f"【{category_zh}】"

            # Banner line 2: Frame info, Time, Failure reason snippet
            cur_sec = cur_frame / max(fps, 1e-6)
            tot_sec = total_frames / max(fps, 1e-6)
            reason_snip = failure_reason_zh or failure_subcategory or ""
            if len(reason_snip) > 55:
                reason_snip = reason_snip[:52] + "..."
            line2_text = f"帧: {cur_frame}/{total_frames} ({cur_sec:.2f}s/{tot_sec:.2f}s)  |  原因: {reason_snip}"

            if font_b1 is not None and font_b2 is not None:
                b_roi = frame[0:banner_h, 0:w]
                pil_b = Image.fromarray(cv2.cvtColor(b_roi, cv2.COLOR_BGR2RGB))
                draw_b = ImageDraw.Draw(pil_b)
                # Line 1 left
                draw_b.text(
                    (int(20 * scale), int(10 * scale)),
                    line1_left,
                    font=font_b1,
                    fill=(248, 189, 56),
                )
                # Line 1 right
                cat_bbox = draw_b.textbbox((0, 0), cat_badge, font=font_b1)
                bw = cat_bbox[2] - cat_bbox[0]
                bx = w - bw - int(25 * scale)
                draw_b.text(
                    (bx, int(10 * scale)),
                    cat_badge,
                    font=font_b1,
                    fill=(cat_color[2], cat_color[1], cat_color[0]),
                )
                # Line 2
                draw_b.text(
                    (int(20 * scale), int(48 * scale)),
                    line2_text,
                    font=font_b2,
                    fill=(220, 220, 220),
                )
                frame[0:banner_h, 0:w] = cv2.cvtColor(np.array(pil_b), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(
                    frame,
                    line1_left,
                    (int(20 * scale), int(30 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.70 * scale,
                    (56, 189, 248),
                    thick,
                    cv2.LINE_AA,
                )
                (bw, bh), _ = cv2.getTextSize(
                    cat_badge, cv2.FONT_HERSHEY_DUPLEX, 0.70 * scale, thick
                )
                bx = w - bw - int(25 * scale)
                cv2.putText(
                    frame,
                    cat_badge,
                    (bx, int(30 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.70 * scale,
                    cat_color,
                    thick,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line2_text,
                    (int(20 * scale), int(64 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * scale,
                    (220, 220, 220),
                    max(1, thick - 1),
                    cv2.LINE_AA,
                )

            # 2. Draw detections for this frame
            detections = tracks_by_frame.get(cur_frame, [])
            for det in detections:
                tid = int(det.get("track_id", -1))
                box = det.get("box")
                if not box or len(box) != 4 or not np.isfinite(box).all():
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                is_target = tid in target_ids_set

                # Determine role name
                role_zh = ""
                if norm_roles:
                    for r_name, r_ids in norm_roles.items():
                        if tid in r_ids:
                            role_zh = ROLE_CHINESE.get(r_name, r_name)
                            break
                if not role_zh and is_target:
                    role_zh = "目标小鼠"

                box_color = target_colors.get(tid, COLOR_GRAY)
                box_thickness = (
                    max(2, int(round(3 * scale))) if is_target else max(1, int(round(1 * scale)))
                )

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                # Draw Skeletons & Keypoints if available
                kps = det.get("keypoints")
                if kps and len(kps) == 7:
                    valid_kps: dict[int, tuple[int, int]] = {}
                    for k_idx, kp in enumerate(kps):
                        if (
                            kp is not None
                            and len(kp) >= 3
                            and kp[0] is not None
                            and kp[1] is not None
                            and kp[2] is not None
                        ):
                            try:
                                conf = float(kp[2])
                                if conf > 0.25:
                                    kx, ky = int(round(float(kp[0]))), int(round(float(kp[1])))
                                    if 0 <= kx < w and 0 <= ky < h:
                                        valid_kps[k_idx] = (kx, ky)
                            except (TypeError, ValueError):
                                pass

                    # Draw skeleton edges
                    edge_th = (
                        max(1, int(round(2 * scale)))
                        if is_target
                        else max(1, int(round(1 * scale)))
                    )
                    for e1, e2 in SKELETON_EDGES:
                        if e1 in valid_kps and e2 in valid_kps:
                            ecolor = (
                                COLOR_SKELETON_HEAD if (e1 < 3 and e2 < 4) else COLOR_SKELETON_BODY
                            )
                            if not is_target:
                                ecolor = (100, 100, 100)
                            cv2.line(
                                frame, valid_kps[e1], valid_kps[e2], ecolor, edge_th, cv2.LINE_AA
                            )

                    # Draw keypoint dots
                    for k_idx, (kx, ky) in valid_kps.items():
                        dot_r = (
                            max(2, int(round(3 * scale)))
                            if is_target
                            else max(1, int(round(2 * scale)))
                        )
                        dot_color = (0, 255, 255) if is_target else (140, 140, 140)
                        if k_idx == 0:  # Nose
                            dot_color = (0, 0, 255)
                        cv2.circle(frame, (kx, ky), dot_r, dot_color, -1, cv2.LINE_AA)

                    # Orientation arrow: neck (3) -> nose (0) for target mouse
                    if is_target and 3 in valid_kps and 0 in valid_kps:
                        p_neck = valid_kps[3]
                        p_nose = valid_kps[0]
                        dx = p_nose[0] - p_neck[0]
                        dy = p_nose[1] - p_neck[1]
                        dist = math.hypot(dx, dy)
                        if dist > 4:
                            p_arrow = (int(p_nose[0] + dx * 0.35), int(p_nose[1] + dy * 0.35))
                            cv2.arrowedLine(
                                frame,
                                p_neck,
                                p_arrow,
                                (0, 255, 255),
                                max(1, int(round(2 * scale))),
                                cv2.LINE_AA,
                                tipLength=0.28,
                            )

                # Draw ID Tag
                if is_target:
                    tag_text = f"ID: {tid}" + (f" [{role_zh}]" if role_zh else "")
                else:
                    tag_text = f"ID: {tid}"

                if font_tag is not None:
                    dummy = Image.new("RGB", (1, 1))
                    d_d = ImageDraw.Draw(dummy)
                    t_bb = d_d.textbbox((0, 0), tag_text, font=font_tag)
                    tw = t_bb[2] - t_bb[0]
                    th_box = t_bb[3] - t_bb[1]
                    tag_y = max(y1 - 6, banner_h + th_box + 8)

                    cv2.rectangle(
                        frame,
                        (x1, tag_y - th_box - 4),
                        (x1 + tw + 8, tag_y + 4),
                        (20, 20, 20),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, tag_y - th_box - 4),
                        (x1 + tw + 8, tag_y + 4),
                        box_color,
                        max(1, int(round(1 * scale))),
                    )
                    ry1 = max(0, tag_y - th_box - 4)
                    ry2 = min(h, tag_y + 6)
                    rx1 = max(0, x1)
                    rx2 = min(w, x1 + tw + 10)
                    if rx2 > rx1 and ry2 > ry1:
                        roi = frame[ry1:ry2, rx1:rx2]
                        p_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        d_roi = ImageDraw.Draw(p_roi)
                        d_roi.text(
                            (x1 + 4 - rx1, tag_y - th_box - 2 - ry1),
                            tag_text,
                            font=font_tag,
                            fill=(255, 255, 255),
                        )
                        frame[ry1:ry2, rx1:rx2] = cv2.cvtColor(np.array(p_roi), cv2.COLOR_RGB2BGR)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.52 * scale
                    font_th = max(1, int(round(1.5 * scale)))
                    (tw, th_box), _ = cv2.getTextSize(tag_text, font, font_scale, font_th)

                    tag_y = max(y1 - 6, banner_h + th_box + 4)
                    # Background pill for tag
                    cv2.rectangle(
                        frame,
                        (x1, tag_y - th_box - 4),
                        (x1 + tw + 6, tag_y + 3),
                        (20, 20, 20),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, tag_y - th_box - 4),
                        (x1 + tw + 6, tag_y + 3),
                        box_color,
                        max(1, int(round(1 * scale))),
                    )
                    cv2.putText(
                        frame,
                        tag_text,
                        (x1 + 3, tag_y - 2),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_th,
                        cv2.LINE_AA,
                    )

            writer.write(frame)
            cur_frame += 1

    finally:
        writer.release()
        cap.release()

    # 3. Transcode to H.264
    tmp_h264 = output_video.with_suffix(".tmp_h264.mp4")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(raw_tmp),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_h264),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw_tmp.unlink(missing_ok=True)

    if proc.returncode == 0 and tmp_h264.is_file() and tmp_h264.stat().st_size > 0:
        tmp_h264.replace(output_video)
        return True

    LOGGER.error(
        "FFmpeg transcode failed for %s: %s", output_video, proc.stderr.decode(errors="replace")
    )
    tmp_h264.unlink(missing_ok=True)
    return False


def batch_render_failed_clips(
    report_path: Path = DEFAULT_REPORT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Find all failed clips from the report and batch render annotated H.264 clips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = [c for c in report["clips"] if not c.get("primary_correct")]

    LOGGER.info("Total failure records in report: %d", len(failures))

    # Deduplicate by (dataset, clip_id)
    tasks: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()

    for item in failures:
        cid = item["clip_id"]
        ds = item["dataset"]
        cdir = Path(item["clip_dir"])
        source_video = cdir / "clip.mp4"
        tracks_path = cdir / "tracks.json"
        annot_path = cdir / "annotation.json"

        task_key = f"{ds}_{cid}"
        if task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)

        # Dataset-prefixed output video: <dataset>_<clip_id>.mp4
        ds_file = output_dir / f"{ds}_{cid}.mp4"
        # Generic clip video: <clip_id>.mp4
        target_file = output_dir / f"{cid}.mp4"

        # Load tracks and annotation
        tracks = []
        if tracks_path.is_file():
            try:
                tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
            except Exception:
                tracks = []

        roles_map = item.get("participants") or {}
        if annot_path.is_file():
            try:
                adata = json.loads(annot_path.read_text(encoding="utf-8"))
                roles_map = adata.get("roles") or adata.get("participants") or roles_map
            except Exception:
                pass

        tasks.append(
            {
                "source_video": source_video,
                "output_video": ds_file,
                "generic_video": target_file,
                "tracks_data": tracks,
                "clip_id": cid,
                "dataset": ds,
                "behavior": item["behavior"],
                "target_mouse_ids": item["target_mouse_ids"],
                "roles_map": roles_map,
                "failure_category": item.get("failure_category"),
                "failure_subcategory": item.get("failure_subcategory"),
                "failure_reason_zh": item.get("failure_reason_zh"),
                "is_union_b": (ds == "B") or item.get("in_union_b_priority", False),
            }
        )

    LOGGER.info("Unique failed clips to render: %d", len(tasks))

    def _render_task(t: dict[str, Any]) -> tuple[str, bool]:
        ds_out = t["output_video"]
        gen_out = t["generic_video"]

        # If dataset output file already exists and valid, skip re-render
        rendered = False
        if ds_out.is_file() and ds_out.stat().st_size > 10240:
            rendered = True
        else:
            rendered = render_failure_overlay_video(
                source_video=t["source_video"],
                output_video=ds_out,
                tracks_data=t["tracks_data"],
                clip_id=t["clip_id"],
                dataset=t["dataset"],
                behavior=t["behavior"],
                target_mouse_ids=t["target_mouse_ids"],
                roles_map=t["roles_map"],
                failure_category=t["failure_category"],
                failure_subcategory=t["failure_subcategory"],
                failure_reason_zh=t["failure_reason_zh"],
                ffmpeg_exe=ffmpeg_exe,
            )

        if rendered:
            if not gen_out.is_file() or t.get("is_union_b"):
                import shutil

                try:
                    shutil.copyfile(ds_out, gen_out)
                except Exception:
                    pass
        return f"{t['dataset']}_{t['clip_id']}", rendered

    t0 = time.time()
    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cid, ok in executor.map(_render_task, tasks):
            results[cid] = ok

    t1 = time.time()
    succeeded = sum(1 for ok in results.values() if ok)
    LOGGER.info("Rendered %d/%d failure clips in %.2fs", succeeded, len(tasks), t1 - t0)

    return {
        "total_failures": len(tasks),
        "succeeded": succeeded,
        "elapsed_seconds": round(t1 - t0, 2),
        "output_dir": str(output_dir),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT_JSON, help="Evaluation JSON path"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output videos directory"
    )
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker threads")
    args = parser.parse_args()

    res = batch_render_failed_clips(
        report_path=args.report, output_dir=args.output_dir, max_workers=args.workers
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["succeeded"] == res["total_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
