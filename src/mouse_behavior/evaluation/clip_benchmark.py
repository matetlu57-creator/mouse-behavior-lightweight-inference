"""Modular benchmark evaluation and report merging for the six calibrated behaviors.

Covers:
- Static (`stationary`)
- Locomotion (`locomotion`)
- Isolation (`isolation`)
- Group locomotion (`group_locomotion`)
- Chasing (`chase` / `chasing`)
- Attack (`attack`)
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mouse_behavior.behavior import segment_clipping
from mouse_behavior.behavior.segment_clipping_common import InMemoryTrackStore
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)
from mouse_behavior.visualization.clip_overlay_renderer import (
    render_annotated_original_video,
    render_cropped_comparison_video,
    video_frame_count,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]

BEHAVIOR_ORDER: tuple[str, ...] = (
    "Attack",
    "Chasing",
    "Isolation",
    "Group locomotion",
    "Static",
    "Locomotion",
)

__all__ = [
    "BEHAVIOR_ORDER",
    "combine_evaluation_reports",
    "compute_temporal_iou",
    "evaluate_attack_chasing_clip",
    "evaluate_isolation_gloco_clip",
    "evaluate_six_behavior_clip",
    "evaluate_static_loco_clip",
]


def compute_temporal_iou(
    pred_interval: tuple[int, int] | None,
    gt_interval: tuple[int, int],
) -> float:
    """Compute temporal Intersection-over-Union (IoU) between two half-open intervals."""
    if pred_interval is None:
        return 0.0
    s_pred, e_pred = int(pred_interval[0]), int(pred_interval[1])
    s_gt, e_gt = int(gt_interval[0]), int(gt_interval[1])

    if e_pred <= s_pred or e_gt <= s_gt:
        return 0.0

    inter_start = max(s_pred, s_gt)
    inter_end = min(e_pred, e_gt)
    inter = max(0, inter_end - inter_start)

    union_start = min(s_pred, s_gt)
    union_end = max(e_pred, e_gt)
    union = max(0, union_end - union_start)

    if union <= 0:
        return 0.0
    return float(inter / union)


def _default_parameters() -> HeuristicParameters:
    try:
        return config_seed_parameters(load_config(REPO_ROOT / "configs/default.yaml"))
    except Exception:
        return HeuristicParameters()


def _load_clip_files(clip_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    anno_path = clip_dir / "annotation.json"
    meta_path = clip_dir / "metadata.json"
    tracks_path = clip_dir / "tracks.json"
    video_path = clip_dir / "clip.mp4"

    if not anno_path.is_file():
        raise FileNotFoundError(f"Missing annotation.json in {clip_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json in {clip_dir}")
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Missing tracks.json in {clip_dir}")

    annotation = json.loads(anno_path.read_text(encoding="utf-8"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    return annotation, metadata, raw_tracks, video_path


def _parse_clip_timing(
    annotation: dict[str, Any],
    metadata: dict[str, Any],
    raw_tracks: list[dict[str, Any]],
) -> tuple[int, int, int, float, int]:
    frame_range = annotation.get("frame_range", {})
    gt_start = int(frame_range.get("start", 0))
    gt_end_inclusive = int(frame_range.get("end", 0))
    gt_end_half_open = gt_end_inclusive + 1

    fps = float(metadata.get("clip", {}).get("fps", 30.0))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(metadata.get("clip", {}).get("frame_count", len(raw_tracks)))
    if gt_end_half_open <= gt_start:
        gt_end_half_open = total_frames
    return gt_start, gt_end_inclusive, gt_end_half_open, fps, total_frames


def evaluate_attack_chasing_clip(
    clip_dir: Path,
    params: HeuristicParameters | None = None,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    crop_output_dir: Path | None = None,
    visualize: bool = True,
) -> dict[str, Any]:
    """Evaluate a single Attack or Chasing clip using pair heuristic rules + segment clipping."""
    clip_dir = Path(clip_dir)
    clip_id = clip_dir.name
    if params is None:
        params = _default_parameters()

    record: dict[str, Any] = {
        "clip_id": clip_id,
        "clip_dir": str(clip_dir),
        "status": "ok",
        "error": None,
        "ground_truth_behavior": None,
        "predicted_behavior": None,
        "is_correct": False,
        "target_mouse_ids": [],
        "semantic_roles": {},
        "ground_truth_range_inclusive": None,
        "ground_truth_range_half_open": None,
        "predicted_range_half_open": None,
        "temporal_iou": 0.0,
        "iou_ge_05": False,
        "attack_frames": 0,
        "chasing_frames": 0,
        "total_frames": 0,
        "fps": 30.0,
        "video_path": None,
        "video_verified": False,
        "video_frame_count": 0,
        "video_fps": 0.0,
        "video_width": 0,
        "video_height": 0,
        "cropped_video_path": None,
    }

    try:
        annotation, metadata, raw_tracks, video_path = _load_clip_files(clip_dir)
        tracks_path = clip_dir / "tracks.json"

        video_verified = False
        vid_fc = 0
        vid_fps = 0.0
        vid_w, vid_h = 0, 0
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            try:
                if cap.isOpened():
                    vid_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    vid_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_verified = vid_fc > 0 and vid_w > 0 and vid_h > 0
            finally:
                cap.release()

        gt_behavior = str(annotation.get("behavior", ""))
        target_ids = sorted(int(m) for m in annotation.get("mouse_ids", []))
        frame_range = annotation.get("frame_range", {})
        gt_start = int(frame_range.get("start", 0))
        gt_end_inclusive = int(frame_range.get("end", 0))
        gt_end_half_open = gt_end_inclusive + 1

        fps = float(metadata.get("clip", {}).get("fps", 30.0))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(metadata.get("clip", {}).get("frame_count", len(raw_tracks)))

        semantic_roles: dict[str, list[int]] = {}
        for p in annotation.get("participants", []):
            role_name = p.get("role_name", "")
            if role_name:
                semantic_roles[role_name] = [int(t) for t in p.get("track_ids", [])]

        record.update(
            ground_truth_behavior=gt_behavior,
            target_mouse_ids=target_ids,
            semantic_roles=semantic_roles,
            ground_truth_range_inclusive={"start": gt_start, "end": gt_end_inclusive},
            ground_truth_range_half_open={"start": gt_start, "end": gt_end_half_open},
            total_frames=total_frames,
            fps=fps,
            video_path=str(video_path) if video_path.exists() else None,
            video_verified=video_verified,
            video_frame_count=vid_fc,
            video_fps=vid_fps,
            video_width=vid_w,
            video_height=vid_h,
        )

        features = load_exported_features(tracks_path, fps=fps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            masks = pair_rule_masks(features, params, social_config)

        target_edge = None
        for edge_idx, (i, j) in enumerate(zip(features.pair_i, features.pair_j)):
            pair_ids = sorted([int(features.track_ids[i]), int(features.track_ids[j])])
            if pair_ids == target_ids:
                target_edge = edge_idx
                break

        if target_edge is None:
            record.update(
                status="target_not_in_tracks",
                error=f"Annotated pair {target_ids} not found in tracks edge combinations",
            )
            return record

        attack_mask = np.asarray(masks.get("attack", np.zeros((total_frames, 1), bool)))[:, target_edge]
        chasing_mask = np.asarray(masks.get("chasing", np.zeros((total_frames, 1), bool)))[:, target_edge]

        attack_frames = int(np.count_nonzero(attack_mask))
        chasing_frames = int(np.count_nonzero(chasing_mask))
        record.update(attack_frames=attack_frames, chasing_frames=chasing_frames)

        pred_behavior = None
        active_mask = None
        if attack_frames > 0 or chasing_frames > 0:
            if attack_frames > chasing_frames:
                pred_behavior = "Attack"
                active_mask = attack_mask
            elif chasing_frames > attack_frames:
                pred_behavior = "Chasing"
                active_mask = chasing_mask
            else:
                pred_behavior = "Attack"
                active_mask = attack_mask

        record["predicted_behavior"] = pred_behavior
        is_correct = bool(pred_behavior and pred_behavior.lower() == gt_behavior.lower())
        record["is_correct"] = is_correct

        pred_range_half_open = None
        pred_start = 0
        pred_end = 0
        if pred_behavior and active_mask is not None and np.any(active_mask):
            obs_indices = np.flatnonzero(active_mask)
            cand_start = int(obs_indices[0])
            cand_end = int(obs_indices[-1]) + 1

            event = {
                "canonical_behavior": pred_behavior.lower(),
                "start_frame": cand_start,
                "end_frame": cand_end,
                "mouse_ids": target_ids,
            }
            store = InMemoryTrackStore(raw_tracks)
            binding = metadata.get("clip", {"fps": fps})

            refined_events: list[dict[str, Any]] = []
            if pred_behavior == "Attack":
                refined_events = segment_clipping.refine_attack_segments([event], store, binding)
            elif pred_behavior == "Chasing":
                refined_events = segment_clipping.refine_chasing_segments([event], store, binding)

            if refined_events:
                pred_start = int(refined_events[0]["start_frame"])
                pred_end = int(refined_events[0]["end_frame"])
            else:
                pred_start, pred_end = cand_start, cand_end

            pred_range_half_open = {"start": pred_start, "end": pred_end}
            record["predicted_range_half_open"] = pred_range_half_open

        if is_correct and pred_range_half_open is not None:
            iou = compute_temporal_iou(
                (pred_range_half_open["start"], pred_range_half_open["end"]),
                (gt_start, gt_end_half_open),
            )
        else:
            iou = 0.0

        record["temporal_iou"] = float(iou)
        record["iou_ge_05"] = bool(iou >= 0.5)

        if crop_output_dir is not None and video_path.exists() and pred_range_half_open is not None:
            out_clip_video = Path(crop_output_dir) / clip_id / "clip_cropped.mp4"
            ok = render_cropped_comparison_video(
                video_path,
                out_clip_video,
                pred_start,
                pred_end,
                fps,
                tracks=raw_tracks,
                target_mouse_ids=target_ids,
                semantic_roles=semantic_roles,
                pred_behavior=pred_behavior,
                gt_behavior=gt_behavior,
                iou=iou,
                visualize=visualize,
            )
            record["cropped_video_path"] = str(out_clip_video) if ok else None

    except Exception as exc:
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            is_correct=False,
            temporal_iou=0.0,
            iou_ge_05=False,
        )

    return record


def evaluate_isolation_gloco_clip(
    clip_dir: Path,
    expected_behavior: str | None = None,
    params: HeuristicParameters | None = None,
    crop_output_dir: Path | None = None,
    orig_annotated_dir: Path | None = None,
    visualize: bool = True,
    render_videos: bool = True,
) -> dict[str, Any]:
    """Evaluate a single Isolation or Group locomotion clip and perform segment cropping."""
    clip_dir = Path(clip_dir)
    clip_id = clip_dir.name
    if params is None:
        params = _default_parameters()

    record: dict[str, Any] = {
        "clip_id": clip_id,
        "clip_dir": str(clip_dir),
        "status": "ok",
        "error": None,
        "ground_truth_behavior": expected_behavior,
        "predicted_behavior": None,
        "is_correct": False,
        "target_mouse_ids": [],
        "ground_truth_range_half_open": None,
        "predicted_range_half_open": None,
        "temporal_iou": 0.0,
        "iou_ge_95": False,
        "iou_ge_05": False,
        "total_frames": 0,
        "fps": 30.0,
        "cropped_video_path": None,
        "annotated_orig_path": None,
    }

    try:
        annotation, metadata, raw_tracks, video_path = _load_clip_files(clip_dir)
        gt_behavior = str(annotation.get("behavior", expected_behavior or "Unknown"))
        target_ids = sorted(int(m) for m in annotation.get("mouse_ids", []))
        gt_start, _, gt_end_half_open, fps, total_frames = _parse_clip_timing(
            annotation, metadata, raw_tracks
        )

        record.update(
            target_mouse_ids=target_ids,
            ground_truth_behavior=gt_behavior,
            ground_truth_range_half_open={"start": gt_start, "end": gt_end_half_open},
            total_frames=total_frames,
            fps=fps,
        )

        store = InMemoryTrackStore(raw_tracks)
        binding = metadata.get("clip", {"fps": fps})

        pred_behavior = None
        refined_events: list[dict[str, Any]] = []

        if len(target_ids) >= 3:
            cand_event = {
                "canonical_behavior": "group_locomotion",
                "behavior": "Group locomotion",
                "start_frame": 0,
                "end_frame": total_frames,
                "mouse_ids": target_ids,
            }
            res_gloco = segment_clipping.refine_group_locomotion_segments([cand_event], store, binding)
            if res_gloco:
                pred_behavior = "Group locomotion"
                refined_events = res_gloco

        if not pred_behavior and len(target_ids) == 1:
            cand_event = {
                "canonical_behavior": "isolation",
                "behavior": "Isolation",
                "start_frame": 0,
                "end_frame": total_frames,
                "mouse_ids": target_ids,
                "target_mouse_id": target_ids[0],
            }
            res_iso = segment_clipping.refine_isolation_segments([cand_event], store, binding)
            if res_iso:
                pred_behavior = "Isolation"
                refined_events = res_iso

        if not pred_behavior:
            cand_events = []
            if len(target_ids) >= 3:
                cand_events.append({
                    "canonical_behavior": "group_locomotion",
                    "behavior": "Group locomotion",
                    "start_frame": 0,
                    "end_frame": total_frames,
                    "mouse_ids": target_ids,
                })
            elif len(target_ids) == 1:
                cand_events.append({
                    "canonical_behavior": "isolation",
                    "behavior": "Isolation",
                    "start_frame": 0,
                    "end_frame": total_frames,
                    "mouse_ids": target_ids,
                    "target_mouse_id": target_ids[0],
                })
            res_all, _ = segment_clipping.refine_behavior_segments(
                cand_events, store, binding, enable_fine_cropping=True
            )
            if res_all:
                refined_events = [res_all[0]]
                can_b = res_all[0].get("canonical_behavior", "")
                if can_b == "isolation":
                    pred_behavior = "Isolation"
                elif can_b == "group_locomotion":
                    pred_behavior = "Group locomotion"

        record["predicted_behavior"] = pred_behavior
        is_correct = bool(pred_behavior and gt_behavior and pred_behavior.lower() == gt_behavior.lower())
        record["is_correct"] = is_correct

        if is_correct and refined_events:
            p_s = int(refined_events[0]["start_frame"])
            p_e = int(refined_events[0]["end_frame"])
            record["predicted_range_half_open"] = {"start": p_s, "end": p_e}
            iou = compute_temporal_iou((p_s, p_e), (gt_start, gt_end_half_open))
        else:
            iou = 0.0

        record["temporal_iou"] = float(iou)
        record["iou_ge_95"] = bool(iou >= 0.95)
        record["iou_ge_05"] = bool(iou >= 0.50)

        if crop_output_dir is not None and record["predicted_range_half_open"]:
            out_cropped = Path(crop_output_dir) / clip_id / "clip_cropped.mp4"
            if out_cropped.is_file() and out_cropped.stat().st_size > 0:
                record["cropped_video_path"] = str(out_cropped)
            elif render_videos and video_path.is_file():
                ok = render_cropped_comparison_video(
                    video_path,
                    out_cropped,
                    record["predicted_range_half_open"]["start"],
                    record["predicted_range_half_open"]["end"],
                    fps,
                    tracks=raw_tracks,
                    target_mouse_ids=target_ids,
                    pred_behavior=pred_behavior,
                    gt_behavior=gt_behavior,
                    iou=iou,
                    visualize=visualize,
                )
                record["cropped_video_path"] = str(out_cropped) if ok else None

        if orig_annotated_dir is not None:
            out_annotated = Path(orig_annotated_dir) / clip_id / "clip_annotated.mp4"
            if out_annotated.is_file() and out_annotated.stat().st_size > 0:
                record["annotated_orig_path"] = str(out_annotated)
            elif render_videos and video_path.is_file():
                ok_orig = render_annotated_original_video(
                    video_path,
                    out_annotated,
                    fps,
                    tracks=raw_tracks,
                    target_mouse_ids=target_ids,
                    behavior=gt_behavior,
                )
                record["annotated_orig_path"] = str(out_annotated) if ok_orig else None

    except Exception as exc:
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            is_correct=False,
            temporal_iou=0.0,
            iou_ge_95=False,
            iou_ge_05=False,
        )

    return record


def evaluate_static_loco_clip(
    clip_dir: Path,
    expected_behavior: str | None = None,
    params: HeuristicParameters | None = None,
    crop_output_dir: Path | None = None,
    orig_annotated_dir: Path | None = None,
    visualize: bool = True,
    render_videos: bool = True,
    overwrite_videos: bool = False,
) -> dict[str, Any]:
    """Evaluate a single Static or Locomotion clip blindly and perform segment cropping."""
    clip_dir = Path(clip_dir)
    clip_id = clip_dir.name
    if params is None:
        params = _default_parameters()

    record: dict[str, Any] = {
        "clip_id": clip_id,
        "clip_dir": str(clip_dir),
        "status": "ok",
        "error": None,
        "ground_truth_behavior": expected_behavior,
        "predicted_behavior": None,
        "is_correct": False,
        "target_mouse_ids": [],
        "ground_truth_range_half_open": None,
        "predicted_range_half_open": None,
        "temporal_iou": 0.0,
        "iou_ge_90": False,
        "iou_ge_95": False,
        "iou_ge_05": False,
        "total_frames": 0,
        "fps": 30.0,
        "cropped_video_path": None,
        "annotated_orig_path": None,
    }

    try:
        annotation, metadata, raw_tracks, video_path = _load_clip_files(clip_dir)
        gt_behavior = str(annotation.get("behavior", expected_behavior or "Unknown"))
        target_ids = sorted(int(m) for m in annotation.get("mouse_ids", []))
        gt_start, _, gt_end_half_open, fps, total_frames = _parse_clip_timing(
            annotation, metadata, raw_tracks
        )

        record.update(
            target_mouse_ids=target_ids,
            ground_truth_behavior=gt_behavior,
            ground_truth_range_half_open={"start": gt_start, "end": gt_end_half_open},
            total_frames=total_frames,
            fps=fps,
        )

        store = InMemoryTrackStore(raw_tracks)
        binding = metadata.get("clip", {"fps": fps})

        pred_behavior = None
        refined_events: list[dict[str, Any]] = []
        target_id = target_ids[0] if target_ids else 1

        cand_static = {
            "canonical_behavior": "static",
            "behavior": "Static",
            "start_frame": 0,
            "end_frame": total_frames,
            "mouse_ids": [target_id],
            "target_mouse_id": target_id,
        }
        res_static = segment_clipping.refine_static_segments([cand_static], store, binding)

        cand_loco = {
            "canonical_behavior": "locomotion",
            "behavior": "Locomotion",
            "start_frame": 0,
            "end_frame": total_frames,
            "mouse_ids": [target_id],
            "target_mouse_id": target_id,
        }
        res_loco = segment_clipping.refine_locomotion_segments([cand_loco], store, binding)

        if res_static and not res_loco:
            pred_behavior = "Static"
            refined_events = res_static
        elif res_loco and not res_static:
            pred_behavior = "Locomotion"
            refined_events = res_loco
        elif res_static and res_loco:
            static_frames = sum(int(e["end_frame"]) - int(e["start_frame"]) for e in res_static)
            loco_frames = sum(int(e["end_frame"]) - int(e["start_frame"]) for e in res_loco)
            if loco_frames > static_frames:
                pred_behavior = "Locomotion"
                refined_events = res_loco
            else:
                pred_behavior = "Static"
                refined_events = res_static

        record["predicted_behavior"] = pred_behavior
        is_correct = bool(pred_behavior and gt_behavior and pred_behavior.lower() == gt_behavior.lower())
        record["is_correct"] = is_correct

        if is_correct and refined_events:
            best_ev = max(refined_events, key=lambda ev: int(ev["end_frame"]) - int(ev["start_frame"]))
            p_s = int(best_ev["start_frame"])
            p_e = int(best_ev["end_frame"])
            record["predicted_range_half_open"] = {"start": p_s, "end": p_e}
            iou = compute_temporal_iou((p_s, p_e), (gt_start, gt_end_half_open))
        else:
            iou = 0.0

        record["temporal_iou"] = float(iou)
        record["iou_ge_90"] = bool(iou >= 0.90)
        record["iou_ge_95"] = bool(iou >= 0.95)
        record["iou_ge_05"] = bool(iou >= 0.50)

        if crop_output_dir is not None and record["predicted_range_half_open"]:
            out_cropped = Path(crop_output_dir) / clip_id / "clip_cropped.mp4"
            expected_crop_frames = (
                record["predicted_range_half_open"]["end"]
                - record["predicted_range_half_open"]["start"]
            )
            reuse_existing = (
                not overwrite_videos
                and out_cropped.is_file()
                and out_cropped.stat().st_size > 0
                and video_frame_count(out_cropped) == expected_crop_frames
            )
            if reuse_existing:
                record["cropped_video_path"] = str(out_cropped)
            elif render_videos and video_path.is_file():
                ok = render_cropped_comparison_video(
                    video_path,
                    out_cropped,
                    record["predicted_range_half_open"]["start"],
                    record["predicted_range_half_open"]["end"],
                    fps,
                    tracks=raw_tracks,
                    target_mouse_ids=target_ids,
                    pred_behavior=pred_behavior,
                    gt_behavior=gt_behavior,
                    iou=iou,
                    visualize=visualize,
                )
                record["cropped_video_path"] = str(out_cropped) if ok else None

        if orig_annotated_dir is not None:
            out_annotated = Path(orig_annotated_dir) / clip_id / "clip_annotated.mp4"
            if out_annotated.is_file() and out_annotated.stat().st_size > 0:
                record["annotated_orig_path"] = str(out_annotated)
            elif render_videos and video_path.is_file():
                ok_orig = render_annotated_original_video(
                    video_path,
                    out_annotated,
                    fps,
                    tracks=raw_tracks,
                    target_mouse_ids=target_ids,
                    behavior=gt_behavior,
                )
                record["annotated_orig_path"] = str(out_annotated) if ok_orig else None

    except Exception as exc:
        record.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            is_correct=False,
            temporal_iou=0.0,
            iou_ge_90=False,
            iou_ge_95=False,
            iou_ge_05=False,
        )

    return record


def evaluate_six_behavior_clip(
    clip_dir: Path,
    *,
    expected_behavior: str | None = None,
    params: HeuristicParameters | None = None,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    crop_output_dir: Path | None = None,
    orig_annotated_dir: Path | None = None,
    visualize: bool = True,
    render_videos: bool = True,
) -> dict[str, Any]:
    """Unified dispatcher evaluating a clip belonging to any of the six benchmark behaviors."""
    anno_path = Path(clip_dir) / "annotation.json"
    behavior_hint = expected_behavior
    if not behavior_hint and anno_path.is_file():
        try:
            behavior_hint = str(json.loads(anno_path.read_text(encoding="utf-8")).get("behavior", ""))
        except Exception:
            behavior_hint = ""
    normalized = str(behavior_hint or "").strip().lower()
    if normalized in {"attack", "chasing", "chase"}:
        return evaluate_attack_chasing_clip(
            clip_dir,
            params=params,
            social_config=social_config,
            crop_output_dir=crop_output_dir,
            visualize=visualize,
        )
    if normalized in {"isolation", "group locomotion", "group_locomotion"}:
        return evaluate_isolation_gloco_clip(
            clip_dir,
            expected_behavior=expected_behavior,
            params=params,
            crop_output_dir=crop_output_dir,
            orig_annotated_dir=orig_annotated_dir,
            visualize=visualize,
            render_videos=render_videos,
        )
    if normalized in {"static", "stationary", "locomotion"}:
        return evaluate_static_loco_clip(
            clip_dir,
            expected_behavior=expected_behavior,
            params=params,
            crop_output_dir=crop_output_dir,
            orig_annotated_dir=orig_annotated_dir,
            visualize=visualize,
            render_videos=render_videos,
        )
    # Fallback based on target participant count
    if anno_path.is_file():
        try:
            mouse_ids = json.loads(anno_path.read_text(encoding="utf-8")).get("mouse_ids", [])
            if len(mouse_ids) == 2:
                return evaluate_attack_chasing_clip(
                    clip_dir,
                    params=params,
                    social_config=social_config,
                    crop_output_dir=crop_output_dir,
                    visualize=visualize,
                )
            if len(mouse_ids) >= 3:
                return evaluate_isolation_gloco_clip(
                    clip_dir,
                    expected_behavior=expected_behavior,
                    params=params,
                    crop_output_dir=crop_output_dir,
                    orig_annotated_dir=orig_annotated_dir,
                    visualize=visualize,
                    render_videos=render_videos,
                )
        except Exception:
            pass
    return evaluate_static_loco_clip(
        clip_dir,
        expected_behavior=expected_behavior,
        params=params,
        crop_output_dir=crop_output_dir,
        orig_annotated_dir=orig_annotated_dir,
        visualize=visualize,
        render_videos=render_videos,
    )


def combine_evaluation_reports(
    report_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Merge per-category evaluation JSON files into one unified six-behavior report."""
    by_id: dict[str, dict[str, Any]] = {}
    for report_path in report_paths:
        if not report_path.is_file():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for clip in payload.get("per_clip_results", []):
            cid = str(clip["clip_id"])
            by_id[cid] = clip

    merged_clips = list(by_id.values())
    merged_clips.sort(key=lambda r: (str(r.get("ground_truth_behavior", "")), str(r["clip_id"])))

    summary: dict[str, Any] = {
        "evaluation_title": (
            f"Multi-Behavior Recognition & Boundary Cropping Unified Evaluation "
            f"(6 Behaviors, {len(merged_clips)} Clips)"
        ),
        "total_clips": len(merged_clips),
    }

    for beh in BEHAVIOR_ORDER:
        b_clips = [c for c in merged_clips if c.get("ground_truth_behavior") == beh]
        b_key = beh.lower().replace(" ", "_")
        summary[f"{b_key}_clips"] = len(b_clips)
        acc = (
            float(np.mean([1.0 if c.get("is_correct") else 0.0 for c in b_clips]))
            if b_clips
            else 0.0
        )
        summary[f"{b_key}_accuracy"] = round(acc, 4)
        corr_clips = [c for c in b_clips if c.get("is_correct")]
        iou = (
            float(np.mean([float(c.get("temporal_iou", 0.0)) for c in corr_clips]))
            if corr_clips
            else 0.0
        )
        summary[f"{b_key}_mean_iou"] = round(iou, 4)

    overall_acc = (
        float(np.mean([1.0 if c.get("is_correct") else 0.0 for c in merged_clips]))
        if merged_clips
        else 0.0
    )
    corr_all = [c for c in merged_clips if c.get("is_correct")]
    overall_iou = (
        float(np.mean([float(c.get("temporal_iou", 0.0)) for c in corr_all]))
        if corr_all
        else 0.0
    )
    iou_90 = (
        float(np.mean([1.0 if float(c.get("temporal_iou", 0.0)) >= 0.90 else 0.0 for c in corr_all]))
        if corr_all
        else 0.0
    )
    iou_95 = (
        float(np.mean([1.0 if float(c.get("temporal_iou", 0.0)) >= 0.95 else 0.0 for c in corr_all]))
        if corr_all
        else 0.0
    )
    iou_50 = (
        float(np.mean([1.0 if float(c.get("temporal_iou", 0.0)) >= 0.50 else 0.0 for c in corr_all]))
        if corr_all
        else 0.0
    )

    summary["overall_accuracy"] = round(overall_acc, 4)
    summary["overall_mean_iou"] = round(overall_iou, 4)
    summary["iou_ge_90_rate"] = round(iou_90, 4)
    summary["iou_ge_95_rate"] = round(iou_95, 4)
    summary["iou_ge_05_rate"] = round(iou_50, 4)
    summary["per_clip_results"] = merged_clips

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
