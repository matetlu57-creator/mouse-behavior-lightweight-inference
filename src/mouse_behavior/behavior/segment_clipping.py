"""Modular behavior segment clipping, boundary verification, fine-cropping, and selection.

Unifies and modularizes:
- Codex source-level segment boundary verification (`endpoint_observations`,
  `is_process_boundary_closed`), sliding huddle output refinement
  (`refine_huddle_segments`), and source-store approach requalification
  (`refine_approach_segments`).
- Calibrated kinematic/postural fine-cropping (`refine_rearing_segments`,
  `refine_attack_segments`, `refine_dispersal_segments`,
  `refine_group_locomotion_segments`).
- Non-overlapping behavior segment selection (`select_behavior_segments`) with
  NO default output count cap (`max_per_behavior=None` by default).

Visual IDs remain strictly tracker-owned (`track_id` / `visual_track_id`); missing
identities or unobserved endpoints are never fabricated.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.behavior.event_boundaries import EventBoundary
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "endpoint_observations",
    "endpoints_fully_observed",
    "is_process_boundary_closed",
    "refine_approach_segments",
    "refine_attack_segments",
    "refine_behavior_segments",
    "refine_dispersal_segments",
    "refine_group_locomotion_segments",
    "refine_huddle_segments",
    "refine_rearing_segments",
    "select_behavior_segments",
    "update_segment_bounds",
]


def _resolve_video_info(binding_or_info: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if binding_or_info is None:
        return {}
    if "source_video" in binding_or_info and isinstance(binding_or_info["source_video"], Mapping):
        return binding_or_info["source_video"]
    return binding_or_info


def _resolve_fps(binding_or_info: Mapping[str, Any] | None, default: float = 30.0) -> float:
    info = _resolve_video_info(binding_or_info)
    fps = float(info.get("fps", default))
    return fps if math.isfinite(fps) and fps > 0 else default


def _resolve_parameters(
    binding: Mapping[str, Any] | None,
    parameters: HeuristicParameters | None = None,
) -> HeuristicParameters:
    if parameters is not None:
        return parameters
    if isinstance(binding, Mapping):
        effective = binding.get("effective_parameters")
        if isinstance(effective, Mapping):
            return HeuristicParameters(**effective)
    return HeuristicParameters()


def _detection_map(row: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(d["track_id"]): d for d in row.get("detections", ()) if "track_id" in d}


def _has_valid_measurement(detection: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    box = np.asarray(detection.get("box", []), dtype=float)
    points = np.asarray(detection.get("keypoints", []), dtype=float)
    box_ok = bool(box.shape == (4,) and np.isfinite(box).all() and (box[2:] > box[:2]).all())
    pts_ok = bool(points.shape == (7, 3) and (np.isfinite(points).all(axis=1) & (points[:, 2] >= 0.25)).any())
    return (box_ok or pts_ok), box_ok, pts_ok


def _extract_center_px(detection: Mapping[str, Any]) -> np.ndarray | None:
    points = np.asarray(detection.get("keypoints", []), dtype=float)
    if points.shape == (7, 3):
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= 0.25)
        body = valid.copy()
        body[[0, 1, 2, 6]] = False
        usable = body if body.any() else valid
        if usable.any():
            return np.median(points[usable, :2], axis=0)
    box = np.asarray(detection.get("box", []), dtype=float)
    if box.shape == (4,) and np.isfinite(box).all() and (box[2:] > box[:2]).all():
        return 0.5 * (box[:2] + box[2:])
    return None


def endpoint_observations(
    store: Any,
    event: Mapping[str, Any],
    video_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether all event participants have measured detections at start and end frames."""
    info = _resolve_video_info(video_info)
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    frame_count = info.get("frame_count")
    if start < 0 or end <= start or (frame_count is not None and end > int(frame_count)):
        return {
            "start_observed": False,
            "end_observed": False,
            "start_boxes": False,
            "end_boxes": False,
            "start_keypoints": False,
            "end_keypoints": False,
        }
    ids = [int(mid) for mid in event.get("mouse_ids", ())]
    if not ids:
        return {
            "start_observed": False,
            "end_observed": False,
            "start_boxes": False,
            "end_boxes": False,
            "start_keypoints": False,
            "end_keypoints": False,
        }

    def _frame_flags(frame_index: int) -> tuple[bool, bool, bool]:
        rows = list(store.rows(frame_index, frame_index + 1))
        if not rows:
            return False, False, False
        detections = _detection_map(rows[0])
        if not all(track in detections for track in ids):
            return False, False, False
        checks = [_has_valid_measurement(detections[track]) for track in ids]
        return (
            all(item[0] for item in checks),
            all(item[1] for item in checks),
            all(item[2] for item in checks),
        )

    start_obs, start_box, start_kp = _frame_flags(start)
    end_obs, end_box, end_kp = _frame_flags(end - 1)
    return {
        "start_observed": start_obs,
        "end_observed": end_obs,
        "start_boxes": start_box,
        "end_boxes": end_box,
        "start_keypoints": start_kp,
        "end_keypoints": end_kp,
    }


def endpoints_fully_observed(
    store: Any,
    event: Mapping[str, Any],
    video_info: Mapping[str, Any] | None = None,
) -> bool:
    """Return True iff all participants in `event` are observed at both `start_frame` and `end_frame - 1`."""
    obs = endpoint_observations(store, event, video_info)
    return bool(obs["start_observed"] and obs["end_observed"])


def is_process_boundary_closed(event: Mapping[str, Any]) -> bool:
    """Verify that a behavior segment has rule-closed boundaries without window truncation."""
    status = event.get("process_boundary_status") or {}
    canonical = str(event.get("canonical_behavior", ""))
    if canonical == "approach":
        if status:
            return bool(
                status.get("requalified_on_source_store") is True
                and not status.get("clipped_by_inference_window", True)
                and status.get("no_body_contact_throughout") is True
            )
        return True
    if canonical == "huddle":
        if status and "output_revision" in status:
            return status.get("output_revision") == "huddle-sliding-output-v37"
        return True
    if canonical in {"isolation", "social_clustering", "dispersal", "group_locomotion"}:
        if status:
            if status.get("clipped_by_inference_window", False):
                return False
            if status.get("starts_and_ends_with_all_targets_observed", True) is False:
                return False
            if "start_closed_in_context" in status or "end_closed_in_context" in status:
                start_ok = bool(status.get("start_closed_in_context") or status.get("starts_at_source_boundary"))
                end_ok = bool(status.get("end_closed_in_context") or status.get("ends_at_source_boundary"))
                return start_ok and end_ok
        return True
    return True


def update_segment_bounds(
    event: Mapping[str, Any],
    new_start: int,
    new_end: int,
    fps: float,
    suffix: str = "crop",
) -> dict[str, Any]:
    """Return a copy of `event` updated to half-open `[new_start, new_end)` source frame bounds."""
    if new_end <= new_start:
        raise ValueError(f"invalid segment bounds [{new_start}, {new_end})")
    updated = copy.deepcopy(dict(event))
    old_start = int(updated["start_frame"])
    old_end = int(updated["end_frame"])
    updated["start_frame"] = int(new_start)
    updated["end_frame"] = int(new_end)
    updated["frame_count"] = int(new_end - new_start)
    updated["duration_s"] = float((new_end - new_start) / fps)
    updated["start_time_s"] = float(new_start / fps)
    updated["end_time_s"] = float(new_end / fps)

    for key in ("canonical_support_frame_intervals", "observed_all_participants_frame_intervals", "role_support_frame_intervals"):
        if key in updated and isinstance(updated[key], list):
            clipped = []
            for interval in updated[key]:
                a, b = int(interval[0]), int(interval[1])
                ca, cb = max(int(new_start), a), min(int(new_end), b)
                if ca < cb:
                    clipped.append([ca, cb])
            updated[key] = clipped
            count_key = key.replace("_intervals", "_count")
            updated[count_key] = int(sum(b - a for a, b in clipped))
        elif key in ("canonical_support_frame_intervals", "observed_all_participants_frame_intervals"):
            updated[key] = [[int(new_start), int(new_end)]]
            updated[key.replace("_intervals", "_count")] = int(new_end - new_start)

    if "observed_all_participants_frame_count" in updated:
        updated["inferred_or_partially_unobserved_frame_count"] = max(
            0,
            int(new_end - new_start) - int(updated["observed_all_participants_frame_count"]),
        )

    updated["boundary"] = EventBoundary(
        int(new_start),
        int(new_end),
        int(new_start),
        int(new_end),
        float(fps),
    ).as_dict()

    if new_start != old_start or new_end != old_end:
        parent_id = updated.get("event_id")
        if parent_id and "parent_candidate_event_id" not in updated:
            updated["parent_candidate_event_id"] = parent_id
        canonical = updated.get("canonical_behavior", "segment")
        digest = hashlib.sha256(
            f"{canonical}_{tuple(updated.get('mouse_ids', ()))}_{new_start}_{new_end}_{suffix}".encode()
        ).hexdigest()[:10]
        updated["event_id"] = f"{canonical}_{suffix}_{int(new_start):08d}_{int(new_end):08d}_{digest}"
    return updated


def refine_huddle_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine huddle candidates with sliding stable-state qualification and observed touching endpoints."""
    from mouse_behavior.visualization.behavior_packages import huddle_output_events

    return huddle_output_events([dict(e) for e in events], store, dict(binding), dict(scale))


def refine_approach_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
    parameters: HeuristicParameters | None = None,
    *,
    scene_guard: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Requalify approach candidates on the source store with optional multi-mouse scene guard."""
    from mouse_behavior.evaluation.approach_events import requalify_source_approach_event

    video_info = _resolve_video_info(binding)
    params = _resolve_parameters(binding, parameters)
    refined: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "approach":
            refined.append(copy.deepcopy(dict(event)))
            continue
        children = requalify_source_approach_event(
            store,
            dict(event),
            dict(video_info),
            dict(scale),
            params,
            scene_guard=scene_guard,
        )
        if not children:
            audit.append({
                "parent_event_id": event.get("event_id"),
                "accepted": [],
                "reason": "rejected_by_source_requalification",
            })
            continue
        accepted_ids = []
        for child in children:
            refined.append(child)
            accepted_ids.append(child.get("event_id"))
        audit.append({
            "parent_event_id": event.get("event_id"),
            "accepted": accepted_ids,
        })
    return refined, audit


def refine_rearing_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float = 0.5,
    max_duration_s: float | None = None,
    max_mounted_fraction: float = 0.20,
    max_box_aspect_ratio: float = 1.65,
    max_ear_neck_px: float = 20.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate standalone upright rearing segments.

    Suppresses social mounting/climbing (bounding box overlap with another
    detected mouse) and curled dorsal grooming (`cos_angle > 0.15`), and trims
    unobserved endpoint frames. `max_duration_s` defaults to `None` so long
    rearing bouts are never arbitrarily discarded unless a caller specifies a cap.
    """
    fps = _resolve_fps(binding)
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "rearing":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mouse_ids = event.get("mouse_ids", ())
        if not mouse_ids or end <= start:
            continue
        mid = int(mouse_ids[0])
        rows = list(store.rows(start, end))
        if not rows:
            continue

        valid_posture = np.zeros(len(rows), dtype=bool)
        present_mask = np.zeros(len(rows), dtype=bool)
        mounted_frames = 0
        ear_neck_vals: list[float] = []
        aspect_vals: list[float] = []
        cos_vals: list[float] = []

        for idx, row in enumerate(rows):
            det_map = _detection_map(row)
            det = det_map.get(mid)
            if det is None:
                continue
            box = np.asarray(det.get("box", []), dtype=float)
            pts = np.asarray(det.get("keypoints", []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
                continue
            present_mask[idx] = True
            bw = max(1.0, float(box[2] - box[0]))
            bh = max(1.0, float(box[3] - box[1]))
            aspect = float(max(bw, bh) / min(bw, bh))
            aspect_vals.append(aspect)

            other_overlap = False
            for oid, odet in det_map.items():
                if oid == mid:
                    continue
                obox = np.asarray(odet.get("box", []), dtype=float)
                if obox.shape == (4,) and np.isfinite(obox).all():
                    ix1 = max(float(box[0]), float(obox[0]))
                    iy1 = max(float(box[1]), float(obox[1]))
                    ix2 = min(float(box[2]), float(obox[2]))
                    iy2 = min(float(box[3]), float(obox[3]))
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / (bw * bh) >= 0.12:
                            other_overlap = True
                            break
            if other_overlap:
                mounted_frames += 1

            if pts.shape == (7, 3) and np.all(pts[[0, 1, 2, 3, 6], 2] >= 0.25):
                ear_mid = 0.5 * (pts[1, :2] + pts[2, :2])
                ear_neck = float(np.linalg.norm(ear_mid - pts[3, :2]))
                nose_tail = float(np.linalg.norm(pts[0, :2] - pts[6, :2]))
                v_hn = pts[0, :2] - pts[3, :2]
                v_tn = pts[6, :2] - pts[3, :2]
                nhn = float(np.linalg.norm(v_hn))
                ntn = float(np.linalg.norm(v_tn))
                cos_ang = float(np.dot(v_hn, v_tn) / (nhn * ntn)) if (nhn > 2.0 and ntn > 2.0) else 0.0
                ear_neck_vals.append(ear_neck)
                cos_vals.append(cos_ang)
                if (
                    not other_overlap
                    and ear_neck <= max_ear_neck_px
                    and nose_tail < 72.0
                    and aspect <= max_box_aspect_ratio
                    and cos_ang <= 0.15
                ):
                    valid_posture[idx] = True

        if not present_mask.any():
            continue

        # Bridge short internal posture jitter gaps (<= 6 frames) when endpoints are unmounted
        bridged = valid_posture.copy()
        pos_idx = np.flatnonzero(bridged)
        if len(pos_idx) >= 2:
            for a, b in zip(pos_idx[:-1], pos_idx[1:]):
                if 1 < (b - a) <= 6 and present_mask[a:b].all():
                    bridged[a:b] = True

        # Extract candidate sub-bouts or fall back to full event if whole-event medians qualify
        candidates: list[tuple[int, int]] = []
        idx = 0
        while idx < len(bridged):
            if not bridged[idx]:
                idx += 1
                continue
            j = idx + 1
            while j < len(bridged) and bridged[j]:
                j += 1
            candidates.append((start + idx, start + j))
            idx = j

        if not candidates:
            med_ear = float(np.median(ear_neck_vals)) if ear_neck_vals else 999.0
            med_asp = float(np.median(aspect_vals)) if aspect_vals else 999.0
            med_cos = float(np.median(cos_vals)) if cos_vals else 1.0
            mounted_ratio = mounted_frames / max(1, int(present_mask.sum()))
            if (
                mounted_ratio <= max_mounted_fraction
                and med_ear <= max_ear_neck_px
                and med_asp <= max_box_aspect_ratio
                and med_cos <= 0.10
            ):
                p_idx = np.flatnonzero(present_mask)
                candidates.append((start + int(p_idx[0]), start + int(p_idx[-1]) + 1))

        for sub_start, sub_end in candidates:
            rel_s = sub_start - start
            rel_e = sub_end - start
            p_sub = np.flatnonzero(present_mask[rel_s:rel_e])
            if not len(p_sub):
                continue
            true_s = sub_start + int(p_sub[0])
            true_e = sub_start + int(p_sub[-1]) + 1
            dur_s = (true_e - true_s) / fps
            if dur_s + 1e-9 < min_duration_s:
                continue
            if max_duration_s is not None and dur_s - 1e-9 > max_duration_s:
                continue
            cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="rearing_crop")
            if qualify_event_duration(cropped, fps, parameters=_resolve_parameters(binding))["ok"]:
                refined.append(cropped)
    return refined


def refine_attack_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float | None = None,
    max_dropout_fraction: float = 0.10,
    min_path_px: float = 60.0,
    min_speed_px_s: float = 40.0,
) -> list[dict[str, Any]]:
    """Validate and trim attack segments to ensure high-confidence co-observed agonistic motion."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding)
    req_s = min_duration_s if min_duration_s is not None else float(params.attack_min_event_duration_s)
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "attack":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 2 or end <= start:
            continue
        rows = list(store.rows(start, end))
        if not rows:
            continue

        both_mask = np.zeros(len(rows), dtype=bool)
        tracks_pos: dict[int, list[np.ndarray]] = {m: [] for m in mids}
        for idx, row in enumerate(rows):
            det_map = _detection_map(row)
            centers = {}
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        centers[m] = c
            if len(centers) == len(mids):
                both_mask[idx] = True
                for m, c in centers.items():
                    tracks_pos[m].append(c)

        obs_idx = np.flatnonzero(both_mask)
        if not len(obs_idx):
            continue
        true_s = start + int(obs_idx[0])
        true_e = start + int(obs_idx[-1]) + 1
        span = true_e - true_s
        dur_s = span / fps
        if dur_s + 1e-9 < req_s:
            continue
        both_in_span = int(both_mask[obs_idx[0] : obs_idx[-1] + 1].sum())
        if (1.0 - both_in_span / max(1, span)) > max_dropout_fraction + 1e-9:
            continue

        total_path = 0.0
        for m in mids:
            pts = np.asarray(tracks_pos[m], dtype=float)
            if len(pts) >= 2:
                total_path += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        speed_px_s = total_path / max(dur_s, 1e-6)
        if total_path + 1e-9 < min_path_px or speed_px_s + 1e-9 < min_speed_px_s:
            continue

        cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="attack_crop")
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
    return refined


def refine_dispersal_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float = 5.0,
    cluster_radius_px: float = 120.0,
    min_radius_gain_px: float = 35.0,
    min_net_gain_px: float = 25.0,
) -> list[dict[str, Any]]:
    """Fine-crop dispersal segments to start at a compact cluster and end after outward expansion."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding)
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "dispersal":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 2 or end <= start:
            continue
        rows = list(store.rows(start, end))
        if not rows:
            continue

        rads: list[tuple[int, float]] = []
        all_obs_frames: set[int] = set()
        for row in rows:
            fn = int(row.get("frame", row.get("source_frame", 0)))
            det_map = _detection_map(row)
            centers = []
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        centers.append(c)
            if len(centers) == len(mids):
                all_obs_frames.add(fn)
            if len(centers) >= 2:
                pts = np.asarray(centers, dtype=float)
                rad = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
                rads.append((fn, rad))

        if len(rads) < 10:
            continue
        frames_arr = np.array([r[0] for r in rads], dtype=int)
        vals_arr = np.array([r[1] for r in rads], dtype=float)

        compact_idx = [
            i for i, r in enumerate(vals_arr)
            if r <= cluster_radius_px and int(frames_arr[i]) in all_obs_frames
        ]
        if not compact_idx:
            continue
        best_i = min(compact_idx, key=lambda idx: vals_arr[idx])
        new_s = int(frames_arr[best_i])
        start_rad = float(vals_arr[best_i])

        after_vals = vals_arr[best_i:]
        peak_offset = int(np.argmax(after_vals))
        peak_i = best_i + peak_offset
        peak_rad = float(vals_arr[peak_i])
        if peak_rad - start_rad < min_radius_gain_px:
            continue

        min_end_frame = new_s + int(math.floor(min_duration_s * fps)) + 1
        target_end = max(int(frames_arr[peak_i]) + int(round(1.0 * fps)), min_end_frame)
        target_end = min(end, target_end)

        valid_end_indices = [
            i for i in range(best_i + 1, len(frames_arr))
            if int(frames_arr[i]) >= min_end_frame
            and int(frames_arr[i]) <= max(target_end, min_end_frame + int(round(2.0 * fps)))
            and int(frames_arr[i]) in all_obs_frames
            and (vals_arr[i] - start_rad) >= min_net_gain_px
        ]
        if not valid_end_indices:
            valid_end_indices = [
                i for i in range(best_i + 1, len(frames_arr))
                if int(frames_arr[i]) >= min_end_frame
                and int(frames_arr[i]) in all_obs_frames
                and (vals_arr[i] - start_rad) >= min_net_gain_px
            ]
        if not valid_end_indices:
            continue

        end_i = valid_end_indices[0]
        new_e = int(frames_arr[end_i]) + 1
        if (new_e - new_s) / fps <= min_duration_s:
            continue

        cropped = update_segment_bounds(event, new_s, new_e, fps, suffix="dispersal_crop")
        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
    return refined


def refine_group_locomotion_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any] | None = None,
    *,
    min_duration_s: float = 3.0,
    min_support_frames: int = 28,
    min_participation_ratio: float = 0.70,
    max_end_radius_px: float = 145.0,
    min_speed_px_s: float = 12.0,
) -> list[dict[str, Any]]:
    """Fine-crop and validate group locomotion segments for coordinated multi-animal movement."""
    fps = _resolve_fps(binding)
    params = _resolve_parameters(binding)
    refined: list[dict[str, Any]] = []
    for event in events:
        if event.get("canonical_behavior") != "group_locomotion":
            refined.append(copy.deepcopy(dict(event)))
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        mids = [int(m) for m in event.get("mouse_ids", ())]
        if len(mids) < 3 or end <= start:
            continue

        # Trim to canonical simultaneous motion support window if available
        supp = event.get("canonical_support_frame_intervals")
        if isinstance(supp, list) and len(supp) > 0:
            s0 = int(supp[0][0])
            e0 = int(supp[-1][1])
            pad = int(round(0.4 * fps))
            cand_s = max(start, s0 - pad)
            cand_e = min(end, e0 + pad)
            if (cand_e - cand_s) / fps > min_duration_s:
                start, end = cand_s, cand_e

        rows = list(store.rows(start, end))
        if not rows:
            continue

        all_obs_indices = []
        rads: list[float] = []
        centroids: list[np.ndarray] = []
        per_mouse_pos: dict[int, list[np.ndarray]] = {m: [] for m in mids}

        for idx, row in enumerate(rows):
            det_map = _detection_map(row)
            centers = {}
            for m in mids:
                if m in det_map:
                    c = _extract_center_px(det_map[m])
                    if c is not None:
                        centers[m] = c
                        per_mouse_pos[m].append(c)
            if len(centers) == len(mids):
                all_obs_indices.append(idx)
            if len(centers) >= 2:
                pts = np.asarray(list(centers.values()), dtype=float)
                c_mean = pts.mean(axis=0)
                centroids.append(c_mean)
                rads.append(float(np.linalg.norm(pts - c_mean, axis=1).mean()))

        if not all_obs_indices or len(centroids) < 5:
            continue
        true_s = start + int(all_obs_indices[0])
        true_e = start + int(all_obs_indices[-1]) + 1
        dur_s = (true_e - true_s) / fps
        if dur_s <= min_duration_s:
            continue

        cropped = update_segment_bounds(event, true_s, true_e, fps, suffix="gloco_crop")
        supp_frames = int(cropped.get("canonical_support_frame_count", true_e - true_s))
        if supp_frames < min_support_frames:
            continue

        end_rad = float(np.mean(rads[-min(15, len(rads)) :]))
        if end_rad > max_end_radius_px:
            continue

        c_arr = np.asarray(centroids, dtype=float)
        c_path = float(np.linalg.norm(np.diff(c_arr, axis=0), axis=1).sum())
        if (c_path / max(dur_s, 1e-6)) < min_speed_px_s:
            continue

        mouse_paths = []
        for m in mids:
            mp = np.asarray(per_mouse_pos[m], dtype=float)
            if len(mp) >= 2:
                mouse_paths.append(float(np.linalg.norm(np.diff(mp, axis=0), axis=1).sum()))
            else:
                mouse_paths.append(0.0)
        if not mouse_paths or max(mouse_paths) <= 0.0:
            continue
        if (min(mouse_paths) / max(mouse_paths)) < min_participation_ratio:
            continue

        if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
            refined.append(cropped)
    return refined


def refine_behavior_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any] | None = None,
    parameters: HeuristicParameters | None = None,
    *,
    enable_huddle_sliding: bool = True,
    enable_approach_requalification: bool = True,
    enable_fine_cropping: bool = False,
    scene_guard: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Unified segment refinement pipeline combining Codex and calibrated fine-cropping."""
    current = [copy.deepcopy(dict(e)) for e in events]
    audit: dict[str, Any] = {
        "input_count": len(current),
        "huddle_audit": [],
        "approach_audit": [],
    }
    if enable_huddle_sliding and scale is not None and any(e.get("canonical_behavior") == "huddle" for e in current):
        current, huddle_audit = refine_huddle_segments(current, store, binding, scale)
        audit["huddle_audit"] = huddle_audit

    if (
        enable_approach_requalification
        and scale is not None
        and any(e.get("canonical_behavior") == "approach" for e in current)
    ):
        current, approach_audit = refine_approach_segments(
            current,
            store,
            binding,
            scale,
            parameters=parameters,
            scene_guard=scene_guard,
        )
        audit["approach_audit"] = approach_audit

    if enable_fine_cropping:
        current = refine_rearing_segments(current, store, binding)
        current = refine_attack_segments(current, store, binding)
        current = refine_dispersal_segments(current, store, binding)
        current = refine_group_locomotion_segments(current, store, binding)

    audit["output_count"] = len(current)
    return current, audit


def _segment_priority_key(event: Mapping[str, Any], fps: float) -> tuple[float, float, float, int]:
    """Higher tuple values indicate stronger measured support for deduplicating overlapping same-ID bouts."""
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    span = max(1, end - start)
    observed = int(event.get("observed_all_participants_frame_count", span))
    support = int(event.get("canonical_support_frame_count", observed))
    obs_ratio = observed / span
    supp_ratio = support / span
    dur_s = span / fps
    return (round(obs_ratio, 4), round(supp_ratio, 4), round(dur_s, 3), -start)


def select_behavior_segments(
    events: Sequence[Mapping[str, Any]],
    store: Any = None,
    binding: Mapping[str, Any] | None = None,
    *,
    behaviors: Sequence[str] | None = None,
    max_per_behavior: int | None = None,
    require_closed_boundaries: bool = True,
    require_endpoint_visibility: bool = True,
    strict_non_overlapping_same_ids: bool = True,
    time_bucket_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select and deduplicate behavior segments without any default output count limit.

    Parameters
    ----------
    events:
        Candidate behavior segment dicts.
    store:
        Optional `TrackStore` used to verify endpoint visibility when
        `require_endpoint_visibility=True`.
    binding:
        Optional run binding or `source_video` dict providing `fps`, `frame_count`,
        and `effective_parameters`.
    behaviors:
        Optional subset of canonical or display behavior names to retain. When
        `None` (default), all behaviors present in `events` are retained.
    max_per_behavior:
        Optional maximum number of segments per behavior. Defaults to `None`
        (unlimited: every qualifying segment is returned).
    require_closed_boundaries:
        Reject segments whose `process_boundary_status` marks window clipping or
        unclosed context boundaries.
    require_endpoint_visibility:
        When `store` is provided, require all participants in `mouse_ids` to be
        observed at both `start_frame` and `end_frame - 1`.
    strict_non_overlapping_same_ids:
        Prevent temporally overlapping segments for the exact same
        `(canonical_behavior, tuple(mouse_ids))`.
    time_bucket_seconds:
        Optional coarse time-bucket spacing in seconds. Defaults to `None`
        (disabled so no valid segments are dropped by time bucketing).
    """
    if max_per_behavior is not None and (type(max_per_behavior) is not int or max_per_behavior < 1):
        raise ValueError("max_per_behavior must be None or a positive integer")
    if time_bucket_seconds is not None and (not math.isfinite(time_bucket_seconds) or time_bucket_seconds <= 0):
        raise ValueError("time_bucket_seconds must be None or a positive finite number")

    fps = _resolve_fps(binding)
    video_info = _resolve_video_info(binding)
    params = _resolve_parameters(binding)
    allowed = set(behaviors) if behaviors is not None else None

    grouped: dict[str, list[dict[str, Any]]] = {}
    rejected_counts: dict[str, int] = {
        "filtered_behavior": 0,
        "duration_policy": 0,
        "unclosed_boundary": 0,
        "unobserved_endpoints": 0,
        "duplicate_or_overlapping": 0,
    }

    seen_exact: set[tuple[str, tuple[int, ...], int, int]] = set()
    for raw in events:
        event = copy.deepcopy(dict(raw))
        canonical = str(event.get("canonical_behavior") or event.get("behavior") or "")
        display = str(event.get("behavior") or canonical)
        if allowed is not None and canonical not in allowed and display not in allowed:
            rejected_counts["filtered_behavior"] += 1
            continue
        if not qualify_event_duration(event, fps, parameters=params)["ok"]:
            rejected_counts["duration_policy"] += 1
            continue
        if require_closed_boundaries and not is_process_boundary_closed(event):
            rejected_counts["unclosed_boundary"] += 1
            continue
        if require_endpoint_visibility and store is not None:
            if not endpoints_fully_observed(store, event, video_info):
                rejected_counts["unobserved_endpoints"] += 1
                continue
        key = (
            canonical,
            tuple(int(m) for m in event.get("mouse_ids", ())),
            int(event["start_frame"]),
            int(event["end_frame"]),
        )
        if key in seen_exact:
            rejected_counts["duplicate_or_overlapping"] += 1
            continue
        seen_exact.add(key)
        grouped.setdefault(canonical, []).append(event)

    selected_all: list[dict[str, Any]] = []
    counts_by_behavior: dict[str, int] = {}

    for canonical, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda ev: _segment_priority_key(ev, fps),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        used_buckets: set[tuple[tuple[int, ...], int]] = set()

        for cand in ranked:
            mids = tuple(int(m) for m in cand.get("mouse_ids", ()))
            s = int(cand["start_frame"])
            e = int(cand["end_frame"])
            if strict_non_overlapping_same_ids:
                overlaps = any(
                    tuple(int(m) for m in existing.get("mouse_ids", ())) == mids
                    and not (e <= int(existing["start_frame"]) or s >= int(existing["end_frame"]))
                    for existing in chosen
                )
                if overlaps:
                    rejected_counts["duplicate_or_overlapping"] += 1
                    continue
            if time_bucket_seconds is not None:
                bucket_frames = max(1, int(round(time_bucket_seconds * fps)))
                bkey = (mids, s // bucket_frames)
                if bkey in used_buckets:
                    rejected_counts["duplicate_or_overlapping"] += 1
                    continue
                used_buckets.add(bkey)
            chosen.append(cand)
            if max_per_behavior is not None and len(chosen) >= max_per_behavior:
                break

        chosen.sort(key=lambda ev: (int(ev["start_frame"]), int(ev["end_frame"]), str(ev.get("event_id", ""))))
        counts_by_behavior[canonical] = len(chosen)
        selected_all.extend(chosen)

    selected_all.sort(
        key=lambda ev: (
            int(ev["start_frame"]),
            int(ev["end_frame"]),
            str(ev.get("canonical_behavior", "")),
            str(ev.get("event_id", "")),
        )
    )
    summary = {
        "total_input_events": len(events),
        "total_selected_events": len(selected_all),
        "max_per_behavior": max_per_behavior,
        "counts_by_behavior": counts_by_behavior,
        "rejected_counts": rejected_counts,
    }
    return selected_all, summary
