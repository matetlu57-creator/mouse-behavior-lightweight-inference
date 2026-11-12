"""Shared trajectory, scale, boundary, and interval utilities for segment clipping."""
from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from mouse_behavior.behavior.event_boundaries import EventBoundary
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

__all__ = [
    "InMemoryTrackStore",
    "endpoint_observations",
    "endpoints_fully_observed",
    "is_process_boundary_closed",
    "update_segment_bounds",
]


class InMemoryTrackStore:
    """Lightweight in-memory track store providing ``.rows(start, end)`` for segment clipping."""

    def __init__(self, tracks: Sequence[Mapping[str, Any]]):
        self._tracks = tracks

    def rows(self, start: int, end: int, *, rebase: bool = False) -> Iterator[dict[str, Any]]:
        for row in self._tracks:
            frame = int(row.get("frame", row.get("source_frame", 0)))
            if start <= frame < end:
                item = dict(row)
                if rebase:
                    item["frame"] = frame - int(start)
                yield item


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
    pts_ok = bool(
        points.shape == (7, 3) and (np.isfinite(points).all(axis=1) & (points[:, 2] >= 0.25)).any()
    )
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


def _resolve_scale_cm_per_px(
    rows: Sequence[Mapping[str, Any]],
    mids: Sequence[int],
    binding_or_info: Mapping[str, Any] | None = None,
    default_body_length_cm: float = 7.918,
    default_cm_per_px: float = 0.10,
    scale: Mapping[str, Any] | None = None,
) -> float:
    """Resolve centimetres-per-pixel ratio from explicit scale, binding, or keypoint sampling."""
    if isinstance(scale, Mapping):
        if "cm_per_pixel" in scale:
            val = float(scale["cm_per_pixel"])
            if math.isfinite(val) and val > 0:
                return val
        if "px_per_cm" in scale:
            val = float(scale["px_per_cm"])
            if math.isfinite(val) and val > 0:
                return 1.0 / val
        if "scale_px_per_cm" in scale:
            val = float(scale["scale_px_per_cm"])
            if math.isfinite(val) and val > 0:
                return 1.0 / val

    for container in (binding_or_info, _resolve_video_info(binding_or_info)):
        if not isinstance(container, Mapping):
            continue
        if "cm_per_pixel" in container:
            val = float(container["cm_per_pixel"])
            if math.isfinite(val) and val > 0:
                return val
        if "px_per_cm" in container:
            val = float(container["px_per_cm"])
            if math.isfinite(val) and val > 0:
                return 1.0 / val
        if "scale_px_per_cm" in container:
            val = float(container["scale_px_per_cm"])
            if math.isfinite(val) and val > 0:
                return 1.0 / val
        scale_blk = container.get("scale")
        if isinstance(scale_blk, Mapping):
            if "cm_per_pixel" in scale_blk:
                val = float(scale_blk["cm_per_pixel"])
                if math.isfinite(val) and val > 0:
                    return val
            if "px_per_cm" in scale_blk:
                val = float(scale_blk["px_per_cm"])
                if math.isfinite(val) and val > 0:
                    return 1.0 / val

    body_lens: list[float] = []
    target_ids = set(int(m) for m in mids)
    for row in rows:
        dmap = _detection_map(row)
        for m in target_ids:
            det = dmap.get(m)
            if not det:
                continue
            kps = np.asarray(det.get("keypoints", []), dtype=float)
            if kps.shape == (7, 3) and kps[0, 2] >= 0.25 and kps[6, 2] >= 0.25:
                blen = float(np.linalg.norm(kps[0, :2] - kps[6, :2]))
                if 15.0 <= blen <= 250.0:
                    body_lens.append(blen)

    if body_lens:
        med_body = float(np.median(body_lens))
        if med_body > 1.0:
            return float(default_body_length_cm / med_body)

    return default_cm_per_px


def _clip_dense_track_series(
    rows: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    target_id: int,
    scale_cm_per_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build frame-indexed (span, 1, 2) centers_cm and (span, 1) observed arrays."""
    span = max(0, end - start)
    centers_cm = np.full((span, 1, 2), np.nan, dtype=float)
    observed = np.zeros((span, 1), dtype=bool)
    for idx, row in enumerate(rows):
        raw_frame = row.get("frame", row.get("source_frame"))
        rel = int(raw_frame) - start if raw_frame is not None else idx
        if not (0 <= rel < span):
            continue
        dmap = _detection_map(row)
        if target_id not in dmap:
            continue
        c = _extract_center_px(dmap[target_id])
        if c is not None:
            centers_cm[rel, 0] = c * scale_cm_per_px
            observed[rel, 0] = True
    return centers_cm, observed


def _observed_support_intervals(observed_slice: np.ndarray, abs_start: int) -> list[list[int]]:
    """Convert a 1D boolean observation mask into half-open [start, end) intervals."""
    obs_idx = np.flatnonzero(observed_slice)
    if not len(obs_idx):
        return []
    intervals: list[list[int]] = []
    curr_s = int(obs_idx[0])
    prev = curr_s
    for idx in obs_idx[1:]:
        idx_i = int(idx)
        if idx_i == prev + 1:
            prev = idx_i
        else:
            intervals.append([abs_start + curr_s, abs_start + prev + 1])
            curr_s = idx_i
            prev = idx_i
    intervals.append([abs_start + curr_s, abs_start + prev + 1])
    return intervals


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

    for key in (
        "canonical_support_frame_intervals",
        "observed_all_participants_frame_intervals",
        "role_support_frame_intervals",
    ):
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
        elif key == "observed_all_participants_frame_intervals":
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
