"""Quantitative validation and audit helpers for modular group-behavior cropping.

Evaluates all five collective/group behaviors across full-length recordings:
- Huddling (``huddle``)
- Social clustering (``social_clustering``)
- Dispersal (``dispersal``)
- Group locomotion (``group_locomotion``)
- Isolation (``isolation``)
"""
from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mouse_behavior.behavior.event_boundaries import EventBoundary
from mouse_behavior.behavior.segment_clipping import (
    refine_behavior_segments,
    select_behavior_segments,
)
from mouse_behavior.behavior.segment_clipping_common import (
    _clip_multi_track_series,
    _detection_map,
    _extract_center_px,
    _observed_run_velocity_cm_s,
    _observed_support_intervals,
    _resolve_scale_cm_per_px,
    endpoints_fully_observed,
    event_participant_ids,
    max_duration_s_for_behavior,
    min_participants_for_behavior,
    normalize_behavior_key,
    qualify_clip_duration,
)
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG
from mouse_behavior.evaluation.segment_export import _sha256, _write_json

GROUP_BEHAVIOR_KEYS: tuple[str, ...] = (
    "huddle",
    "social_clustering",
    "dispersal",
    "group_locomotion",
    "isolation",
)

GROUP_BEHAVIOR_LABELS: dict[str, str] = {
    "huddle": "蜷缩聚堆 (Huddling)",
    "social_clustering": "社交聚集 (Social clustering)",
    "dispersal": "群体分散 (Dispersal)",
    "group_locomotion": "群体同步运动 (Group locomotion)",
    "isolation": "独处 (Isolation)",
}

_EVENT_ID_SPAN_RE = re.compile(r"_(\d{6,10})_(\d{6,10})(?:_|$)")


class CachedTrackStore:
    """Fast in-session reader for ``tracks.sqlite`` preserving ``TrackStore.rows`` semantics."""

    def __init__(self, store_or_path: Any, *, preload_max_frames: int = 150000, max_lru_frames: int = 32768):
        self.path = Path(getattr(store_or_path, "path", store_or_path))
        self._conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        row_count = int(self._conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        self._preloaded: dict[int, dict[str, Any]] | None = None
        self._lru: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._max_lru = max(256, int(max_lru_frames))
        if row_count <= preload_max_frames:
            self._preloaded = {
                int(fn): json.loads(payload)
                for fn, payload in self._conn.execute("SELECT frame, payload FROM frames ORDER BY frame")
            }

    def rows(self, start: int, end: int, *, rebase: bool = False) -> Iterator[dict[str, Any]]:
        s, e = int(start), int(end)
        if e <= s:
            return
        if self._preloaded is not None:
            for fn in range(s, e):
                base = self._preloaded.get(fn)
                if base is None:
                    continue
                row = dict(base)
                if rebase:
                    row["frame"] = int(row["frame"]) - s
                    fps = float(row.pop("source_fps"))
                    row["time"] = row["frame"] / fps
                else:
                    row.pop("source_fps", None)
                yield row
            return

        if any(fn not in self._lru for fn in range(s, e)):
            for fn, payload in self._conn.execute(
                "SELECT frame, payload FROM frames WHERE frame>=? AND frame<? ORDER BY frame",
                (s, e),
            ):
                if int(fn) not in self._lru:
                    self._lru[int(fn)] = json.loads(payload)
                else:
                    self._lru.move_to_end(int(fn))
            while len(self._lru) > self._max_lru:
                self._lru.popitem(last=False)

        for fn in range(s, e):
            base = self._lru.get(fn)
            if base is None:
                continue
            self._lru.move_to_end(fn)
            row = dict(base)
            if rebase:
                row["frame"] = int(row["frame"]) - s
                fps = float(row.pop("source_fps"))
                row["time"] = row["frame"] / fps
            else:
                row.pop("source_fps", None)
            yield row

    def close(self) -> None:
        self._conn.close()


def write_pose_cache_source_receipt(
    video: Path,
    cache_dir: Path,
    model_path: Path,
) -> Path:
    """Create or validate ``source_receipt.json`` required by ``canonical_video_pipeline.bind_cache``."""
    from mouse_behavior.canonical_video_pipeline import video_info

    status_path = cache_dir / "yolo_results_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing yolo_results_status.json in {cache_dir}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"pose cache is incomplete: {status.get('status')}")

    resolved_video = video.resolve()
    resolved_model = model_path.resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"model file not found: {resolved_model}")

    info = video_info(resolved_video)
    video_sha = _sha256(resolved_video)
    model_sha = _sha256(resolved_model)
    if str(status.get("source_video_sha256", "")).lower() != video_sha.lower():
        raise ValueError("pose cache video SHA-256 differs from source video")

    receipt_path = cache_dir / "source_receipt.json"
    payload = {
        "schema_version": 1,
        "source_video": {**info, "sha256": video_sha},
        "model": {"path": str(resolved_model), "sha256": model_sha},
        "pose_cache_status_sha256": _sha256(status_path),
        "new_model_inference": True,
        "identity_semantics": "provisional_visual_track_id",
    }
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            existing.get("source_video", {}).get("sha256", "").lower() == video_sha.lower()
            and existing.get("model", {}).get("sha256", "").lower() == model_sha.lower()
            and Path(existing.get("model", {}).get("path", "")).resolve() == resolved_model
            and Path(existing.get("source_video", {}).get("path", "")).resolve() == resolved_video
        ):
            return receipt_path
        receipt_path.unlink()
    _write_json(receipt_path, payload)
    return receipt_path


def load_historical_review_decisions(review_json: Path | None) -> list[dict[str, Any]]:
    """Parse historical Video22-style backup JSON decisions with recovered frame spans."""
    if review_json is None or not review_json.is_file():
        return []
    raw = json.loads(review_json.read_text(encoding="utf-8-sig"))
    decisions = raw.get("decisions", {})
    if not isinstance(decisions, Mapping):
        return []
    records: list[dict[str, Any]] = []
    for event_id, payload in decisions.items():
        if not isinstance(payload, Mapping):
            continue
        beh = str(payload.get("behavior", ""))
        status = str(payload.get("status", "pending"))
        if status not in {"pass", "fail"}:
            continue
        match = _EVENT_ID_SPAN_RE.search(str(event_id))
        if not match:
            continue
        start_f, end_f = int(match.group(1)), int(match.group(2))
        if end_f <= start_f:
            continue
        records.append(
            {
                "event_id": str(event_id),
                "behavior": beh,
                "status": status,
                "start_frame": start_f,
                "end_frame": end_f,
                "note": str(payload.get("note", "")),
            }
        )
    return records


def refine_all_canonical_and_group_events(
    pre_events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
    parameters: HeuristicParameters,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run full modular refinement including sliding Huddling & calibrated group crops."""
    fps = float(binding["source_video"]["fps"])
    total = int(binding["source_video"]["frame_count"])
    refined, refinement_audit = refine_behavior_segments(
        pre_events,
        store,
        binding,
        scale=scale,
        parameters=parameters,
        enable_huddle_sliding=True,
        enable_approach_requalification=False,
        enable_fine_cropping=True,
    )
    for event in refined:
        s_ev, e_ev = int(event["start_frame"]), int(event["end_frame"])
        mids_ev = {int(m) for m in event.get("mouse_ids", ())}
        if "parent_candidate_event_id" in event or event.get("canonical_behavior") in GROUP_BEHAVIOR_KEYS:
            obs_mask = np.zeros(max(0, e_ev - s_ev), dtype=bool)
            for row in store.rows(s_ev, e_ev):
                fn = int(row.get("source_frame", row.get("frame", -1)))
                if s_ev <= fn < e_ev:
                    present = {int(d["track_id"]) for d in row.get("detections", ()) if "track_id" in d}
                    obs_mask[fn - s_ev] = mids_ev.issubset(present)
            obs_intervals = _observed_support_intervals(obs_mask, s_ev)
            obs_count = int(obs_mask.sum())
            event["observed_all_participants_frame_intervals"] = obs_intervals
            event["observed_all_participants_frame_count"] = obs_count
            event["inferred_or_partially_unobserved_frame_count"] = max(0, (e_ev - s_ev) - obs_count)
            event["boundary"] = EventBoundary(s_ev, e_ev, s_ev, e_ev, fps).as_dict()
        contexts = event.get("inference_parent_contexts", [])
        if contexts and "process_boundary_status" not in event:
            event["process_boundary_status"] = {
                "start_closed_in_context": any(c["context_start"] < s_ev < c["context_end"] for c in contexts),
                "end_closed_in_context": any(c["context_start"] < e_ev < c["context_end"] for c in contexts),
                "starts_at_source_boundary": s_ev == 0,
                "ends_at_source_boundary": e_ev == total,
                "minimum_duration_used_as_crop_length": False,
                "meaning": "Modular fine-cropped group/canonical segment boundaries",
            }
    selected, selection_audit = select_behavior_segments(
        refined,
        store=store,
        binding=binding,
        max_per_behavior=None,
        require_closed_boundaries=False,
        require_endpoint_visibility=False,
        strict_non_overlapping_shared_mouse=True,
    )
    return selected, {
        "refinement_audit": refinement_audit,
        "selection_audit": selection_audit,
    }


def _rows_centers_and_boxes(
    rows: Sequence[Mapping[str, Any]],
    mids: Sequence[int],
) -> tuple[list[int], dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]]]:
    frames: list[int] = []
    centers_by_frame: dict[int, dict[int, np.ndarray]] = {}
    boxes_by_frame: dict[int, dict[int, np.ndarray]] = {}
    target_set = set(mids)
    for row in rows:
        fn = int(row.get("frame", row.get("source_frame", 0)))
        det_map = _detection_map(row)
        c_map: dict[int, np.ndarray] = {}
        b_map: dict[int, np.ndarray] = {}
        for tid, det in det_map.items():
            c = _extract_center_px(det)
            if c is not None:
                c_map[tid] = c
            if tid in target_set:
                box = np.asarray(det.get("box", []), float)
                if box.shape == (4,) and np.isfinite(box).all() and (box[2:] > box[:2]).all():
                    b_map[tid] = box
        frames.append(fn)
        centers_by_frame[fn] = c_map
        boxes_by_frame[fn] = b_map
    return frames, centers_by_frame, boxes_by_frame


def _cluster_radius_cm(pts_px: list[np.ndarray], cm_per_px: float, *, metric: str = "mean") -> float:
    arr = np.asarray(pts_px, dtype=float)
    dists = np.linalg.norm(arr - arr.mean(axis=0), axis=1) * cm_per_px
    return float(dists.max() if metric == "max" else dists.mean())


def audit_group_segment_kinematics(
    event: Mapping[str, Any],
    store: Any,
    scale: Mapping[str, Any],
    fps: float,
    parameters: HeuristicParameters,
) -> dict[str, Any]:
    """Audit geometry, min-participants, duration bounds, and trajectory consistency of a cropped segment."""
    beh = str(event.get("canonical_behavior", ""))
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    mids = [int(m) for m in event.get("mouse_ids", ())]
    unique_mids = event_participant_ids(event)
    min_req_participants = min_participants_for_behavior(beh)
    min_participants_ok = len(unique_mids) >= min_req_participants
    span = max(1, end - start)
    duration_s = span / fps

    dur_check = qualify_clip_duration(dict(event), fps, parameters=parameters)
    min_duration_ok = bool(dur_check.get("ok", False))
    max_dur_s = max_duration_s_for_behavior(beh)
    duration_bounds_ok = bool(min_duration_ok and (max_dur_s is None or duration_s <= max_dur_s + 1e-6))
    duration_ok = duration_bounds_ok
    endpoint_obs = endpoints_fully_observed(store, event)
    obs_count = int(event.get("observed_all_participants_frame_count", 0))
    supp_count = int(event.get("canonical_support_frame_count", obs_count))
    obs_ratio = obs_count / span
    supp_ratio = supp_count / span

    rows = list(store.rows(start, end))
    frames, c_map, b_map = _rows_centers_and_boxes(rows, mids)
    cm_per_px = _resolve_scale_cm_per_px(rows, mids, None, scale=scale)
    if scale and scale.get("cm_per_pixel"):
        cm_per_px = float(scale["cm_per_pixel"])

    metrics: dict[str, Any] = {
        "duration_s": round(duration_s, 3),
        "frame_count": span,
        "participant_count": len(unique_mids),
        "min_participants_required": min_req_participants,
        "min_participants_ok": min_participants_ok,
        "max_duration_s": max_dur_s,
        "duration_bounds_ok": duration_bounds_ok,
        "duration_ok": duration_ok,
        "endpoint_observed_ok": endpoint_obs,
        "observed_all_ratio": round(obs_ratio, 4),
        "support_ratio": round(supp_ratio, 4),
    }

    kinematic_ok = False
    reason_notes: list[str] = []

    if beh == "huddle":
        compact_flags = []
        rads_cm = []
        sep_run = 0
        max_sep_run = 0
        for fn in frames:
            pts = [c_map[fn][m] for m in mids if m in c_map[fn]]
            if len(pts) >= 2:
                rad = _cluster_radius_cm(pts, cm_per_px, metric="max")
                rads_cm.append(rad)
                is_compact = rad <= 10.0 + 1e-6
                compact_flags.append(is_compact)
                if not is_compact:
                    sep_run += 1
                    max_sep_run = max(max_sep_run, sep_run)
                else:
                    sep_run = 0
        touch_flags = []
        for fn in (start, end - 1):
            boxes = [b_map[fn][m] for m in mids if fn in b_map and m in b_map[fn]]
            if len(boxes) == len(mids) and len(boxes) >= 3:
                arr = np.asarray(boxes, float)
                adj = (
                    (arr[:, None, 0] <= arr[None, :, 2])
                    & (arr[None, :, 0] <= arr[:, None, 2])
                    & (arr[:, None, 1] <= arr[None, :, 3])
                    & (arr[None, :, 1] <= arr[:, None, 3])
                )
                # BFS connectivity on bounding boxes
                seen = {0}
                queue = [0]
                while queue:
                    cur = queue.pop()
                    for nxt in range(len(boxes)):
                        if nxt not in seen and adj[cur, nxt]:
                            seen.add(nxt)
                            queue.append(nxt)
                touch_flags.append(len(seen) == len(boxes))
            else:
                touch_flags.append(False)
        compact_ratio = float(sum(compact_flags) / max(1, len(compact_flags)))
        med_rad = float(np.median(rads_cm)) if rads_cm else float("nan")
        max_sep_s = max_sep_run / fps
        metrics.update(
            {
                "compact_frame_ratio": round(compact_ratio, 4),
                "median_radius_cm": None if not math.isfinite(med_rad) else round(med_rad, 3),
                "max_separated_run_s": round(max_sep_s, 3),
                "touching_start": touch_flags[0],
                "touching_end": touch_flags[1],
            }
        )
        kinematic_ok = (
            compact_ratio >= 0.60
            and max_sep_s <= 0.50 + (1.0 / fps) + 1e-6
            and (touch_flags[0] and touch_flags[1])
        )
        if not touch_flags[0] or not touch_flags[1]:
            reason_notes.append("端点体框未完全接触连通")
        if max_sep_s > 0.50 + (1.0 / fps):
            reason_notes.append(f"含超过0.5s的分散间隔({max_sep_s:.2f}s)")

    elif beh == "social_clustering":
        k = len(mids)
        env_cm = 10.0 * (1.0 + 0.20 * max(0, k - 3))
        lb_start = max(0, start - int(round(3.0 * fps)))
        pre_rows = list(store.rows(lb_start, start)) if lb_start < start else []
        _, pre_c, _ = _rows_centers_and_boxes(pre_rows, mids)
        pre_rads = [
            _cluster_radius_cm([pre_c[fn][m] for m in mids if m in pre_c[fn]], cm_per_px, metric="max")
            for fn in sorted(pre_c)
            if sum(1 for m in mids if m in pre_c[fn]) >= 2
        ]
        rads = [
            (fn, _cluster_radius_cm([c_map[fn][m] for m in mids if m in c_map[fn]], cm_per_px, metric="max"))
            for fn in frames
            if sum(1 for m in mids if m in c_map[fn]) >= 2
        ]
        tail_frames = max(1, int(math.ceil(DEFAULT_GROUP_HEURISTIC_CONFIG.clustering_final_stability_s * fps)))
        tail_rads = [r for _, r in rads[-tail_frames:]] if rads else []
        tail_med = float(np.median(tail_rads)) if tail_rads else float("nan")
        pre_p90 = float(np.percentile(pre_rads + [r for _, r in rads[: max(1, len(rads) // 3)]], 90)) if (pre_rads or rads) else float("nan")
        drop_cm = (pre_p90 - tail_med) if (math.isfinite(pre_p90) and math.isfinite(tail_med)) else 0.0
        dense_arrivals = [fn for fn, r in rads if r <= env_cm + 1e-6]
        post_arrival_hold_s = ((end - dense_arrivals[0]) / fps) if dense_arrivals else 0.0
        metrics.update(
            {
                "envelope_limit_cm": round(env_cm, 2),
                "tail_median_radius_cm": None if not math.isfinite(tail_med) else round(tail_med, 3),
                "pre_p90_radius_cm": None if not math.isfinite(pre_p90) else round(pre_p90, 3),
                "radius_drop_cm": round(drop_cm, 3),
                "post_arrival_hold_s": round(post_arrival_hold_s, 3),
            }
        )
        kinematic_ok = (
            math.isfinite(tail_med)
            and tail_med <= env_cm + 0.5
            and drop_cm >= 1.0 - 1e-6
        )
        if not math.isfinite(tail_med) or tail_med > env_cm + 0.5:
            reason_notes.append(f"尾段聚集半径({tail_med:.2f}cm)超出紧凑包络({env_cm:.2f}cm)")
        if drop_cm < 1.0 - 1e-6:
            reason_notes.append(f"半径收敛降幅不足({drop_cm:.2f}cm)")

    elif beh == "dispersal":
        rads_px = []
        for fn in frames:
            pts = [c_map[fn][m] for m in mids if m in c_map[fn]]
            if len(pts) >= 2:
                arr = np.asarray(pts, float)
                rads_px.append(float(np.linalg.norm(arr - arr.mean(axis=0), axis=1).mean()))
        if len(rads_px) >= 4:
            early_k = max(1, min(len(rads_px) // 4, int(round(fps))))
            tail_k = max(1, min(len(rads_px) // 4, int(round(fps))))
            start_rad_px = float(np.min(rads_px[:early_k]))
            peak_rad_px = float(np.max(rads_px))
            end_rad_px = float(np.median(rads_px[-tail_k:]))
            rad_gain_cm = (peak_rad_px - start_rad_px) * cm_per_px
            net_gain_cm = (end_rad_px - start_rad_px) * cm_per_px
        else:
            start_rad_px = peak_rad_px = end_rad_px = 0.0
            rad_gain_cm = net_gain_cm = 0.0
        metrics.update(
            {
                "start_radius_cm": round(start_rad_px * cm_per_px, 3),
                "peak_radius_cm": round(peak_rad_px * cm_per_px, 3),
                "radius_gain_cm": round(rad_gain_cm, 3),
                "net_outward_gain_cm": round(net_gain_cm, 3),
            }
        )
        kinematic_ok = rad_gain_cm >= 2.0 and net_gain_cm >= 1.5
        if rad_gain_cm < 2.0 or net_gain_cm < 1.5:
            reason_notes.append(f"扩散净增益偏小(gain={rad_gain_cm:.2f}cm, net={net_gain_cm:.2f}cm)")

    elif beh == "group_locomotion":
        disps_cm: list[np.ndarray] = []
        paths_cm: list[float] = []
        for m in mids:
            pts = [c_map[fn][m] * cm_per_px for fn in frames if m in c_map[fn]]
            if len(pts) >= 2:
                arr = np.asarray(pts, float)
                disps_cm.append(arr[-1] - arr[0])
                paths_cm.append(float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()))
        if len(disps_cm) == len(mids) and len(disps_cm) >= 2:
            norms = [float(np.linalg.norm(d)) for d in disps_cm]
            units = [d / max(n, 1e-9) for d, n in zip(disps_cm, norms)]
            pair_cos = [
                float(np.dot(units[i], units[j]))
                for i in range(len(units))
                for j in range(i + 1, len(units))
            ]
            mean_cos = float(np.mean(pair_cos)) if pair_cos else 0.0
            min_net_cm = min(norms)
            balance = min(paths_cm) / max(max(paths_cm), 1e-9)
        else:
            mean_cos = 0.0
            min_net_cm = 0.0
            balance = 0.0
        cos_thresh = float(DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_direction_similarity)
        net_thresh = float(DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_net_displacement_cm)
        max_pair_d = float(DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_max_pair_distance_cm)
        sync_frames = 0
        if len(unique_mids) >= 3 and span > 0 and rows:
            centers_cm, obs_m = _clip_multi_track_series(rows, start, end, unique_mids, cm_per_px)
            _, vel_m, spd_m = _observed_run_velocity_cm_s(centers_cm, obs_m, fps, motion_window_s=0.3)
            active_spd = float(DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_active_speed_cm_s)
            for idx in range(span):
                if obs_m[idx].all() and (spd_m[idx] >= active_spd).all():
                    pts_i = centers_cm[idx]
                    pdist = float(np.linalg.norm(pts_i[:, None, :] - pts_i[None, :, :], axis=2).max())
                    if pdist < max_pair_d:
                        u_i = vel_m[idx] / np.maximum(spd_m[idx, :, None], 1e-9)
                        c_i = [
                            float(np.dot(u_i[a], u_i[b]))
                            for a in range(len(unique_mids))
                            for b in range(a + 1, len(unique_mids))
                        ]
                        if c_i and float(np.mean(c_i)) >= cos_thresh - 0.1:
                            sync_frames += 1
        sync_ratio = sync_frames / span
        metrics.update(
            {
                "mean_pairwise_cosine": round(mean_cos, 4),
                "min_net_displacement_cm": round(min_net_cm, 3),
                "path_balance_ratio": round(balance, 3),
                "sync_ratio": round(sync_ratio, 4),
                "cosine_threshold": round(cos_thresh, 3),
                "net_displacement_threshold_cm": round(net_thresh, 2),
            }
        )
        kinematic_ok = (
            mean_cos >= cos_thresh - 0.1
            and min_net_cm >= net_thresh * 0.7
            and balance >= 0.35
            and sync_ratio >= 0.15
        )
        if not kinematic_ok:
            reason_notes.append(
                f"协同一致性偏弱(cos={mean_cos:.2f}, min_disp={min_net_cm:.1f}cm, sync_ratio={sync_ratio:.2f})"
            )

    elif beh == "isolation":
        target_id = int(event.get("target_mouse_id", mids[0]))
        iso_thresh_cm = float(getattr(parameters, "isolation_distance_cm", 20.0))
        nn_dists_cm = []
        violations = 0
        for fn in frames:
            if target_id not in c_map[fn]:
                continue
            ct = c_map[fn][target_id]
            others = [float(np.linalg.norm(ct - c_other)) * cm_per_px for oid, c_other in c_map[fn].items() if oid != target_id]
            if others:
                mn = min(others)
                nn_dists_cm.append(mn)
                if mn < iso_thresh_cm * 0.75:
                    violations += 1
        min_nn = min(nn_dists_cm) if nn_dists_cm else float("inf")
        med_nn = float(np.median(nn_dists_cm)) if nn_dists_cm else float("inf")
        viol_ratio = violations / max(1, len(nn_dists_cm))
        metrics.update(
            {
                "min_nearest_neighbor_cm": None if not math.isfinite(min_nn) else round(min_nn, 2),
                "median_nearest_neighbor_cm": None if not math.isfinite(med_nn) else round(med_nn, 2),
                "close_violation_frames": int(violations),
            }
        )
        kinematic_ok = viol_ratio <= 0.15 and duration_ok
        if viol_ratio > 0.15:
            reason_notes.append(f"存在近邻接近违规帧({violations}帧)")

    if not min_participants_ok:
        reason_notes.append(f"参与目标数不足({len(unique_mids)}<{min_req_participants})")
    if not min_duration_ok:
        reason_notes.append("未通过最短时长门控")
    if max_dur_s is not None and duration_s > max_dur_s + 1e-6:
        reason_notes.append(f"超过最长时长上限({duration_s:.2f}s>{max_dur_s:.1f}s)")
    if not endpoint_obs:
        reason_notes.append("起止端点存在目标缺框")

    full_ok = bool(duration_bounds_ok and endpoint_obs and min_participants_ok and kinematic_ok)
    metrics["kinematic_ok"] = bool(kinematic_ok)
    metrics["full_crop_precision_ok"] = full_ok
    metrics["audit_notes"] = reason_notes if reason_notes else ["通过边界时长、端点可见性与运动学一致性检验"]
    return metrics


def _interval_iou(a_s: int, a_e: int, b_s: int, b_e: int) -> float:
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    if inter == 0:
        return 0.0
    union = max(1, (a_e - a_s) + (b_e - b_s) - inter)
    return inter / union


def _count_shared_mouse_overlaps(events: Sequence[Mapping[str, Any]]) -> tuple[int, dict[str, int]]:
    """Count temporally overlapping pairs sharing >=1 mouse within each canonical behavior."""
    by_beh: dict[str, list[tuple[int, int, set[int]]]] = defaultdict(list)
    for ev in events:
        beh = normalize_behavior_key(str(ev.get("canonical_behavior") or ev.get("behavior") or ""))
        if not beh:
            continue
        s = int(ev["start_frame"])
        e = int(ev["end_frame"])
        mids = set(event_participant_ids(ev))
        by_beh[beh].append((s, e, mids))

    counts: dict[str, int] = {}
    total = 0
    for beh, items in by_beh.items():
        items_sorted = sorted(items, key=lambda it: (it[0], it[1]))
        beh_overlaps = 0
        for i in range(len(items_sorted)):
            s_i, e_i, mids_i = items_sorted[i]
            if not mids_i:
                continue
            for j in range(i + 1, len(items_sorted)):
                s_j, _, mids_j = items_sorted[j]
                if s_j >= e_i:
                    break
                if mids_i & mids_j:
                    beh_overlaps += 1
        counts[beh] = beh_overlaps
        total += beh_overlaps
    return total, counts


def summarize_group_cropping_validation(
    pre_events: Sequence[Mapping[str, Any]],
    post_events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    scale: Mapping[str, Any],
    parameters: HeuristicParameters,
    *,
    historical_reviews: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compare raw rule group events vs fine-cropped segments and compute precision stats."""
    fps = float(binding["source_video"]["fps"])
    pre_by_id = {str(e["event_id"]): dict(e) for e in pre_events if e.get("canonical_behavior") in GROUP_BEHAVIOR_KEYS}
    pre_by_beh: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in pre_events:
        if e.get("canonical_behavior") in GROUP_BEHAVIOR_KEYS:
            pre_by_beh[str(e["canonical_behavior"])].append(dict(e))

    post_by_beh: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in post_events:
        if e.get("canonical_behavior") in GROUP_BEHAVIOR_KEYS:
            post_by_beh[str(e["canonical_behavior"])].append(dict(e))

    all_overlaps, overlaps_by_beh = _count_shared_mouse_overlaps(post_events)
    per_event_audit: dict[str, dict[str, Any]] = {}
    behavior_summaries: dict[str, dict[str, Any]] = {}

    for beh in GROUP_BEHAVIOR_KEYS:
        raw_list = pre_by_beh.get(beh, [])
        crop_list = post_by_beh.get(beh, [])

        parent_to_children: dict[str, list[dict[str, Any]]] = defaultdict(list)
        start_trims_s: list[float] = []
        end_trims_s: list[float] = []
        changed_count = 0
        unchanged_count = 0

        for ev in crop_list:
            eid = str(ev["event_id"])
            kin = audit_group_segment_kinematics(ev, store, scale, fps, parameters)
            pid = str(ev.get("parent_candidate_event_id", ""))
            parent = pre_by_id.get(pid)
            if parent is None:
                mids_t = tuple(ev.get("mouse_ids", ()))
                overlapping = [
                    r
                    for r in raw_list
                    if tuple(r.get("mouse_ids", ())) == mids_t
                    and max(int(r["start_frame"]), int(ev["start_frame"]))
                    < min(int(r["end_frame"]), int(ev["end_frame"]))
                ]
                if overlapping:
                    parent = max(
                        overlapping,
                        key=lambda r: _interval_iou(
                            int(r["start_frame"]), int(r["end_frame"]), int(ev["start_frame"]), int(ev["end_frame"])
                        ),
                    )
            if parent is not None:
                parent_to_children[str(parent["event_id"])].append(ev)
                ds = (int(ev["start_frame"]) - int(parent["start_frame"])) / fps
                de = (int(parent["end_frame"]) - int(ev["end_frame"])) / fps
                start_trims_s.append(ds)
                end_trims_s.append(de)
                changed = (int(ev["start_frame"]) != int(parent["start_frame"])) or (
                    int(ev["end_frame"]) != int(parent["end_frame"])
                )
                if changed:
                    changed_count += 1
                else:
                    unchanged_count += 1
                kin["raw_parent_event_id"] = str(parent["event_id"])
                kin["raw_start_frame"] = int(parent["start_frame"])
                kin["raw_end_frame"] = int(parent["end_frame"])
                kin["raw_duration_s"] = round((int(parent["end_frame"]) - int(parent["start_frame"])) / fps, 3)
                kin["boundary_delta"] = {
                    "changed": changed,
                    "start_shift_s": round(ds, 3),
                    "end_trim_s": round(de, 3),
                    "classification": "trimmed_or_shifted" if changed else "unchanged",
                }
            else:
                kin["boundary_delta"] = {
                    "changed": False,
                    "start_shift_s": 0.0,
                    "end_trim_s": 0.0,
                    "classification": "unmatched",
                }
            per_event_audit[eid] = kin

        split_parents = sum(1 for ch in parent_to_children.values() if len(ch) > 1)
        rejected_raw = sum(1 for r in raw_list if str(r["event_id"]) not in parent_to_children)

        n_post = len(crop_list)
        dur_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["duration_ok"])
        dur_bounds_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["duration_bounds_ok"])
        min_part_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["min_participants_ok"])
        end_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["endpoint_observed_ok"])
        kin_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["kinematic_ok"])
        full_ok_cnt = sum(1 for e in crop_list if per_event_audit[str(e["event_id"])]["full_crop_precision_ok"])

        raw_durs = [(int(e["end_frame"]) - int(e["start_frame"])) / fps for e in raw_list]
        post_durs = [(int(e["end_frame"]) - int(e["start_frame"])) / fps for e in crop_list]

        summary_row: dict[str, Any] = {
            "behavior": beh,
            "label": GROUP_BEHAVIOR_LABELS[beh],
            "raw_candidate_count": len(raw_list),
            "post_crop_count": n_post,
            "changed_bounds_count": changed_count,
            "unchanged_count": unchanged_count,
            "split_parent_count": split_parents,
            "rejected_by_fine_crop_count": rejected_raw,
            "mean_raw_duration_s": round(float(np.mean(raw_durs)), 3) if raw_durs else 0.0,
            "mean_post_duration_s": round(float(np.mean(post_durs)), 3) if post_durs else 0.0,
            "max_post_duration_s": round(float(max(post_durs)), 3) if post_durs else 0.0,
            "mean_start_shift_s": round(float(np.mean(start_trims_s)), 3) if start_trims_s else 0.0,
            "mean_end_trim_s": round(float(np.mean(end_trims_s)), 3) if end_trims_s else 0.0,
            "min_participants_pass_count": min_part_cnt,
            "duration_bounds_pass_count": dur_bounds_cnt,
            "duration_pass_count": dur_ok_cnt,
            "endpoint_observed_count": end_ok_cnt,
            "kinematic_pass_count": kin_ok_cnt,
            "full_crop_precision_count": full_ok_cnt,
            "shared_mouse_overlap_count": int(overlaps_by_beh.get(beh, 0)),
            "min_participants_pass_rate": round(min_part_cnt / n_post, 4) if n_post else None,
            "duration_bounds_pass_rate": round(dur_bounds_cnt / n_post, 4) if n_post else None,
            "duration_pass_rate": round(dur_ok_cnt / n_post, 4) if n_post else None,
            "endpoint_observed_rate": round(end_ok_cnt / n_post, 4) if n_post else None,
            "kinematic_pass_rate": round(kin_ok_cnt / n_post, 4) if n_post else None,
            "full_crop_precision_rate": round(full_ok_cnt / n_post, 4) if n_post else None,
        }

        if historical_reviews:
            hist_beh = [h for h in historical_reviews if h.get("behavior") == beh]
            if hist_beh:
                hist_pass = [h for h in hist_beh if h["status"] == "pass"]
                hist_fail = [h for h in hist_beh if h["status"] == "fail"]
                retained_pass = sum(
                    1
                    for h in hist_pass
                    if any(
                        _interval_iou(int(h["start_frame"]), int(h["end_frame"]), int(e["start_frame"]), int(e["end_frame"]))
                        >= 0.25
                        for e in crop_list
                    )
                )
                eliminated_fail = sum(
                    1
                    for h in hist_fail
                    if not any(
                        _interval_iou(int(h["start_frame"]), int(h["end_frame"]), int(e["start_frame"]), int(e["end_frame"]))
                        >= 0.50
                        for e in crop_list
                    )
                )
                summary_row["historical_alignment"] = {
                    "historical_pass_total": len(hist_pass),
                    "retained_pass_count": retained_pass,
                    "historical_fail_total": len(hist_fail),
                    "eliminated_or_recropped_fail_count": eliminated_fail,
                }
        behavior_summaries[beh] = summary_row

    total_raw = sum(v["raw_candidate_count"] for v in behavior_summaries.values())
    total_post = sum(v["post_crop_count"] for v in behavior_summaries.values())
    total_min_part = sum(v["min_participants_pass_count"] for v in behavior_summaries.values())
    total_dur_bounds = sum(v["duration_bounds_pass_count"] for v in behavior_summaries.values())
    total_dur_ok = sum(v["duration_pass_count"] for v in behavior_summaries.values())
    total_end_ok = sum(v["endpoint_observed_count"] for v in behavior_summaries.values())
    total_kin_ok = sum(v["kinematic_pass_count"] for v in behavior_summaries.values())
    total_full_ok = sum(v["full_crop_precision_count"] for v in behavior_summaries.values())
    group_overlaps = sum(v["shared_mouse_overlap_count"] for v in behavior_summaries.values())

    aggregate = {
        "total_raw_group_candidates": total_raw,
        "total_post_crop_group_segments": total_post,
        "total_rejected_by_fine_crop": sum(v["rejected_by_fine_crop_count"] for v in behavior_summaries.values()),
        "total_boundary_adjusted": sum(v["changed_bounds_count"] for v in behavior_summaries.values()),
        "min_participants_pass_rate": round(total_min_part / total_post, 4) if total_post else None,
        "duration_bounds_pass_rate": round(total_dur_bounds / total_post, 4) if total_post else None,
        "duration_pass_rate": round(total_dur_ok / total_post, 4) if total_post else None,
        "endpoint_observed_rate": round(total_end_ok / total_post, 4) if total_post else None,
        "kinematic_pass_rate": round(total_kin_ok / total_post, 4) if total_post else None,
        "full_crop_precision_rate": round(total_full_ok / total_post, 4) if total_post else None,
        "group_shared_mouse_overlap_count": int(group_overlaps),
        "all_behaviors_shared_mouse_overlap_count": int(all_overlaps),
        "shared_mouse_overlap_by_behavior": {k: v for k, v in sorted(overlaps_by_beh.items()) if v > 0},
        "zero_shared_mouse_overlap_ok": (all_overlaps == 0),
    }

    return {"aggregate": aggregate, "by_behavior": behavior_summaries}, per_event_audit


def select_group_review_clips(
    post_events: Sequence[Mapping[str, Any]],
    store: Any,
    binding: Mapping[str, Any],
    *,
    per_event_audit: Mapping[str, Mapping[str, Any]] | None = None,
    max_per_behavior: int | None = None,
) -> list[dict[str, Any]]:
    """Select diverse endpoint-observed, non-overlapping group segments for MP4 clip export."""
    fps = float(binding["source_video"]["fps"])
    selected: list[dict[str, Any]] = []
    for beh in GROUP_BEHAVIOR_KEYS:
        min_req = min_participants_for_behavior(beh)
        candidates = [
            copy.deepcopy(dict(e))
            for e in post_events
            if e.get("canonical_behavior") == beh and len(event_participant_ids(e)) >= min_req
        ]
        eligible = [e for e in candidates if endpoints_fully_observed(store, e, binding.get("source_video"))]
        pool = eligible if eligible else candidates

        def _sort_key(e: Mapping[str, Any]) -> tuple[int, int, float, int]:
            eid = str(e["event_id"])
            audit = per_event_audit.get(eid, {}) if per_event_audit else {}
            changed = 1 if audit.get("boundary_delta", {}).get("changed") else 0
            full_ok = 1 if audit.get("full_crop_precision_ok") else 0
            dur = (int(e["end_frame"]) - int(e["start_frame"])) / fps
            return (-full_ok, -changed, -dur, int(e["start_frame"]))

        ranked = sorted(pool, key=_sort_key)
        chosen: list[dict[str, Any]] = []
        used_buckets: set[int] = set()
        limit = len(ranked) if (max_per_behavior is None or max_per_behavior <= 0) else int(max_per_behavior)

        def _overlaps_shared_mouse(ev: Mapping[str, Any]) -> bool:
            mids_set = set(event_participant_ids(ev))
            s_e, e_e = int(ev["start_frame"]), int(ev["end_frame"])
            for old in chosen:
                if not (mids_set & set(event_participant_ids(old))):
                    continue
                if not (e_e <= int(old["start_frame"]) or s_e >= int(old["end_frame"])):
                    return True
            return False

        passes = (True, False) if len(ranked) > limit else (False,)
        for diversify in passes:
            for ev in ranked:
                if ev in chosen or _overlaps_shared_mouse(ev):
                    continue
                bucket = int(int(ev["start_frame"]) / fps / 45.0)
                if diversify and bucket in used_buckets:
                    continue
                chosen.append(ev)
                used_buckets.add(bucket)
                if len(chosen) >= limit:
                    break
            if len(chosen) >= limit:
                break
        selected.extend(sorted(chosen, key=lambda e: int(e["start_frame"])))
    return selected
