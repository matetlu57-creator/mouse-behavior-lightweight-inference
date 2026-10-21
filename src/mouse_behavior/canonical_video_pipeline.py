"""Bounded-memory canonical rule inference for a complete tracked video.

This is a separate route from the lightweight social FSM. Visual IDs come only
from the cache, and every published interval is an unconfirmed candidate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from mouse_behavior.behavior.individual_rules import DEFAULT_INDIVIDUAL_CONFIG, individual_rule_masks
from mouse_behavior.behavior.duration_policy import qualify_event_duration
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks, policy_receipt
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.exported_dataset_models import LABEL_TO_CANONICAL
from mouse_behavior.evaluation.group_event_rules import group_rule_candidates, isolation_candidates
from mouse_behavior.evaluation.segment_export import _sha256, _write_json
from mouse_behavior.preprocessing.scale_resolution import robust_reference_body_px
from mouse_behavior.tracking.cache import _iter_cache_records

LOGGER = logging.getLogger(__name__)
DISPLAY_LABELS = {value: key for key, value in LABEL_TO_CANONICAL.items()}
ENABLED_BEHAVIORS = tuple(value for value in DISPLAY_LABELS if value not in {"grooming", "rearing"})
PAIR_LABELS = {"approach": "approach", "chasing": "chase", "attack": "attack",
               "together": "together", "head_contact": "nose_head_contact",
               "tail_contact": "nose_tail_contact"}


def video_info(video: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {video}")
        result = dict(path=str(video.resolve()), bytes=video.stat().st_size,
                      frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                      fps=float(capture.get(cv2.CAP_PROP_FPS)),
                      width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if result["frame_count"] <= 0 or not math.isfinite(result["fps"]) or result["fps"] <= 0:
        raise ValueError("source video has invalid frame count/FPS")
    return result


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _export_detection(payload: dict) -> dict:
    """Preserve observed IDs; convert internal left/right ears to export order."""
    track = payload.get("visual_track_id")
    if not isinstance(track, (int, np.integer)) or track < 0:
        raise ValueError("cache detection lacks a nonnegative tracker-owned visual_track_id")
    points = np.asarray(payload.get("keypoints_px"), float)
    confidence = np.asarray(payload.get("keypoint_conf"), float)
    box = np.asarray(payload.get("bbox_xyxy"), float)
    if points.shape != (7, 2) or confidence.shape != (7,) or box.shape != (4,):
        raise ValueError("cache detection has invalid seven-point pose/box schema")
    keypoints = [[_finite(points[index, 0]), _finite(points[index, 1]), _finite(confidence[index])]
                 for index in (0, 2, 1, 3, 4, 5, 6)]
    return dict(track_id=int(track), box=[_finite(v) for v in box], keypoints=keypoints,
                detection_confidence=_finite(payload.get("box_conf", 0)),
                class_id=int(payload.get("class_id", 0)))


class TrackStore:
    """SQLite is a bounded intermediate, not an identity or tracking layer."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def rows(self, start: int, end: int, *, rebase: bool = False) -> Iterator[dict]:
        connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            for (payload,) in connection.execute(
                "SELECT payload FROM frames WHERE frame>=? AND frame<? ORDER BY frame", (start, end)
            ):
                row = json.loads(payload)
                if rebase:
                    row["frame"] -= start
                    row["time"] = row["frame"] / row.pop("source_fps")
                else:
                    row.pop("source_fps", None)
                yield row
        finally:
            connection.close()

    def window(self, start: int, end: int) -> list[dict]:
        return list(self.rows(start, end, rebase=True))

    def write_tracks(self, path: Path, start: int, end: int) -> None:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("[\n")
            for index, row in enumerate(self.rows(start, end, rebase=True)):
                if index:
                    handle.write(",\n")
                json.dump(row, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n]\n")


def _source_binding(video: Path, status: dict, receipt_path: Path | None,
                    expected_sha256: str | None) -> tuple[dict, dict]:
    info = video_info(video)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path else {}
    supplied = receipt.get("source_video", {})
    digest = expected_sha256 or supplied.get("sha256") or status.get("source_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("cache needs an explicit source SHA-256 or source receipt")
    actual = _sha256(video)
    if actual.lower() != digest.lower():
        raise ValueError("source SHA-256 does not match the authorized cache binding")
    info["sha256"] = actual
    if expected_sha256 and supplied.get("sha256") and supplied["sha256"].lower() != actual:
        raise ValueError("source receipt SHA-256 conflicts with explicit binding")
    if Path(status.get("video", "")).resolve() != video.resolve():
        raise ValueError("cache status belongs to another source video path")
    if supplied:
        if Path(supplied.get("path", "")).resolve() != video.resolve():
            raise ValueError("source receipt belongs to another video path")
        for field in ("bytes", "frame_count", "width", "height"):
            if field in supplied and int(supplied[field]) != info[field]:
                raise ValueError(f"source receipt {field} differs from source video")
        if "fps" in supplied and not math.isclose(float(supplied["fps"]), info["fps"], rel_tol=1e-6):
            raise ValueError("source receipt FPS differs from source video")
    return info, receipt


def bind_cache(video: Path, cache: Path, database: Path, *, source_receipt: Path | None = None,
               expected_source_sha256: str | None = None) -> tuple[dict, dict]:
    """Validate source, completion and every frame before accepting a pose cache."""
    if database.exists():
        raise FileExistsError(f"refusing to overwrite tracks database: {database}")
    status_path = cache / "yolo_results_status.json"
    status_hash = _sha256(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    source, supplied_receipt = _source_binding(video, status, source_receipt, expected_source_sha256)
    frames, fps = source["frame_count"], source["fps"]
    if (status.get("status") != "complete" or status.get("next_frame") != frames
            or status.get("total_frames") != frames):
        raise ValueError("pose cache is incomplete or source frame count differs")
    if not math.isclose(float(status.get("fps", 0)), fps, rel_tol=1e-6):
        raise ValueError("cache FPS differs from source video")
    if status.get("trajectory_schema_version", 0) < 1 or status.get("tracking_backend") != "ultralytics":
        raise ValueError("cache does not declare the supported tracker-owned trajectory schema")
    model = supplied_receipt.get("model")
    if (not isinstance(model, dict) or not isinstance(model.get("path"), str)
            or not model["path"].strip() or not isinstance(model.get("sha256"), str)
            or len(model["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in model["sha256"].lower())):
        raise ValueError("complete model binding path/SHA-256 is required in source_receipt")
    model_path = Path(model["path"])
    if not model_path.is_file():
        raise ValueError("model binding path must reference an existing file")
    if Path(status.get("model", "")).resolve() != model_path.resolve():
        raise ValueError("cache model path differs from source receipt")
    if _sha256(model_path) != model["sha256"].lower():
        raise ValueError("cache model SHA-256 differs from source receipt")
    chunks = sorted(cache.glob("yolo_results.*.*.pkl.gz"))
    if not chunks or len(chunks) != status.get("chunk_count"):
        raise ValueError("cache chunk count disagrees with completion status")
    chunk_hashes = {part.name: _sha256(part) for part in chunks}
    ids: set[int] = set()
    maximum_visible = 0
    detection_counts = []
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE frames(frame INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE lengths(track INTEGER, pixels REAL)")
        count = 0
        for record in _iter_cache_records(cache):
            if type(record.get("frame")) is not int or record["frame"] != count:
                raise ValueError("cache frame indices must be continuous from zero without duplicates")
            if count >= frames:
                raise ValueError("cache extends beyond the source video")
            detections = [_export_detection(payload) for payload in record.get("pose_detections", [])]
            visible = [row["track_id"] for row in detections]
            if len(set(visible)) != len(visible):
                raise ValueError("duplicate visual ID inside one cache frame")
            ids.update(visible)
            maximum_visible = max(maximum_visible, len(visible))
            detection_counts.append(len(visible))
            row = dict(frame=count, time=count / fps, source_frame=count, source_time=count / fps,
                       source_fps=fps, detections=detections)
            connection.execute("INSERT INTO frames VALUES(?,?)", (count, json.dumps(row, allow_nan=False)))
            for detection in detections:
                ends = np.asarray(detection["keypoints"], float)[[0, 6]]
                if np.isfinite(ends).all() and ends[:, 2].min() >= .25:
                    length = float(np.linalg.norm(ends[0, :2] - ends[1, :2]))
                    if 10 <= length <= 300:
                        connection.execute("INSERT INTO lengths VALUES(?,?)", (detection["track_id"], length))
            count += 1
        if count != frames:
            raise ValueError(f"cache frame indices are not continuous through frame {frames - 1}")
        connection.execute("CREATE INDEX length_track ON lengths(track)")
        medians = {}
        samples = {}
        for track in sorted(ids):
            values = [v for (v,) in connection.execute("SELECT pixels FROM lengths WHERE track=?", (track,))]
            if values:
                medians[str(track)] = float(np.median(values))
                samples[str(track)] = len(values)
        if not medians:
            raise ValueError("source cache has no measurable nose-to-tail body length")
        connection.commit()
    finally:
        connection.close()
    if _sha256(status_path) != status_hash or any(_sha256(cache / name) != digest for name, digest in chunk_hashes.items()):
        raise ValueError("pose cache changed during binding")
    reference = robust_reference_body_px(list(medians.values()))
    warnings = ["Body-length scale is a video-wide estimate; perspective and posture remain sources of error."]
    if len(ids) > maximum_visible:
        warnings.append("Visual ID fragmentation is possible; visual IDs are not biological identities.")
    binding = dict(schema_version="1.0", source_video=source, cache_directory=str(cache.resolve()),
                   cache_status=status, cache_status_sha256=status_hash, chunk_sha256=chunk_hashes,
                   source_receipt_sha256=_sha256(source_receipt) if source_receipt else None,
                   model=model, visual_track_ids=sorted(ids), global_visual_track_count=len(ids),
                   detection_count_quantiles={str(q): float(np.quantile(detection_counts, q)) for q in (0, .25, .5, .75, 1)},
                   maximum_simultaneous_detections=maximum_visible, track_truncation=False,
                   database_sha256=_sha256(database), all_frames_continuous=True)
    scale = dict(mode="fixed_video_body_length", reference_body_cm=7.918,
                 reference_body_px=reference, cm_per_pixel=7.918 / reference,
                 per_track_median_pixels=medians, per_track_measurable_frames=samples,
                 estimator="robust_reference_body_px(per-ID whole-video nose-tail medians)",
                 confidence_minimum=.25, measurable_pixel_range=[10, 300],
                 source_sha256=source["sha256"], warnings=warnings)
    return binding, scale


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _event(behavior: str, ids, start: int, end: int, actor: int = -1, target: int = -1) -> dict:
    participants = []
    if actor >= 0 and target >= 0:
        role_names = ("Approaching", "Approached") if behavior == "approach" else ("Chasing", "Chased")
        participants = [dict(role_name=name, role_key=name.lower(), track_ids=[track])
                        for name, track in zip(role_names, (actor, target))]
    return dict(behavior=DISPLAY_LABELS[behavior], canonical_behavior=behavior,
                mouse_ids=sorted(map(int, ids)), actor_id=actor, target_id=target,
                participants=participants, start_frame=start, end_frame=end,
                interval_semantics="source_frames_half_open", confidence="heuristic_candidate",
                screening_status="heuristic_candidate", requires_review=True)


def _evidence(event: dict, observed: np.ndarray, role_support: np.ndarray | None = None) -> dict:
    start, end = event["start_frame"], event["end_frame"]
    bounded = np.zeros(len(observed), bool)
    bounded[start:end] = True
    event["observed_all_participants_frame_intervals"] = _runs(observed & bounded)
    event["observed_all_participants_frame_count"] = int((observed & bounded).sum())
    event["inferred_or_partially_unobserved_frame_count"] = int((~observed & bounded).sum())
    if role_support is not None:
        event["role_status"] = "event_direction_with_measured_support" if event["actor_id"] >= 0 else "unknown"
        event["role_support_frame_intervals"] = _runs(role_support & bounded)
        event["role_unknown_frame_intervals"] = _runs(~role_support & bounded)
        event["role_support_frame_count"] = int((role_support & bounded).sum())
    return event


def _directed_events(mask: np.ndarray, actors: np.ndarray, targets: np.ndarray, ids: list,
                     canonical: str, observed: np.ndarray, *, fps: float) -> list[dict]:
    result = []
    for start, end in _runs(mask):
        known = [(frame, int(actors[frame]), int(targets[frame])) for frame in range(start, end)
                 if actors[frame] in ids and targets[frame] in ids and actors[frame] != targets[frame]]
        if not known:
            result.append(_evidence(_event(canonical, ids, start, end), observed, np.zeros(len(mask), bool)))
            continue
        # Unknown rows remain inside the process; only a measured direction
        # reversal starts another event. The raw support ledger is never filled.
        begin, actor, target = start, known[0][1], known[0][2]
        segments = []
        for frame, a, b in known[1:]:
            if (a, b) != (actor, target):
                segments.append((begin, frame, actor, target))
                begin, actor, target = frame, a, b
        segments.append((begin, end, actor, target))
        for begin, stop, actor, target in segments:
            support = (actors == actor) & (targets == target) & observed
            result.append(_evidence(_event(canonical, ids, begin, stop, actor, target), observed, support))
    return [event for event in result if qualify_event_duration(event, fps)["ok"]]


def _individual_config(params):
    return replace(DEFAULT_INDIVIDUAL_CONFIG,
                   motion_speed_threshold_cm_s=params.individual_motion_speed_threshold_cm_s,
                   static_min_duration_s=params.stationary_min_duration_s,
                   locomotion_min_duration_s=params.locomotion_min_duration_s,
                   static_max_radius_cm=params.stationary_max_center_radius_cm,
                   static_transient_max_s=params.stationary_transient_max_duration_s,
                   static_transient_max_displacement_cm=params.stationary_transient_max_displacement_cm,
                   locomotion_pause_max_s=params.locomotion_stop_gap_s, enable_posture_screen=False)


def infer_window_events(rows: list[dict], fps: float, scale: dict, params, *,
                        posture_mode: str = "disabled") -> tuple[list[dict], dict]:
    """Run the canonical three domains with every observed visual ID retained."""
    if posture_mode not in {"disabled", "v3_review"}:
        raise ValueError("posture_mode must be disabled or v3_review")
    track_ids = sorted({d["track_id"] for row in rows for d in row["detections"]})
    audit = dict(track_ids=track_ids, track_truncation=False, frame_count=len(rows))
    if not track_ids:
        return [], audit
    fixed = dict(mode="fixed", cm_per_pixel=scale["cm_per_pixel"],
                 assumed_mouse_body_length_cm=scale["reference_body_cm"])
    with tempfile.TemporaryDirectory(prefix="canonical_features_") as temporary:
        tracks = Path(temporary) / "tracks.json"
        _write_json(tracks, rows)
        features = load_exported_features(tracks, fps=fps, max_tracks=max(2, len(track_ids)), scale_config=fixed)
    if features.track_selection_truncated or list(map(int, features.track_ids)) != track_ids:
        raise ValueError("canonical feature loading truncated visual IDs")
    result = []
    individual = individual_rule_masks(
        features.centers_cm, features.valid, fps, keypoints_cm=features.collective_keypoints_cm,
        boxes_cm=features.collective_boxes_cm, reference_body_cm=features.reference_body_cm,
        config=_individual_config(params)).masks
    for name in ("stationary", "locomotion"):
        for slot, track in enumerate(features.track_ids):
            for start, end in _runs(individual[name][:, slot]):
                result.append(_evidence(_event(name, [track], start, end), features.valid[:, slot]))
    roles: dict[str, np.ndarray] = {}
    social = pair_rule_masks(features, params, roles=roles)
    for name, canonical in PAIR_LABELS.items():
        for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
            ids = [features.track_ids[left], features.track_ids[right]]
            observed = features.valid[:, left] & features.valid[:, right]
            mask = np.asarray(social[name][:, edge], bool)
            if name in {"approach", "chasing"}:
                actor = np.asarray(roles[name + "_actor"][:, edge], dtype=int)
                target = np.asarray(roles[name + "_target"][:, edge], dtype=int)
                result.extend(_directed_events(mask, actor, target, ids, canonical, observed, fps=fps))
            else:
                for start, end in _runs(mask):
                    result.append(_evidence(_event(canonical, ids, start, end), observed))
    candidates = (*group_rule_candidates(features, params, ()), *isolation_candidates(features, params))
    slots = {int(track): slot for slot, track in enumerate(features.track_ids)}
    for candidate in candidates:
        if candidate.frame_intervals is None:
            raise ValueError("group rule returned score-only evidence without event frames")
        observed = features.valid[:, [slots[int(track)] for track in candidate.target_ids]].all(axis=1)
        for start, end in candidate.frame_intervals:
            event = _evidence(_event(candidate.behavior, candidate.target_ids, start, end), observed)
            if candidate.support_frame_intervals is not None:
                event["canonical_support_frame_intervals"] = [
                    (max(start, a), min(end, b)) for a, b in candidate.support_frame_intervals
                    if max(start, a) < min(end, b)]
            result.append(event)
    if posture_mode == "v3_review":
        from mouse_behavior.behavior.posture_adapter import infer_posture_rows
        posture_events, posture_audit = infer_posture_rows(rows, fps, track_ids=track_ids)
        for event in posture_events:
            slot = slots[event["mouse_ids"][0]]
            result.append(_evidence(event, features.valid[:, slot]))
        audit["posture_review"] = posture_audit
    qualified = []
    audit["duration_rejected"] = []
    for event in result:
        qualification = qualify_event_duration(event, fps, parameters=params)
        if qualification["ok"]:
            qualified.append(event)
        else:
            audit["duration_rejected"].append(dict(event=event, qualification=qualification))
    return qualified, audit


def merge_events(events: list[dict]) -> list[dict]:
    """Join touching support only for the identical behavior, IDs and roles."""
    groups = defaultdict(list)
    for event in events:
        key = (event["canonical_behavior"], tuple(event["mouse_ids"]), event.get("actor_id", -1),
               event.get("target_id", -1), json.dumps(event.get("participants", []), sort_keys=True))
        groups[key].append(dict(event))
    merged = []
    for rows in groups.values():
        current = None
        for row in sorted(rows, key=lambda value: (value["start_frame"], value["end_frame"])):
            if current is not None and row["start_frame"] <= current["end_frame"]:
                current["end_frame"] = max(current["end_frame"], row["end_frame"])
                for field, value in row.items():
                    if field.endswith("_frame_intervals"):
                        current[field] = sorted(current.get(field, []) + value)
            else:
                current = row
                merged.append(current)
    merged.sort(key=lambda row: (row["start_frame"], row["end_frame"], row["canonical_behavior"], row["mouse_ids"]))
    for row in merged:
        for field in list(row):
            if field.endswith("_frame_intervals"):
                intervals = []
                for begin, end in sorted(row[field]):
                    if intervals and begin <= intervals[-1][1]:
                        intervals[-1][1] = max(end, intervals[-1][1])
                    else:
                        intervals.append([begin, end])
                row[field] = intervals
                row[field.replace("_intervals", "_count")] = sum(b - a for a, b in intervals)
        if "observed_all_participants_frame_count" in row:
            row["inferred_or_partially_unobserved_frame_count"] = row["end_frame"] - row["start_frame"] - row["observed_all_participants_frame_count"]
        identity = json.dumps(row, sort_keys=True, allow_nan=False).encode()
        row["event_id"] = f"{row['canonical_behavior']}_{row['start_frame']:08d}_{row['end_frame']:08d}_{hashlib.sha256(identity).hexdigest()[:10]}"
    return merged


def _infer_core_task(spec: tuple) -> tuple[list[dict], dict]:
    database, fps, scale, params, begin, end, window_start, window_end, posture_mode = spec
    rows = TrackStore(Path(database)).window(window_start, window_end)
    selected, audit = infer_window_events(rows, fps, scale, params, posture_mode=posture_mode)
    audit.update(window_start=window_start, window_end=window_end, core_start=begin, core_end=end)
    events = []
    for event in selected:
        start = max(begin, event["start_frame"] + window_start)
        stop = min(end, event["end_frame"] + window_start)
        if start < stop:
            cropped = {**event, "start_frame": start, "end_frame": stop}
            for field, value in event.items():
                if field.endswith("_frame_intervals"):
                    cropped[field] = [(max(start, a + window_start), min(stop, b + window_start))
                                      for a, b in value if max(start, a + window_start) < min(stop, b + window_start)]
            if not qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                # A complete qualified parent is real inferred evidence. Retain
                # it instead of padding its shortened core fragment.
                cropped = {**event, "start_frame": event["start_frame"] + window_start,
                           "end_frame": event["end_frame"] + window_start}
                for field, value in event.items():
                    if field.endswith("_frame_intervals"):
                        cropped[field] = [(a + window_start, b + window_start) for a, b in value]
                cropped["retained_complete_parent_for_duration"] = True
            if qualify_event_duration(cropped, fps, parameters=params)["ok"]:
                events.append(cropped)
    return events, audit


def infer_video_events(store: TrackStore, binding: dict, scale: dict, params, *,
                       core_seconds: float = 20, context_seconds: float = 10,
                       workers: int = 1, posture_mode: str = "disabled") -> tuple[list[dict], dict]:
    fps, total = binding["source_video"]["fps"], binding["source_video"]["frame_count"]
    if not math.isfinite(core_seconds) or core_seconds <= 0 or not math.isfinite(context_seconds) or context_seconds < 10:
        raise ValueError("core seconds must be positive and context must be at least 10 seconds")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    core, context = max(1, round(core_seconds * fps)), math.ceil(context_seconds * fps)
    specs = []
    for begin in range(0, total, core):
        end = min(total, begin + core)
        window_start, window_end = max(0, begin - context), min(total, end + context)
        specs.append((str(store.path.resolve()), fps, scale, params, begin, end, window_start, window_end, posture_mode))
    # Only small immutable task descriptions cross process boundaries. Each
    # child reads its own bounded context from SQLite in read-only mode.
    completed = []

    def collect(result):
        completed.append(result)
        audit = result[1]
        LOGGER.info("[canonical rules] finished core %d..%d/%d, window IDs %d (%d/%d cores)",
                    audit["core_start"], audit["core_end"], total, len(audit["track_ids"]), len(completed), len(specs))

    if workers == 1:
        for spec in specs:
            collect(_infer_core_task(spec))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(_infer_core_task, spec) for spec in specs]
            for future in as_completed(futures):
                collect(future.result())
    completed.sort(key=lambda value: value[1]["core_start"])
    events, audits = [], []
    for selected, audit in completed:
        events.extend(selected)
        audits.append(audit)
    posture_resolution = {}
    if posture_mode == "v3_review":
        from mouse_behavior.behavior.posture_adapter import resolve_posture_window_audits
        posture_events, posture_resolution = resolve_posture_window_audits(audits, fps)
        events = [event for event in events if event.get("posture_engine") != "v3_decoupled"] + posture_events
    merged = merge_events(events)
    for event in merged:
        if not qualify_event_duration(event, fps, parameters=params)["ok"]:
            raise ValueError("merged event lost standalone duration qualification")
    return merged, dict(windows=audits, core_seconds=core_seconds, context_seconds=context_seconds,
                                     global_posture_resolution=posture_resolution,
                                     workers=workers, track_truncation=False, boundary_approximation=True,
                                     boundary_note="Bounded context can change events requiring history beyond the context. A complete qualified parent may extend beyond the core when its shortened child fails duration qualification; no frames are fabricated. Measured support remains separate.")


def run_pipeline(video: Path, cache: Path | None, output: Path, config: Path, *,
                 mode: str = "all", source_receipt: Path | None = None,
                 expected_source_sha256: str | None = None, events_from: Path | None = None,
                 core_seconds: float = 20, context_seconds: float = 10, workers: int = 1,
                 clip_workers: int = 1, clip_preset: str = "veryfast", posture_mode: str = "disabled") -> dict:
    """Create a new independently inspectable run; never overwrite a prior run."""
    from mouse_behavior.visualization.behavior_packages import export_packages, render_full_package, reuse_full_package, write_manifests
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    if mode not in {"all", "infer", "full", "clips"}:
        raise ValueError(f"unknown pipeline mode: {mode}")
    if type(clip_workers) is not int or clip_workers < 1:
        raise ValueError("clip workers must be a positive integer")
    if clip_preset not in ("veryfast", "ultrafast"):
        raise ValueError("clip preset must be veryfast or ultrafast")
    if posture_mode not in {"disabled", "v3_review"}:
        raise ValueError("posture_mode must be disabled or v3_review")
    output.mkdir(parents=True)
    receipt: dict[str, Any] = dict(status="running", mode=mode, requires_review=True,
                                  route="canonical confirmed_rules + individual_rules + group_event_rules",
                                  enabled_behaviors=list(ENABLED_BEHAVIORS), disabled_behaviors=["grooming", "rearing"])
    receipt.update(posture_mode=posture_mode, posture_review_behaviors=["grooming", "rearing"] if posture_mode == "v3_review" else [])
    _write_json(output / "run_receipt.json", receipt)
    try:
        prior = None
        if events_from:
            prior = json.loads((events_from / "run_receipt.json").read_text(encoding="utf-8"))
            if prior.get("status") != "complete":
                raise ValueError("render-only input run is not complete")
            if prior.get("mode") not in {"all", "infer", "full", "clips"}:
                raise ValueError("render-only input run has unknown mode")
            binding = json.loads((events_from / "input_binding.json").read_text(encoding="utf-8"))
            scale = json.loads((events_from / "scale_receipt.json").read_text(encoding="utf-8"))
            if _sha256(video) != binding["source_video"]["sha256"]:
                raise ValueError("render-only source SHA-256 mismatch")
            ledger = prior.get("payload_sha256", {})
            for name in ("events.json", "tracks.sqlite", "scale_receipt.json", "input_binding.json"):
                if name not in ledger or _sha256(events_from / name) != ledger[name]:
                    raise ValueError(f"render-only input changed: {name}")
            events = json.loads((events_from / "events.json").read_text(encoding="utf-8"))
            shutil.copy2(events_from / "tracks.sqlite", output / "tracks.sqlite")
            store = TrackStore(output / "tracks.sqlite")
            receipt["events_from"] = str(events_from.resolve())
            receipt["inference_receipt"] = prior
            receipt["posture_mode"] = prior.get("posture_mode", "disabled")
            receipt["posture_review_behaviors"] = prior.get("posture_review_behaviors", [])
        else:
            if cache is None:
                raise ValueError("pose cache is required for canonical inference")
            binding, scale = bind_cache(video, cache, output / "tracks.sqlite", source_receipt=source_receipt,
                                        expected_source_sha256=expected_source_sha256)
            store = TrackStore(output / "tracks.sqlite")
            params = config_seed_parameters(load_config(config))
            binding["rule_policy"] = policy_receipt()
            binding["posture_mode"] = posture_mode
            binding["effective_parameters"] = asdict(params)
            binding["rule_source_sha256"] = {
                str(Path(function.__code__.co_filename).resolve()): _sha256(Path(function.__code__.co_filename))
                for function in (pair_rule_masks, individual_rule_masks, group_rule_candidates, infer_video_events, qualify_event_duration)
            }
            if posture_mode == "v3_review":
                from mouse_behavior.behavior import posture_adapter, posture_measurements, posture_rules
                binding["rule_source_sha256"].update({str(Path(module.__file__).resolve()): _sha256(Path(module.__file__))
                                                       for module in (posture_adapter, posture_measurements, posture_rules)})
            events, audit = infer_video_events(store, binding, scale, params, core_seconds=core_seconds,
                                               context_seconds=context_seconds, workers=workers, posture_mode=posture_mode)
            receipt.update(inference_audit=audit, seeded_parameters=asdict(params), policy=policy_receipt(),
                           config_path=str(config.resolve()), config_sha256=_sha256(config))
        _write_json(output / "input_binding.json", binding)
        _write_json(output / "scale_receipt.json", scale)
        _write_json(output / "events.json", events)
        rows = []
        if mode != "infer":
            if mode == "clips" and prior is not None and "full" in prior:
                full = reuse_full_package(events_from, output / "full", prior["full"], events, binding, scale)
                receipt["full_reused_from"] = str((events_from / "full").resolve())
                receipt["full_reuse_source_receipt_sha256"] = _sha256(events_from / "run_receipt.json")
            else:
                full = render_full_package(video, output / "full", store, events, binding, scale)
            receipt["full"] = full
            if mode in {"all", "clips"}:
                receipt["clip_workers"] = clip_workers
                receipt["clip_ffmpeg_threads"] = 1
                receipt["clip_preset"] = clip_preset
                receipt["clip_crf"] = 21
                receipt["clip_verification_decoder_threads"] = 1
                context_receipt = receipt
                while "inference_audit" not in context_receipt and isinstance(context_receipt.get("inference_receipt"), dict):
                    context_receipt = context_receipt["inference_receipt"]
                context = context_receipt.get("inference_audit", {})
                rows = export_packages(output / "clips", Path(full["video_path"]), store, events, binding, scale,
                                       workers=clip_workers, qualification_context=context, clip_preset=clip_preset)
        write_manifests(output, rows)
        receipt.update(status="complete", event_count=len(events), clip_count=len(rows),
                       track_truncation=False, source_preserved_sha256=_sha256(video))
        if receipt["source_preserved_sha256"] != binding["source_video"]["sha256"]:
            raise ValueError("source video changed during the run")
        names = ["input_binding.json", "scale_receipt.json", "events.json", "manifest.json", "manifest.csv"]
        names.append("tracks.sqlite")
        receipt["payload_sha256"] = {name: _sha256(output / name) for name in names}
        _write_json(output / "run_receipt.json", receipt)
        return receipt
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(output / "run_receipt.json", receipt)
        raise
