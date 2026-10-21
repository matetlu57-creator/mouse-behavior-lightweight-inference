"""Pixel-pose adapters for independent posture review; never read annotations."""

from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from mouse_behavior.behavior.posture_measurements import MeasurementConfig
from mouse_behavior.behavior.posture_rules import (
    DEFAULT_GROOMING_TEMPORAL, DEFAULT_REARING_TEMPORAL, infer_posture,
)

KEYPOINT_NAMES = (
    "nose",
    "right_ear",
    "left_ear",
    "base_of_neck",
    "left_hip",
    "right_hip",
    "base_of_tail",
)


def posture_clip_from_rows(
    rows, frame_count, fps, track_id, *, keypoint_names=KEYPOINT_NAMES, reader=MeasurementConfig()
):
    """Preserve exact visual ID, coordinate units and unobserved frame slots."""
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, Integral)
        or frame_count < 1
        or isinstance(fps, bool)
        or not isinstance(fps, (float, int))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("positive frame count and finite positive FPS required")
    if isinstance(track_id, bool) or not isinstance(track_id, Integral) or track_id < 0:
        raise ValueError("nonnegative integer visual track ID required")
    if len(keypoint_names) != 7 or set(keypoint_names) != set(KEYPOINT_NAMES):
        raise ValueError("unsupported seven-point keypoint names")
    order = [keypoint_names.index(name) for name in KEYPOINT_NAMES]
    points, boxes = np.full((frame_count, 7, 2), np.nan), np.full((frame_count, 4), np.nan)
    observed = np.zeros(frame_count, bool)
    raw_conf, detection_conf = np.full((frame_count, 7), np.nan), np.full(frame_count, np.nan)
    seen = set()
    for row in rows:
        frame = row["frame"]
        if (
            isinstance(frame, bool)
            or not isinstance(frame, Integral)
            or not 0 <= frame < frame_count
            or frame in seen
        ):
            raise ValueError("duplicate/out-of-bounds frame")
        seen.add(frame)
        time = float(row["time"])
        if not math.isfinite(time) or abs(time - frame / fps) > 1.5 / fps:
            raise ValueError("frame/time mismatch")
        matches = [d for d in row.get("detections", []) if d["track_id"] == track_id]
        if len(matches) > 1:
            raise ValueError("duplicate target detection")
        if not matches:
            continue
        detection = matches[0]
        kp, box = np.asarray(detection["keypoints"], float), np.asarray(detection["box"], float)
        if kp.shape != (7, 3) or box.shape != (4,):
            raise ValueError("expected seven x/y/confidence keypoints and xyxy box")
        kp = kp[order]
        raw_conf[frame] = kp[:, 2]
        confidence = detection.get("detection_confidence")
        confidence = float(confidence) if confidence is not None else float("nan")
        detection_conf[frame] = confidence
        if not math.isfinite(confidence) or not reader.detection_confidence <= confidence <= 1:
            continue
        observed[frame] = True
        good = (
            np.isfinite(kp).all(axis=1) & (kp[:, 2] >= reader.keypoint_confidence) & (kp[:, 2] <= 1)
        )
        points[frame, good] = kp[good, :2]
        if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
            boxes[frame] = box
    return dict(
        points=points,
        boxes=boxes,
        fps=float(fps),
        observed=observed,
        raw_conf=raw_conf,
        detection_conf=detection_conf,
        track_id=int(track_id),
    )


def load_posture_clip(directory: Path, track_id: int, *, reader=MeasurementConfig()):
    """Read tracks+metadata only; truth and human reviews never enter inference."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
    if meta["tracks"]["coordinate_system"] != "clip_pixels":
        raise ValueError("only explicit clip_pixels inputs are supported")
    rows = json.loads((directory / "tracks.json").read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("tracks.json must be a frame list")
    clip = posture_clip_from_rows(
        rows,
        meta["clip"]["frame_count"],
        meta["clip"]["fps"],
        track_id,
        keypoint_names=meta["pose"]["keypoint_names"],
        reader=reader,
    )
    clip["meta"] = meta
    return clip


def infer_posture_rows(rows, fps, *, frame_count=None, track_ids=None, diagnostics=True):
    """Expose resolved review events and both independent evidence ledgers."""
    if frame_count is None:
        frame_count = max((row["frame"] for row in rows), default=-1) + 1
    if track_ids is None:
        track_ids = sorted({d["track_id"] for row in rows for d in row["detections"]})
    events, audit = [], {}
    for track_id in track_ids:
        clip = posture_clip_from_rows(rows, frame_count, fps, int(track_id))
        prediction = infer_posture(clip, diagnostics=diagnostics)
        audit[str(track_id)] = prediction
        for event in prediction["resolution"]["events"]:
            canonical = event["behavior"].lower()
            events.append(
                dict(
                    behavior=event["behavior"],
                    canonical_behavior=canonical,
                    mouse_ids=[int(track_id)],
                    actor_id=-1,
                    target_id=-1,
                    participants=[],
                    start_frame=event["start_frame"],
                    end_frame=event["end_frame_exclusive"],
                    posture_engine="v3_decoupled",
                    posture_profile=prediction["profile"],
                    posture_revision=prediction["revision"],
                    confidence="heuristic_candidate",
                    screening_status="posture_review_candidate",
                    requires_review=True,
                    interval_semantics="source_frames_half_open",
                )
            )
    return events, audit


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def resolve_posture_window_audits(windows, fps):
    """Resolve per-ID candidate interval unions across all overlapping contexts.

    This sweep uses intervals rather than full-video pose arrays. Contexts may
    disagree near their boundaries; any shared same-ID Grooming/Rearing support
    becomes Unknown. Each remaining contiguous interval is requalified. Raw
    per-window candidates, scores and events are never modified.
    """
    tracks = {}
    for window in windows:
        offset = window["window_start"]
        for track, prediction in window.get("posture_review", {}).items():
            data = tracks.setdefault(track, dict(Grooming=[], Rearing=[], profile=prediction["profile"], revision=prediction["revision"]))
            if data["profile"] != prediction["profile"] or data["revision"] != prediction["revision"]:
                raise ValueError("cannot combine different posture profiles in one video")
            for behavior in ("Grooming", "Rearing"):
                for event in prediction["candidates"][behavior]["events"]:
                    data[behavior].append((offset+event["start_frame"], offset+event["end_frame_exclusive"]))
    output, audit = [], {}
    for track, data in tracks.items():
        raw = {name: _union(data[name]) for name in ("Grooming", "Rearing")}
        conflicts, rejected = [], []
        for behavior, opposite in (("Grooming", "Rearing"), ("Rearing", "Grooming")):
            temporal = DEFAULT_GROOMING_TEMPORAL if behavior == "Grooming" else DEFAULT_REARING_TEMPORAL
            for start, end in raw[behavior]:
                cursor = start
                exclusive = []
                for left, right in raw[opposite]:
                    if right <= cursor:
                        continue
                    if left >= end:
                        break
                    if cursor < left:
                        exclusive.append((cursor, min(left, end)))
                    conflicts.append((max(start, left), min(end, right)))
                    cursor = max(cursor, right)
                    if cursor >= end:
                        break
                if cursor < end:
                    exclusive.append((cursor, end))
                for left, right in exclusive:
                    if (right-left)/fps < temporal.min_duration_s:
                        rejected.append(dict(behavior=behavior, start_frame=left, end_frame=right,
                                             reason="post_conflict_fragment_below_minimum"))
                        continue
                    output.append(dict(behavior=behavior, canonical_behavior=behavior.lower(), mouse_ids=[int(track)],
                        actor_id=-1, target_id=-1, participants=[], start_frame=left, end_frame=right,
                        posture_engine="v3_decoupled", posture_profile=data["profile"], posture_revision=data["revision"],
                        confidence="heuristic_candidate", screening_status="posture_review_candidate", requires_review=True,
                        interval_semantics="source_frames_half_open", observed_all_participants_frame_intervals=[(left, right)],
                        observed_all_participants_frame_count=right-left, inferred_or_partially_unobserved_frame_count=0))
        audit[track] = dict(raw_candidate_frame_intervals=raw, conflict_frame_intervals=_union(conflicts),
                           duration_rejected=rejected, resolution="unknown_on_conflict_then_duration_requalification")
    return output, audit
