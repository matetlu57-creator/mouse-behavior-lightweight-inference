"""H.264 full-video overlays and independent frame-exact review packages."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Real
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from mouse_behavior.evaluation.segment_export import _sha256, _verify_video, _write_json
from mouse_behavior.behavior.individual_rules import INDIVIDUAL_RULE_REVISION
from mouse_behavior.behavior.duration_policy import require_event_duration
from mouse_behavior.behavior.event_boundaries import EventBoundary
from mouse_behavior.evaluation.group_event_rules import GROUP_RULE_REVISION
from mouse_behavior.evaluation.confirmed_rules import policy_receipt
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters

LOGGER = logging.getLogger(__name__)
KEYPOINT_NAMES = ["nose", "right_ear", "left_ear", "base_of_neck", "left_hip", "right_hip", "base_of_tail"]
SKELETON = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]]
PAYLOADS = ("clip.mp4", "annotation.json", "metadata.json", "tracks.json")
DISPLAY_PRIORITY = {"attack": 0, "chase": 1, "approach": 2, "nose_head_contact": 3,
                    "nose_tail_contact": 3, "social_clustering": 4, "dispersal": 4,
                    "group_locomotion": 4, "huddle": 4, "isolation": 5, "together": 6,
                    "locomotion": 10, "stationary": 11}


def _duration_parameters(binding: dict):
    """Use the immutable run input binding, never arbitrary event thresholds."""
    effective = binding.get("effective_parameters")
    if effective is None:
        return None
    if not isinstance(effective, dict):
        raise ValueError("effective_parameters binding must be a parameter object")
    return HeuristicParameters(**effective)


def _verify_rule_binding(binding: dict, events: list[dict]) -> None:
    from mouse_behavior.canonical_video_pipeline import verify_rule_execution_binding
    from mouse_behavior.behavior.reviewed_comparisons import bound_comparison_review
    bound_comparison_review(binding)
    verify_rule_execution_binding(binding, required=False)
    if (binding.get("rule_snapshot_schema") == 1 and binding.get("posture_mode") != "v3_review"
            and any(event.get("posture_engine") == "v3_decoupled" for event in events)):
        raise ValueError("posture event lacks its authoritative v3 rule snapshot; rerun inference")


def _new_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite package: {path}")
    path.mkdir(parents=True)


def _point(points: np.ndarray, index: int) -> tuple[int, int] | None:
    row = points[index]
    if np.isfinite(row).all() and row[2] >= .25:
        return tuple(np.rint(row[:2]).astype(int))
    return None


def draw_overlay(frame: np.ndarray, row: dict, events: list[dict], source_frame: int) -> np.ndarray:
    """Head/body arrows use measured neck→nose and tail→neck, with no velocity fallback."""
    measured_positions = {}
    measured_boxes = set()
    active_labels = {}
    active_events = [event for event in events
                     if event["start_frame"] <= source_frame < event["end_frame"]]
    for event in active_events:
        for track in event["mouse_ids"]:
            active_labels.setdefault(track, set()).add(event["behavior"])
    for detection in row["detections"]:
        track = detection["track_id"]
        color = (int(50 + (track * 37) % 200), int(50 + (track * 67) % 200), int(50 + (track * 97) % 200))
        box = np.asarray(detection["box"], float)
        points = np.asarray(detection["keypoints"], float)
        measured_positions[track] = _point(points, 3)
        if np.isfinite(box).all():
            measured_boxes.add(track)
            left, top, right, bottom = np.rint(box).astype(int)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
            cv2.putText(frame, f"V{track}", (max(left, 0), max(top - 4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        for left, right in SKELETON:
            a, b = _point(points, left), _point(points, right)
            if a is not None and b is not None:
                cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
        for start, end, arrow_color in ((3, 0, (0, 255, 255)), (6, 3, (255, 0, 255))):
            a, b = _point(points, start), _point(points, end)
            if a is not None and b is not None and a != b:
                cv2.arrowedLine(frame, a, b, arrow_color, 2, cv2.LINE_AA, tipLength=.3)
        if track in active_labels and np.isfinite(box).all():
            # Mark only measured boxes: an inferred event never fabricates a
            # missing target's position or silently switches to another ID.
            left, top, right, bottom = np.rint(box).astype(int)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 6)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            label = f"V{track} " + "/".join(sorted(active_labels[track]))
            position = (max(left, 0), max(top-7, 16))
            cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2, cv2.LINE_AA)
    displayed = sorted(active_events, key=lambda event: (DISPLAY_PRIORITY.get(event["canonical_behavior"], 7),
                                                 event["start_frame"], event["event_id"]))
    for event in displayed:
        actor, target = event.get("actor_id", -1), event.get("target_id", -1)
        support = event.get("role_support_frame_intervals", [])
        a, b = measured_positions.get(actor), measured_positions.get(target)
        if (actor < 0 or a is None or b is None
                or not any(start <= source_frame < end for start, end in support)):
            continue
        names = ("Approaching", "Approached") if event["canonical_behavior"] == "approach" else ("Chaser", "Chased")
        relation_color = (0, 165, 255)
        cv2.arrowedLine(frame, a, b, relation_color, 2, cv2.LINE_AA, tipLength=.15)
        for position, name, track in zip((a, b), names, (actor, target)):
            cv2.putText(frame, f"{name} V{track}", (max(0, position[0] + 5), max(14, position[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, relation_color, 1, cv2.LINE_AA)
    # Keep original coordinates/dimensions. The panel is explicitly inside the image.
    lines = [f"HEURISTIC CANDIDATES - REVIEW REQUIRED | source frame {source_frame}",
             "Visual IDs only | yellow: neck->nose | magenta: tail->neck",
             "Bright green boxes: measured active behavior participants"]
    for event in displayed[:8]:
        ids = ",".join(map(str, event["mouse_ids"]))
        relation = ""
        if event.get("actor_id", -1) >= 0:
            relation = f" {event['actor_id']}->{event['target_id']}"
            support = event.get("role_support_frame_intervals", [])
            if not any(start <= source_frame < end for start, end in support):
                relation += " (event role; frame unmeasured)"
        elif event["canonical_behavior"] in {"approach", "chase"}:
            relation = " role unknown"
        missing = [f"V{track}" for track in event["mouse_ids"] if track not in measured_boxes]
        if missing:
            relation += f" ({','.join(missing)} not detected)"
        lines.append(f"{event['behavior']} [{ids}]{relation}")
    if len(active_events) > 8:
        lines.append(f"+ {len(active_events) - 8} further active events in annotation/events.json")
    height, width = frame.shape[:2]
    panel_height = min(height, 18 * len(lines) + 8)
    panel_width = min(width, 850)
    overlay = frame[:panel_height, :panel_width].copy()
    frame[:panel_height, :panel_width] = (overlay * .3).astype(np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (5, 17 + index * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _render_video(source: Path, target: Path, store, events: list[dict], info: dict, *,
                  source_start: int = 0, source_end: int | None = None,
                  clip_preset: str = "veryfast", clip_codec: str = "libx264") -> dict:
    """Render source-local half-open frames without decoding beyond a finite end."""
    fps, source_count = info["fps"], info["frame_count"]
    end = source_count if source_end is None else source_end
    if (type(source_start) is not int or type(end) is not int
            or not 0 <= source_start < end <= source_count):
        raise ValueError("render bounds must be source-local half-open integer frame indices")
    if clip_preset not in ("veryfast", "ultrafast"):
        raise ValueError("clip preset must be veryfast or ultrafast")
    if clip_codec not in ("libx264", "h264_nvenc"):
        raise ValueError("clip codec must be libx264 or h264_nvenc")
    count = end - source_start
    width, height = info["width"], info["height"]
    if width % 2 or height % 2:
        raise ValueError("H.264/yuv420p with preserved dimensions requires even width and height")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-c:v", clip_codec, "-threads", "1"]
    command += (["-preset", clip_preset, "-crf", "21"] if clip_codec == "libx264"
                else ["-preset", "p4", "-cq", "21"])
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    process = None
    rows = None
    ordered = sorted(events, key=lambda item: item["start_frame"])
    active, cursor = [], 0
    with tempfile.TemporaryFile() as errors:
        try:
            if not capture.isOpened():
                raise ValueError("source video cannot be opened for rendering")
            if source_start and (not capture.set(cv2.CAP_PROP_POS_FRAMES, source_start)
                                 or not math.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), source_start,
                                                     rel_tol=0., abs_tol=1e-3)):
                raise ValueError("source video could not seek to the bound starting frame")
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            rows = iter(store.rows(source_start, end))
            for index in range(source_start, end):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"source decode ended at {index}; expected through source frame {end - 1}")
                try:
                    row = next(rows)
                except StopIteration as error:
                    raise ValueError(f"render trajectories ended before source frame {index}") from error
                if row["source_frame"] != index or frame.shape[:2] != (height, width):
                    raise ValueError("render frame/trajectory alignment or dimensions changed")
                active = [event for event in active if event["end_frame"] > index]
                while cursor < len(ordered) and ordered[cursor]["start_frame"] <= index:
                    active.append(ordered[cursor])
                    cursor += 1
                process.stdin.write(draw_overlay(frame, row, active, index).tobytes())
                if (index - source_start) % 300 == 0:
                    LOGGER.info("[video render] %d/%d (source frame %d)", index - source_start, count, index)
            if source_end is None and capture.read()[0]:
                raise ValueError("source decodes more frames than the bound frame count")
            process.stdin.close()
            returncode = process.wait(timeout=300)
            if returncode:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg video render failed ({returncode}): {errors.read().decode(errors='replace')[-4000:]}")
        except BaseException as error:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=30)
            if isinstance(error, BrokenPipeError):
                errors.seek(0)
                raise RuntimeError(f"ffmpeg rejected render frames: {errors.read().decode(errors='replace')[-4000:]}") from error
            raise
        finally:
            capture.release()
            if rows is not None and callable(getattr(rows, "close", None)):
                rows.close()
    if source_start or source_end is not None:
        return _verify_bounded_video(target, count, fps)
    return _verify_video(target, count, fps)


def _annotation(event: dict | None, events: list[dict], count: int, fps: float, *, crop_start: int | None = None) -> dict:
    common = dict(confidence="heuristic_candidate", screening_status="heuristic_candidate",
                  requires_review=True, frame_range=dict(start=0, end=count - 1),
                  time_range=dict(start=0., end=count / fps))
    if event is None:
        return dict(behavior="Multiple_behavior_candidates", mouse_ids=sorted({track for e in events for track in e["mouse_ids"]}),
                    participants=[], events=events, **common)
    origin = event["start_frame"] if crop_start is None else crop_start
    boundary = EventBoundary(event["start_frame"], event["end_frame"], origin, origin + count, fps)
    common["frame_range"] = dict(start=event["start_frame"] - origin, end=event["end_frame"] - origin - 1)
    common["time_range"] = dict(start=(event["start_frame"] - origin) / fps, end=(event["end_frame"] - origin) / fps)
    annotation = dict(behavior=event["behavior"], canonical_behavior=event["canonical_behavior"],
                      mouse_ids=event["mouse_ids"], participants=copy.deepcopy(event.get("participants", [])),
                      actor_id=event.get("actor_id", -1), target_id=event.get("target_id", -1),
                      event_id=event["event_id"], role_status=event.get("role_status", "not_applicable"), **common)
    annotation["boundary"] = boundary.as_dict()
    for field, value in event.items():
        if field.endswith("_frame_intervals"):
            annotation[field] = [[start - origin, end - origin]
                                 for start, end in value]
        elif field.endswith("_frame_count"):
            annotation[field] = value
        elif field in {"posture_engine", "posture_profile", "posture_revision", "interaction_status", "interaction_semantics",
                       "actor_observed_path_lower_bound_cm", "observed_motion_by_track",
                       "approach_endpoints_requalified_after_split", "approach_union_endpoints_requalified",
                       "unknown_comparisons_are_not_far_evidence",
                       "parent_candidate_event_id", "review_queue_only", "biological_label_confirmed",
                       "review_reason", "source_static", "source_grooming", "source_rearing",
                       "source_event_ids", "source_canonical_behaviors", "process_boundary_status"}:
            annotation[field] = copy.deepcopy(value)
    if event.get("posture_engine") == "v3_decoupled":
        annotation["screening_status"] = "posture_review_candidate"
    if event.get("review_queue_only"):
        annotation["screening_status"] = "conflict_pending_manual_review"
    if "source_support" in event:
        annotation["source_support_source_frames"] = copy.deepcopy(event["source_support"])
        annotation["source_support_coordinate_system"] = "original_video_half_open_frames"
    return annotation


def _metadata(binding: dict, scale: dict, start: int, end: int, video_variant: str,
              event: dict | None = None, qualification_context: dict | None = None) -> dict:
    info = binding["source_video"]
    binding_digest = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    metadata = dict(schema_version="1.0", clip=dict(filename="clip.mp4", fps=info["fps"],
                width=info["width"], height=info["height"], frame_count=end - start),
                tracks=dict(frame_origin=0, time_origin=0., coordinate_system="clip_pixels",
                            source_frame_origin=0, identity_semantics="provisional_visual_track_id"),
                pose=dict(keypoint_names=KEYPOINT_NAMES, skeleton_edges=SKELETON),
                source=dict(video_filename=Path(info["path"]).name, video_path=info["path"],
                            sha256=info["sha256"], start_frame=start, end_frame=end - 1,
                            start_time=start / info["fps"], end_time=end / info["fps"],
                            crop_region=dict(x=0, y=0, w=info["width"], h=info["height"])),
                provenance=dict(video_variant=video_variant,
                    cache_binding_sha256=binding_digest,
                    cache_status_sha256=binding["cache_status_sha256"],
                    cache_chunk_sha256=binding["chunk_sha256"],
                    inference_route="canonical confirmed_rules + individual_rules + group_event_rules",
                    rule_policy=binding.get("rule_policy") or dict(social=policy_receipt(), individual_revision=INDIVIDUAL_RULE_REVISION,
                                     group_revision=GROUP_RULE_REVISION),
                    rule_source_sha256=binding.get("rule_source_sha256", {}),
                    rule_snapshot_schema=binding.get("rule_snapshot_schema"),
                    rule_runtime_snapshot=binding.get("rule_runtime_snapshot"),
                    isolation_comparison_review=binding.get("isolation_comparison_review"),
                    video_encoding="H.264/yuv420p; original dimensions; audio omitted",
                    event_half_open=dict(start=event["start_frame"], end=event["end_frame"]) if event else dict(start=start, end=end), scale=scale,
                    raw_observations="all original cached detections, not only event participants",
                    source_pose_order=binding.get("source_pose_order", "nose,left_ear,right_ear,neck,left_hip,right_hip,tail"),
                    exported_pose_order="nose,right_ear,left_ear,neck,left_hip,right_hip,tail",
                    confidence="heuristic_candidate", requires_review=True))
    if "screening_stage" in binding:
        metadata["provenance"]["screening_stage"] = binding["screening_stage"]
    if event is not None:
        if "process_boundary_status" in event:
            metadata["provenance"]["process_boundary_status"] = copy.deepcopy(event["process_boundary_status"])
        metadata["provenance"]["boundary"] = EventBoundary(
            event["start_frame"], event["end_frame"], start, end, info["fps"]).as_dict()
        context = qualification_context or {}
        duration_qualification = require_event_duration(event, info["fps"], parameters=_duration_parameters(binding))
        metadata["provenance"]["event_qualification"] = dict(
            export_unit="standalone_candidate_process_interval",
            standalone_duration_qualification=duration_qualification,
            geometry_qualification="upstream canonical candidate; duration verification does not rerun geometry",
            context_seconds=context.get("context_seconds"), core_seconds=context.get("core_seconds"),
            parent_candidate_frame_intervals=event.get("parent_candidate_frame_intervals"),
            source_event_half_open=dict(start=event["start_frame"], end=event["end_frame"]),
            retained_complete_parent_for_semantics=event.get("retained_complete_parent_for_semantics", False),
            retained_complete_parent_for_duration=event.get("retained_complete_parent_for_duration", False),
            persisted_support_intervals={key: value for key, value in event.items() if key.endswith("_frame_intervals")},
            evidence_support_is_separate_from_process_duration=True,
            source_window_bounds=[{key: window[key] for key in ("window_start", "window_end", "core_start", "core_end")}
                                  for window in context.get("windows", [])
                                  if window["core_start"] < end and window["core_end"] > start])
        if event.get("posture_engine") == "v3_decoupled":
            metadata["provenance"]["posture_review"] = dict(
                engine=event["posture_engine"], profile=event["posture_profile"], revision=event["posture_revision"],
                rearing_semantics="standing including wall-supported standing; image-projection candidate only",
                requires_review=True, height_or_wall_support_confirmed=False)
    return metadata


def _verify_bounded_video(path: Path, count: int, fps: float) -> dict:
    """Shared export verification contract with an explicitly bounded decoder.

    The existing segment_export verifier does not accept capture parameters;
    concurrent exports must cap each OpenCV FFmpeg decoder as well as encoders.
    """
    if type(count) is not int or count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("verification requires a positive frame count and finite positive FPS")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("exported video is empty")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_N_THREADS) != 1:
            raise ValueError("exported video cannot be opened with one decoder thread")
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(actual_fps) or not math.isclose(actual_fps, fps, rel_tol=1e-4, abs_tol=1e-3):
            raise ValueError(f"exported FPS {actual_fps} differs from source FPS {fps}")
        decoded = width = height = 0
        retrieved_indices = []
        # FFmpeg's grab decodes every next frame. Avoid the additional BGR
        # conversion/copy on interior frames; retrieve actual first/last images.
        while capture.grab():
            current_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if current_width <= 0 or current_height <= 0:
                raise ValueError("decoded frame has invalid dimensions")
            if decoded == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise ValueError("decoded frame dimensions changed inside exported video")
            decoded += 1
            if decoded == 1 or decoded == count:
                ok, frame = capture.retrieve()
                if not ok or frame is None or frame.shape != (height, width, 3):
                    raise ValueError("decoded boundary frame cannot be retrieved as expected BGR shape")
                retrieved_indices.append(decoded - 1)
        if decoded != count:
            raise ValueError(f"exported video has {decoded} frames; expected {count}")
        return dict(frame_count=decoded, fps=actual_fps, duration_s=decoded / actual_fps, width=width, height=height,
                    bytes=size, decoder_threads=1,
                    decode_method="grab_every_frame_retrieve_first_last", retrieved_frame_indices=retrieved_indices)
    finally:
        capture.release()


def _package_result(path: Path, event: dict | None, verification: dict, info: dict,
                    boundary: EventBoundary | None = None) -> dict:
    result = dict(path=str(path.resolve()), video_path=str((path / "clip.mp4").resolve()),
                event_id=event["event_id"] if event else "full_video",
                behavior=event["behavior"] if event else "Multiple_behavior_candidates",
                canonical_behavior=event["canonical_behavior"] if event else "multiple",
                mouse_ids=event["mouse_ids"] if event else [],
                actor_id=event.get("actor_id", -1) if event else -1,
                target_id=event.get("target_id", -1) if event else -1,
                participants=event.get("participants", []) if event else [],
                start_frame=event["start_frame"] if event else 0,
                end_frame=event["end_frame"] if event else info["frame_count"],
                frame_count=verification["frame_count"], fps=info["fps"],
                sha256={name: _sha256(path / name) for name in PAYLOADS}, verification=verification,
                interaction_status=event.get("interaction_status") if event else None,
                interaction_semantics=event.get("interaction_semantics") if event else None,
                review_queue_only=bool(event and event.get("review_queue_only")),
                biological_label_confirmed=False,
                review_reason=event.get("review_reason") if event else None,
                confidence="heuristic_candidate", requires_review=True)
    if boundary is not None:
        result.update(boundary.as_dict())
    return result


def render_full_package(source: Path, target: Path, store, events: list[dict], binding: dict, scale: dict) -> dict:
    _verify_rule_binding(binding, events)
    _new_directory(target)
    info = binding["source_video"]
    verification = _render_video(source, target / "clip.mp4", store, events, info)
    annotation = _annotation(None, events, info["frame_count"], info["fps"])
    annotation["mouse_ids"] = binding.get("visual_track_ids", annotation["mouse_ids"])
    _write_json(target / "annotation.json", annotation)
    _write_json(target / "metadata.json", _metadata(binding, scale, 0, info["frame_count"], "canonical_overlay_full"))
    store.write_tracks(target / "tracks.json", 0, info["frame_count"])
    return _package_result(target, None, verification, info)


def reuse_full_package(prior_root: Path, target: Path, previous: dict, events: list[dict],
                       binding: dict, scale: dict) -> dict:
    """Copy a complete verified package; paths alone never authorize reuse."""
    _verify_rule_binding(binding, events)
    original = prior_root / "full"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite package: {target}")
    if not isinstance(previous, dict) or set(previous.get("sha256", {})) != set(PAYLOADS):
        raise ValueError("full package has no complete four-payload hash ledger")
    if (Path(previous.get("path", "")).resolve() != original.resolve()
            or Path(previous.get("video_path", "")).resolve() != (original / "clip.mp4").resolve()):
        raise ValueError("full package paths disagree with the completed source run")
    for name in PAYLOADS:
        if not (original / name).is_file() or _sha256(original / name) != previous["sha256"][name]:
            raise ValueError(f"full package payload changed or hash mismatch: {name}")
    info = binding["source_video"]
    count, fps = info["frame_count"], info["fps"]
    if previous.get("frame_count") != count or not math.isclose(float(previous.get("fps", 0)), fps, rel_tol=1e-6):
        raise ValueError("full package receipt frame count/FPS differs from source binding")
    metadata = json.loads((original / "metadata.json").read_text(encoding="utf-8"))
    expected_metadata = _metadata(binding, scale, 0, count, "canonical_overlay_full")
    if metadata != expected_metadata:
        raise ValueError("full metadata differs from source binding, scale or provenance")
    annotation = json.loads((original / "annotation.json").read_text(encoding="utf-8"))
    expected_annotation = _annotation(None, events, count, fps)
    expected_annotation["mouse_ids"] = binding.get("visual_track_ids", expected_annotation["mouse_ids"])
    if annotation != expected_annotation:
        raise ValueError("full annotation differs from the persisted events or candidate contract")
    verified = _verify_bounded_video(original / "clip.mp4", count, fps)
    if verified["width"] != info["width"] or verified["height"] != info["height"]:
        raise ValueError("full video dimensions differ from source binding")
    old_verified = previous.get("verification", {})
    for field in ("frame_count", "width", "height", "bytes"):
        if old_verified.get(field) != verified[field]:
            raise ValueError(f"full video verification changed: {field}")
    _new_directory(target)
    for name in PAYLOADS:
        shutil.copy2(original / name, target / name)
    result = _package_result(target, None, verified, info)
    if result["sha256"] != previous["sha256"]:
        raise ValueError("full package hashes changed while copying")
    return result


def crop_bounds(event: dict, info: dict, clip_context_seconds: float = 0.) -> tuple[int, int, dict]:
    """Shared half-open crop bounds; Approach has only leading playback context."""
    boundary = EventBoundary.for_event(event, info, clip_context_seconds)
    fps = info["fps"]
    approach = str(event.get("canonical_behavior", event.get("behavior", ""))).lower() == "approach"
    before, after = clip_context_seconds, 0. if approach else clip_context_seconds
    start, end = boundary.crop_start, boundary.crop_end
    return start, end, dict(
        clip_context_before_seconds_requested=before,
        clip_context_after_seconds_requested=after,
        clip_context_before_seconds_actual=(event["start_frame"] - start) / fps,
        clip_context_after_seconds_actual=(end - event["end_frame"]) / fps)


def _export_one_package(root: Path, rendered: Path, store, event: dict,
                        binding: dict, scale: dict, seek_fps: float, qualification_context: dict | None,
                        clip_preset: str, clip_context_seconds: float = 0., original_video: bool = False,
                        clip_codec: str = "libx264", target_overlay: bool = False) -> dict:
    info = binding["source_video"]
    start, end, context_metadata = crop_bounds(event, info, clip_context_seconds)
    require_event_duration(event, info["fps"], parameters=_duration_parameters(binding))
    target = root / event["behavior"] / event["event_id"]
    _new_directory(target)
    if target_overlay:
        # Rendering the current event from the bound original prevents another
        # overlapping event's participants from leaking into this review clip.
        verified = _render_video(rendered, target / "clip.mp4", store, [event], info,
                                 source_start=start, source_end=end,
                                 clip_preset=clip_preset, clip_codec=clip_codec)
    else:
        # Accurate input seek decodes at most the preceding GOP. A half-frame
        # margin avoids timestamp rounding at a CFR frame boundary. Reset PTS
        # before CFR output so the margin cannot duplicate the first frame.
        seek_time = max(0., (start - .5) / seek_fps)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-threads", "1", "-ss", f"{seek_time:.12f}", "-i", str(rendered.resolve()),
                   "-map", "0:v:0", "-filter_threads", "1", "-vf",
                   f"setpts=N/({info['fps']}*TB)", "-frames:v", str(end - start),
                   "-c:v", clip_codec, "-threads", "1", "-pix_fmt", "yuv420p"]
        command += (["-preset", clip_preset, "-crf", "21"] if clip_codec == "libx264" else ["-preset", "p4", "-cq", "21"])
        if original_video:
            # The video half-frame seek avoids CFR rounding. Remove that same
            # margin from audio so its first sample starts at the crop boundary.
            audio_margin = start / seek_fps - seek_time
            command += ["-map", "0:a:0?", "-af", f"atrim=start={audio_margin:.12f},asetpts=PTS-STARTPTS",
                        "-c:a", "aac", "-b:a", "160k", "-t", f"{(end-start)/info['fps']:.12f}"]
        else:
            command += ["-an"]
        command += ["-r", str(info["fps"]), "-fps_mode", "cfr", "-movflags", "+faststart", str(target / "clip.mp4")]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=900)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"ffmpeg event export failed: {error.stderr.decode(errors='replace')[-4000:]}") from error
        verified = _verify_bounded_video(target / "clip.mp4", end - start, info["fps"])
    if verified["width"] != info["width"] or verified["height"] != info["height"]:
        raise ValueError("exported video dimensions differ from source binding")
    encoded_event = dict(event)
    # Context is playback only. Recheck just the true event frame count at the
    # actual encoded FPS, never the longer decoded crop duration.
    encoded_qualification = require_event_duration(encoded_event, verified["fps"], parameters=_duration_parameters(binding))
    _write_json(target / "annotation.json", _annotation(event, [], end - start, info["fps"], crop_start=start))
    variant = ("frame_exact_event_target_overlay" if target_overlay else
               "frame_exact_cut_of_original_video" if original_video else "frame_exact_cut_of_canonical_overlay_full")
    metadata = _metadata(binding, scale, start, end, variant, event, qualification_context)
    metadata["provenance"]["crop_half_open"] = dict(start=start, end=end)
    metadata["provenance"]["clip_context_seconds_requested"] = clip_context_seconds
    metadata["provenance"].update(context_metadata)
    metadata["provenance"]["event_offset_half_open"] = dict(start=event["start_frame"]-start, end=event["end_frame"]-start)
    if original_video:
        metadata["provenance"]["video_encoding"] = "H.264/yuv420p; original dimensions; optional source audio encoded AAC"
    if target_overlay:
        metadata["provenance"]["target_overlay"] = dict(
            event_id=event["event_id"], target_visual_track_ids=list(event["mouse_ids"]),
            render_source="bound_original_video",
            highlight_basis="only actual cached detections with finite measured boxes during this event",
            missing_target_policy="not detected; no fabricated boxes or replacement IDs",
            context_policy="no active event means no participant highlight")
    metadata["provenance"]["event_qualification"]["encoded_duration_qualification"] = dict(
        encoded_qualification, duration_basis="true_event_frame_count_over_encoded_fps" if clip_context_seconds else "decoded_frame_count_over_encoded_fps",
        event_frame_count=event["end_frame"] - event["start_frame"])
    metadata["clip"]["decoded_duration_s"] = verified["duration_s"]
    metadata["clip"]["encoded_fps"] = verified["fps"]
    metadata["provenance"]["encoding_settings"] = dict(codec=clip_codec, preset=clip_preset if clip_codec == "libx264" else "p4",
                                                          pixel_format="yuv420p", encoder_threads=1)
    metadata["provenance"]["encoding_settings"]["crf" if clip_codec == "libx264" else "cq"] = 21
    _write_json(target / "metadata.json", metadata)
    if original_video or target_overlay:
        _write_core_tracks(target / "tracks.json", store, start, end, info["fps"])
    else:
        store.write_tracks(target / "tracks.json", start, end)
    return _package_result(target, event, verified, info,
                          EventBoundary(event["start_frame"], event["end_frame"], start, end, info["fps"]))


def _write_core_tracks(path: Path, store, start: int, end: int, fps: float) -> None:
    """Write clip-local observations with exactly the portable core row schema."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write("[\n")
        count = 0
        for index, row in enumerate(store.rows(start, end)):
            if row.get("source_frame", row.get("frame")) != start + index:
                raise ValueError("crop tracks must cover every source frame in order")
            if index:
                handle.write(",\n")
            json.dump(dict(frame=index, time=index / fps, detections=row["detections"]),
                      handle, ensure_ascii=False, allow_nan=False)
            count += 1
        if count != end - start:
            raise ValueError("crop tracks do not cover the whole crop")
        handle.write("\n]\n")


def huddle_output_events(events: list[dict], store, binding: dict, scale: dict) -> tuple[list[dict], list[dict]]:
    """Sliding stable-state qualification followed by frame-level huddle bounds.

    Five seconds is a qualification window, never the exported clip length.
    Each member needs 60% measured visibility in a window; two visible compact
    members supply spatial support, not an invented position for a missing ID.
    Sustained visible separation splits the process at its first failing frame.
    """
    from mouse_behavior.evaluation.group_event_rules import _components
    from mouse_behavior.evaluation.event_frames import mask_intervals

    fps = binding["source_video"]["fps"]
    parameters = _duration_parameters(binding) or HeuristicParameters()
    minimum = parameters.huddle_min_duration_s
    width = int(math.ceil(minimum * fps - 1e-9))
    neighbor_cm, fraction, separation_s = 10., .6, .5
    confirmation = max(1, int(math.ceil(separation_s * fps - 1e-9)))
    result, audit, seen = [], [], set()
    for parent in events:
        if parent["canonical_behavior"] != "huddle":
            result.append(copy.deepcopy(parent))
            continue
        low, high = parent["start_frame"], parent["end_frame"]
        ids = parent["mouse_ids"]
        observed = np.zeros((high-low, len(ids)), bool)
        states = np.full(high-low, -1, np.int8)
        touching_endpoints = np.zeros(high-low, bool)
        for row in store.rows(low, high):
            detections = {d["track_id"]: d for d in row["detections"]}
            centers = np.full((len(ids), 2), np.nan)
            for column, track in enumerate(ids):
                if track not in detections:
                    continue
                detection = detections[track]
                points = np.asarray(detection.get("keypoints", []), float)
                if points.shape == (7, 3):
                    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= .25)
                    body = valid.copy()
                    body[[0, 1, 2, 6]] = False
                    usable = body if body.any() else valid
                    if usable.any():
                        centers[column] = np.median(points[usable, :2], axis=0)
                else:
                    box = np.asarray(detection.get("box", []), float)
                    if box.shape == (4,) and np.isfinite(box).all() and (box[2:] > box[:2]).all():
                        centers[column] = (box[:2]+box[2:])/2
            # Visible body-box contact confirms only the two displayed
            # boundaries. It is not a five-second all-visible support gate.
            if all(track in detections for track in ids) and len(ids) >= 3:
                boxes = np.asarray([detections[track]["box"] for track in ids], float)
                if boxes.shape == (len(ids), 4) and np.isfinite(boxes).all():
                    contact = ((boxes[:, None, 0] <= boxes[None, :, 2])
                               & (boxes[None, :, 0] <= boxes[:, None, 2])
                               & (boxes[:, None, 1] <= boxes[None, :, 3])
                               & (boxes[None, :, 1] <= boxes[:, None, 3]))
                    touching_endpoints[row["frame"]-low] = len(_components(contact, np.ones(len(ids), bool))) == 1
            centers *= scale["cm_per_pixel"]
            valid = np.isfinite(centers).all(axis=1)
            offset = row["frame"]-low
            observed[offset] = valid
            if valid.sum() < 2:
                continue
            distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
            compact = len(_components(distances <= neighbor_cm, valid)) == 1
            if compact:
                compact = bool((np.linalg.norm(centers[valid]-centers[valid].mean(axis=0), axis=1) <= neighbor_cm).all())
            states[offset] = (1 if valid.all() and len(ids) >= 3 else 2) if compact else 0
        # Confirmation suppresses one-frame pose jitter. The confirmed interval
        # itself is excluded; confirmation time is never appended to the crop.
        blocked = np.zeros(len(states), bool)
        for start, end in mask_intervals(states == 0):
            if end-start >= confirmation:
                blocked[start:end] = True
        for start, end in mask_intervals(states < 0):
            if (end-start)/fps > parameters.huddle_max_gap_s:
                blocked[start:end] = True
        spatial = states > 0
        qualified = np.zeros(len(states), bool)
        window_count = 0
        if len(states) >= width and len(ids) >= 3:
            spatial_sum = np.r_[0, np.cumsum(spatial)]
            blocked_sum = np.r_[0, np.cumsum(blocked)]
            member_sum = np.vstack([np.zeros(len(ids), int), np.cumsum(observed, axis=0)])
            good = ((spatial_sum[width:]-spatial_sum[:-width] >= fraction*width)
                    & (blocked_sum[width:]-blocked_sum[:-width] == 0)
                    & ((member_sum[width:]-member_sum[:-width]) >= fraction*width).all(axis=1))
            starts = np.flatnonzero(good)
            window_count = len(starts)
            delta = np.zeros(len(states)+1, int)
            np.add.at(delta, starts, 1)
            np.add.at(delta, starts+width, -1)
            qualified = np.cumsum(delta[:-1]) > 0
        accepted, rejected = [], []
        for start, end in mask_intervals(qualified & ~blocked):
            endpoints = np.flatnonzero((states[start:end] == 1) & touching_endpoints[start:end])
            if not len(endpoints):
                continue
            start, end = start+int(endpoints[0]), start+int(endpoints[-1])+1
            support = spatial[start:end]
            if (end-start)/fps + 1e-12 < minimum or support.sum()/fps + 1e-12 < minimum:
                rejected.append(dict(start_frame=low+start, end_frame=low+end,
                                     measured_compact_seconds=float(support.sum()/fps)))
                continue
            key = (tuple(ids), low+start, low+end)
            if key in seen:
                continue
            seen.add(key)
            event = copy.deepcopy(parent)
            for name in list(event):
                if name.endswith(("_frame_intervals", "_frame_count")):
                    del event[name]
            source_start, source_end = low+start, low+end
            intervals = [[source_start+a, source_start+b] for a, b in mask_intervals(support)]
            all_visible = observed[start:end].all(axis=1)
            event.update(start_frame=source_start, end_frame=source_end,
                event_id=f"huddle_sliding_{source_start:08d}_{source_end:08d}_" +
                         hashlib.sha256(str(tuple(ids)).encode()).hexdigest()[:10],
                parent_candidate_event_id=parent["event_id"],
                canonical_support_frame_intervals=intervals,
                canonical_support_frame_count=int(support.sum()),
                observed_all_participants_frame_intervals=[[source_start+a, source_start+b]
                    for a, b in mask_intervals(all_visible)],
                observed_all_participants_frame_count=int(all_visible.sum()),
                inferred_or_partially_unobserved_frame_count=int(len(support)-all_visible.sum()),
                boundary=EventBoundary(source_start, source_end, source_start, source_end, fps).as_dict(),
                process_boundary_status=dict(output_revision="huddle-sliding-output-v37",
                    center_neighbor_cm=neighbor_cm, centroid_radius_cm=neighbor_cm,
                    sliding_window_seconds=minimum, sliding_step_frames=1,
                    minimum_member_visibility_fraction=fraction, minimum_spatial_support_fraction=fraction,
                    separation_confirmation_seconds=separation_s,
                    support_semantics="at least two observed compact members; missing IDs have no measured position",
                    starts_and_ends_with_all_targets_observed=True, endpoint_body_boxes_connected=True,
                    minimum_duration_used_as_crop_length=False))
            require_event_duration(event, fps, parameters=parameters)
            result.append(event)
            accepted.append(event["event_id"])
        audit.append(dict(parent_event_id=parent["event_id"], accepted=accepted,
                          rejected_bouts_below_minimum=rejected, qualifying_sliding_windows=window_count,
                          visible_separated_frames=int((states == 0).sum()),
                          unknown_frames=int((states == -1).sum())))
    # A simultaneous strict subset is another membership proposal, not an
    # additional review clip for the same fully contained group process.
    retained = []
    for event in result:
        larger = next((other for other in result if event["canonical_behavior"] == "huddle"
                       and other["canonical_behavior"] == "huddle"
                       and set(event["mouse_ids"]) < set(other["mouse_ids"])
                       and other["start_frame"] <= event["start_frame"]
                       and event["end_frame"] <= other["end_frame"]), None)
        if larger is None:
            retained.append(event)
        else:
            audit.append(dict(suppressed_event_id=event["event_id"],
                              covered_by=larger["event_id"], reason="contained_member_subset"))
    return retained, audit


def export_packages(root: Path, rendered: Path, store, events: list[dict], binding: dict,
                    scale: dict, *, workers: int = 1, qualification_context: dict | None = None,
                    clip_preset: str = "veryfast", clip_context_seconds: float = 0.,
                    original_video: bool = False, clip_codec: str = "libx264",
                    target_overlay: bool = False) -> list[dict]:
    """Export every event, with bounded FFmpeg concurrency and stable order."""
    _verify_rule_binding(binding, events)
    if type(workers) is not int or workers < 1:
        raise ValueError("clip workers must be a positive integer")
    if clip_preset not in ("veryfast", "ultrafast"):
        raise ValueError("clip preset must be veryfast or ultrafast")
    if (isinstance(clip_context_seconds, bool) or not isinstance(clip_context_seconds, Real)
            or not math.isfinite(clip_context_seconds) or clip_context_seconds < 0):
        raise ValueError("clip context seconds must be finite and nonnegative")
    if type(original_video) is not bool:
        raise ValueError("original_video must be a boolean")
    if type(target_overlay) is not bool:
        raise ValueError("target_overlay must be a boolean")
    if original_video and target_overlay:
        raise ValueError("original_video and target_overlay are mutually exclusive")
    if clip_codec not in ("libx264", "h264_nvenc"):
        raise ValueError("clip codec must be libx264 or h264_nvenc")
    info = binding["source_video"]
    if original_video or target_overlay:
        rendered = Path(info["path"])
        if _sha256(rendered) != info["sha256"]:
            raise ValueError("original source video SHA-256 mismatch")
    seen = set()
    for event in events:
        start, end = event["start_frame"], event["end_frame"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= info["frame_count"]:
            raise ValueError("event bounds must be source-local half-open integer frame indices")
        event_id = event.get("event_id")
        if event_id in seen:
            raise ValueError(f"duplicate event_id: {event_id}")
        for name in (event_id, event.get("behavior")):
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or any(character in '<>:"/\\|?*' for character in name)):
                raise ValueError("event IDs and behavior labels must be safe directory components")
        seen.add(event_id)
    # Validate the complete batch before any directory, decoder or encoder is
    # opened, so one rejected child interval cannot leave partial packages.
    for event in events:
        require_event_duration(event, info["fps"], parameters=_duration_parameters(binding))
    capture = cv2.VideoCapture(str(rendered))
    try:
        seek_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if (not np.isfinite(seek_fps) or seek_fps <= 0
            or not math.isclose(seek_fps, info["fps"], rel_tol=1e-4, abs_tol=1e-3)):
        raise ValueError("export source FPS differs from bound FPS for frame-exact seek")
    _new_directory(root)
    result: list = [None] * len(events)
    if workers == 1:
        for index, event in enumerate(events):
            result[index] = _export_one_package(root, rendered, store, event, binding, scale, seek_fps, qualification_context, clip_preset, clip_context_seconds, original_video, clip_codec, target_overlay)
            if index % 25 == 0 or index + 1 == len(events):
                LOGGER.info("[event packages] %d/%d", index + 1, len(events))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_export_one_package, root, rendered, store, event, binding, scale, seek_fps, qualification_context, clip_preset, clip_context_seconds, original_video, clip_codec, target_overlay): index
                       for index, event in enumerate(events)}
            for completed, future in enumerate(as_completed(futures), 1):
                result[futures[future]] = future.result()
                if completed % 25 == 0 or completed == len(events):
                    LOGGER.info("[event packages] %d/%d (%d workers, FFmpeg threads=1)", completed, len(events), workers)
    return result


def write_manifests(output: Path, rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["behavior"]] = counts.get(row["behavior"], 0) + 1
    _write_json(output / "manifest.json", dict(schema_version="2.0", clip_count=len(rows),
                counts_by_behavior=counts, confidence="heuristic_candidate", requires_review=True, clips=rows))
    fields = ["event_id", "behavior", "canonical_behavior", "path", "start_frame", "end_frame", "frame_count",
              "fps", "mouse_ids", "actor_id", "target_id", "participants", "sha256", "requires_review", "interaction_status", "interaction_semantics",
              "review_queue_only", "biological_label_confirmed", "review_reason",
              "boundary_schema_version", "coordinate_system", "source_event_half_open", "playback_crop_half_open",
              "event_offset_in_playback_half_open", "event_frame_count", "playback_frame_count",
              "context_before_frames", "context_after_frames", "event_duration_s", "playback_duration_s",
              "playback_scope", "qualification_basis"]
    with (output / "manifest.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})
