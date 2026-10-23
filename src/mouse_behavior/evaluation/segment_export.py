"""Export frame-exact, independent segments of an existing annotation sample.

Windows use clip-local half-open frame indices. The resulting H.264/yuv420p
video is re-encoded without audio; spatial dimensions and visual IDs are retained.
Full-range exports copy the original video, retaining its encoding and audio.
Annotation ranges stay clip-local while metadata.source describes the original
recording. The input files are never modified and existing targets are refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg

from mouse_behavior.evaluation.exported_dataset import SampleRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify_video(path: Path, count: int, fps: float) -> dict[str, Any]:
    """Decode the entire output instead of trusting its container frame count."""
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("exported video is empty")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("exported video cannot be opened")
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(actual_fps) or not math.isclose(
            actual_fps, fps, rel_tol=1e-4, abs_tol=1e-3
        ):
            raise ValueError(f"exported FPS {actual_fps} differs from source FPS {fps}")
        decoded = 0
        width = height = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            decoded += 1
        if decoded != count:
            raise ValueError(f"exported video has {decoded} frames; expected {count}")
        return {
            "frame_count": decoded,
            "fps": actual_fps,
            "width": width,
            "height": height,
            "bytes": size,
            "duration_s": decoded / actual_fps,
        }
    finally:
        capture.release()


def _cropped_annotation(original: dict, start: int, end: int, behavior: str,
                        mouse_ids: tuple[int, ...]) -> dict:
    """Only understood annotation fields become active claims in a new crop.

    Extension fields and the original event ID remain in provenance. Interval
    fields follow the canonical schema's clip-local half-open contract; counts
    are always recomputed from their intersected union, never copied.
    """
    common = {"confidence", "screening_status", "requires_review"}
    result = {key: copy.deepcopy(value) for key, value in original.items() if key in common}
    if behavior != original.get("behavior"):
        return result
    if "canonical_behavior" in original:
        result["canonical_behavior"] = original["canonical_behavior"]
    # Evidence for a different participant set must be re-evaluated upstream.
    if set(mouse_ids) != set(original.get("mouse_ids", [])):
        return result
    if "participants" in original:
        result["participants"] = [
            {**participant, "track_ids": [i for i in participant.get("track_ids", []) if i in mouse_ids]}
            for participant in original["participants"]
            if any(i in mouse_ids for i in participant.get("track_ids", []))
        ]
    for field in ("actor_id", "target_id", "role_status", "posture_engine", "posture_profile", "posture_revision"):
        if field in original:
            result[field] = copy.deepcopy(original[field])
    for field, value in original.items():
        if not field.endswith("_frame_intervals") or field == "parent_candidate_frame_intervals":
            continue
        if not isinstance(value, list):
            raise ValueError(f"{field} must contain half-open frame ranges")
        clipped = []
        for interval in value:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(type(index) is not int for index in interval)
                    or not 0 <= interval[0] < interval[1]):
                raise ValueError(f"invalid half-open range in {field}")
            left, right = max(start, interval[0]), min(end, interval[1])
            if left < right:
                clipped.append([left - start, right - start])
        union = []
        for left, right in sorted(clipped):
            if union and left <= union[-1][1]:
                union[-1][1] = max(union[-1][1], right)
            else:
                union.append([left, right])
        result[field] = union
        result[field.replace("_intervals", "_count")] = sum(right - left for left, right in union)
    if "observed_all_participants_frame_count" in result:
        result["inferred_or_partially_unobserved_frame_count"] = end - start - result["observed_all_participants_frame_count"]
    if result.get("role_support_frame_count") == 0:
        for field in ("actor_id", "target_id", "participants"):
            result.pop(field, None)
        result["role_status"] = "unknown_in_cropped_interval"
    return result


def export_segment(
    record: SampleRecord,
    output_dir: Path,
    *,
    start_frame: int,
    end_frame: int,
    behavior: str,
    mouse_ids: tuple[int, ...],
    provenance: dict,
) -> dict:
    """Export ``[start_frame, end_frame)`` into a new sample directory.

    The returned manifest contains hashes of every source and exported payload,
    caller provenance, both coordinate systems, and full-decode verification.
    On failure only the directory created by this invocation is removed.
    """
    target = Path(output_dir).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite sample directory: {target}")
    fps = float(record.fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source FPS must be finite and positive")
    if (
        type(start_frame) is not int
        or type(end_frame) is not int
        or not 0 <= start_frame < end_frame <= record.frame_count
    ):
        raise ValueError("segment bounds must be integer clip-local 0 <= start < end <= frames")
    if not isinstance(behavior, str) or not behavior.strip():
        raise ValueError("behavior must be a nonempty label")
    if not mouse_ids or any(type(value) is not int or value < 0 for value in mouse_ids):
        raise ValueError("mouse_ids must contain nonnegative integer visual IDs")
    if len(set(mouse_ids)) != len(mouse_ids):
        raise ValueError("mouse_ids must be unique")
    # Round-trip the supplied provenance before creating the output. This also
    # rejects non-JSON values and nonfinite numbers without leaving a directory.
    caller_provenance = json.loads(json.dumps(provenance, allow_nan=False))
    if not isinstance(caller_provenance, dict):
        raise ValueError("provenance must be a JSON object")
    source_paths = {
        "clip.mp4": Path(record.clip_path),
        "annotation.json": Path(record.annotation_path),
        "metadata.json": Path(record.metadata_path),
        "tracks.json": Path(record.tracks_path),
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    annotation = json.loads(source_paths["annotation.json"].read_text(encoding="utf-8"))
    metadata = json.loads(source_paths["metadata.json"].read_text(encoding="utf-8"))
    tracks = json.loads(source_paths["tracks.json"].read_text(encoding="utf-8"))
    if not isinstance(annotation, dict) or not isinstance(metadata, dict):
        raise ValueError("annotation and metadata must be JSON objects")
    if not isinstance(tracks, list):
        raise ValueError("tracks must use the exported per-frame list schema")
    original_annotation = copy.deepcopy(annotation)
    original_metadata = copy.deepcopy(metadata)
    known_ids = {detection.get("track_id") for row in tracks for detection in row.get("detections", [])}
    if not set(mouse_ids).issubset(known_ids):
        raise ValueError("mouse_ids must refer to existing source visual track IDs")
    original_source = copy.deepcopy(metadata.get("source", {}))
    if not isinstance(original_source, dict):
        raise ValueError("metadata.source must be a JSON object")
    recording_origin = original_source.get("start_frame", record.frame_start)
    if type(recording_origin) is not int or recording_origin < 0:
        raise ValueError("metadata.source.start_frame must be a nonnegative integer")
    recording_start = recording_origin + start_frame
    recording_end = recording_origin + end_frame - 1
    original_time = float(original_source.get("start_time", recording_origin / fps))
    if not math.isfinite(original_time):
        raise ValueError("metadata.source.start_time must be finite")
    count = end_frame - start_frame
    full_range = start_frame == 0 and end_frame == record.frame_count
    source_capture = cv2.VideoCapture(str(source_paths["clip.mp4"]))
    try:
        source_container_fps = float(source_capture.get(cv2.CAP_PROP_FPS))
    finally:
        source_capture.release()
    # Old exported containers sometimes round 30.012 FPS to 30. A byte copy
    # would preserve a video/track timeline disagreement; re-encode the same
    # frames at the annotation's FPS instead of relaxing verification.
    copied_video = full_range and math.isfinite(source_container_fps) and math.isclose(
        source_container_fps, fps, rel_tol=1e-4, abs_tol=1e-3
    )
    cropped_tracks = []
    seen = set()
    for entry in tracks:
        if not isinstance(entry, dict) or type(entry.get("frame")) is not int:
            raise ValueError("each track frame must have an integer frame index")
        local_frame = entry["frame"]
        if local_frame in seen or not 0 <= local_frame < record.frame_count:
            raise ValueError("track frame indices must be unique and within the source clip")
        seen.add(local_frame)
        if start_frame <= local_frame < end_frame:
            cropped = copy.deepcopy(entry)
            cropped.setdefault("original_clip_frame", local_frame)
            if "time" in entry:
                cropped.setdefault("original_clip_time", entry["time"])
            cropped["frame"] = local_frame - start_frame
            cropped["time"] = cropped["frame"] / fps
            cropped_tracks.append(cropped)
    cropped_tracks.sort(key=lambda entry: entry["frame"])
    annotation = _cropped_annotation(original_annotation, start_frame, end_frame, behavior, mouse_ids)
    annotation.update(
        behavior=behavior,
        mouse_ids=list(mouse_ids),
        frame_range={"start": 0, "end": count - 1},
        time_range={"start": 0.0, "end": count / fps},
    )
    candidate = caller_provenance.get("candidate", {})
    if isinstance(candidate, dict) and candidate.get("status"):
        annotation["screening_status"] = candidate["status"]
        annotation["requires_review"] = candidate["status"] not in {
            "human_confirmed", "original_label_preserved"
        }
        if annotation["requires_review"]:
            annotation["confidence"] = "screening_candidate"
    metadata = {key: copy.deepcopy(metadata[key]) for key in ("schema_version", "tracks", "pose", "source") if key in metadata}
    metadata["clip"] = dict(filename="clip.mp4", fps=fps, frame_count=count)
    metadata.setdefault("tracks", {}).update(frame_origin=0, time_origin=0.0)
    metadata.setdefault("source", {}).update(
        start_frame=recording_start,
        end_frame=recording_end,
        start_time=original_time + start_frame / fps,
        end_time=original_time + end_frame / fps,
    )
    offsets = {
        "source_clip_local_half_open": {"start": start_frame, "end": end_frame},
        "recording_absolute_inclusive": {"start": recording_start, "end": recording_end},
    }
    metadata["provenance"] = {
        "original_annotation": original_annotation,
        "original_metadata_source": original_source,
        "original_metadata_provenance": original_metadata.get("provenance"),
        "original_metadata": original_metadata,
        "source_sample_id": record.sample_id,
        "source_sha256": source_hashes,
        "offsets": offsets,
        "caller": caller_provenance,
        "video_encoding": (
            "original video copied; encoding and audio retained"
            if copied_video
            else "H.264/yuv420p, frame-exact re-encode, audio omitted"
        ),
        "source_container_fps": source_container_fps if math.isfinite(source_container_fps) else None,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()  # Exclusive creation: a concurrent/preexisting target is untouched.
    try:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(source_paths["clip.mp4"].resolve()),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (
                f"trim=start_frame={start_frame}:end_frame={end_frame},"
                f"setpts=PTS-STARTPTS,setpts=N/({fps}*TB)"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-r",
            str(fps),
            "-fps_mode",
            "cfr",
            "-movflags",
            "+faststart",
            str(target / "clip.mp4"),
        ]
        if copied_video:
            shutil.copy2(source_paths["clip.mp4"], target / "clip.mp4")
        else:
            # Explicit per-frame timestamps and CFR encoding avoid packet-duration
            # quirks in short crops without discarding or duplicating source frames.
            subprocess.run(command, check=True, capture_output=True, timeout=300)
        verified = _verify_video(target / "clip.mp4", count, fps)
        metadata["clip"].update(width=verified["width"], height=verified["height"],
                                decoded_duration_s=verified["duration_s"], encoded_fps=verified["fps"],
                                bytes=verified["bytes"])
        _write_json(target / "annotation.json", annotation)
        _write_json(target / "metadata.json", metadata)
        _write_json(target / "tracks.json", cropped_tracks)
        output_hashes = {name: _sha256(target / name) for name in source_paths}
        return {
            "source_sample_id": record.sample_id,
            "output_dir": str(target),
            "behavior": behavior,
            "mouse_ids": list(mouse_ids),
            "frame_count": count,
            "fps": fps,
            **offsets,
            "source_sha256": source_hashes,
            "output_sha256": output_hashes,
            "provenance": caller_provenance,
            "provenance_sha256": hashlib.sha256(
                json.dumps(
                    caller_provenance, sort_keys=True, ensure_ascii=False, allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
            "verification": verified,
            "audio_preserved": copied_video,
        }
    except BaseException:
        # target.mkdir succeeded in this invocation; no preexisting sample is
        # ever removed, and source files are only opened for reading.
        shutil.rmtree(target)
        raise
