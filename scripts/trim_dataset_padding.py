#!/usr/bin/env python3
"""Trim context padding from cleaned dataset clips to strict zero-padded [0, N-1] ranges.

Reads ``F:\\mouse_behavior_cleaned_dataset_20261121`` strictly read-only while building
a verified zero-padded staging tree in ``F:\\mouse_behavior_cleaned_dataset_20261121_trim_stage``.
After verifying all 764 clips (4 files each + manifest = 3,057 files) with OpenCV
frame-count decode checks, stops ``mouse_clip_review`` (8795) via Site Hub (8788),
atomically swaps the dataset directory, restarts 8795 and verifies ``/api/catalog``,
and atomically rebuilds and CRC-tests ``F:\\mouse_behavior_cleaned_dataset_20261121.zip``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _bootstrap import REPO_ROOT  # noqa: E402, F401
from mouse_behavior.evaluation.review_transaction import write_json  # noqa: E402
from mouse_behavior.visualization.mouse_clip_cropper import (  # noqa: E402
    _encode_segment_h264,
    _verify_exported_video,
    get_ffmpeg_executable,
)

DEFAULT_DATASET_DIR = Path(r"F:\mouse_behavior_cleaned_dataset_20261121")
DEFAULT_STAGE_DIR = Path(r"F:\mouse_behavior_cleaned_dataset_20261121_trim_stage")
DEFAULT_BACKUP_DIR = Path(r"F:\mouse_behavior_cleaned_dataset_20261121_bak_tmp")
DEFAULT_ZIP_PATH = Path(r"F:\mouse_behavior_cleaned_dataset_20261121.zip")
DEFAULT_HUB_URL = "http://127.0.0.1:8788"
DEFAULT_REVIEW_URL = "http://127.0.0.1:8795"
DEFAULT_SITE_ID = "mouse_clip_review"
DEFAULT_WORKERS = 12
REQUIRED_CLIP_FILES = ("clip.mp4", "annotation.json", "metadata.json", "tracks.json")

_LOCAL_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def _safe_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _build_trimmed_tracks(
    raw_tracks: Any,
    start: int,
    end: int,
    fps: float,
) -> list[dict[str, Any]]:
    """Slice ``tracks.json`` to inclusive ``[start, end]`` and re-index to ``0..M-1``."""
    if not isinstance(raw_tracks, list):
        raise ValueError("tracks.json must be a JSON list")

    by_frame: dict[int, Mapping[str, Any]] = {}
    for row in raw_tracks:
        if not isinstance(row, Mapping):
            raise ValueError("tracks.json entry must be a JSON object")
        frame_idx = row.get("frame")
        if not _is_strict_int(frame_idx):
            raise ValueError(f"invalid frame index in tracks.json: {frame_idx!r}")
        if frame_idx in by_frame:
            raise ValueError(f"duplicate frame index {frame_idx} in tracks.json")
        by_frame[frame_idx] = row

    expected_count = end - start + 1
    trimmed: list[dict[str, Any]] = []
    for new_idx, orig_frame in enumerate(range(start, end + 1)):
        row = by_frame.get(orig_frame)
        if row is None:
            raise ValueError(f"missing frame {orig_frame} in tracks.json while slicing [{start}, {end}]")
        entry = copy.deepcopy(dict(row))
        entry["frame"] = new_idx
        entry["time"] = round(new_idx / fps, 6)
        if not _is_strict_int(entry.get("source_frame")):
            entry["source_frame"] = orig_frame
        detections = entry.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"tracks.json frame {orig_frame} detections must be a list")
        trimmed.append(entry)

    if len(trimmed) != expected_count:
        raise RuntimeError(f"trimmed tracks length {len(trimmed)} != expected {expected_count}")
    return trimmed


def _build_trimmed_annotation(
    annotation: Mapping[str, Any],
    *,
    start: int,
    end: int,
    orig_frame_count: int,
    fps: float,
    trimmed_at_utc: str,
) -> dict[str, Any]:
    """Re-index ``annotation.json`` to ``[0, M-1]`` and record trim provenance."""
    new_count = end - start + 1
    out = copy.deepcopy(dict(annotation))
    old_fr = copy.deepcopy(annotation["frame_range"])
    old_tr = copy.deepcopy(annotation.get("time_range"))
    out["frame_range"] = {"start": 0, "end": new_count - 1}
    out["time_range"] = {"start": 0.0, "end": round(new_count / fps, 6)}
    out["padding_trim_provenance"] = {
        "original_frame_range": old_fr,
        "original_time_range": old_tr,
        "original_frame_count": orig_frame_count,
        "new_frame_count": new_count,
        "trimmed_left_frames": start,
        "trimmed_right_frames": orig_frame_count - 1 - end,
        "trimmed_at_utc": trimmed_at_utc,
    }
    return out


def _build_trimmed_metadata(
    metadata: Mapping[str, Any],
    *,
    start: int,
    end: int,
    orig_frame_count: int,
    fps: float,
    verification: Mapping[str, Any],
    trimmed_at_utc: str,
) -> dict[str, Any]:
    """Update ``metadata.json`` counts/source window while preserving scale, pose, and tracks."""
    new_count = end - start + 1
    out = copy.deepcopy(dict(metadata))

    clip_block = out.get("clip")
    if not isinstance(clip_block, dict):
        raise ValueError("metadata.json missing 'clip' object")
    clip_block["frame_count"] = new_count
    clip_block["decoded_duration_s"] = round(new_count / fps, 4)
    clip_block["encoded_fps"] = verification["fps"]

    source_block = out.get("source")
    if isinstance(source_block, dict):
        src_ann = source_block.get("annotation")
        src_clip = source_block.get("clip")
        if isinstance(src_clip, dict) and isinstance(src_ann, Mapping):
            for key in ("start_frame", "end_frame", "start_time", "end_time"):
                if key in src_ann:
                    src_clip[key] = src_ann[key]

    out["padding_trim"] = {
        "original_frame_range": {"start": start, "end": end},
        "original_frame_count": orig_frame_count,
        "new_frame_count": new_count,
        "trimmed_left_frames": start,
        "trimmed_right_frames": orig_frame_count - 1 - end,
        "video_encoding": "H.264/yuv420p/+faststart",
        "verification": dict(verification),
        "trimmed_at_utc": trimmed_at_utc,
    }
    return out


def _process_clip_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Process one clip into staging: lossless copy if zero-padded, else exact H.264 trim."""
    src_dir = Path(job["src_dir"])
    dst_dir = Path(job["dst_dir"])
    behavior = str(job["behavior"])
    clip_id = str(job["clip_id"])
    ffmpeg_exe = str(job["ffmpeg_exe"])
    trimmed_at_utc = str(job["trimmed_at_utc"])

    for fname in REQUIRED_CLIP_FILES:
        fpath = src_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"missing required clip file: {fpath}")

    annotation = _read_json(src_dir / "annotation.json")
    metadata = _read_json(src_dir / "metadata.json")
    raw_tracks = _read_json(src_dir / "tracks.json")

    if not isinstance(annotation, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(f"annotation.json and metadata.json must be JSON objects in {src_dir}")

    clip_meta = metadata.get("clip")
    if not isinstance(clip_meta, Mapping):
        raise ValueError(f"metadata.json missing clip block in {src_dir}")

    orig_frame_count = clip_meta.get("frame_count")
    fps = clip_meta.get("fps")
    if not _is_strict_int(orig_frame_count) or orig_frame_count <= 0:
        raise ValueError(f"invalid frame_count={orig_frame_count!r} in {src_dir}")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError(f"invalid fps={fps!r} in {src_dir}")
    fps_f = float(fps)

    fr = annotation.get("frame_range")
    if not isinstance(fr, Mapping):
        raise ValueError(f"annotation.json missing frame_range in {src_dir}")
    start = fr.get("start")
    end = fr.get("end")
    if not (_is_strict_int(start) and _is_strict_int(end)):
        raise ValueError(f"invalid frame_range={fr!r} in {src_dir}")
    if not (0 <= start <= end < orig_frame_count):
        raise ValueError(
            f"frame_range [{start}, {end}] invalid for frame_count={orig_frame_count} in {src_dir}"
        )
    if not isinstance(raw_tracks, list) or len(raw_tracks) != orig_frame_count:
        raise ValueError(
            f"tracks.json length {len(raw_tracks) if isinstance(raw_tracks, list) else 'non-list'} "
            f"!= frame_count={orig_frame_count} in {src_dir}"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{behavior}/{clip_id}"

    if start == 0 and end == orig_frame_count - 1:
        for fname in REQUIRED_CLIP_FILES:
            shutil.copy2(src_dir / fname, dst_dir / fname)
        return {
            "status": "kept_zero_padded",
            "rel_path": rel_path,
            "behavior": behavior,
            "clip_id": clip_id,
            "original_frame_range": {"start": start, "end": end},
            "original_frame_count": orig_frame_count,
            "new_frame_count": orig_frame_count,
            "trimmed_left_frames": 0,
            "trimmed_right_frames": 0,
        }

    new_count = end - start + 1
    dst_video = dst_dir / "clip.mp4"
    verification = _encode_segment_h264(
        src_dir / "clip.mp4",
        dst_video,
        start,
        end + 1,
        fps_f,
        ffmpeg_exe=ffmpeg_exe,
        overlay_spec=None,
    )
    if verification["frame_count"] != new_count:
        raise RuntimeError(
            f"verified video frame_count {verification['frame_count']} != {new_count} for {rel_path}"
        )

    trimmed_tracks = _build_trimmed_tracks(raw_tracks, start, end, fps_f)
    write_json(dst_dir / "tracks.json", trimmed_tracks)

    trimmed_ann = _build_trimmed_annotation(
        annotation,
        start=start,
        end=end,
        orig_frame_count=orig_frame_count,
        fps=fps_f,
        trimmed_at_utc=trimmed_at_utc,
    )
    write_json(dst_dir / "annotation.json", trimmed_ann)

    trimmed_meta = _build_trimmed_metadata(
        metadata,
        start=start,
        end=end,
        orig_frame_count=orig_frame_count,
        fps=fps_f,
        verification=verification,
        trimmed_at_utc=trimmed_at_utc,
    )
    write_json(dst_dir / "metadata.json", trimmed_meta)

    return {
        "status": "trimmed",
        "rel_path": rel_path,
        "behavior": behavior,
        "clip_id": clip_id,
        "original_frame_range": {"start": start, "end": end},
        "original_frame_count": orig_frame_count,
        "new_frame_range": {"start": 0, "end": new_count - 1},
        "new_frame_count": new_count,
        "trimmed_left_frames": start,
        "trimmed_right_frames": orig_frame_count - 1 - end,
        "verification": verification,
    }


def _verify_staged_clip_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Independently verify one staged clip directory (4 files, JSONs, and OpenCV decode)."""
    stage_clip_dir = Path(job["stage_clip_dir"])
    src_clip_dir = Path(job["src_clip_dir"]) if job.get("src_clip_dir") else None
    expect_unchanged = bool(job.get("expect_unchanged", False))
    behavior = str(job["behavior"])
    clip_id = str(job["clip_id"])

    actual_files = sorted(p.name for p in stage_clip_dir.iterdir() if p.is_file())
    actual_dirs = [p.name for p in stage_clip_dir.iterdir() if p.is_dir()]
    if actual_dirs or actual_files != sorted(REQUIRED_CLIP_FILES):
        raise RuntimeError(
            f"unexpected files in {stage_clip_dir}: files={actual_files}, dirs={actual_dirs}"
        )

    annotation = _read_json(stage_clip_dir / "annotation.json")
    metadata = _read_json(stage_clip_dir / "metadata.json")
    tracks = _read_json(stage_clip_dir / "tracks.json")

    frame_count = metadata["clip"]["frame_count"]
    fps = float(metadata["clip"]["fps"])
    if not _is_strict_int(frame_count) or frame_count <= 0:
        raise RuntimeError(f"invalid staged frame_count in {stage_clip_dir}: {frame_count!r}")

    expected_fr = {"start": 0, "end": frame_count - 1}
    if annotation.get("frame_range") != expected_fr:
        raise RuntimeError(
            f"staged frame_range {annotation.get('frame_range')} != {expected_fr} in {stage_clip_dir}"
        )

    if not isinstance(tracks, list) or len(tracks) != frame_count:
        raise RuntimeError(
            f"staged tracks length {len(tracks) if isinstance(tracks, list) else 'non-list'} "
            f"!= {frame_count} in {stage_clip_dir}"
        )
    frame_indices = [row.get("frame") for row in tracks]
    if frame_indices != list(range(frame_count)):
        raise RuntimeError(f"non-sequential frame indices in {stage_clip_dir}")

    src_block = metadata.get("source")
    if isinstance(src_block, Mapping):
        src_clip = src_block.get("clip")
        src_ann = src_block.get("annotation")
        if isinstance(src_clip, Mapping) and isinstance(src_ann, Mapping):
            for key in ("start_frame", "end_frame", "start_time", "end_time"):
                if key in src_ann and src_clip.get(key) != src_ann.get(key):
                    raise RuntimeError(
                        f"source.clip.{key} ({src_clip.get(key)}) != source.annotation.{key} "
                        f"({src_ann.get(key)}) in {stage_clip_dir}"
                    )

    if expect_unchanged and src_clip_dir is not None:
        for fname in REQUIRED_CLIP_FILES:
            if _sha256_file(stage_clip_dir / fname) != _sha256_file(src_clip_dir / fname):
                raise RuntimeError(f"zero-padded file hash changed for {stage_clip_dir / fname}")

    video_info = _verify_exported_video(stage_clip_dir / "clip.mp4", frame_count, fps)
    return {
        "rel_path": f"{behavior}/{clip_id}",
        "behavior": behavior,
        "clip_id": clip_id,
        "frame_count": frame_count,
        "video_bytes": video_info["bytes"],
        "expect_unchanged": expect_unchanged,
    }


def _discover_clips(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    clips: list[tuple[str, str, Path]] = []
    for beh_dir in sorted(dataset_dir.iterdir(), key=lambda p: p.name.lower()):
        if not beh_dir.is_dir():
            continue
        for clip_dir in sorted(beh_dir.iterdir(), key=lambda p: p.name.lower()):
            if not clip_dir.is_dir():
                continue
            clips.append((beh_dir.name, clip_dir.name, clip_dir))
    return clips


def _update_manifest(
    source_manifest_path: Path,
    stage_manifest_path: Path,
    *,
    results: Sequence[Mapping[str, Any]],
    trimmed_at_utc: str,
) -> dict[str, Any]:
    """Update ``dataset_manifest.json`` in staging with trimmed counts and audit records."""
    manifest = _read_json(source_manifest_path) if source_manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {source_manifest_path}")

    by_rel_path = {str(item["rel_path"]): item for item in results}
    trimmed_items = sorted(
        (dict(item) for item in results if item["status"] == "trimmed"),
        key=lambda r: r["rel_path"].lower(),
    )
    kept_items = sorted(
        (dict(item) for item in results if item["status"] == "kept_zero_padded"),
        key=lambda r: r["rel_path"].lower(),
    )

    relabeled = manifest.get("relabeled_clips")
    if isinstance(relabeled, list):
        for entry in relabeled:
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("new_rel_path")
            res = by_rel_path.get(str(rel_path))
            if res is not None:
                entry.setdefault("original_frame_count", entry.get("frame_count"))
                entry["frame_count"] = int(res["new_frame_count"])

    per_beh_counts = dict(sorted(Counter(item["behavior"] for item in results).items()))
    manifest["padding_trimmed_at"] = trimmed_at_utc
    manifest["padding_trim"] = {
        "schema": "mouse_behavior_padding_trim_v1",
        "trimmed_at_utc": trimmed_at_utc,
        "total_clips": len(results),
        "trimmed_clips_count": len(trimmed_items),
        "already_zero_padded_count": len(kept_items),
        "zero_padded_verified_count": len(results),
        "non_empty_behaviors": len(per_beh_counts),
        "total_files": len(results) * len(REQUIRED_CLIP_FILES) + 1,
        "per_behavior_final_counts": per_beh_counts,
        "trimmed_clips": trimmed_items,
    }
    write_json(stage_manifest_path, manifest)
    return manifest


def _verify_full_dataset(
    target_dir: Path,
    *,
    source_dir_for_unchanged: Path | None,
    unchanged_rel_paths: set[str],
    workers: int,
) -> dict[str, Any]:
    """Run full 764-clip / 3,057-file verification including OpenCV decode on every clip."""
    manifest_path = target_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing dataset_manifest.json in {target_dir}")

    clips = _discover_clips(target_dir)
    all_files = [p for p in target_dir.rglob("*") if p.is_file()]
    expected_file_count = len(clips) * len(REQUIRED_CLIP_FILES) + 1
    if len(all_files) != expected_file_count:
        raise RuntimeError(
            f"total file count {len(all_files)} != expected {expected_file_count} in {target_dir}"
        )

    jobs: list[dict[str, Any]] = []
    for behavior, clip_id, clip_dir in clips:
        rel_path = f"{behavior}/{clip_id}"
        jobs.append(
            {
                "stage_clip_dir": str(clip_dir),
                "src_clip_dir": str(source_dir_for_unchanged / behavior / clip_id)
                if source_dir_for_unchanged is not None
                else None,
                "expect_unchanged": rel_path in unchanged_rel_paths,
                "behavior": behavior,
                "clip_id": clip_id,
            }
        )

    verified: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_verify_staged_clip_job, job) for job in jobs]
        for fut in as_completed(futures):
            verified.append(fut.result())

    total_video_bytes = sum(int(item["video_bytes"]) for item in verified)
    return {
        "total_clips": len(verified),
        "total_files": len(all_files),
        "non_empty_behaviors": len({item["behavior"] for item in verified}),
        "total_video_bytes": total_video_bytes,
    }


def _http_get_json(url: str, *, timeout_s: float = 5.0) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _LOCAL_HTTP_OPENER.open(req, timeout=timeout_s) as resp:
        status = getattr(resp, "status", resp.getcode())
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" not in ctype:
            raise RuntimeError(f"unexpected Content-Type from {url}: {ctype!r}")
        payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"non-object JSON payload from {url}")
        return int(status), payload


def _http_post_json(
    url: str,
    *,
    token: str,
    timeout_s: float = 15.0,
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Hub-Token": token,
        },
    )
    with _LOCAL_HTTP_OPENER.open(req, timeout=timeout_s) as resp:
        status = getattr(resp, "status", resp.getcode())
        payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"non-object JSON payload from {url}")
        return int(status), payload


def _probe_hub(hub_url: str) -> dict[str, Any] | None:
    base = hub_url.rstrip("/")
    try:
        status, payload = _http_get_json(f"{base}/api/health", timeout_s=2.5)
        if status == 200 and payload.get("status") == "ok" and isinstance(payload.get("session_token"), str):
            return payload
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError):
        return None
    return None


def _wait_for_port_down(url: str, *, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _http_get_json(url, timeout_s=1.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            return
        time.sleep(0.25)
    raise RuntimeError(f"service at {url} did not stop within {timeout_s}s")


def _wait_for_review_ready(review_url: str, *, timeout_s: float = 30.0) -> tuple[dict[str, Any], dict[str, Any]]:
    base = review_url.rstrip("/")
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            h_status, health = _http_get_json(f"{base}/api/health", timeout_s=3.0)
            if h_status == 200 and health.get("status") == "ok":
                c_status, catalog = _http_get_json(f"{base}/api/catalog", timeout_s=10.0)
                if c_status == 200 and catalog.get("total_clips") == 764:
                    return health, catalog
        except Exception as exc:
            last_err = exc
        time.sleep(0.4)
    raise RuntimeError(f"review site {base} did not become ready within {timeout_s}s: {last_err}")


def _rename_with_retry(src: Path, dst: Path, *, attempts: int = 8, delay_s: float = 0.5) -> None:
    last_exc: OSError | None = None
    for idx in range(attempts):
        try:
            src.rename(dst)
            return
        except PermissionError as exc:
            last_exc = exc
            if idx + 1 < attempts:
                time.sleep(delay_s)
    assert last_exc is not None
    raise last_exc


def _swap_directories(source_dir: Path, stage_dir: Path, backup_dir: Path) -> None:
    """Atomically replace ``source_dir`` with ``stage_dir`` using ``backup_dir`` for rollback."""
    if backup_dir.exists():
        raise FileExistsError(f"temporary backup dir already exists: {backup_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source dir missing: {source_dir}")
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"stage dir missing: {stage_dir}")

    _rename_with_retry(source_dir, backup_dir)
    try:
        _rename_with_retry(stage_dir, source_dir)
    except BaseException:
        if backup_dir.exists() and not source_dir.exists():
            _rename_with_retry(backup_dir, source_dir)
        raise

    if not (source_dir / "dataset_manifest.json").is_file():
        raise RuntimeError(f"post-swap verification failed: missing manifest in {source_dir}")
    shutil.rmtree(backup_dir)


def _rebuild_zip_archive(dataset_dir: Path, zip_path: Path) -> dict[str, Any]:
    """Rebuild ``zip_path`` atomically with ZIP_STORED for MP4 and ZIP_DEFLATED for JSON."""
    tmp_zip = zip_path.with_name(zip_path.name + ".tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()

    files = sorted((p for p in dataset_dir.rglob("*") if p.is_file()), key=lambda p: p.as_posix().lower())
    root_prefix = dataset_dir.name

    try:
        with zipfile.ZipFile(tmp_zip, mode="w", allowZip64=True) as zf:
            for file_path in files:
                rel = file_path.relative_to(dataset_dir).as_posix()
                arcname = f"{root_prefix}/{rel}"
                if file_path.suffix.lower() == ".mp4":
                    zf.write(file_path, arcname=arcname, compress_type=zipfile.ZIP_STORED)
                else:
                    zf.write(
                        file_path,
                        arcname=arcname,
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=6,
                    )

        with zipfile.ZipFile(tmp_zip, mode="r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"corrupt ZIP member detected in {tmp_zip}: {bad_member}")
            member_count = len(zf.namelist())
            if member_count != len(files):
                raise RuntimeError(f"ZIP member count {member_count} != file count {len(files)}")

        os.replace(tmp_zip, zip_path)
    except BaseException:
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except OSError:
                pass
        raise

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"corrupt ZIP member detected in {zip_path}: {bad_member}")
        infos = zf.infolist()
        member_count = len(infos)

    zip_size = zip_path.stat().st_size
    zip_crc32 = _crc32_file(zip_path)
    zip_sha256 = _sha256_file(zip_path)
    return {
        "zip_path": str(zip_path),
        "bytes": zip_size,
        "member_count": member_count,
        "testzip_passed": True,
        "crc32_hex": f"{zip_crc32:08x}",
        "sha256": zip_sha256,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--hub-url", default=DEFAULT_HUB_URL)
    parser.add_argument("--review-url", default=DEFAULT_REVIEW_URL)
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID)
    parser.add_argument("--skip-hub", action="store_true", help="Skip Site Hub stop/start checks")
    parser.add_argument("--skip-zip", action="store_true", help="Skip rebuilding the .zip archive")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    _safe_stdout_utf8()
    args = _parse_args(argv)

    source_dir = args.source_dir.resolve()
    stage_dir = args.stage_dir.resolve()
    backup_dir = args.backup_dir.resolve()
    zip_path = args.zip_path.resolve()

    if len({source_dir, stage_dir, backup_dir}) != 3:
        raise ValueError("source-dir, stage-dir, and backup-dir must all be distinct paths")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source-dir does not exist: {source_dir}")

    ffmpeg_exe = get_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg executable not available (install imageio-ffmpeg or ffmpeg)")

    if stage_dir.exists():
        print(f"Removing stale staging directory: {stage_dir}", flush=True)
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_clips(source_dir)
    trimmed_at_utc = _utc_now_iso()
    jobs = [
        {
            "src_dir": str(clip_dir),
            "dst_dir": str(stage_dir / behavior / clip_id),
            "behavior": behavior,
            "clip_id": clip_id,
            "ffmpeg_exe": ffmpeg_exe,
            "trimmed_at_utc": trimmed_at_utc,
        }
        for behavior, clip_id, clip_dir in discovered
    ]

    print(
        f"Starting padding trim on {len(jobs)} clips with workers={args.workers} "
        f"(ffmpeg={ffmpeg_exe})",
        flush=True,
    )
    t0 = time.monotonic()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_process_clip_job, job) for job in jobs]
        for idx, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if idx % 50 == 0 or idx == len(jobs):
                elapsed = time.monotonic() - t0
                print(f"  processed {idx}/{len(jobs)} clips ({elapsed:.1f}s)", flush=True)

    trimmed_count = sum(1 for r in results if r["status"] == "trimmed")
    kept_count = sum(1 for r in results if r["status"] == "kept_zero_padded")
    unchanged_rel_paths = {str(r["rel_path"]) for r in results if r["status"] == "kept_zero_padded"}
    print(
        f"Stage encoding complete: trimmed={trimmed_count}, kept_zero_padded={kept_count}, "
        f"total={len(results)}",
        flush=True,
    )

    _update_manifest(
        source_dir / "dataset_manifest.json",
        stage_dir / "dataset_manifest.json",
        results=results,
        trimmed_at_utc=trimmed_at_utc,
    )

    print("Running full verification across all staged clips (OpenCV decode + JSON contracts)...", flush=True)
    stage_verify = _verify_full_dataset(
        stage_dir,
        source_dir_for_unchanged=source_dir,
        unchanged_rel_paths=unchanged_rel_paths,
        workers=int(args.workers),
    )
    print(f"Stage verification passed: {json.dumps(stage_verify, ensure_ascii=False)}", flush=True)

    hub_health_before = None if args.skip_hub else _probe_hub(args.hub_url)
    if hub_health_before is not None:
        hub_base = args.hub_url.rstrip("/")
        token = str(hub_health_before["session_token"])
        print(f"Stopping Site Hub site '{args.site_id}' via {hub_base}...", flush=True)
        _http_post_json(f"{hub_base}/api/sites/{args.site_id}/stop", token=token)
        _wait_for_port_down(f"{args.review_url.rstrip('/')}/api/health", timeout_s=15.0)
    elif not args.skip_hub:
        print(f"Site Hub {args.hub_url} not running; skipping stop/start.", flush=True)

    print(f"Swapping {stage_dir} -> {source_dir}...", flush=True)
    _swap_directories(source_dir, stage_dir, backup_dir)
    print("Atomic directory swap completed and backup removed.", flush=True)

    hub_after: dict[str, Any] | None = None
    review_catalog_summary: dict[str, Any] | None = None
    if hub_health_before is not None:
        hub_base = args.hub_url.rstrip("/")
        hub_health_now = _probe_hub(args.hub_url)
        if hub_health_now is None:
            raise RuntimeError(f"Site Hub {hub_base} became unreachable during dataset swap")
        token_now = str(hub_health_now["session_token"])
        print(f"Starting Site Hub site '{args.site_id}' via {hub_base}...", flush=True)
        _http_post_json(f"{hub_base}/api/sites/{args.site_id}/start", token=token_now)
        review_health, catalog = _wait_for_review_ready(args.review_url, timeout_s=30.0)
        clips_list = catalog.get("clips", [])
        if not isinstance(clips_list, list) or len(clips_list) != 764:
            raise RuntimeError(
                f"unexpected /api/catalog clip count: {len(clips_list) if isinstance(clips_list, list) else type(clips_list)}"
            )
        non_zero_in_catalog = [
            c["clip_uid"]
            for c in clips_list
            if c.get("annotation_range") != [0, int(c.get("frame_count", 0)) - 1]
        ]
        if non_zero_in_catalog:
            raise RuntimeError(
                f"review site catalog still has non-zero-padded clips: {non_zero_in_catalog[:5]}"
            )
        _, hub_sites = _http_get_json(f"{hub_base}/api/sites", timeout_s=5.0)
        hub_after = {
            "hub_status": hub_health_now.get("status"),
            "hub_pid": hub_health_now.get("pid"),
            "summary": hub_sites.get("summary"),
        }
        review_catalog_summary = {
            "review_status": review_health.get("status"),
            "total_clips": catalog.get("total_clips"),
            "clips_len": len(clips_list),
            "behaviors_len": len(catalog.get("behaviors", [])),
            "all_zero_padded_in_catalog": True,
        }
        print(f"Review site verified: {json.dumps(review_catalog_summary, ensure_ascii=False)}", flush=True)

    zip_summary: dict[str, Any] | None = None
    if not args.skip_zip:
        print(f"Rebuilding ZIP archive {zip_path}...", flush=True)
        zip_summary = _rebuild_zip_archive(source_dir, zip_path)
        print(f"ZIP rebuild verified: {json.dumps(zip_summary, ensure_ascii=False)}", flush=True)

    final_report = {
        "trimmed_clips": trimmed_count,
        "kept_zero_padded_clips": kept_count,
        "total_clips": len(results),
        "stage_verification": stage_verify,
        "hub_after": hub_after,
        "review_catalog_summary": review_catalog_summary,
        "zip_summary": zip_summary,
    }
    print("FINAL_SUMMARY_JSON=" + json.dumps(final_report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
