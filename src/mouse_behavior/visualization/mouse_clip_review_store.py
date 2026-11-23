"""Read-only dataset store and atomic JSON ledgers for mouse clip review & collection.

Strictly treats ``source_dir`` (e.g. ``F:\\mouse``) as immutable read-only input.
All review decisions and collected clip packages are persisted exclusively under
a separate ``output_dir`` using atomic writes, preserving corrupt ledger files
intact rather than silently overwriting them.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import threading
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from mouse_behavior.evaluation.review_transaction import safe_path, write_json

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = Path("F:/mouse")
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "mouse_clip_review_site"

CATALOG_SCHEMA = "mouse_clip_catalog_v1"
DETAIL_SCHEMA = "mouse_clip_detail_v1"
REVIEWS_SCHEMA = "mouse_clip_reviews_v1"
COLLECTIONS_SCHEMA = "mouse_clip_collections_v1"

VALID_REVIEW_STATUSES = frozenset({"pending", "pass", "fail", "needs_recrop"})

_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_BAD_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_TRACKS_CACHE_ENTRIES = 8


class CorruptStoreError(RuntimeError):
    """Raised when a persisted JSON ledger exists but is unreadable or schema-invalid."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def sanitize_dir_segment(name: str) -> str:
    """Return a safe single path segment for Windows/POSIX output subdirectories."""
    cleaned = _WINDOWS_BAD_CHARS_RE.sub("_", str(name).strip()).strip(". ")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"invalid directory segment: {name!r}")
    return cleaned


def make_clip_uid(behavior: str, clip_id: str) -> str:
    """Create a deterministic URL-safe clip identifier without spaces or slashes."""
    b_slug = _SLUG_UNSAFE_RE.sub("_", behavior.strip()).strip("_") or "behavior"
    c_slug = _SLUG_UNSAFE_RE.sub("_", clip_id.strip()).strip("_") or "clip"
    return f"{b_slug}__{c_slug}"


def validate_output_dir_separate(source_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Ensure ``output_dir`` is neither equal to nor inside ``source_dir``."""
    src_resolved = Path(source_dir).resolve()
    out_resolved = Path(output_dir).resolve()
    if out_resolved == src_resolved or out_resolved.is_relative_to(src_resolved):
        raise ValueError(
            f"output_dir ({out_resolved}) must be strictly outside read-only source_dir ({src_resolved})"
        )
    return src_resolved, out_resolved


def _read_json_readonly(path: Path) -> Any:
    with path.open("rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8-sig"))


def _probe_video_header(video_path: Path) -> tuple[float, int, int, int]:
    """Fallback header probe when ``metadata.json`` is missing or incomplete."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 30.0, 0, 0, 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)))
        height = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
        frame_count = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)))
        return fps, width, height, frame_count
    finally:
        cap.release()


def _parse_annotation_range(annotation: Mapping[str, Any] | None, frame_count: int) -> list[int] | None:
    if not isinstance(annotation, Mapping):
        return None
    fr = annotation.get("frame_range")
    if not isinstance(fr, Mapping):
        return None
    start = fr.get("start")
    end = fr.get("end")
    if not (_is_strict_int(start) and _is_strict_int(end)):
        return None
    if start < 0 or end < start:
        return None
    if frame_count > 0 and end >= frame_count:
        end = frame_count - 1
    return [int(start), int(end)]


def _parse_target_mouse_ids(annotation: Mapping[str, Any] | None) -> list[int]:
    if not isinstance(annotation, Mapping):
        return []
    raw_ids = annotation.get("mouse_ids")
    if raw_ids is None:
        raw_ids = annotation.get("target_mouse_ids")
    if not isinstance(raw_ids, list):
        return []
    result: list[int] = []
    for item in raw_ids:
        if _is_strict_int(item):
            result.append(int(item))
    return result


class MouseClipReviewStore:
    """Read-only index for ``F:\\mouse`` clips plus writable review/collection ledgers."""

    def __init__(
        self,
        source_dir: Path = DEFAULT_SOURCE_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        src_resolved, out_resolved = validate_output_dir_separate(source_dir, output_dir)
        self.source_dir = src_resolved
        self.output_dir = out_resolved
        self.reviews_path = self.output_dir / "reviews.json"
        self.collections_path = self.output_dir / "collections_manifest.json"
        self.collected_clips_dir = self.output_dir / "collected_clips"

        self._catalog_lock = threading.RLock()
        self._reviews_lock = threading.RLock()
        self._collections_lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._catalog_loaded = False
        self._tracks_cache: OrderedDict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = OrderedDict()

    def scan_catalog(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        """Scan ``source_dir`` reading only ``annotation.json`` and ``metadata.json``."""
        with self._catalog_lock:
            if self._catalog_loaded and not force:
                return self._records
            if not self.source_dir.is_dir():
                self._records = {}
                self._catalog_loaded = True
                return self._records

            records: dict[str, dict[str, Any]] = {}
            used_uids: set[str] = set()

            for behavior_dir in sorted(self.source_dir.iterdir(), key=lambda p: p.name.lower()):
                if not behavior_dir.is_dir():
                    continue
                folder_behavior = behavior_dir.name
                entries = sorted(behavior_dir.iterdir(), key=lambda p: p.name.lower())
                for entry in entries:
                    if entry.is_dir():
                        video_path = entry / "clip.mp4"
                        if not video_path.is_file():
                            continue
                        clip_dir = entry
                        clip_id = entry.name
                        ann_path = clip_dir / "annotation.json"
                        meta_path = clip_dir / "metadata.json"
                        tracks_path = clip_dir / "tracks.json"
                    elif entry.is_file() and entry.suffix.lower() == ".mp4":
                        video_path = entry
                        clip_dir = behavior_dir
                        clip_id = entry.stem
                        ann_path = behavior_dir / f"{entry.stem}.annotation.json"
                        meta_path = behavior_dir / f"{entry.stem}.metadata.json"
                        tracks_path = behavior_dir / f"{entry.stem}.tracks.json"
                    else:
                        continue

                    has_annotation = ann_path.is_file()
                    has_metadata = meta_path.is_file()
                    has_tracks = tracks_path.is_file()

                    annotation: dict[str, Any] | None = None
                    if has_annotation:
                        try:
                            loaded_ann = _read_json_readonly(ann_path)
                            if isinstance(loaded_ann, dict):
                                annotation = loaded_ann
                        except Exception:
                            annotation = None

                    metadata: dict[str, Any] | None = None
                    if has_metadata:
                        try:
                            loaded_meta = _read_json_readonly(meta_path)
                            if isinstance(loaded_meta, dict):
                                metadata = loaded_meta
                        except Exception:
                            metadata = None

                    clip_meta = metadata.get("clip", {}) if isinstance(metadata, dict) else {}
                    fps = clip_meta.get("fps") if isinstance(clip_meta, dict) else None
                    width = clip_meta.get("width") if isinstance(clip_meta, dict) else None
                    height = clip_meta.get("height") if isinstance(clip_meta, dict) else None
                    frame_count = clip_meta.get("frame_count") if isinstance(clip_meta, dict) else None

                    valid_meta = (
                        isinstance(fps, (int, float))
                        and not isinstance(fps, bool)
                        and math.isfinite(float(fps))
                        and float(fps) > 0
                        and _is_strict_int(width)
                        and width > 0
                        and _is_strict_int(height)
                        and height > 0
                        and _is_strict_int(frame_count)
                        and frame_count > 0
                    )
                    if valid_meta:
                        fps_f = float(fps)
                        width_i = int(width)
                        height_i = int(height)
                        count_i = int(frame_count)
                    else:
                        fps_f, width_i, height_i, count_i = _probe_video_header(video_path)

                    behavior_name = (
                        str(annotation.get("behavior")).strip()
                        if annotation and isinstance(annotation.get("behavior"), str) and annotation.get("behavior").strip()
                        else folder_behavior
                    )
                    base_uid = make_clip_uid(folder_behavior, clip_id)
                    clip_uid = base_uid
                    if clip_uid in used_uids:
                        digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:8]
                        clip_uid = f"{base_uid}_{digest}"
                    used_uids.add(clip_uid)

                    source_info = metadata.get("source", {}) if isinstance(metadata, dict) else {}
                    source_video = str(source_info.get("video_filename", "")) if isinstance(source_info, dict) else ""
                    duration_s = round(count_i / fps_f, 4) if fps_f > 0 and count_i > 0 else 0.0

                    records[clip_uid] = {
                        "clip_uid": clip_uid,
                        "behavior": behavior_name,
                        "folder_behavior": folder_behavior,
                        "clip_id": clip_id,
                        "fps": fps_f,
                        "width": width_i,
                        "height": height_i,
                        "frame_count": count_i,
                        "duration_s": duration_s,
                        "annotation_range": _parse_annotation_range(annotation, count_i),
                        "target_mouse_ids": _parse_target_mouse_ids(annotation),
                        "source_video": source_video,
                        "has_tracks": has_tracks,
                        "has_annotation": has_annotation,
                        "has_metadata": has_metadata,
                        "clip_dir": clip_dir,
                        "video_path": video_path,
                        "annotation_path": ann_path if has_annotation else None,
                        "metadata_path": meta_path if has_metadata else None,
                        "tracks_path": tracks_path if has_tracks else None,
                    }

            self._records = records
            self._catalog_loaded = True
            self._tracks_cache.clear()
            return self._records

    def ensure_source_exists(self) -> None:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"source_dir does not exist: {self.source_dir}")

    def get_record(self, clip_uid: str) -> dict[str, Any]:
        self.ensure_source_exists()
        records = self.scan_catalog()
        if clip_uid not in records:
            raise KeyError(f"unknown clip_uid: {clip_uid}")
        return records[clip_uid]

    def get_catalog(self, *, force_rescan: bool = False) -> dict[str, Any]:
        """Build catalog summary enriched with review status and collection counts."""
        self.ensure_source_exists()
        records = self.scan_catalog(force=force_rescan)

        reviews_map: dict[str, dict[str, Any]] = {}
        try:
            reviews_map = self.load_reviews().get("reviews", {})
        except CorruptStoreError:
            reviews_map = {}

        collection_counts: dict[str, int] = {}
        try:
            for item in self.load_collections().get("collections", []):
                uid = str(item.get("clip_uid", ""))
                if uid:
                    collection_counts[uid] = collection_counts.get(uid, 0) + 1
        except CorruptStoreError:
            collection_counts = {}

        behavior_counts: dict[str, int] = {}
        clips: list[dict[str, Any]] = []
        for rec in records.values():
            b_name = rec["behavior"]
            behavior_counts[b_name] = behavior_counts.get(b_name, 0) + 1
            rev = reviews_map.get(rec["clip_uid"])
            rev_status = str(rev.get("status", "pending")) if isinstance(rev, dict) else "pending"
            clips.append(
                {
                    "clip_uid": rec["clip_uid"],
                    "behavior": b_name,
                    "clip_id": rec["clip_id"],
                    "fps": rec["fps"],
                    "width": rec["width"],
                    "height": rec["height"],
                    "frame_count": rec["frame_count"],
                    "duration_s": rec["duration_s"],
                    "annotation_range": rec["annotation_range"],
                    "target_mouse_ids": rec["target_mouse_ids"],
                    "source_video": rec["source_video"],
                    "has_tracks": rec["has_tracks"],
                    "has_annotation": rec["has_annotation"],
                    "has_metadata": rec["has_metadata"],
                    "review_status": rev_status,
                    "collection_count": collection_counts.get(rec["clip_uid"], 0),
                }
            )

        behaviors = [
            {"name": name, "count": count}
            for name, count in sorted(behavior_counts.items(), key=lambda kv: kv[0].lower())
        ]
        return {
            "schema": CATALOG_SCHEMA,
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "total_clips": len(clips),
            "behaviors": behaviors,
            "clips": clips,
        }

    def load_tracks_for_clip(self, clip_uid: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load measured ``tracks.json`` frames and visible-track summary (LRU cached)."""
        rec = self.get_record(clip_uid)
        with self._catalog_lock:
            cached = self._tracks_cache.get(clip_uid)
            if cached is not None:
                self._tracks_cache.move_to_end(clip_uid)
                return cached

        tracks_path: Path | None = rec.get("tracks_path")
        frames: list[dict[str, Any]] = []
        track_stats: dict[int, dict[str, int]] = {}

        if tracks_path is not None and tracks_path.is_file():
            safe_tracks = safe_path(tracks_path, [self.source_dir])
            raw = _read_json_readonly(safe_tracks)
            if not isinstance(raw, list):
                raise ValueError(f"tracks.json is not a frame list: {safe_tracks}")
            fps = rec["fps"] or 30.0
            for item in raw:
                if not isinstance(item, dict):
                    continue
                f_idx = item.get("frame")
                if not _is_strict_int(f_idx) or f_idx < 0:
                    continue
                raw_time = item.get("time")
                t_val = (
                    float(raw_time)
                    if isinstance(raw_time, (int, float))
                    and not isinstance(raw_time, bool)
                    and math.isfinite(float(raw_time))
                    else round(f_idx / fps, 6)
                )
                dets_out: list[dict[str, Any]] = []
                raw_dets = item.get("detections")
                if isinstance(raw_dets, list):
                    for det in raw_dets:
                        if not isinstance(det, dict):
                            continue
                        tid = det.get("track_id")
                        box = det.get("box")
                        if (
                            not _is_strict_int(tid)
                            or not isinstance(box, list)
                            or len(box) != 4
                            or not all(
                                isinstance(v, (int, float))
                                and not isinstance(v, bool)
                                and math.isfinite(float(v))
                                for v in box
                            )
                        ):
                            continue
                        tid_i = int(tid)
                        det_entry: dict[str, Any] = {
                            "track_id": tid_i,
                            "box": [float(v) for v in box],
                        }
                        kpts = det.get("keypoints")
                        if isinstance(kpts, list):
                            det_entry["keypoints"] = kpts
                        det_conf = det.get("detection_confidence")
                        if isinstance(det_conf, (int, float)) and not isinstance(det_conf, bool):
                            det_entry["detection_confidence"] = float(det_conf)
                        if _is_strict_int(det.get("class_id")):
                            det_entry["class_id"] = int(det["class_id"])
                        dets_out.append(det_entry)

                        stat = track_stats.get(tid_i)
                        if stat is None:
                            track_stats[tid_i] = {
                                "track_id": tid_i,
                                "frame_count": 1,
                                "first_frame": int(f_idx),
                                "last_frame": int(f_idx),
                            }
                        else:
                            stat["frame_count"] += 1
                            if f_idx < stat["first_frame"]:
                                stat["first_frame"] = int(f_idx)
                            if f_idx > stat["last_frame"]:
                                stat["last_frame"] = int(f_idx)

                frames.append(
                    {
                        "frame": int(f_idx),
                        "time": t_val,
                        "detections": dets_out,
                    }
                )

        visible_tracks = [track_stats[tid] for tid in sorted(track_stats)]
        result = (frames, visible_tracks)
        with self._catalog_lock:
            self._tracks_cache[clip_uid] = result
            self._tracks_cache.move_to_end(clip_uid)
            while len(self._tracks_cache) > _MAX_TRACKS_CACHE_ENTRIES:
                self._tracks_cache.popitem(last=False)
        return result

    def read_clip_json_files(self, clip_uid: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Read ``annotation.json`` and ``metadata.json`` strictly read-only."""
        rec = self.get_record(clip_uid)
        annotation: dict[str, Any] | None = None
        metadata: dict[str, Any] | None = None

        ann_path: Path | None = rec.get("annotation_path")
        if ann_path is not None and ann_path.is_file():
            safe_ann = safe_path(ann_path, [self.source_dir])
            loaded = _read_json_readonly(safe_ann)
            if isinstance(loaded, dict):
                annotation = loaded

        meta_path: Path | None = rec.get("metadata_path")
        if meta_path is not None and meta_path.is_file():
            safe_meta = safe_path(meta_path, [self.source_dir])
            loaded = _read_json_readonly(safe_meta)
            if isinstance(loaded, dict):
                metadata = loaded

        return annotation, metadata

    def get_clip_detail(self, clip_uid: str) -> dict[str, Any]:
        """Return full clip metadata, measured tracks, review state, and collections."""
        rec = self.get_record(clip_uid)
        annotation, metadata = self.read_clip_json_files(clip_uid)
        frames, visible_tracks = self.load_tracks_for_clip(clip_uid)

        review: dict[str, Any] | None = None
        try:
            review = self.load_reviews().get("reviews", {}).get(clip_uid)
        except CorruptStoreError:
            review = None

        collections: list[dict[str, Any]] = []
        try:
            collections = [
                item
                for item in self.load_collections().get("collections", [])
                if item.get("clip_uid") == clip_uid
            ]
        except CorruptStoreError:
            collections = []

        clip_summary = {
            "clip_uid": rec["clip_uid"],
            "behavior": rec["behavior"],
            "clip_id": rec["clip_id"],
            "fps": rec["fps"],
            "width": rec["width"],
            "height": rec["height"],
            "frame_count": rec["frame_count"],
            "duration_s": rec["duration_s"],
            "annotation_range": rec["annotation_range"],
            "target_mouse_ids": rec["target_mouse_ids"],
            "source_video": rec["source_video"],
            "has_tracks": rec["has_tracks"],
            "has_annotation": rec["has_annotation"],
            "has_metadata": rec["has_metadata"],
            "review_status": str(review.get("status", "pending")) if isinstance(review, dict) else "pending",
            "collection_count": len(collections),
        }

        return {
            "schema": DETAIL_SCHEMA,
            "clip": clip_summary,
            "annotation": annotation,
            "metadata": metadata,
            "video": {
                "url": f"/api/video/{clip_uid}",
                "fps": rec["fps"],
                "frame_count": rec["frame_count"],
                "width": rec["width"],
                "height": rec["height"],
            },
            "frames": frames,
            "visible_tracks": visible_tracks,
            "review": review,
            "collections": collections,
        }

    def get_video_path(self, clip_uid: str) -> Path:
        """Return validated read-only path to a clip's ``clip.mp4``."""
        rec = self.get_record(clip_uid)
        video_path = safe_path(rec["video_path"], [self.source_dir])
        if not video_path.is_file():
            raise FileNotFoundError(f"clip video not found: {video_path}")
        return video_path

    # ------------------------------------------------------------------
    # Reviews Ledger (atomic writes, corrupt-file preservation)
    # ------------------------------------------------------------------

    def _load_reviews_unlocked(self) -> dict[str, Any]:
        if not self.reviews_path.exists():
            return {
                "schema": REVIEWS_SCHEMA,
                "updated_at": None,
                "reviews": {},
            }
        try:
            raw = self.reviews_path.read_bytes().decode("utf-8")
            data = json.loads(raw)
        except Exception as exc:
            raise CorruptStoreError(f"corrupt reviews file {self.reviews_path}: {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("reviews"), dict):
            raise CorruptStoreError(f"invalid reviews schema in {self.reviews_path}")
        return {
            "schema": REVIEWS_SCHEMA,
            "updated_at": data.get("updated_at"),
            "reviews": dict(data["reviews"]),
        }

    def load_reviews(self) -> dict[str, Any]:
        with self._reviews_lock:
            return self._load_reviews_unlocked()

    def save_review(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and atomically persist one clip review decision."""
        if not isinstance(payload, Mapping):
            raise ValueError("review payload must be a JSON object")

        clip_uid = payload.get("clip_uid")
        if not isinstance(clip_uid, str) or not clip_uid.strip():
            raise ValueError("clip_uid is required")
        clip_uid = clip_uid.strip()
        rec = self.get_record(clip_uid)

        status = payload.get("status")
        if not isinstance(status, str) or status not in VALID_REVIEW_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_REVIEW_STATUSES)}")

        raw_tags = payload.get("tags", [])
        if raw_tags is None:
            raw_tags = []
        if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
            raise ValueError("tags must be a list of strings")
        tags: list[str] = []
        for t in raw_tags[:32]:
            cleaned = t.strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned[:120])

        raw_note = payload.get("note", "")
        if raw_note is None:
            raw_note = ""
        if not isinstance(raw_note, str):
            raise ValueError("note must be a string")
        note = raw_note[:4000]

        raw_crop_range = payload.get("crop_range")
        crop_range: list[int] | None = None
        if raw_crop_range is not None:
            if (
                not isinstance(raw_crop_range, list)
                or len(raw_crop_range) != 2
                or not _is_strict_int(raw_crop_range[0])
                or not _is_strict_int(raw_crop_range[1])
            ):
                raise ValueError("crop_range must be null or [start_frame, end_frame] integers")
            s_frame, e_frame = int(raw_crop_range[0]), int(raw_crop_range[1])
            max_frames = rec["frame_count"]
            if s_frame < 0 or e_frame <= s_frame or (max_frames > 0 and e_frame > max_frames):
                raise ValueError(
                    f"crop_range [{s_frame}, {e_frame}) out of bounds for frame_count={max_frames}"
                )
            crop_range = [s_frame, e_frame]

        now = _utc_now_iso()
        entry = {
            "status": status,
            "tags": tags,
            "note": note,
            "crop_range": crop_range,
            "updated_at": now,
        }

        with self._reviews_lock:
            current = self._load_reviews_unlocked()
            current["reviews"][clip_uid] = entry
            current["updated_at"] = now
            self.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.reviews_path, current)

        return {"status": "ok", "clip_uid": clip_uid, "review": entry}

    def export_reviews_csv(self) -> bytes:
        """Render all stored reviews as UTF-8-BOM CSV bytes."""
        data = self.load_reviews()
        reviews = data["reviews"]
        records = self.scan_catalog() if self.source_dir.is_dir() else {}

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "clip_uid",
                "behavior",
                "clip_id",
                "source_video",
                "status",
                "tags",
                "crop_start_frame",
                "crop_end_frame",
                "note",
                "updated_at",
            ]
        )
        for clip_uid, rev in sorted(reviews.items()):
            rec = records.get(clip_uid, {})
            crop_range = rev.get("crop_range") if isinstance(rev, dict) else None
            c_start = crop_range[0] if isinstance(crop_range, list) and len(crop_range) == 2 else ""
            c_end = crop_range[1] if isinstance(crop_range, list) and len(crop_range) == 2 else ""
            tags = rev.get("tags", []) if isinstance(rev, dict) else []
            writer.writerow(
                [
                    clip_uid,
                    rec.get("behavior", ""),
                    rec.get("clip_id", ""),
                    rec.get("source_video", ""),
                    rev.get("status", "pending") if isinstance(rev, dict) else "pending",
                    ";".join(str(t) for t in tags),
                    c_start,
                    c_end,
                    rev.get("note", "") if isinstance(rev, dict) else "",
                    rev.get("updated_at", "") if isinstance(rev, dict) else "",
                ]
            )
        return ("\ufeff" + buf.getvalue()).encode("utf-8")

    # ------------------------------------------------------------------
    # Collections Manifest (atomic writes, corrupt-file preservation)
    # ------------------------------------------------------------------

    def _load_collections_unlocked(self) -> dict[str, Any]:
        if not self.collections_path.exists():
            return {
                "schema": COLLECTIONS_SCHEMA,
                "output_dir": str(self.collected_clips_dir),
                "updated_at": None,
                "total": 0,
                "collections": [],
            }
        try:
            raw = self.collections_path.read_bytes().decode("utf-8")
            data = json.loads(raw)
        except Exception as exc:
            raise CorruptStoreError(f"corrupt collections manifest {self.collections_path}: {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("collections"), list):
            raise CorruptStoreError(f"invalid collections manifest schema in {self.collections_path}")

        items = [dict(item) for item in data["collections"] if isinstance(item, dict)]
        return {
            "schema": COLLECTIONS_SCHEMA,
            "output_dir": str(self.collected_clips_dir),
            "updated_at": data.get("updated_at"),
            "total": len(items),
            "collections": items,
        }

    def load_collections(self) -> dict[str, Any]:
        with self._collections_lock:
            return self._load_collections_unlocked()

    def ensure_collections_writable(self) -> None:
        """Fail fast with ``CorruptStoreError`` before running expensive FFmpeg export."""
        with self._collections_lock:
            self._load_collections_unlocked()

    def record_collection(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        """Insert or update a collection record atomically in ``collections_manifest.json``."""
        export_id = str(entry.get("export_id", "")).strip()
        if not export_id:
            raise ValueError("collection entry missing export_id")

        with self._collections_lock:
            current = self._load_collections_unlocked()
            updated_list = [c for c in current["collections"] if c.get("export_id") != export_id]
            record = dict(entry)
            updated_list.append(record)
            payload = {
                "schema": COLLECTIONS_SCHEMA,
                "output_dir": str(self.collected_clips_dir),
                "updated_at": _utc_now_iso(),
                "total": len(updated_list),
                "collections": updated_list,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.collections_path, payload)
        return record

    def get_collection(self, export_id: str) -> dict[str, Any]:
        manifest = self.load_collections()
        for item in manifest["collections"]:
            if item.get("export_id") == export_id:
                return item
        raise KeyError(f"unknown export_id: {export_id}")

    def delete_collection(self, export_id: str) -> dict[str, Any]:
        """Safely remove an exported package strictly inside ``output_dir/collected_clips``."""
        with self._collections_lock:
            current = self._load_collections_unlocked()
            target_entry = next(
                (c for c in current["collections"] if c.get("export_id") == export_id),
                None,
            )
            if target_entry is None:
                raise KeyError(f"unknown export_id: {export_id}")

            raw_dir = target_entry.get("output_dir")
            if not isinstance(raw_dir, str) or not raw_dir.strip():
                raise ValueError("collection entry has invalid output_dir")

            self.collected_clips_dir.mkdir(parents=True, exist_ok=True)
            target_path = safe_path(Path(raw_dir), [self.collected_clips_dir])
            resolved_root = self.collected_clips_dir.resolve()
            resolved_target = target_path.resolve()
            if resolved_target == resolved_root or resolved_target.parent == resolved_root:
                raise ValueError(f"refusing to delete non-package directory: {resolved_target}")
            if resolved_target.is_relative_to(self.source_dir):
                raise ValueError(f"refusing to touch read-only source_dir: {resolved_target}")

            if resolved_target.exists():
                shutil.rmtree(resolved_target)

            remaining = [c for c in current["collections"] if c.get("export_id") != export_id]
            payload = {
                "schema": COLLECTIONS_SCHEMA,
                "output_dir": str(self.collected_clips_dir),
                "updated_at": _utc_now_iso(),
                "total": len(remaining),
                "collections": remaining,
            }
            write_json(self.collections_path, payload)
        return {"status": "ok", "export_id": export_id}

    def get_exported_video_path(self, export_id: str, variant: str = "raw") -> Path:
        """Resolve an exported clip video (``raw`` or ``overlay``) inside ``collected_clips``."""
        if variant not in {"raw", "overlay"}:
            raise ValueError("variant must be 'raw' or 'overlay'")
        entry = self.get_collection(export_id)
        raw_dir = entry.get("output_dir")
        if not isinstance(raw_dir, str) or not raw_dir.strip():
            raise FileNotFoundError(f"collection directory missing for {export_id}")
        filename = "clip.mp4" if variant == "raw" else "clip_overlay.mp4"
        candidate = Path(raw_dir) / filename
        validated = safe_path(candidate, [self.collected_clips_dir])
        if not validated.is_file():
            raise FileNotFoundError(f"exported video not found: {validated}")
        return validated

    def export_collections_csv(self) -> bytes:
        """Render all collected clip packages as UTF-8-BOM CSV bytes."""
        manifest = self.load_collections()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "export_id",
                "clip_uid",
                "source_behavior",
                "source_clip_id",
                "behavior",
                "mouse_ids",
                "start_frame",
                "end_frame",
                "clip_start_frame",
                "clip_end_frame",
                "padding_frames",
                "frame_count",
                "duration_s",
                "fps",
                "has_overlay",
                "output_dir",
                "note",
                "created_at",
            ]
        )
        for item in manifest["collections"]:
            mids = item.get("mouse_ids", [])
            writer.writerow(
                [
                    item.get("export_id", ""),
                    item.get("clip_uid", ""),
                    item.get("source_behavior", ""),
                    item.get("source_clip_id", ""),
                    item.get("behavior", ""),
                    ";".join(str(m) for m in mids),
                    item.get("start_frame", ""),
                    item.get("end_frame", ""),
                    item.get("clip_start_frame", ""),
                    item.get("clip_end_frame", ""),
                    item.get("padding_frames", 0),
                    item.get("frame_count", ""),
                    item.get("duration_s", ""),
                    item.get("fps", ""),
                    bool(item.get("overlay_video_url")),
                    item.get("output_dir", ""),
                    item.get("note", ""),
                    item.get("created_at", ""),
                ]
            )
        return ("\ufeff" + buf.getvalue()).encode("utf-8")
