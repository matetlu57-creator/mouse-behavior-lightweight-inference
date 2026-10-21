"""Read-only confirmed-data audit, inherited splits and source-scale receipts.

Human labels are never repaired or inferred. All writes belong to a new output
directory; full original export backups provide scale, never the reviewed crop.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mouse_behavior.config import load_config
from mouse_behavior.preprocessing.scale_resolution import resolve_scale, robust_reference_body_px
from .exported_dataset_models import LABEL_TO_CANONICAL


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_reference_pixels(path: Path) -> float:
    """Same robust per-track body estimator as the adapter, without labels."""
    lengths: dict[int, list[float]] = defaultdict(list)
    for frame in json.loads(path.read_text(encoding="utf-8-sig")):
        for detection in frame.get("detections", []):
            points = np.asarray(detection.get("keypoints", []), dtype=float)
            if points.shape != (7, 3) or "track_id" not in detection:
                continue
            endpoints = points[[0, 6]]
            if not np.isfinite(endpoints).all() or np.min(endpoints[:, 2]) < 0.25:
                continue
            length = float(np.linalg.norm(endpoints[0, :2] - endpoints[1, :2]))
            if 10 <= length <= 300:
                lengths[int(detection["track_id"])].append(length)
    if not lengths:
        raise ValueError(f"source has no measurable nose-tail body length: {path}")
    return robust_reference_body_px([float(np.median(values)) for values in lengths.values()])


def load_scale_receipt(path: str, digest: str, tracks_path: str) -> dict[str, Any] | None:
    if not path:
        if digest:
            raise ValueError("scale digest provided without receipt")
        return None
    if not digest or file_sha256(path) != digest:
        raise ValueError("source-scale receipt hash changed or is missing")
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if file_sha256(receipt["source_tracks_path"]) != receipt["source_tracks_sha256"]:
        raise ValueError("source trajectory changed after scale freeze")
    if file_sha256(tracks_path) not in receipt["crop_tracks_sha256"]:
        raise ValueError("source-scale receipt belongs to another crop")
    for asset, asset_digest in receipt["scale_asset_sha256"].items():
        if file_sha256(asset) != asset_digest:
            raise ValueError(f"source-scale asset changed: {asset}")
    return receipt["scale_config"]


def inherited_sessions(train_path: Path, val_path: Path) -> dict[str, str]:
    sessions: dict[str, str] = {}
    sources: dict[str, str] = {}
    for path, expected in ((train_path, "train"), (val_path, "val")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] not in ({"train"} if expected == "train" else {"val", "validation"}):
                    raise ValueError("original manifest contains an unexpected split")
                session, source = row["recording_session"], row["sample_id"]
                if not session or not source:
                    raise ValueError("original split is missing session/source provenance")
                if sessions.get(session, expected) != expected or sources.get(source, expected) != expected:
                    raise ValueError("original train/validation sessions or sources overlap")
                sessions[session] = sources[source] = expected
    return sessions


def prepare_confirmed_dataset(manifest: Path, train_path: Path, val_path: Path,
                              config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new audit output directory")
    entries = json.loads(manifest.read_text(encoding="utf-8-sig"))
    sessions = inherited_sessions(train_path, val_path)
    scale_cfg = dict(load_config(config_path)["scale"])
    assets = {str(config_path.resolve()): file_sha256(config_path)}
    if scale_cfg.get("calibration_file"):
        from mouse_behavior.preprocessing.scale_resolution import resolve_calibration_path
        calibration_path = resolve_calibration_path(scale_cfg["calibration_file"])
        assets[str(calibration_path)] = file_sha256(calibration_path)
    output.mkdir(parents=True)
    (output / "scales").mkdir()
    rows, posture, missing, warnings = [], [], [], []
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frozen_hashes: dict[str, str] = {str(manifest.resolve()): file_sha256(manifest),
                                  str(train_path.resolve()): file_sha256(train_path),
                                  str(val_path.resolve()): file_sha256(val_path)}
    seen: set[str] = set()
    required = {"annotation.json", "metadata.json", "tracks.json", "clip.mp4", "review_receipt.json"}
    for entry in entries:
        sample = entry["review_id"]
        if sample in seen or entry.get("decision") != "approved":
            raise ValueError("repeated review ID or unapproved dataset entry")
        seen.add(sample)
        directory = Path(entry["dataset_path"])
        absent = sorted(name for name in required if not (directory / name).is_file())
        if absent:
            missing.append({"review_id": sample, "path": str(directory), "missing": absent})
            continue
        for name in required:
            path = directory / name
            digest = file_sha256(path)
            if digest != entry["dataset_sha256"].get(name):
                raise ValueError(f"confirmed file hash differs from manifest: {path}")
            frozen_hashes[str(path)] = digest
        annotation = json.loads((directory / "annotation.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (annotation["behavior"] != entry["behavior"]
                or sorted(annotation["mouse_ids"]) != sorted(entry["mouse_ids"])
                or annotation.get("requires_review") is not False
                or annotation.get("manual_review_id") != sample):
            raise ValueError(f"annotation disagrees with human confirmation: {sample}")
        fps, frames = float(metadata["clip"]["fps"]), int(metadata["clip"]["frame_count"])
        if (not math.isfinite(fps) or fps <= 0 or frames <= 0
                or annotation["frame_range"] != {"start": 0, "end": frames - 1}
                or metadata["source"]["end_frame"] - metadata["source"]["start_frame"] + 1 != frames):
            raise ValueError(f"invalid confirmed frame/FPS range: {sample}")
        session = re.sub(r"_part\d+(?=\.mp4$)", "", metadata["source"]["video_filename"],
                         flags=re.IGNORECASE)
        if session not in sessions:
            raise ValueError(f"no inherited split for recording session: {session}")
        if entry.get("note"):
            warnings.append({"review_id": sample, "human_note": entry["note"]})
        canonical = LABEL_TO_CANONICAL.get(entry["behavior"])
        row = {"sample_id": sample, "tracks_path": str(directory / "tracks.json"), "fps": fps,
               "frame_count": frames, "recording_session": session, "canonical_behavior": canonical,
               "mouse_ids": ",".join(map(str, sorted(entry["mouse_ids"]))), "split": sessions[session],
               "source_key": entry["source_key"], "source_sample_id": metadata["provenance"]["source_sample_id"]}
        row["original_tracks_sha256"] = metadata["provenance"]["source_sha256"]["tracks.json"]
        if canonical is None:
            if entry["behavior"] not in {"Grooming", "Rearing"}:
                raise ValueError(f"unsupported human label: {entry['behavior']}")
            posture.append({**row, "behavior": entry["behavior"], "status": "audit_only_detector_disabled"})
        else:
            rows.append(row)
        source_groups[entry["source_key"]].append(row)
    for key, crops in sorted(source_groups.items()):
        if len({row["split"] for row in crops}) != 1:
            raise ValueError("windows from the same source occur on both split sides")
        source = manifest.parent / "backups" / key / "sample" / "tracks.json"
        digest = file_sha256(source)
        if any(row["original_tracks_sha256"] != digest for row in crops):
            raise ValueError(f"original source backup hash differs from provenance: {key}")
        reference_px = source_reference_pixels(source)
        resolved = resolve_scale(scale_cfg, reference_body_px=reference_px)
        receipt_path = output / "scales" / f"{key}.json"
        write_receipt(receipt_path, {"schema": "source_scale_v1", "source_tracks_path": str(source),
            "source_tracks_sha256": digest, "reference_body_px": reference_px,
            "crop_tracks_sha256": sorted({file_sha256(row["tracks_path"]) for row in crops}),
            "scale_asset_sha256": assets, "scale_config": {"mode": "fixed",
                "cm_per_pixel": resolved.cm_per_pixel, "assumed_mouse_body_length_cm": resolved.reference_body_cm,
                "assumed_mouse_body_width_cm": resolved.reference_width_cm},
            "scope": "complete_original_export_backup_not_reviewed_crop", "contains_ground_truth": False})
        frozen_hashes[str(source)] = digest
        for row in crops:
            row["scale_receipt_path"] = str(receipt_path.resolve())
            row["scale_receipt_sha256"] = file_sha256(receipt_path)
    fields = tuple(rows[0]) if rows else ()
    if not fields:
        raise ValueError("no eligible confirmed samples")
    for split in ("train", "val"):
        with (output / f"{split}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
    report = {"manifest_entries": len(entries), "available": len(rows) + len(posture),
              "eligible": len(rows), "missing": missing, "posture_audit_only": posture,
              "warnings": warnings, "counts": dict(Counter(row["canonical_behavior"] for row in rows)),
              "split_counts": dict(Counter(row["split"] for row in rows)),
              "source_scale_count": len(source_groups), "input_sha256": frozen_hashes,
              "original_split_sha256": {str(p): file_sha256(p) for p in (train_path, val_path)},
              "hash_scope": "all_five_confirmed_files_and_original_source_tracks",
              "media_decode_checked": False, "labels_modified": False}
    write_receipt(output / "audit.json", report)
    return report
