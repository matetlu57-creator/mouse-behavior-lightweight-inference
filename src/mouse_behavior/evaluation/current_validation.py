"""Read-only current individual-dataset audit and exact visual-ID metrics."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .blind_top1 import LabeledClip, Top1Prediction, evaluate_top1
from .confirmed_dataset import file_sha256, inherited_sessions
from .exported_dataset_models import INDIVIDUAL_BEHAVIORS, LABEL_TO_CANONICAL
from .scale_migration import rescale_frozen_source_scale


REQUIRED_SAMPLE_FILES = (
    "annotation.json", "metadata.json", "tracks.json", "clip.mp4", "review_receipt.json",
)


def resolved_input_path(value: str | Path, repository: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def assert_hashes_unchanged(hashes: dict[str, str]) -> None:
    """Never seal a run whose inputs, code or model changed during execution."""
    for path, expected in hashes.items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"locked validation input/source changed: {path}")


def verified_source_scale(
    row: dict[str, Any], repository: Path, approved_root: Path, hashes: dict[str, str],
    *, active_rig_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    receipt_path = resolved_input_path(row["scale_receipt_path"], repository)
    receipt_hash = file_sha256(receipt_path)
    if receipt_hash != row["scale_receipt_sha256"]:
        raise ValueError(f"source-scale receipt hash differs: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("scope") != "complete_original_export_backup_not_reviewed_crop"
            or receipt.get("contains_ground_truth") is not False):
        raise ValueError(f"scale is not label-free original-source calibration: {receipt_path}")
    source = resolved_input_path(receipt["source_tracks_path"], repository)
    expected_source = (approved_root / "backups" / row["source_key"] / "sample" / "tracks.json").resolve()
    if source != expected_source:
        raise ValueError(f"scale receipt source-key/path mismatch: {receipt_path}")
    source_hash = file_sha256(source)
    if (source_hash != receipt["source_tracks_sha256"]
            or source_hash != row["original_tracks_sha256"]):
        raise ValueError(f"original source tracks differ from scale/crop provenance: {source}")
    crop = resolved_input_path(row["tracks_path"], repository)
    if file_sha256(crop) not in receipt.get("crop_tracks_sha256", []):
        raise ValueError(f"source-scale receipt belongs to another crop: {crop}")
    scale = dict(receipt["scale_config"])
    value = float(scale.get("cm_per_pixel", 0))
    if scale.get("mode") != "fixed" or not math.isfinite(value) or value <= 0:
        raise ValueError(f"scale receipt has no fixed positive cm/pixel: {receipt_path}")
    drift = []
    active_rig = (Path(active_rig_path).resolve() if active_rig_path is not None else
                  (repository / "configs" / "scale" / "rig_a.json").resolve())
    for text, expected_hash in receipt["scale_asset_sha256"].items():
        asset = resolved_input_path(text, repository)
        actual_hash = file_sha256(asset)
        if actual_hash != expected_hash:
            # The already frozen cm/pixel does not depend on later behavior
            # settings in default.yaml. Rig or any other asset drift fails.
            if asset == (repository / "configs" / "default.yaml").resolve():
                drift.append(str(asset))
            elif asset == (repository / "configs" / "scale" / "rig_a.json").resolve():
                scale, migration = rescale_frozen_source_scale(scale, active_rig)
                drift.append(
                    f"{asset} migrated by factor {migration['factor']:.12g}; "
                    f"active rig: {active_rig}"
                )
            else:
                raise ValueError(f"frozen scale asset changed: {asset}")
        hashes[str(asset)] = actual_hash
    hashes[str(receipt_path)] = receipt_hash
    hashes[str(source)] = source_hash
    hashes[str(crop)] = file_sha256(crop)
    return scale, drift


def prepare_individual_rows(
    *, manifest: Path, decisions_path: Path, train_manifest: Path, val_manifest: Path,
    scale_directory: Path, repository: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extend the inherited session split to all available latest approved postures.

    Missing data or absent scale receipts are disclosed separately. Existing
    receipts with mismatching provenance are rejected, never regenerated.
    """
    sessions = inherited_sessions(train_manifest, val_manifest)
    inherited: dict[str, dict[str, str]] = {}
    for path in (train_manifest, val_manifest):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sample = row["sample_id"]
                if sample in inherited:
                    raise ValueError(f"duplicate sample in inherited split: {sample}")
                inherited[sample] = row
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("latest approved manifest must be a list")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))["decisions"]
    hashes = {str(path.resolve()): file_sha256(path) for path in (
        manifest, decisions_path, train_manifest, val_manifest,
    )}
    rows: list[dict[str, Any]] = []
    unavailable = []
    seen: set[str] = set()
    source_splits: dict[str, str] = {}
    support: Counter[str] = Counter()
    checked_files = 0
    scale_drift: dict[str, list[str]] = {}
    for entry in entries:
        if entry.get("behavior") not in {"Locomotion", "Static", "Grooming", "Rearing"}:
            continue
        canonical = LABEL_TO_CANONICAL.get(entry["behavior"])
        if canonical not in INDIVIDUAL_BEHAVIORS:
            raise ValueError(f"current individual label is not enabled: {entry['behavior']}")
        sample = str(entry.get("review_id", ""))
        if not sample or sample in seen:
            raise ValueError(f"missing/duplicate latest individual sample: {sample!r}")
        seen.add(sample)
        support[canonical] += 1
        decision = decisions.get(sample, {})
        target_ids = tuple(sorted(int(value) for value in entry["mouse_ids"]))
        if (entry.get("decision") != "approved" or decision.get("decision") != "approved"
                or decision.get("behavior") != entry["behavior"]
                or tuple(sorted(int(value) for value in decision.get("mouse_ids", []))) != target_ids
                or len(target_ids) != 1):
            raise ValueError(f"latest individual manifest/manual decision disagrees: {sample}")
        sample_dir = resolved_input_path(entry["dataset_path"], repository)
        absent = [name for name in REQUIRED_SAMPLE_FILES if not (sample_dir / name).is_file()]
        if absent:
            unavailable.append({"sample_id": sample, "behavior": canonical,
                                "reason": "missing_approved_sample_files", "missing": absent,
                                "dataset_path": str(sample_dir), "split": "unknown"})
            continue
        expected_hashes = entry.get("dataset_sha256", {})
        for name in REQUIRED_SAMPLE_FILES:
            path = sample_dir / name
            digest = file_sha256(path)
            if digest != expected_hashes.get(name):
                raise ValueError(f"approved dataset file hash differs: {path}")
            hashes[str(path)] = digest
            checked_files += 1
        annotation = json.loads((sample_dir / "annotation.json").read_text(encoding="utf-8"))
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        if (annotation.get("behavior") != entry["behavior"]
                or tuple(sorted(int(value) for value in annotation.get("mouse_ids", []))) != target_ids
                or annotation.get("manual_review_id") != sample
                or annotation.get("requires_review") is not False):
            raise ValueError(f"individual annotation differs from approval: {sample}")
        fps = float(metadata["clip"]["fps"])
        frames = int(metadata["clip"]["frame_count"])
        if (not math.isfinite(fps) or fps <= 0 or frames <= 0
                or annotation["frame_range"] != {"start": 0, "end": frames - 1}
                or metadata["source"]["end_frame"] - metadata["source"]["start_frame"] + 1 != frames):
            raise ValueError(f"invalid approved individual frame range/FPS: {sample}")
        session = re.sub(r"_part\d+(?=\.mp4$)", "", metadata["source"]["video_filename"], flags=re.I)
        if session not in sessions:
            raise ValueError(f"no inherited recording-session assignment: {session}")
        split = sessions[session]
        source_key = str(entry["source_key"])
        if source_splits.get(source_key, split) != split:
            raise ValueError(f"same original source occurs on both split sides: {source_key}")
        source_splits[source_key] = split
        old = inherited.get(sample)
        row = {
            "sample_id": sample, "tracks_path": str(sample_dir / "tracks.json"),
            "fps": fps, "frame_count": frames, "recording_session": session,
            "canonical_behavior": canonical, "mouse_ids": ",".join(map(str, target_ids)),
            "split": split, "source_key": source_key,
            "source_sample_id": metadata["provenance"]["source_sample_id"],
            "original_tracks_sha256": metadata["provenance"]["source_sha256"]["tracks.json"],
        }
        if old is not None:
            if (old["canonical_behavior"] != canonical
                    or old["mouse_ids"] != row["mouse_ids"]
                    or old["recording_session"] != session or old["split"] != split
                    or resolved_input_path(old["tracks_path"], repository) != Path(row["tracks_path"])):
                raise ValueError(f"inherited individual row differs from latest approval: {sample}")
            row["scale_receipt_path"] = str(resolved_input_path(old["scale_receipt_path"], repository))
            row["scale_receipt_sha256"] = old["scale_receipt_sha256"]
        else:
            receipt_path = (scale_directory / f"{source_key}.json").resolve()
            row["scale_receipt_path"] = str(receipt_path)
            row["scale_receipt_sha256"] = file_sha256(receipt_path) if receipt_path.is_file() else ""
        if not Path(row["scale_receipt_path"]).is_file():
            unavailable.append({**row, "reason": "missing_verified_source_scale_receipt"})
            continue
        _, drift = verified_source_scale(
            row, repository, manifest.parent, hashes,
            active_rig_path=Path(__file__).resolve().parents[3] / "configs" / "scale" / "rig_a.json",
        )
        if drift:
            scale_drift[sample] = drift
        rows.append(row)
    approved_ids = seen
    stale = [row["sample_id"] for row in inherited.values()
             if row["canonical_behavior"] in INDIVIDUAL_BEHAVIORS
             and decisions.get(row["sample_id"], {}).get("decision") == "approved"
             and row["sample_id"] not in approved_ids]
    if stale:
        raise ValueError(f"inherited approved individual IDs absent from latest manifest: {stale[:8]}")
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    audit = {
        "schema": "latest_approved_individual_dataset_audit_v1",
        "approved_individual_samples": len(seen), "eligible_samples": len(rows),
        "approved_behavior_support": dict(sorted(support.items())),
        "eligible_split_behavior_support": {split: dict(sorted(Counter(
            row["canonical_behavior"] for row in rows if row["split"] == split).items()))
            for split in ("train", "val")},
        "unavailable_samples": unavailable, "approved_sample_file_hash_checks": checked_files,
        "new_samples_assigned_by_existing_recording_session": [
            row["sample_id"] for row in rows if row["sample_id"] not in inherited],
        "scale_fitted": False, "scale_receipts_reused": len(rows),
        "allowed_frozen_scale_default_config_drift": scale_drift,
        "frozen_scale_rig_migration": {
            "factor": 7.918 / 10.7,
            "arena_cm": [74.0, 74.0],
            "applied_to_receipt_count": sum(
                any("migrated by factor" in item for item in drift)
                for drift in scale_drift.values()
            ),
            "receipts_modified": False,
        },
        "session_overlap": [], "source_overlap": [], "labels_modified": False,
        "input_sha256": hashes,
    }
    return train, val, audit


def exact_individual_metrics(
    rows: Sequence[tuple[LabeledClip, Top1Prediction]],
) -> dict[str, Any]:
    """Wrong visual IDs and unrecognized outputs are false negatives."""
    strict = evaluate_top1(rows)
    matrix = Counter((label.behavior, prediction.behavior or "unrecognized")
                     for label, prediction in rows)
    per_behavior = {}
    for behavior in INDIVIDUAL_BEHAVIORS:
        support = sum(label.behavior == behavior for label, _ in rows)
        predicted = sum(prediction.behavior == behavior for _, prediction in rows)
        tp = sum(label.behavior == behavior == prediction.behavior
                 and tuple(sorted(label.target_ids)) == tuple(sorted(prediction.target_ids))
                 for label, prediction in rows)
        per_behavior[behavior] = {
            "support": support, "predicted": predicted, "joint_true_positive": tp,
            "false_positive": predicted - tp, "false_negative": support - tp,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "strict_accuracy": tp / support if support else None,
        }
    total = len(rows)
    predicted = sum(item["predicted"] for item in per_behavior.values())
    tp = sum(item["joint_true_positive"] for item in per_behavior.values())
    return {
        "scope": "closed_set_individual_exact_behavior_and_visual_id_set",
        "n_validation_clips": total, "strict_accuracy": tp / total if total else None,
        "joint_micro_precision": tp / predicted if predicted else None,
        "joint_micro_recall": tp / total if total else None,
        "unrecognized": sum(prediction.behavior is None for _, prediction in rows),
        "correct_behavior_wrong_visual_ids": sum(
            label.behavior == prediction.behavior
            and tuple(sorted(label.target_ids)) != tuple(sorted(prediction.target_ids))
            for label, prediction in rows),
        "per_behavior": per_behavior, "strict_top1": strict,
        "confusion": {truth: {pred: matrix[(truth, pred)]
                               for pred in (*INDIVIDUAL_BEHAVIORS, "unrecognized")}
                      for truth in INDIVIDUAL_BEHAVIORS},
    }
