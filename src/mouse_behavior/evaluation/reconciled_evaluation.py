"""Reconciled evaluation and failure diagnosis across Dataset A (429) and Dataset B (605).

Obeys all repository contracts:
- Visual tracking (`visual_track_id` / `track_id`) comes solely from exported
  `tracks.json`; identities are never inferred from position, appearance,
  pose, or motion.
- Always reports strict blind `target-ID joint` (`pf_joint`) accuracy alongside
  target-pair conditioned (`pf_pair`) and specialized `clip_benchmark` accuracy.
- Never grants dual-label credit between `Together` and contact labels
  (`Snout-head_contact` / `Snout-rear_contact`).
- Leaves `DEFAULT_SOCIAL_HEURISTIC_CONFIG` unchanged and evaluates social rules
  with `OPTIMIZED_SOCIAL_HEURISTIC_CONFIG`.
- Preserves canonical defaults unless ``all_behaviors=True`` explicitly enables
  posture evidence for every clip. Exact-ID frame coverage and strict blind
  Top-1 remain separate from the mixed, target-conditioned primary methods.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG,
    individual_rule_masks,
)
from mouse_behavior.config import load_config
from mouse_behavior.evaluation.blind_top1 import candidates_from_evidence, select_top1
from mouse_behavior.evaluation.clip_benchmark import (
    evaluate_attack_chasing_clip,
    evaluate_six_behavior_clip,
)
from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks, social_profiles
from mouse_behavior.evaluation.event_frames import interval_frame_count, mask_intervals
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_features import load_exported_features
from mouse_behavior.evaluation.exported_dataset_metrics import classify_target_ids
from mouse_behavior.evaluation.exported_dataset_models import (
    ExportedFeatures,
    HeuristicParameters,
    LABEL_TO_CANONICAL,
    SampleRecord,
)
from mouse_behavior.evaluation.exported_dataset_prediction import predict_features
from mouse_behavior.evaluation.group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG
from mouse_behavior.evaluation.group_event_rules import isolation_candidates
from mouse_behavior.evaluation.social_event_geometry import (
    body_footprint_distances,
    endpoint_distances,
)
from mouse_behavior.evaluation.social_heuristic_config import (
    OPTIMIZED_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_DATASET_A_ROOT = Path(r"F:\补全行为素材429_261107\四行为训练素材429\dataset")
DEFAULT_DATASET_B_ROOT = Path(r"E:\鼠行为数据集_人工清洗完成_605段_20261024\dataset")
DEFAULT_OUTPUT_JSON = REPO_ROOT / "outputs" / "reconciled_behavior_evaluation.json"
DEFAULT_DATASET_B_SOURCE_SCALES: Path = (
    REPO_ROOT
    / "outputs"
    / "latest_code_repository_20261014"
    / "outputs"
    / "heuristics_v17_20261016_run01"
    / "dataset_statistics_v2"
    / "source_scales.json"
)

SOCIAL_DISPLAY_BEHAVIORS: tuple[str, ...] = (
    "Approach",
    "Attack",
    "Chasing",
    "Snout-head_contact",
    "Snout-rear_contact",
    "Together",
)
INDIVIDUAL_DISPLAY_BEHAVIORS: tuple[str, ...] = (
    "Static",
    "Locomotion",
    "Grooming",
    "Rearing",
)
GROUP_DISPLAY_BEHAVIORS: tuple[str, ...] = (
    "Isolation",
    "Group locomotion",
    "Dispersal",
    "Social clustering",
    "Huddling",
)
BEHAVIOR_DISPLAY_ORDER: tuple[str, ...] = (
    *SOCIAL_DISPLAY_BEHAVIORS,
    *INDIVIDUAL_DISPLAY_BEHAVIORS,
    *GROUP_DISPLAY_BEHAVIORS,
)
SUSPENDED_DISPLAY_BEHAVIORS: frozenset[str] = frozenset({"Grooming", "Rearing"})

CANONICAL_TO_DISPLAY: dict[str, str] = {
    "approach": "Approach",
    "attack": "Attack",
    "chase": "Chasing",
    "chasing": "Chasing",
    "nose_head_contact": "Snout-head_contact",
    "head_contact": "Snout-head_contact",
    "nose_tail_contact": "Snout-rear_contact",
    "tail_contact": "Snout-rear_contact",
    "together": "Together",
    "stationary": "Static",
    "static": "Static",
    "locomotion": "Locomotion",
    "grooming": "Grooming",
    "rearing": "Rearing",
    "isolation": "Isolation",
    "group_locomotion": "Group locomotion",
    "dispersal": "Dispersal",
    "social_clustering": "Social clustering",
    "huddle": "Huddling",
    "huddling": "Huddling",
}
DISPLAY_TO_CANONICAL: dict[str, str] = {
    display: LABEL_TO_CANONICAL[display] for display in BEHAVIOR_DISPLAY_ORDER
}
DISPLAY_BEHAVIOR_ALIASES: dict[str, str] = {
    "Moving": "Locomotion",
}
_DISPLAY_BEHAVIOR_LOOKUP: dict[str, str] = {
    **{name.casefold(): name for name in BEHAVIOR_DISPLAY_ORDER},
    **{alias.casefold(): target for alias, target in DISPLAY_BEHAVIOR_ALIASES.items()},
}


def normalize_display_behavior(raw_behavior: str) -> str:
    """Normalize a display/folder behavior label, mapping ``Moving`` to ``Locomotion``."""
    cleaned = str(raw_behavior or "").strip()
    if not cleaned:
        return cleaned
    if cleaned in DISPLAY_TO_CANONICAL:
        return cleaned
    if cleaned in DISPLAY_BEHAVIOR_ALIASES:
        return DISPLAY_BEHAVIOR_ALIASES[cleaned]
    mapped = _DISPLAY_BEHAVIOR_LOOKUP.get(cleaned.casefold())
    return mapped if mapped is not None else cleaned

SOCIAL_MASK_TO_CANONICAL: dict[str, str] = {
    "together": "together",
    "attack": "attack",
    "chasing": "chase",
    "approach": "approach",
    "head_contact": "nose_head_contact",
    "tail_contact": "nose_tail_contact",
}
CANONICAL_TO_SOCIAL_MASK: dict[str, str] = {v: k for k, v in SOCIAL_MASK_TO_CANONICAL.items()}

PRIMARY_METHOD_BY_BEHAVIOR: dict[str, str] = {
    "Attack": "clip_benchmark_attack_chasing",
    "Chasing": "clip_benchmark_attack_chasing",
    "Approach": "predict_features_pair_conditioned",
    "Snout-head_contact": "predict_features_pair_conditioned",
    "Snout-rear_contact": "predict_features_pair_conditioned",
    "Together": "predict_features_pair_conditioned",
    "Static": "clip_benchmark_six_behavior",
    "Locomotion": "clip_benchmark_six_behavior",
    "Isolation": "clip_benchmark_six_behavior",
    "Group locomotion": "clip_benchmark_six_behavior",
    "Dispersal": "clip_benchmark_six_behavior",
    "Social clustering": "clip_benchmark_six_behavior",
    "Huddling": "predict_features_blind_joint",
    "Grooming": "canonical_default_disabled",
    "Rearing": "canonical_default_disabled",
}

FAILURE_SUBCATEGORIES: tuple[str, ...] = (
    "disabled_rule",
    "target_id_missing_in_tracks",
    "no_candidate_threshold_unmet",
    "priority_or_duration_outranked",
    "wrong_behavior_triggered",
    "other_pair_won_joint",
)

_WORKER_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class ClipIndexEntry:
    """Indexed clip metadata from one exported dataset."""

    dataset: str
    behavior: str
    canonical_behavior: str
    clip_id: str
    clip_dir: str
    annotation_behavior: str
    mouse_ids: tuple[int, ...]
    frame_range_inclusive: dict[str, int]
    frame_range_half_open: dict[str, int]
    participants: tuple[dict[str, Any], ...]
    confidence: str
    manual_review_id: str | None
    fps: float
    frame_count: int
    width: int
    height: int
    schema_version: str
    source_video: str
    source_start_frame: int | None
    source_end_frame: int | None
    tracks_path: str
    tracks_frame_count: int
    observed_track_ids: tuple[int, ...]
    missing_target_ids: tuple[int, ...]
    tracks_sha256: str
    has_video: bool
    metadata_cm_per_pixel: float | None = None
    metadata_scale_source: str | None = None


def _clean_float(value: Any, *, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return round(num, digits)


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        num = float(value)
        return num if math.isfinite(num) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_participants(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        return ()
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        items.append(
            {
                "role_name": str(entry.get("role_name", "")),
                "role_key": str(entry.get("role_key", "")),
                "track_ids": sorted(int(t) for t in entry.get("track_ids", [])),
            }
        )
    items.sort(key=lambda item: (item["role_name"], item["track_ids"]))
    return tuple(items)


def _behavior_layer(display_behavior: str) -> str:
    if display_behavior in SOCIAL_DISPLAY_BEHAVIORS:
        return "social"
    if display_behavior in INDIVIDUAL_DISPLAY_BEHAVIORS:
        return "individual"
    return "group"


def _load_worker_context(config_path: Path) -> tuple[HeuristicParameters, HeuristicParameters]:
    key = str(config_path.resolve())
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        return cached
    params = config_seed_parameters(load_config(config_path))
    opt_params = social_profiles(params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)[0]
    _WORKER_CACHE[key] = (params, opt_params)
    return params, opt_params


def _extract_metadata_distance_scale(metadata: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Read optional calibrated ``distance_scale`` / ``cm_per_px`` from clip metadata."""
    distance_scale = metadata.get("distance_scale")
    raw_val: Any = None
    if isinstance(distance_scale, Mapping):
        raw_val = distance_scale.get("cm_per_pixel", distance_scale.get("cm_per_px"))
    if raw_val is None:
        raw_val = metadata.get("cm_per_pixel", metadata.get("cm_per_px"))
    if raw_val is None:
        return None, None
    try:
        parsed = float(raw_val)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None, None
    return parsed, "metadata_distance_scale"


def index_dataset(dataset_root: Path, dataset_label: str) -> list[ClipIndexEntry]:
    """Index every clip folder under ``dataset_root/<behavior>/<clip_id>``."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    entries: list[ClipIndexEntry] = []
    seen_ids: dict[str, str] = {}
    for beh_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        folder_behavior = beh_dir.name
        normalized_behavior = normalize_display_behavior(folder_behavior)
        if normalized_behavior not in DISPLAY_TO_CANONICAL:
            raise ValueError(f"Unsupported behavior directory {folder_behavior!r} in {root}")
        canonical_behavior = DISPLAY_TO_CANONICAL[normalized_behavior]

        for clip_dir in sorted((p for p in beh_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            clip_id = clip_dir.name
            if clip_id in seen_ids:
                raise ValueError(
                    f"Duplicate clip_id {clip_id!r} in dataset {dataset_label}: "
                    f"{seen_ids[clip_id]} vs {folder_behavior}"
                )
            seen_ids[clip_id] = folder_behavior

            anno_path = clip_dir / "annotation.json"
            meta_path = clip_dir / "metadata.json"
            tracks_path = clip_dir / "tracks.json"
            if not anno_path.is_file() or not meta_path.is_file() or not tracks_path.is_file():
                raise FileNotFoundError(f"Missing required clip files in {clip_dir}")

            annotation = json.loads(anno_path.read_text(encoding="utf-8"))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

            anno_behavior = str(annotation.get("behavior", folder_behavior))
            meta_cm_per_px, meta_scale_source = _extract_metadata_distance_scale(metadata)
            mouse_ids = tuple(sorted({int(m) for m in annotation.get("mouse_ids", [])}))
            frame_range = annotation.get("frame_range", {})
            gt_start = int(frame_range.get("start", 0))
            gt_end_inc = int(frame_range.get("end", max(len(raw_tracks) - 1, 0)))
            clip_meta = metadata.get("clip", {})
            fps_val = float(clip_meta.get("fps", 30.0))
            if not math.isfinite(fps_val) or fps_val <= 0:
                fps_val = 30.0
            frame_count = int(clip_meta.get("frame_count", len(raw_tracks)))
            gt_end_ho = gt_end_inc + 1 if gt_end_inc >= gt_start else frame_count
            source_meta = metadata.get("source", {})
            if isinstance(source_meta.get("clip"), Mapping):
                source_clip = source_meta["clip"]
            else:
                source_clip = source_meta

            observed_ids = sorted(
                {
                    int(det["track_id"])
                    for frame in raw_tracks
                    for det in frame.get("detections", [])
                    if "track_id" in det
                }
            )
            observed_set = set(observed_ids)
            missing_targets = tuple(mid for mid in mouse_ids if mid not in observed_set)

            entries.append(
                ClipIndexEntry(
                    dataset=dataset_label,
                    behavior=normalized_behavior,
                    canonical_behavior=canonical_behavior,
                    clip_id=clip_id,
                    clip_dir=str(clip_dir),
                    annotation_behavior=anno_behavior,
                    mouse_ids=mouse_ids,
                    frame_range_inclusive={"start": gt_start, "end": gt_end_inc},
                    frame_range_half_open={"start": gt_start, "end": gt_end_ho},
                    participants=_normalize_participants(annotation.get("participants")),
                    confidence=str(annotation.get("confidence", "")),
                    manual_review_id=(
                        str(annotation["manual_review_id"])
                        if annotation.get("manual_review_id") is not None
                        else None
                    ),
                    fps=fps_val,
                    frame_count=frame_count,
                    width=int(clip_meta.get("width", 0)),
                    height=int(clip_meta.get("height", 0)),
                    schema_version=str(metadata.get("schema_version", "")),
                    source_video=str(source_meta.get("video_filename", "")),
                    source_start_frame=(
                        int(source_clip["start_frame"])
                        if source_clip.get("start_frame") is not None
                        else None
                    ),
                    source_end_frame=(
                        int(source_clip["end_frame"])
                        if source_clip.get("end_frame") is not None
                        else None
                    ),
                    tracks_path=str(tracks_path),
                    tracks_frame_count=len(raw_tracks),
                    observed_track_ids=tuple(observed_ids),
                    missing_target_ids=missing_targets,
                    tracks_sha256=_sha256_file(tracks_path),
                    has_video=(clip_dir / "clip.mp4").is_file(),
                    metadata_cm_per_pixel=meta_cm_per_px,
                    metadata_scale_source=meta_scale_source,
                )
            )
    return entries


def reconcile_datasets(
    entries_a: Sequence[ClipIndexEntry],
    entries_b: Sequence[ClipIndexEntry],
) -> dict[str, Any]:
    """Compare Dataset A and Dataset B clip IDs, labels, and independent files."""
    by_a = {entry.clip_id: entry for entry in entries_a}
    by_b = {entry.clip_id: entry for entry in entries_b}
    ids_a = set(by_a)
    ids_b = set(by_b)
    overlap_ids = sorted(ids_a & ids_b)
    union_ids = sorted(ids_a | ids_b)
    a_only_ids = sorted(ids_a - ids_b)
    b_only_ids = sorted(ids_b - ids_a)

    label_conflicts: list[dict[str, Any]] = []
    overlap_file_diffs: dict[str, int] = {
        "tracks_bytes_differ": 0,
        "tracks_parsed_differ": 0,
        "tracks_detections_differ": 0,
        "annotation_differ": 0,
        "participants_differ": 0,
        "mouse_ids_differ": 0,
        "frame_range_differ": 0,
        "metadata_differ": 0,
        "label_conflicts": 0,
    }
    overlap_details: list[dict[str, Any]] = []

    for cid in overlap_ids:
        a_entry = by_a[cid]
        b_entry = by_b[cid]
        a_dir = Path(a_entry.clip_dir)
        b_dir = Path(b_entry.clip_dir)

        tracks_bytes_diff = a_entry.tracks_sha256 != b_entry.tracks_sha256
        ta = json.loads((a_dir / "tracks.json").read_text(encoding="utf-8"))
        tb = json.loads((b_dir / "tracks.json").read_text(encoding="utf-8"))
        tracks_parsed_diff = ta != tb
        tracks_det_diff = len(ta) != len(tb) or any(
            fa.get("detections") != fb.get("detections") for fa, fb in zip(ta, tb)
        )

        aa = json.loads((a_dir / "annotation.json").read_text(encoding="utf-8"))
        ab = json.loads((b_dir / "annotation.json").read_text(encoding="utf-8"))
        anno_diff = aa != ab

        ma = json.loads((a_dir / "metadata.json").read_text(encoding="utf-8"))
        mb = json.loads((b_dir / "metadata.json").read_text(encoding="utf-8"))
        meta_diff = ma != mb

        participants_diff = a_entry.participants != b_entry.participants
        mouse_ids_diff = a_entry.mouse_ids != b_entry.mouse_ids
        frame_range_diff = a_entry.frame_range_inclusive != b_entry.frame_range_inclusive
        label_conflict = a_entry.behavior != b_entry.behavior

        overlap_file_diffs["tracks_bytes_differ"] += int(tracks_bytes_diff)
        overlap_file_diffs["tracks_parsed_differ"] += int(tracks_parsed_diff)
        overlap_file_diffs["tracks_detections_differ"] += int(tracks_det_diff)
        overlap_file_diffs["annotation_differ"] += int(anno_diff)
        overlap_file_diffs["participants_differ"] += int(participants_diff)
        overlap_file_diffs["mouse_ids_differ"] += int(mouse_ids_diff)
        overlap_file_diffs["frame_range_differ"] += int(frame_range_diff)
        overlap_file_diffs["metadata_differ"] += int(meta_diff)
        overlap_file_diffs["label_conflicts"] += int(label_conflict)

        if label_conflict:
            label_conflicts.append(
                {
                    "clip_id": cid,
                    "dataset_a_behavior": a_entry.behavior,
                    "dataset_b_behavior": b_entry.behavior,
                    "dataset_a_mouse_ids": list(a_entry.mouse_ids),
                    "dataset_b_mouse_ids": list(b_entry.mouse_ids),
                    "note": (
                        "Label conflict across Dataset A and Dataset B; shared clip IDs also "
                        "retain independent track/annotation/metadata files."
                    ),
                }
            )

        overlap_details.append(
            {
                "clip_id": cid,
                "dataset_a_behavior": a_entry.behavior,
                "dataset_b_behavior": b_entry.behavior,
                "label_conflict": label_conflict,
                "tracks_bytes_differ": tracks_bytes_diff,
                "tracks_parsed_differ": tracks_parsed_diff,
                "tracks_detections_differ": tracks_det_diff,
                "annotation_differ": anno_diff,
                "participants_differ": participants_diff,
                "metadata_differ": meta_diff,
            }
        )

    return {
        "dataset_a_total": len(entries_a),
        "dataset_b_total": len(entries_b),
        "overlap_count": len(overlap_ids),
        "union_count": len(union_ids),
        "a_only_count": len(a_only_ids),
        "b_only_count": len(b_only_ids),
        "overlap_ids": overlap_ids,
        "a_only_ids": a_only_ids,
        "b_only_ids": b_only_ids,
        "label_agreement_count": len(overlap_ids) - len(label_conflicts),
        "label_agreement_rate": (
            round((len(overlap_ids) - len(label_conflicts)) / len(overlap_ids), 6)
            if overlap_ids
            else 1.0
        ),
        "label_conflicts": label_conflicts,
        "overlap_file_differences": overlap_file_diffs,
        "independent_files_note": (
            "Shared clip_ids between Dataset A and Dataset B point to independent "
            "directories with independent tracks.json, annotation.json, and metadata.json files."
        ),
        "overlap_details": overlap_details,
    }


def _padded_features(features: ExportedFeatures, frame_count: int) -> ExportedFeatures:
    loaded = len(features.valid)
    if loaded >= frame_count:
        return features
    from dataclasses import fields, replace

    updates: dict[str, Any] = {}
    for field in fields(features):
        value = getattr(features, field.name)
        if field.name in {
            "track_ids",
            "pair_i",
            "pair_j",
            "body_length_ref_cm",
            "body_width_ref_cm",
            "body_area_ref_cm2",
            "track_presence_fraction",
        }:
            continue
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == loaded:
            fill = False if value.dtype.kind == "b" else np.nan
            updates[field.name] = np.concatenate(
                (
                    value,
                    np.full((frame_count - loaded, *value.shape[1:]), fill, dtype=value.dtype),
                )
            )
    return replace(features, **updates)


def _find_edge_index(features: ExportedFeatures, target_ids: Sequence[int]) -> int | None:
    if len(target_ids) != 2:
        return None
    target_pair = tuple(sorted(int(x) for x in target_ids))
    for edge_idx, (i, j) in enumerate(zip(features.pair_i, features.pair_j)):
        pair = tuple(sorted((int(features.track_ids[int(i)]), int(features.track_ids[int(j)]))))
        if pair == target_pair:
            return edge_idx
    return None


def _intervals_from_mask(mask: np.ndarray) -> list[dict[str, int]]:
    return [{"start": int(s), "end": int(e)} for s, e in mask_intervals(np.asarray(mask, bool))]


def _merge_interval_runs(
    per_behavior_intervals: Mapping[str, list[dict[str, int]]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for display_name, intervals in per_behavior_intervals.items():
        can_name = DISPLAY_TO_CANONICAL.get(display_name, display_name)
        for iv in intervals:
            rows.append(
                {
                    "start": int(iv["start"]),
                    "end": int(iv["end"]),
                    "behavior": display_name,
                    "canonical_behavior": can_name,
                    "source": source,
                }
            )
    rows.sort(key=lambda item: (item["start"], item["end"], item["behavior"]))
    return rows


def _extract_target_kinematics(
    features: ExportedFeatures,
    target_ids: Sequence[int],
    total_frames: int,
) -> dict[str, Any]:
    track_to_slot = {int(tid): slot for slot, tid in enumerate(features.track_ids.tolist())}
    target_slots = [track_to_slot[tid] for tid in target_ids if tid in track_to_slot]
    missing_ids = [int(tid) for tid in target_ids if tid not in track_to_slot]

    observed_ratio_per_target: dict[str, float | None] = {}
    observed_frames_per_target: dict[str, int] = {}
    for tid in target_ids:
        slot = track_to_slot.get(tid)
        if slot is None:
            observed_frames_per_target[str(tid)] = 0
            observed_ratio_per_target[str(tid)] = 0.0
        else:
            cnt = int(np.count_nonzero(features.valid[:, slot]))
            observed_frames_per_target[str(tid)] = cnt
            observed_ratio_per_target[str(tid)] = _clean_float(cnt / max(total_frames, 1))

    if len(target_slots) == len(target_ids) and target_slots:
        coobserved = np.all(features.valid[:, target_slots], axis=1)
    elif target_slots:
        coobserved = np.any(features.valid[:, target_slots], axis=1)
    else:
        coobserved = np.zeros(len(features.valid), dtype=bool)
    coobserved_frames = int(np.count_nonzero(coobserved))

    missing_nose_frac = None
    missing_tail_frac = None
    missing_head_frac = None
    if target_slots and features.collective_keypoints_cm is not None:
        kp = features.collective_keypoints_cm[:, target_slots]  # (T, K, 7, 2)
        obs = features.valid[:, target_slots]
        total_obs = int(np.count_nonzero(obs))
        if total_obs > 0:
            nose_ok = np.isfinite(kp[:, :, 0, :]).all(axis=2) & obs
            tail_ok = np.isfinite(kp[:, :, 6, :]).all(axis=2) & obs
            head_ok = np.isfinite(kp[:, :, :3, :]).all(axis=(2, 3)) & obs
            missing_nose_frac = _clean_float(1.0 - np.count_nonzero(nose_ok) / total_obs)
            missing_tail_frac = _clean_float(1.0 - np.count_nonzero(tail_ok) / total_obs)
            missing_head_frac = _clean_float(1.0 - np.count_nonzero(head_ok) / total_obs)

    speed_vals: list[np.ndarray] = []
    per_target_speed_median: dict[str, float | None] = {}
    per_target_speed_max: dict[str, float | None] = {}
    motion_valid = features.motion_valid if features.motion_valid is not None else features.valid
    for tid in target_ids:
        slot = track_to_slot.get(tid)
        if slot is None:
            per_target_speed_median[str(tid)] = None
            per_target_speed_max[str(tid)] = None
            continue
        mask = features.valid[:, slot] & motion_valid[:, slot] & np.isfinite(features.speed_cm_s[:, slot])
        vals = features.speed_cm_s[mask, slot]
        if vals.size:
            speed_vals.append(vals)
            per_target_speed_median[str(tid)] = _clean_float(np.median(vals), digits=3)
            per_target_speed_max[str(tid)] = _clean_float(np.max(vals), digits=3)
        else:
            per_target_speed_median[str(tid)] = None
            per_target_speed_max[str(tid)] = None

    pooled_speed = np.concatenate(speed_vals) if speed_vals else np.array([], dtype=float)
    nose_accel_vals: list[np.ndarray] = []
    if features.nose_acceleration_cm_s2 is not None:
        for slot in target_slots:
            acc = features.nose_acceleration_cm_s2[:, slot]
            fin = acc[features.valid[:, slot] & np.isfinite(acc)]
            if fin.size:
                nose_accel_vals.append(fin)
    pooled_accel = np.concatenate(nose_accel_vals) if nose_accel_vals else np.array([], dtype=float)

    summary: dict[str, Any] = {
        "total_frames": int(total_frames),
        "fps": _clean_float(features.fps, digits=4),
        "cm_per_pixel": _clean_float(features.cm_per_pixel, digits=6),
        "reference_body_cm": _clean_float(features.reference_body_cm, digits=4),
        "selected_track_count": int(features.selected_track_count),
        "missing_target_ids": missing_ids,
        "observed_frames_per_target": observed_frames_per_target,
        "observed_ratio_per_target": observed_ratio_per_target,
        "coobserved_frames": coobserved_frames,
        "coobserved_ratio": _clean_float(coobserved_frames / max(total_frames, 1)),
        "missing_nose_fraction": missing_nose_frac,
        "missing_tail_fraction": missing_tail_frac,
        "missing_head_fraction": missing_head_frac,
        "speed_median_cm_s": _clean_float(np.median(pooled_speed), digits=3) if pooled_speed.size else None,
        "speed_p90_cm_s": _clean_float(np.quantile(pooled_speed, 0.9), digits=3) if pooled_speed.size else None,
        "speed_max_cm_s": _clean_float(np.max(pooled_speed), digits=3) if pooled_speed.size else None,
        "per_target_speed_median_cm_s": per_target_speed_median,
        "per_target_speed_max_cm_s": per_target_speed_max,
        "nose_accel_p90_cm_s2": _clean_float(np.quantile(pooled_accel, 0.9), digits=2) if pooled_accel.size else None,
        "nose_accel_max_cm_s2": _clean_float(np.max(pooled_accel), digits=2) if pooled_accel.size else None,
    }

    if len(target_ids) == 2 and not missing_ids:
        edge = _find_edge_index(features, target_ids)
        if edge is not None:
            dist = features.pair_distance_cm[:, edge]
            dist_fin = dist[coobserved & np.isfinite(dist)]
            summary["center_distance_min_cm"] = _clean_float(np.min(dist_fin), digits=3) if dist_fin.size else None
            summary["center_distance_median_cm"] = _clean_float(np.median(dist_fin), digits=3) if dist_fin.size else None
            summary["center_distance_max_cm"] = _clean_float(np.max(dist_fin), digits=3) if dist_fin.size else None
            summary["center_distance_start_cm"] = _clean_float(dist_fin[0], digits=3) if dist_fin.size else None
            summary["center_distance_end_cm"] = _clean_float(dist_fin[-1], digits=3) if dist_fin.size else None

            head_ltr, head_rtl = endpoint_distances(features, edge, "head")
            head_min_pair = np.fmin(head_ltr, head_rtl)
            head_fin = head_min_pair[coobserved & np.isfinite(head_min_pair)]
            summary["nose_head_min_cm"] = _clean_float(np.min(head_fin), digits=3) if head_fin.size else None
            summary["nose_head_median_cm"] = _clean_float(np.median(head_fin), digits=3) if head_fin.size else None
            summary["nose_head_max_cm"] = _clean_float(np.max(head_fin), digits=3) if head_fin.size else None

            tail_ltr, tail_rtl = endpoint_distances(features, edge, "tail")
            tail_min_pair = np.fmin(tail_ltr, tail_rtl)
            tail_fin = tail_min_pair[coobserved & np.isfinite(tail_min_pair)]
            summary["nose_tail_min_cm"] = _clean_float(np.min(tail_fin), digits=3) if tail_fin.size else None
            summary["nose_tail_median_cm"] = _clean_float(np.median(tail_fin), digits=3) if tail_fin.size else None
            summary["nose_tail_max_cm"] = _clean_float(np.max(tail_fin), digits=3) if tail_fin.size else None

            if (
                features.pair_nose_nearest_left_to_right_cm is not None
                and features.pair_nose_nearest_right_to_left_cm is not None
            ):
                nn = np.fmin(
                    features.pair_nose_nearest_left_to_right_cm[:, edge],
                    features.pair_nose_nearest_right_to_left_cm[:, edge],
                )
                nn_fin = nn[coobserved & np.isfinite(nn)]
                summary["nose_nearest_min_cm"] = _clean_float(np.min(nn_fin), digits=3) if nn_fin.size else None
                summary["nose_nearest_median_cm"] = _clean_float(np.median(nn_fin), digits=3) if nn_fin.size else None

            closing = features.pair_closing_speed_cm_s[:, edge]
            close_fin = closing[coobserved & np.isfinite(closing)]
            summary["closing_speed_median_cm_s"] = _clean_float(np.median(close_fin), digits=3) if close_fin.size else None
            summary["closing_speed_max_cm_s"] = _clean_float(np.max(close_fin), digits=3) if close_fin.size else None

            dir_sim = features.pair_direction_similarity[:, edge]
            dir_fin = dir_sim[coobserved & np.isfinite(dir_sim)]
            summary["direction_similarity_median"] = _clean_float(np.median(dir_fin), digits=3) if dir_fin.size else None

            body_gap = body_footprint_distances(features, edge, max_body_lengths=2.5)
            gap_fin = body_gap[coobserved & np.isfinite(body_gap)]
            summary["body_footprint_gap_min_cm"] = _clean_float(np.min(gap_fin), digits=3) if gap_fin.size else None
            summary["body_footprint_gap_median_cm"] = _clean_float(np.median(gap_fin), digits=3) if gap_fin.size else None

    elif len(target_ids) == 1 and not missing_ids:
        slot = target_slots[0]
        obs_pts = features.centers_cm[features.valid[:, slot], slot]
        summary["net_displacement_cm"] = (
            _clean_float(np.linalg.norm(obs_pts[-1] - obs_pts[0]), digits=3)
            if len(obs_pts) >= 2
            else None
        )
        nnd = features.nearest_distance_cm[:, slot]
        nnd_fin = nnd[features.valid[:, slot] & np.isfinite(nnd)]
        summary["nearest_neighbor_min_cm"] = _clean_float(np.min(nnd_fin), digits=3) if nnd_fin.size else None
        summary["nearest_neighbor_median_cm"] = _clean_float(np.median(nnd_fin), digits=3) if nnd_fin.size else None
        summary["nearest_neighbor_max_cm"] = _clean_float(np.max(nnd_fin), digits=3) if nnd_fin.size else None

    elif len(target_ids) >= 3 and not missing_ids:
        slot_set = set(target_slots)
        internal_edges = [
            e
            for e, (i, j) in enumerate(zip(features.pair_i, features.pair_j))
            if int(i) in slot_set and int(j) in slot_set
        ]
        if internal_edges:
            internal_dist = features.pair_distance_cm[:, internal_edges]
            fin = internal_dist[np.isfinite(internal_dist)]
            summary["internal_pair_distance_min_cm"] = _clean_float(np.min(fin), digits=3) if fin.size else None
            summary["internal_pair_distance_median_cm"] = _clean_float(np.median(fin), digits=3) if fin.size else None
            summary["internal_pair_distance_max_cm"] = _clean_float(np.max(fin), digits=3) if fin.size else None

        radii: list[float] = []
        for frame_idx in range(len(features.valid)):
            pts = [
                features.centers_cm[frame_idx, s]
                for s in target_slots
                if features.valid[frame_idx, s] and np.isfinite(features.centers_cm[frame_idx, s]).all()
            ]
            if len(pts) >= 2:
                arr = np.asarray(pts, dtype=float)
                radii.append(float(np.linalg.norm(arr - arr.mean(axis=0), axis=1).mean()))
        if radii:
            summary["group_radius_start_cm"] = _clean_float(radii[0], digits=3)
            summary["group_radius_median_cm"] = _clean_float(np.median(radii), digits=3)
            summary["group_radius_end_cm"] = _clean_float(radii[-1], digits=3)

    return summary


def _fmt_num(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "缺失"
    return f"{value:.2f}{suffix}"


def _format_kinematics_zh(kin: Mapping[str, Any], target_ids: Sequence[int]) -> str:
    parts: list[str] = []
    ratios = kin.get("observed_ratio_per_target", {})
    ratio_str = ", ".join(
        f"ID {tid} 观测={_fmt_num(ratios.get(str(tid)) * 100 if ratios.get(str(tid)) is not None else None, '%')}"
        for tid in target_ids
    )
    if ratio_str:
        parts.append(ratio_str)
    if kin.get("coobserved_ratio") is not None:
        parts.append(f"共现={kin['coobserved_frames']}/{kin['total_frames']}帧({kin['coobserved_ratio'] * 100:.1f}%)")
    if kin.get("center_distance_min_cm") is not None:
        parts.append(
            f"中心距 min/med/max={_fmt_num(kin['center_distance_min_cm'])}/"
            f"{_fmt_num(kin['center_distance_median_cm'])}/{_fmt_num(kin['center_distance_max_cm'])} cm"
        )
    if kin.get("nose_head_min_cm") is not None or kin.get("nose_tail_min_cm") is not None:
        parts.append(
            f"鼻-头min={_fmt_num(kin.get('nose_head_min_cm'), 'cm')}, "
            f"鼻-尾min={_fmt_num(kin.get('nose_tail_min_cm'), 'cm')}"
        )
    if kin.get("speed_median_cm_s") is not None:
        parts.append(
            f"速度 med/max={_fmt_num(kin['speed_median_cm_s'])}/{_fmt_num(kin['speed_max_cm_s'], ' cm/s')}"
        )
    if kin.get("missing_nose_fraction") is not None:
        parts.append(
            f"关键点缺失 鼻={kin['missing_nose_fraction'] * 100:.1f}%, "
            f"尾={kin['missing_tail_fraction'] * 100:.1f}%"
        )
    if kin.get("internal_pair_distance_median_cm") is not None:
        parts.append(
            f"组内间距 min/med/max={_fmt_num(kin['internal_pair_distance_min_cm'])}/"
            f"{_fmt_num(kin['internal_pair_distance_median_cm'])}/{_fmt_num(kin['internal_pair_distance_max_cm'])} cm"
        )
    return "；".join(parts)


def _threshold_hint_zh(gt_behavior: str, social_cfg: SocialHeuristicConfig) -> str:
    hints = {
        "Attack": (
            f"阈值参考: attack_min_duration={social_cfg.attack_min_duration_s:g}s, "
            f"center<={social_cfg.dist_attack_max_cm:.2f}cm(或体表gap<={social_cfg.attack_contact_surface_max_gap_cm:.2f}cm), "
            f"加速度>={social_cfg.acc_attack_min_cm_s2:.1f}cm/s^2"
        ),
        "Chasing": (
            f"阈值参考: chasing_min_duration={social_cfg.chasing_min_duration_s:g}s, "
            f"双方均速>={social_cfg.chasing_min_mean_speed_cm_s:g}cm/s, "
            f"净位移>={social_cfg.chasing_min_net_displacement_cm:g}cm"
        ),
        "Approach": (
            f"阈值参考: duration>{social_cfg.approach_min_duration_s:g}s, "
            f"start>={social_cfg.dist_approach_start_min_cm:g}cm, "
            f"end=[{social_cfg.dist_approach_end_min_cm:.2f},{social_cfg.dist_approach_end_max_cm:g}]cm, "
            f"actor速>{social_cfg.vel_approach_actor_min_cm_s:g}cm/s"
        ),
        "Snout-head_contact": (
            f"阈值参考: onset<={social_cfg.contact_head_onset_distance_cm:.2f}cm, "
            f"维持<={social_cfg.dist_nose_head_contact_max_cm:.2f}cm(扩展<={social_cfg.contact_head_extension_distance_cm:.2f}cm), "
            f"时长>={social_cfg.nose_contact_min_duration_s:g}s"
        ),
        "Snout-rear_contact": (
            f"阈值参考: onset<={social_cfg.contact_tail_onset_distance_cm:.2f}cm, "
            f"维持<={social_cfg.dist_nose_tail_contact_max_cm:.2f}cm(扩展<={social_cfg.contact_tail_extension_distance_cm:.2f}cm), "
            f"基线>={social_cfg.nose_tail_min_duration_s:g}s/扩展>={social_cfg.contact_tail_extension_min_duration_s:g}s"
        ),
        "Together": (
            f"阈值参考: 紧距<{social_cfg.dist_together_max_cm:g}cm & 速<{social_cfg.vel_together_max_cm_s:g}cm/s; "
            f"宽距<{social_cfg.together_wide_max_distance_cm:g}cm & 速<{social_cfg.together_wide_max_speed_cm_s:g}cm/s "
            f"& body_gap>={social_cfg.together_wide_min_body_gap_cm:g}cm, 时长>={social_cfg.together_min_duration_s:g}s"
        ),
        "Huddling": (
            f"阈值参考: >=3鼠连通(center<={DEFAULT_GROUP_HEURISTIC_CONFIG.huddle_neighbor_distance_cm:g}cm "
            "或鼻<=4.01cm), 持续>=5.0s, 成员支持>=25%"
        ),
    }
    return hints.get(gt_behavior, "")


def _classify_primary_failure(
    *,
    gt_behavior: str,
    gt_canonical: str,
    target_ids: Sequence[int],
    primary_predicted_behavior: str | None,
    pf_pair_behavior: str | None,
    pf_joint_behavior: str | None,
    pf_joint_target_ids: Sequence[int],
    pf_joint_score: float,
    mask_counts: Mapping[str, int],
    hit_mask: bool,
    target_individual_counts: Mapping[str, int],
    huddle_diag: Mapping[str, Any] | None,
    kinematics: Mapping[str, Any],
    total_frames: int,
    fps: float,
    social_cfg: SocialHeuristicConfig,
    enable_posture_screen: bool = False,
) -> tuple[str, str, str]:
    """Return ``(failure_category, failure_subcategory, failure_reason_zh)``."""
    kin_text = _format_kinematics_zh(kinematics, target_ids)
    hint_text = _threshold_hint_zh(gt_behavior, social_cfg)
    layer = _behavior_layer(gt_behavior)
    mask_summary = (
        ", ".join(f"{b}={cnt}帧({cnt / max(fps, 1e-6):.2f}s)" for b, cnt in mask_counts.items() if cnt > 0)
        or "全部社交掩码均为0帧"
    )

    if gt_behavior in SUSPENDED_DISPLAY_BEHAVIORS and not enable_posture_screen:
        ind_summary = (
            ", ".join(f"{b}={cnt}帧" for b, cnt in target_individual_counts.items() if cnt > 0)
            or "无默认个体行为触发"
        )
        reason = (
            f"行为未识别到（正典规则停用）：{gt_behavior} 在正典默认配置中处于挂起/停用状态 "
            f"（enable_posture_screen=False），未生成任何姿态候选（0/{total_frames} 帧）；"
            f"目标鼠在默认规则下触发：{ind_summary}。{kin_text}。"
        )
        return "missed", "disabled_rule", reason

    if kinematics.get("missing_target_ids"):
        reason = (
            f"行为未识别到（轨迹目标缺失）：标注目标 ID {list(target_ids)} 中存在缺失 "
            f"（缺失 ID={kinematics.get('missing_target_ids', [])}，共现={kinematics.get('coobserved_frames', 0)}/{total_frames} 帧）。"
            f"{kin_text}。"
        )
        return "missed", "target_id_missing_in_tracks", reason

    if gt_behavior == "Huddling" and huddle_diag is not None:
        a1_pos = bool(huddle_diag.get("a1_any_positive", False))
        a2_pos = bool(huddle_diag.get("a2_any_positive", False))
        true_cov = float(huddle_diag.get("coverage_on_true_target", 0.0))
        best_cov = float(huddle_diag.get("coverage_best_any_target", 0.0))
        best_ids = huddle_diag.get("best_huddle_target_ids", [])
        joint_disp = CANONICAL_TO_DISPLAY.get(pf_joint_behavior, str(pf_joint_behavior)) if pf_joint_behavior else "未识别"

        if enable_posture_screen and true_cov < 0.5:
            return (
                "missed", "no_candidate_threshold_unmet",
                f"Huddling 精确目标集合 {list(target_ids)} 的事件并集覆盖 {true_cov * 100:.1f}%，未达到50%；"
                f"最佳候选集合为 ID {best_ids}（覆盖 {best_cov * 100:.1f}%）。{kin_text}。",
            )

        if not a1_pos:
            reason = (
                f"行为未识别到（Huddling 群体规则未触发）：整段 {total_frames} 帧未生成任何 Huddling 群体候选（A1/A2=0），"
                f"盲评 Top-1 为 {joint_disp} (ID {list(pf_joint_target_ids)}, 得分={pf_joint_score:.2f}s)。"
                f"{kin_text}；{hint_text}。"
            )
            return "missed", "no_candidate_threshold_unmet", reason

        if tuple(sorted(pf_joint_target_ids)) == tuple(sorted(target_ids)):
            reason = (
                f"识别行为出错（优先级或时长被压制）：目标集合 {list(target_ids)} 存在 Huddling 候选"
                f"（目标覆盖={true_cov * 100:.1f}%），但同组另一行为 {joint_disp} 得分更高"
                f"（{pf_joint_score:.2f}s）而在盲评 Top-1 胜出。{kin_text}；{hint_text}。"
            )
            return "misclassified", "priority_or_duration_outranked", reason

        reason = (
            "识别行为出错（盲评被其他鼠/子集或竞争行为抢占）："
            + (
                f"真值集合 {list(target_ids)} 已生成 Huddling 候选（覆盖率={true_cov * 100:.1f}%），"
                if a2_pos
                else f"未命中精确真值集合（最佳扎堆组为 ID {best_ids}，覆盖率={best_cov * 100:.1f}%），"
            )
            + f"但盲评 Top-1 被 {joint_disp} (ID {list(pf_joint_target_ids)}, 得分={pf_joint_score:.2f}s) 胜出。"
            f"{kin_text}；{hint_text}。"
        )
        return "misclassified", "other_pair_won_joint", reason

    if layer == "social":
        if gt_behavior in {"Attack", "Chasing"}:
            gt_frames = int(mask_counts.get(gt_behavior, 0))
            if primary_predicted_behavior is not None and primary_predicted_behavior != gt_behavior:
                if hit_mask:
                    reason = (
                        f"识别行为出错（优先级/时长被压制）：真值 {gt_behavior} 在目标对上触发 {gt_frames} 帧"
                        f"（{gt_frames / max(fps, 1e-6):.2f}s），但 {primary_predicted_behavior} 胜出"
                        f"（目标对掩码：{mask_summary}）。{kin_text}；{hint_text}。"
                    )
                    return "misclassified", "priority_or_duration_outranked", reason
                reason = (
                    f"识别行为出错（触发其他行为）：真值 {gt_behavior} 在目标对上仅 0 帧，"
                    f"模型触发并选择了 {primary_predicted_behavior}（目标对掩码：{mask_summary}）。"
                    f"{kin_text}；{hint_text}。"
                )
                return "misclassified", "wrong_behavior_triggered", reason

            other_social = {k: v for k, v in mask_counts.items() if k != gt_behavior and v > 0}
            if other_social:
                other_top = max(other_social.items(), key=lambda kv: kv[1])[0]
                reason = (
                    f"识别行为出错（基准未触发 {gt_behavior}，社会规则触发了 {other_top}）："
                    f"目标对掩码统计：{mask_summary}（pf_pair={CANONICAL_TO_DISPLAY.get(pf_pair_behavior, pf_pair_behavior) or '无'}）。"
                    f"{kin_text}；{hint_text}。"
                )
                return "misclassified", "wrong_behavior_triggered", reason

            reason = (
                f"行为未识别到（未达规则阈值）：目标对上全部 6 类社交掩码均为 0/{total_frames} 帧，"
                f"未能触发 {gt_behavior}。{kin_text}；{hint_text}。"
            )
            return "missed", "no_candidate_threshold_unmet", reason

        # Pair-conditioned social behaviors: Approach, Snout-head_contact, Snout-rear_contact, Together
        gt_frames = int(mask_counts.get(gt_behavior, 0))
        if hit_mask or gt_frames > 0:
            pred_label = primary_predicted_behavior or "无"
            reason = (
                f"识别行为出错（优先级或时长被压制）：真值 {gt_behavior} 在目标对上触发了 {gt_frames} 帧"
                f"（{gt_frames / max(fps, 1e-6):.2f}s），但最终被 {pred_label} 超出（目标对掩码：{mask_summary}）。"
                f"{kin_text}；{hint_text}。"
            )
            return "misclassified", "priority_or_duration_outranked", reason

        if primary_predicted_behavior is not None:
            reason = (
                f"识别行为出错（触发其他社交行为）：真值 {gt_behavior} 为 0 帧，模型在目标对上触发并选择了 "
                f"{primary_predicted_behavior}（目标对掩码：{mask_summary}）。{kin_text}；{hint_text}。"
            )
            return "misclassified", "wrong_behavior_triggered", reason

        reason = (
            f"行为未识别到（未达规则阈值）：目标对上全部社交掩码均为 0/{total_frames} 帧，"
            f"未能触发 {gt_behavior}。{kin_text}；{hint_text}。"
        )
        return "missed", "no_candidate_threshold_unmet", reason

    # Fallback for individual / group benchmark failures
    if primary_predicted_behavior is not None:
        reason = (
            f"识别行为出错：目标上预测为 {primary_predicted_behavior} 而非真值 {gt_behavior}。{kin_text}。"
        )
        return "misclassified", "wrong_behavior_triggered", reason
    reason = f"行为未识别到（未达规则阈值）：目标未满足 {gt_behavior} 判定门槛。{kin_text}。"
    return "missed", "no_candidate_threshold_unmet", reason


def _classify_joint_failure(
    *,
    gt_behavior: str,
    gt_canonical: str,
    target_ids: Sequence[int],
    pf_joint_behavior: str | None,
    pf_joint_target_ids: Sequence[int],
    pf_joint_score: float,
    pf_target_behavior: str | None,
    pf_target_hit: bool,
    kinematics: Mapping[str, Any],
    total_frames: int,
    enable_posture_screen: bool = False,
) -> tuple[str, str, str]:
    """Classify a blind `pf_joint` failure across all 15 classes."""
    if gt_behavior in SUSPENDED_DISPLAY_BEHAVIORS and not enable_posture_screen:
        return (
            "missed",
            "disabled_rule",
            f"盲评失败（规则停用）：{gt_behavior} 在正典默认配置下禁用（enable_posture_screen=False）。",
        )
    if kinematics.get("missing_target_ids"):
        return (
            "missed",
            "target_id_missing_in_tracks",
            f"盲评失败（轨迹缺失）：目标 ID {list(target_ids)} 缺失（{kinematics.get('missing_target_ids', [])}）。",
        )
    if pf_joint_behavior is None:
        return (
            "missed",
            "no_candidate_threshold_unmet",
            f"盲评未识别到：整段 {total_frames} 帧未产生任何正候选（{gt_behavior}）。",
        )
    joint_disp = CANONICAL_TO_DISPLAY.get(pf_joint_behavior, str(pf_joint_behavior))
    if tuple(sorted(pf_joint_target_ids)) != tuple(sorted(target_ids)):
        return (
            "misclassified",
            "other_pair_won_joint",
            (
                f"盲评被其他视觉 ID/鼠对抢占：真值 {gt_behavior} (目标 ID {list(target_ids)}, "
                f"target_hit={pf_target_hit}, target_top1={CANONICAL_TO_DISPLAY.get(pf_target_behavior, pf_target_behavior) or '无'})，"
                f"但盲评 Top-1 被 ID {list(pf_joint_target_ids)} 的 {joint_disp}（{pf_joint_score:.2f}s）胜出。"
            ),
        )
    if pf_target_hit:
        return (
            "misclassified",
            "priority_or_duration_outranked",
            f"盲评同目标集合内被压制：真值 {gt_behavior} 命中但被 {joint_disp}（{pf_joint_score:.2f}s）以时长/得分反超。",
        )
    return (
        "misclassified",
        "wrong_behavior_triggered",
        f"盲评同目标集合触发其他行为：真值 {gt_behavior} 未命中，盲评选择 {joint_disp}（{pf_joint_score:.2f}s）。",
    )


def _evaluate_clip_worker(task: tuple[ClipIndexEntry, str, bool, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Evaluate one indexed clip across primary, blind `pf_joint`, and failure diagnostics."""
    entry, config_path_str, all_behaviors, scale_receipt = task
    config_path = Path(config_path_str)
    params, opt_params = _load_worker_context(config_path)
    social_cfg = OPTIMIZED_SOCIAL_HEURISTIC_CONFIG

    clip_dir = Path(entry.clip_dir)
    gt_behavior = entry.behavior
    gt_canonical = entry.canonical_behavior
    target_ids = list(entry.mouse_ids)
    total_frames = entry.frame_count
    fps = entry.fps
    layer = _behavior_layer(gt_behavior)
    primary_method = PRIMARY_METHOD_BY_BEHAVIOR[gt_behavior]

    features = load_exported_features(
        entry.tracks_path,
        fps=fps,
        max_tracks=max(64, len(entry.observed_track_ids)),
        scale_config=scale_receipt["scale_config"] if scale_receipt is not None else None,
    )
    features = _padded_features(features, total_frames)

    pred = predict_features(
        features,
        opt_params,
        social_config=social_cfg,
        enable_posture_screen=all_behaviors,
    )
    all_candidates = candidates_from_evidence(pred)
    top1 = select_top1(all_candidates)
    pf_joint_behavior = top1.behavior
    pf_joint_behavior_display = CANONICAL_TO_DISPLAY.get(pf_joint_behavior) if pf_joint_behavior else None
    pf_joint_target_ids = sorted(int(x) for x in top1.target_ids)
    pf_joint_score = float(top1.score)
    pf_joint_correct = bool(
        pf_joint_behavior == gt_canonical
        and tuple(pf_joint_target_ids) == tuple(sorted(target_ids))
    )

    sample_record = SampleRecord(
        sample_id=entry.clip_id,
        behavior=gt_behavior,
        canonical_behavior=gt_canonical,
        sample_dir=entry.clip_dir,
        annotation_path=str(clip_dir / "annotation.json"),
        metadata_path=str(clip_dir / "metadata.json"),
        tracks_path=entry.tracks_path,
        clip_path=str(clip_dir / "clip.mp4"),
        source_video=entry.source_video,
        recording_session=entry.source_video,
        fps=fps,
        frame_count=total_frames,
        mouse_ids=entry.mouse_ids,
        confidence=entry.confidence,
        frame_start=entry.frame_range_inclusive["start"],
        frame_end=entry.frame_range_inclusive["end"],
        time_start=0.0,
        time_end=total_frames / max(fps, 1e-6),
    )
    cti = classify_target_ids(sample_record, pred)
    pf_target_behavior = cti.get("predicted_behavior")
    pf_target_behavior_display = (
        CANONICAL_TO_DISPLAY.get(pf_target_behavior) if pf_target_behavior else None
    )
    pf_target_correct = bool(cti.get("correct", False))
    pf_target_hit = bool(cti.get("target_hit", False))
    target_candidate_scores = {
        CANONICAL_TO_DISPLAY.get(k, k): _clean_float(v, digits=4)
        for k, v in dict(cti.get("candidate_scores", {})).items()
        if float(v) > 0.0
    }

    # Individual masks on target animals (fast ~5ms; gives Static/Locomotion context for all layers)
    ind_res = individual_rule_masks(
        features.centers_cm,
        features.valid,
        fps,
        keypoints_cm=features.collective_keypoints_cm,
        boxes_cm=features.collective_boxes_cm,
        reference_body_cm=features.reference_body_cm,
        config=replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=all_behaviors),
    )
    track_to_slot = {int(tid): slot for slot, tid in enumerate(features.track_ids.tolist())}
    target_slots = [track_to_slot[tid] for tid in target_ids if tid in track_to_slot]

    target_individual_counts: dict[str, int] = {}
    target_individual_intervals: dict[str, list[dict[str, int]]] = {}
    for ind_key, disp_name in (
        ("stationary", "Static"),
        ("locomotion", "Locomotion"),
        ("grooming", "Grooming"),
        ("rearing", "Rearing"),
    ):
        arr = np.asarray(ind_res.masks.get(ind_key, np.zeros((total_frames, len(features.track_ids)), bool)), bool)
        if target_slots:
            # classify_target_ids already validates the single-ID label contract.
            combined = arr[:, target_slots[0]] if layer == "individual" else np.any(arr[:, target_slots], axis=1)
        else:
            combined = np.zeros(total_frames, dtype=bool)
        cnt = int(np.count_nonzero(combined))
        if cnt > 0:
            target_individual_counts[disp_name] = cnt
            target_individual_intervals[disp_name] = _intervals_from_mask(combined)

    # Social pair diagnostics
    mask_counts: dict[str, int] = {b: 0 for b in SOCIAL_DISPLAY_BEHAVIORS}
    social_intervals: dict[str, list[dict[str, int]]] = {}
    hit_mask = False
    top1_max_behavior: str | None = None
    top1_max_correct = False
    top1_prio_behavior: str | None = None
    top1_prio_correct = False
    pf_pair_behavior: str | None = pf_target_behavior if layer == "social" else None
    pf_pair_behavior_display: str | None = pf_target_behavior_display if layer == "social" else None
    pf_pair_correct: bool | None = pf_target_correct if layer == "social" else None

    if layer == "social":
        masks = pair_rule_masks(features, opt_params, social_cfg)
        edge = _find_edge_index(features, target_ids)
        if edge is not None:
            for mask_key, can_name in SOCIAL_MASK_TO_CANONICAL.items():
                disp_name = CANONICAL_TO_DISPLAY[can_name]
                col = np.asarray(masks[mask_key][:, edge], bool)
                cnt = int(np.count_nonzero(col))
                mask_counts[disp_name] = cnt
                if cnt > 0:
                    social_intervals[disp_name] = _intervals_from_mask(col)
            hit_mask = mask_counts[gt_behavior] > 0
            prio_display = [CANONICAL_TO_DISPLAY[SOCIAL_MASK_TO_CANONICAL[m]] for m in social_cfg.priority_order]
            positive_display = [b for b in prio_display if mask_counts[b] > 0]
            if positive_display:
                top1_prio_behavior = positive_display[0]
                top1_prio_correct = top1_prio_behavior == gt_behavior
                top1_max_behavior = max(
                    positive_display,
                    key=lambda b: (mask_counts[b], -prio_display.index(b)),
                )
                top1_max_correct = top1_max_behavior == gt_behavior

    # Primary evaluator dispatch per behavior family
    primary_predicted_behavior: str | None = None
    primary_correct = False
    primary_predicted_range: dict[str, int] | None = None
    temporal_iou: float | None = None
    iou_ge_05: bool | None = None
    benchmark_status = "ok"
    benchmark_error: str | None = None
    huddle_diag: dict[str, Any] | None = None
    group_target_intervals: dict[str, list[dict[str, int]]] = {}
    target_set = tuple(sorted(target_ids))
    group_event_masks: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    if layer == "group":
        for event in pred.get("group_events", []):
            members = tuple(sorted(int(value) for value in event["member_ids"]))
            key = (str(event["behavior"]), members)
            mask = group_event_masks.setdefault(key, np.zeros(total_frames, dtype=bool))
            for start, stop in event.get("frame_intervals") or ():
                mask[int(start):int(stop)] = True
        # Union only identical behavior/ID keys; boolean masks count overlaps once.
        group_target_intervals = {
            CANONICAL_TO_DISPLAY.get(behavior, behavior): _intervals_from_mask(mask)
            for (behavior, members), mask in group_event_masks.items() if members == target_set
        }

    if gt_behavior in {"Attack", "Chasing"}:
        bench = evaluate_attack_chasing_clip(
            clip_dir,
            params=params,
            social_config=social_cfg,
            visualize=False,
        )
        primary_predicted_behavior = bench.get("predicted_behavior")
        primary_correct = bool(bench.get("is_correct", False))
        primary_predicted_range = bench.get("predicted_range_half_open")
        temporal_iou = _clean_float(bench.get("temporal_iou"))
        iou_ge_05 = bool(bench.get("iou_ge_05", False))
        benchmark_status = str(bench.get("status", "ok"))
        benchmark_error = bench.get("error")

    elif gt_behavior in {"Approach", "Snout-head_contact", "Snout-rear_contact", "Together"}:
        primary_predicted_behavior = pf_pair_behavior_display
        primary_correct = bool(pf_pair_correct)
        if primary_predicted_behavior and social_intervals.get(primary_predicted_behavior):
            ivs = social_intervals[primary_predicted_behavior]
            primary_predicted_range = {"start": ivs[0]["start"], "end": ivs[-1]["end"]}

    elif gt_behavior in {
        "Static",
        "Locomotion",
        "Isolation",
        "Group locomotion",
        "Dispersal",
        "Social clustering",
    }:
        bench = evaluate_six_behavior_clip(
            clip_dir,
            expected_behavior=gt_behavior,
            params=params,
            social_config=social_cfg,
            visualize=False,
            render_videos=False,
        )
        primary_predicted_behavior = bench.get("predicted_behavior")
        primary_correct = bool(bench.get("is_correct", False))
        primary_predicted_range = bench.get("predicted_range_half_open")
        temporal_iou = _clean_float(bench.get("temporal_iou"))
        iou_ge_05 = bool(bench.get("iou_ge_05", False))
        benchmark_status = str(bench.get("status", "ok"))
        benchmark_error = bench.get("error")

    elif gt_behavior == "Huddling":
        primary_predicted_behavior = pf_joint_behavior_display
        primary_correct = pf_joint_correct
        # Extract Huddling A1/A2 coverage diagnostics from pred['group_events']
        huddle_events = [
            ev for ev in pred.get("group_events", []) if ev.get("behavior") == "huddle"
        ]
        target_set = tuple(sorted(target_ids))
        cov_rows: list[dict[str, Any]] = []
        for ev in huddle_events:
            mids = tuple(sorted(int(x) for x in ev.get("member_ids", [])))
            ivs = ev.get("frame_intervals")
            if ivs is not None:
                frame_cnt = interval_frame_count(tuple((int(s), int(e)) for s, e in ivs), total_frames)
                cov = frame_cnt / max(total_frames, 1)
            else:
                cov = min(1.0, max(0.0, float(ev.get("evidence_seconds", 0.0)) / max(total_frames / fps, 1e-6)))
                frame_cnt = int(round(cov * total_frames))
            cov_rows.append(
                {
                    "member_ids": list(mids),
                    "coverage": cov,
                    "frame_count": frame_cnt,
                    "intervals": [{"start": int(s), "end": int(e)} for s, e in (ivs or ())],
                }
            )

        max_single_event_true_cov = max(
            (r["coverage"] for r in cov_rows if tuple(r["member_ids"]) == target_set), default=0.0,
        )
        if all_behaviors:
            cov_rows = [
                {"member_ids": list(members),
                 "coverage": float(np.count_nonzero(mask)) / max(total_frames, 1),
                 "frame_count": int(np.count_nonzero(mask)),
                 "intervals": _intervals_from_mask(mask)}
                for (behavior, members), mask in group_event_masks.items() if behavior == "huddle"
            ]

        true_cov = max((r["coverage"] for r in cov_rows if tuple(r["member_ids"]) == target_set), default=0.0)
        best_row = max(cov_rows, key=lambda r: r["coverage"], default={"coverage": 0.0, "member_ids": []})
        huddle_diag = {
            "a1_hit": any(r["coverage"] >= 0.5 for r in cov_rows),
            "a2_hit": any(tuple(r["member_ids"]) == target_set and r["coverage"] >= 0.5 for r in cov_rows),
            "a1_any_positive": any(r["coverage"] > 0.0 for r in cov_rows),
            "a2_any_positive": any(tuple(r["member_ids"]) == target_set and r["coverage"] > 0.0 for r in cov_rows),
            "coverage_on_true_target": _clean_float(true_cov),
            "coverage_best_any_target": _clean_float(best_row["coverage"]),
            "best_huddle_target_ids": best_row["member_ids"],
            "huddle_candidate_count": len(cov_rows),
            "a2_max_single_event_hit": max_single_event_true_cov >= 0.5,
            "coverage_on_true_target_max_single_event": max_single_event_true_cov,
        }
        if all_behaviors:
            primary_method = "exact_target_frame_coverage_ge_0.5"
            primary_correct = bool(huddle_diag["a2_hit"])
            primary_predicted_behavior = gt_behavior if primary_correct else None
        if primary_correct:
            ivs = group_target_intervals.get("Huddling", [])
            if ivs:
                primary_predicted_range = {"start": ivs[0]["start"], "end": ivs[-1]["end"]}

    elif gt_behavior in SUSPENDED_DISPLAY_BEHAVIORS:
        if all_behaviors:
            primary_method = "exact_target_frame_coverage_ge_0.5"
            primary_correct = target_individual_counts.get(gt_behavior, 0) / max(total_frames, 1) >= 0.5
            primary_predicted_behavior = gt_behavior if primary_correct else None
            ivs = target_individual_intervals.get(gt_behavior, [])
            if primary_correct and ivs:
                primary_predicted_range = {"start": ivs[0]["start"], "end": ivs[-1]["end"]}
        else:
            primary_predicted_behavior = None
            primary_correct = False
            benchmark_status = "disabled_suspended"

    if gt_behavior == "Isolation":
        # Use rule event support, never a truth-conditioned benchmark envelope.
        isolation_mask = np.zeros(total_frames, dtype=bool)
        for candidate in isolation_candidates(features, opt_params):
            if tuple(sorted(candidate.target_ids)) == target_set:
                for start, stop in candidate.frame_intervals or ():
                    isolation_mask[start:stop] = True
        group_target_intervals["Isolation"] = _intervals_from_mask(isolation_mask)

    coverage_intervals = (
        social_intervals if layer == "social" else
        target_individual_intervals if layer == "individual" else group_target_intervals
    ).get(gt_behavior, [])
    true_target_mask = np.zeros(total_frames, dtype=bool)
    if target_ids and not entry.missing_target_ids:
        for interval in coverage_intervals:
            true_target_mask[interval["start"]:interval["end"]] = True
    true_target_frames = int(np.count_nonzero(true_target_mask))
    true_target_coverage = true_target_frames / max(total_frames, 1)
    coverage_a2_correct = true_target_coverage >= 0.5
    if entry.missing_target_ids:
        primary_correct = False

    # Timeline assembly
    kinematics = _extract_target_kinematics(features, target_ids, total_frames)
    if target_slots:
        coobs_mask = np.all(features.valid[:, target_slots], axis=1)
    else:
        coobs_mask = np.zeros(total_frames, dtype=bool)
    target_observed_intervals = _intervals_from_mask(coobs_mask)

    if layer == "social":
        primary_intervals_map = social_intervals
        primary_source = "pair_rule_masks"
    elif layer == "individual":
        primary_intervals_map = target_individual_intervals
        primary_source = "individual_rule_masks"
    else:
        primary_intervals_map = dict(group_target_intervals)
        if primary_predicted_behavior and primary_predicted_range and primary_predicted_behavior not in primary_intervals_map:
            primary_intervals_map[primary_predicted_behavior] = [primary_predicted_range]
        primary_source = "group_events_and_clip_benchmark"

    target_behavior_intervals = _merge_interval_runs(primary_intervals_map, source=primary_source)
    competing_intervals = (
        _merge_interval_runs(target_individual_intervals, source="individual_rule_masks")
        if layer != "individual"
        else []
    )

    joint_top1_intervals: list[dict[str, Any]] = []
    matching_cand_intervals = next(
        (
            c.frame_intervals
            for c in all_candidates
            if c.behavior == pf_joint_behavior
            and tuple(sorted(c.target_ids)) == tuple(pf_joint_target_ids)
            and c.frame_intervals
        ),
        None,
    )
    if matching_cand_intervals:
        joint_top1_intervals = [
            {
                "start": int(s),
                "end": int(e),
                "behavior": pf_joint_behavior_display or str(pf_joint_behavior),
                "canonical_behavior": pf_joint_behavior,
                "target_ids": pf_joint_target_ids,
            }
            for s, e in matching_cand_intervals
        ]
    elif pf_joint_behavior in {"stationary", "locomotion", "grooming", "rearing"}:
        if pf_joint_target_ids and pf_joint_target_ids[0] in track_to_slot:
            slot = track_to_slot[pf_joint_target_ids[0]]
            m = ind_res.masks.get(pf_joint_behavior)
            if m is not None:
                joint_top1_intervals = [
                    {
                        "start": iv["start"],
                        "end": iv["end"],
                        "behavior": pf_joint_behavior_display,
                        "canonical_behavior": pf_joint_behavior,
                        "target_ids": pf_joint_target_ids,
                    }
                    for iv in _intervals_from_mask(m[:, slot])
                ]
    elif pf_joint_behavior in SOCIAL_MASK_TO_CANONICAL.values() and len(pf_joint_target_ids) == 2:
        if layer == "social" and tuple(pf_joint_target_ids) == tuple(sorted(target_ids)):
            joint_top1_intervals = [
                {
                    "start": iv["start"],
                    "end": iv["end"],
                    "behavior": pf_joint_behavior_display,
                    "canonical_behavior": pf_joint_behavior,
                    "target_ids": pf_joint_target_ids,
                }
                for iv in social_intervals.get(pf_joint_behavior_display or "", [])
            ]

    per_behavior_frame_counts = {
        b: sum(iv["end"] - iv["start"] for iv in ivs)
        for b, ivs in primary_intervals_map.items()
        if ivs
    }

    timeline_series = {
        "frame_count": total_frames,
        "fps": _clean_float(fps, digits=4),
        "ground_truth_interval": {
            "start": entry.frame_range_half_open["start"],
            "end": entry.frame_range_half_open["end"],
            "behavior": gt_behavior,
        },
        "predicted_interval": (
            {
                "start": int(primary_predicted_range["start"]),
                "end": int(primary_predicted_range["end"]),
                "behavior": primary_predicted_behavior,
            }
            if primary_predicted_range and primary_predicted_behavior
            else None
        ),
        "target_observed_intervals": target_observed_intervals,
        "target_behavior_intervals": target_behavior_intervals,
        "competing_behavior_intervals": competing_intervals,
        "per_behavior_frame_counts": per_behavior_frame_counts,
        "per_behavior_intervals": primary_intervals_map,
        "joint_top1_intervals": joint_top1_intervals,
        "kinematics_summary": kinematics,
    }

    # Failure diagnoses
    failure_category: str | None = None
    failure_subcategory: str | None = None
    failure_reason_zh: str | None = None
    if not primary_correct:
        failure_category, failure_subcategory, failure_reason_zh = _classify_primary_failure(
            gt_behavior=gt_behavior,
            gt_canonical=gt_canonical,
            target_ids=target_ids,
            primary_predicted_behavior=primary_predicted_behavior,
            pf_pair_behavior=pf_pair_behavior,
            pf_joint_behavior=pf_joint_behavior,
            pf_joint_target_ids=pf_joint_target_ids,
            pf_joint_score=pf_joint_score,
            mask_counts=mask_counts,
            hit_mask=hit_mask,
            target_individual_counts=target_individual_counts,
            huddle_diag=huddle_diag,
            kinematics=kinematics,
            total_frames=total_frames,
            fps=fps,
            social_cfg=social_cfg,
            enable_posture_screen=all_behaviors,
        )

    joint_failure_category: str | None = None
    joint_failure_subcategory: str | None = None
    joint_failure_reason_zh: str | None = None
    if not pf_joint_correct:
        joint_failure_category, joint_failure_subcategory, joint_failure_reason_zh = _classify_joint_failure(
            gt_behavior=gt_behavior,
            gt_canonical=gt_canonical,
            target_ids=target_ids,
            pf_joint_behavior=pf_joint_behavior,
            pf_joint_target_ids=pf_joint_target_ids,
            pf_joint_score=pf_joint_score,
            pf_target_behavior=pf_target_behavior,
            pf_target_hit=pf_target_hit,
            kinematics=kinematics,
            total_frames=total_frames,
            enable_posture_screen=all_behaviors,
        )

    top_competing = [
        {
            "behavior": CANONICAL_TO_DISPLAY.get(c.behavior, c.behavior),
            "canonical_behavior": c.behavior,
            "target_ids": list(c.target_ids),
            "score": _clean_float(c.evidence, digits=4),
        }
        for c in sorted(all_candidates, key=lambda c: (-c.evidence, c.behavior, c.target_ids))[:8]
    ]

    return _sanitize_json_value(
        {
            "record_key": f"{entry.dataset}:{gt_behavior}:{entry.clip_id}",
            "dataset": entry.dataset,
            "clip_id": entry.clip_id,
            "clip_dir": entry.clip_dir,
            "behavior": gt_behavior,
            "canonical_behavior": gt_canonical,
            "annotation_behavior": entry.annotation_behavior,
            "layer": layer,
            "primary_method": primary_method,
            "canonical_rule_status": (
                "disabled_suspended" if gt_behavior in SUSPENDED_DISPLAY_BEHAVIORS and not all_behaviors else "enabled"
            ),
            "enable_posture_screen": all_behaviors,
            "all_behaviors": all_behaviors,
            "cm_per_pixel": float(features.cm_per_pixel),
            "metadata_distance_scale_cm_per_pixel": entry.metadata_cm_per_pixel,
            "scale_source": (
                scale_receipt.get(
                    "scale_source",
                    "dataset_b_source_scales" if entry.dataset == "B" else "dataset_source_scales",
                )
                if scale_receipt is not None
                else "default_scale_config_body_length"
            ),
            "target_mouse_ids": target_ids,
            "participants": list(entry.participants),
            "frame_range_inclusive": entry.frame_range_inclusive,
            "frame_range_half_open": entry.frame_range_half_open,
            "fps": _clean_float(fps, digits=4),
            "frame_count": total_frames,
            "source_video": entry.source_video,
            "has_video": entry.has_video,
            "tracks_sha256": entry.tracks_sha256,
            "observed_track_ids": list(entry.observed_track_ids),
            "missing_target_ids": list(entry.missing_target_ids),
            "benchmark_status": benchmark_status,
            "benchmark_error": benchmark_error,
            # Primary evaluation result
            "primary_predicted_behavior": primary_predicted_behavior,
            "primary_correct": primary_correct,
            "primary_predicted_range_half_open": primary_predicted_range,
            "temporal_iou": temporal_iou,
            "iou_ge_05": iou_ge_05,
            "true_target_coverage": true_target_coverage,
            "true_target_frame_count": true_target_frames,
            "coverage_a2_correct": coverage_a2_correct,
            "coverage_a2_threshold": 0.5,
            "attack_historical_coverage_ge_0.3": (
                true_target_coverage >= 0.3 if gt_behavior == "Attack" else None
            ),
            # Strict target-ID joint (blind predict_features + select_top1)
            "pf_joint_predicted_behavior": pf_joint_behavior_display,
            "pf_joint_predicted_canonical": pf_joint_behavior,
            "pf_joint_predicted_target_ids": pf_joint_target_ids,
            "pf_joint_score": _clean_float(pf_joint_score, digits=4),
            "pf_joint_correct": pf_joint_correct,
            # Target-conditioned predict_features (pf_pair for social)
            "pf_target_predicted_behavior": pf_target_behavior_display,
            "pf_target_predicted_canonical": pf_target_behavior,
            "pf_target_hit": pf_target_hit,
            "pf_target_correct": pf_target_correct,
            "pf_pair_predicted_behavior": pf_pair_behavior_display,
            "pf_pair_correct": pf_pair_correct,
            "target_candidate_scores": target_candidate_scores,
            # Social mask diagnostics
            "social_mask_counts": mask_counts if layer == "social" else None,
            "hit_mask": hit_mask if layer == "social" else None,
            "top1_max_behavior": top1_max_behavior if layer == "social" else None,
            "top1_max_correct": top1_max_correct if layer == "social" else None,
            "top1_prio_behavior": top1_prio_behavior if layer == "social" else None,
            "top1_prio_correct": top1_prio_correct if layer == "social" else None,
            # Huddling A1/A2 diagnostics
            "huddle_diagnostics": huddle_diag,
            # Competing top candidates across all tracks
            "top_competing_candidates": top_competing,
            # Failure diagnoses
            "failure_category": failure_category,
            "failure_subcategory": failure_subcategory,
            "failure_reason_zh": failure_reason_zh,
            "joint_failure_category": joint_failure_category,
            "joint_failure_subcategory": joint_failure_subcategory,
            "joint_failure_reason_zh": joint_failure_reason_zh,
            # Timeline & kinematics
            "timeline_series": timeline_series,
        }
    )


def _resolve_entry_scale_receipt(
    entry: ClipIndexEntry,
    *,
    dataset_b_scales: Mapping[str, Mapping[str, Any]] | None = None,
    source_scales_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    use_metadata_distance_scale: bool = True,
) -> Mapping[str, Any] | None:
    if source_scales_by_dataset is not None:
        ds_map = source_scales_by_dataset.get(entry.dataset)
        if ds_map is not None and entry.clip_id in ds_map:
            return ds_map[entry.clip_id]
    if entry.dataset == "B" and dataset_b_scales is not None and entry.clip_id in dataset_b_scales:
        return dataset_b_scales[entry.clip_id]
    if use_metadata_distance_scale and entry.metadata_cm_per_pixel is not None:
        return {
            "sample_id": entry.clip_id,
            "scale_source": entry.metadata_scale_source or "metadata_distance_scale",
            "scale_config": {
                "mode": "fixed",
                "cm_per_pixel": float(entry.metadata_cm_per_pixel),
                "assumed_mouse_body_length_cm": 7.918,
                "assumed_mouse_body_width_cm": 3.068558,
            },
        }
    return None


def evaluate_entries(
    entries: Sequence[ClipIndexEntry],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    workers: int = 1,
    all_behaviors: bool = False,
    dataset_b_scales: Mapping[str, Mapping[str, Any]] | None = None,
    source_scales_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    use_metadata_distance_scale: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate indexed clip entries sequentially or in parallel across any dataset label."""
    tasks = [
        (
            entry,
            str(config_path),
            all_behaviors,
            _resolve_entry_scale_receipt(
                entry,
                dataset_b_scales=dataset_b_scales,
                source_scales_by_dataset=source_scales_by_dataset,
                use_metadata_distance_scale=use_metadata_distance_scale,
            ),
        )
        for entry in entries
    ]
    if workers <= 1 or len(tasks) <= 1:
        return [_evaluate_clip_worker(task) for task in tasks]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_evaluate_clip_worker, task): (
                task[0].dataset,
                task[0].behavior,
                task[0].clip_id,
            )
            for task in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    order_map = {
        (entry.dataset, entry.behavior, entry.clip_id): idx for idx, entry in enumerate(entries)
    }
    results.sort(key=lambda r: order_map[(r["dataset"], r["behavior"], r["clip_id"])])
    return results


def _evaluate_entries(
    entries: Sequence[ClipIndexEntry],
    *,
    config_path: Path,
    workers: int,
    all_behaviors: bool = False,
    dataset_b_scales: Mapping[str, Mapping[str, Any]] | None = None,
    source_scales_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    use_metadata_distance_scale: bool = True,
) -> list[dict[str, Any]]:
    return evaluate_entries(
        entries,
        config_path=config_path,
        workers=workers,
        all_behaviors=all_behaviors,
        dataset_b_scales=dataset_b_scales,
        source_scales_by_dataset=source_scales_by_dataset,
        use_metadata_distance_scale=use_metadata_distance_scale,
    )


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(float(num / den), 4)


def aggregate_cohort_metrics(records: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    """Aggregate per-behavior and overall accuracy, strict pf_joint, and failures."""
    total = len(records)
    present_behaviors = [b for b in BEHAVIOR_DISPLAY_ORDER if any(r["behavior"] == b for r in records)]

    per_behavior: dict[str, dict[str, Any]] = {}
    confusion_primary: dict[str, Counter[str]] = defaultdict(Counter)
    confusion_pf_joint: dict[str, Counter[str]] = defaultdict(Counter)

    for rec in records:
        gt = str(rec["behavior"])
        pred_prim = str(rec.get("primary_predicted_behavior") or "<none>")
        pred_joint = str(rec.get("pf_joint_predicted_behavior") or "<none>")
        confusion_primary[gt][pred_prim] += 1
        confusion_pf_joint[gt][pred_joint] += 1

    for beh in present_behaviors:
        subset = [r for r in records if r["behavior"] == beh]
        n = len(subset)
        prim_ok = sum(1 for r in subset if r["primary_correct"])
        joint_ok = sum(1 for r in subset if r["pf_joint_correct"])
        target_ok = sum(1 for r in subset if r["pf_target_correct"])
        coverage_ok = sum(1 for r in subset if r.get("coverage_a2_correct", False))
        fail_subset = [r for r in subset if not r["primary_correct"]]
        missed_cnt = sum(1 for r in fail_subset if r["failure_category"] == "missed")
        miscls_cnt = sum(1 for r in fail_subset if r["failure_category"] == "misclassified")
        subcat_counts = dict(sorted(Counter(r["failure_subcategory"] for r in fail_subset if r["failure_subcategory"]).items()))

        row: dict[str, Any] = {
            "behavior": beh,
            "canonical_behavior": DISPLAY_TO_CANONICAL[beh],
            "layer": _behavior_layer(beh),
            "primary_method": subset[0].get("primary_method", PRIMARY_METHOD_BY_BEHAVIOR[beh]),
            "suspended_in_canonical_default": beh in SUSPENDED_DISPLAY_BEHAVIORS,
            "rule_enabled": any(r.get("canonical_rule_status") == "enabled" for r in subset),
            "support": n,
            "primary_correct": prim_ok,
            "primary_accuracy": _rate(prim_ok, n),
            "pf_joint_correct": joint_ok,
            "pf_joint_accuracy": _rate(joint_ok, n),
            "pf_target_correct": target_ok,
            "pf_target_accuracy": _rate(target_ok, n),
            "coverage_a2_correct": coverage_ok,
            "coverage_a2_accuracy": _rate(coverage_ok, n),
            "coverage_a2_threshold": 0.5,
            "mean_true_target_coverage": _clean_float(np.mean([
                float(r.get("true_target_coverage", 0.0)) for r in subset
            ])),
            "primary_failure_count": len(fail_subset),
            "missed_count": missed_cnt,
            "misclassified_count": miscls_cnt,
            "failure_subcategory_counts": subcat_counts,
        }
        if beh == "Attack":
            historical_ok = sum(1 for r in subset if r.get("attack_historical_coverage_ge_0.3"))
            row["attack_historical_coverage_ge_0.3_count"] = historical_ok
            row["attack_historical_coverage_ge_0.3_rate"] = _rate(historical_ok, n)

        if _behavior_layer(beh) == "social":
            pf_pair_ok = sum(1 for r in subset if r.get("pf_pair_correct"))
            hit_m = sum(1 for r in subset if r.get("hit_mask"))
            t1_max = sum(1 for r in subset if r.get("top1_max_correct"))
            t1_prio = sum(1 for r in subset if r.get("top1_prio_correct"))
            row.update(
                {
                    "pf_pair_correct": pf_pair_ok,
                    "pf_pair_accuracy": _rate(pf_pair_ok, n),
                    "hit_mask_count": hit_m,
                    "hit_mask_rate": _rate(hit_m, n),
                    "top1_max_correct": t1_max,
                    "top1_max_accuracy": _rate(t1_max, n),
                    "top1_prio_correct": t1_prio,
                    "top1_prio_accuracy": _rate(t1_prio, n),
                }
            )

        ious = [r["temporal_iou"] for r in subset if r.get("temporal_iou") is not None]
        if ious:
            corr_ious = [
                r["temporal_iou"]
                for r in subset
                if r["primary_correct"] and r.get("temporal_iou") is not None
            ]
            iou_ge_05_cnt = sum(1 for r in subset if r.get("iou_ge_05"))
            row.update(
                {
                    "mean_temporal_iou_all": _clean_float(np.mean(ious)),
                    "mean_temporal_iou_correct": _clean_float(np.mean(corr_ious)) if corr_ious else 0.0,
                    "iou_ge_05_count": iou_ge_05_cnt,
                    "iou_ge_05_rate": _rate(iou_ge_05_cnt, n),
                }
            )

        if beh == "Huddling":
            hud_rows = [r["huddle_diagnostics"] for r in subset if r.get("huddle_diagnostics")]
            a1_cnt = sum(1 for h in hud_rows if h.get("a1_hit"))
            a2_cnt = sum(1 for h in hud_rows if h.get("a2_hit"))
            a1_pos = sum(1 for h in hud_rows if h.get("a1_any_positive"))
            a2_pos = sum(1 for h in hud_rows if h.get("a2_any_positive"))
            mean_cov = _clean_float(
                np.mean([float(h.get("coverage_on_true_target") or 0.0) for h in hud_rows])
            ) if hud_rows else None
            row.update(
                {
                    "a1_coverage_count": a1_cnt,
                    "a1_coverage_rate": _rate(a1_cnt, n),
                    "a2_coverage_count": a2_cnt,
                    "a2_coverage_rate": _rate(a2_cnt, n),
                    "a1_any_positive_count": a1_pos,
                    "a2_any_positive_count": a2_pos,
                    "huddle_mean_true_target_coverage": mean_cov,
                }
            )

        per_behavior[beh] = row

    prim_total = sum(1 for r in records if r["primary_correct"])
    joint_total = sum(1 for r in records if r["pf_joint_correct"])
    target_total = sum(1 for r in records if r["pf_target_correct"])
    coverage_total = sum(1 for r in records if r.get("coverage_a2_correct", False))
    enabled_records = [r for r in records if r["behavior"] not in SUSPENDED_DISPLAY_BEHAVIORS]
    enabled_prim = sum(1 for r in enabled_records if r["primary_correct"])
    enabled_joint = sum(1 for r in enabled_records if r["pf_joint_correct"])
    enabled_target = sum(1 for r in enabled_records if r["pf_target_correct"])

    primary_fails = [r for r in records if not r["primary_correct"]]
    joint_fails = [r for r in records if not r["pf_joint_correct"]]

    return {
        "cohort": name,
        "total_clips": total,
        "overall_primary_correct": prim_total,
        "overall_primary_accuracy": _rate(prim_total, total),
        "overall_coverage_a2_correct": coverage_total,
        "overall_coverage_a2_accuracy": _rate(coverage_total, total),
        "enabled_13_clips": len(enabled_records),
        "enabled_13_primary_correct": enabled_prim,
        "enabled_13_primary_accuracy": _rate(enabled_prim, len(enabled_records)),
        "overall_pf_joint_correct": joint_total,
        "overall_pf_joint_accuracy": _rate(joint_total, total),
        "enabled_13_pf_joint_correct": enabled_joint,
        "enabled_13_pf_joint_accuracy": _rate(enabled_joint, len(enabled_records)),
        "overall_pf_target_correct": target_total,
        "overall_pf_target_accuracy": _rate(target_total, total),
        "enabled_13_pf_target_correct": enabled_target,
        "enabled_13_pf_target_accuracy": _rate(enabled_target, len(enabled_records)),
        "primary_failure_count": len(primary_fails),
        "primary_missed_count": sum(1 for r in primary_fails if r["failure_category"] == "missed"),
        "primary_misclassified_count": sum(
            1 for r in primary_fails if r["failure_category"] == "misclassified"
        ),
        "failure_subcategory_counts": dict(
            sorted(Counter(r["failure_subcategory"] for r in primary_fails if r["failure_subcategory"]).items())
        ),
        "pf_joint_failure_count": len(joint_fails),
        "joint_failure_subcategory_counts": dict(
            sorted(
                Counter(
                    r["joint_failure_subcategory"] for r in joint_fails if r["joint_failure_subcategory"]
                ).items()
            )
        ),
        "per_behavior": per_behavior,
        "confusion_primary": {truth: dict(sorted(cnts.items())) for truth, cnts in sorted(confusion_primary.items())},
        "confusion_pf_joint": {truth: dict(sorted(cnts.items())) for truth, cnts in sorted(confusion_pf_joint.items())},
    }


def build_overlap_agreement(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare Dataset A and Dataset B predictions on the 270 overlapping clip IDs."""
    by_a = {r["clip_id"]: r for r in records_a}
    by_b = {r["clip_id"]: r for r in records_b}
    overlap_ids: list[str] = list(reconciliation["overlap_ids"])
    detail_by_id = {d["clip_id"]: d for d in reconciliation["overlap_details"]}

    comparisons: list[dict[str, Any]] = []
    label_agree = 0
    prim_pred_agree = 0
    prim_corr_agree = 0
    joint_pred_agree = 0
    joint_corr_agree = 0
    pf_pair_pred_agree = 0

    for cid in overlap_ids:
        ra = by_a[cid]
        rb = by_b[cid]
        detail = detail_by_id[cid]
        lbl_same = ra["behavior"] == rb["behavior"]
        prim_pred_same = ra["primary_predicted_behavior"] == rb["primary_predicted_behavior"]
        prim_corr_same = ra["primary_correct"] == rb["primary_correct"]
        joint_pred_same = (
            ra["pf_joint_predicted_behavior"] == rb["pf_joint_predicted_behavior"]
            and ra["pf_joint_predicted_target_ids"] == rb["pf_joint_predicted_target_ids"]
        )
        joint_corr_same = ra["pf_joint_correct"] == rb["pf_joint_correct"]
        pf_pair_same = ra["pf_pair_predicted_behavior"] == rb["pf_pair_predicted_behavior"]

        label_agree += int(lbl_same)
        prim_pred_agree += int(prim_pred_same)
        prim_corr_agree += int(prim_corr_same)
        joint_pred_agree += int(joint_pred_same)
        joint_corr_agree += int(joint_corr_same)
        pf_pair_pred_agree += int(pf_pair_same)

        comparisons.append(
            {
                "clip_id": cid,
                "dataset_a_behavior": ra["behavior"],
                "dataset_b_behavior": rb["behavior"],
                "label_conflict": not lbl_same,
                "tracks_detections_differ": detail["tracks_detections_differ"],
                "annotation_differ": detail["annotation_differ"],
                "metadata_differ": detail["metadata_differ"],
                "a_primary_pred": ra["primary_predicted_behavior"],
                "b_primary_pred": rb["primary_predicted_behavior"],
                "a_primary_correct": ra["primary_correct"],
                "b_primary_correct": rb["primary_correct"],
                "primary_prediction_agree": prim_pred_same,
                "primary_correctness_agree": prim_corr_same,
                "a_pf_joint_pred": ra["pf_joint_predicted_behavior"],
                "b_pf_joint_pred": rb["pf_joint_predicted_behavior"],
                "a_pf_joint_ids": ra["pf_joint_predicted_target_ids"],
                "b_pf_joint_ids": rb["pf_joint_predicted_target_ids"],
                "pf_joint_prediction_agree": joint_pred_same,
                "a_pf_pair_pred": ra["pf_pair_predicted_behavior"],
                "b_pf_pair_pred": rb["pf_pair_predicted_behavior"],
            }
        )

    n = len(overlap_ids)
    per_beh: dict[str, dict[str, Any]] = {}
    for beh in SOCIAL_DISPLAY_BEHAVIORS:
        sub = [c for c in comparisons if c["dataset_b_behavior"] == beh]
        if not sub:
            continue
        sn = len(sub)
        per_beh[beh] = {
            "support_b_label": sn,
            "a_primary_correct": sum(1 for c in sub if c["a_primary_correct"]),
            "b_primary_correct": sum(1 for c in sub if c["b_primary_correct"]),
            "primary_prediction_agreement_count": sum(1 for c in sub if c["primary_prediction_agree"]),
            "primary_prediction_agreement_rate": _rate(sum(1 for c in sub if c["primary_prediction_agree"]), sn),
            "primary_correctness_agreement_count": sum(1 for c in sub if c["primary_correctness_agree"]),
            "primary_correctness_agreement_rate": _rate(sum(1 for c in sub if c["primary_correctness_agree"]), sn),
            "tracks_detections_differ_count": sum(1 for c in sub if c["tracks_detections_differ"]),
        }

    return {
        "overlap_count": n,
        "label_agreement_count": label_agree,
        "label_agreement_rate": _rate(label_agree, n),
        "primary_prediction_agreement_count": prim_pred_agree,
        "primary_prediction_agreement_rate": _rate(prim_pred_agree, n),
        "primary_correctness_agreement_count": prim_corr_agree,
        "primary_correctness_agreement_rate": _rate(prim_corr_agree, n),
        "pf_pair_prediction_agreement_count": pf_pair_pred_agree,
        "pf_pair_prediction_agreement_rate": _rate(pf_pair_pred_agree, n),
        "pf_joint_prediction_agreement_count": joint_pred_agree,
        "pf_joint_prediction_agreement_rate": _rate(joint_pred_agree, n),
        "pf_joint_correctness_agreement_count": joint_corr_agree,
        "pf_joint_correctness_agreement_rate": _rate(joint_corr_agree, n),
        "per_behavior_agreement": per_beh,
        "comparisons": comparisons,
    }


def _load_source_scales_map(
    source_scales_path: Path,
    entries: Sequence[ClipIndexEntry],
    dataset_label: str,
) -> dict[str, dict[str, Any]]:
    scale_rows = json.loads(Path(source_scales_path).read_text(encoding="utf-8"))
    if not isinstance(scale_rows, list):
        raise ValueError(
            f"Dataset {dataset_label} source scales must be a list of sample_id/scale_config records"
        )
    scales_map: dict[str, dict[str, Any]] = {}
    for row in scale_rows:
        if not isinstance(row, dict) or not row.get("sample_id") or not isinstance(row.get("scale_config"), dict):
            raise ValueError(f"Dataset {dataset_label} source scales require sample_id and scale_config")
        sample_id = str(row["sample_id"])
        if sample_id in scales_map:
            raise ValueError(f"Duplicate Dataset {dataset_label} source scale: {sample_id}")
        scales_map[sample_id] = row
    missing = sorted({entry.clip_id for entry in entries} - scales_map.keys())
    if missing:
        raise ValueError(f"Dataset {dataset_label} source scales missing clips: {', '.join(missing)}")
    return scales_map


def _build_provenance(
    config_path: Path,
    *,
    dataset_b_source_scales: Path | None = None,
    dataset_source_scales: Path | None = None,
    extra_scripts: Sequence[str] = (),
) -> dict[str, Any]:
    source_paths = sorted((REPO_ROOT / "src" / "mouse_behavior").rglob("*.py"))
    script_names = ["_bootstrap.py", "generate_reconciled_failure_report.py", *extra_scripts]
    seen_scripts: set[str] = set()
    for name in script_names:
        if name in seen_scripts:
            continue
        seen_scripts.add(name)
        candidate = REPO_ROOT / "scripts" / name
        if candidate.is_file():
            source_paths.append(candidate)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    provenance: dict[str, Any] = {
        "git_head": git_head,
        "source_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path) for path in source_paths
        },
        "config_sha256": _sha256_file(Path(config_path)),
        "default_scale_config_sha256": _sha256_file(DEFAULT_CONFIG_PATH),
        "dataset_b_source_scales_sha256": (
            _sha256_file(Path(dataset_b_source_scales)) if dataset_b_source_scales is not None else None
        ),
    }
    if dataset_source_scales is not None:
        provenance["dataset_source_scales_sha256"] = _sha256_file(Path(dataset_source_scales))
    return provenance


def evaluate_single_dataset(
    dataset_root: Path,
    dataset_label: str = "C",
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    workers: int | None = None,
    all_behaviors: bool = True,
    dataset_source_scales: Path | None = None,
    use_metadata_distance_scale: bool = True,
) -> dict[str, Any]:
    """Index and evaluate one exported dataset under full 15-behavior reconciled metrics."""
    t0 = time.perf_counter()
    entries = index_dataset(dataset_root, dataset_label)
    scales_map = (
        _load_source_scales_map(dataset_source_scales, entries, dataset_label)
        if dataset_source_scales is not None
        else None
    )
    eff_workers = workers if workers is not None else min(max(os.cpu_count() or 4, 1), 14)
    evaluated = evaluate_entries(
        entries,
        config_path=config_path,
        workers=eff_workers,
        all_behaviors=all_behaviors,
        source_scales_by_dataset={dataset_label: scales_map} if scales_map is not None else None,
        use_metadata_distance_scale=use_metadata_distance_scale,
    )
    cohort_key = f"dataset_{dataset_label.lower()}"
    summary = aggregate_cohort_metrics(evaluated, cohort_key)
    meta_scale_count = sum(1 for entry in entries if entry.metadata_cm_per_pixel is not None)
    if scales_map is not None:
        scale_policy = "explicit_source_scales"
    elif use_metadata_distance_scale and meta_scale_count > 0:
        scale_policy = "metadata_distance_scale_when_present_else_body_length_7.918cm_default"
    else:
        scale_policy = "body_length_7.918cm_default"

    primary_failures = [
        {
            "record_key": r["record_key"],
            "dataset": r["dataset"],
            "clip_id": r["clip_id"],
            "behavior": r["behavior"],
            "annotation_behavior": r["annotation_behavior"],
            "primary_method": r["primary_method"],
            "target_mouse_ids": r["target_mouse_ids"],
            "primary_predicted_behavior": r["primary_predicted_behavior"],
            "pf_joint_predicted_behavior": r["pf_joint_predicted_behavior"],
            "pf_joint_predicted_target_ids": r["pf_joint_predicted_target_ids"],
            "pf_pair_predicted_behavior": r["pf_pair_predicted_behavior"],
            "failure_category": r["failure_category"],
            "failure_subcategory": r["failure_subcategory"],
            "failure_reason_zh": r["failure_reason_zh"],
        }
        for r in evaluated
        if not r["primary_correct"]
    ]
    elapsed_s = round(time.perf_counter() - t0, 2)
    payload = {
        "schema_version": "single_dataset_behavior_evaluation_v1",
        "dataset_label": dataset_label,
        "generated_at_epoch_s": round(time.time(), 3),
        "elapsed_seconds": elapsed_s,
        "workers": eff_workers,
        "provenance": _build_provenance(
            config_path,
            dataset_source_scales=dataset_source_scales,
            extra_scripts=("evaluate_cleaned_dataset_and_merge.py",),
        ),
        "inputs": {
            "dataset_root": str(dataset_root),
            "dataset_label": dataset_label,
            "config_path": str(config_path),
            "scale_policy": scale_policy,
            "dataset_source_scales": str(dataset_source_scales) if dataset_source_scales is not None else None,
            "use_metadata_distance_scale": use_metadata_distance_scale,
            "metadata_distance_scale_clip_count": meta_scale_count,
            "benchmark_scale_policy": "legacy_clip_benchmarks_keep_their_original_internal_scale",
            "social_config_name": "OPTIMIZED_SOCIAL_HEURISTIC_CONFIG",
            "social_config": OPTIMIZED_SOCIAL_HEURISTIC_CONFIG.as_dict(),
            "all_behaviors": all_behaviors,
            "enable_posture_screen": all_behaviors,
            "suspended_posture_behaviors": [] if all_behaviors else sorted(SUSPENDED_DISPLAY_BEHAVIORS),
            "coverage_a2_threshold_all_behaviors": 0.5,
            "coverage_denominator": "full_clip_frame_count",
            "primary_metric_definition": "分项检测通过率（不同专项方法，不等于统一Top-1）",
            "dual_label_credit_enabled": False,
        },
        "raw_behavior_counts": dict(sorted(Counter(e.annotation_behavior for e in entries).items())),
        "normalized_behavior_counts": dict(sorted(Counter(e.behavior for e in entries).items())),
        "scale_source_counts": dict(sorted(Counter(r["scale_source"] for r in evaluated).items())),
        "cohorts": {
            cohort_key: summary,
        },
        "primary_failures": primary_failures,
        "clips": evaluated,
    }
    return _sanitize_json_value(payload)


def evaluate_and_reconcile_datasets(
    *,
    dataset_a_root: Path = DEFAULT_DATASET_A_ROOT,
    dataset_b_root: Path = DEFAULT_DATASET_B_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    workers: int | None = None,
    all_behaviors: bool = False,
    dataset_b_source_scales: Path | None = None,
) -> dict[str, Any]:
    """Run full reconciled evaluation and failure diagnosis across Dataset A & Dataset B."""
    t0 = time.perf_counter()
    if (
        dataset_b_source_scales is None
        and Path(dataset_b_root).resolve() == DEFAULT_DATASET_B_ROOT.resolve()
        and DEFAULT_DATASET_B_SOURCE_SCALES.is_file()
    ):
        dataset_b_source_scales = DEFAULT_DATASET_B_SOURCE_SCALES
    entries_a = index_dataset(dataset_a_root, "A")
    entries_b = index_dataset(dataset_b_root, "B")
    reconciliation = reconcile_datasets(entries_a, entries_b)
    dataset_b_scales = (
        _load_source_scales_map(dataset_b_source_scales, entries_b, "B")
        if dataset_b_source_scales is not None
        else None
    )

    eff_workers = workers if workers is not None else min(max(os.cpu_count() or 4, 1), 14)
    all_entries = [*entries_a, *entries_b]
    evaluated = _evaluate_entries(
        all_entries, config_path=config_path, workers=eff_workers, all_behaviors=all_behaviors,
        dataset_b_scales=dataset_b_scales,
    )

    overlap_set = set(reconciliation["overlap_ids"])
    conflict_ids = {item["clip_id"] for item in reconciliation["label_conflicts"]}
    a_only_set = set(reconciliation["a_only_ids"])
    b_only_set = set(reconciliation["b_only_ids"])

    records_a = [r for r in evaluated if r["dataset"] == "A"]
    records_b = [r for r in evaluated if r["dataset"] == "B"]
    overlap_records_a = [r for r in records_a if r["clip_id"] in overlap_set]
    overlap_records_b = [r for r in records_b if r["clip_id"] in overlap_set]

    for r in records_a:
        cid = r["clip_id"]
        r["in_overlap"] = cid in overlap_set
        r["overlap_label_conflict"] = cid in conflict_ids
        r["in_union_b_priority"] = cid in a_only_set
        r["in_union_a_priority"] = True

    for r in records_b:
        cid = r["clip_id"]
        r["in_overlap"] = cid in overlap_set
        r["overlap_label_conflict"] = cid in conflict_ids
        r["in_union_b_priority"] = True
        r["in_union_a_priority"] = cid in b_only_set

    union_b_records = [*records_b, *[r for r in records_a if r["in_union_b_priority"]]]
    union_a_records = [*records_a, *[r for r in records_b if r["in_union_a_priority"]]]

    summary_a = aggregate_cohort_metrics(records_a, "dataset_a")
    summary_b = aggregate_cohort_metrics(records_b, "dataset_b")
    summary_overlap_a = aggregate_cohort_metrics(overlap_records_a, "overlap_a")
    summary_overlap_b = aggregate_cohort_metrics(overlap_records_b, "overlap_b")
    summary_union_b = aggregate_cohort_metrics(union_b_records, "union_b_priority")
    summary_union_a = aggregate_cohort_metrics(union_a_records, "union_a_priority")
    overlap_agreement = build_overlap_agreement(records_a, records_b, reconciliation)

    union_primary_failures = [
        {
            "record_key": r["record_key"],
            "dataset": r["dataset"],
            "clip_id": r["clip_id"],
            "behavior": r["behavior"],
            "primary_method": r["primary_method"],
            "target_mouse_ids": r["target_mouse_ids"],
            "primary_predicted_behavior": r["primary_predicted_behavior"],
            "pf_joint_predicted_behavior": r["pf_joint_predicted_behavior"],
            "pf_joint_predicted_target_ids": r["pf_joint_predicted_target_ids"],
            "pf_pair_predicted_behavior": r["pf_pair_predicted_behavior"],
            "failure_category": r["failure_category"],
            "failure_subcategory": r["failure_subcategory"],
            "failure_reason_zh": r["failure_reason_zh"],
        }
        for r in union_b_records
        if not r["primary_correct"]
    ]

    elapsed_s = round(time.perf_counter() - t0, 2)
    payload = {
        "schema_version": "reconciled_behavior_evaluation_v1",
        "generated_at_epoch_s": round(time.time(), 3),
        "elapsed_seconds": elapsed_s,
        "workers": eff_workers,
        "provenance": _build_provenance(config_path, dataset_b_source_scales=dataset_b_source_scales),
        "inputs": {
            "dataset_a_root": str(dataset_a_root),
            "dataset_b_root": str(dataset_b_root),
            "config_path": str(config_path),
            "scale_policy": "dataset_b_frozen_source_scales_a_default" if dataset_b_scales is not None else "body_length_7.918cm_default",
            "dataset_b_source_scales": str(dataset_b_source_scales) if dataset_b_source_scales is not None else None,
            "benchmark_scale_policy": "legacy_clip_benchmarks_keep_their_original_internal_scale",
            "social_config_name": "OPTIMIZED_SOCIAL_HEURISTIC_CONFIG",
            "social_config": OPTIMIZED_SOCIAL_HEURISTIC_CONFIG.as_dict(),
            "all_behaviors": all_behaviors,
            "enable_posture_screen": all_behaviors,
            "suspended_posture_behaviors": [] if all_behaviors else sorted(SUSPENDED_DISPLAY_BEHAVIORS),
            "coverage_a2_threshold_all_behaviors": 0.5,
            "coverage_denominator": "full_clip_frame_count",
            "primary_metric_definition": "分项检测通过率（不同专项方法，不等于统一Top-1）",
            "dual_label_credit_enabled": False,
        },
        "reconciliation": {
            key: value
            for key, value in reconciliation.items()
            if key not in {"overlap_ids", "a_only_ids", "b_only_ids", "overlap_details"}
        },
        "reconciliation_ids": {
            "overlap_ids": reconciliation["overlap_ids"],
            "a_only_ids": reconciliation["a_only_ids"],
            "b_only_ids": reconciliation["b_only_ids"],
        },
        "cohorts": {
            "dataset_a": summary_a,
            "dataset_b": summary_b,
            "overlap_a": summary_overlap_a,
            "overlap_b": summary_overlap_b,
            "union_b_priority": summary_union_b,
            "union_a_priority": summary_union_a,
        },
        "overlap_agreement": {
            k: v for k, v in overlap_agreement.items() if k != "comparisons"
        },
        "overlap_comparisons": overlap_agreement["comparisons"],
        "union_b_priority_primary_failures": union_primary_failures,
        "clips": evaluated,
    }
    return _sanitize_json_value(payload)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def build_summary_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise Markdown summary of all cohort tables and failure diagnoses."""
    rec = report["reconciliation"]
    cohorts = report["cohorts"]
    agr = report["overlap_agreement"]

    lines: list[str] = [
        "# Dataset A + Dataset B 全行为评测与失败诊断",
        "",
        "分项检测通过率（不同专项方法，不等于统一Top-1）。专项方法可能使用真值行为类别或目标 ID 提示；保留旧分项口径，不能作为无提示分类准确率。",
        "统一 exact-ID 覆盖：目标行为在完全相同目标 ID 集合上的事件帧并集 / 全片段帧数，15 类均要求 >=50%；同 ID 多事件合并去重，不拼接不同 ID。历史 Huddling A2 按最大单事件计算，全行为模式改为同 ID 多事件并集。",
        "严格盲选 `target-ID joint`（`pf_joint`）要求全体候选的唯一 Top-1 行为及完整目标 ID 集合同时正确，不授予双标签兼容分。",
        f"姿态规则实际启用：{report['inputs']['enable_posture_screen']}；all_behaviors={report['inputs'].get('all_behaviors', False)}。",
        "全行为模式的 Grooming/Rearing/Huddling 分项主指标采用 exact-ID >=50% 覆盖。Huddling 仅改变计分口径，不代表算法恢复。",
        f"尺度策略：{report['inputs']['scale_policy']}。Dataset A 保持默认逐片段尺度；Dataset B 仅在显式提供 source_scales 时使用对应片段的冻结尺度。",
        "原 clip_benchmark 专项保留内部尺度；统一覆盖与严格盲选使用本次显式尺度，不能将两种方法混称同一评测。",
        f"源码 Git HEAD：`{report.get('provenance', {}).get('git_head', '未记录')}`；实际工作区源码与配置 SHA-256 详见 JSON provenance。",
        "",
        "## 1. 数据集对齐与去重汇总",
        f"- Dataset A (`429 补全集`): **{rec['dataset_a_total']}** clips",
        f"- Dataset B (`605 清洗集`): **{rec['dataset_b_total']}** clips",
        f"- Shared Overlap: **{rec['overlap_count']}** clips "
        f"(A-only={rec['a_only_count']}, B-only={rec['b_only_count']}, label agreement={agr['label_agreement_count']}/{rec['overlap_count']})",
        f"- Deduplicated Union: **{rec['union_count']}** unique clip IDs (B-priority 清洗集优先)",
        f"- Label conflicts: {len(rec['label_conflicts'])}，具体冲突保留在 JSON reconciliation.label_conflicts。",
        f"- Shared Clip File Differences: tracks bytes differ={rec['overlap_file_differences']['tracks_bytes_differ']}/{rec['overlap_count']}, "
        f"parsed detections differ={rec['overlap_file_differences']['tracks_detections_differ']}/{rec['overlap_count']}, "
        f"annotation differ={rec['overlap_file_differences']['annotation_differ']}/{rec['overlap_count']}, "
        f"metadata differ={rec['overlap_file_differences']['metadata_differ']}/{rec['overlap_count']}",
        "",
        "## 2. 全部分母的总体指标对比",
        "",
        "| Cohort | Clips | 统一 exact-ID 覆盖 >=50% | 分项检测通过率 | Strict `pf_joint` | 分项失败 (Missed / Misclassified) |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for key, label in (
        ("dataset_a", "Dataset A"),
        ("dataset_b", "Dataset B"),
        ("overlap_a", "Overlap under A"),
        ("overlap_b", "Overlap under B"),
        ("union_b_priority", "Union B-priority"),
        ("union_a_priority", "Union A-priority sensitivity"),
    ):
        c = cohorts[key]
        lines.append(
            f"| {label} | {c['total_clips']} | "
            f"{c['overall_coverage_a2_correct']}/{c['total_clips']} ({_format_pct(c['overall_coverage_a2_accuracy'])}) | "
            f"{c['overall_primary_correct']}/{c['total_clips']} ({_format_pct(c['overall_primary_accuracy'])}) | "
            f"{c['overall_pf_joint_correct']}/{c['total_clips']} ({_format_pct(c['overall_pf_joint_accuracy'])}) | "
            f"{c['primary_failure_count']} ({c['primary_missed_count']} missed / {c['primary_misclassified_count']} misclassified) |"
        )

    lines.extend(
        [
            "",
            f"Overlap Agreement ({rec['overlap_count']} shared IDs): primary prediction agreement={agr['primary_prediction_agreement_count']}/{rec['overlap_count']} "
            f"({_format_pct(agr['primary_prediction_agreement_rate'])}), "
            f"primary correctness agreement={agr['primary_correctness_agreement_count']}/{rec['overlap_count']} "
            f"({_format_pct(agr['primary_correctness_agreement_rate'])}), "
            f"pf_joint agreement={agr['pf_joint_prediction_agreement_count']}/{rec['overlap_count']} "
            f"({_format_pct(agr['pf_joint_prediction_agreement_rate'])}).",
            "",
            "## 3. 各行为 A / B / Union 对照（所有类别保留）",
            "",
            "| Behavior | A 覆盖 >=50% | A Strict joint | B 覆盖 >=50% | B Strict joint | Union 覆盖 >=50% | Union Strict joint |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for behavior in BEHAVIOR_DISPLAY_ORDER:
        cells = []
        for cohort in ("dataset_a", "dataset_b", "union_b_priority"):
            row = cohorts[cohort]["per_behavior"].get(behavior)
            for metric in ("coverage_a2", "pf_joint"):
                cells.append(
                    f"{row[metric + '_correct']}/{row['support']} ({_format_pct(row[metric + '_accuracy'])})"
                    if row else "— (0 clips)"
                )
        lines.append(f"| {behavior} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 4. Union B-priority 分项专项方法明细",
            "",
            "| Behavior | Support | 专项方法 | 分项检测通过率 | Strict `pf_joint` | `pf_target` / `pf_pair` | Failures (Missed / Miscls) |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )

    union_b = cohorts["union_b_priority"]
    for beh, row in union_b["per_behavior"].items():
        extra_note = " (disabled_rule)" if not row["rule_enabled"] else ""
        if beh == "Huddling":
            target_str = (
                f"{row['pf_target_correct']}/{row['support']} ({_format_pct(row['pf_target_accuracy'])}) "
                f"[A1={row['a1_coverage_count']}/{row['support']}, A2={row['a2_coverage_count']}/{row['support']}]"
            )
        else:
            target_str = f"{row['pf_target_correct']}/{row['support']} ({_format_pct(row['pf_target_accuracy'])})"
        lines.append(
            f"| {beh}{extra_note} | {row['support']} | `{row['primary_method']}` | "
            f"{row['primary_correct']}/{row['support']} ({_format_pct(row['primary_accuracy'])}) | "
            f"{row['pf_joint_correct']}/{row['support']} ({_format_pct(row['pf_joint_accuracy'])}) | "
            f"{target_str} | "
            f"{row['primary_failure_count']} ({row['missed_count']} / {row['misclassified_count']}) |"
        )

    lines.extend(
        [
            "",
            "## 5. Union 分项失败子类分布",
            "",
            "| Subcategory | Count |",
            "|---|---:|",
        ]
    )
    for subcat, cnt in union_b["failure_subcategory_counts"].items():
        lines.append(f"| `{subcat}` | {cnt} |")
    attack_row = union_b["per_behavior"].get("Attack")
    if attack_row:
        lines.extend([
            "",
            "Attack 历史辅助口径（exact-ID 覆盖 >=30%，不计入上面的统一 >=50% 指标）："
            f"{attack_row['attack_historical_coverage_ge_0.3_count']}/{attack_row['support']} "
            f"({_format_pct(attack_row['attack_historical_coverage_ge_0.3_rate'])})。",
        ])
    lines.append("")
    return "\n".join(lines)


def write_reconciled_outputs(
    report: Mapping[str, Any],
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> dict[str, Path]:
    """Write the primary JSON report plus companion CSV and Markdown summary files."""
    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(_sanitize_json_value(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    stem = out_json.stem
    summary_csv = out_json.with_name(f"{stem}_summary.csv")
    clips_csv = out_json.with_name(f"{stem}_clips.csv")
    failures_csv = out_json.with_name(f"{stem}_failures.csv")
    summary_md = out_json.with_name(f"{stem}.md")

    summary_fields = (
        "cohort",
        "behavior",
        "layer",
        "primary_method",
        "support",
        "primary_correct",
        "primary_accuracy",
        "coverage_a2_correct",
        "coverage_a2_accuracy",
        "mean_true_target_coverage",
        "pf_joint_correct",
        "pf_joint_accuracy",
        "pf_target_correct",
        "pf_target_accuracy",
        "hit_mask_count",
        "top1_max_correct",
        "top1_prio_correct",
        "a1_coverage_count",
        "a2_coverage_count",
        "primary_failure_count",
        "missed_count",
        "misclassified_count",
    )
    summary_rows: list[dict[str, Any]] = []
    for cohort_name, cohort in report["cohorts"].items():
        for beh, row in cohort["per_behavior"].items():
            summary_rows.append(
                {
                    "cohort": cohort_name,
                    "behavior": beh,
                    "layer": row["layer"],
                    "primary_method": row["primary_method"],
                    "support": row["support"],
                    "primary_correct": row["primary_correct"],
                    "primary_accuracy": row["primary_accuracy"],
                    "coverage_a2_correct": row["coverage_a2_correct"],
                    "coverage_a2_accuracy": row["coverage_a2_accuracy"],
                    "mean_true_target_coverage": row["mean_true_target_coverage"],
                    "pf_joint_correct": row["pf_joint_correct"],
                    "pf_joint_accuracy": row["pf_joint_accuracy"],
                    "pf_target_correct": row["pf_target_correct"],
                    "pf_target_accuracy": row["pf_target_accuracy"],
                    "hit_mask_count": row.get("hit_mask_count", ""),
                    "top1_max_correct": row.get("top1_max_correct", ""),
                    "top1_prio_correct": row.get("top1_prio_correct", ""),
                    "a1_coverage_count": row.get("a1_coverage_count", ""),
                    "a2_coverage_count": row.get("a2_coverage_count", ""),
                    "primary_failure_count": row["primary_failure_count"],
                    "missed_count": row["missed_count"],
                    "misclassified_count": row["misclassified_count"],
                }
            )
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    clip_fields = (
        "record_key",
        "dataset",
        "clip_id",
        "behavior",
        "in_overlap",
        "overlap_label_conflict",
        "in_union_b_priority",
        "target_mouse_ids",
        "primary_method",
        "primary_predicted_behavior",
        "primary_correct",
        "true_target_coverage",
        "coverage_a2_correct",
        "cm_per_pixel",
        "scale_source",
        "enable_posture_screen",
        "temporal_iou",
        "pf_joint_predicted_behavior",
        "pf_joint_predicted_target_ids",
        "pf_joint_correct",
        "pf_target_predicted_behavior",
        "pf_target_correct",
        "hit_mask",
        "top1_max_behavior",
        "top1_prio_behavior",
        "failure_category",
        "failure_subcategory",
        "failure_reason_zh",
        "joint_failure_category",
        "joint_failure_subcategory",
        "joint_failure_reason_zh",
    )

    def _clip_row(r: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "record_key": r["record_key"],
            "dataset": r["dataset"],
            "clip_id": r["clip_id"],
            "behavior": r["behavior"],
            "in_overlap": r["in_overlap"],
            "overlap_label_conflict": r["overlap_label_conflict"],
            "in_union_b_priority": r["in_union_b_priority"],
            "target_mouse_ids": ",".join(map(str, r["target_mouse_ids"])),
            "primary_method": r["primary_method"],
            "primary_predicted_behavior": r["primary_predicted_behavior"] or "",
            "primary_correct": r["primary_correct"],
            "true_target_coverage": r["true_target_coverage"],
            "coverage_a2_correct": r["coverage_a2_correct"],
            "cm_per_pixel": r["cm_per_pixel"],
            "scale_source": r["scale_source"],
            "enable_posture_screen": r["enable_posture_screen"],
            "temporal_iou": r["temporal_iou"] if r["temporal_iou"] is not None else "",
            "pf_joint_predicted_behavior": r["pf_joint_predicted_behavior"] or "",
            "pf_joint_predicted_target_ids": ",".join(map(str, r["pf_joint_predicted_target_ids"])),
            "pf_joint_correct": r["pf_joint_correct"],
            "pf_target_predicted_behavior": r["pf_target_predicted_behavior"] or "",
            "pf_target_correct": r["pf_target_correct"],
            "hit_mask": r["hit_mask"] if r["hit_mask"] is not None else "",
            "top1_max_behavior": r["top1_max_behavior"] or "",
            "top1_prio_behavior": r["top1_prio_behavior"] or "",
            "failure_category": r["failure_category"] or "",
            "failure_subcategory": r["failure_subcategory"] or "",
            "failure_reason_zh": r["failure_reason_zh"] or "",
            "joint_failure_category": r["joint_failure_category"] or "",
            "joint_failure_subcategory": r["joint_failure_subcategory"] or "",
            "joint_failure_reason_zh": r["joint_failure_reason_zh"] or "",
        }

    clip_rows = [_clip_row(r) for r in report["clips"]]
    with clips_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=clip_fields)
        writer.writeheader()
        writer.writerows(clip_rows)

    failure_rows = [_clip_row(r) for r in report["clips"] if not r["primary_correct"]]
    with failures_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=clip_fields)
        writer.writeheader()
        writer.writerows(failure_rows)

    md_text = build_summary_markdown(report)
    summary_md.write_text(md_text, encoding="utf-8")
    return {
        "json": out_json,
        "summary_csv": summary_csv,
        "clips_csv": clips_csv,
        "failures_csv": failures_csv,
        "markdown": summary_md,
    }


__all__ = [
    "BEHAVIOR_DISPLAY_ORDER",
    "CANONICAL_TO_DISPLAY",
    "ClipIndexEntry",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATASET_A_ROOT",
    "DEFAULT_DATASET_B_ROOT",
    "DEFAULT_DATASET_B_SOURCE_SCALES",
    "DEFAULT_OUTPUT_JSON",
    "DISPLAY_BEHAVIOR_ALIASES",
    "DISPLAY_TO_CANONICAL",
    "FAILURE_SUBCATEGORIES",
    "GROUP_DISPLAY_BEHAVIORS",
    "INDIVIDUAL_DISPLAY_BEHAVIORS",
    "PRIMARY_METHOD_BY_BEHAVIOR",
    "SOCIAL_DISPLAY_BEHAVIORS",
    "SUSPENDED_DISPLAY_BEHAVIORS",
    "aggregate_cohort_metrics",
    "build_overlap_agreement",
    "build_summary_markdown",
    "evaluate_and_reconcile_datasets",
    "evaluate_entries",
    "evaluate_single_dataset",
    "index_dataset",
    "normalize_display_behavior",
    "reconcile_datasets",
    "write_reconciled_outputs",
]
