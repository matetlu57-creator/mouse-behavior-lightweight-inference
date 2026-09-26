"""Adapter and compact heuristics for the exported annotation dataset.

The annotation export used for calibration is intentionally kept separate from
the YOLO cache loader.  It contains one short ``tracks.json`` per annotated
clip, while the production pipeline consumes a continuous Ultralytics cache.
This module converts one export sample into scale-normalized kinematic
features and evaluates the same distance/speed/duration concepts used by the
extended ethogram.  It never creates or repairs biological identities: the
export's ``track_id`` values are only clip-local visual identifiers.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RETAINED_BEHAVIORS = (
    "Approach",
    "Attack",
    "Avoiding",
    "Chasing",
    "Dispersal",
    "Following",
    "Group locomotion",
    "Huddling",
    "Isolation",
    "Running",
    "Snout-head_contact",
    "Snout-rear_contact",
    "Social clustering",
    "Static",
    "Together",
    "Walking",
)

LABEL_TO_CANONICAL = {
    "Approach": "approach",
    "Attack": "attack",
    "Avoiding": "avoidance",
    "Chasing": "chase",
    "Dispersal": "dispersal",
    "Following": "following",
    "Group locomotion": "group_locomotion",
    "Huddling": "huddle",
    "Isolation": "isolation",
    "Running": "running",
    "Snout-head_contact": "nose_head_contact",
    "Snout-rear_contact": "nose_tail_contact",
    "Social clustering": "social_clustering",
    "Static": "stationary",
    "Together": "together",
    "Walking": "walking",
}

CANONICAL_BEHAVIORS = tuple(LABEL_TO_CANONICAL.values())
INDIVIDUAL_BEHAVIORS = ("running", "walking", "stationary")
PAIR_BEHAVIORS = (
    "approach",
    "together",
    "chase",
    "avoidance",
    "attack",
    "nose_head_contact",
    "nose_tail_contact",
    "following",
)
GROUP_BEHAVIORS = (
    "huddle",
    "social_clustering",
    "dispersal",
    "group_locomotion",
    "isolation",
)
_PART_SUFFIX = re.compile(r"_part\d+(?=\.mp4$)", re.IGNORECASE)


@dataclass(frozen=True)
class SampleRecord:
    """A manifest row for one exported annotation clip."""

    sample_id: str
    behavior: str
    canonical_behavior: str
    sample_dir: str
    annotation_path: str
    metadata_path: str
    tracks_path: str
    clip_path: str
    source_video: str
    recording_session: str
    fps: float
    frame_count: int
    mouse_ids: tuple[int, ...]
    confidence: str
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float

    def as_row(self, split: str | None = None) -> dict[str, Any]:
        row = asdict(self)
        row["mouse_ids"] = ",".join(str(value) for value in self.mouse_ids)
        if split is not None:
            row["split"] = split
        return row


@dataclass
class ExportedFeatures:
    """Scale-normalized features for one clip.

    Pair arrays are retained only for the current sample.  The evaluator
    processes samples sequentially so a long multi-animal clip cannot make a
    full-dataset pair tensor resident in memory.
    """

    fps: float
    track_ids: np.ndarray
    valid: np.ndarray
    centers_cm: np.ndarray
    speed_cm_s: np.ndarray
    velocity_cm_s: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_distance_cm: np.ndarray
    pair_closing_speed_cm_s: np.ndarray
    pair_direction_similarity: np.ndarray
    pair_nose_head_cm: np.ndarray
    pair_nose_tail_cm: np.ndarray
    nearest_distance_cm: np.ndarray
    mean_nearest_distance_cm: np.ndarray
    cm_per_pixel: float
    track_selection_truncated: bool
    selected_track_count: int
    mouse_width_cm: float = 2.4


@dataclass(frozen=True)
class HeuristicParameters:
    """Tunable parameters for the exported-track calibration adapter."""

    # Train-only tuning moved the Static/Walking boundary to 10 cm/s.
    stationary_max_speed_cm_s: float = 10.0
    walking_min_speed_cm_s: float = 10.0
    # Calibration established 85 cm/s as the inclusive Walking ceiling;
    # speeds above it are reserved for a future Running label.
    walking_max_speed_cm_s: float = 85.0
    walking_max_stop_gap_s: float = 0.5
    running_min_speed_cm_s: float = 85.0
    running_min_directionality: float = 0.55
    running_min_duration_s: float = 0.5
    together_max_distance_cm: float = 8.0
    # Together is a low-motion pair state; brief movement is allowed, but
    # sustained movement above the Static boundary is not.
    together_max_individual_speed_cm_s: float = 16.0
    together_max_combined_speed_cm_s: float = 28.0
    pair_max_distance_cm: float = 5.0
    approach_min_distance_drop_cm: float = 1.5
    approach_min_closing_speed_cm_s: float = 2.0
    approach_terminal_distance_cm: float = 17.0
    contact_distance_cm: float = 2.0
    # Nose-contact labels require at least this much cumulative observed
    # contact time for the same visual-ID pair; brief repeated touches below
    # the threshold do not become a standalone contact behavior.
    contact_min_cumulative_seconds: float = 0.5
    attack_min_speed_cm_s: float = 8.0
    attack_vs_approach_multiplier: float = 1.5
    attack_endpoint_distance_cm: float = 12.0
    attack_reacquisition_max_gap_s: float = 5.0
    chase_min_direction_similarity: float = 0.65
    chase_min_duration_s: float = 2.0
    avoiding_min_distance_increase_cm: float = 3.0
    avoiding_min_duration_s: float = 0.5
    following_min_distance_cm: float = 5.0
    following_max_distance_cm: float = 30.0
    following_min_direction_similarity: float = 0.70
    following_min_duration_s: float = 3.0
    pair_no_contact_seconds: float = 0.5
    huddle_distance_cm: float = 5.0
    huddle_width_multiplier: float = 2.0
    huddle_max_mean_speed_cm_s: float = 10.0
    # Stable tracked IDs can be reacquired after a longer detector dropout.
    # Bridge only when the same members reappear in the same group lineage.
    huddle_min_duration_s: float = 1.0
    huddle_max_gap_s: float = 5.0
    huddle_min_member_support_fraction: float = 0.25
    # A single mouse is isolated only when its nearest visible companion is
    # beyond this threshold.
    isolation_distance_cm: float = 8.0
    # Isolation is a sustained spatial relation, not a one-frame outlier.
    isolation_min_duration_s: float = 10.0
    clustering_max_distance_cm: float = 30.0
    clustering_initial_max_distance_cm: float = 24.0
    clustering_min_nearest_neighbor_drop_cm: float = 2.0
    stationary_min_duration_s: float = 1.0
    walking_min_duration_s: float = 1.0
    pair_min_duration_s: float = 0.1
    together_min_duration_s: float = 1.0
    group_min_duration_s: float = 0.3
    approach_window_s: float = 0.3
    clustering_window_s: float = 2.0
    clustering_min_duration_s: float = 5.0
    clustering_max_gap_s: float = 0.25
    # A qualifying nearest-neighbour decrease is explicit formation evidence;
    # it should outrank a simultaneous quiet huddle of the same IDs.
    clustering_formation_score_bonus: float = 1.0
    # Social clustering describes the formation/migration phase, so require
    # some group motion while Huddling retains its low-motion gate.
    clustering_min_mean_speed_cm_s: float = 3.0
    clustering_min_moving_member_fraction: float = 0.5
    # A fleeting visitor is not a persistent member of a forming group.
    clustering_min_member_support_fraction: float = 0.0
    # Optional train-calibrated path for clips that begin with a close group
    # but retain measurable collective displacement. Zero disables it.
    clustering_motion_max_distance_cm: float = 0.0
    clustering_motion_min_displacement_cm: float = 6.0
    clustering_motion_loose_pair_distance_cm: float = 9.5
    group_locomotion_max_distance_cm: float = 30.0
    group_locomotion_min_direction_similarity: float = 0.70
    group_locomotion_min_walking_fraction: float = 0.60
    group_locomotion_min_duration_s: float = 3.0
    dispersal_start_distance_cm: float = 15.0
    dispersal_min_distance_increase_cm: float = 5.0
    dispersal_min_duration_s: float = 10.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _session_name(source_video: str, sample_dir: Path) -> str:
    source = str(source_video or "").strip()
    if not source:
        source = sample_dir.name
    return _PART_SUFFIX.sub("", Path(source).name)


def discover_samples(dataset_root: str | Path) -> list[SampleRecord]:
    """Read the exported annotation contract without touching video payloads."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    records: list[SampleRecord] = []
    for behavior_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for sample_dir in sorted(path for path in behavior_dir.iterdir() if path.is_dir()):
            annotation_path = sample_dir / "annotation.json"
            metadata_path = sample_dir / "metadata.json"
            tracks_path = sample_dir / "tracks.json"
            if not annotation_path.is_file() or not metadata_path.is_file() or not tracks_path.is_file():
                continue
            annotation = _read_json(annotation_path)
            metadata = _read_json(metadata_path)
            behavior = str(annotation.get("behavior", behavior_dir.name)).strip()
            if behavior != behavior_dir.name:
                raise ValueError(
                    f"behavior directory/annotation mismatch: {sample_dir} "
                    f"({behavior_dir.name!r} vs {behavior!r})"
                )
            clip = dict(metadata.get("clip", {}))
            source = dict(metadata.get("source", {}))
            frame_range = dict(annotation.get("frame_range", {}))
            time_range = dict(annotation.get("time_range", {}))
            source_video = str(source.get("video_filename", clip.get("filename", "clip.mp4")))
            mouse_ids = tuple(sorted({int(value) for value in annotation.get("mouse_ids", [])}))
            records.append(
                SampleRecord(
                    sample_id=sample_dir.relative_to(root).as_posix(),
                    behavior=behavior,
                    canonical_behavior=LABEL_TO_CANONICAL.get(behavior, behavior.lower()),
                    sample_dir=str(sample_dir),
                    annotation_path=str(annotation_path),
                    metadata_path=str(metadata_path),
                    tracks_path=str(tracks_path),
                    clip_path=str(sample_dir / str(clip.get("filename", "clip.mp4"))),
                    source_video=source_video,
                    recording_session=_session_name(source_video, sample_dir),
                    fps=float(clip.get("fps", 30.0) or 30.0),
                    frame_count=int(clip.get("frame_count", 0) or 0),
                    mouse_ids=mouse_ids,
                    confidence=str(annotation.get("confidence", "")),
                    frame_start=int(frame_range.get("start", 0) or 0),
                    frame_end=int(frame_range.get("end", 0) or 0),
                    time_start=float(time_range.get("start", 0.0) or 0.0),
                    time_end=float(time_range.get("end", 0.0) or 0.0),
                )
            )
    if not records:
        raise FileNotFoundError(f"no exported annotation samples below: {root}")
    return records


def filter_behaviors(
    records: Sequence[SampleRecord],
    *,
    minimum_count: int = 50,
) -> tuple[list[SampleRecord], dict[str, int], dict[str, int]]:
    """Keep only behavior folders meeting the user-provided count threshold."""

    counts = Counter(record.behavior for record in records)
    retained = {behavior for behavior, count in counts.items() if count >= int(minimum_count)}
    kept = [record for record in records if record.behavior in retained]
    return kept, dict(sorted(counts.items())), {
        behavior: counts[behavior] for behavior in sorted(counts) if behavior not in retained
    }


def split_by_recording_session(
    records: Sequence[SampleRecord],
    *,
    validation_sessions: Iterable[str] | None = None,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split without putting clips from one source recording in both sets."""

    if validation_sessions is None:
        validation_sessions = {
            "large_arena_2.mp4",
            "WIN_20260821_14_20_49_Pro.mp4",
            "WIN_20260822_18_03_51_Pro.mp4",
            "WIN_20260822_18_53_17_Pro.mp4",
        }
    validation = set(validation_sessions)
    available = {record.recording_session for record in records}
    missing = validation - available
    if missing:
        raise ValueError(f"validation sessions are not present in the dataset: {sorted(missing)}")
    train = [record for record in records if record.recording_session not in validation]
    valid = [record for record in records if record.recording_session in validation]
    if not train or not valid:
        raise ValueError("session split must produce non-empty train and validation sets")
    train_behaviors = {record.behavior for record in train}
    valid_behaviors = {record.behavior for record in valid}
    missing_train = set(RETAINED_BEHAVIORS).intersection({record.behavior for record in records}) - train_behaviors
    missing_valid = set(RETAINED_BEHAVIORS).intersection({record.behavior for record in records}) - valid_behaviors
    if missing_train or missing_valid:
        raise ValueError(
            f"each retained behavior needs both splits; train_missing={sorted(missing_train)}, "
            f"validation_missing={sorted(missing_valid)}"
        )
    return train, valid


def write_split_manifest(
    output_dir: str | Path,
    *,
    all_records: Sequence[SampleRecord],
    train_records: Sequence[SampleRecord],
    validation_records: Sequence[SampleRecord],
    counts: Mapping[str, int],
    excluded_counts: Mapping[str, int],
    minimum_count_inclusive: int | None = 50,
) -> dict[str, Any]:
    """Write human-readable CSVs and a machine-readable split manifest."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_ids = {record.sample_id for record in train_records}
    valid_ids = {record.sample_id for record in validation_records}
    if train_ids.intersection(valid_ids):
        raise ValueError("train and validation sample IDs overlap")

    def write_csv(path: Path, rows: Sequence[SampleRecord], split: str) -> None:
        data = [record.as_row(split) for record in rows]
        fieldnames = list(data[0].keys()) if data else ["sample_id", "split"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv(out / "train.csv", train_records, "train")
    write_csv(out / "val.csv", validation_records, "validation")
    manifest = {
        "schema_version": 1,
        "dataset_root": str(Path(all_records[0].sample_dir).parents[1]) if all_records else None,
        "filter": {
            "minimum_count_inclusive": minimum_count_inclusive,
            "retained_behaviors": sorted(counts),
            "excluded_behaviors": dict(excluded_counts),
        },
        "split": {
            "strategy": "recording_session_holdout",
            "train_count": len(train_records),
            "validation_count": len(validation_records),
            "train_fraction": len(train_records) / max(len(train_records) + len(validation_records), 1),
            "validation_fraction": len(validation_records) / max(len(train_records) + len(validation_records), 1),
            "train_sessions": sorted({record.recording_session for record in train_records}),
            "validation_sessions": sorted({record.recording_session for record in validation_records}),
        },
        "counts": {
            "all_discovered": sum(int(value) for value in counts.values())
            + sum(int(value) for value in excluded_counts.values()),
            "retained": sum(int(value) for value in counts.values()),
            "excluded": sum(int(value) for value in excluded_counts.values()),
            "retained_by_behavior": dict(sorted(counts.items())),
            "train_by_behavior": dict(sorted(Counter(record.behavior for record in train_records).items())),
            "validation_by_behavior": dict(
                sorted(Counter(record.behavior for record in validation_records).items())
            ),
        },
        "files": {"train_csv": "train.csv", "validation_csv": "val.csv"},
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _safe_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _huddle_pair_distance_cm(
    features: ExportedFeatures, parameters: HeuristicParameters
) -> float:
    """Return the Huddling width-based cutoff; Together has its own pair limit."""
    return max(
        min(
            float(features.mouse_width_cm) * float(parameters.huddle_width_multiplier),
            max(float(parameters.huddle_distance_cm), 0.5),
        ),
        0.5,
    )


def load_exported_features(
    tracks_path: str | Path,
    *,
    fps: float,
    max_tracks: int = 64,
) -> ExportedFeatures:
    """Load label-free trajectory input and derive scale-normalized features."""

    path = Path(tracks_path)
    raw = _read_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"tracks.json must contain a non-empty frame list: {path}")
    frame_numbers = [int(frame.get("frame", index)) for index, frame in enumerate(raw)]
    frame_count = max(frame_numbers) + 1
    counts = Counter(
        int(detection["track_id"])
        for frame in raw
        for detection in frame.get("detections", [])
        if "track_id" in detection
    )
    selected_ids = sorted(
        (track_id for track_id, _ in counts.most_common(max(int(max_tracks), 2))),
    )
    track_index = {track_id: index for index, track_id in enumerate(selected_ids)}
    tracks = len(selected_ids)
    keypoints = np.full((frame_count, tracks, 7, 2), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count, tracks), dtype=bool)
    for frame in raw:
        frame_index = int(frame.get("frame", 0))
        if frame_index < 0 or frame_index >= frame_count:
            continue
        for detection in frame.get("detections", []):
            track_id = int(detection.get("track_id", -1))
            slot = track_index.get(track_id)
            if slot is None:
                continue
            points = np.asarray(detection.get("keypoints", []), dtype=float)
            if points.shape != (7, 3):
                continue
            # Export order is nose, right ear, left ear, neck, hips, tail;
            # internal behavior geometry uses left ear then right ear.
            points = points[[0, 2, 1, 3, 4, 5, 6]]
            points_xy = points[:, :2].copy()
            point_valid = (
                np.isfinite(points_xy).all(axis=1)
                & np.isfinite(points[:, 2])
                & (points[:, 2] >= 0.25)
            )
            points_xy[~point_valid] = np.nan
            keypoints[frame_index, slot] = points_xy
            valid[frame_index, slot] = bool(point_valid.any())

        nose = keypoints[:, :, 0]
    tail = keypoints[:, :, 6]
    center_points = keypoints[:, :, [3, 4, 5]]
    center_good = np.isfinite(center_points).all(axis=3)
    body_centers_px = np.ma.median(
        np.ma.masked_where(
            np.broadcast_to(~center_good[..., None], center_points.shape), center_points
        ), axis=2
    ).filled(np.nan)
    any_good = np.isfinite(keypoints).all(axis=3)
    fallback_centers_px = np.ma.median(
        np.ma.masked_where(
            np.broadcast_to(~any_good[..., None], keypoints.shape), keypoints
        ), axis=2
    ).filled(np.nan)
    centers_px = np.where(
        np.isfinite(body_centers_px).all(axis=2, keepdims=True),
        body_centers_px,
        fallback_centers_px,
    )
    valid &= np.isfinite(centers_px).all(axis=2)
    centers_px[~valid] = np.nan
    body_lengths = np.linalg.norm(nose - tail, axis=2)
    body_lengths[~valid] = np.nan
    per_track_scales = []
    for slot in range(tracks):
        values = body_lengths[:, slot]
        values = values[np.isfinite(values) & (values >= 10.0) & (values <= 300.0)]
        if values.size:
            per_track_scales.append(float(np.median(values)))
    reference_body_px = float(np.median(per_track_scales)) if per_track_scales else 60.0
    cm_per_pixel = 8.0 / max(reference_body_px, 1e-6)
    # Hip span is a better body-width proxy than ear span; ears are fallback
    # evidence only when no reliable hip-width samples exist.
    hip_widths_px = np.linalg.norm(keypoints[:, :, 4] - keypoints[:, :, 5], axis=2)
    hip_widths_px = hip_widths_px[
        valid & np.isfinite(hip_widths_px) & (hip_widths_px >= 1.0)
    ]
    ear_widths_px = np.linalg.norm(keypoints[:, :, 1] - keypoints[:, :, 2], axis=2)
    ear_widths_px = ear_widths_px[valid & np.isfinite(ear_widths_px) & (ear_widths_px >= 1.0)]
    width_samples = hip_widths_px if hip_widths_px.size else ear_widths_px
    mouse_width_cm = (
        float(np.median(width_samples) * cm_per_pixel)
        if width_samples.size
        else 2.4
    )
    centers_cm = centers_px * cm_per_pixel
    nose_cm = nose * cm_per_pixel
    tail_cm = tail * cm_per_pixel
    head_cm = _safe_mean(keypoints[:, :, [0, 1, 2]] * cm_per_pixel, axis=2)

    velocity = np.zeros((frame_count, tracks, 2), dtype=np.float32)
    if frame_count > 1:
        delta = centers_cm[1:] - centers_cm[:-1]
        continuity = valid[1:] & valid[:-1] & np.all(np.isfinite(delta), axis=2)
        velocity[1:][continuity] = delta[continuity] * float(fps)
    speed = np.linalg.norm(velocity, axis=2).astype(np.float32)
    speed[~valid] = 0.0
    heading = nose_cm - keypoints[:, :, 3] * cm_per_pixel
    heading_norm = np.linalg.norm(heading, axis=2, keepdims=True)
    velocity_norm = np.linalg.norm(velocity, axis=2, keepdims=True)
    heading = np.divide(
        heading,
        heading_norm,
        out=np.zeros_like(heading, dtype=float),
        where=heading_norm > 1e-6,
    )
    velocity_unit = np.divide(
        velocity,
        velocity_norm,
        out=np.zeros_like(velocity, dtype=float),
        where=velocity_norm > 1e-6,
    )
    heading = np.where((heading_norm > 1e-6), heading, velocity_unit)
    pair_i, pair_j = np.triu_indices(tracks, k=1)
    pairs = len(pair_i)
    pair_distance = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    pair_direction = np.zeros((frame_count, pairs), dtype=np.float32)
    pair_head = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    pair_tail = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    for frame in range(frame_count):
        frame_valid = valid[frame]
        if pairs == 0:
            continue
        good = frame_valid[pair_i] & frame_valid[pair_j]
        if not np.any(good):
            continue
        left = centers_cm[frame, pair_i[good]]
        right = centers_cm[frame, pair_j[good]]
        distances = np.linalg.norm(left - right, axis=1)
        pair_distance[frame, good] = np.where(np.isfinite(distances), distances, np.inf)
        pair_direction[frame, good] = np.clip(
            np.sum(velocity_unit[frame, pair_i[good]] * velocity_unit[frame, pair_j[good]], axis=1),
            -1.0,
            1.0,
        )
        left_nose = nose_cm[frame, pair_i[good]]
        right_nose = nose_cm[frame, pair_j[good]]
        left_head = head_cm[frame, pair_i[good]]
        right_head = head_cm[frame, pair_j[good]]
        left_tail = tail_cm[frame, pair_i[good]]
        right_tail = tail_cm[frame, pair_j[good]]
        pair_head[frame, good] = np.minimum(
            np.linalg.norm(left_nose - right_head, axis=1),
            np.linalg.norm(right_nose - left_head, axis=1),
        )
        pair_tail[frame, good] = np.minimum(
            np.linalg.norm(left_nose - right_tail, axis=1),
            np.linalg.norm(right_nose - left_tail, axis=1),
        )
    closing = np.zeros_like(pair_distance, dtype=np.float32)
    if frame_count > 1:
        previous = pair_distance[:-1]
        current = pair_distance[1:]
        finite = np.isfinite(previous) & np.isfinite(current)
        closing[1:][finite] = (previous[finite] - current[finite]) * float(fps)
    nearest = np.min(pair_distance, axis=1) if pairs else np.full((frame_count, tracks), np.inf)
    if pairs:
        nearest_by_track = np.full((frame_count, tracks), np.inf, dtype=np.float32)
        for pair_index, (left, right) in enumerate(zip(pair_i, pair_j)):
            nearest_by_track[:, left] = np.minimum(nearest_by_track[:, left], pair_distance[:, pair_index])
            nearest_by_track[:, right] = np.minimum(nearest_by_track[:, right], pair_distance[:, pair_index])
        nearest = nearest_by_track
    mean_nearest = _safe_mean(np.where(np.isfinite(nearest), nearest, np.nan), axis=1)
    mean_nearest = np.asarray(mean_nearest, dtype=np.float32)
    mean_nearest[~np.isfinite(mean_nearest)] = np.nan
    return ExportedFeatures(
        fps=max(float(fps), 1e-6),
        track_ids=np.asarray(selected_ids, dtype=int),
        valid=valid,
        centers_cm=centers_cm.astype(np.float32),
        speed_cm_s=speed,
        velocity_cm_s=velocity,
        pair_i=pair_i.astype(int),
        pair_j=pair_j.astype(int),
        pair_distance_cm=pair_distance,
        pair_closing_speed_cm_s=closing,
        pair_direction_similarity=pair_direction,
        pair_nose_head_cm=pair_head,
        pair_nose_tail_cm=pair_tail,
        nearest_distance_cm=nearest.astype(np.float32),
        mean_nearest_distance_cm=mean_nearest,
        cm_per_pixel=float(cm_per_pixel),
        track_selection_truncated=len(counts) > tracks,
        selected_track_count=tracks,
        mouse_width_cm=mouse_width_cm,
    )


def _longest_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _bridge_short_false_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Bridge only interior false runs, preserving clip-boundary gaps."""

    result = np.asarray(mask, dtype=bool).copy()
    if result.ndim != 1 or max_gap_frames <= 0:
        return result
    true_indices = np.flatnonzero(result)
    for before, after in zip(true_indices[:-1], true_indices[1:]):
        gap = int(after - before - 1)
        if 0 < gap <= int(max_gap_frames):
            result[before + 1:after] = True
    return result


def _duration_score(mask: np.ndarray, fps: float, minimum_seconds: float) -> tuple[bool, float]:
    duration = _longest_run(mask) / max(float(fps), 1e-6)
    minimum = max(float(minimum_seconds), 1.0 / max(float(fps), 1e-6))
    return duration >= minimum, duration


def _max_track_duration(mask: np.ndarray, fps: float) -> float:
    if mask.ndim != 2 or not mask.size:
        return 0.0
    return max((_longest_run(mask[:, index]) for index in range(mask.shape[1])), default=0) / max(fps, 1e-6)


def _pair_any_duration(mask: np.ndarray, fps: float) -> float:
    if mask.ndim != 2 or not mask.size:
        return 0.0
    return _longest_run(np.any(mask, axis=1)) / max(fps, 1e-6)


def _durations_by_column(mask: np.ndarray, fps: float) -> np.ndarray:
    """Return the longest continuous run in seconds for every identity/pair column."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"duration mask must be two-dimensional, got shape={values.shape}")
    longest = np.zeros(values.shape[1], dtype=np.int64)
    for column in range(values.shape[1]):
        padded = np.r_[False, values[:, column], False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size:
            longest[column] = int(np.max(ends - starts))
    return longest.astype(float) / max(float(fps), 1e-6)


def _qualifying_score(duration: float, minimum_seconds: float, fps: float) -> float:
    minimum = max(float(minimum_seconds), 1.0 / max(float(fps), 1e-6))
    return float(duration) if float(duration) + 1e-12 >= minimum else 0.0


def _pair_key(left_id: int, right_id: int) -> str:
    left, right = sorted((int(left_id), int(right_id)))
    return f"{left},{right}"


def _prioritize_attack_over_approach(
    scores: dict[str, float], multiplier: float
) -> None:
    """Break an Attack/Approach tie without displacing contact labels."""
    if (
        scores.get("approach", 0.0) > 0.0
        and scores.get("attack", 0.0) > 0.0
        and scores["approach"] >= max(scores.values())
    ):
        scores["attack"] *= max(float(multiplier), 1.0)


def _neighbor_counts(
    pair_mask: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    track_count: int,
) -> np.ndarray:
    """Count qualifying neighbours per frame and identity."""

    counts = np.zeros((pair_mask.shape[0], int(track_count)), dtype=np.int16)
    for pair_index, (left, right) in enumerate(zip(pair_i, pair_j)):
        active = np.asarray(pair_mask[:, pair_index], dtype=np.int16)
        counts[:, int(left)] += active
        counts[:, int(right)] += active
    return counts


def _group_resting_at_onset(
    grouped: np.ndarray,
    speed_cm_s: np.ndarray,
    valid: np.ndarray,
    member_slots: Sequence[int],
    *,
    fps: float,
    minimum_seconds: float,
    speed_limit_cm_s: float,
    minimum_group_fraction: float = 0.2,
    minimum_low_motion_fraction: float = 0.8,
) -> bool:
    """Return whether the first sustained close-group window is resting."""
    onset = np.flatnonzero(grouped)
    required_frames = max(1, int(np.ceil(max(float(minimum_seconds), 0.0) * fps)))
    if onset.size == 0:
        return False
    start = int(onset[0])
    stop = start + required_frames
    if (
        stop > len(grouped)
        or float(np.mean(grouped[start:stop])) < minimum_group_fraction
    ):
        return False

    slots = tuple(int(slot) for slot in member_slots)
    observed = valid[start:stop][:, slots]
    speeds = speed_cm_s[start:stop][:, slots]
    usable = observed & np.isfinite(speeds)
    if not usable.any():
        return False
    onset_speeds = speeds[usable]
    return (
        float(np.mean(onset_speeds <= speed_limit_cm_s)) >= minimum_low_motion_fraction
        and float(np.mean(onset_speeds)) <= speed_limit_cm_s
    )


def _group_episode_scores(
    pair_mask: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    track_ids: Sequence[int],
    *,
    fps: float,
    minimum_seconds: float,
    nearest_drop_cm: np.ndarray | None = None,
    min_nearest_drop_cm: float = 0.0,
    max_gap_seconds: float = 0.0,
    score_multiplier: float = 1.0,
    min_member_support_fraction: float = 0.0,
    formation_evidence: np.ndarray | None = None,
    min_formation_member_fraction: float = 0.0,
    motion_speed_cm_s: np.ndarray | None = None,
    minimum_motion_speed_cm_s: float = 0.0,
    minimum_moving_member_fraction: float = 0.0,
) -> dict[str, float]:
    """Build temporal spatial groups without merging unrelated components.

    A component can change visual IDs as tracklets end; overlapping components
    in consecutive frames are one episode and retain the union of observed
    visual IDs.  This groups event participants, not biological identities.
    """

    active: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    minimum_frames = max(1, int(np.ceil(max(minimum_seconds, 0.0) * fps)))
    max_gap_frames = max(0, int(np.floor(max(float(max_gap_seconds), 0.0) * fps)))

    def finish(episode: Mapping[str, Any]) -> None:
        if episode["frames"] < minimum_frames:
            return
        if nearest_drop_cm is not None and episode["max_drop"] < min_nearest_drop_cm:
            return
        observed_frames = max(int(episode.get("observed_frames", 0)), 1)
        support_fraction = float(np.clip(min_member_support_fraction, 0.0, 1.0))
        members = tuple(
            sorted(
                int(track_ids[slot])
                for slot, count in episode.get("member_frames", {}).items()
                if count / observed_frames >= support_fraction
            )
        )
        if len(members) < 3:
            return
        if formation_evidence is not None:
            member_slots = {
                int(slot)
                for slot, count in episode.get("member_frames", {}).items()
                if count / observed_frames >= support_fraction
            }
            formation_members = set(episode.get("formation_members", set()))
            formation_fraction = (
                len(member_slots.intersection(formation_members)) / len(member_slots)
                if member_slots
                else 0.0
            )
            if formation_fraction + 1e-12 < float(min_formation_member_fraction):
                return
        key = ",".join(str(member) for member in members)
        scores[key] = max(
            scores.get(key, 0.0),
            episode["frames"] / fps * max(float(score_multiplier), 1e-6),
        )

    for frame_index, edge_mask in enumerate(pair_mask):
        adjacency: dict[int, set[int]] = {}
        for edge_index in np.flatnonzero(edge_mask):
            left = int(pair_i[edge_index])
            right = int(pair_j[edge_index])
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        components: list[set[int]] = []
        visited: set[int] = set()
        for start in sorted(adjacency):
            if start in visited:
                continue
            component = {start}
            pending = [start]
            visited.add(start)
            while pending:
                for neighbor in adjacency[pending.pop()]:
                    if neighbor not in visited:
                        component.add(neighbor)
                        visited.add(neighbor)
                        pending.append(neighbor)
            if len(component) < 3:
                continue
            components.append(component)
        next_active: list[dict[str, Any]] = []
        matched: set[int] = set()
        for component in components:
            if motion_speed_cm_s is not None:
                component_speeds = np.asarray(
                    motion_speed_cm_s[frame_index, sorted(component)], dtype=float
                )
                component_speeds = component_speeds[np.isfinite(component_speeds)]
                moving_speeds = component_speeds[component_speeds > 1e-9]
                moving_fraction = len(moving_speeds) / max(len(component_speeds), 1)
                if (
                    moving_speeds.size == 0
                    or moving_fraction + 1e-12 < float(minimum_moving_member_fraction)
                    or float(np.mean(moving_speeds)) + 1e-12
                    < float(minimum_motion_speed_cm_s)
                ):
                    continue
            matches = [
                (len(component & episode["last"]), index)
                for index, episode in enumerate(active)
                if index not in matched
            ]
            overlap, index = max(matches, default=(0, -1))
            if index >= 0 and overlap >= max(2, min(len(component), len(active[index]["last"])) // 2):
                episode = active[index]
                matched.add(index)
                episode["members"].update(component)
                episode["last"] = component
                episode["frames"] += 1
                episode["gap_frames"] = 0
            else:
                episode = {
                    "members": set(component), "last": component,
                    "frames": 1, "max_drop": 0.0, "gap_frames": 0,
                    "observed_frames": 0, "member_frames": {},
                    "formation_members": set(),
                }
            episode["observed_frames"] += 1
            for slot in component:
                episode["member_frames"][slot] = episode["member_frames"].get(slot, 0) + 1
                if (
                    formation_evidence is not None
                    and bool(formation_evidence[frame_index, slot])
                ):
                    episode["formation_members"].add(slot)
            if nearest_drop_cm is not None:
                episode["max_drop"] = max(
                    episode["max_drop"],
                    float(np.max(nearest_drop_cm[frame_index, sorted(component)])),
                )
            next_active.append(episode)
        for index, episode in enumerate(active):
            if index not in matched:
                if int(episode.get("gap_frames", 0)) < max_gap_frames:
                    episode["gap_frames"] = int(episode.get("gap_frames", 0)) + 1
                    episode["frames"] += 1
                    next_active.append(episode)
                else:
                    finish(episode)
        active = next_active
    for episode in active:
        finish(episode)
    return scores


def _rolling_drop(values: np.ndarray, frames: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    frames = max(int(frames), 1)
    if len(values) <= frames:
        return result
    with np.errstate(invalid="ignore"):
        delta = values[:-frames] - values[frames:]
    finite = np.isfinite(delta)
    result[frames:][finite] = delta[finite]
    return result


def predict_features(
    features: ExportedFeatures,
    parameters: HeuristicParameters,
) -> dict[str, Any]:
    """Infer behavior evidence for every visible ID without consulting annotations.

    Identity scores are used for single-mouse and group-membership behaviors.
    Pair scores are keyed by the exact unordered visual-ID pair.  Keeping those
    scores separate prevents a behavior performed by unrelated mice elsewhere
    in the clip from being credited to the annotated target IDs later.
    """

    fps = features.fps
    valid = features.valid
    speed = features.speed_cm_s
    pair_distance = features.pair_distance_cm
    pair_valid = np.isfinite(pair_distance)
    track_ids = [int(value) for value in features.track_ids.tolist()]
    identity_scores: dict[str, dict[str, float]] = {
        str(track_id): {
            behavior: 0.0 for behavior in (*INDIVIDUAL_BEHAVIORS, *GROUP_BEHAVIORS)
        }
        for track_id in track_ids
    }
    pair_scores: dict[str, dict[str, float]] = {}

    stationary = valid & (speed <= parameters.stationary_max_speed_cm_s)
    walking = valid & (speed >= parameters.walking_min_speed_cm_s) & (
        speed < min(parameters.walking_max_speed_cm_s, parameters.running_min_speed_cm_s)
    )
    walking_gap_frames = max(
        0, int(np.floor(max(float(parameters.walking_max_stop_gap_s), 0.0) * fps))
    )
    if walking_gap_frames:
        walking = np.column_stack(
            [
                _bridge_short_false_gaps(walking[:, slot], walking_gap_frames)
                for slot in range(walking.shape[1])
            ]
        )
    stationary_durations = _durations_by_column(stationary, fps)
    walking_durations = _durations_by_column(walking, fps)
    running = valid & (speed >= parameters.running_min_speed_cm_s)
    running_durations = _durations_by_column(running, fps)
    for slot, track_id in enumerate(track_ids):
        identity_scores[str(track_id)]["stationary"] = _qualifying_score(
            stationary_durations[slot], parameters.stationary_min_duration_s, fps
        )
        identity_scores[str(track_id)]["walking"] = _qualifying_score(
            walking_durations[slot], parameters.walking_min_duration_s, fps
        )
        active_velocity = features.velocity_cm_s[running[:, slot], slot]
        active_speed = np.linalg.norm(active_velocity, axis=1)
        directionality = (
            float(np.linalg.norm(active_velocity.sum(axis=0)) / max(active_speed.sum(), 1e-6))
            if active_speed.size
            else 0.0
        )
        identity_scores[str(track_id)]["running"] = (
            _qualifying_score(running_durations[slot], parameters.running_min_duration_s, fps)
            if directionality >= parameters.running_min_directionality
            else 0.0
        )

    pair_left_speed = speed[:, features.pair_i]
    pair_right_speed = speed[:, features.pair_j]
    pair_members_observed = valid[:, features.pair_i] & valid[:, features.pair_j]
    huddle_distance = _huddle_pair_distance_cm(features, parameters)
    together = (
        pair_valid
        & pair_members_observed
        & (pair_distance < parameters.together_max_distance_cm)
        & (pair_left_speed <= parameters.together_max_individual_speed_cm_s)
        & (pair_right_speed <= parameters.together_max_individual_speed_cm_s)
        & (
            pair_left_speed + pair_right_speed
            <= parameters.together_max_combined_speed_cm_s
        )
    )
    together_durations = _durations_by_column(together, fps)

    window = max(int(round(parameters.approach_window_s * fps)), 1)
    distance_drop = np.zeros_like(pair_distance, dtype=float)
    for pair in range(pair_distance.shape[1]):
        distance_drop[:, pair] = _rolling_drop(pair_distance[:, pair], window)
    approach = pair_valid & (pair_distance < parameters.approach_terminal_distance_cm)
    approach &= distance_drop >= parameters.approach_min_distance_drop_cm
    approach &= features.pair_closing_speed_cm_s >= parameters.approach_min_closing_speed_cm_s
    approach_durations = _durations_by_column(approach, fps)

    contact = pair_valid & (
        (features.pair_nose_head_cm < parameters.contact_distance_cm)
        | (features.pair_nose_tail_cm < parameters.contact_distance_cm)
    )
    attack = contact & (
        np.maximum(speed[:, features.pair_i], speed[:, features.pair_j])
        >= parameters.attack_min_speed_cm_s
    )
    attack_durations = _durations_by_column(attack, fps)
    head_contact = pair_valid & (
        features.pair_nose_head_cm < parameters.contact_distance_cm
    )
    tail_contact = pair_valid & (
        features.pair_nose_tail_cm < parameters.contact_distance_cm
    )
    nose_head_cumulative = np.sum(head_contact, axis=0, dtype=np.int64) / max(fps, 1e-6)
    nose_tail_cumulative = np.sum(tail_contact, axis=0, dtype=np.int64) / max(fps, 1e-6)
    no_contact_frames = max(1, int(np.ceil(parameters.pair_no_contact_seconds * fps)))
    chased = (
        pair_valid
        & (pair_distance >= parameters.contact_distance_cm)
        & (pair_distance <= parameters.following_max_distance_cm)
        & (features.pair_direction_similarity >= parameters.chase_min_direction_similarity)
        & (speed[:, features.pair_i] >= parameters.running_min_speed_cm_s)
        & (speed[:, features.pair_j] >= parameters.running_min_speed_cm_s)
    )
    walking_pair = walking[:, features.pair_i] & walking[:, features.pair_j]
    aligned_follow = (
        pair_valid
        & (pair_distance >= parameters.following_min_distance_cm)
        & (pair_distance <= parameters.following_max_distance_cm)
        & (features.pair_direction_similarity >= parameters.following_min_direction_similarity)
        & walking_pair
    )
    contact_or_dropout = contact | ~pair_valid
    no_contact = np.zeros_like(pair_valid)
    for pair in range(pair_distance.shape[1]):
        clean = ~contact_or_dropout[:, pair]
        padded = np.r_[False, clean, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if end - start >= no_contact_frames:
                no_contact[start:end, pair] = True
    # Brief nose touches are intentionally not elevated to separate contact
    # events; Following and Chasing require at least one clean 0.5 s interval.
    chasing = chased & no_contact
    following = aligned_follow & no_contact
    chasing_durations = _durations_by_column(chasing, fps)
    following_durations = _durations_by_column(following, fps)
    # Avoiding is a response after a prior close approach followed by sustained
    # opening distance and a fast change in the pair's relative motion.
    opening = pair_valid & (features.pair_closing_speed_cm_s <= -parameters.approach_min_closing_speed_cm_s)
    opening &= -distance_drop >= parameters.avoiding_min_distance_increase_cm
    avoid_scores = np.zeros(pair_distance.shape[1], dtype=float)
    for pair_index, (left_slot, right_slot) in enumerate(
        zip(features.pair_i, features.pair_j)
    ):
        key = _pair_key(track_ids[int(left_slot)], track_ids[int(right_slot)])
        attack_score = _qualifying_score(
            attack_durations[pair_index], parameters.pair_min_duration_s, fps
        )
        if attack_score <= 0.0:
            contact_frames = np.flatnonzero(attack[:, pair_index])
            endpoint = pair_valid[:, pair_index] & (
                pair_distance[:, pair_index] <= parameters.attack_endpoint_distance_cm
            )
            endpoint_frames = np.flatnonzero(endpoint)
            if contact_frames.size and endpoint_frames.size:
                last_contact = int(contact_frames[-1])
                # Recovery is valid only if this exact visual-ID pair was
                # genuinely unobserved and the first re-observation is nearby.
                # A continuously visible non-contact interval cannot be used
                # to manufacture an attack from an old contact sample.
                later_endpoint = endpoint_frames[endpoint_frames > last_contact]
                if later_endpoint.size:
                    first_reacquired = int(later_endpoint[0])
                    missing_frames = first_reacquired - last_contact - 1
                    gap_seconds = missing_frames / max(fps, 1e-6)
                    missing_pair = not np.all(pair_valid[last_contact + 1:first_reacquired, pair_index])
                    if (
                        missing_frames > 0
                        and missing_pair
                        and gap_seconds <= parameters.attack_reacquisition_max_gap_s
                        and pair_distance[first_reacquired, pair_index]
                        <= parameters.attack_endpoint_distance_cm
                    ):
                        attack_score = max(
                            attack_score,
                            (first_reacquired - int(contact_frames[0]) + 1) / max(fps, 1e-6),
                        )
        values = {
            "approach": _qualifying_score(
                approach_durations[pair_index], parameters.pair_min_duration_s, fps
            ),
            "attack": attack_score,
            "nose_head_contact": _qualifying_score(
                nose_head_cumulative[pair_index], parameters.contact_min_cumulative_seconds, fps
            ),
            "nose_tail_contact": _qualifying_score(
                nose_tail_cumulative[pair_index], parameters.contact_min_cumulative_seconds, fps
            ),
            "together": _qualifying_score(
                together_durations[pair_index], parameters.together_min_duration_s, fps
            ),
            "chase": _qualifying_score(
                chasing_durations[pair_index],
                parameters.chase_min_duration_s,
                fps,
            ),
            "following": _qualifying_score(
                following_durations[pair_index], parameters.following_min_duration_s, fps
            ),
        }
        near_before = np.flatnonzero(
            pair_valid[: max(0, len(pair_distance) - 1), pair_index]
            & (pair_distance[: max(0, len(pair_distance) - 1), pair_index] <= parameters.approach_terminal_distance_cm)
        )
        if near_before.size:
            first_near = int(near_before[0])
            fast_escape = (
                np.maximum(speed[:, left_slot], speed[:, right_slot])
                >= parameters.walking_min_speed_cm_s
            )
            subsequent_opening = opening[first_near:, pair_index] & fast_escape[first_near:]
            avoid_duration = _longest_run(subsequent_opening) / max(fps, 1e-6)
            avoid_scores[pair_index] = _qualifying_score(
                avoid_duration, parameters.avoiding_min_duration_s, fps
            )
        values["avoidance"] = float(avoid_scores[pair_index])
        _prioritize_attack_over_approach(
            values, parameters.attack_vs_approach_multiplier
        )
        if any(score > 0.0 for score in values.values()):
            pair_scores[key] = values

    valid_count = valid.sum(axis=1)
    huddle_speed_limit = min(
        float(parameters.huddle_max_mean_speed_cm_s),
        float(parameters.stationary_max_speed_cm_s),
    )
    huddle_pairs = (
        pair_valid
        & pair_members_observed
        & (pair_distance < huddle_distance)
        & (pair_left_speed <= huddle_speed_limit)
        & (pair_right_speed <= huddle_speed_limit)
    )
    huddle_groups = _group_episode_scores(
        huddle_pairs, features.pair_i, features.pair_j, track_ids,
        fps=fps, minimum_seconds=parameters.huddle_min_duration_s,
        max_gap_seconds=parameters.huddle_max_gap_s,
        min_member_support_fraction=parameters.huddle_min_member_support_fraction,
    )
    # Huddling is a resting state, not a moving formation followed by a pause.
    # Inspect spatial connectivity independently of the speed-gated huddle mask
    # so motion at the start of the close-group episode cannot be averaged away.
    track_slot = {track_id: slot for slot, track_id in enumerate(track_ids)}
    for key in tuple(huddle_groups):
        member_slots = tuple(track_slot[int(value)] for value in key.split(","))
        internal_edges = np.isin(features.pair_i, member_slots) & np.isin(
            features.pair_j, member_slots
        )
        degrees = np.zeros((len(valid), len(member_slots)), dtype=np.int16)
        local_slot = {slot: index for index, slot in enumerate(member_slots)}
        for edge in np.flatnonzero(internal_edges):
            close_edge = (
                pair_valid[:, edge]
                & pair_members_observed[:, edge]
                & (pair_distance[:, edge] < huddle_distance)
            )
            degrees[:, local_slot[int(features.pair_i[edge])]] += close_edge
            degrees[:, local_slot[int(features.pair_j[edge])]] += close_edge
        spatially_grouped = np.max(degrees, axis=1) >= 2
        if not _group_resting_at_onset(
            spatially_grouped,
            speed,
            valid,
            member_slots,
            fps=fps,
            minimum_seconds=parameters.huddle_min_duration_s,
            speed_limit_cm_s=huddle_speed_limit,
        ):
            huddle_groups.pop(key)

    isolated = (
        valid
        & (valid_count >= 2)[:, None]
        & (features.nearest_distance_cm >= parameters.isolation_distance_cm)
    )
    isolation_durations = _durations_by_column(isolated, fps)

    cluster_window = max(int(round(parameters.clustering_window_s * fps)), 1)
    cluster_pairs = pair_valid & (pair_distance < parameters.clustering_max_distance_cm)
    cluster_drop = np.zeros_like(features.nearest_distance_cm, dtype=float)
    for slot in range(cluster_drop.shape[1]):
        cluster_drop[:, slot] = _rolling_drop(
            np.asarray(features.nearest_distance_cm[:, slot], dtype=float), cluster_window
        )
    initial_neighbor_distance = np.full_like(features.nearest_distance_cm, np.inf)
    if cluster_window < len(features.nearest_distance_cm):
        initial_neighbor_distance[cluster_window:] = features.nearest_distance_cm[:-cluster_window]
    formation_evidence = (
        np.isfinite(initial_neighbor_distance)
        & (initial_neighbor_distance <= parameters.clustering_initial_max_distance_cm)
        & (cluster_drop >= parameters.clustering_min_nearest_neighbor_drop_cm)
        & (speed >= parameters.clustering_min_mean_speed_cm_s)
    )
    cluster_groups = _group_episode_scores(
        cluster_pairs, features.pair_i, features.pair_j, track_ids,
        fps=fps, minimum_seconds=parameters.clustering_min_duration_s,
        nearest_drop_cm=cluster_drop,
        min_nearest_drop_cm=parameters.clustering_min_nearest_neighbor_drop_cm,
        max_gap_seconds=parameters.clustering_max_gap_s,
        score_multiplier=parameters.clustering_formation_score_bonus,
        formation_evidence=formation_evidence,
        min_formation_member_fraction=parameters.clustering_min_moving_member_fraction,
        motion_speed_cm_s=speed,
        minimum_motion_speed_cm_s=parameters.clustering_min_mean_speed_cm_s,
        minimum_moving_member_fraction=parameters.clustering_min_moving_member_fraction,
        min_member_support_fraction=parameters.clustering_min_member_support_fraction,
    )
    if parameters.clustering_motion_max_distance_cm > 0.0:
        # Formation may precede the exported clip. A sustained, spatially
        # connected group with net movement is still a clustering candidate;
        # resting Huddles are rejected by the displacement gate. Membership
        # comes only from observed visual IDs, never from inferred identity.
        moving_groups = _group_episode_scores(
            pair_valid & (pair_distance < parameters.clustering_motion_max_distance_cm),
            features.pair_i, features.pair_j, track_ids,
            fps=fps, minimum_seconds=parameters.clustering_min_duration_s,
            max_gap_seconds=parameters.clustering_max_gap_s,
            min_member_support_fraction=0.5,
        )
        for key, episode_score in moving_groups.items():
            member_ids = tuple(int(value) for value in key.split(","))
            member_slots = {track_slot[track_id] for track_id in member_ids}
            member_displacements = []
            for track_id in member_ids:
                slot = track_slot[track_id]
                points = features.centers_cm[valid[:, slot], slot]
                if len(points) < 2:
                    break
                member_displacements.append(float(np.linalg.norm(points[-1] - points[0])))
            internal_edges = np.isin(features.pair_i, list(member_slots)) & np.isin(
                features.pair_j, list(member_slots)
            )
            internal_distances = pair_distance[:, internal_edges]
            observed_distances = internal_distances[np.isfinite(internal_distances)]
            loose_group = bool(
                observed_distances.size
                and np.median(observed_distances)
                >= parameters.clustering_motion_loose_pair_distance_cm
            )
            if (
                len(member_displacements) >= 3
                and np.mean(member_displacements) >= parameters.clustering_motion_min_displacement_cm
                and (
                    sum(
                        displacement >= parameters.clustering_motion_min_displacement_cm
                        for displacement in member_displacements
                    ) >= 2
                    or loose_group
                )
            ):
                cluster_groups[key] = max(cluster_groups.get(key, 0.0), episode_score * 1.25)

    collective_groups = {
        **{key: {"huddle": score} for key, score in huddle_groups.items()},
        **{key: {"social_clustering": score} for key, score in cluster_groups.items()},
    }
    collective_member_ids = {
        int(track_id)
        for key in collective_groups
        for track_id in key.split(",")
    }
    for slot, track_id in enumerate(track_ids):
        values = identity_scores[str(track_id)]
        values["isolation"] = _qualifying_score(
            isolation_durations[slot], parameters.isolation_min_duration_s, fps
        )
        if track_id in collective_member_ids:
            values["isolation"] = 0.0
    group_scores: dict[str, dict[str, float]] = {}
    for slot, track_id in enumerate(track_ids):
        isolation_score = identity_scores[str(track_id)]["isolation"]
        if isolation_score > 0.0:
            group_scores[str(track_id)] = {"isolation": isolation_score}
    for behavior, episodes in (("huddle", huddle_groups), ("social_clustering", cluster_groups)):
        for key, score in episodes.items():
            group_scores.setdefault(key, {})[behavior] = score
            for track_id in key.split(","):
                values = identity_scores[track_id]
                values[behavior] = max(values[behavior], score)

    scores: dict[str, float] = {behavior: 0.0 for behavior in CANONICAL_BEHAVIORS}
    for values in identity_scores.values():
        for behavior, score in values.items():
            scores[behavior] = max(scores[behavior], float(score))
    for values in pair_scores.values():
        for behavior, score in values.items():
            scores[behavior] = max(scores[behavior], float(score))
    detected = sorted(behavior for behavior, score in scores.items() if score > 0.0)

    return {
        "detected": detected,
        "scores": {key: float(value) for key, value in scores.items()},
        "track_ids": track_ids,
        "identity_scores": identity_scores,
        "pair_scores": pair_scores,
        "group_scores": group_scores,
        "track_selection_truncated": features.track_selection_truncated,
        "selected_track_count": features.selected_track_count,
    }


def classify_target_ids(
    record: SampleRecord,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Score the annotated ID set after label-free inference has completed."""

    target = record.canonical_behavior
    target_ids = tuple(sorted(int(value) for value in record.mouse_ids))
    identity_scores = {
        str(key): {str(name): float(score) for name, score in dict(values).items()}
        for key, values in dict(prediction.get("identity_scores", {})).items()
    }
    pair_scores = {
        str(key): {str(name): float(score) for name, score in dict(values).items()}
        for key, values in dict(prediction.get("pair_scores", {})).items()
    }
    group_scores = {
        str(key): {str(name): float(score) for name, score in dict(values).items()}
        for key, values in dict(prediction.get("group_scores", {})).items()
    }
    selected_ids = {
        int(value)
        for value in prediction.get(
            "track_ids", [int(key) for key in identity_scores if str(key).lstrip("-").isdigit()]
        )
    }
    candidate_target_ids: dict[str, tuple[int, ...]] = {}

    def record_candidate(behavior: str, score: float, ids: tuple[int, ...]) -> None:
        previous = float(candidate_scores.get(behavior, 0.0))
        previous_ids = candidate_target_ids.get(behavior, ())
        if score > previous or (
            score == previous
            and score > 0.0
            and (not previous_ids or len(ids) < len(previous_ids))
        ):
            candidate_scores[behavior] = float(score)
            candidate_target_ids[behavior] = ids

    if target in INDIVIDUAL_BEHAVIORS:
        if len(target_ids) != 1:
            raise ValueError(f"{record.behavior} requires exactly one target ID: {record.sample_id}")
        key = str(target_ids[0])
        available = key in identity_scores
        candidate_behaviors = INDIVIDUAL_BEHAVIORS
        candidate_scores = {
            behavior: float(identity_scores.get(key, {}).get(behavior, 0.0))
            for behavior in candidate_behaviors
        }
        candidate_target_ids.update({behavior: target_ids for behavior in candidate_behaviors})
    elif target in PAIR_BEHAVIORS:
        if len(target_ids) != 2:
            raise ValueError(f"{record.behavior} requires exactly two target IDs: {record.sample_id}")
        key = _pair_key(*target_ids)
        available = all(track_id in selected_ids for track_id in target_ids)
        if not selected_ids:
            available = key in pair_scores
        candidate_behaviors = PAIR_BEHAVIORS
        candidate_scores = {
            behavior: float(pair_scores.get(key, {}).get(behavior, 0.0))
            for behavior in candidate_behaviors
        }
        candidate_target_ids.update({behavior: target_ids for behavior in candidate_behaviors})
    elif target == "isolation":
        if len(target_ids) != 1:
            raise ValueError(f"Isolation requires one target ID: {record.sample_id}")
        key = str(target_ids[0])
        available = key in identity_scores
        candidate_behaviors = GROUP_BEHAVIORS
        candidate_scores = {behavior: 0.0 for behavior in candidate_behaviors}
        candidate_scores["isolation"] = float(
            identity_scores.get(key, {}).get("isolation", 0.0)
        )
        candidate_target_ids["isolation"] = target_ids
        has_collective_candidate = False
        for raw_ids, scores in group_scores.items():
            try:
                group_ids = tuple(sorted(int(value) for value in raw_ids.split(",") if value))
            except ValueError:
                continue
            if len(group_ids) < 3 or target_ids[0] not in group_ids:
                continue
            for behavior in GROUP_BEHAVIORS:
                if behavior == "isolation":
                    continue
                score = float(scores.get(behavior, 0.0))
                if score > 0.0:
                    has_collective_candidate = True
                    record_candidate(behavior, score, group_ids)
        if has_collective_candidate:
            candidate_scores["isolation"] = 0.0
    elif target in GROUP_BEHAVIORS:
        if len(target_ids) < 3:
            raise ValueError(f"{record.behavior} requires at least three target IDs: {record.sample_id}")
        available = all(str(track_id) in identity_scores for track_id in target_ids)
        candidate_behaviors = GROUP_BEHAVIORS
        candidate_scores = {behavior: 0.0 for behavior in candidate_behaviors}
        if "group_scores" in prediction:
            target_id_set = set(target_ids)
            for raw_ids, scores in group_scores.items():
                try:
                    group_ids = tuple(sorted(int(value) for value in raw_ids.split(",") if value))
                except ValueError:
                    continue
                if len(group_ids) < 3 or not target_id_set.issubset(group_ids):
                    continue
                for behavior in candidate_behaviors:
                    record_candidate(behavior, float(scores.get(behavior, 0.0)), group_ids)
        else:
            # Read older cached predictions without event-level group candidates.
            candidate_scores = {
                behavior: min(
                    (
                        float(identity_scores.get(str(track_id), {}).get(behavior, 0.0))
                        for track_id in target_ids
                    ),
                    default=0.0,
                )
                for behavior in candidate_behaviors
            }
            candidate_target_ids.update(
                {behavior: target_ids for behavior in candidate_behaviors}
            )
    else:
        raise ValueError(f"unsupported canonical behavior {target!r}: {record.sample_id}")

    ranked = sorted(candidate_behaviors, key=lambda behavior: (-candidate_scores[behavior], behavior))
    predicted = ranked[0] if ranked and candidate_scores[ranked[0]] > 0.0 else None
    predicted_ids = candidate_target_ids.get(predicted, ()) if predicted is not None else ()
    return {
        "target_layer": (
            "individual" if target in INDIVIDUAL_BEHAVIORS
            else "social" if target in PAIR_BEHAVIORS
            else "group"
        ),
        "target_ids": list(target_ids),
        "target_ids_available": bool(available),
        "candidate_scores": candidate_scores,
        "predicted_behavior": predicted,
        "predicted_target_ids": list(predicted_ids),
        "exact_ids_correct": tuple(sorted(predicted_ids)) == target_ids,
        "target_hit": candidate_scores.get(target, 0.0) > 0.0,
        "behavior_correct": predicted == target,
        # A matching group label with additional visual IDs is not a joint
        # behavior-and-ID hit, even when it contains every annotated member.
        "correct": predicted == target and tuple(sorted(predicted_ids)) == target_ids,
    }


def evaluate_predictions(
    rows: Sequence[tuple[SampleRecord, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Calculate strict behavior accuracy for the IDs named by each annotation."""

    classified = [(record, classify_target_ids(record, prediction)) for record, prediction in rows]
    return evaluate_target_classifications(classified)


def evaluate_target_classifications(
    rows: Sequence[tuple[SampleRecord, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate already-classified target-ID results without retaining full pair tensors."""

    per_behavior: dict[str, dict[str, int]] = {
        behavior: {
            "support": 0,
            "correct": 0,
            "strict_top1_correct": 0,
            "exact_id_set_correct": 0,
            "predicted": 0,
            "false_positive": 0,
        }
        for behavior in CANONICAL_BEHAVIORS
    }
    correct = 0
    strict_correct = 0
    covered = 0
    available = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for record, result in rows:
        target = record.canonical_behavior
        per_behavior.setdefault(
            target,
            {
                "support": 0,
                "correct": 0,
                "strict_top1_correct": 0,
                "exact_id_set_correct": 0,
                "predicted": 0,
                "false_positive": 0,
            },
        )["support"] += 1
        detected = {
            str(behavior)
            for behavior, score in dict(result.get("candidate_scores", {})).items()
            if float(score) > 0.0
        }
        for behavior in detected:
            values = per_behavior.setdefault(
                behavior,
                {
                    "support": 0,
                    "correct": 0,
                    "strict_top1_correct": 0,
                    "exact_id_set_correct": 0,
                    "predicted": 0,
                    "false_positive": 0,
                },
            )
            values["predicted"] += 1
            values["false_positive"] += int(behavior != target)
        if result["target_hit"]:
            correct += 1
            per_behavior[target]["correct"] += 1
        predicted = result["predicted_behavior"]
        covered += int(predicted is not None)
        available += int(result["target_ids_available"])
        confusion[(target, str(predicted) if predicted is not None else "<none>")] += 1
        if result["correct"]:
            strict_correct += 1
            per_behavior[target]["strict_top1_correct"] += 1
        if bool(result.get("exact_ids_correct", False)):
            per_behavior[target]["exact_id_set_correct"] += 1
    total = len(rows)
    macro_accuracy = float(
        np.mean(
            [
                values["correct"] / values["support"]
                for values in per_behavior.values()
                if values["support"]
            ]
        )
        if any(values["support"] for values in per_behavior.values())
        else 0.0
    )
    macro_strict_accuracy = float(
        np.mean(
            [
                values["strict_top1_correct"] / values["support"]
                for values in per_behavior.values()
                if values["support"]
            ]
        )
        if any(values["support"] for values in per_behavior.values())
        else 0.0
    )
    per_behavior_metrics: dict[str, dict[str, Any]] = {}
    for key, value in sorted(per_behavior.items()):
        if not value["support"]:
            continue
        precision = value["correct"] / value["predicted"] if value["predicted"] else 0.0
        recall = value["correct"] / value["support"]
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_behavior_metrics[key] = {
            **value,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "strict_top1_accuracy": value["strict_top1_correct"] / value["support"],
            "exact_id_set_accuracy": value["exact_id_set_correct"] / value["support"],
        }
    macro_f1 = float(
        np.mean([values["f1"] for values in per_behavior_metrics.values()])
        if per_behavior_metrics
        else 0.0
    )
    return {
        "n_samples": total,
        "target_id_accuracy": correct / total if total else 0.0,
        "macro_target_id_accuracy": macro_accuracy,
        "strict_target_id_accuracy": strict_correct / total if total else 0.0,
        "exact_target_id_set_accuracy": (
            sum(values["exact_id_set_correct"] for values in per_behavior.values()) / total
            if total
            else 0.0
        ),
        "macro_strict_target_id_accuracy": macro_strict_accuracy,
        "macro_f1": macro_f1,
        "prediction_coverage": covered / total if total else 0.0,
        "target_id_availability_rate": available / total if total else 0.0,
        "per_behavior": per_behavior_metrics,
        "confusion": {
            f"{truth}->{prediction}": count
            for (truth, prediction), count in sorted(confusion.items())
        },
    }


def calibration_objective(metrics: Mapping[str, Any]) -> float:
    """Optimize target-ID recognition while penalizing always-on rules via macro F1."""

    return float(
        0.70 * float(metrics.get("macro_f1", 0.0))
        + 0.30 * float(metrics.get("macro_target_id_accuracy", 0.0))
    )


def group_rule_calibration_rank(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """Rank group-rule candidates by macro F1, then strict target Top-1."""

    rows = dict(metrics.get("per_behavior", {}))
    focus = [
        rows[name]
        for name in ("huddle", "isolation", "social_clustering")
        if int(rows.get(name, {}).get("support", 0)) > 0
    ]
    macro_f1 = (
        sum(float(row.get("f1", 0.0)) for row in focus) / len(focus)
        if focus
        else 0.0
    )
    macro_strict = (
        sum(float(row.get("strict_top1_accuracy", 0.0)) for row in focus) / len(focus)
        if focus
        else 0.0
    )
    return macro_f1, macro_strict


def strict_top1_regressions(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    excluded_behaviors: set[str] | None = None,
) -> list[str]:
    """List behavior classes whose strict Top-1 count fell from a baseline."""

    excluded = excluded_behaviors or set()
    current_rows = dict(metrics.get("per_behavior", {}))
    baseline_rows = dict(baseline.get("per_behavior", {}))
    return sorted(
        behavior
        for behavior, baseline_values in baseline_rows.items()
        if behavior not in excluded
        and int(baseline_values.get("support", 0)) > 0
        and int(current_rows.get(behavior, {}).get("strict_top1_correct", 0))
        < int(baseline_values.get("strict_top1_correct", 0))
    )


def group_rule_parameter_tie_break(
    parameter_name: str, value: float, candidate_index: int
) -> tuple[float, float]:
    """Prefer conservative, empirically supported settings when scores tie."""

    if parameter_name == "clustering_initial_max_distance_cm":
        return -float(value), -float(candidate_index)
    if parameter_name == "huddle_min_member_support_fraction":
        return -abs(float(value) - 0.25), -float(candidate_index)
    return -float(candidate_index), -float(value)


def minimum_behavior_accuracy(metrics: Mapping[str, Any]) -> float:
    """Return the lowest target-ID accuracy among behaviors with labeled support."""

    values = [
        float(details.get("accuracy", 0.0))
        for details in dict(metrics.get("per_behavior", {})).values()
        if int(details.get("support", 0)) > 0
    ]
    return min(values, default=0.0)


def training_target_reached(
    metrics: Mapping[str, Any], target_accuracy: float = 0.95
) -> bool:
    """Require both aggregate and every-class target-ID accuracy to reach target."""

    target = float(target_accuracy)
    return (
        float(metrics.get("target_id_accuracy", 0.0)) + 1e-12 >= target
        and minimum_behavior_accuracy(metrics) + 1e-12 >= target
    )


def calibration_rank(
    metrics: Mapping[str, Any], target_accuracy: float = 0.95
) -> tuple[float, ...]:
    """Rank training candidates under an accuracy constraint.

    Before the target is reached, the worst behavior is improved first so a
    frequent class cannot hide a failed minority class.  Once every behavior
    reaches the requested accuracy, macro F1 and strict top-1 accuracy become
    the tie breakers.  This prevents the search from preferring an always-on
    detector when a more selective feasible candidate exists.
    """

    feasible = training_target_reached(metrics, target_accuracy)
    worst = minimum_behavior_accuracy(metrics)
    macro_accuracy = float(metrics.get("macro_target_id_accuracy", 0.0))
    overall_accuracy = float(metrics.get("target_id_accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    strict_accuracy = float(metrics.get("strict_target_id_accuracy", 0.0))
    if feasible:
        return (1.0, macro_f1, strict_accuracy, macro_accuracy, overall_accuracy)
    return (0.0, worst, macro_accuracy, overall_accuracy, macro_f1)


def config_seed_parameters(config: Mapping[str, Any]) -> HeuristicParameters:
    """Map the repository YAML defaults to the exported-data parameter set."""

    extended = dict(config.get("extended_behavior", {}))
    individual = dict(extended.get("individual", {}))
    social = dict(extended.get("social", {}))
    group = dict(extended.get("group", {}))
    clustering = dict(group.get("social_clustering", {}))
    stationary_max_speed_cm_s = float(individual.get("stationary_max_speed_cm_s", 4.0))
    return HeuristicParameters(
        stationary_max_speed_cm_s=stationary_max_speed_cm_s,
        walking_min_speed_cm_s=float(
            individual.get("walking_min_speed_cm_s", stationary_max_speed_cm_s)
        ),
        walking_max_speed_cm_s=float(individual.get("walking_max_speed_cm_s", 85.0)),
        walking_max_stop_gap_s=float(individual.get("walking_max_stop_gap_seconds", 0.5)),
        running_min_speed_cm_s=float(individual.get("walking_max_speed_cm_s", 85.0)),
        together_max_distance_cm=float(social.get("together_max_distance_cm", 8.0)),
        together_max_individual_speed_cm_s=float(
            social.get("together_max_individual_speed_cm_s", 16.0)
        ),
        together_max_combined_speed_cm_s=float(social.get("together_max_combined_speed_cm_s", 28.0)),
        pair_max_distance_cm=float(social.get("pair_max_distance_cm", 5.0)),
        approach_min_distance_drop_cm=float(social.get("approach_min_distance_drop_cm", 1.5)),
        approach_min_closing_speed_cm_s=float(social.get("approach_min_closing_speed_cm_s", 2.0)),
        approach_terminal_distance_cm=float(social.get("approach_terminal_distance_cm", 17.0)),
        contact_distance_cm=float(config.get("contact_detection", {}).get("nose_head_distance_cm", 3.0)),
        contact_min_cumulative_seconds=max(
            float(config.get("contact_detection", {}).get(
                "nose_head_min_cumulative_duration_seconds", 0.5
            )),
            0.5,
        ),
        running_min_duration_s=max(float(individual.get("running_min_duration_seconds", 0.5)), 0.5),
        attack_min_speed_cm_s=float(dict(social.get("attack_fallback", {})).get("min_raw_actor_speed_cm_s", 8.0)),
        attack_vs_approach_multiplier=float(
            social.get("attack_vs_approach_multiplier", 1.5)
        ),
        attack_endpoint_distance_cm=float(
            dict(social.get("attack_fallback", {})).get("endpoint_distance_cm", 12.0)
        ),
        attack_reacquisition_max_gap_s=float(
            dict(social.get("attack_fallback", {})).get("reacquisition_max_gap_seconds", 5.0)
        ),
        huddle_distance_cm=float(group.get("huddle_distance_cm", 5.0)),
        huddle_width_multiplier=float(group.get("huddle_width_multiplier", 1.0)),
        huddle_max_mean_speed_cm_s=float(group.get("huddle_max_mean_speed_cm_s", 10.0)),
        huddle_min_duration_s=float(
            max(float(group.get("huddle_min_duration_seconds", group.get("confirm_seconds", 1.0))), 1.0)
        ),
        huddle_max_gap_s=float(
            group.get("huddle_fill_gap_seconds", group.get("fill_gap_seconds", 5.0))
        ),
        huddle_min_member_support_fraction=float(
            group.get("huddle_min_member_support_fraction", 0.25)
        ),
        isolation_distance_cm=float(group.get("isolation_distance_cm", 8.0)),
        isolation_min_duration_s=max(float(group.get("isolation_min_duration_seconds", 10.0)), 10.0),
        clustering_max_distance_cm=float(clustering.get("max_neighbor_distance_cm", 30.0)),
        clustering_initial_max_distance_cm=float(
            clustering.get("initial_max_neighbor_distance_cm", 24.0)
        ),
        clustering_min_nearest_neighbor_drop_cm=float(clustering.get("min_nearest_neighbor_drop_cm", 2.0)),
        stationary_min_duration_s=max(float(individual.get("stationary_min_duration_seconds", 1.0)), 1.0),
        walking_min_duration_s=max(float(individual.get("walking_min_duration_seconds", 1.0)), 1.0),
        pair_min_duration_s=float(social.get("approach_min_duration_seconds", 0.1)),
        together_min_duration_s=float(social.get("together_min_duration_seconds", 1.0)),
        group_min_duration_s=float(group.get("confirm_seconds", 0.3)),
        approach_window_s=0.3,
        clustering_window_s=float(clustering.get("formation_window_seconds", 2.0)),
        clustering_min_duration_s=max(float(clustering.get("min_duration_seconds", 5.0)), 5.0),
        clustering_max_gap_s=float(
            clustering.get("fill_gap_seconds", group.get("fill_gap_seconds", 0.25))
        ),
        clustering_formation_score_bonus=float(
            clustering.get("formation_score_bonus", 1.0)
        ),
        clustering_min_mean_speed_cm_s=float(
            clustering.get("min_mean_speed_cm_s", 3.0)
        ),
        clustering_min_moving_member_fraction=float(
            clustering.get("min_moving_member_fraction", 0.5)
        ),
        chase_min_duration_s=max(float(social.get("chase_min_duration_seconds", 2.0)), 2.0),
        following_min_duration_s=max(float(social.get("following_min_duration_seconds", 3.0)), 3.0),
        group_locomotion_max_distance_cm=float(
            dict(group.get("group_locomotion", {})).get("max_neighbor_distance_cm", 30.0)
        ),
        group_locomotion_min_duration_s=max(
            float(dict(group.get("group_locomotion", {})).get("min_duration_seconds", 3.0)), 3.0
        ),
        dispersal_min_duration_s=max(
            float(dict(group.get("dispersal", {})).get("min_duration_seconds", 10.0)), 10.0
        ),
    )


__all__ = [
    "CANONICAL_BEHAVIORS",
    "ExportedFeatures",
    "GROUP_BEHAVIORS",
    "HeuristicParameters",
    "INDIVIDUAL_BEHAVIORS",
    "LABEL_TO_CANONICAL",
    "PAIR_BEHAVIORS",
    "RETAINED_BEHAVIORS",
    "SampleRecord",
    "calibration_rank",
    "calibration_objective",
    "classify_target_ids",
    "config_seed_parameters",
    "discover_samples",
    "evaluate_predictions",
    "evaluate_target_classifications",
    "group_rule_calibration_rank",
    "group_rule_parameter_tie_break",
    "filter_behaviors",
    "load_exported_features",
    "minimum_behavior_accuracy",
    "predict_features",
    "split_by_recording_session",
    "strict_top1_regressions",
    "training_target_reached",
    "write_split_manifest",
]
