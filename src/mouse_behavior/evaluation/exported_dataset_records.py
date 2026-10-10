"""Sample discovery, recording-session splits, and dataset manifests."""

from __future__ import annotations

from mouse_behavior.evaluation.exported_dataset_models import (
    ExportedFeatures,
    HeuristicParameters,
    LABEL_TO_CANONICAL,
    RETAINED_BEHAVIORS,
    SampleRecord,
    _PART_SUFFIX,
)

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np

def collective_track_quality_metrics(
    features: ExportedFeatures,
) -> list[dict[str, Any]]:
    """Summarize clip-local track quality without resolving or remapping IDs."""

    valid = np.asarray(features.valid, dtype=bool)
    frame_count, track_count = valid.shape
    fps = max(float(features.fps), 1e-9)
    observed_counts = np.count_nonzero(valid, axis=0)
    presence = (
        np.asarray(features.track_presence_fraction, dtype=float)
        if features.track_presence_fraction is not None
        else observed_counts.astype(float) / max(frame_count, 1)
    )

    def median_per_track(values: np.ndarray | None) -> list[float | None]:
        if values is None:
            return [None] * track_count
        array = np.asarray(values, dtype=float)
        if array.shape != valid.shape:
            return [None] * track_count
        result: list[float | None] = []
        for slot in range(track_count):
            usable = array[:, slot][np.isfinite(array[:, slot])]
            result.append(float(np.median(usable)) if usable.size else None)
        return result

    median_confidence = median_per_track(features.detection_confidence)
    median_keypoint_coverage = median_per_track(features.keypoint_coverage)
    duplicate_score = np.zeros(track_count, dtype=float)
    if features.pair_box_iou is not None and features.pair_distance_cm is not None:
        overlap = np.asarray(features.pair_box_iou, dtype=float)
        distance = np.asarray(features.pair_distance_cm, dtype=float)
        widths = (
            np.asarray(features.body_width_ref_cm, dtype=float)
            if features.body_width_ref_cm is not None
            else np.full(track_count, max(float(features.mouse_width_cm), 1.0))
        )
        for pair_index, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
            usable = np.isfinite(overlap[:, pair_index]) & np.isfinite(distance[:, pair_index])
            if not np.any(usable):
                continue
            exact_overlap = (
                (overlap[usable, pair_index] >= 0.90)
                & (distance[usable, pair_index] <= max(min(widths[left], widths[right]) * 0.25, 0.1))
            )
            score = float(np.mean(exact_overlap))
            if score < 0.50:
                continue
            left_quality = (
                float(presence[left])
                * (median_confidence[left] if median_confidence[left] is not None else 1.0)
                * (median_keypoint_coverage[left] if median_keypoint_coverage[left] is not None else 1.0)
            )
            right_quality = (
                float(presence[right])
                * (median_confidence[right] if median_confidence[right] is not None else 1.0)
                * (median_keypoint_coverage[right] if median_keypoint_coverage[right] is not None else 1.0)
            )
            duplicate_slot = right if left_quality >= right_quality else left
            duplicate_score[duplicate_slot] = max(duplicate_score[duplicate_slot], score)

    return [
        {
            "track_id": int(features.track_ids[slot]),
            "track_lifetime_s": float(observed_counts[slot] / fps),
            "track_presence_ratio": float(presence[slot]),
            "median_detection_confidence": median_confidence[slot],
            "median_keypoint_coverage": median_keypoint_coverage[slot],
            "spatial_duplicate_score": float(duplicate_score[slot]),
        }
        for slot in range(track_count)
    ]


def collective_track_quality_failures(
    quality: Mapping[str, Any], parameters: HeuristicParameters
) -> list[str]:
    """Return group-graph quality gates that a visual ID does not satisfy."""

    failures: list[str] = []
    if float(quality["track_lifetime_s"]) + 1e-12 < parameters.collective_track_min_lifetime_s:
        failures.append("track_lifetime")
    if float(quality["track_presence_ratio"]) + 1e-12 < parameters.collective_track_min_presence_ratio:
        failures.append("track_presence_ratio")
    confidence = quality.get("median_detection_confidence")
    if (
        confidence is not None
        and float(confidence) + 1e-12
        < parameters.collective_track_min_median_detection_confidence
    ):
        failures.append("median_detection_confidence")
    keypoint_coverage = quality.get("median_keypoint_coverage")
    if (
        keypoint_coverage is not None
        and float(keypoint_coverage) + 1e-12
        < parameters.collective_track_min_median_keypoint_coverage
    ):
        failures.append("median_keypoint_coverage")
    if (
        float(quality["spatial_duplicate_score"]) + 1e-12
        >= parameters.collective_track_max_spatial_duplicate_score
    ):
        failures.append("spatial_duplicate_score")
    return failures


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
            if (
                not annotation_path.is_file()
                or not metadata_path.is_file()
                or not tracks_path.is_file()
            ):
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
    return (
        kept,
        dict(sorted(counts.items())),
        {behavior: counts[behavior] for behavior in sorted(counts) if behavior not in retained},
    )


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
    missing_train = (
        set(RETAINED_BEHAVIORS).intersection({record.behavior for record in records})
        - train_behaviors
    )
    missing_valid = (
        set(RETAINED_BEHAVIORS).intersection({record.behavior for record in records})
        - valid_behaviors
    )
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
            "train_fraction": len(train_records)
            / max(len(train_records) + len(validation_records), 1),
            "validation_fraction": len(validation_records)
            / max(len(train_records) + len(validation_records), 1),
            "train_sessions": sorted({record.recording_session for record in train_records}),
            "validation_sessions": sorted(
                {record.recording_session for record in validation_records}
            ),
        },
        "counts": {
            "all_discovered": sum(int(value) for value in counts.values())
            + sum(int(value) for value in excluded_counts.values()),
            "retained": sum(int(value) for value in counts.values()),
            "excluded": sum(int(value) for value in excluded_counts.values()),
            "retained_by_behavior": dict(sorted(counts.items())),
            "train_by_behavior": dict(
                sorted(Counter(record.behavior for record in train_records).items())
            ),
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
