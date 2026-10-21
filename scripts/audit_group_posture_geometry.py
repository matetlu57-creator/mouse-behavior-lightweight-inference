#!/usr/bin/env python3
"""Compare Huddling and Social clustering geometry on the training split only.

The annotations select the target IDs for this diagnostic. No labels are passed
to trajectory inference, and validation rows are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.exported_dataset import (
    HeuristicParameters,
    SampleRecord,
    classify_target_ids,
    _directional_walker_in_group,
    load_exported_features,
    predict_features,
)


BEHAVIORS = {"Huddling", "Social clustering"}


def _posterior_and_nose(
    path: Path, frame_count: int, target_ids: tuple[int, ...], cm_per_pixel: float
) -> tuple[np.ndarray, np.ndarray]:
    slots = {track_id: index for index, track_id in enumerate(target_ids)}
    posterior = np.full((frame_count, len(slots), 2), np.nan, dtype=np.float32)
    nose = np.full_like(posterior, np.nan)
    with path.open("r", encoding="utf-8") as handle:
        frames = json.load(handle)
    for frame in frames:
        index = int(frame.get("frame", -1))
        if not 0 <= index < frame_count:
            continue
        for detection in frame.get("detections", ()):
            slot = slots.get(int(detection.get("track_id", -1)))
            if slot is None:
                continue
            points = np.asarray(detection.get("keypoints", ()), dtype=float)
            if points.shape != (7, 3):
                continue
            visible = np.isfinite(points).all(axis=1) & (points[:, 2] >= 0.25)
            if visible[0]:
                nose[index, slot] = points[0, :2] * cm_per_pixel
            # Two posterior points are needed: a lone tail estimate can move
            # when the animal changes posture without moving across the floor.
            rear = points[[4, 5, 6], :2][visible[[4, 5, 6]]]
            if len(rear) >= 2:
                posterior[index, slot] = np.median(rear, axis=0) * cm_per_pixel
    return posterior, nose


def _half_second_speed(points: np.ndarray, fps: float) -> np.ndarray:
    shift = max(1, int(round(0.5 * fps)))
    result = np.full(points.shape[:2], np.nan, dtype=np.float32)
    if shift >= len(points):
        return result
    observed = np.isfinite(points).all(axis=2)
    continuous = np.lib.stride_tricks.sliding_window_view(
        observed, shift + 1, axis=0
    ).all(axis=-1)
    distance = np.linalg.norm(points[shift:] - points[:-shift], axis=2)
    result[shift:] = np.where(continuous, distance * fps / shift, np.nan)
    return result


def _finite_median(values: np.ndarray) -> float | None:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    return float(np.median(observed)) if observed.size else None


def _sample(
    row: dict[str, str], max_tracks: int, parameters: HeuristicParameters
) -> dict[str, object]:
    target_ids = tuple(int(value) for value in row["mouse_ids"].split(","))
    path = Path(row["tracks_path"])
    fps = float(row["fps"])
    features = load_exported_features(path, fps=fps, max_tracks=max_tracks)
    slot_by_id = {int(track_id): slot for slot, track_id in enumerate(features.track_ids)}
    result: dict[str, object] = {
        "sample_id": row["sample_id"],
        "behavior": row["behavior"],
        "member_count": len(target_ids),
        "target_ids_available": all(track_id in slot_by_id for track_id in target_ids),
    }
    if not result["target_ids_available"]:
        return result

    record = SampleRecord(
        sample_id=row["sample_id"],
        behavior=row["behavior"],
        canonical_behavior=row["canonical_behavior"],
        sample_dir=row["sample_dir"],
        annotation_path=row["annotation_path"],
        metadata_path=row["metadata_path"],
        tracks_path=row["tracks_path"],
        clip_path=row["clip_path"],
        source_video=row["source_video"],
        recording_session=row["recording_session"],
        fps=fps,
        frame_count=int(row["frame_count"]),
        mouse_ids=target_ids,
        confidence=row["confidence"],
        frame_start=int(row["frame_start"]),
        frame_end=int(row["frame_end"]),
        time_start=float(row["time_start"]),
        time_end=float(row["time_end"]),
    )
    for name, changes in (
        ("baseline", {}),
        ("core", {"group_core_candidate_mode": 1.0}),
        ("geometry_065", {"clustering_formation_rescue_min_geometry_support": 0.65}),
    ):
        prediction = predict_features(
            features,
            replace(parameters, **changes),
        )
        scored = classify_target_ids(record, prediction)
        result[f"{name}_strict_correct"] = bool(scored["correct"])
        result[f"{name}_predicted_behavior"] = scored["predicted_behavior"]
        result[f"{name}_predicted_ids"] = ",".join(
            str(value) for value in scored["predicted_target_ids"]
        )
        if name == "core":
            target_key = ",".join(str(value) for value in sorted(target_ids))
            target_group_scores = prediction["group_scores"].get(target_key, {})
            result["core_target_huddle_score"] = float(target_group_scores.get("huddle", 0.0))
            result["core_target_clustering_score"] = float(
                target_group_scores.get("social_clustering", 0.0)
            )

    slots = tuple(slot_by_id[track_id] for track_id in target_ids)
    result["target_directional_walker"] = _directional_walker_in_group(
        features, target_ids, parameters
    )
    body = features.centers_cm[:, slots]
    posterior, nose = _posterior_and_nose(
        path, len(features.valid), target_ids, features.cm_per_pixel
    )
    body_speed = _half_second_speed(body, fps)
    posterior_speed = _half_second_speed(posterior, fps)
    nose_speed = _half_second_speed(nose, fps)
    nose_only_motion = (
        np.isfinite(nose_speed)
        & np.isfinite(body_speed)
        & np.isfinite(posterior_speed)
        & (nose_speed >= 10.0)
        & (body_speed < 10.0)
        & (posterior_speed < 10.0)
    )
    complete_motion = (
        np.isfinite(nose_speed)
        & np.isfinite(body_speed)
        & np.isfinite(posterior_speed)
    )
    result.update(
        body_speed_median_cm_s=_finite_median(body_speed),
        posterior_speed_median_cm_s=_finite_median(posterior_speed),
        nose_speed_median_cm_s=_finite_median(nose_speed),
        body_speed_p90_cm_s=(
            float(np.nanquantile(body_speed, 0.9)) if np.isfinite(body_speed).any() else None
        ),
        posterior_speed_p90_cm_s=(
            float(np.nanquantile(posterior_speed, 0.9))
            if np.isfinite(posterior_speed).any()
            else None
        ),
        nose_only_motion_fraction=(
            float(nose_only_motion.sum() / complete_motion.sum())
            if complete_motion.any()
            else None
        ),
        posterior_coverage_fraction=float(np.mean(np.isfinite(posterior).all(axis=2))),
    )

    # Compare group area only where every annotated participant is visible;
    # a missing member would make a crowded Huddle look artificially small.
    fully_visible = np.isfinite(body).all(axis=(1, 2))
    result["fully_visible_fraction"] = float(np.mean(fully_visible))
    if fully_visible.any():
        visible_body = body[fully_visible]
        centroid = np.mean(visible_body, axis=1, keepdims=True)
        radius = np.mean(np.linalg.norm(visible_body - centroid, axis=2), axis=1)
        result["radius_median_cm"] = _finite_median(radius)
        result["spread_area_median_cm2"] = _finite_median(np.pi * radius**2)
        boundary = max(1, len(radius) // 5)
        result["radius_onset_cm"] = _finite_median(radius[:boundary])
        result["radius_final_cm"] = _finite_median(radius[-boundary:])
        result["radius_drop_cm"] = float(
            np.median(radius[:boundary]) - np.median(radius[-boundary:])
        )
        internal = np.isin(features.pair_i, slots) & np.isin(features.pair_j, slots)
        distances = features.pair_distance_cm[fully_visible][:, internal]
        overlap = features.pair_box_iou
        result["pair_distance_median_cm"] = _finite_median(distances)
        if overlap is not None:
            iou = overlap[fully_visible][:, internal]
            finite = np.isfinite(iou)
            result["box_overlap_fraction"] = (
                float(np.mean(iou[finite] > 0.0)) if finite.any() else None
            )
            result["box_iou_when_overlapping"] = _finite_median(iou[iou > 0.0])
    displacement = []
    for member_index in range(len(target_ids)):
        points = body[np.isfinite(body[:, member_index]).all(axis=1), member_index]
        if len(points) >= 2:
            displacement.append(float(np.linalg.norm(points[-1] - points[0])))
    result["max_member_net_displacement_cm"] = max(displacement, default=None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.train_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("the geometry audit accepts training rows only")
    with args.parameters.open("r", encoding="utf-8") as handle:
        parameters = HeuristicParameters(**json.load(handle))
    results = [
        _sample(row, args.max_tracks, parameters)
        for row in rows
        if row["behavior"] in BEHAVIORS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        columns = sorted({key for result in results for key in result})
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} training group samples to {args.output}")


if __name__ == "__main__":
    main()
