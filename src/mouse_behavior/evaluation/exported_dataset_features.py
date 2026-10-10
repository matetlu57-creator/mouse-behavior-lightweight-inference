"""Loading exported tracks and computing clip-level trajectory features."""

from __future__ import annotations

from mouse_behavior.evaluation.exported_dataset_models import (
    ExportedFeatures,
    HeuristicParameters,
)
from mouse_behavior.evaluation.exported_dataset_records import (
    _read_json,
)

from collections import Counter
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from mouse_behavior.preprocessing.scale_resolution import robust_reference_body_px, default_scale_config, log_scale_warnings, resolve_scale, scale_block_of
from mouse_behavior.behavior.collective_state import estimate_morphology, robust_body_centers
from mouse_behavior.preprocessing.pose_primitives import nose_to_nearest_keypoint_cm

def _safe_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _huddle_pair_distance_cm(features: ExportedFeatures, parameters: HeuristicParameters) -> float:
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
    scale_config: Mapping[str, Any] | None = None,
) -> ExportedFeatures:
    """Load label-free trajectory input and derive scale-normalized features."""

    path = Path(tracks_path)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive: {fps}")
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
    if not selected_ids:
        raise ValueError(f"tracks.json contains no visual track IDs: {path}")
    track_index = {track_id: index for index, track_id in enumerate(selected_ids)}
    tracks = len(selected_ids)
    keypoints = np.full((frame_count, tracks, 7, 2), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count, tracks), dtype=bool)
    boxes = np.full((frame_count, tracks, 4), np.nan, dtype=np.float32)
    detection_confidence = np.full((frame_count, tracks), np.nan, dtype=np.float32)
    keypoint_coverage = np.full((frame_count, tracks), np.nan, dtype=np.float32)
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
            keypoint_coverage[frame_index, slot] = float(np.mean(point_valid))
            box = np.asarray(detection.get("box", []), dtype=float)
            if (
                box.shape == (4,)
                and np.isfinite(box).all()
                and box[2] > box[0]
                and box[3] > box[1]
            ):
                boxes[frame_index, slot] = box
            confidence = detection.get("detection_confidence")
            if confidence is not None:
                try:
                    confidence_value = float(confidence)
                except (TypeError, ValueError):
                    confidence_value = float("nan")
                if np.isfinite(confidence_value):
                    detection_confidence[frame_index, slot] = confidence_value

        nose = keypoints[:, :, 0]
    tail = keypoints[:, :, 6]
    center_points = keypoints[:, :, [3, 4, 5]]
    center_good = np.isfinite(center_points).all(axis=3)
    body_centers_px = np.ma.median(
        np.ma.masked_where(
            np.broadcast_to(~center_good[..., None], center_points.shape), center_points
        ),
        axis=2,
    ).filled(np.nan)
    any_good = np.isfinite(keypoints).all(axis=3)
    fallback_centers_px = np.ma.median(
        np.ma.masked_where(np.broadcast_to(~any_good[..., None], keypoints.shape), keypoints),
        axis=2,
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
    # # [STEP4+ 2026-10-05] 逐视频体长尺度：剔除姿态坏轨（尾关键点塌缩等）后的鲁棒像素体长
    reference_body_px = (
        robust_reference_body_px(per_track_scales) if per_track_scales else 60.0
    )
    resolved_scale = resolve_scale(
        scale_block_of(scale_config) if scale_config is not None else default_scale_config(),
        reference_body_px=reference_body_px,
    )
    log_scale_warnings(resolved_scale)
    cm_per_pixel = resolved_scale.cm_per_pixel
    # Hip span is a better body-width proxy than ear span; ears are fallback
    # evidence only when no reliable hip-width samples exist.
    hip_widths_px = np.linalg.norm(keypoints[:, :, 4] - keypoints[:, :, 5], axis=2)
    hip_widths_px = hip_widths_px[valid & np.isfinite(hip_widths_px) & (hip_widths_px >= 1.0)]
    ear_widths_px = np.linalg.norm(keypoints[:, :, 1] - keypoints[:, :, 2], axis=2)
    ear_widths_px = ear_widths_px[valid & np.isfinite(ear_widths_px) & (ear_widths_px >= 1.0)]
    width_samples = hip_widths_px if hip_widths_px.size else ear_widths_px
    mouse_width_cm = (
        float(np.median(width_samples) * cm_per_pixel)
        if width_samples.size else resolved_scale.reference_width_cm
    )
    centers_cm = centers_px * cm_per_pixel
    collective_centers_px, collective_valid = robust_body_centers(keypoints)
    collective_valid &= valid & np.isfinite(collective_centers_px).all(axis=2)
    collective_centers_cm = collective_centers_px * cm_per_pixel
    collective_centers_cm[~collective_valid] = np.nan
    collective_velocity = np.zeros((frame_count, tracks, 2), dtype=np.float32)
    collective_speed = np.zeros((frame_count, tracks), dtype=np.float32)
    if frame_count > 1:
        collective_delta = collective_centers_cm[1:] - collective_centers_cm[:-1]
        collective_continuity = (
            collective_valid[1:]
            & collective_valid[:-1]
            & np.isfinite(collective_delta).all(axis=2)
        )
        collective_velocity[1:][collective_continuity] = (
            collective_delta[collective_continuity] * float(fps)
        )
        collective_speed[1:][collective_continuity] = np.linalg.norm(
            collective_velocity[1:][collective_continuity], axis=1
        )
    collective_keypoints_cm = keypoints * cm_per_pixel
    collective_boxes_cm = boxes * cm_per_pixel
    collective_quality = np.nan_to_num(keypoint_coverage, nan=0.0)
    if np.isfinite(detection_confidence).any():
        confidence_quality = np.nan_to_num(detection_confidence, nan=0.0)
        collective_quality = np.minimum(collective_quality, confidence_quality)
    body_length_ref_cm, body_width_ref_cm, body_area_ref_cm2 = estimate_morphology(
        collective_keypoints_cm,
        collective_valid,
        pose_quality=collective_quality,
        boxes=boxes,
        reference_body_cm=resolved_scale.reference_body_cm,
        reference_body_width_cm=resolved_scale.reference_width_cm,
    )
    nose_cm = nose * cm_per_pixel
    tail_cm = tail * cm_per_pixel
    head_points_cm = keypoints[:, :, [0, 1, 2]] * cm_per_pixel

    # Median smoothing keeps one-frame keypoint jitter from masquerading as
    # the post-contact acceleration of an attacking mouse. Missing nose
    # points remain missing; no identity or coordinate is interpolated.
    smoothed_nose = np.full_like(nose_cm, np.nan)
    if frame_count >= 3:
        nose_windows = np.stack((nose_cm[:-2], nose_cm[1:-1], nose_cm[2:]))
        complete_windows = np.isfinite(nose_windows).all(axis=(0, 3))
        smoothed_nose[1:-1] = np.where(
            complete_windows[..., None], np.median(nose_windows, axis=0), np.nan
        )
    nose_speed = np.full((frame_count, tracks), np.nan, dtype=np.float32)
    nose_acceleration = np.full((frame_count, tracks), np.nan, dtype=np.float32)
    if frame_count >= 4:
        nose_delta = smoothed_nose[1:] - smoothed_nose[:-1]
        nose_speed[1:] = np.linalg.norm(nose_delta, axis=2) * float(fps)
    if frame_count >= 5:
        nose_acceleration[2:] = np.linalg.norm(
            (smoothed_nose[2:] - 2 * smoothed_nose[1:-1] + smoothed_nose[:-2])
            * float(fps) ** 2,
            axis=2,
        )

    velocity = np.zeros((frame_count, tracks, 2), dtype=np.float32)
    motion_valid = np.zeros((frame_count, tracks), dtype=bool)
    if frame_count > 1:
        delta = centers_cm[1:] - centers_cm[:-1]
        continuity = valid[1:] & valid[:-1] & np.all(np.isfinite(delta), axis=2)
        velocity[1:][continuity] = delta[continuity] * float(fps)
        motion_valid[1:] = continuity
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
    nose_nearest_ltr = np.full_like(pair_distance, np.inf)
    nose_nearest_rtl = np.full_like(pair_distance, np.inf)
    nose_pair = np.full_like(pair_distance, np.inf)
    pair_tail = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    head_left_to_right = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    head_right_to_left = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    tail_left_to_right = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    tail_right_to_left = np.full((frame_count, pairs), np.inf, dtype=np.float32)
    pair_box_iou = np.full((frame_count, pairs), np.nan, dtype=np.float32)
    pair_box_gap = np.full((frame_count, pairs), np.nan, dtype=np.float32)
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
        left_head = head_points_cm[frame, pair_i[good]]
        right_head = head_points_cm[frame, pair_j[good]]
        left_tail = tail_cm[frame, pair_i[good]]
        right_tail = tail_cm[frame, pair_j[good]]
        # [STEP6 2026-10-05] Approach/Huddling: 鼻头最近关键点/鼻-鼻距离改为帧循环之后按规范用
        # pose_primitives.nose_to_nearest_keypoint_cm 逐对计算，不在本循环内联。
        left_head_distances = np.linalg.norm(right_head - left_nose[:, None, :], axis=2)
        right_head_distances = np.linalg.norm(left_head - right_nose[:, None, :], axis=2)
        head_left_to_right[frame, good] = np.min(
            np.where(np.isfinite(left_head_distances), left_head_distances, np.inf), axis=1
        )
        head_right_to_left[frame, good] = np.min(
            np.where(np.isfinite(right_head_distances), right_head_distances, np.inf), axis=1
        )
        tail_left_to_right[frame, good] = np.linalg.norm(left_nose - right_tail, axis=1)
        tail_right_to_left[frame, good] = np.linalg.norm(right_nose - left_tail, axis=1)
        pair_head[frame, good] = np.minimum(
            head_left_to_right[frame, good], head_right_to_left[frame, good]
        )
        pair_tail[frame, good] = np.minimum(
            tail_left_to_right[frame, good], tail_right_to_left[frame, good]
        )
        good_boxes = good & np.isfinite(boxes[frame, pair_i]).all(axis=1)
        good_boxes &= np.isfinite(boxes[frame, pair_j]).all(axis=1)
        if np.any(good_boxes):
            left_box = boxes[frame, pair_i[good_boxes]]
            right_box = boxes[frame, pair_j[good_boxes]]
            overlap_width = np.maximum(
                np.minimum(left_box[:, 2], right_box[:, 2])
                - np.maximum(left_box[:, 0], right_box[:, 0]),
                0.0,
            )
            overlap_height = np.maximum(
                np.minimum(left_box[:, 3], right_box[:, 3])
                - np.maximum(left_box[:, 1], right_box[:, 1]),
                0.0,
            )
            intersection = overlap_width * overlap_height
            left_area = (left_box[:, 2] - left_box[:, 0]) * (
                left_box[:, 3] - left_box[:, 1]
            )
            right_area = (right_box[:, 2] - right_box[:, 0]) * (
                right_box[:, 3] - right_box[:, 1]
            )
            pair_box_iou[frame, good_boxes] = intersection / np.maximum(
                left_area + right_area - intersection, 1e-6
            )
            dx = np.maximum(
                np.maximum(left_box[:, 0] - right_box[:, 2], right_box[:, 0] - left_box[:, 2]),
                0.0,
            )
            dy = np.maximum(
                np.maximum(left_box[:, 1] - right_box[:, 3], right_box[:, 1] - left_box[:, 3]),
                0.0,
            )
            pair_box_gap[frame, good_boxes] = np.hypot(dx, dy) * cm_per_pixel
    # [STEP6 2026-10-05] Approach/Huddling: 用 pose_primitives.nose_to_nearest_keypoint_cm 逐对计算——
    # actor 鼻 = 该侧 nose 关键点 cm，target keypoints = 对侧 7 关键点 cm，target 逐点 valid=isfinite；
    # 缺失帧保持非有限值，不做插值、不推断身份。pair_nose_distance_cm 用 collective_keypoints_cm 的 nose 列。
    collective_nose_cm = collective_keypoints_cm[:, :, 0]
    collective_keypoint_valid = np.isfinite(collective_keypoints_cm).all(axis=3)
    for pair_index, (left_slot, right_slot) in enumerate(zip(pair_i, pair_j)):
        left_slot = int(left_slot)
        right_slot = int(right_slot)
        nose_nearest_ltr[:, pair_index] = nose_to_nearest_keypoint_cm(
            collective_nose_cm[:, left_slot],
            collective_keypoints_cm[:, right_slot],
            collective_keypoint_valid[:, right_slot],
        )
        nose_nearest_rtl[:, pair_index] = nose_to_nearest_keypoint_cm(
            collective_nose_cm[:, right_slot],
            collective_keypoints_cm[:, left_slot],
            collective_keypoint_valid[:, left_slot],
        )
        with np.errstate(invalid="ignore"):
            nose_pair[:, pair_index] = np.linalg.norm(
                collective_nose_cm[:, left_slot] - collective_nose_cm[:, right_slot], axis=1
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
            nearest_by_track[:, left] = np.minimum(
                nearest_by_track[:, left], pair_distance[:, pair_index]
            )
            nearest_by_track[:, right] = np.minimum(
                nearest_by_track[:, right], pair_distance[:, pair_index]
            )
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
        pair_nose_nearest_left_to_right_cm=nose_nearest_ltr,
        pair_nose_nearest_right_to_left_cm=nose_nearest_rtl,
        pair_nose_distance_cm=nose_pair,
        pair_nose_tail_cm=pair_tail,
        nearest_distance_cm=nearest.astype(np.float32),
        mean_nearest_distance_cm=mean_nearest,
        cm_per_pixel=float(cm_per_pixel),
        reference_body_cm=resolved_scale.reference_body_cm,
        track_selection_truncated=len(counts) > tracks,
        selected_track_count=tracks,
        mouse_width_cm=mouse_width_cm,
        motion_valid=motion_valid,
        detection_confidence=detection_confidence,
        keypoint_coverage=keypoint_coverage,
        track_presence_fraction=valid.mean(axis=0).astype(np.float32),
        pair_box_iou=pair_box_iou,
        pair_box_gap_cm=pair_box_gap,
        heading_unit=heading.astype(np.float32),
        pair_nose_head_left_to_right_cm=head_left_to_right,
        pair_nose_head_right_to_left_cm=head_right_to_left,
        pair_nose_tail_left_to_right_cm=tail_left_to_right,
        pair_nose_tail_right_to_left_cm=tail_right_to_left,
        nose_speed_cm_s=nose_speed,
        nose_acceleration_cm_s2=nose_acceleration,
        collective_centers_cm=collective_centers_cm.astype(np.float32),
        collective_valid=collective_valid,
        collective_velocity_cm_s=collective_velocity,
        collective_speed_cm_s=collective_speed,
        collective_keypoints_cm=collective_keypoints_cm.astype(np.float32),
        collective_boxes_cm=collective_boxes_cm.astype(np.float32),
        body_length_ref_cm=body_length_ref_cm.astype(np.float32),
        body_width_ref_cm=body_width_ref_cm.astype(np.float32),
        body_area_ref_cm2=body_area_ref_cm2.astype(np.float32),
    )
