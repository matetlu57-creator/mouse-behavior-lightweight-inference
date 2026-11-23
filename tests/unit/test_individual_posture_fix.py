"""Regression tests for posture screen precision, grooming recovery, and rearing detection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG,
    IndividualHeuristicConfig,
    individual_rule_masks,
)


def _make_mouse_trajectory(
    frames: int,
    *,
    fps: float = 30.0,
    ref: float = 7.918,
    tail_xy: np.ndarray,
    neck_xy: np.ndarray,
    nose_xy: np.ndarray,
    boxes: np.ndarray | None = None,
    hip_half_w: float = 0.5,
    ear_half_w: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct (centers, valid, keypoints, boxes) for a single mouse."""
    valid = np.ones((frames, 1), dtype=bool)
    keypoints = np.zeros((frames, 1, 7, 2), dtype=float)
    axis = neck_xy - tail_xy
    norms = np.linalg.norm(axis, axis=1, keepdims=True)
    unit = np.divide(axis, norms, out=np.broadcast_to([1.0, 0.0], axis.shape).copy(), where=norms > 1e-6)
    perp = np.stack((-unit[:, 1], unit[:, 0]), axis=1)

    hip_center = tail_xy + 0.25 * axis
    ear_center = 0.45 * neck_xy + 0.55 * nose_xy
    keypoints[:, 0, 0] = nose_xy
    keypoints[:, 0, 1] = ear_center + ear_half_w * perp
    keypoints[:, 0, 2] = ear_center - ear_half_w * perp
    keypoints[:, 0, 3] = neck_xy
    keypoints[:, 0, 4] = hip_center + hip_half_w * perp
    keypoints[:, 0, 5] = hip_center - hip_half_w * perp
    keypoints[:, 0, 6] = tail_xy

    centers = 0.5 * (neck_xy + hip_center)[:, None, :]
    if boxes is None:
        all_xy = keypoints[:, 0, :, :]
        pad = 0.05 * ref
        x0 = np.min(all_xy[:, :, 0], axis=1) - pad
        y0 = np.min(all_xy[:, :, 1], axis=1) - pad
        x1 = np.max(all_xy[:, :, 0], axis=1) + pad
        y1 = np.max(all_xy[:, :, 1], axis=1) + pad
        boxes = np.stack((x0, y0, x1, y1), axis=1)[:, None, :]
    return centers, valid, keypoints, boxes


def test_default_opt_in_and_signature_contracts() -> None:
    import inspect

    assert DEFAULT_INDIVIDUAL_CONFIG.enable_posture_screen is False
    assert IndividualHeuristicConfig(3.0).motion_speed_threshold_cm_s == 3.0
    params = inspect.signature(individual_rule_masks).parameters
    assert "boxes_cm" in params


def test_wall_adjacent_static_jitter_does_not_false_trigger_rearing_or_grooming() -> None:
    fps = 30.0
    frames = 120
    ref = 7.918
    t = np.arange(frames, dtype=float)

    # Mouse resting near arena wall (x=0 boundary) with realistic ~4px (~0.48 cm)
    # head/neck jitter (~3.3 cm/s < grooming sustained nose threshold)
    tail = np.column_stack((np.full(frames, 1.0), 10.0 + 0.04 * np.sin(0.25 * t)))
    neck = np.column_stack((np.full(frames, 5.2) + 0.40 * np.sin(0.25 * t), 10.0 + 0.06 * np.cos(0.30 * t)))
    nose = np.column_stack((np.full(frames, 7.6) + 0.40 * np.sin(0.28 * t), 10.0 + 0.08 * np.sin(0.35 * t)))

    centers, valid, kp, boxes = _make_mouse_trajectory(
        frames, fps=fps, ref=ref, tail_xy=tail, neck_xy=neck, nose_xy=nose,
    )
    # Add second mouse spanning wider arena envelope so wall margin stays realistic
    c2 = np.broadcast_to([40.0, 40.0], (frames, 1, 2)).copy()
    v2 = np.ones((frames, 1), dtype=bool)
    k2 = np.broadcast_to(kp[:, 0:1], (frames, 1, 7, 2)).copy() + 30.0
    b2 = np.broadcast_to([35.0, 35.0, 45.0, 45.0], (frames, 1, 4)).copy()

    cfg = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res = individual_rule_masks(
        np.concatenate((centers, c2), axis=1),
        np.concatenate((valid, v2), axis=1),
        fps,
        keypoints_cm=np.concatenate((kp, k2), axis=1),
        boxes_cm=np.concatenate((boxes, b2), axis=1),
        reference_body_cm=ref,
        config=cfg,
    )
    assert not res.masks["rearing"][:, 0].any(), "Wall-adjacent static jitter must not trigger Rearing"
    assert not res.masks["grooming"][:, 0].any(), "Sub-threshold static jitter must not trigger Grooming"
    assert res.masks["stationary"][:, 0].mean() >= 0.85

    # Stretched mouse lying flat along the wall with minor neck oscillation
    tail_flat = np.column_stack((np.full(frames, 1.0), np.full(frames, 10.0)))
    neck_flat = np.column_stack((np.full(frames, 6.8) + 0.45 * np.sin(0.2 * t), np.full(frames, 10.0)))
    nose_flat = np.column_stack((np.full(frames, 9.6) + 0.35 * np.sin(0.2 * t), np.full(frames, 10.0)))
    c_f, v_f, k_f, _ = _make_mouse_trajectory(
        frames, fps=fps, ref=ref, tail_xy=tail_flat, neck_xy=neck_flat, nose_xy=nose_flat,
    )
    b_f = np.broadcast_to([0.4, 8.8, 10.4, 11.2], (frames, 1, 4)).copy()
    res_flat = individual_rule_masks(
        np.concatenate((c_f, c2), axis=1),
        np.concatenate((v_f, v2), axis=1),
        fps,
        keypoints_cm=np.concatenate((k_f, k2), axis=1),
        boxes_cm=np.concatenate((b_f, b2), axis=1),
        reference_body_cm=ref,
        config=cfg,
    )
    assert not res_flat.masks["rearing"][:, 0].any(), "Stretched wall-lying mouse must not trigger Rearing"


def test_seated_grooming_not_killed_by_rearing_veto() -> None:
    fps = 30.0
    frames = 120
    ref = 7.918
    t = np.arange(frames, dtype=float)

    # Seated grooming with slight posterior sway (~0.18*ref), lateral turn (~0.09*ref),
    # crouched projection (~0.62*ref) near wall, and fast 5 Hz axial nose/head oscillations.
    sway = 0.65 * np.sin(0.15 * t)
    wash = 0.72 * np.sin(2.0 * np.pi * 4.5 * (t / fps))
    lat_turn = 0.35 * np.sin(0.35 * t)

    tail = np.column_stack((np.full(frames, 1.5), 10.0 + sway))
    neck = np.column_stack((np.full(frames, 4.8), 10.0 + sway + 0.35 * lat_turn))
    nose = np.column_stack((np.full(frames, 6.2) + wash, 10.0 + sway + lat_turn))

    centers, valid, kp, _ = _make_mouse_trajectory(
        frames, fps=fps, ref=ref, tail_xy=tail, neck_xy=neck, nose_xy=nose,
    )
    # Compact grooming detection box (mild pulsation ~0.11*ref)
    boxes = np.column_stack((
        np.full(frames, 1.0),
        9.0 + 0.4 * np.sin(0.25 * t),
        np.full(frames, 7.2) + 0.35 * np.sin(0.25 * t),
        11.2 + 0.4 * np.sin(0.25 * t),
    ))[:, None, :]

    c2 = np.broadcast_to([40.0, 40.0], (frames, 1, 2)).copy()
    v2 = np.ones((frames, 1), dtype=bool)
    k2 = np.broadcast_to(kp[:, 0:1], (frames, 1, 7, 2)).copy() + 30.0
    b2 = np.broadcast_to([35.0, 35.0, 45.0, 45.0], (frames, 1, 4)).copy()

    cfg = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res = individual_rule_masks(
        np.concatenate((centers, c2), axis=1),
        np.concatenate((valid, v2), axis=1),
        fps,
        keypoints_cm=np.concatenate((kp, k2), axis=1),
        boxes_cm=np.concatenate((boxes, b2), axis=1),
        reference_body_cm=ref,
        config=cfg,
    )
    assert res.masks["grooming"][:, 0].mean() >= 0.70, (
        f"Seated grooming with mild sway should be detected, got {res.masks['grooming'][:, 0].mean():.2f}"
    )
    assert not res.masks["rearing"][:, 0].any(), "Seated grooming must not be misclassified as Rearing"

    # Keypoint-only (boxes_cm=None) stretched-seat grooming should also qualify without rearing veto
    res_nobox = individual_rule_masks(centers, valid, fps, keypoints_cm=kp, boxes_cm=None, reference_body_cm=ref, config=cfg)
    assert res_nobox.masks["grooming"][:, 0].mean() >= 0.70


def test_wall_and_open_field_rearing_stably_detected() -> None:
    fps = 30.0
    frames = 120
    ref = 7.918
    t = np.arange(frames, dtype=float)

    # 1) Wall-supported upright rear: large vertical rise/drop cycle, box shrinks cleanly
    lift = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / frames)))  # smooth 0 -> 1 -> 0
    tail = np.column_stack((np.full(frames, 1.5), np.full(frames, 10.0)))
    neck = np.column_stack((5.8 - 2.6 * lift, np.full(frames, 10.0)))
    nose = np.column_stack((8.8 - 4.4 * lift, np.full(frames, 10.0)))

    c_w, v_w, k_w, b_w = _make_mouse_trajectory(
        frames, fps=fps, ref=ref, tail_xy=tail, neck_xy=neck, nose_xy=nose,
    )
    c2 = np.broadcast_to([40.0, 40.0], (frames, 1, 2)).copy()
    v2 = np.ones((frames, 1), dtype=bool)
    k2 = np.broadcast_to(k_w[:, 0:1], (frames, 1, 7, 2)).copy() + 30.0
    b2 = np.broadcast_to([35.0, 35.0, 45.0, 45.0], (frames, 1, 4)).copy()

    cfg = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res_wall = individual_rule_masks(
        np.concatenate((c_w, c2), axis=1),
        np.concatenate((v_w, v2), axis=1),
        fps,
        keypoints_cm=np.concatenate((k_w, k2), axis=1),
        boxes_cm=np.concatenate((b_w, b2), axis=1),
        reference_body_cm=ref,
        config=cfg,
    )
    assert res_wall.masks["rearing"][:, 0].mean() >= 0.60, (
        f"Wall rearing coverage too low: {res_wall.masks['rearing'][:, 0].mean():.2f}"
    )
    assert not res_wall.masks["grooming"][:, 0].any()

    # 2) Open-field rearing (far from walls) with extended peak projection > 0.95*ref
    tail_o = np.column_stack((np.full(frames, 20.0), np.full(frames, 20.0)))
    neck_o = np.column_stack((24.0 + 2.4 * lift, np.full(frames, 20.0)))
    nose_o = np.column_stack((26.2 + 3.4 * lift, np.full(frames, 20.0)))
    c_o, v_o, k_o, b_o = _make_mouse_trajectory(
        frames, fps=fps, ref=ref, tail_xy=tail_o, neck_xy=neck_o, nose_xy=nose_o,
    )
    # Envelope walls at x=0 and x=45 so mouse at x=20..30 is strictly open-field (d_wall > 1.5*ref)
    b_far_lo = np.broadcast_to([0.0, 0.0, 3.0, 3.0], (frames, 1, 4)).copy()
    b_far_hi = np.broadcast_to([42.0, 42.0, 45.0, 45.0], (frames, 1, 4)).copy()
    c_lo = np.broadcast_to([1.5, 1.5], (frames, 1, 2)).copy()
    c_hi = np.broadcast_to([43.5, 43.5], (frames, 1, 2)).copy()
    k_lo = np.zeros((frames, 1, 7, 2), dtype=float)
    k_hi = np.full((frames, 1, 7, 2), 43.5, dtype=float)

    res_open = individual_rule_masks(
        np.concatenate((c_o, c_lo, c_hi), axis=1),
        np.concatenate((v_o, v2, v2), axis=1),
        fps,
        keypoints_cm=np.concatenate((k_o, k_lo, k_hi), axis=1),
        boxes_cm=np.concatenate((b_o, b_far_lo, b_far_hi), axis=1),
        reference_body_cm=ref,
        config=cfg,
    )
    assert res_open.masks["rearing"][:, 0].mean() >= 0.60, (
        f"Open-field rearing coverage too low: {res_open.masks['rearing'][:, 0].mean():.2f}"
    )
    assert not res_open.masks["grooming"][:, 0].any()


def test_unobserved_frames_never_claimed_without_bounded_bridge() -> None:
    fps = 30.0
    frames = 120
    ref = 7.918
    t = np.arange(frames, dtype=float)
    wash = 0.72 * np.sin(2.0 * np.pi * 4.5 * (t / fps))
    tail = np.column_stack((np.full(frames, 10.0), np.full(frames, 10.0)))
    neck = np.column_stack((np.full(frames, 13.5), np.full(frames, 10.0)))
    nose = np.column_stack((np.full(frames, 15.0) + wash, np.full(frames, 10.0)))
    centers, valid, kp, boxes = _make_mouse_trajectory(
        frames, fps=fps, ref=ref, tail_xy=tail, neck_xy=neck, nose_xy=nose,
    )
    # Create an extended unobserved dropout (> both grooming_bridge_max_s=0.5s and static_detection_gap_max_s=1.0s)
    valid[30:70, 0] = False
    cfg = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res = individual_rule_masks(
        centers, valid, fps, keypoints_cm=kp, boxes_cm=boxes, reference_body_cm=ref, config=cfg,
    )
    for mask in res.masks.values():
        assert not mask[30:70, 0].any(), "Long unobserved gap must not be fabricated by posture rules"
