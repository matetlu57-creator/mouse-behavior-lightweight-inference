#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Randomized exact-equivalence check for v1.42.1 Identity Cascade."""

from __future__ import annotations
import numpy as np
from regression_performance_test import load_modules, make_detection


def one_case(new_base, seed: int, n: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    cfg = {
        "keypoint_motion": {},
        "instance_mask_memory": {},
        "performance": {
            "identity_cost_backend": "numpy",
            "identity_cascade": {
                "enabled": True,
                "min_cells": 1,
                "sparse_density_threshold": 1.0,
            },
        },
    }
    assigner = new_base.KeypointMotionIdentityAssigner(cfg, max_mice=20)
    detections = []
    assigner.tracks = {}
    for logical_id in range(n):
        det = make_detection(new_base, rng, logical_id)
        shift = np.array([rng.uniform(-250, 800), rng.uniform(-150, 500)])
        det.keypoints_px = np.asarray(det.keypoints_px, dtype=np.float64) + shift
        det.bbox_xyxy = np.asarray(det.bbox_xyxy, dtype=np.float64) + np.array(
            [shift[0], shift[1], shift[0], shift[1]]
        )
        scale = float(rng.uniform(0.75, 1.30))
        center = (det.bbox_xyxy[:2] + det.bbox_xyxy[2:]) / 2.0
        det.bbox_xyxy = np.r_[
            center + (det.bbox_xyxy[:2] - center) * scale,
            center + (det.bbox_xyxy[2:] - center) * scale,
        ]
        det.invalidate_derived_geometry_cache()
        det.refresh_derived_geometry_cache()
        detections.append(det)

        offset = rng.normal(0.0, 45.0, size=2)
        track_center = det.center_px + offset
        track_box = det.bbox_xyxy + np.array([offset[0], offset[1], offset[0], offset[1]])
        track_kpts = np.asarray(det.keypoints_px, dtype=np.float64) + offset
        assigner.tracks[logical_id] = new_base.IdentityTrack(
            logical_id=logical_id,
            last_center_px=track_center,
            velocity_px_per_frame=rng.normal(0.0, 1.0, size=2),
            last_frame=int(rng.integers(0, 4)),
            raw_track_id=logical_id,
            body_length_px=float(det.body_length_px * rng.uniform(0.8, 1.2)),
            heading_vector=(rng.normal(size=2) if rng.random() < 0.7 else None),
            last_keypoints_px=track_kpts,
            last_keypoint_conf=np.asarray(det.keypoint_conf, dtype=np.float64).copy(),
            last_bbox_xyxy=track_box,
            last_box_conf=float(det.box_conf),
        )
        if rng.random() < 0.7:
            det.heading_vector = rng.normal(size=2)
        assigner.kpt_missing[logical_id] = int(rng.integers(0, 3))

    track_ids = list(range(n))
    frame = 5
    assigner.identity_cascade_enabled = True
    assigner.identity_cascade_min_cells = 1
    assigner.identity_cascade_sparse_density = 1.0
    sparse = assigner._base_cost_matrix_numpy(track_ids, detections, frame)
    candidate_count = int(assigner.last_fast_gate_candidate_count)
    total_count = int(assigner.last_fast_gate_total_count)

    assigner.identity_cascade_enabled = False
    dense = assigner._base_cost_matrix_numpy(track_ids, detections, frame)
    np.testing.assert_array_equal(sparse, dense)
    return candidate_count, total_count


def main() -> int:
    _, new_base, _, _ = load_modules()
    stats = []
    for offset in range(50):
        n = 5 + (offset % 16)
        stats.append(one_case(new_base, 1000 + offset, n))
    densities = [c / t for c, t in stats if t]
    print("IDENTITY CASCADE FUZZ: PASS 50/50 exact matrices")
    print(f"candidate_density_min={min(densities):.6f}")
    print(f"candidate_density_max={max(densities):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
