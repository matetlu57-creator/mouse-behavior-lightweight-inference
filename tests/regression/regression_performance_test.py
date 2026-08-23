from __future__ import annotations

import importlib
import importlib.util
import io
from dataclasses import asdict
import pickle
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__test__ = False
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = Path(__file__).resolve().parent / "fixtures" / "legacy_v138"


def install_stubs() -> None:
    disk = types.ModuleType("disk_sequence_guard")

    class DiskSequenceIdentityGuard:
        def __init__(self, config: Any = None, num_keypoints: int = 7) -> None:
            self.enabled = False
            self.min_reliability = 0.0

        def export_state(self) -> dict[str, Any]:
            return {}

        def import_state(self, state: Any) -> None:
            return None

    def choose_adaptive_thread_count(
        rows: int,
        cols: int,
        fixed_threads: int = 1,
        auto_enabled: bool = True,
        max_threads: int = 4,
        parallel_min_cells: int = 16384,
    ) -> int:
        return max(1, min(int(fixed_threads), int(max_threads)))

    disk.DiskSequenceIdentityGuard = DiskSequenceIdentityGuard
    disk.choose_adaptive_thread_count = choose_adaptive_thread_count
    sys.modules["disk_sequence_guard"] = disk

    pose = types.ModuleType("pose_quality_recovery")

    class FrameRecoveryStats:
        pass

    pose.FrameRecoveryStats = FrameRecoveryStats
    sys.modules["pose_quality_recovery"] = pose
    sys.modules["mask_cluster_reid"] = types.ModuleType("mask_cluster_reid")
    sys.modules["adaptive_arena_boundary"] = types.ModuleType("adaptive_arena_boundary")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules():
    install_stubs()
    old_base = load_module(
        "base_original_for_test", ORIGINAL / "mouse_chase_attack_extractor_base.py"
    )
    sys.modules["mouse_chase_attack_extractor_base"] = old_base
    old_main = load_module("main_original_for_test", ORIGINAL / "mouse_chase_attack_high_recall.py")
    # Other tests may have imported the package before these controlled stubs
    # were installed. Reload both modules so this regression is independent of
    # pytest collection/execution order and always exercises the same fixtures.
    new_base = importlib.reload(
        importlib.import_module("mouse_behavior.full_pipeline.extractor_base")
    )
    new_main = importlib.reload(importlib.import_module("mouse_behavior.full_pipeline.high_recall"))
    return old_base, new_base, old_main, new_main


def assert_same_value(a: Any, b: Any, path: str = "root") -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=path)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        assert list(a.keys()) == list(b.keys()), f"key order mismatch at {path}"
        for key in a:
            assert_same_value(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"length mismatch at {path}"
        for index, (left, right) in enumerate(zip(a, b)):
            assert_same_value(left, right, f"{path}[{index}]")
        return
    if hasattr(a, "__dataclass_fields__") and hasattr(b, "__dataclass_fields__"):
        for key in a.__dataclass_fields__:
            if key.startswith("_cached_") or key not in b.__dataclass_fields__:
                continue
            assert_same_value(getattr(a, key), getattr(b, key), f"{path}.{key}")
        return
    if isinstance(a, float) or isinstance(b, float):
        av, bv = float(a), float(b)
        if np.isnan(av) and np.isnan(bv):
            return
        assert av == bv, f"float mismatch at {path}: {av!r} != {bv!r}"
        return
    assert a == b, f"mismatch at {path}: {a!r} != {b!r}"


def make_detection(module, rng: np.random.Generator, index: int):
    center = np.array([80.0 + index * 11.0, 90.0 + index * 7.0])
    template = np.array(
        [[18, 0], [11, -5], [11, 5], [4, 0], [-7, -6], [-7, 6], [-18, 0]],
        dtype=np.float64,
    )
    points = template + center + rng.normal(0.0, 0.5, size=(7, 2))
    confidence = rng.uniform(0.05, 0.99, size=7).astype(np.float64)
    if index % 4 == 0:
        points[0] = np.nan
        confidence[0] = 0.0
    box = np.array(
        [center[0] - 24, center[1] - 14, center[0] + 24, center[1] + 14],
        dtype=np.float64,
    )
    return module.Detection(
        raw_track_id=index,
        keypoints_px=points,
        keypoint_conf=confidence,
        bbox_xyxy=box,
        box_conf=0.8,
        prefer_bbox_center=bool(index % 5 == 0),
    )


def test_detection_and_appearance(old_base, new_base) -> dict[str, float]:
    rng = np.random.default_rng(138401)
    frame = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
    old_detections = [make_detection(old_base, rng, i) for i in range(20)]
    rng = np.random.default_rng(138401)
    _ = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
    new_detections = [make_detection(new_base, rng, i) for i in range(20)]

    # Internal memoization must not change the public dataclass schema or asdict output.
    assert tuple(old_base.Detection.__dataclass_fields__) == tuple(
        new_base.Detection.__dataclass_fields__
    )
    assert list(asdict(old_detections[0])) == list(asdict(new_detections[0]))

    for old_det, new_det in zip(old_detections, new_detections):
        np.testing.assert_array_equal(old_det.center_px, new_det.center_px)
        assert old_det.body_length_px == new_det.body_length_px
        new_det.refresh_derived_geometry_cache()
        np.testing.assert_array_equal(old_det.center_px, new_det.center_px)
        assert old_det.body_length_px == new_det.body_length_px

    restored = pickle.loads(pickle.dumps(new_detections[1], protocol=pickle.HIGHEST_PROTOCOL))
    assert restored._cached_center_px is None
    assert restored._cached_body_length_px is None
    np.testing.assert_array_equal(restored.center_px, old_detections[1].center_px)
    assert restored.body_length_px == old_detections[1].body_length_px

    old_mutated = make_detection(old_base, np.random.default_rng(99), 3)
    new_mutated = make_detection(new_base, np.random.default_rng(99), 3)
    new_mutated.refresh_derived_geometry_cache()
    old_mutated.keypoints_px[:, 0] += 2.0
    new_mutated.keypoints_px[:, 0] += 2.0
    new_mutated.invalidate_derived_geometry_cache()
    np.testing.assert_array_equal(old_mutated.center_px, new_mutated.center_px)
    assert old_mutated.body_length_px == new_mutated.body_length_px

    old_enriched = old_base.enrich_detections_with_appearance(frame, old_detections, {})
    new_enriched = new_base.enrich_detections_with_appearance(frame, new_detections, {})
    for index, (old_det, new_det) in enumerate(zip(old_enriched, new_enriched)):
        # OpenCV histogram/statistical kernels may differ by a few float32 ULPs
        # across calls/builds.  Keep this tolerance isolated to appearance
        # descriptors; identity/behavior decisions remain exact elsewhere.
        np.testing.assert_allclose(
            np.asarray(old_det.appearance_feature),
            np.asarray(new_det.appearance_feature),
            rtol=2e-7,
            atol=2e-7,
            err_msg=f"appearance[{index}]",
        )
        assert_same_value(old_det.normalized_pose, new_det.normalized_pose, f"pose[{index}]")
        assert_same_value(old_det.anchor_feature, new_det.anchor_feature, f"anchor[{index}]")
        assert_same_value(old_det.heading_vector, new_det.heading_vector, f"heading[{index}]")
        assert old_det.brightness_score == new_det.brightness_score
        assert old_det.white_score == new_det.white_score

    repeats = 1000
    started = time.perf_counter()
    for _ in range(repeats):
        for det in old_enriched:
            _ = det.center_px
            _ = det.body_length_px
    old_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        for det in new_enriched:
            _ = det.center_px
            _ = det.body_length_px
    new_seconds = time.perf_counter() - started
    return {
        "derived_geometry_old_seconds": old_seconds,
        "derived_geometry_new_seconds": new_seconds,
        "derived_geometry_speedup": old_seconds / max(new_seconds, 1e-12),
    }


def make_observation(module, frame: int, logical_id: int) -> Any:
    angle = 0.03 * frame + 0.15 * logical_id
    center = np.array(
        [
            30.0 + logical_id * 3.2 + frame * (0.08 + logical_id * 0.001),
            40.0 + (logical_id % 5) * 4.5 + np.sin(angle),
        ],
        dtype=np.float64,
    )
    heading = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    velocity = np.array([0.08 + logical_id * 0.001, np.cos(angle) * 0.03], dtype=np.float64)
    keypoints = np.stack(
        [center + heading * offset for offset in (2.2, 1.6, 1.6, 0.5, -1.0, -1.0, -2.3)],
        axis=0,
    )
    keypoints[1, 1] -= 0.2
    keypoints[2, 1] += 0.2
    confidence = np.full(7, 0.85, dtype=np.float64)
    bbox = np.array([center[0] - 3, center[1] - 2, center[0] + 3, center[1] + 2])
    return module.MouseObservation(
        frame=frame,
        logical_id=logical_id,
        raw_track_id=logical_id,
        keypoints_px=keypoints * 10,
        keypoints_cm=keypoints,
        keypoint_conf=confidence,
        bbox_xyxy=bbox * 10,
        box_conf=0.9,
        center_cm=center,
        head_cm=np.mean(keypoints[:3], axis=0),
        rear_cm=np.mean(keypoints[4:], axis=0),
        heading=heading,
        velocity_cm_s=velocity,
        speed_cm_s=float(np.linalg.norm(velocity)),
        acceleration_cm_s2=0.1,
        angular_speed_deg_s=2.0,
        nose_speed_cm_s=float(np.linalg.norm(velocity) * 1.2),
        body_length_cm=float(np.linalg.norm(keypoints[0] - keypoints[6])),
    )


def behavior_config() -> dict[str, Any]:
    common_chase = {
        "max_distance_cm": 20.0,
        "actor_min_speed_cm_s": 0.01,
        "target_min_speed_cm_s": 0.01,
        "direction_similarity_min": -1.0,
        "pursuit_alignment_min": -1.0,
        "target_escape_alignment_min": -1.0,
        "trajectory_correlation_min": -1.0,
        "require_all_conditions": False,
        "candidate_score_min": 1,
    }
    common_attack = {
        "contact_distance_cm": 4.0,
        "actor_lunge_speed_cm_s": 0.1,
        "rapid_closing_distance_cm": 0.01,
        "target_escape_speed_cm_s": 0.01,
        "target_escape_alignment_min": -1.0,
        "target_turn_angle_deg": 1.0,
        "repeated_contact_count": 1,
        "head_motion_speed_cm_s": 0.01,
        "head_to_center_speed_ratio": 0.1,
        "attack_pursuit_alignment_min": -1.0,
        "stationary_fight_distance_cm": 4.0,
        "stationary_fight_max_center_speed_cm_s": 100.0,
        "stationary_fight_min_angular_speed_deg_s": 1.0,
        "min_dynamic_evidence": 1,
        "repeated_contact_window_seconds": 1.0,
    }
    strong_chase = dict(common_chase)
    strong_chase.update({"max_distance_cm": 10.0, "candidate_score_min": 2})
    strong_attack = dict(common_attack)
    strong_attack.update(
        {"contact_distance_cm": 3.0, "stationary_fight_distance_cm": 3.0, "min_dynamic_evidence": 2}
    )
    return {
        "features": {"history_seconds": 1.0, "response_lookback_seconds": 0.3},
        "chase": {"weak": common_chase, "strong": strong_chase},
        "attack": {"weak": common_attack, "strong": strong_attack},
    }


def base_behavior_config() -> dict[str, Any]:
    cfg = behavior_config()
    chase = dict(cfg["chase"]["weak"])
    chase["high_confidence_score_min"] = 3
    return {
        "features": dict(cfg["features"]),
        "chase": chase,
        "attack": dict(cfg["attack"]["weak"]),
    }


def test_cluster_context_cache(old_base, new_base) -> None:
    """Verify the single-frame evidence memoization is output-identical."""

    class Assigner:
        def __init__(self, tracks: dict[int, Any]) -> None:
            self.tracks = tracks

        @staticmethod
        def _prediction(track: Any, frame: int) -> np.ndarray:
            dt = max(int(frame) - int(track.last_frame), 0)
            return np.asarray(track.last_center_px, dtype=np.float64) + np.asarray(
                track.velocity_px_per_frame, dtype=np.float64
            ) * min(dt, 12)

    def tracks_for(module: Any) -> dict[int, Any]:
        tracks: dict[int, Any] = {}
        for logical_id, center_x in ((1, 100.0), (2, 110.0)):
            center = np.array([center_x, 100.0], dtype=np.float64)
            tracks[logical_id] = module.IdentityTrack(
                logical_id=logical_id,
                last_center_px=center,
                velocity_px_per_frame=np.array([0.5, 0.0], dtype=np.float64),
                last_frame=9,
                raw_track_id=logical_id,
                body_length_px=20.0,
                last_bbox_xyxy=np.array(
                    [center_x - 12.0, 92.0, center_x + 12.0, 108.0],
                    dtype=np.float64,
                ),
            )
        return tracks

    config = {
        "enabled": True,
        "enter_distance_body_lengths": 1.2,
        "min_entry_separation_body_lengths": 0.3,
        "release_distance_body_lengths": 1.65,
        "release_stable_frames": 5,
        "recovery_cooldown_frames": 3,
    }
    old_manager = old_base.OcclusionClusterManager(config)
    new_manager = new_base.OcclusionClusterManager(config)
    old_calls = 0
    new_calls = 0

    # Keep two independent counters while returning exactly the same evidence.
    def evidence_payload() -> dict[str, Any]:
        return {
            "observed_count": 1,
            "observed_indices": [],
            "deficit": True,
            "merged_like": True,
            "merged_member_count": 2,
            "max_iou": 0.25,
            "max_det_area_ratio": 1.8,
            "locally_visible_count": 1,
        }

    def old_evidence(self: Any, members: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal old_calls
        old_calls += 1
        return evidence_payload()

    def new_evidence(self: Any, members: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal new_calls
        new_calls += 1
        return evidence_payload()

    old_manager._member_detection_evidence = types.MethodType(old_evidence, old_manager)
    new_manager._member_detection_evidence = types.MethodType(new_evidence, new_manager)
    old_context = old_manager.build_context(Assigner(tracks_for(old_base)), [], 10, (240, 320))
    new_context = new_manager.build_context(Assigner(tracks_for(new_base)), [], 10, (240, 320))
    assert_same_value(old_context, new_context, "cluster_context")
    assert_same_value(old_manager.states, new_manager.states, "cluster_states")
    assert_same_value(old_manager.debug_rows, new_manager.debug_rows, "cluster_debug")
    assert old_calls == 2, f"reference path should recompute twice, got {old_calls}"
    assert new_calls == 1, f"optimized path should reuse evidence, got {new_calls}"


def test_pair_features(old_base, new_base, old_main, new_main) -> dict[str, float]:
    fps = 30.0
    cfg = behavior_config()
    old_history = old_base.ObservationHistory(max_frames=40)
    new_history = new_base.ObservationHistory(max_frames=40)
    for frame in range(35):
        for logical_id in range(20):
            old_history.add(make_observation(old_base, frame, logical_id))
            new_history.add(make_observation(new_base, frame, logical_id))

    old_computer = old_main.PairFeatureComputer(fps, cfg)
    new_computer = new_main.PairFeatureComputer(fps, cfg)
    old_current = [old_history.previous(i) for i in range(20)]
    new_current = [new_history.previous(i) for i in range(20)]
    assert all(item is not None for item in old_current + new_current)

    old_outputs: list[Any] = []
    new_outputs: list[Any] = []
    started = time.perf_counter()
    for _ in range(20):
        old_outputs.clear()
        for i in range(20):
            for j in range(i + 1, 20):
                old_outputs.append(
                    old_computer.compute(old_current[i], old_current[j], old_history, 1)
                )
                old_outputs.append(
                    old_computer.compute(old_current[j], old_current[i], old_history, 1)
                )
    old_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(20):
        new_outputs.clear()
        for i in range(20):
            for j in range(i + 1, 20):
                new_outputs.append(
                    new_computer.compute(new_current[i], new_current[j], new_history, 1)
                )
                new_outputs.append(
                    new_computer.compute(new_current[j], new_current[i], new_history, 1)
                )
    new_seconds = time.perf_counter() - started

    for index, (old_value, new_value) in enumerate(zip(old_outputs, new_outputs)):
        assert_same_value(old_value, new_value, f"pair_feature[{index}]")

    old_base_computer = old_base.PairFeatureComputer(fps, base_behavior_config())
    new_base_computer = new_base.PairFeatureComputer(fps, base_behavior_config())
    for i in range(20):
        for j in range(i + 1, 20):
            assert_same_value(
                old_base_computer.compute(old_current[i], old_current[j], old_history, 1),
                new_base_computer.compute(new_current[i], new_current[j], new_history, 1),
                f"base_pair_feature[{i},{j}]",
            )
            assert_same_value(
                old_base_computer.compute(old_current[j], old_current[i], old_history, 1),
                new_base_computer.compute(new_current[j], new_current[i], new_history, 1),
                f"base_pair_feature[{j},{i}]",
            )

    for logical_id in range(20):
        old_history.add(make_observation(old_base, 35, logical_id))
        new_history.add(make_observation(new_base, 35, logical_id))
    for i in range(20):
        for j in range(i + 1, 20):
            assert_same_value(
                old_computer.compute(
                    old_history.previous(i), old_history.previous(j), old_history, 1
                ),
                new_computer.compute(
                    new_history.previous(i), new_history.previous(j), new_history, 1
                ),
                f"next_frame_pair_feature[{i},{j}]",
            )

    return {
        "pair_features_old_seconds": old_seconds,
        "pair_features_new_seconds": new_seconds,
        "pair_features_speedup": old_seconds / max(new_seconds, 1e-12),
    }


def test_identity_cascade_equivalence(new_base) -> None:
    """Sparse candidate evaluation must be exactly equal to dense NumPy cost."""
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
    rng = np.random.default_rng(1421)
    detections = [make_detection(new_base, rng, i) for i in range(20)]
    assigner.tracks = {}
    for logical_id, det in enumerate(detections):
        shift_x = float(logical_id * 100.0)
        det.keypoints_px = np.asarray(det.keypoints_px, dtype=np.float64) + np.array([shift_x, 0.0])
        det.bbox_xyxy = np.asarray(det.bbox_xyxy, dtype=np.float64) + np.array(
            [shift_x, 0.0, shift_x, 0.0]
        )
        det.invalidate_derived_geometry_cache()
        det.refresh_derived_geometry_cache()
        assigner.tracks[logical_id] = new_base.IdentityTrack(
            logical_id=logical_id,
            last_center_px=det.center_px.copy(),
            velocity_px_per_frame=np.zeros(2, dtype=np.float64),
            last_frame=0,
            raw_track_id=logical_id,
            body_length_px=det.body_length_px,
            normalized_pose=det.normalized_pose,
            anchor_feature=det.anchor_feature,
            heading_vector=det.heading_vector,
            last_keypoints_px=np.asarray(det.keypoints_px, dtype=np.float64).copy(),
            last_keypoint_conf=np.asarray(det.keypoint_conf, dtype=np.float64).copy(),
            last_bbox_xyxy=np.asarray(det.bbox_xyxy, dtype=np.float64).copy(),
            last_box_conf=float(det.box_conf),
        )
        assigner.kpt_missing[logical_id] = 0

    track_ids = list(range(20))
    assigner.identity_cascade_enabled = True
    assigner.identity_cascade_min_cells = 1
    assigner.identity_cascade_sparse_density = 1.0
    sparse = assigner._base_cost_matrix_numpy(track_ids, detections, 1)
    assert assigner.last_base_cost_mode == "cascade_sparse_numpy"
    assert assigner.last_fast_gate_candidate_count < assigner.last_fast_gate_total_count

    assigner.identity_cascade_enabled = False
    dense = assigner._base_cost_matrix_numpy(track_ids, detections, 1)
    assert assigner.last_base_cost_mode == "numpy_dense"
    np.testing.assert_array_equal(sparse, dense)


def test_pair_dataframe_store(old_main, new_main) -> dict[str, float]:
    rng = np.random.default_rng(138402)
    keys = [f"{i}_{j}" for i in range(20) for j in range(i + 1, 20)]
    row_count = 190_000
    table = pd.DataFrame(
        {
            "frame": np.arange(row_count, dtype=np.int64) // len(keys),
            "pair_key": np.resize(np.asarray(keys, dtype=object), row_count),
            "value": rng.normal(size=row_count),
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pairs.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        old_store = old_main.PairDataFrameStore(path)
        new_store = new_main.PairDataFrameStore(path)
        assert old_store.pair_keys() == new_store.pair_keys()

        started = time.perf_counter()
        old_tables = [old_store.read_pair(key) for key in keys]
        old_seconds = time.perf_counter() - started
        started = time.perf_counter()
        new_tables = [new_store.read_pair(key) for key in keys]
        new_seconds = time.perf_counter() - started
        for old_table, new_table in zip(old_tables, new_tables):
            pd.testing.assert_frame_equal(old_table, new_table, check_exact=True)
        pd.testing.assert_frame_equal(
            old_store.read_pair("missing_pair"),
            new_store.read_pair("missing_pair"),
            check_exact=True,
        )
    return {
        "pair_store_old_seconds": old_seconds,
        "pair_store_new_seconds": new_seconds,
        "pair_store_speedup": old_seconds / max(new_seconds, 1e-12),
    }


def test_frame_records_and_sqlite(old_base, new_base, old_main, new_main) -> None:
    fps = 30.0
    cfg = behavior_config()
    cfg["wall_jump"] = {"enabled": False}
    old_history = old_base.ObservationHistory(max_frames=40)
    new_history = new_base.ObservationHistory(max_frames=40)
    for frame in range(12):
        for logical_id in range(6):
            old_history.add(make_observation(old_base, frame, logical_id))
            new_history.add(make_observation(new_base, frame, logical_id))
    old_observations = [old_history.previous(i) for i in range(6)]
    new_observations = [new_history.previous(i) for i in range(6)]

    old_records = old_main._compute_behavior_frame_records(
        frame_idx=11,
        fps=fps,
        width=640,
        height=360,
        observations=old_observations,
        cluster_context={},
        transformer=old_main._Stage3TransformerStub("fixed", 0.1),
        config=cfg,
        history=old_history,
        feature_computer=old_main.PairFeatureComputer(fps, cfg),
        individual_behavior_gate=old_main.IndividualBehaviorGate(fps, cfg),
        contact_tracker=old_base.PairContactTracker(fps, 1.0),
        pair_compute_mode="numpy",
    )
    new_records = new_main._compute_behavior_frame_records(
        frame_idx=11,
        fps=fps,
        width=640,
        height=360,
        observations=new_observations,
        cluster_context={},
        transformer=new_main._Stage3TransformerStub("fixed", 0.1),
        config=cfg,
        history=new_history,
        feature_computer=new_main.PairFeatureComputer(fps, cfg),
        individual_behavior_gate=new_main.IndividualBehaviorGate(fps, cfg),
        contact_tracker=new_base.PairContactTracker(fps, 1.0),
        pair_compute_mode="numpy",
    )
    assert len(old_records) == len(new_records)
    for index, (old_record, new_record) in enumerate(zip(old_records, new_records)):
        # v1.43 intentionally adds standard-engine feature/quality columns.
        # The regression contract here is that every legacy field remains
        # numerically identical before the new temporal engine is applied.
        assert set(old_record).issubset(set(new_record)), (
            f"legacy schema missing at frame_record[{index}]"
        )
        for key in old_record:
            assert_same_value(old_record[key], new_record[key], f"frame_record[{index}].{key}")

    with tempfile.TemporaryDirectory() as directory:
        old_store = old_main.PairSQLiteStore(Path(directory) / "old.sqlite", batch_size=100)
        new_store = new_main.PairSQLiteStore(Path(directory) / "new.sqlite", batch_size=100)
        for record in old_records:
            old_store.add(record)
        new_store.add_many(new_records)
        old_store.finalize()
        new_store.finalize()
        assert old_store.pair_keys() == new_store.pair_keys()
        for key in old_store.pair_keys():
            old_table = old_store.read_pair(key)
            new_table = new_store.read_pair(key)
            assert set(old_table.columns).issubset(set(new_table.columns))
            pd.testing.assert_frame_equal(
                old_table, new_table.loc[:, old_table.columns], check_exact=True
            )
        old_store.close()
        new_store.close()


def test_checkpoint_cache_exclusion(new_base, new_main) -> None:
    history = new_base.ObservationHistory(max_frames=20)
    for frame in range(12):
        for logical_id in range(4):
            history.add(make_observation(new_base, frame, logical_id))
    assert history.get_window(0, 10)
    assert history.near_frame(0, 5) is not None
    assert history._window_cache and history._near_cache

    buffer = io.BytesIO()
    new_main._CheckpointPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(history)
    buffer.seek(0)
    restored = pickle.load(buffer)
    assert restored._window_cache == {}
    assert restored._near_cache == {}
    assert_same_value(history.get(0), restored.get(0), "checkpoint_history")
    assert_same_value(history.near_frame(0, 5), restored.near_frame(0, 5), "checkpoint_near")


def main() -> None:
    old_base, new_base, old_main, new_main = load_modules()
    metrics: dict[str, float] = {}
    metrics.update(test_detection_and_appearance(old_base, new_base))
    test_cluster_context_cache(old_base, new_base)
    test_identity_cascade_equivalence(new_base)
    metrics.update(test_pair_features(old_base, new_base, old_main, new_main))
    metrics.update(test_pair_dataframe_store(old_main, new_main))
    test_frame_records_and_sqlite(old_base, new_base, old_main, new_main)
    test_checkpoint_cache_exclusion(new_base, new_main)
    print("REGRESSION: PASS")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
