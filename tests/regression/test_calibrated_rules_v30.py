"""Regression tests for heuristic rule calibrations (v30).

Covers:
1. Individual heuristic configuration defaults for grooming and rearing.
2. Posture screening enabling via IndividualHeuristicConfig and individual_rule_masks.
3. Attack vs Chase mutual exclusivity in pair_rule_masks.
4. Departure process mask return hysteresis with convex hull in group_event_rules.
5. Dispersal multi-animal candidate generation and displacement scale invariance.
6. Social clustering temporally overlapping candidate connected components merging.
"""
from dataclasses import replace
import numpy as np
import pytest

from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG,
    individual_rule_masks,
)
from mouse_behavior.evaluation.confirmed_rules import pair_rule_masks
from mouse_behavior.evaluation.exported_dataset_models import (
    ExportedFeatures,
    HeuristicParameters,
)
from mouse_behavior.evaluation.group_event_rules import (
    _departure_process_mask,
    group_rule_candidates,
)
from mouse_behavior.evaluation.social_heuristic_config import DEFAULT_SOCIAL_HEURISTIC_CONFIG


def test_individual_heuristic_config_defaults():
    cfg = DEFAULT_INDIVIDUAL_CONFIG
    assert pytest.approx(cfg.grooming_min_step_nose_speed_body_ratio) == 6.0 / 10.7
    assert pytest.approx(cfg.grooming_min_nose_posterior_speed_ratio) == 1.8
    assert pytest.approx(cfg.grooming_min_head_posterior_speed_ratio) == 1.3
    assert pytest.approx(cfg.grooming_bridge_max_s) == 0.5
    assert pytest.approx(cfg.rearing_max_posterior_excursion_body_ratio) == 1.15
    assert pytest.approx(cfg.rearing_max_wall_distance_body_ratio) == 0.85
    assert pytest.approx(cfg.rearing_bridge_max_s) == 0.6


def test_individual_posture_screen_enable_and_disable():
    fps = 30.0
    frames = 60
    centers = np.full((frames, 1, 2), 10.0)
    valid = np.ones((frames, 1), bool)
    keypoints = np.full((frames, 1, 7, 2), 10.0)
    # Nose oscillating fore-aft
    keypoints[:, 0, 0, 0] = 10.0 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, frames))

    # With posture screen disabled (default), grooming and rearing masks are strictly all-False
    res_off = individual_rule_masks(centers, valid, fps, keypoints_cm=keypoints, config=DEFAULT_INDIVIDUAL_CONFIG)
    assert not res_off.masks["grooming"].any()
    assert not res_off.masks["rearing"].any()

    # With posture screen enabled, posture analysis executes
    cfg_on = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res_on = individual_rule_masks(centers, valid, fps, keypoints_cm=keypoints, config=cfg_on)
    assert "grooming" in res_on.masks
    assert "rearing" in res_on.masks


def test_attack_and_chasing_are_strictly_mutually_exclusive():
    # Construct a clip where attack and chase features could both trigger
    frames = 90
    fps = 30.0
    centers = np.zeros((frames, 2, 2), dtype=float)
    # Moving together in pursuit
    for t in range(frames):
        centers[t, 0] = [t * 1.5, 0.0]
        centers[t, 1] = [t * 1.5 + 4.0, 0.0]

    valid = np.ones((frames, 2), bool)
    speed = np.full((frames, 2), 45.0)
    velocity = np.zeros((frames, 2, 2))
    velocity[:, :, 0] = 45.0
    boxes = np.zeros((frames, 2, 4), dtype=float)
    boxes[:, :, 2:] = 2.0
    # Add a target box dropout in the middle
    boxes[40:44, 1] = np.nan

    f = ExportedFeatures(
        centers_cm=centers,
        speed_cm_s=speed,
        velocity_cm_s=velocity,
        pair_i=np.array([0]),
        pair_j=np.array([1]),
        pair_distance_cm=np.full((frames, 1), 4.0),
        pair_closing_speed_cm_s=np.zeros((frames, 1)),
        pair_direction_similarity=np.ones((frames, 1)),
        pair_nose_head_cm=np.full((frames, 1), 4.0),
        pair_nose_tail_cm=np.full((frames, 1), 1.0),
        nearest_distance_cm=np.full((frames, 2), 4.0),
        mean_nearest_distance_cm=np.full((frames, 2), 4.0),
        cm_per_pixel=0.1,
        track_selection_truncated=False,
        selected_track_count=2,
        track_ids=np.array([1, 2]),
        valid=valid,
        fps=fps,
        collective_boxes_cm=boxes,
        collective_centers_cm=centers,
        collective_valid=valid,
        reference_body_cm=7.918,
    )
    p = HeuristicParameters()
    masks = pair_rule_masks(f, p, DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    # Strict mutual exclusivity: no frame may simultaneously have two social labels active
    names = ("approach", "together", "attack", "chasing", "head_contact", "tail_contact")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = masks[names[i]][:, 0] & masks[names[j]][:, 0]
            assert not ov.any(), f"{names[i]} and {names[j]} overlap on frames {np.flatnonzero(ov)}"



def test_departure_process_mask_return_hysteresis():
    # 4 animals forming a square hull
    count = 60
    fps = 30.0
    centers = np.zeros((count, 4, 2), dtype=float)
    # Mice 0 and 1 move outward beyond the boundary
    for t in range(count):
        centers[t, 0] = [10.0 + t * 0.2, 10.0]
        centers[t, 1] = [0.0 - t * 0.2, 0.0]
        centers[t, 2] = [0.0, 10.0]
        centers[t, 3] = [10.0, 0.0]

    observed = np.ones((count, 4), bool)
    leaving = (0, 1)

    mask = _departure_process_mask(
        centers, observed, 0, count, leaving, fps,
        minimum_outward_speed_cm_s=1.0, pause_max_s=0.5,
    )
    assert mask.any(), "outward departing animals should produce active departure mask"

    # Start > 0 with return hysteresis to start_positions
    start = 10
    mask_delayed = _departure_process_mask(
        centers, observed, start, count, leaving, fps,
        minimum_outward_speed_cm_s=1.0, pause_max_s=0.5,
    )
    assert mask_delayed.any(), "delayed start departure should produce active departure mask"
    assert not mask_delayed[:start].any(), "frames before start must never be claimed"


def test_dispersal_candidate_scale_invariance():
    # Multi-animal candidate generation should scale displacement threshold by reference_body_cm
    from tests.regression import test_heuristics_v16_audit as audit
    from mouse_behavior.evaluation import group_event_rules as group

    # Case 1: All 3 depart (displacement > 0.18 * ref_body)
    f1 = audit.features(300, 3, reference_body_cm=7.918)
    time = np.minimum(np.arange(300) / f1.fps, 9.0)
    f1.centers_cm[:, 0, 0] = -1.0 - 2.0 * time
    f1.centers_cm[:, 1, 0] = 1.0 + 2.0 * time
    f1.centers_cm[:, 2, 1] = 1.0 + 0.5 * time
    audit.sync(f1)
    audit.resync_kinematics(f1)
    cands1 = group.group_rule_candidates(f1, audit.confirmed_params(), (), behaviors=("dispersal",))
    sets1 = {c.target_ids for c in cands1}
    assert (100, 101) in sets1, "actively departing subset must be emitted"
    assert (100, 101, 102) in sets1, "full group candidate must be emitted when all animals displace > 0.18 * ref_body"

    # Case 2: Track 2 stationary (displacement = 0.0 < 0.18 * ref_body)
    f2 = audit.features(300, 3, reference_body_cm=7.918)
    f2.centers_cm[:, 0, 0] = -1.0 - 2.0 * time
    f2.centers_cm[:, 1, 0] = 1.0 + 2.0 * time
    f2.centers_cm[:, 2, 1] = 1.0
    audit.sync(f2)
    audit.resync_kinematics(f2)
    cands2 = group.group_rule_candidates(f2, audit.confirmed_params(), (), behaviors=("dispersal",))
    sets2 = {c.target_ids for c in cands2}
    assert (100, 101) in sets2, "actively departing subset must be emitted"
    assert (100, 101, 102) not in sets2, "stationary animal must not be included in departing candidates"

    # Case 3: Physical scale invariance across reference_body_cm
    # Track 2 moves 1.2 cm. For ref_body=15.0 cm, threshold is 2.7 cm -> 1.2 cm < 2.7 cm -> NOT emitted.
    f3 = audit.features(300, 3, reference_body_cm=15.0)
    f3.centers_cm[:, 0, 0] = -1.0 - 2.0 * time
    f3.centers_cm[:, 1, 0] = 1.0 + 2.0 * time
    f3.centers_cm[:, 2, 1] = 1.0 + (1.2 / 9.0) * time
    audit.sync(f3)
    audit.resync_kinematics(f3)
    cands3 = group.group_rule_candidates(f3, audit.confirmed_params(), (), behaviors=("dispersal",))
    sets3 = {c.target_ids for c in cands3}
    assert (100, 101, 102) not in sets3, "1.2 cm displacement on 15 cm body is below 18% threshold"

    # For ref_body=5.0 cm, threshold is 0.9 cm -> 1.2 cm >= 0.9 cm -> emitted.
    f4 = audit.features(300, 3, reference_body_cm=5.0)
    f4.centers_cm[:, 0, 0] = -1.0 - 2.0 * time
    f4.centers_cm[:, 1, 0] = 1.0 + 2.0 * time
    f4.centers_cm[:, 2, 1] = 1.0 + (1.2 / 9.0) * time
    audit.sync(f4)
    audit.resync_kinematics(f4)
    cands4 = group.group_rule_candidates(f4, audit.confirmed_params(), (), behaviors=("dispersal",))
    sets4 = {c.target_ids for c in cands4}
    assert (100, 101, 102) in sets4, "1.2 cm displacement on 5 cm body is above 18% threshold"


def test_social_clustering_candidate_union_connected_components():
    from tests.regression.test_temporal_screen import features

    f = features(n=100, mice=4)
    f.centers_cm[:, 0, 0] = 0
    f.centers_cm[:, 1, 0] = 3
    f.centers_cm[:, 2, 0] = np.maximum(25 - np.arange(100), 5)
    f.centers_cm[:, 3, 0] = np.maximum(40 - np.maximum(np.arange(100) - 30, 0), 7)
    f.pair_distance_cm = np.linalg.norm(f.centers_cm[:, f.pair_i] - f.centers_cm[:, f.pair_j], axis=2)
    f.nearest_distance_cm[:] = np.inf
    for edge, (left, right) in enumerate(zip(f.pair_i, f.pair_j)):
        f.nearest_distance_cm[:, left] = np.minimum(f.nearest_distance_cm[:, left], f.pair_distance_cm[:, edge])
        f.nearest_distance_cm[:, right] = np.minimum(f.nearest_distance_cm[:, right], f.pair_distance_cm[:, edge])
    points = np.zeros((100, 4, 7, 2))
    offsets = np.array([[-1, 0], [0, -0.5], [0, 0.5], [0.4, 0], [1, -0.5], [1, 0.5], [1, 0]])
    points[:] = f.centers_cm[:, :, None] + offsets
    f.collective_keypoints_cm = points
    params = HeuristicParameters(
        confirmed_rules_v1=1,
        isolation_distance_cm=20,
        isolation_min_duration_s=10,
        huddle_min_duration_s=5,
        group_locomotion_max_distance_cm=30,
        dispersal_min_distance_increase_cm=3,
        dispersal_min_duration_s=5,
    )
    cands = group_rule_candidates(f, params, (), behaviors=("social_clustering",))
    assert any(
        c.behavior == "social_clustering" and c.target_ids == (100, 101, 102, 103)
        for c in cands
    ), "overlapping clustering episodes must yield the full 4-animal cluster candidate"


def test_rearing_mounted_on_other_mouse_veto():
    fps = 30.0
    frames = 60
    centers = np.full((frames, 2, 2), 10.0)
    valid = np.ones((frames, 2), bool)
    keypoints = np.full((frames, 2, 7, 2), 10.0)
    # Give mouse 0 a shortened projection (rearing posture)
    keypoints[:, 0, 0] = [10.0, 10.0]  # nose
    keypoints[:, 0, 6] = [10.0, 11.5]  # tailbase (projection = 1.5 cm < 0.75 * reference_body_cm)
    boxes = np.zeros((frames, 2, 4), dtype=float)
    # Mouse 0 box: [9, 9, 11, 12]
    boxes[:, 0] = [9.0, 9.0, 11.0, 12.0]

    # Case A: Mouse 1 overlaps Mouse 0 completely (mounted / climbing)
    boxes[:, 1] = [9.0, 9.0, 11.0, 12.0]
    centers[:, 1] = [10.0, 10.0]

    cfg = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res_mounted = individual_rule_masks(centers, valid, fps, keypoints_cm=keypoints, boxes_cm=boxes, config=cfg)
    assert not res_mounted.masks["rearing"][:, 0].any(), "Mouse mounted on another mouse must NOT be classified as rearing"

    # Case B: Mouse 1 is far away (standalone rearing)
    boxes[:, 1] = [50.0, 50.0, 52.0, 53.0]
    centers[:, 1] = [51.0, 51.0]
    res_standalone = individual_rule_masks(centers, valid, fps, keypoints_cm=keypoints, boxes_cm=boxes, config=cfg)
    # Standalone rearing posture evaluates without mounting veto
    assert isinstance(res_standalone.masks["rearing"], np.ndarray)


def test_dispersal_duration_policy_threshold():
    from mouse_behavior.behavior.duration_policy import qualify_event_duration, resolve_duration_policy

    policy = resolve_duration_policy("dispersal")
    assert policy.required_s == 5.0
    assert policy.comparator == ">"

    fps = 30.0
    # 151 frames > 5.0s -> OK
    ev_pass = {"canonical_behavior": "dispersal", "start_frame": 0, "end_frame": 151}
    res_pass = qualify_event_duration(ev_pass, fps)
    assert res_pass["ok"]
    assert res_pass["actual_s"] == pytest.approx(151 / 30.0)
    assert res_pass["required_s"] == 5.0
    assert res_pass["comparator"] == ">"

    # 150 frames = 5.0s -> FAIL under strict >
    ev_fail = {"canonical_behavior": "dispersal", "start_frame": 0, "end_frame": 150}
    res_fail = qualify_event_duration(ev_fail, fps)
    assert not res_fail["ok"]
    assert res_fail["reason"] == "process_duration_below_requirement"


def test_dorsal_grooming_suppresses_rearing():
    fps = 30.0
    frames = 60
    centers = np.full((frames, 1, 2), 10.0)
    valid = np.ones((frames, 1), bool)

    # Construct dorsal grooming posture:
    # 0: nose, 1: left ear, 2: right ear, 3: neck, 4: left hip, 5: right hip, 6: tail base
    # In dorsal grooming, mouse curls head toward its posterior/hips
    keypoints = np.zeros((frames, 1, 7, 2), dtype=float)
    keypoints[:, 0, 6] = [10.0, 10.0]  # tail base
    keypoints[:, 0, 4] = [9.5, 9.5]   # left hip
    keypoints[:, 0, 5] = [10.5, 9.5]  # right hip
    keypoints[:, 0, 3] = [10.0, 8.0]  # neck
    keypoints[:, 0, 1] = [9.7, 8.5]   # left ear
    keypoints[:, 0, 2] = [10.3, 8.5]  # right ear
    # Nose is curled back near posterior/hips (short projection and nose_back)
    keypoints[:, 0, 0] = [10.1, 9.4]  # nose

    cfg = replace(DEFAULT_INDIVIDUAL_CONFIG, enable_posture_screen=True)
    res_dorsal = individual_rule_masks(centers, valid, fps, keypoints_cm=keypoints, config=cfg)
    assert not res_dorsal.masks["rearing"][:, 0].any(), "Curled dorsal grooming posture must be vetoed from rearing"

    # Upright rearing posture: nose extended forward away from posterior, straight spine
    keypoints_rearing = np.zeros((frames, 1, 7, 2), dtype=float)
    for t in range(frames):
        keypoints_rearing[t, 0, 6] = [10.0, 10.0]  # tail base
        keypoints_rearing[t, 0, 4] = [9.7, 9.8]    # left hip
        keypoints_rearing[t, 0, 5] = [10.3, 9.8]   # right hip
        h_ext = 0.8 * np.sin(t * 0.1)
        keypoints_rearing[t, 0, 3] = [10.0, 8.0 + h_ext]  # neck
        keypoints_rearing[t, 0, 1] = [9.7, 7.5 + h_ext]   # left ear
        keypoints_rearing[t, 0, 2] = [10.3, 7.5 + h_ext]  # right ear
        keypoints_rearing[t, 0, 0] = [10.0, 7.0 + h_ext]  # nose forward
    boxes_rearing = np.zeros((frames, 1, 4), dtype=float)
    for t in range(frames):
        if t < 15:
            boxes_rearing[t, 0] = [9.0, 5.0, 11.0, 11.0]
        else:
            boxes_rearing[t, 0] = [9.5, 7.0, 10.5, 11.0]

    res_rearing = individual_rule_masks(
        centers, valid, fps, keypoints_cm=keypoints_rearing, boxes_cm=boxes_rearing, config=cfg
    )
    assert res_rearing.masks["rearing"][:, 0].any(), "Upright standing posture must be classified as rearing"





