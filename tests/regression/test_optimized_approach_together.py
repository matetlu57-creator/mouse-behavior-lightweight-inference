"""Regression tests for optimized Approach and Together heuristics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mouse_behavior.config import load_config
from mouse_behavior.evaluation import confirmed_rules as rules
from mouse_behavior.evaluation.exported_dataset_config import config_seed_parameters
from mouse_behavior.evaluation.exported_dataset_prediction import predict_features
from mouse_behavior.evaluation.social_event_geometry import body_footprint_distances
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    OPTIMIZED_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)
from tests.regression import test_heuristics_v16_audit as audit

ROOT = Path(__file__).resolve().parents[2]


def _params():
    return config_seed_parameters(load_config(ROOT / "configs" / "default.yaml"))


def _add_pose_facing_axis(
    f,
    *,
    slot: int,
    direction: tuple[float, float],
    body_length_cm: float = 7.918,
    half_width_cm: float = 1.0,
) -> None:
    dx, dy = float(direction[0]), float(direction[1])
    norm = float(np.hypot(dx, dy))
    fwd = np.array([dx / norm, dy / norm], dtype=float)
    side = np.array([-fwd[1], fwd[0]], dtype=float)
    half_len = 0.5 * body_length_cm
    shape = (
        (half_len, 0.0),
        (0.32 * body_length_cm, 0.8 * half_width_cm),
        (0.32 * body_length_cm, -0.8 * half_width_cm),
        (0.12 * body_length_cm, 0.0),
        (-0.25 * body_length_cm, half_width_cm),
        (-0.25 * body_length_cm, -half_width_cm),
        (-half_len, 0.0),
    )
    if f.collective_keypoints_cm is None:
        f.collective_keypoints_cm = np.full(
            (len(f.valid), f.centers_cm.shape[1], 7, 2), np.nan, dtype=float
        )
    f.collective_keypoints_cm[:, slot] = np.stack(
        [f.centers_cm[:, slot] + x * fwd + y * side for x, y in shape],
        axis=1,
    )
    f.heading_unit[:, slot] = fwd


def _sync_pair(f) -> None:
    audit.sync(f)
    audit.resync_kinematics(f)
    f.speed_cm_s[0] = 0.0
    f.velocity_cm_s[0] = 0.0
    if hasattr(f, "body_length_ref_cm") and f.body_length_ref_cm is not None:
        f.body_length_ref_cm[:] = float(f.reference_body_cm)
    if hasattr(f, "body_width_ref_cm") and f.body_width_ref_cm is not None:
        f.body_width_ref_cm[:] = float(f.mouse_width_cm)


def test_optimized_short_fast_approach_recovers_in_masks_and_predict_features() -> None:
    """Fast 7-frame (~0.23s) approaches qualify in both masks and predict_features."""
    f = audit.features(7, 2, fps=30.0, reference_body_cm=7.918)
    time = np.arange(7, dtype=float) / f.fps
    f.centers_cm[:, 0, 0] = 40.0 * time
    f.centers_cm[:, 0, 1] = 0.0
    f.centers_cm[:, 1, 0] = 19.0
    f.centers_cm[:, 1, 1] = 0.0
    _sync_pair(f)
    _add_pose_facing_axis(f, slot=0, direction=(1.0, 0.0))
    _add_pose_facing_axis(f, slot=1, direction=(-1.0, 0.0))
    audit.sync(f)

    params = _params()
    legacy_masks = rules.pair_rule_masks(f, params, DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert not legacy_masks["approach"][:, 0].any()

    opt_masks = rules.pair_rule_masks(f, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert opt_masks["approach"][:, 0].any()

    pred = predict_features(f, params, social_config=OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    pair_score = next(iter(pred["pair_scores"].values()))
    assert pair_score["approach"] > 0.0


def test_optimized_wide_endpoint_approach_accepts_closing_and_rejects_receding() -> None:
    """Wider approach endpoint cap (22.5 cm) accepts closing but still rejects receding."""
    n = 30
    f = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    f.centers_cm[:, 0, 0] = np.linspace(0.0, 12.0, n)
    f.centers_cm[:, 0, 1] = 0.0
    f.centers_cm[:, 1, 0] = 33.2
    f.centers_cm[:, 1, 1] = 0.0
    _sync_pair(f)
    # Give target large elongated pose? No, use realistic snout-to-surface <= 6.0 cm when centers ~21.2 cm:
    # Two 7.9 cm mice face each other at 21.2 cm -> nose-to-nose ~ 13.3 cm; to keep surface <= 5.5 cm
    # place box/hull or shorter closing where endpoint <= 22.0 cm AND surface <= 5.8 cm (large mouse body):
    f.reference_body_cm = 12.0
    _add_pose_facing_axis(f, slot=0, direction=(1.0, 0.0), body_length_cm=16.5)
    _add_pose_facing_axis(f, slot=1, direction=(-1.0, 0.0), body_length_cm=16.5)
    audit.sync(f)

    params = _params()
    legacy_masks = rules.pair_rule_masks(f, params, DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert not legacy_masks["approach"][:, 0].any()

    opt_masks = rules.pair_rule_masks(f, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert opt_masks["approach"][:, 0].any()

    # Reversing actor velocity so the pair recedes must not trigger approach.
    receding = audit.features(n, 2, fps=30.0, reference_body_cm=12.0)
    receding.centers_cm[:, 0, 0] = np.linspace(12.0, 0.0, n)
    receding.centers_cm[:, 0, 1] = 0.0
    receding.centers_cm[:, 1, 0] = 33.2
    receding.centers_cm[:, 1, 1] = 0.0
    _sync_pair(receding)
    _add_pose_facing_axis(receding, slot=0, direction=(-1.0, 0.0), body_length_cm=16.5)
    _add_pose_facing_axis(receding, slot=1, direction=(-1.0, 0.0), body_length_cm=16.5)
    audit.sync(receding)
    receding_masks = rules.pair_rule_masks(
        receding, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG
    )
    assert not receding_masks["approach"][:, 0].any()


def test_optimized_together_tight_and_wide_copresence_discriminates_sniffing_and_motion() -> None:
    """Optimized Together captures tight huddles & side-by-side rest while rejecting sniffing/fast passes."""
    params = _params()
    n = 45

    # 1. Tight cuddle at center distance 5.36 cm (< 5.40 cm, > DEFAULT 5.318 cm): qualifies with overlap.
    tight = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    tight.centers_cm[:, 0, :] = [0.0, 0.0]
    tight.centers_cm[:, 1, :] = [5.36, 0.0]
    _sync_pair(tight)
    _add_pose_facing_axis(tight, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(tight, slot=1, direction=(-1.0, 0.0), half_width_cm=1.0)
    audit.sync(tight)
    assert not rules.pair_rule_masks(tight, params, DEFAULT_SOCIAL_HEURISTIC_CONFIG)["together"].any()
    tight_masks = rules.pair_rule_masks(tight, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert tight_masks["together"][:, 0].all()

    # 2. Calm side-by-side co-presence at center distance 7.2 cm with >= 1.8 cm body-footprint gap.
    side_by_side = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    side_by_side.centers_cm[:, 0, :] = [0.0, 0.0]
    side_by_side.centers_cm[:, 1, :] = [0.0, 7.2]
    _sync_pair(side_by_side)
    _add_pose_facing_axis(side_by_side, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(side_by_side, slot=1, direction=(1.0, 0.0), half_width_cm=1.0)
    audit.sync(side_by_side)
    gap = body_footprint_distances(side_by_side, 0, max_body_lengths=2.5)
    assert float(np.nanmedian(gap)) >= 1.8
    assert not rules.pair_rule_masks(side_by_side, params, DEFAULT_SOCIAL_HEURISTIC_CONFIG)["together"].any()
    wide_masks = rules.pair_rule_masks(side_by_side, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert wide_masks["together"][:, 0].all()
    pred_wide = predict_features(side_by_side, params, social_config=OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    pair_scores = next(iter(pred_wide["pair_scores"].values()))
    assert pair_scores["together"] > 0.0

    # 3. Active nose-to-head sniffing at d=8.4 cm (noses ~0.5 cm apart, body_gap < 1.6 cm) stays head_contact.
    sniff = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    sniff.centers_cm[:, 0, :] = [0.0, 0.0]
    sniff.centers_cm[:, 1, :] = [8.4, 0.0]
    _sync_pair(sniff)
    _add_pose_facing_axis(sniff, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(sniff, slot=1, direction=(-1.0, 0.0), half_width_cm=1.0)
    audit.sync(sniff)
    sniff_masks = rules.pair_rule_masks(sniff, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert not sniff_masks["together"][:, 0].any()
    assert sniff_masks["head_contact"][:, 0].any()

    # 4. Fast parallel movement at d=7.2 cm with body_gap >= 1.6 cm must not trigger Together.
    fast_pair = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    time = np.arange(n, dtype=float) / fast_pair.fps
    fast_pair.centers_cm[:, 0, 0] = 10.0 * time
    fast_pair.centers_cm[:, 0, 1] = 0.0
    fast_pair.centers_cm[:, 1, 0] = 10.0 * time
    fast_pair.centers_cm[:, 1, 1] = 7.2
    _sync_pair(fast_pair)
    _add_pose_facing_axis(fast_pair, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(fast_pair, slot=1, direction=(1.0, 0.0), half_width_cm=1.0)
    audit.sync(fast_pair)
    fast_masks = rules.pair_rule_masks(fast_pair, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert not fast_masks["together"][:, 0].any()


def test_optimized_leading_approach_does_not_displace_sustained_together() -> None:
    """Brief approach lead-in before >=1.0s Together preserves Together dominance."""
    n = 70
    f = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    # First 15 frames approach, remaining 55 frames calm side-by-side at d=7.0 cm.
    x0 = np.concatenate([np.linspace(-20.0, 0.0, 15), np.zeros(55)])
    y0 = np.zeros(n)
    x1 = np.zeros(n)
    y1 = np.concatenate([np.full(15, 7.0), np.full(55, 7.0)])
    f.centers_cm[:, 0, 0] = x0
    f.centers_cm[:, 0, 1] = y0
    f.centers_cm[:, 1, 0] = x1
    f.centers_cm[:, 1, 1] = y1
    _sync_pair(f)
    _add_pose_facing_axis(f, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(f, slot=1, direction=(1.0, 0.0), half_width_cm=1.0)
    audit.sync(f)

    params = _params()
    masks = rules.pair_rule_masks(f, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert masks["together"][:, 0].sum() >= 45
    pred = predict_features(f, params, social_config=OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    pair_scores = next(iter(pred["pair_scores"].values()))
    assert pair_scores["together"] > pair_scores["approach"]


def test_optimized_unobserved_pair_and_config_validation() -> None:
    """Unobserved frames produce empty masks safely and invalid wide fields raise."""
    f = audit.features(45, 2, fps=30.0, reference_body_cm=7.918)
    f.valid[:] = False
    f.centers_cm[:] = np.nan
    audit.sync(f)
    masks = rules.pair_rule_masks(f, _params(), OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert not masks["together"].any()
    assert not masks["approach"].any()

    with pytest.raises(ValueError, match="together_wide_max_distance_cm"):
        SocialHeuristicConfig(dist_together_max_cm=6.0, together_wide_max_distance_cm=5.0)
    with pytest.raises(ValueError, match="together_wide_min_body_gap_cm"):
        SocialHeuristicConfig(together_wide_min_body_gap_cm=-0.1)
    with pytest.raises(ValueError, match="together_wide_max_speed_cm_s"):
        SocialHeuristicConfig(together_wide_max_speed_cm_s=0.0)
    with pytest.raises(ValueError, match="together_wide_min_endpoint_gap_cm"):
        SocialHeuristicConfig(together_wide_min_endpoint_gap_cm=-0.1)


def test_wide_together_passes_frame_mask_and_handles_missing_keypoints(monkeypatch) -> None:
    """Wide Together passes candidate frame_mask and stays safe when keypoints are absent."""
    n = 60
    f = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    # Frames [0:15] fast (>10 cm/s), frames [15:60] calm side-by-side at d=10.5 cm.
    f.centers_cm[:15, 0, 0] = np.linspace(-6.0, 0.0, 15)
    f.centers_cm[15:, 0, 0] = 0.0
    f.centers_cm[:, 0, 1] = 0.0
    f.centers_cm[:, 1, 0] = 0.0
    f.centers_cm[:, 1, 1] = 10.5
    _sync_pair(f)
    _add_pose_facing_axis(f, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(f, slot=1, direction=(1.0, 0.0), half_width_cm=1.0)
    audit.sync(f)

    captured_masks: list[np.ndarray | None] = []
    real_bfd = rules.body_footprint_distances

    def spy_bfd(features, edge, *, max_body_lengths, frame_mask=None):
        if np.isclose(max_body_lengths, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG.together_wide_max_footprint_body_lengths):
            captured_masks.append(None if frame_mask is None else np.asarray(frame_mask, bool).copy())
        return real_bfd(features, edge, max_body_lengths=max_body_lengths, frame_mask=frame_mask)

    monkeypatch.setattr(rules, "body_footprint_distances", spy_bfd)
    masks = rules.pair_rule_masks(f, _params(), OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert masks["together"][20:, 0].any()
    assert len(captured_masks) == 1
    assert captured_masks[0] is not None
    assert not captured_masks[0][:10].any(), "fast frames must be masked out before hull geometry"
    assert captured_masks[0][20:].all()

    # Missing collective_keypoints_cm must not crash wide Together and tight Together still works.
    f.collective_keypoints_cm = None
    no_kp_wide = rules.pair_rule_masks(f, _params(), OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert not no_kp_wide["together"][:, 0].any()

    f_tight = audit.features(50, 2, fps=30.0, reference_body_cm=7.918)
    f_tight.centers_cm[:, 1, 0] = 4.2
    _sync_pair(f_tight)
    audit.sync(f_tight)
    f_tight.collective_keypoints_cm = None
    tight_masks = rules.pair_rule_masks(f_tight, _params(), OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert tight_masks["together"][:, 0].all()


def test_short_approach_yields_to_longer_qualified_head_contact() -> None:
    """Short (<0.30s) approach bursts must not split or preempt longer head_contact."""
    n = 30
    f = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    # Actor approaches slowly during [0:6] (~0.17s) while noses stay close across [0:30] (1.0s),
    # simulating continuous Snout-head_contact with a brief mid/initial closing shift.
    f.centers_cm[:6, 0, 0] = np.linspace(-13.0, -7.5, 6)
    f.centers_cm[6:, 0, 0] = -7.5
    f.centers_cm[:, 1, 0] = 0.0
    _sync_pair(f)
    _add_pose_facing_axis(f, slot=0, direction=(1.0, 0.0))
    _add_pose_facing_axis(f, slot=1, direction=(-1.0, 0.0))
    audit.sync(f)
    # Place noses within head-contact range across the whole bout while keeping centers >=7.5cm.
    f.collective_keypoints_cm[:, 0, 0, :2] = np.column_stack(
        [f.centers_cm[:, 1, 0] - 1.5, np.zeros(n)]
    )
    f.collective_keypoints_cm[:, 1, 0, :2] = np.column_stack(
        [f.centers_cm[:, 1, 0] + 0.0, np.zeros(n)]
    )
    audit.sync(f)

    params = _params()
    opt_masks = rules.pair_rule_masks(f, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert opt_masks["head_contact"][:, 0].sum() >= 24
    assert opt_masks["head_contact"][:, 0].sum() > opt_masks["approach"][:, 0].sum()


def test_wide_together_endpoint_guard_protects_distant_snout_tail_contact() -> None:
    """Wide Together (>=1.2cm endpoint gap) preserves distant Snout-rear_contact vs side-by-side resting."""
    n = 60
    params = _params()

    # Case 1: Calm pair at center=9.5cm, body_gap >= 2.0cm, but snout touches tailroot (td=0.8cm).
    snout_tail = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    snout_tail.centers_cm[:, 0, 0] = -9.5
    snout_tail.centers_cm[:, 1, 0] = 0.0
    _sync_pair(snout_tail)
    _add_pose_facing_axis(snout_tail, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(snout_tail, slot=1, direction=(1.0, 0.0), half_width_cm=1.0)
    # Move mouse 0 nose (kp 0) close to mouse 1 tailroot (kp 6 at -3.959).
    snout_tail.collective_keypoints_cm[:, 0, 0, :] = [-4.8, 0.0]
    audit.sync(snout_tail)
    st_masks = rules.pair_rule_masks(snout_tail, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert not st_masks["together"][:, 0].any()
    assert st_masks["tail_contact"][:, 0].sum() >= 45

    # Case 2: Calm parallel side-by-side mice at center=12.0cm with no endpoint contact.
    parallel = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    parallel.centers_cm[:, 0, :] = [0.0, 0.0]
    parallel.centers_cm[:, 1, :] = [0.0, 12.0]
    _sync_pair(parallel)
    _add_pose_facing_axis(parallel, slot=0, direction=(1.0, 0.0), half_width_cm=1.0)
    _add_pose_facing_axis(parallel, slot=1, direction=(1.0, 0.0), half_width_cm=1.0)
    audit.sync(parallel)
    par_masks = rules.pair_rule_masks(parallel, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert par_masks["together"][:, 0].sum() >= 50


def test_predict_features_syncs_attack_and_chase_durations_and_receipt() -> None:
    """predict_features honors OPTIMIZED attack/chase min durations and policy_receipt updates."""
    params = _params()

    # Synthetic ~0.80s chase (< default 1.0s, >= optimized 0.50s).
    n = 24
    f_chase = audit.features(n, 2, fps=30.0, reference_body_cm=7.918)
    t = np.arange(n, dtype=float) / f_chase.fps
    f_chase.centers_cm[:, 0, 0] = 45.0 * t
    f_chase.centers_cm[:, 1, 0] = 7.0 + 44.0 * t
    _sync_pair(f_chase)
    _add_pose_facing_axis(f_chase, slot=0, direction=(1.0, 0.0))
    _add_pose_facing_axis(f_chase, slot=1, direction=(1.0, 0.0))
    audit.sync(f_chase)
    masks = rules.pair_rule_masks(f_chase, params, OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    assert 15 <= int(masks["chasing"][:, 0].sum()) < 30
    pred_def = predict_features(f_chase, params, social_config=DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert next(iter(pred_def["pair_scores"].values()))["chase"] == 0.0
    pred = predict_features(f_chase, params, social_config=OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    chase_score = next(iter(pred["pair_scores"].values()))["chase"]
    assert chase_score >= 0.50

    default_receipt = rules.policy_receipt(DEFAULT_SOCIAL_HEURISTIC_CONFIG)
    assert "continuous>=1s; bounded unknown gaps<=0.15s; no contact veto; Approach wins" in (
        default_receipt["rules"]["together"]
    )
    opt_receipt = rules.policy_receipt(OPTIMIZED_SOCIAL_HEURISTIC_CONFIG)
    opt_tog_text = opt_receipt["rules"]["together"]
    assert "wide_branch: center<13cm" in opt_tog_text
    assert "body_gap>=1.8cm" in opt_tog_text
    assert "endpoint_gap>=1.2cm" in opt_tog_text
    assert "priority_order=together,attack,chasing,approach,head_contact,tail_contact" in opt_tog_text
