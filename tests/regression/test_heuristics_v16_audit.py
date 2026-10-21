"""Independent audit regressions for the 2026-10-15 v16 heuristic change set.

Purpose
-------
This module is NOT a restatement of ``test_heuristics_v16.py``.  It was written
from a source-level review of the v16 edits and has three jobs:

1. ``GUARD`` tests prove that a fixture actually reaches the code path under
   test.  The shipped v16 suite asserts ``not masks["approach"].any()`` on a
   fixture whose ``heading_unit`` is ``None``; ``_open_approach_mask`` returns
   all-False on line 2 of the function in that case, so the assertion cannot
   fail no matter what the rule does.  Every fixture here is checked for
   reachability before it is used for a behavioural claim.

2. ``LOCK`` tests pin the v16 requirements that the review verified as
   genuinely implemented, so a later edit cannot silently undo them.

3. ``DEFECT`` tests encode the *correct* behaviour for each defect found in
   the review and are marked ``xfail(strict=True, reason="DEFECT <id> ...")``.
   They fail today, they keep the suite green, and ``strict=True`` turns them
   red the moment somebody fixes the code without removing the marker.  Run
   ``pytest -rxX`` to see every defect reason in the short summary.

Defect ids referenced here: IND-1..IND-6, SOC-1..SOC-17, GRP-1..GRP-18,
POL-1..POL-2, PROV-1..PROV-4, DOC-1..DOC-3, POST-1..POST-4, SITE-1, EMP-1..EMP-7.
Each id maps to a ``file:line`` in the review.

Acceptance criterion (corrected after review)
--------------------------------------------
The goal is NOT whole-clip Top-1 agreement with the annotation. It is
segment-level detection: inside the annotated clip, does the rule layer emit the
target behavior on the correct target as a time segment? Clip Top-1 gives every
clip exactly one label, so a background pair or a background lone mouse can win
the slot even when the true behavior was detected; measuring only Top-1 conflates
RULE loss (never detected) with RANKING loss (detected, then stolen). Section 7
therefore reads the three segment-level artifacts directly and reports the two
losses separately.

Retraction: an earlier revision of this review concluded from
``individual_validation_metrics.json`` that the v16 individual-layer change
produced "exactly zero net improvement", because ``strict_accuracy`` is
bit-identical (0.4406779661016949) in the v14 and v16 runs. That conclusion was
an artifact of the whole-clip Top-1 口径, which folds ``correct_behavior_wrong_visual_ids``
(10 in both runs) into the same number. Under the segment-level criterion
(``target_id_conditioned_metrics.json``) locomotion recall went 0.903 -> 1.000
with false negatives 3 -> 0 and precision 0.824 -> 0.939. The module-1 pause
bridge works; the Top-1 metric simply cannot see it. ``test_lock_individual_
locomotion_is_fully_recovered_on_the_correct_target`` locks the corrected reading.

Fixture discipline
------------------
* ``speed_cm_s`` / ``velocity_cm_s`` are always derived from ``centers_cm``.
  The shipped fixture leaves them at zero while the centres move, which makes
  every speed gate pass for free and hides the missing ``speed <= 2.0`` guard
  in ``_huddle_geometry``.
* Group fixtures leave ``collective_keypoints_cm = None`` so that
  ``screening_signals.body_motion`` passes the supplied speed/velocity through
  unchanged instead of re-deriving them.
* ``reference_body_cm`` is set explicitly.  The spec's distances (13.375 cm
  neighbour radius, 1.3375 cm static radius) are derived from a 10.7 cm body
  length, while the shipped default is 7.918 cm; see DEFECT PROV-3.
"""

from __future__ import annotations

import ast
import csv
import itertools
import inspect
import json
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from mouse_behavior.behavior.individual_rules import (
    DEFAULT_INDIVIDUAL_CONFIG,
    INDIVIDUAL_RULE_REVISION,
    individual_rule_masks,
)
from mouse_behavior.behavior.static_rule import bounded_stationary_mask
from mouse_behavior.evaluation import confirmed_rules as confirmed_rules_module
from mouse_behavior.evaluation import group_event_rules as group_module
from mouse_behavior.evaluation.calibrated_rule_ranker import (
    CandidateTable,
    build_candidate_table,
    predict_table,
)
from mouse_behavior.evaluation.confirmed_rules import (
    _open_approach_mask,
    pair_rule_masks,
    policy_receipt,
)
from mouse_behavior.evaluation.exported_dataset_models import (
    CANONICAL_BEHAVIORS,
    GROUP_BEHAVIORS,
    INDIVIDUAL_BEHAVIORS,
    ExportedFeatures,
    HeuristicParameters,
)
from mouse_behavior.evaluation.group_event_rules import (
    GROUP_RULE_REVISION,
    NEIGHBOR_BODY_LENGTHS,
    _huddle_event,
    _huddle_geometry,
    _initial_compact_components,
    dispersal_phase,
    group_rule_candidates,
)
from mouse_behavior.evaluation.screening_signals import runs
from mouse_behavior.evaluation.social_event_geometry import (
    endpoint_distances,
    snout_surface_distances,
)
from mouse_behavior.evaluation.social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    SocialHeuristicConfig,
)

FPS = 30.0
BODY_CM = 10.7  # the reference body length every spec distance is derived from
LEGACY_BODY_CM = 7.918  # the default actually shipped in ExportedFeatures
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mouse_behavior"

xfail_defect = pytest.mark.xfail(strict=True)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    return list(runs(np.asarray(mask, dtype=bool)))


def _kinematics(centers: np.ndarray, valid: np.ndarray, fps: float):
    """Derive speed/velocity from centres so the fixture cannot lie."""
    points = np.where(valid[..., None], centers, np.nan)
    velocity = np.zeros_like(points, dtype=float)
    if len(points) > 1:
        velocity[1:] = (points[1:] - points[:-1]) * fps
        velocity[0] = velocity[1]
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.linalg.norm(velocity, axis=2)
    return speed, velocity


def _heading(velocity: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(velocity, axis=2, keepdims=True)
    unit = np.where(norm > 1e-9, velocity / np.maximum(norm, 1e-9), np.nan)
    forward = np.zeros_like(unit)
    forward[..., 0] = 1.0
    return np.where(np.isfinite(unit).all(axis=2, keepdims=True), unit, forward)


def features(
    frames: int,
    mice: int,
    fps: float = FPS,
    *,
    reference_body_cm: float = BODY_CM,
    with_kinematics: bool = True,
) -> ExportedFeatures:
    """A complete ExportedFeatures whose pair geometry is always consistent."""
    pair_i, pair_j = np.triu_indices(mice, k=1)
    edges = len(pair_i)
    valid = np.ones((frames, mice), dtype=bool)
    centers = np.zeros((frames, mice, 2), dtype=float)
    far = np.full((frames, edges), 50.0, dtype=float)
    if with_kinematics:
        speed, velocity = _kinematics(centers, valid, fps)
    else:
        speed = np.zeros((frames, mice), dtype=float)
        velocity = np.zeros((frames, mice, 2), dtype=float)
    return ExportedFeatures(
        fps=fps,
        track_ids=np.arange(100, 100 + mice, dtype=int),
        valid=valid,
        centers_cm=centers,
        speed_cm_s=speed,
        velocity_cm_s=velocity,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_distance_cm=np.full((frames, edges), np.nan),
        pair_closing_speed_cm_s=np.zeros((frames, edges)),
        pair_direction_similarity=np.zeros((frames, edges)),
        pair_nose_head_cm=far.copy(),
        pair_nose_tail_cm=far.copy(),
        nearest_distance_cm=np.full((frames, mice), np.nan),
        mean_nearest_distance_cm=np.full((frames, mice), np.nan),
        cm_per_pixel=1.0,
        track_selection_truncated=False,
        selected_track_count=mice,
        reference_body_cm=reference_body_cm,
        pair_nose_head_left_to_right_cm=far.copy(),
        pair_nose_head_right_to_left_cm=far.copy(),
        pair_nose_tail_left_to_right_cm=far.copy(),
        pair_nose_tail_right_to_left_cm=far.copy(),
    )


def sync(f: ExportedFeatures) -> ExportedFeatures:
    """Recompute every derived array after centres/valid were edited."""
    frames, mice = f.valid.shape
    if not hasattr(f, "_audit_kept_speed"):
        pass
    f.pair_distance_cm[:] = np.nan
    for edge, (left, right) in enumerate(zip(f.pair_i, f.pair_j, strict=True)):
        both = f.valid[:, left] & f.valid[:, right]
        gap = np.linalg.norm(f.centers_cm[:, left] - f.centers_cm[:, right], axis=1)
        f.pair_distance_cm[:, edge] = np.where(both, gap, np.nan)
    f.nearest_distance_cm[:] = np.nan
    for frame in range(frames):
        for slot in range(mice):
            if not f.valid[frame, slot]:
                continue
            others = [o for o in range(mice) if o != slot and f.valid[frame, o]]
            if others:
                f.nearest_distance_cm[frame, slot] = min(
                    float(np.linalg.norm(f.centers_cm[frame, slot] - f.centers_cm[frame, o]))
                    for o in others
                )
    f.mean_nearest_distance_cm[:] = f.nearest_distance_cm
    return f


def set_centers(f: ExportedFeatures, positions, *, start: int = 0, end: int | None = None):
    """Assign centres over a frame range, NaN-out invalid frames, refresh."""
    end = len(f.valid) if end is None else end
    f.centers_cm[start:end] = np.asarray(positions, dtype=float)
    f.centers_cm[~f.valid] = np.nan
    return f


def resync_kinematics(f: ExportedFeatures) -> ExportedFeatures:
    f.speed_cm_s[:], f.velocity_cm_s[:] = _kinematics(f.centers_cm, f.valid, f.fps)
    f.heading_unit = _heading(f.velocity_cm_s)
    return f


def confirmed_params(**overrides) -> HeuristicParameters:
    return replace(HeuristicParameters(confirmed_rules_v1=1.0), **overrides)


def component_social_config(**overrides) -> SocialHeuristicConfig:
    """Historical numeric fixtures isolate rule semantics from v17 full-set fitting.

    These thresholds are intentional local test inputs, not production defaults.
    Keep the user's six-way priority and Together's 7 cm / 4 cm/s gates.
    """
    return replace(
        SocialHeuristicConfig(
            dist_nose_head_contact_max_cm=1.5, dist_nose_tail_contact_max_cm=1.5,
            dist_attack_max_cm=4.0, vel_attack_min_cm_s=4.0,
            acc_attack_min_cm_s2=25.0,
            nose_contact_min_duration_s=.5, contact_bridge_max_gap_s=.15,
            together_contact_veto_min_duration_s=.15, chasing_min_mean_speed_cm_s=0.,
            vel_chasing_min_cm_s=26.75, vel_chasing_target_min_cm_s=26.75,
            dist_approach_start_min_cm=12.0, dist_approach_end_min_cm=9.0,
            dist_approach_end_max_cm=12.0, approach_surface_endpoint_cm=2.0,
            approach_surface_endpoint_tolerance_cm=0.5,
        ),
        **overrides,
    )


def positive_evidence_model(table: CandidateTable) -> dict:
    return {
        "feature_names": list(table.feature_names),
        "coefficients": {b: [0.0] * len(table.feature_names) for b in CANONICAL_BEHAVIORS},
        "require_positive_evidence": True,
    }


def evidence_of(table: CandidateTable, behavior: str, ids: tuple[int, ...]) -> float:
    for candidate, row in zip(table.candidates, table.values, strict=True):
        if candidate.behavior == behavior and candidate.target_ids == ids:
            return float(candidate.evidence)
    return float("nan")


# ==========================================================================
# 1. GUARD - prove the fixtures reach the code under test
# ==========================================================================
def test_guard_group_fixture_reaches_group_rule_candidates():
    """A 4-mouse compact pile must actually produce group candidates."""
    f = features(180, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    sync(f)
    resync_kinematics(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    candidates = group_rule_candidates(f, confirmed_params(), ())
    assert candidates, "fixture never reached group_rule_candidates emission"
    assert any(c.behavior == "huddle" and c.evidence > 0 for c in candidates)


def test_guard_social_contact_path_is_reachable():
    """Directed contact distances must be finite, otherwise every contact
    assertion below would be vacuous (the trap that makes the shipped suite's
    ``not approach.any()`` assertions untestable)."""
    f = features(60, 2)
    set_centers(f, [[0.0, 0.0], [7.5, 0.0]])  # Outside Together's center gate.
    sync(f)
    f.pair_nose_head_left_to_right_cm[:, 0] = 1.0
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    head, _ = endpoint_distances(f, 0, "head")
    assert np.isfinite(head).all(), "endpoint_distances fell through to NaN"
    assert masks["head_contact"][:, 0].any(), "contact path unreachable in fixture"


def test_guard_approach_path_requires_heading_and_keypoints():
    """Documents why the shipped v16 approach assertions are vacuous.

    ``_open_approach_mask`` returns all-False immediately when ``heading_unit``
    is None (confirmed_rules.py:505-506) and ``snout_surface_distances``
    returns all-NaN when ``collective_keypoints_cm`` is None
    (social_event_geometry.py:61-62), which makes ``eligible`` empty.  A
    fixture without both can never emit approach, so ``assert not
    masks["approach"].any()`` on such a fixture proves nothing.
    """
    f = features(60, 2)
    set_centers(f, [[0.0, 0.0], [4.0, 0.0]])
    sync(f)
    assert f.heading_unit is None
    assert _open_approach_mask(f, 0, confirmed_params()).sum() == 0
    surface = snout_surface_distances(f, 0)
    assert not np.isfinite(surface[0]).any()
    assert not np.isfinite(surface[1]).any()
    # The same fixture with heading supplied still cannot emit approach,
    # because the surface geometry is unavailable without keypoints.
    f.heading_unit = _heading(np.ones((60, 2, 2)))
    assert _open_approach_mask(f, 0, confirmed_params()).sum() == 0


# ==========================================================================
# 2. Individual layer
# ==========================================================================
def _locomotion_track(frames: int, fps: float, *, pause: tuple[int, int] | None,
                      speed_cm_s: float = 10.0, missing: tuple[int, int] | None = None):
    """Straight +x walk with an optional standstill pause and dropout."""
    centers = np.zeros((frames, 1, 2), dtype=float)
    valid = np.ones((frames, 1), dtype=bool)
    step = speed_cm_s / fps
    x = 0.0
    for index in range(frames):
        if pause is not None and pause[0] <= index < pause[1]:
            pass
        else:
            x += step
        centers[index, 0, 0] = x
    if missing is not None:
        valid[missing[0]:missing[1], 0] = False
    centers[~valid] = np.nan
    return centers, valid


def test_lock_locomotion_bridges_a_pure_stationary_pause_under_one_second():
    """v16 requirement 1.2.2: a 0.8 s mid-run pause plus a 2-frame dropout is
    connected into a single locomotion bout.  Verified implemented."""
    frames = 150
    centers, valid = _locomotion_track(frames, FPS, pause=(60, 84), missing=(70, 72))
    result = individual_rule_masks(centers, valid, FPS)
    bouts = _runs_of(result.masks["locomotion"][:, 0])
    assert len(bouts) == 1, f"expected one connected bout, got {bouts}"
    assert bouts[0][0] < 60 and bouts[0][1] > 84


def test_lock_locomotion_pause_bridge_respects_the_one_second_ceiling():
    """Boundary: a 1.2 s pause must NOT be bridged (locomotion_pause_max_s=1.0)."""
    frames = 180
    centers, valid = _locomotion_track(frames, FPS, pause=(60, 96))
    result = individual_rule_masks(centers, valid, FPS)
    bouts = _runs_of(result.masks["locomotion"][:, 0])
    assert len(bouts) == 2, f"a 1.2 s pause was bridged: {bouts}"


def test_lock_shortened_body_length_does_not_veto_stationary():
    """v16 requirement 1.1.1: a 4.5 cm projection with a 3-frame dropout still
    yields stationary.  Verified implemented (static_posture_eligible_mask is
    no longer wired into the individual classifier)."""
    frames = 90
    centers = np.zeros((frames, 1, 2), dtype=float)
    valid = np.ones((frames, 1), dtype=bool)
    valid[40:43, 0] = False
    keypoints = np.zeros((frames, 1, 7, 2), dtype=float)
    keypoints[:, 0, 0, 0] = 4.5
    keypoints[~valid] = np.nan
    centers[~valid] = np.nan
    result = individual_rule_masks(centers, valid, FPS, keypoints_cm=keypoints)
    assert result.masks["stationary"][:, 0].any()
    assert result.masks["stationary"][40:43, 0].all(), "dropout frames not held"
    assert not result.masks["locomotion"][:, 0].any()


def test_lock_bounded_stationary_bridges_a_one_second_detection_gap():
    """v16 requirement 1.1.2: max_missing_gap_s = 1.0 s, anchors inside radius."""
    frames = 120
    centers = np.zeros((frames, 1, 2), dtype=float)
    valid = np.ones((frames, 1), dtype=bool)
    valid[40:70, 0] = False  # exactly 1.0 s of dropout
    low = np.ones((frames, 1), dtype=bool)
    low[40:70, 0] = False
    out = bounded_stationary_mask(
        low, centers, valid, fps=FPS, max_gap_s=0.2, max_missing_gap_s=1.0,
        max_transient_displacement_cm=1.3375, max_center_radius_cm=1.3375,
    )
    assert out[40:70, 0].all(), "a 1.0 s detection gap was not bridged"


def test_lock_bounded_stationary_rejects_a_gap_above_the_missing_limit():
    frames = 120
    centers = np.zeros((frames, 1), dtype=float)
    centers = np.repeat(centers[:, :, None], 2, axis=2)
    valid = np.ones((frames, 1), dtype=bool)
    valid[40:72, 0] = False  # 32 frames > 1.0 s
    low = np.ones((frames, 1), dtype=bool)
    low[40:72, 0] = False
    out = bounded_stationary_mask(
        low, centers, valid, fps=FPS, max_gap_s=0.2, max_missing_gap_s=1.0,
        max_transient_displacement_cm=1.3375, max_center_radius_cm=1.3375,
    )
    assert not out[40:72, 0].any(), "a 1.07 s detection gap was bridged"


@xfail_defect(reason="DEFECT IND-1 individual_rules.py:322-327 - the 'tiny posture "
                     "flicker' bridge is unreachable: grooming/rearing are both "
                     "_qualify()-ed to >=0.3 s / >=0.5 s, i.e. >=9 frames at 30 fps, "
                     "while posture_gap_limit is max(2, floor(0.1*fps)) = 3 frames, so "
                     "`posture[gap].any()` always implies gap_frames > limit. "
                     "Spec 1.2.3 (motion must not be shredded by posture noise) is "
                     "therefore unimplemented and the branch is dead code.")
def test_defect_ind_1_posture_flicker_bridge_is_reachable():
    config = DEFAULT_INDIVIDUAL_CONFIG
    posture_gap_limit = max(2, int(np.floor(0.1 * FPS + 1e-9)))
    shortest_posture_bout = min(config.grooming_min_duration_s,
                                config.rearing_min_duration_s) * FPS
    assert shortest_posture_bout <= posture_gap_limit


def test_defect_ind_2_minimum_stationary_bout_survives_a_flanking_walk():
    frames = 150
    centers = np.zeros((frames, 1, 2), dtype=float)
    valid = np.ones((frames, 1), dtype=bool)
    step = 10.0 / FPS
    x = 0.0
    for index in range(frames):
        if not (45 <= index < 75):  # exactly 1.0 s standstill
            x += step
        centers[index, 0, 0] = x
    result = individual_rule_masks(centers, valid, FPS)
    assert result.masks["stationary"][45:75, 0].all(), (
        "a 1.0 s stationary bout between two walks was erased; "
        f"stationary={result.masks['stationary'][:, 0].sum()} "
        f"locomotion={result.masks['locomotion'][:, 0].sum()}"
    )

# NOTE: an earlier draft of this module carried a DEFECT IND-4 test claiming
# that bounded_stationary_mask()'s excursion anchor (static_rule.py:143,
# `anchor = points[0]`) was order-dependent. It was executed and XPASSed: the
# anchor excursion bound and the median-radius bound are complementary, so the
# hypothesis was wrong and the test was removed rather than weakened.

def test_defect_ind_5_short_clips_still_get_a_speed_estimate():
    frames = 9
    centers, valid = _locomotion_track(frames, FPS, pause=None)
    result = individual_rule_masks(centers, valid, FPS)
    speed = result.diagnostics["body_speed_cm_s"][:, 0]
    assert np.isfinite(speed).any(), f"all-NaN speed for a {frames}-frame clip"


# ==========================================================================
# 3. Social layer
# ==========================================================================
def _pair_clip(frames: int, separation_cm: float, *, fps: float = FPS,
               body_cm: float = BODY_CM) -> ExportedFeatures:
    """Two mice held a fixed distance apart, no keypoints (so the directed
    ``pair_nose_*`` arrays drive ``endpoint_distances``)."""
    f = features(frames, 2, fps, reference_body_cm=body_cm)
    set_centers(f, [[0.0, 0.0], [separation_cm, 0.0]])
    sync(f)
    resync_kinematics(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    return f


def test_lock_head_priority_beats_a_longer_tail_bout():
    """v17 makes head and tail separate priority entries; duration cannot swap them."""
    f = _pair_clip(60, 7.5)
    f.pair_nose_tail_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_tail_left_to_right_cm[:45, 0] = 1.0
    f.pair_nose_head_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_head_left_to_right_cm[15:30, 0] = 1.0
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["head_contact"][:, 0].sum() == 15
    assert masks["head_contact"][15:30, 0].all()
    assert not masks["tail_contact"][15:30, 0].any()
    assert masks["tail_contact"][:15, 0].all()
    assert masks["tail_contact"][30:45, 0].all()
    assert masks["tail_contact"][:, 0].sum() == 30


def test_lock_head_priority_also_wins_when_head_is_longer():
    """Both contact subtypes qualify, but head keeps its explicit priority."""
    f = _pair_clip(60, 7.5)
    f.pair_nose_head_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_head_left_to_right_cm[:45, 0] = 1.0
    f.pair_nose_tail_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_tail_left_to_right_cm[15:30, 0] = 1.0
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["head_contact"][:, 0].sum() == 45
    assert not masks["tail_contact"][:, 0].any()


def test_lock_contact_bridges_a_short_keypoint_flicker():
    """A contact bout bridges unknown keypoints outside Together's distance gate."""
    f = _pair_clip(45, 7.5)
    channel = f.pair_nose_head_left_to_right_cm[:, 0]
    channel[:] = 50.0
    channel[:30] = 1.0
    channel[14:16] = np.nan  # 2-frame keypoint dropout
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["head_contact"][:, 0].sum() == 30


def test_lock_chase_is_no_longer_vetoed_by_a_single_head_contact_frame():
    """v16 requirement 2.1-B1: the ``if head_contact.any(): return result``
    single-frame veto is gone. Verified implemented."""
    frames = 90
    f = features(frames, 2)
    f.centers_cm[:, 0, 0] = np.arange(frames, dtype=float)
    f.centers_cm[:, 1, 0] = np.arange(frames, dtype=float) + 8.0
    sync(f)
    resync_kinematics(f)
    assert np.isclose(f.speed_cm_s[:, 0], FPS).all(), "fixture must run at 30 cm/s"
    f.pair_nose_head_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_head_left_to_right_cm[45, 0] = 1.0  # one-frame擦碰
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["chasing"][:, 0].sum() >= frames - 1


def test_lock_pair_masks_are_mutually_exclusive():
    f = _pair_clip(60, 4.0)
    f.pair_nose_tail_left_to_right_cm[:45, 0] = 1.0
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    names = ("approach", "attack", "chasing", "together", "head_contact", "tail_contact")
    active = [name for name in names if masks[name][:, 0].any()]
    assert len(active) <= 1, f"more than one label active: {active}"


def test_defect_soc_1_attack_needs_directional_evidence_across_a_box_gap():
    frames = 90
    f = _pair_clip(frames, 3.0)
    f.collective_boxes_cm = np.zeros((frames, 2, 4), dtype=float)
    f.collective_boxes_cm[:, :, 2:] = 1.0
    f.collective_boxes_cm[40:45] = np.nan  # 5-frame box dropout
    f.speed_cm_s[:] = 0.0
    f.speed_cm_s[38, 0] = 5.0  # one ordinary weight-shift frame
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert not masks["attack"][:, 0].any(), (
        "two motionless mice 3 cm apart were labelled attack because of a box gap"
    )


def test_lock_v17_together_then_attack_outrank_independent_contact():
    """Real lower-priority signals remain present when the user priority wins."""
    frames = 90
    f = _pair_clip(frames, 4.0)
    f.pair_nose_head_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_head_left_to_right_cm[:60, 0] = 1.0  # 2.0 s real contact
    f.collective_boxes_cm = np.zeros((frames, 2, 4), dtype=float)
    f.collective_boxes_cm[:, :, 2:] = 1.0
    f.collective_boxes_cm[30:31, 1] = np.nan
    f.collective_boxes_cm[33:35, 1] = np.nan
    f.centers_cm[20:36, 0, 0] = np.arange(16) * 0.2
    f.centers_cm[36:, 0, 0] = 3.0
    f.valid[30:31, 1] = False
    f.valid[33:35, 1] = False
    f.centers_cm[30:31, 1] = np.nan
    f.centers_cm[33:35, 1] = np.nan
    sync(f)
    resync_kinematics(f)
    # Isolate the historical 7cm/4cm/s priority fixture from fitted defaults.
    config = component_social_config(dist_together_max_cm=7.0, vel_together_max_cm_s=4.0)
    raw_attack = confirmed_rules_module._attack_action_mask(f, 0, config)
    assert raw_attack.any(), "fixture must supply a genuine directed dropout attack"
    head = endpoint_distances(f, 0, "head")[0]
    raw_contact, _ = confirmed_rules_module._bridged_contact_bout_mask(
        strict=f.valid.all(axis=1) & (head <= config.dist_nose_head_contact_max_cm),
        measured=head, center=f.pair_distance_cm[:, 0], body_cm=f.reference_body_cm,
        fps=f.fps, social_config=config,
    )
    assert raw_contact.sum() >= 30, "fixture must independently qualify head contact"
    masks = pair_rule_masks(f, confirmed_params(), config)
    assert _runs_of(masks["together"][:, 0]) == [(36, 90)]
    assert _runs_of(masks["attack"][:, 0]) == [(29, 36)]
    assert _runs_of(masks["head_contact"][:, 0]) == [(0, 29)]
    assert not (masks["together"] & masks["attack"]).any()
    assert not (masks["attack"] & masks["head_contact"]).any()
    # Removing only Together eligibility exposes Attack ahead of the same contact.
    no_together = replace(config, dist_together_max_cm=0.1)
    masks = pair_rule_masks(f, confirmed_params(), no_together)
    assert np.array_equal(masks["attack"][:, 0], raw_attack)
    assert _runs_of(masks["head_contact"][:, 0]) == [(0, 29), (36, 60)]
    f.collective_boxes_cm = None
    masks = pair_rule_masks(f, confirmed_params(), no_together)
    assert np.array_equal(masks["head_contact"][:, 0], raw_contact)


def test_defect_soc_7_attack_bout_duration_is_bounded():
    frames = 180
    f = _pair_clip(frames, 3.0)
    f.collective_boxes_cm = np.zeros((frames, 2, 4), dtype=float)
    f.collective_boxes_cm[:, :, 2:] = 1.0
    f.collective_boxes_cm[60:150] = np.nan  # 3.0 s dropout
    f.speed_cm_s[58, 0] = 5.0
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    limit = DEFAULT_SOCIAL_HEURISTIC_CONFIG.attack_pounce_max_duration_s
    for start, end in _runs_of(masks["attack"][:, 0]):
        assert (end - start) / FPS <= limit + 1e-9, (
            f"attack bout {(end - start) / FPS:.2f}s exceeds the {limit}s pounce bound"
        )


def test_defect_soc_5_bridge_does_not_convert_measured_separation_into_contact():
    f = _pair_clip(24, 7.5)
    channel = f.pair_nose_head_left_to_right_cm[:, 0]
    channel[:] = 5.0  # confidently measured, clearly not contact
    for frame in (0, 5, 10, 15, 20, 23):
        channel[frame] = 1.0
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    real_seconds = 6 / FPS
    assert not masks["head_contact"][:, 0].any(), (
        f"{real_seconds:.2f}s of real contact was reported as "
        f"{masks['head_contact_seconds'][0]:.2f}s"
    )


def test_defect_soc_4_tail_contact_flicker_bridges_at_realistic_centre_distance():
    f = _pair_clip(20, 7.5)  # realistic nose-on-tail-base centre separation
    channel = f.pair_nose_tail_left_to_right_cm[:, 0]
    channel[:] = 1.0
    channel[9:11] = 50.0  # 2-frame flicker
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["tail_contact"][:, 0].sum() == 20


def test_defect_soc_9_together_is_not_fragmented_by_one_contact_frame():
    f = _pair_clip(35, 4.0)
    f.speed_cm_s[:] = 0.5
    f.pair_nose_head_left_to_right_cm[:, 0] = 50.0
    f.pair_nose_head_left_to_right_cm[20, 0] = 1.0  # single noise frame
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["together"][:, 0].any(), (
        "a 1.17 s quiet 4 cm huddle was destroyed by one noisy keypoint frame"
    )


def test_defect_soc_16_chase_bridge_counts_toward_support_duration():
    frames = 120
    f = features(frames, 2)
    f.centers_cm[:, 0, 0] = np.arange(frames, dtype=float)
    f.centers_cm[:, 1, 0] = np.arange(frames, dtype=float) + 8.0
    sync(f)
    resync_kinematics(f)
    head = f.pair_nose_head_left_to_right_cm[:, 0]
    head[:] = np.nan
    head[:12] = 50.0
    head[108:] = 50.0  # only 0.8 s of triangle-valid observation, split by a 3.2 s gap
    masks = pair_rule_masks(f, confirmed_params(), component_social_config())
    assert masks["chasing"][:, 0].any(), (
        "a 4 s pursuit at 30 cm/s with a 3.2 s ID gap produced no chase label"
    )


def test_defect_soc_11_config_permits_a_single_dropout_cycle():
    SocialHeuristicConfig(attack_box_reappearance_min_cycles=1)


def test_defect_soc_3_tail_follow_suppression_targets_a_moving_leader():
    source = (SRC / "evaluation" / "confirmed_rules.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_attack_action_mask")
    body = ast.get_source_segment(source, fn)
    assert "target_speed >=" in body or "target_speed >" in body, (
        "the tail-follow guard still requires the leader to be slow; the spec "
        "requires it to be walking"
    )


def test_defect_soc_6_approach_window_has_no_zero_tolerance_gates():
    source = (SRC / "evaluation" / "confirmed_rules.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_open_approach_mask")
    body = ast.get_source_segment(source, fn)
    for token in ("visible[onset:end].all()", "directed[onset:end].all()",
                  "np.isfinite(distance[onset:end]).all()"):
        assert token not in body, f"zero-tolerance gate still present: {token}"


def test_lock_v17_approach_endpoint_window_is_scale_feasible():
    """SOC-17's original geometric assertion now passes with the v17 fitted windows.

    Historical [9, 12] cm center and 2 +/- 0.5 cm surface gates could not fit
    a 10.7 cm collinear body. The new measured ranges must retain a nonempty
    joint feasible band at both reference scales; this does not establish recall.

    CAVEAT, stated explicitly: this is a CONFIG-CONSISTENCY argument, not a
    measurement. It models a collinear head-on closure, where the actor's nose sits
    ~L/2 in front of its own centre and the target's convex hull extends ~L/2 from the
    target's centre, so the snout-to-surface gap is approximately ``d - L``. Real hulls
    are neither symmetric nor collinear, so the feasible band is shifted, not widened:
    this test checks feasibility under that explicit approximation, rather than
    claiming that all real orientations satisfy it or that fitted limits are
    independently validated physical boundaries."""
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    for body_cm in (BODY_CM, LEGACY_BODY_CM):
        best = min(
            abs((distance - body_cm) - config.approach_surface_endpoint_cm)
            for distance in np.linspace(config.dist_approach_end_min_cm,
                                        config.dist_approach_end_max_cm, 301)
        )
        assert best <= config.approach_surface_endpoint_tolerance_cm, (
            f"no centre distance in [{config.dist_approach_end_min_cm}, "
            f"{config.dist_approach_end_max_cm}] can produce a surface gap of "
            f"{config.approach_surface_endpoint_cm} +/- "
            f"{config.approach_surface_endpoint_tolerance_cm} at body length "
            f"{body_cm} cm (closest miss {best:.3f} cm)"
        )


def test_defect_soc_10_no_unreferenced_private_helpers():
    source = (SRC / "evaluation" / "confirmed_rules.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("_")}
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    repo_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in itertools.chain(SRC.rglob("*.py"), (REPO_ROOT / "scripts").rglob("*.py"))
    )
    dead = sorted(
        name for name in defined
        if name not in referenced and repo_text.count(name) <= 1
    )
    assert not dead, f"unreferenced private helpers: {dead}"


def test_defect_soc_12_pair_rule_masks_uses_its_parameters_argument():
    source = (SRC / "evaluation" / "confirmed_rules.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wrapper = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "pair_rule_masks")
    delegated = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_base_pair_rule_masks"]
    assert len(delegated) == 1
    assert isinstance(delegated[0].args[1], ast.Name) and delegated[0].args[1].id == "p"
    for name in ("_base_pair_rule_masks", "_open_approach_mask"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        args = {a.arg for a in fn.args.args}
        used = {n.value.id for n in ast.walk(fn)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        if "p" in args:
            assert "p" in used, f"{name}() accepts `p` and never reads it"


def test_defect_soc_14_policy_receipt_describes_the_shipped_rules():
    receipt = policy_receipt()
    revision = receipt["social_rule_revision"]
    assert "v13" not in revision, f"social_rule_revision is stale: {revision}"
    chase = receipt["rules"]["chase"] if "rules" in receipt else json.dumps(receipt)
    assert "0.4_s" not in chase and "3.5" in json.dumps(receipt), (
        "the receipt still documents the retired 0.4 s chase gap limit"
    )


def test_lock_soc_16b_retired_social_settings_are_explicit_warned_and_inert():
    """Retired compatibility fields are declared, warn on tuning, and change no mask."""
    config = component_social_config()
    f = _pair_clip(45, 7.5)
    f.pair_nose_head_left_to_right_cm[:, 0] = 1.0
    baseline = pair_rule_masks(f, confirmed_params(), config)
    assert baseline["head_contact"][:, 0].all(), "fixture must exercise a live rule"
    for name in (
            "chasing_track_gap_max_s", "nose_contact_gap_grace_s",
            "nose_contact_exit_distance_cm", "vel_nose_probing_min_cm_s",
            "angle_nose_probing_max_deg",
    ):
        assert name in config.retired_fields
        assert policy_receipt(config)["retired_social_config_fields"][name]["status"]
        with pytest.warns(RuntimeWarning, match="Retired social settings"):
            changed = replace(config, **{name: getattr(config, name) * 1.1})
        actual = pair_rule_masks(f, confirmed_params(), changed)
        for output in baseline:
            assert np.array_equal(actual[output], baseline[output]), (name, output)


# ==========================================================================
# 4. Group layer
# ==========================================================================
def _near_matrix(f: ExportedFeatures, neighbor_cm: float) -> np.ndarray:
    return (np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= neighbor_cm)
            & f.valid[:, f.pair_i] & f.valid[:, f.pair_j])


def _static_pile(frames: int, mice: int, *, fps: float = FPS,
                 body_cm: float = BODY_CM) -> ExportedFeatures:
    f = features(frames, mice, fps, reference_body_cm=body_cm)
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    return f


def test_lock_initial_compact_components_backfills_a_late_id_inside_the_pile():
    """v16 requirement 3.1-B. The `not compact -> break` early stop is gone and a
    track first seen inside the initial pile envelope joins the member set."""
    frames = 210
    f = _static_pile(frames, 4)
    f.valid[:3, 3] = False
    set_centers(f, [[-0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, -0.5]])
    f.centers_cm[30:, 0, 0] = np.linspace(-0.5, -14.0, frames - 30)
    f.centers_cm[45:, 1, 0] = np.linspace(0.5, 14.0, frames - 45)
    f.centers_cm[~f.valid] = np.nan
    sync(f)
    neighbor = NEIGHBOR_BODY_LENGTHS * BODY_CM
    assert (0, 1, 2, 3) in _initial_compact_components(
        f, _near_matrix(f, neighbor), neighbor)


def test_lock_dispersal_requires_two_genuinely_observed_outward_members():
    """v16 requirement 3.1-A3. A peripheral ID that merely blinks out and back
    must not create a dispersal."""
    frames = 300
    f = _static_pile(frames, 4)
    positions = np.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    set_centers(f, positions)
    f.valid[250:260, 3] = False  # peripheral blink, no real outward travel
    f.centers_cm[~f.valid] = np.nan
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    neighbor = NEIGHBOR_BODY_LENGTHS * BODY_CM
    near = _near_matrix(f, neighbor)
    members = (0, 1, 2, 3)
    from mouse_behavior.evaluation.group_event_rules import _geometry
    compact, enough, _ = _geometry(f, members, near)
    internal = np.isin(f.pair_i, members) & np.isin(f.pair_j, members)
    pair_observed = f.valid[:, f.pair_i[internal]] & f.valid[:, f.pair_j[internal]]
    distances = np.where(pair_observed, f.pair_distance_cm[:, internal], np.nan)
    phase = dispersal_phase(
        distances, enough, compact, confirmed_params(), FPS,
        centers_cm=f.centers_cm[:, members],
        member_valid=f.valid[:, members] & np.isfinite(f.centers_cm[:, members]).all(axis=2),
        initial_group_radius_cm=neighbor,
    )
    assert not phase.any(), "an ID blink was accepted as a dispersal"


def test_lock_isolation_evidence_is_zeroed_in_the_group_only_path():
    """v16 requirement 3.2-D as far as it is implemented: with a group-only
    behavior subset, a positive multi-mouse candidate zeroes isolation."""
    frames = 360
    f = _static_pile(frames, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [70.0, 70.0]])
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    table = build_candidate_table(f, [confirmed_params()], behavior_subset=GROUP_BEHAVIORS)
    assert evidence_of(table, "isolation", (103,)) == 0.0
    assert evidence_of(table, "huddle", (100, 101, 102)) > 0.0


def test_defect_grp_ev_huddle_evidence_is_compact_support_not_episode_span():
    frames = 180
    compact = np.zeros(frames, dtype=bool)
    compact[:5] = True  # 0.167 s of genuine compact evidence
    enough = compact.copy()
    observed = np.zeros((frames, 3), dtype=bool)
    observed[:5] = True
    duration = _huddle_event(compact, enough, observed, FPS, 5.0)
    assert duration < 5.0, (
        f"0.17 s of compact evidence produced a {duration:.2f} s huddle"
    )
    assert duration == pytest.approx(5 / FPS, abs=1e-9)


def test_defect_grp_4_departed_mouse_is_not_held_in_the_pile():
    """An observed departure cannot become an occlusion hold at its old position."""
    frames = 300
    f = _static_pile(frames, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [6.0, 6.0]])
    f.centers_cm[160:165, 3] = [25.0, 25.0]  # departure really observed first
    f.valid[165:, 3] = False
    f.centers_cm[~f.valid] = np.nan
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    compact, _, observed, positions, held = _huddle_geometry(
        f, (0, 1, 2, 3), _near_matrix(f, 10.0), neighbor_cm=10.0, max_gap_s=5.0)
    assert compact[:160].all(), "fixture must start as a real compact pile"
    assert observed[160:165, 3].all()
    assert np.all(positions[160:165, 3] == [25.0, 25.0])
    assert not compact[160:165].any()
    assert not held[165:, 3].any(), (
        "a mouse observed outside the pile was subsequently held in it"
    )


@pytest.mark.parametrize("pile_moves_away", [False, True])
def test_lock_grp_5_no_reappearance_hold_is_bounded_by_time_and_visible_support(pile_moves_away):
    """v17 permits forward hold without reappearance, for at most five seconds."""
    frames = 240
    f = _static_pile(frames, 3)
    set_centers(f, [[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    f.valid[10:, 2] = False  # no reappearance anywhere in the clip
    if pile_moves_away:
        f.centers_cm[70:, :2, 0] += 25.0
    f.centers_cm[~f.valid] = np.nan
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    _, _, observed, positions, held = _huddle_geometry(
        f, (0, 1, 2), _near_matrix(f, 10.0), neighbor_cm=10.0, max_gap_s=5.0)
    assert not observed[10:, 2].any()
    stop = 70 if pile_moves_away else 160  # last observed row 9 + 150 held rows
    assert held[10:stop, 2].all()
    assert np.all(positions[10:stop, 2] == [3.0, 0.0])
    assert not held[stop:, 2].any()


def test_defect_grp_6_huddle_union_is_bounded_by_the_stable_window():
    frames = 600
    f = _static_pile(frames, 4)
    f.valid[:, 2] = False
    f.valid[100:106, 2] = True  # 0.2 s pass-through
    f.valid[:240, 3] = False
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    f.centers_cm[~f.valid] = np.nan
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    candidates = group_rule_candidates(f, confirmed_params(), ())
    huddles = [c for c in candidates if c.behavior == "huddle" and c.evidence > 0]
    assert huddles, "fixture produced no huddle candidate"
    assert all(102 not in c.target_ids for c in huddles), (
        f"a 0.2 s passer-by joined the huddle union: "
        f"{[(c.target_ids, c.evidence) for c in huddles]}"
    )


def test_defect_grp_3_backfill_does_not_destroy_the_dispersal_candidate():
    frames = 300
    f = _static_pile(frames, 4)
    head = np.asarray([[-4.9, 0.0], [4.9, 0.0], [0.0, 4.9], [0.0, -13.3]])
    tail = np.asarray([[-40.0, 0.0], [40.0, 0.0], [0.0, 40.0], [0.0, -40.0]])
    set_centers(f, head, end=30)
    f.centers_cm[30:] = head + (tail - head) * (
        np.arange(frames - 30, dtype=float) / (frames - 30))[:, None, None]
    f.centers_cm[~f.valid] = np.nan
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    neighbor = NEIGHBOR_BODY_LENGTHS * BODY_CM
    assert (0, 1, 2, 3) in _initial_compact_components(
        f, _near_matrix(f, neighbor), neighbor), "back-fill itself did not fire"
    candidates = group_rule_candidates(f, confirmed_params(), ())
    assert any(c.behavior == "dispersal" for c in candidates), (
        f"four mice visibly dispersing produced no dispersal candidate; got "
        f"{[(c.behavior, c.target_ids, round(c.evidence, 2)) for c in candidates]}"
    )


def test_defect_grp_1_isolation_suppression_works_on_the_full_canonical_path():
    frames = 360
    f = _static_pile(frames, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [70.0, 70.0]])
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    table = build_candidate_table(f, [confirmed_params()])  # no behavior_subset
    assert evidence_of(table, "huddle", (100, 101, 102)) > 0.0, "no huddle evidence"
    assert evidence_of(table, "isolation", (103,)) == 0.0, (
        "isolation kept positive evidence on the 13-class path despite a huddle"
    )


def test_defect_grp_2_isolation_suppression_is_clip_level_not_profile_level():
    frames = 360
    f = _static_pile(frames, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [70.0, 70.0]])
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    base = confirmed_params()
    profiles = (replace(base, huddle_min_duration_s=5.0),
                replace(base, huddle_min_duration_s=60.0))
    table = build_candidate_table(f, list(profiles), behavior_subset=GROUP_BEHAVIORS)
    assert evidence_of(table, "isolation", (103,)) == 0.0, (
        "a non-firing profile restored isolation evidence for the whole clip"
    )


def test_defect_grp_11_genuinely_isolated_mice_survive_an_unrelated_huddle():
    frames = 360
    f = _static_pile(frames, 6)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0],
                    [80.0, 0.0], [0.0, 80.0], [80.0, 80.0]])
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    table = build_candidate_table(f, [confirmed_params()], behavior_subset=GROUP_BEHAVIORS)
    for track in (103, 104, 105):
        assert evidence_of(table, "isolation", (track,)) > 0.0, (
            f"track {track} is >20 cm from everything yet lost its isolation evidence"
        )


def test_defect_grp_12_suppression_ignores_the_allowed_behavior_filter():
    frames = 360
    f = _static_pile(frames, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [70.0, 70.0]])
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    table = build_candidate_table(f, [confirmed_params()], behavior_subset=("isolation",))
    prediction = predict_table(table, positive_evidence_model(table))
    assert prediction.behavior == "isolation", (
        f"an isolation-only request returned {prediction.behavior!r}"
    )


def test_defect_grp_7_dispersal_shim_exposes_the_hardened_signature():
    import inspect
    shim = inspect.signature(confirmed_rules_module.dispersal_phase)
    real = inspect.signature(group_module.dispersal_phase)
    missing = set(real.parameters) - set(shim.parameters)
    assert not missing, f"the shim cannot forward: {sorted(missing)}"


def test_defect_grp_17_threshold_validation_precedes_the_early_returns():
    with pytest.raises(ValueError):
        dispersal_phase(
            np.zeros((10, 3)), np.ones(10, dtype=bool), np.ones(10, dtype=bool),
            confirmed_params(), FPS,
            centers_cm=np.zeros((10, 3, 2)), member_valid=np.ones((10, 3), dtype=bool),
            minimum_visible_fraction=0.0,
        )


def test_defect_grp_10_huddle_geometry_is_computed_once_per_proposal(monkeypatch):
    frames = 120
    f = _static_pile(frames, 4)
    set_centers(f, [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    sync(f)
    f.speed_cm_s[:] = 0.0
    f.velocity_cm_s[:] = 0.0
    calls: list[tuple[tuple[int, ...], float]] = []
    original = group_module._huddle_geometry

    def counting(*args, **kwargs):
        calls.append((tuple(args[1]), float(kwargs["neighbor_cm"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(group_module, "_huddle_geometry", counting)
    proposals = group_module._proposal_sets(
        f, _near_matrix(f, NEIGHBOR_BODY_LENGTHS * BODY_CM),
        f.velocity_cm_s, f.speed_cm_s, confirmed_params(), ())
    calls.clear()
    group_rule_candidates(f, confirmed_params(), ())
    duplicates = {m for m in calls if calls.count(m) > 1}
    assert not duplicates, f"_huddle_geometry recomputed for member sets {duplicates}"
    calls.clear()
    same_radius = replace(group_module.DEFAULT_GROUP_HEURISTIC_CONFIG,
                          huddle_neighbor_distance_cm=group_module.DEFAULT_GROUP_HEURISTIC_CONFIG.neighbor_distance_cm)
    group_rule_candidates(f, confirmed_params(), (), same_radius)
    assert all(calls.count(key) == 1 for key in calls), "identical process geometry must be shared"
    assert proposals, "fixture produced no proposal sets"


# v17: retained unchanged literal-multiplier assertions as a historical lock.
# Retirement of strict-XFAIL is backed by the 16 real canonical/ranker behavior
# regressions in test_group_evidence_seconds_v17.py, including exact durations,
# unknown/held rows, quiet-tail invariance and bounded actual feature fractions.
# Common seconds units do not imply equal biological evidence or probabilities.
def test_defect_grp_8_group_evidence_is_commensurable_in_seconds():
    source = (SRC / "evaluation" / "calibrated_rule_ranker.py").read_text(encoding="utf-8")
    assert "* 1.25" not in source, "social_clustering evidence carries a 1.25 multiplier"
    assert "* 0.75" not in source, "union-huddle evidence carries a 0.75 multiplier"


# NOTE: an earlier draft carried a DEFECT SOC-20 test claiming that
# _bridge_internal_gaps (confirmed_rules.py:241-242) accepts a non-positive fps or a
# negative maximum_gap_s. Executed and XPASSed: the guard already raises ValueError
# for both. The claim came from a sub-review and was rejected on evidence.


# ==========================================================================
# 5. Parameter provenance, policy and validation-gate adequacy
# ==========================================================================
PROJECT_ROOT = REPO_ROOT.parents[1]
NEW_RUN = REPO_ROOT / "outputs" / "current_validation_20261015_v16_final"
OLD_RUN = PROJECT_ROOT / "outputs" / "current_heuristic_validation_20261014_v14"
NEW_SITE = REPO_ROOT / "outputs" / "behavior_error_review_20261015_v16_final"
OLD_SITE = PROJECT_ROOT / "outputs" / "behavior_error_review_20261013"
NEW_STATIC = REPO_ROOT / "outputs" / "static_locomotion_v16_20261015_final"
OLD_STATIC = PROJECT_ROOT / "outputs" / "static_locomotion_v15_20261014"


def _read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"artifact not available: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _layer_errors(root: pathlib.Path) -> dict[str, int]:
    out = {}
    social = _read_csv(root / "social" / "social_validation_predictions.csv")
    out["social"] = sum(1 for r in social if not _flag(r["strict_behavior_and_id_correct"]))
    group = _read_csv(root / "group" / "scored_predictions.csv")
    out["group"] = sum(1 for r in group if not _flag(r["requested_joint_correct"]))
    individual = _read_csv(root / "individual" / "validation_rows.csv")
    out["individual"] = sum(1 for r in individual if not _flag(r["strict_correct"]))
    return out


@xfail_defect(reason="DEFECT PROV-3 - the frozen physical constants are derived from a "
                     "DIFFERENT body-length reference than the one in force. "
                     "static_max_radius_cm = static_transient_max_displacement_cm = "
                     "1.3375 cm = 0.125 x 10.7 cm, and the spec's neighbour radius "
                     "13.375 cm = 1.25 x 10.7 cm, but ExportedFeatures.reference_body_cm "
                     "and individual_rule_masks() both default to 7.918 cm, which gives "
                     "NEIGHBOR_BODY_LENGTHS * body = 9.8975 cm. The spec text, the "
                     "shipped default and the frozen constants therefore describe three "
                     "different scales, and no test or receipt binds them together. "
                     "Because approach's endpoint window is a surface gap measured in "
                     "absolute cm, this is a live correctness risk, not cosmetics.")
def test_defect_prov_3_frozen_radii_match_the_active_reference_body_length():
    config = DEFAULT_INDIVIDUAL_CONFIG
    expected_radius = 0.125 * LEGACY_BODY_CM
    assert config.static_max_radius_cm == pytest.approx(expected_radius), (
        f"static_max_radius_cm={config.static_max_radius_cm} was derived from a "
        f"{config.static_max_radius_cm / 0.125:.3f} cm body, but the shipped default "
        f"reference_body_cm is {LEGACY_BODY_CM}"
    )
    assert NEIGHBOR_BODY_LENGTHS * LEGACY_BODY_CM == pytest.approx(13.375)


@xfail_defect(reason="DEFECT PROV-1 / GRP-15 group_event_rules.py:24 - "
                     "GROUP_RULE_REVISION is still "
                     "'confirmed_group_rules_v14_total_support_whole_component' even "
                     "though v16 added _huddle_geometry, rewrote "
                     "_initial_compact_components, extended the dispersal_phase "
                     "signature and relaxed _clustering_event. The string is published "
                     "by policy_receipt() and written into validation receipts, so a v16 "
                     "group run is indistinguishable by receipt from a v14 run. The "
                     "individual layer WAS bumped to 2026-10-15-static-locomotion-v16 and "
                     "the social layer was not (SOC-14): two of three layers cannot be "
                     "version-separated, which defeats the spec's own step-2/step-3 "
                     "receipt checks.")
def test_defect_prov_1_every_layer_revision_identifies_the_v16_change_set():
    assert "v14" not in GROUP_RULE_REVISION, GROUP_RULE_REVISION
    receipt = policy_receipt()
    assert "v13" not in receipt["social_rule_revision"]
    assert "2026-10-15" in INDIVIDUAL_RULE_REVISION  # the one layer that was bumped


@xfail_defect(reason="DEFECT PROV-2 - at least four different 'isolation distance' "
                     "values are simultaneously in force: "
                     "confirmed_rules.GROUP_ISOLATION_DISTANCE_CM = 20.0, "
                     "configs/default.yaml isolation_min_distance_cm = 26.75, "
                     "hierarchical_core.isolation_min_distance_cm = 26.75, and the "
                     "rule_profiles sweep 20.0625 .. 66.875. The v16 run receipt says "
                     "'fixed_policy_20_cm_no_validation_tuning'. There is no single "
                     "receipt-backed active source, so a reader cannot tell which "
                     "threshold produced a given prediction.")
def test_defect_prov_2_isolation_distance_has_a_single_source_of_truth():
    texts = {
        "confirmed_rules.py": (SRC / "evaluation" / "confirmed_rules.py").read_text("utf-8"),
        "hierarchical_core.py": (SRC / "evaluation" / "hierarchical_core.py").read_text("utf-8"),
        "default.yaml": (REPO_ROOT / "configs" / "default.yaml").read_text("utf-8"),
    }
    found = {}
    for name, text in texts.items():
        for line in text.splitlines():
            if "isolation" in line and ("20.0" in line or "26.75" in line or "20.0625" in line):
                found.setdefault(name, []).append(line.strip())
    distinct = {name: len(lines) for name, lines in found.items()}
    assert len(found) <= 1, f"competing isolation distances: {distinct}"


@xfail_defect(reason="DEFECT PROV-4 - the v16 review site's manifest is not "
                     "self-consistent: `sourceFiles` records paths under "
                     "outputs/current_validation_20261015_v16_final/ while the per-item "
                     "`validationSource` strings recorded for the same items point at "
                     "outputs/current_validation_20261015_v16/ (no _final). The two runs "
                     "happen to hold identical predictions here, but the manifest can no "
                     "longer prove which run produced the site, which is exactly the "
                     "'改了新代码却读回旧结果' failure the spec's section 二 red lines "
                     "exist to prevent. Step-4's check only inspects `sourceFiles` and so "
                     "passes.")
def test_defect_prov_4_review_manifest_provenance_is_self_consistent():
    manifest_path = NEW_SITE / "review_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"artifact not available: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    def run_root(path: str) -> str | None:
        parts = pathlib.PurePosixPath(path.replace("\\", "/")).parts
        if "outputs" in parts:
            index = parts.index("outputs")
            if index + 1 < len(parts):
                return parts[index + 1]
        return None

    declared = {entry["path"] for entry in manifest["sourceFiles"].values()}
    roots = {run_root(path) for path in declared} - {None}
    used = {item["validationSource"] for item in manifest["items"]}
    used_roots = {run_root(path) for path in used} - {None}
    assert roots and used_roots, f"could not resolve validation roots: {declared}"
    assert used_roots <= roots, (
        f"items cite validation roots {sorted(used_roots)} that sourceFiles does not "
        f"declare {sorted(roots)}"
    )


@xfail_defect(reason="DEFECT POL-2 exported_dataset_models.py:231 vs "
                     "individual_rules.py:28 vs configs/default.yaml:573 - the same "
                     "quantity has three values: HeuristicParameters.locomotion_stop_gap_s "
                     "defaults to 0.5 (the v15 behaviour), "
                     "IndividualHeuristicConfig.locomotion_pause_max_s defaults to 1.0 and "
                     "default.yaml sets locomotion_stop_gap_seconds: 1.0. Any caller that "
                     "builds HeuristicParameters() without loading the yaml - including "
                     "the shipped v16 regression suite, which uses "
                     "HeuristicParameters(confirmed_rules_v1=1.0) - silently runs the "
                     "pre-v16 pause window while reporting the v16 revision string.")
def test_defect_pol_2_parameter_defaults_match_the_shipped_config():
    import yaml
    config_path = REPO_ROOT / "configs" / "default.yaml"
    if not config_path.exists():
        pytest.skip(f"artifact not available: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                hit = find(value, key)
                if hit is not None:
                    return hit
        return None

    shipped = find(raw, "locomotion_stop_gap_seconds")
    assert shipped is not None, "locomotion_stop_gap_seconds missing from default.yaml"
    assert HeuristicParameters().locomotion_stop_gap_s == pytest.approx(float(shipped))
    assert DEFAULT_INDIVIDUAL_CONFIG.locomotion_pause_max_s == pytest.approx(float(shipped))


@xfail_defect(reason="DEFECT POL-1 calibrated_rule_ranker.py:671-675 - the comment on "
                     "line 671 states that grooming/rearing must not be ranker "
                     "candidates before Phase B, but line 672 iterates "
                     "INDIVIDUAL_BEHAVIORS = ('locomotion','stationary','grooming',"
                     "'rearing') and seeds all four. Empirically the v16 individual run "
                     "emits rearing for 142 of 570 per-visual-id rows (25%) and grooming "
                     "for 2. NOTE, verified: this predates v16 (the v14 run emitted "
                     "rearing 126 times), so it is a standing policy violation and NOT a "
                     "v16 regression - it is reported here because the standing decision "
                     "is that grooming/rearing stay audited, disabled evidence.")
def test_defect_pol_1_posture_screens_are_not_canonical_predictions():
    rows = _read_csv(NEW_RUN / "individual" / "per_visual_id_predictions.csv")
    predicted = {r["predicted_behavior"] for r in rows if r.get("predicted_behavior")}
    posture = predicted & {"grooming", "rearing"}
    assert not posture, (
        f"disabled posture screens are canonical predictions: {sorted(posture)} "
        f"(rearing={sum(1 for r in rows if r['predicted_behavior'] == 'rearing')} rows)"
    )


@xfail_defect(reason="DEFECT EMP-1 - 5 of the 13 canonical behaviors are NEVER predicted "
                     "anywhere on the v16 joint validation: approach (support 28), "
                     "together (6), chase (4), isolation (8), group_locomotion (1). That "
                     "is 47 of 247 scored samples and 47 of the 167 like-for-like errors "
                     "(28%). social_clustering is predicted exactly once against a "
                     "support of 17 and scores 0 correct; grooming is predicted twice "
                     "against a support of 8 and scores 0 correct; rearing is the "
                     "predicted label on 142 of 570 per-visual-id rows against a "
                     "clip-level support of 4 and scores 0 correct. Structurally "
                     "unreachable classes cannot be recovered by tuning the thresholds of "
                     "the classes that do fire. The spec's baseline table hides this "
                     "because it was built from three older vintages in which some of "
                     "these classes still fired. See DEFECT EMP-5 for the same fact measured at the "
                     "rule layer, which separates rule loss from Top-1 ranking loss.")
def test_defect_emp_1_every_canonical_behavior_is_reachable():
    predicted: set[str] = set()
    for rows, column in (
        (_read_csv(NEW_RUN / "individual" / "per_visual_id_predictions.csv"), "predicted_behavior"),
        (_read_csv(NEW_RUN / "social" / "social_validation_predictions.csv"), "predicted_behavior"),
        (_read_csv(NEW_RUN / "group" / "scored_predictions.csv"), "predicted_behavior"),
    ):
        predicted |= {r[column] for r in rows if r.get(column)}
    unreachable = sorted(set(CANONICAL_BEHAVIORS) - predicted)
    assert not unreachable, f"canonical behaviors never predicted: {unreachable}"


@xfail_defect(reason="DEFECT EMP-2 - the spec's step-5 before/after table is not a valid "
                     "comparison. The 114-error baseline site draws its individual layer "
                     "from outputs/static_locomotion_validation_20261012/report.json "
                     "(47 rows) and scores 11 behavior classes, while the v16 site draws "
                     "its individual layer from "
                     "current_validation_20261015_v16/individual/per_visual_id_"
                     "predictions.csv (570 rows over 59 clips) and scores 15 classes, "
                     "adding grooming/rearing/isolation/group_locomotion that the "
                     "baseline never counted. Different script, different denominator, "
                     "different granularity, different taxonomy. The published 114 -> 154 "
                     "delta is therefore not attributable to the v16 code change.")
def test_defect_emp_2_baseline_and_current_sites_are_like_for_like():
    old_path = OLD_SITE / "review_manifest.json"
    new_path = NEW_SITE / "review_manifest.json"
    if not (old_path.exists() and new_path.exists()):
        pytest.skip("baseline or current review manifest not available")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    assert set(old["behaviorCounts"]) == set(new["behaviorCounts"]), (
        f"taxonomy differs: baseline {sorted(set(new['behaviorCounts']) - set(old['behaviorCounts']))} "
        f"new-only, {sorted(set(old['behaviorCounts']) - set(new['behaviorCounts']))} baseline-only"
    )
    old_layers = {item["layer"]: item["validationSource"] for item in old["items"]}
    new_layers = {item["layer"]: item["validationSource"] for item in new["items"]}
    assert old_layers.keys() == new_layers.keys()
    mismatched = {layer for layer in old_layers
                  if pathlib.Path(old_layers[layer]).name != pathlib.Path(new_layers[layer]).name}
    assert not mismatched, f"layers scored from different artifacts: {sorted(mismatched)}"


@xfail_defect(reason="DEFECT EMP-3 (BLOCKING, process) - the individual-layer acceptance "
                     "gate cannot detect a quality regression. Running the SAME script "
                     "over the SAME 157 frozen samples, non-empty predictions fell from "
                     "70 (v15: locomotion 61 + stationary 9) to 40 (v16: locomotion 38 + "
                     "stationary 2), a 43% coverage collapse, yet "
                     "verification.json reports passed=true and summary.json reports "
                     "rule_revision=2026-10-15-static-locomotion-v16. Both of the spec's "
                     "step-2 checks are satisfied by a run that detects strictly less. "
                     "The gate validates provenance strings and input hashes, never "
                     "detection coverage.")
def test_defect_emp_3_the_verification_gate_detects_a_coverage_regression():
    new_path = NEW_STATIC / "verification.json"
    old_csv = OLD_STATIC / "predictions.csv"
    new_csv = NEW_STATIC / "predictions.csv"
    if not (new_path.exists() and old_csv.exists() and new_csv.exists()):
        pytest.skip("static/locomotion artifacts not available")
    verification = json.loads(new_path.read_text(encoding="utf-8"))

    def coverage(path: pathlib.Path) -> int:
        rows = _read_csv(path)
        return sum(1 for r in rows if (r.get("predicted_behavior") or "").strip())

    old_covered, new_covered = coverage(old_csv), coverage(new_csv)
    assert verification.get("passed") is True
    assert new_covered >= old_covered, (
        f"prediction coverage fell {old_covered} -> {new_covered} of "
        f"{len(_read_csv(new_csv))} samples while verification.passed stayed true"
    )


@xfail_defect(reason="DEFECT EMP-4 - measured against its true immediate predecessor "
                     "(current_heuristic_validation_20261014_v14, identical 59/115/73 "
                     "denominators), v16 improved the social layer by 11 errors "
                     "(98 -> 87: nose_head_contact -6, nose_tail_contact -4, attack -1), "
                     "left the individual layer bit-identical (33 -> 33, strict_accuracy "
                     "0.4406779661016949 in both runs, locomotion -2 offset by stationary "
                     "+2), and REGRESSED the group layer on every metric (41 -> 47 "
                     "errors; requested joint precision 0.640 -> 0.510, joint recall "
                     "0.438 -> 0.356, behavior precision 0.880 -> 0.824, isolation recall "
                     "4/8 -> 0/8, huddle 15/33 -> 17/33). Net across layers: 172 -> 167.")
def test_defect_emp_4_v16_does_not_regress_any_layer_against_v14():
    if not (OLD_RUN.exists() and NEW_RUN.exists()):
        pytest.skip("v14 or v16 joint validation artifacts not available")
    before, after = _layer_errors(OLD_RUN), _layer_errors(NEW_RUN)
    regressed = {layer: (before[layer], after[layer])
                 for layer in before if after[layer] > before[layer]}
    assert not regressed, f"layers that regressed v14 -> v16: {regressed}"


# ==========================================================================
# 6. Conformance with codex_v16_complete_heuristics_and_posture_plan.md
#    (the merged spec: v16 heuristics + grooming/rearing construction)
#
#    That document states in section 三 that the module specifications were
#    "已在沙箱完成实现与测试验证" and in section 四 that
#    tests/unit/test_heuristic_fixes_v16.py exists with 8 passing tests.
#    Every claim below was checked mechanically against this repository.
# ==========================================================================
POSTURE_PLAN = "codex_v16_complete_heuristics_and_posture_plan.md"


@xfail_defect(reason=f"DEFECT DOC-1 - {POSTURE_PLAN} section 四 states that "
                     "tests/unit/test_heuristic_fixes_v16.py was created in the sandbox "
                     "and that all 8 core scenarios pass 100%. No such file exists in the "
                     "target repository; tests/unit/ contains only README.md, and the "
                     "whole snapshot ships exactly one pre-existing test file "
                     "(tests/regression/test_heuristics_v16.py, 8 tests). PACKAGE_INFO.json "
                     "records excluded_test_related_file_count=59, so the packaging step "
                     "dropped the suite. Section 五 step 1 therefore runs a pytest gate "
                     "over 8 tests, and the 8 claimed posture tests are unverifiable here.")
def test_defect_doc_1_claimed_posture_unit_test_file_exists():
    expected = REPO_ROOT / "tests" / "unit" / "test_heuristic_fixes_v16.py"
    assert expected.exists(), f"missing: {expected}"
    claimed = (
        "test_individual_locomotion_pause_and_dropout_bridging",
        "test_individual_curled_static_and_missing_frames",
        "test_individual_grooming_7kp_and_box_with_nose_occlusion",
        "test_individual_open_field_rearing_7kp_and_box",
        "test_social_attack_long_middle_id_dropout_anchored_by_start_end",
        "test_social_contact_short_gap_bridge_and_tail_over_head_duration",
        "test_group_huddle_asynchronous_id_persistence_and_isolation_fallback",
        "test_group_dispersal_occluded_member_backfill",
    )
    source = expected.read_text(encoding="utf-8")
    missing = [name for name in claimed if f"def {name}" not in source]
    assert not missing, f"claimed tests absent: {missing}"


def test_defect_post_1_individual_rules_accept_box_morphology():
    import inspect
    parameters = inspect.signature(individual_rule_masks).parameters
    assert "boxes_cm" in parameters, (
        f"individual_rule_masks signature: {list(parameters)}"
    )


def test_defect_post_2_posture_operators_are_implemented():
    source = (SRC / "behavior" / "individual_rules.py").read_text(encoding="utf-8")
    for name in ("_robust_posterior_and_head", "_box_morphology_and_wall"):
        assert f"def {name}" in source, f"{name} is not defined"


def test_defect_post_3_posture_does_not_require_all_seven_keypoints():
    source = (SRC / "behavior" / "individual_rules.py").read_text(encoding="utf-8")
    assert "np.isfinite(points).all(axis=(2, 3))" not in source, (
        "posture still requires all 7 keypoints finite on every frame"
    )


@xfail_defect(reason="DEFECT POST-4 exported_dataset_prediction.py:933-937 - the comment "
                     "justifying the posture slots says 'identity_scores 含 "
                     "grooming/rearing 占位槽, scores 初始化必须覆盖 INDIVIDUAL_BEHAVIORS, "
                     "否则 KeyError'. The stated rationale is false: CANONICAL_BEHAVIORS "
                     "already contains all four INDIVIDUAL_BEHAVIORS, so the extra "
                     "splat is redundant and no KeyError is possible without it. The "
                     "slots are also not placeholders - lines 93-102 populate them with "
                     "real _qualifying_score values and line 944 promotes any score > 0 "
                     "into `detected`, which is how rearing became the predicted label on "
                     "142 of 570 per-visual-id rows while scoring 0 of 4 at clip level. MEASURED: "
                     "rearing precision fell 0.125 -> 0.091 and false positives rose 7 -> 10 "
                     "between v14 and v16 (target_id_conditioned_metrics.json).")
def test_defect_post_4_posture_slot_comment_rationale_is_accurate():
    assert set(INDIVIDUAL_BEHAVIORS) - set(CANONICAL_BEHAVIORS), (
        "INDIVIDUAL_BEHAVIORS is already a subset of CANONICAL_BEHAVIORS, so the "
        "'otherwise KeyError' comment does not hold and the slots are live, not placeholders"
    )




def test_v21_cross_gap_speed_factor_is_retired_and_cannot_supply_chase_travel():
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert "chasing_cross_gap_speed_factor" in config.retired_fields
    source = inspect.getsource(confirmed_rules_module._chase_mask)
    assert "chasing_cross_gap_speed_factor" not in source
    # Functional guards for missing time/path/net are in test_semantics_v21.

def test_defect_doc_3_static_posture_mask_exposes_the_stationary_override():
    import inspect
    parameters = inspect.signature(
        __import__("mouse_behavior.behavior.static_rule", fromlist=["x"])
        .static_posture_eligible_mask
    ).parameters
    for name in ("allow_shortened_when_stationary", "stationary_mask"):
        assert name in parameters, f"missing parameter: {name}; got {list(parameters)}"


def test_defect_grp_18_region_gap_survives_one_degraded_recipient():
    from mouse_behavior.evaluation.group_event_rules import _region_gap
    frames = 12
    f = features(frames, 3)
    keypoints = np.zeros((frames, 3, 7, 2), dtype=float)
    keypoints[:, 1, 0] = (3.0, 0.0)
    keypoints[:, 1, 1] = (2.6, 0.4)
    keypoints[:, 1, 2] = (2.6, -0.4)
    keypoints[:, 1, 3:] = np.nan          # recipient 1 keeps only 3 of 7 points
    keypoints[:, 2, 0] = (0.0, 3.0)
    keypoints[:, 2, 1] = (0.4, 3.4)
    keypoints[:, 2, 2] = (-0.4, 3.4)
    keypoints[:, 2, 3] = (0.0, 2.4)
    keypoints[:, 2, 4] = (0.2, 2.0)
    keypoints[:, 2, 5] = (-0.2, 2.0)
    keypoints[:, 2, 6] = (0.0, 1.8)
    f.collective_keypoints_cm = keypoints
    gap = _region_gap(f, 0, (1, 2))
    assert np.isfinite(gap).any(), (
        "one recipient below the 4-point floor voided the region gap for the whole group"
    )


def test_lock_individual_exclusion_order_is_grooming_rearing_locomotion_stationary():
    """Spec 2.2.4's four-class strict order IS honoured, even though the posture
    definitions themselves are not the spec's. Locking it so a later edit cannot
    silently reorder the waterfall."""
    source = (SRC / "behavior" / "individual_rules.py").read_text(encoding="utf-8")
    assert "& ~grooming" in source, "rearing must exclude grooming (Step 2 on ~grooming)"
    assert "locomotion &= ~posture" in source, "locomotion must exclude posture (Step 3)"
    assert "stationary &= ~locomotion" in source, "stationary must exclude locomotion (Step 4)"
    assert '_qualify(stationary & ~posture' in source, "stationary must exclude posture"


def test_lock_attack_endpoint_distance_has_a_single_active_source():
    """The receipt and actual near gate use the same behavior-specific bound."""
    config = DEFAULT_SOCIAL_HEURISTIC_CONFIG
    assert config.dist_attack_max_cm == 7.321547985076904
    receipt = policy_receipt()
    assert receipt["social_heuristic_config"]["dist_attack_max_cm"] == config.dist_attack_max_cm
    bound = config.dist_attack_max_cm
    for separation in (bound, bound + .001):
        f = _pair_clip(60, separation)
        motion = np.zeros(60)
        motion[10:20] = np.arange(10) * 0.2
        motion[20:] = 1.8
        f.centers_cm[:, 0, 0] = motion
        f.centers_cm[:, 1, 0] = motion + separation
        sync(f)
        # Isolate an exact physical-distance input from subtraction roundoff.
        f.pair_distance_cm[:] = separation
        resync_kinematics(f)
        f.collective_boxes_cm = np.ones((60, 2, 4))
        f.collective_boxes_cm[20:25] = np.nan
        # Isolate the distance gate; the fixture's measured speed is 6 cm/s.
        single_cycle = replace(config, attack_box_reappearance_min_cycles=1,
                               vel_attack_min_cm_s=4.0)
        raw = confirmed_rules_module._attack_action_mask(f, 0, single_cycle)
        assert bool(raw.any()) is (separation == bound), separation




# ==========================================================================
# 7. Segment-level ("片段里识别出片段") metrics
#
#    The acceptance criterion is NOT whole-clip Top-1. It is: inside the
#    annotated clip, does the rule layer emit the target behavior on the
#    correct target, as a time segment? Three artifacts measure this:
#      individual -> individual/target_id_conditioned_metrics.json
#                    (conditioned on the approved visual target ID)
#      social     -> social/social_rule_gate_metrics.json
#                    `true_pair_positive_rule_candidates` = the true behavior
#                    fired on the exact true pair, before any Top-1 competition
#      group      -> group/training_candidate_coverage.json
#                    `target_candidate` = a candidate with the right behavior
#                    AND the right ID set was enumerated (TRAIN split, n=123)
#    Comparing those against the Top-1 diagonal separates RULE LOSS (the
#    segment was never detected) from RANKING LOSS (it was detected and then
#    stolen by another pair/group competing for the single clip-level slot).
# ==========================================================================
def _metrics(root: pathlib.Path, rel: str):
    path = root / rel
    if not path.exists():
        pytest.skip(f"artifact not available: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


SOCIAL_BEHS = ("approach", "together", "chase", "attack",
               "nose_head_contact", "nose_tail_contact")


def _social_rule_and_top1(root: pathlib.Path) -> dict[str, dict[str, int]]:
    gate = _metrics(root, "social/social_rule_gate_metrics.json")
    per = {r["behavior"]: r for r in _read_csv(root / "social/social_per_behavior_precision_recall.csv")}
    out = {}
    for behavior in SOCIAL_BEHS:
        out[behavior] = {
            "support": int(gate[behavior]["support"]),
            "rule": int(gate[behavior]["true_pair_positive_rule_candidates"]),
            "exact": int(per[behavior]["exact_behavior_target_id_tp"]),
            "wrong_ids": int(per[behavior]["correct_behavior_wrong_target_ids"]),
        }
    return out


def test_lock_individual_locomotion_is_fully_recovered_on_the_correct_target():
    """The v16 module-1 pause bridge DID work at segment level, and the clip
    Top-1 metric hides it. Conditioned on the approved visual target ID,
    locomotion went recall 0.903 -> 1.000 (false negatives 3 -> 0) and
    precision 0.824 -> 0.939, while the unconditioned clip Top-1 accuracy did
    not move at all (0.4406779661016949 in both runs). Locking the win."""
    after = _metrics(NEW_RUN, "individual/target_id_conditioned_metrics.json")
    locomotion = after["per_behavior"]["locomotion"]
    assert locomotion["false_negative"] == 0
    assert locomotion["recall"] == pytest.approx(1.0)
    assert locomotion["precision"] >= 0.90
    before = _metrics(OLD_RUN, "individual/target_id_conditioned_metrics.json")
    assert locomotion["recall"] > before["per_behavior"]["locomotion"]["recall"]


@xfail_defect(reason="DEFECT EMP-5 - at segment level the social rule layer never "
                     "emits approach, together or chase on the true pair at all: "
                     "true_pair_positive_rule_candidates = 0/28, 0/6 and 0/4 in BOTH "
                     "v14 and v16 (coverage 0.000). This is RULE loss, not ranking "
                     "loss - there is no candidate for the ranker to steal. 38 of 115 "
                     "validation clips (33%) are unreachable before Top-1 is even "
                     "considered, and the merged spec's own section 3.4 requires the "
                     "fix (relax `visible` to 80%) that was not implemented. Overall "
                     "social rule coverage is 18/115 -> 32/115.")
def test_defect_emp_5_social_rule_layer_emits_every_target_behavior():
    after = _social_rule_and_top1(NEW_RUN)
    dead = {b: f"{after[b]['rule']}/{after[b]['support']}"
            for b in SOCIAL_BEHS if after[b]["rule"] == 0}
    assert not dead, f"behaviors the rule layer never emits on the true pair: {dead}"


@xfail_defect(reason="DEFECT EMP-6 - at segment level the group layer REGRESSED. "
                     "target_candidate_rate (right behavior AND right ID set "
                     "enumerated, train split n=123) fell 0.4797 -> 0.3171. Per "
                     "behavior: isolation 24/33 -> 2/33 (0.727 -> 0.061), dispersal "
                     "14/27 -> 11/27 (0.519 -> 0.407), group_locomotion 0/7 and "
                     "social_clustering 0/3 unchanged. Only huddle improved, 21/53 -> "
                     "26/53 (0.396 -> 0.491). The isolation collapse happens at "
                     "CANDIDATE GENERATION, not at ranking: GRP-11 zeroes every "
                     "isolation candidate whenever any multi-mouse candidate has "
                     "evidence > 0, so 31 correct candidates are destroyed before the "
                     "ranker sees them.")
def test_defect_emp_6_group_candidate_coverage_does_not_regress():
    before = _metrics(OLD_RUN, "group/training_candidate_coverage.json")
    after = _metrics(NEW_RUN, "group/training_candidate_coverage.json")
    assert after["target_candidate_rate"] >= before["target_candidate_rate"], (
        f"overall {before['target_candidate_rate']:.4f} -> "
        f"{after['target_candidate_rate']:.4f}"
    )
    regressed = {
        b: (before["per_behavior"][b]["target_candidate"],
            after["per_behavior"][b]["target_candidate"])
        for b in before["per_behavior"]
        if after["per_behavior"][b]["target_candidate"] < before["per_behavior"][b]["target_candidate"]
    }
    assert not regressed, f"behaviors losing candidates (before, after): {regressed}"


@xfail_defect(reason="DEFECT EMP-7 - ranking theft grew. nose_head_contact: the rule "
                     "layer emitted the true behavior on the true pair in 22 clips "
                     "(up from 13), but exact behavior+ID Top-1 captured only 18, so 4 "
                     "detections were stolen by another pair competing for the single "
                     "clip-level slot (v14: 13 emitted, 12 captured, 1 stolen). Of the "
                     "9 new detections the relaxation bought, only 6 reached Top-1. "
                     "A further 4 clips are correct-behavior/wrong-pair. This is the "
                     "spec's root cause #2 and it is now measurable: clip-level Top-1 "
                     "is the wrong output shape for a clip that contains several "
                     "sequential social bouts.")
def test_defect_emp_7_top1_does_not_discard_detected_segments():
    after = _social_rule_and_top1(NEW_RUN)
    stolen = {b: after[b]["rule"] - after[b]["exact"] for b in SOCIAL_BEHS}
    total_stolen = sum(v for v in stolen.values() if v > 0)
    assert total_stolen == 0, f"detected then lost at Top-1: {stolen}"


@xfail_defect(reason="DEFECT IND-6 - the module-1 pause bridge was paid for with "
                     "stationary recall. Conditioned on the approved visual target ID, "
                     "stationary went precision 0.615 -> 0.545 and recall 0.500 -> "
                     "0.375, false negatives 8 -> 10, while locomotion false negatives "
                     "went 3 -> 0. This is the measured consequence of DEFECT IND-2: "
                     "locomotion_pause_max_s (1.0) == static_min_duration_s (1.0), so a "
                     "stationary bout of exactly the minimum qualifying duration that "
                     "sits between two walks is filled in by the bridge at "
                     "individual_rules.py:327 and then deleted from `stationary` at "
                     "line 328. Rearing precision also fell 0.125 -> 0.091 (false "
                     "positives 7 -> 10), the measured cost of DEFECT POST-4.")
def test_defect_ind_6_stationary_recall_is_not_the_price_of_locomotion_recall():
    before = _metrics(OLD_RUN, "individual/target_id_conditioned_metrics.json")
    after = _metrics(NEW_RUN, "individual/target_id_conditioned_metrics.json")
    regressed = {}
    for behavior in ("stationary", "grooming", "rearing"):
        b, a = before["per_behavior"][behavior], after["per_behavior"][behavior]
        if a["recall"] < b["recall"] or a["precision"] < b["precision"]:
            regressed[behavior] = (
                f"P {b['precision']:.3f}->{a['precision']:.3f} "
                f"R {b['recall']:.3f}->{a['recall']:.3f} "
                f"FN {b['false_negative']}->{a['false_negative']} "
                f"FP {b['false_positive']}->{a['false_positive']}"
            )
    assert not regressed, regressed


# ==========================================================================
# 8. Segment-level acceptance (>=70% coverage), the two accuracies, and the
#    scale-era defect found while establishing that baseline.
#
#    Criterion agreed with the user: a sample counts as detected when the
#    predicted mask for the annotated behavior covers >= 70% of the clip's
#    frames. Two accuracies are reported:
#      A1 = behavior correct, target ID ignored (wrong / too many / too few
#           are all forgiven) -> these go to the manual-review queue
#      A2 = behavior correct AND the target ID set matches exactly
#    A1 - A2 is the population that is currently scored as an error but whose
#    behavior is actually right.
# ==========================================================================
SEG_RUN = REPO_ROOT / "outputs" / "segment_level_20261016_v16"
SEG_SUMMARY = SEG_RUN / "segment_level_summary.json"
SEG_BASELINE = {"a1_hits": 106, "a2_hits": 96, "n_scored": 247, "manual_review_queue": 10}


def _segment_summary():
    if not SEG_SUMMARY.exists():
        pytest.skip(f"segment-level baseline not available: {SEG_SUMMARY}")
    return json.loads(SEG_SUMMARY.read_text(encoding="utf-8"))


def test_lock_segment_scorer_exists_with_the_agreed_threshold():
    script = REPO_ROOT / "scripts" / "score_segment_level_v16.py"
    assert script.exists(), f"missing scorer: {script}"
    source = script.read_text(encoding="utf-8")
    assert "--overlap-threshold" in source
    assert "0.7" in source, "the agreed 70% coverage threshold is not the default"
    assert "FileExistsError" in source, "the scorer must refuse to overwrite an output dir"


def test_lock_segment_level_baseline_is_recorded():
    """Freeze the v16 segment-level baseline so any later run must beat it.

    Measured on the frozen v16 code (all five rule files hash-verified
    unchanged during the 280 s run): A1 = 106/247 = 42.9%, A2 = 96/247 = 38.9%,
    manual-review queue = 10. Per domain: individual A1 41/59, social A1 21/115,
    group A1 44/73 (group coverage is an evidence/clip-duration proxy because
    the group layer emits durations, not masks).
    """
    summary = _segment_summary()
    overall = summary["overall"]
    for key, expected in SEG_BASELINE.items():
        assert overall[key] == expected, f"{key}: {overall[key]} != baseline {expected}"
    assert overall["a2_hits"] <= overall["a1_hits"]
    assert summary["threshold"] == pytest.approx(0.70)
    assert summary["group_coverage_is_evidence_over_clip_duration_proxy"] is True


@xfail_defect(reason="DEFECT SEG-1 - the coverage distribution is BIMODAL, so "
                     "retuning the 70% threshold buys almost nothing and this is a "
                     "rewrite problem, not a retune problem. Of 247 samples: 136 sit at "
                     "exactly 0 coverage and 87 at [0.9,1.0]; only 15 land between 0.3 "
                     "and 0.9 and just 7 in [0.5,0.7). Lowering the threshold to 50% "
                     "would add at most 7 samples (2.8 points). The 136 zero-coverage "
                     "samples are concentrated in the classes whose rules never fire at "
                     "all: approach 28/28 silent, attack 20/22, chase 4/4, together 4/6, "
                     "social_clustering 17/17.")
def test_defect_seg_1_failures_are_not_threshold_sensitive():
    summary = _segment_summary()
    bands = summary["overall"]["coverage_band_histogram"]
    near_miss = sum(int(bands.get(b, 0)) for b in ("[0.3,0.5)", "[0.5,0.7)"))
    total = int(summary["overall"]["n_scored"])
    assert near_miss / total >= 0.10, (
        f"only {near_miss}/{total} samples are near misses; the failure mode is "
        f"all-or-nothing, so the threshold is not the lever"
    )


@xfail_defect(reason="DEFECT ERA-1 (BLOCKING, systemic, in NEITHER spec document) - "
                     "scale_migration.rescale_frozen_source_scale multiplies every "
                     "sample's cm_per_pixel by NEW_BODY_LENGTH_CM/OLD_BODY_LENGTH_CM = "
                     "7.918/10.7 = 0.74 and rewrites assumed_mouse_body_length_cm to "
                     "7.918, so at runtime ALL distances and speeds are 0.74x the frozen "
                     "receipt values. Ratio-of-body-length gates survive this because "
                     "reference_body_cm is rescaled too. But 41 ABSOLUTE cm / cm_s / "
                     "cm_s2 thresholds across HeuristicParameters, SocialHeuristicConfig, "
                     "GroupHeuristicConfig and IndividualHeuristicConfig are exact "
                     "multiples of the OLD 10.7 cm body length (k = 0.125, 0.25, 0.375, "
                     "0.625, 0.75, 1.0, 1.5, 1.875, 2.5, 3.0, 3.125, 3.75, 5.0, 5.625, "
                     "6.25, 10.625) and were never re-derived, so each is 1.35x "
                     "mis-scaled against the quantity it is compared to. Minimums became "
                     "1.35x too STRICT (vel_chasing_min_cm_s and "
                     "vel_chasing_target_min_cm_s = 26.75 = 2.5x10.7; "
                     "chasing_min_net/path_displacement_cm = 10.7; isolation_distance_cm = "
                     "20.0625/26.75; approach_min_closing_speed_cm_s = 2.675) and maximums "
                     "1.35x too PERMISSIVE (huddle_distance_cm = 8.025 = 0.75x10.7; "
                     "static_max_radius_cm = static_transient_max_displacement_cm = "
                     "1.3375 = 0.125x10.7; together_max_distance_cm = 8.025; "
                     "clustering_stability_radius_cm = 1.3375). This alone accounts for "
                     "chase 0/4, the isolation candidate collapse 24/33 -> 2/33, and "
                     "huddle over-firing. Fixing it means re-deriving frozen physical "
                     "constants, which BOTH specs forbid ('不改动各行为核心物理定义参数') "
                     "and which invalidates the fitted ranker models - so it is escalated "
                     "as a decision, not patched unilaterally.")
def test_defect_era_1_absolute_thresholds_match_the_active_scale_era():
    from mouse_behavior.evaluation.group_heuristic_config import (
        DEFAULT_GROUP_HEURISTIC_CONFIG,
    )
    from mouse_behavior.preprocessing.scale_resolution import LEGACY_REFERENCE_BODY_CM
    old_era, new_era = 10.7, LEGACY_REFERENCE_BODY_CM
    assert new_era == pytest.approx(7.918), "the migration target body length changed"
    stale = []
    for label, obj in (("HeuristicParameters", HeuristicParameters()),
                       ("SocialHeuristicConfig", DEFAULT_SOCIAL_HEURISTIC_CONFIG),
                       ("GroupHeuristicConfig", DEFAULT_GROUP_HEURISTIC_CONFIG),
                       ("IndividualHeuristicConfig", DEFAULT_INDIVIDUAL_CONFIG)):
        for field, value in vars(obj).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not field.endswith(("_cm", "_cm_s", "_cm_s2")):
                continue
            ratio = float(value) / old_era
            if abs(ratio - round(ratio, 3)) < 1e-9 and 0.03 <= abs(ratio) <= 40:
                stale.append(f"{label}.{field}={float(value):g} ({round(ratio, 4):g}x10.7)")
    assert not stale, f"{len(stale)} thresholds still in the 10.7 cm era: {stale}"


@xfail_defect(reason="DEFECT IND-7 (BLOCKING) - the locomotion pause bridge has no "
                     "observation bound and the final `_qualify(locomotion & ~posture, "
                     "...)` never re-intersects with `observed`, so short gaps CHAIN and "
                     "the mask claims frames where the mouse was never detected. Measured "
                     "on the real human-approved rearing sample "
                     "(2bc41eee9c6844d466038347): visual track 3384 is observed in only "
                     "10 of 60 frames yet is labelled locomotion for 33 frames, 29 of "
                     "which are rows where valid == False. Track 3379: 40 observed -> 44 "
                     "claimed (6 fabricated). Track 3762 in the grooming sample: 33 "
                     "observed -> 40 claimed (7 fabricated). Because line 328 then does "
                     "`stationary &= ~locomotion`, fabricated locomotion also ERASES real "
                     "stationary frames - a second mechanism, alongside IND-2, for the "
                     "measured stationary recall drop 0.500 -> 0.375.")
def test_defect_ind_7_masks_never_claim_frames_the_mouse_was_not_detected_in():
    frames = 60
    centers = np.full((frames, 1, 2), np.nan)
    valid = np.zeros((frames, 1), dtype=bool)
    for block in range(6):  # 6 two-frame sightings, 8-frame gaps between them
        start = block * 10
        valid[start:start + 2, 0] = True
        centers[start:start + 2, 0, 0] = np.asarray([start, start + 1], dtype=float)
        centers[start:start + 2, 0, 1] = 0.0
    # Guard: the fixture must actually produce locomotion, otherwise the
    # assertion below passes vacuously (this is the trap that made the first
    # draft of this test XPASS - the y coordinate was left NaN, so
    # `observed & isfinite(centers).all(axis=2)` was False on every frame).
    result = individual_rule_masks(centers, valid, FPS)
    assert result.masks["locomotion"][:, 0].any(), "fixture produced no locomotion at all"
    locomotion = result.masks["locomotion"][:, 0]
    claimed_unobserved = int(locomotion[~valid[:, 0]].sum())
    claimed_observed = int(locomotion[valid[:, 0]].sum())
    assert claimed_unobserved <= claimed_observed, (
        f"{claimed_unobserved} frames labelled locomotion where the mouse was never "
        f"detected, against only {claimed_observed} observed frames"
    )


CALIB_SAMPLES = (
    ("7f180b7276de46e3", "Grooming", "3b4f5b31fd6f543422fcf6d3",
     {3458: "grooming", 3524: "stationary", 3746: "stationary", 3575: "locomotion"}),
    ("9d00b2ae2e01b39a", "Rearing", "2bc41eee9c6844d466038347",
     {3382: "rearing", 6: "stationary", 3141: "locomotion"}),
)
DATASET_ROOT = REPO_ROOT / "behavior_confirmed_dataset_cleaned_20261014_original_video_v3" / "dataset"
SCALES_ROOT = REPO_ROOT.parents[1] / "outputs" / "confirmed_calibration_v1_20261010" / "dataset_v2" / "scales"


def _load_calibrated_sample(source_key: str, behavior_dir: str, sample_id: str):
    """Load one real human-approved posture sample in the MIGRATED scale era.

    Production never sees the raw receipt scale: verified_source_scale runs
    rescale_frozen_source_scale, which multiplies cm_per_pixel by 7.918/10.7 and
    rewrites assumed_mouse_body_length_cm to 7.918. Tuning against the raw era
    would optimise numbers the pipeline never produces.
    """
    from mouse_behavior.evaluation.exported_dataset import load_exported_features
    from mouse_behavior.evaluation.scale_migration import rescale_frozen_source_scale
    sample_dir = DATASET_ROOT / behavior_dir / sample_id
    receipt = SCALES_ROOT / f"{source_key}.json"
    if not (sample_dir / "tracks.json").exists() or not receipt.exists():
        pytest.skip(f"calibration sample unavailable: {sample_dir} / {receipt}")
    raw = json.loads(receipt.read_text(encoding="utf-8"))["scale_config"]
    scale, migration = rescale_frozen_source_scale(
        raw, REPO_ROOT / "configs" / "scale" / "rig_a.json")
    assert migration["factor"] == pytest.approx(7.918 / 10.7, abs=1e-9)
    meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    features = load_exported_features(
        sample_dir / "tracks.json", fps=float(meta["clip"]["fps"]),
        max_tracks=64, scale_config=scale)
    assert features.reference_body_cm == pytest.approx(7.918, abs=1e-6), (
        "the sample did not load in the migrated era; posture thresholds derived "
        "from the spec's 10.7 cm arithmetic would be 1.35x off"
    )
    return features


@xfail_defect(reason="DEFECT POST-5 - the merged spec's section 二/三 presents a "
                     "measured 7-track physical table and a claimed run output "
                     "(track 3458 grooming 40/40, 3524 stationary 40/40, 3746 stationary "
                     "38/39, 3575 locomotion 40/40, 3382 rearing 47, track 6 stationary "
                     "60/60, 3141 locomotion 60/60) as already achieved. Measured with "
                     "the code actually in this repository: 3458 = ALL ZEROS (no behavior "
                     "at all), 3524 = all zeros, 3746 = all zeros, 3382 = rearing 18 (not "
                     "47), track 6 = stationary 50 (not 60); only the two locomotion "
                     "tracks match. individual_rule_masks takes no boxes_cm argument, "
                     "_robust_posterior_and_head and _box_morphology_and_wall do not "
                     "exist, and pose_observed still requires all 7 keypoints finite. The "
                     "spec's section 四 heading 'already implemented and verified in the "
                     "sandbox' is false for this repository. This test is the acceptance "
                     "gate for module 2: it must pass on the annotated track of each "
                     "sample with ZERO false positives on the same-cage static and "
                     "locomotion tracks.")
def test_defect_post_5_real_calibration_samples_reproduce_the_spec_results():
    from mouse_behavior.behavior.individual_rules import individual_rule_masks
    for source_key, behavior_dir, sample_id, expected in CALIB_SAMPLES:
        features = _load_calibrated_sample(source_key, behavior_dir, sample_id)
        result = individual_rule_masks(
            features.centers_cm, features.valid, features.fps,
            keypoints_cm=features.collective_keypoints_cm,
            boxes_cm=getattr(features, "collective_boxes_cm", None),
            reference_body_cm=features.reference_body_cm,
        )
        counts = {}
        for slot, track in enumerate(features.track_ids):
            observed = int(features.valid[:, slot].sum())
            if observed == 0:
                continue
            counts[int(track)] = {name: int(result.masks[name][:, slot].sum())
                                  for name in ("stationary", "locomotion", "grooming", "rearing")}
        for track, behavior in expected.items():
            assert track in counts, f"{sample_id}: track {track} absent"
            observed = int(features.valid[:, list(features.track_ids).index(track)].sum())
            hits = counts[track][behavior]
            others = sum(v for k, v in counts[track].items() if k != behavior)
            assert hits >= 0.7 * observed, (
                f"{sample_id} track {track}: expected {behavior} on >=70% of "
                f"{observed} frames, got {hits} (other classes {others})"
            )
            assert others == 0, (
                f"{sample_id} track {track}: {others} frames of competing labels "
                f"alongside {behavior}={hits}"
            )
