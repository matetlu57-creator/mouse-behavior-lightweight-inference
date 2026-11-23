"""Deterministic unit tests for Stage 1 group behavior recognition upgrades."""
from __future__ import annotations

import numpy as np

from tests.regression import test_heuristics_v16_audit as audit
from mouse_behavior.evaluation import group_event_rules as group
from mouse_behavior.evaluation.exported_dataset_models import HeuristicParameters


def _sync_features(f):
    f.centers_cm[~f.valid] = np.nan
    audit.sync(f)
    audit.resync_kinematics(f)
    return f


def _sc_near(f, neighbor_cm: float = 10.0) -> np.ndarray:
    return np.isfinite(f.pair_distance_cm) & (f.pair_distance_cm <= neighbor_cm)


def _frames_to_mask(frames, n_frames: int) -> np.ndarray:
    out = np.zeros(n_frames, dtype=bool)
    for start, stop in frames:
        out[start:stop] = True
    return out


def test_preexisting_cluster_plus_outsider_vetoed_after_half_second():
    """Already-clustered >=3 mice plus an arriving outsider (onset >= 0.5s) is vetoed."""
    f = audit.features(160, 4)
    f.centers_cm[:, 0] = [0.0, 0.0]
    f.centers_cm[:, 1] = [2.0, 0.0]
    f.centers_cm[:, 2] = [0.0, 2.0]
    f.centers_cm[:45, 3] = [26.0, 0.0]
    f.centers_cm[45:96, 3, 0] = np.linspace(26.0, 3.0, 51)
    f.centers_cm[45:96, 3, 1] = 1.0
    f.centers_cm[96:, 3] = [3.0, 1.0]
    _sync_features(f)

    p = HeuristicParameters(confirmed_rules_v1=1.0)
    cands = group.group_rule_candidates(f, p, (), behaviors=("social_clustering",))
    assert not cands, f"Pre-existing cluster + outsider should be vetoed, got {cands}"


def test_half_second_guard_preserves_frame_zero_clustering():
    """Clips starting at frame 0 (< max(10, 0.5*fps)) are not rejected by history veto."""
    f = audit.features(150, 4)
    f.centers_cm[:, 0] = [0.0, 0.0]
    f.centers_cm[:, 1] = [2.0, 0.0]
    f.centers_cm[:, 2] = [0.0, 2.0]
    f.centers_cm[:51, 3, 0] = np.linspace(24.0, 3.0, 51)
    f.centers_cm[:51, 3, 1] = 1.0
    f.centers_cm[51:, 3] = [3.0, 1.0]
    _sync_features(f)

    p = HeuristicParameters(confirmed_rules_v1=1.0)
    cands = group.group_rule_candidates(f, p, (), behaviors=("social_clustering",))
    expected_ids = tuple(int(t) for t in f.track_ids)
    assert any(c.behavior == "social_clustering" and c.target_ids == expected_ids for c in cands)


def test_occlusion_tolerant_convergence_and_negative_guards():
    """Recover bounded pile occlusion while strictly blocking teleports, rigid translation, and all-unknown gaps."""
    # 1. k=4 mutual convergence with occluded 4th track inside the final pile.
    f_occ = audit.features(150, 4)
    initial = np.array([[-20.0, 0.0], [20.0, 0.0], [0.0, -20.0], [0.0, 20.0]])
    final = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]])
    frac = np.minimum(np.arange(150) / 70.0, 1.0)
    f_occ.centers_cm[:] = initial + frac[:, None, None] * (final - initial)
    # Hide mouse 3 inside the settled pile so no contiguous all-observed run reaches 0.5s.
    f_occ.valid[72:125, 3] = False
    _sync_features(f_occ)
    proc_occ = group._clustering_process(
        f_occ, (0, 1, 2, 3), _sc_near(f_occ), f_occ.velocity_cm_s, f_occ.speed_cm_s,
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10.0, max_gap_s=5.0,
    )
    assert proc_occ.evidence > 0.0
    obs_mask = _frames_to_mask(proc_occ.frames, len(f_occ.valid))
    assert not obs_mask[72:125].any(), "Missing track frames must never be credited"

    # 2. Teleport guard must still block unobserved jumps.
    f_tp = audit.features(150, 3)
    f_tp.centers_cm[:, 1] = [0.0, 0.0]
    f_tp.centers_cm[:, 2] = [0.0, 2.0]
    f_tp.centers_cm[:50, 0] = [22.0, 0.0]
    f_tp.centers_cm[50:, 0] = [3.0, 0.0]
    f_tp.valid[49:51, 0] = False
    _sync_features(f_tp)
    assert not group._clustering_process(
        f_tp, (0, 1, 2), _sc_near(f_tp), f_tp.velocity_cm_s, f_tp.speed_cm_s,
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10.0, max_gap_s=5.0,
    ).frames

    # 3. Rigid translation of existing pile must remain rejected.
    f_tr = audit.features(150, 3)
    f_tr.centers_cm[:] = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    f_tr.centers_cm[:, :, 0] += np.arange(150)[:, None] * 2.0
    _sync_features(f_tr)
    assert not group._clustering_process(
        f_tr, (0, 1, 2), _sc_near(f_tr), f_tr.velocity_cm_s, f_tr.speed_cm_s,
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10.0, max_gap_s=5.0,
    ).frames

    # 4. All-unknown gap splitting short dense segments must remain rejected.
    f_gap = audit.features(150, 3)
    f_gap.centers_cm[:, 1] = [0.0, 0.0]
    f_gap.centers_cm[:, 2] = [0.0, 2.0]
    f_gap.centers_cm[:, 0] = [3.0, 0.0]
    f_gap.centers_cm[:55, 0] = [22.0, 0.0]
    f_gap.valid[60:63, :] = False
    f_gap.centers_cm[75:, 0] = [22.0, 0.0]
    _sync_features(f_gap)
    assert not group._clustering_process(
        f_gap, (0, 1, 2), _sc_near(f_gap), f_gap.velocity_cm_s, f_gap.speed_cm_s,
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10.0, max_gap_s=5.0,
    ).frames


def test_parallel_group_locomotion_not_mislabeled_as_social_clustering():
    """Cohesive parallel group travel without sharp radial collapse suppresses Social clustering."""
    f = audit.features(150, 3)
    t = np.arange(150, dtype=float)
    y = t * 0.8  # 24 cm/s forward travel (~120 cm net)
    f.centers_cm[:, 0, 0] = np.linspace(-10.5, -2.5, 150)
    f.centers_cm[:, 1, 0] = 0.0
    f.centers_cm[:, 2, 0] = np.linspace(10.5, 2.5, 150)
    f.centers_cm[:, :, 1] = y[:, None]
    _sync_features(f)

    proc = group._clustering_process(
        f, (0, 1, 2), _sc_near(f), f.velocity_cm_s, f.speed_cm_s,
        group.DEFAULT_GROUP_HEURISTIC_CONFIG, neighbor_cm=10.0, max_gap_s=5.0,
    )
    assert proc.evidence == 0.0

    p = HeuristicParameters(confirmed_rules_v1=1.0)
    cands = group.group_rule_candidates(
        f, p, (), behaviors=("social_clustering", "group_locomotion")
    )
    assert not any(c.behavior == "social_clustering" for c in cands)
    expected_ids = tuple(int(t) for t in f.track_ids)
    assert any(c.behavior == "group_locomotion" and c.target_ids == expected_ids for c in cands)


def test_synchronous_process_splits_stationary_pauses_into_separate_bouts():
    """Stationary pauses > 1.5s inside long sequences must split Group locomotion into separate bouts."""
    f = audit.features(360, 3)
    x = np.zeros(360, dtype=float)
    x[:110] = np.linspace(0.0, 55.0, 110)
    x[110:210] = 55.0  # 3.33s stationary pause (> 1.5s)
    x[210:320] = np.linspace(55.0, 110.0, 110)
    x[320:] = 110.0

    f.centers_cm[:, 0, 0] = x
    f.centers_cm[:, 0, 1] = 0.0
    f.centers_cm[:, 1, 0] = x
    f.centers_cm[:, 1, 1] = 3.0
    f.centers_cm[:, 2, 0] = x
    f.centers_cm[:, 2, 1] = 6.0
    _sync_features(f)

    p = HeuristicParameters(
        confirmed_rules_v1=1.0,
        collective_active_speed_cm_s=group.DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_active_speed_cm_s,
        group_locomotion_min_direction_similarity=group.DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_direction_similarity,
        group_locomotion_max_distance_cm=group.DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_max_pair_distance_cm,
    )
    proc = group._synchronous_process(
        f, (0, 1, 2), f.velocity_cm_s, f.speed_cm_s, p, group.DEFAULT_GROUP_HEURISTIC_CONFIG
    )
    assert len(proc.frames) == 2, f"Expected 2 split locomotion bouts, got {proc.frames}"
    assert proc.frames[0][1] <= 115
    assert proc.frames[1][0] >= 205
    assert proc.evidence < 8.0


def test_synchronous_process_and_candidates_emit_pure_python_int_bounds_for_json_merge():
    """Frame and support intervals must use pure Python int bounds so merge_events/json.dumps succeeds."""
    import json
    from mouse_behavior.canonical_video_pipeline import merge_events

    f = audit.features(150, 3)
    x = np.linspace(0.0, 75.0, 150)
    f.centers_cm[:, 0, 0] = x
    f.centers_cm[:, 0, 1] = 0.0
    f.centers_cm[:, 1, 0] = x
    f.centers_cm[:, 1, 1] = 3.0
    f.centers_cm[:, 2, 0] = x
    f.centers_cm[:, 2, 1] = 6.0
    _sync_features(f)

    p = HeuristicParameters(
        confirmed_rules_v1=1.0,
        collective_active_speed_cm_s=group.DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_active_speed_cm_s,
        group_locomotion_min_direction_similarity=group.DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_direction_similarity,
        group_locomotion_max_distance_cm=group.DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_max_pair_distance_cm,
    )
    proc = group._synchronous_process(
        f, (0, 1, 2), f.velocity_cm_s, f.speed_cm_s, p, group.DEFAULT_GROUP_HEURISTIC_CONFIG
    )
    assert proc.frames
    assert type(proc.evidence) is float
    for interval in proc.frames:
        assert type(interval[0]) is int and type(interval[1]) is int
    for interval in proc.support_frames or ():
        assert type(interval[0]) is int and type(interval[1]) is int

    cands = group.group_rule_candidates(
        f, HeuristicParameters(confirmed_rules_v1=1.0), (), behaviors=("group_locomotion", "social_clustering")
    )
    gl_cands = [c for c in cands if c.behavior == "group_locomotion"]
    assert gl_cands
    events = []
    for cand in gl_cands:
        for tid in cand.target_ids:
            assert type(tid) is int
        for s, e in cand.frame_intervals or ():
            assert type(s) is int and type(e) is int
            events.append({
                "canonical_behavior": cand.behavior,
                "mouse_ids": list(cand.target_ids),
                "start_frame": s,
                "end_frame": e,
                "observed_all_participants_frame_intervals": [[s, e]],
                "canonical_support_frame_intervals": [
                    [a, b] for a, b in (cand.support_frame_intervals or ())
                ],
                "inference_parent_contexts": [
                    json.dumps({"start_frame": s, "end_frame": e}, sort_keys=True)
                ],
            })
        for a, b in cand.support_frame_intervals or ():
            assert type(a) is int and type(b) is int

    merged = merge_events(events)
    assert len(merged) == 1
    serialized = json.dumps(merged, sort_keys=True)
    assert '"group_locomotion"' in serialized
