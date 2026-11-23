"""Observed formation prefixes and state memory, independent of annotations.

Unknown rows may preserve an episode but never contribute event time, path,
velocity, or net displacement. A measured non-dense row always breaks a dense
state episode, even when surrounded by otherwise qualifying observations.
"""
from __future__ import annotations

import numpy as np

from .group_heuristic_config import GroupHeuristicConfig
from .screening_signals import runs


def _unknown_envelope(state, observed, fps, maximum_gap_s):
    envelope = state.copy()
    for low, high in runs(~state):
        if (low > 0 and high < len(state) and state[low - 1] and state[high]
                and not observed[low:high].any()
                and (high - low) / fps <= maximum_gap_s + 1e-12):
            envelope[low:high] = True
    return envelope


def _occluded_members_verified(
    safe_centers: np.ndarray,
    observed: np.ndarray,
    occ_dense: np.ndarray,
    low: int,
    high: int,
    lookback: int,
    env_cm: float,
    fps: float,
) -> bool:
    """Verify partially occluded members entered near the observed pile without distant contradiction."""
    if observed[low:high].all():
        return True
    obs_cnt = observed.sum(axis=1)
    pile_valid_idx = np.flatnonzero(occ_dense[low:high] & (obs_cnt[low:high] >= 2)) + low
    if not len(pile_valid_idx):
        return False
    pile_centers = safe_centers[pile_valid_idx].sum(axis=1) / obs_cnt[pile_valid_idx, None]
    search_lo = max(0, low - lookback)
    edge_limit = env_cm + 5.0
    for col in range(observed.shape[1]):
        if observed[low:high, col].all():
            continue
        if not observed[search_lo:high, col].any():
            return False
        for gap_lo, gap_hi in runs(~observed[:, col]):
            if gap_hi <= low or gap_lo >= high:
                continue
            last_t = gap_lo - 1
            if last_t < search_lo:
                return False
            others_mask = observed[last_t].copy()
            others_mask[col] = False
            if others_mask.any():
                ref = safe_centers[last_t, others_mask].mean(axis=0)
            else:
                nearest_pos = int(np.argmin(np.abs(pile_valid_idx - last_t)))
                ref = pile_centers[nearest_pos]
            if float(np.linalg.norm(safe_centers[last_t, col] - ref)) > edge_limit:
                return False
    return True


def formation_masks(centers, observed, compact, fps, config: GroupHeuristicConfig,
                    neighbor_cm, legacy_mask, legacy_motion):
    """Qualify joining or mutual convergence, then emit only actual observations.

    The recipient center supplies relative motion, not a continuous dense-seed
    veto. A final observed dense state confirms the retrospective process. New
    onsets must precede that state; translation of an existing pile cannot
    qualify this channel. The older observed-arrival channel remains available.
    Supports bounded top-down pile occlusion when k >= 3 while emitting only
    actually observed rows.
    """
    n_frames, n_members, _ = centers.shape
    finite_centers = np.isfinite(centers).all(axis=2)
    observed = observed & finite_centers
    safe_centers = np.where(observed[:, :, None], centers, 0.0)

    known = observed.all(axis=1)
    obs_cnt = observed.sum(axis=1)
    env_cm = neighbor_cm * (1.0 + 0.20 * max(0, n_members - 3))
    mean_all = safe_centers.sum(axis=1) / np.maximum(obs_cnt[:, None], 1)
    dist_to_mean = np.linalg.norm(safe_centers - mean_all[:, None], axis=2)
    dist_masked = np.where(observed, dist_to_mean, 0.0)
    radius = dist_masked.max(axis=1)
    obs_rad = np.where(obs_cnt >= 2, dist_masked.max(axis=1), np.inf)

    dense = compact & known & (radius <= env_cm)
    minimum = max(1, int(np.ceil(config.clustering_final_stability_s * fps - 1e-9)))
    state_envelope = _unknown_envelope(dense, known, fps, config.clustering_unknown_gap_s)
    episodes = [(low, high) for low, high in runs(state_envelope)
                if any(b - a >= minimum for a, b in runs(dense[low:high]))
                and dense[low:high].mean() >= config.clustering_min_observed_fraction]

    output, support = legacy_mask.copy(), legacy_motion.copy()

    # Vectorized bounded occlusion-tolerant dense check (k >= 3): >=2 observed connected & <= env_cm.
    obs2 = obs_cnt >= 2
    occ_dense = dense.copy()
    occ_edge_limit = max(neighbor_cm * 1.35, env_cm)
    partial_sel = np.flatnonzero(obs2 & (obs_rad <= env_cm) & ~known)
    if len(partial_sel):
        pts = safe_centers[partial_sel]
        obs_s = observed[partial_sel]
        diffs = pts[:, :, None, :] - pts[:, None, :, :]
        pair_ok = (
            obs_s[:, :, None] & obs_s[:, None, :]
            & (np.sum(diffs * diffs, axis=3) <= occ_edge_limit * occ_edge_limit + 1e-9)
        )
        reach = pair_ok.astype(np.int8)
        steps_needed = max(1, int(np.ceil(np.log2(max(n_members, 2)))))
        for _ in range(steps_needed):
            reach = ((reach @ reach) > 0).astype(np.int8)
        first_obs = np.argmax(obs_s, axis=1)
        connected = (reach[np.arange(len(partial_sel)), first_obs].astype(bool) == obs_s).all(axis=1)
        occ_dense[partial_sel] = connected

    occ_env = _unknown_envelope(occ_dense, obs_cnt > 0, fps, config.clustering_unknown_gap_s)
    occ_episodes = [(low, high) for low, high in runs(occ_env)
                    if any(b - a >= minimum for a, b in runs(occ_dense[low:high]))
                    and occ_dense[low:high].mean() >= config.clustering_min_observed_fraction]

    if not episodes and not occ_episodes:
        return output & known, support & known

    for low, high in episodes:
        if legacy_mask[low:high].any():
            output[low:high] |= dense[low:high]

    # Lookback contexts use obs_cnt >= 2 (bridging only brief 1-mouse-visible pile moments <= gap),
    # while never bridging total-missing gaps or teleports.
    ctx_mask = obs2.copy()
    for gap_lo, gap_hi in runs(~obs2):
        if (gap_lo > 0 and gap_hi < n_frames and obs2[gap_lo - 1] and obs2[gap_hi]
                and (obs_cnt[gap_lo:gap_hi] > 0).all()
                and (gap_hi - gap_lo) / fps <= config.clustering_unknown_gap_s + 1e-12):
            ctx_mask[gap_lo:gap_hi] = True
    base_ctx_start = np.full(n_frames, -1, dtype=int)
    for a, b in runs(ctx_mask):
        base_ctx_start[a:b] = a
    lookback = int(np.floor(config.clustering_motion_lookback_s * fps + 1e-9))

    def prefix(values):
        return np.concatenate((np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)))

    def _run_channel(ep_list, state_mask, *, occluded_channel: bool):
        ep_valid_flags = [
            (not occluded_channel)
            or (
                not (output[low:high].any() and known[low:high].all())
                and _occluded_members_verified(
                    safe_centers, observed, state_mask, low, high, lookback, env_cm, fps
                )
            )
            for low, high in ep_list
        ]
        if not any(ep_valid_flags):
            return

        for column in range(n_members):
            actor = safe_centers[:, column]
            obs_a = observed[:, column]
            others = np.delete(safe_centers, column, axis=1)
            obs_o = np.delete(observed, column, axis=1)
            o_cnt = obs_o.sum(axis=1)
            valid_pair = obs_a & (o_cnt >= 1)

            common_o = obs_o[1:] & obs_o[:-1]
            c_cnt = common_o.sum(axis=1)
            adj = obs2[1:] & obs2[:-1] & obs_a[1:] & obs_a[:-1] & (c_cnt >= 1)
            if not adj.any():
                continue

            recipient = others.sum(axis=1) / np.maximum(o_cnt[:, None], 1)
            offset = recipient - actor
            rec_prev = (others[:-1] * common_o[:, :, None]).sum(axis=1) / np.maximum(c_cnt[:, None], 1)
            rec_curr = (others[1:] * common_o[:, :, None]).sum(axis=1) / np.maximum(c_cnt[:, None], 1)
            steps = np.where(adj[:, None], actor[1:] - actor[:-1], 0.)
            relative = np.where(adj[:, None], (actor[1:] - rec_curr) - (actor[:-1] - rec_prev), 0.)
            lengths = np.linalg.norm(steps, axis=1)
            common_off_prev = np.where(adj[:, None], rec_prev - actor[:-1], 0.)
            common_off_curr = np.where(adj[:, None], rec_curr - actor[1:], 0.)
            step_drop = np.where(
                adj,
                np.linalg.norm(common_off_prev, axis=1) - np.linalg.norm(common_off_curr, axis=1),
                0.,
            )

            denominator = np.linalg.norm(relative, axis=1) * np.linalg.norm(common_off_prev, axis=1)
            cosine = np.divide((relative * common_off_prev).sum(axis=1), denominator,
                               out=np.full(len(steps), np.nan), where=denominator > 1e-9)
            toward = np.r_[False, adj & (lengths * fps >= config.clustering_actor_speed_cm_s)
                           & (cosine >= config.clustering_toward_cosine)
                           & (step_drop > 0.)]
            if not toward.any():
                continue

            actor_ctx = ctx_mask.copy()
            broken_actor = False
            for gap_lo, gap_hi in runs(~obs_a):
                if gap_lo > 0 and gap_hi < n_frames and obs_a[gap_lo - 1] and obs_a[gap_hi]:
                    if float(np.linalg.norm(actor[gap_hi] - actor[gap_lo - 1])) > neighbor_cm:
                        actor_ctx[gap_lo:gap_hi] = False
                        broken_actor = True
            if broken_actor:
                col_ctx_start = np.full(n_frames, -1, dtype=int)
                for a, b in runs(actor_ctx):
                    col_ctx_start[a:b] = a
            else:
                col_ctx_start = base_ctx_start

            counts = np.r_[0, np.cumsum(adj)]
            paths, vectors, relatives = prefix(lengths), prefix(steps), prefix(relative)
            drops = prefix(step_drop)
            toward_counts = np.r_[0, np.cumsum(toward[1:])]

            for (low, high), ep_ok in zip(ep_list, ep_valid_flags):
                if not ep_ok:
                    continue
                if not toward[max(0, low - lookback):high].any():
                    continue
                arrivals = [i + low for a, b in runs(state_mask[low:high])
                            for i in range(a, b - minimum + 1)]
                ep_min_rad = float(obs_rad[low:high][state_mask[low:high]].min()) if state_mask[low:high].any() else np.inf
                for arrival in arrivals:
                    if not valid_pair[arrival]:
                        continue
                    context = int(col_ctx_start[arrival])
                    if context < 0:
                        continue
                    earliest = max(context, int(arrival) - lookback)
                    if not toward[earliest + 1:arrival + 1].any():
                        continue
                    onsets = np.flatnonzero(toward[earliest + 1:arrival + 1]) + earliest
                    onsets = onsets[~state_mask[onsets] & valid_pair[onsets]]
                    if not len(onsets):
                        continue
                    path = paths[arrival] - paths[onsets]
                    net = np.linalg.norm(vectors[arrival] - vectors[onsets], axis=1)
                    relative_net = relatives[arrival] - relatives[onsets]
                    den = np.linalg.norm(relative_net, axis=1) * np.linalg.norm(offset[onsets], axis=1)
                    direction = np.divide((relative_net * offset[onsets]).sum(axis=1), den,
                                          out=np.full(len(onsets), -1.), where=den > 1e-9)
                    measured_steps = counts[arrival] - counts[onsets]
                    directed_steps = toward_counts[arrival] - toward_counts[onsets]

                    base_qualified = (
                        (path > config.clustering_actor_min_path_cm + 1e-9)
                        & (net + 1e-9 >= config.clustering_actor_min_displacement_cm)
                        & (drops[arrival] - drops[onsets] + 1e-9 >= config.clustering_distance_drop_cm)
                        & (direction >= config.clustering_toward_cosine)
                        & (path * fps + 1e-9 >= config.clustering_actor_speed_cm_s * measured_steps)
                        & (directed_steps / fps + 1e-9 >= .15)
                    )
                    strict_toward = directed_steps + 1e-9 >= config.clustering_min_toward_fraction * measured_steps
                    if occluded_channel and n_members >= 4 and not known[low:high].all():
                        rad_onset = obs_rad[onsets]
                        rad_collapse = (
                            np.isfinite(rad_onset)
                            & (rad_onset > env_cm)
                            & ((rad_onset - min(float(obs_rad[arrival]), ep_min_rad)) >= 5.0)
                            & (obs_rad[arrival] <= 0.75 * rad_onset)
                        )
                        occ_toward = (
                            (directed_steps + 1e-9 >= 0.50 * measured_steps)
                            & (direction >= 0.85)
                            & rad_collapse
                        )
                        qualified = base_qualified & (strict_toward | occ_toward)
                    else:
                        qualified = base_qualified & strict_toward

                    valid_onsets = onsets[qualified]
                    if not len(valid_onsets):
                        continue
                    onset = int(valid_onsets[0])
                    if occluded_channel:
                        cand_span = known[onset:max(arrival + 1, high)].copy()
                        if low > onset and arrival + 1 < low:
                            cand_span[arrival + 1 - onset:low - onset] = False
                        if high > onset:
                            cand_span[max(0, low - onset):high - onset] |= dense[max(low, onset):high]
                        if int(cand_span.sum()) < minimum:
                            continue
                    output[onset:arrival + 1] |= known[onset:arrival + 1]
                    output[low:high] |= dense[low:high]
                    support[onset:arrival + 1] |= toward[onset:arrival + 1]
                    break

    if episodes:
        _run_channel(episodes, dense, occluded_channel=False)
    if occ_episodes:
        _run_channel(occ_episodes, occ_dense, occluded_channel=True)

    # Preserve the invariant even for callers supplying an old serialized mask.
    return output & known, support & known
