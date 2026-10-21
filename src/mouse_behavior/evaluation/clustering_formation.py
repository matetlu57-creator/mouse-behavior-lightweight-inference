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


def formation_masks(centers, observed, compact, fps, config: GroupHeuristicConfig,
                    neighbor_cm, legacy_mask, legacy_motion):
    """Qualify joining or mutual convergence, then emit only actual observations.

    The recipient center supplies relative motion, not a continuous dense-seed
    veto. A final observed dense state confirms the retrospective process. New
    onsets must precede that state; translation of an existing pile cannot
    qualify this channel. The older observed-arrival channel remains available.
    """
    known = observed.all(axis=1)
    radius = np.linalg.norm(centers - centers.mean(axis=1)[:, None], axis=2).max(axis=1)
    dense = compact & known & (radius <= neighbor_cm)
    minimum = max(1, int(np.ceil(config.clustering_final_stability_s * fps - 1e-9)))
    state_envelope = _unknown_envelope(dense, known, fps, config.clustering_unknown_gap_s)
    episodes = [(low, high) for low, high in runs(state_envelope)
                if any(b - a >= minimum for a, b in runs(dense[low:high]))
                and dense[low:high].mean() >= config.clustering_min_observed_fraction]
    output, support = legacy_mask.copy(), legacy_motion.copy()
    if not episodes:
        return output, support
    for low, high in episodes:
        if legacy_mask[low:high].any():
            output[low:high] |= dense[low:high]
    contexts = runs(_unknown_envelope(known, known, fps, config.clustering_unknown_gap_s))
    adjacent = known[1:] & known[:-1]
    counts = np.r_[0, np.cumsum(adjacent)]
    lookback = int(np.floor(config.clustering_motion_lookback_s * fps + 1e-9))

    def prefix(values):
        return np.concatenate((np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)))

    for column in range(centers.shape[1]):
        actor = centers[:, column]
        recipient = np.delete(centers, column, axis=1).mean(axis=1)
        offset = recipient - actor
        distance = np.linalg.norm(offset, axis=1)
        steps = np.where(adjacent[:, None], np.diff(actor, axis=0), 0.)
        relative = np.where(adjacent[:, None], np.diff(actor - recipient, axis=0), 0.)
        lengths = np.linalg.norm(steps, axis=1)
        denominator = np.linalg.norm(relative, axis=1) * np.linalg.norm(offset[:-1], axis=1)
        cosine = np.divide((relative * offset[:-1]).sum(axis=1), denominator,
                           out=np.full(len(steps), np.nan), where=denominator > 1e-9)
        toward = np.r_[False, adjacent & (lengths * fps >= config.clustering_actor_speed_cm_s)
                       & (cosine >= config.clustering_toward_cosine) & (np.diff(distance) < 0.)]
        paths, vectors, relatives = prefix(lengths), prefix(steps), prefix(relative)
        drops = prefix(np.where(adjacent, distance[:-1] - distance[1:], 0.))
        toward_counts = np.r_[0, np.cumsum(toward[1:])]
        for low, high in episodes:
            # First confirmation requires a CONTIGUOUS measured final run.
            # Memory can retain confirmed state; it cannot create stability
            # by summing shorter runs on opposite sides of an unknown gap.
            arrivals = [i + low for a, b in runs(dense[low:high])
                        for i in range(a, b - minimum + 1)]
            for arrival in arrivals:
                context = next((a for a, b in contexts if a <= arrival < b), None)
                if context is None:
                    continue
                earliest = max(context, int(arrival) - lookback)
                onsets = np.flatnonzero(toward[earliest + 1:arrival + 1]) + earliest
                onsets = onsets[~dense[onsets]]
                if not len(onsets):
                    continue
                path = paths[arrival] - paths[onsets]
                net = np.linalg.norm(vectors[arrival] - vectors[onsets], axis=1)
                relative_net = relatives[arrival] - relatives[onsets]
                denominator = np.linalg.norm(relative_net, axis=1) * np.linalg.norm(offset[onsets], axis=1)
                direction = np.divide((relative_net * offset[onsets]).sum(axis=1), denominator,
                                      out=np.full(len(onsets), -1.), where=denominator > 1e-9)
                measured_steps = counts[arrival] - counts[onsets]
                directed_steps = toward_counts[arrival] - toward_counts[onsets]
                qualified = ((path > config.clustering_actor_min_path_cm + 1e-9)
                             & (net + 1e-9 >= config.clustering_actor_min_displacement_cm)
                             & (drops[arrival] - drops[onsets] + 1e-9 >= config.clustering_distance_drop_cm)
                             & (direction >= config.clustering_toward_cosine)
                             & (path * fps + 1e-9 >= config.clustering_actor_speed_cm_s * measured_steps)
                             & (directed_steps / fps + 1e-9 >= .15)
                             & (directed_steps + 1e-9 >= config.clustering_min_toward_fraction * measured_steps))
                valid_onsets = onsets[qualified]
                if not len(valid_onsets):
                    continue
                onset = int(valid_onsets[0])
                output[onset:arrival + 1] |= known[onset:arrival + 1]
                output[low:high] |= dense[low:high]
                support[onset:arrival + 1] |= toward[onset:arrival + 1]
                break
    # Preserve the invariant even for callers supplying an old serialized mask.
    return output & known, support & known
