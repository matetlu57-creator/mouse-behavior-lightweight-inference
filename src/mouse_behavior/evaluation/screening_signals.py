"""Observable phase evidence for dataset trimming, separate from Top-1 ranking."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import math

import cv2
import numpy as np

from mouse_behavior.evaluation.exported_dataset import ExportedFeatures, HeuristicParameters


@dataclass(frozen=True)
class ScreenPolicy:
    posture_export_mode: str = "human_confirmed_only"
    approach_stop_distance_cm: float = 3.5
    approach_endpoint_tolerance_cm: float = .5
    contact_guard_s: float = 0.
    contact_clearance_width_fraction: float = 0.
    surface_contact_width_fraction: float = .1
    rear_transition_ratio: float = 1.25
    rear_transition_context_s: float = 2.
    chase_heading_lag_s: float = .5
    chase_support_speed_fraction: float = .4
    chase_ahead_cosine: float = .5
    chase_path_width_fraction: float = .5
    evidence_gap_s: float = .4
    motion_window_s: float = .3
    posture_window_s: float = 1.2
    posture_min_s: float = 1.
    quiet_speed_cm_s: float = 2.675
    groom_foreaft_body_fraction: float = .07
    rear_foreaft_body_fraction: float = .06
    rear_length_fraction: float = .78
    export_event_min_s: float = .5

    def __post_init__(self):
        if self.posture_export_mode not in {"human_confirmed_only", "review_candidates"}:
            raise ValueError("posture_export_mode must be human_confirmed_only or review_candidates")
        for name, value in vars(self).items():
            if name == "posture_export_mode":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"segment_screening.{name} must be a finite nonnegative number")
        if self.rear_transition_ratio <= 1 or self.motion_window_s <= 0 or self.posture_window_s <= 0:
            raise ValueError("screening windows must be positive and rear_transition_ratio must exceed one")
        if not .4 <= self.chase_support_speed_fraction <= 1:
            raise ValueError("chase_support_speed_fraction must be between 0.4 and one")
        if not 0 < self.chase_ahead_cosine <= 1 or not 0 < self.chase_path_width_fraction <= 1:
            raise ValueError("chase ahead cosine and path width fraction must be in (0, 1]")
        if self.chase_heading_lag_s > .5:
            raise ValueError("chase_heading_lag_s cannot exceed 0.5 seconds")
        if not 0 < self.approach_endpoint_tolerance_cm <= .5:
            raise ValueError("approach_endpoint_tolerance_cm must be in (0, 0.5]")
        if self.approach_stop_distance_cm <= 0:
            raise ValueError("approach_stop_distance_cm must be positive")
        if not 0 < self.surface_contact_width_fraction < 1:
            raise ValueError("surface_contact_width_fraction must be between zero and one")


def smooth(values, frames):
    """Centered median; missing observations stay missing at the current frame."""
    values = np.asarray(values, float)
    size = max(1, int(frames) | 1)
    pad = [(size // 2, size // 2)] + [(0, 0)] * (values.ndim - 1)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, pad, constant_values=np.nan), size, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(windows, axis=-1)
    return np.where(np.isfinite(values), result, np.nan)


def envelope(values, frames):
    values = np.asarray(values, float)
    size = max(3, int(frames) | 1)
    pad = [(size // 2, size // 2)] + [(0, 0)] * (values.ndim - 1)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, pad, constant_values=np.nan), size, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(windows, [.1, .9], axis=-1)
    return hi - lo


def runs(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def bridge(mask, allowed, frames):
    """Join evidence dips only when every intervening frame supports the phase."""
    result = np.asarray(mask, bool).copy()
    bounds = runs(result)
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        if start - end <= frames and np.all(allowed[end:start]):
            result[end:start] = True
    return result


def supported_episodes(seed, support, frames):
    """Recover the observed ongoing event around seeds; no arbitrary padding."""
    support = bridge(support, np.asarray(support) | np.asarray(seed), frames)
    result = np.zeros_like(seed, bool)
    for start, end in runs(support):
        active = np.flatnonzero(seed[start:end])
        if active.size:
            # Leading and trailing calm support cannot lengthen an event.
            result[start + active[0]:start + active[-1] + 1] = True
    return result


def observed_run_motion(positions: np.ndarray, fps: float, policy: ScreenPolicy):
    """Motion within one entirely observed run, including short reacquisitions.

    Use the existing window/lag when supported. Short runs use the largest
    available odd window and a shorter measured lag; two observations suffice
    for a displacement, while a singleton has unknown motion. No operand comes
    from the far side of a missing interval.
    """
    positions = np.asarray(positions, float)
    if positions.ndim < 2 or positions.shape[-1] != 2 or not np.isfinite(positions).all():
        raise ValueError("observed run positions must be finite (...,2) coordinates")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    count = len(positions)
    velocity = np.full_like(positions, np.nan)
    if count < 2:
        return positions.copy(), velocity, np.linalg.norm(velocity, axis=-1)
    desired = max(1, int(fps * policy.motion_window_s) | 1)
    window = min(desired, count if count % 2 else count - 1)
    centers = smooth(positions, window)
    lag = min(max(1, round(fps * policy.motion_window_s)), count // 2)
    velocity[lag:] = (centers[lag:] - centers[:-lag]) * fps / lag
    velocity[:lag] = velocity[lag:lag * 2]
    return centers, velocity, np.linalg.norm(velocity, axis=-1)


def body_motion(features: ExportedFeatures, policy: ScreenPolicy):
    if features.collective_keypoints_cm is None:
        return features.centers_cm, features.velocity_cm_s, features.speed_cm_s
    fps = features.fps
    centers = smooth(features.centers_cm, fps * policy.motion_window_s)
    lag = max(1, round(fps * policy.motion_window_s))
    velocity = np.full_like(centers, np.nan)
    velocity[lag:] = (centers[lag:] - centers[:-lag]) * fps / lag
    # At clip onset only, use observed forward displacement rather than a zero.
    velocity[:lag] = velocity[lag:lag * 2] if len(centers) >= lag * 2 else np.nan
    good = features.valid & np.isfinite(velocity).all(axis=2)
    return centers, velocity, np.where(good, np.linalg.norm(velocity, axis=2), np.nan)


def pursuit_similarity(centers, velocity, speed, left, right, fps, policy, width_cm):
    """A pursuing mouse may reach the leader's turn up to half a second later."""
    result = np.full(len(centers), -np.inf)
    frames = np.arange(len(centers))
    distance = np.linalg.norm(centers[:, right] - centers[:, left], axis=1)
    for follower, leader in ((left, right), (right, left)):
        offset = centers[:, leader] - centers[:, follower]
        ahead = np.sum(offset * velocity[:, follower], axis=1) / np.maximum(distance * speed[:, follower], 1e-9)
        for lag in range(max(0, round(fps * policy.chase_heading_lag_s)) + 1):
            previous = np.maximum(0, frames-lag)
            similarity = np.sum(velocity[:, follower] * velocity[previous, leader], axis=1) / np.maximum(speed[:, follower] * speed[previous, leader], 1e-9)
            path_gap = np.linalg.norm(centers[:, follower] - centers[previous, leader], axis=1)
            advance = np.linalg.norm(centers[:, leader] - centers[previous, leader], axis=1)
            follows_path = ((lag > 0) & (path_gap <= policy.chase_path_width_fraction * width_cm)
                            & (advance >= policy.chase_path_width_fraction * width_cm))
            usable = ((frames >= lag) & ((ahead >= policy.chase_ahead_cosine) | follows_path)
                      & (speed[:, follower] > policy.quiet_speed_cm_s)
                      & (speed[previous, leader] > policy.quiet_speed_cm_s) & np.isfinite(similarity))
            result = np.maximum(result, np.where(usable, similarity, -np.inf))
    return result


def pair_phases(features: ExportedFeatures, pair: int, params: HeuristicParameters,
                policy: ScreenPolicy, *, accept_observed_endpoint: bool = False,
                measured_motion: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None):
    """Contact includes snout-to-torso surface; its safety margin is never bridged."""
    left, right = int(features.pair_i[pair]), int(features.pair_j[pair])
    slots = [left, right]
    valid = features.valid[:, left] & features.valid[:, right]
    nose_dist = np.minimum(features.pair_nose_head_cm[:, pair], features.pair_nose_tail_cm[:, pair])
    nearest = nose_dist.copy()
    for values in (features.pair_nose_nearest_left_to_right_cm, features.pair_nose_nearest_right_to_left_cm):
        if values is not None:
            nearest = np.minimum(nearest, values[:, pair])
    body_gap = np.full(len(valid), np.inf)
    geometry_known = valid.copy()
    points = features.collective_keypoints_cm
    if points is not None:
        for frame in np.flatnonzero(valid):
            hulls = []
            for slot in slots:
                pts = points[frame, slot]
                finite = np.isfinite(pts).all(axis=1)
                if finite.sum() < 4 or not finite[0]:
                    geometry_known[frame] = False
                    break
                hulls.append(cv2.convexHull(pts[finite].astype(np.float32)))
            if len(hulls) != 2:
                continue
            gaps = []
            for a, b in ((0, 1), (1, 0)):
                nose = points[frame, slots[a], 0]
                nearest[frame] = min(nearest[frame], max(0., -cv2.pointPolygonTest(hulls[b], tuple(map(float, nose)), True)))
                for point in hulls[a][:, 0]:
                    gaps.append(max(0., -cv2.pointPolygonTest(hulls[b], tuple(map(float, point)), True)))
            body_gap[frame] = min(gaps)
            intersection, _ = cv2.intersectConvexConvex(hulls[0], hulls[1])
            if intersection > 0:
                body_gap[frame] = 0.
    # Landmark distances and snout-to-hull surface gaps have different origins.
    # Applying the old 2.675 cm landmark gate to the surface gap cuts early.
    contact_limit = (policy.surface_contact_width_fraction * features.mouse_width_cm
                     if points is not None else params.contact_distance_cm)
    contact = valid & geometry_known & ((nearest <= contact_limit) | (body_gap <= contact_limit))
    gap_frames = max(1, round(policy.evidence_gap_s * features.fps))
    contact = bridge(contact, valid & geometry_known & (nearest <= contact_limit), gap_frames)
    clearance = contact_limit + policy.contact_clearance_width_fraction * features.mouse_width_cm
    unsafe = contact | ~geometry_known | (nearest <= clearance)
    guard = max(0, int(np.ceil(policy.contact_guard_s * features.fps)))
    guarded = unsafe.copy()
    for offset in range(1, guard + 1):
        guarded[:-offset] |= unsafe[offset:]
        guarded[offset:] |= unsafe[:-offset]
    centers, velocity, speed = (body_motion(features, policy)
                                if measured_motion is None else measured_motion)
    distance = smooth(features.pair_distance_cm[:, pair], features.fps * policy.motion_window_s)
    lag = max(1, round(features.fps * policy.motion_window_s))
    closing = np.zeros(len(valid))
    good_distance = np.isfinite(distance[:-lag]) & np.isfinite(distance[lag:])
    closing[lag:][good_distance] = (distance[:-lag][good_distance] - distance[lag:][good_distance]) * features.fps / lag
    keypoint_nearest = nose_dist.copy()
    for values in (features.pair_nose_nearest_left_to_right_cm, features.pair_nose_nearest_right_to_left_cm):
        if values is not None:
            keypoint_nearest = np.minimum(keypoint_nearest, values[:, pair])
    finite = np.isfinite(keypoint_nearest[lag:]) & np.isfinite(keypoint_nearest[:-lag])
    nose_closing = np.zeros(len(valid))
    nose_closing[lag:][finite] = (keypoint_nearest[:-lag][finite] - keypoint_nearest[lag:][finite]) * features.fps / lag
    closing = np.maximum(closing, nose_closing)
    motion_known = np.isfinite(speed[:, slots]).all(axis=1)
    max_speed = np.max(np.nan_to_num(speed[:, slots], nan=0.), axis=1)
    approach_guarded = guarded | (nearest <= policy.approach_stop_distance_cm)
    approach_support = valid & motion_known & ~approach_guarded & (max_speed > params.approach_min_actor_speed_cm_s) & (closing > .5)
    approach_seed = approach_support & (closing >= params.approach_min_closing_speed_cm_s)
    approach_support = bridge(approach_support, valid & motion_known & ~approach_guarded & (max_speed > policy.quiet_speed_cm_s), gap_frames)
    approach = np.zeros(len(valid), bool)
    for start, end in runs(approach_support):
        # A distance observed before the crop cannot qualify its first frame.
        # Recover only the measured onset hidden by the backward closing-speed
        # window, never an unobserved prefix, pause or departing mouse.
        onset = start
        while (keypoint_nearest[onset] < params.approach_min_start_distance_cm
               and onset > max(0, start-lag)):
            previous = onset-1
            if not (valid[previous] and geometry_known[previous] and motion_known[previous]
                    and not guarded[previous] and max_speed[previous] > policy.quiet_speed_cm_s
                    and np.isfinite(keypoint_nearest[previous])
                    and keypoint_nearest[previous] >= keypoint_nearest[onset]
                    and nearest[previous] >= nearest[onset]):
                break
            onset = previous
        if (approach_seed[start:end].any() and np.isfinite(keypoint_nearest[onset])
                and keypoint_nearest[onset] >= params.approach_min_start_distance_cm):
            start = onset
            # A completed approach must actually reach the requested boundary.
            # A clip ending far away or losing motion/geometry stays for review.
            if (accept_observed_endpoint and end == len(valid)
                    and geometry_known[end - 1] and motion_known[end - 1]
                    and not guarded[end - 1]
                    and abs(nearest[end - 1] - policy.approach_stop_distance_cm)
                    <= policy.approach_endpoint_tolerance_cm):
                # The measured final frame is sufficient at an already cropped
                # input boundary. No fabricated successor or extra duration.
                approach[start:end] = True
                continue
            if not (end < len(valid) and geometry_known[end]
                    and nearest[end] <= max(clearance, policy.approach_stop_distance_cm)):
                continue
            # Compare the two actual frames around the crossing. At high speed
            # the preceding frame can still be 5 cm away; no interpolated frames.
            next_frame = end
            local = nearest[start:end]
            if np.isfinite(local).any():
                end = start + int(np.argmin(local)) + 1
            before_error = abs(nearest[end-1] - policy.approach_stop_distance_cm)
            after_error = abs(nearest[next_frame] - policy.approach_stop_distance_cm)
            if (end == next_frame and not guarded[next_frame] and motion_known[next_frame]
                    and after_error < before_error):
                end += 1
            if abs(nearest[end-1] - policy.approach_stop_distance_cm) > policy.approach_endpoint_tolerance_cm:
                continue
            approach[start:end] = True
    near = valid & (features.pair_distance_cm[:, pair] <= params.attack_endpoint_distance_cm)
    active = motion_known & (max_speed > policy.quiet_speed_cm_s)
    # Acceleration alone is dominated by keypoint jitter. Original Attack
    # labels may retain that seed, but novel attack mining needs body action.
    vigorous = motion_known & (max_speed >= params.attack_min_speed_cm_s)
    strict_attack_seed = contact & vigorous
    if points is None:
        direction = features.pair_direction_similarity[:, pair]
    else:
        direction = np.sum(velocity[:, left] * velocity[:, right], axis=1) / np.maximum(speed[:, left] * speed[:, right], 1e-9)
    if points is not None:
        direction = pursuit_similarity(centers, velocity, speed, left, right, features.fps, policy, features.mouse_width_cm)
    chase_allowed = valid & geometry_known & motion_known & (features.pair_nose_head_cm[:, pair] > params.contact_distance_cm)
    chasing = (chase_allowed & (distance <= params.chase_max_distance_cm)
               & (speed[:, left] >= params.chase_min_speed_cm_s)
               & (speed[:, right] >= params.chase_min_speed_cm_s)
               & (direction >= params.chase_min_direction_similarity))
    chasing = bridge(chasing, chase_allowed & (distance <= params.chase_max_distance_cm)
                     & (np.minimum(speed[:, left], speed[:, right]) >= policy.chase_support_speed_fraction * params.chase_min_speed_cm_s)
                     & (direction >= params.chase_min_direction_similarity), gap_frames)
    # Quiet touching may be Together. Active touching/sniffing, head-to-head
    # approaches, pursuit and rapid relative motion are interactions.
    head_active = np.zeros(len(valid), bool)
    if points is not None:
        pair_points = smooth(points[:, slots], .15 * features.fps)
        head = (pair_points[:, :, 0] + (pair_points[:, :, 1] + pair_points[:, :, 2]) / 2) / 2
        hind = (pair_points[:, :, 4] + pair_points[:, :, 5]) / 2
        relative_head = head - hind
        head_speed = np.full((len(valid), 2), np.nan)
        head_speed[lag:] = np.linalg.norm(relative_head[lag:] - relative_head[:-lag], axis=2) * features.fps / lag
        if len(valid) >= 2 * lag:
            head_speed[:lag] = head_speed[lag:2*lag]
        head_active = np.max(np.nan_to_num(head_speed, nan=np.inf), axis=1) > params.approach_min_actor_speed_cm_s
        head_amplitude = np.linalg.norm(envelope(relative_head, .6 * features.fps), axis=2)
        head_action = (head_active & np.isfinite(head_speed).all(axis=1)
                       & (np.max(np.nan_to_num(head_amplitude), axis=1) >= .05 * features.reference_body_cm))
    elif features.nose_speed_cm_s is not None:
        head_speed = smooth(features.nose_speed_cm_s[:, slots], features.fps * .3)
        head_active = np.max(np.nan_to_num(head_speed, nan=np.inf), axis=1) > params.approach_min_actor_speed_cm_s
        head_action = head_active & np.isfinite(head_speed).all(axis=1)
    else:
        head_action = np.zeros(len(valid), bool)
    strict_attack_seed &= ~chasing
    attack_seed = contact & (vigorous | head_action) & ~chasing
    attack_support = bridge(near & (active | head_action) & ~approach, near & ~approach, gap_frames)
    attack = supported_episodes(attack_seed, attack_support | attack_seed, gap_frames)
    strict_attack = supported_episodes(strict_attack_seed, attack_support | strict_attack_seed, gap_frames)
    quiet = motion_known & (max_speed <= policy.quiet_speed_cm_s) & ~head_active
    separated = ~contact & (nearest > clearance) & (distance >= params.together_min_distance_cm)
    relative = np.linalg.norm(velocity[:, left] - velocity[:, right], axis=1)
    interacting = approach | attack | chasing | (guarded & ~quiet) | (relative > policy.quiet_speed_cm_s) | head_active
    together = (valid & geometry_known & motion_known & (distance <= params.together_max_distance_cm)
                & (separated | quiet) & ~interacting)
    return {"contact": contact, "guarded_contact": guarded, "approach": approach,
            "attack": attack, "strict_attack": strict_attack, "attack_seed": attack_seed, "chasing": chasing,
            "together": together, "quiet": quiet, "nearest_surface_cm": nearest,
            "nearest_keypoint_cm": keypoint_nearest, "pursuit_similarity": direction,
            "speed_cm_s": speed, "centers_cm": centers, "velocity_cm_s": velocity,
            "geometry_known": geometry_known, "head_active": head_active}


def posture_evidence(features: ExportedFeatures, policy: ScreenPolicy, *, arena_quad_cm=None):
    """Head-on-neck action separates grooming from upright forebody motion."""
    points = features.collective_keypoints_cm
    if points is None:
        return {}, {}
    fps = features.fps
    points = smooth(points, .15 * fps)
    hind = (points[:, :, 4] + points[:, :, 5]) / 2
    neck, nose, tail = points[:, :, 3], points[:, :, 0], points[:, :, 6]
    axis = smooth(neck - tail, .5 * fps)
    norm = np.linalg.norm(axis, axis=2)
    axis = axis / np.maximum(norm[..., None], 1e-9)
    transverse = np.stack((-axis[:, :, 1], axis[:, :, 0]), axis=2)
    head = (nose + (points[:, :, 1] + points[:, :, 2]) / 2) / 2
    axial = np.sum((head - neck) * axis, axis=2)
    lateral = np.sum((head - hind) * transverse, axis=2)
    amplitude = envelope(axial, fps * policy.posture_window_s)
    lateral_amp = envelope(lateral, fps * policy.posture_window_s)
    delta = np.zeros_like(axial)
    lag = max(1, round(.12 * fps))
    delta[lag:] = axial[lag:] - axial[:-lag]
    sign = np.where(np.abs(delta) > .012 * features.reference_body_cm, np.sign(delta), 0.)
    reversals = np.zeros_like(sign)
    for slot in range(sign.shape[1]):
        prior = 0
        for frame in range(len(sign)):
            now = sign[frame, slot]
            if now and prior and now != prior:
                reversals[frame, slot] = 1
            if now:
                prior = now
            if not features.valid[frame, slot]:
                prior = 0
    window = max(3, round(fps * policy.posture_window_s))
    pad = ((window//2, window-window//2-1), (0, 0))
    cycles = np.lib.stride_tricks.sliding_window_view(np.pad(reversals, pad), window, axis=0).sum(axis=-1)
    body_length = np.linalg.norm(nose - tail, axis=2)
    _, _, body_speed = body_motion(features, policy)
    hind_amp = np.linalg.norm(envelope(hind, fps * policy.posture_window_s), axis=2)
    stable = features.valid & (body_speed <= policy.quiet_speed_cm_s) & (hind_amp <= .2 * features.reference_body_cm)
    quality = stable & np.isfinite(amplitude) & np.isfinite(norm) & (norm > .12 * features.reference_body_cm)
    hip_width = np.linalg.norm(points[:, :, 4] - points[:, :, 5], axis=2)
    ear_width = np.linalg.norm(points[:, :, 1] - points[:, :, 2], axis=2)
    quality &= ((hip_width >= .12 * features.reference_body_cm)
                & (hip_width <= .65 * features.reference_body_cm)
                & (ear_width >= .06 * features.reference_body_cm)
                & (body_length >= .25 * features.reference_body_cm)
                & (body_length <= 1.6 * features.reference_body_cm))
    if features.detection_confidence is not None:
        quality &= np.isnan(features.detection_confidence) | (features.detection_confidence >= .4)
    if features.keypoint_coverage is not None:
        quality &= features.keypoint_coverage >= .85
    rhythmic = (amplitude >= policy.groom_foreaft_body_fraction * features.reference_body_cm) & (cycles >= 2)
    rhythmic &= amplitude >= .65 * lateral_amp
    nose_axial = np.sum((nose - neck) * axis, axis=2)
    ear_axial = np.sum(((points[:, :, 1] + points[:, :, 2]) / 2 - neck) * axis, axis=2)
    nose_delta, ear_delta = np.zeros_like(axial), np.zeros_like(axial)
    nose_delta[lag:] = nose_axial[lag:] - nose_axial[:-lag]
    ear_delta[lag:] = ear_axial[lag:] - ear_axial[:-lag]
    moving_head = (np.abs(nose_delta) > .01 * features.reference_body_cm) & (np.abs(ear_delta) > .005 * features.reference_body_cm)
    coherent = moving_head & (nose_delta * ear_delta > 0)
    moving_windows = np.lib.stride_tricks.sliding_window_view(np.pad(moving_head, pad), window, axis=0).sum(axis=-1)
    coherent_windows = np.lib.stride_tricks.sliding_window_view(np.pad(coherent, pad), window, axis=0).sum(axis=-1)
    rhythmic &= coherent_windows >= np.maximum(3, moving_windows * .6)
    backwards = np.sum((nose - neck) * axis, axis=2) < -.02 * features.reference_body_cm
    close_back = np.linalg.norm(nose - hind, axis=2) <= .4 * features.reference_body_cm
    # One isolated exploratory head turn is insufficient for back grooming.
    back_amplitude = np.linalg.norm(envelope(head - hind, fps * policy.posture_window_s), axis=2)
    back_groom = backwards & close_back & (cycles >= 2) & (back_amplitude >= .07 * features.reference_body_cm)
    groom = quality & (rhythmic | back_groom)
    short = ((body_length <= policy.rear_length_fraction * features.reference_body_cm)
             & (norm <= .6 * features.reference_body_cm)
             & (np.sum((nose-neck) * axis, axis=2) >= 0))
    small_head = amplitude <= policy.rear_foreaft_body_fraction * features.reference_body_cm
    # Constant compact posture also occurs during curled rest. Require observed
    # extension next to the compressed episode before proposing upright posture.
    upright_evidence = np.zeros_like(short)
    context = max(1, round(policy.rear_transition_context_s * fps))
    for slot in range(short.shape[1]):
        for start, end in runs(short[:, slot] & features.valid[:, slot]):
            before = norm[max(0, start-context):start, slot]
            after = norm[end:min(len(norm), end+context), slot]
            adjacent = np.r_[before, after]
            adjacent = adjacent[np.isfinite(adjacent)]
            compressed = np.nanmedian(norm[start:end, slot])
            if adjacent.size and np.quantile(adjacent, .8) >= policy.rear_transition_ratio * compressed:
                upright_evidence[start:end, slot] = True
    rear = quality & short & upright_evidence & small_head & ~groom
    masks = {"grooming": groom, "rearing": rear}
    for name, values in masks.items():
        for slot in range(values.shape[1]):
            allowed = quality[:, slot] & (rhythmic[:, slot] | back_groom[:, slot] if name == "grooming" else short[:, slot] & upright_evidence[:, slot] & small_head[:, slot] & ~groom[:, slot])
            values[:, slot] = bridge(values[:, slot], allowed, round(policy.evidence_gap_s * fps))
    metrics = {"foreaft_amplitude_cm": amplitude, "lateral_amplitude_cm": lateral_amp,
               "reversals": cycles, "body_length_cm": body_length, "hind_amplitude_cm": hind_amp,
               "body_speed_cm_s": body_speed, "back_groom": back_groom, "upright_transition": upright_evidence}
    if arena_quad_cm is not None:
        quad = np.asarray(arena_quad_cm, np.float32).reshape(-1, 1, 2)
        wall = np.full_like(axial, np.nan)
        for frame, slot in zip(*np.where(quality)):
            wall[frame, slot] = min(abs(cv2.pointPolygonTest(quad, tuple(map(float, point)), True))
                                    for point in (nose[frame, slot], hind[frame, slot]))
        metrics['wall_distance_cm'] = wall
    return masks, metrics
