"""Shared individual rules in physical units, independent of annotations.

Short projected length alone cannot identify standing in a top-view recording.
Posture rules require a change over time around a stable posterior anchor.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from .static_rule import bounded_stationary_mask

INDIVIDUAL_RULE_REVISION = "2026-10-20-individual-v27-four-cm-bounded-prefix"
SUSPENDED_POSTURE_BEHAVIORS = frozenset(("grooming", "rearing"))


@dataclass(frozen=True)
class IndividualHeuristicConfig:
    motion_speed_threshold_cm_s: float = 4.0
    motion_window_s: float = 0.3
    smooth_window_s: float = 0.1
    static_min_duration_s: float = 1.0
    locomotion_min_duration_s: float = 0.5
    static_max_radius_cm: float = 4.0
    static_transient_max_s: float = 1.0
    static_transient_max_displacement_cm: float = 4.0
    static_detection_gap_max_s: float = 1.0
    locomotion_pause_max_s: float = 1.0
    # [v16.2 IND-7] Bounds on how much fabricated (unobserved) time the locomotion
    # pause bridge may introduce. Without them short dropout gaps chain: the real
    # rearing sample had visual track 3384 observed in only 10 of 60 frames yet
    # labelled locomotion for 33, 29 of them rows where valid == False.
    # Per single bridged gap the unobserved span may not exceed the detection-gap
    # limit already used for Static; across one final bout the cumulative bridged
    # unobserved time is capped in seconds AND as a fraction of the bout, so a bout
    # always keeps a majority of genuinely detected frames.
    locomotion_bridged_unobserved_max_s: float = 1.0
    locomotion_bridged_unobserved_max_fraction: float = 0.5
    # Same cumulative bound for the Static path: a per-gap detection limit alone
    # still lets many short dropouts chain into a bout that is mostly fabricated.
    static_bridged_unobserved_max_fraction: float = 0.5
    grooming_signal_window_s: float = 0.3
    grooming_context_window_s: float = 1.2
    grooming_min_duration_s: float = 0.3
    grooming_min_head_amplitude_body_ratio: float = 0.07
    grooming_min_reversals: int = 3
    posterior_max_excursion_body_ratio: float = 0.12
    back_groom_max_nose_posterior_body_ratio: float = 0.45
    rearing_context_window_s: float = 2.0
    rearing_min_duration_s: float = 0.5
    rearing_min_projection_change_body_ratio: float = 0.15
    rearing_max_short_projection_body_ratio: float = 0.55
    wall_min_neck_excursion_body_ratio: float = 0.12
    # --- v16.2 module 2: posture rebuilt on 7-point trajectory + detection box ---
    #
    # ERA INVARIANCE. evaluation/scale_migration.py multiplies cm_per_pixel by
    # NEW_BODY_LENGTH_CM / OLD_BODY_LENGTH_CM = 7.918 / 10.7 = 0.74 and rewrites
    # assumed_mouse_body_length_cm to 7.918 whenever a frozen receipt predates the
    # active rig, which is the case for every scale receipt in the confirmed
    # calibration set. At runtime reference_body_cm therefore resolves to 7.918 and
    # every centimetre in `features` is 0.74x the value the plan document measured.
    # A threshold written as a cm literal would be 1.35x mis-scaled against the
    # quantity it is compared to, so every gate below is a FRACTION of
    # reference_body_cm (or of reference_body_cm per second) and rescales with it.
    # The fractions are the plan's own arithmetic divided by its 10.7 cm L_ref.
    posterior_min_valid_points: int = 2
    grooming_step_window_s: float = 0.07
    # 1.50 cm/s quantisation dead zone / 10.7 cm. Expressed per body length rather
    # than in cm or px because the noise it suppresses is 0.5-1.0 px of detector
    # jitter, and cm_per_pixel is itself multiplied by 0.74 under migration.
    grooming_step_dead_zone_body_ratio: float = 1.50 / 10.7
    # 7.5 cm/s step-nose gate / 10.7 cm. The plan derived 7.5 from its measured
    # grooming v_nose of 12.29 cm/s (gate = 0.61x measured) while its fastest
    # same-cage static track measured 9.14 cm/s, so the gate sits between them.
    # Keeping it per body length preserves that ordering in every era.
    grooming_min_step_nose_speed_body_ratio: float = 7.5 / 10.7
    grooming_min_nose_posterior_speed_ratio: float = 3.0
    grooming_min_head_posterior_speed_ratio: float = 2.3
    grooming_max_seated_projection_body_ratio: float = 0.92
    grooming_box_pulsation_body_ratio: float = 0.07
    grooming_min_nose_ear_codirection: float = 0.55
    grooming_bridge_max_s: float = 0.2
    rearing_max_posterior_excursion_body_ratio: float = 0.15
    rearing_max_box_shrink_ratio: float = 0.78
    rearing_min_spine_collinearity: float = 0.5
    rearing_max_wall_distance_body_ratio: float = 0.3
    # The plan states "window head-neck displacement covariance > 0.05". A raw
    # covariance carries cm^2 units and would scale by 0.74^2 under migration, so
    # this is implemented as the NORMALISED (dimensionless, Pearson) covariance of
    # the head and neck axial displacements, which is era-invariant by construction
    # and reads 0.05 as "very weakly co-directional is already enough" - the gate
    # is an OR with the instantaneous head_neck_cos > 0 arm.
    rearing_min_head_neck_covariance: float = 0.05
    rearing_bridge_max_s: float = 0.2
    box_baseline_percentile: float = 85.0
    # Keep the velocity estimator independent of the permitted Static excursion.
    motion_interpolation_max_gap_s: float = 0.3
    # Appended for positional compatibility; opt-in remains audit-only.
    enable_posture_screen: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.motion_window_s <= 0 or self.grooming_signal_window_s <= 0:
            raise ValueError("signal windows must be positive")
        for fraction_name in (
            "locomotion_bridged_unobserved_max_fraction",
            "static_bridged_unobserved_max_fraction",
        ):
            if not 0.0 <= getattr(self, fraction_name) <= 1.0:
                raise ValueError(f"{fraction_name} must be finite and within [0, 1]")
        if self.locomotion_bridged_unobserved_max_s > self.static_detection_gap_max_s:
            raise ValueError(
                "locomotion_bridged_unobserved_max_s must not exceed static_detection_gap_max_s"
            )
        if self.grooming_step_window_s <= 0 or self.box_baseline_percentile <= 0:
            raise ValueError("grooming_step_window_s and box_baseline_percentile must be positive")
        if not 0.0 < self.box_baseline_percentile <= 100.0:
            raise ValueError("box_baseline_percentile must be within (0, 100]")
        if int(self.posterior_min_valid_points) not in (1, 2, 3):
            raise ValueError("posterior_min_valid_points must be 1, 2 or 3")
        for unitless in (
            "grooming_min_nose_posterior_speed_ratio",
            "grooming_min_head_posterior_speed_ratio",
            "grooming_min_nose_ear_codirection",
            "rearing_max_box_shrink_ratio",
            "rearing_min_spine_collinearity",
        ):
            value = float(getattr(self, unitless))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{unitless} must be finite and non-negative")
        if not -1.0 <= self.rearing_min_head_neck_covariance <= 1.0:
            raise ValueError("rearing_min_head_neck_covariance must be within [-1, 1]")
        if not 0.0 <= self.rearing_min_spine_collinearity <= 1.0:
            raise ValueError("rearing_min_spine_collinearity must be within [0, 1]")


DEFAULT_INDIVIDUAL_CONFIG = IndividualHeuristicConfig()


@dataclass
class IndividualRuleResult:
    masks: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def _runs(mask: np.ndarray):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def _qualify(mask: np.ndarray, minimum: float, fps: float) -> np.ndarray:
    result = np.zeros_like(mask)
    for slot in range(mask.shape[1]):
        for start, end in _runs(mask[:, slot]):
            if (end - start) / fps + 1e-9 >= minimum:
                result[start:end, slot] = True
    return result


def _enforce_observation_bound(
    mask: np.ndarray,
    observed: np.ndarray,
    bridged_allowed: np.ndarray,
    label: str,
) -> np.ndarray:
    """Invariant (DEFECT IND-7): never claim a frame the detector did not report.

    Gap bridging is the only mechanism allowed to label an unobserved frame, and
    every bridge is bounded at creation time. ``bridged_allowed`` is the exact set
    of frames that a bounded bridge filled in. Any True-while-unobserved frame
    outside that set is a fabrication bug, so it raises instead of being silently
    trimmed: a quiet clamp would let a later edit reintroduce unbounded chaining
    and only show up as unexplained recall changes downstream.
    """

    offenders = mask & ~observed & ~bridged_allowed
    if offenders.any():
        rows = np.flatnonzero(offenders.any(axis=1))[:5].tolist()
        slots = np.flatnonzero(offenders.any(axis=0))[:5].tolist()
        raise RuntimeError(
            f"{label} claims {int(offenders.sum())} unobserved frame(s) that no "
            f"bounded bridge authorised (first rows {rows}, first tracks {slots})"
        )
    return mask


def _bridge_gaps(mask: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Connect runs separated by at most ``max_gap_frames``; report what was filled.

    Returns ``(bridged, filled)``. ``filled`` marks exactly the frames this call
    added, so the caller can hand them to :func:`_enforce_observation_bound` as the
    authoritative allow-list of frames that may be reported while unobserved.
    """

    result = np.asarray(mask, dtype=bool).copy()
    filled = np.zeros_like(result)
    limit = max(int(max_gap_frames), 0)
    if limit <= 0:
        return result, filled
    for slot in range(result.shape[1]):
        segments = list(_runs(result[:, slot]))
        for (_, left_end), (right_start, _) in zip(segments, segments[1:]):
            if 0 < right_start - left_end <= limit:
                result[left_end:right_start, slot] = True
                filled[left_end:right_start, slot] = True
    return result, filled

def _window(values: np.ndarray, frames: int) -> np.ndarray:
    frames = max(1, int(frames)) | 1
    radius = frames // 2
    padded = np.pad(values, ((radius, radius), *((0, 0) for _ in values.shape[1:])), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, frames, axis=0)


def _median(values: np.ndarray, frames: int) -> np.ndarray:
    windows = _window(values, frames)
    return np.ma.median(np.ma.masked_invalid(windows), axis=-1).filled(np.nan)


def _range(values: np.ndarray, frames: int) -> np.ndarray:
    windows = _window(values, frames)
    finite = np.isfinite(windows)
    result = np.max(np.where(finite, windows, -np.inf), axis=-1) - np.min(
        np.where(finite, windows, np.inf), axis=-1
    )
    return np.where(np.any(finite, axis=-1), result, np.nan)


def _velocity(points: np.ndarray, observed: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """Centered derivatives use contiguous observations of the same visual ID."""
    frames = len(points)
    if frames < 2:
        return np.full_like(points, np.nan, dtype=float)
    half = max(1, int(round(seconds * fps / 2)))
    left = np.maximum(0, np.arange(frames) - half)
    right = np.minimum(frames - 1, np.arange(frames) + half)
    delta = points[right] - points[left]
    counts = np.vstack((np.zeros((1, observed.shape[1]), int), np.cumsum(~observed, axis=0)))
    continuous = (counts[right + 1] - counts[left]) == 0
    result = delta / ((right - left) / fps)[:, None, None]
    return np.where((continuous & observed)[..., None], result, np.nan)


def _rolling_mean_path_speed(
    points: np.ndarray, observed: np.ndarray, fps: float, seconds: float,
    *, max_gap_s: float = 0.3,
) -> np.ndarray:
    """Trailing time-weighted mean of adjacent center-speed magnitudes.

    Short gaps are represented by the net displacement between the two
    surrounding observations, divided by their actual elapsed time. Windows
    need at least 60% valid interval samples; averages use only valid samples.
    """
    values = np.asarray(points, dtype=float)
    visible = np.asarray(observed, dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 2 or visible.shape != values.shape[:2]:
        raise ValueError("points must be (frames, mice, 2) and observed (frames, mice)")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("fps and seconds must be positive and finite")

    frame_count, mouse_count = visible.shape
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError("max_gap_s must be finite and non-negative")
    interval_speed = np.full((frame_count, mouse_count), np.nan, dtype=float)
    if frame_count < 2:
        return interval_speed
    max_span = int(np.floor(float(max_gap_s) * float(fps) + 1e-9)) + 1
    for mouse in range(mouse_count):
        indices = np.flatnonzero(visible[:, mouse] & np.isfinite(values[:, mouse]).all(axis=1))
        for left, right in zip(indices, indices[1:]):
            span = int(right - left)
            if span <= 0 or span > max_span:
                continue
            speed = float(np.linalg.norm(values[right, mouse] - values[left, mouse])) * fps / span
            interval_speed[left + 1:right + 1, mouse] = speed

    window_frames = float(seconds) * float(fps)
    full_intervals = int(np.floor(window_frames + 1e-10))
    fractional_interval = window_frames - full_intervals
    if fractional_interval < 1e-10:
        fractional_interval = 0.0
    support = full_intervals + int(fractional_interval > 0.0)
    result = np.full((frame_count, mouse_count), np.nan, dtype=float)
    if support <= 0:
        return result

    finite = np.isfinite(interval_speed)
    finite_prefix = np.vstack((np.zeros((1, mouse_count)), np.cumsum(finite, axis=0)))
    speed_prefix = np.vstack((
        np.zeros((1, mouse_count)),
        np.cumsum(np.where(finite, interval_speed, 0.0), axis=0),
    ))

    ends = np.arange(support, frame_count)
    if len(ends):
        starts = ends - support + 1
        finite_count = finite_prefix[ends + 1] - finite_prefix[starts]
        minimum_count = max(1, int(np.ceil(0.6 * support - 1e-12)))
        adequate = finite_count >= minimum_count

        if full_intervals:
            full_start = ends - full_intervals + 1
            totals = speed_prefix[ends + 1] - speed_prefix[full_start]
        else:
            totals = np.zeros((len(ends), mouse_count), dtype=float)
        if fractional_interval:
            oldest = ends - full_intervals
            totals += fractional_interval * np.where(
                finite[oldest], interval_speed[oldest], 0.0
            )
        valid_weight = finite_count.astype(float)
        if fractional_interval:
            oldest = ends - full_intervals
            valid_weight += np.where(finite[oldest], fractional_interval - 1.0, 0.0)
        means = np.divide(totals, valid_weight, out=np.full_like(totals, np.nan),
                          where=valid_weight > 0)
        result[ends] = np.where(adequate, means, np.nan)

    # [v16.2 module 1 item 3] Backfill the leading window. A trailing mean cannot
    # exist before `support` intervals have elapsed, which left frames
    # 0 .. support-1 NaN: on a 40-frame (1.33 s) clip the first 10 frames carried
    # no speed evidence at all, so a genuinely static mouse kept only 30 frames =
    # 0.9996 s and fell just below static_min_duration_s = 1.0 s. Each leading
    # frame now averages the intervals that actually touch it through an expanding
    # window capped at `support`, under the same 60% adequacy rule. Frame 0 borrows
    # interval 1, i.e. the 0->1 displacement, which is real evidence anchored at
    # frame 0; no speed is invented where nothing was observed to move.
    last_interval = frame_count - 1
    for index in range(min(support, frame_count)):
        high = max(1, min(index, last_interval))
        low = max(1, high - support + 1)
        span = high - low + 1
        available = finite_prefix[high + 1] - finite_prefix[low]
        required = max(1, int(np.ceil(0.6 * span - 1e-12)))
        total = speed_prefix[high + 1] - speed_prefix[low]
        filled = np.divide(total, available, out=np.full_like(total, np.nan),
                           where=available >= required)
        result[index] = np.where(np.isfinite(result[index]), result[index], filled)
    return result


def _trailing_mean(values: np.ndarray, frames: int) -> np.ndarray:
    """Trailing mean over ``frames`` samples, ignoring NaN and never inventing one.

    The leading pad is NaN, not zero, and a window with no finite sample at all
    returns NaN. Both matter: padding with zero would read the start of a clip, and
    the far side of a detection dropout, as evidence that the animal was perfectly
    still - a fabricated measurement rather than a missing one. Callers compare
    these means against ``>=`` gates, and NaN fails those, so "no evidence" stays
    "no evidence" instead of silently becoming "stationary".
    """

    frames = max(1, int(frames))
    padded = np.pad(values, ((frames - 1, 0), *((0, 0) for _ in values.shape[1:])),
                    mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, frames, axis=0)
    finite = np.isfinite(windows)
    counts = finite.sum(axis=-1)
    totals = np.where(finite, windows, 0.0).sum(axis=-1)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)


def _positive_motion_dominance(numerator: np.ndarray, denominator: np.ndarray,
                               minimum_ratio: float) -> np.ndarray:
    """A moving head may dominate a still rear; 0/0 and missing data prove nothing."""
    return (np.isfinite(numerator) & np.isfinite(denominator)
            & (numerator > 0) & (denominator >= 0)
            & (numerator >= minimum_ratio * denominator))


def _step_speed(values: np.ndarray, fps: float, frames: int, dead_zone: float) -> np.ndarray:
    """Mean adjacent-frame step magnitude over a short trailing window.

    Deliberately computed on the UNSMOOTHED series. The v16 grooming rule applied a
    0.1 s median filter and then differenced over a 0.3 s window, which averages
    away the 4-6 Hz fore/aft reversal that IS the grooming signal (a reversal every
    2-3 frames). Steps whose magnitude falls below ``dead_zone`` are set to zero so
    0.5-1.0 px of detector quantisation is not read as movement; steps that cannot
    be computed at all stay NaN so a dropout is never counted as stillness.
    """

    if values.shape[0] < 2:
        return np.zeros(values.shape[:2], dtype=float)
    delta = np.linalg.norm(np.diff(values, axis=0), axis=-1) * fps
    gated = np.where(np.isfinite(delta), np.where(delta >= dead_zone, delta, 0.0), np.nan)
    arrival = np.full(delta.shape, np.nan, dtype=float)
    arrival[:] = gated
    return _trailing_mean(np.vstack((np.full((1, *arrival.shape[1:]), np.nan), arrival)), frames)


def _robust_posterior_and_head(
    points: np.ndarray,
    observed: np.ndarray,
    boxes: np.ndarray | None,
    *,
    min_posterior_points: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior anchor and head proxy that survive partial keypoint loss.

    ``points`` is (frames, mice, 7, 2) ordered nose, right ear, left ear, neck,
    left hip, right hip, tail base. The v16 rule gated ALL posture evidence on
    every one of the 7 keypoints being finite, so a paw over the nose while washing
    the face, or a raised head while standing, voided an entire bout (DEFECT
    POST-3). Here:

    * the posterior anchor (hips + tail base, indices 4/5/6) needs at least
      ``min_posterior_points`` of its 3 points finite;
    * the head proxy (indices 0/1/2) averages the finite head points when the nose
      is present, and otherwise extrapolates the ear midpoint along ears-minus-neck,
      i.e. onward in the direction the head points away from the neck;
    * when neither the nose nor a usable ear/neck pair exists, it falls back to the
      detection-box edge midpoint farthest from the posterior anchor - the front of
      the animal as the box sees it.

    Returns ``(posterior, head, anchor_ok)``. ``head`` is NaN where no source
    produced a finite point; nothing is extrapolated from nothing.
    """

    nan2 = np.full(points.shape[:2] + (2,), np.nan, dtype=float)
    rear = points[:, :, [4, 5, 6], :]
    rear_ok = np.isfinite(rear).all(axis=3)
    rear_count = rear_ok.sum(axis=2)
    posterior = np.divide(
        np.where(rear_ok[..., None], rear, 0.0).sum(axis=2),
        rear_count[..., None],
        out=nan2.copy(),
        where=rear_count[..., None] > 0,
    )
    anchor_ok = rear_count >= max(1, int(min_posterior_points))
    posterior = np.where(anchor_ok[..., None], posterior, np.nan)

    ears = points[:, :, [1, 2], :]
    ears_ok = np.isfinite(ears).all(axis=3)
    ears_count = ears_ok.sum(axis=2)
    ear_mid = np.divide(
        np.where(ears_ok[..., None], ears, 0.0).sum(axis=2),
        ears_count[..., None],
        out=nan2.copy(),
        where=ears_count[..., None] > 0,
    )
    neck = points[:, :, 3, :]
    nose = points[:, :, 0, :]
    head_points = points[:, :, [0, 1, 2], :]
    head_ok = np.isfinite(head_points).all(axis=3)
    head_count = head_ok.sum(axis=2)
    head = np.divide(
        np.where(head_ok[..., None], head_points, 0.0).sum(axis=2),
        head_count[..., None],
        out=nan2.copy(),
        where=head_count[..., None] > 0,
    )
    nose_missing = ~np.isfinite(nose).all(axis=2)
    extrapolated = np.isfinite(ear_mid).all(axis=2) & np.isfinite(neck).all(axis=2)
    head = np.where((nose_missing & extrapolated)[..., None], ear_mid + (ear_mid - neck), head)

    still_missing = nose_missing & ~extrapolated
    if boxes is not None and np.any(still_missing):
        front = _box_front_edge(boxes, posterior)
        head = np.where(still_missing[..., None], front, head)
    return posterior, head, anchor_ok


def _box_front_edge(boxes: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    """Box edge midpoint farthest from the posterior anchor, per frame per mouse."""

    x0, y0, x1, y1 = (boxes[..., index] for index in range(4))
    centre_x = (x0 + x1) / 2.0
    centre_y = (y0 + y1) / 2.0
    candidates = np.stack((
        np.stack((centre_x, y0), axis=-1),
        np.stack((centre_x, y1), axis=-1),
        np.stack((x0, centre_y), axis=-1),
        np.stack((x1, centre_y), axis=-1),
    ), axis=0)
    distances = np.linalg.norm(candidates - posterior[None, ...], axis=-1)
    best = np.argmax(np.where(np.isfinite(distances), distances, -np.inf), axis=0)
    gathered = np.take_along_axis(
        candidates.reshape(4, -1, 2), best.reshape(1, -1, 1), axis=0
    ).reshape(candidates.shape[1:])
    return np.where(np.isfinite(posterior).all(axis=2, keepdims=True), gathered, np.nan)


def _box_morphology_and_wall(
    boxes: np.ndarray,
    *,
    baseline_percentile: float = 85.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Box shrink ratio and distance to the arena envelope.

    ``boxes`` is (frames, mice, 4) as x0, y0, x1, y1 in cm.

    ``rho_box`` is the box major axis max(W, H) divided by that track's own
    ``baseline_percentile``-th percentile major axis, so a mouse that rears up and
    foreshortens in a top view drops below 1 against its own upright baseline
    instead of against an absolute centimetre guess.

    ``d_wall`` is the smallest margin from the box edges to the whole-video
    trajectory envelope [Xmin, Ymin, Xmax, Ymax] taken over every track and frame.
    That envelope is the arena extent the recording actually evidences - no wall is
    located in the data - so the track that reaches an extreme measures 0 there,
    which is exactly the "touching the wall" reading the plan asks for.

    Returns ``(width, height, rho_box, d_wall)``.
    """

    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[..., 3] - boxes[..., 1]
    major = np.maximum(width, height)
    finite = np.isfinite(major)
    baseline = np.full(major.shape[1:], np.nan, dtype=float)
    for slot in range(major.shape[1]):
        values = major[:, slot][finite[:, slot]]
        if len(values):
            baseline[slot] = float(np.percentile(values, float(baseline_percentile)))
    rho_box = np.divide(major, baseline[None, ...], out=np.full_like(major, np.nan),
                        where=finite & (baseline[None, ...] > 0))

    usable = np.isfinite(boxes).all(axis=2)
    flat = boxes[usable]
    if not len(flat):
        return width, height, rho_box, np.full_like(major, np.nan)
    envelope = (flat[:, 0].min(), flat[:, 1].min(), flat[:, 2].max(), flat[:, 3].max())
    margins = np.stack((
        boxes[..., 0] - envelope[0],
        boxes[..., 1] - envelope[1],
        envelope[2] - boxes[..., 2],
        envelope[3] - boxes[..., 3],
    ), axis=-1)
    d_wall = np.where(usable, margins.min(axis=-1), np.nan)
    return width, height, rho_box, d_wall


def _context_range(values: np.ndarray, frames: int) -> np.ndarray:
    """Peak-to-peak range over a context window that never shrinks at clip edges.

    ``_range`` centres the window and edge-pads it, so on a clip whose length is
    about one window the first and last frames only ever see a one-sided half of
    the data and their excursion is systematically underestimated. The rearing
    context is 2.0 s and the calibrated rearing clip is 2.0 s long, which is
    exactly that degenerate case: the window slid off both ends and cost the
    annotated track 13 of its 47 frames. This clamps the window back inside the
    real samples instead, so a boundary frame is judged on the full context that
    actually exists rather than on a truncated slice of it.
    """

    values = np.asarray(values, dtype=float)
    length = values.shape[0]
    frames = max(1, int(frames))
    if length <= 1:
        return np.zeros(values.shape, dtype=float)
    width = min(max(1, int(frames)), length)
    radius = width // 2
    # Clamp the window back inside the real samples instead of padding it, so a
    # boundary frame is judged on `width` genuine observations.
    starts = np.clip(np.arange(length) - radius, 0, max(length - width, 0))
    windows = np.stack([values[starts + offset] for offset in range(width)], axis=0)
    finite = np.isfinite(windows)
    high = np.max(np.where(finite, windows, -np.inf), axis=0)
    low = np.min(np.where(finite, windows, np.inf), axis=0)
    return np.where(np.any(finite, axis=0), high - low, np.nan)

def _windowed_correlation(left: np.ndarray, right: np.ndarray, frames: int) -> np.ndarray:
    """Dimensionless (Pearson) covariance of two signals over a trailing window.

    Used for the head/neck co-direction arm of wall rearing. A raw covariance would
    carry cm^2 units and scale by 0.74^2 under the frozen-scale migration; the
    normalised form is invariant, so the plan's 0.05 threshold stays meaningful.
    """

    frames = max(3, int(frames))
    left_mean = _trailing_mean(left, frames)
    right_mean = _trailing_mean(right, frames)
    product = _trailing_mean(left * right, frames)
    left_square = _trailing_mean(left * left, frames)
    right_square = _trailing_mean(right * right, frames)
    covariance = product - left_mean * right_mean
    spread = np.sqrt(np.maximum(left_square - left_mean ** 2, 0.0) *
                     np.maximum(right_square - right_mean ** 2, 0.0))
    return np.divide(covariance, spread, out=np.zeros_like(covariance), where=spread > 1e-9)

def individual_rule_masks(
    centers_cm: np.ndarray,
    valid: np.ndarray,
    fps: float,
    keypoints_cm: np.ndarray | None = None,
    boxes_cm: np.ndarray | None = None,
    reference_body_cm: float = 7.918,
    config: IndividualHeuristicConfig = DEFAULT_INDIVIDUAL_CONFIG,
) -> IndividualRuleResult:
    """Return mutually exclusive masks; missing posture does not veto motion.

    Keypoint order: nose, two ears, neck, two hind limbs, tail base.
    ``boxes_cm`` is the optional (frames, mice, 4) detection box as x0, y0, x1, y1
    in centimetres; posture rules use it for box shrinkage, wall proximity and
    grooming pulsation, and degrade to keypoint-only evidence when it is absent.
    The returned rearing mask is evidence from a top-view projection and box
    morphology, not a measurement of vertical body angle or proof of a known wall
    location.
    """
    centers = np.asarray(centers_cm, float)
    observed = np.asarray(valid, bool)
    if centers.shape != (*observed.shape, 2) or observed.ndim != 2:
        raise ValueError("centers must be (frames, mice, 2), valid (frames, mice)")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(reference_body_cm) or reference_body_cm <= 0:
        raise ValueError("fps and reference_body_cm must be positive and finite")
    if boxes_cm is not None:
        boxes = np.asarray(boxes_cm, float)
        if boxes.shape != (*np.asarray(valid).shape, 4):
            raise ValueError("boxes must be (frames, mice, 4) as x0, y0, x1, y1")
    else:
        boxes = None
    masks = {name: np.zeros_like(observed) for name in ("stationary", "locomotion", "grooming", "rearing")}
    if not len(centers):
        return IndividualRuleResult(masks, {})
    observed = observed & np.isfinite(centers).all(axis=2)
    centers = np.where(observed[..., None], centers, np.nan)
    smoothed = _median(centers, round(config.smooth_window_s * fps))
    smoothed[~observed] = np.nan
    velocity = _velocity(smoothed, observed, fps, config.motion_window_s)
    speed = _rolling_mean_path_speed(
        smoothed, observed, fps, config.motion_window_s,
        max_gap_s=config.motion_interpolation_max_gap_s,
    )
    diagnostics = {"body_speed_cm_s": speed, "body_velocity_cm_s": velocity}
    low = observed & np.isfinite(speed) & (speed <= config.motion_speed_threshold_cm_s)
    # Measure a short excursion itself using adjacent observed centers.
    raw_step_speed = np.full_like(speed, np.nan)
    raw_step_speed[1:] = np.linalg.norm(centers[1:] - centers[:-1], axis=2) * fps
    # [v16.2 IND-2] A trailing 0.3 s mean lags by roughly half its window, so it
    # erodes BOTH ends of a standstill: a true 1.0 s pause (30 frames at 30 fps)
    # leaves only 0.9333 s of `low` and therefore fails static_min_duration_s =
    # 1.0 s before pause bridging is even considered. The instantaneous
    # adjacent-frame step is sharp, so a frame that genuinely did not move is
    # stationary evidence in its own right. This only widens the CANDIDATE set:
    # bounded_stationary_mask still requires every observed centre of an accepted
    # bout to stay within static_max_radius_cm of the bout median, so a slow drift
    # that travels further than the static bound can never be absorbed. Frame 0 has
    # no preceding sample and stays NaN rather than being asserted "at rest".
    low |= observed & np.isfinite(raw_step_speed) & (
        raw_step_speed <= config.motion_speed_threshold_cm_s
    )
    for slot in range(observed.shape[1]):
        for start, end in _runs(~low[:, slot]):
            if not (0 < start < end < len(centers) and observed[start:end, slot].all()):
                continue
            raw_active = np.count_nonzero(raw_step_speed[start:end, slot] > config.motion_speed_threshold_cm_s)
            excursion = np.linalg.norm(centers[start:end, slot] - centers[start - 1, slot], axis=1)
            if (raw_active / fps <= config.static_transient_max_s + 1e-9
                    and np.max(excursion) <= config.static_transient_max_displacement_cm
                    and np.linalg.norm(centers[end, slot] - centers[start - 1, slot]) <= config.static_max_radius_cm):
                low[start:end, slot] = True
    stationary_gaps: list[tuple[int, int, int]] = []
    stationary = bounded_stationary_mask(
        low, smoothed, observed, fps=fps,
        max_gap_s=config.static_transient_max_s,
        max_missing_gap_s=config.static_detection_gap_max_s,
        max_transient_displacement_cm=config.static_transient_max_displacement_cm,
        max_center_radius_cm=config.static_max_radius_cm,
        bridged_out=stationary_gaps,
        max_bridged_missing_fraction=config.static_bridged_unobserved_max_fraction,
    )
    stationary = _qualify(stationary, config.static_min_duration_s, fps)
    locomotion = observed & np.isfinite(speed) & (speed > config.motion_speed_threshold_cm_s)
    # Displacement prevents a long, slow drift from disappearing between two
    # speed categories. Rest confined to the spatial bound remains Static.
    local_excursion = np.linalg.norm(_range(smoothed, round(2 * fps)), axis=2)
    locomotion |= observed & ~stationary & (local_excursion > config.static_max_radius_cm)
    # [v16.2 IND-2] A bout that already qualified as Static survived the spatial
    # bound, which is stronger evidence of rest than the lagging trailing mean at
    # that bout's edges. Let it keep its own frames instead of losing them to
    # smeared locomotion here and then to `stationary &= ~locomotion` below. This
    # is the second half of "an already-qualified stationary bout is never consumed
    # by locomotion"; the pause bridge guard is the first half.
    locomotion &= ~stationary
    # Frames a bounded posture bridge filled in; see _enforce_observation_bound.
    posture_bridged = np.zeros_like(observed)
    if keypoints_cm is not None and config.enable_posture_screen:
        points = np.asarray(keypoints_cm, float)
        if points.shape != (*observed.shape, 7, 2):
            raise ValueError("keypoints must be (frames, mice, 7, 2)")
        raw_points = np.where(observed[..., None, None], points, np.nan)
        points = _median(raw_points, round(config.smooth_window_s * fps))
        points[~observed] = np.nan

        # [v16.2 module 2 / DEFECT POST-3] The all-7-keypoints gate is gone. It used
        # to require every one of the 7 keypoints to be finite on every frame,
        # so one paw over the nose while washing the face, or one raised-head frame
        # while standing, voided an entire posture bout. The posterior anchor now
        # needs >= 2 of its 3 points and the head proxy is reconstructed when the
        # nose drops out; smoothed points drive position ranges, unsmoothed points
        # drive the high-frequency step speeds.
        posterior, head, anchor_ok = _robust_posterior_and_head(
            points, observed, boxes,
            min_posterior_points=config.posterior_min_valid_points,
        )
        raw_posterior, raw_head, _ = _robust_posterior_and_head(
            raw_points, observed, boxes,
            min_posterior_points=config.posterior_min_valid_points,
        )
        pose_observed = observed & anchor_ok & np.isfinite(head).all(axis=2)

        neck = points[:, :, 3]
        tail_base = points[:, :, 6]
        axis = neck - tail_base
        axis_length = np.linalg.norm(axis, axis=2)
        unit = np.divide(axis, axis_length[..., None], out=np.zeros_like(axis),
                         where=axis_length[..., None] > 1e-6)
        relative_head = head - posterior
        relative_neck = neck - posterior

        context = round(config.grooming_context_window_s * fps)
        rear_frames = round(config.rearing_context_window_s * fps)
        posterior_excursion = np.linalg.norm(_range(posterior, context), axis=2)
        rear_posterior_excursion = np.linalg.norm(_context_range(posterior, rear_frames), axis=2)
        projection = np.linalg.norm(points[:, :, 0] - tail_base, axis=2)
        projection_change = _context_range(projection, rear_frames)

        # --- detection-box morphology (optional) --------------------------
        if boxes is not None:
            box_width, box_height, rho_box, d_wall = _box_morphology_and_wall(
                boxes, baseline_percentile=config.box_baseline_percentile,
            )
            box_major_change = _context_range(np.maximum(box_width, box_height), rear_frames)
            box_pulsation = np.maximum(_range(box_width, context), _range(box_height, context))
            box_pulsation = np.where(np.isfinite(box_pulsation), box_pulsation, -np.inf)
            near_wall = np.isfinite(d_wall) & (
                d_wall <= config.rearing_max_wall_distance_body_ratio * reference_body_cm)
            shrink_ok = np.isfinite(rho_box) & (rho_box <= config.rearing_max_box_shrink_ratio)
            pulsation_ok = box_pulsation >= config.grooming_box_pulsation_body_ratio * reference_body_cm
        else:
            # No box evidence: the box-only gates stay non-restrictive so keypoint
            # callers keep their previous behaviour, and the grooming pulsation OR
            # arm falls back to the nose/ear co-direction term alone.
            rho_box = np.full(observed.shape, np.nan)
            d_wall = np.full(observed.shape, np.nan)
            box_major_change = np.full(observed.shape, np.nan)
            box_pulsation = np.full(observed.shape, np.nan)
            shrink_ok = np.ones(observed.shape, dtype=bool)
            near_wall = np.ones(observed.shape, dtype=bool)
            pulsation_ok = np.zeros(observed.shape, dtype=bool)

        # --- short-window step speeds on UNSMOOTHED keypoints --------------
        step_frames = max(1, int(round(config.grooming_step_window_s * fps)))
        dead_zone = config.grooming_step_dead_zone_body_ratio * reference_body_cm
        step_nose = _step_speed(raw_points[:, :, 0] - raw_posterior, fps, step_frames, dead_zone)
        step_posterior = _step_speed(raw_posterior, fps, step_frames, dead_zone)
        step_head = _step_speed(raw_head - raw_posterior, fps, step_frames, dead_zone)
        speed_ratio = np.divide(step_nose, step_posterior, out=np.full_like(step_nose, np.nan),
                                where=np.isfinite(step_posterior) & (step_posterior > 0))
        head_ratio = np.divide(step_head, step_posterior, out=np.full_like(step_head, np.nan),
                               where=np.isfinite(step_posterior) & (step_posterior > 0))

        # --- grooming ------------------------------------------------------
        stable = pose_observed & np.isfinite(speed) & (
            speed <= config.motion_speed_threshold_cm_s)
        stable &= posterior_excursion <= config.posterior_max_excursion_body_ratio * reference_body_cm
        axial = np.sum((head - neck) * unit, axis=2)
        lateral = (head - neck)[..., 0] * unit[..., 1] - (head - neck)[..., 1] * unit[..., 0]
        head_velocity = _velocity(relative_head, pose_observed, fps, config.grooming_step_window_s)
        axial_velocity = np.sum(head_velocity * unit, axis=2)
        lateral_velocity = head_velocity[..., 0] * unit[..., 1] - head_velocity[..., 1] * unit[..., 0]

        def reversal_count(signal: np.ndarray) -> np.ndarray:
            signs = np.sign(np.where(np.abs(signal) >= 0.02 * reference_body_cm, signal, 0))
            reversals = np.zeros_like(signs)
            reversals[1:] = ((signs[1:] * signs[:-1]) < 0).astype(float)
            return np.sum(_window(reversals, context), axis=-1)

        axial_amp = _range(axial, context)
        lateral_amp = _range(lateral, context)
        min_amplitude = config.grooming_min_head_amplitude_body_ratio * reference_body_cm
        seated = np.isfinite(projection) & (
            projection <= config.grooming_max_seated_projection_body_ratio * reference_body_cm)
        fore_aft = step_nose >= config.grooming_min_step_nose_speed_body_ratio * reference_body_cm
        fore_aft &= _positive_motion_dominance(
            step_nose, step_posterior, config.grooming_min_nose_posterior_speed_ratio)
        fore_aft &= _positive_motion_dominance(
            step_head, step_posterior, config.grooming_min_head_posterior_speed_ratio)
        fore_aft &= seated
        fore_aft &= reversal_count(axial_velocity) >= config.grooming_min_reversals

        nose_relative = raw_points[:, :, 0] - raw_posterior
        ear_pair = raw_points[:, :, [1, 2], :]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ear_mean = np.nanmean(ear_pair, axis=2)
        ears_relative = ear_mean - raw_posterior
        nose_velocity = _velocity(nose_relative, pose_observed, fps, config.grooming_step_window_s)
        ears_velocity = _velocity(ears_relative, pose_observed, fps, config.grooming_step_window_s)
        nose_axial_velocity = np.sum(nose_velocity * unit, axis=2)
        ears_axial_velocity = np.sum(ears_velocity * unit, axis=2)
        active_nose = np.abs(nose_axial_velocity) >= .02 * reference_body_cm
        coherent = (nose_axial_velocity * ears_axial_velocity > 0) & active_nose
        coherence = np.sum(_window(coherent, context), axis=-1) / np.maximum(
            np.sum(_window(active_nose, context), axis=-1), 1)
        fore_aft &= pulsation_ok | (coherence >= config.grooming_min_nose_ear_codirection)

        nose_back = np.sum((points[:, :, 0] - neck) * unit, axis=2) < 0
        back_distance = np.linalg.norm(points[:, :, 0] - posterior, axis=2)
        back = nose_back & (
            back_distance <= config.back_groom_max_nose_posterior_body_ratio * reference_body_cm)
        back &= (np.maximum(axial_amp, lateral_amp) >= min_amplitude)
        back &= (np.maximum(reversal_count(axial_velocity), reversal_count(lateral_velocity))
                 >= config.grooming_min_reversals)
        back &= np.linalg.norm(_range(nose_relative, context), axis=2) >= min_amplitude

        grooming_evidence = stable & (fore_aft | back)
        grooming_evidence, grooming_filled = _bridge_gaps(
            grooming_evidence, round(config.grooming_bridge_max_s * fps))
        grooming = _qualify(grooming_evidence, config.grooming_min_duration_s, fps)

        # --- rearing -------------------------------------------------------
        rear_stable = pose_observed & (
            rear_posterior_excursion
            <= config.rearing_max_posterior_excursion_body_ratio * reference_body_cm)
        short_projection = np.isfinite(projection) & (
            projection <= config.rearing_max_short_projection_body_ratio * reference_body_cm)
        rear_segment = neck - posterior
        front_segment = head - neck
        rear_length = np.linalg.norm(rear_segment, axis=2)
        front_length = np.linalg.norm(front_segment, axis=2)
        spine_collinearity = np.divide(
            np.sum(rear_segment * front_segment, axis=2),
            rear_length * front_length,
            out=np.zeros_like(rear_length),
            where=(rear_length * front_length) > 1e-9,
        )
        dynamic_change = np.maximum(
            np.where(np.isfinite(projection_change), projection_change, -np.inf),
            np.where(np.isfinite(box_major_change), box_major_change, -np.inf),
        )
        # Scene A: open-field rearing, the body foreshortens in the top view.
        scene_a = short_projection & shrink_ok
        scene_a &= spine_collinearity >= config.rearing_min_spine_collinearity
        scene_a &= dynamic_change >= (
            config.rearing_min_projection_change_body_ratio * reference_body_cm)
        # Scene B: wall/climbing rearing, the projection need not shorten but the
        # box sits against the arena envelope and head and neck both sweep.
        neck_excursion = np.linalg.norm(_context_range(relative_neck, rear_frames), axis=2)
        head_excursion = np.linalg.norm(_context_range(relative_head, rear_frames), axis=2)
        head_neck_cos = np.sum(_velocity(relative_head, pose_observed, fps, 0.3)
                               * _velocity(relative_neck, pose_observed, fps, 0.3), axis=2)
        # The plan asks for the head/neck DISPLACEMENT covariance, so this
        # correlates per-frame increments, not positions. Correlating the position
        # projections returned ~1.0 for every track in both calibration samples -
        # two slowly varying projections of a near-rigid body co-vary trivially -
        # which made the whole OR arm a no-op. Increments only co-vary when the
        # head and the neck are genuinely being driven together.
        head_axial = np.sum(relative_head * unit, axis=2)
        neck_axial = np.sum(relative_neck * unit, axis=2)
        window_covariance = _windowed_correlation(
            np.diff(head_axial, axis=0, prepend=head_axial[:1]),
            np.diff(neck_axial, axis=0, prepend=neck_axial[:1]),
            rear_frames,
        )
        sweep = (neck_excursion >= config.wall_min_neck_excursion_body_ratio * reference_body_cm)
        sweep &= head_excursion >= config.wall_min_neck_excursion_body_ratio * reference_body_cm
        scene_b = near_wall & sweep
        scene_b &= (head_neck_cos > 0) | (
            window_covariance >= config.rearing_min_head_neck_covariance)

        rearing_evidence = rear_stable & (scene_a | scene_b) & ~grooming
        rearing_evidence, rearing_filled = _bridge_gaps(
            rearing_evidence, round(config.rearing_bridge_max_s * fps))
        rearing_evidence &= ~grooming
        rearing = _qualify(rearing_evidence, config.rearing_min_duration_s, fps)

        posture_bridged |= grooming_filled | rearing_filled
        wall_pattern = scene_b & rear_stable
        masks["grooming"], masks["rearing"] = grooming, rearing
        diagnostics.update({"head_relative_speed_cm_s": np.linalg.norm(head_velocity, axis=2),
                            "posterior_excursion_cm": posterior_excursion,
                            "projection_cm": projection, "projection_change_cm": projection_change,
                            "grooming_fore_aft": fore_aft & stable,
                            "grooming_back": back & stable, "wall_like_pattern": wall_pattern,
                            "rho_box": rho_box, "d_wall_cm": d_wall,
                            "spine_collinearity": spine_collinearity,
                            "head_neck_window_covariance": window_covariance,
                            "grooming_step_nose_speed_cm_s": step_nose,
                            "grooming_step_posterior_speed_cm_s": step_posterior,
                            "grooming_nose_posterior_speed_ratio": speed_ratio,
                            "grooming_head_posterior_speed_ratio": head_ratio})
    posture = masks["grooming"] | masks["rearing"]
    locomotion &= ~posture
    # [v16.2 IND-7 / IND-2 / IND-1] Bounded locomotion pause bridging.
    #
    # A gap is refused when any of these hold:
    #   * it is longer than locomotion_pause_max_s;
    #   * it holds an already-qualified Static bout (IND-2: the pause ceiling and
    #     static_min_duration_s are both 1.0 s, so such a bout used to be filled in
    #     here and then erased by `stationary &= ~locomotion` before it could ever
    #     be reported - measured stationary recall 0.500 -> 0.375, FN 8 -> 10, while
    #     locomotion FN went 3 -> 0). Sub-threshold pauses and short dropouts carry
    #     no qualified Static bout and still bridge, which preserves that win;
    #   * it holds any qualified posture (IND-1: grooming and rearing are both
    #     _qualify()-ed to >= grooming_min_duration_s / rearing_min_duration_s, i.e.
    #     >= 9 frames at 30 fps, so a posture interruption can never be a sub-frame
    #     flicker. The old `gap_frames > posture_gap_limit` guard was therefore
    #     unreachable, and bridging across posture is undone by the final
    #     `& ~posture` below regardless, so refusing outright loses nothing);
    #   * its own unobserved span exceeds static_detection_gap_max_s (IND-7 (a));
    #   * the cumulative unobserved time of the resulting bout would exceed
    #     locomotion_bridged_unobserved_max_s, or
    #     locomotion_bridged_unobserved_max_fraction of that bout (IND-7 (b)). This
    #     is what stops short dropouts chaining into a mostly fabricated bout.
    # Every frame this loop fills is recorded in `locomotion_bridged` so each
    # emitted mask can be checked against it (IND-7 (c)).
    locomotion_bridged = np.zeros_like(observed)
    for slot in range(observed.shape[1]):
        segments = list(_runs(locomotion[:, slot]))
        if not segments:
            continue
        anchor_start = segments[0][0]
        bridged_unobserved = 0
        for (_, left_end), (right_start, right_end) in zip(segments, segments[1:]):
            gap = slice(left_end, right_start)
            gap_frames = right_start - left_end
            gap_unobserved = int(np.count_nonzero(~observed[gap, slot]))
            refused = (
                gap_frames <= 0
                or gap_frames / fps > config.locomotion_pause_max_s + 1e-9
                or stationary[gap, slot].any()
                or posture[gap, slot].any()
                or gap_unobserved / fps > config.static_detection_gap_max_s + 1e-9
            )
            if not refused:
                cumulative = bridged_unobserved + gap_unobserved
                bout_length = right_end - anchor_start
                refused = (
                    cumulative / fps > config.locomotion_bridged_unobserved_max_s + 1e-9
                    or cumulative
                    > config.locomotion_bridged_unobserved_max_fraction * bout_length
                )
            if refused:
                anchor_start = right_start
                bridged_unobserved = 0
                continue
            locomotion[gap, slot] = True
            locomotion_bridged[gap, slot] = True
            bridged_unobserved += gap_unobserved
    stationary &= ~locomotion
    stationary_bridged = np.zeros_like(observed)
    for slot, gap_start, gap_end in stationary_gaps:
        stationary_bridged[gap_start:gap_end, slot] = True
    masks["grooming"] = _enforce_observation_bound(
        masks["grooming"], observed, posture_bridged, "grooming")
    masks["rearing"] = _enforce_observation_bound(
        masks["rearing"], observed, posture_bridged, "rearing")
    masks["stationary"] = _enforce_observation_bound(
        _qualify(stationary & ~posture, config.static_min_duration_s, fps),
        observed, stationary_bridged, "stationary",
    )
    masks["locomotion"] = _enforce_observation_bound(
        _qualify(locomotion & ~posture, config.locomotion_min_duration_s, fps),
        observed, locomotion_bridged, "locomotion",
    )
    return IndividualRuleResult(masks, diagnostics)
