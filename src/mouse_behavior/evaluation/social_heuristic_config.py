"""Versioned, decoupled kinematic limits for social behavior rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping
import warnings

from .retrospective_thresholds_v17 import SOCIAL_EXTREMA
from .retrospective_thresholds_v18 import SOCIAL_FITTED
from .requested_thresholds_v28 import (
    APPROACH_CENTER_END_MIN_CM, APPROACH_SURFACE_END_MIN_CM, APPROACH_SURFACE_END_MAX_CM,
    TOGETHER_MAX_SPEED_CM_S, CONTACT_MIN_DURATION_S, CHASING_MIN_MEAN_SPEED_CM_S,
    CONTACT_HEAD_ONSET_DISTANCE_CM, CONTACT_TAIL_ONSET_DISTANCE_CM,
)

SOCIAL_CONFIG_REVISION = "2026-10-20-v28-contact-context-and-v26-restoration"


@dataclass(frozen=True)
class SocialHeuristicConfig:
    """Full-dataset retrospective limits at the 7.918 cm reference scale.

    v17 follows the user's 2026-10-16 policy. Full-dataset extrema are fitted
    engineering settings, not independently validated biological boundaries.
    Fields marked RETIRED remain serializable and warn on custom values.
    """

    attack_priority: int = 3
    attack_min_duration_s: float = 0.15
    dist_attack_max_cm: float = SOCIAL_FITTED["dist_attack_max_cm"]
    # Maximum no-tail footprint gap at the five video-audited center-far
    # contact frames (2b003:18,19,23,52; 892262:12). Retrospective proxy fit.
    attack_contact_surface_max_gap_cm: float = 1.1925119161605835
    attack_contact_max_footprint_body_lengths: float = 2.0
    # Maximum of within-action minimum actor-nose/target-non-tail-hull gaps
    # across three video-reviewed recovered pairs. Selected-case retrospective
    # confirmation, not a full-cohort extremum or a universal contact radius.
    attack_new_action_target_surface_max_gap_cm: float = 1.9986556302093281
    acc_attack_min_cm_s2: float = SOCIAL_FITTED["acc_attack_min_cm_s2"]
    vel_attack_min_cm_s: float = SOCIAL_FITTED["vel_attack_min_cm_s"]
    # Longest bounded target dropout in the twelve v21-loss video audits:
    # 977efbcaa730be97b4fc0dd1, target291, frames [5,33), real fps.
    # Retrospective engineering fit; repeated same-target cycles still required.
    attack_box_gap_max_s: float = 28. / 29.967457227456368
    attack_box_bridge_max_s: float = 3.5
    # SOC-11 (v16.2): the anchored single-long-gap path is genuine >=1-cycle
    # evidence, so __post_init__ now accepts one dropout/reappearance cycle and
    # its message no longer describes a rule that was retired. The default stays
    # at 2 because the repeated-cycle path (Path 2A) is unchanged.
    attack_box_reappearance_min_cycles: int = 2
    attack_box_repeat_window_s: float = 2.0
    attack_pounce_max_duration_s: float = 0.75  # anchored occlusion paint only
    attack_deceleration_max_s: float = 0.75
    angle_attack_toward_max_deg: float = 60.0
    attack_acceleration_fit_half_window_s: float = 0.125
    attack_max_track_jump_speed_cm_s: float = 120.0
    # SOC-1 (v16.2, era-safe durations): anchor windows for the anchored
    # box-gap path (Path 2B). `attack_anchor_window_s` replaces the hardcoded
    # +/-0.3 s literal; `attack_anchor_min_sustained_s` is the reciprocal
    # motion each side of the gap must show when neither a closing pair nor a
    # partner-directed velocity supplies engagement evidence. Together they
    # replace "ONE frame with speed >= vel_attack_min_cm_s on EITHER side" as
    # the whole positive case.
    attack_anchor_window_s: float = 0.3
    attack_anchor_min_sustained_s: float = 0.15
    # SOC-3 (v16.2, era-safe angle + duration): stable tail-follow suppression.
    # The leader's walking floor is NOT a new number - confirmed_rules uses the
    # individual layer's own Locomotion boundary
    # (DEFAULT_INDIVIDUAL_CONFIG.motion_speed_threshold_cm_s), so it stays in
    # the same scale era as the individual layer.
    attack_tail_follow_min_duration_s: float = 0.3
    attack_tail_follow_max_heading_angle_deg: float = 45.0

    nose_contact_priority: int = 5
    nose_tail_priority: int = 6
    nose_contact_min_duration_s: float = CONTACT_MIN_DURATION_S
    nose_contact_max_duration_s: float = 15.0  # review flag only; long contacts remain positive
    nose_contact_candidate_distance_cm: float = 3.0
    nose_contact_context_max_displacement_cm: float = 1.3375
    nose_contact_exit_distance_cm: float = 2.0  # RETIRED gate; ordering check + v12 mapping only
    nose_contact_gap_grace_s: float = 0.1  # RETIRED; superseded by contact_bridge_max_gap_s
    dist_nose_head_contact_max_cm: float = SOCIAL_FITTED["dist_nose_head_contact_max_cm"]
    dist_nose_tail_contact_max_cm: float = SOCIAL_EXTREMA["dist_nose_tail_contact_max_cm"]
    # Retrospective p90 of per-clip median directed endpoint distances. These
    # tighter short-bout limits add measured support; legacy limits remain the
    # baseline and still serve the attack tail-follow guard.
    contact_head_extension_distance_cm: float = 2.2344581844846205
    contact_tail_extension_distance_cm: float = 3.127273859004017
    contact_head_extension_min_duration_s: float = CONTACT_MIN_DURATION_S
    contact_tail_extension_min_duration_s: float = 0.15  # restore V26 tail timing
    vel_nose_probing_min_cm_s: float = 0.2  # RETIRED optional context, never required
    angle_nose_probing_max_deg: float = 60.0  # RETIRED optional context, never required
    contact_target_margin_cm: float = 0.35  # RETIRED v12 setting; endpoints are independent
    # SOC-5 / SOC-4 (v16.2, era-safe): contact gap bridging.
    # * contact_bridge_max_gap_s centralizes the three hardcoded 0.15 s
    #   literals (head contact, tail contact, approach contact veto).
    # * contact_bridge_support_max_body_lengths is a DEDICATED support radius
    #   for the bridge. It must not reuse dist_together_max_cm: nose-to-tail-root
    #   contact puts the two body centres around 7-8 cm apart, so the 6.0 cm
    #   Together threshold made tail-contact flicker unbridgeable. It is a
    #   multiple of reference_body_cm, so it is invariant to the scale migration.
    #   1.25 matches the body-length factor the group huddle rule already uses.
    # * contact_signal_max_body_lengths decides when a FINITE endpoint distance
    #   is a credible non-contact measurement. An endpoint distance above
    #   centre_distance + k * reference_body_cm is geometrically impossible, so
    #   it is an occlusion artefact (unmeasured, bridgeable) rather than proof
    #   of separation. A finite value inside that ceiling WAS measured and must
    #   split the bout, which is what stops six 1-frame blips separated by
    #   measured 5.0 cm frames from chaining into a 0.8 s "contact".
    contact_bridge_max_gap_s: float = 0.05
    contact_bridge_support_max_body_lengths: float = 1.25
    contact_signal_max_body_lengths: float = 1.25

    chasing_priority: int = 4
    chasing_min_duration_s: float = 1.0
    dist_chasing_max_cm: float = 10.0
    vel_chasing_min_cm_s: float = SOCIAL_EXTREMA["vel_chasing_min_cm_s"]
    vel_chasing_target_min_cm_s: float = SOCIAL_EXTREMA["vel_chasing_target_min_cm_s"]
    chasing_min_net_displacement_cm: float = 10.7
    chasing_min_path_displacement_cm: float = 10.7
    chasing_track_gap_max_s: float = 0.4  # RETIRED; superseded by chasing_reacquisition_gap_max_s
    chasing_reacquisition_gap_max_s: float = 3.5
    angle_chasing_dir_max_deg: float = 45.0
    angle_chasing_behind_max_deg: float = 60.0
    # DOC-2 / spec section 3.2 (v16.2, dimensionless): the cross-gap
    # consistency test compares the NET speed across the gap against this factor
    # of the in-window chase speed floor. Demanding the full floor as a net
    # speed over a 3.5 s gap needs >= 93.6 cm of net travel per mouse, which
    # made the long-gap bridge inert. The factor is dimensionless, so it is
    # era-safe; vel_chasing_target_min_cm_s itself is NOT changed.
    chasing_cross_gap_speed_factor: float = 0.75
    # SOC-16 (v16.2, dimensionless count): how many directed nose-to-head-
    # triangle measurements must be finite for a frame to count as chase
    # support. 2 is the pre-v16.2 behaviour (a non-degenerate cv2 head triangle
    # on BOTH mice simultaneously), which a "only head and tail have IDs" clip
    # can never satisfy. 1 accepts either side.
    chasing_min_observed_sides: int = 1  # RETIRED: pursuit uses observed center motion
    # Full755 retrospective audit: pooled observed annotated actor-frame p99,
    # not independently validated classification limits. See directed_roles_v26
    # geometry_statistics.json and parameter_selection.json for sample support.
    chasing_head_facing_max_deg: float = 53.17451262688489
    chasing_body_facing_max_deg: float = 60.95376744220115
    chasing_role_turn_grace_s: float = 0.5  # existing bounded leader-turn context

    approach_priority: int = 1
    approach_min_duration_s: float = 0.3
    dist_approach_start_min_cm: float = 12.0
    dist_approach_end_min_cm: float = APPROACH_CENTER_END_MIN_CM
    dist_approach_end_max_cm: float = SOCIAL_EXTREMA["dist_approach_end_max_cm"]
    # V28 restores V26 endpoints while retaining the serialized midpoint API.
    approach_surface_endpoint_cm: float = (APPROACH_SURFACE_END_MIN_CM + APPROACH_SURFACE_END_MAX_CM) / 2
    approach_surface_endpoint_tolerance_cm: float = (APPROACH_SURFACE_END_MAX_CM - APPROACH_SURFACE_END_MIN_CM) / 2
    approach_stationary_target_max_path_cm: float = 1.3375  # RETIRED target-motion gate
    vel_approach_actor_min_cm_s: float = 3.0
    vel_approach_actor_target_ratio_min: float = 1.2  # RETIRED target-motion gate
    angle_approach_facing_max_deg: float = 60.0
    approach_head_facing_max_deg: float = 69.08646603254526
    approach_body_facing_max_deg: float = 72.34189423033574
    approach_min_closing_speed_cm_s: float = 0.2
    # SOC-6 / spec module 3 item 4 (v16.2, era-safe ratio + durations): the
    # approach run gate. `approach_window_min_valid_ratio` is the >= 80% valid-
    # frame ratio that replaces the whole-window zero-tolerance `.all()` tests
    # on visible / directed / distance-finite; the onset and endpoint frames are
    # still required valid individually. `approach_window_max_gap_s` bounds the
    # interior frames that may be bridged before runs are formed, and
    # `approach_trailing_contact_max_s` replaces the hardcoded 0.1 s (= 1-3
    # frames at 30 fps) trailing contact-noise trim the spec asks for.
    approach_window_min_valid_ratio: float = 0.8
    approach_window_max_gap_s: float = 0.15
    approach_trailing_contact_max_s: float = 0.1

    together_priority: int = 2
    together_min_duration_s: float = 1.0
    dist_together_max_cm: float = SOCIAL_FITTED["dist_together_max_cm"]
    vel_together_max_cm_s: float = TOGETHER_MAX_SPEED_CM_S
    together_use_smoothed_speed: bool = True
    # SOC-9 (v16.2, era-safe duration): a contact run at least this long vetoes
    # `together`. It is deliberately far below nose_contact_min_duration_s: a
    # SINGLE noisy keypoint frame must not fragment a huddle, but a real
    # sub-threshold contact run must not be relabelled as Together either.
    together_contact_veto_min_duration_s: float = 0.1
    # SOC-9 (v16.2, era-safe duration): `together` was the only social label
    # with no gap bridging, so one unobserved frame split a huddle and the 1.0 s
    # minimum rejected both halves. Same bounded unknown-frame bridging as the
    # contact rules; kept as its own field so the two can be tuned apart.
    together_bridge_max_gap_s: float = 0.15
    # Mean = sum of real adjacent eligible step lengths / adjacent step time.
    # Each mouse must pass; missing rows/jumps across gaps add neither operand.
    # Zero is supported only for explicit historical experiment configurations.
    chasing_min_mean_speed_cm_s: float = CHASING_MIN_MEAN_SPEED_CM_S
    # Separate tail baseline/bridge restore V26 without undoing head's gain.
    nose_tail_min_duration_s: float = 0.5
    contact_tail_bridge_max_gap_s: float = 0.15
    contact_head_onset_distance_cm: float = CONTACT_HEAD_ONSET_DISTANCE_CM
    contact_tail_onset_distance_cm: float = CONTACT_TAIL_ONSET_DISTANCE_CM

    def contact_minimum_s(self, kind: str) -> float:
        if kind not in ("head", "tail"):
            raise ValueError("contact kind must be head or tail")
        return self.nose_contact_min_duration_s if kind == "head" else self.nose_tail_min_duration_s

    def contact_onset_cm(self, kind: str) -> float:
        if kind not in ("head", "tail"):
            raise ValueError("contact kind must be head or tail")
        return self.contact_head_onset_distance_cm if kind == "head" else self.contact_tail_onset_distance_cm

    @property
    def priority_order(self) -> tuple[str, ...]:
        priorities = {
            "approach": self.approach_priority, "together": self.together_priority,
            "attack": self.attack_priority, "chasing": self.chasing_priority,
            "head_contact": self.nose_contact_priority, "tail_contact": self.nose_tail_priority,
        }
        return tuple(sorted(priorities, key=priorities.__getitem__))

    @property
    def retired_fields(self) -> dict[str, str]:
        """Fields no rule reads, with the reason they are still carried."""
        return {
            "chasing_min_observed_sides": "retired_unrelated_head_triangle_gate; observed_center_motion_supplies_direction",
            "approach_stationary_target_max_path_cm": "retired_target_path_gate; stationary_lateral_and_receding_targets_allowed",
            "vel_approach_actor_target_ratio_min": "retired_relative_speed_gate; actor_toward_and_measured_closing_required",
            "angle_attack_toward_max_deg": "retired_angle_cap; positive_partner_directed_motion_required",
            "angle_approach_facing_max_deg": "retired_heading_cap; measured_closing_motion_required",
            "dist_chasing_max_cm": "retired_center_distance_cap; directed_pursuit_required",
            "approach_trailing_contact_max_s": "retired_contact_veto; six_way_priority_resolves_overlap",
            "together_contact_veto_min_duration_s": "retired_veto; retained_for_contact_evidence_diagnostic",
            "attack_box_bridge_max_s": "retired_v21_single_long_gap_is_unknown",
            "attack_pounce_max_duration_s": "retired_v21_single_long_gap_paint_removed",
            "attack_anchor_min_sustained_s": "retired_v21_single_long_gap_reciprocal_gate_removed",
            "chasing_cross_gap_speed_factor": "retired_v21_missing_intervals_add_no_measured_travel",
            "chasing_track_gap_max_s": (
                "retired_not_read_by_any_rule; superseded_by chasing_reacquisition_gap_max_s"
            ),
            "nose_contact_gap_grace_s": (
                "retired_not_read_by_any_rule; superseded_by contact_bridge_max_gap_s"
            ),
            "nose_contact_exit_distance_cm": (
                "retired_gate; retained only for "
                "from_mapping() compatibility with serialized v12 settings"
            ),
            "vel_nose_probing_min_cm_s": "optional_context_never_required; not_read_by_any_rule",
            "angle_nose_probing_max_deg": "optional_context_never_required; not_read_by_any_rule",
            "contact_target_margin_cm": (
                "retired_v12_setting; head_and_tail_endpoints_are_independent_so_no_margin_is_read"
            ),
        }

    def __post_init__(self) -> None:
        if not isinstance(self.together_use_smoothed_speed, bool):
            raise ValueError("together_use_smoothed_speed must be bool")
        positive_fields = (
            "attack_min_duration_s", "dist_attack_max_cm",
            "attack_contact_surface_max_gap_cm", "attack_contact_max_footprint_body_lengths",
            "attack_new_action_target_surface_max_gap_cm",
            "attack_box_gap_max_s", "attack_box_bridge_max_s",
            "attack_box_repeat_window_s", "attack_pounce_max_duration_s",
            "attack_deceleration_max_s",
            "attack_acceleration_fit_half_window_s", "attack_max_track_jump_speed_cm_s",
            "attack_anchor_window_s", "attack_anchor_min_sustained_s",
            "attack_tail_follow_min_duration_s",
            "nose_contact_min_duration_s", "nose_contact_max_duration_s",
            "nose_contact_candidate_distance_cm", "nose_contact_exit_distance_cm",
            "nose_contact_context_max_displacement_cm",
            "nose_contact_gap_grace_s",
            "dist_nose_head_contact_max_cm", "dist_nose_tail_contact_max_cm",
            "contact_head_extension_distance_cm", "contact_tail_extension_distance_cm",
            "contact_head_extension_min_duration_s", "contact_tail_extension_min_duration_s",
            "vel_nose_probing_min_cm_s", "contact_target_margin_cm",
            "contact_bridge_max_gap_s",
            "nose_tail_min_duration_s", "contact_tail_bridge_max_gap_s",
            "contact_head_onset_distance_cm", "contact_tail_onset_distance_cm",
            "contact_bridge_support_max_body_lengths", "contact_signal_max_body_lengths",
            "chasing_min_duration_s", "dist_chasing_max_cm",
            "approach_min_duration_s",
            "chasing_min_net_displacement_cm", "chasing_min_path_displacement_cm",
            "chasing_track_gap_max_s", "chasing_reacquisition_gap_max_s",
            "chasing_role_turn_grace_s",
            "dist_approach_start_min_cm",
            "dist_approach_end_max_cm", "vel_approach_actor_min_cm_s",
            "approach_surface_endpoint_cm",
            "approach_stationary_target_max_path_cm",
            "vel_approach_actor_target_ratio_min", "approach_min_closing_speed_cm_s",
            "approach_window_max_gap_s", "approach_trailing_contact_max_s",
            "together_min_duration_s", "together_bridge_max_gap_s",
            "together_contact_veto_min_duration_s",
            "dist_together_max_cm", "vel_together_max_cm_s",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("vel_attack_min_cm_s", "acc_attack_min_cm_s2",
                     "vel_chasing_min_cm_s", "vel_chasing_target_min_cm_s",
                     "dist_approach_end_min_cm", "approach_surface_endpoint_tolerance_cm",
                     "chasing_min_mean_speed_cm_s"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "angle_attack_toward_max_deg", "angle_nose_probing_max_deg", "angle_chasing_dir_max_deg",
            "angle_chasing_behind_max_deg", "angle_approach_facing_max_deg",
            "attack_tail_follow_max_heading_angle_deg",
            "approach_head_facing_max_deg", "approach_body_facing_max_deg",
            "chasing_head_facing_max_deg", "chasing_body_facing_max_deg",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0 < value <= 180:
                raise ValueError(f"{name} must be in (0, 180]")
        for name in ("approach_window_min_valid_ratio", "chasing_cross_gap_speed_factor"):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be a dimensionless ratio in (0, 1]")
        if self.nose_contact_min_duration_s > self.nose_contact_max_duration_s:
            raise ValueError("nose-contact minimum duration cannot exceed its maximum")
        for name in ("attack_box_reappearance_min_cycles", "chasing_min_observed_sides"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        # SOC-11: one anchored dropout/reappearance cycle is genuine occluded-
        # attack evidence, so the floor is 1, not 2.
        if self.attack_box_reappearance_min_cycles < 1:
            raise ValueError(
                "occluded attack requires at least one dropout/reappearance cycle"
            )
        if not 1 <= self.chasing_min_observed_sides <= 2:
            raise ValueError("chasing_min_observed_sides must be 1 (either mouse) or 2 (both)")
        if self.attack_min_duration_s > self.attack_pounce_max_duration_s:
            raise ValueError("pounce minimum duration cannot exceed its maximum")
        if self.attack_anchor_min_sustained_s > self.attack_anchor_window_s:
            raise ValueError("anchor sustained motion must fit inside the anchor window")
        if self.attack_box_gap_max_s > self.attack_box_repeat_window_s:
            raise ValueError("attack repeat window must contain a bounded detection gap")
        if self.contact_bridge_max_gap_s >= self.nose_contact_min_duration_s:
            raise ValueError("a contact bridge must be shorter than the bout it supports")
        if self.contact_tail_bridge_max_gap_s >= self.nose_tail_min_duration_s:
            raise ValueError("a tail bridge must be shorter than the tail bout it supports")
        if self.nose_tail_min_duration_s > self.nose_contact_max_duration_s:
            raise ValueError("tail-contact minimum duration cannot exceed its maximum")
        if self.together_bridge_max_gap_s >= self.together_min_duration_s:
            raise ValueError("a together bridge must be shorter than the bout it supports")
        if self.together_contact_veto_min_duration_s > self.nose_contact_min_duration_s:
            raise ValueError(
                "the together contact veto cannot be longer than a qualifying contact bout"
            )
        if self.dist_approach_start_min_cm <= self.dist_approach_end_min_cm:
            raise ValueError("approach start distance must exceed the endpoint minimum")
        if self.dist_approach_end_min_cm > self.dist_approach_end_max_cm:
            raise ValueError("approach endpoint range must be increasing")
        if self.approach_surface_endpoint_tolerance_cm > self.approach_surface_endpoint_cm:
            raise ValueError("approach surface endpoint tolerance must keep the gap positive")
        priorities = (
            self.approach_priority, self.together_priority, self.attack_priority,
            self.chasing_priority, self.nose_contact_priority, self.nose_tail_priority,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in priorities):
            raise ValueError("social behavior priorities must be integers")
        if sorted(priorities) != list(range(1, 7)):
            raise ValueError("social behavior priorities must be a permutation of 1 through 6")
        changed_retired = [
            name for name in self.retired_fields
            if getattr(self, name) != self.__dataclass_fields__[name].default
        ]
        if changed_retired:
            warnings.warn(
                "Retired social settings do not tune rule thresholds: "
                + ", ".join(sorted(changed_retired))
                + ". Migrate to the live settings described by retired_fields; "
                "legacy values are retained for receipt compatibility only.",
                RuntimeWarning, stacklevel=2,
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SocialHeuristicConfig:
        data = dict(values)
        allowed = set(cls.__dataclass_fields__)
        unexpected = set(data) - allowed
        if unexpected:
            raise ValueError(f"unknown social heuristic settings: {sorted(unexpected)}")
        return cls(**data)


DEFAULT_SOCIAL_HEURISTIC_CONFIG = SocialHeuristicConfig()
