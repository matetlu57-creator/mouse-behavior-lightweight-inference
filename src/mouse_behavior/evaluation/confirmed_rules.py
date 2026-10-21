"""Opt-in, annotation-free rules for confirmed-data calibration v1.

Existing production policy stays separate. Distances, roles and visible temporal
support are hard gates here; ranking cannot turn zero evidence into an event.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, replace
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from .blind_top1 import Candidate
from .exported_dataset_models import ExportedFeatures, HeuristicParameters
from .requested_thresholds_v27 import ISOLATION_DISTANCE_CM
from .group_event_rules import GROUP_RULE_REVISION
from .group_heuristic_config import DEFAULT_GROUP_HEURISTIC_CONFIG
from mouse_behavior.behavior.individual_rules import DEFAULT_INDIVIDUAL_CONFIG, INDIVIDUAL_RULE_REVISION
from .social_heuristic_config import (
    DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    SOCIAL_CONFIG_REVISION,
    SocialHeuristicConfig,
)
from .screening_signals import ScreenPolicy, runs, body_motion, observed_run_motion
from .pair_roles import (
    assign_event_roles, bridge_pose_turns, new_role_buffers,
    observed_orientations, pose_direction_support,
)
from .social_event_geometry import (
    body_footprint_distances, endpoint_distances, observed_path_and_net, snout_surface_distances,
    directed_snout_body_footprint_distances,
)


CONFIRMED_POLICY = ScreenPolicy()
SOCIAL_RULE_REVISION = "2026-10-20-social-v28-closest-endpoint-and-directed-pursuit"
SOCIAL_NOSE_HEAD_CONTACT_DISTANCE_CM = DEFAULT_SOCIAL_HEURISTIC_CONFIG.dist_nose_head_contact_max_cm
SOCIAL_NOSE_TAIL_CONTACT_DISTANCE_CM = DEFAULT_SOCIAL_HEURISTIC_CONFIG.dist_nose_tail_contact_max_cm
SOCIAL_ALIGNED_HEADING_COSINE = 0.25
SOCIAL_OPPOSING_HEADING_COSINE = -0.25
# SOC-3: "walking normally" is the individual layer's own Locomotion
# boundary, not a new scale-era speed constant, so both layers stay in the
# same 0.74x scale era without either being re-derived.
SOCIAL_TAIL_FOLLOW_MIN_TARGET_SPEED_CM_S = DEFAULT_INDIVIDUAL_CONFIG.motion_speed_threshold_cm_s
GROUP_ISOLATION_DISTANCE_CM = ISOLATION_DISTANCE_CM
GROUP_HUDDLE_MIN_DURATION_S = 5.0
GROUP_CLUSTERING_MAX_DURATION_S = 5.0
GROUP_LOCOMOTION_MIN_DURATION_S = 3.0
GROUP_DISPERSAL_MIN_INCREASE_CM = 3.0
GROUP_DISPERSAL_MIN_DURATION_S = 5.0
GROUP_PROPOSAL_SET_BUDGET = 160
SOCIAL_CONTACT_ACCELERATION_TRAIN_P90_CM_S2 = {
    "nose_head_contact_clip_max": 1300.4284820709004,
    "nose_tail_contact_clip_max": 953.3508845552554,
}


def policy_receipt(
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
) -> dict:
    """Publish executable behavior-specific policy and full-cohort provenance."""
    from .retrospective_thresholds_v17 import (
        FIT_SCOPE, STATISTICS_SHA256, ROLE_STATISTICS_SHA256,
        ISOLATION_STATISTICS_SHA256, ISOLATION_EXCLUSIONS_SHA256,
        ISOLATION_REVIEWED_COMPARATOR_POLICY,
        ISOLATION_USER_DECISION_SHA256, ISOLATION_AUTHORIZATION_BINDING_SHA256,
        ACTIVE_DATASET_MANIFEST_SHA256, ACTIVE_DATASET_REVISION_RECEIPT_SHA256,
        ACTIVE_DATASET_REVISION, ACTIVE_DATASET_SAMPLE_COUNT, ROLE_MEMBERSHIP_PROOF_SHA256,
    )
    from .retrospective_thresholds_v18 import (
        FIT_SCOPE as V18_FIT_SCOPE, PROFILE_REVISION, PARAMETER_SOURCES, SOURCE_ARTIFACT_SHA256,
    )
    from .retrospective_thresholds_v19 import (
        PROCESS_PARAMETER_REVISION, PROCESS_PARAMETER_SOURCES,
        SOURCE_ARTIFACT_SHA256 as PROCESS_SOURCE_SHA256,
    )
    from .retrospective_thresholds_v20 import (
        SEMANTIC_PARAMETER_REVISION, PARAMETER_SOURCES as V20_PARAMETER_SOURCES,
        SOURCE_ARTIFACT_SHA256 as V20_SOURCE_SHA256,
    )
    rules = {
        "attack": f"light: jointly visible IDs and attacker nose/complete-head directed acceleration>={social_config.acc_attack_min_cm_s2}, center<={social_config.dist_attack_max_cm}cm; V23 footprint cue cannot create a light onset; legacy trigger>=0.15s, no legacy light onset speed floor; additional coherent trigger: same actor simultaneous nose AND complete-head measured acceleration and directed velocity meet existing floors, one-frame seed permitted, whole observed action>=attack_min_duration_s; V25 retains original finite fits and estimates only unavailable observed point derivatives within >=5-point contiguous jump-free runs; an additional center-near dual pulse grows backward through center-near support and forward through center-near OR validated body-footprint contact, within its own jointly observed point-jump-free epoch; each expanded window independently meets the action minimum before union; directed deceleration continuation<={social_config.attack_deceleration_max_s}s; heavy: same target bbox reappears after at least {social_config.attack_box_reappearance_min_cycles} bounded gaps<={social_config.attack_box_gap_max_s}s within {social_config.attack_box_repeat_window_s}s, nearest joint-near anchors on both sides within {social_config.attack_anchor_window_s}s and directed fast actor anchor; heavy proximity: center<={social_config.dist_attack_max_cm}cm OR jointly observed non-tail body footprint gap<={social_config.attack_contact_surface_max_gap_cm}cm, >=4 finite points per mouse and footprint span<={social_config.attack_contact_max_footprint_body_lengths} reference body lengths; observed separation otherwise vetoes; contact rescue requires overlap with original center-heavy in the same expanded process and adds only jointly observed rows and cannot enlarge center-only unknown support, including after priority; single-gap route retired; missing interior rows are inferred process, never measured kinematics; subtype is evidence channel, not manual severity truth",
        "standalone_contact": f"independent directed measured onset: head<={social_config.contact_head_onset_distance_cm}cm, tail<={social_config.contact_tail_onset_distance_cm}cm; baseline continuation head<={social_config.dist_nose_head_contact_max_cm}cm for>={social_config.nose_contact_min_duration_s}s, tail<={social_config.dist_nose_tail_contact_max_cm}cm for>={social_config.nose_tail_min_duration_s}s; head unknown bridge<={social_config.contact_bridge_max_gap_s}s, tail<={social_config.contact_tail_bridge_max_gap_s}s; short strictly observed nearest-endpoint channels: head<={social_config.contact_head_extension_distance_cm}cm for>={social_config.contact_head_extension_min_duration_s}s, tail<={social_config.contact_tail_extension_distance_cm}cm for>={social_config.contact_tail_extension_min_duration_s}s; each channel rechecks its own measured support after priority; qualified Together and sustained directed Chase exclude contact; head/tail lowest priorities; over15s review flag",
        "chase": "jointly visible measured actor/target motion and sustained front-back pursuit, including existing bounded0.5s observed leader-turn path; actual neck-to-nose head and tailroot-to-neck body directions support roles, never legacy velocity-fallback heading; observed incompatible facing splits except bounded interior0.5s pose-only turns with continuous pursuit motion, whose contradictory rows keep unknown roles; eligible observed support>=1s and adjacent observed path/net>=10.7cm for both; missing rows never positive frames/time/travel or roles; a qualified chase can reclaim Approach/Chasing and lower-priority contact occupancy after original six-way arbitration, preserving selected Attack/Together; lower-priority contacts cannot reserve its remainder and block the full directed process",
        "approach": "actor moves toward partner and mutual distance shrinks; stationary, lateral or receding target allowed without target-path or relative-speed gates; measured head/body facing caps use full755 retrospective p99 support envelopes; missing measured head may retain an undirected closing candidate but never obtains a role from velocity; observed reversed pose is not positive direction; start>=12cm at first measured directed anchor in bounded lag lookback; >=80percent candidate-prefix validity and duration>=0.3s with existing exact-final-frame fitted snout-surface envelope; sustained qualified Chase reclaims overlap while preserving Attack/Together and superseding lower-priority contacts",
        "together": f"center strictly<{social_config.dist_together_max_cm}cm AND each participant center speed strictly<{social_config.vel_together_max_cm_s}cm/s, symmetric independent AND; smoothed_speed={social_config.together_use_smoothed_speed}; continuous>=1s; bounded unknown gaps<=0.15s; no contact veto; Approach wins",
    }
    groups = {
        "huddling": f"at_least3 observed visual IDs; connected graph center<={DEFAULT_GROUP_HEURISTIC_CONFIG.huddle_neighbor_distance_cm}cm OR nose<=4.0125cm; compact support>=5s; quiet last observed in-pile member held<=5s without reappearance; observed departure or moving pile ends hold; raw missing rows preserved",
        "isolation": "all finite coobserved target nearest-neighbor distances strictly>20cm; observed support strictly>10s; missing unknown; no individual action gate; sub20 positive exclusions are derived evaluation artifacts only, not runtime identity merges",
        "isolation_distance_cm": GROUP_ISOLATION_DISTANCE_CM,
        "social_clustering": "observed joining or mutual convergence precedes confirmed dense state; adjacent real center path strictly>10cm, measured path/time>=2cm/s, observed vector-sum net>=1cm, adjacent measured distance-drop>=1cm, relative direction cosine>=0.5 and directed support>=0.15s; convergence lookback and toward fraction from group config; final connected all-observed shared-center radius<=10cm requires a contiguous measured run>=0.5s; bounded unknown-only memory retains confirmed state with minimum observed fraction, never combines shorter runs to create stability; output actual all-member-observed formation prefix and dense rows only; no recipient speed cap, cross-gap jumps or missing-frame credit; retained legacy joining channel remains separately qualified; support intervals publish observed directed motion",
        "group_locomotion": f"at_least3 IDs; each smoothed speed>={DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_active_speed_cm_s}cm/s; synchronized support>={DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_simultaneous_min_s}s; all pair distances<{DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_max_pair_distance_cm}cm; velocity and net coherence>={DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_direction_similarity}; each net displacement>={DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_min_net_displacement_cm}cm; process span>3s",
        "dispersal": "initial observed connected pair anchors fixed origin; >=3 real IDs individually first observed in that origin by confirmation, progressive appearance allowed; departing subset>=2,>=60percent visible since own entry, net/radial departure>=4cm and outside observed origin hull; checkpoint strictly after5s; each departed mouse may have at most one departed neighbor<3cm (multiple disjoint pairs allowed, exactly3cm not close); actual process strictly>5s; stop on ended motion or observed return; retain distinct independently qualified checkpoint member/process sets, never union incompatible episodes; legacy mask API retains first event",
        "dispersal_distance_statistic": "maximum_close_neighbor_degree_le1_at_observed_checkpoint_strictly_after5s",
        "proposal_member_set_budget": GROUP_PROPOSAL_SET_BUDGET,
    }
    receipt = {
        "version": "confirmed_rules_v28",
        "requested_policy_revision": "2026-10-20-v28-user-overrides-v26-restorations-and-contact-onsets",
        "chasing_mean_speed_contract": "each participant real adjacent eligible path divided by adjacent-step time >= chasing_min_mean_speed_cm_s; no cross-gap jumps, missing time or imputed frames; 15cm/s explicit request, not calibrated optimum",
        "contact_onset_contract": "independent head-triangle and tail-root first-measured onset gates; per-class first-finite annotated-frame p90 from159 head/75 tail samples; onset must survive higher-priority arbitration; no retrospective painting before onset; full-data retrospective fit, no independent validation",
        "contact_onset_statistics_sha256": "c99e75282843b88844630044e96a98bd29c1d319a47c518190a20ec3b6ed00f6",
        "contact_subtype_competition_contract": "baseline and short extensions compare each measured primary endpoint with both competing subtype directions of the same pair; only the closer subtype survives, unknown competitor supplies no veto, equal distances defer to configured head/tail priority; no endpoint or frame is inferred; selected over broad-head temporal tightening in full234 retrospective ablation, not independent validation",
        "directed_role_revision": "2026-10-19-v26-observed-directed-pair-roles",
        "directed_role_fit_scope": "all755_retrospective_parameter_selection_not_independent_validation",
        "directed_role_artifacts_sha256": {
            "outputs/directed_roles_v26_20261019/geometry_statistics.json": "b0424def1727a77786d77878ac79dbc98ce69473833c193e0ce7bbd4c609b1f6",
            "outputs/directed_roles_v26_20261019/parameter_selection.json": "b6b8f4433153b638b7b7d614b4bdadd1990d0f4e015b14800aee0c7a1dce5f11",
        },
        "directed_role_contract": "optional roles dict contains approach_actor/approach_target/chasing_actor/chasing_target int64 arrays shaped frames,pairs; values are actual feature.track_ids visual IDs, never column slots; -1 means missing, ambiguous or nonbehavior; role scoring is separate from unordered A1/A2 target-ID set metrics; prediction never reads annotation or this retrospective audit",
        "observed_run_parameter_revision": "2026-10-19-v25-observed-run-fit-and-center-pulse-contact",
        "observed_run_fit_support_contract": "retain every finite original point derivative; estimate unavailable derivatives only at real observed point rows using at least5 contiguous actual points; missing rows and consecutive jumps>120cm/s split estimation windows; no raw position, valid flag or ID is modified",
        "observed_run_action_support_contract": "retain exact V24 coherent cue; additional same-actor center-near dual-point pulse needs its own >=0.15s action within one joint-observed, point-jump-free epoch; qualify each seed-expanded window before actor/epoch union; backward remains center-near, forward permits validated measured footprint proximity; no footprint onset and no new unknown support",
        "observed_run_contact_unknown_guard": "after new attack arbitration, each contact subtype may retain unknown rows only from the exact V24 canonical public path computed with observed attack extension disabled; no head-to-tail transfer; removing unknown rows requalifies each fragment at its subtype minimum without reallocating deleted support",
        "new_action_target_confirmation": "each additional actor-specific seed-expanded action window must contain at least one jointly observed actual actor-nose to assigned-target validated non-tail footprint distance within attack_new_action_target_surface_max_gap_cm; target hull uses points0..5, at least4 finite points, span<=2 reference body lengths, nondegenerate; never borrow confirmation across windows, actors or unknown rows; retained V24 branch unchanged",
        "new_action_target_surface_cap_cm": social_config.attack_new_action_target_surface_max_gap_cm,
        "new_action_target_surface_fit_scope": "maximum of five independently qualified actor-window minimum nose-to-target-non-tail-hull gaps in three video-reviewed recovered annotated pairs; selected-case retrospective fit, not full93 extremum or generic contact truth",
        "v25_fit_artifact_sha256": {
            "outputs/heuristics_v25_attack_misses_20261018/fit_decision_target_confirmed.json": "a280985ea95e5f3ff7de1399ac50bc388f03a9c7b0823670bccdba65f5674af5",
        },
        "attack_acceptance_default_coverage": 0.30,
        "other_behavior_acceptance_default_coverage": 0.50,
        "acceptance_is_detection_rule": False,
        "coherent_attack_parameter_revision": "2026-10-18-v24-coherent-attack-observed-subtype-contact",
        "coherent_attack_support_contract": "same actor nose AND complete head simultaneously meet existing acceleration and directed fast-velocity gates; a seed grows only through jointly observed, fitted directed motion including deceleration; whole action meets existing minimum duration; no new unknown positive rows; established v23 light/heavy routes retained",
        "contact_short_bout_support_contract": "same-newAttack canonical contact support OR subtype-specific qualified strict observed support; compare head/tail within each actor direction when alternate is finite; no new gap bridging; requalify after priority using independent subtype minima; legacy unknown support cannot enlarge",
        "v24_fit_artifact_sha256": {
            "outputs/heuristics_v24_attack_contacts_20261018/fit_decision.json": "8077c1e0bc76d776fd17f04c0c4b1b43c86780aa93ec874ada7308f8f27b7753",
        },
        "observed_contact_parameter_revision": "2026-10-18-v23-eight-loss-case-video-review",
        "observed_contact_parameter_sources": {
            "attack_contact_surface_max_gap_cm": "maximum non-tail body-footprint gap across five video-reviewed contact frames: 2b00345a15169ac2f3b009cb frames18/19/23/52 and 89226228d39f3bb902edddfa frame12; selected eight-loss cohort retrospective fit, not whole-93 extrema or manual frame truth",
            "attack_contact_max_footprint_body_lengths": "2 reference body lengths; conservative keypoint-span validity guard, not fitted optimal threshold",
        },
        "observed_contact_fit_artifact_sha256": {
            "outputs/heuristics_v23_observed_contact_20261018/observed_contact_fit.json": "1e23550454cd76f21ad983d3a550fe776c03a88845252897677656efc046fa8e",
        },
        "observed_contact_legacy_scope": "observed_contact_* describes retained V23 heavy rescue; the additional V25 coherent center-pulse continuation has its separate observed_run_action_support_contract",
        "observed_contact_rescue_channel": "heavy_repeated_target_boxes_only for V23 rescue; light onset remains center-near; V25 coherent forward continuation may use validated measured body contact",
        "observed_contact_channel_review_sha256": {"outputs/heuristics_v23_observed_contact_20261018/contact_channel_review.json": "14878e96549a1623e1625ed2656b8c8ec0460a4e27ee55d8c6666bf427c63424"},
        "observed_contact_extension_requires_center_heavy_overlap": True,
        "observed_contact_extension_review_sha256": {"outputs/heuristics_v23_observed_contact_20261018/contact_extension_scope_review.json": "63956d8a984a5897abcdd34323add967976bb87aaddea6aca135615da6c8ee8b"},
        "observed_contact_support_contract": "retained V23 rescue: body geometry only extends an expanded-heavy contiguous process overlapping original center-only heavy raw support; standalone new heavy events forbidden; original directed-action or repeated-same-target bbox-cycle requirements still apply; additions require both tracks valid; center-only unknown support cannot enlarge before or after priority",
        "boundary_parameter_revision": "2026-10-18-v22-video-audited-loss-cohort",
        "boundary_parameter_sources": {
            "attack_box_gap_max_s": "28 missing target291 bbox frames / 29.967457227456368fps in 977efbcaa730be97b4fc0dd1; maximum in twelve selected loss-case traces; retrospective, not a whole-93 extrema or independent validation",
        },
        "boundary_fit_artifact_sha256": {
            "outputs/heuristics_v22_video_boundary_20261018/attack_gap_fit.json": "d13345cb042d380e31d88c0054e147a4c8fe5d3fe0d6612c1a5f69242d048c2d",
        },
        "fit_scope": "full_dataset_retrospective_with_user_requested_v28_overrides", "independent_validation": False,
        "clustering_fit_scope": "v29_all8_retrospective_process_fit_not_independent_validation",
        "legacy_v18_fit_scope": V18_FIT_SCOPE,
        "legacy_v17_fit_scope": FIT_SCOPE,
        "retrospective_profile_revision": PROFILE_REVISION,
        "semantic_parameter_revision": SEMANTIC_PARAMETER_REVISION,
        "semantic_parameter_sources": V20_PARAMETER_SOURCES,
        "semantic_fit_artifact_sha256": V20_SOURCE_SHA256,
        "retrospective_process_parameter_revision": PROCESS_PARAMETER_REVISION,
        "process_parameter_sources": PROCESS_PARAMETER_SOURCES,
        "process_fit_artifact_sha256": PROCESS_SOURCE_SHA256,
        "behavior_specific_parameter_sources": PARAMETER_SOURCES,
        "behavior_specific_fit_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "active_dataset_revision": ACTIVE_DATASET_REVISION,
        "active_dataset_sample_count": ACTIVE_DATASET_SAMPLE_COUNT,
        "active_dataset_manifest_sha256": ACTIVE_DATASET_MANIFEST_SHA256,
        "active_dataset_revision_receipt_sha256": ACTIVE_DATASET_REVISION_RECEIPT_SHA256,
        "role_membership_proof_sha256": ROLE_MEMBERSHIP_PROOF_SHA256,
        "statistics_sha256": STATISTICS_SHA256,
        "role_statistics_sha256": ROLE_STATISTICS_SHA256,
        "isolation_statistics_sha256": ISOLATION_STATISTICS_SHA256,
        "isolation_exclusions_sha256": ISOLATION_EXCLUSIONS_SHA256,
        "isolation_user_decision_sha256": ISOLATION_USER_DECISION_SHA256,
        "isolation_authorization_binding_sha256": ISOLATION_AUTHORIZATION_BINDING_SHA256,
        "isolation_threshold_fit_policy": ISOLATION_REVIEWED_COMPARATOR_POLICY,
        "runtime_comparators": "current visible visual IDs; reviewed per-sample exclusions clean fitting statistics only, not a global runtime ID merge",

        "reference_body_cm": 7.918,
        "social_rule_revision": SOCIAL_RULE_REVISION,
        "social_config_revision": SOCIAL_CONFIG_REVISION,
        "group_rule_revision": GROUP_RULE_REVISION,
        "individual_rule_revision": INDIVIDUAL_RULE_REVISION,
        "individual_heuristic_config": asdict(DEFAULT_INDIVIDUAL_CONFIG),
        "group_heuristic_config": DEFAULT_GROUP_HEURISTIC_CONFIG.as_dict(),
        "social_heuristic_config": social_config.as_dict(),
        "screen_policy": asdict(CONFIRMED_POLICY),
        "exclusive_pair_priority": list(social_config.priority_order),
        "exclusive_pair_priority_scope": "overlapping frames of qualified events; disjoint temporal events survive; fragments requalified after priority subtraction",
        "group_event_frame_contract": "clip-local half-open event intervals; group-motion simultaneous support exported separately; legacy ranking evidence remains seconds",
        "rules": rules, "group_rules": groups,
        "retired_social_config_fields": {
            name: {"value": getattr(social_config, name), "status": status}
            for name, status in social_config.retired_fields.items()
        },
        "retired_group_config_fields": {
            name: {"value": getattr(DEFAULT_GROUP_HEURISTIC_CONFIG, name),
                   "status": "compatibility_only_v21_final_spatial_state"}
            for name in DEFAULT_GROUP_HEURISTIC_CONFIG.retired_fields
        },
        "attack_min_head_or_nose_acceleration_cm_s2": social_config.acc_attack_min_cm_s2,
        "contact_distance_below_approach_endpoint_cm": {
            "strict_nose_head_head_triangle": social_config.dist_nose_head_contact_max_cm,
            "strict_nose_tail_nose_to_tailbase": social_config.dist_nose_tail_contact_max_cm,
            "approach_endpoint_minimum": social_config.dist_approach_end_min_cm,
            "ordering_constraint": "retired; independently fitted ranges may overlap",
        },
        "group_members": "real_visual_ids_only; huddle bounded holds; dispersal only actual departed subset",
        "grooming_rearing": "audited_candidates_disabled_canonically_until_manual_review_and_calibration; offline_screen_is_not_canonical_prediction",
        "priority_changed": True,
    }
    receipt.update({name: rules[name] for name in rules})
    return receipt


def confirmed_profiles(seed: HeuristicParameters) -> tuple[HeuristicParameters, ...]:
    """One current fixed profile; historical motion threshold variants retired."""
    base = replace(
        seed,
        confirmed_rules_v1=1.0,
        group_core_candidate_mode=1.0,
        isolation_distance_cm=GROUP_ISOLATION_DISTANCE_CM,
        isolation_min_duration_s=10.0,
        individual_motion_speed_threshold_cm_s=DEFAULT_INDIVIDUAL_CONFIG.motion_speed_threshold_cm_s,
        group_locomotion_max_distance_cm=DEFAULT_GROUP_HEURISTIC_CONFIG.locomotion_max_pair_distance_cm,
        huddle_min_duration_s=GROUP_HUDDLE_MIN_DURATION_S,
        group_locomotion_min_duration_s=GROUP_LOCOMOTION_MIN_DURATION_S,
        dispersal_min_distance_increase_cm=GROUP_DISPERSAL_MIN_INCREASE_CM,
        dispersal_min_duration_s=GROUP_DISPERSAL_MIN_DURATION_S,
    )
    return (base,)


def social_profiles(
    seed: HeuristicParameters,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
) -> tuple[HeuristicParameters, ...]:
    """Use one social rule profile; unrelated group-core variants cannot compete."""
    return (replace(
        seed,
        confirmed_rules_v1=1.0,
        group_core_candidate_mode=0.0,
        pair_min_duration_s=social_config.approach_min_duration_s,
        approach_min_start_distance_cm=social_config.dist_approach_start_min_cm,
        approach_min_end_distance_cm=social_config.dist_approach_end_min_cm,
        approach_terminal_distance_cm=social_config.dist_approach_end_min_cm,
        approach_min_actor_speed_cm_s=social_config.vel_approach_actor_min_cm_s,
        approach_min_closing_speed_cm_s=social_config.approach_min_closing_speed_cm_s,
        contact_distance_cm=max(
            social_config.dist_nose_head_contact_max_cm,
            social_config.dist_nose_tail_contact_max_cm,
        ),
        contact_min_cumulative_seconds=social_config.nose_contact_min_duration_s,
        together_min_distance_cm=0.0,
        together_max_distance_cm=social_config.dist_together_max_cm,
        together_min_duration_s=social_config.together_min_duration_s,
        chase_max_distance_cm=social_config.dist_chasing_max_cm,
        chase_min_chaser_speed_cm_s=social_config.vel_chasing_min_cm_s,
        chase_min_chaser_facing_cosine=float(np.cos(np.deg2rad(social_config.angle_chasing_dir_max_deg))),
        chase_min_chaser_toward_cosine=float(np.cos(np.deg2rad(social_config.angle_chasing_behind_max_deg))),
        chase_min_net_displacement_cm=social_config.chasing_min_net_displacement_cm,
        chase_min_directed_support_s=social_config.chasing_min_duration_s,
        attack_min_nose_acceleration_toward_cm_s2=social_config.acc_attack_min_cm_s2,
        attack_endpoint_distance_cm=social_config.dist_attack_max_cm,
        attack_min_event_duration_s=social_config.attack_min_duration_s,
    ),)


def _toward_cosine(vector: np.ndarray, offset: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, float)
    offset = np.asarray(offset, float)
    denominator = np.linalg.norm(vector, axis=1) * np.linalg.norm(offset, axis=1)
    cosine = np.divide(
        np.sum(vector * offset, axis=1), denominator,
        out=np.full(len(vector), -np.inf), where=denominator > 1e-9,
    )
    return np.clip(cosine, -1.0, 1.0)


def _single_bout_mask(mask: np.ndarray, fps: float, minimum_seconds: float) -> np.ndarray:
    """Keep only individual continuous bouts meeting the event minimum."""
    values = np.asarray(mask, bool)
    result = np.zeros_like(values)
    minimum = max(1, int(np.ceil(float(minimum_seconds) * fps - 1e-9)))
    for start, end in runs(values):
        if end - start >= minimum:
            result[start:end] = True
    return result


def _bridge_internal_gaps(mask: np.ndarray, support: np.ndarray, fps: float,
                          *, maximum_gap_s: float,
                          unknown: np.ndarray | None = None) -> np.ndarray:
    """Bridge bounded interior gaps between two runs of an evidence mask.

    Two 2026-10-16-social-v16.2 corrections:

    * SOC-5 - ``unknown`` marks frames whose signal was NOT measured. Only those
      frames may be bridged. A frame that was measured and found outside the
      contact distance is positive evidence of NON-contact and must split the
      bout; bridging it fabricates event duration out of frames that were
      measured as separated, and the fabricated frames then count toward the
      minimum duration through ``_single_bout_mask``.
    * SOC-4 - ``support`` is evaluated at the two gap ANCHORS, not on every
      interior frame. The interior is exactly where the measurement is missing,
      so requiring support there makes the bridge impossible for the dropouts it
      exists for (and made any detection dropout, ``pair_distance = inf``,
      unbridgeable).

    ``unknown=None`` treats every interior frame as unmeasured, which reproduces
    the pre-v16.2 behaviour for a caller with no measurement-provenance signal.
    """
    values = np.asarray(mask, dtype=bool).copy()
    maintained = np.asarray(support, dtype=bool)
    if values.ndim != 1 or maintained.shape != values.shape:
        raise ValueError("gap masks must be aligned one-dimensional arrays")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(maximum_gap_s) or maximum_gap_s < 0:
        raise ValueError("fps must be positive and maximum_gap_s non-negative")
    limit = int(np.floor(maximum_gap_s * fps + 1e-9))
    unmeasured = None if unknown is None else np.asarray(unknown, dtype=bool)
    if unmeasured is not None and unmeasured.shape != values.shape:
        raise ValueError("unknown mask must be aligned with the gap mask")
    bounds = list(runs(values))
    for (_, left_end), (right_start, _) in zip(bounds, bounds[1:]):
        if not 0 < right_start - left_end <= limit:
            continue
        if unmeasured is not None and not unmeasured[left_end:right_start].all():
            continue
        # left_end >= 1 and right_start <= len-1 for every interior gap, because
        # both bounding runs contain at least one frame.
        if not (maintained[left_end - 1] and maintained[right_start]):
            continue
        values[left_end:right_start] = True
    return values


def _contact_bridge_support(
    center: np.ndarray, body_cm: float, social_config: SocialHeuristicConfig,
) -> np.ndarray:
    """SOC-4: dedicated contact-bridge support radius, in reference body lengths.

    Deliberately NOT ``dist_together_max_cm`` (6.0 cm): nose-to-tail-root contact
    puts the two body centres around 7-8 cm apart, so reusing the Together
    threshold made tail-contact flicker permanently unbridgeable. A multiple of
    ``reference_body_cm`` is invariant to the 0.74x scale migration, so no new
    absolute cm literal is introduced.
    """
    limit = social_config.contact_bridge_support_max_body_lengths * float(body_cm)
    finite = np.isfinite(center)
    return finite & (np.asarray(center, dtype=float) <= limit)


def _contact_signal_unknown(
    measured: np.ndarray, center: np.ndarray, body_cm: float, *, max_body_lengths: float,
) -> np.ndarray:
    """SOC-5: True where an endpoint distance is NOT a credible non-contact reading.

    A non-finite endpoint distance is unmeasured. A finite one is also treated as
    unmeasured when it is geometrically impossible for the observed pair: an
    actor's snout cannot sit further from a target body part than the two centres
    are apart plus one body scale. Such a value is an occlusion artefact (a tail
    root lost behind a body, or an exporter sentinel), not proof of separation,
    so it may be bridged. A finite value inside that ceiling WAS measured and
    must split the bout.
    """
    measured = np.asarray(measured, dtype=float)
    finite = np.isfinite(measured)
    ceiling = np.asarray(center, dtype=float) + max_body_lengths * float(body_cm)
    impossible = finite & np.isfinite(ceiling) & (measured > ceiling)
    return (~finite) | impossible


def _measured_contact_bout_mask(mask, directed_support, fps, minimum_s, directed_onsets=None):
    """Qualify each retained fragment with one real directed endpoint signal.

    Unknown continuity supplies neither the minimum contact time nor an orphan
    start/end anchor after higher-priority behaviors remove measured rows.
    """
    result = np.zeros_like(mask, dtype=bool)
    for low, high in runs(np.asarray(mask, bool)):
        for direction, support in enumerate(directed_support):
            onset = low
            if directed_onsets is not None:
                anchors = np.flatnonzero(np.asarray(directed_onsets[direction], bool)[low:high]
                                         & np.asarray(support, bool)[low:high])
                if not len(anchors):
                    continue
                onset = low + int(anchors[0])
            measured = np.flatnonzero(np.asarray(support, bool)[onset:high])
            if len(measured) / fps + 1e-12 >= minimum_s:
                start, end = onset + int(measured[0]), onset + int(measured[-1]) + 1
                result[start:end] = True
    return result


def _closest_contact_endpoint(primary, alternate_directions):
    """Compare actual endpoint proximity across both directions of this pair.

    The opposite mouse's short head reading must not steal a closer measured
    tail contact merely because head has the earlier tie-breaking priority.
    Unknown competing geometry supplies no veto; ties retain configured order.
    """
    alternate = np.fmin(*alternate_directions)
    return ~np.isfinite(alternate) | (primary <= alternate)


def _bridged_contact_bout_mask(
    *, strict: np.ndarray, measured: np.ndarray, center: np.ndarray, body_cm: float,
    fps: float, social_config: SocialHeuristicConfig, kind: str = "head",
) -> tuple[np.ndarray, np.ndarray]:
    """Bridge bounded keypoint flicker inside one directed contact signal.

    Spec module 3 item 3 names this operator. Bridging crosses only genuinely
    UNMEASURED interior frames of at most ``contact_bridge_max_gap_s`` whose two
    anchors sit inside the dedicated body-length support radius.

    Returns ``(qualified, evidence)``:

    * ``qualified`` meets ``nose_contact_min_duration_s`` and is what the contact
      labels emit.
    * ``evidence`` meets the much shorter
      ``together_contact_veto_min_duration_s`` for legacy diagnostic callers.
      v17 does not use it as a Together veto.
    """
    bridged = _bridge_internal_gaps(
        strict, _contact_bridge_support(center, body_cm, social_config), fps,
        maximum_gap_s=(social_config.contact_bridge_max_gap_s if kind == "head"
                       else social_config.contact_tail_bridge_max_gap_s),
        unknown=_contact_signal_unknown(
            measured, center, body_cm,
            max_body_lengths=social_config.contact_signal_max_body_lengths,
        ),
    )
    qualified = _measured_contact_bout_mask(
        bridged, (strict,), fps, social_config.contact_minimum_s(kind),
        (strict & (measured <= social_config.contact_onset_cm(kind)),))
    return (
        qualified,
        _single_bout_mask(bridged, fps, social_config.together_contact_veto_min_duration_s),
    )


def _qualify_contact_channels(mask, joint, endpoints, kind, fps, config, *, extension=True):
    """Each channel must supply its own distance, onset and measured duration."""
    values = endpoints[kind]
    cap = config.dist_nose_head_contact_max_cm if kind == 'head' else config.dist_nose_tail_contact_max_cm
    onsets = tuple(joint & np.isfinite(v) & (v <= config.contact_onset_cm(kind)) for v in values)
    output = _measured_contact_bout_mask(
        mask, tuple(joint & np.isfinite(v) & (v <= cap) for v in values),
        fps, config.contact_minimum_s(kind), onsets)
    if extension:
        distance = config.contact_head_extension_distance_cm if kind == 'head' else config.contact_tail_extension_distance_cm
        minimum = config.contact_head_extension_min_duration_s if kind == 'head' else config.contact_tail_extension_min_duration_s
        alternate = endpoints['tail' if kind == 'head' else 'head']
        supports = tuple(joint & np.isfinite(v) & (v <= distance)
                         & (~np.isfinite(other) | (v <= other))
                         for v, other in zip(values, alternate, strict=True))
        # Short extensions consist only of observed endpoint support. A broad
        # baseline fragment must not borrow their lower duration threshold.
        for support, onset in zip(supports, onsets, strict=True):
            output |= _measured_contact_bout_mask(mask & support, (support,), fps, minimum, (onset,))
    return output


@lru_cache(maxsize=8)
def _warn_social_mirror_drift(scope: str, drifted: tuple[str, ...]) -> None:
    """Emit one RuntimeWarning per distinct drifted field set (SOC-12)."""
    warnings.warn(
        "pair_rule_masks() received a HeuristicParameters whose social mirror "
        "fields disagree with social_config for the " + scope + " rules: "
        + ", ".join(drifted) + ". The social masks are driven by social_config "
        "alone; tune SocialHeuristicConfig. HeuristicParameters is read for "
        "provenance only and is never a threshold source.",
        RuntimeWarning,
        stacklevel=4,
    )


def _report_social_mirror_drift(
    scope: str, entries: Iterable[tuple[str, float, float]],
) -> None:
    """SOC-12: report a stale HeuristicParameters mirror instead of ignoring it.

    ``social_profiles()`` copies the social thresholds into HeuristicParameters
    so the ranking layer sees one policy. ``pair_rule_masks`` used to accept that
    object and never read it, which misled callers and calibrators into believing
    HeuristicParameters drives these rules. It is now read and compared; the
    comparison never gates a mask, because scripts/score_segment_level_v16.py
    builds its parameters from configs/default.yaml (which does not set
    confirmed_rules_v1) and must keep receiving the social masks.
    """
    drifted = tuple(
        name for name, actual, expected in entries
        if abs(float(actual) - float(expected)) > 1e-9
    )
    if drifted:
        _warn_social_mirror_drift(scope, drifted)


def _fill_short_nan_gaps(values: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only bounded interior geometry gaps and retain its provenance mask."""
    result = np.asarray(values, dtype=float).copy()
    known = np.isfinite(result)
    if result.ndim != 1:
        raise ValueError("geometry values must be one-dimensional")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    for start, end in runs(~known):
        if (start == 0 or end == len(result) or end - start > max_gap_frames
                or not known[start - 1] or not known[end]):
            continue
        result[start:end] = np.linspace(result[start - 1], result[end], end - start + 2)[1:-1]
    return result, known


def _local_quadratic_point_kinematics(
    positions_cm: np.ndarray,
    valid: np.ndarray,
    fps: float,
    *,
    half_window_s: float,
    max_jump_speed_cm_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit observed point velocity/acceleration; reject gaps and tracker spikes."""
    points = np.asarray(positions_cm, dtype=float)
    observed = np.asarray(valid, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or observed.shape != (len(points),):
        raise ValueError("center positions and observation mask have incompatible shapes")
    speed = np.full_like(points, np.nan)
    acceleration = np.full_like(points, np.nan)
    if len(points) < 5 or not np.isfinite(fps) or fps <= 0:
        return speed, acceleration
    radius = max(2, int(np.ceil(half_window_s * fps)))
    times = np.arange(len(points), dtype=float) / float(fps)
    for index in range(2, len(points) - 2):
        left, right = max(0, index - radius), min(len(points), index + radius + 1)
        indices = np.arange(left, right)
        if len(indices) < 5 or not observed[indices].all():
            continue
        sample = points[indices]
        if not np.isfinite(sample).all():
            continue
        jumps = np.linalg.norm(np.diff(sample, axis=0), axis=1) * float(fps)
        if jumps.size and float(np.max(jumps)) > max_jump_speed_cm_s:
            continue
        local_time = times[indices] - times[index]
        design = np.column_stack((np.ones(len(indices)), local_time, local_time**2))
        coefficients, *_ = np.linalg.lstsq(design, sample, rcond=None)
        speed[index] = coefficients[1]
        acceleration[index] = 2.0 * coefficients[2]
    return speed, acceleration


def _observed_run_point_kinematics(
    positions_cm: np.ndarray, valid: np.ndarray, fps: float, *,
    half_window_s: float, max_jump_speed_cm_s: float,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate unavailable derivatives only where five real run points exist.

    Missing positions remain missing. A gap or excessive consecutive-point jump
    separates estimation windows; existing finite fits are preserved verbatim.
    One-sided windows at a run boundary estimate observed rows, not new rows.
    """
    points = np.asarray(positions_cm, dtype=float)
    observed = np.asarray(valid, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or observed.shape != (len(points),):
        raise ValueError("point positions and observation mask have incompatible shapes")
    if baseline is None:
        baseline = _local_quadratic_point_kinematics(
            points, observed, fps, half_window_s=half_window_s,
            max_jump_speed_cm_s=max_jump_speed_cm_s)
    velocity, acceleration = (np.array(value, copy=True) for value in baseline)
    if velocity.shape != points.shape or acceleration.shape != points.shape:
        raise ValueError("baseline derivative shapes must match point positions")
    if len(points) < 5 or not np.isfinite(fps) or fps <= 0:
        return velocity, acceleration
    observed = observed & np.isfinite(points).all(axis=1)
    radius = max(2, int(np.ceil(half_window_s * fps)))
    cuts = np.zeros(len(points), bool)
    cuts[1:] = (observed[1:] & observed[:-1]
                & (np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
                   > max_jump_speed_cm_s))
    for start, end in runs(observed):
        edges = [start, *(np.flatnonzero(cuts[start:end]) + start).tolist(), end]
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            if high - low < 5:
                continue
            for index in range(low, high):
                if np.isfinite(velocity[index]).all() and np.isfinite(acceleration[index]).all():
                    continue
                left, right = max(low, index - radius), min(high, index + radius + 1)
                if right - left < 5:
                    left = max(low, min(left, high - 5))
                    right = min(high, max(right, low + 5))
                sample = points[left:right]
                local_time = (np.arange(left, right) - index) / fps
                design = np.column_stack((np.ones(len(sample)), local_time, local_time**2))
                coefficients, *_ = np.linalg.lstsq(design, sample, rcond=None)
                velocity[index] = coefficients[1]
                acceleration[index] = 2.0 * coefficients[2]
    return velocity, acceleration


def _coherent_epoch_action(
    seed, active, joint, actor_valid, nose_positions, head_positions, center_near,
    fps, social_config, *, confirmation,
) -> np.ndarray:
    """Every new action needs its own seed within one observed, jump-free epoch."""
    observed = (joint & actor_valid & np.isfinite(nose_positions).all(axis=1)
                & np.isfinite(head_positions).all(axis=1))
    confirmation = np.asarray(confirmation, bool)
    if confirmation.shape != np.shape(seed):
        raise ValueError("target confirmation must align with action frames")
    confirmation = confirmation & observed
    if (seed & (~observed | ~center_near | ~active)).any():
        raise ValueError("coherent seed must be observed, center-near, and active")
    cuts = np.zeros(len(seed), bool)
    for points in (nose_positions, head_positions):
        known = actor_valid & np.isfinite(points).all(axis=1)
        cuts[1:] |= (known[1:] & known[:-1]
                     & (np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
                        > social_config.attack_max_track_jump_speed_cm_s))
    minimum = max(1, int(np.ceil(social_config.attack_min_duration_s * fps - 1e-9)))
    bound = max(1, int(np.floor(social_config.attack_deceleration_max_s * fps + 1e-9)))
    result = np.zeros(len(seed), bool)
    for start_epoch, end_epoch in runs(observed):
        edges = [start_epoch, *(np.flatnonzero(cuts[start_epoch:end_epoch])
                                + start_epoch).tolist(), end_epoch]
        for low_epoch, high_epoch in zip(edges[:-1], edges[1:], strict=True):
            for local_start, local_end in runs(seed[low_epoch:high_epoch]):
                start, end = local_start + low_epoch, local_end + low_epoch
                low, high = start, end
                while (low > low_epoch and start - low < bound
                       and active[low - 1] and center_near[low - 1]):
                    low -= 1
                while high < high_epoch and high - end < bound and active[high]:
                    high += 1
                if high - low >= minimum and confirmation[low:high].any():
                    result[low:high] = True
    return result


def _coherent_observed_attack_action(
    f, edge, social_config, *, point_kinematics=None,
    allow_observed_run_extension=True, continuation_proximity=None,
):
    """Return retained V24 or epoch-qualified action with a center-near onset."""
    result = np.zeros(len(f.valid), dtype=bool)
    points = f.collective_keypoints_cm
    if f.centers_cm is None or points is None:
        return result
    left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
    joint = f.valid[:, left] & f.valid[:, right]
    distance = f.pair_distance_cm[:, edge]
    near = np.isfinite(distance) & (distance <= social_config.dist_attack_max_cm)
    continuation_near = near
    if continuation_proximity is not None:
        continuation_near = np.asarray(continuation_proximity, bool)
        if continuation_near.shape != joint.shape or (continuation_near & ~joint).any():
            raise ValueError("coherent continuation proximity must contain only observed pair rows")
    tail = endpoint_distances(f, edge, "tail")
    cache = point_kinematics if point_kinematics is not None else {}
    continuation = max(1, int(np.floor(social_config.attack_deceleration_max_s * f.fps + 1e-9)))
    for direction, (actor, target) in enumerate(((left, right), (right, left))):
        follow = np.zeros(len(joint), dtype=bool)
        if f.heading_unit is not None:
            same = (_toward_cosine(f.heading_unit[:, actor], f.heading_unit[:, target])
                    >= np.cos(np.deg2rad(social_config.attack_tail_follow_max_heading_angle_deg)))
            follow = _single_bout_mask(
                joint & np.isfinite(tail[direction])
                & (tail[direction] <= social_config.dist_nose_tail_contact_max_cm)
                & (f.speed_cm_s[:, target] > SOCIAL_TAIL_FOLLOW_MIN_TARGET_SPEED_CM_S)
                & same, f.fps, social_config.attack_tail_follow_min_duration_s,
            )
        head = np.asarray(points[:, actor, :3], dtype=float)
        complete = np.isfinite(head).all(axis=(1, 2))
        center = np.where(complete[:, None], head.mean(axis=1), np.nan)
        seeds, actions, positions_by_kind = [], [], []
        for kind, positions in enumerate((head[:, 0], center)):
            positions_by_kind.append(positions)
            key = (actor, kind, social_config.attack_acceleration_fit_half_window_s,
                   social_config.attack_max_track_jump_speed_cm_s)
            if key not in cache:
                cache[key] = _local_quadratic_point_kinematics(
                    positions, f.valid[:, actor], f.fps,
                    half_window_s=social_config.attack_acceleration_fit_half_window_s,
                    max_jump_speed_cm_s=social_config.attack_max_track_jump_speed_cm_s,
                )
            if allow_observed_run_extension:
                observed_key = ("observed_run", *key)
                if observed_key not in cache:
                    cache[observed_key] = _observed_run_point_kinematics(
                        positions, f.valid[:, actor], f.fps,
                        half_window_s=social_config.attack_acceleration_fit_half_window_s,
                        max_jump_speed_cm_s=social_config.attack_max_track_jump_speed_cm_s,
                        baseline=cache[key],
                    )
                velocity, acceleration = cache[observed_key]
            else:
                velocity, acceleration = cache[key]
            offset = f.centers_cm[:, target] - positions
            finite = np.isfinite(velocity).all(axis=1) & np.isfinite(acceleration).all(axis=1)
            speed = np.linalg.norm(velocity, axis=1)
            directed_velocity = _toward_cosine(velocity, offset) > 0
            directed_acceleration = _toward_cosine(acceleration, offset) > 0
            common = joint & near & finite & ~follow
            seeds.append(common & directed_velocity & directed_acceleration
                         & (speed >= social_config.vel_attack_min_cm_s)
                         & (np.linalg.norm(acceleration, axis=1) >= social_config.acc_attack_min_cm_s2))
            slowing = np.sum(velocity * acceleration, axis=1) < 0
            action_common = (joint & continuation_near & finite & ~follow
                             if allow_observed_run_extension else common)
            actions.append(action_common & directed_velocity & (speed > 1e-9)
                           & (slowing | (speed >= social_config.vel_attack_min_cm_s)))
        # Two independently fitted points agree at onset and throughout the
        # added measured action. A nose swing alone cannot trigger this cue.
        seed = seeds[0] & seeds[1]
        active = actions[0] & actions[1]
        if allow_observed_run_extension:
            if not (seed & active).any():
                continue
            target_gap = directed_snout_body_footprint_distances(
                f, edge, actor=actor,
                max_body_lengths=social_config.attack_contact_max_footprint_body_lengths,
                frame_mask=active)
            confirmation = (np.isfinite(target_gap)
                            & (target_gap <= social_config.attack_new_action_target_surface_max_gap_cm))
            result |= _coherent_epoch_action(
                seed & active, active, joint, f.valid[:, actor],
                positions_by_kind[0], positions_by_kind[1], near, f.fps, social_config,
                confirmation=confirmation)
            continue
        for start, end in runs(seed):
            low, high = start, end
            while low > 0 and start - low < continuation and active[low - 1]:
                low -= 1
            while high < len(active) and high - end < continuation and active[high]:
                high += 1
            result[low:high] = True
    return _single_bout_mask(result, f.fps, social_config.attack_min_duration_s)


def _attack_action_mask(
    f: ExportedFeatures, edge: int,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, point_kinematics: dict | None = None, channels: dict | None = None,
    allow_contact_rescue: bool = True, proximity_mask: np.ndarray | None = None,
    allow_observed_attack_extension: bool = True,
) -> np.ndarray:
    """Light: measured directed nose/head acceleration with both IDs visible.

    Heavy: repeated bbox disappearance of the SAME target, with bounded
    joint-near and directed actor-motion anchors. One long gap is unknown, not a third
    attack route. No missing point kinematics are invented by either channel.
    """
    result = np.zeros(len(f.valid), dtype=bool)
    heavy_result = np.zeros_like(result)
    if channels is not None:
        channels.update(light_observed_acceleration=result, heavy_repeated_target_boxes=heavy_result)
    if f.centers_cm is None:
        return result
    left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
    visible = f.valid[:, left] & f.valid[:, right]
    distance = f.pair_distance_cm[:, edge]
    near = np.isfinite(distance) & (distance <= social_config.dist_attack_max_cm)
    if allow_contact_rescue:
        # Keep center-only inferred intervals unchanged. Geometry may recover
        # observed action rows, but cannot paint any additional missing row.
        cache = point_kinematics if point_kinematics is not None else {}
        baseline_channels = {}
        baseline = _attack_action_mask(
            f, edge, social_config, point_kinematics=cache,
            channels=baseline_channels, allow_contact_rescue=False,
        )
        gap = body_footprint_distances(
            f, edge, max_body_lengths=social_config.attack_contact_max_footprint_body_lengths,
            frame_mask=visible & ~near,
        )
        contact = visible & np.isfinite(gap) & (gap <= social_config.attack_contact_surface_max_gap_cm)
        candidate_channels = {}
        if contact.any():
            _attack_action_mask(
                f, edge, social_config, point_kinematics=cache,
                channels=candidate_channels, allow_contact_rescue=False,
                proximity_mask=(near | contact) & visible,
            )
            # Loss-video review found normal head turns in the newly eligible
            # light route. Only repeated-target-box action has video-supported
            # center-far gains in this cohort; keep light proximity unchanged.
            expanded_heavy = candidate_channels['heavy_repeated_target_boxes']
            established_process = np.zeros_like(baseline)
            for start, end in runs(expanded_heavy):
                if baseline_channels['heavy_repeated_target_boxes'][start:end].any():
                    established_process[start:end] = True
            # Repair truncation of an established center-based heavy process.
            # A new footprint cue must not create a standalone heavy event in
            # a normal contact clip whose tracker also happens to blink.
            measured_candidate = expanded_heavy & established_process & visible
        else:
            measured_candidate = np.zeros_like(baseline)
        coherent_baseline = _coherent_observed_attack_action(
            f, edge, social_config, point_kinematics=cache,
            allow_observed_run_extension=False)
        coherent_extension = (_coherent_observed_attack_action(
            f, edge, social_config, point_kinematics=cache,
            continuation_proximity=(near & visible) | contact)
            if allow_observed_attack_extension else np.zeros_like(coherent_baseline))
        coherent = coherent_baseline | coherent_extension
        if channels is not None:
            channels.update(
                center_only_raw=baseline,
                observed_contact_rescue=measured_candidate & ~baseline,
                light_observed_acceleration=baseline_channels['light_observed_acceleration'] | coherent,
                coherent_observed_action=coherent,
                coherent_v24_baseline=coherent_baseline,
                coherent_v25_observed_extension=coherent_extension & ~coherent_baseline,
                heavy_repeated_target_boxes=(baseline_channels['heavy_repeated_target_boxes']
                    | measured_candidate),
            )
        return baseline | measured_candidate | coherent
    if proximity_mask is not None:
        near = np.asarray(proximity_mask, bool)
        if near.shape != visible.shape or (near & ~visible).any():
            raise ValueError("attack proximity must contain only observed pair rows")
    minimum = max(1, int(np.ceil(social_config.attack_min_duration_s * f.fps - 1e-9)))

    points = f.collective_keypoints_cm
    # SOC-3: direction-specific. A stable nose-on-tail follow by `actor` must
    # only suppress `actor`'s own pounce; the v16 code OR-ed both orderings into
    # one mask. The leader must be WALKING NORMALLY IN THE SAME DIRECTION (spec
    # module 3 item 1 / 2.1-A3): the walking floor is the individual layer's own
    # Locomotion boundary, so no new scale-era speed constant is introduced, and
    # the heading test is new. A missing heading maps to -1.0 through
    # _toward_cosine, so an unknown heading never suppresses anything.
    tail_follow_by_actor: dict = {
        left: np.zeros(len(f.valid), dtype=bool),
        right: np.zeros(len(f.valid), dtype=bool),
    }
    if points is not None:
        tail = endpoint_distances(f, edge, "tail")
        heading = f.heading_unit
        aligned_cosine = float(np.cos(
            np.deg2rad(social_config.attack_tail_follow_max_heading_angle_deg)
        ))
        for direction, (actor, target) in enumerate(((left, right), (right, left))):
            directed_tail = tail[direction]
            target_speed = np.asarray(f.speed_cm_s[:, target], dtype=float)
            same_direction = np.zeros(len(f.valid), dtype=bool)
            if heading is not None:
                both_known = (np.isfinite(heading[:, actor]).all(axis=1)
                              & np.isfinite(heading[:, target]).all(axis=1))
                same_direction = both_known & (
                    _toward_cosine(heading[:, actor], heading[:, target]) >= aligned_cosine
                )
            tail_follow = (
                visible & np.isfinite(directed_tail)
                & (directed_tail <= social_config.dist_nose_tail_contact_max_cm)
                & np.isfinite(target_speed)
                & (target_speed > SOCIAL_TAIL_FOLLOW_MIN_TARGET_SPEED_CM_S)
                & same_direction
            )
            tail_follow_by_actor[actor] = _single_bout_mask(
                tail_follow, f.fps, social_config.attack_tail_follow_min_duration_s,
            )
    toward_min = 0.0
    if points is not None:
        for actor, target in ((left, right), (right, left)):
            head = np.asarray(points[:, actor, :3], float)
            complete_head = np.isfinite(head).all(axis=(1, 2))
            head_center = np.where(complete_head[:, None], head.mean(axis=1), np.nan)
            for point_kind, positions in enumerate((head[:, 0], head_center)):
                key = (actor, point_kind,
                       social_config.attack_acceleration_fit_half_window_s,
                       social_config.attack_max_track_jump_speed_cm_s)
                if point_kinematics is not None and key in point_kinematics:
                    velocity, acceleration = point_kinematics[key]
                else:
                    velocity, acceleration = _local_quadratic_point_kinematics(
                        positions, f.valid[:, actor], f.fps,
                        half_window_s=social_config.attack_acceleration_fit_half_window_s,
                        max_jump_speed_cm_s=social_config.attack_max_track_jump_speed_cm_s,
                    )
                    if point_kinematics is not None:
                        point_kinematics[key] = velocity, acceleration
                offset = f.centers_cm[:, target] - positions
                toward_acceleration = _toward_cosine(acceleration, offset)
                toward_velocity = _toward_cosine(velocity, offset)
                seed = (
                    visible & near & np.isfinite(acceleration).all(axis=1)
                    & (np.linalg.norm(acceleration, axis=1) >= social_config.acc_attack_min_cm_s2)
                    & (toward_acceleration > toward_min)
                    & ~tail_follow_by_actor[actor]
                )
                continuation_frames = max(1, int(np.floor(
                    social_config.attack_deceleration_max_s * f.fps + 1e-9)))
                # Trigger thresholds confirm onset, not every frame of an action.
                # A measured slowdown can fall below the former trigger speed.
                # Its directed continuation is bounded from the last real trigger;
                # stops, reversals, separation and missing observations still end it.
                slowing = (np.sum(velocity * acceleration, axis=1) < 0.)
                active = (visible & near & np.isfinite(velocity).all(axis=1)
                          & np.isfinite(acceleration).all(axis=1)
                          & (np.linalg.norm(velocity, axis=1) > 1e-9)
                          & (toward_velocity > toward_min)
                          & (slowing | (np.linalg.norm(velocity, axis=1)
                                        >= social_config.vel_attack_min_cm_s))
                          & ~tail_follow_by_actor[actor])
                for start, end in runs(seed):
                    if end - start < minimum:
                        continue
                    stop = end
                    while (stop < len(active) and stop - end < continuation_frames
                           and active[stop]):
                        stop += 1
                    result[start:stop] = True

    boxes = f.collective_boxes_cm
    if boxes is None:
        return result
    anchor_width = max(1, int(np.ceil(social_config.attack_anchor_window_s * f.fps)))
    cycle_limit = int(np.floor(social_config.attack_box_gap_max_s * f.fps + 1e-9))
    for actor, target in ((left, right), (right, left)):
        target_box_visible = np.isfinite(boxes[:, target]).all(axis=1)
        offset = f.centers_cm[:, target] - f.centers_cm[:, actor]
        actor_velocity = f.velocity_cm_s[:, actor]
        directed = _toward_cosine(actor_velocity, offset) > toward_min
        anchor_motion = (visible & near & directed
                         & (f.speed_cm_s[:, actor] >= social_config.vel_attack_min_cm_s)
                         & ~tail_follow_by_actor[actor])
        cycles = []
        for start, end in runs(~target_box_visible):
            if (start <= 0 or end >= len(visible) or end - start > cycle_limit
                    or not target_box_visible[start - 1] or not target_box_visible[end]):
                continue
            # The same target ID bounds each cycle. The actor may also be
            # occluded inside it; only observed anchors supply motion. Another
            # target's blink cannot complete this target's repeated cycle.
            before = slice(max(0, start - anchor_width), start)
            after = slice(end, min(len(visible), end + anchor_width))
            # A target can reappear before the actor's keypoints recover. Use
            # the nearest jointly observed pair on each side, within the same
            # bounded anchor windows, rather than requiring adjacent valid rows.
            before_anchors = np.flatnonzero(visible[before] & near[before]) + before.start
            after_anchors = np.flatnonzero(visible[after] & near[after]) + after.start
            if not len(before_anchors) or not len(after_anchors):
                continue
            low, high = int(before_anchors[-1]), int(after_anchors[0])
            if not (anchor_motion[before].any() or anchor_motion[after].any()):
                continue
            if (visible[low:high + 1] & ~near[low:high + 1]).any():
                continue
            cycles.append((low, high))
        for index, (start, _) in enumerate(cycles):
            repeated = [cycles[index]]
            for next_start, next_end in cycles[index + 1:]:
                if (next_end - start) / f.fps > social_config.attack_box_repeat_window_s:
                    break
                if (visible[start:next_end + 1] & ~near[start:next_end + 1]).any():
                    break
                if any(gap_end - gap_start > cycle_limit for gap_start, gap_end
                       in runs(~target_box_visible[start:next_end + 1])):
                    break
                repeated.append((next_start, next_end))
            if len(repeated) >= social_config.attack_box_reappearance_min_cycles:
                end = repeated[-1][1] + 1
                if end - start >= minimum:
                    heavy_result[start:end] = True
    return result | heavy_result

def _open_approach_mask(
    f: ExportedFeatures, edge: int, p: HeuristicParameters,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, direction_evidence: dict | None = None,
) -> np.ndarray:
    """Directed closing with a fitted endpoint envelope, independent of contacts.

    v17.1 permits stationary, lateral and receding targets. The actor must still
    move toward the target while mutual distance shrinks. Only the completed
    candidate prefix supplies onset, endpoint and validity evidence.
    """
    result = np.zeros(len(f.valid), bool)
    directed_candidates = np.zeros((2, len(f.valid)), bool)
    facing_evidence = np.zeros_like(directed_candidates)
    if direction_evidence is not None:
        direction_evidence.update(candidates=directed_candidates, facing=facing_evidence)
    if f.centers_cm is None:
        return result
    _report_social_mirror_drift("approach", (
        ("pair_min_duration_s", p.pair_min_duration_s, social_config.approach_min_duration_s),
        ("approach_min_actor_speed_cm_s", p.approach_min_actor_speed_cm_s,
         social_config.vel_approach_actor_min_cm_s),
    ))
    left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
    visible = f.valid[:, left] & f.valid[:, right]
    distance = np.asarray(f.pair_distance_cm[:, edge], float)
    minimum = max(1, int(np.ceil(social_config.approach_min_duration_s * f.fps - 1e-9)))
    lag = max(1, int(round(social_config.approach_min_duration_s * f.fps)))
    gap_frames = max(0, int(np.floor(social_config.approach_window_max_gap_s * f.fps + 1e-9)))
    filled_distance = _fill_short_nan_gaps(distance, gap_frames)[0]
    closing = np.zeros(len(visible), bool)
    if len(visible) > lag:
        earlier, later = filled_distance[:-lag], filled_distance[lag:]
        measured = np.isfinite(earlier) & np.isfinite(later)
        difference = np.full(len(earlier), np.nan)
        np.subtract(earlier, later, out=difference, where=measured)
        closing[lag:] = difference * f.fps / lag > social_config.approach_min_closing_speed_cm_s
    surfaces = snout_surface_distances(f, edge)
    orientation = observed_orientations(f)
    ratio = social_config.approach_window_min_valid_ratio
    for direction, (actor, target) in enumerate(((left, right), (right, left))):
        surface_raw = np.asarray(surfaces[direction], float)
        surface = _fill_short_nan_gaps(surface_raw, gap_frames)[0]
        offset = f.centers_cm[:, target] - f.centers_cm[:, actor]
        velocity = f.velocity_cm_s[:, actor]
        actor_speed = f.speed_cm_s[:, actor]
        motion = _toward_cosine(velocity, offset)
        directed = (actor_speed > social_config.vel_approach_actor_min_cm_s) & (motion > 0.0)
        pose_known, facing = pose_direction_support(
            f, actor, target, social_config, "approach", orientation=orientation)
        facing_evidence[direction] = facing
        directed &= ~pose_known | facing
        known = visible & np.isfinite(actor_speed) & np.isfinite(velocity).all(axis=1)
        hesitant = known & ~directed & ~(pose_known & ~facing) & np.isfinite(filled_distance)
        eligible = visible & np.isfinite(filled_distance) & directed & closing
        eligible = _bridge_internal_gaps(
            eligible, np.isfinite(filled_distance), f.fps,
            maximum_gap_s=social_config.approach_window_max_gap_s,
            unknown=~known | hesitant,
        )
        for start, end in runs(eligible):
            # The lag is distance evidence, not a demand that an unmeasured
            # crop-boundary velocity already be directed. Anchor at the first
            # observed directed row inside that bounded lookback instead.
            lookback = np.arange(max(0, start - lag), start + 1)
            anchors = lookback[
                visible[lookback] & directed[lookback]
                & np.isfinite(surface_raw[lookback])
                & np.isfinite(filled_distance[lookback])
            ]
            if not len(anchors):
                continue
            onset = int(anchors[0])
            if filled_distance[onset] < social_config.dist_approach_start_min_cm:
                continue
            possible = np.arange(onset + minimum - 1, end)
            endpoint_ok = (
                (filled_distance[possible] >= social_config.dist_approach_end_min_cm)
                & (filled_distance[possible] <= social_config.dist_approach_end_max_cm)
                & (np.abs(surface[possible] - social_config.approach_surface_endpoint_cm)
                   <= social_config.approach_surface_endpoint_tolerance_cm + 1e-9)
                & (filled_distance[possible] < filled_distance[onset])
                & visible[possible] & directed[possible]
                & np.isfinite(surface_raw[possible])
            )
            possible = possible[endpoint_ok]
            # Validate the candidate event itself. Unavailable geometry or
            # changed motion after its endpoint cannot erase a completed prefix.
            for endpoint in possible[::-1]:
                event = slice(onset, int(endpoint) + 1)
                if (visible[event].mean() < ratio or directed[event].mean() < ratio
                        or np.isfinite(surface_raw[event]).mean() < ratio):
                    continue
                result[event] = True
                directed_candidates[direction, event] = True
                break
    return result


def _pursuit_direction_cosine(f: ExportedFeatures, actor: int, target: int,
                             social_config: SocialHeuristicConfig) -> np.ndarray:
    """Same direction now, or follower reaches the leader's observed turn later.

    Reuses the screening policy's bounded 0.5-second lag and body-width corridor.
    A later trajectory is never used to infer a missing visual identity.
    """
    frames = np.arange(len(f.valid))
    best = np.full(len(frames), -1.)
    cosine_min = float(np.cos(np.deg2rad(social_config.angle_chasing_behind_max_deg)))
    ahead = _toward_cosine(f.velocity_cm_s[:, actor], f.centers_cm[:, target] - f.centers_cm[:, actor])
    behind = _toward_cosine(f.velocity_cm_s[:, target], f.centers_cm[:, target] - f.centers_cm[:, actor])
    corridor = CONFIRMED_POLICY.chase_path_width_fraction * f.mouse_width_cm
    for lag in range(round(f.fps * CONFIRMED_POLICY.chase_heading_lag_s) + 1):
        previous = np.maximum(0, frames - lag)
        similarity = _toward_cosine(f.velocity_cm_s[:, actor], f.velocity_cm_s[previous, target])
        path_gap = np.linalg.norm(f.centers_cm[:, actor] - f.centers_cm[previous, target], axis=1)
        advance = np.linalg.norm(f.centers_cm[:, target] - f.centers_cm[previous, target], axis=1)
        follows_path = (lag > 0) & (path_gap <= corridor) & (advance >= corridor)
        forward = (behind >= cosine_min) if lag == 0 else ((ahead >= cosine_min) | follows_path)
        usable = ((frames >= lag) & f.valid[:, actor] & f.valid[:, target] & f.valid[previous, target]
                  & f.valid[previous, actor] & forward & (f.speed_cm_s[previous, target] >= social_config.vel_chasing_target_min_cm_s))
        best = np.maximum(best, np.where(usable, similarity, -1.))
    return best


def _qualify_observed_chase(mask: np.ndarray, known: np.ndarray, centers: np.ndarray,
                            left: int, right: int, fps: float,
                            config: SocialHeuristicConfig) -> np.ndarray:
    """Combine visible support across bounded unknowns without filling them.

    Time, path and net displacement all use jointly observed eligible frames.
    Net is the vector sum of adjacent observed increments, excluding every
    cross-gap jump. Observed stops, reversals and priority losses split support.
    """
    mask = np.asarray(mask, bool) & known
    result = np.zeros_like(mask)
    bounds = runs(mask)
    episodes = []
    limit = int(np.floor(config.chasing_reacquisition_gap_max_s * fps + 1e-9))
    for low, high in bounds:
        if (episodes and low - episodes[-1][1] <= limit
                and (~known[episodes[-1][1]:low]).all()):
            episodes[-1] = episodes[-1][0], high
        else:
            episodes.append((low, high))
    minimum = max(1, int(np.ceil(config.chasing_min_duration_s * fps - 1e-9)))
    for low, high in episodes:
        support = mask[low:high]
        if support.sum() < minimum:
            continue
        adjacent = support[1:] & support[:-1]
        adjacent_seconds = float(adjacent.sum()) / fps
        if adjacent_seconds <= 0:
            continue
        travels = []
        for slot in (left, right):
            steps = np.diff(centers[low:high, slot], axis=0)[adjacent]
            path = float(np.linalg.norm(steps, axis=1).sum())
            net = float(np.linalg.norm(steps.sum(axis=0)))
            travels.append((path, net, path / adjacent_seconds))
        if all(path >= config.chasing_min_path_displacement_cm
               and net >= config.chasing_min_net_displacement_cm
               and mean_speed + 1e-9 >= config.chasing_min_mean_speed_cm_s
               for path, net, mean_speed in travels):
            result[low:high] |= support
    return result


def _coobserved_chase_motion(
    f: ExportedFeatures, edge: int,
    measured_motion: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute pose-based center motion inside each joint observation run.

    A finite lagged velocity must not jump across a missing observation. Median
    smoothing also stays inside the joint run, so the other side of a dropout
    cannot affect a speed or direction operand. Other behaviors keep their own
    motion definitions. Non-pose callers supply measured motion directly.
    """
    if f.collective_keypoints_cm is None:
        return measured_motion
    slots = [int(f.pair_i[edge]), int(f.pair_j[edge])]
    joint = f.valid[:, slots].all(axis=1) & np.isfinite(f.centers_cm[:, slots]).all(axis=(1, 2))
    centers, velocity, speed = (np.array(value, copy=True) for value in measured_motion)
    centers[:, slots] = np.nan
    velocity[:, slots] = np.nan
    speed[:, slots] = np.nan
    for low, high in runs(joint):
        local_centers, local_velocity, local_speed = observed_run_motion(
            f.centers_cm[low:high, slots], f.fps, CONFIRMED_POLICY)
        centers[low:high, slots] = local_centers
        velocity[low:high, slots] = local_velocity
        speed[low:high, slots] = local_speed
    return centers, velocity, speed


def _chase_mask(
    f: ExportedFeatures, edge: int,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, measured_motion: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    direction_evidence: dict | None = None,
) -> np.ndarray:
    """Constrain pursuit only with jointly visible measured body motion."""
    result = np.zeros(len(f.valid), dtype=bool)
    directed_candidates = np.zeros((2, len(f.valid)), bool)
    facing_evidence = np.zeros_like(directed_candidates)
    if direction_evidence is not None:
        direction_evidence.update(candidates=directed_candidates, facing=facing_evidence)
    if f.centers_cm is None:
        return result
    if measured_motion is None:
        measured_motion = body_motion(f, CONFIRMED_POLICY)
    measured_motion = _coobserved_chase_motion(f, edge, measured_motion)
    _, velocity, speed = measured_motion
    f = replace(f, velocity_cm_s=velocity, speed_cm_s=speed)
    left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
    visible = f.valid[:, left] & f.valid[:, right]
    motion_known = (visible & np.isfinite(velocity[:, [left, right]]).all(axis=(1, 2))
                    & np.isfinite(speed[:, [left, right]]).all(axis=1)
                    & np.isfinite(f.centers_cm[:, [left, right]]).all(axis=(1, 2)))
    direction_cosine_min = float(np.cos(np.deg2rad(social_config.angle_chasing_dir_max_deg)))
    orientation = observed_orientations(f)
    for direction, (actor, target) in enumerate(((left, right), (right, left))):
        direction_cosine = _pursuit_direction_cosine(f, actor, target, social_config)
        motion_support = (motion_known
                    & (speed[:, actor] >= social_config.vel_chasing_min_cm_s)
                    & (speed[:, target] >= social_config.vel_chasing_target_min_cm_s)
                    & (direction_cosine >= direction_cosine_min))
        # Only genuine observed path-following permits a pose reference along
        # the follower's path at a leader turn. Parallel side-by-side motion
        # does not satisfy the existing directed pursuit geometry.
        direct = _toward_cosine(velocity[:, target], f.centers_cm[:, target] - f.centers_cm[:, actor])
        turn_path = motion_support & (direct < np.cos(np.deg2rad(social_config.angle_chasing_behind_max_deg)))
        pose_known, facing = pose_direction_support(
            f, actor, target, social_config, "chasing", orientation=orientation,
            turn_path=turn_path)
        facing_evidence[direction] = facing
        eligible = bridge_pose_turns(
            motion_support & (~pose_known | facing), motion_support, f.fps,
            social_config.chasing_role_turn_grace_s)
        directed_candidates[direction] = _qualify_observed_chase(
            eligible, motion_known, f.centers_cm, left, right, f.fps, social_config)
        result |= directed_candidates[direction]
    return result

def _base_pair_rule_masks(
    f: ExportedFeatures,
    p: HeuristicParameters,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, allow_observed_attack_extension: bool = True, diagnostics: dict | None = None,
    role_evidence: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build six independent candidates, then resolve their temporal overlaps.

    Arbitration acts only on overlapping frames for each unordered visual-ID pair. Contact never
    vetoes a higher-priority Approach or Together candidate. Each continuous
    contact bout must independently meet its duration; measured negatives split it.
    """
    shape = f.pair_distance_cm.shape
    names = ("approach", "attack", "chasing", "together", "head_contact", "tail_contact")
    result = {name: np.zeros(shape, bool) for name in names}
    result["head_contact_seconds"] = np.zeros(shape[1], float)
    result["tail_contact_seconds"] = np.zeros(shape[1], float)
    result["long_contact_review"] = np.zeros(shape, bool)
    result["contact_start_context"] = np.zeros(shape, bool)
    # SOC-12: `p` IS read. These are the ranking layer's mirror of the non-approach
    # social policy; _open_approach_mask() reports the approach half. The masks
    # below are driven by social_config alone, so a drifted mirror is reported
    # once instead of silently ignored. It is never a threshold source:
    # scripts/score_segment_level_v16.py builds `p` from configs/default.yaml,
    # which does not set confirmed_rules_v1, and must keep receiving these masks.
    _report_social_mirror_drift("contact/together/chase/attack", (
        ("contact_distance_cm", p.contact_distance_cm,
         max(social_config.dist_nose_head_contact_max_cm,
             social_config.dist_nose_tail_contact_max_cm)),
        ("contact_min_cumulative_seconds", p.contact_min_cumulative_seconds,
         social_config.nose_contact_min_duration_s),
        ("together_max_distance_cm", p.together_max_distance_cm,
         social_config.dist_together_max_cm),
        ("together_min_duration_s", p.together_min_duration_s,
         social_config.together_min_duration_s),
        ("chase_min_chaser_speed_cm_s", p.chase_min_chaser_speed_cm_s,
         social_config.vel_chasing_min_cm_s),
        ("chase_min_net_displacement_cm", p.chase_min_net_displacement_cm,
         social_config.chasing_min_net_displacement_cm),
        ("chase_min_directed_support_s", p.chase_min_directed_support_s,
         social_config.chasing_min_duration_s),
        ("attack_min_nose_acceleration_toward_cm_s2",
         p.attack_min_nose_acceleration_toward_cm_s2,
         social_config.acc_attack_min_cm_s2),
        ("attack_endpoint_distance_cm", p.attack_endpoint_distance_cm,
         social_config.dist_attack_max_cm),
        ("attack_min_event_duration_s", p.attack_min_event_duration_s,
         social_config.attack_min_duration_s),
    ))
    body_cm = float(f.reference_body_cm)
    visible = f.valid[:, f.pair_i] & f.valid[:, f.pair_j]
    # A track's point kinematics do not depend on its partner. Bound this cache
    # to one prediction so no later clip can inherit an earlier track ID.
    point_kinematics: dict = {}
    chase_motion = body_motion(f, CONFIRMED_POLICY)
    together_speed = (chase_motion[2] if social_config.together_use_smoothed_speed
                      else f.speed_cm_s)
    for edge in range(shape[1]):
        if not visible[:, edge].any():
            continue
        left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
        approach_evidence, chase_evidence = {}, {}
        approach = _open_approach_mask(
            f, edge, p, social_config, direction_evidence=approach_evidence)
        head_directed = endpoint_distances(f, edge, "head")
        tail_directed = endpoint_distances(f, edge, "tail")
        observed_pair = visible[:, edge]
        center = f.pair_distance_cm[:, edge]
        chase = _chase_mask(f, edge, social_config, measured_motion=chase_motion,
                            direction_evidence=chase_evidence)
        if role_evidence is not None:
            role_evidence[edge] = {"approach": approach_evidence, "chasing": chase_evidence}
        attack_channels = {}
        attack = _attack_action_mask(f, edge, social_config, point_kinematics=point_kinematics,
                                     channels=attack_channels,
                                     allow_observed_attack_extension=allow_observed_attack_extension)
        if diagnostics is not None and np.any(attack_channels.get('coherent_v25_observed_extension', False)):
            diagnostics['new_coherent_candidate'] = True
        center_only_attack = attack_channels.get('center_only_raw', attack)

        head = np.zeros(len(f.valid), bool)
        tail = np.zeros(len(f.valid), bool)
        contact_bout_seconds = {"head": 0.0, "tail": 0.0}
        overlong_head = np.zeros(len(f.valid), dtype=bool)
        overlong_tail = np.zeros(len(f.valid), dtype=bool)
        surface_distances = snout_surface_distances(f, edge)
        for name, directed, requested_distance in (
            ("head", head_directed, social_config.dist_nose_head_contact_max_cm),
            ("tail", tail_directed, social_config.dist_nose_tail_contact_max_cm),
        ):
            bout_mask = np.zeros(len(f.valid), bool)
            overlong_union = np.zeros(len(f.valid), bool)
            for direction, (primary, (actor, _)) in enumerate(zip(
                directed, ((left, right), (right, left)), strict=True
            )):
                primary = np.asarray(primary, dtype=float)
                strict = (observed_pair & np.isfinite(primary)
                          & (primary <= requested_distance)
                          & _closest_contact_endpoint(
                              primary, tail_directed if name == "head" else head_directed))
                qualified, _ = _bridged_contact_bout_mask(
                    strict=strict, measured=primary, center=center, body_cm=body_cm,
                    fps=f.fps, social_config=social_config, kind=name,
                )
                overlong = np.zeros(len(f.valid), bool)
                for start, end in runs(qualified):
                    if (end - start) / f.fps > social_config.nose_contact_max_duration_s:
                        overlong[start:end] = True
                    path, _ = observed_path_and_net(
                        f.centers_cm[start:end, actor], observed_pair[start:end]
                    )
                    # Optional close-start/small-displacement context. This
                    # explains a crop that starts at contact without requiring
                    # an earlier Approach or replacing subtype geometry.
                    if (np.isfinite(surface_distances[direction][start])
                            and surface_distances[direction][start] <= social_config.nose_contact_candidate_distance_cm
                            and path <= social_config.nose_contact_context_max_displacement_cm):
                        result["contact_start_context"][start:end, edge] = True
                bout_mask |= qualified
                overlong_union |= overlong
                contact_bout_seconds[name] = max(contact_bout_seconds[name], max(
                    ((end - start) / max(f.fps, 1e-9) for start, end in runs(qualified)),
                    default=0.0,
                ))
            if name == "head":
                head = bout_mask
                overlong_head = overlong_union
            else:
                tail = bout_mask
                overlong_tail = overlong_union
        result["long_contact_review"][:, edge] = overlong_head | overlong_tail
        # Together is independent evidence. Six-way arbitration below decides
        # its overlap with Approach, Attack, Chase and the two contact subtypes.
        together_support = np.isfinite(center) & (center < social_config.dist_together_max_cm)
        together_raw = (
            observed_pair & together_support
            & (together_speed[:, left] < social_config.vel_together_max_cm_s)
            & (together_speed[:, right] < social_config.vel_together_max_cm_s)
        )
        together_unknown = ~(
            observed_pair & np.isfinite(center)
            & np.isfinite(together_speed[:, left]) & np.isfinite(together_speed[:, right])
        )
        together = _single_bout_mask(
            _bridge_internal_gaps(
                together_raw, together_support, f.fps,
                maximum_gap_s=social_config.together_bridge_max_gap_s,
                unknown=together_unknown,
            ),
            f.fps, social_config.together_min_duration_s,
        )
        # Resolve all labels for a pair before ranking pairs in the clip.
        # Contact is independent of its previous approach when the exported
        # crop begins at contact; source-context may lie outside that crop.
        attack_s = max(
            ((end - start) / max(f.fps, 1e-9) for start, end in runs(attack)),
            default=0.0,
        )
        head_s = contact_bout_seconds["head"]
        tail_s = contact_bout_seconds["tail"]
        approach_s = max(
            ((end - start) / max(f.fps, 1e-9) for start, end in runs(approach)),
            default=0.0,
        )
        together_s = max(
            ((end - start) / max(f.fps, 1e-9) for start, end in runs(together)),
            default=0.0,
        )
        attack_qualified = attack_s + 1e-12 >= social_config.attack_min_duration_s
        chase_qualified = bool(chase.any())
        approach_qualified = approach_s + 1e-12 >= social_config.approach_min_duration_s
        together_qualified = together_s + 1e-12 >= social_config.together_min_duration_s
        candidates = {
            "approach": (approach, approach_qualified),
            "together": (together, together_qualified),
            "attack": (attack, attack_qualified),
            "chasing": (chase, chase_qualified),
            "head_contact": (head, head_s + 1e-12 >= social_config.nose_contact_min_duration_s),
            "tail_contact": (tail, tail_s + 1e-12 >= social_config.nose_tail_min_duration_s),
        }
        claimed = np.zeros(len(f.valid), bool)
        minima = {"approach": social_config.approach_min_duration_s,
                  "together": social_config.together_min_duration_s,
                  "attack": social_config.attack_min_duration_s,
                  "chasing": social_config.chasing_min_duration_s,
                  "head_contact": social_config.nose_contact_min_duration_s,
                  "tail_contact": social_config.nose_tail_min_duration_s}
        for behavior in social_config.priority_order:
            mask, qualified = candidates[behavior]
            if not qualified:
                continue
            # Priority resolves simultaneous evidence. An earlier Approach
            # cannot delete a later Attack/contact on this same visual pair.
            if behavior == "chasing":
                observed_motion = _coobserved_chase_motion(f, edge, chase_motion)
                motion_known = (visible[:, edge]
                                & np.isfinite(observed_motion[1][:, [left, right]]).all(axis=(1, 2))
                                & np.isfinite(observed_motion[2][:, [left, right]]).all(axis=1)
                                & np.isfinite(f.centers_cm[:, [left, right]]).all(axis=(1, 2)))
                selected = _qualify_observed_chase(
                    mask & ~claimed, motion_known, f.centers_cm, left, right, f.fps, social_config)
            else:
                selected = _single_bout_mask(mask & ~claimed, f.fps, minima[behavior])
                if behavior in ("head_contact", "tail_contact"):
                    values = head_directed if behavior == "head_contact" else tail_directed
                    cap = (social_config.dist_nose_head_contact_max_cm if behavior == "head_contact"
                           else social_config.dist_nose_tail_contact_max_cm)
                    selected = _measured_contact_bout_mask(
                        selected, tuple(visible[:, edge] & np.isfinite(v) & (v <= cap) for v in values),
                        f.fps, minima[behavior],
                        tuple(visible[:, edge] & np.isfinite(v)
                              & (v <= social_config.contact_onset_cm(
                                  "head" if behavior == "head_contact" else "tail")) for v in values))
                if behavior == "attack":
                    baseline = _single_bout_mask(center_only_attack & ~claimed, f.fps, minima[behavior])
                    # Observed rescue must not requalify previously subthreshold
                    # unknown tails after priority has removed their anchors.
                    selected = _single_bout_mask(
                        selected & (visible[:, edge] | baseline), f.fps, minima[behavior],
                    )
            result[behavior][:, edge] = selected
            claimed |= selected
            seconds = max(((end - start) / f.fps for start, end in runs(selected)), default=0.)
            if behavior == "head_contact":
                result["head_contact_seconds"][edge] = seconds
            elif behavior == "tail_contact":
                result["tail_contact_seconds"][edge] = seconds

        # A fully qualified, sustained pursuit is more specific than its
        # closing-distance phase. Transfer only Approach/Chasing occupancy
        # after the existing arbitration; never promote Chase over an already
        # selected Attack or Together on the pair. Lower-priority contacts
        # cannot reserve a leftover fragment that prevented this requalification.
        if chase.any() and (chase & result["approach"][:, edge]).any():
            occupied = result["attack"][:, edge] | result["together"][:, edge]
            for name, priority in (("head_contact", social_config.nose_contact_priority),
                                   ("tail_contact", social_config.nose_tail_priority)):
                if priority < social_config.chasing_priority:
                    occupied |= result[name][:, edge]
            observed_motion = _coobserved_chase_motion(f, edge, chase_motion)
            known = (visible[:, edge]
                     & np.isfinite(observed_motion[1][:, [left, right]]).all(axis=(1, 2))
                     & np.isfinite(observed_motion[2][:, [left, right]]).all(axis=1)
                     & np.isfinite(f.centers_cm[:, [left, right]]).all(axis=(1, 2)))
            restored = np.zeros(len(f.valid), bool)
            for directed_chase in chase_evidence.get("candidates", (chase,)):
                restored |= _qualify_observed_chase(
                    directed_chase & ~occupied, known, f.centers_cm,
                    left, right, f.fps, social_config)
            result["chasing"][:, edge] |= restored
            result["approach"][:, edge] = _single_bout_mask(
                result["approach"][:, edge] & ~restored, f.fps,
                social_config.approach_min_duration_s)
            for name, kind, values, cap in (
                ("head_contact", "head", head_directed, social_config.dist_nose_head_contact_max_cm),
                ("tail_contact", "tail", tail_directed, social_config.dist_nose_tail_contact_max_cm),
            ):
                result[name][:, edge] = _measured_contact_bout_mask(
                    result[name][:, edge] & ~restored,
                    tuple(observed_pair & np.isfinite(v) & (v <= cap) for v in values),
                    f.fps, social_config.contact_minimum_s(kind),
                    tuple(observed_pair & np.isfinite(v) & (v <= social_config.contact_onset_cm(kind))
                          for v in values))
                result[name + "_seconds"][edge] = max(
                    ((end - start) / f.fps for start, end in runs(result[name][:, edge])), default=0.)

    return result

def _restrict_contact_unknown_support(
    result: dict, reference: dict, f: ExportedFeatures,
    social_config: SocialHeuristicConfig, *, allow_contact_extension: bool = True,
) -> None:
    """Keep historical unknown rows only in the same canonical contact subtype.

    New observed attack support can split an older head bout and expose a tail
    candidate underneath. Its old head-owned unknown row is not tail evidence.
    Requalify each remaining fragment without reallocating deleted support.
    """
    visible = f.valid[:, f.pair_i] & f.valid[:, f.pair_j]
    for name, minimum in (
        ('head_contact', social_config.contact_head_extension_min_duration_s),
        ('tail_contact', social_config.contact_tail_extension_min_duration_s),
    ):
        if not allow_contact_extension:
            minimum = social_config.contact_minimum_s('head' if name == 'head_contact' else 'tail')
        for edge in range(visible.shape[1]):
            selected = _single_bout_mask(
                result[name][:, edge] & (visible[:, edge] | reference[name][:, edge]),
                f.fps, minimum,
            )
            if selected.any():
                kind = 'head' if name == 'head_contact' else 'tail'
                selected = _qualify_contact_channels(
                    selected, visible[:, edge],
                    {k: endpoint_distances(f, edge, k) for k in ('head', 'tail')},
                    kind, f.fps, social_config, extension=allow_contact_extension)
            result[name][:, edge] = selected
            result[name + '_seconds'][edge] = max(
                ((end - start) / f.fps for start, end in runs(selected)), default=0.)


def pair_rule_masks(
    f: ExportedFeatures,
    p: HeuristicParameters,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, allow_contact_extension: bool = True,
    allow_observed_attack_extension: bool = True,
    roles: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Add subtype-specific, jointly observed short bouts to canonical support.

    The baseline includes this revision's attack arbitration. New contact
    evidence never bridges gaps and compares head/tail on the same actor. With
    the default order, all four higher-priority masks remain exactly unchanged.
    """
    diagnostics, role_evidence = {}, {}
    role_options = {"role_evidence": role_evidence} if roles is not None else {}
    result = _base_pair_rule_masks(
        f, p, social_config, allow_observed_attack_extension=allow_observed_attack_extension,
        diagnostics=diagnostics, **role_options)

    def guard_contact_unknowns() -> dict:
        visible = f.valid[:, f.pair_i] & f.valid[:, f.pair_j]
        if (allow_observed_attack_extension and diagnostics.get('new_coherent_candidate', False)
                and ((result['head_contact'] | result['tail_contact']) & ~visible).any()):
            # Recompute the exact retained V24 public path, including priority
            # and short contacts. A same-new-attack baseline cannot detect a
            # head-to-tail transfer, even when total unknown support is stable.
            reference = pair_rule_masks(
                f, p, social_config, allow_contact_extension=allow_contact_extension,
                allow_observed_attack_extension=False)
            _restrict_contact_unknown_support(
                result, reference, f, social_config,
                allow_contact_extension=allow_contact_extension)
        if roles is not None:
            _collect_pair_roles(f, result, role_evidence, roles)
        return result

    if not allow_contact_extension:
        return guard_contact_unknowns()
    minima = {
        "approach": social_config.approach_min_duration_s,
        "together": social_config.together_min_duration_s,
        "attack": social_config.attack_min_duration_s,
        "chasing": social_config.chasing_min_duration_s,
        "head_contact": social_config.contact_head_extension_min_duration_s,
        "tail_contact": social_config.contact_tail_extension_min_duration_s,
    }
    contacts = {
        "head_contact": ("head", social_config.contact_head_extension_distance_cm),
        "tail_contact": ("tail", social_config.contact_tail_extension_distance_cm),
    }
    for edge in range(f.pair_distance_cm.shape[1]):
        left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
        joint = f.valid[:, left] & f.valid[:, right]
        if not joint.any():
            continue
        endpoints = {kind: endpoint_distances(f, edge, kind) for kind in ("head", "tail")}
        claimed = np.zeros(len(f.valid), bool)
        higher_noncontact = np.zeros(len(f.valid), bool)
        for name in social_config.priority_order:
            baseline = result[name][:, edge].copy()
            candidate = baseline.copy()
            if name in contacts:
                kind, distance = contacts[name]
                alternate = "tail" if kind == "head" else "head"
                for primary in endpoints[kind]:
                    strict = (joint & np.isfinite(primary) & (primary <= distance)
                              & _closest_contact_endpoint(primary, endpoints[alternate])
                              & ~higher_noncontact)
                    candidate |= _single_bout_mask(strict, f.fps, minima[name])
            candidate &= ~claimed
            if name == "chasing" and not np.array_equal(candidate, baseline):
                # Custom priority orders may let contact split pursuit. Recheck
                # its observed displacement as well as its duration.
                motion = _coobserved_chase_motion(f, edge, body_motion(f, CONFIRMED_POLICY))
                known = (joint & np.isfinite(motion[1][:, [left, right]]).all(axis=(1, 2))
                         & np.isfinite(motion[2][:, [left, right]]).all(axis=1)
                         & np.isfinite(f.centers_cm[:, [left, right]]).all(axis=(1, 2)))
                selected = _qualify_observed_chase(
                    candidate, known, f.centers_cm, left, right, f.fps, social_config)
            elif name in contacts or not np.array_equal(candidate, baseline):
                selected = _single_bout_mask(candidate, f.fps, minima[name])
            else:
                selected = baseline
            if name in contacts:
                kind, _ = contacts[name]
                selected = _qualify_contact_channels(
                    selected, joint, endpoints, kind, f.fps, social_config)
            result[name][:, edge] = selected
            claimed |= selected
            if name not in contacts:
                higher_noncontact |= selected
            if name in contacts:
                seconds = max(((end - start) / f.fps for start, end in runs(selected)), default=0.)
                result[name.replace("_contact", "_contact_seconds")][edge] = seconds
                for start, end in runs(selected):
                    if (end - start) / f.fps > social_config.nose_contact_max_duration_s:
                        result["long_contact_review"][start:end, edge] = True
    return guard_contact_unknowns()


def _collect_pair_roles(f: ExportedFeatures, masks: dict, evidence: dict,
                        roles: dict[str, np.ndarray]) -> None:
    """Publish actual visual IDs, never slots; -1 is unknown/nonbehavior."""
    roles.update(new_role_buffers(f.pair_distance_cm.shape))
    for edge, directions in evidence.items():
        left, right = int(f.pair_i[edge]), int(f.pair_j[edge])
        for behavior, observed in directions.items():
            if not observed:
                continue
            actor, target = assign_event_roles(
                masks[behavior][:, edge], observed["candidates"], observed["facing"],
                f.track_ids, left, right)
            roles[behavior + "_actor"][:, edge] = actor
            roles[behavior + "_target"][:, edge] = target


def pair_rule_roles(
    f: ExportedFeatures, p: HeuristicParameters,
    social_config: SocialHeuristicConfig = DEFAULT_SOCIAL_HEURISTIC_CONFIG,
    *, masks: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return directed roles aligned to canonical masks, shape (frames,pairs).

    Prefer ``roles={}; pair_rule_masks(..., roles=roles)`` to collect these in
    the same pass. This convenience API can also annotate existing masks
    without rerunning Attack/contact generation. Every value is a real
    ``f.track_ids`` visual ID or -1. An unordered pair-ID score is separate
    from direction accuracy; roles never read annotation participants.
    """
    roles: dict[str, np.ndarray] = {}
    if masks is None:
        pair_rule_masks(f, p, social_config, roles=roles)
        return roles
    evidence = {}
    motion = body_motion(f, CONFIRMED_POLICY)
    for edge in range(f.pair_distance_cm.shape[1]):
        approach, chasing = {}, {}
        _open_approach_mask(f, edge, p, social_config, direction_evidence=approach)
        _chase_mask(f, edge, social_config, measured_motion=motion, direction_evidence=chasing)
        evidence[edge] = {"approach": approach, "chasing": chasing}
    _collect_pair_roles(f, masks, evidence, roles)
    return roles


def dispersal_phase(distances: np.ndarray, visible: np.ndarray, compact: np.ndarray,
                    p: HeuristicParameters, fps: float, *,
                    centers_cm: np.ndarray | None = None,
                    member_valid: np.ndarray | None = None,
                    initial_group_radius_cm: float = 10.0,
                    minimum_outward_members: int = 2,
                    minimum_outward_displacement_cm: float = 4.0,
                    minimum_visible_fraction: float = 0.6,
                    extend_actual_motion: bool = True,
                    minimum_outward_speed_cm_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.dispersal_min_outward_speed_cm_s,
                    pause_max_s: float = DEFAULT_GROUP_HEURISTIC_CONFIG.dispersal_pause_max_s,
                    member_velocity_cm_s: np.ndarray | None = None) -> np.ndarray:
    from .group_event_rules import dispersal_phase as event_phase
    return event_phase(
        distances, visible, compact, p, fps, centers_cm=centers_cm,
        member_valid=member_valid, initial_group_radius_cm=initial_group_radius_cm,
        minimum_outward_members=minimum_outward_members,
        minimum_outward_displacement_cm=minimum_outward_displacement_cm,
        minimum_visible_fraction=minimum_visible_fraction,
        extend_actual_motion=extend_actual_motion,
        minimum_outward_speed_cm_s=minimum_outward_speed_cm_s,
        pause_max_s=pause_max_s,
        member_velocity_cm_s=member_velocity_cm_s,
    )


def group_rule_candidates(f: ExportedFeatures, p: HeuristicParameters,
                          proposals: Sequence[Candidate]) -> tuple[Candidate, ...]:
    from .group_event_rules import group_rule_candidates as event_candidates
    return event_candidates(f, p, proposals)
