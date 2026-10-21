"""Behavior-specific, full-cohort retrospective profile; no held-out claim.

The profile was selected on the active 755 annotated crops. It must not be
described as independently validated accuracy or population specificity.
Frozen v17 extrema and the inactive train-quantile alternative remain intact.
"""

PROFILE_REVISION = "2026-10-18-v18-behavior-specific-retrospective"
FIT_SCOPE = "full_dataset_retrospective_behavior_specific_joint_selection"
INDEPENDENT_VALIDATION = False
ACTIVE_DATASET_SAMPLE_COUNT = 755
ACTIVE_DATASET_MANIFEST_SHA256 = "274243ef8bc054d7ac0fad701217b75e114847e540f2ee8cf4e9de32b208206c"

SOCIAL_FITTED = {
    "dist_nose_head_contact_max_cm": 4.380314658914379,
    "dist_together_max_cm": 5.318182945251465,
    "vel_together_max_cm_s": 14.612846374511719,
    "dist_attack_max_cm": 7.321547985076904,
    "acc_attack_min_cm_s2": 99.59917210877003,
    "vel_attack_min_cm_s": 10.231697660707422,
}

GROUP_FITTED = {
    "huddle_neighbor_distance_cm": 12.987381935119629,
    "locomotion_active_speed_cm_s": 0.11513036230661561,
    "locomotion_min_direction_similarity": 0.6284568309783936,
    "locomotion_max_pair_distance_cm": 64.03693389892578,
    "locomotion_min_net_displacement_cm": 13.836385726928711,
    "dispersal_min_outward_speed_cm_s": 0.11536234840750695,
    "dispersal_pause_max_s": 0.3998359188212399,
}

PARAMETER_SOURCES = {
    "dist_nose_head_contact_max_cm": "full-data annotated head-contact onset-distance p90; joint attack/contact/Together check",
    "dist_together_max_cm": "observed Together per-crop maximum center distance, strict float32 upper bound; joint selection",
    "vel_together_max_cm_s": "observed Together per-crop maximum instantaneous center speed, strict float32 upper bound; joint selection",
    "dist_attack_max_cm": "maximum of annotated Attacker nose/head directed-frame center-distance medians",
    "acc_attack_min_cm_s2": "minimum of annotated Attacker nose/head acceleration-magnitude medians",
    "vel_attack_min_cm_s": "minimum of annotated Attacker nose/head velocity-magnitude medians",
    "huddle_neighbor_distance_cm": "full-data annotated visible-member MST bottleneck p95; exact-set proposal check",
    "locomotion_active_speed_cm_s": "minimum strictly positive smoothed member speed; observed zeros separately retained",
    "locomotion_min_direction_similarity": "minimum annotated whole-crop net-direction coherence",
    "locomotion_max_pair_distance_cm": "maximum annotated observed pair distance; next float32 bound for strict comparison",
    "locomotion_min_net_displacement_cm": "minimum annotated observed first-to-last member displacement; guards low-speed noise",
    "dispersal_min_outward_speed_cm_s": "p5 positive radial speeds of annotated departing IDs relative to their initial component anchor; 31/41 clips with an initial component",
    "dispersal_pause_max_s": "p95 inactive interior gaps between observed outward supports in usable Dispersal clips",
}

POSTURE_CANONICALLY_ENABLED = False

SOURCE_ARTIFACT_SHA256 = {
    "outputs/heuristics_v18_retune_20261018/audit_full_v2/statistics.json": "4d5a9633bf2de784916661d32fd2efa2e5ea816d8e3df8b479be2aec6dc06617",
    "outputs/heuristics_v18_retune_20261018/social_refined_v1/candidate_comparison.json": "a2e56eef3b1c3f33f84f9c54dcede3b88cd9c105e57b47eb77cb2c1b4b4dc2c7",
    "outputs/heuristics_v18_retune_20261018/group_detector_p95_a2_v1/summary.json": "9b22de55ffe3e21d9c7b828abfeae77b7ff6b4b977dc0a84317a73587f8cc541",
    "outputs/heuristics_v18_retune_20261018/dispersal_motion_fit.json": "3843f81925568028a34a63066e629ae84dba21b5f80beca10055363dd2b46ad1",
}
