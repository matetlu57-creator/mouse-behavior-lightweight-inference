"""Models, canonical labels, and heuristic parameters for exported clips."""

from __future__ import annotations

import re
import weakref
from dataclasses import asdict, dataclass
from typing import Any, Callable
import numpy as np
from mouse_behavior.preprocessing.scale_resolution import LEGACY_REFERENCE_BODY_CM

RETAINED_BEHAVIORS = (
    "Approach",
    "Attack",
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "Avoiding",
    "Chasing",
    "Dispersal",
    # "Following",
    "Group locomotion",
    "Huddling",
    "Isolation",
    "Locomotion",
    "Snout-head_contact",
    "Snout-rear_contact",
    "Social clustering",
    "Static",
    "Together",
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    # "Walking",
)

LABEL_TO_CANONICAL = {
    "Approach": "approach",
    "Attack": "attack",
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "Avoiding": "avoidance",
    "Chasing": "chase",
    "Dispersal": "dispersal",
    # "Following": "following",
    "Group locomotion": "group_locomotion",
    "Huddling": "huddle",
    "Isolation": "isolation",
    "Locomotion": "locomotion",
    "Snout-head_contact": "nose_head_contact",
    "Snout-rear_contact": "nose_tail_contact",
    "Social clustering": "social_clustering",
    "Static": "stationary",
    "Together": "together",
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    # "Walking": "walking",
}

CANONICAL_BEHAVIORS = tuple(LABEL_TO_CANONICAL.values())
INDIVIDUAL_BEHAVIORS = ("locomotion", "stationary", "grooming", "rearing")
PAIR_BEHAVIORS = (
    "approach",
    "together",
    "chase",
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # "avoidance",
    "attack",
    "nose_head_contact",
    "nose_tail_contact",
    # "following",
)
GROUP_BEHAVIORS = (
    "huddle",
    "social_clustering",
    "dispersal",
    "group_locomotion",
    "isolation",
)
_PART_SUFFIX = re.compile(r"_part\d+(?=\.mp4$)", re.IGNORECASE)
_COLLECTIVE_WINDOW_CACHE: dict[
    tuple[Any, ...], tuple[weakref.ReferenceType[Any], tuple[Any, ...]]
] = {}


def _remove_collective_window_cache_entry(
    cache_key: tuple[Any, ...], owner_ref: weakref.ReferenceType[Any]
) -> None:
    """Release cached windows when their owning clip features are collected."""

    cached = _COLLECTIVE_WINDOW_CACHE.get(cache_key)
    if cached is not None and cached[0] is owner_ref:
        _COLLECTIVE_WINDOW_CACHE.pop(cache_key, None)


def _cached_collective_windows(
    cache_key: tuple[Any, ...],
    owner: Any,
    compute: Callable[[], tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Reuse windows only while the exact feature object that owns them lives.

    ``id(owner)`` is part of the cache key, but Python may recycle object IDs
    after a clip is released. The weak reference prevents a later clip from
    receiving a previous clip's collective windows when that happens.
    """

    try:
        owner_ref = weakref.ref(
            owner,
            lambda reference: _remove_collective_window_cache_entry(cache_key, reference),
        )
    except TypeError:
        return compute()

    cached = _COLLECTIVE_WINDOW_CACHE.get(cache_key)
    if cached is not None:
        cached_owner, windows = cached
        if cached_owner() is owner:
            return windows
        _COLLECTIVE_WINDOW_CACHE.pop(cache_key, None)

    windows = compute()
    _COLLECTIVE_WINDOW_CACHE[cache_key] = (owner_ref, windows)
    return windows


@dataclass(frozen=True)
class SampleRecord:
    """A manifest row for one exported annotation clip."""

    sample_id: str
    behavior: str
    canonical_behavior: str
    sample_dir: str
    annotation_path: str
    metadata_path: str
    tracks_path: str
    clip_path: str
    source_video: str
    recording_session: str
    fps: float
    frame_count: int
    mouse_ids: tuple[int, ...]
    confidence: str
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float

    def as_row(self, split: str | None = None) -> dict[str, Any]:
        row = asdict(self)
        row["mouse_ids"] = ",".join(str(value) for value in self.mouse_ids)
        if split is not None:
            row["split"] = split
        return row


@dataclass
class ExportedFeatures:
    """Scale-normalized features for one clip.

    Pair arrays are retained only for the current sample.  The evaluator
    processes samples sequentially so a long multi-animal clip cannot make a
    full-dataset pair tensor resident in memory.
    """

    fps: float
    track_ids: np.ndarray
    valid: np.ndarray
    centers_cm: np.ndarray
    speed_cm_s: np.ndarray
    velocity_cm_s: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_distance_cm: np.ndarray
    pair_closing_speed_cm_s: np.ndarray
    pair_direction_similarity: np.ndarray
    pair_nose_head_cm: np.ndarray
    pair_nose_tail_cm: np.ndarray
    nearest_distance_cm: np.ndarray
    mean_nearest_distance_cm: np.ndarray
    cm_per_pixel: float
    track_selection_truncated: bool
    selected_track_count: int
    mouse_width_cm: float = 2.4
    # Centimetre-scale era marker.  Artefacts serialized before arena
    # calibration defaulted to the historical 8 cm assumption; new builds
    # always store the resolved reference body length.
    reference_body_cm: float = LEGACY_REFERENCE_BODY_CM
    # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Approach 度量=接近鼠鼻头到被接近鼠最近关键点；Huddling 边含鼻-鼻距离。
    pair_nose_nearest_left_to_right_cm: np.ndarray | None = None
    pair_nose_nearest_right_to_left_cm: np.ndarray | None = None
    pair_nose_distance_cm: np.ndarray | None = None
    motion_valid: np.ndarray | None = None
    detection_confidence: np.ndarray | None = None
    keypoint_coverage: np.ndarray | None = None
    track_presence_fraction: np.ndarray | None = None
    pair_box_iou: np.ndarray | None = None
    pair_box_gap_cm: np.ndarray | None = None
    heading_unit: np.ndarray | None = None
    pair_nose_head_left_to_right_cm: np.ndarray | None = None
    pair_nose_head_right_to_left_cm: np.ndarray | None = None
    pair_nose_tail_left_to_right_cm: np.ndarray | None = None
    pair_nose_tail_right_to_left_cm: np.ndarray | None = None
    nose_speed_cm_s: np.ndarray | None = None
    nose_acceleration_cm_s2: np.ndarray | None = None
    # Separate group-only geometry. Existing individual/social features keep
    # using centers_cm and the original global scale above.
    collective_centers_cm: np.ndarray | None = None
    collective_valid: np.ndarray | None = None
    collective_velocity_cm_s: np.ndarray | None = None
    collective_speed_cm_s: np.ndarray | None = None
    collective_keypoints_cm: np.ndarray | None = None
    collective_boxes_cm: np.ndarray | None = None
    body_length_ref_cm: np.ndarray | None = None
    body_width_ref_cm: np.ndarray | None = None
    body_area_ref_cm2: np.ndarray | None = None


@dataclass(frozen=True)
class HeuristicParameters:
    """Tunable parameters for the exported-track calibration adapter."""

    stationary_max_speed_cm_s: float = 13.375
    stationary_transient_max_duration_s: float = 0.2
    stationary_transient_max_displacement_cm: float = 1.3375
    stationary_max_center_radius_cm: float = 1.3375
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    # walking_min_speed_cm_s: float = 10.0
    # walking_max_speed_cm_s: float = 85.0
    locomotion_stop_gap_s: float = 0.5
    # Retained: high-speed locomotion floor used elsewhere (score normalization/external scripts).
    # [STEP6 2026-10-05] Chasing: chased 速度门改用 chase_min_speed_cm_s；running_min_speed_cm_s 保留给别处使用。
    running_min_speed_cm_s: float = 113.6875
    # running_min_directionality: float = 0.55
    # running_min_duration_s: float = 0.5
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地
    together_max_distance_cm: float = 8.025
    together_min_distance_cm: float = 4.0125
    # [STEP6 2026-10-05] Together: 旧低速度语义注释废弃（速度门已从掩码删除）：
    # Together is a low-motion pair state; brief movement is allowed, but
    # sustained movement above the Static boundary is not.
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Together 删除速度门
    # [STEP6 2026-10-05] Together: 不再参与掩码；字段必须保留——旧 JSON 模型/config_seed/现有测试
    # 仍按字段名传参或断言取值，删除会引发 TypeError/AttributeError（验收禁止）。
    together_max_individual_speed_cm_s: float = 16.0
    together_max_combined_speed_cm_s: float = 28.0
    pair_max_distance_cm: float = 6.6875
    # [STEP6 2026-10-05] Approach: 旧 center-distance drop 阈值不再参与掩码（旧掩码注释保留），字段保留兼容旧 JSON 模型。
    approach_min_distance_drop_cm: float = 1.5
    approach_min_closing_speed_cm_s: float = 2.675
    # [STEP6 2026-10-05] Approach: bout 级起始距离下限——仅当 run 首帧 nose_nearest ≥ 该值才计分。
    approach_min_start_distance_cm: float = 6.6875
    # [STEP6 2026-10-05] Approach: yaml 镜像键（social.approach_min_end_distance_cm），与 approach_terminal_distance_cm 同义。
    approach_min_end_distance_cm: float = 2.675
    # [STEP6 2026-10-05] Approach: 语义改为接触截断下限——nose_nearest 必须 > 该值才视为仍在接近（17→2 cm）。
    approach_terminal_distance_cm: float = 2.675
    contact_distance_cm: float = 2.675
    # Nose-contact labels require at least this much cumulative observed
    # contact time for the same visual-ID pair; brief repeated touches below
    # the threshold do not become a standalone contact behavior.
    contact_min_cumulative_seconds: float = 0.5
    attack_min_speed_cm_s: float = 10.7
    approach_min_actor_speed_cm_s: float = 4.0125
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：轻度攻击=攻击下限（鼻加速度 OR 门）
    attack_mild_nose_accel_cm_s2: float = 66.875
    attack_vs_approach_multiplier: float = 1.5
    attack_endpoint_distance_cm: float = 16.05
    attack_reacquisition_max_gap_s: float = 5.0
    # Optional train-calibrated evidence from the contact keypoint. Zero
    # multiplier disables the extra attack ranking path for older profiles.
    attack_contact_nose_min_speed_cm_s: float = 0.0
    attack_contact_nose_min_acceleration_cm_s2: float = 0.0
    attack_contact_body_min_speed_cm_s: float = 0.0
    attack_contact_score_multiplier: float = 0.0
    attack_contact_post_window_s: float = 0.2
    # A quiet nearby pair may walk briefly without completing a continuous
    # per-frame Together bout. These optional limits enable a clip summary.
    together_initial_max_distance_cm: float = 0.0
    together_median_max_distance_cm: float = 0.0
    # [STEP6 2026-10-05] Together: recovery 块 mean_speed 条件删除，该字段不再参与判定；
    # 字段保留因旧 JSON 模型（behavior_rules_v2）仍按字段名传入。
    together_mean_max_speed_cm_s: float = 0.0
    together_max_distance_std_cm: float = 0.0
    together_min_observed_seconds: float = 1.0
    chase_min_direction_similarity: float = 0.707
    chase_min_duration_s: float = 1.0
    chase_min_speed_cm_s: float = 26.75
    chase_fill_gap_s: float = 0.5
    # Chase keeps its own distance ceiling now that Following is suspended.
    chase_max_distance_cm: float = 16.05
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留待恢复
    # avoiding_min_distance_increase_cm: float = 3.0
    # avoiding_min_duration_s: float = 0.5
    # following_min_distance_cm: float = 5.0
    # following_max_distance_cm: float = 30.0
    # following_min_direction_similarity: float = 0.70
    # following_min_duration_s: float = 3.0
    pair_no_contact_seconds: float = 0.5
    huddle_distance_cm: float = 8.025
    huddle_nose_distance_cm: float = 4.0125
    huddle_width_multiplier: float = 2.0
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Huddling 删除全部速度门
    # [STEP6 2026-10-05] Huddling: 速度门删除（不再参与边掩码/onset 判定）；字段必须保留——旧 JSON 模型、
    # calibrated_rule_ranker 与现有测试仍按字段名引用，删除会引发 TypeError/AttributeError（验收禁止）。
    huddle_max_mean_speed_cm_s: float = 10.0
    # Stable tracked IDs can be reacquired after a longer detector dropout.
    # Bridge only when the same members reappear in the same group lineage.
    # [STEP6 2026-10-05] Huddling: ≥5 s 时长门按“保持”经 yaml（group.huddle_min_duration_seconds: 5.0）落地；
    # 数据类默认维持 1.0，避免改变现状并防止既有短样本测试出现 KeyError 级失败。
    huddle_min_duration_s: float = 5.0
    huddle_max_gap_s: float = 5.0
    huddle_min_member_support_fraction: float = 0.25
    huddle_onset_min_low_motion_fraction: float = 0.8
    # Emit persistent, exact visual-ID cores rather than episode-wide unions.
    group_core_candidate_mode: float = 0.0
    # Optional group-only locomotion estimate over a positional window.
    # This rejects posture/keypoint jitter without changing individual rules.
    group_motion_window_s: float = 0.0
    # A single mouse is isolated only when its nearest visible companion is
    # beyond this threshold.
    isolation_distance_cm: float = 26.75
    # Count observed isolation within an episode; a short loss of the same ID
    # may connect its evidence without adding unobserved time.
    isolation_min_duration_s: float = 10.0
    isolation_max_gap_s: float = 0.2
    clustering_max_distance_cm: float = 40.125
    clustering_initial_max_distance_cm: float = 32.1
    clustering_min_nearest_neighbor_drop_cm: float = 2.675
    stationary_min_duration_s: float = 1.0
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    # walking_min_duration_s: float = 1.0
    locomotion_min_duration_s: float = 0.5
    pair_min_duration_s: float = 0.1
    together_min_duration_s: float = 1.0
    group_min_duration_s: float = 0.3
    approach_window_s: float = 0.3
    clustering_window_s: float = 2.0
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Social clustering 不规定具体时长
    clustering_min_duration_s: float = 0.0
    clustering_max_gap_s: float = 0.25
    # A qualifying nearest-neighbour decrease is explicit formation evidence;
    # it should outrank a simultaneous quiet huddle of the same IDs.
    clustering_formation_score_bonus: float = 1.0
    # Social clustering describes the formation/migration phase, so require
    # some group motion while Huddling retains its low-motion gate.
    clustering_min_mean_speed_cm_s: float = 4.0125
    clustering_min_moving_member_fraction: float = 0.5
    # A fleeting visitor is not a persistent member of a forming group.
    clustering_min_member_support_fraction: float = 0.0
    clustering_directional_walk_min_seconds: float = 0.0
    clustering_directional_min_distance_drop_cm: float = 0.0
    clustering_directional_min_cosine: float = 0.6
    # Optional train-calibrated path for clips that begin with a close group
    # but retain measurable collective displacement. Zero disables it.
    clustering_motion_max_distance_cm: float = 0.0
    clustering_motion_min_displacement_cm: float = 8.025
    clustering_motion_loose_pair_distance_cm: float = 12.70625
    # Optional formation rescue for a compact-looking group that is visibly
    # moving toward a common location. Zero disables this path.
    clustering_formation_rescue_min_geometry_support: float = 0.0
    # Opt-in shared CLU/HUD handcrafted feature path. Zero preserves the
    # historical detector exactly; all other behavior scores are untouched.
    collective_feature_rules_enabled: float = 0.0
    collective_proximity_distance_bw: float = 3.0
    collective_contact_score_threshold: float = 0.30
    collective_active_speed_cm_s: float = 6.6875
    collective_low_motion_speed_cm_s: float = 13.375
    # Discard short-lived/low-confidence detector fragments before forming
    # Huddling and Social-clustering graphs. Other behavior paths keep all IDs.
    collective_track_min_lifetime_s: float = 1.0
    collective_track_min_presence_ratio: float = 0.05
    collective_track_min_median_detection_confidence: float = 0.25
    collective_track_min_median_keypoint_coverage: float = 0.10
    collective_track_max_spatial_duplicate_score: float = 0.85
    social_nnd_drop_bw_min: float = 0.20
    social_nnd_slope_bw_s_max: float = -0.02
    social_radius_drop_bw_min: float = 0.20
    social_radius_slope_bw_s_max: float = -0.02
    social_inward_fraction_min: float = 0.45
    social_active_fraction_min: float = 0.35
    # Fraction of candidate members that show sustained inward movement above
    # the collective-active speed boundary during a formation window.
    social_active_inward_member_fraction_min: float = 0.20
    social_edge_gain_rate_min: float = 0.02
    social_density_slope_min: float = 0.005
    social_soft_score_min: float = 0.60
    huddle_nnd_bw_max: float = 2.50
    huddle_radius_bw_max: float = 3.00
    huddle_bbox_iom_min: float = 0.14
    collective_huddle_min_group_visible_fraction: float = 0.30
    collective_huddle_min_member_support_fraction: float = 0.40
    collective_huddle_recovery_near_group_max_bw: float = 3.0
    huddle_nnd_slope_abs_max: float = 0.15
    huddle_radius_slope_abs_max: float = 0.15
    huddle_low_motion_fraction_min: float = 0.60
    huddle_contact_density_min: float = 0.25
    huddle_proximity_density_min: float = 0.60
    huddle_graph_stability_min: float = 0.50
    huddle_soft_score_min: float = 0.60
    huddle_attack_speed_p90_max_cm_s: float = 60.1875
    huddle_attack_relative_motion_max_cm_s: float = 40.125
    group_locomotion_max_distance_cm: float = 33.4375
    group_locomotion_min_direction_similarity: float = 0.70
    group_locomotion_min_moving_fraction: float = 0.60
    group_locomotion_min_duration_s: float = 3.0
    dispersal_start_distance_cm: float = 20.0625
    dispersal_min_distance_increase_cm: float = 6.6875
    dispersal_min_duration_s: float = 5.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}
