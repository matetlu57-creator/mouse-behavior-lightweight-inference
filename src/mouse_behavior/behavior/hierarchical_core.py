"""Composable, track-ID-preserving heuristic behavior rules.

This module is the rule core, not a CSV/video adapter. Callers provide windowed
features in centimeters, centimeters per second, and seconds. It deliberately
retains individual, social, and collective decisions together; only
``primary_by_mouse`` is collapsed to the highest active layer for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_ear", "right_ear", "base_of_neck", "left_hip", "right_hip", "base_of_tail"
)
KP = {name: index for index, name in enumerate(KEYPOINT_NAMES)}


class Level(str, Enum):
    INDIVIDUAL = "individual"
    SOCIAL = "social"
    COLLECTIVE = "collective"


class MotionLabel(str, Enum):
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    LOCOMOTION = "Locomotion"
    STATIC = "Static"
    GROOMING = "Grooming"
    REARING = "Rearing"
    UNKNOWN = "Individual-Unknown"


class SocialLabel(str, Enum):
    APPROACH = "Approach"
    TOGETHER = "Together"
    CHASING = "Chasing"
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留代码待恢复
    # AVOIDING = "Avoiding"
    ATTACK = "Attack"
    SNOUT_HEAD = "Snout-head contact"
    SNOUT_REAR = "Snout-rear contact"
    # FOLLOWING = "Following"
    UNKNOWN = "Social-Unknown"


class CollectiveLabel(str, Enum):
    HUDDLING = "Huddling"
    SOCIAL_CLUSTERING = "Social clustering"
    DISPERSAL = "Dispersal"
    GROUP_LOCOMOTION = "Group locomotion"
    ISOLATION = "Isolation"
    UNKNOWN = "Collective-Unknown"


class EventPhase(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    GRACE = "grace"


# Priorities resolve conflicts within a layer only.
SOCIAL_PRIORITY: Mapping[str, int] = {
    SocialLabel.ATTACK.value: 100,
    SocialLabel.SNOUT_HEAD.value: 90,
    SocialLabel.SNOUT_REAR.value: 90,
    SocialLabel.CHASING.value: 80,
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留代码待恢复
    # SocialLabel.AVOIDING.value: 70,
    # SocialLabel.FOLLOWING.value: 60,
    SocialLabel.APPROACH.value: 50,
    SocialLabel.TOGETHER.value: 40,
    SocialLabel.UNKNOWN.value: 1,
}
COLLECTIVE_PRIORITY: Mapping[str, int] = {
    CollectiveLabel.HUDDLING.value: 100,
    CollectiveLabel.SOCIAL_CLUSTERING.value: 90,
    CollectiveLabel.DISPERSAL.value: 80,
    CollectiveLabel.GROUP_LOCOMOTION.value: 70,
    CollectiveLabel.ISOLATION.value: 60,
    CollectiveLabel.UNKNOWN.value: 1,
}


def _finite_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _unit_interval(name: str, value: float) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")


@dataclass(frozen=True)
class PoseObservation:
    mouse_id: int
    frame_index: int
    timestamp_s: float
    keypoints_xy_px: np.ndarray
    keypoint_conf: Optional[np.ndarray] = None

    def validate(self) -> None:
        points = np.asarray(self.keypoints_xy_px)
        if points.shape != (7, 2):
            raise ValueError(f"keypoints_xy_px must be (7, 2), got {points.shape}")
        if self.keypoint_conf is not None and np.asarray(self.keypoint_conf).shape != (7,):
            raise ValueError("keypoint_conf must be (7,)")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")


@dataclass(frozen=True)
class PoseQuality:
    valid: np.ndarray
    completeness: float


@dataclass(frozen=True)
class IndividualGeometry:
    mouse_id: int
    center_cm: np.ndarray
    head_center_cm: np.ndarray
    rear_center_cm: np.ndarray
    body_axis_unit: np.ndarray
    head_axis_unit: np.ndarray
    intra_dist_cm: np.ndarray
    quality: PoseQuality
    mouse_width_cm: float = float("nan")


@dataclass(frozen=True)
class PairGeometry:
    mouse_i: int
    mouse_j: int
    center_distance_cm: float
    inter_dist_cm: np.ndarray
    inter_valid: np.ndarray
    coverage: float
    facing_i_to_j_deg: float
    facing_j_to_i_deg: float


@dataclass(frozen=True)
class BehaviorDecision:
    level: Level
    label: str
    members: Tuple[int, ...]
    confidence: float = 0.0
    priority: int = 0
    actor: Optional[int] = None
    target: Optional[int] = None
    evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("members must be non-empty and unique")
        _unit_interval("confidence", self.confidence)
        if self.actor is not None and self.actor not in self.members:
            raise ValueError("actor must be one of members")
        if self.target is not None and self.target not in self.members:
            raise ValueError("target must be one of members")


@dataclass(frozen=True)
class BehaviorHierarchySnapshot:
    individual_by_mouse: Mapping[int, BehaviorDecision]
    social_events: Tuple[BehaviorDecision, ...]
    collective_events: Tuple[BehaviorDecision, ...]
    primary_by_mouse: Mapping[int, BehaviorDecision]


@dataclass(frozen=True)
class MotionThresholds:
    # Retained: high-speed locomotion floor used by the Chasing gate.
    running_min_speed_cm_s: float = 53.5
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）
    moving_min_speed_cm_s: float = 0.66875
    static_max_speed_cm_s: float = 0.66875
    locomotion_min_duration_s: float = 0.5
    static_min_duration_s: float = 1.0
    static_max_center_radius_cm: float = 1.3375
    static_min_projected_body_length_ratio: float = 0.55
    static_max_transient_duration_s: float = 0.2
    static_max_transient_displacement_cm: float = 1.3375
    # walking_min_speed_cm_s: float = 0.5
    # walking_max_speed_cm_s: float = 40.0
    # running_min_duration_s: float = 0.5
    # walking_min_duration_s: float = 1.0
    # running_min_directionality: float = 0.65
    # walking_min_active_fraction: float = 0.40

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            _finite_nonnegative(name, float(value))
        _unit_interval(
            "static_min_projected_body_length_ratio",
            self.static_min_projected_body_length_ratio,
        )
        if self.static_max_speed_cm_s > self.moving_min_speed_cm_s:
            raise ValueError("Static and Locomotion speed intervals overlap")
        if self.moving_min_speed_cm_s > self.running_min_speed_cm_s:
            raise ValueError("moving min speed must not exceed the chasing speed floor")


@dataclass(frozen=True)
class SocialThresholds:
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地
    together_max_distance_cm: float = 8.025
    together_min_distance_cm: float = 4.0125
    approach_final_distance_cm: float = 6.6875
    approach_min_end_distance_cm: float = 2.675
    chasing_max_distance_cm: float = 16.05
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）：合并后 Chasing 用显式速度下限替代原 Running 标签隐含的高速语义。
    chasing_min_speed_cm_s: float = 53.5
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留代码待恢复
    # following_min_distance_cm: float = 5.0
    # following_max_distance_cm: float = 30.0
    min_contact_duration_s: float = 0.5
    chasing_min_duration_s: float = 1.0
    chasing_min_speed_cm_s: float = 26.75
    # following_min_duration_s: float = 3.0
    together_min_duration_s: float = 1.0
    approach_min_duration_s: float = 0.0
    # avoiding_min_duration_s: float = 0.0
    contact_distance_cm: float = 2.675
    min_heading_similarity: float = 0.707
    approach_min_closing_speed_cm_s: float = 2.675
    # avoiding_min_speed_cm_s: float = 10.0
    # avoiding_min_turn_deg: float = 50.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            _finite_nonnegative(name, float(value))
        # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留代码待恢复
        # if self.following_min_distance_cm > self.following_max_distance_cm:
        #     raise ValueError("following distance interval is inverted")
        _unit_interval("min_heading_similarity", self.min_heading_similarity)


@dataclass(frozen=True)
class AttackThresholds:
    """Train-calibratable conflict score; defaults are provisional, not fitted."""

    min_conflict_score: float = 0.60
    rapid_closing_weight: float = 0.15
    pounce_or_bite_weight: float = 0.25
    entanglement_weight: float = 0.20
    local_motion_weight: float = 0.15
    observation_loss_weight: float = 0.10
    recovery_escape_weight: float = 0.15

    def validate(self) -> None:
        _unit_interval("min_conflict_score", self.min_conflict_score)
        weights = (
            self.rapid_closing_weight,
            self.pounce_or_bite_weight,
            self.entanglement_weight,
            self.local_motion_weight,
            self.observation_loss_weight,
            self.recovery_escape_weight,
        )
        if any(not np.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("attack evidence weights must be finite, non-negative, and non-zero")


@dataclass(frozen=True)
class CollectiveThresholds:
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地
    huddling_min_duration_s: float = 5.0
    social_clustering_min_duration_s: float = 0.0
    dispersal_min_duration_s: float = 5.0
    group_locomotion_min_duration_s: float = 3.0
    isolation_min_duration_s: float = 10.0
    huddle_max_neighbor_width_ratio: float = 1.0
    # A huddle is a rest state. For a three-member core this allows no more
    # than one quarter of members to exceed the individual Static speed band.
    huddle_max_high_motion_fraction: float = 0.20
    huddle_min_continuity_fraction: float = 0.70
    recovery_min_near_group_fraction: float = 0.50
    clustering_max_nnd_slope_cm_s: float = -0.2675
    clustering_max_radius_slope_cm_s: float = -0.2675
    clustering_min_moving_fraction: float = 0.50
    clustering_min_inward_fraction: float = 0.50
    dispersal_min_nnd_slope_cm_s: float = 0.2675
    dispersal_min_radius_slope_cm_s: float = 0.2675
    dispersal_min_moving_fraction: float = 0.30
    group_locomotion_max_neighbor_distance_cm: float = 33.4375
    group_locomotion_min_coherence: float = 0.70
    group_locomotion_min_moving_fraction: float = 0.60
    group_locomotion_max_spacing_cv: float = 0.30
    # Isolation uses the nearest currently visible companion.
    isolation_min_distance_cm: float = 26.75

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "huddle_max_neighbor_width_ratio",
            "group_locomotion_max_neighbor_distance_cm",
            "isolation_min_distance_cm",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in (
            "huddle_max_high_motion_fraction", "huddle_min_continuity_fraction",
            "recovery_min_near_group_fraction", "clustering_min_moving_fraction",
            "clustering_min_inward_fraction", "dispersal_min_moving_fraction",
            "group_locomotion_min_coherence", "group_locomotion_min_moving_fraction",
            "group_locomotion_max_spacing_cv",
        ):
            _unit_interval(name, getattr(self, name))


@dataclass(frozen=True)
class HeuristicThresholds:
    motion: MotionThresholds = field(default_factory=MotionThresholds)
    social: SocialThresholds = field(default_factory=SocialThresholds)
    attack: AttackThresholds = field(default_factory=AttackThresholds)
    collective: CollectiveThresholds = field(default_factory=CollectiveThresholds)

    def validate(self) -> None:
        self.motion.validate()
        self.social.validate()
        self.attack.validate()
        self.collective.validate()


def thresholds_from_config(config: Mapping[str, object]) -> HeuristicThresholds:
    """Load only declared core thresholds; reject typos instead of ignoring them."""
    root = config.get("hierarchical_heuristic_core", config)
    if not isinstance(root, Mapping):
        raise ValueError("hierarchical_heuristic_core config must be a mapping")
    sections = {
        "motion": MotionThresholds,
        "social": SocialThresholds,
        "attack": AttackThresholds,
        "collective": CollectiveThresholds,
    }
    unknown_sections = set(root) - set(sections)
    if unknown_sections:
        raise ValueError(f"unknown heuristic threshold sections: {sorted(unknown_sections)}")
    values = {}
    for name, cls in sections.items():
        section = root.get(name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} threshold section must be a mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"unknown {name} thresholds: {sorted(unknown)}")
        values[name] = cls(**section)
    result = HeuristicThresholds(**values)
    result.validate()
    return result


@dataclass(frozen=True)
class MotionWindowStats:
    """Torso translation statistics; head turns alone are compatible with Static.

    ``mean_speed_cm_s`` and ``center_radius_cm`` must describe the body center,
    never nose/ear motion. Grooming and rearing require their own evidence.
    """
    mouse_id: int
    duration_s: float
    mean_speed_cm_s: float
    path_length_cm: float
    net_displacement_cm: float
    active_fraction: float
    center_radius_cm: float
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion（用户决策/文档）：理毛/站立攀爬证据门（检测器默认关闭，Step 5-A 标定后启用）。
    rearing_evidence: bool = False
    grooming_evidence: bool = False
    # Optional independently measured bout summary. When absent, the strict
    # speed-only Static rule remains in force.
    transient_motion_duration_s: float | None = None
    transient_displacement_cm: float | None = None
    # Minimum 2D nose-to-tail projected length divided by calibrated reference
    # body length over this window. None keeps older summary callers compatible.
    min_projected_body_length_ratio: float | None = None

    @property
    def directionality(self) -> float:
        if not np.isfinite(self.path_length_cm) or self.path_length_cm <= 1e-12:
            return 0.0
        return float(np.clip(self.net_displacement_cm / self.path_length_cm, 0.0, 1.0))


@dataclass(frozen=True)
class PairWindowStats:
    """Window features for an ordered source/target pair; units are cm and s."""

    actor: int
    target: int
    duration_s: float
    center_distance_cm: float
    distance_slope_cm_s: float
    heading_similarity: float
    actor_toward_target_cos: float
    target_escape_cos: float
    actor_speed_cm_s: float
    target_speed_cm_s: float
    actor_motion: str
    target_motion: str
    pair_contact: bool
    nose_to_target_head_cm: float = float("nan")
    source_head_to_target_rear_cm: float = float("nan")
    snout_head_contact_duration_s: float = 0.0
    snout_rear_contact_duration_s: float = 0.0
    approach_recent: bool = False
    target_turn_deg: float = 0.0


@dataclass(frozen=True)
class ConflictWindowStats:
    actor: int
    target: int
    duration_s: float
    body_contact: bool
    rapid_closing: float
    pounce_or_bite: float
    entanglement: float
    local_motion_energy: float
    occlusion_or_observation_loss: float

    recovery_escape: float

    def validate(self) -> None:
        _finite_nonnegative("duration_s", self.duration_s)
        for name in (
            "rapid_closing", "pounce_or_bite", "entanglement", "local_motion_energy",
            "occlusion_or_observation_loss", "recovery_escape",
        ):
            _unit_interval(name, getattr(self, name))


@dataclass(frozen=True)
class GroupWindowStats:
    members: Tuple[int, ...]
    duration_s: float
    median_neighbor_distance_cm: float
    median_neighbor_width_ratio: float
    nnd_slope_cm_s: float
    group_radius_slope_cm_s: float
    moving_fraction: float
    inward_motion_fraction: float
    high_motion_fraction: float
    direction_coherence: float
    spacing_cv: float
    member_recovery_near_group: float = 0.0
    previous_collective_label: Optional[str] = None
    member_continuity_fraction: float = 1.0

    def validate(self) -> None:
        _finite_nonnegative("duration_s", self.duration_s)
        for name in ("median_neighbor_distance_cm", "median_neighbor_width_ratio", "spacing_cv"):
            _finite_nonnegative(name, getattr(self, name))
        for name in (
            "moving_fraction", "inward_motion_fraction", "high_motion_fraction",
            "direction_coherence", "member_recovery_near_group", "member_continuity_fraction",
        ):
            _unit_interval(name, getattr(self, name))
        if not np.isfinite(self.nnd_slope_cm_s) or not np.isfinite(self.group_radius_slope_cm_s):
            raise ValueError("group trend slopes must be finite")
        if len(set(self.members)) != len(self.members):
            raise ValueError("group members must be unique")


@dataclass(frozen=True)
class IsolationWindowStats:
    mouse_id: int
    duration_s: float
    min_distance_to_any_mouse_cm: float


@dataclass
class HysteresisStateMachine:
    enter_threshold: float
    maintain_threshold: float
    enter_time_s: float
    grace_time_s: float
    phase: EventPhase = EventPhase.IDLE
    phase_since_s: Optional[float] = None

    def __post_init__(self) -> None:
        _unit_interval("enter_threshold", self.enter_threshold)
        _unit_interval("maintain_threshold", self.maintain_threshold)
        if self.maintain_threshold > self.enter_threshold:
            raise ValueError("maintain_threshold must not exceed enter_threshold")
        _finite_nonnegative("enter_time_s", self.enter_time_s)
        _finite_nonnegative("grace_time_s", self.grace_time_s)

    def update(self, evidence: float, t_s: float) -> EventPhase:
        _unit_interval("evidence", evidence)
        if not np.isfinite(t_s):
            raise ValueError("t_s must be finite")
        if self.phase_since_s is None:
            self.phase_since_s = t_s
        elapsed = max(0.0, t_s - self.phase_since_s)
        if self.phase == EventPhase.IDLE:
            if evidence >= self.enter_threshold:
                self.phase, self.phase_since_s = EventPhase.CANDIDATE, t_s
        elif self.phase == EventPhase.CANDIDATE:
            if evidence < self.enter_threshold:
                self.phase, self.phase_since_s = EventPhase.IDLE, t_s
            elif elapsed >= self.enter_time_s:
                self.phase, self.phase_since_s = EventPhase.ACTIVE, t_s
        elif self.phase == EventPhase.ACTIVE:
            if evidence < self.maintain_threshold:
                self.phase, self.phase_since_s = EventPhase.GRACE, t_s
        elif self.phase == EventPhase.GRACE:
            if evidence >= self.maintain_threshold:
                self.phase, self.phase_since_s = EventPhase.ACTIVE, t_s
            elif elapsed >= self.grace_time_s:
                self.phase, self.phase_since_s = EventPhase.IDLE, t_s
        return self.phase


def valid_keypoints(obs: PoseObservation, conf_thr: float = 0.25) -> PoseQuality:
    obs.validate()
    if not np.isfinite(conf_thr):
        raise ValueError("conf_thr must be finite")
    points = np.asarray(obs.keypoints_xy_px, dtype=float)
    valid = np.isfinite(points).all(axis=1)
    if obs.keypoint_conf is not None:
        confidence = np.asarray(obs.keypoint_conf, dtype=float)
        valid &= np.isfinite(confidence) & (confidence >= conf_thr)
    return PoseQuality(valid=valid, completeness=float(valid.mean()))


def _nan_point() -> np.ndarray:
    return np.full(2, np.nan, dtype=float)


def _median_valid(points: np.ndarray, valid: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    available = [index for index in indices if valid[index]]
    return np.median(points[available], axis=0) if available else _nan_point()


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    if vector.shape != (2,) or not np.isfinite(vector).all():
        return _nan_point()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else _nan_point()


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def compute_individual_geometry(
    obs: PoseObservation, px_per_cm: float, conf_thr: float = 0.25
) -> IndividualGeometry:
    if not np.isfinite(px_per_cm) or px_per_cm <= 0:
        raise ValueError("px_per_cm must be finite and > 0")
    quality = valid_keypoints(obs, conf_thr)
    points = np.asarray(obs.keypoints_xy_px, dtype=float) / px_per_cm
    body = (KP["base_of_neck"], KP["left_hip"], KP["right_hip"], KP["base_of_tail"])
    center = _median_valid(points, quality.valid, body)
    if not np.isfinite(center).all():
        center = _median_valid(points, quality.valid, tuple(range(7)))
    head = _median_valid(points, quality.valid, (KP["nose"], KP["left_ear"], KP["right_ear"]))
    rear = _median_valid(points, quality.valid, (KP["left_hip"], KP["right_hip"], KP["base_of_tail"]))
    if quality.valid[KP["base_of_neck"]] and quality.valid[KP["base_of_tail"]]:
        body_axis = _unit_vector(points[KP["base_of_neck"]] - points[KP["base_of_tail"]])
    else:
        body_axis = _unit_vector(head - rear)
    head_axis = (
        _unit_vector(points[KP["nose"]] - points[KP["base_of_neck"]])
        if quality.valid[KP["nose"]] and quality.valid[KP["base_of_neck"]]
        else _nan_point()
    )
    intra = np.full((7, 7), np.nan)
    for i in range(7):
        if quality.valid[i]:
            intra[i, i] = 0.0
            for j in range(i + 1, 7):
                if quality.valid[j]:
                    intra[i, j] = intra[j, i] = np.linalg.norm(points[i] - points[j])
    width = float("nan")
    if quality.valid[KP["left_hip"]] and quality.valid[KP["right_hip"]]:
        width = float(np.linalg.norm(points[KP["left_hip"]] - points[KP["right_hip"]]))
    elif quality.valid[KP["left_ear"]] and quality.valid[KP["right_ear"]]:
        width = float(np.linalg.norm(points[KP["left_ear"]] - points[KP["right_ear"]]))
    return IndividualGeometry(obs.mouse_id, center, head, rear, body_axis, head_axis,
                              intra, quality, width)


def compute_pair_geometry(
    obs_i: PoseObservation, obs_j: PoseObservation,
    geom_i: IndividualGeometry, geom_j: IndividualGeometry, px_per_cm: float,
) -> PairGeometry:
    if obs_i.mouse_id == obs_j.mouse_id:
        raise ValueError("pair requires different mouse IDs")
    if (geom_i.mouse_id, geom_j.mouse_id) != (obs_i.mouse_id, obs_j.mouse_id):
        raise ValueError("geometry IDs must match observation IDs and order")
    if not np.isfinite(px_per_cm) or px_per_cm <= 0:
        raise ValueError("px_per_cm must be finite and > 0")
    distance = float("nan")
    if np.isfinite(geom_i.center_cm).all() and np.isfinite(geom_j.center_cm).all():
        distance = float(np.linalg.norm(geom_i.center_cm - geom_j.center_cm))
    points_i = np.asarray(obs_i.keypoints_xy_px, dtype=float) / px_per_cm
    points_j = np.asarray(obs_j.keypoints_xy_px, dtype=float) / px_per_cm
    mask = geom_i.quality.valid[:, None] & geom_j.quality.valid[None, :]
    inter = np.full((7, 7), np.nan)
    rows, cols = np.where(mask)
    if rows.size:
        inter[rows, cols] = np.linalg.norm(points_i[rows] - points_j[cols], axis=1)
    facing_i = facing_j = float("nan")
    if np.isfinite(distance):
        i_to_j = _unit_vector(geom_j.center_cm - geom_i.center_cm)
        j_to_i = -i_to_j if np.isfinite(i_to_j).all() else _nan_point()
        axis_i = geom_i.head_axis_unit if np.isfinite(geom_i.head_axis_unit).all() else geom_i.body_axis_unit
        axis_j = geom_j.head_axis_unit if np.isfinite(geom_j.head_axis_unit).all() else geom_j.body_axis_unit
        facing_i, facing_j = _angle_deg(axis_i, i_to_j), _angle_deg(axis_j, j_to_i)
    return PairGeometry(obs_i.mouse_id, obs_j.mouse_id, distance, inter, mask,
                        float(mask.mean()), facing_i, facing_j)


def pairwise_center_distances(geometries: Sequence[IndividualGeometry]) -> np.ndarray:
    centers = np.asarray([item.center_cm for item in geometries], dtype=float)
    if centers.size == 0:
        return np.empty((0, 0), dtype=float)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    distances[~np.isfinite(centers).all(axis=1), :] = np.nan
    distances[:, ~np.isfinite(centers).all(axis=1)] = np.nan
    return distances


def connected_components_from_distance(
    distances_cm: np.ndarray, max_edge_cm: float
) -> Tuple[Tuple[int, ...], ...]:
    distances = np.asarray(distances_cm, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances_cm must be square")
    if not np.isfinite(max_edge_cm) or max_edge_cm <= 0:
        raise ValueError("max_edge_cm must be finite and > 0")
    unseen = set(range(len(distances)))
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            neighbors = [other for other in unseen
                         if np.isfinite(distances[index, other]) and distances[index, other] <= max_edge_cm]
            for other in neighbors:
                unseen.remove(other)
                stack.append(other)
        components.append(tuple(sorted(component)))
    return tuple(components)


def velocity_coherence(velocities: np.ndarray, min_speed: float = 1e-9) -> float:
    values = np.asarray(velocities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("velocities must have shape (N,2)")
    norms = np.linalg.norm(values, axis=1)
    good = np.isfinite(values).all(axis=1) & (norms > min_speed)
    if not np.any(good):
        return float("nan")
    return float(np.clip(np.linalg.norm((values[good] / norms[good, None]).mean(axis=0)), 0, 1))


def detect_individual_motion(stats: MotionWindowStats, cfg: MotionThresholds) -> BehaviorDecision:
    """Classify body translation while allowing a stationary mouse to turn its head."""
    cfg.validate()
    _finite_nonnegative("duration_s", stats.duration_s)
    _unit_interval("active_fraction", stats.active_fraction)
    if stats.transient_motion_duration_s is not None:
        _finite_nonnegative("transient_motion_duration_s", stats.transient_motion_duration_s)
    if stats.transient_displacement_cm is not None:
        _finite_nonnegative("transient_displacement_cm", stats.transient_displacement_cm)
    if stats.min_projected_body_length_ratio is not None:
        _unit_interval(
            "min_projected_body_length_ratio",
            stats.min_projected_body_length_ratio,
        )
    speed = stats.mean_speed_cm_s
    if not np.isfinite(speed):
        return BehaviorDecision(Level.INDIVIDUAL, MotionLabel.UNKNOWN.value, (stats.mouse_id,))
    # Summary adapter for v13; the temporal detector owns posture evidence.
    transient_excursion = (
        stats.transient_motion_duration_s is not None
        and stats.transient_displacement_cm is not None
        and 0 < stats.transient_motion_duration_s <= cfg.static_max_transient_duration_s
        and stats.transient_displacement_cm <= cfg.static_max_transient_displacement_cm
    )
    bounded_segment_displacement = (
        stats.transient_motion_duration_s in (None, 0.0)
        and stats.net_displacement_cm <= cfg.static_max_transient_displacement_cm
        and stats.center_radius_cm <= cfg.static_max_center_radius_cm
    )
    locomotion = (stats.duration_s >= cfg.locomotion_min_duration_s
                  and speed > cfg.moving_min_speed_cm_s
                  and not transient_excursion
                  and not bounded_segment_displacement
                  and not stats.rearing_evidence and not stats.grooming_evidence)
    static = (
        stats.duration_s >= cfg.static_min_duration_s
        and (
            (
                (0 <= speed <= cfg.static_max_speed_cm_s or transient_excursion)
                and stats.center_radius_cm <= cfg.static_max_center_radius_cm
            )
            or bounded_segment_displacement
        )
        and not stats.rearing_evidence
        and not stats.grooming_evidence
    )
    if stats.grooming_evidence:
        return BehaviorDecision(Level.INDIVIDUAL, MotionLabel.GROOMING.value, (stats.mouse_id,), 1.0,
                                evidence={"speed_cm_s": speed})
    if stats.rearing_evidence:
        return BehaviorDecision(Level.INDIVIDUAL, MotionLabel.REARING.value, (stats.mouse_id,), 1.0,
                                evidence={"speed_cm_s": speed})
    if locomotion and static:
        raise RuntimeError("individual motion gates overlap; do not resolve by priority")
    static_evidence = {
        "speed_cm_s": speed,
        "center_radius_cm": stats.center_radius_cm,
        "net_displacement_cm": float(stats.net_displacement_cm),
        "transient_motion_duration_s": float(stats.transient_motion_duration_s or 0.0),
        "transient_displacement_cm": float(stats.transient_displacement_cm or 0.0),
    }
    if stats.min_projected_body_length_ratio is not None:
        static_evidence["min_projected_body_length_ratio"] = float(
            stats.min_projected_body_length_ratio
        )
    for matched, label, evidence in (
        (locomotion, MotionLabel.LOCOMOTION, {"speed_cm_s": speed, "active_fraction": stats.active_fraction}),
        (static, MotionLabel.STATIC, static_evidence),
    ):
        if matched:
            return BehaviorDecision(Level.INDIVIDUAL, label.value, (stats.mouse_id,), 1.0, evidence=evidence)
    return BehaviorDecision(Level.INDIVIDUAL, MotionLabel.UNKNOWN.value, (stats.mouse_id,), evidence={"speed_cm_s": speed})


def _social_decision(label: SocialLabel, actor: int, target: int,
                     evidence: Mapping[str, float]) -> BehaviorDecision:
    return BehaviorDecision(Level.SOCIAL, label.value, tuple(sorted((actor, target))), 1.0,
                            SOCIAL_PRIORITY[label.value], actor, target, evidence)


def social_candidates_from_pair(
    stats: PairWindowStats, cfg: SocialThresholds
) -> Tuple[BehaviorDecision, ...]:
    cfg.validate()
    if stats.actor == stats.target:
        raise ValueError("pair members must have different IDs")
    candidates = []
    pair = (stats.actor, stats.target)
    if (stats.pair_contact and np.isfinite(stats.nose_to_target_head_cm)
            and stats.nose_to_target_head_cm <= cfg.contact_distance_cm
            and stats.snout_head_contact_duration_s >= cfg.min_contact_duration_s):
        candidates.append(_social_decision(SocialLabel.SNOUT_HEAD, *pair,
            {"contact_distance_cm": stats.nose_to_target_head_cm,
             "contact_duration_s": stats.snout_head_contact_duration_s}))
    if (stats.pair_contact and np.isfinite(stats.source_head_to_target_rear_cm)
            and stats.source_head_to_target_rear_cm <= cfg.contact_distance_cm
            and stats.snout_rear_contact_duration_s >= cfg.min_contact_duration_s):
        candidates.append(_social_decision(SocialLabel.SNOUT_REAR, *pair,
            {"contact_distance_cm": stats.source_head_to_target_rear_cm,
             "contact_duration_s": stats.snout_rear_contact_duration_s}))
    if (stats.duration_s >= cfg.chasing_min_duration_s
            and stats.center_distance_cm <= cfg.chasing_max_distance_cm
            and not stats.pair_contact
            and stats.actor_motion == MotionLabel.LOCOMOTION.value
            and stats.target_motion == MotionLabel.LOCOMOTION.value
            and stats.actor_speed_cm_s >= cfg.chasing_min_speed_cm_s
            and stats.target_speed_cm_s >= cfg.chasing_min_speed_cm_s
            and stats.heading_similarity >= cfg.min_heading_similarity
            and stats.actor_toward_target_cos > 0 and stats.target_escape_cos > 0):
        candidates.append(_social_decision(SocialLabel.CHASING, *pair,
            {"distance_cm": stats.center_distance_cm,
             "heading_similarity": stats.heading_similarity}))
    # [SUSPENDED-STEP3 2026-10-04] Following/Avoiding 暂停（用户决策），保留代码待恢复
    # if (stats.duration_s >= cfg.avoiding_min_duration_s and stats.approach_recent
    #         and stats.distance_slope_cm_s > 0
    #         and stats.target_speed_cm_s >= cfg.avoiding_min_speed_cm_s
    #         and stats.target_turn_deg >= cfg.avoiding_min_turn_deg):
    #     candidates.append(BehaviorDecision(
    #         Level.SOCIAL, SocialLabel.AVOIDING.value, tuple(sorted(pair)), 1.0,
    #         SOCIAL_PRIORITY[SocialLabel.AVOIDING.value], stats.target, stats.actor,
    #         {"distance_slope_cm_s": stats.distance_slope_cm_s,
    #          "avoider_speed_cm_s": stats.target_speed_cm_s,
    #          "turn_deg": stats.target_turn_deg}))
    # if (stats.duration_s >= cfg.following_min_duration_s
    #         and cfg.following_min_distance_cm <= stats.center_distance_cm <= cfg.following_max_distance_cm
    #         and not stats.pair_contact
    #         and stats.actor_motion == MotionLabel.WALKING.value
    #         and stats.target_motion == MotionLabel.WALKING.value
    #         and stats.heading_similarity >= cfg.min_heading_similarity
    #         and stats.actor_toward_target_cos > 0 and stats.target_escape_cos <= 0):
    #     candidates.append(_social_decision(SocialLabel.FOLLOWING, *pair,
    #         {"distance_cm": stats.center_distance_cm,
    #          "heading_similarity": stats.heading_similarity}))
    if (stats.duration_s >= cfg.approach_min_duration_s
            and stats.actor_motion == MotionLabel.LOCOMOTION.value
            and stats.center_distance_cm <= cfg.approach_final_distance_cm
            and stats.center_distance_cm >= cfg.approach_min_end_distance_cm
            and stats.distance_slope_cm_s < -cfg.approach_min_closing_speed_cm_s
            and stats.actor_toward_target_cos > 0):
        candidates.append(_social_decision(SocialLabel.APPROACH, *pair,
            {"distance_cm": stats.center_distance_cm,
             "distance_slope_cm_s": stats.distance_slope_cm_s}))
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Together 去掉速度门；[3,6] cm 带，接触帧（pair_contact）容忍。
    if (stats.duration_s >= cfg.together_min_duration_s
            and stats.center_distance_cm <= cfg.together_max_distance_cm
            and (stats.center_distance_cm >= cfg.together_min_distance_cm
                 or stats.pair_contact)):
        candidates.append(BehaviorDecision(Level.SOCIAL, SocialLabel.TOGETHER.value,
            tuple(sorted(pair)), 1.0, SOCIAL_PRIORITY[SocialLabel.TOGETHER.value],
            evidence={"distance_cm": stats.center_distance_cm}))
    return tuple(candidates)


def detect_attack(
    stats: ConflictWindowStats, cfg: AttackThresholds = AttackThresholds()
) -> Optional[BehaviorDecision]:
    stats.validate()
    cfg.validate()
    if not stats.body_contact:
        return None
    evidence = (
        stats.rapid_closing, stats.pounce_or_bite, stats.entanglement,
        stats.local_motion_energy, stats.occlusion_or_observation_loss, stats.recovery_escape,
    )
    weights = (
        cfg.rapid_closing_weight, cfg.pounce_or_bite_weight, cfg.entanglement_weight,
        cfg.local_motion_weight, cfg.observation_loss_weight, cfg.recovery_escape_weight,
    )
    score = float(np.dot(evidence, weights) / sum(weights))
    if score < cfg.min_conflict_score:
        return None
    return BehaviorDecision(Level.SOCIAL, SocialLabel.ATTACK.value,
        tuple(sorted((stats.actor, stats.target))), score,
        SOCIAL_PRIORITY[SocialLabel.ATTACK.value], stats.actor, stats.target,
        {"conflict_score": score, "body_contact": 1.0})


def resolve_social_candidates(candidates: Iterable[BehaviorDecision]) -> Optional[BehaviorDecision]:
    items = tuple(candidates)
    if any(item.level != Level.SOCIAL for item in items):
        raise ValueError("all candidates must be social level")
    return max(items, key=lambda item: (item.priority, item.confidence)) if items else None


def collective_candidates_from_group(
    stats: GroupWindowStats, cfg: CollectiveThresholds
) -> Tuple[BehaviorDecision, ...]:
    cfg.validate()
    stats.validate()
    if len(stats.members) < 3:
        return ()
    candidates = []
    recovery_supported = stats.member_recovery_near_group >= cfg.recovery_min_near_group_fraction
    continuous = stats.member_continuity_fraction >= cfg.huddle_min_continuity_fraction
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Huddling 删除全部速度门（用户决策）。
    if (stats.duration_s >= cfg.huddling_min_duration_s
            and stats.median_neighbor_width_ratio < cfg.huddle_max_neighbor_width_ratio
            and (continuous or recovery_supported)):
        candidates.append(BehaviorDecision(Level.COLLECTIVE, CollectiveLabel.HUDDLING.value,
            stats.members, 1.0, COLLECTIVE_PRIORITY[CollectiveLabel.HUDDLING.value],
            evidence={"neighbor_width_ratio": stats.median_neighbor_width_ratio,
                      "high_motion_fraction": stats.high_motion_fraction,
                      "member_continuity_fraction": stats.member_continuity_fraction,
                      "recovery_near_group": stats.member_recovery_near_group}))
    # # [STEP6 2026-10-05] 社交/群体参数按文档 v4 + 用户覆盖落地：Social clustering 不规定具体时长（0 = 无时长门）。
    clustering = ((cfg.social_clustering_min_duration_s <= 0.0
                   or stats.duration_s >= cfg.social_clustering_min_duration_s)
        and stats.moving_fraction >= cfg.clustering_min_moving_fraction
        and stats.inward_motion_fraction >= cfg.clustering_min_inward_fraction
        and stats.nnd_slope_cm_s <= cfg.clustering_max_nnd_slope_cm_s
        and stats.group_radius_slope_cm_s <= cfg.clustering_max_radius_slope_cm_s
        and stats.member_continuity_fraction >= cfg.huddle_min_continuity_fraction)
    if clustering:
        candidates.append(BehaviorDecision(Level.COLLECTIVE, CollectiveLabel.SOCIAL_CLUSTERING.value,
            stats.members, 1.0, COLLECTIVE_PRIORITY[CollectiveLabel.SOCIAL_CLUSTERING.value],
            evidence={"nnd_slope_cm_s": stats.nnd_slope_cm_s,
                      "radius_slope_cm_s": stats.group_radius_slope_cm_s,
                      "moving_fraction": stats.moving_fraction,
                      "inward_motion_fraction": stats.inward_motion_fraction}))
    prior_cluster = stats.previous_collective_label in {
        CollectiveLabel.HUDDLING.value, CollectiveLabel.SOCIAL_CLUSTERING.value}
    dispersal = (stats.duration_s >= cfg.dispersal_min_duration_s and prior_cluster
        and stats.moving_fraction >= cfg.dispersal_min_moving_fraction
        and stats.nnd_slope_cm_s >= cfg.dispersal_min_nnd_slope_cm_s
        and stats.group_radius_slope_cm_s >= cfg.dispersal_min_radius_slope_cm_s
        and stats.member_continuity_fraction >= cfg.huddle_min_continuity_fraction)
    if dispersal:
        candidates.append(BehaviorDecision(Level.COLLECTIVE, CollectiveLabel.DISPERSAL.value,
            stats.members, 1.0, COLLECTIVE_PRIORITY[CollectiveLabel.DISPERSAL.value],
            evidence={"nnd_slope_cm_s": stats.nnd_slope_cm_s,
                      "radius_slope_cm_s": stats.group_radius_slope_cm_s}))
    stable_spacing = (stats.nnd_slope_cm_s > cfg.clustering_max_nnd_slope_cm_s
                      and stats.group_radius_slope_cm_s > cfg.clustering_max_radius_slope_cm_s
                      and stats.nnd_slope_cm_s < cfg.dispersal_min_nnd_slope_cm_s
                      and stats.group_radius_slope_cm_s < cfg.dispersal_min_radius_slope_cm_s)
    group_move = (stats.duration_s >= cfg.group_locomotion_min_duration_s
        and stats.median_neighbor_distance_cm <= cfg.group_locomotion_max_neighbor_distance_cm
        and stats.moving_fraction >= cfg.group_locomotion_min_moving_fraction
        and stats.direction_coherence >= cfg.group_locomotion_min_coherence
        and stats.spacing_cv <= cfg.group_locomotion_max_spacing_cv and stable_spacing
        and stats.member_continuity_fraction >= cfg.huddle_min_continuity_fraction)
    if group_move:
        candidates.append(BehaviorDecision(Level.COLLECTIVE, CollectiveLabel.GROUP_LOCOMOTION.value,
            stats.members, 1.0, COLLECTIVE_PRIORITY[CollectiveLabel.GROUP_LOCOMOTION.value],
            evidence={"direction_coherence": stats.direction_coherence,
                      "moving_fraction": stats.moving_fraction,
                      "spacing_cv": stats.spacing_cv}))
    return tuple(candidates)


def accept_huddle_recovery(
    *, previous_members: Sequence[int], recovered_mouse_id: int,
    distance_to_current_group_cm: float, reference_mouse_width_cm: float,
    max_recovery_width_ratio: float = 1.5,
) -> bool:
    if recovered_mouse_id not in set(previous_members):
        return False
    distance = distance_to_current_group_cm
    width, ratio = reference_mouse_width_cm, max_recovery_width_ratio
    return (np.isfinite(distance) and distance >= 0 and np.isfinite(width) and width > 0
            and np.isfinite(ratio) and ratio > 0 and distance <= width * ratio)


def detect_isolation(
    stats: IsolationWindowStats, cfg: CollectiveThresholds
) -> Optional[BehaviorDecision]:
    cfg.validate()
    if (stats.duration_s >= max(cfg.isolation_min_duration_s, 10.0)
            and np.isfinite(stats.min_distance_to_any_mouse_cm)
            and stats.min_distance_to_any_mouse_cm >= cfg.isolation_min_distance_cm):
        return BehaviorDecision(Level.COLLECTIVE, CollectiveLabel.ISOLATION.value,
            (stats.mouse_id,), 1.0, COLLECTIVE_PRIORITY[CollectiveLabel.ISOLATION.value],
            evidence={"min_distance_to_any_cm": stats.min_distance_to_any_mouse_cm})
    return None


def resolve_collective_candidates(
    candidates: Iterable[BehaviorDecision]
) -> Optional[BehaviorDecision]:
    items = tuple(candidates)
    if any(item.level != Level.COLLECTIVE for item in items):
        raise ValueError("all candidates must be collective level")
    return max(items, key=lambda item: (item.priority, item.confidence)) if items else None


def _best_event_for_mouse(
    mouse_id: int, events: Iterable[BehaviorDecision], expected_level: Level
) -> Optional[BehaviorDecision]:
    matches = []
    for event in events:
        if event.level != expected_level:
            raise ValueError(f"event level must be {expected_level.value}")
        if mouse_id in event.members:
            matches.append(event)
    return max(matches, key=lambda item: (item.priority, item.confidence)) if matches else None


def resolve_behavior_hierarchy(
    *, individual_by_mouse: Mapping[int, BehaviorDecision],
    social_events: Sequence[BehaviorDecision] = (),
    collective_events: Sequence[BehaviorDecision] = (),
    social_locks: Iterable[int] = (), collective_locks: Iterable[int] = (),
) -> BehaviorHierarchySnapshot:
    """Keep all events, while selecting one highest-layer primary per mouse."""
    ids = set(individual_by_mouse)
    for mouse_id, decision in individual_by_mouse.items():
        if decision.level != Level.INDIVIDUAL or decision.members != (mouse_id,):
            raise ValueError("individual_by_mouse needs one singleton individual decision per ID")
    for level, events in ((Level.SOCIAL, social_events), (Level.COLLECTIVE, collective_events)):
        for event in events:
            if event.level != level or not set(event.members).issubset(ids):
                raise ValueError(f"{level.value} events must reference known IDs at that level")
    social_lock_set, collective_lock_set = set(social_locks), set(collective_locks)
    if not social_lock_set.issubset(ids) or not collective_lock_set.issubset(ids):
        raise ValueError("hierarchy locks may only reference observed mouse IDs")
    primary = dict(individual_by_mouse)
    for mouse_id in ids:
        event = _best_event_for_mouse(mouse_id, social_events, Level.SOCIAL)
        if event:
            primary[mouse_id] = event
        elif mouse_id in social_lock_set:
            primary[mouse_id] = BehaviorDecision(Level.SOCIAL, SocialLabel.UNKNOWN.value,
                (mouse_id,), priority=SOCIAL_PRIORITY[SocialLabel.UNKNOWN.value],
                evidence={"social_context_lock": 1.0})
    for mouse_id in ids:
        event = _best_event_for_mouse(mouse_id, collective_events, Level.COLLECTIVE)
        if event:
            primary[mouse_id] = event
        elif mouse_id in collective_lock_set:
            primary[mouse_id] = BehaviorDecision(Level.COLLECTIVE, CollectiveLabel.UNKNOWN.value,
                (mouse_id,), priority=COLLECTIVE_PRIORITY[CollectiveLabel.UNKNOWN.value],
                evidence={"collective_context_lock": 1.0})
    return BehaviorHierarchySnapshot(dict(individual_by_mouse), tuple(social_events),
                                     tuple(collective_events), primary)


__all__ = [
    "KEYPOINT_NAMES", "KP", "Level", "MotionLabel", "SocialLabel", "CollectiveLabel",
    "EventPhase", "SOCIAL_PRIORITY", "COLLECTIVE_PRIORITY", "PoseObservation", "PoseQuality",
    "IndividualGeometry", "PairGeometry", "BehaviorDecision", "BehaviorHierarchySnapshot",
    "MotionThresholds", "SocialThresholds", "AttackThresholds", "CollectiveThresholds",
    "HeuristicThresholds", "thresholds_from_config",
    "MotionWindowStats", "PairWindowStats", "ConflictWindowStats", "GroupWindowStats",
    "IsolationWindowStats", "HysteresisStateMachine", "valid_keypoints",
    "compute_individual_geometry", "compute_pair_geometry", "pairwise_center_distances",
    "connected_components_from_distance", "velocity_coherence", "detect_individual_motion",
    "social_candidates_from_pair", "detect_attack", "resolve_social_candidates",
    "collective_candidates_from_group", "accept_huddle_recovery", "detect_isolation",
    "resolve_collective_candidates", "resolve_behavior_hierarchy",
]
