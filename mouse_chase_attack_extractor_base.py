#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多鼠追逐/攻击启发式初筛与视频片段提取

四类标签：
0 非追逐非攻击
1 非攻击性追逐
2 非追逐攻击
3 攻击性追逐

输入：Ultralytics YOLO26（或兼容Ultralytics API的YOLO Pose）自定义7关键点模型 + 视频
关键点顺序固定：
    nose, left ear, right ear, base of neck, left hip, right hip, base of tail

主要输出（文件名均为中文）：
- 逐帧行为标签.csv：逐帧特征、原始判定、时序后处理判定
- 行为事件表.csv：事件级汇总与片段路径
- 事件片段/：按四类标签分类的视频片段（00_非追逐非攻击 等中文子目录）
- 行为标注视频.mp4：带关键点和最终标签的核查视频

注意：本程序定位为高召回初筛器，不应把启发式输出直接当作人工真值。
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from disk_sequence_guard import (
    DiskSequenceIdentityGuard,
    choose_adaptive_thread_count,
)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # scipy不可用时使用贪心匹配回退
    linear_sum_assignment = None


# 模型关键点的严格索引顺序。名称按用户数据集中的显示名称保存。
KEYPOINT_NAMES = [
    "nose",
    "left ear",
    "right ear",
    "base of neck",
    "left hip",
    "right hip",
    "base of tail",
]

# 同时保留Python友好的别名及旧版本别名，避免行为计算模块因名称变化失效。
# 所有别名都只映射索引，不改变模型输出的7点顺序。
KP = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}
KP.update({
    "left_ear": 1,
    "right_ear": 2,
    "base_of_neck": 3,
    "left_hip": 4,
    "right_hip": 5,
    "base_of_tail": 6,
    # 向后兼容旧版内部字段
    "neck": 3,
    "left_hind": 4,
    "right_hind": 5,
    "tail": 6,
})

LABELS = {
    0: ("non_chase_non_attack", "非追逐非攻击"),
    1: ("non_aggressive_chase", "非攻击性追逐"),
    2: ("non_chase_attack", "非追逐攻击"),
    3: ("aggressive_chase", "攻击性追逐"),
}

CLIP_DIRS = {
    0: "00_非追逐非攻击",
    1: "01_非攻击性追逐",
    2: "02_非追逐攻击",
    3: "03_攻击性追逐",
}

SKELETON_EDGES = [
    (KP["nose"], KP["left_ear"]),
    (KP["nose"], KP["right_ear"]),
    (KP["left_ear"], KP["neck"]),
    (KP["right_ear"], KP["neck"]),
    (KP["neck"], KP["left_hind"]),
    (KP["neck"], KP["right_hind"]),
    (KP["left_hind"], KP["tail"]),
    (KP["right_hind"], KP["tail"]),
]

BASE_MODULE_VERSION = "1.42.1-final-code-merge"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


# OpenCV CLAHE objects allocate internal lookup tables and temporary buffers.
# Reusing one object per worker thread is mathematically identical to creating
# one per detection, while avoiding thousands of allocator calls per video.
_APPEARANCE_THREAD_LOCAL = threading.local()


def _get_appearance_clahe() -> Any:
    clahe = getattr(_APPEARANCE_THREAD_LOCAL, "clahe", None)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
        _APPEARANCE_THREAD_LOCAL.clahe = clahe
    return clahe


# ----------------------------- 通用工具 -----------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_point(point: np.ndarray) -> bool:
    return point.shape == (2,) and bool(np.all(np.isfinite(point)))


def safe_norm(vec: np.ndarray) -> float:
    if vec.shape != (2,) or not np.all(np.isfinite(vec)):
        return float("nan")
    return float(np.linalg.norm(vec))


def unit_vector(vec: np.ndarray) -> np.ndarray:
    n = safe_norm(vec)
    if not np.isfinite(n) or n < 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return vec.astype(np.float64) / n


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    ua = unit_vector(vec_a)
    ub = unit_vector(vec_b)
    if not np.all(np.isfinite(ua)) or not np.all(np.isfinite(ub)):
        return 0.0
    return float(np.clip(np.dot(ua, ub), -1.0, 1.0))


def angle_difference_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    sim = cosine_similarity(vec_a, vec_b)
    return float(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0))))


def nanmean_points(points: Sequence[np.ndarray]) -> np.ndarray:
    valid = [p for p in points if finite_point(np.asarray(p))]
    if not valid:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.nanmean(np.stack(valid, axis=0), axis=0).astype(np.float64)


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not finite_point(a) or not finite_point(b):
        return float("nan")
    return float(np.linalg.norm(a - b))


def min_point_distance(point: np.ndarray, points: np.ndarray) -> float:
    if not finite_point(point):
        return float("nan")
    valid = points[np.all(np.isfinite(points), axis=1)]
    if len(valid) == 0:
        return float("nan")
    return float(np.min(np.linalg.norm(valid - point[None, :], axis=1)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if np.nanstd(a) < 1e-8 or np.nanstd(b) < 1e-8:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def sanitize_filename(text: str) -> str:
    # \w 在 Python3 的 str 模式下匹配 Unicode 字符（含中文），因此中文视频名/标签名得以保留
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text.strip("_.") or "片段"


def mode_or_default(values: Iterable[Any], default: Any = -1) -> Any:
    filtered = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


# ----------------------------- 数据结构 -----------------------------


@dataclass
class Detection:
    raw_track_id: Optional[int]
    keypoints_px: np.ndarray  # (7, 2)
    keypoint_conf: np.ndarray  # (7,)
    bbox_xyxy: np.ndarray  # (4,)
    box_conf: float
    # v1.5: 用于多鼠身份保持的外观/姿态描述。默认值保证旧调用仍兼容。
    # 外观只使用结构纹理。绝对毛色/亮度不参与身份代价。
    # 白鼠仅用 white_score 选择“灰度反转+CLAHE”预处理，不作为ID证据。
    appearance_feature: Optional[np.ndarray] = None
    brightness_score: float = float("nan")
    white_score: float = float("nan")
    is_white_candidate: bool = False
    appearance_mode: str = "normal_clahe"
    appearance_reliable: bool = False
    max_overlap_iou: float = 0.0
    pose_quality: float = 0.0
    normalized_pose: Optional[np.ndarray] = None
    anchor_feature: Optional[np.ndarray] = None
    heading_vector: Optional[np.ndarray] = None
    synthetic_recovery: bool = False
    # global: 全图YOLO跟踪结果；local_recovery: 接触簇局部二次推理；
    # predicted_hold: 仅供内部ID保持的运动预测，不可用于渲染和行为几何。
    detection_source: str = "global"
    # v1.12.7：骨架解剖学不可信时，跟踪/几何中心回退到检测框中心。
    # 模型在域外场地（如裸板）上关键点会系统性错位但检测框仍然可靠，
    # 关键点加权中心会被拉偏→污染匹配代价与行为几何。由检测流按
    # skeleton_anatomy_ok 判定后置位。
    prefer_bbox_center: bool = False
    # v1.20：关键点来源与异常骨架恢复元数据。
    # keypoint_sources元素取值：RAW / ROI / PREDICTED / TEMPLATE / MISSING。
    keypoint_sources: Optional[np.ndarray] = None
    pose_recovery_score: float = float("nan")
    pose_recovery_reason: str = ""
    # v1.22：Pose框约束的伪实例掩码。当前权重不是分割模型，因此这些字段
    # 来自GrabCut/关键点种子生成的伪掩码，只在质量可靠时参与身份代价。
    mask_feature: Optional[np.ndarray] = None
    mask_shape_feature: Optional[np.ndarray] = None
    mask_texture_feature: Optional[np.ndarray] = None
    mask_color_feature: Optional[np.ndarray] = None
    mask_quality: float = 0.0
    mask_reliable: bool = False
    mask_source: str = "none"
    mask_area_ratio: float = float("nan")
    mask_bbox_xyxy: Optional[np.ndarray] = None
    instance_mask_local: Optional[np.ndarray] = None
    # A track-gap ROI is requested for one already-confirmed logical ID.  The
    # target is only an assignment routing constraint; ordinary cost, margin,
    # jump and confidence gates still decide whether the observation is safe.
    recovery_target_logical_id: int = -1

    # v1.40.1：在Pose恢复和去重完成后显式刷新的一帧内派生几何缓存。
    # 缓存是普通运行时属性，不进入dataclass字段、比较或asdict输出；调用方在
    # 最终候选确定后通过 refresh_derived_geometry_cache() 建立。
    def __post_init__(self) -> None:
        self._cached_center_px: Optional[np.ndarray] = None
        self._cached_body_length_px: Optional[float] = None

    def __getstate__(self) -> Dict[str, Any]:
        """Serialize only authoritative detection data, never derived caches."""
        state = dict(self.__dict__)
        state.pop("_cached_center_px", None)
        state.pop("_cached_body_length_px", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Accept caches/checkpoints produced before v1.40.1 safely."""
        self.__dict__.update(dict(state))
        self._cached_center_px = None
        self._cached_body_length_px = None

    def invalidate_derived_geometry_cache(self) -> None:
        """Invalidate one-frame geometry after keypoints/bbox policy changes."""
        self._cached_center_px = None
        self._cached_body_length_px = None

    def refresh_derived_geometry_cache(self) -> None:
        """Precompute immutable derived geometry for downstream hot loops."""
        self._cached_center_px = self._compute_center_px_uncached()
        self._cached_body_length_px = self._compute_body_length_px_uncached()

    def _compute_center_px_uncached(self) -> np.ndarray:
        if self.prefer_bbox_center:
            x1, y1, x2, y2 = self.bbox_xyxy
            return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
        p = np.asarray(self.keypoints_px, dtype=np.float64)
        c = np.asarray(self.keypoint_conf, dtype=np.float64).reshape(-1)
        n = min(len(p), len(c), len(KEYPOINT_NAMES))
        if n > 0:
            valid = (
                np.isfinite(p[:n, 0]) & np.isfinite(p[:n, 1])
                & (p[:n, 0] > 0) & (p[:n, 1] > 0)
                & np.isfinite(c[:n]) & (c[:n] >= 0.08)
            )
            core_idx = [
                idx
                for idx in (KP["neck"], KP["left_hind"], KP["right_hind"], KP["tail"])
                if idx < n
            ]
            core = np.zeros(n, dtype=bool)
            core[core_idx] = True
            use = valid & core
            if use.sum() >= 2:
                weights = np.clip(c[:n][use], 0.05, 1.0)
                return np.average(p[:n][use], axis=0, weights=weights).astype(np.float64)
            if valid.sum() >= 2:
                weights = np.clip(c[:n][valid], 0.05, 1.0)
                return np.average(p[:n][valid], axis=0, weights=weights).astype(np.float64)
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)

    def _compute_body_length_px_uncached(self) -> float:
        p = np.asarray(self.keypoints_px, dtype=np.float64)
        c = np.asarray(self.keypoint_conf, dtype=np.float64).reshape(-1)
        ni, ti = KP["nose"], KP["tail"]
        if ni < len(p) and ti < len(p) and ni < len(c) and ti < len(c):
            if c[ni] >= 0.08 and c[ti] >= 0.08 and finite_point(p[ni]) and finite_point(p[ti]):
                length = point_distance(p[ni], p[ti])
                if np.isfinite(length) and length > 3.0:
                    return length
        x1, y1, x2, y2 = self.bbox_xyxy
        # 检测框长边更稳定，乘0.80近似鼻尾有效体长。
        return float(max(abs(x2 - x1), abs(y2 - y1), 1.0) * 0.80)

    @property
    def center_px(self) -> np.ndarray:
        """身份跟踪中心：优先使用高置信身体核心点，避免低置信飞点拉偏中心。"""
        cached = getattr(self, "_cached_center_px", None)
        return cached if cached is not None else self._compute_center_px_uncached()

    @property
    def body_length_px(self) -> float:
        cached = getattr(self, "_cached_body_length_px", None)
        return float(cached) if cached is not None else self._compute_body_length_px_uncached()


@dataclass
class IdentityTrack:
    logical_id: int
    last_center_px: np.ndarray
    velocity_px_per_frame: np.ndarray
    last_frame: int
    raw_track_id: Optional[int]
    body_length_px: float
    # short/long双模板：短期适应光照，长期模板仅在无遮挡高质量帧更新。
    appearance_feature: Optional[np.ndarray] = None
    appearance_feature_long: Optional[np.ndarray] = None
    # v1.22：每个逻辑ID的短期/长期伪实例掩码身份模板。
    mask_feature_short: Optional[np.ndarray] = None
    mask_feature_long: Optional[np.ndarray] = None
    mask_quality_ema: float = 0.0
    mask_updates: int = 0
    mask_long_updates: int = 0
    brightness_score: float = float("nan")
    white_score: float = float("nan")
    is_white_candidate: bool = False
    normalized_pose: Optional[np.ndarray] = None
    anchor_feature: Optional[np.ndarray] = None
    heading_vector: Optional[np.ndarray] = None
    bbox_wh: Optional[np.ndarray] = None
    last_keypoints_px: Optional[np.ndarray] = None
    last_keypoint_conf: Optional[np.ndarray] = None
    last_bbox_xyxy: Optional[np.ndarray] = None
    last_box_conf: float = 0.0
    appearance_updates: int = 0
    hits: int = 1
    lock_strength: float = 0.0
    # 轨迹状态机：tracked（稳定）/ suspicious（歧义或接触中，冻结外观）/
    # lost（长期无观测，重激活需要外观或OKS联合证据）。
    state: str = "tracked"
    clean_streak: int = 0
    # 短时身份记忆（修复文档v1.1 §12.2）：每条轨迹独立维护
    # 位置/速度/方向/姿态/体型/外观/邻近关系的动态对象档案。
    memory: Optional["TemporaryIdentityMemory"] = None


@dataclass
class TemporaryIdentityMemory:
    """无标记同外观小鼠的短时身份记忆（修复文档v1.1 §11~§12）。

    模拟研究员维持临时身份的“动态对象档案”：它上一时刻在哪里、朝什么方向
    运动、下一时刻应该出现在哪里、周围有哪些小鼠。该模块只增强身份稳定性，
    绝不作为检测过滤器——没有记忆或记忆不完整的检测照样保留并渲染。
    每个视频或独立短片段开始时重新初始化，不承诺跨视频编号对应真实个体。
    """

    track_id: int
    state: str = "tentative"  # tentative / confirmed / suspicious / lost
    last_frame: int = -1
    hits: int = 0  # 连续命中帧数（漏检即清零）
    misses: int = 0  # 连续漏检帧数
    identity_confidence: float = 0.30
    center_history: Deque[np.ndarray] = field(default_factory=deque)
    velocity_history: Deque[np.ndarray] = field(default_factory=deque)
    keypoint_history: Deque[Tuple[np.ndarray, np.ndarray]] = field(default_factory=deque)
    body_center_history: Deque[np.ndarray] = field(default_factory=deque)
    direction_history: Deque[np.ndarray] = field(default_factory=deque)
    body_length_ema: float = float("nan")
    body_width_ema: float = float("nan")
    appearance_feature_ema: Optional[np.ndarray] = None
    neighbor_relation_history: Deque[Dict[int, Tuple[float, float]]] = field(default_factory=deque)

    def median_velocity(self, window: int) -> np.ndarray:
        """最近 window 帧的中位速度（px/帧），降低单帧抖动影响（§12.7 VELOCITY_WINDOW）。"""
        if not self.velocity_history:
            return np.zeros(2, dtype=np.float64)
        samples = list(self.velocity_history)[-max(int(window), 1):]
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in samples], axis=0)
        return np.median(arr, axis=0).astype(np.float64)

    def last_direction(self) -> Optional[np.ndarray]:
        for vec in reversed(self.direction_history):
            if vec is not None and np.all(np.isfinite(vec)):
                return np.asarray(vec, dtype=np.float64)
        return None


@dataclass
class MouseObservation:
    frame: int
    logical_id: int
    raw_track_id: Optional[int]
    keypoints_px: np.ndarray
    keypoints_cm: np.ndarray
    keypoint_conf: np.ndarray
    bbox_xyxy: np.ndarray
    box_conf: float
    center_cm: np.ndarray
    head_cm: np.ndarray
    rear_cm: np.ndarray
    heading: np.ndarray
    velocity_cm_s: np.ndarray
    speed_cm_s: float
    acceleration_cm_s2: float
    angular_speed_deg_s: float
    nose_speed_cm_s: float
    body_length_cm: float
    # 轨迹状态（渲染用）：tracked稳定 / tentative临时ID / suspicious身份待确认
    track_state: str = "tracked"
    # 渲染标签：confirmed→"ID n"，tentative→"TMP n"，suspicious→"ID? n"（§3.2）。
    display_label: str = ""
    # v1.20：逐关键点来源，仅用于审计和渲染；行为模块仍以keypoint_conf为准。
    keypoint_sources: Optional[np.ndarray] = None


@dataclass
class PairFeatures:
    actor_id: int
    target_id: int
    center_distance_cm: float
    head_distance_cm: float
    actor_nose_to_target_body_cm: float
    actor_nose_to_target_tail_cm: float
    actor_speed_cm_s: float
    target_speed_cm_s: float
    actor_acceleration_cm_s2: float
    target_acceleration_cm_s2: float
    actor_nose_speed_cm_s: float
    target_nose_speed_cm_s: float
    direction_similarity: float
    pursuit_alignment: float
    actor_behind_target: bool
    trajectory_correlation: float
    actor_path_window_cm: float
    target_path_window_cm: float
    distance_drop_cm: float
    target_turn_angle_deg: float
    contact: bool
    repeated_contact_count: int
    chase_score: int
    chase_candidate: bool
    chase_high_confidence: bool
    attack_dynamic_evidence: int
    stationary_fight_candidate: bool
    attack_candidate: bool


@dataclass
class IdentityDebug:
    frame: int
    logical_id: int
    raw_track_id: Optional[int]
    assignment_cost: float
    proposed_logical_id: int = -1
    assignment_gain: float = float("nan")
    dwell_count: int = 0
    dwell_required: int = 0
    cooldown_remaining: int = 0
    commit_status: str = "accepted"
    switch_rejected_reason: str = ""
    appearance_mode: str = ""
    detection_source: str = "global"
    occlusion_cluster_id: int = -1
    cluster_expected_count: int = 0
    cluster_observed_count: int = 0
    track_state: str = ""


# ----------------------------- 关键点平滑与尺度 -----------------------------


class KeypointSmoother:
    def __init__(self, alpha: float, min_conf: float, max_missing: int,
                 interp_decay: float = 0.85) -> None:
        self.alpha = float(alpha)
        self.min_conf = float(min_conf)
        self.max_missing = int(max_missing)
        # v1.12.4：插值点置信度按 last_raw×decay^缺失帧 衰减，但不低于min_conf——
        # 旧版插值点effective_conf=0.0，被渲染阈值(≥0.1)直接过滤，
        # 插值机制从未生效（端点丢失就消失，骨架稀疏）。窗口内保点、窗口外消失。
        self.interp_decay = float(interp_decay)
        self.points: Dict[int, np.ndarray] = {}
        self.missing_counts: Dict[int, np.ndarray] = {}
        self.last_raw_conf: Dict[int, np.ndarray] = {}

    def update(
        self,
        logical_id: int,
        points: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = points.astype(np.float64, copy=True)
        confidence = confidence.astype(np.float64, copy=True)

        previous = self.points.get(logical_id)
        missing = self.missing_counts.get(logical_id, np.zeros(len(KEYPOINT_NAMES), dtype=np.int32))
        last_conf = self.last_raw_conf.get(logical_id, np.zeros(len(KEYPOINT_NAMES), dtype=np.float64))

        if previous is None:
            previous = np.full_like(points, np.nan, dtype=np.float64)

        output = np.full_like(points, np.nan, dtype=np.float64)
        effective_conf = confidence.copy()

        for idx in range(len(KEYPOINT_NAMES)):
            raw_valid = (
                idx < len(confidence)
                and confidence[idx] >= self.min_conf
                and finite_point(points[idx])
            )
            prev_valid = finite_point(previous[idx])

            if raw_valid:
                if prev_valid:
                    output[idx] = self.alpha * points[idx] + (1.0 - self.alpha) * previous[idx]
                else:
                    output[idx] = points[idx]
                missing[idx] = 0
                last_conf[idx] = float(confidence[idx])
            else:
                missing[idx] += 1
                if prev_valid and missing[idx] <= self.max_missing:
                    output[idx] = previous[idx]
                    effective_conf[idx] = max(
                        float(last_conf[idx]) * (self.interp_decay ** int(missing[idx])),
                        self.min_conf,
                    )
                else:
                    output[idx] = np.array([np.nan, np.nan])
                    effective_conf[idx] = 0.0

        self.points[logical_id] = output.copy()
        self.missing_counts[logical_id] = missing.copy()
        self.last_raw_conf[logical_id] = last_conf.copy()
        return output, effective_conf


class ScaleEstimator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.mode = str(config.get("mode", "body_length"))
        self.fixed_cm_per_pixel = config.get("cm_per_pixel")
        self.assumed_body_length_cm = float(config.get("assumed_mouse_body_length_cm", 8.0))
        self.alpha = float(config.get("scale_smoothing_alpha", 0.05))
        self.current_cm_per_pixel: Optional[float] = None

        if self.mode == "fixed":
            if self.fixed_cm_per_pixel is None or float(self.fixed_cm_per_pixel) <= 0:
                raise ValueError("scale.mode=fixed时，必须设置正数cm_per_pixel。")
            self.current_cm_per_pixel = float(self.fixed_cm_per_pixel)

    def update(self, detections: Sequence[Detection]) -> float:
        if self.mode == "fixed":
            assert self.current_cm_per_pixel is not None
            return self.current_cm_per_pixel

        lengths = [d.body_length_px for d in detections if np.isfinite(d.body_length_px) and d.body_length_px > 5]
        if lengths:
            estimated = self.assumed_body_length_cm / float(np.median(lengths))
            if self.current_cm_per_pixel is None:
                self.current_cm_per_pixel = estimated
            else:
                self.current_cm_per_pixel = (
                    self.alpha * estimated + (1.0 - self.alpha) * self.current_cm_per_pixel
                )

        if self.current_cm_per_pixel is None:
            # 极少数情况下第一帧关键点全部无效，先用1.0保证程序不中断；有效检测出现后会自动更新。
            self.current_cm_per_pixel = 1.0
        return float(self.current_cm_per_pixel)


# ----------------------------- 多鼠外观与姿态描述 -----------------------------


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 4 or b.size < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _pose_quality_for_identity(det: Detection, min_conf: float = 0.10) -> float:
    c = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
    p = np.asarray(det.keypoints_px, dtype=np.float64)
    n = min(len(KEYPOINT_NAMES), len(c), len(p))
    if n <= 0:
        return 0.0
    valid = (
        np.isfinite(p[:n, 0]) & np.isfinite(p[:n, 1])
        & (p[:n, 0] > 0) & (p[:n, 1] > 0)
        & np.isfinite(c[:n]) & (c[:n] >= min_conf)
    )
    return float(valid.sum() / max(n, 1))


def _heading_for_identity(det: Detection, min_conf: float = 0.08) -> Optional[np.ndarray]:
    p = np.asarray(det.keypoints_px, dtype=np.float64)
    c = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
    for start, end in ((KP["tail"], KP["nose"]), (KP["neck"], KP["nose"]), (KP["tail"], KP["neck"])):
        if start >= len(p) or end >= len(p) or start >= len(c) or end >= len(c):
            continue
        if c[start] < min_conf or c[end] < min_conf:
            continue
        if not finite_point(p[start]) or not finite_point(p[end]):
            continue
        v = p[end] - p[start]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            return (v / n).astype(np.float64)
    return None


def _normalized_pose_for_identity(det: Detection, min_conf: float = 0.08) -> Optional[np.ndarray]:
    p = np.asarray(det.keypoints_px, dtype=np.float64)
    c = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
    n = min(len(KEYPOINT_NAMES), len(p), len(c))
    if n <= 0:
        return None
    valid = (
        np.isfinite(p[:n, 0]) & np.isfinite(p[:n, 1])
        & (p[:n, 0] > 0) & (p[:n, 1] > 0)
        & np.isfinite(c[:n]) & (c[:n] >= min_conf)
    )
    if valid.sum() < 3:
        return None
    center = det.center_px
    scale = max(float(det.body_length_px), 5.0)
    out = np.full((len(KEYPOINT_NAMES), 2), np.nan, dtype=np.float64)
    valid_idx = np.where(valid)[0]
    out[valid_idx] = (p[valid_idx] - center[None, :]) / scale
    heading = _heading_for_identity(det, min_conf=min_conf)
    if heading is not None:
        # 将身体朝向旋转到+x，使姿态特征对转向更稳定。
        angle = -math.atan2(float(heading[1]), float(heading[0]))
        ca, sa = math.cos(angle), math.sin(angle)
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
        m = np.all(np.isfinite(out), axis=1)
        out[m] = out[m] @ rot.T
    return out.reshape(-1)



def _identity_anchor_descriptor(det: Detection, min_conf: float = 0.08) -> Optional[np.ndarray]:
    """虚拟锚点身份描述：nose/neck/tail、身体中心、nose-neck中点、neck-tail中点。

    坐标以身体中心平移、体长归一化，并旋转到统一朝向。与绝对颜色无关。
    """
    p = np.asarray(det.keypoints_px, dtype=np.float64)
    c = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
    n = min(len(p), len(c), len(KEYPOINT_NAMES))
    if n < len(KEYPOINT_NAMES):
        return None
    valid = (
        np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])
        & (p[:, 0] > 0) & (p[:, 1] > 0)
        & np.isfinite(c) & (c >= min_conf)
    )
    ni, ne, lh, rh, ti = KP["nose"], KP["neck"], KP["left_hind"], KP["right_hind"], KP["tail"]
    pts = []
    # 关键真实点
    for idx in (ni, ne, ti):
        pts.append(p[idx] if valid[idx] else np.array([np.nan, np.nan]))
    # 身体中心优先 neck+双髋+tail
    core_idx = [idx for idx in (ne, lh, rh, ti) if valid[idx]]
    body_center = np.mean(p[core_idx], axis=0) if len(core_idx) >= 2 else det.center_px
    pts.append(body_center)
    pts.append((p[ni] + p[ne]) / 2.0 if valid[ni] and valid[ne] else np.array([np.nan, np.nan]))
    pts.append((p[ne] + p[ti]) / 2.0 if valid[ne] and valid[ti] else np.array([np.nan, np.nan]))
    arr = np.asarray(pts, dtype=np.float64)
    center = np.asarray(body_center, dtype=np.float64)
    scale = max(float(det.body_length_px), 5.0)
    arr = (arr - center[None, :]) / scale
    heading = _heading_for_identity(det, min_conf=min_conf)
    if heading is not None:
        angle = -math.atan2(float(heading[1]), float(heading[0]))
        ca, sa = math.cos(angle), math.sin(angle)
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
        good = np.all(np.isfinite(arr), axis=1)
        arr[good] = arr[good] @ rot.T
    return arr.reshape(-1)

def _appearance_descriptor(
    frame: np.ndarray,
    det: Detection,
    descriptor_width: int = 24,
    descriptor_height: int = 12,
    white_invert_threshold: float = 0.55,
) -> Tuple[Optional[np.ndarray], float, float, str]:
    """提取颜色无关的结构纹理描述。

    关键原则：
    - 绝对亮度、白色比例、HSV颜色不进入身份特征；
    - white_score只用于决定是否先做灰度反转；
    - 白鼠ROI：255-gray -> CLAHE，使轮廓、阴影和标签纹理更明显；
    - 黑鼠ROI：gray -> CLAHE；
    - 最终只保留局部标准化纹理、梯度和边缘结构。
    """
    h, w = frame.shape[:2]
    box = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)
    if box.size < 4:
        return None, float("nan"), float("nan"), "invalid"
    x1, y1, x2, y2 = box[:4]
    bw = max(float(x2 - x1), 2.0)
    bh = max(float(y2 - y1), 2.0)
    mx, my = 0.08 * bw, 0.08 * bh
    ix1 = max(0, min(w - 1, int(round(x1 + mx))))
    iy1 = max(0, min(h - 1, int(round(y1 + my))))
    ix2 = max(ix1 + 1, min(w, int(round(x2 - mx))))
    iy2 = max(iy1 + 1, min(h, int(round(y2 - my))))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0 or min(crop.shape[:2]) < 3:
        return None, float("nan"), float("nan"), "invalid"

    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.ellipse(mask, (cw // 2, ch // 2),
                (max(1, int(cw * 0.46)), max(1, int(ch * 0.46))),
                0, 0, 360, 255, -1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = gray[mask > 0]
    sat = hsv[..., 1][mask > 0]
    if pixels.size < 16:
        return None, float("nan"), float("nan"), "invalid"

    q25, q50 = np.percentile(pixels, [25, 50]).astype(np.float64) / 255.0
    mean_b = float(np.mean(pixels) / 255.0)
    mean_s = float(np.mean(sat) / 255.0)
    white_ratio = float(np.mean(pixels >= 185))
    brightness_score = float(0.55 * q25 + 0.25 * q50 + 0.20 * mean_b)
    white_score = float(np.clip(0.45 * q25 + 0.30 * white_ratio + 0.15 * q50 + 0.10 * (1.0 - mean_s), 0.0, 1.0))
    is_white = bool(white_score >= white_invert_threshold)

    work = 255 - gray if is_white else gray.copy()
    mode = "white_invert_clahe" if is_white else "normal_clahe"
    work = _get_appearance_clahe().apply(work)

    # 朝向对齐，减少转身对纹理模板的影响。
    heading = _heading_for_identity(det, min_conf=0.06)
    aligned = work
    aligned_mask = mask
    if heading is not None:
        angle_deg = math.degrees(math.atan2(float(heading[1]), float(heading[0])))
        center = (cw / 2.0, ch / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        aligned = cv2.warpAffine(work, mat, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned_mask = cv2.warpAffine(mask, mat, (cw, ch), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    small = cv2.resize(aligned, (descriptor_width, descriptor_height), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
    small_mask = cv2.resize(aligned_mask, (descriptor_width, descriptor_height), interpolation=cv2.INTER_NEAREST) > 0
    vals = small[small_mask]
    mean_v = float(np.mean(vals)) if vals.size else float(np.mean(small))
    std_v = float(np.std(vals)) if vals.size else float(np.std(small))
    norm = (small - mean_v) / (std_v + 1e-6)
    norm = np.clip(norm, -2.5, 2.5) / 5.0 + 0.5

    gx = cv2.Sobel(norm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / (float(np.percentile(grad, 95)) + 1e-6)
    grad = np.clip(grad, 0.0, 1.0)
    edges = cv2.Canny((norm * 255).astype(np.uint8), 45, 120).astype(np.float64) / 255.0

    # 直方图使用梯度幅值而非灰度，进一步去除毛色影响。
    hist_grad = cv2.calcHist([(grad * 255).astype(np.uint8)], [0], small_mask.astype(np.uint8) * 255, [12], [0, 256]).reshape(-1)
    hist_grad = hist_grad / max(float(hist_grad.sum()), 1.0)
    feature = np.concatenate([
        hist_grad,
        norm.reshape(-1),
        grad.reshape(-1),
        edges.reshape(-1),
    ]).astype(np.float64)
    feature = np.clip(feature, 0.0, 1.0)
    return feature, brightness_score, white_score, mode

def enrich_detections_with_appearance(
    frame: np.ndarray,
    detections: Sequence[Detection],
    config: Optional[Mapping[str, Any]] = None,
    skip_appearance: bool = False,
) -> List[Detection]:
    """添加颜色无关结构纹理、白鼠反转模式、姿态与锚点。

    白鼠判定只控制预处理方式，不进入身份匹配代价，也不做黑/白硬门控。
    v1.13.0：skip_appearance=True（简洁运动模式）时只算姿态质量/朝向/
    重叠度等轻量字段，跳过外观直方图（身份不靠它，省每帧每检测开销）。
    """
    cfg = dict(config or {})
    max_overlap = float(cfg.get("appearance_max_iou", 0.18))
    min_pose_quality = float(cfg.get("appearance_min_pose_quality", 0.35))
    invert_threshold = float(cfg.get("white_invert_threshold", 0.55))
    detections = list(detections)
    # Pose恢复、候选融合和去重均已完成；从这里开始几何只读。显式刷新可避免
    # cluster、mask和identity在同一帧重复构造valid mask/加权中心/鼻尾体长。
    for det in detections:
        det.refresh_derived_geometry_cache()
    if len(detections) > 1:
        _, appearance_iou_matrix, _ = _pairwise_bbox_geometry(detections)
        np.fill_diagonal(appearance_iou_matrix, 0.0)
        max_iou_values = np.max(appearance_iou_matrix, axis=1)
    else:
        max_iou_values = np.zeros(len(detections), dtype=np.float64)
    for i, det in enumerate(detections):
        det.pose_quality = _pose_quality_for_identity(det, min_conf=0.08)
        det.normalized_pose = _normalized_pose_for_identity(det, min_conf=0.08)
        det.anchor_feature = _identity_anchor_descriptor(det, min_conf=0.08)
        det.heading_vector = _heading_for_identity(det, min_conf=0.08)
        max_iou = float(max_iou_values[i]) if i < len(max_iou_values) else 0.0
        if skip_appearance:
            det.appearance_feature = None
            det.brightness_score = float("nan")
            det.white_score = float("nan")
            det.is_white_candidate = False
            det.appearance_mode = "skipped_simple_motion"
            det.max_overlap_iou = float(max_iou)
            det.appearance_reliable = False
            continue
        feat, brightness, white, mode = _appearance_descriptor(
            frame, det, white_invert_threshold=invert_threshold
        )
        det.appearance_feature = feat
        det.brightness_score = brightness
        det.white_score = white
        det.is_white_candidate = bool(white >= invert_threshold) if np.isfinite(white) else False
        det.appearance_mode = mode
        det.max_overlap_iou = float(max_iou)
        det.appearance_reliable = bool(
            feat is not None
            and det.pose_quality >= min_pose_quality
            and max_iou <= max_overlap
            and det.box_conf >= float(cfg.get("appearance_min_box_conf", 0.12))
        )
    return detections


# ----------------------------- 检测去重与接触簇 -----------------------------


def _bbox_intersection_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """交集面积占较小框面积的比例。比IoU更适合识别同一实例的重复框。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 4 or b.size < 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    small = min(aa, ab)
    return float(inter / small) if small > 1e-9 else 0.0


def _pairwise_bbox_geometry(detections: Sequence[Detection]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量返回中心距离、IoU和交集/小框面积矩阵。

    只替代重复的几何内核；上层排序、阈值、tie-break与合并顺序仍由原Python
    控制流决定，避免性能优化改变检测进入身份模块的顺序。
    """
    dets = list(detections)
    count = len(dets)
    if count == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty, empty
    boxes = np.asarray(
        [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for det in dets],
        dtype=np.float64,
    )
    centers = np.asarray(
        [np.asarray(det.center_px, dtype=np.float64).reshape(-1)[:2] for det in dets],
        dtype=np.float64,
    )
    center_delta = centers[:, None, :] - centers[None, :, :]
    center_distance = np.linalg.norm(center_delta, axis=2)
    invalid_center = ~np.all(np.isfinite(centers), axis=1)
    if np.any(invalid_center):
        center_distance[invalid_center, :] = np.nan
        center_distance[:, invalid_center] = np.nan

    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    union = area[:, None] + area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-9)
    small = np.minimum(area[:, None], area[None, :])
    small_overlap = np.divide(inter, small, out=np.zeros_like(inter), where=small > 1e-9)
    invalid_box = ~np.all(np.isfinite(boxes), axis=1)
    if np.any(invalid_box):
        iou[invalid_box, :] = 0.0
        iou[:, invalid_box] = 0.0
        small_overlap[invalid_box, :] = 0.0
        small_overlap[:, invalid_box] = 0.0
    return center_distance, iou, small_overlap


def _pairwise_pose_distance_px(
    detections: Sequence[Detection],
    min_conf: float = 0.08,
    min_points: int = 3,
) -> np.ndarray:
    """批量计算检测间关键点中位距离；无足够共同点保持inf。"""
    dets = list(detections)
    count = len(dets)
    out = np.full((count, count), np.inf, dtype=np.float64)
    if count < 2:
        if count == 1:
            out[0, 0] = 0.0
        return out
    point_count = len(KEYPOINT_NAMES)
    points = np.full((count, point_count, 2), np.nan, dtype=np.float64)
    conf = np.zeros((count, point_count), dtype=np.float64)
    for idx, det in enumerate(dets):
        p = np.asarray(det.keypoints_px, dtype=np.float64)
        c = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
        n = min(len(p), len(c), point_count)
        if n > 0:
            points[idx, :n] = p[:n]
            conf[idx, :n] = c[:n]
    valid = np.all(np.isfinite(points), axis=2) & (conf >= float(min_conf))
    common = valid[:, None, :] & valid[None, :, :]
    delta = points[:, None, :, :] - points[None, :, :, :]
    dist = np.linalg.norm(delta, axis=3)
    common_count = np.sum(common, axis=2)
    med = np.ma.median(np.ma.array(dist, mask=~common), axis=2).filled(np.inf)
    ok = common_count >= int(min_points)
    out[ok] = med[ok]
    np.fill_diagonal(out, 0.0)
    return out


def _detection_pose_distance_px(a: Detection, b: Detection) -> float:
    """两套关键点的中位像素距离。仅比较双方都可靠的点。"""
    pa = np.asarray(a.keypoints_px, dtype=np.float64)
    pb = np.asarray(b.keypoints_px, dtype=np.float64)
    ca = np.asarray(a.keypoint_conf, dtype=np.float64).reshape(-1)
    cb = np.asarray(b.keypoint_conf, dtype=np.float64).reshape(-1)
    n = min(len(pa), len(pb), len(ca), len(cb), len(KEYPOINT_NAMES))
    if n <= 0:
        return float("inf")
    valid = (
        np.all(np.isfinite(pa[:n]), axis=1)
        & np.all(np.isfinite(pb[:n]), axis=1)
        & (ca[:n] >= 0.08)
        & (cb[:n] >= 0.08)
    )
    if valid.sum() < 3:
        return float("inf")
    return float(np.median(np.linalg.norm(pa[:n][valid] - pb[:n][valid], axis=1)))


def suppress_duplicate_detections(
    detections: Sequence[Detection],
    config: Optional[Mapping[str, Any]] = None,
) -> List[Detection]:
    """在身份分配前删除“同一只鼠的重复实例”，但尽量不合并真正重叠的两只鼠。

    只有当框高度重叠、中心几乎一致，并且关键点也近乎相同才判定为重复。
    打斗时两只真实小鼠虽然框可能重叠，但两套姿态通常不会逐点一致，因此会保留。
    """
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", True)):
        return list(detections)
    iou_th = float(cfg.get("iou_threshold", 0.72))
    small_overlap_th = float(cfg.get("small_box_overlap_threshold", 0.86))
    center_bl_th = float(cfg.get("center_distance_body_lengths", 0.20))
    pose_bl_th = float(cfg.get("pose_distance_body_lengths", 0.20))
    ordered = sorted(
        list(detections),
        key=lambda d: (
            float(d.pose_quality),
            float(np.nanmean(d.keypoint_conf)) if len(d.keypoint_conf) else 0.0,
            float(d.box_conf),
        ),
        reverse=True,
    )
    center_matrix, iou_matrix, small_overlap_matrix = _pairwise_bbox_geometry(ordered)
    pose_matrix = _pairwise_pose_distance_px(ordered, min_conf=0.08, min_points=3)
    body_values = np.asarray([float(det.body_length_px) for det in ordered], dtype=np.float64)
    kept_indices: List[int] = []
    for det_index, det in enumerate(ordered):
        duplicate = False
        for good_index in kept_indices:
            with np.errstate(all="ignore"):
                body = max(float(np.nanmedian([body_values[det_index], body_values[good_index]])), 8.0)
            center_d = float(center_matrix[det_index, good_index]) / body
            iou = float(iou_matrix[det_index, good_index])
            small_overlap = float(small_overlap_matrix[det_index, good_index])
            pose_d = float(pose_matrix[det_index, good_index]) / body
            good = ordered[good_index]
            same_raw = (
                det.raw_track_id is not None
                and good.raw_track_id is not None
                and det.raw_track_id == good.raw_track_id
            )
            if (
                center_d <= center_bl_th
                and (iou >= iou_th or small_overlap >= small_overlap_th or same_raw)
                and (pose_d <= pose_bl_th or same_raw)
            ):
                duplicate = True
                break
        if not duplicate:
            kept_indices.append(det_index)
    kept = [ordered[index] for index in kept_indices]
    return sorted(kept, key=lambda d: (float(d.center_px[0]), float(d.center_px[1])))


def _point_inside_bbox(point: np.ndarray, bbox: np.ndarray) -> bool:
    if not finite_point(point):
        return False
    b = np.asarray(bbox, dtype=np.float64).reshape(-1)
    return bool(b.size >= 4 and b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3])


def _bbox_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 4 or b.size < 4:
        return False
    return bool(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))


class OcclusionClusterManager:
    """多鼠接触/打斗簇管理器。

    作用：
    - 在两只或多只鼠进入近距离后冻结已有逻辑ID集合；
    - 检测“预期有N只、YOLO只给出M<N只”的实例丢失；
    - 禁止簇内临时检测创建新逻辑ID；
    - 为局部二次推理提供ROI；
    - 在检测数下降、框重叠且运动剧烈时给攻击粗筛提供兜底提示。
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.track_max_age = int(cfg.get("track_max_age_frames", 20))
        self.enter_distance_bl = float(cfg.get("enter_distance_body_lengths", 1.20))
        self.release_distance_bl = float(cfg.get("release_distance_body_lengths", 1.65))
        self.release_stable_frames = int(cfg.get("release_stable_frames", 5))
        self.region_padding_ratio = float(cfg.get("region_padding_ratio", 0.40))
        self.overlap_iou_threshold = float(cfg.get("overlap_iou_threshold", 0.12))
        self.merged_box_area_ratio = float(cfg.get("merged_box_area_ratio", 1.45))
        # A detection is considered available to a predicted member only inside this local gate.
        self.observation_match_gate_bl = float(cfg.get("observation_match_gate_body_lengths", 0.72))
        # IoU provides a fallback when the center of a merged detection lies between two mice.
        self.observation_match_iou = float(cfg.get("observation_match_iou", 0.05))
        # A merged box must cover at least two predicted members, not merely be naturally large.
        self.merged_member_iou = float(cfg.get("merged_member_iou", 0.10))
        # A reappearing mouse may move quickly after a fight, so visibility uses a wider pair-local box.
        self.visibility_region_padding_ratio = float(cfg.get("visibility_region_padding_ratio", 0.60))
        # Nearly coincident old tracks are usually duplicate IDs for one mouse, not two hidden mice.
        self.min_entry_separation_bl = float(cfg.get("min_entry_separation_body_lengths", 0.30))
        self.attack_min_active_frames = int(cfg.get("attack_min_active_frames", 2))
        self.attack_motion_bl_per_frame = float(cfg.get("attack_motion_body_lengths_per_frame", 0.08))
        self.recovery_cooldown_frames = int(cfg.get("recovery_cooldown_frames", 3))
        self.states: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.next_cluster_id = 1
        self.last_recovery_frame: Dict[int, int] = defaultdict(lambda: -10**9)
        self.debug_rows: List[Dict[str, Any]] = []
        # Per-frame detection geometry shared by all candidate mouse groups.
        # It is rebuilt when local ROI recovery changes the detection objects.
        self._visibility_detection_signature: Optional[Tuple[int, Tuple[int, ...]]] = None
        self._visibility_detection_centers = np.empty((0, 2), dtype=np.float64)
        self._visibility_detection_boxes = np.empty((0, 4), dtype=np.float64)
        self._visibility_detection_areas = np.empty(0, dtype=np.float64)
        self._visibility_detection_iou = np.empty((0, 0), dtype=np.float64)

    @staticmethod
    def _predicted_bbox(track: IdentityTrack, frame: int) -> Optional[np.ndarray]:
        if track.last_bbox_xyxy is None:
            return None
        dt = max(frame - track.last_frame, 0)
        shift = track.velocity_px_per_frame * min(dt, 12)
        box = np.asarray(track.last_bbox_xyxy, dtype=np.float64).copy()
        box[[0, 2]] += shift[0]
        box[[1, 3]] += shift[1]
        return box

    @staticmethod
    def _expand_bbox(box: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
        b = np.asarray(box, dtype=np.float64).copy()
        w = max(b[2] - b[0], 1.0)
        h = max(b[3] - b[1], 1.0)
        b[0] -= w * ratio
        b[2] += w * ratio
        b[1] -= h * ratio
        b[3] += h * ratio
        b[0] = np.clip(b[0], 0, max(width - 1, 0))
        b[2] = np.clip(b[2], 0, max(width - 1, 0))
        b[1] = np.clip(b[1], 0, max(height - 1, 0))
        b[3] = np.clip(b[3], 0, max(height - 1, 0))
        return b

    @staticmethod
    def _components(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
        graph: Dict[int, set[int]] = {int(n): set() for n in nodes}
        for a, b in edges:
            graph[int(a)].add(int(b))
            graph[int(b)].add(int(a))
        seen, comps = set(), []
        for n in nodes:
            if n in seen:
                continue
            stack, comp = [int(n)], []
            seen.add(int(n))
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if len(comp) >= 2:
                comps.append(sorted(comp))
        return comps

    def _member_detection_evidence(
        self,
        members: Sequence[int],
        tracks_map: Mapping[int, IdentityTrack],
        assigner: "StableIdentityAssigner",
        detections: Sequence[Detection],
        frame: int,
    ) -> Dict[str, Any]:
        """Measure physical visibility for one candidate group with one-to-one matching."""
        member_ids = [int(lid) for lid in members]
        detection_signature = (int(frame), tuple(id(det) for det in detections))
        if detection_signature != self._visibility_detection_signature:
            detection_count = len(detections)
            detection_centers = np.asarray(
                [np.asarray(det.center_px, dtype=np.float64).reshape(-1)[:2]
                 for det in detections],
                dtype=np.float64,
            ).reshape(detection_count, 2)
            detection_boxes = np.asarray(
                [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4]
                 for det in detections],
                dtype=np.float64,
            ).reshape(detection_count, 4)
            detection_areas = np.maximum(
                detection_boxes[:, 2] - detection_boxes[:, 0], 0.0
            ) * np.maximum(
                detection_boxes[:, 3] - detection_boxes[:, 1], 0.0
            )
            if detection_count:
                x1 = np.maximum(detection_boxes[:, None, 0], detection_boxes[None, :, 0])
                y1 = np.maximum(detection_boxes[:, None, 1], detection_boxes[None, :, 1])
                x2 = np.minimum(detection_boxes[:, None, 2], detection_boxes[None, :, 2])
                y2 = np.minimum(detection_boxes[:, None, 3], detection_boxes[None, :, 3])
                intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
                union = detection_areas[:, None] + detection_areas[None, :] - intersection
                detection_iou = np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 1.0e-9,
                )
                np.fill_diagonal(detection_iou, 0.0)
            else:
                detection_iou = np.empty((0, 0), dtype=np.float64)
            self._visibility_detection_signature = detection_signature
            self._visibility_detection_centers = detection_centers
            self._visibility_detection_boxes = detection_boxes
            self._visibility_detection_areas = detection_areas
            self._visibility_detection_iou = detection_iou

        detection_centers = self._visibility_detection_centers
        detection_boxes = self._visibility_detection_boxes
        detection_areas = self._visibility_detection_areas
        detection_iou = self._visibility_detection_iou
        member_count = len(member_ids)
        predicted_boxes = np.full((member_count, 4), np.nan, dtype=np.float64)
        predicted_centers = np.full((member_count, 2), np.nan, dtype=np.float64)
        bodies = np.full(member_count, 8.0, dtype=np.float64)
        valid_rows: List[int] = []
        for row, lid in enumerate(member_ids):
            box = self._predicted_bbox(tracks_map[lid], frame)
            if box is not None:
                values = np.asarray(box, dtype=np.float64).reshape(-1)
                if values.size >= 4 and np.all(np.isfinite(values[:4])):
                    predicted_boxes[row] = values[:4]
                    valid_rows.append(row)
            center = np.asarray(
                assigner._prediction(tracks_map[lid], frame), dtype=np.float64
            ).reshape(-1)
            if center.size >= 2:
                predicted_centers[row] = center[:2]
            body = float(tracks_map[lid].body_length_px)
            if np.isfinite(body):
                bodies[row] = max(body, 8.0)
        if not valid_rows or not detections:
            return {
                "observed_count": 0,
                "observed_indices": [],
                "deficit": True,
                "merged_like": False,
                "merged_member_count": 0,
                "max_iou": 0.0,
                "max_det_area_ratio": 0.0,
                "locally_visible_count": 0,
            }

        # Build all member-to-detection geometry in vectorized arrays.  Every
        # nearby pair now reuses the same detection centers/boxes/IoUs instead
        # of invoking Python properties and scalar IoU helpers repeatedly.
        center_delta = predicted_centers[:, None, :] - detection_centers[None, :, :]
        distance_bl = np.linalg.norm(center_delta, axis=2) / bodies[:, None]
        x1 = np.maximum(predicted_boxes[:, None, 0], detection_boxes[None, :, 0])
        y1 = np.maximum(predicted_boxes[:, None, 1], detection_boxes[None, :, 1])
        x2 = np.minimum(predicted_boxes[:, None, 2], detection_boxes[None, :, 2])
        y2 = np.minimum(predicted_boxes[:, None, 3], detection_boxes[None, :, 3])
        intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        predicted_area = np.maximum(
            predicted_boxes[:, 2] - predicted_boxes[:, 0], 0.0
        ) * np.maximum(
            predicted_boxes[:, 3] - predicted_boxes[:, 1], 0.0
        )
        union = predicted_area[:, None] + detection_areas[None, :] - intersection
        track_detection_iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 1.0e-9,
        )
        eligible = (
            (distance_bl <= self.observation_match_gate_bl)
            | (track_detection_iou >= self.observation_match_iou)
        )
        valid_row_mask = np.zeros(member_count, dtype=bool)
        valid_row_mask[np.asarray(valid_rows, dtype=np.int64)] = True
        eligible &= valid_row_mask[:, None]
        cost = np.where(
            eligible,
            distance_bl + 0.15 * (1.0 - track_detection_iou),
            1.0e6,
        )
        relevant_mask = np.any(eligible, axis=0)

        # A split mouse can move beyond its stale ID prediction while remaining visibly distinct.
        valid_boxes = predicted_boxes[np.asarray(valid_rows, dtype=np.int64)]
        union = np.array([
            float(np.min(valid_boxes[:, 0])),
            float(np.min(valid_boxes[:, 1])),
            float(np.max(valid_boxes[:, 2])),
            float(np.max(valid_boxes[:, 3])),
        ], dtype=np.float64)
        union_width = max(float(union[2] - union[0]), 1.0)
        union_height = max(float(union[3] - union[1]), 1.0)
        local_box = union.copy()
        local_box[[0, 2]] += np.array([-union_width, union_width]) * self.visibility_region_padding_ratio
        local_box[[1, 3]] += np.array([-union_height, union_height]) * self.visibility_region_padding_ratio
        local_mask = (
            (detection_centers[:, 0] >= local_box[0])
            & (detection_centers[:, 0] <= local_box[2])
            & (detection_centers[:, 1] >= local_box[1])
            & (detection_centers[:, 1] <= local_box[3])
        )
        relevant_mask |= local_mask
        relevant_indices = np.flatnonzero(relevant_mask).astype(int).tolist()

        # Count only distinct detections that can be assigned to distinct predicted members.
        observed_count = 0
        if relevant_indices:
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(cost)
                observed_count = sum(float(cost[row, col]) < 1e6 for row, col in zip(rows, cols))
            else:
                used_rows: set[int] = set()
                used_cols: set[int] = set()
                candidates = sorted(
                    (float(cost[row, col]), row, col)
                    for row in range(cost.shape[0])
                    for col in range(cost.shape[1])
                    if float(cost[row, col]) < 1e6
                )
                for _, row, col in candidates:
                    if row not in used_rows and col not in used_cols:
                        used_rows.add(row)
                        used_cols.add(col)
                        observed_count += 1
        locally_visible_count = min(int(np.count_nonzero(local_mask)), len(member_ids))
        observed_count = max(observed_count, locally_visible_count)

        relevant_array = np.asarray(sorted(relevant_indices), dtype=np.int64)
        max_iou = (
            float(np.max(detection_iou[np.ix_(relevant_array, relevant_array)]))
            if relevant_array.size >= 2
            else 0.0
        )

        track_areas = np.maximum(predicted_area[np.asarray(valid_rows, dtype=np.int64)], 1.0)
        median_area = float(np.median(track_areas)) if track_areas.size else 1.0
        max_covered_members = 0
        max_det_area_ratio = 0.0
        if relevant_array.size:
            valid_row_array = np.asarray(valid_rows, dtype=np.int64)
            relevant_boxes = detection_boxes[relevant_array]
            member_centers = predicted_centers[valid_row_array]
            center_covered = (
                (member_centers[None, :, 0] >= relevant_boxes[:, None, 0])
                & (member_centers[None, :, 0] <= relevant_boxes[:, None, 2])
                & (member_centers[None, :, 1] >= relevant_boxes[:, None, 1])
                & (member_centers[None, :, 1] <= relevant_boxes[:, None, 3])
            )
            box_covered = (
                track_detection_iou[np.ix_(valid_row_array, relevant_array)].T
                >= self.merged_member_iou
            )
            covered_counts = np.count_nonzero(center_covered | box_covered, axis=1)
            max_covered_members = int(np.max(covered_counts, initial=0))
            matching = covered_counts == max_covered_members
            area_ratios = np.maximum(detection_areas[relevant_array], 1.0) / median_area
            if np.any(matching):
                max_det_area_ratio = float(np.max(area_ratios[matching]))

        expected_count = len(member_ids)
        deficit = observed_count < expected_count
        merged_like = (
            max_covered_members >= 2
            and max_det_area_ratio >= self.merged_box_area_ratio
        )
        return {
            "observed_count": int(observed_count),
            "observed_indices": sorted(relevant_indices),
            "deficit": bool(deficit),
            "merged_like": bool(merged_like),
            "merged_member_count": int(max_covered_members),
            "max_iou": float(max_iou),
            "max_det_area_ratio": float(max_det_area_ratio),
            "locally_visible_count": int(locally_visible_count),
        }

    def build_context(
        self,
        assigner: "StableIdentityAssigner",
        detections: Sequence[Detection],
        frame: int,
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "regions": [], "frozen_ids": set(), "forbidden_new_regions": [],
                "attack_pairs": set(), "recovery_regions": [], "id_to_cluster": {},
            }
        height, width = int(frame_shape[0]), int(frame_shape[1])
        # 记忆分配器会把TMP临时轨迹也纳入簇计算（cluster_tracks属性）；
        # 旧分配器没有该属性，回退为仅固定槽轨迹。
        tracks_map = getattr(assigner, "cluster_tracks", None) or assigner.tracks
        recent_ids = [
            lid for lid, tr in tracks_map.items()
            if frame - tr.last_frame <= self.track_max_age and finite_point(tr.last_center_px)
        ]
        # The same two-member group is evaluated once while building edges and
        # again when materializing its component. Cache within this immutable
        # pre-assignment frame context; no state or threshold is changed.
        evidence_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

        def evidence_for(members: Sequence[int]) -> Dict[str, Any]:
            key = tuple(sorted(int(member) for member in members))
            cached = evidence_cache.get(key)
            if cached is None:
                cached = self._member_detection_evidence(
                    key, tracks_map, assigner, detections, frame
                )
                evidence_cache[key] = cached
            return cached

        active_state_pairs = {
            tuple(sorted((int(a), int(b))))
            for state_members in self.states
            for index, a in enumerate(state_members)
            for b in state_members[index + 1 :]
        }
        # Build edges only from verified two-mouse merge evidence. Distance alone is not an occlusion.
        edges: List[Tuple[int, int]] = []
        for i, a in enumerate(recent_ids):
            ta = tracks_map[a]
            pa = assigner._prediction(ta, frame)
            for b in recent_ids[i + 1:]:
                tb = tracks_map[b]
                pb = assigner._prediction(tb, frame)
                body = max(float(np.nanmedian([ta.body_length_px, tb.body_length_px])), 8.0)
                separation_bl = point_distance(pa, pb) / body
                pair_was_active = tuple(sorted((int(a), int(b)))) in active_state_pairs
                entry_separation_ok = separation_bl >= self.min_entry_separation_bl or pair_was_active
                if separation_bl > self.enter_distance_bl or not entry_separation_ok:
                    continue
                pair_evidence = evidence_for((a, b))
                if pair_evidence["deficit"] and pair_evidence["merged_like"]:
                    edges.append((a, b))
        components = self._components(recent_ids, edges)

        current_keys = set()
        regions: List[Dict[str, Any]] = []
        frozen_ids: set[int] = set()
        forbidden: List[np.ndarray] = []
        attack_pairs: set[Tuple[int, int]] = set()
        recovery_regions: List[Dict[str, Any]] = []
        id_to_cluster: Dict[int, int] = {}

        for members_list in components:
            members = tuple(sorted(members_list))
            current_keys.add(members)
            state = self.states.get(members)
            if state is None:
                state = {
                    "cluster_id": self.next_cluster_id,
                    "active_frames": 0,
                    "release_count": 0,
                }
                self.next_cluster_id += 1
                self.states[members] = state

            boxes = []
            body_lengths = []
            speeds_bl = []
            max_pair_bl = 0.0
            for lid in members:
                tr = tracks_map[lid]
                pb = self._predicted_bbox(tr, frame)
                if pb is not None:
                    boxes.append(pb)
                body_lengths.append(max(float(tr.body_length_px), 8.0))
                speeds_bl.append(float(np.linalg.norm(tr.velocity_px_per_frame)) / max(float(tr.body_length_px), 8.0))
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    ta, tb = tracks_map[a], tracks_map[b]
                    body = max(float(np.nanmedian([ta.body_length_px, tb.body_length_px])), 8.0)
                    max_pair_bl = max(max_pair_bl, point_distance(assigner._prediction(ta, frame), assigner._prediction(tb, frame)) / body)
            if not boxes:
                continue
            union = np.array([
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            ], dtype=np.float64)
            region_box = self._expand_bbox(union, self.region_padding_ratio, width, height)
            evidence = evidence_for(members)
            obs_indices = list(evidence["observed_indices"])
            obs = [detections[idx] for idx in obs_indices]
            observed_count = int(evidence["observed_count"])
            expected_count = len(members)

            max_iou = float(evidence["max_iou"])
            deficit = bool(evidence["deficit"])
            merged_like = bool(evidence["merged_like"])
            separated = max_pair_bl >= self.release_distance_bl and observed_count >= expected_count
            if separated:
                state["release_count"] = int(state.get("release_count", 0)) + 1
            else:
                state["release_count"] = 0
            if state["release_count"] >= self.release_stable_frames:
                continue

            # Evidence must be consecutive; isolated detector misses must never accumulate into CXX.
            previous_evidence_frame = int(state.get("last_frame", frame - 1))
            if previous_evidence_frame != frame - 1:
                state["active_frames"] = 0
            state["active_frames"] = int(state.get("active_frames", 0)) + 1
            state["last_frame"] = int(frame)
            state["bbox"] = region_box
            state["expected_count"] = expected_count
            state["observed_count"] = observed_count

            cluster_id = int(state["cluster_id"])
            frozen_ids.update(members)
            forbidden.append(region_box)
            for lid in members:
                id_to_cluster[int(lid)] = cluster_id

            motion = float(np.mean(speeds_bl)) if speeds_bl else 0.0
            attack_hint = bool(
                state["active_frames"] >= self.attack_min_active_frames
                and (deficit or merged_like or max_iou >= self.overlap_iou_threshold)
                and (motion >= self.attack_motion_bl_per_frame or deficit)
            )
            if attack_hint:
                for i, a in enumerate(members):
                    for b in members[i + 1:]:
                        attack_pairs.add(tuple(sorted((int(a), int(b)))))

            region = {
                "cluster_id": cluster_id,
                "members": members,
                "bbox": region_box,
                "expected_count": expected_count,
                "observed_count": observed_count,
                "observed_indices": obs_indices,
                "deficit": bool(deficit),
                "merged_like": bool(merged_like),
                "merged_member_count": int(evidence["merged_member_count"]),
                "max_det_area_ratio": float(evidence["max_det_area_ratio"]),
                "locally_visible_count": int(evidence["locally_visible_count"]),
                "ambiguity_frames": int(state["active_frames"]),
                "max_iou": float(max_iou),
                "motion_bl_per_frame": motion,
                "active_frames": int(state["active_frames"]),
                "attack_hint": attack_hint,
            }
            regions.append(region)
            recovery_requested = False
            if (
                (deficit or merged_like)
                and frame - self.last_recovery_frame[cluster_id] >= self.recovery_cooldown_frames
            ):
                recovery_regions.append(region)
                self.last_recovery_frame[cluster_id] = frame
                recovery_requested = True

            self.debug_rows.append({
                "frame": frame,
                "cluster_id": cluster_id,
                "members": ",".join(map(str, members)),
                "expected_count": expected_count,
                "observed_count": observed_count,
                "deficit": int(deficit),
                "merged_like": int(merged_like),
                "merged_member_count": int(evidence["merged_member_count"]),
                "max_det_area_ratio": float(evidence["max_det_area_ratio"]),
                "locally_visible_count": int(evidence["locally_visible_count"]),
                "ambiguity_frames": int(state["active_frames"]),
                "max_iou": max_iou,
                "motion_bl_per_frame": motion,
                "active_frames": int(state["active_frames"]),
                "attack_hint": int(attack_hint),
                "recovery_requested": int(recovery_requested),
            })

        # 清理已不再出现的状态。
        for key in list(self.states):
            if key not in current_keys and frame - int(self.states[key].get("last_frame", frame)) > self.release_stable_frames:
                self.states.pop(key, None)

        return {
            "regions": regions,
            "frozen_ids": frozen_ids,
            "forbidden_new_regions": forbidden,
            "attack_pairs": attack_pairs,
            "recovery_regions": recovery_regions,
            "id_to_cluster": id_to_cluster,
        }

    @staticmethod
    def detection_inside_forbidden(det: Detection, context: Optional[Mapping[str, Any]]) -> bool:
        if not context:
            return False
        return any(
            _point_inside_bbox(det.center_px, b) or _bbox_intersects(det.bbox_xyxy, b)
            for b in context.get("forbidden_new_regions", [])
        )


# ----------------------------- 多鼠身份连续性 -----------------------------


class StableIdentityAssigner:
    """多鼠稳定身份分配器 v1.6。

    设计原则：
    1. 颜色不参与ID代价；白鼠只在外观ROI中做灰度反转+CLAHE。
    2. 融合预测中心、速度、姿态、虚拟锚点、朝向、体型和结构纹理。
    3. 匈牙利算法只提出候选；唯一提交仲裁器用收益、驻留和冷却决定是否提交。
    4. 模糊/近距离阶段宁可短时使用预测框，也不立即把ID交给另一只鼠。
    5. 远距离和高代价匹配硬拒绝；YOLO原始Track ID仅为弱证据。
    """

    INF_COST = 1e6

    def __init__(self, config: Mapping[str, Any], max_mice: int = 20) -> None:
        self.mode = str(config.get("mode", "hybrid"))
        self.max_mice = int(max_mice)
        self.candidate_extra = int(config.get("candidate_extra", 10))
        self.max_missing_frames = int(config.get("max_missing_frames", 90))
        self.prediction_output_frames = int(config.get("prediction_output_frames", 12))
        self.max_jump_body_lengths = float(config.get("max_jump_body_lengths", 2.8))
        self.max_assignment_cost = float(config.get("max_assignment_cost", 1.20))
        self.raw_id_mismatch_penalty = float(config.get("raw_id_mismatch_penalty", 0.05))
        self.raw_id_match_reward = float(config.get("raw_id_match_reward", 0.03))
        self.velocity_weight = float(config.get("velocity_prediction_weight", 1.0))

        self.appearance_ema_alpha = float(config.get("appearance_ema_alpha", 0.22))
        self.long_term_memory_enabled = bool(config.get("long_term_memory_enabled", True))  # 接收主程序按视频时长计算的长期记忆开关。
        self.appearance_long_ema_alpha = float(config.get("appearance_long_ema_alpha", 0.035))
        self.appearance_long_max_iou = float(config.get("appearance_long_max_iou", 0.06))
        self.appearance_long_min_pose_quality = float(config.get("appearance_long_min_pose_quality", 0.55))
        self.pose_ema_alpha = float(config.get("pose_ema_alpha", 0.30))
        self.ignore_color_for_identity = bool(config.get("ignore_color_for_identity", True))

        self.close_contact_body_lengths = float(config.get("close_contact_body_lengths", 1.30))
        self.close_contact_gate_scale = float(config.get("close_contact_gate_scale", 0.78))
        self.missing_lock_decay = float(config.get("missing_lock_decay", 0.008))

        commit = dict(config.get("commit_arbiter", {}))
        self.commit_enabled = bool(commit.get("enabled", True))
        self.commit_cooldown_frames = int(commit.get("cooldown_frames", 60))
        self.commit_dwell_close = int(commit.get("dwell_close", 12))
        self.commit_dwell_normal = int(commit.get("dwell_normal", 7))
        self.commit_dwell_far = int(commit.get("dwell_far", 4))
        self.commit_min_gain_close = float(commit.get("min_gain_close", 0.12))
        self.commit_min_gain_normal = float(commit.get("min_gain_normal", 0.08))
        self.commit_min_gain_far = float(commit.get("min_gain_far", 0.05))
        self.commit_immediate_gain = float(commit.get("immediate_gain", 0.28))
        self.commit_lock_threshold = float(commit.get("lock_threshold", 0.55))
        self.commit_missing_bypass_frames = int(commit.get("missing_bypass_frames", 6))

        weights = dict(config.get("weights", {}))
        self.w_center = float(weights.get("center", 0.34))
        self.w_velocity = float(weights.get("velocity", 0.17))
        self.w_pose = float(weights.get("pose", 0.12))
        self.w_anchor = float(weights.get("anchor", 0.13))
        self.w_heading = float(weights.get("heading", 0.06))
        self.w_size = float(weights.get("size", 0.07))
        self.w_appearance = float(weights.get("appearance", 0.11))
        # 保留字段兼容旧配置，但默认和推荐均为0。
        self.w_white = 0.0 if self.ignore_color_for_identity else float(weights.get("white", 0.0))

        self.tracks: Dict[int, IdentityTrack] = {}
        self.raw_to_logical: Dict[int, int] = {}
        self.debug_records: List[IdentityDebug] = []
        self.pending_commits: Dict[int, Dict[str, Any]] = {}
        self.last_commit_by_track: Dict[int, int] = defaultdict(lambda: -10**9)
        self.last_commit_by_pair: Dict[Tuple[int, int], int] = defaultdict(lambda: -10**9)

        # 新身份不得由接触簇中的单帧临时检测直接创建。
        self.new_track_confirm_frames = int(config.get("new_track_confirm_frames", 8))
        self.new_track_max_gap_frames = int(config.get("new_track_max_gap_frames", 2))
        self.new_track_min_separation_bl = float(config.get("new_track_min_separation_body_lengths", 0.45))
        self.new_track_dup_iou = float(config.get("new_track_duplicate_iou", 0.20))
        self.pending_new_tracks: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        # 接触簇内只允许“明显唯一”的检测更新旧ID，否则维持内部预测。
        self.cluster_accept_cost = float(config.get("cluster_accept_cost", 0.48))
        self.cluster_min_margin = float(config.get("cluster_min_assignment_margin", 0.10))
        self.cluster_preserve_raw_id = bool(config.get("cluster_preserve_raw_id", True))

        # v1.42.1: result-preserving cascade gate.  This gate is deliberately
        # identical to the distance hard gate already used at the beginning of
        # _cost(); it only avoids evaluating pose/anchor/heading/appearance for
        # pairs that _cost() would return as INF_COST anyway.
        cascade_cfg = dict(config.get("cascade_matching", {}))
        self.cascade_matching_enabled = bool(cascade_cfg.get("enabled", True))
        self.last_fast_gate_candidate_count = 0
        self.last_fast_gate_total_count = 0

    def _expire(self, frame: int) -> None:
        stale = [
            lid for lid, track in self.tracks.items()
            if frame - track.last_frame > self.max_missing_frames
        ]
        for lid in stale:
            raw = self.tracks[lid].raw_track_id
            if raw is not None and self.raw_to_logical.get(raw) == lid:
                self.raw_to_logical.pop(raw, None)
            self.tracks.pop(lid, None)
            self.pending_commits.pop(lid, None)
        for signature in list(self.pending_new_tracks):
            if frame - int(self.pending_new_tracks[signature].get("last_frame", frame)) > self.new_track_max_gap_frames:
                self.pending_new_tracks.pop(signature, None)

    def _prediction(self, track: IdentityTrack, frame: int) -> np.ndarray:
        dt = max(frame - track.last_frame, 0)
        pred_dt = min(dt, 12)
        return track.last_center_px + self.velocity_weight * track.velocity_px_per_frame * pred_dt

    def build_fast_gate(
        self,
        tracks: Sequence[IdentityTrack],
        detections: Sequence[Detection],
        frame: int,
    ) -> np.ndarray:
        """Return the exact pre-cost candidate mask used by the legacy cost gate.

        No new heuristic threshold is introduced here: a False cell is a pair
        for which ``_distance_and_gate`` already makes ``_cost`` return
        ``INF_COST``.  Therefore enabling the cascade cannot create a new
        rejection relative to the legacy StableIdentityAssigner.
        """
        nt, nd = len(tracks), len(detections)
        if nt == 0 or nd == 0:
            return np.zeros((nt, nd), dtype=bool)
        if not self.cascade_matching_enabled:
            return np.ones((nt, nd), dtype=bool)
        mask = np.zeros((nt, nd), dtype=bool)
        for r, track in enumerate(tracks):
            for c, det in enumerate(detections):
                _, _, allowed = self._distance_and_gate(track, det, frame)
                mask[r, c] = bool(allowed)
        self.last_fast_gate_candidate_count = int(np.count_nonzero(mask))
        self.last_fast_gate_total_count = int(mask.size)
        return mask

    @staticmethod
    def _feature_l1(a: Optional[np.ndarray], b: Optional[np.ndarray], fallback: float = 0.45, scale: float = 2.2) -> float:
        if a is None or b is None:
            return fallback
        aa = np.asarray(a, dtype=np.float64).reshape(-1)
        bb = np.asarray(b, dtype=np.float64).reshape(-1)
        n = min(len(aa), len(bb))
        if n == 0:
            return fallback
        valid = np.isfinite(aa[:n]) & np.isfinite(bb[:n])
        if valid.sum() < 8:
            return fallback
        return float(np.clip(np.mean(np.abs(aa[:n][valid] - bb[:n][valid])) * scale, 0.0, 1.5))

    def _appearance_distance(self, track: IdentityTrack, detection: Detection) -> float:
        if detection.appearance_feature is None:
            return 0.45
        distances = []
        if track.appearance_feature is not None:
            distances.append(self._feature_l1(track.appearance_feature, detection.appearance_feature, 0.45, 2.0))
        if self.long_term_memory_enabled and track.appearance_feature_long is not None:  # 短视频禁止长期模板参与身份代价。
            distances.append(self._feature_l1(track.appearance_feature_long, detection.appearance_feature, 0.45, 2.0))
        return min(distances) if distances else 0.45

    def _appearance_long_distance(self, track: IdentityTrack, detection: Detection) -> Optional[float]:
        """只对照长期外观模板。长期模板仅在无遮挡高质量帧更新，
        是重激活、冲突仲裁和滑窗重拼接中最可信的身份证据。"""
        if not self.long_term_memory_enabled:  # 短视频不允许长期模板参与重激活或冲突仲裁。
            return None  # 返回无长期证据，让现有短时运动与姿态逻辑自然接管。
        if track.appearance_feature_long is None or detection.appearance_feature is None:
            return None
        return self._feature_l1(track.appearance_feature_long, detection.appearance_feature, 0.45, 2.0)

    @staticmethod
    def _pose_distance(track: IdentityTrack, detection: Detection) -> float:
        if track.normalized_pose is None or detection.normalized_pose is None:
            return 0.55
        a = np.asarray(track.normalized_pose, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(detection.normalized_pose, dtype=np.float64).reshape(-1, 2)
        n = min(len(a), len(b))
        valid = np.all(np.isfinite(a[:n]), axis=1) & np.all(np.isfinite(b[:n]), axis=1)
        if valid.sum() < 3:
            return 0.55
        d = np.linalg.norm(a[:n][valid] - b[:n][valid], axis=1)
        return float(np.clip(np.median(d) / 0.55, 0.0, 1.5))

    @staticmethod
    def _oks_pose_cost(track: IdentityTrack, detection: Detection) -> Optional[float]:
        """按关键点置信度加权的OKS代价（1-OKS）。

        相比归一化姿态的L2中位距，OKS按各关键点的自然尺度（鼻尖/耳根更严格，
        尾根更宽松）加权，并直接用检测置信度做权重，对飞点和遮挡点天然免疫。
        关键点不足时返回None，由调用方回退到归一化姿态距离。
        """
        tk = track.last_keypoints_px
        tc = track.last_keypoint_conf
        if tk is None or tc is None:
            return None
        dk = np.asarray(detection.keypoints_px, dtype=np.float64)
        dc = np.asarray(detection.keypoint_conf, dtype=np.float64).reshape(-1)
        tk = np.asarray(tk, dtype=np.float64)
        tc = np.asarray(tc, dtype=np.float64).reshape(-1)
        n = min(len(KEYPOINT_NAMES), len(tk), len(dk), len(tc), len(dc))
        if n < 4:
            return None
        valid = (
            np.all(np.isfinite(tk[:n]), axis=1) & np.all(np.isfinite(dk[:n]), axis=1)
            & (dk[:n, 0] > 0) & (dk[:n, 1] > 0)
            & np.isfinite(dc[:n]) & (dc[:n] >= 0.10)
            & np.isfinite(tc[:n]) & (tc[:n] >= 0.10)
        )
        if valid.sum() < 4:
            return None
        # 各关键点容差（相对体长）：头部小、躯干中、尾根大。
        # 取值以“相邻帧抖动（≤15%体长）几乎不惩罚、另一只鼠（≥70%体长）接近满分”为标定目标。
        kappa = np.array([0.20, 0.18, 0.18, 0.25, 0.30, 0.30, 0.40], dtype=np.float64)[:n]
        lengths = [v for v in (track.body_length_px, detection.body_length_px)
                   if np.isfinite(v) and v > 3]
        scale_px = max(float(np.nanmedian(lengths)) if lengths else 20.0, 8.0)
        d2 = np.sum((tk[:n][valid] - dk[:n][valid]) ** 2, axis=1)
        denom = 2.0 * (kappa[valid] * scale_px) ** 2
        per_kp = np.exp(-d2 / np.maximum(denom, 1e-6))
        weights = np.clip(dc[:n][valid] * tc[:n][valid], 0.01, 1.0)
        oks = float(np.average(per_kp, weights=weights))
        return float(np.clip(1.0 - oks, 0.0, 1.0))

    @staticmethod
    def _anchor_distance(track: IdentityTrack, detection: Detection) -> float:
        if track.anchor_feature is None or detection.anchor_feature is None:
            return 0.50
        a = np.asarray(track.anchor_feature, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(detection.anchor_feature, dtype=np.float64).reshape(-1, 2)
        n = min(len(a), len(b))
        valid = np.all(np.isfinite(a[:n]), axis=1) & np.all(np.isfinite(b[:n]), axis=1)
        if valid.sum() < 3:
            return 0.50
        weights = np.ones(n, dtype=np.float64)
        # nose、neck、tail权重更高。
        weights[: min(3, n)] = np.array([1.3, 1.5, 1.4])[: min(3, n)]
        d = np.linalg.norm(a[:n][valid] - b[:n][valid], axis=1)
        return float(np.clip(np.average(d, weights=weights[:n][valid]) / 0.45, 0.0, 1.5))

    @staticmethod
    def _heading_distance(track: IdentityTrack, detection: Detection) -> float:
        if track.heading_vector is None or detection.heading_vector is None:
            return 0.45
        a = np.asarray(track.heading_vector, dtype=np.float64)
        b = np.asarray(detection.heading_vector, dtype=np.float64)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 0.45
        return float((1.0 - np.clip(np.dot(a, b), -1.0, 1.0)) / 2.0)

    @staticmethod
    def _size_distance(track: IdentityTrack, detection: Detection) -> float:
        vals = []
        if np.isfinite(track.body_length_px) and track.body_length_px > 1 and np.isfinite(detection.body_length_px) and detection.body_length_px > 1:
            vals.append(abs(math.log(max(detection.body_length_px, 1.0) / max(track.body_length_px, 1.0))))
        box = np.asarray(detection.bbox_xyxy, dtype=np.float64)
        det_wh = np.array([max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)], dtype=np.float64)
        if track.bbox_wh is not None and np.all(np.isfinite(track.bbox_wh)):
            vals.extend(np.abs(np.log(det_wh / np.maximum(track.bbox_wh, 1.0))).tolist())
        return float(np.clip(np.mean(vals), 0.0, 1.5)) if vals else 0.4

    def _white_distance(self, track: IdentityTrack, detection: Detection) -> float:
        # 用户要求：颜色不参与ID判定。white_score仅选择是否对白鼠ROI反转。
        if self.ignore_color_for_identity:
            return 0.0
        if not np.isfinite(track.white_score) or not np.isfinite(detection.white_score):
            return 0.35
        return float(np.clip(abs(track.white_score - detection.white_score) / 0.55, 0.0, 1.5))

    def _is_hard_white_mismatch(self, track: IdentityTrack, detection: Detection) -> bool:
        return False

    def _nearest_other_distance_bl(self, track: IdentityTrack) -> float:
        best = np.inf
        for other_id, other in self.tracks.items():
            if other_id == track.logical_id or not finite_point(other.last_center_px):
                continue
            body = max(float(np.nanmedian([track.body_length_px, other.body_length_px])), 8.0)
            best = min(best, point_distance(track.last_center_px, other.last_center_px) / body)
        return float(best)

    def _distance_and_gate(self, track: IdentityTrack, detection: Detection, frame: int) -> Tuple[float, float, bool]:
        predicted = self._prediction(track, frame)
        lengths = [v for v in (track.body_length_px, detection.body_length_px) if np.isfinite(v) and v > 3]
        norm = max(float(np.nanmedian(lengths)) if lengths else 20.0, 8.0)
        d = point_distance(predicted, detection.center_px)
        if not np.isfinite(d):
            return self.INF_COST, norm, False
        distance_bl = d / norm
        dt = max(frame - track.last_frame, 1)
        speed_bl = float(np.linalg.norm(track.velocity_px_per_frame)) / norm
        gate = self.max_jump_body_lengths + min(dt - 1, 8) * 0.38 + min(speed_bl, 2.5) * 0.75
        near_other = self._nearest_other_distance_bl(track) <= self.close_contact_body_lengths
        if track.lock_strength >= 0.70 and dt <= 2:
            gate *= self.close_contact_gate_scale if near_other else 0.90
        return float(distance_bl), norm, bool(distance_bl <= gate)

    def _cost(self, track: IdentityTrack, detection: Detection, frame: int) -> float:
        distance_bl, _, allowed = self._distance_and_gate(track, detection, frame)
        if not allowed:
            return self.INF_COST
        dt = max(frame - track.last_frame, 1)
        actual_delta = detection.center_px - track.last_center_px
        expected_delta = track.velocity_px_per_frame * min(dt, 8)
        norm = max(float(np.nanmedian([track.body_length_px, detection.body_length_px])), 8.0)
        velocity_cost = float(np.clip(np.linalg.norm(actual_delta - expected_delta) / (norm * 2.0), 0.0, 1.5))
        center_cost = float(np.clip(distance_bl / max(self.max_jump_body_lengths, 1.0), 0.0, 1.5))
        pose_cost = self._pose_distance(track, detection)
        anchor_cost = self._anchor_distance(track, detection)
        heading_cost = self._heading_distance(track, detection)
        size_cost = self._size_distance(track, detection)
        appearance_cost = self._appearance_distance(track, detection)
        white_cost = self._white_distance(track, detection)
        cost = (
            self.w_center * center_cost
            + self.w_velocity * velocity_cost
            + self.w_pose * pose_cost
            + self.w_anchor * anchor_cost
            + self.w_heading * heading_cost
            + self.w_size * size_cost
            + self.w_appearance * appearance_cost
            + self.w_white * white_cost
        )
        # 原始Track ID只作很弱的证据。
        if track.raw_track_id is not None and detection.raw_track_id is not None:
            cost += -self.raw_id_match_reward if track.raw_track_id == detection.raw_track_id else self.raw_id_mismatch_penalty
        mapped = self.raw_to_logical.get(detection.raw_track_id) if detection.raw_track_id is not None else None
        if mapped is not None and mapped != track.logical_id:
            cost += self.raw_id_mismatch_penalty * 0.5
        return float(max(cost, 0.0))

    @staticmethod
    def _ema_array(old: Optional[np.ndarray], new: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
        if new is None:
            return old
        n = np.asarray(new, dtype=np.float64)
        if old is None or np.asarray(old).shape != n.shape:
            return n.copy()
        o = np.asarray(old, dtype=np.float64)
        valid_n = np.isfinite(n)
        out = o.copy()
        both = valid_n & np.isfinite(o)
        out[both] = (1.0 - alpha) * o[both] + alpha * n[both]
        out[valid_n & ~np.isfinite(o)] = n[valid_n & ~np.isfinite(o)]
        return out

    def _update_track(
        self,
        logical_id: int,
        detection: Detection,
        frame: int,
        freeze_appearance: bool = False,
        preserve_raw_id: bool = False,
        store: Optional[MutableMapping[int, "IdentityTrack"]] = None,
    ) -> None:
        # store默认self.tracks；记忆分配器用它把TMP临时轨迹写入独立存储。
        track_store: MutableMapping[int, IdentityTrack] = self.tracks if store is None else store
        center = detection.center_px.astype(np.float64)
        old = track_store.get(logical_id)
        if old is not None and finite_point(old.last_center_px):
            dt = max(frame - old.last_frame, 1)
            measured_velocity = (center - old.last_center_px) / dt
            speed = float(np.linalg.norm(measured_velocity))
            body = max(float(np.nanmedian([old.body_length_px, detection.body_length_px])), 8.0)
            beta = 0.72 if speed > body * 0.65 else 0.45
            velocity = beta * measured_velocity + (1.0 - beta) * old.velocity_px_per_frame
            hits = old.hits + 1
            lock_strength = min(1.0, old.lock_strength + 0.055)
            appearance_feature = old.appearance_feature
            appearance_feature_long = old.appearance_feature_long if self.long_term_memory_enabled else None  # 短视频不在轨迹更新间保留长期外观模板。
            brightness_score = old.brightness_score
            white_score = old.white_score
            is_white_candidate = old.is_white_candidate
            appearance_updates = old.appearance_updates
            normalized_pose = old.normalized_pose
            anchor_feature = old.anchor_feature
            heading_vector = old.heading_vector
            state = old.state
            clean_streak = old.clean_streak
        else:
            velocity = np.zeros(2, dtype=np.float64)
            hits = 1
            lock_strength = 0.10
            appearance_feature = None
            appearance_feature_long = None
            brightness_score = float("nan")
            white_score = float("nan")
            is_white_candidate = bool(detection.is_white_candidate)
            appearance_updates = 0
            normalized_pose = None
            anchor_feature = None
            heading_vector = None
            state = "tracked"
            clean_streak = 0

        if (not freeze_appearance) and detection.appearance_reliable and detection.appearance_feature is not None:
            appearance_feature = self._ema_array(appearance_feature, detection.appearance_feature, self.appearance_ema_alpha)
            # 长期模板只在几乎无遮挡、高姿态质量时更新，避免接触时被另一只鼠污染。
            if (self.long_term_memory_enabled  # 只有超过十分钟的视频才允许更新长期外观模板。
                    and detection.max_overlap_iou <= self.appearance_long_max_iou
                    and detection.pose_quality >= self.appearance_long_min_pose_quality):
                appearance_feature_long = self._ema_array(
                    appearance_feature_long, detection.appearance_feature, self.appearance_long_ema_alpha
                )
            if np.isfinite(detection.brightness_score):
                brightness_score = detection.brightness_score if not np.isfinite(brightness_score) else (
                    0.85 * brightness_score + 0.15 * detection.brightness_score
                )
            if np.isfinite(detection.white_score):
                white_score = detection.white_score if not np.isfinite(white_score) else (
                    0.85 * white_score + 0.15 * detection.white_score
                )
                is_white_candidate = bool(white_score >= 0.55)
            appearance_updates += 1

        if detection.normalized_pose is not None and detection.pose_quality >= 0.35:
            normalized_pose = self._ema_array(normalized_pose, detection.normalized_pose, self.pose_ema_alpha)
        if detection.anchor_feature is not None and detection.pose_quality >= 0.28:
            anchor_feature = self._ema_array(anchor_feature, detection.anchor_feature, self.pose_ema_alpha)
        if detection.heading_vector is not None:
            if heading_vector is None:
                heading_vector = np.asarray(detection.heading_vector, dtype=np.float64).copy()
            else:
                hv = (1.0 - self.pose_ema_alpha) * np.asarray(heading_vector) + self.pose_ema_alpha * np.asarray(detection.heading_vector)
                n = float(np.linalg.norm(hv))
                heading_vector = hv / n if n > 1e-6 else heading_vector

        box = np.asarray(detection.bbox_xyxy, dtype=np.float64)
        bbox_wh = np.array([max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)], dtype=np.float64)
        if old is not None and old.bbox_wh is not None:
            bbox_wh = 0.70 * old.bbox_wh + 0.30 * bbox_wh
        body_length = detection.body_length_px
        if old is not None and np.isfinite(old.body_length_px) and np.isfinite(body_length):
            body_length = 0.75 * old.body_length_px + 0.25 * body_length

        old_raw = old.raw_track_id if old is not None else None
        assigned_raw_id = old_raw if preserve_raw_id and old_raw is not None else detection.raw_track_id
        if old_raw is not None and old_raw != assigned_raw_id and self.raw_to_logical.get(old_raw) == logical_id:
            self.raw_to_logical.pop(old_raw, None)

        track_store[logical_id] = IdentityTrack(
            logical_id=logical_id,
            last_center_px=center,
            velocity_px_per_frame=velocity,
            last_frame=frame,
            raw_track_id=assigned_raw_id,
            body_length_px=float(body_length),
            appearance_feature=appearance_feature,
            appearance_feature_long=appearance_feature_long,
            brightness_score=float(brightness_score),
            white_score=float(white_score),
            is_white_candidate=bool(is_white_candidate),
            normalized_pose=normalized_pose,
            anchor_feature=anchor_feature,
            heading_vector=heading_vector,
            bbox_wh=bbox_wh,
            last_keypoints_px=np.asarray(detection.keypoints_px, dtype=np.float64).copy(),
            last_keypoint_conf=np.asarray(detection.keypoint_conf, dtype=np.float64).copy(),
            last_bbox_xyxy=np.asarray(detection.bbox_xyxy, dtype=np.float64).copy(),
            last_box_conf=float(detection.box_conf),
            appearance_updates=appearance_updates,
            hits=hits,
            lock_strength=lock_strength,
            state=state,
            clean_streak=clean_streak,
            # 记忆对象随轨迹迁移（_update_track重建dataclass时保留）。
            memory=old.memory if old is not None else None,
        )
        if assigned_raw_id is not None:
            self.raw_to_logical[assigned_raw_id] = logical_id

    @staticmethod
    def _greedy_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        pairs = [(float(cost_matrix[r, c]), r, c)
                 for r in range(cost_matrix.shape[0])
                 for c in range(cost_matrix.shape[1])]
        pairs.sort(key=lambda x: x[0])
        used_rows, used_cols, result = set(), set(), []
        for cost, r, c in pairs:
            if cost >= StableIdentityAssigner.INF_COST or r in used_rows or c in used_cols:
                continue
            used_rows.add(r); used_cols.add(c); result.append((r, c))
        return result

    def _assignment_is_accepted(self, track: IdentityTrack, detection: Detection, frame: int, cost: float) -> bool:
        if not np.isfinite(cost) or cost >= self.INF_COST or cost > self.max_assignment_cost:
            return False
        _, _, allowed = self._distance_and_gate(track, detection, frame)
        return bool(allowed)

    @staticmethod
    def _candidate_signature(detection: Detection) -> Tuple[Any, ...]:
        if detection.raw_track_id is not None:
            return ("raw", int(detection.raw_track_id))
        c = detection.center_px
        b = max(float(detection.body_length_px), 8.0)
        return ("pos", int(round(float(c[0]) / b * 4)), int(round(float(c[1]) / b * 4)))

    def _assignment_gain(
        self,
        cost_matrix: np.ndarray,
        row_idx: int,
        det_idx: int,
        proposed_by_track: Mapping[int, int],
        track_ids: Sequence[int],
        conflict_lid: Optional[int],
    ) -> float:
        chosen = float(cost_matrix[row_idx, det_idx])
        alternatives = []
        row = cost_matrix[row_idx]
        finite_row = row[np.isfinite(row) & (row < self.INF_COST)]
        if finite_row.size >= 2:
            alternatives.append(float(np.partition(finite_row, 1)[1] - chosen))
        col = cost_matrix[:, det_idx]
        finite_col = col[np.isfinite(col) & (col < self.INF_COST)]
        if finite_col.size >= 2:
            alternatives.append(float(np.partition(finite_col, 1)[1] - chosen))
        # 若底层raw ID当前映射到另一逻辑ID，计算该局部两鼠“正常/交叉”总代价差。
        if conflict_lid is not None and conflict_lid in proposed_by_track and conflict_lid in track_ids:
            other_row = track_ids.index(conflict_lid)
            other_det = proposed_by_track[conflict_lid]
            current = float(cost_matrix[row_idx, det_idx] + cost_matrix[other_row, other_det])
            swapped = float(cost_matrix[row_idx, other_det] + cost_matrix[other_row, det_idx])
            if np.isfinite(swapped) and swapped < self.INF_COST:
                alternatives.append(swapped - current)
        if not alternatives:
            return 1.0
        return float(min(alternatives))

    def _commit_decision(
        self,
        track: IdentityTrack,
        detection: Detection,
        frame: int,
        gain: float,
        conflict_lid: Optional[int],
    ) -> Tuple[bool, int, int, int, str]:
        if not self.commit_enabled:
            self.pending_commits.pop(track.logical_id, None)
            return True, 0, 0, 0, "commit_disabled"
        dt_missing = max(frame - track.last_frame, 0)
        nearest_bl = self._nearest_other_distance_bl(track)
        near = nearest_bl <= self.close_contact_body_lengths
        far = nearest_bl >= 2.0
        raw_changed = (
            track.raw_track_id is not None and detection.raw_track_id is not None
            and track.raw_track_id != detection.raw_track_id
        )
        conflict = conflict_lid is not None and conflict_lid != track.logical_id
        ambiguous = gain < self.commit_immediate_gain
        has_pending = track.logical_id in self.pending_commits
        requires = (
            track.lock_strength >= self.commit_lock_threshold
            and (dt_missing <= self.commit_missing_bypass_frames or has_pending)
            and (conflict or (near and (raw_changed or ambiguous)))
        )
        if not requires:
            self.pending_commits.pop(track.logical_id, None)
            return True, 0, 0, 0, "direct"

        if near:
            dwell_req, min_gain = self.commit_dwell_close, self.commit_min_gain_close
        elif far:
            dwell_req, min_gain = self.commit_dwell_far, self.commit_min_gain_far
        else:
            dwell_req, min_gain = self.commit_dwell_normal, self.commit_min_gain_normal

        pair = tuple(sorted((track.logical_id, conflict_lid))) if conflict else (track.logical_id, track.logical_id)
        last_commit = max(self.last_commit_by_track[track.logical_id], self.last_commit_by_pair[pair])
        cooldown_remaining = max(0, self.commit_cooldown_frames - (frame - last_commit))
        if cooldown_remaining > 0:
            self.pending_commits.pop(track.logical_id, None)
            return False, 0, dwell_req, cooldown_remaining, "cooldown"
        if not np.isfinite(gain) or gain < min_gain:
            self.pending_commits.pop(track.logical_id, None)
            return False, 0, dwell_req, 0, f"gain<{min_gain:.3f}"

        signature = self._candidate_signature(detection)
        state = self.pending_commits.get(track.logical_id)
        if state and state.get("signature") == signature and frame - int(state.get("last_frame", frame)) <= 2:
            count = int(state.get("count", 0)) + 1
        else:
            count = 1
        self.pending_commits[track.logical_id] = {
            "signature": signature, "count": count, "last_frame": frame, "pair": pair
        }
        if count < dwell_req:
            return False, count, dwell_req, 0, f"dwell({count}/{dwell_req})"

        self.pending_commits.pop(track.logical_id, None)
        self.last_commit_by_track[track.logical_id] = frame
        self.last_commit_by_pair[pair] = frame
        return True, count, dwell_req, 0, "committed"

    def _make_predicted_detection(self, track: IdentityTrack, frame: int) -> Optional[Detection]:
        if track.last_keypoints_px is None or track.last_bbox_xyxy is None:
            return None
        if frame - track.last_frame > self.prediction_output_frames:
            return None
        pred = self._prediction(track, frame)
        shift = pred - track.last_center_px
        kpts = np.asarray(track.last_keypoints_px, dtype=np.float64).copy() + shift[None, :]
        bbox = np.asarray(track.last_bbox_xyxy, dtype=np.float64).copy()
        bbox[[0, 2]] += shift[0]
        bbox[[1, 3]] += shift[1]
        conf = np.asarray(track.last_keypoint_conf, dtype=np.float64).copy()
        conf = np.minimum(conf, 0.12)
        return Detection(
            raw_track_id=track.raw_track_id,
            keypoints_px=kpts,
            keypoint_conf=conf,
            bbox_xyxy=bbox,
            box_conf=min(float(track.last_box_conf), 0.25),
            appearance_feature=track.appearance_feature,
            brightness_score=track.brightness_score,
            white_score=track.white_score,
            is_white_candidate=track.is_white_candidate,
            appearance_mode="predicted_hold",
            appearance_reliable=False,
            detection_source="predicted_hold",
            normalized_pose=track.normalized_pose,
            anchor_feature=track.anchor_feature,
            heading_vector=track.heading_vector,
            synthetic_recovery=True,
        )

    def assign(
        self,
        detections: Sequence[Detection],
        frame: int,
        occlusion_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[int, Detection]]:
        self._expire(frame)
        context = dict(occlusion_context or {})
        frozen_ids = {int(v) for v in context.get("frozen_ids", set())}
        id_to_cluster = {int(k): int(v) for k, v in dict(context.get("id_to_cluster", {})).items()}
        regions_by_id = {
            int(region.get("cluster_id", -1)): region
            for region in context.get("regions", [])
        }

        detections = list(detections)[: self.max_mice + self.candidate_extra]
        if not detections:
            output = []
            for lid, track in sorted(self.tracks.items()):
                track.lock_strength = max(0.0, track.lock_strength - self.missing_lock_decay)
                pred = self._make_predicted_detection(track, frame)
                if pred is not None:
                    output.append((lid, pred))
            return output

        if self.mode == "tracker":
            assigned, used = [], set()
            for idx, det in enumerate(detections):
                logical_id = int(det.raw_track_id) if det.raw_track_id is not None else idx
                logical_id %= max(self.max_mice, 1)
                if logical_id in used:
                    free = [i for i in range(self.max_mice) if i not in used]
                    if not free:
                        break
                    logical_id = free[0]
                used.add(logical_id)
                self._update_track(logical_id, det, frame)
                self.debug_records.append(IdentityDebug(
                    frame, logical_id, det.raw_track_id, 0.0,
                    logical_id, 1.0, 0, 0, 0, "tracker", "",
                    det.appearance_mode, det.detection_source
                ))
                assigned.append((logical_id, det))
            return sorted(assigned, key=lambda item: item[0])

        if not self.tracks:
            ordered = sorted(detections, key=lambda d: (float(d.center_px[0]), float(d.center_px[1])))
            output = []
            for logical_id, det in enumerate(ordered[: self.max_mice]):
                self._update_track(logical_id, det, frame)
                self.debug_records.append(IdentityDebug(
                    frame, logical_id, det.raw_track_id, 0.0,
                    logical_id, 1.0, 0, 0, 0, "initialized", "",
                    det.appearance_mode, det.detection_source
                ))
                output.append((logical_id, det))
            return output

        track_ids = sorted(self.tracks.keys())
        tracks = [self.tracks[logical_id] for logical_id in track_ids]
        candidate_mask = self.build_fast_gate(tracks, detections, frame)
        cost_matrix = np.full((len(track_ids), len(detections)), self.INF_COST, dtype=np.float64)
        for r, logical_id in enumerate(track_ids):
            for c, det in enumerate(detections):
                if candidate_mask[r, c]:
                    cost_matrix[r, c] = self._cost(self.tracks[logical_id], det, frame)

        if linear_sum_assignment is not None:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            assignment = list(zip(row_indices.tolist(), col_indices.tolist()))
        else:
            assignment = self._greedy_assignment(cost_matrix)

        proposed_by_track = {track_ids[r]: c for r, c in assignment if r < len(track_ids)}
        reserved_detection_indices = {c for _, c in assignment}
        output_by_id: Dict[int, Detection] = {}
        updated_ids: set[int] = set()

        for row_idx, det_idx in assignment:
            logical_id = track_ids[row_idx]
            track = self.tracks[logical_id]
            det = detections[det_idx]
            cost = float(cost_matrix[row_idx, det_idx])
            cluster_id = int(id_to_cluster.get(logical_id, -1))
            cluster = regions_by_id.get(cluster_id, {})
            cluster_expected = int(cluster.get("expected_count", 0))
            cluster_observed = int(cluster.get("observed_count", 0))

            if not self._assignment_is_accepted(track, det, frame, cost):
                pred = self._make_predicted_detection(track, frame)
                if pred is not None:
                    output_by_id[logical_id] = pred
                self.debug_records.append(IdentityDebug(
                    frame, logical_id, det.raw_track_id, cost, logical_id, float("nan"), 0, 0, 0,
                    "rejected", "hard_gate_or_cost", det.appearance_mode, det.detection_source,
                    cluster_id, cluster_expected, cluster_observed,
                ))
                continue

            conflict_lid = self.raw_to_logical.get(det.raw_track_id) if det.raw_track_id is not None else None
            if conflict_lid == logical_id:
                conflict_lid = None
            gain = self._assignment_gain(cost_matrix, row_idx, det_idx, proposed_by_track, track_ids, conflict_lid)

            # 接触簇中不执行逐帧swap。只有当前检测对该ID明显唯一时才更新位置，
            # 并冻结外观模板、保留进入接触簇前的raw ID。
            if logical_id in frozen_ids:
                finite_row = cost_matrix[row_idx]
                finite_row = finite_row[np.isfinite(finite_row) & (finite_row < self.INF_COST)]
                if finite_row.size >= 2:
                    second = float(np.partition(finite_row, 1)[1])
                    margin = second - cost
                else:
                    margin = 1.0
                strong_unique = bool(cost <= self.cluster_accept_cost and margin >= self.cluster_min_margin)
                if not strong_unique:
                    pred = self._make_predicted_detection(track, frame)
                    if pred is not None:
                        output_by_id[logical_id] = pred
                    self.debug_records.append(IdentityDebug(
                        frame, logical_id, det.raw_track_id, cost, logical_id, margin,
                        0, 0, 0, "cluster_hold_predict",
                        f"cluster_ambiguous_margin={margin:.3f}", det.appearance_mode,
                        det.detection_source, cluster_id, cluster_expected, cluster_observed,
                    ))
                    continue

                self._update_track(
                    logical_id, det, frame,
                    freeze_appearance=True,
                    preserve_raw_id=self.cluster_preserve_raw_id,
                )
                output_by_id[logical_id] = det
                updated_ids.add(logical_id)
                self.debug_records.append(IdentityDebug(
                    frame, logical_id, det.raw_track_id, cost, logical_id, margin,
                    0, 0, 0, "cluster_position_update",
                    "appearance_frozen_raw_preserved", det.appearance_mode,
                    det.detection_source, cluster_id, cluster_expected, cluster_observed,
                ))
                continue

            accepted, dwell_count, dwell_req, cooldown_remaining, reason = self._commit_decision(
                track, det, frame, gain, conflict_lid
            )
            if not accepted:
                pred = self._make_predicted_detection(track, frame)
                if pred is not None:
                    output_by_id[logical_id] = pred
                self.debug_records.append(IdentityDebug(
                    frame, logical_id, det.raw_track_id, cost, logical_id, gain,
                    dwell_count, dwell_req, cooldown_remaining, "hold_predict", reason,
                    det.appearance_mode, det.detection_source,
                    cluster_id, cluster_expected, cluster_observed,
                ))
                continue

            self._update_track(logical_id, det, frame)
            output_by_id[logical_id] = det
            updated_ids.add(logical_id)
            self.debug_records.append(IdentityDebug(
                frame, logical_id, det.raw_track_id, cost, logical_id, gain,
                dwell_count, dwell_req, cooldown_remaining, "accepted", reason,
                det.appearance_mode, det.detection_source,
                cluster_id, cluster_expected, cluster_observed,
            ))

        # 未更新旧ID只做慢衰减和内部预测。预测检测由主程序排除渲染与行为几何。
        for lid, track in list(self.tracks.items()):
            if lid in updated_ids:
                continue
            track.lock_strength = max(0.0, track.lock_strength - self.missing_lock_decay)
            if lid not in output_by_id:
                pred = self._make_predicted_detection(track, frame)
                if pred is not None:
                    output_by_id[lid] = pred

        used_logical = set(self.tracks.keys())
        free_ids = [i for i in range(self.max_mice) if i not in used_logical]
        for det_idx, det in enumerate(detections):
            if det_idx in reserved_detection_indices or not free_ids:
                continue

            # 接触/打斗区域中的临时实例绝不创建新ID。
            if OcclusionClusterManager.detection_inside_forbidden(det, context):
                self.debug_records.append(IdentityDebug(
                    frame, -1, det.raw_track_id, 0.0, -1, float("nan"),
                    0, self.new_track_confirm_frames, 0,
                    "suppressed_new_in_occlusion", "inside_existing_cluster",
                    det.appearance_mode, det.detection_source,
                ))
                continue

            # 距离任一旧轨迹过近“且框重叠”的未分配候选才视为重复/碎片，
            # 密集群体中单纯相邻的新鼠不应被永久拦截。
            nearest_bl = np.inf
            nearest_iou = 0.0
            for tr in self.tracks.values():
                body = max(float(np.nanmedian([tr.body_length_px, det.body_length_px])), 8.0)
                pred_center = self._prediction(tr, frame)
                d_bl = point_distance(pred_center, det.center_px) / body
                if d_bl < nearest_bl:
                    nearest_bl = d_bl
                    nearest_iou = 0.0
                    if tr.last_bbox_xyxy is not None:
                        shift = pred_center - tr.last_center_px
                        pred_box = np.asarray(tr.last_bbox_xyxy, dtype=np.float64).copy()
                        pred_box[[0, 2]] += shift[0]
                        pred_box[[1, 3]] += shift[1]
                        nearest_iou = bbox_iou_xyxy(pred_box, det.bbox_xyxy)
            if nearest_bl < self.new_track_min_separation_bl and nearest_iou >= self.new_track_dup_iou:
                self.debug_records.append(IdentityDebug(
                    frame, -1, det.raw_track_id, 0.0, -1, float("nan"),
                    0, self.new_track_confirm_frames, 0,
                    "suppressed_near_existing", f"nearest_bl={nearest_bl:.3f};iou={nearest_iou:.3f}",
                    det.appearance_mode, det.detection_source,
                ))
                continue

            signature = self._candidate_signature(det)
            state = self.pending_new_tracks.get(signature)
            if state and frame - int(state.get("last_frame", frame)) <= self.new_track_max_gap_frames:
                count = int(state.get("count", 0)) + 1
            else:
                count = 1
            self.pending_new_tracks[signature] = {
                "count": count,
                "last_frame": frame,
                "detection": det,
            }
            if count < self.new_track_confirm_frames:
                self.debug_records.append(IdentityDebug(
                    frame, -1, det.raw_track_id, 0.0, -1, float("nan"),
                    count, self.new_track_confirm_frames, 0,
                    "pending_new_track", f"confirm({count}/{self.new_track_confirm_frames})",
                    det.appearance_mode, det.detection_source,
                ))
                continue

            logical_id = free_ids.pop(0)
            self.pending_new_tracks.pop(signature, None)
            self._update_track(logical_id, det, frame)
            output_by_id[logical_id] = det
            updated_ids.add(logical_id)
            self.debug_records.append(IdentityDebug(
                frame, logical_id, det.raw_track_id, 0.0, logical_id, 1.0,
                count, self.new_track_confirm_frames, 0, "new_track_confirmed", "",
                det.appearance_mode, det.detection_source,
            ))

        return sorted(output_by_id.items(), key=lambda item: item[0])


class LegacyStableSlotAssigner(StableIdentityAssigner):
    """固定逻辑槽身份分配器。

    该模式吸收旧 ``MouseTracker`` 看起来稳定的核心原则，但修复其即时删除、
    任意复用 lost ID 和仅依赖中心点的缺陷：

    * 逻辑ID与YOLO/ByteTrack原始ID完全解耦；
    * 一旦创建的逻辑槽在整个视频内不删除、不复用；
    * 漏检时保留槽位和运动预测，不把其它小鼠强行塞入该ID；
    * 先锁定互为最近邻的明确匹配，再对剩余目标使用匈牙利算法；
    * 接触/打斗阶段若匹配边际不足，宁可保持遮挡状态，也不交换ID；
    * 未匹配检测必须连续稳定出现若干帧，才允许占用新的空闲槽位。

    预测检测只供内部保持使用，主程序会排除其渲染和行为几何。
    """

    def __init__(self, config: Mapping[str, Any], max_mice: int = 20) -> None:
        super().__init__(config, max_mice=max_mice)
        slot = dict(config.get("legacy_slot", {}))
        self.slot_never_expire = bool(slot.get("never_expire", True))
        self.slot_center_weight = float(slot.get("center_weight", 0.68))
        self.slot_iou_weight = float(slot.get("bbox_iou_weight", 0.16))
        self.slot_pose_weight = float(slot.get("pose_weight", 0.07))
        self.slot_anchor_weight = float(slot.get("anchor_weight", 0.06))
        self.slot_heading_weight = float(slot.get("heading_weight", 0.03))
        self.slot_max_cost = float(slot.get("max_cost", 1.05))
        self.slot_max_jump_bl = float(slot.get("max_jump_body_lengths", 3.20))
        self.slot_mutual_lock = bool(slot.get("mutual_nearest_lock", True))
        self.slot_mutual_lock_cost = float(slot.get("mutual_lock_cost", 0.72))
        self.slot_close_hold = bool(slot.get("close_contact_hold", True))
        self.slot_close_margin = float(slot.get("close_contact_min_margin", 0.10))
        self.slot_general_margin = float(slot.get("general_min_margin", 0.025))
        self.slot_prediction_frames = int(slot.get("prediction_frames", 30))
        self.slot_new_track_confirm = int(slot.get("new_track_confirm_frames", 10))
        self.slot_new_track_max_gap = int(slot.get("new_track_max_gap_frames", 2))
        self.slot_new_track_min_separation_bl = float(
            slot.get("new_track_min_separation_body_lengths", 0.55)
        )
        # 重复判定需要“贴近+框重叠”同时成立。密集群体里新鼠常与旧鼠相邻，
        # 仅靠中心距会把真实新鼠永久误判为重复检测。
        self.slot_new_track_dup_iou = float(slot.get("new_track_duplicate_iou", 0.20))
        # OKS姿态代价权重（报告推荐：姿态几何是同种群体中最可靠的身份辅助线索之一）。
        self.slot_oks_weight = float(slot.get("oks_weight", 0.06))

        # 两阶段关联（ByteTrack思想）：高分检测全特征匹配，
        # 未匹配轨迹再用低分检测做纯运动/IoU回收——低分检测外观不可靠。
        # 但关键点质量高的检测（如检测器优先模式的ROI姿态结果）即使box_conf低，
        # 其姿态/OKS证据仍然可信，应升入高分组参与全特征匹配。
        two_stage = dict(config.get("two_stage", {}))
        self.two_stage_enabled = bool(two_stage.get("enabled", True))
        self.two_stage_high_conf = float(two_stage.get("high_conf", 0.20))
        self.two_stage_high_pose_quality = float(two_stage.get("high_pose_quality", 0.35))
        self.two_stage_low_max_cost = float(two_stage.get("low_max_cost", 0.42))

        # 轨迹状态机：suspicious冻结外观并提高接受门槛；
        # lost轨迹重激活必须同时满足运动可达+外观或OKS证据，错绑比短时丢失更糟。
        sm = dict(config.get("state_machine", {}))
        self.sm_enabled = bool(sm.get("enabled", True))
        self.sm_lost_after_frames = int(sm.get("lost_after_frames", 60))
        self.sm_recover_confirm_frames = int(sm.get("recover_confirm_frames", 3))
        self.sm_suspicious_margin = float(sm.get("suspicious_margin", 0.06))
        self.sm_lost_accept_cost = float(sm.get("lost_accept_cost", 0.55))
        self.sm_lost_appearance_gate = float(sm.get("lost_appearance_gate", 0.38))
        self.sm_lost_oks_gate = float(sm.get("lost_oks_gate", 0.40))

        # 冲突对仲裁：边际不足且竞争者贴近时，用OKS+长期外观+锚点做最终裁决。
        hp = dict(config.get("hard_pair", {}))
        self.hp_enabled = bool(hp.get("enabled", True))
        self.hp_margin = float(hp.get("margin", 0.08))
        self.hp_close_bl = float(hp.get("close_body_lengths", 1.8))
        self.hp_min_arbitration_margin = float(hp.get("min_arbitration_margin", 0.04))

        # 短窗重拼接：若两条轨迹的近期观测持续更像对方的长期外观模板，
        # 说明此前发生过ID互换，执行身份回滚而不是将错就错。
        rst = dict(config.get("restitch", {}))
        self.restitch_enabled = bool(rst.get("enabled", True))
        self.restitch_confirm_frames = int(rst.get("confirm_frames", 8))
        self.restitch_window_frames = int(rst.get("window_frames", 60))
        self.restitch_cooldown_frames = int(rst.get("cooldown_frames", 90))
        self.restitch_min_improvement = float(rst.get("min_improvement", 0.06))
        self.restitch_min_appearance_updates = int(rst.get("min_appearance_updates", 6))
        self._restitch_votes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self._restitch_last_swap: Dict[Tuple[int, int], int] = defaultdict(lambda: -10**9)

        # 临时ID（文档§3/§5）：每一个当前帧检测都必须有框和标签。
        # 未匹配检测立即创建TMP轨迹，连续确认后才晋升为固定逻辑ID；
        # 身份冲突时显示ID?，检测框绝不丢弃。
        self.tmp_ttl_frames = int(config.get("tmp_ttl_frames", 15))
        self._tmp_next = 10001
        self._tmp_states: Dict[int, str] = {}  # tmp_id -> tentative / suspicious

        # 本模式明确不使用raw Track ID影响逻辑身份。
        self.raw_id_mismatch_penalty = 0.0
        self.raw_id_match_reward = 0.0

    def _expire(self, frame: int) -> None:
        """固定槽模式不删除已创建逻辑ID，只清理未确认的新目标候选。"""
        if not self.slot_never_expire:
            super()._expire(frame)
            return
        for signature in list(self.pending_new_tracks):
            if frame - int(self.pending_new_tracks[signature].get("last_frame", frame)) > self.slot_new_track_max_gap:
                self.pending_new_tracks.pop(signature, None)

    @staticmethod
    def _position_signature(detection: Detection) -> Tuple[Any, ...]:
        c = detection.center_px
        b = max(float(detection.body_length_px), 8.0)
        return (
            "slot_pos",
            int(round(float(c[0]) / b * 4)),
            int(round(float(c[1]) / b * 4)),
        )

    def _slot_cost(self, track: IdentityTrack, detection: Detection, frame: int) -> float:
        pred = self._prediction(track, frame)
        body = max(float(np.nanmedian([track.body_length_px, detection.body_length_px])), 8.0)
        d_pred_bl = point_distance(pred, detection.center_px) / body
        if not np.isfinite(d_pred_bl) or d_pred_bl > self.slot_max_jump_bl:
            return self.INF_COST
        d_last_bl = point_distance(track.last_center_px, detection.center_px) / body
        center_cost = float(np.clip((0.82 * d_pred_bl + 0.18 * d_last_bl) / max(self.slot_max_jump_bl, 1e-6), 0.0, 1.5))
        if track.last_bbox_xyxy is None:
            iou_cost = 0.50
        else:
            iou_cost = 1.0 - bbox_iou_xyxy(track.last_bbox_xyxy, detection.bbox_xyxy)
        pose_cost = self._pose_distance(track, detection)
        anchor_cost = self._anchor_distance(track, detection)
        heading_cost = self._heading_distance(track, detection)
        total = (
            self.slot_center_weight * center_cost
            + self.slot_iou_weight * iou_cost
            + self.slot_pose_weight * pose_cost
            + self.slot_anchor_weight * anchor_cost
            + self.slot_heading_weight * heading_cost
        )
        if self.slot_oks_weight > 0.0:
            oks_cost = self._oks_pose_cost(track, detection)
            if oks_cost is not None:
                total += self.slot_oks_weight * oks_cost
            else:
                # 关键点不足时把OKS权重退还给归一化姿态距离，保持代价尺度稳定。
                total += self.slot_oks_weight * pose_cost
        return float(total)

    def _slot_low_cost(self, track: IdentityTrack, detection: Detection, frame: int) -> float:
        """低分检测回收代价：只用运动、IoU和尺度。

        与ByteTrack第二阶段一致——低置信检测通常伴随模糊、局部裁剪或姿态异常，
        外观与姿态特征不可靠，不参与代价。
        """
        pred = self._prediction(track, frame)
        body = max(float(np.nanmedian([track.body_length_px, detection.body_length_px])), 8.0)
        d_pred_bl = point_distance(pred, detection.center_px) / body
        if not np.isfinite(d_pred_bl) or d_pred_bl > self.slot_max_jump_bl:
            return self.INF_COST
        d_last_bl = point_distance(track.last_center_px, detection.center_px) / body
        center_cost = float(np.clip((0.82 * d_pred_bl + 0.18 * d_last_bl) / max(self.slot_max_jump_bl, 1e-6), 0.0, 1.5))
        if track.last_bbox_xyxy is None:
            iou_cost = 0.50
        else:
            iou_cost = 1.0 - bbox_iou_xyxy(track.last_bbox_xyxy, detection.bbox_xyxy)
        size_cost = self._size_distance(track, detection)
        total = 0.60 * center_cost + 0.25 * iou_cost + 0.15 * min(size_cost, 1.0)
        return float(total)

    def _arbitration_cost(self, track: IdentityTrack, detection: Detection) -> float:
        """冲突对裁决代价：OKS + 长期外观 + 锚点。只用对身份最敏感的分量。"""
        parts: List[float] = []
        oks = self._oks_pose_cost(track, detection)
        if oks is not None:
            parts.append(oks)
        app_long = self._appearance_long_distance(track, detection)
        if app_long is not None:
            parts.append(app_long)
        anchor = self._anchor_distance(track, detection)
        parts.append(min(anchor, 1.0))
        return float(np.mean(parts)) if parts else 0.5

    @staticmethod
    def _row_col_margin(cost_matrix: np.ndarray, row: int, col: int, inf_cost: float) -> float:
        chosen = float(cost_matrix[row, col])
        margins: List[float] = []
        r = cost_matrix[row]
        rv = r[np.isfinite(r) & (r < inf_cost)]
        if rv.size >= 2:
            margins.append(float(np.partition(rv, 1)[1] - chosen))
        c = cost_matrix[:, col]
        cv = c[np.isfinite(c) & (c < inf_cost)]
        if cv.size >= 2:
            margins.append(float(np.partition(cv, 1)[1] - chosen))
        return min(margins) if margins else 1.0

    def _find_hard_pair_competitor(
        self,
        cost_matrix: np.ndarray,
        row: int,
        col: int,
        track_ids: Sequence[int],
        detection: Detection,
        frame: int,
    ) -> Optional[int]:
        """寻找对同一检测构成竞争、且空间上确实贴近的另一条轨迹。"""
        column = cost_matrix[:, col]
        for idx in np.argsort(column).tolist():
            idx = int(idx)
            if idx == row or not np.isfinite(column[idx]) or column[idx] >= self.INF_COST:
                continue
            other = self.tracks[track_ids[idx]]
            body = max(float(np.nanmedian([other.body_length_px, detection.body_length_px])), 8.0)
            d_bl = point_distance(self._prediction(other, frame), detection.center_px) / body
            if d_bl <= self.hp_close_bl:
                return track_ids[idx]
        return None

    def _make_slot_prediction(self, track: IdentityTrack, frame: int) -> Optional[Detection]:
        old = self.prediction_output_frames
        try:
            self.prediction_output_frames = max(old, self.slot_prediction_frames)
            return self._make_predicted_detection(track, frame)
        finally:
            self.prediction_output_frames = old

    # ------------------ 轨迹状态机（tracked / suspicious / lost） ------------------

    def _sm_transition_on_miss(self, track: IdentityTrack, frame: int) -> None:
        if track.state != "lost" and frame - track.last_frame > self.sm_lost_after_frames:
            track.state = "lost"
            track.clean_streak = 0

    def _sm_mark_suspicious(self, track: IdentityTrack) -> None:
        if track.state == "tracked":
            track.state = "suspicious"
        track.clean_streak = 0

    def _sm_mark_accepted(self, track: IdentityTrack, margin: float) -> None:
        if track.state == "lost":
            # 重激活成功后先置于观察期，不直接恢复稳定状态。
            track.state = "suspicious"
            track.clean_streak = 1
            return
        if track.state != "suspicious":
            return
        if margin >= self.sm_suspicious_margin:
            track.clean_streak += 1
            if track.clean_streak >= self.sm_recover_confirm_frames:
                track.state = "tracked"
                track.clean_streak = 0
        else:
            track.clean_streak = 0

    def _lost_reactivation_ok(self, track: IdentityTrack, detection: Detection, cost: float) -> Tuple[bool, str]:
        """lost轨迹重激活：运动可达 + 长期外观或OKS身份证据。

        报告原则：回收必须同时满足外观相似+速度可达+尺度合理+时间间隔合理，
        否则宁可保持lost也不强行绑定错误ID。完全没有可用证据时退化为更严格的纯运动门槛。
        """
        if cost > self.sm_lost_accept_cost:
            return False, f"lost_cost>{self.sm_lost_accept_cost:.2f}"
        if getattr(self, "simple_motion", False):
            # v1.13.0：纯运动+朝向门槛——主恢复路径是轨迹段融合，
            # lost重激活只收运动门内且朝向不矛盾的（代价已含朝向项）。
            return True, "lost_reactivated_motion_simple"
        app = self._appearance_long_distance(track, detection)
        oks = self._oks_pose_cost(track, detection)
        if app is None and oks is None:
            return True, "lost_motion_only_no_evidence_available"
        if app is not None and app <= self.sm_lost_appearance_gate:
            return True, f"lost_reactivated_appearance={app:.3f}"
        if oks is not None and oks <= self.sm_lost_oks_gate:
            return True, f"lost_reactivated_oks={oks:.3f}"
        return False, "lost_identity_evidence_contradicts"

    # ------------------ 短窗重拼接（ID互换回滚） ------------------

    def _restitch_check(self, frame: int, frame_updates: Mapping[int, Detection], frozen_ids: set) -> None:
        if not self.long_term_memory_enabled:  # 十分钟以内视频不运行任何依赖长期外观模板的重拼接逻辑。
            return  # 直接保留现有短时运动身份结果，避免长期模块产生隐式影响。
        mature = {
            lid for lid, t in self.tracks.items()
            if t.appearance_feature_long is not None
            and t.appearance_updates >= self.restitch_min_appearance_updates
        }
        if len(mature) < 2:
            return
        # 投票：轨迹lid的当前观测更像other的长期模板 → 可能发生过ID互换。
        for lid, det in frame_updates.items():
            if lid not in mature or lid in frozen_ids:
                continue
            if det.appearance_feature is None or not det.appearance_reliable:
                continue
            own = self._appearance_long_distance(self.tracks[lid], det)
            if own is None:
                continue
            for other in mature:
                if other == lid or other in frozen_ids:
                    continue
                other_dist = self._appearance_long_distance(self.tracks[other], det)
                if other_dist is not None and other_dist + self.restitch_min_improvement < own:
                    self._restitch_votes[(lid, other)].append(frame)
        cutoff = frame - self.restitch_window_frames
        for key in list(self._restitch_votes):
            kept = [f for f in self._restitch_votes[key] if f >= cutoff]
            if kept:
                self._restitch_votes[key] = kept
            else:
                self._restitch_votes.pop(key, None)
        checked: set = set()
        for a, b in list(self._restitch_votes):
            pair = (min(a, b), max(a, b))
            if pair in checked:
                continue
            checked.add(pair)
            if len(self._restitch_votes.get(pair, [])) < self.restitch_confirm_frames:
                continue
            if len(self._restitch_votes.get((pair[1], pair[0]), [])) < self.restitch_confirm_frames:
                continue
            if frame - self._restitch_last_swap[pair] < self.restitch_cooldown_frames:
                continue
            x, y = pair
            if x not in self.tracks or y not in self.tracks:
                continue
            if x in frozen_ids or y in frozen_ids:
                continue
            self._execute_restitch(x, y, frame)

    def _execute_restitch(self, a: int, b: int, frame: int) -> None:
        """交换两条轨迹的身份（轨迹对象互换槽位），并记录调试事件。"""
        ta, tb = self.tracks[a], self.tracks[b]
        ta.logical_id, tb.logical_id = b, a
        self.tracks[a], self.tracks[b] = tb, ta
        for raw, lid in list(self.raw_to_logical.items()):
            if lid in (a, b):
                self.raw_to_logical.pop(raw, None)
        for t in (self.tracks[a], self.tracks[b]):
            if t.raw_track_id is not None:
                self.raw_to_logical[t.raw_track_id] = t.logical_id
            t.state = "suspicious"
            t.clean_streak = 0
            t.lock_strength = min(t.lock_strength, 0.40)
        pair = (min(a, b), max(a, b))
        self._restitch_last_swap[pair] = frame
        for key in list(self._restitch_votes):
            if a in key or b in key:
                self._restitch_votes.pop(key, None)
        logging.info("滑窗重拼接：逻辑ID %d 与 %d 在帧 %d 执行身份回滚", a, b, frame)
        for lid in pair:
            self.debug_records.append(IdentityDebug(
                frame, lid, self.tracks[lid].raw_track_id, 0.0, lid, float("nan"),
                0, 0, 0, "restitch_swap_back", f"pair={pair}",
                "restitch", "restitch", -1, 0, 0, self.tracks[lid].state,
            ))

    def assign(
        self,
        detections: Sequence[Detection],
        frame: int,
        occlusion_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[int, Detection]]:
        self._expire(frame)
        context = dict(occlusion_context or {})
        frozen_ids = {int(v) for v in context.get("frozen_ids", set())}
        id_to_cluster = {int(k): int(v) for k, v in dict(context.get("id_to_cluster", {})).items()}
        regions_by_id = {
            int(region.get("cluster_id", -1)): region
            for region in context.get("regions", [])
        }
        detections = list(detections)[: self.max_mice + self.candidate_extra]

        if not detections:
            out: List[Tuple[int, Detection]] = []
            for lid, track in sorted(self.tracks.items()):
                track.lock_strength = max(0.0, track.lock_strength - self.missing_lock_decay)
                if self.sm_enabled:
                    self._sm_transition_on_miss(track, frame)
                pred = self._make_slot_prediction(track, frame)
                if pred is not None:
                    out.append((lid, pred))
                self.debug_records.append(IdentityDebug(
                    frame, lid, None, float("nan"), lid, float("nan"),
                    0, 0, 0, "slot_missing_hold", "no_detection_keep_slot",
                    "predicted_hold", "predicted_hold",
                    int(id_to_cluster.get(lid, -1)), 0, 0, track.state,
                ))
            return out

        if not self.tracks:
            ordered = sorted(detections, key=lambda d: (float(d.center_px[0]), float(d.center_px[1])))
            out: List[Tuple[int, Detection]] = []
            for logical_id, det in enumerate(ordered[: self.max_mice]):
                self._update_track(logical_id, det, frame, preserve_raw_id=True)
                self.debug_records.append(IdentityDebug(
                    frame, logical_id, det.raw_track_id, 0.0, logical_id, 1.0,
                    0, 0, 0, "slot_initialized", "x_y_sorted_initial_slots",
                    det.appearance_mode, det.detection_source,
                    -1, 0, 0, self.tracks[logical_id].state,
                ))
                out.append((logical_id, det))
            return out

        if self.sm_enabled:
            for track in self.tracks.values():
                self._sm_transition_on_miss(track, frame)

        # 两阶段关联：高分检测先全特征匹配；低分检测只做纯运动回收。
        if self.two_stage_enabled:
            high_cols = [
                i for i, d in enumerate(detections)
                if float(d.box_conf) >= self.two_stage_high_conf
                or float(d.pose_quality) >= self.two_stage_high_pose_quality
            ]
            high_set = set(high_cols)
            low_cols = [i for i in range(len(detections)) if i not in high_set]
        else:
            high_cols = list(range(len(detections)))
            low_cols = []

        track_ids = sorted(self.tracks)
        stage1_dets = [detections[i] for i in high_cols]
        cost = np.full((len(track_ids), len(stage1_dets)), self.INF_COST, dtype=np.float64)
        for r, lid in enumerate(track_ids):
            for c, det in enumerate(stage1_dets):
                cost[r, c] = self._slot_cost(self.tracks[lid], det, frame)

        assignments: List[Tuple[int, int, str]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()

        if self.slot_mutual_lock and cost.size:
            row_best = np.argmin(cost, axis=1)
            col_best = np.argmin(cost, axis=0)
            for r, c0 in enumerate(row_best.tolist()):
                c = int(c0)
                if c >= cost.shape[1] or int(col_best[c]) != r:
                    continue
                if not np.isfinite(cost[r, c]) or cost[r, c] >= self.INF_COST or cost[r, c] > self.slot_mutual_lock_cost:
                    continue
                assignments.append((r, c, "mutual"))
                used_rows.add(r)
                used_cols.add(c)

        rem_rows = [r for r in range(cost.shape[0]) if r not in used_rows]
        rem_cols = [c for c in range(cost.shape[1]) if c not in used_cols]
        if rem_rows and rem_cols:
            sub = cost[np.ix_(rem_rows, rem_cols)]
            if linear_sum_assignment is not None:
                rr, cc = linear_sum_assignment(sub)
                for sr, sc in zip(rr.tolist(), cc.tolist()):
                    assignments.append((rem_rows[sr], rem_cols[sc], "hungarian"))
            else:
                for sr, sc in self._greedy_assignment(sub):
                    assignments.append((rem_rows[sr], rem_cols[sc], "greedy"))

        output: Dict[int, Detection] = {}
        updated: set[int] = set()
        assigned_det: set[int] = set()
        assigned_rows: set[int] = set()
        rejected_rows: set[int] = set()
        frame_updates: Dict[int, Detection] = {}

        for r, c, method in assignments:
            lid = track_ids[r]
            track = self.tracks[lid]
            det = stage1_dets[c]
            det_global_idx = high_cols[c]
            chosen = float(cost[r, c])
            assigned_rows.add(r)
            cluster_id = int(id_to_cluster.get(lid, -1))
            cluster = regions_by_id.get(cluster_id, {})
            cluster_expected = int(cluster.get("expected_count", 0))
            cluster_observed = int(cluster.get("observed_count", 0))
            margin = self._row_col_margin(cost, r, c, self.INF_COST)
            nearest_other = self._nearest_other_distance_bl(track)
            close = nearest_other <= self.close_contact_body_lengths or lid in frozen_ids

            if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.slot_max_cost:
                # 不立即输出预测框：该轨迹是第二阶段低分回收的候选，
                # 回收失败后再由末尾的兜底逻辑输出预测。
                rejected_rows.add(r)
                self.debug_records.append(IdentityDebug(
                    frame, lid, det.raw_track_id, chosen, lid, margin,
                    0, 0, 0, "slot_rejected", "cost_or_jump_gate",
                    det.appearance_mode, det.detection_source,
                    cluster_id, cluster_expected, cluster_observed, track.state,
                ))
                continue

            min_margin = self.slot_close_margin if close else self.slot_general_margin
            if (close and self.slot_close_hold and margin < min_margin) or (method != "mutual" and margin < min_margin):
                # 歧义保持前先给姿态/外观证据一次裁决机会（报告：昂贵特征只用于冲突对tie-break）。
                # 证据明确支持本轨迹则照常接受；不支持或同样歧义才回退到预测保持。
                resolved = False
                arb_note = ""
                if self.hp_enabled:
                    competitor = self._find_hard_pair_competitor(cost, r, c, track_ids, det, frame)
                    if competitor is not None:
                        comp_track = self.tracks[competitor]
                        arb_self = self._arbitration_cost(track, det)
                        arb_comp = self._arbitration_cost(comp_track, det)
                        if arb_self + self.hp_min_arbitration_margin < arb_comp:
                            resolved = True
                        elif arb_comp + self.hp_min_arbitration_margin < arb_self:
                            arb_note = f"competitor={competitor};arb={arb_self:.3f}vs{arb_comp:.3f}"
                if not resolved:
                    pred = self._make_slot_prediction(track, frame)
                    if pred is not None:
                        output[lid] = pred
                    if self.sm_enabled:
                        self._sm_mark_suspicious(track)
                    if arb_note:
                        self.debug_records.append(IdentityDebug(
                            frame, lid, det.raw_track_id, chosen, lid, margin,
                            0, 0, 0, "hard_pair_arbitration_hold", arb_note,
                            det.appearance_mode, det.detection_source,
                            cluster_id, cluster_expected, cluster_observed, track.state,
                        ))
                    else:
                        self.debug_records.append(IdentityDebug(
                            frame, lid, det.raw_track_id, chosen, lid, margin,
                            0, 0, 0, "slot_ambiguous_hold",
                            f"{method}_margin<{min_margin:.3f}",
                            det.appearance_mode, det.detection_source,
                            cluster_id, cluster_expected, cluster_observed, track.state,
                        ))
                    continue

            # lost轨迹重激活：运动可达之外必须附加长期外观或OKS身份证据。
            reactivation_note = "raw_id_ignored_fixed_slot"
            if self.sm_enabled and track.state == "lost":
                ok, reason = self._lost_reactivation_ok(track, det, chosen)
                if not ok:
                    self.debug_records.append(IdentityDebug(
                        frame, lid, det.raw_track_id, chosen, lid, margin,
                        0, 0, 0, "slot_lost_reactivation_rejected", reason,
                        det.appearance_mode, det.detection_source,
                        cluster_id, cluster_expected, cluster_observed, track.state,
                    ))
                    continue
                reactivation_note = reason

            # 冲突对仲裁：边际不足且存在贴近的竞争轨迹时，用OKS+长期外观+锚点最终裁决。
            if self.hp_enabled and margin < self.hp_margin:
                competitor = self._find_hard_pair_competitor(cost, r, c, track_ids, det, frame)
                if competitor is not None:
                    comp_track = self.tracks[competitor]
                    arb_self = self._arbitration_cost(track, det)
                    arb_comp = self._arbitration_cost(comp_track, det)
                    if arb_comp + self.hp_min_arbitration_margin < arb_self:
                        pred = self._make_slot_prediction(track, frame)
                        if pred is not None:
                            output[lid] = pred
                        if self.sm_enabled:
                            self._sm_mark_suspicious(track)
                        self.debug_records.append(IdentityDebug(
                            frame, lid, det.raw_track_id, chosen, lid, margin,
                            0, 0, 0, "hard_pair_arbitration_hold",
                            f"competitor={competitor};arb={arb_self:.3f}vs{arb_comp:.3f}",
                            det.appearance_mode, det.detection_source,
                            cluster_id, cluster_expected, cluster_observed, track.state,
                        ))
                        continue

            self._update_track(
                lid, det, frame,
                freeze_appearance=close,
                preserve_raw_id=True,
            )
            if self.sm_enabled:
                self._sm_mark_accepted(self.tracks[lid], margin)
            output[lid] = det
            updated.add(lid)
            assigned_det.add(det_global_idx)
            frame_updates[lid] = det
            self.debug_records.append(IdentityDebug(
                frame, lid, det.raw_track_id, chosen, lid, margin,
                0, 0, 0, f"slot_{method}_accepted",
                reactivation_note,
                det.appearance_mode, det.detection_source,
                cluster_id, cluster_expected, cluster_observed, self.tracks[lid].state,
            ))

        # 第二阶段：被硬拒绝或无高分匹配的轨迹，用低分检测做纯运动/IoU回收。
        # 刻意保持歧义的轨迹不参与，避免低置信检测把ID带偏。
        if self.two_stage_enabled and low_cols:
            rescue_rows = [
                r for r in range(len(track_ids))
                if track_ids[r] not in updated
                and track_ids[r] not in output
                and (r in rejected_rows or r not in assigned_rows)
                and self.tracks[track_ids[r]].state != "lost"
            ]
            if rescue_rows:
                low_dets = [detections[i] for i in low_cols]
                low_cost = np.full((len(rescue_rows), len(low_dets)), self.INF_COST, dtype=np.float64)
                for rr_, r in enumerate(rescue_rows):
                    for cc_, det in enumerate(low_dets):
                        low_cost[rr_, cc_] = self._slot_low_cost(self.tracks[track_ids[r]], det, frame)
                if linear_sum_assignment is not None:
                    lr, lc = linear_sum_assignment(low_cost)
                    rescue_matches = list(zip(lr.tolist(), lc.tolist()))
                else:
                    rescue_matches = self._greedy_assignment(low_cost)
                for rr_, cc_ in rescue_matches:
                    r = rescue_rows[rr_]
                    lid = track_ids[r]
                    track = self.tracks[lid]
                    det = low_dets[cc_]
                    chosen = float(low_cost[rr_, cc_])
                    if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.two_stage_low_max_cost:
                        continue
                    # 低分回收只更新位置与运动，外观和长期模板全部冻结。
                    self._update_track(lid, det, frame, freeze_appearance=True, preserve_raw_id=True)
                    output[lid] = det
                    updated.add(lid)
                    assigned_det.add(low_cols[cc_])
                    self.debug_records.append(IdentityDebug(
                        frame, lid, det.raw_track_id, chosen, lid, float("nan"),
                        0, 0, 0, "slot_low_rescue_accepted", "bytetrack_style_motion_only",
                        det.appearance_mode, det.detection_source,
                        int(id_to_cluster.get(lid, -1)), 0, 0, self.tracks[lid].state,
                    ))

        for lid, track in self.tracks.items():
            if lid in updated:
                continue
            track.lock_strength = max(0.0, track.lock_strength - self.missing_lock_decay)
            if lid not in output:
                pred = self._make_slot_prediction(track, frame)
                if pred is not None:
                    output[lid] = pred

        # 新身份只允许由高分检测创建；低分检测不足以证明新动物出现。
        free_ids = [i for i in range(self.max_mice) if i not in self.tracks]
        for c in high_cols:
            det = detections[c]
            if c in assigned_det or not free_ids:
                continue
            if OcclusionClusterManager.detection_inside_forbidden(det, context):
                self.debug_records.append(IdentityDebug(
                    frame, -1, det.raw_track_id, 0.0, -1, float("nan"),
                    0, self.slot_new_track_confirm, 0,
                    "slot_suppressed_new_in_occlusion", "inside_existing_cluster",
                    det.appearance_mode, det.detection_source,
                ))
                continue
            nearest_bl = np.inf
            nearest_iou = 0.0
            for tr in self.tracks.values():
                body = max(float(np.nanmedian([tr.body_length_px, det.body_length_px])), 8.0)
                pred_center = self._prediction(tr, frame)
                d_bl = point_distance(pred_center, det.center_px) / body
                if d_bl < nearest_bl:
                    nearest_bl = d_bl
                    nearest_iou = 0.0
                    if tr.last_bbox_xyxy is not None:
                        shift = pred_center - tr.last_center_px
                        pred_box = np.asarray(tr.last_bbox_xyxy, dtype=np.float64).copy()
                        pred_box[[0, 2]] += shift[0]
                        pred_box[[1, 3]] += shift[1]
                        nearest_iou = bbox_iou_xyxy(pred_box, det.bbox_xyxy)
            if nearest_bl < self.slot_new_track_min_separation_bl and nearest_iou >= self.slot_new_track_dup_iou:
                self.debug_records.append(IdentityDebug(
                    frame, -1, det.raw_track_id, 0.0, -1, float("nan"),
                    0, self.slot_new_track_confirm, 0,
                    "slot_suppressed_duplicate", f"nearest_bl={nearest_bl:.3f};iou={nearest_iou:.3f}",
                    det.appearance_mode, det.detection_source,
                ))
                continue
            sig = self._position_signature(det)
            state = self.pending_new_tracks.get(sig)
            if state and frame - int(state.get("last_frame", frame)) <= self.slot_new_track_max_gap:
                count = int(state.get("count", 0)) + 1
            else:
                count = 1
            self.pending_new_tracks[sig] = {"count": count, "last_frame": frame, "detection": det}
            if count < self.slot_new_track_confirm:
                self.debug_records.append(IdentityDebug(
                    frame, -1, det.raw_track_id, 0.0, -1, float("nan"),
                    count, self.slot_new_track_confirm, 0,
                    "slot_pending_new", f"confirm({count}/{self.slot_new_track_confirm})",
                    det.appearance_mode, det.detection_source,
                ))
                continue
            lid = free_ids.pop(0)
            self.pending_new_tracks.pop(sig, None)
            self._update_track(lid, det, frame, preserve_raw_id=True)
            output[lid] = det
            updated.add(lid)
            assigned_det.add(c)
            frame_updates[lid] = det
            self.debug_records.append(IdentityDebug(
                frame, lid, det.raw_track_id, 0.0, lid, 1.0,
                count, self.slot_new_track_confirm, 0,
                "slot_new_confirmed", "fixed_slot_created_once",
                det.appearance_mode, det.detection_source,
                -1, 0, 0, self.tracks[lid].state,
            ))

        # 滑窗重拼接：双向外观证据持续成立时回滚此前的ID互换。
        if self.restitch_enabled and frame_updates:
            self._restitch_check(frame, frame_updates, frozen_ids)

        return sorted(output.items(), key=lambda item: item[0])


class MemoryIdentityAssigner(LegacyStableSlotAssigner):
    """短时身份记忆分配器（修复文档v1.1 §3、§5、§6、§7、§11、§12）。

    与 LegacyStableSlotAssigner 的核心差异：

    1. **检测与身份分离（§3）**：当前帧每一个有效检测都输出并渲染，
       身份只能是 confirmed / tentative(TMP) / suspicious(ID?)，绝不消失。
    2. **未匹配检测立即创建TMP临时轨迹（§5第6步、§6.4）**：第一帧起就有框
       和标签；连续 confirm_frames 帧命中后晋升为固定槽位；TMP有容量上限和TTL。
    3. **每条轨迹维护 TemporaryIdentityMemory（§12.2）**：位置/速度中位数、
       方向、姿态历史、体长体宽EMA、外观EMA、邻近关系。
    4. **匹配代价（§12.4）**：0.35运动预测 + 0.20身体中心 + 0.15姿态 +
       0.10方向 + 0.10体型 + 0.10外观；关键点不足自动退化为
       运动+中心+尺度匹配（§6.6，MIN_KEYPOINTS_FOR_TRACKING=0）。
    5. **距离门限按体长归一化（§7）**：普通1.2倍体长、高速2.0倍体长两套门限。
    6. **身份冲突保留检测（§6.7、§12.9）**：匹配边际不足时轨迹进入suspicious、
       冻结外观/体型记忆更新，但检测框与关键点持续渲染；配合OKS+长期外观
       仲裁与滑窗重拼接，在分开后回溯修正（§12.6）。
    7. **记忆绝不成为新的检测过滤器（§12.1）**。
    """

    TMP_ID_BASE = 10000
    # 溢出显示标签：TMP容量满且全部为新鲜轨迹时，额外检测用纯显示标签渲染，
    # 不占追踪容量、不参与晋升——任何情况下检测框都不消失（§9）。
    OVERFLOW_ID_BASE = 20000

    def __init__(self, config: Mapping[str, Any], max_mice: int = 20) -> None:
        super().__init__(config, max_mice=max_mice)
        mem = dict(config.get("memory", {}))
        self.mem_enabled = bool(mem.get("enabled", True))
        self.mem_window = max(int(mem.get("window_frames", 30)), 3)
        self.mem_velocity_window = max(int(mem.get("velocity_window_frames", 7)), 1)
        self.mem_suspicious_window = max(int(mem.get("suspicious_window_frames", 5)), 1)
        self.mem_id_margin = float(mem.get("id_margin_threshold", 0.08))
        self.mem_lost_ttl = max(int(mem.get("lost_ttl_frames", 15)), 1)
        self.mem_ema_alpha = float(mem.get("ema_alpha", 0.10))
        self.mem_confirm_frames = max(int(mem.get("confirm_frames", 2)), 1)
        self.mem_tmp_ttl = max(int(mem.get("tmp_ttl_frames", 15)), 1)
        self.mem_max_tentative = max(int(mem.get("max_tentative_tracks", 10)), 1)
        self.mem_recycle_lost_frames = int(mem.get("recycle_lost_after_frames", 300))
        self.mem_normal_gate_bl = float(mem.get("normal_distance_gate_body_lengths", 1.2))
        self.mem_fast_gate_bl = float(mem.get("fast_distance_gate_body_lengths", 2.0))
        self.mem_fast_speed_bl = float(mem.get("fast_speed_gate_body_lengths_per_frame", 0.6))
        self.mem_max_cost = float(mem.get("max_assignment_cost", 1.05))
        self.mem_low_max_cost = float(mem.get("low_max_cost", 0.50))
        # 运动门控晋升：来自这些通道（前缀匹配）的TMP轨迹必须观察到足够位移
        # 才能晋升固定槽位。v1.11.3起默认"*"（全部通道）——墙上水痕等静止假目标
        # 也可能被Pose模型画上骨架而携带姿态证据，只有"持续位移"是真鼠的可靠证据；
        # 真鼠TMP保持命中不会过期（tmp_ttl按misses计），静止真鼠维持TMP显示直至移动。
        self.mem_promotion_motion_sources = tuple(
            str(s) for s in mem.get("promotion_motion_required_sources", ["*"])
        )
        self.mem_promotion_min_disp_bl = float(mem.get("promotion_min_displacement_body_lengths", 0.6))
        # v1.12.4 并列判据：活动范围直径（体长倍数）。抱团聚拢鼠净位移虽小，
        # 但扭动使访问点集直径增大；静止水痕两项都停在抖动量级。
        self.mem_promotion_min_diameter_bl = float(mem.get("promotion_min_diameter_body_lengths", 0.5))
        # v1.12.6：接触簇内TMP累计稳定命中≥N帧后解除禁升（默认60帧=2秒，
        # 远严于confirm_frames=2）。§12.8簇内禁升是为防ID给错鼠，但长期抱团
        # 的两只鼠永远等不到"离开簇"解禁（实跑左下角TMP持续200+帧）。
        # 累计2秒稳定命中足以证明追踪的是真实个体。设0恢复严格§12.8。
        self.mem_contact_promote_total_hits = int(mem.get("contact_cluster_promote_min_total_hits", 60))
        # v1.12.10 慢车道晋升：累计命中≥N帧且骨架解剖学有效占比≥ratio的TMP，
        # 跳过"连续命中"与"运动门控"直接晋升——治静止抱团真鼠（位移/直径
        # 双双不达标的真个体）与白鼠闪烁通道（连续命中凑不齐）永远TMP。
        # 骨架证据把静止水痕/反光挡在门外（水痕的骨架解剖学必坏）。
        self.mem_slow_lane_min_total_hits = int(mem.get("slow_lane_promote_min_total_hits", 90))
        self.mem_slow_lane_min_kp_ratio = float(mem.get("slow_lane_min_keypoint_ok_ratio", 0.6))
        # 抖动级运动下限（体长）：静止水痕抖动≈0.1体长，慢车道要求≥0.25，
        # 防止"骨架画得像样的水痕"借慢车道晋升（双保险）。
        self.mem_slow_lane_min_motion_bl = float(mem.get("slow_lane_min_motion_body_lengths", 0.25))
        # v1.12.11 速度自适应跳变门：低速/静止的confirmed轨迹在短间隔（≤2帧）
        # 内不得接受远超其运动能力的跳变——实跑中静止的ID4在1帧漏检后抓了
        # 0.9体长外的别鼠检测（身份被偷、轨迹端点污染、尾段4→10跳变根源）。
        # cap = max(min, scale×中位速度+base)：静止轨迹0.5体长；
        # 追逐快鼠（0.2体长/帧）1.4体长，不影响正常追踪。
        self.mem_jump_gate_min_bl = float(mem.get("jump_gate_min_body_lengths", 0.5))
        self.mem_jump_gate_speed_scale = float(mem.get("jump_gate_speed_scale", 6.0))
        self.mem_jump_gate_base_bl = float(mem.get("jump_gate_base_body_lengths", 0.2))
        # v1.13.0 简洁运动身份模式（默认开）：只靠轨迹段+位置+朝向+速度判ID。
        # 匹配代价=运动预测+中心+朝向+尺度（不算外观直方图、不算OKS——更快，
        # 接触时关键点交叉也不漂移）；匹配边际不足宁可断轨转lost，交轨迹段
        # 融合（位置+朝向+速度）事后裁决，不冻结、不仲裁、不标可疑。
        self.simple_motion = bool(mem.get("simple_motion_mode", True))
        self.simple_ambiguous_reject = bool(mem.get("simple_ambiguous_reject", True))
        # v1.12.5 轨迹段融合（参考苏峰博士论文§4.2.8 MOT-SP / §4.2.9 MOT-TP）：
        # 严格在线匹配（宁可断轨也不错配）会产生轨迹段；旧ID回收后把片段摘要
        # 入池，新TMP晋升前用"最大值+差值"双条件门控融合回原ID——
        # 距离足够近且领先第二名足够多才融合，缺一按新个体处理，绝不勉强。
        tf = dict(mem.get("tracklet_fusion", {}))
        self.tf_enabled = bool(tf.get("enabled", True))
        self.tf_max_gap_frames = int(tf.get("max_gap_frames", 600))
        self.tf_fragment_ttl = int(tf.get("fragment_ttl_frames", 300))
        self.tf_dist_thre_bl = float(tf.get("dist_thre_body_lengths", 0.5))
        self.tf_dist_diff_bl = float(tf.get("dist_diff_body_lengths", 0.25))
        self.tf_iou_thre = float(tf.get("iou_thre", 0.15))
        self.tf_iou_diff = float(tf.get("iou_diff_thre", 0.10))
        # IoU的OR门只用于短间隔：长间隔后任何鼠途经同一区域框都会重叠，
        # 端点框IoU是弱证据（实测会误融合）；长间隔只认距离门+速度预测。
        self.tf_iou_max_gap = int(tf.get("iou_max_gap_frames", 30))
        # v1.12.8 lost轨迹吸收：实跑残余跳变的主因——接触中老轨迹变lost但
        # 仍占着槽位，同一只鼠被新TMP接管，TMP晋升只能领新号（6→7）。
        # 晋升时对"活着的lost轨迹"跑同一套双条件门控，无争议就接回原ID。
        self.tf_absorb_lost = bool(tf.get("absorb_lost_tracks", True))
        self.tf_absorb_min_misses = int(tf.get("absorb_min_misses", 10))
        self.tf_absorb_coexist_tol = int(tf.get("absorb_coexist_tolerance_frames", 5))
        # v1.12.11：共存守卫的位置豁免距离（体长）——lost轨迹在TMP创建后的
        # 命中点若仍在TMP起点附近，是"同一只鼠漏过去的检测"而非共存。
        self.tf_absorb_coexist_override_bl = float(
            tf.get("absorb_coexist_override_body_lengths", 0.6)
        )
        # v1.13.0：融合评分的朝向权重（用户点名证据维度：位置+朝向+速度）。
        self.tf_heading_weight = float(tf.get("fusion_heading_weight", 0.25))
        self.tf_predict_max = int(tf.get("velocity_predict_max_frames", 30))
        # v1.12.10 TMP死亡片段证据继承：闪烁通道（白鼠亮斑）的TMP反复
        # 死亡重建，每次晋升证据（累计命中/位移/直径/骨架有效命中）清零，
        # 永远凑不够晋升条件。死亡的TMP把证据存为短时效片段，同一只鼠的
        # 下一个TMP就近无争议地继承——继承的只是"晋升就绪度证据"，TMP
        # 没有身份，不涉及任何ID转移，不会引入互换风险。
        self.tf_tmp_chain_enabled = bool(tf.get("tmp_chain_inherit", True))
        self.tf_tmp_chain_ttl = int(tf.get("tmp_chain_ttl_frames", 90))
        self.tf_tmp_chain_dist = float(tf.get("tmp_chain_dist_body_lengths", 0.5))
        self.tf_tmp_chain_margin = float(tf.get("tmp_chain_margin_body_lengths", 0.25))
        mw = dict(mem.get("weights", {}))
        self.mw_motion = float(mw.get("motion", 0.35))
        self.mw_center = float(mw.get("center", 0.20))
        self.mw_pose = float(mw.get("pose", 0.15))
        self.mw_heading = float(mw.get("heading", 0.10))
        self.mw_size = float(mw.get("size", 0.10))
        self.mw_appearance = float(mw.get("appearance", 0.10))
        # v1.11.3 近距离接触保护：两只鼠接触时关键点互相交叉，姿态/外观证据
        # 不可靠（ID互换的主要来源）。接触中的检测对自动降姿态/外观权重，
        # 同时身份边际阈值加倍——宁可标ID?冻结身份，也不做低置信互换。
        self.mem_contact_crowd_bl = float(mem.get("contact_crowd_body_lengths", 1.0))
        self.mem_contact_pose_scale = float(mem.get("contact_pose_weight_scale", 0.33))
        self.mem_contact_app_scale = float(mem.get("contact_appearance_weight_scale", 0.5))
        self.mem_contact_margin_scale = float(mem.get("contact_margin_scale", 2.0))

        # TMP临时轨迹独立存储；固定槽位仍只在self.tracks中。
        self.tmp_tracks: Dict[int, IdentityTrack] = {}
        self._tmp_promotion_blocked: set[int] = set()
        self._tmp_motion_required: set[int] = set()
        # v1.11.3：TMP自创建点起的最大位移（不受记忆窗口30帧上限影响），
        # 慢速鼠也能在足够时间后达标；静止假目标（水痕）永远为0。
        self._tmp_origin_center: Dict[int, np.ndarray] = {}
        self._tmp_max_displacement: Dict[int, float] = {}
        # v1.12.4：活动范围直径估计（创建点/最远点/新点的最大两两距离）。
        self._tmp_far_point: Dict[int, np.ndarray] = {}
        self._tmp_diameter: Dict[int, float] = {}
        # v1.12.6：TMP累计命中计数（不随miss清零），簇内解禁用。
        self._tmp_total_hits: Dict[int, int] = {}
        # v1.12.8：TMP最新一次检测（吸收lost轨迹时用来复活老轨迹）。
        self._tmp_last_det: Dict[int, Detection] = {}
        # v1.12.8：TMP创建帧（lost吸收的共存守卫用）。
        self._tmp_created_frame: Dict[int, int] = {}
        # v1.12.10：TMP命中中骨架解剖学有效的次数（慢车道晋升证据）。
        self._tmp_kp_ok_hits: Dict[int, int] = {}
        # v1.12.10：死亡TMP的晋升证据片段池（短时效，证据继承用）。
        self._tmp_chain_pool: Dict[int, Dict[str, Any]] = {}
        self._tmp_chain_seq = 0
        # v1.12.5：已回收轨迹的片段摘要池（轨迹段融合候选）。
        self._fragment_pool: Dict[int, Dict[str, Any]] = {}
        self._overflow_ids: set[int] = set()
        # 每帧输出元信息：主程序据此渲染TMP/ID?标签并写检测级CSV。
        self.output_info: Dict[int, Dict[str, Any]] = {}
        # 每帧阶段统计（§8日志与硬性校验）。
        self.frame_stats: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 轨迹集合视图
    # ------------------------------------------------------------------

    @property
    def cluster_tracks(self) -> Dict[int, IdentityTrack]:
        """接触簇计算使用固定槽+TMP全部轨迹。"""
        if not self.tmp_tracks:
            return self.tracks
        merged = dict(self.tracks)
        merged.update(self.tmp_tracks)
        return merged

    def _all_track_items(self) -> List[Tuple[int, IdentityTrack]]:
        return list(self.tracks.items()) + list(self.tmp_tracks.items())

    # ------------------------------------------------------------------
    # 记忆维护（§12.2、§12.8）
    # ------------------------------------------------------------------

    def _ensure_memory(self, track: IdentityTrack) -> TemporaryIdentityMemory:
        mem = track.memory
        if mem is None:
            mem = TemporaryIdentityMemory(track_id=track.logical_id)
            for name in (
                "center_history", "velocity_history", "keypoint_history",
                "body_center_history", "direction_history", "neighbor_relation_history",
            ):
                setattr(mem, name, deque(maxlen=self.mem_window))
            track.memory = mem
        return mem

    def _memory_update(self, track: IdentityTrack, det: Detection, frame: int, allow_identity: bool) -> None:
        """每次命中更新运动历史；体型与外观EMA仅在安全条件满足时更新（§12.8）。

        安全条件由调用方保证：高置信、关键点相对完整、邻近小鼠足够远、
        匹配边际明确、轨迹非suspicious。接触/冲突/可疑时只更新运动状态。
        """
        mem = self._ensure_memory(track)
        center = det.center_px.astype(np.float64)
        if mem.center_history and mem.last_frame >= 0 and frame > mem.last_frame:
            previous = np.asarray(mem.center_history[-1], dtype=np.float64)
            if np.all(np.isfinite(previous)):
                mem.velocity_history.append((center - previous) / float(frame - mem.last_frame))
        mem.center_history.append(center)

        box = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)
        box_valid = box.size >= 4 and bool(np.all(np.isfinite(box[:4])))
        if box_valid:
            mem.body_center_history.append(np.array(
                [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64
            ))
        mem.keypoint_history.append((
            np.asarray(det.keypoints_px, dtype=np.float64).copy(),
            np.asarray(det.keypoint_conf, dtype=np.float64).copy(),
        ))
        if det.heading_vector is not None and np.all(np.isfinite(det.heading_vector)):
            mem.direction_history.append(np.asarray(det.heading_vector, dtype=np.float64))

        # 邻近关系：其他轨迹中心相对本轨迹的体长归一化位移（§12.2）。
        body = max(float(np.nanmedian([track.body_length_px, det.body_length_px])), 8.0)
        relations: Dict[int, Tuple[float, float]] = {}
        for other_id, other in self._all_track_items():
            if other_id == track.logical_id or not finite_point(other.last_center_px):
                continue
            delta = (np.asarray(other.last_center_px, dtype=np.float64) - center) / body
            if np.all(np.isfinite(delta)):
                relations[int(other_id)] = (float(delta[0]), float(delta[1]))
        mem.neighbor_relation_history.append(relations)
        mem.last_frame = frame

        if not allow_identity:
            return
        body_length = float(det.body_length_px)
        if np.isfinite(body_length) and body_length > 3:
            if not np.isfinite(mem.body_length_ema):
                mem.body_length_ema = body_length
            else:
                mem.body_length_ema = (
                    (1.0 - self.mem_ema_alpha) * mem.body_length_ema
                    + self.mem_ema_alpha * body_length
                )
        if box_valid:
            width = float(min(max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)))
            if not np.isfinite(mem.body_width_ema):
                mem.body_width_ema = width
            else:
                mem.body_width_ema = (
                    (1.0 - self.mem_ema_alpha) * mem.body_width_ema
                    + self.mem_ema_alpha * width
                )
        if det.appearance_reliable and det.appearance_feature is not None:
            mem.appearance_feature_ema = self._ema_array(
                mem.appearance_feature_ema, det.appearance_feature, self.mem_ema_alpha
            )

    def _memory_mark_miss(self, track: IdentityTrack) -> None:
        mem = self._ensure_memory(track)
        mem.misses += 1
        mem.hits = 0
        mem.identity_confidence = max(0.0, mem.identity_confidence * 0.90)

    def _memory_transition_on_miss(self, track: IdentityTrack) -> None:
        """连续漏检超过LOST_TTL后进入lost；lost不得伪装为当前检测（§3.2）。"""
        mem = track.memory
        misses = mem.misses if mem is not None else 0
        if track.state != "lost" and misses > self.mem_lost_ttl:
            track.state = "lost"
            track.clean_streak = 0
            if mem is not None:
                mem.state = "lost"

    # ------------------------------------------------------------------
    # 记忆预测、门限与代价（§7、§12.4、§6.6）
    # ------------------------------------------------------------------

    def _memory_prediction(self, track: IdentityTrack, frame: int) -> np.ndarray:
        mem = track.memory
        dt = max(frame - track.last_frame, 0)
        if mem is not None and mem.velocity_history:
            velocity = mem.median_velocity(self.mem_velocity_window)
        else:
            velocity = track.velocity_px_per_frame
        return track.last_center_px + self.velocity_weight * np.asarray(velocity, dtype=np.float64) * min(dt, 12)

    def _memory_norm_body(self, track: IdentityTrack, det: Detection) -> float:
        lengths = [track.body_length_px, det.body_length_px]
        if track.memory is not None and np.isfinite(track.memory.body_length_ema):
            lengths.append(track.memory.body_length_ema)
        valid = [float(v) for v in lengths if np.isfinite(v) and v > 3]
        return max(float(np.nanmedian(valid)) if valid else 20.0, 8.0)

    def _memory_gate(self, track: IdentityTrack, frame: int) -> float:
        """普通1.2倍/高速2.0倍体长双门限（§7），漏检越久可达范围越大。"""
        body = max(float(track.body_length_px), 8.0) if np.isfinite(track.body_length_px) else 8.0
        mem = track.memory
        if mem is not None and mem.velocity_history:
            speed_bl = float(np.linalg.norm(mem.median_velocity(self.mem_velocity_window))) / body
        else:
            speed_bl = float(np.linalg.norm(track.velocity_px_per_frame)) / body
        gate = self.mem_fast_gate_bl if speed_bl >= self.mem_fast_speed_bl else self.mem_normal_gate_bl
        dt = max(frame - track.last_frame, 1)
        gate += min(dt - 1, 8) * 0.38
        if track.state == "lost":
            gate = max(gate, self.mem_fast_gate_bl)
        return float(gate)

    def _memory_jump_cap_bl(self, track: IdentityTrack, norm: float) -> float:
        """v1.12.11 速度自适应跳变上限（体长）：max(min, scale×中位速度+base)。

        实跑教训：静止的confirmed轨迹在1帧漏检后抓了0.9体长外的别鼠检测
        （门限1.2体长放行）——身份被偷、端点被污染，吸收都救不回来。
        静止轨迹 cap=0.5 体长：30fps下真实鼠单帧位移极少超过0.5体长；
        追逐快鼠按中位速度放宽（0.2体长/帧 → cap=1.4），正常追踪不受影响。"""
        mem = track.memory
        body = max(float(norm), 8.0)
        if mem is not None and mem.velocity_history:
            speed_px = float(np.linalg.norm(mem.median_velocity(self.mem_velocity_window)))
        else:
            v = np.asarray(track.velocity_px_per_frame, dtype=np.float64)
            speed_px = float(np.linalg.norm(v)) if np.all(np.isfinite(v)) else 0.0
        speed_bl = speed_px / body
        return max(
            self.mem_jump_gate_min_bl,
            self.mem_jump_gate_speed_scale * speed_bl + self.mem_jump_gate_base_bl,
        )

    @staticmethod
    def _valid_keypoint_count(det: Detection, min_conf: float = 0.10) -> int:
        points = np.asarray(det.keypoints_px, dtype=np.float64)
        conf = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
        n = min(len(KEYPOINT_NAMES), len(points), len(conf))
        if n <= 0:
            return 0
        valid = (
            np.isfinite(points[:n, 0]) & np.isfinite(points[:n, 1])
            & (points[:n, 0] > 0) & (points[:n, 1] > 0)
            & np.isfinite(conf[:n]) & (conf[:n] >= min_conf)
        )
        return int(valid.sum())

    def _memory_size_distance(self, track: IdentityTrack, det: Detection) -> float:
        vals: List[float] = []
        mem = track.memory
        ref_length = mem.body_length_ema if (mem is not None and np.isfinite(mem.body_length_ema)) else track.body_length_px
        if np.isfinite(ref_length) and ref_length > 1 and np.isfinite(det.body_length_px) and det.body_length_px > 1:
            vals.append(abs(math.log(max(det.body_length_px, 1.0) / max(float(ref_length), 1.0))))
        box = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)
        if box.size >= 4 and np.all(np.isfinite(box[:4])):
            det_width = float(min(max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)))
            ref_width: Optional[float] = None
            if mem is not None and np.isfinite(mem.body_width_ema):
                ref_width = float(mem.body_width_ema)
            elif track.bbox_wh is not None and np.all(np.isfinite(track.bbox_wh)):
                ref_width = float(np.min(np.maximum(track.bbox_wh, 1.0)))
            if ref_width is not None and ref_width > 1:
                vals.append(abs(math.log(det_width / ref_width)))
        return float(np.clip(np.mean(vals), 0.0, 1.5)) if vals else 0.4

    def _detection_crowding(self, dets: Sequence[Detection]) -> List[bool]:
        """标记"接触簇"内的检测：与其他检测中心距 < contact_crowd_bl × 体长。

        接触中的小鼠关键点互相交叉，姿态代价会把身份拉向邻居（ID互换来源）。
        """
        flags = [False] * len(dets)
        centers = [d.center_px for d in dets]
        bodies = [max(float(d.body_length_px), 8.0) for d in dets]
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                dist = point_distance(centers[i], centers[j])
                if not np.isfinite(dist):
                    continue
                if dist < self.mem_contact_crowd_bl * max(bodies[i], bodies[j]):
                    flags[i] = True
                    flags[j] = True
        return flags

    def _memory_cost(
        self,
        track: IdentityTrack,
        det: Detection,
        frame: int,
        crowded: bool = False,
    ) -> float:
        """§12.4总匹配代价；关键点不足自动退化（§6.6）；接触簇内降姿态权重。"""
        pred = self._memory_prediction(track, frame)
        norm = self._memory_norm_body(track, det)
        d_pred = point_distance(pred, det.center_px)
        if not np.isfinite(d_pred):
            return self.INF_COST
        gate = self._memory_gate(track, frame)
        # v1.12.11：confirmed轨迹在≤2帧短间隔内套用速度自适应跳变门
        # （只拦"预测外跳变"；预测已含速度补偿，匀速快鼠d_pred≈0不受影响；
        # TMP暂不适用——它抓错的代价低，且避免闪烁通道TMP跟不上快鼠而churn）。
        if track.logical_id < self.TMP_ID_BASE and frame - int(track.last_frame) <= 2:
            gate = min(gate, self._memory_jump_cap_bl(track, norm))
        d_pred_bl = d_pred / norm
        if d_pred_bl > gate:
            return self.INF_COST
        d_last = point_distance(track.last_center_px, det.center_px)
        d_last_bl = d_last / norm if np.isfinite(d_last) else d_pred_bl

        motion_cost = float(np.clip(d_pred_bl / max(gate, 1e-6), 0.0, 1.5))
        center_cost = float(np.clip(d_last_bl / max(gate, 1e-6), 0.0, 1.5))
        size_cost = self._memory_size_distance(track, det)

        det_kpts = self._valid_keypoint_count(det)
        track_kpts = 0
        if track.last_keypoint_conf is not None:
            tc = np.asarray(track.last_keypoint_conf, dtype=np.float64).reshape(-1)
            track_kpts = int(np.sum(np.isfinite(tc) & (tc >= 0.10)))

        # §6.6：≥4点完整姿态；2~3点部分姿态；<2点只用中心/尺度/运动。
        if det_kpts >= 4 and track_kpts >= 4:
            pose_cost = self._oks_pose_cost(track, det)
            if pose_cost is None:
                pose_cost = self._pose_distance(track, det)
            heading_cost = self._heading_distance(track, det)
            pose_ok = True
        elif det_kpts >= 2 and track_kpts >= 2:
            pose_cost = min(self._pose_distance(track, det), self._anchor_distance(track, det))
            heading_cost = self._heading_distance(track, det)
            pose_ok = True
        else:
            pose_cost = heading_cost = 0.0
            pose_ok = False

        appearance_cost: Optional[float] = None
        mem = track.memory
        if det.appearance_reliable and det.appearance_feature is not None:
            candidates: List[float] = []
            if mem is not None and mem.appearance_feature_ema is not None:
                candidates.append(self._feature_l1(mem.appearance_feature_ema, det.appearance_feature, 0.45, 2.0))
            long_distance = self._appearance_long_distance(track, det)
            if long_distance is not None:
                candidates.append(long_distance)
            if track.appearance_feature is not None:
                candidates.append(self._feature_l1(track.appearance_feature, det.appearance_feature, 0.45, 2.0))
            if candidates:
                appearance_cost = min(candidates)

        if pose_ok:
            w_pose = self.mw_pose
            w_app = self.mw_appearance
            if crowded:
                # 接触中关键点交叉，姿态/外观证据不可靠（v1.11.3）。
                w_pose *= self.mem_contact_pose_scale
                w_app *= self.mem_contact_app_scale
            terms = [
                (self.mw_motion, motion_cost),
                (self.mw_center, center_cost),
                (w_pose, pose_cost),
                (self.mw_heading, heading_cost),
                (self.mw_size, size_cost),
            ]
            if appearance_cost is not None:
                terms.append((w_app, appearance_cost))
        else:
            # 白鼠bbox-only、严重遮挡：退化为运动+中心+尺度（§6.6禁止删框）。
            terms = [
                (self.mw_motion, motion_cost),
                (self.mw_center, center_cost),
                (self.mw_size, size_cost),
            ]
        total_weight = sum(w for w, _ in terms)
        if total_weight <= 1e-9:
            return self.INF_COST
        return float(sum(w * c for w, c in terms) / total_weight)

    def _memory_low_cost(self, track: IdentityTrack, det: Detection, frame: int) -> float:
        """低分检测回收代价：只用运动预测、IoU和尺度（外观/姿态不可靠）。"""
        pred = self._memory_prediction(track, frame)
        norm = self._memory_norm_body(track, det)
        d_pred = point_distance(pred, det.center_px)
        if not np.isfinite(d_pred):
            return self.INF_COST
        gate = self._memory_gate(track, frame)
        d_pred_bl = d_pred / norm
        if d_pred_bl > gate:
            return self.INF_COST
        d_last = point_distance(track.last_center_px, det.center_px)
        d_last_bl = d_last / norm if np.isfinite(d_last) else d_pred_bl
        center_cost = float(np.clip((0.82 * d_pred_bl + 0.18 * d_last_bl) / max(gate, 1e-6), 0.0, 1.5))
        if track.last_bbox_xyxy is None:
            iou_cost = 0.50
        else:
            shift = pred - track.last_center_px
            pred_box = np.asarray(track.last_bbox_xyxy, dtype=np.float64).copy()
            pred_box[[0, 2]] += shift[0]
            pred_box[[1, 3]] += shift[1]
            iou_cost = 1.0 - bbox_iou_xyxy(pred_box, det.bbox_xyxy)
        size_cost = self._memory_size_distance(track, det)
        return float(0.60 * center_cost + 0.25 * iou_cost + 0.15 * min(size_cost, 1.0))

    def _memory_nearest_other_bl(self, track: IdentityTrack) -> float:
        best = np.inf
        for other_id, other in self._all_track_items():
            if other_id == track.logical_id or not finite_point(other.last_center_px):
                continue
            body = max(float(np.nanmedian([track.body_length_px, other.body_length_px])), 8.0)
            best = min(best, point_distance(track.last_center_px, other.last_center_px) / body)
        return float(best)

    def _memory_find_competitor(
        self,
        cost_matrix: np.ndarray,
        row: int,
        col: int,
        track_ids: Sequence[int],
        track_map: Mapping[int, IdentityTrack],
        detection: Detection,
        frame: int,
    ) -> Optional[int]:
        """寻找对同一检测构成竞争、且空间上确实贴近的另一条轨迹（含TMP）。"""
        column = cost_matrix[:, col]
        for idx in np.argsort(column).tolist():
            idx = int(idx)
            if idx == row or not np.isfinite(column[idx]) or column[idx] >= self.INF_COST:
                continue
            other = track_map[track_ids[idx]]
            body = max(float(np.nanmedian([other.body_length_px, detection.body_length_px])), 8.0)
            d_bl = point_distance(self._memory_prediction(other, frame), detection.center_px) / body
            if np.isfinite(d_bl) and d_bl <= self.hp_close_bl:
                return track_ids[idx]
        return None

    # ------------------------------------------------------------------
    # TMP临时轨迹：创建、过期、晋升（§5第6步、§6.4、§4.7）
    # ------------------------------------------------------------------

    def _next_tmp_id(self) -> int:
        for n in range(1, self.mem_max_tentative + 1):
            candidate = self.TMP_ID_BASE + n
            if candidate not in self.tmp_tracks:
                return candidate
        return -1

    def _drop_tmp(self, tmp_id: int, frame: int, reason: str) -> None:
        track = self.tmp_tracks.pop(tmp_id, None)
        # v1.12.10：死亡前把晋升证据存为短时效片段，供同一只鼠的下一个TMP继承
        # （闪烁通道反复重建不再从零累计）。至少累计过3次命中才值得留片段。
        if self.tf_tmp_chain_enabled and track is not None:
            total = int(self._tmp_total_hits.get(tmp_id, 0))
            if total >= 3:
                self._tmp_chain_seq += 1
                bbox = track.last_bbox_xyxy
                self._tmp_chain_pool[self._tmp_chain_seq] = {
                    "end_frame": int(track.last_frame),
                    "end_center": np.asarray(track.last_center_px, dtype=np.float64).copy(),
                    "end_bbox": (np.asarray(bbox, dtype=np.float64).copy()
                                 if bbox is not None else np.full(4, np.nan)),
                    "body_length": float(track.body_length_px),
                    "total_hits": total,
                    "max_displacement": float(self._tmp_max_displacement.get(tmp_id, 0.0)),
                    "diameter": float(self._tmp_diameter.get(tmp_id, 0.0)),
                    "kp_ok_hits": int(self._tmp_kp_ok_hits.get(tmp_id, 0)),
                }
        self._tmp_promotion_blocked.discard(tmp_id)
        self._tmp_motion_required.discard(tmp_id)
        self._tmp_origin_center.pop(tmp_id, None)
        self._tmp_max_displacement.pop(tmp_id, None)
        self._tmp_far_point.pop(tmp_id, None)
        self._tmp_diameter.pop(tmp_id, None)
        self._tmp_total_hits.pop(tmp_id, None)
        self._tmp_kp_ok_hits.pop(tmp_id, None)
        self._tmp_last_det.pop(tmp_id, None)
        self._tmp_created_frame.pop(tmp_id, None)
        if track is not None and track.raw_track_id is not None:
            if self.raw_to_logical.get(track.raw_track_id) == tmp_id:
                self.raw_to_logical.pop(track.raw_track_id, None)
        if track is not None:
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=tmp_id, raw_track_id=track.raw_track_id,
                assignment_cost=0.0, proposed_logical_id=tmp_id, assignment_gain=float("nan"),
                dwell_count=0, dwell_required=0, cooldown_remaining=0,
                commit_status="tmp_dropped", switch_rejected_reason=reason,
                appearance_mode="", detection_source="",
                occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                track_state="tentative",
            ))

    def _create_tmp_track(self, det: Detection, frame: int, forbid_promote: bool) -> int:
        tmp_id = self._next_tmp_id()
        if tmp_id < 0:
            # 容量满：优先回收“陈旧TMP”（本帧未命中）；全部新鲜时用溢出显示标签，
            # 保证同帧每个检测都有独立框和标签（§9 rendered==detections）。
            stale = [
                i for i, t in self.tmp_tracks.items()
                if (t.memory.misses if t.memory else 0) > 0 or t.last_frame < frame
            ]
            if stale:
                worst_id = max(
                    stale,
                    key=lambda i: (
                        self.tmp_tracks[i].memory.misses if self.tmp_tracks[i].memory else 0,
                        -self.tmp_tracks[i].last_frame,
                    ),
                )
                self._drop_tmp(worst_id, frame, "tmp_capacity_recycle")
                tmp_id = worst_id
            else:
                overflow_id = self.OVERFLOW_ID_BASE + len(self._overflow_ids) + 1
                self._overflow_ids.add(overflow_id)
                return overflow_id
        self._update_track(tmp_id, det, frame, freeze_appearance=False,
                           preserve_raw_id=True, store=self.tmp_tracks)
        track = self.tmp_tracks[tmp_id]
        track.state = "tentative"
        mem = self._ensure_memory(track)
        mem.track_id = tmp_id
        mem.state = "tentative"
        mem.hits = 1
        mem.misses = 0
        mem.identity_confidence = 0.30
        self._memory_update(track, det, frame, allow_identity=False)
        if forbid_promote:
            self._tmp_promotion_blocked.add(tmp_id)
        # v1.11.3：记录TMP创建点，运动门控用"自创建点起最大位移"判定。
        self._tmp_origin_center[tmp_id] = np.asarray(det.center_px, dtype=np.float64).copy()
        self._tmp_max_displacement[tmp_id] = 0.0
        # v1.12.4：活动范围直径估计（创建点↔最远点↔新点的最大两两距离，
        # 单调递增、对静止水痕的中心抖动免疫）。抱团聚拢鼠位移虽小但
        # 活动范围直径会随扭动累积，作为运动门控的并列判据。
        self._tmp_far_point[tmp_id] = np.asarray(det.center_px, dtype=np.float64).copy()
        self._tmp_diameter[tmp_id] = 0.0
        self._tmp_total_hits[tmp_id] = 0
        self._tmp_kp_ok_hits[tmp_id] = 0
        self._tmp_created_frame[tmp_id] = int(frame)
        # v1.12.10：就近无争议地继承死亡TMP的晋升证据（防闪烁通道反复清零）。
        self._inherit_tmp_chain(tmp_id, frame)
        # 运动门控（v1.11.3默认全源）：晋升前必须观察到足够位移，
        # 排除静止反光假斑与墙上水痕（后者可能被Pose误画骨架）。
        if self.mem_promotion_motion_sources:
            src = str(det.detection_source)
            if "*" in self.mem_promotion_motion_sources or src.startswith(
                self.mem_promotion_motion_sources
            ):
                self._tmp_motion_required.add(tmp_id)
        return tmp_id

    def _inherit_tmp_chain(self, tmp_id: int, frame: int) -> None:
        """v1.12.10：新TMP就近无争议地继承死亡TMP的晋升证据。

        闪烁通道（白鼠亮斑）的TMP反复死亡重建，每次晋升证据清零，永远
        凑不够条件。继承只搬"晋升就绪度证据"（累计命中/位移/直径/骨架
        有效命中），TMP没有身份，不涉及任何ID转移；距离门+边际门保证
        就近且无争议，有争议宁可从零累计。"""
        if not self.tf_tmp_chain_enabled or not self._tmp_chain_pool:
            return
        track = self.tmp_tracks.get(tmp_id)
        start = self._tmp_origin_center.get(tmp_id)
        if track is None or start is None:
            return
        body_t = max(float(track.body_length_px), 8.0)
        scored: List[Tuple[float, int]] = []
        for seq, frag in self._tmp_chain_pool.items():
            gap = frame - int(frag["end_frame"])
            if gap < 1 or gap > self.tf_tmp_chain_ttl:
                continue
            body = max(body_t, float(frag["body_length"]), 8.0)
            d = point_distance(frag["end_center"], start)
            if np.isfinite(d):
                scored.append((d / body, seq))
        if not scored:
            return
        scored.sort(key=lambda x: x[0])
        best_d, best_seq = scored[0]
        second_d = scored[1][0] if len(scored) > 1 else float("inf")
        if best_d > self.tf_tmp_chain_dist or (second_d - best_d) < self.tf_tmp_chain_margin:
            return
        frag = self._tmp_chain_pool.pop(best_seq)
        self._tmp_total_hits[tmp_id] = int(frag["total_hits"])
        self._tmp_max_displacement[tmp_id] = float(frag["max_displacement"])
        self._tmp_diameter[tmp_id] = float(frag["diameter"])
        self._tmp_kp_ok_hits[tmp_id] = int(frag["kp_ok_hits"])
        logging.info(
            "TMP %d 继承死亡TMP的晋升证据（累计命中%d，v1.12.10证据链）",
            tmp_id - self.TMP_ID_BASE, int(frag["total_hits"]),
        )

    def _expire_tmp(self, frame: int) -> None:
        for tmp_id in list(self.tmp_tracks):
            track = self.tmp_tracks[tmp_id]
            mem = track.memory
            misses = mem.misses if mem is not None else (frame - track.last_frame)
            if misses > self.mem_tmp_ttl:
                self._drop_tmp(tmp_id, frame, "tmp_ttl_expired")
        for seq in list(self._tmp_chain_pool):
            if frame - int(self._tmp_chain_pool[seq]["end_frame"]) > self.tf_tmp_chain_ttl:
                self._tmp_chain_pool.pop(seq, None)

    def _register_fragment(self, track: IdentityTrack, frame: int) -> None:
        """轨迹段摘要入池（v1.12.5，论文MOT-SP：严格在线匹配宁可断轨，
        片段留待后续"最大值+差值"双条件融合）。保存端点位置/框/速度，
        供新TMP晋升时融合回原ID，避免旧ID回收后重领新ID的跳变。"""
        if not self.tf_enabled:
            return
        mem = track.memory
        if mem is not None and mem.velocity_history:
            velocity = np.asarray(mem.median_velocity(self.mem_velocity_window), dtype=np.float64)
        else:
            velocity = np.asarray(track.velocity_px_per_frame, dtype=np.float64)
        bbox = track.last_bbox_xyxy
        heading = None
        if track.heading_vector is not None:
            hv = np.asarray(track.heading_vector, dtype=np.float64)
            if np.all(np.isfinite(hv)):
                heading = hv.copy()
        self._fragment_pool[int(track.logical_id)] = {
            "end_frame": int(track.last_frame),
            "end_center": np.asarray(track.last_center_px, dtype=np.float64).copy(),
            "end_bbox": (np.asarray(bbox, dtype=np.float64).copy()
                         if bbox is not None else np.full(4, np.nan)),
            "velocity": velocity,
            "heading": heading,
            "body_length": float(track.body_length_px),
            "removed_frame": int(frame),
        }

    def _expire_fragments(self, frame: int) -> None:
        for lid in list(self._fragment_pool):
            if frame - int(self._fragment_pool[lid]["removed_frame"]) > self.tf_fragment_ttl:
                self._fragment_pool.pop(lid, None)

    def _fuse_tracklet(self, tmp_id: int, track: IdentityTrack, frame: int) -> Optional[Tuple[int, str]]:
        """论文§4.2.8/4.2.9轨迹段融合门控（MOT-SP距离主导 + MOT-TP预测延长）。

        对每个候选片段：端点经匀速预测延长（MOT-TP的轻量等价——论文用C-ANN
        延长≤6点，这里用已有速度向量延长≤predict_max帧）后与新片段起点
        （TMP创建点）求体长归一化距离；双条件缺一不可——
        最优距离≤DistThre 且 次优-最优≥DistDiffThre（防错配边际），
        或短间隔下IoU≥阈值且边际≥阈值。不满足就按新个体处理，绝不勉强融合。

        v1.12.8：候选从"已回收片段池"扩展到"活着的lost轨迹"（僵尸占槽吸收）——
        实跑残余跳变主因：接触中老轨迹变lost但仍占槽，同一只鼠被新TMP接管，
        TMP晋升只能领新号（6→7）。lost候选额外加两道守卫：
        1. misses≥absorb_min_misses（饿够时间，刚lost的不动）；
        2. 共存守卫：last_frame ≤ TMP创建帧+容差（它停止命中的时间不晚于
           TMP出现——一致于"TMP接管了它的鼠"；若之后还在命中=两只鼠共存，不吸收）。
        返回 (逻辑ID, "fragment"|"lost") 或 None。
        """
        if not self.tf_enabled:
            return None
        start = self._tmp_origin_center.get(tmp_id)
        if start is None or not np.all(np.isfinite(start)):
            start = np.asarray(track.last_center_px, dtype=np.float64)
        body_t = max(float(track.body_length_px), 8.0)
        t_vel = np.asarray(track.velocity_px_per_frame, dtype=np.float64)
        t_bbox = track.last_bbox_xyxy
        created = int(self._tmp_created_frame.get(tmp_id, frame))
        t_heading = None
        if track.heading_vector is not None:
            _hv = np.asarray(track.heading_vector, dtype=np.float64)
            if np.all(np.isfinite(_hv)):
                t_heading = _hv

        def _score(end_frame: int, end_center: np.ndarray, end_bbox: np.ndarray,
                   velocity: np.ndarray, body_f: float,
                   cand_heading: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
            gap = frame - end_frame
            if gap < 1 or gap > self.tf_max_gap_frames:
                return None
            body = max(body_f, body_t, 8.0)
            # v1.12.10：速度外推只跨"未观测区间"（候选端点→TMP创建点）。
            # 创建→晋升这段路TMP自己就是被观测轨迹，无需外推；旧版按
            # 晋升帧外推（gap可达50帧、顶满30帧上限），速度稍有噪声就
            # 过冲——实跑中lost端点距TMP创建点仅0.2体长的真匹配曾被
            # 误判>0.5体长而吸收失败（ID2→ID11跳变）。
            pred_len = min(max(1, created - end_frame), self.tf_predict_max)
            f_pred = end_center + self.velocity_weight * velocity * pred_len
            t_back = start - self.velocity_weight * t_vel * pred_len
            candidates = [
                point_distance(f_pred, start),
                point_distance(end_center, t_back),
                point_distance(end_center, start),
            ]
            finite = [d for d in candidates if np.isfinite(d)]
            if not finite:
                return None
            dist_bl = min(finite) / body
            # v1.13.0 朝向一致性（用户点名证据维度）：候选端点朝向↔TMP当前
            # 朝向的余弦距离∈[0,1]，无关键点证据取中性0.5。评分=
            # (1-w)×体长距离 + w×朝向差——位置说不清的朝向说清：实跑0.56体长
            # 无竞争的同鼠continuation曾被纯距离阈值拒绝，朝向佐证后放行。
            hdiff = 0.5
            if cand_heading is not None and t_heading is not None:
                hdiff = float((1.0 - float(np.clip(np.dot(cand_heading, t_heading), -1.0, 1.0))) / 2.0)
            score = (1.0 - self.tf_heading_weight) * dist_bl + self.tf_heading_weight * hdiff
            iou = 0.0
            if (gap <= self.tf_iou_max_gap and t_bbox is not None
                    and end_bbox is not None and np.all(np.isfinite(end_bbox))):
                iou = float(bbox_iou_xyxy(end_bbox, t_bbox))
            return (score, iou)

        scored: List[Tuple[float, float, int, str]] = []
        for lid, frag in self._fragment_pool.items():
            s = _score(int(frag["end_frame"]), frag["end_center"], frag["end_bbox"],
                       frag["velocity"], float(frag["body_length"]), frag.get("heading"))
            if s is not None:
                scored.append((s[0], s[1], lid, "fragment"))
        if self.tf_absorb_lost:
            for lid, lt in self.tracks.items():
                if lt.state != "lost":
                    continue
                mem = lt.memory
                misses = int(mem.misses) if mem is not None else (frame - int(lt.last_frame))
                if misses < self.tf_absorb_min_misses:
                    continue
                if int(lt.last_frame) > created + self.tf_absorb_coexist_tol:
                    # v1.12.11 位置豁免：它停止命中晚于TMP创建+t容差——若最后命中点
                    # 就落在TMP起点附近（≤override体长），说明那些"命中"本来就是
                    # 这只鼠漏过去的检测（TMP漏检间隙老轨迹偷了一两帧后又断），
                    # 而非另一只鼠共存——放行交给评分门控裁决；最后命中点离得远
                    # 才是真共存，维持拒绝。实跑案例：ID2在TMP漏检的1帧里在原位
                    # 命中一次，被守卫误判共存，TMP晋升领了新号ID11。
                    last_hit = np.asarray(lt.last_center_px, dtype=np.float64)
                    override_d = point_distance(last_hit, start)
                    body_o = max(float(lt.body_length_px), body_t, 8.0)
                    if (not np.isfinite(override_d)
                            or override_d / body_o > self.tf_absorb_coexist_override_bl):
                        continue
                if mem is not None and mem.velocity_history:
                    vel = np.asarray(mem.median_velocity(self.mem_velocity_window), dtype=np.float64)
                else:
                    vel = np.asarray(lt.velocity_px_per_frame, dtype=np.float64)
                lt_heading = None
                if lt.heading_vector is not None:
                    _hv = np.asarray(lt.heading_vector, dtype=np.float64)
                    if np.all(np.isfinite(_hv)):
                        lt_heading = _hv
                s = _score(int(lt.last_frame), np.asarray(lt.last_center_px, dtype=np.float64),
                           lt.last_bbox_xyxy, vel, float(lt.body_length_px), lt_heading)
                if s is not None:
                    scored.append((s[0], s[1], lid, "lost"))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        best_d, best_iou, best_lid, best_kind = scored[0]
        second_d = scored[1][0] if len(scored) > 1 else float("inf")
        second_iou = scored[1][1] if len(scored) > 1 else 0.0
        dist_ok = best_d <= self.tf_dist_thre_bl and (second_d - best_d) >= self.tf_dist_diff_bl
        iou_ok = best_iou >= self.tf_iou_thre and (best_iou - second_iou) >= self.tf_iou_diff
        if not (dist_ok or iou_ok):
            # v1.12.10：融合拒绝留痕——写明最优候选、距离与边际，
            # 实跑后可直接判读是"距离超阈"还是"边际不足"。
            margin = second_d - best_d
            margin_txt = "inf" if not np.isfinite(margin) else f"{margin:.2f}"
            cause = "距离超阈" if best_d > self.tf_dist_thre_bl else "边际不足"
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=tmp_id, raw_track_id=track.raw_track_id,
                assignment_cost=float(best_d), proposed_logical_id=best_lid,
                assignment_gain=float("nan"), dwell_count=0, dwell_required=0,
                cooldown_remaining=0, commit_status="fuse_rejected",
                switch_rejected_reason=(
                    f"best_lid={best_lid}({best_kind})_dist={best_d:.2f}BL"
                    f"_margin={margin_txt}BL_{cause}"
                ),
                appearance_mode="", detection_source="",
                occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                track_state="tentative",
            ))
            return None
        if best_kind == "fragment":
            self._fragment_pool.pop(best_lid, None)
        return (best_lid, best_kind)

    def _free_logical_ids(self, frame: int) -> List[int]:
        """§4.7：过期lost槽位及时清理，保证confirmed容量可恢复。"""
        self._expire_fragments(frame)
        if self.mem_recycle_lost_frames > 0:
            for lid in list(self.tracks):
                track = self.tracks[lid]
                if track.state == "lost" and frame - track.last_frame > self.mem_recycle_lost_frames:
                    self._register_fragment(track, frame)  # v1.12.5：片段入池待融合
                    if track.raw_track_id is not None and self.raw_to_logical.get(track.raw_track_id) == lid:
                        self.raw_to_logical.pop(track.raw_track_id, None)
                    self.tracks.pop(lid, None)
                    self.pending_commits.pop(lid, None)
        return [i for i in range(self.max_mice) if i not in self.tracks]

    def _promote_tmp_if_ready(self, tmp_id: int, frame: int) -> Optional[int]:
        track = self.tmp_tracks.get(tmp_id)
        if track is None or tmp_id in self._tmp_promotion_blocked:
            return None
        mem = self._ensure_memory(track)
        # v1.12.10 慢车道：累计命中≥N帧（默认90=3秒）且骨架解剖学有效占比
        # 达标的TMP，本帧在检即可跳过"连续命中"与"运动门控"——治静止抱团
        # 真鼠（位移/直径不达标）与闪烁通道（连续命中凑不齐）永远TMP。
        # 双保险：骨架证据（水痕解剖学必坏）+ 抖动级运动下限0.25体长
        # （水痕抖动≈0.1体长；万一骨架画得像样也升不上来）。
        slow_lane = False
        total_hits = int(self._tmp_total_hits.get(tmp_id, 0))
        if (self.mem_slow_lane_min_total_hits > 0
                and total_hits >= self.mem_slow_lane_min_total_hits
                and mem.misses == 0):
            kp_ok = int(self._tmp_kp_ok_hits.get(tmp_id, 0))
            body = max(float(track.body_length_px), 8.0)
            motion_bl = max(
                float(self._tmp_max_displacement.get(tmp_id, 0.0)),
                float(self._tmp_diameter.get(tmp_id, 0.0)),
            ) / body
            slow_lane = (
                kp_ok >= 3
                and kp_ok / max(total_hits, 1) >= self.mem_slow_lane_min_kp_ratio
                and motion_bl >= self.mem_slow_lane_min_motion_bl
            )
        if not slow_lane and (mem.hits < self.mem_confirm_frames or mem.misses > 0):
            return None
        # 运动门控（v1.11.3默认全源）：TMP必须观察到足够位移才晋升
        # （真鼠会动，反光假斑/水痕不动）。位移按"自TMP创建点起的最大位移"
        # 累计，不受记忆窗口30帧上限影响——慢速鼠也能在足够时间后达标。
        # v1.12.4并列判据"活动范围直径"：抱团聚拢鼠净位移虽小，但扭动会让
        # 访问点集直径持续增大；静止水痕两项都停在抖动量级（≈0.1倍体长）。
        if not slow_lane and tmp_id in self._tmp_motion_required:
            body = max(float(track.body_length_px), 8.0)
            disp_bl = float(self._tmp_max_displacement.get(tmp_id, 0.0)) / body
            diam_bl = float(self._tmp_diameter.get(tmp_id, 0.0)) / body
            if (disp_bl < self.mem_promotion_min_disp_bl
                    and diam_bl < self.mem_promotion_min_diameter_bl):
                return None
            self._tmp_motion_required.discard(tmp_id)
        free = self._free_logical_ids(frame)
        # v1.12.5：轨迹段融合优先于领取新ID——该TMP若能无争议地融合到
        # 某个已回收的旧片段（论文式双条件门控），直接继承原ID，防止跳变。
        fused = self._fuse_tracklet(tmp_id, track, frame)
        fused_id: Optional[int] = None
        if fused is not None:
            cand_id, cand_kind = fused
            if cand_kind == "lost":
                # v1.12.8 lost轨迹吸收：同一只鼠接回被僵尸lost占着的原ID。
                old = self.tracks.get(cand_id)
                det = self._tmp_last_det.get(tmp_id)
                if old is not None and old.state == "lost" and det is not None:
                    self._update_track(
                        cand_id, det, frame,
                        freeze_appearance=False, preserve_raw_id=False, store=self.tracks,
                    )
                    revived = self.tracks[cand_id]
                    revived.state = "tracked"
                    rmem = self._ensure_memory(revived)
                    rmem.misses = 0
                    rmem.hits = max(rmem.hits, self.mem_confirm_frames)
                    rmem.state = "tracked"
                    rmem.identity_confidence = max(rmem.identity_confidence, 0.5)
                    self.tmp_tracks.pop(tmp_id, None)
                    self._tmp_promotion_blocked.discard(tmp_id)
                    self._tmp_origin_center.pop(tmp_id, None)
                    self._tmp_max_displacement.pop(tmp_id, None)
                    self._tmp_far_point.pop(tmp_id, None)
                    self._tmp_diameter.pop(tmp_id, None)
                    self._tmp_total_hits.pop(tmp_id, None)
                    self._tmp_kp_ok_hits.pop(tmp_id, None)
                    self._tmp_last_det.pop(tmp_id, None)
                    self._tmp_created_frame.pop(tmp_id, None)
                    logging.info(
                        "lost轨迹吸收：TMP %d 在帧 %d 接回原ID %d（僵尸占槽修复，v1.12.8%s）",
                        tmp_id - self.TMP_ID_BASE, frame, cand_id,
                        "，慢车道" if slow_lane else "",
                    )
                    self.debug_records.append(IdentityDebug(
                        frame=frame, logical_id=cand_id, raw_track_id=det.raw_track_id,
                        assignment_cost=0.0, proposed_logical_id=cand_id, assignment_gain=float("nan"),
                        dwell_count=0, dwell_required=0, cooldown_remaining=0,
                        commit_status="lost_absorbed", switch_rejected_reason="",
                        appearance_mode="", detection_source="",
                        occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                        track_state="tracked",
                    ))
                    return cand_id
            else:  # fragment
                if cand_id not in self.tracks:
                    fused_id = cand_id
                # 极小概率ID刚被占用，放弃融合按新个体处理
        if fused_id is None and not free:
            return None
        new_id = fused_id if fused_id is not None else free[0]
        self.tmp_tracks.pop(tmp_id, None)
        self._tmp_promotion_blocked.discard(tmp_id)
        self._tmp_origin_center.pop(tmp_id, None)
        self._tmp_max_displacement.pop(tmp_id, None)
        self._tmp_far_point.pop(tmp_id, None)
        self._tmp_diameter.pop(tmp_id, None)
        self._tmp_total_hits.pop(tmp_id, None)
        self._tmp_kp_ok_hits.pop(tmp_id, None)
        self._tmp_last_det.pop(tmp_id, None)
        self._tmp_created_frame.pop(tmp_id, None)
        if slow_lane:
            logging.info(
                "慢车道晋升：TMP %d 在帧 %d 晋升为ID %d（累计命中%d帧、骨架有效占比达标，v1.12.10）",
                tmp_id - self.TMP_ID_BASE, frame, new_id, total_hits,
            )
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=new_id, raw_track_id=track.raw_track_id,
                assignment_cost=0.0, proposed_logical_id=new_id, assignment_gain=float("nan"),
                dwell_count=total_hits, dwell_required=self.mem_slow_lane_min_total_hits,
                cooldown_remaining=0, commit_status="tmp_promoted_slow_lane",
                switch_rejected_reason="",
                appearance_mode="", detection_source="",
                occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                track_state="tracked",
            ))
        track.logical_id = new_id
        track.state = "tracked"
        track.lock_strength = max(track.lock_strength, 0.35)
        mem.track_id = new_id
        mem.state = "tracked"
        mem.identity_confidence = max(mem.identity_confidence, 0.55)
        self.tracks[new_id] = track
        if track.raw_track_id is not None:
            if self.raw_to_logical.get(track.raw_track_id) == tmp_id:
                self.raw_to_logical.pop(track.raw_track_id, None)
            self.raw_to_logical[track.raw_track_id] = new_id
        if fused_id is not None:
            logging.info(
                "轨迹段融合：TMP %d 在帧 %d 融合回已回收的旧ID %d（论文式双条件门控）",
                tmp_id - self.TMP_ID_BASE, frame, new_id,
            )
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=new_id, raw_track_id=track.raw_track_id,
                assignment_cost=0.0, proposed_logical_id=new_id, assignment_gain=float("nan"),
                dwell_count=0, dwell_required=0, cooldown_remaining=0,
                commit_status="tracklet_fused", switch_rejected_reason="",
                appearance_mode="", detection_source="",
                occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                track_state="tracked",
            ))
        return new_id

    # ------------------------------------------------------------------
    # 标签与匹配接受
    # ------------------------------------------------------------------

    def _display_state(self, logical_id: int) -> str:
        if logical_id >= self.TMP_ID_BASE:
            return "tentative"
        track = self.tracks.get(logical_id)
        return track.state if track is not None else "tracked"

    def _display_label(self, logical_id: int) -> str:
        """§3.2渲染标签：TMP n / TMP ?（溢出）/ ID? n / ID n。"""
        if logical_id >= self.OVERFLOW_ID_BASE:
            return "TMP ?"
        if logical_id >= self.TMP_ID_BASE:
            return f"TMP {logical_id - self.TMP_ID_BASE}"
        state = self._display_state(logical_id)
        if state == "suspicious":
            return f"ID? {logical_id}"
        return f"ID {logical_id}"

    def _accept_memory_match(
        self,
        logical_id: int,
        det: Detection,
        frame: int,
        cost_value: float,
        margin: float,
        close: bool,
        force_suspicious: bool,
        method: str,
        note: str,
        cluster_id: int,
        cluster_expected: int,
        cluster_observed: int,
    ) -> None:
        is_tmp = logical_id in self.tmp_tracks
        store: MutableMapping[int, IdentityTrack] = self.tmp_tracks if is_tmp else self.tracks
        self._update_track(
            logical_id, det, frame,
            freeze_appearance=close or force_suspicious,
            preserve_raw_id=True,
            store=store,
        )
        track = store[logical_id]
        mem = self._ensure_memory(track)
        mem.hits += 1
        mem.misses = 0
        if is_tmp and logical_id in self._tmp_origin_center:
            origin = self._tmp_origin_center[logical_id]
            dist = point_distance(origin, det.center_px)
            if np.isfinite(dist):
                self._tmp_max_displacement[logical_id] = max(
                    self._tmp_max_displacement.get(logical_id, 0.0), float(dist)
                )
            # v1.12.4：活动范围直径估计——新点与创建点、最远点的两两距离取大。
            far = self._tmp_far_point.get(logical_id)
            d0 = dist if np.isfinite(dist) else 0.0
            d1 = point_distance(far, det.center_px) if far is not None else float("nan")
            d1 = float(d1) if np.isfinite(d1) else 0.0
            self._tmp_diameter[logical_id] = max(
                self._tmp_diameter.get(logical_id, 0.0), d0, d1
            )
            if far is None or d0 > point_distance(far, origin):
                self._tmp_far_point[logical_id] = np.asarray(det.center_px, dtype=np.float64).copy()
            # v1.12.6：累计命中（不随miss清零），簇内解禁的证据时长。
            self._tmp_total_hits[logical_id] = self._tmp_total_hits.get(logical_id, 0) + 1
            # v1.12.10：骨架解剖学有效命中计数（慢车道晋升证据；
            # 水痕/反光被Pose误画的骨架必然解剖学判坏，天然被挡）。
            if skeleton_anatomy_ok(det):
                self._tmp_kp_ok_hits[logical_id] = self._tmp_kp_ok_hits.get(logical_id, 0) + 1
            # v1.12.8：记录最新检测，lost吸收时复活老轨迹用。
            self._tmp_last_det[logical_id] = det
        if force_suspicious:
            # §6.7/§12.9：冲突期间只降低身份置信度，检测框必须保留。
            if not is_tmp:
                track.state = "suspicious"
                track.clean_streak = 0
                mem.state = "suspicious"
            mem.identity_confidence = min(mem.identity_confidence, 0.35)
        else:
            mem.identity_confidence = min(
                1.0, mem.identity_confidence + (0.15 if margin >= self.mem_id_margin else 0.05)
            )
            if self.sm_enabled and not is_tmp:
                self._sm_mark_accepted(track, margin)
            if not is_tmp:
                mem.state = track.state
        # §12.8 记忆更新安全规则：仅高置信+姿态完整+邻居远+边际明确+非suspicious时，
        # 才更新体型/外观等核心身份记忆；否则只更新运动状态。
        safe_identity = bool(
            not close
            and not force_suspicious
            and track.state != "suspicious"
            and margin >= self.mem_id_margin
            and float(det.box_conf) >= self.two_stage_high_conf
            and float(det.pose_quality) >= 0.5
            and float(det.max_overlap_iou) <= self.appearance_long_max_iou
        )
        self._memory_update(track, det, frame, allow_identity=safe_identity)
        self.debug_records.append(IdentityDebug(
            frame=frame, logical_id=logical_id, raw_track_id=det.raw_track_id,
            assignment_cost=cost_value, proposed_logical_id=logical_id, assignment_gain=margin,
            dwell_count=0, dwell_required=0, cooldown_remaining=0,
            commit_status=method, switch_rejected_reason=note,
            appearance_mode=det.appearance_mode, detection_source=det.detection_source,
            occlusion_cluster_id=cluster_id, cluster_expected_count=cluster_expected,
            cluster_observed_count=cluster_observed, track_state=track.state,
        ))
        self.output_info[logical_id] = {
            "state": self._display_state(logical_id),
            "label": self._display_label(logical_id),
            "cost": float(cost_value),
            "method": method,
        }

    def _finalize_stats(self, stats: Dict[str, int], outputs: List[Detection]) -> None:
        stats["active"] = len(self.tracks)
        stats["tentative"] = len(self.tmp_tracks)
        stats["suspicious"] = sum(1 for t in self.tracks.values() if t.state == "suspicious")
        stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
        stats["rendered"] = int(sum(
            1 for d in outputs
            if not bool(getattr(d, "synthetic_recovery", False))
            and str(getattr(d, "detection_source", "global")) != "predicted_hold"
        ))
        stats["unmatched_det"] = int(max(
            stats.get("raw", 0) - stats.get("matched", 0)
            - stats.get("low_rescued", 0) - stats.get("new_tentative", 0), 0
        ))
        self.frame_stats = stats

    # ------------------------------------------------------------------
    # 主关联流程（§5逐帧流程）
    # ------------------------------------------------------------------

    def assign(
        self,
        detections: Sequence[Detection],
        frame: int,
        occlusion_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[int, Detection]]:
        self._expire(frame)
        self._expire_tmp(frame)
        self._free_logical_ids(frame)
        context = dict(occlusion_context or {})
        frozen_ids = {int(v) for v in context.get("frozen_ids", set())}
        id_to_cluster = {int(k): int(v) for k, v in dict(context.get("id_to_cluster", {})).items()}
        regions_by_id = {
            int(region.get("cluster_id", -1)): region
            for region in context.get("regions", [])
        }

        detections = list(detections)[: self.max_mice + self.candidate_extra + self.mem_max_tentative]
        self.output_info = {}
        self._overflow_ids = set()
        stats: Dict[str, int] = {
            "raw": len(detections), "after_conf": len(detections), "after_kpt_filter": len(detections),
            "matched": 0, "low_rescued": 0, "lost_recovered": 0, "new_tentative": 0,
            "unmatched_det": 0, "active": 0, "tentative": 0, "suspicious": 0,
            "lost": 0, "rendered": 0,
        }

        # §5第1/2步：空帧只维护记忆与状态；lost不得伪装为当前检测（§3.2）。
        if not detections:
            for lid, track in self._all_track_items():
                track.lock_strength = max(0.0, track.lock_strength - self.missing_lock_decay)
                self._memory_mark_miss(track)
                if self.sm_enabled and lid in self.tracks:
                    self._memory_transition_on_miss(track)
                self.debug_records.append(IdentityDebug(
                    frame=frame, logical_id=lid, raw_track_id=None,
                    assignment_cost=float("nan"), proposed_logical_id=lid, assignment_gain=float("nan"),
                    dwell_count=0, dwell_required=0, cooldown_remaining=0,
                    commit_status="memory_miss_hold", switch_rejected_reason="no_detection_keep_memory",
                    appearance_mode="", detection_source="predicted_hold",
                    occlusion_cluster_id=int(id_to_cluster.get(lid, -1)),
                    cluster_expected_count=0, cluster_observed_count=0, track_state=track.state,
                ))
            self._finalize_stats(stats, [])
            return []

        # §5第3/4步：检测按置信度分两级；身份匹配分两阶段。
        if self.two_stage_enabled:
            high_cols = [
                i for i, d in enumerate(detections)
                if float(d.box_conf) >= self.two_stage_high_conf
                or float(d.pose_quality) >= self.two_stage_high_pose_quality
            ]
            high_set = set(high_cols)
            low_cols = [i for i in range(len(detections)) if i not in high_set]
        else:
            high_cols, low_cols = list(range(len(detections))), []

        all_items = self._all_track_items()
        live_items = [(lid, t) for lid, t in all_items if t.state != "lost"]
        lost_items = [(lid, t) for lid, t in all_items if t.state == "lost" and lid not in self.tmp_tracks]
        track_map = dict(all_items)
        track_ids = [lid for lid, _ in live_items]
        stage1 = [detections[i] for i in high_cols]
        # v1.11.3：接触簇内的检测姿态证据不可靠，代价计算时降姿态权重。
        stage1_crowded = self._detection_crowding(stage1)

        cost = np.full((len(track_ids), len(stage1)), self.INF_COST, dtype=np.float64)
        for r, lid in enumerate(track_ids):
            for c, det in enumerate(stage1):
                cost[r, c] = self._memory_cost(track_map[lid], det, frame, crowded=stage1_crowded[c])

        assignments: List[Tuple[int, int, str]] = []
        used_rows: set = set()
        used_cols: set = set()
        if self.slot_mutual_lock and cost.size:
            row_best = np.argmin(cost, axis=1)
            col_best = np.argmin(cost, axis=0)
            for r, c0 in enumerate(row_best.tolist()):
                c = int(c0)
                if int(col_best[c]) != r:
                    continue
                if not np.isfinite(cost[r, c]) or cost[r, c] >= self.INF_COST or cost[r, c] > self.slot_mutual_lock_cost:
                    continue
                assignments.append((r, c, "mutual"))
                used_rows.add(r)
                used_cols.add(c)
        rem_rows = [r for r in range(cost.shape[0]) if r not in used_rows]
        rem_cols = [c for c in range(cost.shape[1]) if c not in used_cols]
        if rem_rows and rem_cols:
            sub = cost[np.ix_(rem_rows, rem_cols)]
            if linear_sum_assignment is not None:
                rr, cc = linear_sum_assignment(sub)
                for sr, sc in zip(rr.tolist(), cc.tolist()):
                    assignments.append((rem_rows[sr], rem_cols[sc], "hungarian"))
            else:
                for sr, sc in self._greedy_assignment(sub):
                    assignments.append((rem_rows[sr], rem_cols[sc], "greedy"))

        output: Dict[int, Detection] = {}
        updated: set = set()
        assigned_det: set = set()
        assigned_rows: set = set()
        rejected_rows: set = set()
        held_rows: set = set()
        frame_updates: Dict[int, Detection] = {}

        for r, c, method in assignments:
            lid = track_ids[r]
            track = track_map[lid]
            det = stage1[c]
            det_global_idx = high_cols[c]
            chosen = float(cost[r, c])
            assigned_rows.add(r)
            is_tmp = lid in self.tmp_tracks
            cluster_id = int(id_to_cluster.get(lid, -1))
            cluster = regions_by_id.get(cluster_id, {})
            cluster_expected = int(cluster.get("expected_count", 0))
            cluster_observed = int(cluster.get("observed_count", 0))
            margin = self._row_col_margin(cost, r, c, self.INF_COST)
            nearest_other = self._memory_nearest_other_bl(track)
            close = nearest_other <= self.close_contact_body_lengths or lid in frozen_ids

            if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.mem_max_cost:
                rejected_rows.add(r)
                self.debug_records.append(IdentityDebug(
                    frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                    assignment_cost=chosen, proposed_logical_id=lid, assignment_gain=margin,
                    dwell_count=0, dwell_required=0, cooldown_remaining=0,
                    commit_status="memory_rejected", switch_rejected_reason="cost_or_distance_gate",
                    appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                    occlusion_cluster_id=cluster_id, cluster_expected_count=cluster_expected,
                    cluster_observed_count=cluster_observed, track_state=track.state,
                ))
                continue

            # §6.7/§12.9：匹配边际不足且贴近邻居 → 身份冲突。
            # 冻结身份但不删检测框；有竞争者时用OKS+长期外观仲裁。
            # v1.11.3：接触簇内边际阈值加倍——宁可冻结标ID?，不做低置信互换。
            margin_thresh = self.mem_id_margin
            if close or stage1_crowded[c]:
                margin_thresh = self.mem_id_margin * self.mem_contact_margin_scale
            ambiguous = margin < margin_thresh
            if self.simple_motion and ambiguous and self.simple_ambiguous_reject:
                # v1.13.0 简洁模式：边际不足不强制匹配——轨迹转lost，
                # 检测去当TMP，由轨迹段融合（位置+朝向+速度）事后裁决。
                # 不冻结、不仲裁、不标可疑：断轨可由融合自愈，错配会污染。
                rejected_rows.add(r)
                self.debug_records.append(IdentityDebug(
                    frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                    assignment_cost=chosen, proposed_logical_id=lid, assignment_gain=margin,
                    dwell_count=0, dwell_required=0, cooldown_remaining=0,
                    commit_status="memory_ambiguous_rejected",
                    switch_rejected_reason=f"margin={margin:.3f}<{margin_thresh:.3f}_simple_motion",
                    appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                    occlusion_cluster_id=cluster_id, cluster_expected_count=cluster_expected,
                    cluster_observed_count=cluster_observed, track_state=track.state,
                ))
                continue
            if ambiguous and close:
                resolved = False
                if self.hp_enabled:
                    competitor = self._memory_find_competitor(cost, r, c, track_ids, track_map, det, frame)
                    if competitor is not None:
                        comp_track = track_map[competitor]
                        arb_self = self._arbitration_cost(track, det)
                        arb_comp = self._arbitration_cost(comp_track, det)
                        if arb_comp + self.hp_min_arbitration_margin < arb_self:
                            held_rows.add(r)
                            if self.sm_enabled and not is_tmp:
                                self._sm_mark_suspicious(track)
                                if track.memory is not None:
                                    track.memory.state = "suspicious"
                            self.debug_records.append(IdentityDebug(
                                frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                                assignment_cost=chosen, proposed_logical_id=lid, assignment_gain=margin,
                                dwell_count=0, dwell_required=0, cooldown_remaining=0,
                                commit_status="hard_pair_arbitration_hold",
                                switch_rejected_reason=f"competitor={competitor};arb={arb_self:.3f}vs{arb_comp:.3f}",
                                appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                                occlusion_cluster_id=cluster_id, cluster_expected_count=cluster_expected,
                                cluster_observed_count=cluster_observed, track_state=track.state,
                            ))
                            continue
                        resolved = arb_self + self.hp_min_arbitration_margin < arb_comp
                if resolved:
                    self._accept_memory_match(
                        lid, det, frame, chosen, margin, close, False,
                        f"memory_{method}_arbitrated", "arbitration_resolved_by_oks_long_appearance",
                        cluster_id, cluster_expected, cluster_observed,
                    )
                else:
                    self._accept_memory_match(
                        lid, det, frame, chosen, margin, close, True,
                        f"memory_{method}_suspicious",
                        f"id_margin<{self.mem_id_margin:.3f}_freeze_identity_keep_detection",
                        cluster_id, cluster_expected, cluster_observed,
                    )
                output[lid] = det
                updated.add(lid)
                assigned_det.add(det_global_idx)
                if not is_tmp:
                    frame_updates[lid] = det
                stats["matched"] += 1
                continue

            self._accept_memory_match(
                lid, det, frame, chosen, margin, close, False,
                f"memory_{method}_accepted", "memory_cost",
                cluster_id, cluster_expected, cluster_observed,
            )
            output[lid] = det
            updated.add(lid)
            assigned_det.add(det_global_idx)
            if not is_tmp:
                frame_updates[lid] = det
            stats["matched"] += 1

        # §5第4步：低分检测第二阶段回收（ByteTrack思想：纯运动+IoU+尺度）。
        if self.two_stage_enabled and low_cols:
            rescue_rows = [
                r for r in range(len(track_ids))
                if track_ids[r] not in updated
                and (r in rejected_rows or r not in assigned_rows)
                and r not in held_rows
                and track_map[track_ids[r]].state != "lost"
            ]
            if rescue_rows:
                low_dets = [detections[i] for i in low_cols]
                low_cost = np.full((len(rescue_rows), len(low_dets)), self.INF_COST, dtype=np.float64)
                for rr_i, r in enumerate(rescue_rows):
                    for cc_i, det in enumerate(low_dets):
                        low_cost[rr_i, cc_i] = self._memory_low_cost(track_map[track_ids[r]], det, frame)
                if linear_sum_assignment is not None:
                    lr, lc = linear_sum_assignment(low_cost)
                    rescue_matches = list(zip(lr.tolist(), lc.tolist()))
                else:
                    rescue_matches = self._greedy_assignment(low_cost)
                for rr_i, cc_i in rescue_matches:
                    r = rescue_rows[rr_i]
                    lid = track_ids[r]
                    det = low_dets[cc_i]
                    chosen = float(low_cost[rr_i, cc_i])
                    if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.mem_low_max_cost:
                        continue
                    cluster_id = int(id_to_cluster.get(lid, -1))
                    cluster = regions_by_id.get(cluster_id, {})
                    is_tmp = lid in self.tmp_tracks
                    close = self._memory_nearest_other_bl(track_map[lid]) <= self.close_contact_body_lengths or lid in frozen_ids
                    # 低分回收只更新位置与运动；外观/体型身份记忆全部冻结。
                    self._accept_memory_match(
                        lid, det, frame, chosen, float("nan"), close, False,
                        "memory_low_rescue_accepted", "bytetrack_style_motion_only_freeze_identity",
                        cluster_id, int(cluster.get("expected_count", 0)), int(cluster.get("observed_count", 0)),
                    )
                    output[lid] = det
                    updated.add(lid)
                    assigned_det.add(low_cols[cc_i])
                    if not is_tmp:
                        frame_updates[lid] = det
                    stats["low_rescued"] += 1

        # §5第5步：仍未匹配的检测优先尝试恢复 Lost ID（运动可达+外观/OKS证据）。
        if self.sm_enabled and lost_items:
            remaining = [i for i in range(len(detections)) if i not in assigned_det]
            if remaining:
                rec_cost = np.full((len(lost_items), len(remaining)), self.INF_COST, dtype=np.float64)
                for rr_i, (lid, track) in enumerate(lost_items):
                    for cc_i, det_idx in enumerate(remaining):
                        rec_cost[rr_i, cc_i] = self._memory_cost(track, detections[det_idx], frame)
                if linear_sum_assignment is not None:
                    lr, lc = linear_sum_assignment(rec_cost)
                    rec_matches = list(zip(lr.tolist(), lc.tolist()))
                else:
                    rec_matches = self._greedy_assignment(rec_cost)
                for rr_i, cc_i in rec_matches:
                    lid, track = lost_items[rr_i]
                    det_idx = remaining[cc_i]
                    det = detections[det_idx]
                    chosen = float(rec_cost[rr_i, cc_i])
                    if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.mem_max_cost:
                        continue
                    ok, reason = self._lost_reactivation_ok(track, det, chosen)
                    if not ok:
                        self.debug_records.append(IdentityDebug(
                            frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                            assignment_cost=chosen, proposed_logical_id=lid, assignment_gain=float("nan"),
                            dwell_count=0, dwell_required=0, cooldown_remaining=0,
                            commit_status="memory_lost_reactivation_rejected", switch_rejected_reason=reason,
                            appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                            occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                            track_state=track.state,
                        ))
                        continue
                    self._accept_memory_match(
                        lid, det, frame, chosen, float("nan"), False, False,
                        "memory_lost_reactivated", reason, -1, 0, 0,
                    )
                    output[lid] = det
                    updated.add(lid)
                    assigned_det.add(det_idx)
                    frame_updates[lid] = det
                    stats["lost_recovered"] += 1

        # §12.8：TMP连续命中confirm_frames帧且不在接触簇禁区 → 晋升为固定槽位。
        for tmp_id in list(self.tmp_tracks):
            if tmp_id not in updated:
                continue
            if tmp_id in self._tmp_promotion_blocked:
                center = self.tmp_tracks[tmp_id].last_center_px
                still_inside = any(
                    _point_inside_bbox(center, box)
                    for box in context.get("forbidden_new_regions", [])
                )
                if not still_inside:
                    self._tmp_promotion_blocked.discard(tmp_id)
                elif (self.mem_contact_promote_total_hits > 0
                        and self._tmp_total_hits.get(tmp_id, 0) >= self.mem_contact_promote_total_hits):
                    # v1.12.6：簇内TMP累计稳定命中≥N帧（默认60=2秒）解除禁升。
                    # §12.8簇内禁升防ID给错鼠，但长期抱团的两只鼠永远等不到
                    # "离开簇"解禁——累计2秒稳定命中足以证明追踪真实个体。
                    self._tmp_promotion_blocked.discard(tmp_id)
                    logging.info(
                        "接触簇内TMP %d 累计命中≥%d帧，解除禁升（v1.12.6）",
                        tmp_id - self.TMP_ID_BASE, self.mem_contact_promote_total_hits,
                    )
            new_id = self._promote_tmp_if_ready(tmp_id, frame)
            if new_id is None:
                continue
            det = output.pop(tmp_id, None)
            if det is not None:
                output[new_id] = det
                frame_updates[new_id] = det
            prev_info = self.output_info.pop(tmp_id, {})
            self.output_info[new_id] = {
                "state": self._display_state(new_id),
                "label": self._display_label(new_id),
                "cost": float(prev_info.get("cost", 0.0)),
                "method": "tmp_promoted",
            }
            updated.discard(tmp_id)
            updated.add(new_id)
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=new_id, raw_track_id=self.tracks[new_id].raw_track_id,
                assignment_cost=0.0, proposed_logical_id=new_id, assignment_gain=1.0,
                dwell_count=self.mem_confirm_frames, dwell_required=self.mem_confirm_frames,
                cooldown_remaining=0, commit_status="tmp_promoted",
                switch_rejected_reason=f"TMP{tmp_id - self.TMP_ID_BASE}_confirmed_to_slot_{new_id}",
                appearance_mode=det.appearance_mode if det is not None else "",
                detection_source=det.detection_source if det is not None else "",
                occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                track_state=self.tracks[new_id].state,
            ))

        # 未匹配轨迹：miss计数、锁定衰减、lost转换；confirmed轨迹内部预测保持
        # （预测框只用于内部连续性，主程序不渲染——lost不得伪装成当前检测）。
        for lid, track in self._all_track_items():
            if lid in updated:
                continue
            track.lock_strength = max(0.0, track.lock_strength - self.missing_lock_decay)
            self._memory_mark_miss(track)
            if lid in self.tracks:
                if self.sm_enabled:
                    self._memory_transition_on_miss(track)
                if lid not in output:
                    pred = self._make_slot_prediction(track, frame)
                    if pred is not None:
                        output[lid] = pred

        # §5第6步：仍未匹配的检测立即创建Tentative临时ID并渲染（§6.4、§9）。
        for det_idx in range(len(detections)):
            if det_idx in assigned_det:
                continue
            det = detections[det_idx]
            forbid = OcclusionClusterManager.detection_inside_forbidden(det, context)
            tmp_id = self._create_tmp_track(det, frame, forbid)
            output[tmp_id] = det
            assigned_det.add(det_idx)
            stats["new_tentative"] += 1
            overflow = tmp_id >= self.OVERFLOW_ID_BASE
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=tmp_id, raw_track_id=det.raw_track_id,
                assignment_cost=0.0, proposed_logical_id=tmp_id, assignment_gain=float("nan"),
                dwell_count=1, dwell_required=self.mem_confirm_frames, cooldown_remaining=0,
                commit_status=(
                    "tmp_overflow_display_only" if overflow
                    else "tmp_created_in_cluster" if forbid else "tmp_created"
                ),
                switch_rejected_reason="unmatched_detection_gets_temporary_id",
                appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                occlusion_cluster_id=-1, cluster_expected_count=0, cluster_observed_count=0,
                track_state="tentative",
            ))
            self.output_info[tmp_id] = {
                "state": "tentative",
                "label": self._display_label(tmp_id),
                "cost": 0.0,
                "method": "tmp_created",
            }

        # §12.6：滑窗重拼接——双向外观证据持续成立时回滚此前的ID互换。
        if self.restitch_enabled and frame_updates:
            self._restitch_check(frame, frame_updates, frozen_ids)

        self._finalize_stats(stats, list(output.values()))
        return sorted(output.items(), key=lambda item: item[0])


class KeypointMotionIdentityAssigner:
    """面向多鼠的逐关键点运动身份分配器。

    该实现把旧双鼠 ``PureKeypointTracker`` 的核心机制扩展到最多 ``max_mice`` 只：
    - 每个关键点独立维护速度与加速度；
    - 用预测关键点与当前检测关键点的置信度加权中位距离作为主代价；
    - 匈牙利算法执行全局一对一匹配；
    - 门限随轨迹速度和漏检时长自适应放宽；
    - 已建立轨迹不因候选边际接近而主动断成TMP，避免ID碎裂；
    - 当前帧没有检测时只保留内部预测，不渲染假框/假骨架。

    这是一种在线身份连续性模块，不负责补回检测器完全漏掉的实例。
    """

    INF_COST = 1e6

    def __init__(self, config: Mapping[str, Any], max_mice: int = 20) -> None:
        cfg = dict(config.get("keypoint_motion", {}))
        self.max_mice = int(max_mice)
        self.max_missing_frames = int(cfg.get("max_missing_frames", 90))
        self.min_keypoint_conf = float(cfg.get("min_keypoint_confidence", 0.10))
        self.min_common_keypoints = int(cfg.get("min_common_keypoints", 2))
        self.prediction_max_frames = int(cfg.get("prediction_max_frames", 10))
        self.prediction_acceleration_max_frames = max(
            int(cfg.get("prediction_acceleration_max_frames", 3)), 0
        )
        self.prediction_max_speed_bl_per_frame = max(
            float(cfg.get("prediction_max_speed_body_lengths_per_frame", 0.35)),
            0.01,
        )
        self.prediction_max_displacement_bl = max(
            float(cfg.get("prediction_max_displacement_body_lengths", 1.25)),
            self.prediction_max_speed_bl_per_frame,
        )
        self.velocity_alpha = float(cfg.get("velocity_alpha", 0.60))
        self.acceleration_alpha = float(cfg.get("acceleration_alpha", 0.35))
        self.center_velocity_alpha = float(cfg.get("center_velocity_alpha", 0.65))
        self.body_length_alpha = float(cfg.get("body_length_alpha", 0.15))

        self.base_gate_body_lengths = float(cfg.get("base_gate_body_lengths", 1.15))
        self.speed_gate_scale = float(cfg.get("speed_gate_scale", 4.0))
        self.missing_gate_growth = float(cfg.get("missing_gate_growth", 0.20))
        self.max_gate_body_lengths = float(cfg.get("max_gate_body_lengths", 3.20))
        self.max_assignment_cost = float(cfg.get("max_assignment_cost", 1.25))

        # v1.29: 轨迹桥接只负责把持续出现的临时轨迹接回旧正式 ID，
        # 不改变普通匈牙利匹配的门限，避免为修复漏检而放宽所有帧的匹配。
        bridge_cfg = dict(cfg.get("tracklet_bridge", {}))
        self.bridge_enabled = bool(bridge_cfg.get("enabled", True))
        self.bridge_min_provisional_hits = max(int(bridge_cfg.get("min_provisional_hits", 3)), 1)
        self.bridge_min_missing_frames = max(int(bridge_cfg.get("min_missing_frames", 2)), 1)
        self.bridge_max_missing_frames = max(
            int(bridge_cfg.get("max_missing_frames", self.max_missing_frames)),
            self.bridge_min_missing_frames,
        )
        self.bridge_max_score = float(bridge_cfg.get("max_bridge_score", 1.15))
        self.bridge_max_pose_cost = float(bridge_cfg.get("max_pose_cost", 1.20))
        self.bridge_max_body_ratio = max(float(bridge_cfg.get("max_body_ratio", 2.0)), 1.0)
        self.bridge_min_box_conf = float(bridge_cfg.get("min_box_confidence", 0.18))
        self.bridge_min_pose_quality = float(bridge_cfg.get("min_pose_quality", 0.42))
        self.bridge_min_motion_bl = max(float(bridge_cfg.get("min_motion_body_lengths", 0.10)), 0.0)
        self.bridge_max_center_bl = max(float(bridge_cfg.get("max_center_body_lengths", 5.0)), 1.0)
        self.bridge_min_margin = max(float(bridge_cfg.get("min_bridge_margin", 0.05)), 0.0)
        self.render_max_missing_frames = max(int(cfg.get("render_max_missing_frames", 10)), 0)
        self.render_confidence_decay = float(cfg.get("render_confidence_decay", 0.88))
        self.render_block_gate_bl = max(float(cfg.get("render_block_gate_body_lengths", 0.75)), 0.1)

        weights = dict(cfg.get("weights", {}))
        self.w_keypoint = float(weights.get("keypoint", 0.56))
        self.w_center = float(weights.get("center", 0.24))
        self.w_iou = float(weights.get("iou", 0.08))
        self.w_heading = float(weights.get("heading", 0.06))
        self.w_size = float(weights.get("size", 0.06))

        self.initial_order = str(cfg.get("initial_order", "top_to_bottom_left_to_right")).lower()
        self.new_track_min_separation_bl = float(cfg.get("new_track_min_separation_body_lengths", 0.35))
        self.weak_track_drop_hits = int(cfg.get("weak_track_drop_hits", 2))
        self.weak_track_drop_missing = int(cfg.get("weak_track_drop_missing_frames", 12))
        self.reuse_expired_ids = bool(cfg.get("reuse_expired_ids", False))

        self.tracks: Dict[int, IdentityTrack] = {}
        self.kpt_velocity: Dict[int, np.ndarray] = {}
        self.kpt_acceleration: Dict[int, np.ndarray] = {}
        self.kpt_missing: Dict[int, int] = defaultdict(int)
        self.next_logical_id = 0
        self.free_ids: List[int] = []
        self.debug_records: List[IdentityDebug] = []
        self.output_info: Dict[int, Dict[str, Any]] = {}
        self.frame_stats: Dict[str, int] = {}

    @property
    def cluster_tracks(self) -> Dict[int, IdentityTrack]:
        return self.tracks

    @staticmethod
    def _valid_points(points: np.ndarray, conf: np.ndarray, threshold: float) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        c = np.asarray(conf, dtype=np.float64).reshape(-1)
        n = min(len(p), len(c), len(KEYPOINT_NAMES))
        valid = np.zeros(len(KEYPOINT_NAMES), dtype=bool)
        if n <= 0:
            return valid
        valid[:n] = (
            np.isfinite(p[:n, 0]) & np.isfinite(p[:n, 1])
            & (p[:n, 0] > 0) & (p[:n, 1] > 0)
            & np.isfinite(c[:n]) & (c[:n] >= threshold)
        )
        return valid

    def _allocate_id(self) -> Optional[int]:
        if self.reuse_expired_ids and self.free_ids:
            return int(self.free_ids.pop(0))
        used = set(self.tracks)
        while self.next_logical_id in used:
            self.next_logical_id += 1
        if len(self.tracks) >= self.max_mice:
            return None
        lid = int(self.next_logical_id)
        self.next_logical_id += 1
        return lid

    def _initial_sort_key(self, det: Detection) -> Tuple[float, float]:
        center = np.asarray(det.center_px, dtype=np.float64)
        if self.initial_order in {"left_to_right_top_to_bottom", "x_then_y"}:
            return float(center[0]), float(center[1])
        return float(center[1]), float(center[0])

    def _expire(self, frame: int) -> None:
        stale: List[int] = []
        frozen_now = set(int(x) for x in getattr(self, "_current_frozen_ids", set()))
        for lid, track in self.tracks.items():
            # 聚集身份池中的ID必须保留到延迟ReID完成，不能按普通弱轨迹清理。
            if int(lid) in frozen_now:
                continue
            missing = max(int(frame - track.last_frame), int(self.kpt_missing.get(lid, 0)))
            weak = track.hits <= self.weak_track_drop_hits and missing > self.weak_track_drop_missing
            if missing > self.max_missing_frames or weak:
                stale.append(lid)
        for lid in stale:
            self.tracks.pop(lid, None)
            self.kpt_velocity.pop(lid, None)
            self.kpt_acceleration.pop(lid, None)
            self.kpt_missing.pop(lid, None)
            if self.reuse_expired_ids:
                self.free_ids.append(int(lid))
                self.free_ids.sort()

    @staticmethod
    def _prediction_body_scale(track: IdentityTrack) -> float:
        body = float(getattr(track, "body_length_px", 0.0))
        box = getattr(track, "last_bbox_xyxy", None)
        if box is not None:
            values = np.asarray(box, dtype=np.float64).reshape(-1)
            if values.size >= 4 and np.all(np.isfinite(values[:4])):
                body = max(
                    body,
                    0.75
                    * max(
                        abs(float(values[2] - values[0])),
                        abs(float(values[3] - values[1])),
                    ),
                )
        return max(body if np.isfinite(body) else 0.0, 8.0)

    def _bounded_prediction_delta(
        self,
        track: IdentityTrack,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Cap missing-frame extrapolation so an internal hold cannot fly away.

        Acceleration estimates are useful across a very short occlusion, but a
        noisy pose derivative grows quadratically when it is extrapolated for
        tens of frames.  The cap is expressed in body lengths and is applied to
        each keypoint (or to the fallback centre vector), preserving direction
        without allowing a stale prediction to cross the cage.
        """
        values = np.asarray(delta, dtype=np.float64).copy()
        if values.size == 0:
            return values
        body = self._prediction_body_scale(track)
        maximum = self.prediction_max_displacement_bl * body
        if values.ndim == 1:
            norm = float(np.linalg.norm(values))
            if np.isfinite(norm) and norm > maximum:
                values *= maximum / max(norm, 1.0e-9)
            return values
        norms = np.linalg.norm(values, axis=-1)
        scale = np.ones_like(norms, dtype=np.float64)
        over = np.isfinite(norms) & (norms > maximum)
        scale[over] = maximum / np.maximum(norms[over], 1.0e-9)
        values *= scale[..., None]
        return values

    def _predicted_keypoints(self, lid: int, track: IdentityTrack, frame: int) -> np.ndarray:
        last = track.last_keypoints_px
        if last is None:
            return np.full((len(KEYPOINT_NAMES), 2), np.nan, dtype=np.float64)
        points = np.asarray(last, dtype=np.float64).copy()
        velocity = self.kpt_velocity.get(lid)
        acceleration = self.kpt_acceleration.get(lid)
        if velocity is None:
            velocity = np.zeros_like(points)
        if acceleration is None:
            acceleration = np.zeros_like(points)
        dt = min(max(int(frame - track.last_frame), 0), self.prediction_max_frames)
        valid = np.all(np.isfinite(points), axis=1)
        safe_velocity = np.where(np.isfinite(velocity), velocity, 0.0)
        safe_acceleration = np.where(np.isfinite(acceleration), acceleration, 0.0)
        body = self._prediction_body_scale(track)
        speed_limit = self.prediction_max_speed_bl_per_frame * body
        velocity_norm = np.linalg.norm(safe_velocity, axis=-1)
        velocity_scale = np.ones_like(velocity_norm, dtype=np.float64)
        over_speed = np.isfinite(velocity_norm) & (velocity_norm > speed_limit)
        velocity_scale[over_speed] = speed_limit / np.maximum(
            velocity_norm[over_speed], 1.0e-9
        )
        safe_velocity *= velocity_scale[..., None]
        acceleration_dt = min(dt, self.prediction_acceleration_max_frames)
        displacement = (
            safe_velocity * float(dt)
            + 0.5 * safe_acceleration * float(acceleration_dt * acceleration_dt)
        )
        displacement = self._bounded_prediction_delta(track, displacement)
        points[valid] = points[valid] + displacement[valid]
        return points

    def _prediction(self, track: IdentityTrack, frame: int) -> np.ndarray:
        lid = int(track.logical_id)
        predicted = self._predicted_keypoints(lid, track, frame)
        conf = (
            np.asarray(track.last_keypoint_conf, dtype=np.float64)
            if track.last_keypoint_conf is not None
            else np.zeros(len(KEYPOINT_NAMES), dtype=np.float64)
        )
        valid = self._valid_points(predicted, conf, self.min_keypoint_conf)
        core = np.zeros(len(KEYPOINT_NAMES), dtype=bool)
        for idx in (KP["neck"], KP["left_hind"], KP["right_hind"], KP["tail"]):
            if idx < len(core):
                core[idx] = True
        use = valid & core
        if np.sum(use) >= 2:
            return np.median(predicted[use], axis=0).astype(np.float64)
        if np.sum(valid) >= 2:
            return np.median(predicted[valid], axis=0).astype(np.float64)
        dt = min(max(int(frame - track.last_frame), 0), self.prediction_max_frames)
        displacement = self._bounded_prediction_delta(
            track,
            np.asarray(track.velocity_px_per_frame, dtype=np.float64) * float(dt),
        )
        return np.asarray(track.last_center_px, dtype=np.float64) + displacement

    @staticmethod
    def _bridge_pose_cost(track: IdentityTrack, det: Detection) -> float:
        """Return a bounded normalized-pose distance for long-gap reattachment."""
        old = track.normalized_pose
        new = det.normalized_pose
        if old is None or new is None:
            return float("nan")
        old_arr = np.asarray(old, dtype=np.float64)
        new_arr = np.asarray(new, dtype=np.float64)
        if old_arr.shape != new_arr.shape or old_arr.size == 0:
            return float("nan")
        valid = np.all(np.isfinite(old_arr), axis=-1) & np.all(np.isfinite(new_arr), axis=-1)
        if int(np.sum(valid)) < 3:
            return float("nan")
        distance = np.linalg.norm(old_arr[valid] - new_arr[valid], axis=-1)
        return float(np.clip(np.median(distance) / 0.35, 0.0, 2.0))

    def bridge_provisional_track(
        self,
        det: Detection,
        frame: int,
        occupied_ids: Optional[Iterable[int]] = None,
        provisional_hits: int = 0,
        provisional_motion_bl: float = float("inf"),
    ) -> Optional[int]:
        """Reconnect a persistent Pxx detection to the best still-held formal ID.

        The bridge is deliberately narrower than the normal assignment path.  It
        requires a multi-frame, high-quality candidate and a stale-but-retained
        track.  The caller supplies IDs already assigned in this frame so a
        currently visible mouse can never be stolen by the bridge.
        """
        if not self.bridge_enabled or int(provisional_hits) < self.bridge_min_provisional_hits:
            return None
        if float(det.box_conf) < self.bridge_min_box_conf:
            return None
        conf = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
        pose_quality = float(
            np.mean(np.isfinite(conf) & (conf >= self.min_keypoint_conf))
        ) if conf.size else 0.0
        if pose_quality < self.bridge_min_pose_quality:
            return None
        if np.isfinite(provisional_motion_bl) and provisional_motion_bl < self.bridge_min_motion_bl:
            return None

        occupied = {int(value) for value in (occupied_ids or ())}
        candidates: List[Tuple[float, int]] = []
        det_body = max(float(det.body_length_px), 8.0)
        for lid, track in self.tracks.items():
            lid = int(lid)
            if lid in occupied:
                continue
            missing = max(int(frame) - int(track.last_frame), int(self.kpt_missing.get(lid, 0)))
            if missing < self.bridge_min_missing_frames or missing > self.bridge_max_missing_frames:
                continue
            if int(track.hits) <= self.weak_track_drop_hits:
                continue
            track_body = max(float(track.body_length_px), 8.0)
            body_ratio = max(det_body / track_body, track_body / det_body)
            if body_ratio > self.bridge_max_body_ratio:
                continue
            center_distance = point_distance(np.asarray(track.last_center_px), np.asarray(det.center_px))
            center_bl = center_distance / max(det_body, track_body)
            if not np.isfinite(center_bl) or center_bl > self.bridge_max_center_bl:
                continue
            pose_cost = self._bridge_pose_cost(track, det)
            if np.isfinite(pose_cost) and pose_cost > self.bridge_max_pose_cost:
                continue
            pose_term = pose_cost if np.isfinite(pose_cost) else 0.65
            size_term = min(abs(math.log(max(body_ratio, 1.0))), 1.5)
            center_term = min(center_bl / max(self.bridge_max_center_bl, 1.0), 1.5)
            score = 0.48 * center_term + 0.34 * pose_term + 0.18 * size_term
            candidates.append((float(score), lid))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_score, lid = candidates[0]
        if best_score > self.bridge_max_score:
            return None
        if len(candidates) > 1 and candidates[1][0] - best_score < self.bridge_min_margin:
            return None

        track = self.tracks.get(int(lid))
        if track is None:
            return None
        missing_before_update = max(
            int(frame) - int(track.last_frame),
            int(self.kpt_missing.get(int(lid), 0)),
        )
        self._update_track(int(lid), det, int(frame))
        self.output_info[int(lid)] = {
            "state": "tracked",
            "label": f"ID {int(lid)}",
            "cost": float(best_score),
            "method": "keypoint_tracklet_bridge",
        }
        self.debug_records.append(IdentityDebug(
            frame=int(frame),
            logical_id=int(lid),
            raw_track_id=det.raw_track_id,
            assignment_cost=float(best_score),
            proposed_logical_id=int(lid),
            assignment_gain=float("nan"),
            commit_status="keypoint_tracklet_bridge",
            switch_rejected_reason=(
                f"persistent_provisional_hits={int(provisional_hits)};missing={int(missing_before_update)}"
            ),
            appearance_mode=det.appearance_mode,
            detection_source=det.detection_source,
            track_state="tracked",
        ))
        self.frame_stats["tracklet_bridged"] = int(self.frame_stats.get("tracklet_bridged", 0)) + 1
        return int(lid)

    def render_predictions(
        self,
        frame: int,
        blocked_detections: Sequence[Detection] = (),
    ) -> List[Tuple[int, Detection, Dict[str, Any]]]:
        """Create short, render-only holds for formal IDs during detector gaps."""
        if self.render_max_missing_frames <= 0:
            return []
        blocked = list(blocked_detections)
        output: List[Tuple[int, Detection, Dict[str, Any]]] = []
        for lid, track in sorted(self.tracks.items()):
            lid = int(lid)
            missing = int(frame) - int(track.last_frame)
            if missing <= 0 or missing > self.render_max_missing_frames:
                continue
            if str(track.state) == "lost":
                continue
            predicted_points = self._predicted_keypoints(lid, track, int(frame))
            predicted_box = self._predicted_bbox(track, int(frame))
            if predicted_box is None:
                continue
            finite_points = np.all(np.isfinite(predicted_points), axis=1)
            if int(np.sum(finite_points)) < 2:
                continue
            pred_body = max(float(track.body_length_px), 8.0)
            too_close = False
            for det in blocked:
                distance_bl = point_distance(np.asarray(predicted_box[:2]) + (predicted_box[2:4] - predicted_box[:2]) / 2.0, det.center_px) / max(pred_body, float(det.body_length_px), 8.0)
                if np.isfinite(distance_bl) and distance_bl <= self.render_block_gate_bl:
                    too_close = True
                    break
            if too_close:
                continue
            last_conf = np.asarray(
                track.last_keypoint_conf if track.last_keypoint_conf is not None else np.ones(len(KEYPOINT_NAMES)),
                dtype=np.float64,
            )
            decay = float(np.clip(self.render_confidence_decay, 0.50, 0.99)) ** missing
            predicted_conf = np.where(finite_points, np.clip(last_conf * decay, 0.05, 0.65), 0.0)
            det = Detection(
                raw_track_id=track.raw_track_id,
                keypoints_px=predicted_points.astype(np.float64, copy=True),
                keypoint_conf=predicted_conf.astype(np.float64, copy=True),
                bbox_xyxy=np.asarray(predicted_box, dtype=np.float64).copy(),
                box_conf=float(np.clip(track.last_box_conf * decay, 0.05, 0.65)),
                normalized_pose=track.normalized_pose.copy() if track.normalized_pose is not None else None,
                anchor_feature=track.anchor_feature.copy() if track.anchor_feature is not None else None,
                heading_vector=track.heading_vector.copy() if track.heading_vector is not None else None,
                appearance_mode="predicted_hold",
                detection_source="predicted_hold",
                prefer_bbox_center=True,
                keypoint_sources=np.full(len(KEYPOINT_NAMES), "PREDICTED", dtype=object),
            )
            output.append((lid, det, {
                "state": "predicted_hold",
                "label": f"ID {lid}",
                "cost": float("nan"),
                "method": "keypoint_render_hold",
            }))
        return output

    def _predicted_bbox(self, track: IdentityTrack, frame: int) -> Optional[np.ndarray]:
        if track.last_bbox_xyxy is None:
            return None
        pred_center = self._prediction(track, frame)
        last_center = np.asarray(track.last_center_px, dtype=np.float64)
        shift = pred_center - last_center
        box = np.asarray(track.last_bbox_xyxy, dtype=np.float64).copy()
        if box.size < 4 or not np.all(np.isfinite(box[:4])):
            return None
        box[[0, 2]] += shift[0]
        box[[1, 3]] += shift[1]
        return box

    def _adaptive_gate_bl(self, track: IdentityTrack, frame: int) -> float:
        body = max(float(track.body_length_px), 8.0) if np.isfinite(track.body_length_px) else 8.0
        speed_bl = float(np.linalg.norm(track.velocity_px_per_frame)) / body
        missing = max(int(frame - track.last_frame), 0)
        gate = (
            self.base_gate_body_lengths
            + self.speed_gate_scale * speed_bl
            + self.missing_gate_growth * min(missing, self.prediction_max_frames)
        )
        return float(np.clip(gate, self.base_gate_body_lengths, self.max_gate_body_lengths))

    def _keypoint_cost(self, lid: int, track: IdentityTrack, det: Detection, body: float, gate: float, frame: int) -> Tuple[float, int]:
        pred = self._predicted_keypoints(lid, track, frame)
        old_conf = (
            np.asarray(track.last_keypoint_conf, dtype=np.float64)
            if track.last_keypoint_conf is not None
            else np.zeros(len(KEYPOINT_NAMES), dtype=np.float64)
        )
        new_points = np.asarray(det.keypoints_px, dtype=np.float64)
        new_conf = np.asarray(det.keypoint_conf, dtype=np.float64)
        valid = (
            self._valid_points(pred, old_conf, self.min_keypoint_conf)
            & self._valid_points(new_points, new_conf, self.min_keypoint_conf)
        )
        count = int(np.sum(valid))
        if count < self.min_common_keypoints:
            return 0.65, count
        distances = np.linalg.norm(pred[valid] - new_points[valid], axis=1) / max(body, 1e-6)
        joint_conf = np.sqrt(np.clip(old_conf[valid], 0.01, 1.0) * np.clip(new_conf[valid], 0.01, 1.0))
        # 继承旧双鼠实现：中位距离为主，低置信度会放大代价；再加入加权中位近似。
        order = np.argsort(distances)
        d_sorted = distances[order]
        w_sorted = joint_conf[order]
        cutoff = 0.5 * float(np.sum(w_sorted))
        weighted_median = float(d_sorted[np.searchsorted(np.cumsum(w_sorted), cutoff, side="left")])
        confidence_factor = 1.50 - float(np.mean(joint_conf))
        raw = weighted_median * max(confidence_factor, 0.50)
        return float(np.clip(raw / max(gate, 1e-6), 0.0, 2.0)), count

    def _cost(self, lid: int, track: IdentityTrack, det: Detection, frame: int) -> float:
        lengths = [track.body_length_px, det.body_length_px]
        body = max(float(np.nanmedian([v for v in lengths if np.isfinite(v) and v > 3])) if any(np.isfinite(v) and v > 3 for v in lengths) else 20.0, 8.0)
        gate = self._adaptive_gate_bl(track, frame)
        pred_center = self._prediction(track, frame)
        center_dist = point_distance(pred_center, det.center_px)
        if not np.isfinite(center_dist):
            return self.INF_COST
        center_bl = center_dist / body
        if center_bl > gate:
            return self.INF_COST

        keypoint_cost, common = self._keypoint_cost(lid, track, det, body, gate, frame)
        center_cost = float(np.clip(center_bl / max(gate, 1e-6), 0.0, 2.0))

        pred_box = self._predicted_bbox(track, frame)
        iou_cost = 1.0 - bbox_iou_xyxy(pred_box, det.bbox_xyxy) if pred_box is not None else 0.55

        old_heading = track.heading_vector
        new_heading = det.heading_vector
        if old_heading is not None and new_heading is not None:
            heading_cost = float(np.clip((1.0 - cosine_similarity(old_heading, new_heading)) / 2.0, 0.0, 1.0))
        else:
            heading_cost = 0.45

        if np.isfinite(track.body_length_px) and track.body_length_px > 3 and np.isfinite(det.body_length_px) and det.body_length_px > 3:
            size_cost = float(np.clip(abs(math.log(det.body_length_px / track.body_length_px)), 0.0, 1.5))
        else:
            size_cost = 0.35

        # 关键点不足时自动把权重转移给中心和IoU，避免bbox-only检测被无意义姿态项拒绝。
        wk = self.w_keypoint if common >= self.min_common_keypoints else 0.0
        wc = self.w_center + (self.w_keypoint * 0.75 if wk == 0.0 else 0.0)
        wi = self.w_iou + (self.w_keypoint * 0.25 if wk == 0.0 else 0.0)
        terms = [
            (wk, keypoint_cost),
            (wc, center_cost),
            (wi, iou_cost),
            (self.w_heading, heading_cost),
            (self.w_size, size_cost),
        ]
        denom = sum(w for w, _ in terms if w > 0)
        if denom <= 1e-9:
            return self.INF_COST
        return float(sum(w * c for w, c in terms if w > 0) / denom)

    def _new_track(self, det: Detection, frame: int, logical_id: Optional[int] = None) -> Optional[int]:
        lid = self._allocate_id() if logical_id is None else int(logical_id)
        if lid is None:
            return None
        points = np.asarray(det.keypoints_px, dtype=np.float64).copy()
        conf = np.asarray(det.keypoint_conf, dtype=np.float64).copy()
        box = np.asarray(det.bbox_xyxy, dtype=np.float64).copy()
        wh = np.array([max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)], dtype=np.float64)
        track = IdentityTrack(
            logical_id=lid,
            last_center_px=np.asarray(det.center_px, dtype=np.float64).copy(),
            velocity_px_per_frame=np.zeros(2, dtype=np.float64),
            last_frame=int(frame),
            raw_track_id=det.raw_track_id,
            body_length_px=max(float(det.body_length_px), 8.0),
            normalized_pose=det.normalized_pose.copy() if det.normalized_pose is not None else None,
            anchor_feature=det.anchor_feature.copy() if det.anchor_feature is not None else None,
            heading_vector=det.heading_vector.copy() if det.heading_vector is not None else None,
            bbox_wh=wh,
            last_keypoints_px=points,
            last_keypoint_conf=conf,
            last_bbox_xyxy=box,
            last_box_conf=float(det.box_conf),
            hits=1,
            lock_strength=0.25,
            state="tracked",
        )
        self.tracks[lid] = track
        self.kpt_velocity[lid] = np.zeros_like(points, dtype=np.float64)
        self.kpt_acceleration[lid] = np.zeros_like(points, dtype=np.float64)
        self.kpt_missing[lid] = 0
        return lid

    def _update_track(self, lid: int, det: Detection, frame: int) -> None:
        track = self.tracks[lid]
        old_points = np.asarray(track.last_keypoints_px, dtype=np.float64) if track.last_keypoints_px is not None else np.full((len(KEYPOINT_NAMES), 2), np.nan)
        old_conf = np.asarray(track.last_keypoint_conf, dtype=np.float64) if track.last_keypoint_conf is not None else np.zeros(len(KEYPOINT_NAMES))
        new_points = np.asarray(det.keypoints_px, dtype=np.float64).copy()
        new_conf = np.asarray(det.keypoint_conf, dtype=np.float64).copy()
        dt = max(int(frame - track.last_frame), 1)

        velocity = self.kpt_velocity.get(lid, np.zeros_like(new_points, dtype=np.float64))
        acceleration = self.kpt_acceleration.get(lid, np.zeros_like(new_points, dtype=np.float64))
        valid = (
            self._valid_points(old_points, old_conf, self.min_keypoint_conf)
            & self._valid_points(new_points, new_conf, self.min_keypoint_conf)
        )
        if np.any(valid):
            measured_velocity = (new_points[valid] - old_points[valid]) / float(dt)
            old_velocity = velocity[valid].copy()
            velocity[valid] = (
                (1.0 - self.velocity_alpha) * old_velocity
                + self.velocity_alpha * measured_velocity
            )
            measured_acc = (measured_velocity - old_velocity) / float(dt)
            acceleration[valid] = (
                (1.0 - self.acceleration_alpha) * acceleration[valid]
                + self.acceleration_alpha * measured_acc
            )

        old_center = np.asarray(track.last_center_px, dtype=np.float64)
        new_center = np.asarray(det.center_px, dtype=np.float64)
        measured_center_velocity = (new_center - old_center) / float(dt)
        track.velocity_px_per_frame = (
            (1.0 - self.center_velocity_alpha) * np.asarray(track.velocity_px_per_frame, dtype=np.float64)
            + self.center_velocity_alpha * measured_center_velocity
        )
        track.last_center_px = new_center.copy()
        track.last_frame = int(frame)
        track.raw_track_id = det.raw_track_id
        if np.isfinite(det.body_length_px) and det.body_length_px > 3:
            track.body_length_px = (
                (1.0 - self.body_length_alpha) * max(float(track.body_length_px), 8.0)
                + self.body_length_alpha * float(det.body_length_px)
            )
        box = np.asarray(det.bbox_xyxy, dtype=np.float64).copy()
        track.bbox_wh = np.array([max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)])
        track.last_bbox_xyxy = box
        track.last_box_conf = float(det.box_conf)
        track.last_keypoints_px = new_points
        track.last_keypoint_conf = new_conf
        track.normalized_pose = det.normalized_pose.copy() if det.normalized_pose is not None else track.normalized_pose
        track.anchor_feature = det.anchor_feature.copy() if det.anchor_feature is not None else track.anchor_feature
        track.heading_vector = det.heading_vector.copy() if det.heading_vector is not None else track.heading_vector
        track.hits += 1
        track.lock_strength = min(1.0, track.lock_strength + 0.04)
        track.state = "tracked"
        track.clean_streak += 1
        self.kpt_velocity[lid] = velocity
        self.kpt_acceleration[lid] = acceleration
        self.kpt_missing[lid] = 0

    @staticmethod
    def _greedy_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        flat = np.argsort(cost, axis=None)
        rows, cols = np.unravel_index(flat, cost.shape)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r in used_r or c in used_c:
                continue
            if not np.isfinite(cost[r, c]) or cost[r, c] >= KeypointMotionIdentityAssigner.INF_COST:
                continue
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
        return pairs

    def assign(
        self,
        detections: Sequence[Detection],
        frame: int,
        occlusion_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[int, Detection]]:
        self._expire(frame)
        detections = list(detections)
        self.output_info = {}
        self.debug_records = []
        stats = {
            "raw": len(detections), "after_conf": len(detections), "after_kpt_filter": len(detections),
            "matched": 0, "low_rescued": 0, "lost_recovered": 0, "new_tentative": 0,
            "unmatched_det": 0, "active": len(self.tracks), "tentative": 0,
            "suspicious": 0, "lost": 0, "rendered": len(detections),
        }

        if not detections:
            for lid, track in self.tracks.items():
                self.kpt_missing[lid] = int(self.kpt_missing.get(lid, 0)) + 1
                track.lock_strength = max(0.0, track.lock_strength - 0.01)
                if self.kpt_missing[lid] > self.max_missing_frames:
                    track.state = "lost"
            stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
            self.frame_stats = stats
            return []

        # 第一批检测直接得到固定ID，按空间顺序初始化，避免TMP→ID造成视觉上的改号。
        if not self.tracks:
            output: List[Tuple[int, Detection]] = []
            for det in sorted(detections, key=self._initial_sort_key)[: self.max_mice]:
                lid = self._new_track(det, frame)
                if lid is None:
                    continue
                output.append((lid, det))
                self.output_info[lid] = {"state": "tracked", "label": f"ID {lid}", "cost": 0.0, "method": "keypoint_initial"}
                self.debug_records.append(IdentityDebug(
                    frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                    assignment_cost=0.0, proposed_logical_id=lid, assignment_gain=1.0,
                    commit_status="keypoint_initial", switch_rejected_reason="",
                    appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                    track_state="tracked",
                ))
            stats["matched"] = len(output)
            stats["active"] = len(self.tracks)
            stats["rendered"] = len(output)
            self.frame_stats = stats
            return sorted(output, key=lambda item: item[0])

        track_ids = sorted(self.tracks)
        cost = np.full((len(track_ids), len(detections)), self.INF_COST, dtype=np.float64)
        for r, lid in enumerate(track_ids):
            track = self.tracks[lid]
            for c, det in enumerate(detections):
                cost[r, c] = self._cost(lid, track, det, frame)

        if cost.size and linear_sum_assignment is not None:
            rr, cc = linear_sum_assignment(cost)
            proposed = list(zip(rr.tolist(), cc.tolist()))
        elif cost.size:
            proposed = self._greedy_assignment(cost)
        else:
            proposed = []

        output: Dict[int, Detection] = {}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        blocked_ambiguous_detections: set[int] = set()
        for r, c in proposed:
            lid = track_ids[r]
            chosen = float(cost[r, c])
            if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.max_assignment_cost:
                continue
            det = detections[c]
            was_missing = int(self.kpt_missing.get(lid, 0))
            self._update_track(lid, det, frame)
            output[lid] = det
            matched_tracks.add(lid)
            matched_detections.add(c)
            method = "keypoint_hungarian_recovered" if was_missing > 0 else "keypoint_hungarian"
            self.output_info[lid] = {"state": "tracked", "label": f"ID {lid}", "cost": chosen, "method": method}
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                assignment_cost=chosen, proposed_logical_id=lid, assignment_gain=float("nan"),
                commit_status=method, switch_rejected_reason="global_assignment_no_margin_break",
                appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                track_state="tracked",
            ))
            stats["matched"] += 1
            if was_missing > 0:
                stats["lost_recovered"] += 1

        # v1.28：密集中心旁边仍完整可见的鼠不应因为普通门限过严连续显示Pxx。
        # 仅对ClusterReID明确标出的“中心外围清洁检测”做一次旧轨迹重接；
        # 不创建新ID，不接触中心C检测，也不处理水渍/影子等低质量候选。
        partial_edge_indices = {
            int(index)
            for index in context.get("partial_dense_visible_detection_indices", set())
            if int(index) not in self._current_reserved_detection_indices
        }
        if self.partial_edge_recovery_enabled and partial_edge_indices:
            edge_detections = [
                index
                for index in sorted(partial_edge_indices)
                if index not in matched_detections
                and 0 <= index < len(detections)
                and float(detections[index].box_conf) >= self.partial_edge_recovery_min_conf
                and not str(getattr(detections[index], "pose_recovery_reason", "") or "")
                and bool(getattr(detections[index], "mask_reliable", False))
            ]
            edge_tracks = [
                lid
                for lid in track_ids
                if lid not in matched_tracks
                and lid not in self._current_frozen_ids
                and int(self.kpt_missing.get(lid, 0)) <= self.partial_edge_recovery_max_missing
            ]
            if edge_tracks and edge_detections:
                edge_cost = np.full(
                    (len(edge_tracks), len(edge_detections)),
                    self.INF_COST,
                    dtype=np.float64,
                )
                for row, lid in enumerate(edge_tracks):
                    track = self.tracks[lid]
                    prediction = self._prediction(track, frame)
                    body = max(float(track.body_length_px), 8.0)
                    for column, detection_index in enumerate(edge_detections):
                        det = detections[detection_index]
                        distance_bl = point_distance(prediction, det.center_px) / body
                        overlap = (
                            bbox_iou_xyxy(track.last_bbox_xyxy, det.bbox_xyxy)
                            if track.last_bbox_xyxy is not None else 0.0
                        )
                        if distance_bl <= self.partial_edge_recovery_gate_bl or overlap >= 0.04:
                            mask_distance = self._track_mask_distance(track, det)
                            edge_cost[row, column] = (
                                distance_bl
                                + 0.15 * (1.0 - overlap)
                                + 0.12 * (
                                    mask_distance if mask_distance is not None else 0.50
                                )
                            )
                if linear_sum_assignment is not None:
                    edge_rows, edge_columns = linear_sum_assignment(edge_cost)
                    edge_pairs = zip(edge_rows.tolist(), edge_columns.tolist())
                else:
                    edge_pairs = self._greedy_assignment(edge_cost)
                for row, column in edge_pairs:
                    chosen = float(edge_cost[row, column])
                    if not np.isfinite(chosen) or chosen >= self.INF_COST:
                        continue
                    lid = edge_tracks[row]
                    detection_index = edge_detections[column]
                    det = detections[detection_index]
                    was_missing = int(self.kpt_missing.get(lid, 0))
                    self._update_track(lid, det, frame)
                    output[lid] = det
                    matched_tracks.add(lid)
                    matched_detections.add(detection_index)
                    self.output_info[lid] = {
                        "state": "tracked",
                        "label": f"ID {lid}",
                        "cost": chosen,
                        "method": "partial_dense_edge_recovery",
                    }
                    stats["matched"] += 1
                    if was_missing > 0:
                        stats["lost_recovered"] += 1

        for lid in track_ids:
            if lid in matched_tracks:
                continue
            self.kpt_missing[lid] = int(self.kpt_missing.get(lid, 0)) + 1
            track = self.tracks[lid]
            track.lock_strength = max(0.0, track.lock_strength - 0.01)
            if self.kpt_missing[lid] > self.max_missing_frames:
                track.state = "lost"

        # 新出现的检测只在与所有当前轨迹明显分离时创建固定ID，防止重复框制造新号。
        for c, det in enumerate(detections):
            if c in matched_detections or len(self.tracks) >= self.max_mice:
                continue
            det_body = max(float(det.body_length_px), 8.0)
            nearest = float("inf")
            for track in self.tracks.values():
                pred = self._prediction(track, frame)
                dist = point_distance(pred, det.center_px)
                body = max(det_body, float(track.body_length_px) if np.isfinite(track.body_length_px) else det_body)
                if np.isfinite(dist):
                    nearest = min(nearest, dist / max(body, 1e-6))
            if nearest < self.new_track_min_separation_bl:
                stats["unmatched_det"] += 1
                continue
            lid = self._new_track(det, frame)
            if lid is None:
                stats["unmatched_det"] += 1
                continue
            output[lid] = det
            self.output_info[lid] = {"state": "tracked", "label": f"ID {lid}", "cost": 0.0, "method": "keypoint_new_track"}
            self.debug_records.append(IdentityDebug(
                frame=frame, logical_id=lid, raw_track_id=det.raw_track_id,
                assignment_cost=0.0, proposed_logical_id=lid, assignment_gain=float("nan"),
                commit_status="keypoint_new_track", switch_rejected_reason="separated_unmatched_detection",
                appearance_mode=det.appearance_mode, detection_source=det.detection_source,
                track_state="tracked",
            ))
            stats["new_tentative"] += 1

        stats["active"] = len(self.tracks)
        stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
        stats["rendered"] = len(output)
        self.frame_stats = stats
        return sorted(output.items(), key=lambda item: item[0])

# ----------------------------- 轨迹历史与几何特征 -----------------------------


class ObservationHistory:
    """按逻辑ID保存有限长度的观测历史。

    高召回主程序依赖以下接口：previous/add/get/near_frame。
    历史使用deque限长保存，不会随视频时长无限增长。
    """

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max(int(max_frames), 1)
        self.data: Dict[int, Deque[MouseObservation]] = defaultdict(
            lambda: deque(maxlen=self.max_frames)
        )
        # A frame evaluates every logical ID against many partners. Cache only
        # immutable read views and near-frame lookups; adding an observation
        # invalidates caches for that ID before pair calculation begins.
        self._window_cache: Dict[int, Dict[int, Tuple[MouseObservation, ...]]] = {}
        self._near_cache: Dict[int, Dict[int, Optional[MouseObservation]]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        # Read caches are derivable and may duplicate observations in checkpoints.
        state = dict(self.__dict__)
        state["_window_cache"] = {}
        state["_near_cache"] = {}
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._window_cache = {}
        self._near_cache = {}

    def previous(self, logical_id: int) -> Optional[MouseObservation]:
        history = self.data.get(int(logical_id))
        return history[-1] if history else None

    def add(self, observation: MouseObservation) -> None:
        logical_id = int(observation.logical_id)
        self.data[logical_id].append(observation)
        self._window_cache.pop(logical_id, None)
        self._near_cache.pop(logical_id, None)

    def get(self, logical_id: int) -> List[MouseObservation]:
        # Preserve the historical API contract: callers receive a mutable copy.
        return list(self.data.get(int(logical_id), []))

    def get_window(
        self, logical_id: int, max_items: int
    ) -> Tuple[MouseObservation, ...]:
        """Return a cached read-only tail window for pair-feature hot loops."""
        logical_id = int(logical_id)
        max_items = max(int(max_items), 0)
        by_size = self._window_cache.setdefault(logical_id, {})
        cached = by_size.get(max_items)
        if cached is not None:
            return cached
        history = self.data.get(logical_id)
        if not history or max_items == 0:
            result: Tuple[MouseObservation, ...] = ()
        else:
            values = tuple(history)
            result = values[-max_items:] if len(values) > max_items else values
        by_size[max_items] = result
        return result

    def near_frame(self, logical_id: int, frame: int) -> Optional[MouseObservation]:
        logical_id = int(logical_id)
        target_frame = int(frame)
        cache = self._near_cache.setdefault(logical_id, {})
        if target_frame in cache:
            return cache[target_frame]
        history = self.data.get(logical_id)
        result: Optional[MouseObservation] = None
        if history:
            # deque长度很小（通常约1秒），反向寻找最近且不晚于目标帧的观测。
            for obs in reversed(history):
                if obs.frame <= target_frame:
                    result = obs
                    break
        cache[target_frame] = result
        return result


def derive_geometry(
    kpts_cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """由7个关键点派生身体中心、头部、后躯、朝向和鼻尾体长。"""
    points = np.asarray(kpts_cm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < len(KEYPOINT_NAMES):
        nan2 = np.array([np.nan, np.nan], dtype=np.float64)
        return nan2.copy(), nan2.copy(), nan2.copy(), nan2.copy(), float("nan")

    nose = points[KP["nose"]]
    left_ear = points[KP["left_ear"]]
    right_ear = points[KP["right_ear"]]
    neck = points[KP["neck"]]
    left_hind = points[KP["left_hind"]]
    right_hind = points[KP["right_hind"]]
    tail = points[KP["tail"]]

    head = nanmean_points([nose, left_ear, right_ear])
    rear = nanmean_points([left_hind, right_hind, tail])
    # 行为轨迹中心优先使用颈基和双髋，避免鼻尖/尾基飞点拉动中心。
    center = nanmean_points([neck, left_hind, right_hind])
    if not finite_point(center):
        center = nanmean_points(list(points))

    if finite_point(nose) and finite_point(neck):
        heading = unit_vector(nose - neck)
    elif finite_point(neck) and finite_point(tail):
        heading = unit_vector(neck - tail)
    else:
        heading = np.array([np.nan, np.nan], dtype=np.float64)

    body_length = point_distance(nose, tail)
    return center, head, rear, heading, body_length


class PairContactTracker:
    def __init__(self, fps: float, window_seconds: float) -> None:
        self.window_frames = max(int(round(fps * window_seconds)), 1)
        self.previous_contact: Dict[Tuple[int, int], bool] = defaultdict(bool)
        self.onsets: Dict[Tuple[int, int], Deque[int]] = defaultdict(deque)

    def update(self, id_a: int, id_b: int, frame: int, contact: bool) -> int:
        key = tuple(sorted((id_a, id_b)))
        prev = self.previous_contact[key]
        if contact and not prev:
            self.onsets[key].append(frame)
        self.previous_contact[key] = contact

        while self.onsets[key] and frame - self.onsets[key][0] > self.window_frames:
            self.onsets[key].popleft()
        return len(self.onsets[key])


class PairFeatureComputer:
    def __init__(self, fps: float, config: Mapping[str, Any]) -> None:
        self.fps = float(fps)
        self.feature_cfg = config["features"]
        self.chase_cfg = config["chase"]
        self.attack_cfg = config["attack"]
        history_seconds = float(self.feature_cfg.get("history_seconds", 1.0))
        self.history_frames = max(int(round(self.fps * history_seconds)), 3)
        self.lookback_frames = max(
            int(round(self.fps * float(self.feature_cfg.get("response_lookback_seconds", 0.3)))),
            1,
        )

        self._cache_frame = -1
        self._history_map_cache: Dict[int, Dict[int, MouseObservation]] = {}
        self._trajectory_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._target_turn_cache: Dict[int, float] = {}
        self._distance_drop_cache: Dict[Tuple[int, int], float] = {}

    def _ensure_frame_cache(self, frame: int) -> None:
        frame = int(frame)
        if frame == self._cache_frame:
            return
        self._cache_frame = frame
        self._history_map_cache.clear()
        self._trajectory_cache.clear()
        self._target_turn_cache.clear()
        self._distance_drop_cache.clear()

    def _history_map(
        self, logical_id: int, history: ObservationHistory
    ) -> Dict[int, MouseObservation]:
        logical_id = int(logical_id)
        cached = self._history_map_cache.get(logical_id)
        if cached is not None:
            return cached
        values = history.get_window(logical_id, self.history_frames)
        cached = {observation.frame: observation for observation in values}
        self._history_map_cache[logical_id] = cached
        return cached

    def _trajectory_features(
        self,
        actor_id: int,
        target_id: int,
        history: ObservationHistory,
        frame: int,
    ) -> Tuple[float, float, float]:
        self._ensure_frame_cache(frame)
        key = (int(actor_id), int(target_id))
        cached = self._trajectory_cache.get(key)
        if cached is not None:
            return cached
        actor_by_frame = self._history_map(actor_id, history)
        target_by_frame = self._history_map(target_id, history)
        common_frames = sorted(set(actor_by_frame) & set(target_by_frame))
        actor_centers = np.empty((0, 2), dtype=np.float64)
        target_centers = np.empty((0, 2), dtype=np.float64)
        da = np.empty((0, 2), dtype=np.float64)
        dt = np.empty((0, 2), dtype=np.float64)
        if len(common_frames) < 4:
            result = (0.0, 0.0, 0.0)
        else:
            actor_centers = np.stack([actor_by_frame[f].center_cm for f in common_frames])
            target_centers = np.stack([target_by_frame[f].center_cm for f in common_frames])
            valid = np.all(np.isfinite(actor_centers), axis=1) & np.all(
                np.isfinite(target_centers), axis=1
            )
            actor_centers = actor_centers[valid]
            target_centers = target_centers[valid]
            if len(actor_centers) < 4:
                result = (0.0, 0.0, 0.0)
            else:
                da = np.diff(actor_centers, axis=0)
                dt = np.diff(target_centers, axis=0)
                corr_x = safe_corr(da[:, 0], dt[:, 0])
                corr_y = safe_corr(da[:, 1], dt[:, 1])
                corr = float(np.clip((corr_x + corr_y) / 2.0, -1.0, 1.0))
                result = (
                    corr,
                    float(np.sum(np.linalg.norm(da, axis=1))),
                    float(np.sum(np.linalg.norm(dt, axis=1))),
                )
        self._trajectory_cache[key] = result
        if key[0] != key[1]:
            # Recompute correlation in the historical reversed argument order.
            # np.corrcoef can differ by one ULP when inputs are swapped.
            if len(common_frames) < 4 or len(actor_centers) < 4:
                reverse_corr = 0.0
            else:
                reverse_corr_x = safe_corr(dt[:, 0], da[:, 0])
                reverse_corr_y = safe_corr(dt[:, 1], da[:, 1])
                reverse_corr = float(
                    np.clip((reverse_corr_x + reverse_corr_y) / 2.0, -1.0, 1.0)
                )
            self._trajectory_cache[(key[1], key[0])] = (
                reverse_corr, result[2], result[1]
            )
        return result

    def _distance_drop(
        self,
        actor: MouseObservation,
        target: MouseObservation,
        history: ObservationHistory,
    ) -> float:
        self._ensure_frame_cache(actor.frame)
        key = tuple(sorted((int(actor.logical_id), int(target.logical_id))))
        cached = self._distance_drop_cache.get(key)
        if cached is not None:
            return cached
        old_frame = actor.frame - self.lookback_frames
        old_actor = history.near_frame(actor.logical_id, old_frame)
        old_target = history.near_frame(target.logical_id, old_frame)
        current_distance = point_distance(actor.center_cm, target.center_cm)
        if old_actor is None or old_target is None or not np.isfinite(current_distance):
            value = 0.0
        else:
            previous_distance = point_distance(old_actor.center_cm, old_target.center_cm)
            value = (
                float(previous_distance - current_distance)
                if np.isfinite(previous_distance)
                else 0.0
            )
        self._distance_drop_cache[key] = float(value)
        return float(value)

    def _target_turn_angle(
        self,
        target: MouseObservation,
        history: ObservationHistory,
    ) -> float:
        self._ensure_frame_cache(target.frame)
        logical_id = int(target.logical_id)
        cached = self._target_turn_cache.get(logical_id)
        if cached is not None:
            return cached
        old = history.near_frame(logical_id, target.frame - self.lookback_frames)
        value = 0.0 if old is None else angle_difference_deg(old.heading, target.heading)
        self._target_turn_cache[logical_id] = float(value)
        return float(value)

    def compute(
        self,
        actor: MouseObservation,
        target: MouseObservation,
        history: ObservationHistory,
        repeated_contact_count: int,
    ) -> PairFeatures:
        center_distance = point_distance(actor.center_cm, target.center_cm)
        head_distance = point_distance(actor.head_cm, target.head_cm)
        actor_nose = actor.keypoints_cm[KP["nose"]]
        target_tail = target.keypoints_cm[KP["tail"]]
        nose_to_body = min_point_distance(actor_nose, target.keypoints_cm)
        nose_to_tail = point_distance(actor_nose, target_tail)

        direction_similarity = cosine_similarity(actor.velocity_cm_s, target.velocity_cm_s)
        vector_to_target = target.center_cm - actor.center_cm
        pursuit_alignment = cosine_similarity(actor.heading, vector_to_target)
        actor_behind_target = False
        if finite_point(actor.center_cm) and finite_point(target.center_cm) and np.all(np.isfinite(target.heading)):
            actor_behind_target = bool(np.dot(actor.center_cm - target.center_cm, target.heading) < 0)

        trajectory_corr, actor_path, target_path = self._trajectory_features(
            actor.logical_id, target.logical_id, history, actor.frame
        )
        distance_drop = self._distance_drop(actor, target, history)
        target_turn = self._target_turn_angle(target, history)

        chase_conditions = [
            np.isfinite(center_distance) and center_distance <= float(self.chase_cfg["max_distance_cm"]),
            actor.speed_cm_s > float(self.chase_cfg["actor_min_speed_cm_s"]),
            target.speed_cm_s > float(self.chase_cfg["target_min_speed_cm_s"]),
            direction_similarity > float(self.chase_cfg["direction_similarity_min"]),
            pursuit_alignment > float(self.chase_cfg["pursuit_alignment_min"]),
            actor_behind_target,
            trajectory_corr > float(self.chase_cfg["trajectory_correlation_min"]),
        ]
        chase_score = int(sum(bool(v) for v in chase_conditions))
        chase_candidate = chase_score >= int(self.chase_cfg["candidate_score_min"])
        chase_high = chase_score >= int(self.chase_cfg["high_confidence_score_min"])

        contact = bool(
            np.isfinite(nose_to_body)
            and nose_to_body < float(self.attack_cfg["contact_distance_cm"])
        )

        lunge = actor.speed_cm_s > float(self.attack_cfg["actor_lunge_speed_cm_s"])
        rapid_closing = distance_drop > float(self.attack_cfg["rapid_closing_distance_cm"])
        target_escape = target.speed_cm_s > float(self.attack_cfg["target_escape_speed_cm_s"])
        target_turning = target_turn > float(self.attack_cfg["target_turn_angle_deg"])
        repeated_contact = repeated_contact_count >= int(self.attack_cfg["repeated_contact_count"])
        head_motion = (
            actor.nose_speed_cm_s > float(self.attack_cfg["head_motion_speed_cm_s"])
            and actor.nose_speed_cm_s
            > float(self.attack_cfg["head_to_center_speed_ratio"]) * max(actor.speed_cm_s, 1.0)
        )
        dynamic_evidence = int(sum([lunge, rapid_closing, target_escape, target_turning, repeated_contact, head_motion]))

        stationary_fight = bool(
            np.isfinite(center_distance)
            and center_distance < float(self.attack_cfg["stationary_fight_distance_cm"])
            and max(actor.speed_cm_s, target.speed_cm_s)
            < float(self.attack_cfg["stationary_fight_max_center_speed_cm_s"])
            and max(actor.angular_speed_deg_s, target.angular_speed_deg_s)
            > float(self.attack_cfg["stationary_fight_min_angular_speed_deg_s"])
            and contact
        )

        attack_candidate = bool(
            (contact and dynamic_evidence >= int(self.attack_cfg["min_dynamic_evidence"]))
            or stationary_fight
        )

        return PairFeatures(
            actor_id=actor.logical_id,
            target_id=target.logical_id,
            center_distance_cm=float(center_distance),
            head_distance_cm=float(head_distance),
            actor_nose_to_target_body_cm=float(nose_to_body),
            actor_nose_to_target_tail_cm=float(nose_to_tail),
            actor_speed_cm_s=float(actor.speed_cm_s),
            target_speed_cm_s=float(target.speed_cm_s),
            actor_acceleration_cm_s2=float(actor.acceleration_cm_s2),
            target_acceleration_cm_s2=float(target.acceleration_cm_s2),
            actor_nose_speed_cm_s=float(actor.nose_speed_cm_s),
            target_nose_speed_cm_s=float(target.nose_speed_cm_s),
            direction_similarity=float(direction_similarity),
            pursuit_alignment=float(pursuit_alignment),
            actor_behind_target=bool(actor_behind_target),
            trajectory_correlation=float(trajectory_corr),
            actor_path_window_cm=float(actor_path),
            target_path_window_cm=float(target_path),
            distance_drop_cm=float(distance_drop),
            target_turn_angle_deg=float(target_turn),
            contact=bool(contact),
            repeated_contact_count=int(repeated_contact_count),
            chase_score=int(chase_score),
            chase_candidate=bool(chase_candidate),
            chase_high_confidence=bool(chase_high),
            attack_dynamic_evidence=int(dynamic_evidence),
            stationary_fight_candidate=bool(stationary_fight),
            attack_candidate=bool(attack_candidate),
        )


# ----------------------------- YOLO解析 -----------------------------


def parse_yolo_result(
    result: Any,
    expected_keypoints: int,
    max_mice: int,
    profiling: Optional[MutableMapping[str, float]] = None,
) -> List[Detection]:
    """Parse one Ultralytics result without changing legacy ordering.

    When ``profiling`` is supplied, GPU->CPU tensor transfer time and pure CPU
    object parsing time are accumulated separately.  The optional argument is
    intentionally backward compatible with all existing callers.
    """
    if result is None or result.boxes is None or result.keypoints is None:
        return []
    if result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return []

    transfer_started = time.perf_counter()
    xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float64)
    if xy.ndim != 3 or xy.shape[1] != expected_keypoints:
        raise ValueError(
            f"模型关键点数量为{xy.shape[1] if xy.ndim == 3 else '未知'}，"
            f"但程序要求{expected_keypoints}个：{KEYPOINT_NAMES}"
        )

    if getattr(result.keypoints, "conf", None) is not None:
        kp_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float64)
    else:
        kp_conf = np.ones((xy.shape[0], expected_keypoints), dtype=np.float64)

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
    raw_ids: List[Optional[int]]
    if result.boxes.id is not None:
        raw_ids = [int(v) for v in result.boxes.id.detach().cpu().numpy().tolist()]
    else:
        raw_ids = [None] * len(xy)
    transfer_seconds = time.perf_counter() - transfer_started
    if profiling is not None:
        profiling["yolo_result_transfer_seconds"] = float(
            profiling.get("yolo_result_transfer_seconds", 0.0)
        ) + transfer_seconds

    parse_started = time.perf_counter()
    order = np.argsort(-box_conf)[:max_mice]
    detections = []
    for idx in order:
        # v1.11.2：pose_quality在此落盘（有效关键点的平均置信度），
        # 供两阶段分组、镜像对判别、记忆安全更新一致使用。
        conf_row = np.asarray(kp_conf[idx], dtype=np.float64).reshape(-1)
        pts_row = np.asarray(xy[idx], dtype=np.float64)
        n_kp = min(len(conf_row), len(pts_row))
        valid_kp = (
            np.isfinite(pts_row[:n_kp]).all(axis=1)
            & (pts_row[:n_kp, 0] > 0) & (pts_row[:n_kp, 1] > 0)
            & np.isfinite(conf_row[:n_kp]) & (conf_row[:n_kp] >= 0.10)
        )
        pose_quality = float(np.mean(conf_row[:n_kp][valid_kp])) if np.any(valid_kp) else 0.0
        detections.append(
            Detection(
                raw_track_id=raw_ids[idx],
                keypoints_px=xy[idx],
                keypoint_conf=kp_conf[idx],
                bbox_xyxy=boxes_xyxy[idx],
                box_conf=float(box_conf[idx]),
                pose_quality=pose_quality,
            )
        )
    if profiling is not None:
        profiling["result_parse_seconds"] = float(
            profiling.get("result_parse_seconds", 0.0)
        ) + (time.perf_counter() - parse_started)
    return detections


# ----------------------------- 白鼠亮斑通道与全图Pose匹配（v1.11.1） -----------------------------


def detect_bright_blob_candidates(
    frame_bgr: np.ndarray,
    cfg: Optional[Mapping[str, Any]] = None,
    expected_keypoints: int = 7,
) -> List[Detection]:
    """白色小鼠亮斑检测通道（无需训练模型）。

    原理：白鼠在灰色地面上是显著亮目标，顶帽（TOPHAT）变换提取"比局部背景亮"
    的小尺度目标，对缓慢变化的背景和画面中央的聚光灯照明天然免疫。
    输出 detection_source='white_blob' 的候选（无关键点、固定低置信度），
    后续由候选过滤器（静态碎屑拒绝）和记忆分配器的运动门控晋升共同防假。
    """
    config = dict(cfg or {})
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ksize = max(int(config.get("kernel_px", 41)) | 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    delta = float(config.get("bright_delta", 22))
    mask = (tophat > delta).astype(np.uint8) * 255
    open_k = int(config.get("open_kernel_px", 5))
    close_k = int(config.get("close_kernel_px", 21))
    if open_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = float(config.get("min_area_px", 2500))
    # v1.11.3：上限放宽到45000——贴近镜头边缘的白鼠体型明显更大（实测约160×200）。
    max_area = float(config.get("max_area_px", 45000))
    min_aspect = float(config.get("min_aspect_ratio", 0.30))
    max_aspect = float(config.get("max_aspect_ratio", 3.5))
    blob_conf = float(config.get("blob_conf", 0.20))
    # 掩码像素亮度必须比周围环带地面高出该阈值——真白鼠约+60，
    # 黑鼠体表高光/反光形成的假亮斑为负值（实测-20以下）。
    min_contrast = float(config.get("min_mask_surround_contrast", 20.0))
    ring_px = int(config.get("surround_ring_px", 25))
    # v1.11.3：亮斑贴画面边缘时环带大部分出画，surround不足50像素，
    # 旧逻辑直接跳过导致边缘白鼠被误杀。改为绝对亮度兜底——
    # 真白鼠体表掩码亮度实测约180~255，地面/墙角反光极少达到165。
    edge_abs_bright_min = float(config.get("edge_abs_bright_min", 165.0))
    height, width = gray.shape[:2]
    gray_f = gray.astype(np.float32)
    out: List[Detection] = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or area > max_area:
            continue
        aspect = w / max(h, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if min_contrast > 0:
            pix_mean = float(gray_f[labels == i].mean())
            x1, y1 = max(x - ring_px, 0), max(y - ring_px, 0)
            x2, y2 = min(x + w + ring_px, width), min(y + h + ring_px, height)
            ring = np.ones((y2 - y1, x2 - x1), dtype=bool)
            ring[y - y1:y - y1 + h, x - x1:x - x1 + w] = False
            ring &= labels[y1:y2, x1:x2] == 0
            surround = gray_f[y1:y2, x1:x2][ring]
            if surround.size < 50:
                # 画面边缘：环带样本不足，用绝对亮度兜底判别。
                if pix_mean < edge_abs_bright_min:
                    continue
            elif pix_mean - float(surround.mean()) < min_contrast:
                continue
        out.append(Detection(
            raw_track_id=None,
            keypoints_px=np.full((expected_keypoints, 2), np.nan, dtype=np.float64),
            keypoint_conf=np.zeros(expected_keypoints, dtype=np.float64),
            bbox_xyxy=np.array([x, y, x + w, y + h], dtype=np.float64),
            box_conf=blob_conf,
            pose_quality=0.0,
            # 不标记为白鼠候选：让静态碎屑规则照常生效（反光假斑45帧内被清除）；
            # 真白鼠持续移动，本来就不会触发静态拒绝。
            white_score=0.0,
            is_white_candidate=False,
            appearance_mode="white_blob_tophat",
            appearance_reliable=False,
            detection_source="white_blob",
        ))
    max_blobs = int(config.get("max_blobs", 12))
    out.sort(key=lambda d: -float((d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1])))
    return out[:max_blobs]


def attach_pose_to_candidates(
    candidates: List[Detection],
    pose_detections: List[Detection],
    expand_ratio: float = 0.15,
    min_inbox_ratio: float = 0.5,
    min_iou: float = 0.20,
) -> Tuple[List[Detection], List[Detection]]:
    """把全图Pose实例的7个关键点按1:1贪心匹配挂到候选框上。

    Pose模型按训练时的全图方式推理，关键点为全图坐标，直接拷贝；
    候选保留自己的检测框（检测器框/亮斑框）。匹配不上的Pose实例原样返回，
    由调用方作为独立检测保留（Pose找到但检测器漏掉的鼠）。
    """
    if not candidates:
        return list(candidates), list(pose_detections)
    if not pose_detections:
        return list(candidates), []
    candidate_boxes = np.asarray(
        [np.asarray(cand.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for cand in candidates],
        dtype=np.float64,
    )
    widths = np.maximum(candidate_boxes[:, 2] - candidate_boxes[:, 0], 1.0)
    heights = np.maximum(candidate_boxes[:, 3] - candidate_boxes[:, 1], 1.0)
    expanded = candidate_boxes.copy()
    expanded[:, 0] -= widths * float(expand_ratio)
    expanded[:, 2] += widths * float(expand_ratio)
    expanded[:, 1] -= heights * float(expand_ratio)
    expanded[:, 3] += heights * float(expand_ratio)

    pose_count = len(pose_detections)
    point_count = len(KEYPOINT_NAMES)
    pose_points = np.full((pose_count, point_count, 2), np.nan, dtype=np.float64)
    pose_conf = np.zeros((pose_count, point_count), dtype=np.float64)
    pose_boxes = np.asarray(
        [np.asarray(pdet.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for pdet in pose_detections],
        dtype=np.float64,
    )
    for pi, pdet in enumerate(pose_detections):
        kpts = np.asarray(pdet.keypoints_px, dtype=np.float64)
        conf = np.asarray(pdet.keypoint_conf, dtype=np.float64).reshape(-1)
        n = min(len(kpts), len(conf), point_count)
        if n > 0:
            pose_points[pi, :n] = kpts[:n]
            pose_conf[pi, :n] = conf[:n]
    pose_valid = (
        np.all(np.isfinite(pose_points), axis=2)
        & (pose_points[:, :, 0] > 0)
        & (pose_points[:, :, 1] > 0)
        & np.isfinite(pose_conf)
        & (pose_conf >= 0.10)
    )
    pose_valid_count = np.sum(pose_valid, axis=1)
    px = pose_points[None, :, :, 0]
    py = pose_points[None, :, :, 1]
    inside_mask = (
        pose_valid[None, :, :]
        & (px >= expanded[:, None, None, 0])
        & (px <= expanded[:, None, None, 2])
        & (py >= expanded[:, None, None, 1])
        & (py <= expanded[:, None, None, 3])
    )
    inside_count = np.sum(inside_mask, axis=2)
    inside_ratio = np.divide(
        inside_count,
        pose_valid_count[None, :],
        out=np.zeros((len(candidates), pose_count), dtype=np.float64),
        where=pose_valid_count[None, :] > 0,
    )
    x1 = np.maximum(candidate_boxes[:, None, 0], pose_boxes[None, :, 0])
    y1 = np.maximum(candidate_boxes[:, None, 1], pose_boxes[None, :, 1])
    x2 = np.minimum(candidate_boxes[:, None, 2], pose_boxes[None, :, 2])
    y2 = np.minimum(candidate_boxes[:, None, 3], pose_boxes[None, :, 3])
    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    cand_area = np.maximum(candidate_boxes[:, 2] - candidate_boxes[:, 0], 0.0) * np.maximum(candidate_boxes[:, 3] - candidate_boxes[:, 1], 0.0)
    pose_area = np.maximum(pose_boxes[:, 2] - pose_boxes[:, 0], 0.0) * np.maximum(pose_boxes[:, 3] - pose_boxes[:, 1], 0.0)
    union = cand_area[:, None] + pose_area[None, :] - inter
    iou_matrix = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-9)
    scored: List[Tuple[float, int, int]] = []
    for ci, cand in enumerate(candidates):
        box = candidate_boxes[ci]
        if box.size < 4 or not np.all(np.isfinite(box[:4])):
            continue
        for pi, pdet in enumerate(pose_detections):
            if int(pose_valid_count[pi]) < 3:
                continue
            inside = float(inside_ratio[ci, pi])
            iou = float(iou_matrix[ci, pi])
            if inside < min_inbox_ratio and iou < min_iou:
                continue
            scored.append((0.65 * inside + 0.35 * float(iou), ci, pi))
    scored.sort(key=lambda t: -t[0])
    used_candidates: set = set()
    used_pose: set = set()
    for _, ci, pi in scored:
        if ci in used_candidates or pi in used_pose:
            continue
        used_candidates.add(ci)
        used_pose.add(pi)
        cand = candidates[ci]
        pdet = pose_detections[pi]
        cand.keypoints_px = np.asarray(pdet.keypoints_px, dtype=np.float64).copy()
        cand.keypoint_conf = np.asarray(pdet.keypoint_conf, dtype=np.float64).copy()
        conf = cand.keypoint_conf.reshape(-1)
        good = np.isfinite(conf) & (conf >= 0.10)
        cand.pose_quality = float(np.mean(conf[good])) if np.any(good) else 0.0
        src = str(cand.detection_source)
        if "posefull" not in src:
            cand.detection_source = f"{src}_posefull"
    unmatched = [p for pi, p in enumerate(pose_detections) if pi not in used_pose]
    return candidates, unmatched


def fuse_pose_primary_detections(
    pose_dets: Sequence[Detection],
    supplement_dets: Sequence[Detection],
    same_mouse_center_bl: float = 0.45,
    same_mouse_min_iou: float = 0.15,
    mega_size_ratio: float = 1.6,
    kp_same_mouse_bl: float = 0.30,
    kp_same_mouse_bl_unkeyed: float = 0.20,
    kp_min_conf: float = 0.10,
) -> List[Detection]:
    """Pose主通道融合（v1.12.0，按文档§5：第1步YOLO-Pose输出当前帧所有检测）。

    Pose模型全图输出（框+关键点）是唯一权威检测流；普通检测器框与白鼠亮斑
    只是"补缺补充框"：

    0. **Pose框自身去重**（v1.12.4）：Pose模型NMS不完美时同一只鼠会出多个
       重叠实例，旧版全部保留 → 一鼠多框多TMP。先按关键点锚定规则合并。
    1. 补充框与某Pose框判为同一目标（中心距<0.45体长且IoU≥0.15）且尺寸相近
       （面积比<mega_size_ratio）→ 丢弃补充框：Pose框已覆盖这只鼠。
    2. 尺寸悬殊（Pose框面积≥补充框×mega_size_ratio）→ 该Pose框是**覆盖多只鼠的
       超大框**，补充框是其中独立一只（典型：白鼠亮斑）→ 保留补充框。
    3. 超大Pose框若被≥2个保留补充框嵌入（各代表一只鼠）→ 丢弃超大框，
       其关键点转挂到中心最近的补充框上（拆分成多只鼠）。
    4. 补充框之间仍按"中心距+IoU"去重合并（检测器框优先于亮斑框）。
    5. **最终关键点锚定去重**（v1.12.4）：以上各步仍可能漏掉"小补充框落在
       松散Pose大框里被误判为独立鼠"等同目标残留，统一按关键点簇质心/中心距
       终审一次，保证每只鼠只剩一个框进入后续匹配。

    与v1.11.x的区别：不再有"检测器框为主、Pose关键点往回挂"的方向，
    Pose框不再被合并吞掉，白鼠亮斑也不会被并进超大框。
    """
    pose_list = list(pose_dets)
    sups = list(supplement_dets)

    # 0)：Pose框自身同目标去重（v1.12.4）。
    pose_list = dedup_same_object_detections(
        pose_list,
        kp_same_mouse_bl=kp_same_mouse_bl,
        kp_same_mouse_bl_unkeyed=kp_same_mouse_bl_unkeyed,
        same_mouse_center_bl=same_mouse_center_bl,
        same_mouse_min_iou=same_mouse_min_iou,
        kp_min_conf=kp_min_conf,
    )

    def _same_object(a: Detection, b: Detection) -> bool:
        dist = point_distance(a.center_px, b.center_px)
        if not np.isfinite(dist):
            return False
        bl = max(float(a.body_length_px), float(b.body_length_px), 8.0)
        if dist >= same_mouse_center_bl * bl:
            return False
        return bbox_iou_xyxy(a.bbox_xyxy, b.bbox_xyxy) >= same_mouse_min_iou

    def _size_ratio(a: Detection, b: Detection) -> float:
        aa = max(float(a.bbox_xyxy[2] - a.bbox_xyxy[0]) * float(a.bbox_xyxy[3] - a.bbox_xyxy[1]), 1.0)
        bb = max(float(b.bbox_xyxy[2] - b.bbox_xyxy[0]) * float(b.bbox_xyxy[3] - b.bbox_xyxy[1]), 1.0)
        return max(aa, bb) / min(aa, bb)

    def _center_inside(inner: Detection, outer: Detection, margin: float = 4.0) -> bool:
        c = inner.center_px
        b = outer.bbox_xyxy
        return bool(
            np.isfinite(c[0]) and np.isfinite(c[1])
            and b[0] - margin <= c[0] <= b[2] + margin
            and b[1] - margin <= c[1] <= b[3] + margin
        )

    # 1)+2)：补充框 vs Pose框
    kept_sup: List[Detection] = []
    if sups and pose_list:
        sup_centers = np.asarray([np.asarray(det.center_px, dtype=np.float64) for det in sups], dtype=np.float64)
        pose_centers = np.asarray([np.asarray(det.center_px, dtype=np.float64) for det in pose_list], dtype=np.float64)
        cross_center = np.linalg.norm(sup_centers[:, None, :] - pose_centers[None, :, :], axis=2)
        sup_boxes = np.asarray([np.asarray(det.bbox_xyxy, dtype=np.float64)[:4] for det in sups], dtype=np.float64)
        pose_boxes = np.asarray([np.asarray(det.bbox_xyxy, dtype=np.float64)[:4] for det in pose_list], dtype=np.float64)
        x1 = np.maximum(sup_boxes[:, None, 0], pose_boxes[None, :, 0])
        y1 = np.maximum(sup_boxes[:, None, 1], pose_boxes[None, :, 1])
        x2 = np.minimum(sup_boxes[:, None, 2], pose_boxes[None, :, 2])
        y2 = np.minimum(sup_boxes[:, None, 3], pose_boxes[None, :, 3])
        inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        sup_area = np.maximum(sup_boxes[:, 2] - sup_boxes[:, 0], 0.0) * np.maximum(sup_boxes[:, 3] - sup_boxes[:, 1], 0.0)
        pose_area = np.maximum(pose_boxes[:, 2] - pose_boxes[:, 0], 0.0) * np.maximum(pose_boxes[:, 3] - pose_boxes[:, 1], 0.0)
        union = sup_area[:, None] + pose_area[None, :] - inter
        cross_iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-9)
        cross_body = np.maximum(
            np.asarray([max(float(det.body_length_px), 8.0) for det in sups], dtype=np.float64)[:, None],
            np.asarray([max(float(det.body_length_px), 8.0) for det in pose_list], dtype=np.float64)[None, :],
        )
        cross_ratio = np.maximum(
            np.maximum(sup_area[:, None], 1.0) / np.maximum(pose_area[None, :], 1.0),
            np.maximum(pose_area[None, :], 1.0) / np.maximum(sup_area[:, None], 1.0),
        )
    else:
        cross_center = np.zeros((len(sups), len(pose_list)), dtype=np.float64)
        cross_iou = np.zeros_like(cross_center)
        cross_body = np.ones_like(cross_center)
        cross_ratio = np.ones_like(cross_center)

    for sup_index, sup in enumerate(sups):
        covered = False
        for pose_index, pd in enumerate(pose_list):
            distance = float(cross_center[sup_index, pose_index])
            if not np.isfinite(distance) or distance >= same_mouse_center_bl * float(cross_body[sup_index, pose_index]):
                continue
            if float(cross_iou[sup_index, pose_index]) < same_mouse_min_iou:
                continue
            if float(cross_ratio[sup_index, pose_index]) < mega_size_ratio:
                covered = True  # 同一目标且尺寸相近：Pose已覆盖
                break
            # 尺寸悬殊：Pose是超大框，补充框是其中独立一只 → 保留
        if not covered:
            kept_sup.append(sup)
    # 3)：补充框之间去重（同鼠合并）
    kept_sup = merge_cross_channel_duplicates(kept_sup, same_mouse_center_bl, same_mouse_min_iou)
    # 4)：超大Pose框拆分——被≥2个保留补充框嵌入则丢弃超大框。
    # 超大框中心落在两鼠之间，与各鼠中心距必然>0.45体长，
    # 故"嵌入"按补充框中心是否落在超大框内部判定，不用中心距。
    out_pose: List[Detection] = []
    if kept_sup and pose_list:
        kept_centers = np.asarray([np.asarray(det.center_px, dtype=np.float64) for det in kept_sup], dtype=np.float64)
        kept_boxes = np.asarray([np.asarray(det.bbox_xyxy, dtype=np.float64)[:4] for det in kept_sup], dtype=np.float64)
        kept_area = np.maximum(kept_boxes[:, 2] - kept_boxes[:, 0], 0.0) * np.maximum(kept_boxes[:, 3] - kept_boxes[:, 1], 0.0)
        pose_boxes2 = np.asarray([np.asarray(det.bbox_xyxy, dtype=np.float64)[:4] for det in pose_list], dtype=np.float64)
        pose_area2 = np.maximum(pose_boxes2[:, 2] - pose_boxes2[:, 0], 0.0) * np.maximum(pose_boxes2[:, 3] - pose_boxes2[:, 1], 0.0)
        center_inside = (
            (kept_centers[:, None, 0] >= pose_boxes2[None, :, 0] - 4.0)
            & (kept_centers[:, None, 0] <= pose_boxes2[None, :, 2] + 4.0)
            & (kept_centers[:, None, 1] >= pose_boxes2[None, :, 1] - 4.0)
            & (kept_centers[:, None, 1] <= pose_boxes2[None, :, 3] + 4.0)
        )
        size_ratio2 = np.maximum(
            np.maximum(kept_area[:, None], 1.0) / np.maximum(pose_area2[None, :], 1.0),
            np.maximum(pose_area2[None, :], 1.0) / np.maximum(kept_area[:, None], 1.0),
        )
    else:
        center_inside = np.zeros((len(kept_sup), len(pose_list)), dtype=bool)
        size_ratio2 = np.ones((len(kept_sup), len(pose_list)), dtype=np.float64)

    for pose_index, pd in enumerate(pose_list):
        embedded = [
            s for sup_index, s in enumerate(kept_sup)
            if bool(center_inside[sup_index, pose_index])
            and float(size_ratio2[sup_index, pose_index]) >= mega_size_ratio
        ]
        if len(embedded) >= 2:
            nearest = min(embedded, key=lambda s: point_distance(s.center_px, pd.center_px))
            if float(nearest.pose_quality) <= 0.0 and float(pd.pose_quality) > 0.0:
                nearest.keypoints_px = np.asarray(pd.keypoints_px, dtype=np.float64).copy()
                nearest.keypoint_conf = np.asarray(pd.keypoint_conf, dtype=np.float64).copy()
                nearest.pose_quality = float(pd.pose_quality)
            continue  # 超大框被补充框拆分替代
        out_pose.append(pd)
    # 5)：最终关键点锚定去重（v1.12.4）——每只鼠只剩一个框进入匹配。
    return dedup_same_object_detections(
        out_pose + kept_sup,
        kp_same_mouse_bl=kp_same_mouse_bl,
        kp_same_mouse_bl_unkeyed=kp_same_mouse_bl_unkeyed,
        same_mouse_center_bl=same_mouse_center_bl,
        same_mouse_min_iou=same_mouse_min_iou,
        kp_min_conf=kp_min_conf,
    )


def correct_head_tail_orientation(
    points_px: np.ndarray,
    keypoint_conf: np.ndarray,
    reference_px: Optional[np.ndarray] = None,
    velocity_px: Optional[np.ndarray] = None,
    min_conf: float = 0.15,
    min_speed_px: float = 1.5,
    hard_margin: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """头尾方向校正（v1.12.7，速度一致性守卫）。

    模型在域外场地上可能把整套骨架预测成180°颠倒（nose标在尾端、
    耳朵标在髋端）。手性校正只换左右，不管头尾；本函数用两个独立
    证据判断是否需要整体翻转：
    1. **速度方向**：小鼠绝大多数时间头朝运动方向走——候选骨架的
       nose→tail轴与速度方向的一致性（体长归一化点积）；
    2. **帧间连续性**：候选nose应靠近上一帧（已校正）的nose位置。
    仅当翻转候选的总分领先原始≥hard_margin才翻转（防噪声来回跳），
    证据不足（速度慢/端点低置信/无参考）时保持原样。
    返回 (修正后关键点, 修正后置信度, 是否翻转)。
    """
    pts = np.asarray(points_px, dtype=np.float64)
    conf = np.asarray(keypoint_conf, dtype=np.float64).reshape(-1)
    n = min(len(pts), len(conf), len(KEYPOINT_NAMES))
    if n < len(KEYPOINT_NAMES):
        return pts.copy(), conf.copy(), False
    nose_i, tail_i = KP["nose"], KP["tail"]

    def _valid(idx: int) -> bool:
        return (idx < n and np.isfinite(conf[idx]) and conf[idx] >= min_conf
                and finite_point(pts[idx]))

    if not (_valid(nose_i) and _valid(tail_i)):
        return pts.copy(), conf.copy(), False

    body = max(float(point_distance(pts[nose_i], pts[tail_i])), 8.0)
    # 180°翻转映射：鼻↔尾、左耳↔左髋、右耳↔右髋、颈保留
    flip_map = [tail_i, KP["left_hind"], KP["right_hind"], KP["neck"],
                KP["left_ear"], KP["right_ear"], nose_i]
    cand_pts = [pts, pts[flip_map]]
    cand_conf = [conf, conf[flip_map]]

    def _score(p: np.ndarray) -> float:
        score = 0.0
        if velocity_px is not None:
            v = np.asarray(velocity_px, dtype=np.float64)
            speed = float(np.linalg.norm(v))
            if np.isfinite(speed) and speed >= min_speed_px:
                axis = p[nose_i] - p[tail_i]
                score += float(np.dot(axis, v / speed)) / body  # 头朝运动方向为正
        if reference_px is not None:
            ref = np.asarray(reference_px, dtype=np.float64)
            if nose_i < len(ref) and finite_point(ref[nose_i]):
                d = point_distance(p[nose_i], ref[nose_i])
                if np.isfinite(d):
                    score += 0.5 * (-float(d) / body)  # 连续性（越近越好）
        return score

    s_orig, s_flip = _score(cand_pts[0]), _score(cand_pts[1])
    if s_flip > s_orig + hard_margin:
        return cand_pts[1].copy(), cand_conf[1].copy(), True
    return pts.copy(), conf.copy(), False


def stabilize_keypoint_chirality(
    points_px: np.ndarray,
    keypoint_conf: np.ndarray,
    reference_px: Optional[np.ndarray] = None,
    min_conf: float = 0.10,
    min_body_px: float = 8.0,
    improve_margin: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """校正左右耳/左右髋的单帧镜像翻转（v1.11.3，"蝴蝶结"骨架的来源）。

    Pose模型对俯视小鼠偶尔会单帧把左右耳或左右髋预测到身体对侧，
    耳-颈连线交叉成X形。绝对左右从俯视不可知，但**帧间一致性**可知：
    以轨迹上一帧关键点为基准，在"不换/换耳对/换髋对/两对都换"四种组合中
    选与参考平均距离最小者；只有明显更优才交换，避免正常帧被误翻。
    返回值：(校正后关键点, 置信度, 是否发生交换)。
    """
    pts = np.asarray(points_px, dtype=np.float64).copy()
    conf = np.asarray(keypoint_conf, dtype=np.float64).reshape(-1).copy()
    n = min(len(pts), len(conf), len(KEYPOINT_NAMES))
    if n < 7 or reference_px is None:
        return pts, conf, False
    ref = np.asarray(reference_px, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[0] < n:
        return pts, conf, False
    le, re = KP["left_ear"], KP["right_ear"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    valid = (
        np.isfinite(conf[:n]) & (conf[:n] >= min_conf)
        & np.all(np.isfinite(pts[:n]), axis=1)
        & np.all(np.isfinite(ref[:n]), axis=1)
    )
    if int(valid.sum()) < 4:
        return pts, conf, False
    nose, tail = KP["nose"], KP["tail"]
    body = point_distance(ref[nose], ref[tail]) if (valid[nose] and valid[tail]) else float("nan")
    if not np.isfinite(body) or body < min_body_px:
        diffs = np.linalg.norm(pts[:n][valid] - ref[:n][valid], axis=1)
        body = max(float(np.median(diffs)) * 4.0, min_body_px)

    def _combo(swap_ears: bool, swap_hips: bool) -> Tuple[float, np.ndarray]:
        cand = pts[:n].copy()
        if swap_ears:
            cand[[le, re]] = cand[[re, le]]
        if swap_hips:
            cand[[lh, rh]] = cand[[rh, lh]]
        d = float(np.mean(np.linalg.norm(cand[valid] - ref[:n][valid], axis=1))) / body
        return d, cand

    base_dist, _ = _combo(False, False)
    best_dist, best_cand, best_combo = base_dist, None, (False, False)
    for combo in ((True, False), (False, True), (True, True)):
        d, cand = _combo(*combo)
        if d < best_dist:
            best_dist, best_cand, best_combo = d, cand, combo
    if best_combo == (False, False) or best_cand is None:
        return pts, conf, False
    # 需明显更优才交换：至少改善0.02倍体长或相对15%（正常帧抖动远小于此）。
    if best_dist > base_dist - max(0.02, improve_margin * base_dist):
        return pts, conf, False
    out = pts.copy()
    out[:n] = best_cand
    return out, conf, True


def keypoint_cluster_centroid(
    det: Detection,
    min_conf: float = 0.10,
    min_points: int = 3,
) -> Optional[np.ndarray]:
    """检测的有效关键点簇质心（v1.12.4）。

    有效点：置信度≥min_conf 且坐标有限；不足 min_points 个有效点返回 None
    （此时该检测按"无关键点"处理，退回框规则）。关键点扎在鼠身上，
    比松散/跨鼠的检测框更能代表"这只鼠实际在哪"。
    """
    pts = getattr(det, "keypoints_px", None)
    conf = getattr(det, "keypoint_conf", None)
    if pts is None or conf is None:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    n = min(len(pts), len(conf))
    if n < min_points:
        return None
    valid = (
        np.isfinite(conf[:n]) & (conf[:n] >= min_conf)
        & np.all(np.isfinite(pts[:n]), axis=1)
    )
    if int(valid.sum()) < min_points:
        return None
    centroid = np.mean(pts[:n][valid], axis=0)
    if not np.all(np.isfinite(centroid)):
        return None
    return centroid


def skeleton_anatomy_ok(
    det: Detection,
    min_points: int = 3,
    min_conf: float = 0.20,
    span_ratio: float = 0.45,
    centroid_margin: float = 0.15,
) -> bool:
    """骨架解剖学健全性检查（v1.12.7）。

    模型在域外场地（如裸板）上的典型失效：关键点自信地挤作一团、
    跨度远小于身体，或散到框外——位置全错但置信度不低，单靠置信度
    无法识别。检查（全部满足才算健全）：
    1. conf≥min_conf 的有效点 ≥ min_points；
    2. 有效点最大两两距离（骨架跨度）≥ span_ratio×框对角线——
       正常骨架必然从鼻到尾横跨身体，挤作一团即判坏；
    3. 有效点质心落在按 centroid_margin 外扩的框内——散到框外即判坏。
    判坏的检测应把跟踪/几何中心回退到检测框中心（框在域外仍可靠）。
    """
    pts = getattr(det, "keypoints_px", None)
    conf = getattr(det, "keypoint_conf", None)
    if pts is None or conf is None:
        return False
    pts = np.asarray(pts, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    n = min(len(pts), len(conf))
    if n < min_points:
        return False
    valid = (
        np.isfinite(conf[:n]) & (conf[:n] >= min_conf)
        & np.all(np.isfinite(pts[:n]), axis=1)
    )
    if int(valid.sum()) < min_points:
        return False
    good = pts[:n][valid]
    span = 0.0
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            span = max(span, float(point_distance(good[i], good[j])))
    x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
    diag = float(np.hypot(x2 - x1, y2 - y1))
    if diag <= 1e-6:
        return False
    if span < span_ratio * diag:
        return False
    centroid = np.mean(good, axis=0)
    mx = centroid_margin * (x2 - x1)
    my = centroid_margin * (y2 - y1)
    if not (x1 - mx <= centroid[0] <= x2 + mx and y1 - my <= centroid[1] <= y2 + my):
        return False
    return True


def dedup_same_object_detections(
    detections: Sequence[Detection],
    kp_same_mouse_bl: float = 0.30,
    kp_same_mouse_bl_unkeyed: float = 0.20,
    same_mouse_center_bl: float = 0.45,
    same_mouse_min_iou: float = 0.15,
    kp_min_conf: float = 0.10,
) -> List[Detection]:
    """同目标叠框去重（v1.12.4，关键点锚定）。

    Pose模型自身NMS不完美时会对同一只鼠输出多个重叠实例；旧管线只去重
    "补充框 vs Pose框"，从不对Pose框彼此去重——叠框各自建TMP轨迹互相抢命中、
    连续命中被打断，正是"一只鼠多个框多个TMP、ID迟迟不晋升"的根因。

    判定锚定**关键点簇**而非框（框可能松散/跨两只鼠，关键点扎在鼠身上）：
    - 双方都有有效关键点簇：簇质心距 < kp_same_mouse_bl×体长 → 同鼠。
      两只相邻真鼠的骨架簇相距≥0.5倍体长，不会被误并。
    - 仅一方有关键点：无关键点框中心距关键点簇质心
      < kp_same_mouse_bl_unkeyed×体长 → 同鼠。阈值更紧，保护贴背/骑跨的
      相邻白鼠亮斑不被并进黑鼠。
    - 双方都无关键点：退回"中心距<same_mouse_center_bl×体长 且 IoU≥阈值"。

    组内代表：有关键点者优先，其余按 pose_quality→box_conf→面积 排序；
    代表无姿态而组内有姿态时，把最优姿态拷贝给代表（与
    merge_cross_channel_duplicates 的移交规则一致）。
    """
    dets = list(detections)
    n = len(dets)
    if n < 2:
        return dets
    bodies = [max(float(d.body_length_px), 8.0) for d in dets]
    centroids = [keypoint_cluster_centroid(d, min_conf=kp_min_conf) for d in dets]
    center_matrix, iou_matrix, _ = _pairwise_bbox_geometry(dets)
    centroid_array = np.full((n, 2), np.nan, dtype=np.float64)
    centroid_valid = np.zeros(n, dtype=bool)
    for idx, centroid in enumerate(centroids):
        if centroid is not None and finite_point(np.asarray(centroid, dtype=np.float64)):
            centroid_array[idx] = np.asarray(centroid, dtype=np.float64)
            centroid_valid[idx] = True
    centroid_delta = centroid_array[:, None, :] - centroid_array[None, :, :]
    centroid_distance = np.linalg.norm(centroid_delta, axis=2)
    centroid_distance[~(centroid_valid[:, None] & centroid_valid[None, :])] = np.nan
    det_centers = np.asarray([np.asarray(d.center_px, dtype=np.float64) for d in dets], dtype=np.float64)
    center_to_centroid = np.linalg.norm(det_centers[:, None, :] - centroid_array[None, :, :], axis=2)
    center_to_centroid[:, ~centroid_valid] = np.nan
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _same(i: int, j: int) -> bool:
        bl = max(bodies[i], bodies[j])
        ci, cj = centroids[i], centroids[j]
        if ci is not None and cj is not None:
            dist = float(centroid_distance[i, j])
            return bool(np.isfinite(dist) and dist < kp_same_mouse_bl * bl)
        if ci is not None or cj is not None:
            keyed, unkeyed = (i, j) if ci is not None else (j, i)
            dist = float(center_to_centroid[unkeyed, keyed])
            return bool(np.isfinite(dist) and dist < kp_same_mouse_bl_unkeyed * bl)
        dist = float(center_matrix[i, j])
        if not np.isfinite(dist) or dist >= same_mouse_center_bl * bl:
            return False
        return float(iou_matrix[i, j]) >= same_mouse_min_iou

    for i in range(n):
        for j in range(i + 1, n):
            if not _same(i, j):
                continue
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    def _rank(idx: int) -> Tuple[int, float, float, float]:
        d = dets[idx]
        has_kp = 1 if centroids[idx] is not None else 0
        area = float(max(d.bbox_xyxy[2] - d.bbox_xyxy[0], 0.0)) * float(
            max(d.bbox_xyxy[3] - d.bbox_xyxy[1], 0.0))
        return (has_kp, float(d.pose_quality), float(d.box_conf), area)

    out: List[Detection] = []
    for members in groups.values():
        rep_idx = max(members, key=_rank)
        rep = dets[rep_idx]
        if len(members) > 1:
            # 代表无有效姿态而组内成员有 → 移交最优姿态。
            if centroids[rep_idx] is None:
                donor = max(
                    (m for m in members if centroids[m] is not None),
                    key=lambda m: float(dets[m].pose_quality),
                    default=None,
                )
                if donor is not None:
                    dd = dets[donor]
                    rep.keypoints_px = np.asarray(dd.keypoints_px, dtype=np.float64).copy()
                    rep.keypoint_conf = np.asarray(dd.keypoint_conf, dtype=np.float64).copy()
                    rep.pose_quality = float(dd.pose_quality)
            rep.box_conf = max(float(rep.box_conf), max(float(dets[m].box_conf) for m in members))
            if "+dedup" not in str(rep.detection_source):
                rep.detection_source = f"{rep.detection_source}+dedup"
        out.append(rep)
    return out


def merge_cross_channel_duplicates(
    detections: Sequence[Detection],
    same_mouse_center_bl: float = 0.45,
    same_mouse_min_iou: float = 0.15,
) -> List[Detection]:
    """跨通道重复检测合并（v1.11.3）。

    三路融合（检测器/亮斑/全图Pose）可能对同一只鼠产出多个候选框。
    叠框会为同一只鼠建多条TMP轨迹互相竞争命中——谁都达不到连续晋升条件，
    是白鼠ID不稳定与"骨架混乱"（一鼠多套骨架）的直接来源。

    同一目标判定（两个条件缺一不可）：
    - 中心距 < same_mouse_center_bl × 体长（ bbox长边近似）；
    - IoU ≥ same_mouse_min_iou。
    两只紧挨的不同小鼠（如黑鼠+白鼠）框可能重叠但中心相距较远，
    必须都保留——旧规则按IoU直接丢弃亮斑，正是贴黑鼠的白鼠被误杀的原因。

    组内代表优先级：detector_box+pose > pose_full > white_blob+pose
    > detector_box > white_blob；代表置信度取组内最大，来源追加"+merged"；
    代表无姿态而组内有姿态时，把最优姿态拷贝给代表。
    """
    dets = list(detections)
    n = len(dets)
    if n < 2:
        return dets
    centers = [d.center_px for d in dets]
    bodies = [max(float(d.body_length_px), 8.0) for d in dets]
    center_matrix, iou_matrix, _ = _pairwise_bbox_geometry(dets)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(center_matrix[i, j])
            if not np.isfinite(dist):
                continue
            if dist >= same_mouse_center_bl * max(bodies[i], bodies[j]):
                continue
            if float(iou_matrix[i, j]) < same_mouse_min_iou:
                continue
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    def _rank(d: Detection) -> int:
        src = str(d.detection_source)
        has_pose = float(d.pose_quality) > 0.0
        if src.startswith("detector_box") and has_pose:
            return 0
        if src == "pose_full":
            return 1
        if src.startswith("white_blob") and has_pose:
            return 2
        if src.startswith("detector_box"):
            return 3
        if src.startswith("white_blob"):
            return 4
        return 5

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)
    out: List[Detection] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(dets[members[0]])
            continue
        members.sort(key=lambda k: (_rank(dets[k]), -float(dets[k].box_conf)))
        keep = dets[members[0]]
        keep.box_conf = max(float(dets[k].box_conf) for k in members)
        if "+merged" not in str(keep.detection_source):
            keep.detection_source = f"{keep.detection_source}+merged"
        if float(keep.pose_quality) <= 0.0:
            donor = max(members, key=lambda k: float(dets[k].pose_quality))
            if float(dets[donor].pose_quality) > 0.0:
                keep.keypoints_px = np.asarray(dets[donor].keypoints_px, dtype=np.float64).copy()
                keep.keypoint_conf = np.asarray(dets[donor].keypoint_conf, dtype=np.float64).copy()
                keep.pose_quality = float(dets[donor].pose_quality)
        out.append(keep)
    return out


class ArenaFloorMask:
    """场地地面多边形掩码（v1.11.2）。

    中心落在地面多边形（+容差）之外的候选一律拒绝——墙壁上的镜像影子、
    场地外的桌面杂物等在几何上就不可能是场地内的小鼠。
    多边形用像素坐标给出（顺时针/逆时针均可）；未给多边形时可用
    frame_border_margin 四边距形成矩形代替。
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        config = dict(cfg or {})
        self.enabled = bool(config.get("enabled", False))
        self.tolerance_px = float(config.get("tolerance_px", 12))
        points = config.get("polygon", []) or []
        self.polygon: Optional[np.ndarray] = (
            np.asarray(points, dtype=np.float64) if len(points) >= 3 else None
        )
        self.margins = dict(config.get("frame_border_margin", {}) or {})
        self._contour: Optional[np.ndarray] = None

    def _ensure_contour(self, frame_shape: Optional[Tuple[int, int]]) -> None:
        if self._contour is not None:
            return
        if self.polygon is not None:
            self._contour = self.polygon.astype(np.float32).reshape(-1, 1, 2)
            return
        if frame_shape is not None and self.margins:
            h, w = int(frame_shape[0]), int(frame_shape[1])
            left = float(self.margins.get("left_px", 0))
            right = float(self.margins.get("right_px", 0))
            top = float(self.margins.get("top_px", 0))
            bottom = float(self.margins.get("bottom_px", 0))
            if (left or right or top or bottom) and (w - left - right > 10) and (h - top - bottom > 10):
                self.polygon = np.array(
                    [[left, top], [w - right, top], [w - right, h - bottom], [left, h - bottom]],
                    dtype=np.float64,
                )
                self._contour = self.polygon.astype(np.float32).reshape(-1, 1, 2)

    def allows(self, x: float, y: float) -> bool:
        if not self.enabled or self._contour is None:
            return True
        return cv2.pointPolygonTest(self._contour, (float(x), float(y)), True) >= -self.tolerance_px

    def boundary_distance(self, x: float, y: float) -> float:
        """点到多边形边界的带符号距离（内部为正），用于镜像对判别。"""
        if self._contour is None:
            return float("inf")
        return float(cv2.pointPolygonTest(self._contour, (float(x), float(y)), True))

    def filter(
        self,
        detections: Sequence[Detection],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        self._ensure_contour(frame_shape)
        if not self.enabled or self._contour is None:
            return list(detections)
        return [d for d in detections if self.allows(d.center_px[0], d.center_px[1])]


def _box_contrast_features(
    frame_gray: np.ndarray,
    box_xyxy: Sequence[float],
    ring_px: int = 15,
) -> Optional[Tuple[float, float, float]]:
    """盒内对比度特征：(最暗40%像素均值, 环带背景均值, 最亮40%像素均值)。

    真黑鼠：暗部均值远低于背景（比值约0.4）；墙上影子/水痕：暗部与背景
    接近（比值约0.65+）——"更淡"是影子区别于真鼠的稳定物证。
    白鼠是亮目标：最亮40%显著高于背景，用于亮目标豁免（绝不判为影子）。
    """
    if frame_gray is None or frame_gray.size == 0:
        return None
    h_img, w_img = frame_gray.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box_xyxy[:4])
    x1, x2 = max(0, min(x1, w_img)), max(0, min(x2, w_img))
    y1, y2 = max(0, min(y1, h_img)), max(0, min(y2, h_img))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    roi = frame_gray[y1:y2, x1:x2].astype(np.float32).reshape(-1)
    k = max(int(roi.size * 0.40), 8)
    part = np.partition(roi, k)
    obj_mean = float(part[:k].mean())
    bright_mean = float(part[-k:].mean()) if roi.size > k else float(part.mean())
    sx1, sy1 = max(0, x1 - ring_px), max(0, y1 - ring_px)
    sx2, sy2 = min(w_img, x2 + ring_px), min(h_img, y2 + ring_px)
    ring = frame_gray[sy1:sy2, sx1:sx2].astype(np.float32)
    keep = np.ones(ring.shape, dtype=bool)
    keep[y1 - sy1:y2 - sy1, x1 - sx1:x2 - sx1] = False
    bg = ring[keep]
    if bg.size < 30:
        return None
    return obj_mean, float(bg.mean()), bright_mean


def suppress_reflection_pairs(
    detections: Sequence[Detection],
    mask: ArenaFloorMask,
    max_distance_body_lengths: float = 1.3,
    boundary_band_px: float = 90.0,
    max_size_log_ratio: float = 0.35,
    max_size_log_ratio_with_contrast: float = 1.00,
    frame_gray: Optional[np.ndarray] = None,
    pose_quality_ratio: float = 0.9,
    contrast_margin: float = 0.15,
    bright_object_factor: float = 1.3,
) -> List[Detection]:
    """镜像影子对判别（v1.11.2；v1.11.3扩展带姿态嫌疑者）。

    判定规则（候选B被判为候选A的镜像）：
    - A、B中心距离 ≤ max_distance_body_lengths × 体长，且体型相近；
    - B比A更贴近场地边界（boundary_distance更小且 < boundary_band_px）；
    - B无姿态证据（反光/影子是侧视图，Pose模型不会给它关键点）；
    - A有姿态证据（A是真实小鼠的强证据）且置信度不比B弱太多。

    v1.11.3：实测墙上的影子/水痕也会被Pose模型画上骨架（携带姿态证据），
    旧的"B必须无姿态"条件被绕过。对带姿态嫌疑者追加两条物证：
    - B的姿态置信度显著弱于A（pose_quality < A × pose_quality_ratio）；
    - B本体相对背景明显更淡（暗部/背景比值 > A + contrast_margin），
      且B最亮部分不比背景亮（bright_object_factor 豁免真白鼠——亮目标不是影子）。
    需要传入 frame_gray 才能启用带姿态分支；未传入时保持旧行为。

    白鼠亮斑通道豁免判别：真白鼠本就贴墙活动，其反光大多在掩码外已被剔除。
    两只都无姿态时宁可都保留（防止误删贴墙真鼠），遗留反光由TMP的TTL消化。
    """
    dets = list(detections)
    mask._ensure_contour(None)
    if not mask.enabled or mask._contour is None or len(dets) < 2:
        return dets
    centers = [d.center_px for d in dets]
    bodies = [max(float(d.body_length_px), 8.0) for d in dets]

    contour_points = np.asarray(mask._contour, dtype=np.float64).reshape(-1, 2)

    # Adaptive heat-map boundaries can contain more than one hundred hull
    # vertices.  The previous nested Python loop evaluated nine box samples
    # against every segment one by one, producing roughly 60,000 tiny NumPy
    # calls in only 30 frames.  Keep the exact per-wall profile semantics while
    # evaluating all segment/sample projections in one vectorized operation.
    contour_starts = np.ascontiguousarray(contour_points, dtype=np.float64)
    contour_ends = np.roll(contour_starts, -1, axis=0)
    contour_segments = contour_ends - contour_starts
    contour_denominator = np.einsum(
        "ij,ij->i", contour_segments, contour_segments
    )

    def _boundary_profile(detection: Detection) -> np.ndarray:
        """分别计算候选框到每一段墙面的距离，避免角落处比较错墙。"""
        x1, y1, x2, y2 = (
            float(value) for value in detection.bbox_xyxy[:4]
        )
        middle_x = (x1 + x2) / 2.0
        middle_y = (y1 + y2) / 2.0
        samples = np.asarray(
            [
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2],
                [middle_x, y1],
                [middle_x, y2],
                [x1, middle_y],
                [x2, middle_y],
                [middle_x, middle_y],
            ],
            dtype=np.float64,
        )
        delta = samples[None, :, :] - contour_starts[:, None, :]
        numerator = np.einsum(
            "sni,si->sn", delta, contour_segments, optimize=True
        )
        fractions = np.divide(
            numerator,
            contour_denominator[:, None],
            out=np.zeros_like(numerator),
            where=contour_denominator[:, None] > 1.0e-9,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = (
            contour_starts[:, None, :]
            + fractions[:, :, None] * contour_segments[:, None, :]
        )
        distances = np.linalg.norm(samples[None, :, :] - projections, axis=2)
        return np.min(distances, axis=1)

    boundary_profiles = [_boundary_profile(detection) for detection in dets]
    suppressed: set = set()
    for i in range(len(dets)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            body = float(np.nanmedian([bodies[i], bodies[j]]))
            dist = point_distance(centers[i], centers[j])
            if not np.isfinite(dist) or dist > max_distance_body_lengths * body:
                continue
            size_log = abs(math.log(max(bodies[i], 1.0) / max(bodies[j], 1.0)))
            # 无灰度物证时继续使用严格体型门；传入原始画面后允许影子框比
            # 真鼠更窄或更长，再由“贴墙 + 对比度”联合判别。旧代码在此处
            # 提前跳过，导致墙影虽然灰度特征明确，仍没有机会进入主判据。
            if size_log > (
                max_size_log_ratio_with_contrast if frame_gray is not None else max_size_log_ratio
            ):
                continue
            for suspect_idx, real_idx in ((j, i), (i, j)):
                suspect = dets[suspect_idx]
                real = dets[real_idx]
                if suspect_idx in suppressed:
                    continue
                if str(suspect.detection_source).startswith("white_blob"):
                    continue  # 亮斑通道豁免：真白鼠贴墙活动
                if str(real.detection_source).startswith("white_blob"):
                    continue
                suspect_profile = boundary_profiles[suspect_idx]
                real_profile = boundary_profiles[real_idx]
                closer_to_same_wall = bool(
                    np.any(
                        (suspect_profile + 1.0 < real_profile)
                        & (suspect_profile <= boundary_band_px)
                    )
                )
                if not closer_to_same_wall:
                    continue  # 必须至少在同一段墙面上比真实候选更贴墙
                if float(np.min(suspect_profile)) > boundary_band_px:
                    continue
                if float(suspect.box_conf) > float(real.box_conf) + 0.05:
                    continue
                sp = float(suspect.pose_quality)
                rp = float(real.pose_quality)
                if sp <= 0.12:
                    if rp <= 0.12:
                        continue  # 双方均无姿态时宁可保留，防误删贴墙真鼠
                    suppressed.add(suspect_idx)
                    continue
                # ---- v1.12.0：带姿态嫌疑者（被误画骨架的影子/水痕）----
                # 实测影子的骨架置信度与真鼠相当（v1.11.3的"姿态更弱"条件放行），
                # 但对比度物证稳定可分：影子暗部/背景0.72 vs 真鼠0.29（差0.43）。
                # 对比度升级为主证，不再要求姿态更弱。
                if frame_gray is None or rp <= 0.12:
                    continue
                fs = _box_contrast_features(frame_gray, suspect.bbox_xyxy)
                fr = _box_contrast_features(frame_gray, real.bbox_xyxy)
                if fs is None or fr is None:
                    continue
                s_ratio = fs[0] / max(fs[1], 1.0)
                r_ratio = fr[0] / max(fr[1], 1.0)
                if s_ratio <= r_ratio + contrast_margin:
                    continue  # 与真鼠对比度相当→不像影子，保留
                if fs[2] > fs[1] * bright_object_factor:
                    continue  # 亮目标（真白鼠）豁免：影子不会比背景亮
                suppressed.add(suspect_idx)
    return [d for k, d in enumerate(dets) if k not in suppressed]


# ----------------------------- 时序后处理 -----------------------------


def runs(mask: np.ndarray) -> Iterable[Tuple[bool, int, int]]:
    if len(mask) == 0:
        return
    start = 0
    value = bool(mask[0])
    for idx in range(1, len(mask)):
        current = bool(mask[idx])
        if current != value:
            yield value, start, idx - 1
            start = idx
            value = current
    yield value, start, len(mask) - 1


def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    output = mask.astype(bool, copy=True)
    if max_gap <= 0:
        return output
    all_runs = list(runs(output))
    for idx, (value, start, end) in enumerate(all_runs):
        length = end - start + 1
        if value or length > max_gap or idx == 0 or idx == len(all_runs) - 1:
            continue
        if all_runs[idx - 1][0] and all_runs[idx + 1][0]:
            output[start : end + 1] = True
    return output


def remove_short_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    output = mask.astype(bool, copy=True)
    if min_length <= 1:
        return output
    for value, start, end in list(runs(output)):
        if value and end - start + 1 < min_length:
            output[start : end + 1] = False
    return output


def temporal_filter(mask: np.ndarray, min_length: int, max_gap: int) -> np.ndarray:
    output = fill_short_false_gaps(mask, max_gap=max_gap)
    output = remove_short_true_runs(output, min_length=min_length)
    return output


def postprocess_frame_labels(frame_df: pd.DataFrame, fps: float, config: Mapping[str, Any]) -> pd.DataFrame:
    df = frame_df.copy()
    raw_chase = df["raw_chase"].fillna(False).astype(bool).to_numpy()
    raw_attack = df["raw_attack"].fillna(False).astype(bool).to_numpy()

    chase_min = max(int(math.ceil(float(config["chase"]["min_duration_seconds"]) * fps)), 1)
    chase_gap = max(int(round(float(config["chase"]["fill_gap_seconds"]) * fps)), 0)
    attack_min = max(int(math.ceil(float(config["attack"]["min_duration_seconds"]) * fps)), 1)
    attack_gap = max(int(round(float(config["attack"]["fill_gap_seconds"]) * fps)), 0)

    final_chase = temporal_filter(raw_chase, chase_min, chase_gap)
    final_attack = temporal_filter(raw_attack, attack_min, attack_gap)

    # 长时间无有效双鼠检测时，不允许跨越缺失区段合并事件。
    valid_pair = df["valid_pair"].fillna(False).astype(bool).to_numpy()
    final_chase &= valid_pair
    final_attack &= valid_pair

    final_label = final_chase.astype(np.int8) + 2 * final_attack.astype(np.int8)

    df["final_chase"] = final_chase
    df["final_attack"] = final_attack
    df["final_label_id"] = final_label
    df["final_label_en"] = [LABELS[int(v)][0] for v in final_label]
    df["final_label_zh"] = [LABELS[int(v)][1] for v in final_label]
    return df


# ----------------------------- 事件生成 -----------------------------


def positive_events_from_frames(
    df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    labels = df["final_label_id"].astype(int).to_numpy()
    events: List[Dict[str, Any]] = []
    event_index = 1

    start = 0
    while start < len(labels):
        label = int(labels[start])
        end = start
        while end + 1 < len(labels) and int(labels[end + 1]) == label:
            end += 1

        if label != 0:
            segment = df.iloc[start : end + 1]
            actor_id = int(mode_or_default(segment["selected_actor_id"].tolist(), -1))
            target_id = int(mode_or_default(segment["selected_target_id"].tolist(), -1))
            actor_path = float(segment["selected_actor_speed_cm_s"].fillna(0).sum() / fps)
            target_path = float(segment["selected_target_speed_cm_s"].fillna(0).sum() / fps)
            distance_fraction = float(
                (segment["center_distance_cm"] <= float(config["chase"]["max_distance_cm"])).mean()
            )
            mean_corr = float(segment["trajectory_correlation"].fillna(0).mean())
            strict_chase = bool(
                label in (1, 3)
                and actor_path >= float(config["chase"]["strict_min_path_cm"])
                and target_path >= float(config["chase"]["strict_min_path_cm"])
                and distance_fraction >= 0.8
                and mean_corr >= float(config["chase"]["trajectory_correlation_min"])
            )

            events.append(
                {
                    "event_id": f"E{event_index:05d}",
                    "label_id": label,
                    "label_en": LABELS[label][0],
                    "label_zh": LABELS[label][1],
                    "actor_id": actor_id,
                    "target_id": target_id,
                    "start_frame": int(segment["frame"].iloc[0]),
                    "end_frame": int(segment["frame"].iloc[-1]),
                    "start_time_s": float(segment["time_s"].iloc[0]),
                    "end_time_s": float(segment["time_s"].iloc[-1]),
                    "duration_s": float((end - start + 1) / fps),
                    "max_chase_score": int(segment["selected_chase_score"].fillna(0).max()),
                    "max_attack_evidence": int(segment["selected_attack_evidence"].fillna(0).max()),
                    "min_center_distance_cm": float(segment["center_distance_cm"].min()),
                    "min_nose_body_distance_cm": float(segment["selected_nose_body_distance_cm"].min()),
                    "mean_trajectory_correlation": mean_corr,
                    "actor_path_cm": actor_path,
                    "target_path_cm": target_path,
                    "actor_max_speed_cm_s": float(segment["selected_actor_speed_cm_s"].max()),
                    "target_max_speed_cm_s": float(segment["selected_target_speed_cm_s"].max()),
                    "target_max_turn_angle_deg": float(segment["selected_target_turn_angle_deg"].max()),
                    "strict_chase": strict_chase,
                    "uncertain_attack": bool(
                        label in (2, 3)
                        and segment["selected_attack_evidence"].fillna(0).max()
                        == int(config["attack"]["min_dynamic_evidence"])
                    ),
                    "is_hard_negative": False,
                }
            )
            event_index += 1
        start = end + 1

    return events


def hard_negative_events(
    df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
    start_index: int,
) -> List[Dict[str, Any]]:
    clip_cfg = config["clips"]
    if not bool(clip_cfg.get("extract_hard_negatives", True)):
        return []

    labels = df["final_label_id"].astype(int).to_numpy()
    valid = df["valid_pair"].fillna(False).astype(bool).to_numpy()
    close = df["center_distance_cm"].fillna(np.inf).to_numpy() < float(
        clip_cfg["hard_negative_close_distance_cm"]
    )
    fast = np.maximum(
        df["mouse_a_speed_cm_s"].fillna(0).to_numpy(),
        df["mouse_b_speed_cm_s"].fillna(0).to_numpy(),
    ) > float(clip_cfg["hard_negative_min_speed_cm_s"])

    positive = labels != 0
    exclusion = positive.copy()
    pad_frames = int(round(
        max(float(clip_cfg["pre_padding_seconds"]), float(clip_cfg["post_padding_seconds"])) * fps
    ))
    positive_indices = np.flatnonzero(positive)
    for idx in positive_indices:
        lo = max(0, idx - pad_frames)
        hi = min(len(exclusion), idx + pad_frames + 1)
        exclusion[lo:hi] = True

    candidate = valid & (labels == 0) & (close | fast) & (~exclusion)
    candidate_frames = np.flatnonzero(candidate)

    clip_frames = max(int(round(float(clip_cfg["hard_negative_clip_seconds"]) * fps)), 1)
    half = clip_frames // 2
    min_interval = max(int(round(float(clip_cfg["hard_negative_min_interval_seconds"]) * fps)), 1)
    max_clips = int(clip_cfg.get("max_hard_negative_clips", 200))

    selected: List[int] = []
    last = -10**9
    for idx in candidate_frames:
        if idx - last >= min_interval:
            selected.append(int(idx))
            last = int(idx)
        if len(selected) >= max_clips:
            break

    events: List[Dict[str, Any]] = []
    for offset, center_idx in enumerate(selected):
        start_idx = max(0, center_idx - half)
        end_idx = min(len(df) - 1, start_idx + clip_frames - 1)
        segment = df.iloc[start_idx : end_idx + 1]
        events.append(
            {
                "event_id": f"E{start_index + offset:05d}",
                "label_id": 0,
                "label_en": LABELS[0][0],
                "label_zh": LABELS[0][1],
                "actor_id": int(mode_or_default(segment["selected_actor_id"].tolist(), -1)),
                "target_id": int(mode_or_default(segment["selected_target_id"].tolist(), -1)),
                "start_frame": int(segment["frame"].iloc[0]),
                "end_frame": int(segment["frame"].iloc[-1]),
                "start_time_s": float(segment["time_s"].iloc[0]),
                "end_time_s": float(segment["time_s"].iloc[-1]),
                "duration_s": float(len(segment) / fps),
                "max_chase_score": int(segment["selected_chase_score"].fillna(0).max()),
                "max_attack_evidence": int(segment["selected_attack_evidence"].fillna(0).max()),
                "min_center_distance_cm": float(segment["center_distance_cm"].min()),
                "min_nose_body_distance_cm": float(segment["selected_nose_body_distance_cm"].min()),
                "mean_trajectory_correlation": float(segment["trajectory_correlation"].fillna(0).mean()),
                "actor_path_cm": float(segment["selected_actor_speed_cm_s"].fillna(0).sum() / fps),
                "target_path_cm": float(segment["selected_target_speed_cm_s"].fillna(0).sum() / fps),
                "actor_max_speed_cm_s": float(segment["selected_actor_speed_cm_s"].fillna(0).max()),
                "target_max_speed_cm_s": float(segment["selected_target_speed_cm_s"].fillna(0).max()),
                "target_max_turn_angle_deg": float(segment["selected_target_turn_angle_deg"].fillna(0).max()),
                "strict_chase": False,
                "uncertain_attack": False,
                "is_hard_negative": True,
            }
        )
    return events


def add_clip_boundaries(
    events: List[Dict[str, Any]],
    total_frames: int,
    fps: float,
    config: Mapping[str, Any],
) -> None:
    clips_cfg = config["clips"]
    fixed_seconds = float(clips_cfg.get("fixed_clip_seconds", 0.0))
    fixed_frames = max(int(round(fixed_seconds * fps)), 1) if fixed_seconds > 0 else 0
    pre = int(round(float(clips_cfg.get("pre_padding_seconds", 1.5)) * fps))
    post = int(round(float(clips_cfg.get("post_padding_seconds", 1.5)) * fps))
    for event in events:
        event_start = max(0, int(event["start_frame"]))
        event_end = min(total_frames - 1, int(event["end_frame"]))
        if fixed_frames > 0:
            window_frames = min(fixed_frames, total_frames)
            center = int(round((event_start + event_end) / 2.0))
            clip_start = max(0, min(center - window_frames // 2, total_frames - window_frames))
            clip_end = clip_start + window_frames - 1
        elif event.get("is_hard_negative", False):
            clip_start, clip_end = event_start, event_end
        else:
            clip_start = max(0, event_start - pre)
            clip_end = min(total_frames - 1, event_end + post)
        event["clip_start_frame"] = int(clip_start)
        event["clip_end_frame"] = int(clip_end)
        event["clip_start_time_s"] = clip_start / fps
        event["clip_end_time_s"] = clip_end / fps
        event["clip_duration_s"] = (clip_end - clip_start + 1) / fps
        event.setdefault("clip_selected", True)
        event.setdefault("clip_skip_reason", "")
        event["clip_path"] = ""
        event["review_status"] = "待复核"


# ----------------------------- 视频输出 -----------------------------


def extract_event_clips(
    video_path: Path,
    events: List[Dict[str, Any]],
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
    *,
    filename_stem: Optional[str] = None,
) -> None:
    if not events:
        logging.info("没有可提取的事件片段。")
        return

    clips_root = ensure_dir(output_dir / "事件片段")
    for folder in CLIP_DIRS.values():
        ensure_dir(clips_root / folder)

    start_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    source_stem = str(filename_stem or video_path.stem)
    for event in events:
        if not bool(event.get("clip_selected", True)):
            continue
        label = int(event["label_id"])
        filename = sanitize_filename(
            f"{source_stem}_{event['event_id']}_A{event['actor_id']}_B{event['target_id']}_"
            f"{LABELS[label][1]}_{event['start_time_s']:.2f}s_{event['end_time_s']:.2f}s.mp4"
        )
        path = clips_root / CLIP_DIRS[label] / filename
        event["clip_path"] = str(path.resolve())
        event["_writer_path"] = path
        start_map[int(event["clip_start_frame"])].append(event)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法再次打开视频用于片段提取：{video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    active: Dict[str, Tuple[cv2.VideoWriter, Dict[str, Any]]] = {}
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total, desc="提取行为片段", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for event in start_map.get(frame_idx, []):
                writer = cv2.VideoWriter(
                    str(event["_writer_path"]), fourcc, fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建片段视频：{event['_writer_path']}")
                active[event["event_id"]] = (writer, event)

            finished: List[str] = []
            for event_id, (writer, event) in active.items():
                if frame_idx <= int(event["clip_end_frame"]):
                    writer.write(frame)
                if frame_idx >= int(event["clip_end_frame"]):
                    writer.release()
                    finished.append(event_id)
            for event_id in finished:
                active.pop(event_id, None)

            frame_idx += 1
            pbar.update(1)

    for writer, _ in active.values():
        writer.release()
    cap.release()

    for event in events:
        event.pop("_writer_path", None)


def draw_observation(frame: np.ndarray, obs: MouseObservation, color: Tuple[int, int, int]) -> None:
    points = obs.keypoints_px
    for a, b in SKELETON_EDGES:
        pa, pb = points[a], points[b]
        if finite_point(pa) and finite_point(pb):
            cv2.line(frame, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), color, 2)
    for idx, point in enumerate(points):
        if finite_point(point):
            radius = 4 if idx == KP["nose"] else 3
            cv2.circle(frame, tuple(np.round(point).astype(int)), radius, color, -1)
    valid = points[np.all(np.isfinite(points), axis=1)]
    if len(valid):
        anchor = tuple(np.round(valid[0]).astype(int))
        cv2.putText(
            frame,
            f"Mouse {obs.logical_id}",
            (anchor[0] + 5, anchor[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def save_annotated_video(
    video_path: Path,
    output_path: Path,
    frame_df: pd.DataFrame,
    visual_records: Mapping[int, Sequence[MouseObservation]],
    fps: float,
    width: int,
    height: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频用于绘制核查结果：{video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建核查视频：{output_path}")

    rows = frame_df.set_index("frame")
    colors = [(0, 220, 0), (0, 140, 255), (255, 100, 0), (220, 0, 220)]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    with tqdm(total=total, desc="生成标注核查视频", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for obs in visual_records.get(frame_idx, []):
                draw_observation(frame, obs, colors[obs.logical_id % len(colors)])

            if frame_idx in rows.index:
                row = rows.loc[frame_idx]
                label_id = int(row["final_label_id"])
                text = LABELS[label_id][0]
                actor = int(row["selected_actor_id"]) if pd.notna(row["selected_actor_id"]) else -1
                target = int(row["selected_target_id"]) if pd.notna(row["selected_target_id"]) else -1
                cv2.rectangle(frame, (10, 10), (720, 82), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Label {label_id}: {text} | actor={actor} target={target}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"chase={int(row['selected_chase_score'])} attack_ev={int(row['selected_attack_evidence'])} "
                    f"distance={row['center_distance_cm']:.1f}cm",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA,
                )
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    writer.release()
    cap.release()


# ----------------------------- 主分析流程 -----------------------------


def empty_frame_record(frame_idx: int, fps: float, cm_per_pixel: float) -> Dict[str, Any]:
    return {
        "frame": frame_idx,
        "time_s": frame_idx / fps,
        "valid_pair": False,
        "cm_per_pixel": cm_per_pixel,
        "mouse_a_id": np.nan,
        "mouse_b_id": np.nan,
        "mouse_a_raw_track_id": np.nan,
        "mouse_b_raw_track_id": np.nan,
        "mouse_a_speed_cm_s": 0.0,
        "mouse_b_speed_cm_s": 0.0,
        "center_distance_cm": np.nan,
        "head_distance_cm": np.nan,
        "trajectory_correlation": 0.0,
        "direction_similarity": 0.0,
        "selected_actor_id": np.nan,
        "selected_target_id": np.nan,
        "selected_chase_score": 0,
        "selected_attack_evidence": 0,
        "selected_actor_speed_cm_s": 0.0,
        "selected_target_speed_cm_s": 0.0,
        "selected_nose_body_distance_cm": np.nan,
        "selected_target_turn_angle_deg": 0.0,
        "a_to_b_chase_score": 0,
        "b_to_a_chase_score": 0,
        "a_to_b_attack_evidence": 0,
        "b_to_a_attack_evidence": 0,
        "a_to_b_chase": False,
        "b_to_a_chase": False,
        "a_to_b_attack": False,
        "b_to_a_attack": False,
        "contact": False,
        "repeated_contact_count": 0,
        "raw_chase": False,
        "raw_attack": False,
        "raw_label_id": 0,
        "raw_label_en": LABELS[0][0],
        "raw_label_zh": LABELS[0][1],
    }


def choose_direction(features_ab: PairFeatures, features_ba: PairFeatures) -> PairFeatures:
    # 先从真正触发候选的方向中选择，避免另一方向的零散分数抢占发起者。
    if features_ab.attack_candidate != features_ba.attack_candidate:
        return features_ab if features_ab.attack_candidate else features_ba
    if features_ab.attack_candidate and features_ba.attack_candidate:
        if features_ab.attack_dynamic_evidence != features_ba.attack_dynamic_evidence:
            return features_ab if features_ab.attack_dynamic_evidence > features_ba.attack_dynamic_evidence else features_ba

    if features_ab.chase_candidate != features_ba.chase_candidate:
        return features_ab if features_ab.chase_candidate else features_ba
    if features_ab.chase_candidate and features_ba.chase_candidate:
        if features_ab.chase_score != features_ba.chase_score:
            return features_ab if features_ab.chase_score > features_ba.chase_score else features_ba

    score_ab = features_ab.attack_dynamic_evidence + features_ab.chase_score
    score_ba = features_ba.attack_dynamic_evidence + features_ba.chase_score
    if score_ab != score_ba:
        return features_ab if score_ab > score_ba else features_ba
    return features_ab if features_ab.actor_speed_cm_s >= features_ba.actor_speed_cm_s else features_ba


def process_video(
    video_path: Path,
    model_path: Path,
    output_root: Path,
    config: MutableMapping[str, Any],
    save_clips: bool = True,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装ultralytics，请先执行：pip install -r requirements.txt") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"视频元数据异常：fps={fps}, size={width}x{height}")

    output_dir = ensure_dir(output_root / video_path.stem)
    logging.info("处理视频：%s", video_path)
    logging.info("视频信息：%d帧，%.3f FPS，%dx%d", total_frames, fps, width, height)

    model_cfg = config["model"]
    max_mice = int(model_cfg.get("max_mice", 20))

    model = YOLO(str(model_path))
    identity = StableIdentityAssigner(config["identity"], max_mice=max_mice)
    smoother = KeypointSmoother(
        alpha=float(config["keypoints"]["smoothing_alpha"]),
        # v1.11.4：平滑阈值独立于渲染阈值（0.1）。旧值0.2会把模型以0.10~0.20
        # 置信度给出的nose/tail端点直接抹除——骨架缺端点呈"蝴蝶结"的直接原因。
        min_conf=float(config["keypoints"].get(
            "smoothing_min_confidence", config["keypoints"].get("min_confidence", 0.1)
        )),
        max_missing=int(config["keypoints"]["interpolation_max_frames"]),
        interp_decay=float(config["keypoints"].get("interpolation_conf_decay", 0.85)),
    )
    scale_estimator = ScaleEstimator(config["scale"])
    history_seconds = max(float(config["features"]["history_seconds"]), 1.0)
    history = ObservationHistory(max_frames=max(int(round(fps * history_seconds)) + 5, 10))
    feature_computer = PairFeatureComputer(fps, config)
    contact_tracker = PairContactTracker(
        fps=fps,
        window_seconds=float(config["attack"]["repeated_contact_window_seconds"]),
    )

    frame_records: List[Dict[str, Any]] = []
    visual_records: Dict[int, List[MouseObservation]] = {}
    frame_idx = 0

    tracker_path = str(model_cfg.get("tracker", "botsort.yaml"))
    device = model_cfg.get("device", 0)

    with tqdm(total=total_frames, desc=f"姿态推理与行为特征 {video_path.name}", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_path,
                imgsz=int(model_cfg.get("imgsz", 960)),
                conf=float(model_cfg.get("conf", 0.25)),
                iou=float(model_cfg.get("iou", 0.5)),
                device=device,
                verbose=False,
            )
            result = results[0] if results else None
            detections = parse_yolo_result(result, len(KEYPOINT_NAMES), max_mice)
            cm_per_pixel = scale_estimator.update(detections)
            assigned = identity.assign(detections, frame_idx)

            observations: List[MouseObservation] = []
            for logical_id, detection in assigned:
                smoothed_px, effective_conf = smoother.update(
                    logical_id, detection.keypoints_px, detection.keypoint_conf
                )
                previous = history.previous(logical_id)
                obs = build_observation(
                    frame=frame_idx,
                    fps=fps,
                    logical_id=logical_id,
                    detection=detection,
                    smoothed_keypoints_px=smoothed_px,
                    effective_conf=effective_conf,
                    cm_per_pixel=cm_per_pixel,
                    previous=previous,
                )
                if finite_point(obs.center_cm):
                    observations.append(obs)

            # 先把当前观测加入历史，使轨迹相关性窗口包含当前帧。
            for obs in observations:
                history.add(obs)
            visual_records[frame_idx] = observations

            record = empty_frame_record(frame_idx, fps, cm_per_pixel)
            if len(observations) >= 2:
                observations = sorted(observations, key=lambda o: o.logical_id)[:2]
                a, b = observations
                contact_distances = np.array([
                    min_point_distance(a.keypoints_cm[KP["nose"]], b.keypoints_cm),
                    min_point_distance(b.keypoints_cm[KP["nose"]], a.keypoints_cm),
                ], dtype=np.float64)
                finite_contact_distances = contact_distances[np.isfinite(contact_distances)]
                symmetric_contact = bool(
                    len(finite_contact_distances) > 0
                    and float(np.min(finite_contact_distances))
                    < float(config["attack"]["contact_distance_cm"])
                )
                repeated_count = contact_tracker.update(
                    a.logical_id, b.logical_id, frame_idx, symmetric_contact
                )
                features_ab = feature_computer.compute(a, b, history, repeated_count)
                features_ba = feature_computer.compute(b, a, history, repeated_count)
                selected = choose_direction(features_ab, features_ba)

                raw_chase = bool(features_ab.chase_candidate or features_ba.chase_candidate)
                raw_attack = bool(features_ab.attack_candidate or features_ba.attack_candidate)
                raw_label = int(raw_chase) + 2 * int(raw_attack)

                record.update(
                    {
                        "valid_pair": True,
                        "mouse_a_id": a.logical_id,
                        "mouse_b_id": b.logical_id,
                        "mouse_a_raw_track_id": a.raw_track_id if a.raw_track_id is not None else np.nan,
                        "mouse_b_raw_track_id": b.raw_track_id if b.raw_track_id is not None else np.nan,
                        "mouse_a_speed_cm_s": a.speed_cm_s,
                        "mouse_b_speed_cm_s": b.speed_cm_s,
                        "center_distance_cm": point_distance(a.center_cm, b.center_cm),
                        "head_distance_cm": point_distance(a.head_cm, b.head_cm),
                        "trajectory_correlation": selected.trajectory_correlation,
                        "direction_similarity": selected.direction_similarity,
                        "selected_actor_id": selected.actor_id,
                        "selected_target_id": selected.target_id,
                        "selected_chase_score": selected.chase_score,
                        "selected_attack_evidence": selected.attack_dynamic_evidence,
                        "selected_actor_speed_cm_s": selected.actor_speed_cm_s,
                        "selected_target_speed_cm_s": selected.target_speed_cm_s,
                        "selected_nose_body_distance_cm": selected.actor_nose_to_target_body_cm,
                        "selected_target_turn_angle_deg": selected.target_turn_angle_deg,
                        "a_to_b_chase_score": features_ab.chase_score,
                        "b_to_a_chase_score": features_ba.chase_score,
                        "a_to_b_attack_evidence": features_ab.attack_dynamic_evidence,
                        "b_to_a_attack_evidence": features_ba.attack_dynamic_evidence,
                        "a_to_b_chase": features_ab.chase_candidate,
                        "b_to_a_chase": features_ba.chase_candidate,
                        "a_to_b_attack": features_ab.attack_candidate,
                        "b_to_a_attack": features_ba.attack_candidate,
                        "contact": symmetric_contact,
                        "repeated_contact_count": repeated_count,
                        "raw_chase": raw_chase,
                        "raw_attack": raw_attack,
                        "raw_label_id": raw_label,
                        "raw_label_en": LABELS[raw_label][0],
                        "raw_label_zh": LABELS[raw_label][1],
                    }
                )

            frame_records.append(record)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    if not frame_records:
        raise RuntimeError("视频没有读取到任何有效帧。")

    frame_df = pd.DataFrame(frame_records)
    frame_df = postprocess_frame_labels(frame_df, fps, config)

    positive_events = positive_events_from_frames(frame_df, fps, config)
    negative_events = hard_negative_events(
        frame_df, fps, config, start_index=len(positive_events) + 1
    )
    events = positive_events + negative_events
    add_clip_boundaries(events, len(frame_df), fps, config)

    if save_clips:
        extract_event_clips(video_path, events, output_dir, fps, width, height)

    if bool(config["output"].get("save_annotated_video", True)):
        save_annotated_video(
            video_path,
            output_dir / "行为标注视频.mp4",
            frame_df,
            visual_records,
            fps,
            width,
            height,
        )

    if bool(config["output"].get("save_frame_csv", True)):
        frame_df.to_csv(output_dir / "逐帧行为标签.csv", index=False, encoding="utf-8-sig")

    event_df = pd.DataFrame(events)
    if bool(config["output"].get("save_event_csv", True)):
        event_df.to_csv(output_dir / "行为事件表.csv", index=False, encoding="utf-8-sig")

    identity_df = pd.DataFrame([asdict(x) for x in identity.debug_records])
    identity_df.to_csv(output_dir / "身份分配调试记录.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "运行元数据.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path.resolve()),
                "model": str(model_path.resolve()),
                "fps": fps,
                "total_frames": len(frame_df),
                "width": width,
                "height": height,
                "keypoints": KEYPOINT_NAMES,
                "event_counts": {
                    LABELS[label][1]: int((event_df["label_id"] == label).sum()) if not event_df.empty else 0
                    for label in LABELS
                },
                "config": to_builtin(config),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("完成：%s", output_dir)
    if not event_df.empty:
        summary = event_df.groupby(["label_id", "label_zh"]).size().reset_index(name="事件数")
        logging.info("事件统计：\n%s", summary.to_string(index=False))
    return output_dir


def discover_videos(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式：{path.suffix}")
        return [path]
    if path.is_dir():
        videos = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise FileNotFoundError(f"目录中没有找到视频：{path}")
        return videos
    raise FileNotFoundError(f"输入路径不存在：{path}")


def apply_cli_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    if args.device is not None:
        config["model"]["device"] = args.device
    if args.tracker is not None:
        config["model"]["tracker"] = args.tracker
    if args.cm_per_pixel is not None:
        if args.cm_per_pixel <= 0:
            raise ValueError("--cm-per-pixel必须大于0。")
        config["scale"]["mode"] = "fixed"
        config["scale"]["cm_per_pixel"] = float(args.cm_per_pixel)
    if args.no_annotated:
        config["output"]["save_annotated_video"] = False
    if args.no_hard_negatives:
        config["clips"]["extract_hard_negatives"] = False


def validate_config(config: Mapping[str, Any]) -> None:
    required = ["model", "keypoints", "identity", "scale", "features", "chase", "attack", "clips", "output"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段：{missing}")
    names = list(config["keypoints"].get("names", []))
    if names != KEYPOINT_NAMES:
        raise ValueError(
            "配置中的关键点顺序必须严格为：" + ", ".join(KEYPOINT_NAMES)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用YOLO26 Pose关键点对双鼠追逐/攻击进行四分类初筛并提取视频片段。"
    )
    parser.add_argument("--model", required=True, type=Path, help="自定义7关键点YOLO Pose权重，例如best.pt")
    parser.add_argument("--video", required=True, type=Path, help="单个视频或包含视频的目录")
    parser.add_argument("--output", required=True, type=Path, help="输出根目录")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("behavior_config.yaml"),
        help="YAML配置文件",
    )
    parser.add_argument("--device", default=None, help="覆盖配置中的设备，例如0、1或cpu")
    parser.add_argument("--tracker", default=None, help="覆盖追踪器，例如botsort.yaml或bytetrack.yaml")
    parser.add_argument("--cm-per-pixel", type=float, default=None, help="已标定的厘米/像素；提供后覆盖身体长度估算")
    parser.add_argument("--no-clips", action="store_true", help="不提取分类片段，仅输出CSV")
    parser.add_argument("--no-annotated", action="store_true", help="不生成带关键点核查视频")
    parser.add_argument("--no-hard-negatives", action="store_true", help="不提取困难负样本")
    parser.add_argument("--verbose", action="store_true", help="显示调试日志")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        config = load_yaml(args.config)
        validate_config(config)
        apply_cli_overrides(config, args)

        model_path = args.model.expanduser().resolve()
        video_input = args.video.expanduser().resolve()
        output_root = ensure_dir(args.output.expanduser().resolve())
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        videos = discover_videos(video_input)
        logging.info("共发现%d个视频。", len(videos))
        completed = []
        failed = []
        for video in videos:
            try:
                completed.append(
                    process_video(
                        video_path=video,
                        model_path=model_path,
                        output_root=output_root,
                        config=config,
                        save_clips=not args.no_clips,
                    )
                )
            except Exception as exc:  # 单个视频失败时继续批处理其他视频
                logging.exception("视频处理失败：%s", video)
                failed.append({"video": str(video), "error": str(exc)})

        with (output_root / "批处理汇总.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"completed": [str(p) for p in completed], "failed": failed},
                f,
                ensure_ascii=False,
                indent=2,
            )

        if failed:
            logging.warning("完成%d个，失败%d个。详情见批处理汇总.json。", len(completed), len(failed))
            return 2
        logging.info("全部完成，共%d个视频。", len(completed))
        return 0
    except Exception as exc:
        logging.exception("程序终止：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

# ============================================================================
# v1.19: 固定鼠数硬槽 + Dummy空匹配 + 鲁棒体长 + 新轨迹连续确认
# ============================================================================
_OriginalKeypointMotionIdentityAssigner = KeypointMotionIdentityAssigner


class _KeypointMotionIdentityAssignerV120Base(_OriginalKeypointMotionIdentityAssigner):
    """v1.20鲁棒运动身份分配基础类（供v1.21自适应版本继承）。

    在旧双鼠逐关键点预测机制之上补齐四个多鼠场景必需约束：
    1. expected_mice_count 是真正的硬容量，固定模式绝不产生第 N+1 个ID；
    2. 匈牙利矩阵加入每条轨迹自己的 Dummy 空匹配，检测不可信时允许本帧断观测；
    3. 身份尺度使用 max(鼻尾长度, bbox长边×比例)，防遮挡关键点收缩；
    4. 自动鼠数模式中的新轨迹必须连续出现若干帧才正式建ID。
    """

    def __init__(self, config: Mapping[str, Any], max_mice: int = 20) -> None:
        super().__init__(config, max_mice=max_mice)
        cfg = dict(config.get("keypoint_motion", {}))
        self.expected_mice_count = max(0, int(cfg.get("expected_mice_count", 0)))
        self.identity_capacity = (
            min(self.max_mice, self.expected_mice_count)
            if self.expected_mice_count > 0 else self.max_mice
        )
        self.unmatched_cost = float(cfg.get("unmatched_cost", 1.08))
        self.unmatched_cost_growth = float(cfg.get("unmatched_cost_growth", 0.025))
        self.robust_bbox_body_scale = float(cfg.get("robust_bbox_body_scale", 0.75))
        self.hard_size_ratio_min = float(cfg.get("hard_size_ratio_min", 0.38))
        self.hard_size_ratio_max = float(cfg.get("hard_size_ratio_max", 2.65))
        self.hard_gate_base_bl = float(cfg.get("hard_gate_base_body_lengths", 1.90))
        self.hard_gate_missing_growth = float(cfg.get("hard_gate_missing_growth", 0.32))
        self.hard_gate_max_bl = float(cfg.get("hard_gate_max_body_lengths", 3.30))
        self.confidence_cost_weight = float(cfg.get("confidence_cost_weight", 0.02))

        self.new_track_confirm_frames = max(1, int(cfg.get("new_track_confirm_frames", 3)))
        self.pending_match_distance_bl = float(cfg.get("pending_match_distance_body_lengths", 0.80))
        self.pending_ttl_frames = max(1, int(cfg.get("pending_ttl_frames", 10)))
        self.max_new_tracks_per_frame = max(1, int(cfg.get("max_new_tracks_per_frame", 2)))
        self.pending_candidates: Dict[int, Dict[str, Any]] = {}
        self.next_pending_id = 0
        performance_cfg = dict(config.get("performance", {}))
        self.cost_matrix_backend = str(
            performance_cfg.get("identity_cost_backend", "auto")
        ).strip().lower()
        if self.cost_matrix_backend not in {"python", "numpy", "auto", "cpp"}:
            self.cost_matrix_backend = "auto"
        self.cost_matrix_tie_fallback_epsilon = max(
            0.0, float(performance_cfg.get("identity_cost_tie_fallback_epsilon", 1.0e-10))
        )
        self.identity_cpp_threads = max(
            1,
            int(performance_cfg.get("identity_cpp_threads", 1)),
        )
        self.identity_cpp_selftest = bool(
            performance_cfg.get("identity_cpp_selftest", True)
        )
        self.identity_cpp_fallback_on_tie = bool(
            performance_cfg.get("identity_cpp_fallback_on_tie", True)
        )
        self._cpp_module = None
        self.cpp_backend_status = "not_requested"
        self.cpp_backend_failure_reason = ""
        self.last_cost_backend_used = self.cost_matrix_backend
        self.last_cost_build_seconds = 0.0
        self.last_assignment_seconds = 0.0

        # v1.42.1 sparse identity cascade.  The candidate gate is exactly the
        # existing v1.20 hard size + hard center gate, so rejected cells are
        # guaranteed to be INF_COST in the scalar baseline.
        cascade_cfg = dict(performance_cfg.get("identity_cascade", {}))
        self.identity_cascade_enabled = bool(cascade_cfg.get("enabled", True))
        self.identity_cascade_min_cells = max(1, int(cascade_cfg.get("min_cells", 64)))
        self.identity_cascade_sparse_density = float(
            np.clip(cascade_cfg.get("sparse_density_threshold", 0.35), 0.0, 1.0)
        )
        self.last_fast_gate_candidate_count = 0
        self.last_fast_gate_total_count = 0
        self.last_fast_gate_density = 1.0
        self.last_base_cost_mode = "uninitialized"

    def _allocate_id(self) -> Optional[int]:
        if self.expected_mice_count > 0:
            for lid in range(self.identity_capacity):
                if lid not in self.tracks:
                    return int(lid)
            return None
        return super()._allocate_id()

    def _expire(self, frame: int) -> None:
        # 固定鼠数时身份槽永久保留。长遮挡只进入lost，不删除、不换号。
        if self.expected_mice_count > 0:
            for lid, track in self.tracks.items():
                missing = max(int(frame - track.last_frame), int(self.kpt_missing.get(lid, 0)))
                if missing > self.max_missing_frames:
                    track.state = "lost"
            return
        super()._expire(frame)

    def _robust_det_body(self, det: Detection) -> float:
        box = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)
        bbox_len = 0.0
        if box.size >= 4 and np.all(np.isfinite(box[:4])):
            bbox_len = self.robust_bbox_body_scale * max(
                abs(float(box[2] - box[0])), abs(float(box[3] - box[1]))
            )
        kpt_len = float(det.body_length_px) if np.isfinite(det.body_length_px) else 0.0
        return max(kpt_len, bbox_len, 8.0)

    def _robust_track_body(self, track: IdentityTrack) -> float:
        bbox_len = 0.0
        if track.last_bbox_xyxy is not None:
            box = np.asarray(track.last_bbox_xyxy, dtype=np.float64).reshape(-1)
            if box.size >= 4 and np.all(np.isfinite(box[:4])):
                bbox_len = self.robust_bbox_body_scale * max(
                    abs(float(box[2] - box[0])), abs(float(box[3] - box[1]))
                )
        tracked = float(track.body_length_px) if np.isfinite(track.body_length_px) else 0.0
        return max(tracked, bbox_len, 8.0)

    def _hard_gate_bl(self, track: IdentityTrack, frame: int) -> float:
        missing = max(int(frame - track.last_frame), int(self.kpt_missing.get(track.logical_id, 0)))
        soft = self._adaptive_gate_bl(track, frame)
        hard = self.hard_gate_base_bl + self.hard_gate_missing_growth * min(
            missing, self.prediction_max_frames
        )
        return float(np.clip(max(soft, hard), self.hard_gate_base_bl, self.hard_gate_max_bl))

    def _cost(self, lid: int, track: IdentityTrack, det: Detection, frame: int) -> float:
        track_body = self._robust_track_body(track)
        det_body = self._robust_det_body(det)
        ratio = det_body / max(track_body, 1e-6)
        if ratio < self.hard_size_ratio_min or ratio > self.hard_size_ratio_max:
            return self.INF_COST
        body = max(float(np.nanmedian([track_body, det_body])), 8.0)

        pred_center = self._prediction(track, frame)
        center_dist = point_distance(pred_center, det.center_px)
        if not np.isfinite(center_dist):
            return self.INF_COST
        center_bl = center_dist / body
        hard_gate = self._hard_gate_bl(track, frame)
        if center_bl > hard_gate:
            return self.INF_COST

        soft_gate = max(self._adaptive_gate_bl(track, frame), 0.25)
        keypoint_cost, common = self._keypoint_cost(lid, track, det, body, soft_gate, frame)
        center_cost = float(np.clip(center_bl / soft_gate, 0.0, 2.0))

        pred_box = self._predicted_bbox(track, frame)
        iou_cost = 1.0 - bbox_iou_xyxy(pred_box, det.bbox_xyxy) if pred_box is not None else 0.55

        if track.heading_vector is not None and det.heading_vector is not None:
            heading_cost = float(np.clip(
                (1.0 - cosine_similarity(track.heading_vector, det.heading_vector)) / 2.0,
                0.0, 1.0,
            ))
        else:
            heading_cost = 0.45

        size_cost = float(np.clip(abs(math.log(ratio)), 0.0, 1.5))
        confidence_cost = 1.0 - float(np.clip(det.box_conf, 0.0, 1.0))

        wk = self.w_keypoint if common >= self.min_common_keypoints else 0.0
        wc = self.w_center + (self.w_keypoint * 0.75 if wk == 0.0 else 0.0)
        wi = self.w_iou + (self.w_keypoint * 0.25 if wk == 0.0 else 0.0)
        terms = [
            (wk, keypoint_cost),
            (wc, center_cost),
            (wi, iou_cost),
            (self.w_heading, heading_cost),
            (self.w_size, size_cost),
            (self.confidence_cost_weight, confidence_cost),
        ]
        denom = sum(w for w, _ in terms if w > 0)
        if denom <= 1e-9:
            return self.INF_COST
        return float(sum(w * c for w, c in terms if w > 0) / denom)

    def build_fast_gate(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        frame: int,
    ) -> np.ndarray:
        """Vectorized exact gate for KeypointMotion identity matching.

        This duplicates only the hard checks at the beginning of the v1.20
        scalar ``_cost`` implementation: robust body-size ratio and predicted
        center distance.  It never rejects a pair that the scalar cost could
        score as finite.
        """
        tids = [int(value) for value in track_ids]
        dets = list(detections)
        nt, nd = len(tids), len(dets)
        if nt == 0 or nd == 0:
            return np.zeros((nt, nd), dtype=bool)
        if not self.identity_cascade_enabled:
            mask = np.ones((nt, nd), dtype=bool)
            self.last_fast_gate_candidate_count = int(mask.size)
            self.last_fast_gate_total_count = int(mask.size)
            self.last_fast_gate_density = 1.0
            return mask

        tracks = [self.tracks[lid] for lid in tids]
        track_body = np.asarray([self._robust_track_body(track) for track in tracks], dtype=np.float64)
        det_body = np.asarray([self._robust_det_body(det) for det in dets], dtype=np.float64)
        ratio = det_body[None, :] / np.maximum(track_body[:, None], 1e-6)
        size_ok = (ratio >= self.hard_size_ratio_min) & (ratio <= self.hard_size_ratio_max)
        body = np.maximum((track_body[:, None] + det_body[None, :]) * 0.5, 8.0)

        pred_centers = np.asarray([self._prediction(track, frame) for track in tracks], dtype=np.float64)
        det_centers = np.asarray([np.asarray(det.center_px, dtype=np.float64) for det in dets], dtype=np.float64)
        finite = np.all(np.isfinite(pred_centers), axis=1)[:, None] & np.all(
            np.isfinite(det_centers), axis=1
        )[None, :]
        center_dist = np.linalg.norm(pred_centers[:, None, :] - det_centers[None, :, :], axis=2)
        center_bl = np.divide(center_dist, body, out=np.full_like(center_dist, np.inf), where=body > 1e-9)
        hard_gate = np.asarray([self._hard_gate_bl(track, frame) for track in tracks], dtype=np.float64)
        mask = size_ok & finite & (center_bl <= hard_gate[:, None])
        self.last_fast_gate_candidate_count = int(np.count_nonzero(mask))
        self.last_fast_gate_total_count = int(mask.size)
        self.last_fast_gate_density = (
            float(self.last_fast_gate_candidate_count) / float(mask.size) if mask.size else 0.0
        )
        return mask

    def _base_cost_matrix_numpy(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        frame: int,
    ) -> np.ndarray:
        """批量计算v1.20基础身份代价，保持单对 ``_cost`` 数学定义不变。"""
        tids = [int(value) for value in track_ids]
        dets = list(detections)
        nt, nd = len(tids), len(dets)
        if nt == 0 or nd == 0:
            return np.full((nt, nd), self.INF_COST, dtype=np.float64)

        tracks = [self.tracks[lid] for lid in tids]
        track_body = np.asarray([self._robust_track_body(track) for track in tracks], dtype=np.float64)
        det_body = np.asarray([self._robust_det_body(det) for det in dets], dtype=np.float64)
        ratio = det_body[None, :] / np.maximum(track_body[:, None], 1e-6)
        body = np.maximum((track_body[:, None] + det_body[None, :]) * 0.5, 8.0)
        size_ok = (ratio >= self.hard_size_ratio_min) & (ratio <= self.hard_size_ratio_max)

        pred_centers = np.asarray([self._prediction(track, frame) for track in tracks], dtype=np.float64)
        det_centers = np.asarray([np.asarray(det.center_px, dtype=np.float64) for det in dets], dtype=np.float64)
        center_delta = pred_centers[:, None, :] - det_centers[None, :, :]
        center_dist = np.linalg.norm(center_delta, axis=2)
        center_finite = np.all(np.isfinite(pred_centers), axis=1)[:, None] & np.all(
            np.isfinite(det_centers), axis=1
        )[None, :]
        hard_gate = np.asarray([self._hard_gate_bl(track, frame) for track in tracks], dtype=np.float64)
        soft_gate = np.maximum(
            np.asarray([self._adaptive_gate_bl(track, frame) for track in tracks], dtype=np.float64),
            0.25,
        )
        center_bl = np.divide(center_dist, body, out=np.full_like(center_dist, np.inf), where=body > 1e-9)
        hard_ok = center_finite & (center_bl <= hard_gate[:, None])
        candidate_mask = size_ok & hard_ok
        self.last_fast_gate_candidate_count = int(np.count_nonzero(candidate_mask))
        self.last_fast_gate_total_count = int(candidate_mask.size)
        self.last_fast_gate_density = (
            float(self.last_fast_gate_candidate_count) / float(candidate_mask.size)
            if candidate_mask.size else 0.0
        )
        # For the common 20-mouse sparse case, evaluate the exact scalar base
        # cost only for physically possible cells.  This is more important than
        # vectorizing 300+ impossible pairs, and it returns the same scalar
        # values used by the pre-optimization implementation.
        if (
            self.identity_cascade_enabled
            and candidate_mask.size >= self.identity_cascade_min_cells
            and self.last_fast_gate_density <= self.identity_cascade_sparse_density
        ):
            pair_rows, pair_cols = np.nonzero(candidate_mask)
            pair_count = int(len(pair_rows))
            result = np.full((nt, nd), self.INF_COST, dtype=np.float64)
            point_count = len(KEYPOINT_NAMES)

            pred_points = np.full((nt, point_count, 2), np.nan, dtype=np.float64)
            old_conf = np.zeros((nt, point_count), dtype=np.float64)
            for r, (lid, track) in enumerate(zip(tids, tracks)):
                pred = np.asarray(self._predicted_keypoints(lid, track, frame), dtype=np.float64)
                conf = (
                    np.asarray(track.last_keypoint_conf, dtype=np.float64).reshape(-1)
                    if track.last_keypoint_conf is not None else np.zeros(point_count, dtype=np.float64)
                )
                n = min(len(pred), len(conf), point_count)
                if n > 0:
                    pred_points[r, :n] = pred[:n]
                    old_conf[r, :n] = conf[:n]
            new_points = np.full((nd, point_count, 2), np.nan, dtype=np.float64)
            new_conf = np.zeros((nd, point_count), dtype=np.float64)
            for c, det in enumerate(dets):
                points = np.asarray(det.keypoints_px, dtype=np.float64)
                conf = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
                n = min(len(points), len(conf), point_count)
                if n > 0:
                    new_points[c, :n] = points[:n]
                    new_conf[c, :n] = conf[:n]

            pp = pred_points[pair_rows]
            npnt = new_points[pair_cols]
            oc = old_conf[pair_rows]
            nc = new_conf[pair_cols]
            old_valid = (
                np.all(np.isfinite(pp), axis=2) & (pp[:, :, 0] > 0) & (pp[:, :, 1] > 0)
                & np.isfinite(oc) & (oc >= self.min_keypoint_conf)
            )
            new_valid = (
                np.all(np.isfinite(npnt), axis=2) & (npnt[:, :, 0] > 0) & (npnt[:, :, 1] > 0)
                & np.isfinite(nc) & (nc >= self.min_keypoint_conf)
            )
            common = old_valid & new_valid
            common_count = np.sum(common, axis=1)
            pair_body = body[pair_rows, pair_cols]
            kp_distance = np.linalg.norm(pp - npnt, axis=2) / np.maximum(pair_body[:, None], 1e-6)
            joint_conf = np.sqrt(np.clip(oc, 0.01, 1.0) * np.clip(nc, 0.01, 1.0))
            masked_distance = np.where(common, kp_distance, np.inf)
            masked_weight = np.where(common, joint_conf, 0.0)
            order = np.argsort(masked_distance, axis=1, kind="stable")
            sorted_distance = np.take_along_axis(masked_distance, order, axis=1)
            sorted_weight = np.take_along_axis(masked_weight, order, axis=1)
            cumulative = np.cumsum(sorted_weight, axis=1)
            cutoff = 0.5 * np.sum(sorted_weight, axis=1)
            median_index = np.argmax(cumulative >= cutoff[:, None], axis=1)
            weighted_median = sorted_distance[np.arange(pair_count), median_index]
            mean_joint_conf = np.divide(
                np.sum(masked_weight, axis=1), common_count,
                out=np.zeros(pair_count, dtype=np.float64), where=common_count > 0,
            )
            confidence_factor = np.maximum(1.50 - mean_joint_conf, 0.50)
            pair_soft_gate = soft_gate[pair_rows]
            keypoint_cost = np.clip(
                weighted_median * confidence_factor / np.maximum(pair_soft_gate, 1e-6), 0.0, 2.0
            )
            keypoint_cost = np.where(common_count >= self.min_common_keypoints, keypoint_cost, 0.65)
            pair_center_cost = np.clip(
                center_bl[pair_rows, pair_cols] / np.maximum(pair_soft_gate, 1e-6), 0.0, 2.0
            )

            pred_boxes = np.full((nt, 4), np.nan, dtype=np.float64)
            pred_box_valid = np.zeros(nt, dtype=bool)
            for r, track in enumerate(tracks):
                pred_box = self._predicted_bbox(track, frame)
                if pred_box is not None:
                    arr = np.asarray(pred_box, dtype=np.float64).reshape(-1)
                    if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
                        pred_boxes[r] = arr[:4]
                        pred_box_valid[r] = True
            det_boxes = np.asarray(
                [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for det in dets], dtype=np.float64
            )
            pb = pred_boxes[pair_rows]
            db = det_boxes[pair_cols]
            x1 = np.maximum(pb[:, 0], db[:, 0]); y1 = np.maximum(pb[:, 1], db[:, 1])
            x2 = np.minimum(pb[:, 2], db[:, 2]); y2 = np.minimum(pb[:, 3], db[:, 3])
            inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
            pa = np.maximum(pb[:, 2] - pb[:, 0], 0.0) * np.maximum(pb[:, 3] - pb[:, 1], 0.0)
            da = np.maximum(db[:, 2] - db[:, 0], 0.0) * np.maximum(db[:, 3] - db[:, 1], 0.0)
            union = pa + da - inter
            pair_iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-9)
            iou_cost = 1.0 - pair_iou
            iou_cost[~pred_box_valid[pair_rows]] = 0.55

            heading_cost = np.full(pair_count, 0.45, dtype=np.float64)
            for k, (r, c) in enumerate(zip(pair_rows.tolist(), pair_cols.tolist())):
                old_heading = tracks[r].heading_vector
                new_heading = dets[c].heading_vector
                if old_heading is not None and new_heading is not None:
                    heading_cost[k] = float(np.clip(
                        (1.0 - cosine_similarity(old_heading, new_heading)) / 2.0, 0.0, 1.0
                    ))
            pair_ratio = ratio[pair_rows, pair_cols]
            size_cost = np.clip(np.abs(np.log(np.maximum(pair_ratio, 1e-300))), 0.0, 1.5)
            confidence_cost = 1.0 - np.clip(
                np.asarray([float(dets[c].box_conf) for c in pair_cols], dtype=np.float64), 0.0, 1.0
            )
            enough_kp = common_count >= self.min_common_keypoints
            wk = np.where(enough_kp, self.w_keypoint, 0.0)
            wc = self.w_center + np.where(enough_kp, 0.0, self.w_keypoint * 0.75)
            wi = self.w_iou + np.where(enough_kp, 0.0, self.w_keypoint * 0.25)
            numerator = wk * keypoint_cost + wc * pair_center_cost + wi * iou_cost
            denominator = wk + wc + wi
            if self.w_heading > 0:
                numerator += self.w_heading * heading_cost; denominator += self.w_heading
            if self.w_size > 0:
                numerator += self.w_size * size_cost; denominator += self.w_size
            if self.confidence_cost_weight > 0:
                numerator += self.confidence_cost_weight * confidence_cost
                denominator += self.confidence_cost_weight
            pair_cost = np.divide(
                numerator, denominator, out=np.full(pair_count, self.INF_COST, dtype=np.float64),
                where=denominator > 1e-9,
            )
            result[pair_rows, pair_cols] = pair_cost
            result[~np.isfinite(result)] = self.INF_COST
            self.last_base_cost_mode = "cascade_sparse_numpy"
            return result

        self.last_base_cost_mode = "numpy_dense"
        center_cost = np.clip(center_bl / np.maximum(soft_gate[:, None], 1e-6), 0.0, 2.0)

        point_count = len(KEYPOINT_NAMES)
        pred_points = np.full((nt, point_count, 2), np.nan, dtype=np.float64)
        old_conf = np.zeros((nt, point_count), dtype=np.float64)
        for r, (lid, track) in enumerate(zip(tids, tracks)):
            pred = np.asarray(self._predicted_keypoints(lid, track, frame), dtype=np.float64)
            conf = (
                np.asarray(track.last_keypoint_conf, dtype=np.float64).reshape(-1)
                if track.last_keypoint_conf is not None
                else np.zeros(point_count, dtype=np.float64)
            )
            n = min(len(pred), len(conf), point_count)
            if n > 0:
                pred_points[r, :n] = pred[:n]
                old_conf[r, :n] = conf[:n]
        new_points = np.full((nd, point_count, 2), np.nan, dtype=np.float64)
        new_conf = np.zeros((nd, point_count), dtype=np.float64)
        for c, det in enumerate(dets):
            points = np.asarray(det.keypoints_px, dtype=np.float64)
            conf = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
            n = min(len(points), len(conf), point_count)
            if n > 0:
                new_points[c, :n] = points[:n]
                new_conf[c, :n] = conf[:n]

        old_valid = (
            np.all(np.isfinite(pred_points), axis=2)
            & (pred_points[:, :, 0] > 0)
            & (pred_points[:, :, 1] > 0)
            & np.isfinite(old_conf)
            & (old_conf >= self.min_keypoint_conf)
        )
        new_valid = (
            np.all(np.isfinite(new_points), axis=2)
            & (new_points[:, :, 0] > 0)
            & (new_points[:, :, 1] > 0)
            & np.isfinite(new_conf)
            & (new_conf >= self.min_keypoint_conf)
        )
        common = old_valid[:, None, :] & new_valid[None, :, :]
        common_count = np.sum(common, axis=2)
        kp_delta = pred_points[:, None, :, :] - new_points[None, :, :, :]
        kp_distance = np.linalg.norm(kp_delta, axis=3) / np.maximum(body[:, :, None], 1e-6)
        joint_conf = np.sqrt(
            np.clip(old_conf[:, None, :], 0.01, 1.0)
            * np.clip(new_conf[None, :, :], 0.01, 1.0)
        )
        masked_distance = np.where(common, kp_distance, np.inf)
        masked_weight = np.where(common, joint_conf, 0.0)
        order = np.argsort(masked_distance, axis=2, kind="stable")
        sorted_distance = np.take_along_axis(masked_distance, order, axis=2)
        sorted_weight = np.take_along_axis(masked_weight, order, axis=2)
        cumulative = np.cumsum(sorted_weight, axis=2)
        cutoff = 0.5 * np.sum(sorted_weight, axis=2)
        median_index = np.argmax(cumulative >= cutoff[:, :, None], axis=2)
        weighted_median = np.take_along_axis(
            sorted_distance, median_index[:, :, None], axis=2
        )[:, :, 0]
        mean_joint_conf = np.divide(
            np.sum(masked_weight, axis=2),
            common_count,
            out=np.zeros((nt, nd), dtype=np.float64),
            where=common_count > 0,
        )
        confidence_factor = np.maximum(1.50 - mean_joint_conf, 0.50)
        keypoint_cost = np.clip(
            weighted_median * confidence_factor / np.maximum(soft_gate[:, None], 1e-6),
            0.0,
            2.0,
        )
        keypoint_cost = np.where(common_count >= self.min_common_keypoints, keypoint_cost, 0.65)

        pred_boxes = np.full((nt, 4), np.nan, dtype=np.float64)
        pred_box_valid = np.zeros(nt, dtype=bool)
        for r, track in enumerate(tracks):
            pred_box = self._predicted_bbox(track, frame)
            if pred_box is not None:
                arr = np.asarray(pred_box, dtype=np.float64).reshape(-1)
                if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
                    pred_boxes[r] = arr[:4]
                    pred_box_valid[r] = True
        det_boxes = np.asarray(
            [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for det in dets],
            dtype=np.float64,
        )
        x1 = np.maximum(pred_boxes[:, None, 0], det_boxes[None, :, 0])
        y1 = np.maximum(pred_boxes[:, None, 1], det_boxes[None, :, 1])
        x2 = np.minimum(pred_boxes[:, None, 2], det_boxes[None, :, 2])
        y2 = np.minimum(pred_boxes[:, None, 3], det_boxes[None, :, 3])
        inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        pred_area = np.maximum(pred_boxes[:, 2] - pred_boxes[:, 0], 0.0) * np.maximum(
            pred_boxes[:, 3] - pred_boxes[:, 1], 0.0
        )
        det_area = np.maximum(det_boxes[:, 2] - det_boxes[:, 0], 0.0) * np.maximum(
            det_boxes[:, 3] - det_boxes[:, 1], 0.0
        )
        union = pred_area[:, None] + det_area[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-9)
        iou_cost = 1.0 - iou
        iou_cost[~pred_box_valid, :] = 0.55

        heading_cost = np.full((nt, nd), 0.45, dtype=np.float64)
        for r, track in enumerate(tracks):
            old_heading = track.heading_vector
            if old_heading is None:
                continue
            for c, det in enumerate(dets):
                if det.heading_vector is None:
                    continue
                heading_cost[r, c] = float(
                    np.clip(
                        (1.0 - cosine_similarity(old_heading, det.heading_vector)) / 2.0,
                        0.0,
                        1.0,
                    )
                )

        size_cost = np.clip(np.abs(np.log(np.maximum(ratio, 1e-300))), 0.0, 1.5)
        confidence_cost = 1.0 - np.clip(
            np.asarray([float(det.box_conf) for det in dets], dtype=np.float64), 0.0, 1.0
        )
        enough_kp = common_count >= self.min_common_keypoints
        wk = np.where(enough_kp, self.w_keypoint, 0.0)
        wc = self.w_center + np.where(enough_kp, 0.0, self.w_keypoint * 0.75)
        wi = self.w_iou + np.where(enough_kp, 0.0, self.w_keypoint * 0.25)

        numerator = np.zeros((nt, nd), dtype=np.float64)
        denominator = np.zeros((nt, nd), dtype=np.float64)
        for weight, term in (
            (wk, keypoint_cost),
            (wc, center_cost),
            (wi, iou_cost),
            (self.w_heading, heading_cost),
            (self.w_size, size_cost),
            (self.confidence_cost_weight, confidence_cost[None, :]),
        ):
            if np.isscalar(weight):
                if float(weight) <= 0:
                    continue
                numerator += float(weight) * term
                denominator += float(weight)
            else:
                positive = weight > 0
                numerator += np.where(positive, weight * term, 0.0)
                denominator += np.where(positive, weight, 0.0)
        result = np.full((nt, nd), self.INF_COST, dtype=np.float64)
        valid = size_ok & hard_ok & (denominator > 1e-9)
        result[valid] = numerator[valid] / denominator[valid]
        result[~np.isfinite(result)] = self.INF_COST
        return result

    def _new_track(self, det: Detection, frame: int, logical_id: Optional[int] = None) -> Optional[int]:
        if len(self.tracks) >= self.identity_capacity:
            return None
        lid = super()._new_track(det, frame, logical_id=logical_id)
        if lid is not None:
            self.tracks[lid].body_length_px = self._robust_det_body(det)
        return lid

    def _update_track(self, lid: int, det: Detection, frame: int) -> None:
        old_body = self._robust_track_body(self.tracks[lid])
        super()._update_track(lid, det, frame)
        det_body = self._robust_det_body(det)
        self.tracks[lid].body_length_px = (
            (1.0 - self.body_length_alpha) * old_body
            + self.body_length_alpha * det_body
        )

    def _separation_from_tracks(self, det: Detection, frame: int) -> float:
        det_body = self._robust_det_body(det)
        nearest = float("inf")
        for track in self.tracks.values():
            pred = self._prediction(track, frame)
            dist = point_distance(pred, det.center_px)
            body = max(det_body, self._robust_track_body(track))
            if np.isfinite(dist):
                nearest = min(nearest, dist / max(body, 1e-6))
        return nearest

    def _process_pending(
        self,
        detections: Sequence[Detection],
        unmatched_indices: Sequence[int],
        frame: int,
    ) -> List[Tuple[int, Detection]]:
        # 固定槽已满时，任何额外候选都只能是重复框/假阳性，绝不创建新ID。
        if len(self.tracks) >= self.identity_capacity:
            self.pending_candidates.clear()
            return []

        stale = [
            pid for pid, item in self.pending_candidates.items()
            if frame - int(item["last_frame"]) > self.pending_ttl_frames
        ]
        for pid in stale:
            self.pending_candidates.pop(pid, None)

        candidates = [
            int(i) for i in unmatched_indices
            if self._separation_from_tracks(detections[int(i)], frame)
            >= self.new_track_min_separation_bl
        ]
        pending_ids = sorted(self.pending_candidates)
        used_candidates: set[int] = set()
        used_pending: set[int] = set()

        if candidates and pending_ids:
            pcost = np.full((len(pending_ids), len(candidates)), self.INF_COST, dtype=np.float64)
            for r, pid in enumerate(pending_ids):
                item = self.pending_candidates[pid]
                old_center = np.asarray(item["center"], dtype=np.float64)
                old_body = max(float(item["body"]), 8.0)
                for c, idx in enumerate(candidates):
                    det = detections[idx]
                    body = max(old_body, self._robust_det_body(det))
                    dist = point_distance(old_center, det.center_px)
                    if np.isfinite(dist):
                        norm = dist / max(body, 1e-6)
                        if norm <= self.pending_match_distance_bl:
                            pcost[r, c] = norm
            if linear_sum_assignment is not None and pcost.size:
                rr, cc = linear_sum_assignment(pcost)
                pairs = zip(rr.tolist(), cc.tolist())
            else:
                pairs = self._greedy_assignment(pcost)
            for r, c in pairs:
                if pcost[r, c] >= self.INF_COST:
                    continue
                pid = pending_ids[r]
                idx = candidates[c]
                item = self.pending_candidates[pid]
                consecutive = frame - int(item["last_frame"]) <= 2
                item["hits"] = int(item["hits"]) + 1 if consecutive else 1
                det = detections[idx]
                item["center"] = np.asarray(det.center_px, dtype=np.float64).copy()
                item["body"] = self._robust_det_body(det)
                item["last_frame"] = int(frame)
                item["det"] = det
                used_candidates.add(idx)
                used_pending.add(pid)

        for idx in candidates:
            if idx in used_candidates:
                continue
            det = detections[idx]
            pid = int(self.next_pending_id)
            self.next_pending_id += 1
            self.pending_candidates[pid] = {
                "center": np.asarray(det.center_px, dtype=np.float64).copy(),
                "body": self._robust_det_body(det),
                "hits": 1,
                "last_frame": int(frame),
                "det": det,
            }
            used_pending.add(pid)

        ready = [
            (pid, item) for pid, item in self.pending_candidates.items()
            if int(item["last_frame"]) == frame
            and int(item["hits"]) >= self.new_track_confirm_frames
        ]
        ready.sort(key=lambda x: (
            -int(x[1]["hits"]),
            -float(getattr(x[1]["det"], "box_conf", 0.0)),
        ))
        output: List[Tuple[int, Detection]] = []
        for pid, item in ready[: self.max_new_tracks_per_frame]:
            if len(self.tracks) >= self.identity_capacity:
                break
            det = item["det"]
            lid = self._new_track(det, frame)
            if lid is not None:
                output.append((lid, det))
            self.pending_candidates.pop(pid, None)
        return output

    def assign(
        self,
        detections: Sequence[Detection],
        frame: int,
        occlusion_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[int, Detection]]:
        self._expire(frame)
        detections = list(detections)
        self.output_info = {}
        self.debug_records = []
        stats = {
            "raw": len(detections), "after_conf": len(detections), "after_kpt_filter": len(detections),
            "matched": 0, "low_rescued": 0, "lost_recovered": 0, "new_tentative": 0,
            "unmatched_det": 0, "active": len(self.tracks), "tentative": len(self.pending_candidates),
            "suspicious": 0, "lost": 0, "rendered": 0,
        }

        if not detections:
            for lid, track in self.tracks.items():
                self.kpt_missing[lid] = int(self.kpt_missing.get(lid, 0)) + 1
                track.lock_strength = max(0.0, track.lock_strength - 0.01)
                if self.kpt_missing[lid] > self.max_missing_frames:
                    track.state = "lost"
            stale = [pid for pid, item in self.pending_candidates.items()
                     if frame - int(item["last_frame"]) > self.pending_ttl_frames]
            for pid in stale:
                self.pending_candidates.pop(pid, None)
            stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
            stats["tentative"] = len(self.pending_candidates)
            self.frame_stats = stats
            return []

        if not self.tracks:
            # 先按置信度/姿态质量选容量内最可靠候选，再按空间顺序给固定ID。
            selected = sorted(
                detections,
                key=lambda d: (float(d.box_conf) + 0.10 * float(getattr(d, "pose_quality", 0.0))),
                reverse=True,
            )[: self.identity_capacity]
            selected = sorted(selected, key=self._initial_sort_key)
            output: List[Tuple[int, Detection]] = []
            for det in selected:
                lid = self._new_track(det, frame)
                if lid is None:
                    continue
                output.append((lid, det))
                self.output_info[lid] = {
                    "state": "tracked", "label": f"ID {lid}", "cost": 0.0,
                    "method": "keypoint_initial_fixed_capacity",
                }
            stats["matched"] = len(output)
            stats["active"] = len(self.tracks)
            stats["rendered"] = len(output)
            self.frame_stats = stats
            return sorted(output, key=lambda item: item[0])

        track_ids = sorted(self.tracks)
        ns, nd = len(track_ids), len(detections)
        # 每条轨迹拥有一个独立Dummy列，允许“本帧不匹配”。
        cost = np.full((ns, nd + ns), self.INF_COST, dtype=np.float64)
        for r, lid in enumerate(track_ids):
            track = self.tracks[lid]
            for c, det in enumerate(detections):
                cost[r, c] = self._cost(lid, track, det, frame)
            missing = int(self.kpt_missing.get(lid, 0))
            cost[r, nd + r] = self.unmatched_cost + self.unmatched_cost_growth * min(
                missing, self.prediction_max_frames
            )

        if linear_sum_assignment is not None:
            rr, cc = linear_sum_assignment(cost)
            proposed = list(zip(rr.tolist(), cc.tolist()))
        else:
            proposed = self._greedy_assignment(cost)

        output: Dict[int, Detection] = {}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for r, c in proposed:
            lid = track_ids[r]
            if c >= nd:
                continue
            chosen = float(cost[r, c])
            if not np.isfinite(chosen) or chosen >= self.INF_COST or chosen > self.max_assignment_cost:
                continue
            det = detections[c]
            was_missing = int(self.kpt_missing.get(lid, 0))
            self._update_track(lid, det, frame)
            output[lid] = det
            matched_tracks.add(lid)
            matched_detections.add(c)
            method = "keypoint_hungarian_recovered" if was_missing > 0 else "keypoint_hungarian"
            self.output_info[lid] = {
                "state": "tracked", "label": f"ID {lid}", "cost": chosen, "method": method,
            }
            stats["matched"] += 1
            if was_missing > 0:
                stats["lost_recovered"] += 1

        for lid in track_ids:
            if lid in matched_tracks:
                continue
            self.kpt_missing[lid] = int(self.kpt_missing.get(lid, 0)) + 1
            track = self.tracks[lid]
            track.lock_strength = max(0.0, track.lock_strength - 0.01)
            if self.kpt_missing[lid] > self.max_missing_frames:
                track.state = "lost"

        unmatched = [i for i in range(nd) if i not in matched_detections]
        new_output = self._process_pending(detections, unmatched, frame)
        for lid, det in new_output:
            output[lid] = det
            self.output_info[lid] = {
                "state": "tracked", "label": f"ID {lid}", "cost": 0.0,
                "method": "keypoint_new_track_confirmed",
            }
            stats["new_tentative"] += 1

        stats["unmatched_det"] = len(unmatched)
        stats["active"] = len(self.tracks)
        stats["tentative"] = len(self.pending_candidates)
        stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
        stats["rendered"] = len(output)
        self.frame_stats = stats
        return sorted(output.items(), key=lambda item: item[0])


BASE_MODULE_VERSION = "1.42.1-final-code-merge"

# ============================================================================
# v1.21: 全自适应鼠数（启动确认 + 动态增鼠 + 自动退役）
# ============================================================================
_V120KeypointMotionIdentityAssigner = _KeypointMotionIdentityAssignerV120Base


class KeypointMotionIdentityAssigner(_V120KeypointMotionIdentityAssigner):
    """v1.21 全自适应鼠数身份分配器。

    与固定槽模式不同，本实现不需要预先填写视频中的小鼠数量：
    - 开始阶段的检测先进入候选池，持续出现若干帧后才建立正式ID；
    - 视频中途出现的新小鼠同样经过连续确认后动态增加ID；
    - 单帧反光、重复框和短暂假阳性不会立即创建新ID；
    - 已确认轨迹短时漏检仍保留，长期离场后自动退役；
    - ``model.max_mice`` 仅作为安全上限，不代表固定鼠数。

    为避免第一帧误检污染整个视频，自适应模式默认不会在第一帧立即建ID。
    通常在第3帧完成首批身份确认。
    """

    def __init__(self, config: Mapping[str, Any], max_mice: int = 20) -> None:
        super().__init__(config, max_mice=max_mice)
        cfg = dict(config.get("keypoint_motion", {}))
        adaptive = dict(cfg.get("adaptive_count", {}))

        # v1.21 强制使用自适应容量。max_mice只是同时存在轨迹的安全上限。
        self.expected_mice_count = 0
        self.identity_capacity = self.max_mice

        self.adaptive_enabled = bool(adaptive.get("enabled", True))
        self.initial_confirm_hits = max(1, int(adaptive.get("initial_confirm_hits", 3)))
        self.new_track_confirm_hits = max(1, int(adaptive.get("new_track_confirm_hits", 4)))
        self.confirmation_window_frames = max(
            self.initial_confirm_hits,
            int(adaptive.get("confirmation_window_frames", 7)),
        )
        self.pending_max_gap_frames = max(1, int(adaptive.get("pending_max_gap_frames", 2)))
        self.pending_ttl_frames = max(
            self.confirmation_window_frames,
            int(adaptive.get("pending_ttl_frames", 12)),
        )
        self.pending_match_distance_bl = float(
            adaptive.get("pending_match_distance_body_lengths", self.pending_match_distance_bl)
        )
        self.pending_match_iou_weight = float(adaptive.get("pending_match_iou_weight", 0.18))
        self.pending_match_size_weight = float(adaptive.get("pending_match_size_weight", 0.12))
        self.pending_max_match_cost = float(adaptive.get("pending_max_match_cost", 1.05))

        self.pending_min_box_conf = float(adaptive.get("min_box_confidence", 0.06))
        self.pending_min_pose_quality = float(adaptive.get("min_pose_quality", 0.0))
        self.pending_min_candidate_score = float(adaptive.get("min_candidate_score", 0.10))
        self.pending_min_average_score = float(adaptive.get("min_average_score", 0.13))
        self.initial_promotion_min_separation_bl = float(
            adaptive.get("initial_promotion_min_separation_body_lengths", 0.18)
        )
        self.later_promotion_min_separation_bl = float(
            adaptive.get("later_promotion_min_separation_body_lengths", self.new_track_min_separation_bl)
        )
        self.initial_max_new_tracks_per_frame = max(
            1, int(adaptive.get("initial_max_new_tracks_per_frame", self.max_mice))
        )
        self.max_new_tracks_per_frame = max(
            1, int(adaptive.get("max_new_tracks_per_frame", self.max_new_tracks_per_frame))
        )

        # 自动数量统计仅用于日志与结果元数据，不反过来硬截断身份容量。
        self.count_history_window = max(3, int(adaptive.get("count_history_window_frames", 15)))
        self.count_history: Deque[int] = deque(maxlen=self.count_history_window)
        self.estimated_mouse_count = 0
        self.visible_mouse_count = 0

        # v1.20父类已经创建pending_candidates；v1.21扩充其中的时序字段。
        self.pending_candidates.clear()
        self.next_pending_id = 0

        # v1.40: the final adaptive assign() implementation used to commit the
        # Hungarian optimum without the row/column ambiguity margin that older
        # assigners already enforced.  These hard gates prefer a short missing
        # observation over an unsupported ID exchange or cross-screen jump.
        ambiguity_cfg = dict(cfg.get("assignment_ambiguity_guard", {}))
        self.assignment_ambiguity_enabled = bool(ambiguity_cfg.get("enabled", False))
        self.assignment_general_min_margin = max(
            float(ambiguity_cfg.get("general_min_margin", 0.08)), 0.0
        )
        self.assignment_contact_min_margin = max(
            float(ambiguity_cfg.get("contact_min_margin", 0.18)),
            self.assignment_general_min_margin,
        )
        self.assignment_recovery_min_margin = max(
            float(ambiguity_cfg.get("recovery_min_margin", 0.12)),
            self.assignment_general_min_margin,
        )
        self.assignment_max_jump_bl_per_frame = max(
            float(ambiguity_cfg.get("max_jump_body_lengths_per_frame", 0.80)), 0.05
        )
        self.assignment_max_recovery_jump_bl = max(
            float(ambiguity_cfg.get("max_recovery_jump_body_lengths", 1.60)),
            self.assignment_max_jump_bl_per_frame,
        )
        self.assignment_unique_override_margin = max(
            float(ambiguity_cfg.get("unique_override_margin", 0.35)),
            self.assignment_contact_min_margin,
        )
        self.assignment_unique_override_max_cost = max(
            float(ambiguity_cfg.get("unique_override_max_cost", 0.28)), 0.0
        )
        # A low row/column margin does not always mean that the selected
        # detection is wrong.  In dense scenes, two nearby tracks can have
        # almost equal global costs even though the current observation is a
        # tiny, one-frame continuation of the same trajectory.  Rejecting all
        # such rows creates the visible ID-box flicker reported in production.
        # Keep this compatibility-off for old/minimal configs; production
        # explicitly enables it with tight motion/cost bounds.  The accepted
        # observation uses the contact-safe updater so ambiguous evidence can
        # never contaminate appearance or long-term mask templates.
        self.assignment_motion_hold_enabled = bool(
            ambiguity_cfg.get("motion_hold_enabled", False)
        )
        self.assignment_motion_hold_max_jump_bl = max(
            float(ambiguity_cfg.get("motion_hold_max_jump_body_lengths", 0.30)),
            0.01,
        )
        self.assignment_motion_hold_max_cost = max(
            float(ambiguity_cfg.get("motion_hold_max_cost", 0.70)),
            0.0,
        )
        self.assignment_motion_hold_min_margin = max(
            float(ambiguity_cfg.get("motion_hold_min_margin", 0.0)),
            0.0,
        )

        # Keep mature 0..max_mice-1 identity slots alive through long occlusions.
        # Weak startup debris can still expire and its slot is reused, but a
        # well-established animal is marked lost instead of being deleted and
        # replaced by ID20/ID21 later in the same video.
        slot_cfg = dict(cfg.get("persistent_slots", {}))
        self.preserve_confirmed_slots = bool(slot_cfg.get("enabled", False))
        self.preserve_slot_min_hits = max(int(slot_cfg.get("min_hits", 30)), 1)
        if self.preserve_confirmed_slots:
            self.reuse_expired_ids = True

        # v1.22：真正进入身份匹配链路的伪实例掩码记忆。
        mask_cfg = dict(config.get("instance_mask_memory", {}))
        self.mask_memory_enabled = bool(mask_cfg.get("enabled", True))
        self.mask_long_memory_enabled = bool(mask_cfg.get("long_term_memory_enabled", True))  # 接收主程序为当前视频计算的长期掩码记忆开关。
        self.mask_cost_weight = float(mask_cfg.get("identity_cost_weight", 0.18))
        self.mask_short_alpha = float(mask_cfg.get("short_ema_alpha", 0.22))
        self.mask_long_alpha = float(mask_cfg.get("long_ema_alpha", 0.035))
        self.mask_update_min_quality = float(mask_cfg.get("update_min_quality", 0.38))
        self.mask_long_min_quality = float(mask_cfg.get("long_update_min_quality", 0.52))
        self.mask_short_max_overlap = float(mask_cfg.get("short_update_max_iou", 0.12))
        self.mask_long_max_overlap = float(mask_cfg.get("long_update_max_iou", 0.04))
        self._current_frozen_ids: set[int] = set()
        self._current_reserved_detection_indices: set[int] = set()
        contact_cfg = dict(cfg.get("contact_identity_guard", {}))
        self.contact_guard_enabled = bool(contact_cfg.get("enabled", True))
        self.contact_guard_distance_bl = float(
            contact_cfg.get("distance_body_lengths", 1.35)
        )
        self.contact_guard_iou = float(contact_cfg.get("overlap_iou", 0.02))
        self.contact_guard_hold_frames = max(
            1, int(contact_cfg.get("hold_frames", 6))
        )
        self.contact_guard_motion_weight = float(
            contact_cfg.get("motion_cost_weight", 0.90)
        )
        self.contact_guard_velocity_update_alpha = float(
            contact_cfg.get("velocity_update_alpha", 0.15)
        )
        self.contact_guard_max_assignment_cost = float(
            contact_cfg.get("max_assignment_cost", 0.70)
        )
        # Keep pre-contact longitudinal order long enough to survive a merged chase or fight.
        self.contact_order_hold_frames = max(
            self.contact_guard_hold_frames,
            int(contact_cfg.get("order_hold_frames", 45)),
        )
        # Require a visible reversal before overriding the globally cheapest Hungarian solution.
        self.contact_order_min_reversal_bl = float(
            contact_cfg.get("order_min_reversal_body_lengths", 0.12)
        )
        # 只有双方都确实在移动，才把接近解释为需要保持顺序的追逐接触。
        self.contact_order_min_speed_bl = float(
            contact_cfg.get("order_min_speed_body_lengths_per_frame", 0.015)
        )
        # 双方运动方向至少大体一致，排除静止邻鼠和迎面相遇的无关鼠对。
        self.contact_order_min_direction_cosine = float(
            contact_cfg.get("order_min_direction_cosine", 0.20)
        )
        # 默认关闭v1.34孤立一动一停顺序锁，回到接触时更少强制交换ID的旧基线。
        # 测试或专门实验可显式打开该兼容分支，不改变生产配置的回退行为。
        self.contact_order_allow_isolated_pair = bool(
            contact_cfg.get("allow_isolated_pair", False)
        )
        # 已确认发生顺序反转时允许使用稍宽的代价门，避免错误运动预测阻断纠正。
        self.contact_order_max_assignment_cost = float(
            contact_cfg.get("order_max_assignment_cost", self.max_assignment_cost)
        )
        # 低置信度墙边阴影、尾部残影和脱落标签不能作为成对顺序纠正的实体。
        self.contact_order_min_detection_confidence = float(
            contact_cfg.get("order_min_detection_confidence", 0.12)
        )
        # A strong long-window sequence preference may veto the spatial-order
        # swap heuristic when two mice truly cross instead of bouncing apart.
        self.contact_order_disk_veto_margin = float(
            contact_cfg.get("order_disk_veto_margin", 0.10)
        )
        # A spatial-order repair is a heuristic override of the Hungarian
        # optimum.  When DISK evidence is unavailable or inconclusive, do not
        # permit that override to buy physical order at an arbitrarily large
        # pairwise assignment penalty.  This specifically protects true
        # crossings where the old order memory would otherwise exchange IDs.
        self.contact_order_max_pair_cost_increase = float(
            contact_cfg.get("order_max_pair_cost_increase", 0.20)
        )
        self._current_contact_detection_ids: set[int] = set()
        self._contact_guard_until: Dict[int, int] = {}
        # Each entry stores the last unambiguous axis and expiry for one formal-ID pair.
        self._contact_pair_order: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # 记录当前帧经过严格顺序证据纠正的ID，供最终提交阶段选择专用代价门。
        self._contact_order_repaired_ids: set[int] = set()
        self._disk_contact_order_veto_ids: set[int] = set()
        self._contact_order_regret_veto_ids: set[int] = set()
        self.disk_contact_order_veto_count = 0
        self.contact_order_regret_veto_count = 0
        # v1.39：把DISK论文中的“缺失掩码 + 长时序 + 不确定性”思想接入在线身份分配。
        # 这里是因果工程实现，不冒充需要离线训练和未来帧上下文的原版Transformer。
        disk_sequence_cfg = dict(cfg.get("disk_sequence_guard", {}))
        if not disk_sequence_cfg:
            # Preserve old/minimal test configurations; production enables the
            # new guard explicitly in mouse_chase_attack_config.yaml.
            disk_sequence_cfg["enabled"] = False
        self.disk_sequence_guard = DiskSequenceIdentityGuard(
            disk_sequence_cfg,
            num_keypoints=len(KEYPOINT_NAMES),
        )
        performance_cfg = dict(config.get("performance", {}))
        self.identity_cpp_auto_threads = bool(
            performance_cfg.get("identity_cpp_auto_threads", True)
        )
        self.identity_cpp_max_threads = max(
            1, int(performance_cfg.get("identity_cpp_max_threads", 4))
        )
        self.identity_cpp_parallel_min_cells = max(
            1, int(performance_cfg.get("identity_cpp_parallel_min_cells", 16384))
        )
        self._disk_cost_cache_frame = -1
        self._disk_cost_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._disk_last_reliability_matrix = np.zeros((0, 0), dtype=np.float64)
        self._disk_last_cost_matrix = np.full((0, 0), np.nan, dtype=np.float64)
        self._configure_cpp_backend()

    def _allocate_id(self) -> Optional[int]:
        """Allocate only stable slots in ``0..max_mice-1``."""
        if self.free_ids:
            candidate = int(self.free_ids.pop(0))
            if 0 <= candidate < self.max_mice:
                return candidate
        used = set(int(value) for value in self.tracks)
        for candidate in range(self.max_mice):
            if candidate not in used:
                self.next_logical_id = max(int(self.next_logical_id), candidate + 1)
                return int(candidate)
        return None

    def _expire(self, frame: int) -> None:
        """Retire only weak slots; mature identities survive as lost tracks."""
        stale: List[int] = []
        frozen_now = set(int(x) for x in getattr(self, "_current_frozen_ids", set()))
        for lid, track in self.tracks.items():
            if int(lid) in frozen_now:
                continue
            missing = max(int(frame - track.last_frame), int(self.kpt_missing.get(lid, 0)))
            mature = bool(
                self.preserve_confirmed_slots
                and int(getattr(track, "hits", 0)) >= self.preserve_slot_min_hits
            )
            if mature:
                if missing > self.max_missing_frames:
                    track.state = "lost"
                continue
            weak = track.hits <= self.weak_track_drop_hits and missing > self.weak_track_drop_missing
            if missing > self.max_missing_frames or weak:
                stale.append(int(lid))
        for lid in stale:
            self.tracks.pop(lid, None)
            self.kpt_velocity.pop(lid, None)
            self.kpt_acceleration.pop(lid, None)
            self.kpt_missing.pop(lid, None)
            if lid not in self.free_ids:
                self.free_ids.append(int(lid))
        self.free_ids.sort()

    def _assignment_row_column_margin(
        self,
        detection_cost: np.ndarray,
        row: int,
        column: int,
    ) -> float:
        chosen = float(detection_cost[row, column])
        alternatives: List[float] = []
        row_values = np.delete(np.asarray(detection_cost[row], dtype=np.float64), column)
        row_values = row_values[np.isfinite(row_values) & (row_values < self.INF_COST)]
        if row_values.size:
            alternatives.append(float(np.min(row_values) - chosen))
        column_values = np.delete(
            np.asarray(detection_cost[:, column], dtype=np.float64), row
        )
        column_values = column_values[
            np.isfinite(column_values) & (column_values < self.INF_COST)
        ]
        if column_values.size:
            alternatives.append(float(np.min(column_values) - chosen))
        return min(alternatives) if alternatives else float("inf")

    def _assignment_jump_body_lengths(
        self,
        logical_id: int,
        detection: Detection,
        frame: int,
    ) -> Tuple[float, float]:
        track = self.tracks[int(logical_id)]
        predicted = np.asarray(self._prediction(track, frame), dtype=np.float64)
        current = np.asarray(detection.center_px, dtype=np.float64)
        body = max(
            float(getattr(track, "body_length_px", 0.0)),
            float(getattr(detection, "body_length_px", 0.0)),
            8.0,
        )
        jump = point_distance(predicted, current) / body
        gap = max(int(frame - track.last_frame), 1)
        allowed = min(
            self.assignment_max_recovery_jump_bl,
            self.assignment_max_jump_bl_per_frame * float(gap),
        )
        return float(jump), float(allowed)

    def __getstate__(self) -> Dict[str, Any]:
        """Keep native handles out of inference checkpoints.

        A pybind11 extension module is a live process object and cannot be
        serialized by pickle.  The numerical backend is optional, so the
        checkpoint stores the established Python state and restores the
        handle through the normal import/self-test path after loading.
        """
        state = dict(self.__dict__)
        state["_cpp_module"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore Python state without probing native code during copy/pickle."""
        self.__dict__.update(dict(state))
        self._cpp_module = None
        if str(getattr(self, "cpp_backend_status", "")).startswith("cpp_ready"):
            # _run_cpp_startup_selftest uses copy.copy(self).  Re-probing here
            # would recursively acquire the process-wide self-test lock.
            # A real checkpoint restore is re-probed lazily before its first
            # native matrix call instead.
            self.cpp_backend_status = "cpp_pending_restore"

    def _configure_cpp_backend(self) -> None:
        """Load and validate the optional isolated C++ matrix backend.

        ``auto`` is deliberately conservative: the extension is used only
        after import and the fixed-seed equivalence/Hungarian self-test pass.
        Every failure leaves the established NumPy path available.
        """
        if self.cost_matrix_backend not in {"auto", "cpp"}:
            self.cpp_backend_status = "not_requested"
            return

        try:
            import mouse_cpu_kernels_backend as cpp_backend
        except Exception as exc:  # pragma: no cover - defensive import guard
            self.cpp_backend_status = "numpy_fallback_import_helper"
            self.cpp_backend_failure_reason = (
                f"backend loader unavailable: {type(exc).__name__}: {exc}"
            )
            logging.warning("C++ identity backend回退NumPy：%s", self.cpp_backend_failure_reason)
            return

        module, diagnostic = cpp_backend.get_cpp_backend()
        if module is None:
            self.cpp_backend_status = "numpy_fallback_import"
            self.cpp_backend_failure_reason = str(diagnostic)
            logging.warning("C++ identity backend回退NumPy：%s", diagnostic)
            return

        if self.identity_cpp_selftest:
            cache_key = "|".join(
                [
                    str(getattr(module, "__file__", "mouse_cpu_kernels")),
                    str(self.min_common_keypoints),
                    repr(self.w_keypoint),
                    repr(self.w_center),
                    repr(self.w_iou),
                    repr(self.w_heading),
                    repr(self.w_size),
                    repr(self.confidence_cost_weight),
                ]
            )
            passed, selftest_message = cpp_backend.run_cached_selftest(
                cache_key,
                lambda: self._run_cpp_startup_selftest(module, cpp_backend),
            )
            if not passed:
                self.cpp_backend_status = "numpy_fallback_selftest"
                self.cpp_backend_failure_reason = str(selftest_message)
                logging.warning(
                    "C++ identity backend自检失败，回退NumPy：%s",
                    selftest_message,
                )
                return
            self.cpp_backend_status = "cpp_ready_selftested"
        else:
            self.cpp_backend_status = "cpp_ready_selftest_disabled"

        self._cpp_module = module
        logging.info(
            "C++ identity backend可用：%s | threads=%d | selftest=%s",
            diagnostic,
            self.identity_cpp_threads,
            self.identity_cpp_selftest,
        )

    @staticmethod
    def _contiguous_array(value: Any, dtype: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(value, dtype=dtype))

    def _cpp_identity_matrix_payload(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        frame: int,
    ) -> Dict[str, Any]:
        """Pack the v1.38 NumPy matrix inputs without exposing Python objects."""
        tids = [int(value) for value in track_ids]
        dets = list(detections)
        tracks = [self.tracks[lid] for lid in tids]
        nt, nd = len(tracks), len(dets)
        point_count = len(KEYPOINT_NAMES)

        track_body = np.asarray(
            [self._robust_track_body(track) for track in tracks],
            dtype=np.float64,
        )
        det_body = np.asarray(
            [self._robust_det_body(det) for det in dets],
            dtype=np.float64,
        )
        pred_centers = np.asarray(
            [self._prediction(track, frame) for track in tracks],
            dtype=np.float64,
        ).reshape(nt, 2)
        det_centers = np.asarray(
            [np.asarray(det.center_px, dtype=np.float64) for det in dets],
            dtype=np.float64,
        ).reshape(nd, 2)
        hard_gates = np.asarray(
            [self._hard_gate_bl(track, frame) for track in tracks],
            dtype=np.float64,
        )
        soft_gates = np.maximum(
            np.asarray(
                [self._adaptive_gate_bl(track, frame) for track in tracks],
                dtype=np.float64,
            ),
            0.25,
        )

        pred_points = np.full((nt, point_count, 2), np.nan, dtype=np.float64)
        old_conf = np.zeros((nt, point_count), dtype=np.float64)
        track_velocities = np.zeros((nt, 2), dtype=np.float64)
        track_accelerations = np.zeros((nt, 2), dtype=np.float64)
        for row, (lid, track) in enumerate(zip(tids, tracks)):
            pred = np.asarray(
                self._predicted_keypoints(lid, track, frame),
                dtype=np.float64,
            )
            conf = (
                np.asarray(track.last_keypoint_conf, dtype=np.float64).reshape(-1)
                if track.last_keypoint_conf is not None
                else np.zeros(point_count, dtype=np.float64)
            )
            count = min(len(pred), len(conf), point_count)
            if count > 0:
                pred_points[row, :count] = pred[:count]
                old_conf[row, :count] = conf[:count]
            velocity = np.asarray(track.velocity_px_per_frame, dtype=np.float64).reshape(-1)
            if velocity.size >= 2:
                track_velocities[row] = velocity[:2]
            acceleration = np.asarray(
                self.kpt_acceleration.get(lid, np.zeros((point_count, 2))),
                dtype=np.float64,
            )
            if acceleration.ndim == 2 and acceleration.shape[1] >= 2:
                valid_acc = np.all(np.isfinite(acceleration[:, :2]), axis=1)
                if np.any(valid_acc):
                    track_accelerations[row] = np.mean(
                        acceleration[valid_acc, :2],
                        axis=0,
                    )

        new_points = np.full((nd, point_count, 2), np.nan, dtype=np.float64)
        new_conf = np.zeros((nd, point_count), dtype=np.float64)
        for column, det in enumerate(dets):
            points = np.asarray(det.keypoints_px, dtype=np.float64)
            conf = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
            count = min(len(points), len(conf), point_count)
            if count > 0:
                new_points[column, :count] = points[:count]
                new_conf[column, :count] = conf[:count]

        old_valid = (
            np.all(np.isfinite(pred_points), axis=2)
            & (pred_points[:, :, 0] > 0)
            & (pred_points[:, :, 1] > 0)
            & np.isfinite(old_conf)
            & (old_conf >= self.min_keypoint_conf)
        )
        new_valid = (
            np.all(np.isfinite(new_points), axis=2)
            & (new_points[:, :, 0] > 0)
            & (new_points[:, :, 1] > 0)
            & np.isfinite(new_conf)
            & (new_conf >= self.min_keypoint_conf)
        )

        track_headings = np.full((nt, 2), np.nan, dtype=np.float64)
        for row, track in enumerate(tracks):
            if track.heading_vector is not None:
                heading = np.asarray(track.heading_vector, dtype=np.float64).reshape(-1)
                if heading.size >= 2:
                    track_headings[row] = (
                        heading[:2]
                        if np.all(np.isfinite(heading[:2]))
                        else np.zeros(2, dtype=np.float64)
                    )
        detection_headings = np.full((nd, 2), np.nan, dtype=np.float64)
        for column, det in enumerate(dets):
            if det.heading_vector is not None:
                heading = np.asarray(det.heading_vector, dtype=np.float64).reshape(-1)
                if heading.size >= 2:
                    detection_headings[column] = (
                        heading[:2]
                        if np.all(np.isfinite(heading[:2]))
                        else np.zeros(2, dtype=np.float64)
                    )

        pred_boxes = np.full((nt, 4), np.nan, dtype=np.float64)
        for row, track in enumerate(tracks):
            box = self._predicted_bbox(track, frame)
            if box is not None:
                values = np.asarray(box, dtype=np.float64).reshape(-1)
                if values.size >= 4:
                    pred_boxes[row] = values[:4]
        det_boxes = np.asarray(
            [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for det in dets],
            dtype=np.float64,
        ).reshape(nd, 4)
        detection_confidence = np.asarray(
            [float(det.box_conf) for det in dets],
            dtype=np.float64,
        )

        required = {
            "track_centers": self._contiguous_array(pred_centers, np.float64),
            "track_velocities": self._contiguous_array(track_velocities, np.float64),
            "track_accelerations": self._contiguous_array(track_accelerations, np.float64),
            "track_keypoints": self._contiguous_array(pred_points, np.float64),
            "track_kpt_valid": self._contiguous_array(old_valid, np.bool_),
            "track_body_lengths": self._contiguous_array(track_body, np.float64),
            "track_headings": self._contiguous_array(track_headings, np.float64),
            "detection_centers": self._contiguous_array(det_centers, np.float64),
            "detection_keypoints": self._contiguous_array(new_points, np.float64),
            "detection_kpt_valid": self._contiguous_array(new_valid, np.bool_),
            "detection_body_lengths": self._contiguous_array(det_body, np.float64),
            "detection_headings": self._contiguous_array(detection_headings, np.float64),
            "detection_confidence": self._contiguous_array(detection_confidence, np.float64),
        }
        optional = {
            "track_bboxes": self._contiguous_array(pred_boxes, np.float64),
            "detection_bboxes": self._contiguous_array(det_boxes, np.float64),
            "track_hard_gates": self._contiguous_array(hard_gates, np.float64),
            "track_soft_gates": self._contiguous_array(soft_gates, np.float64),
            "track_kpt_confidence": self._contiguous_array(old_conf, np.float64),
            "detection_kpt_confidence": self._contiguous_array(new_conf, np.float64),
            "hard_size_ratio_min": float(self.hard_size_ratio_min),
            "hard_size_ratio_max": float(self.hard_size_ratio_max),
            "min_common_keypoints": int(self.min_common_keypoints),
            "weight_keypoint": float(self.w_keypoint),
            "weight_center": float(self.w_center),
            "weight_iou": float(self.w_iou),
            "weight_heading": float(self.w_heading),
            "weight_size": float(self.w_size),
            "confidence_cost_weight": float(self.confidence_cost_weight),
            "inf_cost": float(self.INF_COST),
        }
        return {
            "required": required,
            "optional": optional,
            "threads": choose_adaptive_thread_count(
                nt,
                nd,
                fixed_threads=int(self.identity_cpp_threads),
                auto_enabled=bool(self.identity_cpp_auto_threads),
                max_threads=int(self.identity_cpp_max_threads),
                parallel_min_cells=int(self.identity_cpp_parallel_min_cells),
            ),
        }

    def _run_cpp_startup_selftest(self, module: Any, cpp_backend: Any) -> Tuple[bool, str]:
        """Compare scalar, NumPy, C++ and Hungarian results on 40 fixed cases."""
        rng = np.random.default_rng(138017)
        probe = copy.copy(self)
        probe.mask_memory_enabled = False
        probe.mask_long_memory_enabled = False
        # The native startup probe validates only the stateless v1.38 kernel.
        # Disable the new stateful sequence layer so histories from successive
        # synthetic cases cannot leak into the scalar reference matrix.
        probe.disk_sequence_guard = DiskSequenceIdentityGuard(
            {"enabled": False}, num_keypoints=len(KEYPOINT_NAMES)
        )
        probe._disk_cost_cache_frame = -1
        probe._disk_cost_cache = {}
        probe._current_contact_detection_ids = set()
        probe._contact_guard_until = {}
        probe._current_frozen_ids = set()
        probe._current_reserved_detection_indices = set()
        probe.tracks = {}
        probe.kpt_velocity = {}
        probe.kpt_acceleration = {}
        probe.kpt_missing = defaultdict(int)
        probe.pending_candidates = {}
        probe.next_logical_id = 0
        # The production assigner may intentionally be configured with fewer
        # slots than the fixed self-test matrix cases.  The self-test is a
        # kernel-equivalence probe, so give its isolated copy enough capacity
        # for the largest generated case instead of letting _new_track reject
        # a row and the following _update_track raise KeyError.
        probe.identity_capacity = max(
            int(getattr(probe, "identity_capacity", 0)),
            6,
        )
        probe.cost_matrix_backend = "cpp"
        probe._cpp_module = module
        probe.identity_cpp_threads = 1
        probe.cost_matrix_tie_fallback_epsilon = 1.0e-10
        probe.identity_cpp_fallback_on_tie = True

        template = np.asarray(
            [[22, 0], [12, -6], [12, 6], [5, 0], [-8, -7], [-8, 7], [-22, 0]],
            dtype=np.float64,
        )

        def make_detection(center: np.ndarray, angle: float, scale: float, case: int, index: int) -> Detection:
            ca, sa = np.cos(angle), np.sin(angle)
            rotation = np.asarray([[ca, -sa], [sa, ca]], dtype=np.float64)
            points = template * (scale / 44.0)
            points = points @ rotation.T + np.asarray(center, dtype=np.float64)
            confidence = rng.uniform(0.05, 0.98, size=len(KEYPOINT_NAMES)).astype(np.float64)
            if case % 8 == 0:
                confidence[index % len(KEYPOINT_NAMES)] = 0.0
                points[index % len(KEYPOINT_NAMES)] = np.nan
            if case % 11 == 0 and index % 2 == 0:
                confidence[0] = np.nan
            box = np.asarray(
                [center[0] - scale * 0.55, center[1] - scale * 0.35,
                 center[0] + scale * 0.55, center[1] + scale * 0.35],
                dtype=np.float64,
            )
            heading = rotation @ np.asarray([1.0, 0.0], dtype=np.float64)
            return Detection(
                raw_track_id=index,
                keypoints_px=points,
                keypoint_conf=confidence,
                bbox_xyxy=box,
                box_conf=float(rng.uniform(0.05, 0.99)),
                heading_vector=heading,
                detection_source="selftest",
            )

        for case in range(40):
            probe.tracks.clear()
            probe.kpt_velocity.clear()
            probe.kpt_acceleration.clear()
            track_count = case % 7
            detection_count = (case * 3 + 1) % 8
            if track_count == 0:
                track_count = 1
            if detection_count == 0:
                detection_count = 1
            base_detections = [
                make_detection(
                    np.asarray([80.0 + 70.0 * (index % 4), 100.0 + 65.0 * (index // 4)]),
                    0.03 * index,
                    30.0 + 3.0 * (index % 4),
                    case,
                    index,
                )
                for index in range(track_count)
            ]
            for lid, detection in enumerate(base_detections):
                probe._new_track(detection, 0, logical_id=lid)
                if case % 3 == 1:
                    shifted = copy.deepcopy(detection)
                    shifted.keypoints_px = np.asarray(detection.keypoints_px, dtype=np.float64).copy()
                    shifted.keypoints_px[:, 0] += 0.5 + 0.1 * lid
                    shifted.bbox_xyxy = np.asarray(detection.bbox_xyxy, dtype=np.float64).copy()
                    shifted.bbox_xyxy[[0, 2]] += 0.5 + 0.1 * lid
                    probe._update_track(lid, shifted, 1)
            detections = [
                make_detection(
                    np.asarray([80.0 + 70.0 * (index % 4) + rng.normal(0, 8),
                                100.0 + 65.0 * (index // 4) + rng.normal(0, 8)]),
                    0.03 * index + rng.normal(0, 0.04),
                    float(28.0 + rng.uniform(-5.0, 8.0)),
                    case,
                    index + 13,
                )
                for index in range(detection_count)
            ]
            frame = 2 + (case % 5)
            tids = sorted(probe.tracks)
            scalar = np.asarray(
                [
                    [probe._cost(lid, probe.tracks[lid], detection, frame) for detection in detections]
                    for lid in tids
                ],
                dtype=np.float64,
            )
            numpy_matrix = probe._base_cost_matrix_numpy(tids, detections, frame)
            if not np.array_equal(scalar >= probe.INF_COST, numpy_matrix >= probe.INF_COST):
                return False, f"scalar/NumPy gate mismatch in case {case}"
            finite = (scalar < probe.INF_COST) & (numpy_matrix < probe.INF_COST)
            if np.any(finite) and not np.allclose(
                scalar[finite], numpy_matrix[finite], rtol=0.0, atol=1.0e-12
            ):
                return False, f"scalar/NumPy value mismatch in case {case}"

            payload = probe._cpp_identity_matrix_payload(tids, detections, frame)
            payload["threads"] = 1
            cpp_matrix = cpp_backend.call_identity_cost_matrix(module, payload)
            if cpp_matrix.shape != numpy_matrix.shape:
                return False, f"shape mismatch in case {case}: {cpp_matrix.shape} vs {numpy_matrix.shape}"
            if not np.array_equal(cpp_matrix >= probe.INF_COST, numpy_matrix >= probe.INF_COST):
                return False, f"C++ gate mismatch in case {case}"
            finite = (cpp_matrix < probe.INF_COST) & (numpy_matrix < probe.INF_COST)
            if np.any(finite):
                max_error = float(np.max(np.abs(cpp_matrix[finite] - numpy_matrix[finite])))
                if max_error > 1.0e-12:
                    return False, f"C++ max_abs_error={max_error:.3e} in case {case}"
            if linear_sum_assignment is not None and cpp_matrix.size:
                left = linear_sum_assignment(numpy_matrix)
                right = linear_sum_assignment(cpp_matrix)
                if not (
                    np.array_equal(left[0], right[0])
                    and np.array_equal(left[1], right[1])
                ):
                    return False, f"Hungarian mismatch in case {case}"

        # Duplicate detections exercise the existing approximate-tie safety
        # path.  The result must be the scalar matrix, not a C++ approximation.
        tie_probe = copy.copy(probe)
        tie_probe.tracks = {}
        tie_probe.kpt_velocity = {}
        tie_probe.kpt_acceleration = {}
        tie_detection = make_detection(np.asarray([160.0, 140.0]), 0.1, 42.0, 41, 0)
        tie_probe._new_track(tie_detection, 0, logical_id=0)
        tie_detections = [copy.deepcopy(tie_detection), copy.deepcopy(tie_detection)]
        tie_probe._detection_cost_matrix([0], tie_detections, 1)
        if tie_probe.last_cost_backend_used != "python_tie_fallback":
            return False, "near-tie did not use the scalar fallback"
        return True, "40 fixed boundary/random cases; scalar, NumPy, C++, and Hungarian matched"

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        return bbox_iou_xyxy(a, b)

    @staticmethod
    def _mask_ema(old: Optional[np.ndarray], new: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
        if new is None:
            return None if old is None else np.asarray(old, dtype=np.float32).copy()
        new_arr = np.asarray(new, dtype=np.float32).reshape(-1)
        if old is None or np.asarray(old).size != new_arr.size:
            return new_arr.copy()
        out = (1.0 - float(alpha)) * np.asarray(old, dtype=np.float32).reshape(-1) + float(alpha) * new_arr
        norm = float(np.linalg.norm(out))
        return (out / norm).astype(np.float32) if norm > 1e-8 else out.astype(np.float32)

    @staticmethod
    def _mask_feature_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        aa = np.asarray(a, dtype=np.float64).reshape(-1)
        bb = np.asarray(b, dtype=np.float64).reshape(-1)
        n = min(len(aa), len(bb))
        if n < 8:
            return None
        valid = np.isfinite(aa[:n]) & np.isfinite(bb[:n])
        if int(valid.sum()) < 8:
            return None
        aa, bb = aa[:n][valid], bb[:n][valid]
        l1 = float(np.mean(np.abs(aa - bb)))
        denom = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1e-9)
        cosine = 1.0 - float(np.dot(aa, bb) / denom)
        return float(np.clip(0.55 * l1 * 2.5 + 0.45 * cosine, 0.0, 1.5))

    def _track_mask_distance(self, track: IdentityTrack, det: Detection) -> Optional[float]:
        if not self.mask_memory_enabled or not bool(getattr(det, "mask_reliable", False)):
            return None
        feature = getattr(det, "mask_feature", None)
        distances = []
        templates = [getattr(track, "mask_feature_short", None)]  # 短时掩码模板始终服务当前连续跟踪。
        if self.mask_long_memory_enabled:  # 仅长于十分钟的视频允许长期掩码模板参与匹配。
            templates.insert(0, getattr(track, "mask_feature_long", None))  # 长视频优先比较稳定的长期模板。
        for template in templates:  # 逐一计算当前策略允许使用的掩码模板距离。
            value = self._mask_feature_distance(template, feature)
            if value is not None:
                distances.append(value)
        return min(distances) if distances else None

    def _mask_cost_matrix(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
    ) -> np.ndarray:
        """Batch the mask-template comparisons that used to run pair by pair."""
        tids = [int(value) for value in track_ids]
        dets = list(detections)
        result = np.full((len(tids), len(dets)), np.nan, dtype=np.float64)
        if not self.mask_memory_enabled or not tids or not dets:
            return result

        detection_features: List[Optional[np.ndarray]] = []
        max_length = 0
        for det in dets:
            feature = getattr(det, "mask_feature", None)
            if not bool(getattr(det, "mask_reliable", False)) or feature is None:
                detection_features.append(None)
                continue
            values = np.asarray(feature, dtype=np.float64).reshape(-1)
            detection_features.append(values)
            max_length = max(max_length, int(values.size))
        templates_by_track: List[List[np.ndarray]] = []
        for lid in tids:
            track = self.tracks[lid]
            templates: List[np.ndarray] = []
            if self.mask_long_memory_enabled:
                value = getattr(track, "mask_feature_long", None)
                if value is not None:
                    values = np.asarray(value, dtype=np.float64).reshape(-1)
                    templates.append(values)
                    max_length = max(max_length, int(values.size))
            value = getattr(track, "mask_feature_short", None)
            if value is not None:
                values = np.asarray(value, dtype=np.float64).reshape(-1)
                templates.append(values)
                max_length = max(max_length, int(values.size))
            templates_by_track.append(templates)
        if max_length < 8:
            return result

        detection_matrix = np.full(
            (len(dets), max_length), np.nan, dtype=np.float64
        )
        for column, values in enumerate(detection_features):
            if values is not None and values.size:
                detection_matrix[column, : values.size] = values

        for row, templates in enumerate(templates_by_track):
            row_cost = np.full(len(dets), np.nan, dtype=np.float64)
            for template in templates:
                count = min(int(template.size), max_length)
                if count < 8:
                    continue
                candidate = detection_matrix[:, :count]
                reference = template[:count]
                valid = np.isfinite(candidate) & np.isfinite(reference[None, :])
                valid_count = np.sum(valid, axis=1)
                usable = valid_count >= 8
                if not np.any(usable):
                    continue
                safe_count = np.maximum(valid_count, 1)
                difference = np.where(
                    valid,
                    np.abs(candidate - reference[None, :]),
                    0.0,
                )
                l1 = np.sum(difference, axis=1) / safe_count
                masked_candidate = np.where(valid, candidate, 0.0)
                masked_reference = np.where(valid, reference[None, :], 0.0)
                dot = np.sum(masked_candidate * masked_reference, axis=1)
                norm_candidate = np.sqrt(
                    np.sum(masked_candidate * masked_candidate, axis=1)
                )
                norm_reference = np.sqrt(
                    np.sum(masked_reference * masked_reference, axis=1)
                )
                cosine = 1.0 - dot / np.maximum(
                    norm_candidate * norm_reference,
                    1.0e-9,
                )
                distance = np.clip(0.55 * l1 * 2.5 + 0.45 * cosine, 0.0, 1.5)
                distance[~usable] = np.nan
                replace = np.isfinite(distance) & (
                    ~np.isfinite(row_cost) | (distance < row_cost)
                )
                row_cost[replace] = distance[replace]
            result[row] = row_cost
        return result

    def _center_motion_prediction(
        self,
        track: IdentityTrack,
        frame: int,
    ) -> np.ndarray:
        """只用框中心速度预测，避免接触时交叉骨架把预测方向带到邻鼠。"""
        elapsed = min(
            max(int(frame - track.last_frame), 0),
            self.prediction_max_frames,
        )
        displacement = self._bounded_prediction_delta(
            track,
            np.asarray(track.velocity_px_per_frame, dtype=np.float64)
            * float(elapsed),
        )
        return np.asarray(track.last_center_px, dtype=np.float64) + displacement

    def _refresh_contact_guard(
        self,
        detections: Sequence[Detection],
        frame: int,
    ) -> None:
        """标记当前接触检测和需要短时保持运动优先的正式轨迹。"""
        self._current_contact_detection_ids = set()
        if not self.contact_guard_enabled:
            self._contact_guard_until.clear()
            return
        detection_count = len(detections)
        if detection_count >= 2:
            detection_centers = np.asarray(
                [np.asarray(det.center_px, dtype=np.float64).reshape(-1)[:2]
                 for det in detections],
                dtype=np.float64,
            ).reshape(detection_count, 2)
            detection_bodies = np.asarray(
                [self._robust_det_body(det) for det in detections],
                dtype=np.float64,
            )
            detection_boxes = np.asarray(
                [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4]
                 for det in detections],
                dtype=np.float64,
            ).reshape(detection_count, 4)
            detection_delta = (
                detection_centers[:, None, :] - detection_centers[None, :, :]
            )
            detection_distance = np.linalg.norm(detection_delta, axis=2)
            detection_pair_body = np.maximum(
                detection_bodies[:, None], detection_bodies[None, :]
            )
            detection_pair_body = np.maximum(detection_pair_body, 8.0)
            x1 = np.maximum(detection_boxes[:, None, 0], detection_boxes[None, :, 0])
            y1 = np.maximum(detection_boxes[:, None, 1], detection_boxes[None, :, 1])
            x2 = np.minimum(detection_boxes[:, None, 2], detection_boxes[None, :, 2])
            y2 = np.minimum(detection_boxes[:, None, 3], detection_boxes[None, :, 3])
            intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
            area = np.maximum(detection_boxes[:, 2] - detection_boxes[:, 0], 0.0) * np.maximum(
                detection_boxes[:, 3] - detection_boxes[:, 1], 0.0
            )
            union = area[:, None] + area[None, :] - intersection
            detection_iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 1.0e-9,
            )
            detection_contact = (
                (
                    np.isfinite(detection_distance)
                    & (
                        detection_distance / detection_pair_body
                        <= self.contact_guard_distance_bl
                    )
                )
                | (detection_iou >= self.contact_guard_iou)
            )
            for left_index, right_index in np.argwhere(
                np.triu(detection_contact, k=1)
            ):
                self._current_contact_detection_ids.add(id(detections[int(left_index)]))
                self._current_contact_detection_ids.add(id(detections[int(right_index)]))
        track_ids = [
            int(logical_id)
            for logical_id in sorted(self.tracks)
            if int(logical_id) not in self._current_frozen_ids
        ]
        # Compute the track contact graph once.  The old path evaluated all 190
        # pairs twice (once for neighbours and once for holds/order), including
        # repeated robust-body and prediction calls.
        track_count = len(track_ids)
        contact_neighbors: Dict[int, set[int]] = {
            logical_id: set() for logical_id in track_ids
        }
        if track_count:
            predicted_center_array = np.asarray(
                [self._center_motion_prediction(self.tracks[lid], frame)
                 for lid in track_ids],
                dtype=np.float64,
            ).reshape(track_count, 2)
            predicted_body_array = np.asarray(
                [self._robust_track_body(self.tracks[lid]) for lid in track_ids],
                dtype=np.float64,
            )
            track_delta = (
                predicted_center_array[:, None, :] - predicted_center_array[None, :, :]
            )
            track_distance = np.linalg.norm(track_delta, axis=2)
            track_pair_body = np.maximum(
                predicted_body_array[:, None], predicted_body_array[None, :]
            )
            track_pair_body = np.maximum(track_pair_body, 8.0)
            track_contact = (
                np.isfinite(track_distance)
                & (track_distance / track_pair_body <= self.contact_guard_distance_bl)
            )
            np.fill_diagonal(track_contact, False)
        else:
            predicted_center_array = np.empty((0, 2), dtype=np.float64)
            track_pair_body = np.empty((0, 0), dtype=np.float64)
            track_contact = np.empty((0, 0), dtype=bool)
        if self.contact_order_allow_isolated_pair:
            for left_index, right_index in np.argwhere(np.triu(track_contact, k=1)):
                left_id = track_ids[int(left_index)]
                right_id = track_ids[int(right_index)]
                contact_neighbors[left_id].add(right_id)
                contact_neighbors[right_id].add(left_id)
        for left_index, right_index in np.argwhere(np.triu(track_contact, k=1)):
            left_index = int(left_index)
            right_index = int(right_index)
            left_id = track_ids[left_index]
            right_id = track_ids[right_index]
            left_track = self.tracks[left_id]
            right_track = self.tracks[right_id]
            left_center = predicted_center_array[left_index]
            right_center = predicted_center_array[right_index]
            body = float(track_pair_body[left_index, right_index])
            hold_until = int(frame + self.contact_guard_hold_frames)
            self._contact_guard_until[left_id] = max(
                self._contact_guard_until.get(left_id, frame),
                hold_until,
            )
            self._contact_guard_until[right_id] = max(
                self._contact_guard_until.get(right_id, frame),
                hold_until,
            )
            # 用体长归一化速度，避免把追逐路线旁边的静止鼠记成接触顺序对。
            left_velocity = np.asarray(
                left_track.velocity_px_per_frame,
                dtype=np.float64,
            )
            right_velocity = np.asarray(
                right_track.velocity_px_per_frame,
                dtype=np.float64,
            )
            left_speed_bl = float(np.linalg.norm(left_velocity)) / body
            right_speed_bl = float(np.linalg.norm(right_velocity)) / body
            # 速度过低的一方通常是路旁静止鼠，不参与追逐顺序保护。
            both_moving = (
                left_speed_bl >= self.contact_order_min_speed_bl
                and right_speed_bl >= self.contact_order_min_speed_bl
            )
            one_moving = (
                self.contact_order_allow_isolated_pair
                and (
                    left_speed_bl >= self.contact_order_min_speed_bl
                    or right_speed_bl >= self.contact_order_min_speed_bl
                )
            )
            isolated_pair = bool(
                self.contact_order_allow_isolated_pair
                and contact_neighbors.get(left_id, set()) == {right_id}
                and contact_neighbors.get(right_id, set()) == {left_id}
            )
            # 方向一致性只在双方速度可靠时计算，避免零向量产生伪余弦。
            direction_cosine = (
                float(np.dot(left_velocity, right_velocity))
                / max(
                    float(
                        np.linalg.norm(left_velocity)
                        * np.linalg.norm(right_velocity)
                    ),
                    1e-9,
                )
                if both_moving
                else -1.0
            )
            # 非共同运动鼠对仍保留普通接触保护，但不建立跨帧顺序约束。
            if both_moving:
                if direction_cosine < self.contact_order_min_direction_cosine:
                    continue
            elif not (one_moving and isolated_pair):
                continue
            # Preserve the pair's pre-contact order instead of letting predictions cross.
            pair_key = (left_id, right_id)
            low_center = left_center
            high_center = right_center
            relative = np.asarray(
                high_center - low_center,
                dtype=np.float64,
            )
            relative_norm = float(np.linalg.norm(relative))
            state = self._contact_pair_order.get(pair_key)
            if relative_norm > 1e-6:
                proposed_axis = relative / relative_norm
                if state is None:
                    state = {
                        "axis": proposed_axis,
                        "until": int(frame + self.contact_order_hold_frames),
                    }
                    self._contact_pair_order[pair_key] = state
                else:
                    old_axis = np.asarray(state["axis"], dtype=np.float64)
                    # Never learn a reversed observation; it may already be an ID swap.
                    if float(np.dot(relative, old_axis)) > 0.0:
                        blended = 0.85 * old_axis + 0.15 * proposed_axis
                        blended_norm = float(np.linalg.norm(blended))
                        if blended_norm > 1e-6:
                            state["axis"] = blended / blended_norm
                    state["until"] = max(
                        int(state.get("until", frame)),
                        int(frame + self.contact_order_hold_frames),
                    )
        for logical_id in list(self._contact_guard_until):
            if self._contact_guard_until[logical_id] < frame:
                self._contact_guard_until.pop(logical_id, None)
        # Expired pair-order memories are removed independently of per-track contact holds.
        for pair_key in list(self._contact_pair_order):
            if int(self._contact_pair_order[pair_key].get("until", -1)) < frame:
                self._contact_pair_order.pop(pair_key, None)

    def _repair_contact_order_assignments(
        self,
        proposed: Sequence[Tuple[int, int]],
        cost: np.ndarray,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        frame: int,
    ) -> List[Tuple[int, int]]:
        """Swap two proposed columns when contact would reverse their stored physical order."""
        # 每帧重新计算纠正集合，禁止上一帧状态泄漏到当前提交阶段。
        self._contact_order_repaired_ids = set()
        self._disk_contact_order_veto_ids = set()
        self._contact_order_regret_veto_ids = set()
        # Convert the assignment list to a mutable row-to-column map.
        assignment = {int(row): int(column) for row, column in proposed}
        row_by_id = {
            int(logical_id): int(row)
            for row, logical_id in enumerate(track_ids)
        }
        detection_count = len(detections)
        # 同一帧中每个正式ID只能参与一次顺序修复，禁止0/2/5式链式换位。
        repaired_ids: set[int] = set()
        # 先处理反转幅度最大的鼠对，保证最明确的纠正拥有优先权。
        repair_candidates: List[
            Tuple[float, Tuple[int, int], int, int, int, int]
        ] = []
        for pair_key, state in self._contact_pair_order.items():
            # Ignore expired or currently absent tracks.
            if int(state.get("until", -1)) < frame:
                continue
            if pair_key[0] not in row_by_id or pair_key[1] not in row_by_id:
                continue
            low_row = row_by_id[pair_key[0]]
            high_row = row_by_id[pair_key[1]]
            low_column = assignment.get(low_row, detection_count)
            high_column = assignment.get(high_row, detection_count)
            # Pairwise order is observable only when both tracks received real detections.
            if low_column >= detection_count or high_column >= detection_count:
                continue
            if low_column == high_column:
                continue
            low_detection = detections[low_column]
            high_detection = detections[high_column]
            # 顺序纠正需要两只真实鼠都被可靠检出；任一低置信度候选都保持未匹配。
            if min(
                float(getattr(low_detection, "box_conf", 0.0)),
                float(getattr(high_detection, "box_conf", 0.0)),
            ) < self.contact_order_min_detection_confidence:
                continue
            # The repair applies only while the two assigned detections are still in contact.
            pair_body = max(
                self._robust_det_body(low_detection),
                self._robust_det_body(high_detection),
                8.0,
            )
            separation = point_distance(
                low_detection.center_px,
                high_detection.center_px,
            )
            if (
                not np.isfinite(separation)
                or separation / pair_body > self.contact_guard_distance_bl
            ):
                continue
            axis = np.asarray(state.get("axis"), dtype=np.float64)
            if axis.size != 2 or not np.all(np.isfinite(axis)):
                continue
            observed_relative = (
                np.asarray(high_detection.center_px, dtype=np.float64)
                - np.asarray(low_detection.center_px, dtype=np.float64)
            )
            projection = float(np.dot(observed_relative, axis))
            # Small projection noise near complete overlap is not a reliable crossing.
            if projection >= -self.contact_order_min_reversal_bl * pair_body:
                continue
            # 暂存候选；完成全部安全检查后再统一按反转强度执行。
            swapped_low_cost = float(cost[low_row, high_column])
            swapped_high_cost = float(cost[high_row, low_column])
            original_low_cost = float(cost[low_row, low_column])
            original_high_cost = float(cost[high_row, high_column])
            # A repaired assignment must still satisfy the normal contact safety limit.
            repair_limit = min(
                self.max_assignment_cost,
                self.contact_order_max_assignment_cost,
            )
            if (
                not np.isfinite(swapped_low_cost)
                or not np.isfinite(swapped_high_cost)
                or swapped_low_cost >= self.INF_COST
                or swapped_high_cost >= self.INF_COST
                or swapped_low_cost > repair_limit
                or swapped_high_cost > repair_limit
            ):
                continue
            sequence_supports_repair = False
            sequence_cost = np.asarray(
                getattr(self, "_disk_last_cost_matrix", np.full((0, 0), np.nan)),
                dtype=np.float64,
            )
            sequence_reliability = np.asarray(
                getattr(self, "_disk_last_reliability_matrix", np.zeros((0, 0))),
                dtype=np.float64,
            )
            if (
                self.disk_sequence_guard.enabled
                and sequence_cost.shape[:2] == (len(track_ids), detection_count)
                and sequence_reliability.shape[:2] == (len(track_ids), detection_count)
            ):
                sequence_indices = (
                    (low_row, low_column),
                    (high_row, high_column),
                    (low_row, high_column),
                    (high_row, low_column),
                )
                sequence_values = np.asarray(
                    [sequence_cost[row, column] for row, column in sequence_indices],
                    dtype=np.float64,
                )
                reliability_values = np.asarray(
                    [sequence_reliability[row, column] for row, column in sequence_indices],
                    dtype=np.float64,
                )
                if (
                    np.all(np.isfinite(sequence_values))
                    and np.all(
                        reliability_values
                        >= self.disk_sequence_guard.min_reliability
                    )
                ):
                    original_sequence_total = float(
                        sequence_values[0] + sequence_values[1]
                    )
                    swapped_sequence_total = float(
                        sequence_values[2] + sequence_values[3]
                    )
                    if (
                        original_sequence_total
                        + self.contact_order_disk_veto_margin
                        < swapped_sequence_total
                    ):
                        self._disk_contact_order_veto_ids.update(pair_key)
                        self.disk_contact_order_veto_count += 1
                        # The accepted assignment is a true crossing.  Rebase
                        # the order memory to the new physical order so the
                        # stale pre-crossing axis cannot request the same swap
                        # on every subsequent contact frame.
                        relative_norm = float(np.linalg.norm(observed_relative))
                        if relative_norm > 1.0e-6:
                            state["axis"] = observed_relative / relative_norm
                            state["until"] = int(
                                frame + self.contact_order_hold_frames
                            )
                        continue
                    sequence_supports_repair = bool(
                        swapped_sequence_total
                        + self.contact_order_disk_veto_margin
                        < original_sequence_total
                    )
            original_pair_cost = original_low_cost + original_high_cost
            swapped_pair_cost = swapped_low_cost + swapped_high_cost
            if (
                not sequence_supports_repair
                and np.isfinite(original_pair_cost)
                and original_pair_cost < self.INF_COST
                and swapped_pair_cost
                > original_pair_cost + self.contact_order_max_pair_cost_increase
            ):
                self._contact_order_regret_veto_ids.update(pair_key)
                self.contact_order_regret_veto_count += 1
                relative_norm = float(np.linalg.norm(observed_relative))
                if relative_norm > 1.0e-6:
                    state["axis"] = observed_relative / relative_norm
                    state["until"] = int(frame + self.contact_order_hold_frames)
                continue
            repair_candidates.append(
                (
                    float(-projection / pair_body),
                    pair_key,
                    low_row,
                    high_row,
                    low_column,
                    high_column,
                )
            )
        # 反转最明显的鼠对优先，且任何ID本帧最多交换一次。
        for (
            _reversal_strength,
            pair_key,
            low_row,
            high_row,
            low_column,
            high_column,
        ) in sorted(repair_candidates, key=lambda item: item[0], reverse=True):
            # 已参与更强纠正的ID不能再次被别的鼠对交换。
            if pair_key[0] in repaired_ids or pair_key[1] in repaired_ids:
                continue
            # 前一项纠正可能改变列号；列号已变化时放弃过期候选。
            if (
                assignment.get(low_row, detection_count) != low_column
                or assignment.get(high_row, detection_count) != high_column
            ):
                continue
            assignment[low_row] = high_column
            assignment[high_row] = low_column
            repaired_ids.update(pair_key)
            self._contact_order_repaired_ids.update(pair_key)
        # Preserve rows not represented in the mutable map and return deterministic row order.
        return sorted(
            [(row, column) for row, column in assignment.items()],
            key=lambda item: item[0],
        )

    def _contact_guarded(
        self,
        logical_id: int,
        det: Detection,
        frame: int,
    ) -> bool:
        return bool(
            self.contact_guard_enabled
            and (
                id(det) in self._current_contact_detection_ids
                or self._contact_guard_until.get(int(logical_id), -1) >= frame
            )
        )

    def _contact_motion_cost(
        self,
        track: IdentityTrack,
        det: Detection,
        frame: int,
    ) -> float:
        """接触期只依赖中心运动、框重叠和体型，不信任易串鼠的骨架。"""
        track_body = self._robust_track_body(track)
        det_body = self._robust_det_body(det)
        ratio = det_body / max(track_body, 1e-6)
        if ratio < self.hard_size_ratio_min or ratio > self.hard_size_ratio_max:
            return self.INF_COST
        body = max(float(np.nanmedian([track_body, det_body])), 8.0)
        predicted_center = self._center_motion_prediction(track, frame)
        center_distance = point_distance(predicted_center, det.center_px)
        if not np.isfinite(center_distance):
            return self.INF_COST
        center_bl = center_distance / body
        hard_gate = self._hard_gate_bl(track, frame)
        if center_bl > hard_gate:
            return self.INF_COST
        soft_gate = max(self._adaptive_gate_bl(track, frame), 0.35)
        center_cost = float(np.clip(center_bl / soft_gate, 0.0, 2.0))
        predicted_box = None
        if track.last_bbox_xyxy is not None:
            predicted_box = np.asarray(track.last_bbox_xyxy, dtype=np.float64).copy()
            shift = predicted_center - np.asarray(
                track.last_center_px,
                dtype=np.float64,
            )
            predicted_box[[0, 2]] += shift[0]
            predicted_box[[1, 3]] += shift[1]
        overlap_cost = (
            1.0 - bbox_iou_xyxy(predicted_box, det.bbox_xyxy)
            if predicted_box is not None
            else 0.55
        )
        displacement = (
            np.asarray(det.center_px, dtype=np.float64)
            - np.asarray(track.last_center_px, dtype=np.float64)
        )
        velocity = np.asarray(track.velocity_px_per_frame, dtype=np.float64)
        if np.linalg.norm(velocity) > 0.03 * body and np.linalg.norm(displacement) > 1.0:
            direction_cost = float(
                np.clip(
                    (1.0 - cosine_similarity(velocity, displacement)) / 2.0,
                    0.0,
                    1.0,
                )
            )
        else:
            direction_cost = 0.35
        size_cost = float(
            np.clip(abs(math.log(max(ratio, 1e-6))), 0.0, 1.5)
        )
        confidence_cost = 1.0 - float(np.clip(det.box_conf, 0.0, 1.0))
        return float(
            0.74 * center_cost
            + 0.18 * overlap_cost
            + 0.02 * direction_cost
            + 0.04 * size_cost
            + 0.02 * confidence_cost
        )

    def _contact_motion_cost_matrix(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        frame: int,
    ) -> np.ndarray:
        """Vectorized equivalent of ``_contact_motion_cost`` for all pairs."""
        tids = [int(value) for value in track_ids]
        dets = list(detections)
        nt, nd = len(tids), len(dets)
        output = np.full((nt, nd), self.INF_COST, dtype=np.float64)
        if nt == 0 or nd == 0:
            return output
        tracks = [self.tracks[lid] for lid in tids]
        track_body = np.asarray(
            [self._robust_track_body(track) for track in tracks], dtype=np.float64
        )
        det_body = np.asarray(
            [self._robust_det_body(det) for det in dets], dtype=np.float64
        )
        ratio = det_body[None, :] / np.maximum(track_body[:, None], 1.0e-6)
        body = np.maximum(
            0.5 * (track_body[:, None] + det_body[None, :]), 8.0
        )
        predicted_center = np.asarray(
            [self._center_motion_prediction(track, int(frame)) for track in tracks],
            dtype=np.float64,
        ).reshape(nt, 2)
        detection_center = np.asarray(
            [np.asarray(det.center_px, dtype=np.float64) for det in dets],
            dtype=np.float64,
        ).reshape(nd, 2)
        center_distance = np.linalg.norm(
            predicted_center[:, None, :] - detection_center[None, :, :], axis=2
        )
        center_bl = center_distance / body
        hard_gate = np.asarray(
            [self._hard_gate_bl(track, int(frame)) for track in tracks],
            dtype=np.float64,
        )
        soft_gate = np.maximum(
            np.asarray(
                [self._adaptive_gate_bl(track, int(frame)) for track in tracks],
                dtype=np.float64,
            ),
            0.35,
        )
        center_cost = np.clip(center_bl / soft_gate[:, None], 0.0, 2.0)

        predicted_boxes = np.full((nt, 4), np.nan, dtype=np.float64)
        for row, track in enumerate(tracks):
            if track.last_bbox_xyxy is None:
                continue
            box = np.asarray(track.last_bbox_xyxy, dtype=np.float64).reshape(-1)
            if box.size < 4 or not np.all(np.isfinite(box[:4])):
                continue
            predicted_boxes[row] = box[:4]
            shift = predicted_center[row] - np.asarray(
                track.last_center_px, dtype=np.float64
            )
            predicted_boxes[row, [0, 2]] += shift[0]
            predicted_boxes[row, [1, 3]] += shift[1]
        detection_boxes = np.asarray(
            [np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)[:4] for det in dets],
            dtype=np.float64,
        ).reshape(nd, 4)
        x1 = np.maximum(predicted_boxes[:, None, 0], detection_boxes[None, :, 0])
        y1 = np.maximum(predicted_boxes[:, None, 1], detection_boxes[None, :, 1])
        x2 = np.minimum(predicted_boxes[:, None, 2], detection_boxes[None, :, 2])
        y2 = np.minimum(predicted_boxes[:, None, 3], detection_boxes[None, :, 3])
        intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        track_area = np.maximum(predicted_boxes[:, 2] - predicted_boxes[:, 0], 0.0) * np.maximum(
            predicted_boxes[:, 3] - predicted_boxes[:, 1], 0.0
        )
        detection_area = np.maximum(detection_boxes[:, 2] - detection_boxes[:, 0], 0.0) * np.maximum(
            detection_boxes[:, 3] - detection_boxes[:, 1], 0.0
        )
        union = track_area[:, None] + detection_area[None, :] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=np.isfinite(union) & (union > 1.0e-9),
        )
        overlap_cost = 1.0 - iou
        no_box = ~np.all(np.isfinite(predicted_boxes), axis=1)
        overlap_cost[no_box, :] = 0.55

        last_center = np.asarray(
            [np.asarray(track.last_center_px, dtype=np.float64) for track in tracks],
            dtype=np.float64,
        ).reshape(nt, 2)
        displacement = detection_center[None, :, :] - last_center[:, None, :]
        velocity = np.asarray(
            [np.asarray(track.velocity_px_per_frame, dtype=np.float64) for track in tracks],
            dtype=np.float64,
        ).reshape(nt, 2)
        velocity_norm = np.linalg.norm(velocity, axis=1)
        displacement_norm = np.linalg.norm(displacement, axis=2)
        moving = (
            (velocity_norm[:, None] > 0.03 * body)
            & (displacement_norm > 1.0)
        )
        cosine = np.sum(velocity[:, None, :] * displacement, axis=2) / np.maximum(
            velocity_norm[:, None] * displacement_norm,
            1.0e-9,
        )
        direction_cost = np.full((nt, nd), 0.35, dtype=np.float64)
        direction_cost[moving] = np.clip(
            (1.0 - cosine[moving]) / 2.0, 0.0, 1.0
        )
        size_cost = np.clip(np.abs(np.log(np.maximum(ratio, 1.0e-6))), 0.0, 1.5)
        confidence_cost = 1.0 - np.clip(
            np.asarray([float(det.box_conf) for det in dets], dtype=np.float64),
            0.0,
            1.0,
        )
        combined = (
            0.74 * center_cost
            + 0.18 * overlap_cost
            + 0.02 * direction_cost
            + 0.04 * size_cost
            + 0.02 * confidence_cost[None, :]
        )
        valid = (
            np.isfinite(combined)
            & np.isfinite(center_distance)
            & (ratio >= self.hard_size_ratio_min)
            & (ratio <= self.hard_size_ratio_max)
            & (center_bl <= hard_gate[:, None])
        )
        output[valid] = combined[valid]
        return output

    def _disk_pair_score(
        self,
        lid: int,
        det: Detection,
        frame: int,
    ) -> Tuple[float, float]:
        """Return cached sequence evidence for one committed diagnostic pair."""
        if int(frame) != int(self._disk_cost_cache_frame):
            self._disk_cost_cache_frame = int(frame)
            self._disk_cost_cache = {}
        key = (int(lid), id(det))
        cached = self._disk_cost_cache.get(key)
        if cached is not None:
            return cached
        value = self.disk_sequence_guard.score_detection(int(lid), det, int(frame))
        self._disk_cost_cache[key] = value
        return value

    def _blend_disk_pair_cost(
        self,
        base_cost: float,
        lid: int,
        det: Detection,
        frame: int,
    ) -> float:
        """Blend causal sequence evidence without weakening any physical hard gate."""
        if (
            not self.disk_sequence_guard.enabled
            or not np.isfinite(base_cost)
            or base_cost >= self.INF_COST
        ):
            return float(base_cost)
        sequence_cost, reliability = self._disk_pair_score(lid, det, frame)
        if (
            not np.isfinite(sequence_cost)
            or reliability < self.disk_sequence_guard.min_reliability
            or self.disk_sequence_guard.cost_weight <= 0.0
        ):
            return float(base_cost)
        weight = float(
            np.clip(
                self.disk_sequence_guard.cost_weight * reliability,
                0.0,
                self.disk_sequence_guard.cost_weight,
            )
        )
        return float((1.0 - weight) * base_cost + weight * sequence_cost)

    def _cost(self, lid: int, track: IdentityTrack, det: Detection, frame: int) -> float:
        base_cost = super()._cost(lid, track, det, frame)
        if self._contact_guarded(lid, det, frame):
            motion_cost = self._contact_motion_cost(track, det, frame)
            if not np.isfinite(motion_cost) or motion_cost >= self.INF_COST:
                return self.INF_COST
            if not np.isfinite(base_cost) or base_cost >= self.INF_COST:
                return float(motion_cost)
            motion_weight = float(
                np.clip(self.contact_guard_motion_weight, 0.0, 1.0)
            )
            return float(
                motion_weight * motion_cost
                + (1.0 - motion_weight) * base_cost
            )
        if not np.isfinite(base_cost) or base_cost >= self.INF_COST:
            return base_cost
        mask_cost = self._track_mask_distance(track, det)
        combined_cost = float(base_cost)
        if mask_cost is not None and self.mask_cost_weight > 0:
            # 聚集外的干净掩码作为辅助证据，不允许压过运动硬门。
            weight = float(np.clip(self.mask_cost_weight * max(float(getattr(det, "mask_quality", 0.0)), 0.25), 0.0, 0.35))
            combined_cost = float((1.0 - weight) * combined_cost + weight * mask_cost)
        return self._blend_disk_pair_cost(combined_cost, lid, det, frame)

    def _detection_cost_matrix(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        frame: int,
    ) -> np.ndarray:
        """构造正式轨迹×检测代价矩阵；只批量化无状态数值内核。"""
        tids = [int(value) for value in track_ids]
        dets = list(detections)
        backend = str(getattr(self, "cost_matrix_backend", "numpy")).lower()
        if backend == "python":
            self.last_cost_backend_used = "python"
            matrix = np.full((len(tids), len(dets)), self.INF_COST, dtype=np.float64)
            for r, lid in enumerate(tids):
                track = self.tracks[lid]
                for c, det in enumerate(dets):
                    matrix[r, c] = self._cost(lid, track, det, frame)
            return matrix

        # 基础运动/关键点/IoU/体型代价批量计算；接触守卫和mask记忆仍按原逻辑覆盖。
        # C++ 只替换这个无状态矩阵内核；Hungarian、contact order、mask记忆、
        # 轨迹状态和正式 assignment_cost 仍全部由下面的 Python 逻辑负责。
        active_backend = "numpy"
        matrix: Optional[np.ndarray] = None
        if (
            backend in {"auto", "cpp"}
            and getattr(self, "_cpp_module", None) is None
            and getattr(self, "cpp_backend_status", "") == "cpp_pending_restore"
        ):
            self._configure_cpp_backend()
        if backend in {"auto", "cpp"} and getattr(self, "_cpp_module", None) is not None:
            try:
                import mouse_cpu_kernels_backend as cpp_backend

                payload = self._cpp_identity_matrix_payload(tids, dets, frame)
                matrix = cpp_backend.call_identity_cost_matrix(self._cpp_module, payload)
                expected_shape = (len(tids), len(dets))
                if matrix.shape != expected_shape:
                    raise ValueError(
                        f"returned shape {matrix.shape}, expected {expected_shape}"
                    )
                if not np.all(np.isfinite(matrix)):
                    raise ValueError("returned matrix contains NaN/Inf")
                active_backend = "cpp"
            except Exception as exc:
                # A runtime ABI/buffer/backend error is no more fatal than an
                # import error: discard this process's native handle and use
                # the established NumPy implementation for the current frame.
                self._cpp_module = None
                self.cpp_backend_status = "numpy_fallback_runtime"
                self.cpp_backend_failure_reason = (
                    f"runtime call failed: {type(exc).__name__}: {exc}"
                )
                logging.warning(
                    "C++ identity backend运行失败，当前帧回退NumPy：%s",
                    self.cpp_backend_failure_reason,
                )

        if matrix is None:
            matrix = super()._base_cost_matrix_numpy(tids, dets, frame)
            base_mode = str(getattr(self, "last_base_cost_mode", "numpy_dense"))
            if base_mode == "cascade_sparse_numpy":
                active_backend = "cascade_sparse_numpy"
            else:
                active_backend = "numpy_fallback" if backend in {"auto", "cpp"} else "numpy"
        row_contact = np.asarray(
            [
                self.contact_guard_enabled
                and self._contact_guard_until.get(int(lid), -1) >= int(frame)
                for lid in tids
            ],
            dtype=bool,
        )
        column_contact = np.asarray(
            [
                self.contact_guard_enabled
                and id(det) in self._current_contact_detection_ids
                for det in dets
            ],
            dtype=bool,
        )
        contact_pairs = row_contact[:, None] | column_contact[None, :]
        if np.any(contact_pairs):
            motion_matrix = self._contact_motion_cost_matrix(tids, dets, int(frame))
            motion_valid = np.isfinite(motion_matrix) & (motion_matrix < self.INF_COST)
            base_valid = np.isfinite(matrix) & (matrix < self.INF_COST)
            invalid_contact = contact_pairs & ~motion_valid
            matrix[invalid_contact] = self.INF_COST
            motion_only = contact_pairs & motion_valid & ~base_valid
            matrix[motion_only] = motion_matrix[motion_only]
            blend_contact = contact_pairs & motion_valid & base_valid
            motion_weight = float(
                np.clip(self.contact_guard_motion_weight, 0.0, 1.0)
            )
            matrix[blend_contact] = (
                motion_weight * motion_matrix[blend_contact]
                + (1.0 - motion_weight) * matrix[blend_contact]
            )

        non_contact = ~contact_pairs
        if self.mask_cost_weight > 0.0 and np.any(non_contact):
            mask_matrix = self._mask_cost_matrix(tids, dets)
            mask_quality = np.asarray(
                [
                    max(float(getattr(det, "mask_quality", 0.0)), 0.25)
                    for det in dets
                ],
                dtype=np.float64,
            )
            mask_weight = np.clip(
                self.mask_cost_weight * mask_quality,
                0.0,
                0.35,
            )
            mask_valid = (
                non_contact
                & np.isfinite(matrix)
                & (matrix < self.INF_COST)
                & np.isfinite(mask_matrix)
            )
            matrix[mask_valid] = (
                (1.0 - np.broadcast_to(mask_weight, matrix.shape)[mask_valid])
                * matrix[mask_valid]
                + np.broadcast_to(mask_weight, matrix.shape)[mask_valid]
                * mask_matrix[mask_valid]
            )

        # DISK-inspired long-window sequence consistency is auxiliary only:
        # contact pairs keep the established motion-only guard, and physical
        # hard gates from the base matrix remain impossible assignments.
        sequence_cost = np.full_like(matrix, np.nan, dtype=np.float64)
        sequence_reliability = np.zeros_like(matrix, dtype=np.float64)
        sequence_rows: List[int] = []
        active_contact_order_ids = {
            int(logical_id)
            for pair_key, state in self._contact_pair_order.items()
            if int(state.get("until", -1)) >= int(frame)
            for logical_id in pair_key
        }
        if self.disk_sequence_guard.enabled:
            for row, lid in enumerate(tids):
                eligible = (
                    np.isfinite(matrix[row])
                    & (matrix[row] < self.INF_COST)
                )
                values = matrix[row, eligible]
                if values.size == 0:
                    continue
                best = float(np.min(values))
                ambiguous = False
                if values.size >= 2:
                    two = np.partition(values, 1)[:2]
                    ambiguous = bool(
                        float(two[1] - two[0])
                        <= self.disk_sequence_guard.activation_margin
                    )
                missing = int(self.kpt_missing.get(int(lid), 0))
                activate_for_gap = bool(
                    self.disk_sequence_guard.activation_after_missing_frames == 0
                    or missing
                    >= self.disk_sequence_guard.activation_after_missing_frames
                )
                state = str(getattr(self.tracks[lid], "state", "tracked"))
                if (
                    ambiguous
                    or best >= self.disk_sequence_guard.activation_min_best_cost
                    or activate_for_gap
                    or state in {"suspicious", "lost"}
                    or int(lid) in active_contact_order_ids
                ):
                    sequence_rows.append(row)
        if sequence_rows:
            selected_ids = [tids[row] for row in sequence_rows]
            selected_cost, selected_reliability = self.disk_sequence_guard.cost_matrix(
                selected_ids,
                dets,
                int(frame),
            )
            sequence_cost[sequence_rows, :] = selected_cost
            sequence_reliability[sequence_rows, :] = selected_reliability
        else:
            self.disk_sequence_guard.last_active_tracks = 0
            self.disk_sequence_guard.last_mean_reliability = 0.0
        self._disk_last_cost_matrix = sequence_cost
        self._disk_last_reliability_matrix = sequence_reliability
        if int(frame) != int(self._disk_cost_cache_frame):
            self._disk_cost_cache_frame = int(frame)
            self._disk_cost_cache = {}
        for row, lid in enumerate(tids):
            for column, det in enumerate(dets):
                self._disk_cost_cache[(int(lid), id(det))] = (
                    float(sequence_cost[row, column]),
                    float(sequence_reliability[row, column]),
                )
        sequence_valid = (
            non_contact
            & np.isfinite(matrix)
            & (matrix < self.INF_COST)
            & np.isfinite(sequence_cost)
            & (
                sequence_reliability
                >= self.disk_sequence_guard.min_reliability
            )
        )
        if np.any(sequence_valid) and self.disk_sequence_guard.cost_weight > 0.0:
            sequence_weight = np.clip(
                self.disk_sequence_guard.cost_weight * sequence_reliability,
                0.0,
                self.disk_sequence_guard.cost_weight,
            )
            matrix[sequence_valid] = (
                (1.0 - sequence_weight[sequence_valid]) * matrix[sequence_valid]
                + sequence_weight[sequence_valid] * sequence_cost[sequence_valid]
            )

        # NumPy reductions can differ from the scalar baseline at the final bits.
        # Clear-cost frames are safe, but a true near-tie is exactly where ID stability
        # matters most.  Fall back to the legacy scalar matrix only for such frames.
        epsilon = float(getattr(self, "cost_matrix_tie_fallback_epsilon", 1.0e-10))
        near_tie = False
        if epsilon > 0.0 and matrix.size:
            finite = np.isfinite(matrix) & (matrix < self.INF_COST)
            for row in range(matrix.shape[0]):
                values = matrix[row, finite[row]]
                if values.size >= 2:
                    two = np.partition(values, 1)[:2]
                    if abs(float(two[1]) - float(two[0])) <= epsilon:
                        near_tie = True
                        break
            if not near_tie:
                for col in range(matrix.shape[1]):
                    values = matrix[finite[:, col], col]
                    if values.size >= 2:
                        two = np.partition(values, 1)[:2]
                        if abs(float(two[1]) - float(two[0])) <= epsilon:
                            near_tie = True
                            break
        if near_tie and not (
            active_backend == "cpp"
            and not bool(getattr(self, "identity_cpp_fallback_on_tie", True))
        ):
            exact = np.full_like(matrix, self.INF_COST)
            for r, lid in enumerate(tids):
                track = self.tracks[lid]
                for c, det in enumerate(dets):
                    exact[r, c] = self._cost(lid, track, det, frame)
            self.last_cost_backend_used = "python_tie_fallback"
            return exact
        self.last_cost_backend_used = active_backend
        return matrix

    def _update_mask_memory(self, lid: int, det: Detection, allow_long: bool = True) -> None:
        if not self.mask_memory_enabled or lid not in self.tracks:
            return
        if lid in self._current_frozen_ids:
            return
        if not bool(getattr(det, "mask_reliable", False)):
            return
        quality = float(getattr(det, "mask_quality", 0.0))
        overlap = float(getattr(det, "max_overlap_iou", 0.0))
        feature = getattr(det, "mask_feature", None)
        if feature is None or quality < self.mask_update_min_quality or overlap > self.mask_short_max_overlap:
            return
        track = self.tracks[lid]
        track.mask_feature_short = self._mask_ema(track.mask_feature_short, feature, self.mask_short_alpha)
        track.mask_quality_ema = 0.85 * float(track.mask_quality_ema) + 0.15 * quality
        track.mask_updates += 1
        if (self.mask_long_memory_enabled  # 主程序判定短视频时禁止创建或更新长期掩码模板。
                and allow_long  # 调用方还可以在聚集或重识别阶段临时禁止长期更新。
                and quality >= self.mask_long_min_quality  # 只有高质量掩码才有资格更新长期模板。
                and overlap <= self.mask_long_max_overlap):  # 只有几乎无遮挡时才避免长期模板被邻鼠污染。
            track.mask_feature_long = self._mask_ema(track.mask_feature_long, feature, self.mask_long_alpha)
            track.mask_long_updates += 1

    def _new_track(self, det: Detection, frame: int, logical_id: Optional[int] = None) -> Optional[int]:
        lid = super()._new_track(det, frame, logical_id=logical_id)
        if lid is not None and bool(getattr(det, "mask_reliable", False)) and getattr(det, "mask_feature", None) is not None:
            feature = np.asarray(det.mask_feature, dtype=np.float32).copy()
            track = self.tracks[lid]
            track.mask_feature_short = feature.copy()
            if (self.mask_long_memory_enabled  # 短视频的新轨迹只初始化短时掩码模板。
                    and float(getattr(det, "mask_quality", 0.0)) >= self.mask_long_min_quality  # 长期模板要求高质量初始掩码。
                    and float(getattr(det, "max_overlap_iou", 0.0)) <= self.mask_long_max_overlap):  # 长期模板要求初始观测无明显重叠。
                track.mask_feature_long = feature.copy()
                track.mask_long_updates = 1
            track.mask_quality_ema = float(getattr(det, "mask_quality", 0.0))
            track.mask_updates = 1
        if lid is not None:
            self.disk_sequence_guard.observe(
                int(lid),
                int(frame),
                det,
                allow_update=(int(lid) not in self._current_frozen_ids),
            )
        return lid

    def _update_track(self, lid: int, det: Detection, frame: int) -> None:
        allow_sequence_update = bool(
            int(lid) not in self._current_frozen_ids
            and not self._contact_guarded(int(lid), det, int(frame))
            and str(getattr(self.tracks.get(int(lid)), "state", "tracked"))
            not in {"suspicious", "cluster_occluded", "lost"}
        )
        cached_sequence_evidence = self._disk_cost_cache.get((int(lid), id(det)))
        if cached_sequence_evidence is not None and not np.isfinite(
            float(cached_sequence_evidence[0])
        ):
            cached_sequence_evidence = None
        super()._update_track(lid, det, frame)
        self._update_mask_memory(lid, det, allow_long=True)
        self.disk_sequence_guard.observe(
            int(lid),
            int(frame),
            det,
            allow_update=allow_sequence_update,
            sequence_evidence=cached_sequence_evidence,
        )

    def _update_track_contact_safe(
        self,
        lid: int,
        det: Detection,
        frame: int,
    ) -> None:
        """更新接触期位置，但保留接触前骨架、掩码和速度模板。"""
        track = self.tracks[lid]
        old_center = np.asarray(track.last_center_px, dtype=np.float64).copy()
        old_velocity = np.asarray(
            track.velocity_px_per_frame,
            dtype=np.float64,
        ).copy()
        old_body = float(track.body_length_px)
        old_points = (
            None
            if track.last_keypoints_px is None
            else np.asarray(track.last_keypoints_px, dtype=np.float64).copy()
        )
        old_conf = (
            None
            if track.last_keypoint_conf is None
            else np.asarray(track.last_keypoint_conf, dtype=np.float64).copy()
        )
        old_pose = (
            None
            if track.normalized_pose is None
            else np.asarray(track.normalized_pose).copy()
        )
        old_anchor = (
            None
            if track.anchor_feature is None
            else np.asarray(track.anchor_feature).copy()
        )
        old_heading = (
            None
            if track.heading_vector is None
            else np.asarray(track.heading_vector).copy()
        )
        old_kpt_velocity = np.asarray(
            self.kpt_velocity.get(
                lid,
                np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float64),
            ),
            dtype=np.float64,
        ).copy()
        old_kpt_acceleration = np.asarray(
            self.kpt_acceleration.get(
                lid,
                np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float64),
            ),
            dtype=np.float64,
        ).copy()
        old_mask_short = getattr(track, "mask_feature_short", None)
        old_mask_long = getattr(track, "mask_feature_long", None)
        old_mask_short = (
            None
            if old_mask_short is None
            else np.asarray(old_mask_short).copy()
        )
        old_mask_long = (
            None
            if old_mask_long is None
            else np.asarray(old_mask_long).copy()
        )
        old_mask_quality = float(getattr(track, "mask_quality_ema", 0.0))
        old_mask_updates = int(getattr(track, "mask_updates", 0))
        old_mask_long_updates = int(getattr(track, "mask_long_updates", 0))
        old_clean_streak = int(getattr(track, "clean_streak", 0))
        old_frame = int(track.last_frame)
        self._update_track(lid, det, frame)
        updated_track = self.tracks[lid]
        new_center = np.asarray(updated_track.last_center_px, dtype=np.float64)
        elapsed = max(int(frame - old_frame), 1)
        measured_velocity = (new_center - old_center) / float(elapsed)
        alpha = float(
            np.clip(self.contact_guard_velocity_update_alpha, 0.0, 1.0)
        )
        updated_track.velocity_px_per_frame = (
            (1.0 - alpha) * old_velocity + alpha * measured_velocity
        )
        updated_track.body_length_px = (
            0.95 * old_body + 0.05 * self._robust_det_body(det)
        )
        if old_points is not None:
            updated_track.last_keypoints_px = (
                old_points + (new_center - old_center)
            )
        if old_conf is not None:
            updated_track.last_keypoint_conf = old_conf
        updated_track.normalized_pose = old_pose
        updated_track.anchor_feature = old_anchor
        updated_track.heading_vector = old_heading
        updated_track.mask_feature_short = old_mask_short
        updated_track.mask_feature_long = old_mask_long
        updated_track.mask_quality_ema = old_mask_quality
        updated_track.mask_updates = old_mask_updates
        updated_track.mask_long_updates = old_mask_long_updates
        updated_track.clean_streak = old_clean_streak
        self.kpt_velocity[lid] = old_kpt_velocity
        self.kpt_acceleration[lid] = old_kpt_acceleration
        self._contact_guard_until[lid] = max(
            self._contact_guard_until.get(lid, frame),
            int(frame + self.contact_guard_hold_frames),
        )

    def reidentify_track(self, lid: int, det: Detection, frame: int, method: str = "cluster_delayed_reid") -> None:
        """聚集后延迟判决成功时，把匿名tracklet重新接回原逻辑ID。"""
        lid = int(lid)
        if lid not in self.tracks:
            self._new_track(det, frame, logical_id=lid)
        else:
            # 提交后允许当前干净掩码更新短模板；长期模板仍按质量门控制。
            was_frozen = lid in self._current_frozen_ids
            self._current_frozen_ids.discard(lid)
            self._update_track(lid, det, frame)
            if was_frozen:
                self._current_frozen_ids.add(lid)
        self.kpt_missing[lid] = 0
        self.tracks[lid].state = "tracked"
        self.output_info[lid] = {
            "state": "reid_resolved", "label": f"ID {lid}", "cost": 0.0, "method": method,
        }

    def _candidate_score(self, det: Detection) -> float:
        box_conf = float(np.clip(getattr(det, "box_conf", 0.0), 0.0, 1.0))
        pose_quality = float(getattr(det, "pose_quality", 0.0))
        if not np.isfinite(pose_quality) or pose_quality <= 0:
            pose_quality = _pose_quality_for_identity(det, min_conf=self.min_keypoint_conf)
        pose_quality = float(np.clip(pose_quality, 0.0, 1.0))
        valid_conf = np.asarray(det.keypoint_conf, dtype=np.float64).reshape(-1)
        valid_conf = valid_conf[np.isfinite(valid_conf)]
        mean_kpt_conf = float(np.mean(valid_conf)) if valid_conf.size else 0.0
        # 检测框置信度为主，Pose质量只作为加分；坏骨架仍可依靠bbox进入追踪。
        return float(0.68 * box_conf + 0.20 * pose_quality + 0.12 * mean_kpt_conf)

    def _candidate_allowed(self, det: Detection) -> bool:
        # A track-gap ROI is requested because a confirmed identity is missing.
        # It may recover that existing track, but must never accumulate into a
        # brand-new formal ID when the crop finds debris or a neighbouring mouse.
        if str(getattr(det, "detection_source", "")).lower() == "local_recovery_track_gap":
            return False
        score = self._candidate_score(det)
        box_conf = float(getattr(det, "box_conf", 0.0))
        pose_quality = float(getattr(det, "pose_quality", 0.0))
        if not np.isfinite(pose_quality) or pose_quality <= 0:
            pose_quality = _pose_quality_for_identity(det, min_conf=self.min_keypoint_conf)
        if box_conf < self.pending_min_box_conf:
            return False
        if pose_quality < self.pending_min_pose_quality:
            return False
        return score >= self.pending_min_candidate_score

    def _pending_cost(self, item: Mapping[str, Any], det: Detection) -> float:
        old_center = np.asarray(item["center"], dtype=np.float64)
        old_body = max(float(item["body"]), 8.0)
        det_body = self._robust_det_body(det)
        body = max(old_body, det_body, 8.0)
        dist = point_distance(old_center, det.center_px)
        if not np.isfinite(dist):
            return self.INF_COST
        center_cost = dist / max(body, 1e-6)
        if center_cost > self.pending_match_distance_bl:
            return self.INF_COST

        old_bbox = np.asarray(item.get("bbox", det.bbox_xyxy), dtype=np.float64)
        iou_cost = 1.0 - self._box_iou(old_bbox, det.bbox_xyxy)
        size_ratio = det_body / max(old_body, 1e-6)
        if size_ratio <= 0:
            return self.INF_COST
        size_cost = min(abs(math.log(size_ratio)), 2.0)
        return float(
            center_cost
            + self.pending_match_iou_weight * iou_cost
            + self.pending_match_size_weight * size_cost
        )

    def _update_count_estimate(self, visible: int) -> None:
        self.visible_mouse_count = int(visible)
        self.count_history.append(int(visible))
        # 已确认且尚未过期的轨迹数是总鼠数的稳定估计；可见数只反映当前帧。
        self.estimated_mouse_count = int(len(self.tracks))

    def _remove_stale_pending(self, frame: int) -> None:
        stale = [
            pid for pid, item in self.pending_candidates.items()
            if frame - int(item.get("last_frame", frame)) > self.pending_ttl_frames
        ]
        for pid in stale:
            self.pending_candidates.pop(pid, None)

    def _separated_from_confirmed(self, det: Detection, frame: int) -> bool:
        if not self.tracks:
            return True
        return self._separation_from_tracks(det, frame) >= self.later_promotion_min_separation_bl

    def _select_ready_pending(
        self,
        ready: Sequence[Tuple[int, Dict[str, Any]]],
        initial_batch: bool,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        if not ready:
            return []
        # 先按时序稳定度和平均质量选最可靠候选，避免重复框先占ID。
        ranked = sorted(
            ready,
            key=lambda x: (
                -len(x[1].get("hit_frames", [])),
                -float(x[1].get("average_score", 0.0)),
                -float(getattr(x[1].get("det"), "box_conf", 0.0)),
            ),
        )
        sep_gate = (
            self.initial_promotion_min_separation_bl
            if initial_batch else self.later_promotion_min_separation_bl
        )
        selected: List[Tuple[int, Dict[str, Any]]] = []
        for pid, item in ranked:
            det = item["det"]
            det_body = self._robust_det_body(det)
            duplicate = False
            for _, chosen in selected:
                other = chosen["det"]
                body = max(det_body, self._robust_det_body(other), 8.0)
                dist = point_distance(det.center_px, other.center_px)
                if np.isfinite(dist) and dist / body < sep_gate:
                    duplicate = True
                    break
            if not duplicate:
                selected.append((pid, item))
        if initial_batch:
            # 首批ID按画面空间顺序赋值，使每次运行的初始编号可重复。
            selected.sort(key=lambda x: self._initial_sort_key(x[1]["det"]))
        return selected

    def _process_pending(
        self,
        detections: Sequence[Detection],
        unmatched_indices: Sequence[int],
        frame: int,
    ) -> List[Tuple[int, Detection]]:
        if len(self.tracks) >= self.identity_capacity:
            self.pending_candidates.clear()
            return []

        self._remove_stale_pending(frame)
        candidates = [
            int(i) for i in unmatched_indices
            if self._candidate_allowed(detections[int(i)])
            and self._separated_from_confirmed(detections[int(i)], frame)
        ]
        pending_ids = sorted(self.pending_candidates)
        used_candidates: set[int] = set()

        if candidates and pending_ids:
            pcost = np.full((len(pending_ids), len(candidates)), self.INF_COST, dtype=np.float64)
            for r, pid in enumerate(pending_ids):
                item = self.pending_candidates[pid]
                for c, idx in enumerate(candidates):
                    pcost[r, c] = self._pending_cost(item, detections[idx])
            if linear_sum_assignment is not None and pcost.size:
                rr, cc = linear_sum_assignment(pcost)
                pairs = zip(rr.tolist(), cc.tolist())
            else:
                pairs = self._greedy_assignment(pcost)
            for r, c in pairs:
                chosen_cost = float(pcost[r, c])
                if not np.isfinite(chosen_cost) or chosen_cost > self.pending_max_match_cost:
                    continue
                pid = pending_ids[r]
                idx = candidates[c]
                item = self.pending_candidates[pid]
                det = detections[idx]
                last_frame = int(item.get("last_frame", frame))
                if frame - last_frame > self.pending_max_gap_frames:
                    item["hit_frames"] = []
                    item["scores"] = []
                hit_frames = list(item.get("hit_frames", []))
                if not hit_frames or hit_frames[-1] != frame:
                    hit_frames.append(int(frame))
                hit_frames = [f for f in hit_frames if frame - f < self.confirmation_window_frames]
                scores = list(item.get("scores", []))
                scores.append(self._candidate_score(det))
                scores = scores[-self.confirmation_window_frames :]
                item.update({
                    "center": np.asarray(det.center_px, dtype=np.float64).copy(),
                    "body": self._robust_det_body(det),
                    "bbox": np.asarray(det.bbox_xyxy, dtype=np.float64).copy(),
                    "last_frame": int(frame),
                    "det": det,
                    "hit_frames": hit_frames,
                    "scores": scores,
                    "average_score": float(np.mean(scores)) if scores else 0.0,
                })
                used_candidates.add(idx)

        for idx in candidates:
            if idx in used_candidates:
                continue
            det = detections[idx]
            pid = int(self.next_pending_id)
            self.next_pending_id += 1
            score = self._candidate_score(det)
            self.pending_candidates[pid] = {
                "center": np.asarray(det.center_px, dtype=np.float64).copy(),
                "body": self._robust_det_body(det),
                "bbox": np.asarray(det.bbox_xyxy, dtype=np.float64).copy(),
                "last_frame": int(frame),
                "det": det,
                "hit_frames": [int(frame)],
                "scores": [score],
                "average_score": score,
            }

        initial_batch = len(self.tracks) == 0
        required_hits = self.initial_confirm_hits if initial_batch else self.new_track_confirm_hits
        ready: List[Tuple[int, Dict[str, Any]]] = []
        for pid, item in self.pending_candidates.items():
            hit_frames = list(item.get("hit_frames", []))
            if not hit_frames or int(item.get("last_frame", -1)) != frame:
                continue
            recent_hits = [f for f in hit_frames if frame - f < self.confirmation_window_frames]
            avg_score = float(item.get("average_score", 0.0))
            if len(recent_hits) >= required_hits and avg_score >= self.pending_min_average_score:
                ready.append((pid, item))

        selected = self._select_ready_pending(ready, initial_batch=initial_batch)
        limit = self.initial_max_new_tracks_per_frame if initial_batch else self.max_new_tracks_per_frame
        output: List[Tuple[int, Detection]] = []
        for pid, item in selected[:limit]:
            if len(self.tracks) >= self.identity_capacity:
                break
            det = item["det"]
            lid = self._new_track(det, frame)
            if lid is not None:
                output.append((lid, det))
            self.pending_candidates.pop(pid, None)
        return output

    def assign(
        self,
        detections: Sequence[Detection],
        frame: int,
        occlusion_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[int, Detection]]:
        # v1.22.1：必须先读取当前帧聚集冻结集合，再执行过期清理。
        # 旧顺序先_expire()，刚建立但已进入聚集的轨迹会在第12帧左右被
        # weak_track_drop误删，导致自适应鼠数从6降为4且散开后无法恢复原ID。
        detections = list(detections)
        self.last_cost_build_seconds = 0.0
        self.last_assignment_seconds = 0.0
        context = dict(occlusion_context or {})
        self._current_frozen_ids = set(int(x) for x in context.get("frozen_ids", set()))
        self._current_reserved_detection_indices = set(int(x) for x in context.get("reserved_detection_indices", set()))
        self._expire(frame)
        self.disk_sequence_guard.prune(sorted(self.tracks))
        self._disk_cost_cache_frame = int(frame)
        self._disk_cost_cache = {}
        self._disk_contact_order_veto_ids = set()
        self._contact_order_regret_veto_ids = set()
        self.output_info = {}
        self.debug_records = []
        stats = {
            "raw": len(detections), "after_conf": len(detections), "after_kpt_filter": len(detections),
            "matched": 0, "low_rescued": 0, "lost_recovered": 0, "new_tentative": 0,
            "unmatched_det": 0, "active": len(self.tracks), "tentative": len(self.pending_candidates),
            "suspicious": 0, "lost": 0, "rendered": 0,
            "adaptive_estimated_count": len(self.tracks), "adaptive_visible_count": 0,
            "cluster_frozen_count": len(self._current_frozen_ids),
            "cluster_reserved_detection_count": len(self._current_reserved_detection_indices),
            "disk_sequence_active_tracks": 0,
            "disk_sequence_mean_reliability": 0.0,
            "disk_sequence_rejected_updates": int(
                self.disk_sequence_guard.rejected_updates
            ),
            "disk_contact_order_veto_count": 0,
            "disk_contact_order_veto_total": int(
                self.disk_contact_order_veto_count
            ),
            "contact_order_regret_veto_count": 0,
            "contact_order_regret_veto_total": int(
                self.contact_order_regret_veto_count
            ),
        }
        self._refresh_contact_guard(detections, frame)

        if not detections:
            for lid, track in self.tracks.items():
                if lid in self._current_frozen_ids:
                    track.state = "cluster_occluded"
                    continue
                self.kpt_missing[lid] = int(self.kpt_missing.get(lid, 0)) + 1
                track.lock_strength = max(0.0, track.lock_strength - 0.01)
                if self.kpt_missing[lid] > self.max_missing_frames:
                    track.state = "lost"
            self._remove_stale_pending(frame)
            self._update_count_estimate(0)
            stats["active"] = len(self.tracks)
            stats["tentative"] = len(self.pending_candidates)
            stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
            stats["adaptive_estimated_count"] = self.estimated_mouse_count
            stats["disk_sequence_rejected_updates"] = int(
                self.disk_sequence_guard.rejected_updates
            )
            self.frame_stats = stats
            return []

        available_indices = [i for i in range(len(detections)) if i not in self._current_reserved_detection_indices]
        # 没有已确认轨迹时也先做时序确认；簇保留区域内候选绝不创建新ID。
        if not self.tracks:
            new_output = self._process_pending(detections, available_indices, frame)
            output: Dict[int, Detection] = {}
            for lid, det in new_output:
                output[lid] = det
                self.output_info[lid] = {
                    "state": "tracked", "label": f"ID {lid}", "cost": 0.0,
                    "method": "adaptive_initial_confirmed",
                }
            self._update_count_estimate(len(output))
            stats["matched"] = len(output)
            stats["new_tentative"] = len(output)
            stats["unmatched_det"] = max(0, len(available_indices) - len(output))
            stats["active"] = len(self.tracks)
            stats["tentative"] = len(self.pending_candidates)
            stats["rendered"] = len(output)
            stats["adaptive_estimated_count"] = self.estimated_mouse_count
            stats["adaptive_visible_count"] = self.visible_mouse_count
            stats["disk_sequence_rejected_updates"] = int(
                self.disk_sequence_guard.rejected_updates
            )
            self.frame_stats = stats
            return sorted(output.items(), key=lambda item: item[0])

        track_ids = sorted(self.tracks)
        ns, nd = len(track_ids), len(detections)
        cost = np.full((ns, nd + ns), self.INF_COST, dtype=np.float64)
        cost_started = time.perf_counter()
        detection_cost = self._detection_cost_matrix(track_ids, detections, frame)
        for r, lid in enumerate(track_ids):
            if lid not in self._current_frozen_ids:
                for c in range(nd):
                    if c in self._current_reserved_detection_indices:
                        continue
                    recovery_target = int(
                        getattr(detections[c], "recovery_target_logical_id", -1)
                    )
                    if recovery_target >= 0 and int(lid) != recovery_target:
                        # Keep detection_cost untouched: the row/column margin
                        # below must still see real competitors.  Only route the
                        # Hungarian edge to the ID whose missing-track ROI
                        # produced this candidate.
                        continue
                    cost[r, c] = float(detection_cost[r, c])
            missing = int(self.kpt_missing.get(lid, 0))
            cost[r, nd + r] = 0.01 if lid in self._current_frozen_ids else (
                self.unmatched_cost + self.unmatched_cost_growth * min(missing, self.prediction_max_frames)
            )
        self.last_cost_build_seconds = time.perf_counter() - cost_started

        assignment_started = time.perf_counter()
        if linear_sum_assignment is not None:
            rr, cc = linear_sum_assignment(cost)
            proposed = list(zip(rr.tolist(), cc.tolist()))
        else:
            proposed = self._greedy_assignment(cost)
        # Enforce the pre-contact physical order before any track template is updated.
        proposed = self._repair_contact_order_assignments(
            proposed,
            cost,
            track_ids,
            detections,
            frame,
        )
        self.last_assignment_seconds = time.perf_counter() - assignment_started

        output: Dict[int, Detection] = {}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        blocked_ambiguous_detections: set[int] = set()
        for r, c in proposed:
            lid = track_ids[r]
            if c >= nd or lid in self._current_frozen_ids or c in self._current_reserved_detection_indices:
                continue
            chosen = float(cost[r, c])
            det = detections[c]
            contact_guarded = self._contact_guarded(lid, det, frame)
            assignment_limit = (
                min(
                    self.max_assignment_cost,
                    self.contact_order_max_assignment_cost,
                )
                if lid in self._contact_order_repaired_ids
                else
                min(
                    self.max_assignment_cost,
                    self.contact_guard_max_assignment_cost,
                )
                if contact_guarded
                else self.max_assignment_cost
            )
            if (
                not np.isfinite(chosen)
                or chosen >= self.INF_COST
                or chosen > assignment_limit
            ):
                continue
            was_missing = int(self.kpt_missing.get(lid, 0))
            margin = self._assignment_row_column_margin(detection_cost, r, c)
            recovery_source = str(getattr(det, "detection_source", "")) in {
                "local_recovery_track_gap",
                "local_recovery",
                "occlusion_recovery",
            }
            minimum_margin = (
                self.assignment_contact_min_margin
                if contact_guarded
                else self.assignment_recovery_min_margin
                if was_missing > 0 or recovery_source
                else self.assignment_general_min_margin
            )
            jump_bl, allowed_jump_bl = self._assignment_jump_body_lengths(
                lid,
                det,
                frame,
            )
            strong_unique = bool(
                chosen <= self.assignment_unique_override_max_cost
                and margin >= self.assignment_unique_override_margin
            )
            # The continuation exception is deliberately narrower than the
            # ordinary assignment gate: no prior miss, no recovery ROI, no
            # negative two-sided margin, a small predicted displacement, and a
            # bounded total cost are all required.  Negative margins remain
            # rejected because they mean another row/column locally prefers
            # this detection, which is the classic ID-exchange configuration.
            ambiguous_motion_hold = bool(
                self.assignment_ambiguity_enabled
                and self.assignment_motion_hold_enabled
                and margin < minimum_margin
                and margin >= self.assignment_motion_hold_min_margin
                and was_missing == 0
                and not recovery_source
                and jump_bl <= self.assignment_motion_hold_max_jump_bl
                and chosen <= self.assignment_motion_hold_max_cost
            )
            rejection_reason = ""
            if (
                self.assignment_ambiguity_enabled
                and margin < minimum_margin
                and not ambiguous_motion_hold
            ):
                rejection_reason = (
                    f"row_col_margin={margin:.3f}<{minimum_margin:.3f}"
                )
            elif (
                self.assignment_ambiguity_enabled
                and jump_bl > allowed_jump_bl
                and not strong_unique
            ):
                rejection_reason = (
                    f"jump={jump_bl:.3f}BL>{allowed_jump_bl:.3f}BL"
                )
            if rejection_reason:
                # The detection remains available to the provisional display
                # tracker, but may not create/update a formal slot this frame.
                blocked_ambiguous_detections.add(int(c))
                self.debug_records.append(IdentityDebug(
                    frame=int(frame),
                    logical_id=int(lid),
                    raw_track_id=det.raw_track_id,
                    assignment_cost=float(chosen),
                    proposed_logical_id=int(lid),
                    assignment_gain=float(margin),
                    commit_status="rejected_ambiguous_assignment",
                    switch_rejected_reason=rejection_reason,
                    appearance_mode=det.appearance_mode,
                    detection_source=det.detection_source,
                    track_state="lost" if was_missing > 0 else "tracked",
                ))
                continue
            # Preserve the exact scalar legacy value written to diagnostics/trajectory CSVs.
            # Compute it before mutating the track state.
            persisted_cost = float(self._cost(lid, self.tracks[lid], det, frame))
            if contact_guarded or ambiguous_motion_hold:
                self._update_track_contact_safe(lid, det, frame)
            else:
                self._update_track(lid, det, frame)
            output[lid] = det
            matched_tracks.add(lid); matched_detections.add(c)
            method = (
                "adaptive_ambiguous_motion_hold"
                if ambiguous_motion_hold
                else
                "adaptive_contact_guard_recovered"
                if contact_guarded and was_missing > 0
                else "adaptive_contact_guard"
                if contact_guarded
                else "adaptive_recovered"
                if was_missing > 0
                else "adaptive_hungarian_mask"
            )
            if (
                not contact_guarded
                and self._disk_last_reliability_matrix.shape == (ns, nd)
                and float(self._disk_last_reliability_matrix[r, c])
                >= self.disk_sequence_guard.min_reliability
            ):
                method += "_disk_sequence"
            if lid in self._disk_contact_order_veto_ids:
                method += "_disk_order_veto"
            if lid in self._contact_order_regret_veto_ids:
                method += "_contact_regret_veto"
            # The vectorized matrix drives assignment, while persisted_cost keeps
            # byte-for-byte compatibility with the legacy scalar diagnostics.
            self.output_info[lid] = {
                "state": "tracked", "label": f"ID {lid}", "cost": persisted_cost, "method": method,
            }
            stats["matched"] += 1
            if was_missing > 0:
                stats["lost_recovered"] += 1

        for lid in track_ids:
            if lid in matched_tracks:
                continue
            track = self.tracks[lid]
            if lid in self._current_frozen_ids:
                track.state = "cluster_occluded"
                continue
            self.kpt_missing[lid] = int(self.kpt_missing.get(lid, 0)) + 1
            track.lock_strength = max(0.0, track.lock_strength - 0.01)
            if self.kpt_missing[lid] > self.max_missing_frames:
                track.state = "lost"

        unmatched = [
            i
            for i in available_indices
            if i not in matched_detections
            and i not in blocked_ambiguous_detections
        ]
        new_output = self._process_pending(detections, unmatched, frame)
        for lid, det in new_output:
            output[lid] = det
            self.output_info[lid] = {
                "state": "tracked", "label": f"ID {lid}", "cost": 0.0,
                "method": "adaptive_new_track_confirmed",
            }
            stats["new_tentative"] += 1

        self._update_count_estimate(len(output))
        stats["unmatched_det"] = len(unmatched)
        stats["active"] = len(self.tracks)
        stats["tentative"] = len(self.pending_candidates)
        stats["lost"] = sum(1 for t in self.tracks.values() if t.state == "lost")
        stats["rendered"] = len(output)
        stats["adaptive_estimated_count"] = self.estimated_mouse_count
        stats["adaptive_visible_count"] = self.visible_mouse_count
        stats["disk_sequence_active_tracks"] = int(
            self.disk_sequence_guard.last_active_tracks
        )
        stats["disk_sequence_mean_reliability"] = float(
            self.disk_sequence_guard.last_mean_reliability
        )
        stats["disk_sequence_rejected_updates"] = int(
            self.disk_sequence_guard.rejected_updates
        )
        stats["disk_contact_order_veto_count"] = int(
            len(self._disk_contact_order_veto_ids)
        )
        stats["disk_contact_order_veto_total"] = int(
            self.disk_contact_order_veto_count
        )
        stats["contact_order_regret_veto_count"] = int(
            len(self._contact_order_regret_veto_ids)
        )
        stats["contact_order_regret_veto_total"] = int(
            self.contact_order_regret_veto_count
        )
        self.frame_stats = stats
        return sorted(output.items(), key=lambda item: item[0])


BASE_MODULE_VERSION = "1.42.1-final-code-merge"
