#!/usr/bin/env python3
"""Apply deterministic, result-preserving v1.40.1 optimizations to v1.40.0 files."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "original"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def extract_method(
    source: str,
    class_marker: str,
    method_marker: str,
    next_method_marker: str,
) -> str:
    """Extract one indented method block from the pristine source file."""
    class_start = source.index(class_marker)
    start = source.index(method_marker, class_start)
    end = source.index(next_method_marker, start + len(method_marker))
    return source[start:end]


def optimize_base() -> None:
    path = ROOT / "mouse_chase_attack_extractor_base.py"
    source = (ORIGINAL / path.name).read_text(encoding="utf-8")
    text = source
    text = replace_once(text, "import sys\nimport time\n", "import sys\nimport threading\nimport time\n", "base threading import")
    text = text.replace(
        'BASE_MODULE_VERSION = "1.40.0-ambiguity-margin-persistent-slots"',
        'BASE_MODULE_VERSION = "1.40.1-performance-preserving"',
    )
    clahe_helper = '''VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}\n\n\n# OpenCV CLAHE objects allocate internal lookup tables and temporary buffers.\n# Reusing one object per worker thread is mathematically identical to creating\n# one per detection, while avoiding thousands of allocator calls per video.\n_APPEARANCE_THREAD_LOCAL = threading.local()\n\n\ndef _get_appearance_clahe() -> Any:\n    clahe = getattr(_APPEARANCE_THREAD_LOCAL, "clahe", None)\n    if clahe is None:\n        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))\n        _APPEARANCE_THREAD_LOCAL.clahe = clahe\n    return clahe\n'''
    text = replace_once(
        text,
        'VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}\n',
        clahe_helper,
        "base CLAHE helper",
    )

    detection_start = text.index("@dataclass\nclass Detection:")
    detection_end = text.index("@dataclass\nclass IdentityTrack:", detection_start)
    original_detection = text[detection_start:detection_end]
    fields_end = original_detection.index("    @property\n    def center_px")
    fields = original_detection[:fields_end]
    optimized_methods = '''    # v1.40.1：在Pose恢复和去重完成后显式刷新的一帧内派生几何缓存。
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


'''
    text = text[:detection_start] + fields + optimized_methods + text[detection_end:]

    text = replace_once(
        text,
        '    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))\n    work = clahe.apply(work)\n',
        '    work = _get_appearance_clahe().apply(work)\n',
        "base CLAHE reuse",
    )
    text = replace_once(
        text,
        '    detections = list(detections)\n    if len(detections) > 1:\n',
        '    detections = list(detections)\n'
        '    # Pose恢复、候选融合和去重均已完成；从这里开始几何只读。显式刷新可避免\n'
        '    # cluster、mask和identity在同一帧重复构造valid mask/加权中心/鼻尾体长。\n'
        '    for det in detections:\n'
        '        det.refresh_derived_geometry_cache()\n'
        '    if len(detections) > 1:\n',
        "base geometry refresh",
    )

    recent_ids_block = '''        recent_ids = [\n            lid for lid, tr in tracks_map.items()\n            if frame - tr.last_frame <= self.track_max_age and finite_point(tr.last_center_px)\n        ]\n'''
    cache_block = recent_ids_block + '''        # The same two-member group is evaluated once while building edges and\n        # again when materializing its component. Cache within this immutable\n        # pre-assignment frame context; no state or threshold is changed.\n        evidence_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}\n\n        def evidence_for(members: Sequence[int]) -> Dict[str, Any]:\n            key = tuple(sorted(int(member) for member in members))\n            cached = evidence_cache.get(key)\n            if cached is None:\n                cached = self._member_detection_evidence(\n                    key, tracks_map, assigner, detections, frame\n                )\n                evidence_cache[key] = cached\n            return cached\n\n        active_state_pairs = {\n            tuple(sorted((int(a), int(b))))\n            for state_members in self.states\n            for index, a in enumerate(state_members)\n            for b in state_members[index + 1 :]\n        }\n'''
    text = replace_once(text, recent_ids_block, cache_block, "cluster evidence cache insert")
    text = replace_once(
        text,
        '                pair_was_active = any(a in key and b in key for key in self.states)\n',
        '                pair_was_active = tuple(sorted((int(a), int(b)))) in active_state_pairs\n',
        "cluster active pair lookup",
    )
    text = replace_once(
        text,
        '                pair_evidence = self._member_detection_evidence(\n                    (a, b), tracks_map, assigner, detections, frame\n                )\n',
        '                pair_evidence = evidence_for((a, b))\n',
        "cluster pair evidence reuse",
    )
    text = replace_once(
        text,
        '            evidence = self._member_detection_evidence(\n                members, tracks_map, assigner, detections, frame\n            )\n',
        '            evidence = evidence_for(members)\n',
        "cluster component evidence reuse",
    )

    history_start = text.index("class ObservationHistory:")
    history_end = text.index("\ndef derive_geometry(", history_start)
    optimized_history = '''class ObservationHistory:\n    """按逻辑ID保存有限长度的观测历史。\n\n    高召回主程序依赖以下接口：previous/add/get/near_frame。\n    历史使用deque限长保存，不会随视频时长无限增长。\n    """\n\n    def __init__(self, max_frames: int) -> None:\n        self.max_frames = max(int(max_frames), 1)\n        self.data: Dict[int, Deque[MouseObservation]] = defaultdict(\n            lambda: deque(maxlen=self.max_frames)\n        )\n        # A frame evaluates every logical ID against many partners. Cache only\n        # immutable read views and near-frame lookups; adding an observation\n        # invalidates caches for that ID before pair calculation begins.\n        self._window_cache: Dict[int, Dict[int, Tuple[MouseObservation, ...]]] = {}\n        self._near_cache: Dict[int, Dict[int, Optional[MouseObservation]]] = {}\n\n    def __getstate__(self) -> Dict[str, Any]:\n        # Read caches are derivable and may duplicate observations in checkpoints.\n        state = dict(self.__dict__)\n        state["_window_cache"] = {}\n        state["_near_cache"] = {}\n        return state\n\n    def __setstate__(self, state: Mapping[str, Any]) -> None:\n        self.__dict__.update(dict(state))\n        self._window_cache = {}\n        self._near_cache = {}\n\n    def previous(self, logical_id: int) -> Optional[MouseObservation]:\n        history = self.data.get(int(logical_id))\n        return history[-1] if history else None\n\n    def add(self, observation: MouseObservation) -> None:\n        logical_id = int(observation.logical_id)\n        self.data[logical_id].append(observation)\n        self._window_cache.pop(logical_id, None)\n        self._near_cache.pop(logical_id, None)\n\n    def get(self, logical_id: int) -> List[MouseObservation]:\n        # Preserve the historical API contract: callers receive a mutable copy.\n        return list(self.data.get(int(logical_id), []))\n\n    def get_window(\n        self, logical_id: int, max_items: int\n    ) -> Tuple[MouseObservation, ...]:\n        """Return a cached read-only tail window for pair-feature hot loops."""\n        logical_id = int(logical_id)\n        max_items = max(int(max_items), 0)\n        by_size = self._window_cache.setdefault(logical_id, {})\n        cached = by_size.get(max_items)\n        if cached is not None:\n            return cached\n        history = self.data.get(logical_id)\n        if not history or max_items == 0:\n            result: Tuple[MouseObservation, ...] = ()\n        else:\n            values = tuple(history)\n            result = values[-max_items:] if len(values) > max_items else values\n        by_size[max_items] = result\n        return result\n\n    def near_frame(self, logical_id: int, frame: int) -> Optional[MouseObservation]:\n        logical_id = int(logical_id)\n        target_frame = int(frame)\n        cache = self._near_cache.setdefault(logical_id, {})\n        if target_frame in cache:\n            return cache[target_frame]\n        history = self.data.get(logical_id)\n        result: Optional[MouseObservation] = None\n        if history:\n            # deque长度很小（通常约1秒），反向寻找最近且不晚于目标帧的观测。\n            for obs in reversed(history):\n                if obs.frame <= target_frame:\n                    result = obs\n                    break\n        cache[target_frame] = result\n        return result\n\n'''
    text = text[:history_start] + optimized_history + text[history_end:]

    base_init_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def __init__(",
        "    def _trajectory_features(",
    )
    base_init_new = base_init_old + '''        self._cache_frame = -1\n        self._history_map_cache: Dict[int, Dict[int, MouseObservation]] = {}\n        self._trajectory_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}\n        self._target_turn_cache: Dict[int, float] = {}\n        self._distance_drop_cache: Dict[Tuple[int, int], float] = {}\n\n    def _ensure_frame_cache(self, frame: int) -> None:\n        frame = int(frame)\n        if frame == self._cache_frame:\n            return\n        self._cache_frame = frame\n        self._history_map_cache.clear()\n        self._trajectory_cache.clear()\n        self._target_turn_cache.clear()\n        self._distance_drop_cache.clear()\n\n    def _history_map(\n        self, logical_id: int, history: ObservationHistory\n    ) -> Dict[int, MouseObservation]:\n        logical_id = int(logical_id)\n        cached = self._history_map_cache.get(logical_id)\n        if cached is not None:\n            return cached\n        values = history.get_window(logical_id, self.history_frames)\n        cached = {observation.frame: observation for observation in values}\n        self._history_map_cache[logical_id] = cached\n        return cached\n\n'''
    text = replace_once(text, base_init_old, base_init_new, "base pair init/cache helpers")

    base_traj_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def _trajectory_features(",
        "    def _distance_drop(",
    )
    base_traj_new = '''    def _trajectory_features(\n        self,\n        actor_id: int,\n        target_id: int,\n        history: ObservationHistory,\n        frame: int,\n    ) -> Tuple[float, float, float]:\n        self._ensure_frame_cache(frame)\n        key = (int(actor_id), int(target_id))\n        cached = self._trajectory_cache.get(key)\n        if cached is not None:\n            return cached\n        actor_by_frame = self._history_map(actor_id, history)\n        target_by_frame = self._history_map(target_id, history)\n        common_frames = sorted(set(actor_by_frame) & set(target_by_frame))\n        actor_centers = np.empty((0, 2), dtype=np.float64)\n        target_centers = np.empty((0, 2), dtype=np.float64)\n        da = np.empty((0, 2), dtype=np.float64)\n        dt = np.empty((0, 2), dtype=np.float64)\n        if len(common_frames) < 4:\n            result = (0.0, 0.0, 0.0)\n        else:\n            actor_centers = np.stack([actor_by_frame[f].center_cm for f in common_frames])\n            target_centers = np.stack([target_by_frame[f].center_cm for f in common_frames])\n            valid = np.all(np.isfinite(actor_centers), axis=1) & np.all(\n                np.isfinite(target_centers), axis=1\n            )\n            actor_centers = actor_centers[valid]\n            target_centers = target_centers[valid]\n            if len(actor_centers) < 4:\n                result = (0.0, 0.0, 0.0)\n            else:\n                da = np.diff(actor_centers, axis=0)\n                dt = np.diff(target_centers, axis=0)\n                corr_x = safe_corr(da[:, 0], dt[:, 0])\n                corr_y = safe_corr(da[:, 1], dt[:, 1])\n                corr = float(np.clip((corr_x + corr_y) / 2.0, -1.0, 1.0))\n                result = (\n                    corr,\n                    float(np.sum(np.linalg.norm(da, axis=1))),\n                    float(np.sum(np.linalg.norm(dt, axis=1))),\n                )\n        self._trajectory_cache[key] = result\n        if key[0] != key[1]:\n            # Recompute correlation in the historical reversed argument order.\n            # np.corrcoef can differ by one ULP when inputs are swapped.\n            if len(common_frames) < 4 or len(actor_centers) < 4:\n                reverse_corr = 0.0\n            else:\n                reverse_corr_x = safe_corr(dt[:, 0], da[:, 0])\n                reverse_corr_y = safe_corr(dt[:, 1], da[:, 1])\n                reverse_corr = float(\n                    np.clip((reverse_corr_x + reverse_corr_y) / 2.0, -1.0, 1.0)\n                )\n            self._trajectory_cache[(key[1], key[0])] = (\n                reverse_corr, result[2], result[1]\n            )\n        return result\n\n'''
    text = replace_once(text, base_traj_old, base_traj_new, "base trajectory cache")

    base_drop_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def _distance_drop(",
        "    def _target_turn_angle(",
    )
    base_drop_new = '''    def _distance_drop(\n        self,\n        actor: MouseObservation,\n        target: MouseObservation,\n        history: ObservationHistory,\n    ) -> float:\n        self._ensure_frame_cache(actor.frame)\n        key = tuple(sorted((int(actor.logical_id), int(target.logical_id))))\n        cached = self._distance_drop_cache.get(key)\n        if cached is not None:\n            return cached\n        old_frame = actor.frame - self.lookback_frames\n        old_actor = history.near_frame(actor.logical_id, old_frame)\n        old_target = history.near_frame(target.logical_id, old_frame)\n        current_distance = point_distance(actor.center_cm, target.center_cm)\n        if old_actor is None or old_target is None or not np.isfinite(current_distance):\n            value = 0.0\n        else:\n            previous_distance = point_distance(old_actor.center_cm, old_target.center_cm)\n            value = (\n                float(previous_distance - current_distance)\n                if np.isfinite(previous_distance)\n                else 0.0\n            )\n        self._distance_drop_cache[key] = float(value)\n        return float(value)\n\n'''
    text = replace_once(text, base_drop_old, base_drop_new, "base distance drop cache")

    base_turn_old = '''    def _target_turn_angle(\n        self,\n        target: MouseObservation,\n        history: ObservationHistory,\n    ) -> float:\n        old = history.near_frame(target.logical_id, target.frame - self.lookback_frames)\n        if old is None:\n            return 0.0\n        return angle_difference_deg(old.heading, target.heading)\n\n'''
    base_turn_new = '''    def _target_turn_angle(\n        self,\n        target: MouseObservation,\n        history: ObservationHistory,\n    ) -> float:\n        self._ensure_frame_cache(target.frame)\n        logical_id = int(target.logical_id)\n        cached = self._target_turn_cache.get(logical_id)\n        if cached is not None:\n            return cached\n        old = history.near_frame(logical_id, target.frame - self.lookback_frames)\n        value = 0.0 if old is None else angle_difference_deg(old.heading, target.heading)\n        self._target_turn_cache[logical_id] = float(value)\n        return float(value)\n\n'''
    text = replace_once(text, base_turn_old, base_turn_new, "base target turn cache")
    text = replace_once(
        text,
        '            actor.logical_id, target.logical_id, history\n        )\n',
        '            actor.logical_id, target.logical_id, history, actor.frame\n        )\n',
        "base trajectory call",
    )
    path.write_text(text, encoding="utf-8")


def optimize_main() -> None:
    path = ROOT / "mouse_chase_attack_high_recall.py"
    source = (ORIGINAL / path.name).read_text(encoding="utf-8")
    text = source
    text = replace_once(
        text,
        'PROGRAM_VERSION = "1.40.0-two-stage-arena-id-stability"',
        'PROGRAM_VERSION = "1.40.1-performance-preserving"',
        "main version",
    )

    main_init_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def __init__(",
        "    def _trajectory_features(",
    )
    main_init_new = main_init_old + '''        # Per-frame memoization is safe because history is fully populated\n        # before pair enumeration and is not mutated until the next frame.\n        self._cache_frame = -1\n        self._history_map_cache: Dict[int, Dict[int, base.MouseObservation]] = {}\n        self._trajectory_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}\n        self._target_turn_cache: Dict[int, float] = {}\n        self._distance_drop_cache: Dict[Tuple[int, int], float] = {}\n\n    def _ensure_frame_cache(self, frame: int) -> None:\n        frame = int(frame)\n        if frame == self._cache_frame:\n            return\n        self._cache_frame = frame\n        self._history_map_cache.clear()\n        self._trajectory_cache.clear()\n        self._target_turn_cache.clear()\n        self._distance_drop_cache.clear()\n\n    def _history_map(\n        self, logical_id: int, history: base.ObservationHistory\n    ) -> Dict[int, base.MouseObservation]:\n        logical_id = int(logical_id)\n        cached = self._history_map_cache.get(logical_id)\n        if cached is not None:\n            return cached\n        if hasattr(history, "get_window"):\n            values = history.get_window(logical_id, self.history_frames)\n        else:\n            values = history.get(logical_id)[-self.history_frames :]\n        cached = {observation.frame: observation for observation in values}\n        self._history_map_cache[logical_id] = cached\n        return cached\n\n'''
    text = replace_once(text, main_init_old, main_init_new, "main pair init/cache helpers")

    main_traj_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def _trajectory_features(",
        "    def _distance_drop(",
    )
    main_traj_new = '''    def _trajectory_features(\n        self,\n        actor_id: int,\n        target_id: int,\n        history: base.ObservationHistory,\n        frame: int,\n    ) -> Tuple[float, float, float]:\n        self._ensure_frame_cache(frame)\n        key = (int(actor_id), int(target_id))\n        cached = self._trajectory_cache.get(key)\n        if cached is not None:\n            return cached\n        actor_by_frame = self._history_map(actor_id, history)\n        target_by_frame = self._history_map(target_id, history)\n        common = sorted(set(actor_by_frame) & set(target_by_frame))\n        a = np.empty((0, 2), dtype=np.float64)\n        b = np.empty((0, 2), dtype=np.float64)\n        da = np.empty((0, 2), dtype=np.float64)\n        db = np.empty((0, 2), dtype=np.float64)\n        if len(common) < 4:\n            result = (0.0, 0.0, 0.0)\n        else:\n            a = np.stack([actor_by_frame[f].center_cm for f in common])\n            b = np.stack([target_by_frame[f].center_cm for f in common])\n            valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)\n            a, b = a[valid], b[valid]\n            if len(a) < 4:\n                result = (0.0, 0.0, 0.0)\n            else:\n                da, db = np.diff(a, axis=0), np.diff(b, axis=0)\n                # 对水平/垂直直线运动，单轴可能是常数，逐轴Pearson会错误给0。\n                # 展平二维位移后再计算相关性，保持“同轨迹>0.7”的文档含义。\n                corr = float(\n                    np.clip(safe_corr(da.reshape(-1), db.reshape(-1)), -1.0, 1.0)\n                )\n                result = (\n                    corr,\n                    float(np.linalg.norm(da, axis=1).sum()),\n                    float(np.linalg.norm(db, axis=1).sum()),\n                )\n        self._trajectory_cache[key] = result\n        if key[0] != key[1]:\n            # np.corrcoef can differ by one ULP when its arguments are swapped.\n            # Reproduce the historical directed order so CSV values stay exact.\n            if len(common) < 4 or len(a) < 4:\n                reverse_corr = 0.0\n            else:\n                reverse_corr = float(\n                    np.clip(safe_corr(db.reshape(-1), da.reshape(-1)), -1.0, 1.0)\n                )\n            self._trajectory_cache[(key[1], key[0])] = (\n                reverse_corr, result[2], result[1]\n            )\n        return result\n\n'''
    text = replace_once(text, main_traj_old, main_traj_new, "main trajectory cache")

    main_drop_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def _distance_drop(",
        "    def _target_turn(",
    )
    main_drop_new = '''    def _distance_drop(\n        self,\n        actor: base.MouseObservation,\n        target: base.MouseObservation,\n        history: base.ObservationHistory,\n    ) -> float:\n        self._ensure_frame_cache(actor.frame)\n        key = tuple(sorted((int(actor.logical_id), int(target.logical_id))))\n        cached = self._distance_drop_cache.get(key)\n        if cached is not None:\n            return cached\n        old_frame = actor.frame - self.lookback_frames\n        old_actor = history.near_frame(actor.logical_id, old_frame)\n        old_target = history.near_frame(target.logical_id, old_frame)\n        current = point_distance(actor.center_cm, target.center_cm)\n        if old_actor is None or old_target is None or not np.isfinite(current):\n            value = 0.0\n        else:\n            previous = point_distance(old_actor.center_cm, old_target.center_cm)\n            value = float(previous - current) if np.isfinite(previous) else 0.0\n        self._distance_drop_cache[key] = float(value)\n        return float(value)\n\n'''
    text = replace_once(text, main_drop_old, main_drop_new, "main distance drop cache")

    main_turn_old = extract_method(
        source,
        "class PairFeatureComputer:",
        "    def _target_turn(",
        "    def _evaluate_chase(",
    )
    # Extracted block includes the following @staticmethod line; retain it once.
    main_turn_new = '''    def _target_turn(\n        self,\n        target: base.MouseObservation,\n        history: base.ObservationHistory,\n    ) -> float:\n        self._ensure_frame_cache(target.frame)\n        logical_id = int(target.logical_id)\n        cached = self._target_turn_cache.get(logical_id)\n        if cached is not None:\n            return cached\n        old = history.near_frame(logical_id, target.frame - self.lookback_frames)\n        value = 0.0 if old is None else angle_difference_deg(old.heading, target.heading)\n        self._target_turn_cache[logical_id] = float(value)\n        return float(value)\n\n    @staticmethod\n'''
    text = replace_once(text, main_turn_old, main_turn_new, "main target turn cache")
    text = replace_once(
        text,
        '            actor.logical_id, target.logical_id, history\n        )\n',
        '            actor.logical_id, target.logical_id, history, actor.frame\n        )\n',
        "main trajectory call",
    )

    add_old = '''    def add(self, row: Mapping[str, Any]) -> None:\n        clean: Dict[str, Any] = {}\n        for key, value in row.items():\n            value = to_builtin(value)\n            if isinstance(value, float) and not np.isfinite(value):\n                value = None\n            elif isinstance(value, bool):\n                value = int(value)\n            clean[str(key)] = value\n        self.buffer.append(clean)\n        if len(self.buffer) >= self.batch_size:\n            self.flush()\n\n'''
    add_new = '''    @staticmethod\n    def _clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:\n        clean: Dict[str, Any] = {}\n        for key, value in row.items():\n            value = to_builtin(value)\n            if isinstance(value, float) and not np.isfinite(value):\n                value = None\n            elif isinstance(value, bool):\n                value = int(value)\n            clean[str(key)] = value\n        return clean\n\n    def add(self, row: Mapping[str, Any]) -> None:\n        self.buffer.append(self._clean_row(row))\n        if len(self.buffer) >= self.batch_size:\n            self.flush()\n\n    def add_many(self, rows: Iterable[Mapping[str, Any]]) -> None:\n        """Append one deterministic frame/chunk with fewer call boundaries."""\n        clean_row = self._clean_row\n        for row in rows:\n            self.buffer.append(clean_row(row))\n            if len(self.buffer) >= self.batch_size:\n                self.flush()\n\n'''
    text = replace_once(text, add_old, add_new, "SQLite add_many")

    text = replace_once(
        text,
        '    pair_cluster_evidence: Mapping[str, Any],\n) -> Dict[str, Any]:\n',
        '    pair_cluster_evidence: Mapping[str, Any],\n'
        '    record_template: Optional[Mapping[str, Any]] = None,\n'
        ') -> Dict[str, Any]:\n',
        "pair record template signature",
    )
    text = replace_once(
        text,
        '    record = empty_frame_record(frame_idx, fps, transformer)\n    record.update({\n',
        '    record = (\n'
        '        dict(record_template)\n'
        '        if record_template is not None\n'
        '        else empty_frame_record(frame_idx, fps, transformer)\n'
        '    )\n'
        '    record.update({\n',
        "pair record template use",
    )
    text = replace_once(
        text,
        '    cluster_reid_active_count: int,\n) -> Dict[str, Any]:\n',
        '    cluster_reid_active_count: int,\n'
        '    record_template: Optional[Mapping[str, Any]] = None,\n'
        ') -> Dict[str, Any]:\n',
        "fallback template signature",
    )
    # The next occurrence is the fallback record initialization.
    text = replace_once(
        text,
        '    """Create the legacy missing-detection cluster fallback row."""\n'
        '    record = empty_frame_record(frame_idx, fps, transformer)\n',
        '    """Create the legacy missing-detection cluster fallback row."""\n'
        '    record = (\n'
        '        dict(record_template)\n'
        '        if record_template is not None\n'
        '        else empty_frame_record(frame_idx, fps, transformer)\n'
        '    )\n',
        "fallback template use",
    )
    text = replace_once(
        text,
        '    if tracking_only:\n        return []\n    behavior_observations = _behavior_eligible_observations(observations)\n',
        '    if tracking_only:\n'
        '        return []\n'
        '    # Every pair row starts from the same frame metadata. Copying one\n'
        '    # prepared template preserves key order and values while avoiding\n'
        '    # construction of the large record dictionary up to 190 times/frame.\n'
        '    record_template = empty_frame_record(frame_idx, fps, transformer)\n'
        '    behavior_observations = _behavior_eligible_observations(observations)\n',
        "frame record template creation",
    )
    text = replace_once(
        text,
        '            pair_tuple in cluster_attack_pairs,\n            pair_cluster_evidence,\n        )\n',
        '            pair_tuple in cluster_attack_pairs,\n'
        '            pair_cluster_evidence,\n'
        '            record_template=record_template,\n'
        '        )\n',
        "pair record template pass",
    )
    text = replace_once(
        text,
        '                cluster_reid_active_count,\n            )\n',
        '                cluster_reid_active_count,\n'
        '                record_template=record_template,\n'
        '            )\n',
        "fallback template pass",
    )
    text = text.replace(
        '                for record in records:\n                    raw_store.add(record)\n',
        '                raw_store.add_many(records)\n',
        1,
    )
    text = replace_once(
        text,
        '        for record in records:\n            raw_store.add(record)\n',
        '        raw_store.add_many(records)\n',
        "sequential add_many",
    )

    store_start = text.index("class PairDataFrameStore:")
    store_end = text.index("\ndef _open_dict_writer(", store_start)
    optimized_store = '''class PairDataFrameStore:\n    """从已导出的鼠对CSV重放行为后处理，避免重新执行YOLO姿态推理。"""\n\n    def __init__(self, path: Path) -> None:\n        # 一次性载入当前短视频的鼠对特征；这里只用于快速行为规则复算。\n        self.table = pd.read_csv(path, encoding="utf-8-sig")\n        # 缓存文件必须保留原始鼠对键，否则无法按独立鼠对执行时序滤波。\n        if "pair_key" not in self.table.columns:\n            raise ValueError(f"行为缓存缺少pair_key字段：{path}")\n        # Build the row index once. The previous implementation converted and\n        # scanned the complete pair_key column for every pair: O(K × N).\n        key_series = self.table["pair_key"].astype("string")\n        grouped = key_series.groupby(key_series, sort=False).groups\n        self._pair_indices: Dict[str, np.ndarray] = {\n            str(key): np.asarray(indices, dtype=np.int64)\n            for key, indices in grouped.items()\n            if not pd.isna(key)\n        }\n        self._pair_keys = sorted(self._pair_indices)\n\n    def pair_keys(self) -> List[str]:\n        # 返回副本，避免调用方修改内部稳定顺序。\n        return list(self._pair_keys)\n\n    def read_pair(self, pair_key: str) -> pd.DataFrame:\n        indices = self._pair_indices.get(str(pair_key))\n        if indices is None:\n            return self.table.iloc[0:0].copy()\n        # 返回副本，避免后处理新增列时污染其他鼠对或后续重复读取。\n        pair = self.table.iloc[indices].copy()\n        # 帧号排序保持时间滤波输入严格单调。\n        return pair.sort_values("frame", kind="stable").reset_index(drop=True)\n\n'''
    text = text[:store_start] + optimized_store + text[store_end:]

    text = replace_once(
        text,
        '                for row in mask_stats.debug_rows:\n'
        '                    out_row = dict(row)\n'
        '                    out_row["frame"] = int(frame_idx)\n'
        '                    mask_writer.writerow(out_row)\n',
        '                if mask_stats.debug_rows:\n'
        '                    mask_writer.writerows(\n'
        '                        {**dict(row), "frame": int(frame_idx)}\n'
        '                        for row in mask_stats.debug_rows\n'
        '                    )\n',
        "mask writerows",
    )
    text = replace_once(
        text,
        '                for row in occlusion_manager.debug_rows:\n'
        '                    row = dict(row)\n'
        '                    row["local_recovery_added"] = local_added\n'
        '                    cluster_writer.writerow(row)\n'
        '                occlusion_manager.debug_rows.clear()\n'
        '                for row in cluster_reid.debug_rows:\n'
        '                    reid_writer.writerow(dict(row))\n'
        '                cluster_reid.debug_rows.clear()\n\n'
        '                for debug in identity.debug_records:\n'
        '                    debug_writer.writerow(asdict(debug))\n'
        '                identity.debug_records.clear()\n',
        '                if occlusion_manager.debug_rows:\n'
        '                    cluster_writer.writerows(\n'
        '                        {**dict(row), "local_recovery_added": local_added}\n'
        '                        for row in occlusion_manager.debug_rows\n'
        '                    )\n'
        '                occlusion_manager.debug_rows.clear()\n'
        '                if cluster_reid.debug_rows:\n'
        '                    reid_writer.writerows(dict(row) for row in cluster_reid.debug_rows)\n'
        '                cluster_reid.debug_rows.clear()\n\n'
        '                if identity.debug_records:\n'
        '                    debug_writer.writerows(asdict(debug) for debug in identity.debug_records)\n'
        '                identity.debug_records.clear()\n',
        "debug writerows",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    optimize_base()
    optimize_main()
    print("Applied v1.40.1 performance-preserving optimizations.")


if __name__ == "__main__":
    main()
