#!/usr/bin/env python3
"""Build a frozen, post-hoc validation review dataset for the video viewer.

The prediction export and trajectory evidence are read as label-free inference
outputs. Validation annotations are joined only after that export is frozen to
compute metrics and explain errors. This script never calls the behavior
predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import REPO_ROOT  # noqa: F401 - ensure the src package is importable
from mouse_behavior.evaluation.exported_dataset import (
    HeuristicParameters,
    _isolation_cumulative_durations,
    _longest_run,
    _rolling_drop,
    _windowed_body_speed,
    load_exported_features,
)
from mouse_behavior.evaluation.handcrafted_features import behavior_feature_vector


DISPLAY_NAMES = {
    "approach": "接近",
    "attack": "攻击/被攻击",
    "huddle": "扎堆",
    "isolation": "孤立",
    "nose_head_contact": "鼻头接触",
    "nose_tail_contact": "鼻尾接触",
    "social_clustering": "社会聚集",
    "group_locomotion": "群体同步运动",
    "dispersal": "群体分散",
    "stationary": "静止",
    "together": "一起",
    # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
    "locomotion": "移动",
    "following": "跟随",
    "chase": "追逐/被追逐",
    "avoidance": "回避/被回避",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))


def _canonical(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"none", "null", "unrecognized", "未识别"}:
        return ""
    aliases = {
        "huddling": "huddle",
        "snout_head_contact": "nose_head_contact",
        "snout_rear_contact": "nose_tail_contact",
        "snout_tail_contact": "nose_tail_contact",
        "static": "stationary",
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion: legacy per-frame labels collapse onto the merged class.
        "running": "locomotion",
        "walking": "locomotion",
        "group_locomotion": "group_locomotion",
        "chasing": "chase",
        "avoiding": "avoidance",
    }
    return aliases.get(normalized, normalized)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percent(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def _strict_and_compatible(truth: str, predicted: str, target_ids: tuple[int, ...], predicted_ids: tuple[int, ...]) -> tuple[bool, bool]:
    # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
    # 同层双标签信用（Together↔鼻头/鼻尾接触，同一对目标 ID）已整体删除；
    # 本脚本的 compatible 口径退化为严格 Top-1 等价路径，两个返回值相同。
    ids_match = bool(target_ids) and predicted_ids == target_ids
    strict = predicted == truth and ids_match
    return strict, strict


def _target_scores(evidence: dict[str, Any], behavior: str, target_ids: tuple[int, ...]) -> dict[str, float]:
    scores: dict[str, float] = {}
    identities = evidence.get("identity_scores", {})
    pairs = evidence.get("pair_scores", {})
    groups = evidence.get("group_scores", {})
    if len(target_ids) == 1:
        scores.update(
            {
                str(key): float(value)
                for key, value in identities.get(str(target_ids[0]), {}).items()
            }
        )
        # Isolation is represented as an identity-in-a-group score in the
        # evidence export, but its target unit remains the one labelled ID.
        if behavior == "isolation":
            scores["isolation"] = max(
                scores.get("isolation", 0.0),
                float(groups.get(str(target_ids[0]), {}).get("isolation", 0.0)),
            )
    elif len(target_ids) == 2:
        key = ",".join(map(str, target_ids))
        scores.update({str(k): float(v) for k, v in pairs.get(key, {}).items()})
    else:
        target_set = set(target_ids)
        for raw_ids, values in groups.items():
            try:
                candidate_ids = set(_ids(str(raw_ids)))
            except ValueError:
                continue
            if target_set.issubset(candidate_ids):
                for name, value in values.items():
                    scores[str(name)] = max(scores.get(str(name), 0.0), float(value))
    return scores


def _feature_failures(
    behavior: str,
    values: dict[str, float | None],
    parameters: HeuristicParameters,
    mouse_width_cm: float,
) -> list[str]:
    """Report measured primary gates below/above rule thresholds.

    These are trajectory-summary comparisons, not a second inference pass.
    Temporal conjunction details not represented in summaries are called out
    separately by the caller rather than guessed.
    """

    checks: list[tuple[str, float | None, str, float, str]] = []
    if behavior == "isolation":
        checks.append(("最近邻距离 P10", values.get("nearest_p10_cm"), ">=", parameters.isolation_distance_cm, "cm"))
    elif behavior == "stationary":
        checks.append(("身体中心速度 P90", values.get("center_speed_p90_cm_s"), "<=", parameters.stationary_max_speed_cm_s, "cm/s"))
    elif behavior == "locomotion":
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion
        # Locomotion is the negative definition of Static: speed above the static
        # ceiling, with no directionality gate and no upper speed band.
        checks.extend(
            (
                ("速度 P90", values.get("speed_p90_cm_s"), ">", parameters.stationary_max_speed_cm_s, "cm/s"),
                ("活动速度中位数", values.get("active_speed_median_cm_s"), ">", parameters.stationary_max_speed_cm_s, "cm/s"),
            )
        )
    elif behavior == "approach":
        checks.extend(
            (
                ("末段双鼠距离", values.get("final_distance_cm"), "<", parameters.approach_terminal_distance_cm, "cm"),
                ("闭合速度 P90", values.get("closing_speed_p90_cm_s"), ">=", parameters.approach_min_closing_speed_cm_s, "cm/s"),
            )
        )
    elif behavior == "together":
        checks.extend(
            (
                ("双鼠距离中位数", values.get("distance_median_cm"), "<", parameters.together_max_distance_cm, "cm"),
                ("同鼠对的个体速度 P90 最大值", values.get("individual_speed_p90_max_cm_s"), "<=", parameters.together_max_individual_speed_cm_s, "cm/s"),
                ("双鼠合计速度中位数", values.get("combined_speed_median_cm_s"), "<=", parameters.together_max_combined_speed_cm_s, "cm/s"),
            )
        )
    elif behavior == "nose_head_contact":
        checks.append(("鼻头接触累计时长", values.get("nose_head_under_2cm_s"), ">=", parameters.contact_min_cumulative_seconds, "s"))
    elif behavior == "nose_tail_contact":
        checks.append(("鼻尾接触累计时长", values.get("nose_rear_under_2cm_s"), ">=", parameters.contact_min_cumulative_seconds, "s"))
    elif behavior == "attack":
        checks.append(("双鼠峰值速度", values.get("peak_speed_cm_s"), ">=", parameters.attack_min_speed_cm_s, "cm/s"))
        if parameters.attack_contact_score_multiplier > 1.0:
            checks.extend(
                (
                    ("接触后鼻尖速度 P90", values.get("contact_nose_speed_p90_cm_s"), ">=", parameters.attack_contact_nose_min_speed_cm_s, "cm/s"),
                    ("接触后鼻尖加速度 P90", values.get("contact_nose_acceleration_p90_cm_s2"), ">=", parameters.attack_contact_nose_min_acceleration_cm_s2, "cm/s²"),
                    ("接触后身体速度 P90", values.get("contact_body_speed_p90_cm_s"), ">=", parameters.attack_contact_body_min_speed_cm_s, "cm/s"),
                )
            )
    elif behavior == "huddle":
        checks.extend(
            (
                ("群体相邻距离中位数", values.get("pair_distance_median_cm"), "<", min(parameters.huddle_distance_cm, parameters.huddle_width_multiplier * mouse_width_cm), "cm"),
                ("群体速度 P90", values.get("speed_p90_cm_s"), "<=", min(parameters.huddle_max_mean_speed_cm_s, parameters.stationary_max_speed_cm_s), "cm/s"),
            )
        )
    elif behavior == "social_clustering":
        checks.extend(
            (
                ("群体平均距离变化（负值为收拢）", values.get("pair_distance_end_change_cm"), "<=", -parameters.clustering_min_nearest_neighbor_drop_cm, "cm"),
                ("群体速度中位数", values.get("speed_median_cm_s"), ">=", parameters.clustering_min_mean_speed_cm_s, "cm/s"),
                ("目标成员数", values.get("member_count"), ">=", 3.0, "只"),
            )
        )
    elif behavior == "dispersal":
        checks.extend(
            (
                ("初始群体半径", values.get("initial_radius_cm"), "<=", parameters.dispersal_start_distance_cm, "cm"),
                ("群体距离变化（正值为散开）", values.get("pair_distance_end_change_cm"), ">=", parameters.dispersal_min_distance_increase_cm, "cm"),
            )
        )
    elif behavior == "group_locomotion":
        checks.extend(
            (
                ("群体相邻距离中位数", values.get("pair_distance_median_cm"), "<", parameters.group_locomotion_max_distance_cm, "cm"),
                # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion: the merged mask is speed above the static ceiling.
                ("群体速度中位数（移动下限）", values.get("speed_median_cm_s"), ">", parameters.stationary_max_speed_cm_s, "cm/s"),
            )
        )

    failures: list[str] = []
    for label, actual, op, threshold, unit in checks:
        if actual is None or not math.isfinite(float(actual)):
            failures.append(f"{label}：轨迹摘要不可用，需检查关键点/ID 缺失")
            continue
        number = float(actual)
        passed = {
            "<": number < threshold,
            "<=": number <= threshold,
            ">": number > threshold,
            ">=": number >= threshold,
        }[op]
        if not passed:
            failures.append(
                f"{label}={number:.2f}{unit}，未满足 {op}{threshold:g}{unit}"
            )
    return failures


def _exact_temporal_failures(
    behavior: str,
    features: Any,
    target_ids: tuple[int, ...],
    parameters: HeuristicParameters,
    evidence: dict[str, Any],
) -> list[str]:
    """Recheck the target's primary temporal conjunction after frozen inference."""

    slots_by_id = {int(track_id): index for index, track_id in enumerate(features.track_ids)}
    if not set(target_ids).issubset(slots_by_id):
        return ["标注目标 ID 在有效轨迹特征中缺失，无法形成该目标候选"]
    slots = tuple(slots_by_id[value] for value in target_ids)
    fps = max(float(features.fps), 1e-6)
    valid = features.valid
    motion_valid = features.motion_valid if features.motion_valid is not None else valid
    failures: list[str] = []

    if behavior == "isolation" and len(slots) == 1:
        slot = slots[0]
        valid_count = valid.sum(axis=1)
        far = valid[:, slot] & (valid_count >= 2) & (
            features.nearest_distance_cm[:, slot] >= parameters.isolation_distance_cm
        )
        unknown = ~valid[:, slot] | (valid_count < 2)
        duration = float(
            _isolation_cumulative_durations(
                far[:, None], unknown[:, None], fps, parameters.isolation_max_gap_s
            )[0]
        )
        required = max(float(parameters.isolation_min_duration_s), 10.0)
        if duration + 1e-12 < required:
            failures.append(
                f"满足最近邻距离≥{parameters.isolation_distance_cm:g}cm 的累计有效时长最长 {duration:.2f}s，未达到 {required:g}s（短时 ID 缺失不计入时长）"
            )
        group_candidates = []
        for raw_ids, scores in dict(evidence.get("group_scores", {})).items():
            try:
                member_ids = _ids(str(raw_ids))
            except ValueError:
                continue
            if target_ids[0] in member_ids:
                for name in ("huddle", "social_clustering"):
                    if float(scores.get(name, 0.0)) > 0.0:
                        group_candidates.append(DISPLAY_NAMES.get(name, name))
        if group_candidates:
            if duration + 1e-12 >= required:
                failures.append(
                    f"距离与累计时长门槛已满足（{duration:.2f}s）；随后因同一 ID 进入群体候选而被 Isolation 抑制"
                )
            failures.append(
                f"同一 ID 同时进入群体候选（{'、'.join(sorted(set(group_candidates)))}）；当前规则会抑制该 ID 的 Isolation 候选"
            )
        return failures

    if behavior not in {"approach", "together"} or len(slots) != 2:
        return failures
    left, right = slots
    pair_indices = np.flatnonzero(
        ((features.pair_i == left) & (features.pair_j == right))
        | ((features.pair_i == right) & (features.pair_j == left))
    )
    if len(pair_indices) != 1:
        return ["目标双鼠对未形成唯一的轨迹关系，无法核对配对特征"]
    pair = int(pair_indices[0])
    distance = np.asarray(features.pair_distance_cm[:, pair], dtype=float)
    observed_pair = np.isfinite(distance) & valid[:, left] & valid[:, right]
    left_speed = np.asarray(features.speed_cm_s[:, left], dtype=float)
    right_speed = np.asarray(features.speed_cm_s[:, right], dtype=float)
    if behavior == "together":
        observed_pair &= motion_valid[:, left] & motion_valid[:, right]
        near = observed_pair & (distance < parameters.together_max_distance_cm)
        slow_individual = (
            (left_speed <= parameters.together_max_individual_speed_cm_s)
            & (right_speed <= parameters.together_max_individual_speed_cm_s)
        )
        slow_pair = (
            left_speed + right_speed <= parameters.together_max_combined_speed_cm_s
        )
        if not near.any():
            failures.append(
                f"目标鼠对没有可见帧满足距离<{parameters.together_max_distance_cm:g}cm"
            )
        elif not np.any(near & slow_individual):
            failures.append(
                f"两鼠距离虽曾<{parameters.together_max_distance_cm:g}cm，但没有帧同时满足各自速度≤{parameters.together_max_individual_speed_cm_s:g}cm/s"
            )
        elif not np.any(near & slow_individual & slow_pair):
            failures.append(
                f"近距离且个体速度合格时，两鼠合计速度仍未满足≤{parameters.together_max_combined_speed_cm_s:g}cm/s"
            )
        qualifying = near & slow_individual & slow_pair
        observed_seconds = _longest_run(qualifying) / fps
        required = float(parameters.together_min_duration_s)
        if observed_seconds + 1e-12 < required:
            failures.append(
                f"距离与速度同时合格的最长连续时长为 {observed_seconds:.2f}s，未达到 {required:g}s"
            )
        return failures

    closing = np.asarray(features.pair_closing_speed_cm_s[:, pair], dtype=float)
    window = max(int(round(float(parameters.approach_window_s) * fps)), 1)
    drop = _rolling_drop(distance, window)
    near = observed_pair & (distance < parameters.approach_terminal_distance_cm)
    dropping = drop >= parameters.approach_min_distance_drop_cm
    closing_enough = closing >= parameters.approach_min_closing_speed_cm_s
    if not near.any():
        failures.append(
            f"目标鼠对没有可见帧进入接近阈值< {parameters.approach_terminal_distance_cm:g}cm"
        )
    if not np.any(observed_pair & dropping):
        maximum_drop = float(np.nanmax(drop[observed_pair])) if observed_pair.any() else 0.0
        failures.append(
            f"{parameters.approach_window_s:g}s 窗口内最大距离下降 {maximum_drop:.2f}cm，未达到≥{parameters.approach_min_distance_drop_cm:g}cm"
        )
    if not np.any(observed_pair & closing_enough):
        maximum_closing = float(np.nanmax(closing[observed_pair])) if observed_pair.any() else 0.0
        failures.append(
            f"目标鼠对最大闭合速度 {maximum_closing:.2f}cm/s，未达到≥{parameters.approach_min_closing_speed_cm_s:g}cm/s"
        )
    joint = near & dropping & closing_enough
    duration = _longest_run(joint) / fps
    required = float(parameters.pair_min_duration_s)
    if duration + 1e-12 < required and not failures:
        failures.append(
            f"进入距离、距离下降和闭合速度三项条件同时成立的最长时长为 {duration:.2f}s，未达到 {required:g}s"
        )
    return failures


def _social_clustering_failures(
    features: Any, target_ids: tuple[int, ...], parameters: HeuristicParameters
) -> list[str]:
    """Report the exact directional-walker gate used by Social clustering."""

    positions = {int(track_id): index for index, track_id in enumerate(features.track_ids)}
    slots = [positions[value] for value in target_ids if value in positions]
    if len(slots) < 3:
        return ["社会聚集至少需要 3 个标注目标 ID 的有效轨迹"]
    frame_count = len(features.valid)
    if frame_count < 3:
        return ["目标群体轨迹帧数不足，无法核对聚集形成方向"]
    connected = _group_connected_mask(
        features,
        slots,
        float(parameters.clustering_max_distance_cm),
    )
    connected_duration = _longest_run(
        _bridge_gaps(connected, int(np.floor(parameters.clustering_max_gap_s * features.fps)))
    ) / max(float(features.fps), 1e-6)
    failures: list[str] = []
    if connected_duration + 1e-12 < parameters.clustering_min_duration_s:
        failures.append(
            f"同一标注 ID 集合在相邻距离<{parameters.clustering_max_distance_cm:g}cm 时形成连通群的最长时长 {connected_duration:.2f}s，未达到 {parameters.clustering_min_duration_s:g}s"
        )

    # The formation score requires at least half of the candidate members to
    # show a nearest-neighbour decrease from within the initial-distance gate.
    window = max(int(round(parameters.clustering_window_s * features.fps)), 1)
    group_speed = _windowed_body_speed(features, parameters.group_motion_window_s)
    initial_member_frames = np.zeros((frame_count, len(slots)), dtype=bool)
    for local_slot, track_slot in enumerate(slots):
        nearest = features.nearest_distance_cm[:, track_slot]
        drop = _rolling_drop(nearest, window)
        initial = np.full_like(nearest, np.inf, dtype=float)
        if window < frame_count:
            initial[window:] = nearest[:-window]
        initial_member_frames[:, local_slot] = (
            features.valid[:, track_slot]
            & (initial <= parameters.clustering_initial_max_distance_cm)
            & (drop >= parameters.clustering_min_nearest_neighbor_drop_cm)
            & np.isfinite(group_speed[:, track_slot])
            & (group_speed[:, track_slot] >= parameters.clustering_min_mean_speed_cm_s)
        )
    formation_member_fraction = float(np.mean(initial_member_frames.any(axis=0)))
    if formation_member_fraction + 1e-12 < parameters.clustering_min_moving_member_fraction:
        failures.append(
            f"只有 {formation_member_fraction:.0%} 的标注成员满足“初始最近邻≤{parameters.clustering_initial_max_distance_cm:g}cm、{parameters.clustering_window_s:g}s 内下降≥{parameters.clustering_min_nearest_neighbor_drop_cm:g}cm、群体速度≥{parameters.clustering_min_mean_speed_cm_s:g}cm/s”形成证据，要求至少 {parameters.clustering_min_moving_member_fraction:.0%}"
        )
    last_start = max(0, int(frame_count * 0.8))
    endpoints = features.centers_cm[last_start:, slots].reshape(-1, 2)
    endpoints = endpoints[np.isfinite(endpoints).all(axis=1)]
    if len(endpoints) < 3:
        return [*failures, "视频末段可用身体中心不足 3 个，无法确定聚集目的位置"]
    destination = np.median(endpoints, axis=0)
    all_motion_valid = features.motion_valid if features.motion_valid is not None else features.valid
    motion_valid = all_motion_valid
    walker_summaries: list[tuple[float, float]] = []
    for slot in slots:
        points = features.centers_cm[:, slot]
        to_group = destination - points
        distance = np.linalg.norm(to_group, axis=1)
        velocity = features.velocity_cm_s[:, slot]
        speed = np.linalg.norm(velocity, axis=1)
        denominator = speed * distance
        direction = np.divide(
            np.sum(velocity * to_group, axis=1),
            denominator,
            out=np.zeros(frame_count, dtype=float),
            where=denominator > 1e-6,
        )
        # [STEP4 2026-10-04] 个体层合并：Walking+Running -> Locomotion: the merged mask replaces the old walking speed band.
        approaching_locomotion = (
            motion_valid[:, slot]
            & (speed > parameters.stationary_max_speed_cm_s)
            & (direction >= parameters.clustering_directional_min_cosine)
        )
        observed_distances = distance[np.isfinite(points).all(axis=1)]
        if len(observed_distances) < 3:
            walker_summaries.append(
                (float(np.count_nonzero(approaching_locomotion) / features.fps), 0.0)
            )
            continue
        boundary = max(1, len(observed_distances) // 5)
        toward_drop = float(
            np.median(observed_distances[:boundary])
            - np.median(observed_distances[-boundary:])
        )
        walker_summaries.append(
            (float(np.count_nonzero(approaching_locomotion) / features.fps), toward_drop)
        )
        if (
            walker_summaries[-1][0] + 1e-12 >= parameters.clustering_directional_walk_min_seconds
            and toward_drop + 1e-12 >= parameters.clustering_directional_min_distance_drop_cm
        ):
            return failures
    best_seconds, best_drop = max(
        walker_summaries,
        key=lambda item: (
            min(item[0] / max(parameters.clustering_directional_walk_min_seconds, 1e-6), 1.0)
            + min(item[1] / max(parameters.clustering_directional_min_distance_drop_cm, 1e-6), 1.0),
            item[0],
        ),
    )
    if best_seconds + 1e-12 < parameters.clustering_directional_walk_min_seconds:
        failures.append(
            f"没有成员向末段聚集中心以移动速度且方向余弦≥{parameters.clustering_directional_min_cosine:g}持续足够久；最佳成员仅 {best_seconds:.2f}s，要求≥{parameters.clustering_directional_walk_min_seconds:g}s"
        )
    if best_drop + 1e-12 < parameters.clustering_directional_min_distance_drop_cm:
        failures.append(
            f"最佳移动成员到聚集中心的距离下降 {best_drop:.2f}cm，未达到≥{parameters.clustering_directional_min_distance_drop_cm:g}cm"
        )
    return failures


def _group_connected_mask(features: Any, slots: list[int], distance_cm: float) -> np.ndarray:
    """Whether all requested members form one observed, close spatial component."""

    if len(slots) < 3:
        return np.zeros(len(features.valid), dtype=bool)
    local = {slot: index for index, slot in enumerate(slots)}
    internal_edges = [
        (edge, local[int(left)], local[int(right)])
        for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j))
        if int(left) in local and int(right) in local
    ]
    result = np.zeros(len(features.valid), dtype=bool)
    for frame in range(len(result)):
        if not bool(np.all(features.valid[frame, slots])):
            continue
        adjacency = [set() for _ in slots]
        for edge, left, right in internal_edges:
            if features.pair_distance_cm[frame, edge] < distance_cm:
                adjacency[left].add(right)
                adjacency[right].add(left)
        visited = {0}
        pending = [0]
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        result[frame] = len(visited) == len(slots)
    return result


def _bridge_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    bridged = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0:
        return bridged
    false_positions = np.flatnonzero(~bridged)
    if not false_positions.size:
        return bridged
    runs: list[tuple[int, int]] = []
    start = previous = int(false_positions[0])
    for position in false_positions[1:]:
        current = int(position)
        if current != previous + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    for start, stop in runs:
        if start > 0 and stop < len(bridged) - 1 and stop - start + 1 <= max_gap_frames:
            bridged[start : stop + 1] = True
    return bridged


def _huddling_onset_failures(
    features: Any, target_ids: tuple[int, ...], parameters: HeuristicParameters
) -> list[str]:
    positions = {int(track_id): index for index, track_id in enumerate(features.track_ids)}
    slots = [positions[value] for value in target_ids if value in positions]
    if len(slots) < 3:
        return ["扎堆至少需要 3 个标注目标 ID 的有效轨迹"]
    pair_limit = min(
        parameters.huddle_distance_cm,
        float(features.mouse_width_cm) * parameters.huddle_width_multiplier,
    )
    grouped = np.zeros(len(features.valid), dtype=bool)
    local = {slot: index for index, slot in enumerate(slots)}
    for frame in range(len(grouped)):
        degrees = np.zeros(len(slots), dtype=np.int16)
        for edge, (left, right) in enumerate(zip(features.pair_i, features.pair_j)):
            left, right = int(left), int(right)
            if left not in local or right not in local:
                continue
            if (
                features.valid[frame, left]
                and features.valid[frame, right]
                and features.pair_distance_cm[frame, edge] < pair_limit
            ):
                degrees[local[left]] += 1
                degrees[local[right]] += 1
        grouped[frame] = bool(np.max(degrees, initial=0) >= 2)
    onset = np.flatnonzero(grouped)
    if not onset.size:
        return [
            f"未观察到任一目标鼠同时与至少两只目标鼠保持<{pair_limit:.2f}cm 的紧凑邻接关系"
        ]
    fps = max(float(features.fps), 1e-6)
    required_frames = max(1, int(np.ceil(parameters.huddle_min_duration_s * fps)))
    start = int(onset[0])
    stop = start + required_frames
    if stop > len(grouped):
        return [
            f"紧凑群体从第 {start} 帧才出现，剩余时长不足 {parameters.huddle_min_duration_s:g}s 扎堆窗口"
        ]
    grouped_fraction = float(np.mean(grouped[start:stop]))
    minimum_group_fraction = 0.2
    if grouped_fraction < minimum_group_fraction:
        return [
            f"扎堆起始 {parameters.huddle_min_duration_s:g}s 内紧凑邻接只占 {grouped_fraction:.0%}，低于要求的 {minimum_group_fraction:.0%}"
        ]
    group_speed = _windowed_body_speed(features, parameters.group_motion_window_s)
    observed = features.valid[start:stop][:, slots]
    speeds = group_speed[start:stop][:, slots]
    usable = observed & np.isfinite(speeds)
    if not usable.any():
        return ["扎堆起始窗口缺少可用的目标鼠身体速度观测"]
    speed_values = speeds[usable]
    speed_limit = min(parameters.huddle_max_mean_speed_cm_s, parameters.stationary_max_speed_cm_s)
    low_motion_fraction = float(np.mean(speed_values <= speed_limit))
    mean_speed = float(np.mean(speed_values))
    failures = []
    if low_motion_fraction + 1e-12 < parameters.huddle_onset_min_low_motion_fraction:
        failures.append(
            f"扎堆起始窗口内低运动帧占 {low_motion_fraction:.0%}，未达到≥{parameters.huddle_onset_min_low_motion_fraction:.0%}（速度阈值≤{speed_limit:g}cm/s）"
        )
    if mean_speed > speed_limit:
        failures.append(
            f"扎堆起始窗口目标鼠平均身体速度 {mean_speed:.2f}cm/s，高于≤{speed_limit:g}cm/s"
        )
    return failures


def _error_diagnostic(
    row: dict[str, str],
    prediction: dict[str, str],
    evidence: dict[str, Any],
    parameters: HeuristicParameters,
    collective_debug_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    truth = _canonical(row["canonical_behavior"])
    predicted = _canonical(prediction.get("predicted_behavior", ""))
    target_ids = _ids(row.get("mouse_ids", ""))
    predicted_ids = _ids(prediction.get("predicted_target_ids", ""))
    target_scores = _target_scores(evidence, truth, target_ids)
    target_score = float(target_scores.get(truth, 0.0))
    if truth in {"huddle", "social_clustering"} and target_score <= 0.0:
        expected_state = "HUDDLING" if truth == "huddle" else "SOCIAL_CLUSTERING"
        matching = [
            item
            for item in collective_debug_rows or []
            if item.get("state", "").upper() == expected_state
            and set(target_ids).issubset(_ids(item.get("member_ids", "")))
        ]
        if matching:
            selected = min(
                matching,
                key=lambda item: sum(
                    not passed
                    for passed in json.loads(item.get("checks", "{}")).values()
                ),
            )
            checks = json.loads(selected.get("checks", "{}"))
            names = {
                "member_count": "群体数量不足 3 只",
                "duration": "持续时长未达到行为要求",
                "group_visibility": "群体同时可见比例不足",
                "member_support": "目标成员有效轨迹比例不足",
                "nnd_contraction": "最近邻距离未明显收缩",
                "radius_contraction": "群体半径未明显收缩",
                "inward_motion": "目标成员向群体中心运动比例不足",
                "active_members": "活跃运动成员比例不足",
                "graph_growth": "邻近图未显示增长或群体合并",
                "compact_nnd": "最近邻距离未达到紧凑阈值",
                "compact_radius": "群体半径未达到紧凑阈值",
                "stable_nnd": "最近邻距离不够稳定",
                "stable_radius": "群体半径不够稳定",
                "low_motion": "低运动成员比例不足",
                "contact_density": "接触图密度不足",
                "proximity_density": "邻近图密度不足",
                "graph_stability": "群体邻接结构稳定性不足",
                "not_attack": "检测到攻击性高运动，排除 Huddling",
            }
            failures = [names.get(key, key) for key, passed in checks.items() if not passed]
            if failures:
                causes = [
                    "冻结启发式的目标群体候选未通过：" + "；".join(failures)
                ]
            else:
                causes = ["目标群体候选条件通过，但被更高分的 Top-1 候选覆盖"]
            return {
                "predictedBehavior": predicted,
                "predictedBehaviorName": DISPLAY_NAMES.get(predicted, predicted or "未识别"),
                "targetBehaviorScore": target_score,
                "targetCandidateScores": target_scores,
                "failedConditions": causes,
            }
        return {
            "predictedBehavior": predicted,
            "predictedBehaviorName": DISPLAY_NAMES.get(predicted, predicted or "未识别"),
            "targetBehaviorScore": target_score,
            "targetCandidateScores": target_scores,
            "failedConditions": [
                "未形成包含全部标注目标 ID 的群体候选窗口（成员数、时长或邻接范围门槛未建立候选）"
            ],
        }
    if target_score > 0.0 and predicted != truth:
        highest_other = max(
            ((key, value) for key, value in target_scores.items() if key != truth and value > 0.0),
            key=lambda item: item[1],
            default=None,
        )
        note = f"标签行为规则在目标 ID 集合上已有候选分数 {target_score:.2f}s，但 Top‑1 选择了{DISPLAY_NAMES.get(predicted, predicted or '未识别')}。"
        if predicted_ids != target_ids:
            note += f" 预测 ID 为 {','.join(map(str, predicted_ids)) or '空'}，标注 ID 为 {','.join(map(str, target_ids))}。"
        if highest_other:
            note += f" 同一目标候选中竞争行为为{DISPLAY_NAMES.get(highest_other[0], highest_other[0])}（{highest_other[1]:.2f}s）；属于候选排序冲突，不是标签行为门槛未触发。"
        causes = [note]
    elif target_score > 0.0 and predicted == truth and predicted_ids != target_ids:
        causes = [
            f"标签行为规则已触发（{target_score:.2f}s），但预测目标 ID 集合 {','.join(map(str, predicted_ids)) or '空'} 与标注集合 {','.join(map(str, target_ids))} 不一致。"
        ]
    else:
        causes = []
        try:
            features = load_exported_features(row["tracks_path"], fps=float(row["fps"]), max_tracks=64)
            measured = behavior_feature_vector(features, truth, target_ids)
            if truth == "together" and len(target_ids) == 2:
                positions = {int(track_id): index for index, track_id in enumerate(features.track_ids)}
                if set(target_ids).issubset(positions):
                    slots = [positions[value] for value in target_ids]
                    speed_p90s = [
                        float(np.nanquantile(features.speed_cm_s[:, slot], 0.90))
                        if np.isfinite(features.speed_cm_s[:, slot]).any()
                        else None
                        for slot in slots
                    ]
                    measured["individual_speed_p90_max_cm_s"] = (
                        max(value for value in speed_p90s if value is not None)
                        if all(value is not None for value in speed_p90s)
                        else None
                    )
            exact_conditions = _exact_temporal_failures(
                truth, features, target_ids, parameters, evidence
            )
            if truth == "social_clustering":
                exact_conditions.extend(
                    _social_clustering_failures(features, target_ids, parameters)
                )
            elif truth == "huddle":
                exact_conditions.extend(
                    _huddling_onset_failures(features, target_ids, parameters)
                )
            if exact_conditions:
                causes = exact_conditions
                if truth == "social_clustering":
                    causes.extend(
                        _feature_failures(
                            truth, measured, parameters, features.mouse_width_cm
                        )
                    )
            else:
                causes = _feature_failures(truth, measured, parameters, features.mouse_width_cm)
        except (OSError, ValueError, KeyError, TypeError) as error:
            causes = [f"目标轨迹特征无法完整计算：{error}"]
        if not causes:
            causes = [
                "目标行为候选分数为 0。已导出的摘要未显示单一主阈值明显越界；更可能未通过持续时间、逐帧条件同时成立、群体连通/方向性或关键点有效性门槛。"
            ]
        if target_score <= 0.0 and predicted and predicted != truth:
            causes.insert(0, f"当前识别为{DISPLAY_NAMES.get(predicted, predicted)}，不是标签行为{DISPLAY_NAMES.get(truth, truth)}。")
    return {
        "predictedBehavior": predicted,
        "predictedBehaviorName": DISPLAY_NAMES.get(predicted, predicted or "未识别"),
        "targetBehaviorScore": target_score,
        "targetCandidateScores": target_scores,
        "failedConditions": causes,
    }


def _video_url(video_path: str, output_root: Path) -> str:
    root = output_root.resolve()
    source = Path(video_path).resolve()
    relative = source.relative_to(root)
    browser_copy = root / "video_viewer_media" / relative
    if browser_copy.exists():
        relative = browser_copy.relative_to(root)
    try:
        output_prefix = root.relative_to((REPO_ROOT / "outputs").resolve())
    except ValueError:
        output_prefix = Path(root.name)
    return f"/video-files/{(output_prefix / relative).as_posix()}"


def build(args: argparse.Namespace) -> dict[str, Any]:
    receipt = _read_json(args.receipt)
    evidence_path = args.evidence.resolve()
    params_path = args.parameters.resolve()
    if receipt.get("contains_labels") is not False:
        raise ValueError("prediction receipt must confirm contains_labels=false")
    if int(receipt.get("count", 0)) < 1:
        raise ValueError("invalid prediction receipt count")
    if receipt.get("evidence_sha256") != _sha256(evidence_path):
        raise ValueError("trajectory evidence hash does not match the frozen prediction receipt")
    if receipt.get("parameters_sha256") != _sha256(params_path):
        raise ValueError("heuristic parameter hash does not match the frozen prediction receipt")

    split_rows = _read_csv(args.validation_split)
    prediction_rows = _read_csv(args.predictions)
    if len(prediction_rows) != int(receipt["count"]):
        raise ValueError("prediction count does not match the frozen receipt")
    evidence_by_id: dict[str, dict[str, Any]] = {}
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                evidence_by_id[str(item["opaque_id"])] = item
    collective_debug_by_id: dict[str, list[dict[str, str]]] = {}
    debug_path = args.feature_debug
    if debug_path is not None and debug_path.is_file():
        for item in _read_csv(debug_path):
            collective_debug_by_id.setdefault(item.get("opaque_id", ""), []).append(item)
    predictions = {row["sample_id"]: row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("prediction sample IDs are not unique")
    if set(predictions) != {row["sample_id"] for row in split_rows}:
        raise ValueError("validation split and frozen prediction sample IDs differ")
    opaque_ids = {str(row.get("opaque_id", "")) for row in prediction_rows}
    if "" in opaque_ids or len(opaque_ids) != len(prediction_rows) or opaque_ids != set(evidence_by_id):
        raise ValueError("opaque IDs do not match the frozen trajectory evidence")

    raw_parameters = _read_json(params_path)
    parameters = HeuristicParameters(
        **{key: float(value) for key, value in raw_parameters.items()}
    )
    label_order: list[str] = []
    stats: dict[str, dict[str, int]] = {}
    predicted_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    manifests_by_sample = {
        str(item.get("sample_id", "")): item for item in _read_json(args.manifest)
    }
    confusion: dict[str, int] = {}
    strict_total = compatible_total = target_hit_total = prediction_total = 0
    for index, row in enumerate(split_rows):
        sample_id = row["sample_id"]
        pred = predictions[sample_id]
        truth = _canonical(row["canonical_behavior"])
        predicted = _canonical(pred.get("predicted_behavior", ""))
        truth_ids = _ids(row.get("mouse_ids", ""))
        predicted_ids = _ids(pred.get("predicted_target_ids", ""))
        strict, compatible = _strict_and_compatible(truth, predicted, truth_ids, predicted_ids)
        csv_strict = str(pred.get("strict_top1_correct", "")).strip() in {"1", "true", "True"}
        if strict != csv_strict:
            raise ValueError(f"strict Top-1 disagrees with frozen export for {sample_id}")
        # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
        # 与 frozen export 的一致性校验只保留 strict Top-1；旧的 compatible_top1_correct 列
        # 属于双标签口径，读取时容忍其存在，但不再参与校验或计分。
        if truth not in stats:
            label_order.append(truth)
            stats[truth] = {"support": 0, "strict": 0, "compatible": 0}
        stats[truth]["support"] += 1
        stats[truth]["strict"] += int(strict)
        stats[truth]["compatible"] += int(compatible)
        strict_total += int(strict)
        compatible_total += int(compatible)
        if predicted:
            predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        target_hit = bool(truth_ids and predicted_ids == truth_ids)
        target_hit_total += int(target_hit)
        prediction_total += int(bool(predicted))
        if not strict:
            confusion_key = f"{DISPLAY_NAMES.get(truth, truth)} → {DISPLAY_NAMES.get(predicted, predicted or '未识别')}"
            confusion[confusion_key] = confusion.get(confusion_key, 0) + 1

        evidence = evidence_by_id.get(str(pred.get("opaque_id", "")), {})
        target_scores = _target_scores(evidence, truth, truth_ids)
        diagnostics = _error_diagnostic(
            row,
            pred,
            evidence,
            parameters,
            collective_debug_by_id.get(str(pred.get("opaque_id", "")), []),
        ) if not strict else {
            "predictedBehavior": predicted,
            "predictedBehaviorName": DISPLAY_NAMES.get(predicted, predicted),
            "targetBehaviorScore": float(target_scores.get(truth, 0.0)),
            "targetCandidateScores": target_scores,
            "failedConditions": [],
        }
        manifest = manifests_by_sample.get(sample_id)
        if manifest is None:
            raise ValueError(f"render manifest missing sample {sample_id}")
        predicted_name = DISPLAY_NAMES.get(predicted, predicted or "未识别")
        truth_name = DISPLAY_NAMES.get(truth, truth)
        aux = [
            {
                "behavior": str(event.get("behavior", "")),
                "behaviorName": DISPLAY_NAMES.get(str(event.get("behavior", "")), str(event.get("behavior", ""))),
                "ids": [str(value) for value in event.get("ids", [])],
                "score": float(event.get("score", 0.0)),
                "eventScope": str(event.get("event_scope", "")),
            }
            for event in manifest.get("auxiliary_events", [])
            if not event.get("ids")
            or any(
                str(value) not in set(map(str, truth_ids + predicted_ids))
                for value in event.get("ids", [])
            )
        ]
        records.append(
            {
                "id": f"{args.version}-{index + 1:04d}",
                "sampleId": sample_id,
                "truth": truth,
                "truthName": truth_name,
                "predicted": predicted,
                "predictedName": predicted_name,
                "targetHit": target_hit,
                "strictTop1": strict,
                "compatibleTop1": compatible,
                "status": "success" if strict else "error",
                "sourceKind": "behavior-example" if strict else "error",
                "errorReason": "行为和目标 ID 均严格匹配" if strict else f"标签为{truth_name}；实际识别为{predicted_name}。",
                "failedConditions": diagnostics["failedConditions"],
                "targetBehaviorScore": diagnostics["targetBehaviorScore"],
                "candidateScores": diagnostics["targetCandidateScores"],
                "sourceClip": str(row.get("clip_path", "")),
                "sourceTracks": str(row.get("tracks_path", "")),
                "videoUrl": _video_url(str(manifest["output_video"]), args.output_root),
                "frames": int(manifest.get("frames") or 0),
                "fps": round(float(manifest.get("fps") or row.get("fps") or 0), 3),
                "width": int(manifest.get("width") or 0),
                "height": int(manifest.get("height") or 0),
                "recordingSession": str(row.get("recording_session", "")),
                "targetIds": [str(value) for value in truth_ids],
                "predictedTargetIds": [str(value) for value in predicted_ids],
                "selectedTrackCount": int(evidence.get("selected_track_count", len(evidence.get("track_ids", []))) or 0),
                "auxiliaryEvents": aux,
                "auxiliaryBehaviors": sorted({item["behavior"] for item in aux}),
            }
        )

    behavior_metrics = []
    for behavior in label_order:
        item = stats[behavior]
        strict_hits = item["strict"]
        predicted_count = predicted_counts.get(behavior, 0)
        behavior_metrics.append(
            {
                "key": behavior,
                "name": DISPLAY_NAMES.get(behavior, behavior),
                "support": item["support"],
                "predictedCount": predicted_count,
                "truePositives": strict_hits,
                "precision": _percent(_rate(strict_hits, predicted_count)),
                "recall": _percent(_rate(strict_hits, item["support"])),
                "strictTop1": _percent(_rate(strict_hits, item["support"])),
                # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
                # dualTop1/dualCorrect 仅为消费端 schema 兼容保留；双标签信用删除后其数值与严格 Top-1 相同。
                "dualTop1": _percent(_rate(item["compatible"], item["support"])),
                "dualCorrect": item["compatible"],
            }
        )
    records.sort(key=lambda item: (item["status"] != "success", item["truthName"], item["sampleId"]))
    payload = {
        "reviewVersion": args.version,
        "generatedFrom": [
            str(args.validation_split),
            str(args.predictions),
            str(args.evidence),
            str(args.manifest),
            str(args.parameters),
        ],
        "notes": [
            "推理阶段只读轨迹与视觉 ID；标签仅在预测冻结后用于评分和错误归因。",
            "精确率/召回率按严格目标 ID 联合命中计 TP；错行为、错 ID 均不算 TP。",
            # [SUSPENDED-STEP3 2026-10-04] dual-label credit removed: Together now allows contact without interaction
            "计分仅保留严格 Top-1：行为相同且目标 ID 集合完全相同；同层双标签兼容信用已随 Together 新定义（允许接触、无互动）删除。",
            "错误条件是轨迹摘要与当前主阈值的事后对照；时序复合规则未由单个摘要字段表达时会明确提示，不伪称唯一根因。",
            "该验证集此前已被查看，本报告属于诊断性验证，不是新的盲测。",
        ],
        "metrics": {
            "total": len(records),
            "success": strict_total,
            "errors": len(records) - strict_total,
            "strictTop1Rate": _percent(_rate(strict_total, len(records))) or 0.0,
            "compatibleCorrect": compatible_total,
            "compatibleTop1Rate": _percent(_rate(compatible_total, len(records))) or 0.0,
            "targetIdHits": target_hit_total,
            "targetIdHitRate": _percent(_rate(target_hit_total, len(records))) or 0.0,
            "overallPrecision": _percent(_rate(strict_total, prediction_total)),
            "overallRecall": _percent(_rate(strict_total, len(records))),
            "microPredictedCount": prediction_total,
            "perBehavior": behavior_metrics,
            "errorConfusion": sorted(
                ({"pair": key, "count": value} for key, value in confusion.items()),
                key=lambda value: (-value["count"], value["pair"]),
            ),
        },
        "behaviors": [{"key": item["key"], "name": item["name"]} for item in behavior_metrics],
        "items": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-debug", type=Path)
    parser.add_argument("--version", choices=("v7", "v8", "v9", "v10"), default="v7")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args)
    metrics = payload["metrics"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": metrics["total"],
                "strict": f"{metrics['success']}/{metrics['total']} ({metrics['strictTop1Rate']}%)",
                "compatible": f"{metrics['compatibleCorrect']}/{metrics['total']} ({metrics['compatibleTop1Rate']}%)",
                "errors": metrics["errors"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
