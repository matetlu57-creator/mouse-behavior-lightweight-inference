"""Observed social geometry; missing points never imply contact or identity."""

from __future__ import annotations

import inspect

import cv2
import numpy as np

from .exported_dataset_models import ExportedFeatures


def filter_optional_kwargs(
    fn, optional_kwargs: dict[str, object],
) -> dict[str, object]:
    """Return subset of `optional_kwargs` accepted by `fn`'s signature."""
    if not optional_kwargs:
        return {}
    try:
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return optional_kwargs
        return {k: v for k, v in optional_kwargs.items() if k in params}
    except (TypeError, ValueError):
        return {}


def _slot_hull_table(features: ExportedFeatures, kind: str, cache: dict | None) -> dict[tuple[int, int], tuple[np.ndarray, float, float, float, float] | None]:
    if cache is None:
        return {}
    key = ("slot_hulls", kind)
    table = cache.get(key)
    if table is None:
        table = {}
        cache[key] = table
    return table


def _get_body_bbox_stats(
    features: ExportedFeatures, max_body_lengths: float, cache: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    points = features.collective_keypoints_cm
    if points is None or not np.isfinite(features.reference_body_cm) or features.reference_body_cm <= 0:
        return None
    ckey = ("footprint_aabb", float(max_body_lengths))
    if cache is not None and ckey in cache:
        return cache[ckey]
    maximum_span = max_body_lengths * features.reference_body_cm
    pts6 = np.asarray(points[:, :, :6, :], float)
    fin6 = np.isfinite(pts6).all(axis=3)
    enough4 = fin6.sum(axis=2) >= 4
    inf_pos = np.where(fin6, pts6[:, :, :, 0], np.inf)
    ninf_pos = np.where(fin6, pts6[:, :, :, 0], -np.inf)
    min_x = inf_pos.min(axis=2)
    max_x = ninf_pos.max(axis=2)
    inf_y = np.where(fin6, pts6[:, :, :, 1], np.inf)
    ninf_y = np.where(fin6, pts6[:, :, :, 1], -np.inf)
    min_y = inf_y.min(axis=2)
    max_y = ninf_y.max(axis=2)
    span = np.where(enough4, np.hypot(max_x - min_x, max_y - min_y), np.inf)
    cand = features.valid & enough4 & (span <= maximum_span)
    res = (cand, min_x, max_x, min_y, max_y)
    if cache is not None:
        cache[ckey] = res
    return res


def _get_non_tail_hull(
    points: np.ndarray, frame: int, slot: int, maximum_span: float, hull_table: dict,
) -> np.ndarray | None:
    key = (frame, slot)
    if key in hull_table:
        return hull_table[key]
    body = np.asarray(points[frame, slot, :6], float)
    body = body[np.isfinite(body).all(axis=1)]
    if len(body) < 4 or np.linalg.norm(np.ptp(body, axis=0)) > maximum_span:
        hull_table[key] = None
        return None
    hull = cv2.convexHull(body.astype(np.float32))
    if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
        hull_table[key] = None
        return None
    hull_table[key] = hull
    return hull


def endpoint_distances(
    features: ExportedFeatures, edge: int, endpoint: str,
    *, _geom_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure actor snout to the target head triangle or target tail root.

    Legacy directed head measurements are minimum distances to the three head
    vertices. They are a conservative sufficient contact measurement: a vertex
    is part of the triangle. A nose-to-nose summary is never substituted.
    """
    count = len(features.valid)
    left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
    result = [np.full(count, np.nan), np.full(count, np.nan)]
    points = features.collective_keypoints_cm
    if points is None:
        for index, direction in enumerate(("left_to_right", "right_to_left")):
            measured = getattr(features, f"pair_nose_{endpoint}_{direction}_cm", None)
            if measured is not None:
                result[index] = np.asarray(measured[:, edge], float)
        return result[0], result[1]
    points = np.asarray(points, float)
    head_hulls = _slot_hull_table(features, "head3", _geom_cache) if endpoint == "head" else {}
    for index, (actor, target) in enumerate(((left, right), (right, left))):
        observed = features.valid[:, actor] & features.valid[:, target] & np.isfinite(points[:, actor, 0]).all(axis=1)
        if endpoint == "tail":
            valid_frames = observed & np.isfinite(points[:, target, 6]).all(axis=1)
            if valid_frames.any():
                diff = points[valid_frames, actor, 0] - points[valid_frames, target, 6]
                result[index][valid_frames] = np.hypot(diff[:, 0], diff[:, 1])
        elif endpoint == "head":
            valid_frames = observed & np.isfinite(points[:, target, :3]).all(axis=(1, 2))
            for frame in np.flatnonzero(valid_frames):
                hkey = (int(frame), target)
                if hkey in head_hulls:
                    hull = head_hulls[hkey]
                else:
                    triangle = points[frame, target, :3]
                    hull = cv2.convexHull(triangle.astype(np.float32))
                    if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
                        hull = None
                    if _geom_cache is not None:
                        head_hulls[hkey] = hull
                if hull is None:
                    continue
                nose = points[frame, actor, 0]
                result[index][frame] = max(
                    0.0, -cv2.pointPolygonTest(hull, (float(nose[0]), float(nose[1])), True)
                )
        else:
            raise ValueError("social endpoint must be head or tail")
    return result[0], result[1]


def snout_surface_distances(
    features: ExportedFeatures, edge: int,
    *, _geom_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed snout-to-body convex-hull gaps, not center distances."""
    count = len(features.valid)
    result = [np.full(count, np.nan), np.full(count, np.nan)]
    points = features.collective_keypoints_cm
    if points is None:
        return result[0], result[1]
    left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
    body7_hulls = _slot_hull_table(features, "body7", _geom_cache)
    for index, (actor, target) in enumerate(((left, right), (right, left))):
        observed = (
            features.valid[:, actor]
            & features.valid[:, target]
            & np.isfinite(points[:, actor, 0]).all(axis=1)
        )
        for frame in np.flatnonzero(observed):
            hkey = (int(frame), target)
            if hkey in body7_hulls:
                hull = body7_hulls[hkey]
            else:
                body = points[frame, target]
                finite = np.isfinite(body).all(axis=1)
                if int(finite.sum()) < 4:
                    hull = None
                else:
                    hull = cv2.convexHull(body[finite].astype(np.float32))
                    if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
                        hull = None
                if _geom_cache is not None:
                    body7_hulls[hkey] = hull
            if hull is None:
                continue
            nose = points[frame, actor, 0]
            result[index][frame] = max(
                0.0, -cv2.pointPolygonTest(hull, (float(nose[0]), float(nose[1])), True)
            )
    return result[0], result[1]


def body_footprint_distances(
    features: ExportedFeatures, edge: int, *, max_body_lengths: float,
    frame_mask: np.ndarray | None = None,
    _geom_cache: dict | None = None,
    _overlap_only: bool = False,
) -> np.ndarray:
    """Observed projected body gap, excluding tailroot and coarse boxes.

    At least four finite head/body points per mouse and a nondegenerate hull
    are required. A large keypoint span is unknown, not a contact shortcut.
    Polygon overlap is only a geometric cue; it never establishes an action.
    """
    if not np.isfinite(max_body_lengths) or max_body_lengths <= 0:
        raise ValueError("maximum footprint size must be finite and positive")
    count = len(features.valid)
    result = np.full(count, np.nan)
    points = features.collective_keypoints_cm
    if points is None:
        return result
    left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
    observed = features.valid[:, left] & features.valid[:, right]
    if frame_mask is not None:
        selected = np.asarray(frame_mask, bool)
        if selected.shape != (count,):
            raise ValueError("footprint frame mask must align with frames")
        observed &= selected
    if not np.isfinite(features.reference_body_cm) or features.reference_body_cm <= 0:
        return result
    maximum_span = max_body_lengths * features.reference_body_cm
    stats = _get_body_bbox_stats(features, max_body_lengths, _geom_cache)
    if stats is None:
        return result
    cand, min_x, max_x, min_y, max_y = stats
    observed = observed & cand[:, left] & cand[:, right]
    if not observed.any():
        return result
    hull_table = _slot_hull_table(features, f"body6_{float(max_body_lengths)}", _geom_cache)
    if _overlap_only:
        dx = np.maximum(0.0, np.maximum(min_x[:, left] - max_x[:, right], min_x[:, right] - max_x[:, left]))
        dy = np.maximum(0.0, np.maximum(min_y[:, left] - max_y[:, right], min_y[:, right] - max_y[:, left]))
        aabb_gap = np.hypot(dx, dy)
        disjoint = observed & (aabb_gap > 0.0)
        result[disjoint] = aabb_gap[disjoint]
        check_frames = np.flatnonzero(observed & (aabb_gap <= 0.0))
    else:
        check_frames = np.flatnonzero(observed)
    for frame in check_frames:
        f_idx = int(frame)
        first = _get_non_tail_hull(points, f_idx, left, maximum_span, hull_table)
        if first is None:
            continue
        second = _get_non_tail_hull(points, f_idx, right, maximum_span, hull_table)
        if second is None:
            continue
        overlap, _ = cv2.intersectConvexConvex(first, second)
        if overlap > 1e-9:
            result[f_idx] = 0.
            continue
        result[f_idx] = min(
            max(0., -cv2.pointPolygonTest(target, (float(point[0]), float(point[1])), True))
            for source, target in ((first, second), (second, first))
            for point in source.reshape(-1, 2)
        )
    return result


def observed_body_separation(features: ExportedFeatures, edge: int, *,
                             max_body_lengths: float,
                             _geom_cache: dict | None = None) -> np.ndarray:
    """Known positive projected gap; touching boxes or non-tail hulls veto it.

    This conservative image-plane proxy is not proof of physical separation.
    Unknown geometry never supplies a positive frame or bridges an episode.
    """
    box_gap = np.full(len(features.valid), np.nan)
    left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
    boxes = features.collective_boxes_cm
    if boxes is not None:
        first, second = np.asarray(boxes[:, left], float), np.asarray(boxes[:, right], float)
        known = (np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
                 & (first[:, 2:] > first[:, :2]).all(axis=1)
                 & (second[:, 2:] > second[:, :2]).all(axis=1))
        gaps = np.maximum(0., np.maximum(first[:, :2] - second[:, 2:],
                                         second[:, :2] - first[:, 2:]))
        box_gap[known] = np.linalg.norm(gaps[known], axis=1)
    elif features.pair_box_gap_cm is not None:
        box_gap = np.asarray(features.pair_box_gap_cm[:, edge], float)
    box_known = np.isfinite(box_gap)
    # When bounding box itself overlaps (box_gap <= 0), outcome is always False regardless of hulls;
    # when box_known is True and box_gap > 0, outcome depends strictly on hull_gap <= 0 (so _overlap_only=True is exact).
    pair_vis = features.valid[:, left] & features.valid[:, right]
    possible = pair_vis & ~(box_known & (box_gap <= 0.))
    if not possible.any():
        return np.zeros(len(features.valid), bool)
    hull_gap = body_footprint_distances(
        features,
        edge,
        max_body_lengths=max_body_lengths,
        frame_mask=possible,
        **filter_optional_kwargs(
            body_footprint_distances,
            {
                "_geom_cache": _geom_cache,
                "_overlap_only": bool(boxes is not None or features.pair_box_gap_cm is not None)
                and bool(box_known[possible].all()),
            },
        ),
    )
    hull_known = np.isfinite(hull_gap)
    return ((hull_known | box_known) & ~(hull_known & (hull_gap <= 0.))
            & ~(box_known & (box_gap <= 0.))
            & pair_vis)


def directed_snout_body_footprint_distances(
    features: ExportedFeatures, edge: int, *, actor: int, max_body_lengths: float,
    frame_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Measured actor nose to its own target's valid non-tail body hull.

    Body overlap can concern the actor's flank while its head turns toward a
    third mouse. This directional geometry keeps target confirmation separate
    from the symmetric footprint cue used for continuation.
    """
    if not np.isfinite(max_body_lengths) or max_body_lengths <= 0:
        raise ValueError("maximum footprint size must be finite and positive")
    left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
    if actor not in (left, right):
        raise ValueError("actor must belong to the requested pair")
    target = right if actor == left else left
    count = len(features.valid)
    result = np.full(count, np.nan)
    observed = features.valid[:, actor] & features.valid[:, target]
    if frame_mask is not None:
        selected = np.asarray(frame_mask, bool)
        if selected.shape != (count,):
            raise ValueError("footprint frame mask must align with frames")
        observed &= selected
    points = features.collective_keypoints_cm
    if points is None or not np.isfinite(features.reference_body_cm) or features.reference_body_cm <= 0:
        return result
    maximum_span = max_body_lengths * features.reference_body_cm
    for frame in np.flatnonzero(observed):
        nose = np.asarray(points[frame, actor, 0], float)
        body = np.asarray(points[frame, target, :6], float)
        body = body[np.isfinite(body).all(axis=1)]
        if (not np.isfinite(nose).all() or len(body) < 4
                or np.linalg.norm(np.ptp(body, axis=0)) > maximum_span):
            continue
        hull = cv2.convexHull(body.astype(np.float32))
        if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
            continue
        result[frame] = max(0., -cv2.pointPolygonTest(hull, tuple(map(float, nose)), True))
    return result


def observed_path_and_net(positions: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    """Path uses adjacent observed frames; a dropout adds no invented travel."""
    positions = np.asarray(positions, float)
    observed = np.asarray(observed, bool) & np.isfinite(positions).all(axis=1)
    frames = np.flatnonzero(observed)
    if len(frames) < 2:
        return 0.0, 0.0
    adjacent = observed[1:] & observed[:-1]
    path = float(np.linalg.norm(np.diff(positions, axis=0)[adjacent], axis=1).sum())
    net = float(np.linalg.norm(positions[frames[-1]] - positions[frames[0]]))
    return path, net


def observed_motion_statistics(positions: np.ndarray, observed: np.ndarray, fps: float) -> dict:
    """Motion of one published event; absent steps add neither time nor travel."""
    positions = np.asarray(positions, float)
    observed = np.asarray(observed, bool) & np.isfinite(positions).all(axis=1)
    adjacent = observed[1:] & observed[:-1]
    steps = np.diff(positions, axis=0)[adjacent]
    seconds = float(adjacent.sum()) / fps
    path = float(np.linalg.norm(steps, axis=1).sum())
    return dict(path_cm=path, net_cm=float(np.linalg.norm(steps.sum(axis=0))),
                adjacent_seconds=seconds, mean_speed_cm_s=path / seconds if seconds else 0.)
