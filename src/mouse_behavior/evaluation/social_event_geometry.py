"""Observed social geometry; missing points never imply contact or identity."""

from __future__ import annotations

import cv2
import numpy as np

from .exported_dataset_models import ExportedFeatures


def endpoint_distances(
    features: ExportedFeatures, edge: int, endpoint: str,
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
    for index, (actor, target) in enumerate(((left, right), (right, left))):
        observed = features.valid[:, actor] & features.valid[:, target]
        for frame in np.flatnonzero(observed):
            nose = points[frame, actor, 0]
            if not np.isfinite(nose).all():
                continue
            if endpoint == "tail":
                tail = points[frame, target, 6]
                if np.isfinite(tail).all():
                    result[index][frame] = np.linalg.norm(nose - tail)
            elif endpoint == "head":
                triangle = points[frame, target, :3]
                if not np.isfinite(triangle).all():
                    continue
                hull = cv2.convexHull(triangle.astype(np.float32))
                if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
                    continue
                result[index][frame] = max(
                    0.0, -cv2.pointPolygonTest(hull, tuple(map(float, nose)), True)
                )
            else:
                raise ValueError("social endpoint must be head or tail")
    return result[0], result[1]


def snout_surface_distances(features: ExportedFeatures, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """Return directed snout-to-body convex-hull gaps, not center distances."""
    count = len(features.valid)
    result = [np.full(count, np.nan), np.full(count, np.nan)]
    points = features.collective_keypoints_cm
    if points is None:
        return result[0], result[1]
    left, right = int(features.pair_i[edge]), int(features.pair_j[edge])
    for index, (actor, target) in enumerate(((left, right), (right, left))):
        observed = features.valid[:, actor] & features.valid[:, target]
        for frame in np.flatnonzero(observed):
            nose = points[frame, actor, 0]
            body = points[frame, target]
            finite = np.isfinite(body).all(axis=1)
            if not np.isfinite(nose).all() or finite.sum() < 4:
                continue
            hull = cv2.convexHull(body[finite].astype(np.float32))
            if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
                continue
            result[index][frame] = max(
                0.0, -cv2.pointPolygonTest(hull, tuple(map(float, nose)), True)
            )
    return result[0], result[1]


def body_footprint_distances(
    features: ExportedFeatures, edge: int, *, max_body_lengths: float,
    frame_mask: np.ndarray | None = None,
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
    for frame in np.flatnonzero(observed):
        hulls = []
        for slot in (left, right):
            body = np.asarray(points[frame, slot, :6], float)
            body = body[np.isfinite(body).all(axis=1)]
            if len(body) < 4 or np.linalg.norm(np.ptp(body, axis=0)) > maximum_span:
                break
            hull = cv2.convexHull(body.astype(np.float32))
            if len(hull) < 3 or cv2.contourArea(hull) <= 1e-9:
                break
            hulls.append(hull)
        if len(hulls) != 2:
            continue
        first, second = hulls
        overlap, _ = cv2.intersectConvexConvex(first, second)
        if overlap > 1e-9:
            result[frame] = 0.
            continue
        result[frame] = min(
            max(0., -cv2.pointPolygonTest(target, tuple(map(float, point)), True))
            for source, target in ((first, second), (second, first))
            for point in source.reshape(-1, 2)
        )
    return result


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
