"""Physical scale calibration from a user-confirmed cage-floor quadrilateral.

The historical pipeline derived the scene scale from an *assumed* mouse body
length, which couples posture to scale: a frame where the mouse rears
foreshortens its nose-to-tail span and therefore biases every centimetre
feature.  This module inverts the causality: the cage floor is a physical
object of known size, so four clicked floor corners plus the real inner
dimensions define the scale, and the true body length becomes a derived
quantity.

Why four points and not two measurements
----------------------------------------
A top-down camera is rarely perfectly perpendicular.  On the project footage
the floor appears as a slightly tilted trapezoid, so a single "width in
pixels" is not defined.  Four corners also carry the perspective information
needed for a homography that maps image pixels onto the true floor plane.

The existing ``adaptive_arena`` boundary cannot be reused for this: it learns
the *mouse activity* heatmap (mice never touch every wall) and then shrinks it
by ``boundary_expansion_ratio`` (0.97 by default), and ``_coerce_polygon``
collapses any polygon into an axis-aligned bounding box, which would silently
absorb the white walls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from .arena_learning import _video_fingerprint

CALIBRATION_SCHEMA_VERSION = 1

# Corner order used everywhere in this module and in the calibration file.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def order_quad_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""

    quad = np.asarray(points, dtype=float).reshape(-1, 2)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError(f"expected four finite 2-D points, got shape={quad.shape}")
    total = quad[:, 0] + quad[:, 1]
    difference = quad[:, 0] - quad[:, 1]
    return np.asarray(
        [
            quad[int(np.argmin(total))],
            quad[int(np.argmax(difference))],
            quad[int(np.argmax(total))],
            quad[int(np.argmin(difference))],
        ],
        dtype=float,
    )


def quad_edge_lengths(quad: np.ndarray) -> dict[str, float]:
    """Return the four edge lengths of an ordered quadrilateral."""

    ordered = order_quad_corners(quad)
    top_left, top_right, bottom_right, bottom_left = ordered
    return {
        "top_px": float(np.linalg.norm(top_right - top_left)),
        "right_px": float(np.linalg.norm(bottom_right - top_right)),
        "bottom_px": float(np.linalg.norm(bottom_right - bottom_left)),
        "left_px": float(np.linalg.norm(bottom_left - top_left)),
    }


def perspective_metrics(quad: np.ndarray) -> dict[str, float]:
    """Quantify how far the floor is from an axis-aligned rectangle.

    ``max_relative_edge_deviation`` is the number that decides whether a single
    ``cm_per_pixel`` is acceptable: it compares the opposing edges that image
    the same physical dimension.  A few tenths of a percent is harmless; beyond
    a couple of percent the homography path should be used.
    """

    edges = quad_edge_lengths(quad)
    ordered = order_quad_corners(quad)
    top_left, top_right, _, bottom_left = ordered

    def tilt_deg(a: np.ndarray, b: np.ndarray) -> float:
        vector = b - a
        return float(np.degrees(np.arctan2(vector[1], vector[0])))

    width_pair = (edges["top_px"], edges["bottom_px"])
    height_pair = (edges["left_px"], edges["right_px"])
    deviations = []
    for low, high in (width_pair, height_pair):
        span = max(low, high)
        deviations.append(abs(high - low) / span if span > 1e-9 else 0.0)
    return {
        "top_tilt_deg": tilt_deg(top_left, top_right),
        "left_tilt_deg": tilt_deg(top_left, bottom_left) - 90.0,
        "width_edge_ratio": width_pair[0] / width_pair[1] if width_pair[1] > 1e-9 else float("nan"),
        "height_edge_ratio": height_pair[0] / height_pair[1] if height_pair[1] > 1e-9 else float("nan"),
        "max_relative_edge_deviation": float(max(deviations)),
    }


def cm_per_pixel_from_quad(
    quad: np.ndarray,
    width_cm: float,
    height_cm: float,
) -> dict[str, float]:
    """Derive the scene scale from floor corners and the real inner size."""

    for name, value in (("width_cm", width_cm), ("height_cm", height_cm)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    edges = quad_edge_lengths(quad)
    px_per_cm_width = 0.5 * (edges["top_px"] + edges["bottom_px"]) / width_cm
    px_per_cm_height = 0.5 * (edges["left_px"] + edges["right_px"]) / height_cm
    if min(px_per_cm_width, px_per_cm_height) <= 1e-9:
        raise ValueError("floor quad is degenerate; cannot derive a scale")
    px_per_cm = 0.5 * (px_per_cm_width + px_per_cm_height)
    return {
        "px_per_cm_width": px_per_cm_width,
        "px_per_cm_height": px_per_cm_height,
        "axis_consistency_ratio": px_per_cm_width / px_per_cm_height,
        "px_per_cm": px_per_cm,
        "cm_per_pixel": 1.0 / px_per_cm,
    }


def homography_from_quad(
    quad: np.ndarray,
    width_cm: float,
    height_cm: float,
    px_per_cm: float | None = None,
) -> tuple[np.ndarray, float]:
    """Map image pixels onto the true floor plane, in centimetre units.

    The destination rectangle starts at the origin and is scaled by
    ``px_per_cm`` so the homography output stays in pixels of a virtual
    metric camera; divide by ``px_per_cm`` to obtain centimetres.
    """

    scale = cm_per_pixel_from_quad(quad, width_cm, height_cm)
    resolution = float(px_per_cm) if px_per_cm is not None else float(scale["px_per_cm"])
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(f"px_per_cm must be finite and > 0, got {px_per_cm}")
    ordered = order_quad_corners(quad).astype(np.float32)
    destination = np.asarray(
        [
            [0.0, 0.0],
            [width_cm * resolution, 0.0],
            [width_cm * resolution, height_cm * resolution],
            [0.0, height_cm * resolution],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return matrix, resolution


def propose_floor_quad(
    frame: np.ndarray,
    *,
    max_saturation: int = 30,
    min_value: int = 70,
    max_value: int = 160,
    min_area_ratio: float = 0.05,
) -> Optional[np.ndarray]:
    """Auto-suggest the cage floor quadrilateral from one frame.

    The floor of a typical open-field box is a neutral mid-grey surface bounded
    by near-white walls, while the surrounding table is saturated.  On the
    project footage the floor sits at V in roughly [108, 154] with S <= 23,
    the walls at V in [164, 195], and the table at S >= 48, so a low-saturation
    mid-value band isolates the floor; the largest connected component is the
    floor itself even when a laptop or other grey object sits outside the box.
    Mice punch holes in the mask, which the external contour ignores.  The
    suggestion is only a starting point: a human must confirm or nudge the four
    corners.
    """

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame must be an HxWx3 image, got shape={image.shape}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] <= max_saturation)
        & (hsv[:, :, 2] >= min_value)
        & (hsv[:, :, 2] <= max_value)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_ratio * float(image.shape[0] * image.shape[1]):
        return None
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
    if len(approx) == 4:
        return order_quad_corners(approx.astype(float))
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(hull) < 4:
        return None
    rect = cv2.minAreaRect(hull.astype(np.float32))
    return order_quad_corners(np.asarray(cv2.boxPoints(rect), dtype=float))


def _fit_line_robust(points: np.ndarray, max_deviation_px: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through points, dropping outliers and refitting."""

    current = np.asarray(points, dtype=float)
    for _ in range(2):
        if len(current) < 2:
            break
        mean = current.mean(axis=0)
        _, _, vt = np.linalg.svd(current - mean, full_matrices=False)
        direction = vt[0]
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        deviation = np.abs((current - mean) @ normal)
        keep = deviation <= max_deviation_px
        if keep.sum() >= max(4, len(current) // 2):
            current = current[keep]
    mean = current.mean(axis=0)
    _, _, vt = np.linalg.svd(current - mean, full_matrices=False)
    return mean, vt[0]


def _line_intersection(
    point_a: np.ndarray, direction_a: np.ndarray,
    point_b: np.ndarray, direction_b: np.ndarray,
) -> Optional[np.ndarray]:
    matrix = np.column_stack([direction_a, -direction_b])
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        return None
    try:
        factors = np.linalg.solve(matrix, point_b - point_a)
    except np.linalg.LinAlgError:
        return None
    return point_a + factors[0] * direction_a


def refine_quad_to_edges(
    frame: np.ndarray,
    quad: np.ndarray,
    *,
    search_px: int = 28,
    samples_per_edge: int = 64,
) -> np.ndarray:
    """Snap a coarse floor quad onto the floor/wall brightness step.

    The colour band that proposes the quad also catches the shaded strip of
    wall next to the floor, which biases every edge outward by tens of pixels.
    The true junction is the strongest brightness step near each coarse edge,
    so each edge is re-fit through the per-sample gradient maxima and the four
    refined lines are intersected into corners.  Mice or tails that cross an
    edge are rejected by the robust line fit.
    """

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame must be an HxWx3 image, got shape={image.shape}")
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float)
    height, width = gray.shape
    ordered = order_quad_corners(quad)
    centroid = ordered.mean(axis=0)
    offsets = np.arange(-search_px, search_px + 1, dtype=float)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(4):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        midpoint = 0.5 * (start + end)
        outward = midpoint - centroid
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-9:
            lines.append(_fit_line_robust(np.vstack([start, end])))
            continue
        normal = outward / norm
        tangents = np.linspace(0.10, 0.90, samples_per_edge)[:, None]
        base = start[None, :] + tangents * (end - start)[None, :]
        refined: list[np.ndarray] = []
        for point in base:
            probe = point[None, :] + normal[None, :] * offsets[:, None]
            inside = (
                (probe[:, 0] >= 0) & (probe[:, 0] <= width - 1)
                & (probe[:, 1] >= 0) & (probe[:, 1] <= height - 1)
            )
            if not inside.all():
                continue
            values = gray[
                np.round(probe[:, 1]).astype(int),
                np.round(probe[:, 0]).astype(int),
            ]
            gradient = np.abs(np.diff(values))
            if gradient.size == 0:
                continue
            best = int(np.argmax(gradient))
            refined.append(point + normal * offsets[best])
        if len(refined) >= 8:
            lines.append(_fit_line_robust(np.asarray(refined)))
        else:
            lines.append(_fit_line_robust(np.vstack([start, end])))

    corners = []
    for index in range(4):
        point_a, direction_a = lines[index]
        point_b, direction_b = lines[(index + 1) % 4]
        corner = _line_intersection(point_a, direction_a, point_b, direction_b)
        if corner is None:
            corner = ordered[(index + 1) % 4]
        corners.append(corner)
    return order_quad_corners(np.asarray(corners, dtype=float))


def build_calibration(
    quad: np.ndarray,
    *,
    width_cm: float,
    height_cm: float,
    source_video: str | os.PathLike[str] | None = None,
    method: str = "manual",
) -> dict[str, Any]:
    """Assemble the auditable payload stored in ``scale.calibration_file``."""

    ordered = order_quad_corners(quad)
    scale = cm_per_pixel_from_quad(ordered, width_cm, height_cm)
    matrix, resolution = homography_from_quad(ordered, width_cm, height_cm)
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": str(method),
        "corner_order": list(CORNER_ORDER),
        "quad_px": [[round(float(x), 3), round(float(y), 3)] for x, y in ordered],
        "box_inner_cm": {"width": float(width_cm), "height": float(height_cm)},
        "scale": {key: float(value) for key, value in scale.items()},
        "homography_px_per_cm": float(resolution),
        "homography": [[float(value) for value in row] for row in matrix],
        "diagnostics": {
            key: float(value)
            for key, value in {**quad_edge_lengths(ordered), **perspective_metrics(ordered)}.items()
        },
    }
    if source_video is not None:
        resolved = str(os.path.abspath(source_video))
        payload["source_video"] = resolved
        payload["source_video_fingerprint"] = _video_fingerprint(resolved)
    return payload


def save_calibration(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    target = os.fspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_calibration(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(os.fspath(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"calibration JSON top level must be an object: {path}")
    if int(payload.get("schema_version", 0)) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration schema: {payload.get('schema_version')}")
    quad = np.asarray(payload.get("quad_px", []), dtype=float).reshape(-1, 2)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError(f"calibration quad_px must hold four finite points: {path}")
    return dict(payload)


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CORNER_ORDER",
    "order_quad_corners",
    "quad_edge_lengths",
    "perspective_metrics",
    "cm_per_pixel_from_quad",
    "homography_from_quad",
    "propose_floor_quad",
    "refine_quad_to_edges",
    "build_calibration",
    "save_calibration",
    "load_calibration",
]