"""Unit tests for cage-floor scale calibration primitives."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from mouse_behavior.preprocessing import scale_calibration as sc

RECT = np.asarray(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]], dtype=float
)


def _synthetic_floor_frame(
    quad: np.ndarray,
    *,
    shape: tuple[int, int] = (600, 800),
    band_px: float = 18.0,
    steps: int = 5,
) -> np.ndarray:
    """Grey floor inside a smooth shading band inside white walls.

    The band values stay inside the proposal mask, so the coarse proposal is
    inflated by roughly ``band_px``; the only sharp brightness step is the
    true floor/wall junction, which is what the refinement must find.
    """

    height, width = shape
    ordered = sc.order_quad_corners(quad)
    centroid = ordered.mean(axis=0)
    image = np.full((height, width, 3), 190, dtype=np.uint8)
    image[:, :30] = (200, 150, 100)  # saturated table outside the box

    def inflated(amount: float) -> np.ndarray:
        offset = ordered - centroid
        norm = np.linalg.norm(offset, axis=1, keepdims=True)
        return ordered + offset / np.maximum(norm, 1e-9) * amount

    for index in range(steps, 0, -1):
        value = 150 + int((190 - 150) * (steps - index) / steps)
        cv2.fillPoly(image, [inflated(band_px * index / steps).astype(np.int32)], (value,) * 3)
    cv2.fillPoly(image, [ordered.astype(np.int32)], (130,) * 3)
    return image


def test_order_quad_corners_is_rotation_stable():
    shuffled = np.roll(RECT, 2, axis=0)

    np.testing.assert_allclose(sc.order_quad_corners(shuffled), RECT)


def test_order_quad_corners_rejects_wrong_input():
    with pytest.raises(ValueError):
        sc.order_quad_corners(np.zeros((3, 2)))


def test_quad_edge_lengths_on_a_rectangle():
    edges = sc.quad_edge_lengths(RECT)

    assert edges == {
        "top_px": 400.0,
        "right_px": 300.0,
        "bottom_px": 400.0,
        "left_px": 300.0,
    }


def test_perspective_metrics_flag_a_trapezoid():
    flat = sc.perspective_metrics(RECT)
    trapezoid = np.asarray(
        [[200.0, 100.0], [400.0, 100.0], [500.0, 400.0], [100.0, 400.0]], dtype=float
    )

    assert flat["max_relative_edge_deviation"] == 0.0
    assert flat["width_edge_ratio"] == 1.0
    tilted = sc.perspective_metrics(trapezoid)
    assert tilted["max_relative_edge_deviation"] > 0.2


def test_cm_per_pixel_from_quad_matches_the_tape_measurement():
    scale = sc.cm_per_pixel_from_quad(RECT, width_cm=40.0, height_cm=30.0)

    assert scale["px_per_cm_width"] == 10.0
    assert scale["px_per_cm_height"] == 10.0
    assert scale["axis_consistency_ratio"] == 1.0
    assert scale["cm_per_pixel"] == 0.1


def test_cm_per_pixel_rejects_non_positive_box():
    with pytest.raises(ValueError):
        sc.cm_per_pixel_from_quad(RECT, width_cm=0.0, height_cm=30.0)


def test_homography_maps_floor_corners_onto_the_metric_plane():
    matrix, resolution = sc.homography_from_quad(RECT, width_cm=40.0, height_cm=30.0)

    mapped = cv2.perspectiveTransform(
        sc.order_quad_corners(RECT).astype(np.float32).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    expected = np.asarray(
        [
            [0.0, 0.0],
            [40.0 * resolution, 0.0],
            [40.0 * resolution, 30.0 * resolution],
            [0.0, 30.0 * resolution],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(mapped, expected, atol=1e-6)


def test_propose_floor_quad_finds_an_unshaded_floor():
    frame = _synthetic_floor_frame(RECT, band_px=0.0)

    quad = sc.propose_floor_quad(frame)

    assert quad is not None
    np.testing.assert_allclose(sc.order_quad_corners(quad), RECT, atol=2.0)


def test_refine_snaps_back_through_the_shading_band():
    """The coarse proposal is inflated by the band; refinement must undo it."""

    frame = _synthetic_floor_frame(RECT, band_px=18.0)
    coarse = sc.propose_floor_quad(frame)

    assert coarse is not None
    inflated = sc.quad_edge_lengths(coarse)["top_px"]
    assert inflated > 400.0 + 10.0  # the band really did leak into the proposal

    refined = sc.refine_quad_to_edges(frame, coarse)

    np.testing.assert_allclose(sc.order_quad_corners(refined), RECT, atol=3.0)


def test_refine_ignores_a_mouse_crossing_the_floor_edge():
    frame = _synthetic_floor_frame(RECT, band_px=18.0)
    cv2.ellipse(frame, (300, 100), (40, 25), 0, 0, 360, (20, 20, 20), -1)

    refined = sc.refine_quad_to_edges(frame, sc.propose_floor_quad(frame))

    np.testing.assert_allclose(sc.order_quad_corners(refined), RECT, atol=3.0)


def test_refine_rejects_non_images():
    with pytest.raises(ValueError):
        sc.refine_quad_to_edges(np.zeros((10, 10)), RECT)


def test_calibration_round_trip_through_disk(tmp_path):
    payload = sc.build_calibration(
        RECT, width_cm=40.0, height_cm=30.0, source_video=None, method="unit_test"
    )
    target = tmp_path / "calib.json"

    sc.save_calibration(payload, target)
    loaded = sc.load_calibration(target)

    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1
    np.testing.assert_allclose(np.asarray(loaded["quad_px"], dtype=float), RECT)
    assert loaded["scale"]["cm_per_pixel"] == 0.1
    assert loaded["corner_order"] == list(sc.CORNER_ORDER)


def test_load_calibration_rejects_foreign_payloads(tmp_path):
    bad_schema = tmp_path / "old.json"
    bad_schema.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    bad_quad = tmp_path / "quad.json"
    bad_quad.write_text(
        json.dumps({"schema_version": 1, "quad_px": [[0, 0], [1, 1]]}), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        sc.load_calibration(bad_schema)
    with pytest.raises(ValueError):
        sc.load_calibration(bad_quad)