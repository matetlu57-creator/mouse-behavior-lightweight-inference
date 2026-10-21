"""Unit tests for the unified scene-scale resolution policy (Step 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mouse_behavior.behavior.collective_state import estimate_morphology
from mouse_behavior.evaluation.exported_dataset import (
    ExportedFeatures,
    load_exported_features,
)
from mouse_behavior.preprocessing import scale_calibration as sc
from mouse_behavior.preprocessing.scale_resolution import (
    LEGACY_REFERENCE_BODY_CM,
    default_scale_config,
    resolve_reference_body_cm,
    resolve_scale,
    scale_block_of,
)

QUAD = np.array([[0.0, 0.0], [775.0, 0.0], [775.0, 775.0], [0.0, 775.0]])
ARENA_CPP = 0.09548616806593231


def _make_calibration(tmp_path, *, with_subject=True, source_video=None, name="rig.json"):
    payload = sc.build_calibration(
        QUAD, width_cm=100.0, height_cm=100.0, source_video=source_video
    )
    if with_subject:
        payload["subject_metrics"] = {"unified_body_length_cm": 10.7}
    path = tmp_path / name
    sc.save_calibration(payload, path)
    return path


def _cal_cfg(path, **extra):
    cfg = {
        "mode": "calibration",
        "calibration_file": str(path),
        "assumed_mouse_body_length_cm": 10.7,
    }
    cfg.update(extra)
    return cfg


def test_calibration_mode_resolves_scale_and_subject(tmp_path):
    path = _make_calibration(tmp_path)
    resolved = resolve_scale(_cal_cfg(path))
    assert resolved.mode == "calibration"
    assert resolved.cm_per_pixel == pytest.approx(1.0 / 7.75, rel=1e-6)
    assert resolved.reference_body_cm == pytest.approx(10.7)
    assert resolved.calibration_path == str(path)
    assert resolved.warnings == ()


def test_calibration_without_subject_metrics_falls_back_with_warning(tmp_path):
    path = _make_calibration(tmp_path, with_subject=False)
    resolved = resolve_scale(_cal_cfg(path))
    assert resolved.reference_body_cm == pytest.approx(10.7)
    assert any("subject_metrics" in w for w in resolved.warnings)
    with pytest.raises(ValueError):
        resolve_scale({k: v for k, v in _cal_cfg(path).items() if k != "assumed_mouse_body_length_cm"})


def test_calibration_fingerprint_mismatch_warns_but_resolves(tmp_path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a" * 128)
    video_b.write_bytes(b"b" * 256)
    path = _make_calibration(tmp_path, source_video=video_a)
    same = resolve_scale(_cal_cfg(path), source_video=video_a)
    assert same.warnings == ()
    other = resolve_scale(_cal_cfg(path), source_video=video_b)
    assert any("different source video" in w for w in other.warnings)
    assert other.cm_per_pixel == same.cm_per_pixel


def test_fixed_mode_and_missing_value():
    resolved = resolve_scale({"mode": "fixed", "cm_per_pixel": 0.5, "assumed_mouse_body_length_cm": 10.7})
    assert resolved.cm_per_pixel == 0.5
    assert resolved.reference_body_cm == 10.7
    with pytest.raises(ValueError):
        resolve_scale({"mode": "fixed", "assumed_mouse_body_length_cm": 10.7})


def test_body_length_mode_scales_each_video_by_unified_measured_body(tmp_path):
    # [STEP4+ 2026-10-05] 用户规定流程：逐视频像素体长 × 统一实测体长 = 尺度。
    cfg = {"mode": "body_length", "assumed_mouse_body_length_cm": 10.7}
    with pytest.raises(ValueError):
        resolve_scale(cfg)
    resolved = resolve_scale(cfg, reference_body_px=82.5)
    assert resolved.cm_per_pixel == pytest.approx(10.7 / 82.5)
    assert resolved.warnings == ()
    path = _make_calibration(tmp_path)
    unified_cfg = {"mode": "body_length", "calibration_file": str(path)}
    resolved = resolve_scale(unified_cfg, reference_body_px=82.5)
    assert resolved.reference_body_cm == pytest.approx(10.7)
    assert resolved.cm_per_pixel == pytest.approx(10.7 / 82.5)


def test_missing_or_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_scale({})
    with pytest.raises(ValueError):
        resolve_scale({"mode": "magic"})
    with pytest.raises(ValueError):
        resolve_reference_body_cm({})


def test_fixed_mode_requires_assumed_body_length():
    with pytest.raises(ValueError):
        resolve_scale({"mode": "fixed", "cm_per_pixel": 0.5})


def test_calibration_requires_file():
    with pytest.raises(ValueError):
        resolve_scale({"mode": "calibration", "assumed_mouse_body_length_cm": 10.7})


def test_relative_calibration_path_resolves_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_scale({"mode": "calibration", "calibration_file": "configs/scale/rig_a.json"})
    assert resolved.cm_per_pixel == pytest.approx(ARENA_CPP, rel=1e-9)
    assert resolved.reference_body_cm == pytest.approx(7.918)


def test_default_scale_config_is_unified_per_video_body_length():
    cfg = default_scale_config()
    assert cfg["mode"] == "body_length"
    resolved = resolve_scale(cfg, reference_body_px=82.45)
    assert resolved.cm_per_pixel == pytest.approx(7.918 / 82.45)
    assert resolved.reference_body_cm == pytest.approx(7.918)


def test_scale_block_of_accepts_full_config_or_block():
    block = {"mode": "fixed", "cm_per_pixel": 0.25, "assumed_mouse_body_length_cm": 9.0}
    assert scale_block_of({"scale": block}) == block
    assert scale_block_of(block) == block
    assert scale_block_of(None) == {}


def test_resolve_reference_body_cm_needs_no_pixel_evidence(tmp_path):
    path = _make_calibration(tmp_path)
    assert resolve_reference_body_cm(_cal_cfg(path)) == pytest.approx(10.7)
    assert resolve_reference_body_cm({"mode": "body_length", "assumed_mouse_body_length_cm": 9.5}) == 9.5


def test_estimate_morphology_requires_reference_when_blind():
    keypoints = np.full((4, 2, 7, 2), np.nan)
    valid = np.zeros((4, 2), dtype=bool)
    with pytest.raises(ValueError):
        estimate_morphology(keypoints, valid)
    lengths, widths, _ = estimate_morphology(keypoints, valid, reference_body_cm=10.7)
    assert np.allclose(lengths, 10.7)
    assert np.allclose(widths, max(10.7 * 0.30, 1.0))


def _tiny_tracks(path: Path, frames: int = 6) -> None:
    payload = []
    for frame in range(frames):
        detections = []
        for track in (1, 2):
            x = 100.0 + 10.0 * track + frame
            keypoints = [[x + dx, 100.0 + dy, 0.9] for dx, dy in
                         [(0, 0), (3, -2), (-3, -2), (6, 2), (10, 4), (10, -4), (16, 0)]]
            detections.append(
                {
                    "track_id": track,
                    "keypoints": keypoints,
                    "box": [x - 8, 88.0, 32.0, 24.0],
                    "detection_confidence": 0.9,
                }
            )
        payload.append({"frame": frame, "detections": detections})
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_exported_features_uses_unified_default_scale(tmp_path):
    tracks_path = tmp_path / "tracks.json"
    _tiny_tracks(tracks_path)
    features = load_exported_features(tracks_path, fps=30.0)
    # The current 74 cm arena calibration provides the shared 7.918 cm ruler.
    assert features.cm_per_pixel == pytest.approx(7.918 / 16.0)
    assert features.reference_body_cm == pytest.approx(7.918)


def test_dataclass_default_uses_the_current_reference_scale():
    assert LEGACY_REFERENCE_BODY_CM == 7.918
    field = ExportedFeatures.__dataclass_fields__["reference_body_cm"]
    assert field.default == 7.918

def test_robust_reference_body_px_rejects_pose_broken_tracks():
    from mouse_behavior.preprocessing.scale_resolution import robust_reference_body_px

    values = [55.7, 74.0, 80.9, 80.9, 81.7, 83.0, 83.1, 84.9, 88.1, 89.8]
    robust = robust_reference_body_px(values)
    naive = float(np.median(values))
    assert robust > naive  # collapsed-tail track must not shrink the ruler
    with pytest.raises(ValueError):
        robust_reference_body_px([])
