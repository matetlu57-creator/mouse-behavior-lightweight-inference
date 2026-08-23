from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_behavior import mask_trigger_controller, nvenc_video_writer

ROOT = Path(__file__).resolve().parents[2]


def test_mask_trigger_strict_preserves_mask_dependency():
    ctrl = mask_trigger_controller.MaskTriggerController(
        {"enabled": True, "result_preserving_only": True},
        {"identity_cost_weight": 0.32},
        {"mask_weight": 0.26},
    )
    decision = ctrl.decide(1, [], {}, [], SimpleNamespace(frame_stats={}))
    assert decision.run_mask is True
    assert decision.reason == "strict_downstream_mask_dependency"


def test_mask_trigger_aggressive_low_risk_can_skip():
    ctrl = mask_trigger_controller.MaskTriggerController(
        {
            "enabled": True,
            "result_preserving_only": False,
            "force_refresh_interval_frames": 15,
            "overlap_iou_threshold": 0.02,
        },
        {"identity_cost_weight": 0.32},
        {"mask_weight": 0.26},
    )
    det = SimpleNamespace(bbox_xyxy=np.array([0, 0, 10, 10], dtype=float))
    decision = ctrl.decide(1, [det], {}, [], SimpleNamespace(frame_stats={}))
    assert decision.run_mask is False
    assert decision.reason == "low_risk"


def test_nvenc_module_exposes_fallback_factory():
    assert callable(nvenc_video_writer.ffmpeg_nvenc_available)
    assert callable(nvenc_video_writer.create_video_writer)
    assert hasattr(nvenc_video_writer.NVENCWriter, "write")
    assert hasattr(nvenc_video_writer.NVENCWriter, "release")


def test_main_wires_identity_cascade_into_runtime_config():
    source = (ROOT / "src" / "mouse_behavior" / "full_pipeline" / "high_recall.py").read_text(
        encoding="utf-8"
    )
    assert '"identity_cascade": dict(performance_cfg.get("identity_cascade", {}))' in source
