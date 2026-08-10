from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_mask_trigger_strict_preserves_mask_dependency():
    mod = load("mask_trigger_test", ROOT / "mask_trigger_controller.py")
    ctrl = mod.MaskTriggerController(
        {"enabled": True, "result_preserving_only": True},
        {"identity_cost_weight": 0.32},
        {"mask_weight": 0.26},
    )
    decision = ctrl.decide(1, [], {}, [], SimpleNamespace(frame_stats={}))
    assert decision.run_mask is True
    assert decision.reason == "strict_downstream_mask_dependency"


def test_mask_trigger_aggressive_low_risk_can_skip():
    mod = load("mask_trigger_test2", ROOT / "mask_trigger_controller.py")
    ctrl = mod.MaskTriggerController(
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
    mod = load("nvenc_writer_test", ROOT / "nvenc_video_writer.py")
    assert callable(mod.ffmpeg_nvenc_available)
    assert callable(mod.create_video_writer)
    assert hasattr(mod.NVENCWriter, "write")
    assert hasattr(mod.NVENCWriter, "release")

def test_main_wires_identity_cascade_into_runtime_config():
    source = (ROOT / "mouse_chase_attack_high_recall.py").read_text(encoding="utf-8")
    assert '"identity_cascade": dict(performance_cfg.get("identity_cascade", {}))' in source

