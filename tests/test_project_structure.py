from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reusable_modules_live_in_package_and_legacy_imports_still_work():
    package_engine = importlib.import_module("mouse_behavior.standard_behavior_engine")
    legacy_engine = importlib.import_module("standard_behavior_engine")

    assert package_engine.ENGINE_VERSION == "1.43.0-standard-behavior-engine"
    assert legacy_engine.apply_standard_behavior_engine is package_engine.apply_standard_behavior_engine
    assert (ROOT / "src" / "mouse_behavior" / "lightweight_behavior_inference.py").is_file()
    assert (ROOT / "src" / "mouse_behavior" / "pose_cache.py").is_file()
    assert (ROOT / "scripts" / "build_lightweight_pose_cache.py").is_file()


def test_cli_and_library_boundaries_are_explicit():
    lightweight = importlib.import_module("mouse_behavior.lightweight_behavior_inference")

    assert callable(lightweight.analyze)
    assert callable(lightweight.main)
    assert not hasattr(lightweight, "_PARSED_ARGUMENTS")
