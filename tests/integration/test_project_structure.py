from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_reusable_modules_live_in_package_and_legacy_imports_still_work():
    package_engine = importlib.import_module("mouse_behavior.standard_behavior_engine")
    legacy_engine = importlib.import_module("standard_behavior_engine")

    assert package_engine.ENGINE_VERSION == "1.43.0-standard-behavior-engine"
    assert (
        legacy_engine.apply_standard_behavior_engine
        is package_engine.apply_standard_behavior_engine
    )
    assert (ROOT / "src" / "mouse_behavior" / "lightweight_behavior_inference.py").is_file()
    assert (ROOT / "src" / "mouse_behavior" / "pose_cache.py").is_file()
    assert (ROOT / "scripts" / "build_lightweight_pose_cache.py").is_file()


def test_cli_and_library_boundaries_are_explicit():
    lightweight = importlib.import_module("mouse_behavior.lightweight_behavior_inference")

    assert callable(lightweight.analyze)
    assert callable(lightweight.main)
    assert not hasattr(lightweight, "_PARSED_ARGUMENTS")


def test_obsolete_root_cli_wrappers_are_removed():
    for filename in (
        "_script_compat.py",
        "build_lightweight_pose_cache.py",
        "calibrate_standard_behavior.py",
        "rerun_beiyi_lightweight_rules.py",
        "sweep_standard_behavior.py",
        "validate_beiyi_extended_ethogram.py",
    ):
        assert not (ROOT / filename).exists(), filename

    for filename in (
        "build_lightweight_pose_cache.py",
        "calibrate_standard_behavior.py",
        "rerun_beiyi_lightweight_rules.py",
        "sweep_standard_behavior.py",
        "validate_beiyi_extended_ethogram.py",
    ):
        assert (ROOT / "scripts" / filename).is_file(), filename


def test_repository_has_executable_quality_and_test_layers():
    for relative in (
        ".quality-gate.toml",
        "scripts/run_quality.py",
        "scripts/validate_repository.py",
        "tests/unit",
        "tests/integration",
        "tests/regression",
        "tests/e2e",
    ):
        assert (ROOT / relative).exists(), relative
