from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_reusable_modules_live_in_package():
    package_engine = importlib.import_module("mouse_behavior.standard_behavior_engine")
    renamed_lightweight = importlib.import_module(
        "mouse_behavior.lightweight_cache_behavior_analysis"
    )

    assert package_engine.ENGINE_VERSION == "1.43.0-standard-behavior-engine"
    assert callable(renamed_lightweight.main)
    assert (ROOT / "src" / "mouse_behavior" / "lightweight_behavior_inference.py").is_file()
    assert (ROOT / "src" / "mouse_behavior" / "pose_cache.py").is_file()
    assert (ROOT / "scripts" / "build_lightweight_pose_cache.py").is_file()
    for relative in (
        "src/mouse_behavior/behavior/standard_evidence.py",
        "src/mouse_behavior/behavior/standard_fsm.py",
        "src/mouse_behavior/behavior/ethogram.py",
        "src/mouse_behavior/behavior/pair_analysis.py",
        "src/mouse_behavior/preprocessing/geometry.py",
        "src/mouse_behavior/preprocessing/kinematics.py",
        "src/mouse_behavior/preprocessing/pair_features.py",
        "src/mouse_behavior/preprocessing/arena_learning.py",
        "src/mouse_behavior/tracking/cache.py",
        "src/mouse_behavior/io/arena_boundary.py",
        "src/mouse_behavior/visualization/overlay.py",
        "src/mouse_behavior/visualization/rendering.py",
        "src/mouse_behavior/full_pipeline/extractor_base.py",
        "src/mouse_behavior/full_pipeline/high_recall.py",
        "scripts/run_full_behavior_pipeline.py",
    ):
        assert (ROOT / relative).is_file(), relative

    lightweight = importlib.import_module("mouse_behavior.lightweight_behavior_inference")
    for name in (
        "_pair_prefilter",
        "_pair_metrics",
        "_extended_individual_and_group_events",
        "extract_behavior_clips",
    ):
        assert hasattr(lightweight, name), name


def test_cli_and_library_boundaries_are_explicit():
    lightweight = importlib.import_module("mouse_behavior.lightweight_behavior_inference")

    assert callable(lightweight.analyze)
    assert callable(lightweight.main)
    assert not hasattr(lightweight, "_PARSED_ARGUMENTS")


def test_repository_root_contains_no_python_entrypoints():
    for filename in (
        "_script_compat.py",
        "adaptive_arena_boundary.py",
        "annotation_website_export.py",
        "build_lightweight_pose_cache.py",
        "calibrate_standard_behavior.py",
        "lightweight_behavior_inference.py",
        "lightweight_cache_behavior_analysis.py",
        "mask_trigger_controller.py",
        "mouse_chase_attack_extractor_base.py",
        "mouse_chase_attack_high_recall.py",
        "nvenc_video_writer.py",
        "rerun_beiyi_lightweight_rules.py",
        "standard_behavior_engine.py",
        "sweep_standard_behavior.py",
        "validate_beiyi_extended_ethogram.py",
    ):
        assert not (ROOT / filename).exists(), filename

    assert list(ROOT.glob("*.py")) == []

    for filename in (
        "build_lightweight_pose_cache.py",
        "calibrate_standard_behavior.py",
        "rerun_beiyi_lightweight_rules.py",
        "run_full_behavior_pipeline.py",
        "sweep_standard_behavior.py",
        "validate_beiyi_extended_ethogram.py",
    ):
        assert (ROOT / "scripts" / filename).is_file(), filename


def test_reusable_package_uses_logging_instead_of_print() -> None:
    offenders: list[str] = []
    package_root = ROOT / "src" / "mouse_behavior"
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        ):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


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
