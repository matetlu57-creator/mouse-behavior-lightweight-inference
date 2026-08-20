from __future__ import annotations

from pathlib import Path

import pytest

from mouse_behavior.config import load_config


def test_load_config_resolves_relative_extends_and_deep_merges(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "profiles" / "balanced.yaml"
    child.parent.mkdir()
    base.write_text(
        """\nengine:\n  enabled: true\n  thresholds:\n    enter: 2\n    exit: 1\nitems: [base]\n""",
        encoding="utf-8",
    )
    child.write_text(
        """\nextends: ../base.yaml\nengine:\n  thresholds:\n    enter: 3\nprofile: balanced\n""",
        encoding="utf-8",
    )

    result = load_config(child)

    assert result["engine"]["enabled"] is True
    assert result["engine"]["thresholds"] == {"enter": 3, "exit": 1}
    assert result["items"] == ["base"]
    assert result["profile"] == "balanced"


def test_load_config_rejects_inheritance_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="循环"):
        load_config(first)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
