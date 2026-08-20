from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _pythonpath_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    entries = [str(ROOT / "src")]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def test_package_and_script_help_are_runnable() -> None:
    commands = (
        [sys.executable, "-m", "mouse_behavior", "--help"],
        [
            sys.executable,
            str(ROOT / "scripts" / "run_lightweight_behavior_inference.py"),
            "--help",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=_pythonpath_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()
