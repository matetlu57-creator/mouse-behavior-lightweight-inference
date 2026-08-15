"""Make the repository's ``src`` package importable for direct script runs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def ensure_importable() -> None:
    """Add local source roots once, without requiring editable installation."""

    for path in (SRC_ROOT, REPO_ROOT, SCRIPTS_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


ensure_importable()
