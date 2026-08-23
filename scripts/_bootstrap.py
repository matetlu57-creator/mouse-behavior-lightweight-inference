"""Make the repository's ``src`` package importable for direct script runs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def ensure_importable() -> None:
    """Add the local ``src`` package once for direct repository script runs."""

    value = str(SRC_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


ensure_importable()
