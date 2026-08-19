"""Root-level compatibility loader for scripts moved under ``scripts/``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


def load_script(script_name: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Load a relocated script without executing its CLI automatically."""

    repo_root = Path(__file__).resolve().parent
    scripts_root = repo_root / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    loaded = runpy.run_path(
        str(scripts_root / script_name),
        run_name=f"_compat_{Path(script_name).stem}",
    )
    for name, value in loaded.items():
        if name not in {"__name__", "__file__", "__package__", "__spec__", "__builtins__"}:
            namespace[name] = value
    return loaded
