"""Backward-compatible import path for mask-trigger decisions."""

from __future__ import annotations

import sys
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mouse_behavior._compat import reexport


_IMPLEMENTATION = reexport("mouse_behavior.mask_trigger_controller", globals())
