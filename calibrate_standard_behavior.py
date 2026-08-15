"""Backward-compatible CLI path for ``scripts/calibrate_standard_behavior.py``."""

from __future__ import annotations

from _script_compat import load_script


_SCRIPT = load_script("calibrate_standard_behavior.py", globals())


if __name__ == "__main__":
    raise SystemExit(_SCRIPT["main"]())
