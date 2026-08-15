"""Backward-compatible CLI path for ``scripts/rerun_beiyi_lightweight_rules.py``."""

from __future__ import annotations

from _script_compat import load_script


_SCRIPT = load_script("rerun_beiyi_lightweight_rules.py", globals())


if __name__ == "__main__":
    raise SystemExit(_SCRIPT["main"]())
