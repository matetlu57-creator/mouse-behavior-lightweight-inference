"""Backward-compatible CLI path for ``scripts/validate_beiyi_extended_ethogram.py``."""

from __future__ import annotations

from _script_compat import load_script


_SCRIPT = load_script("validate_beiyi_extended_ethogram.py", globals())


if __name__ == "__main__":
    raise SystemExit(_SCRIPT["main"]())
