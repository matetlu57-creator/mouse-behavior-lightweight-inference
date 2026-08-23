"""Run the full multi-stage behavior pipeline with ``python -m``."""

from __future__ import annotations

from .high_recall import main


if __name__ == "__main__":
    raise SystemExit(main())
