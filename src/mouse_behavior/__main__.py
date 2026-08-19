"""Run the lightweight inference entry point with ``python -m mouse_behavior``."""
from __future__ import annotations

from .lightweight_behavior_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
