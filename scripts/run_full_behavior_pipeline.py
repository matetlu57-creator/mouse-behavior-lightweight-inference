#!/usr/bin/env python3
"""CLI wrapper for the packaged full multi-stage behavior pipeline."""

from __future__ import annotations

from _bootstrap import ensure_importable


ensure_importable()

from mouse_behavior.full_pipeline.high_recall import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
