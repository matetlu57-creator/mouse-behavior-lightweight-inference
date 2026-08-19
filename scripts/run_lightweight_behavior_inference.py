#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for the reusable lightweight inference module."""

from __future__ import annotations

from _bootstrap import ensure_importable


ensure_importable()

from mouse_behavior.lightweight_behavior_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
