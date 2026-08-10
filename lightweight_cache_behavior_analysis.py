#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible module name for the renamed lightweight entry point.

The canonical implementation is ``lightweight_behavior_inference.py``.
This shim keeps existing scripts and notebooks that import the former module
name working while the project is migrated to its new name.
"""

from lightweight_behavior_inference import *  # noqa: F401,F403
from lightweight_behavior_inference import main


if __name__ == "__main__":
    raise SystemExit(main())
