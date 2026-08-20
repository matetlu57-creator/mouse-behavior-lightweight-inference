#!/usr/bin/env python3
"""Run the canonical repository-boundary and publication-safety checks."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_repository import check  # noqa: E402


def main() -> int:
    errors = check(ROOT)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("repository validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
