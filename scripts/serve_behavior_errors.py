"""Serve the read-only all-behavior error review on http://127.0.0.1:8776/."""
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.visualization.behavior_error_site import (
    DEFAULT_DIAGNOSTICS, DEFAULT_REPORT, create_error_review_server,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--port", type=int, default=8776)
    args = parser.parse_args()
    server = create_error_review_server(args.report, port=args.port, diagnostics_path=args.diagnostics)
    print(f"行为错误复核网站：http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"评测记录：{args.report}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
