"""Run one lightweight analysis through the reusable pipeline API.

Install the project first with ``python -m pip install -e .``.  The example
does not build a pose cache; pass a completed cache produced by the pose-cache
script or another approved upstream pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mouse_behavior.core.pipeline import LightweightPipeline
from mouse_behavior.utils.logger import configure_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/profiles/balanced.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--expected-mice", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    run = LightweightPipeline(
        args.config,
        expected_mice=args.expected_mice,
        sample_stride=args.sample_stride,
    ).run(args.video, args.yolo_cache, args.output, fps=args.fps)
    LOGGER.info("analysis complete: output=%s", run.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
