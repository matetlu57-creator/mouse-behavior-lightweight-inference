#!/usr/bin/env python3
"""Generate the reconciled Dataset A (429) + Dataset B (605) evaluation & failure report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.reconciled_evaluation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_A_ROOT,
    DEFAULT_DATASET_B_ROOT,
    DEFAULT_OUTPUT_JSON,
    build_summary_markdown,
    evaluate_and_reconcile_datasets,
    write_reconciled_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-a",
        type=Path,
        default=DEFAULT_DATASET_A_ROOT,
        help="Path to Dataset A (429-clip social dataset root)",
    )
    parser.add_argument(
        "--dataset-b",
        type=Path,
        default=DEFAULT_DATASET_B_ROOT,
        help="Path to Dataset B (605-clip cleaned dataset root)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        dest="output",
        help="Output JSON report path (default: outputs/reconciled_behavior_evaluation.json)",
    )
    parser.add_argument(
        "--dataset-b-source-scales",
        type=Path,
        default=None,
        help="Frozen source_scales.json for every Dataset B clip; Dataset A keeps default scale",
    )
    parser.add_argument(
        "--all-behaviors",
        action="store_true",
        help="Enable posture rules for every clip; score Grooming/Rearing/Huddling primary by exact-ID >=50%% frame coverage",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel ProcessPoolExecutor workers (<=1 runs sequentially)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args(argv)
    report = evaluate_and_reconcile_datasets(
        dataset_a_root=args.dataset_a,
        dataset_b_root=args.dataset_b,
        config_path=args.config,
        workers=args.workers,
        all_behaviors=args.all_behaviors,
        dataset_b_source_scales=args.dataset_b_source_scales,
    )
    written = write_reconciled_outputs(report, output_json=args.output)
    print(build_summary_markdown(report))
    print(f"Elapsed: {report['elapsed_seconds']}s (workers={report['workers']})")
    for kind, path in written.items():
        print(f"  wrote {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
