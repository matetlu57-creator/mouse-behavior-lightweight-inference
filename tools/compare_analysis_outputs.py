#!/usr/bin/env python3
"""Compare two lightweight-analysis output directories as a release gate.

Every generated file is compared. Runtime-only metadata fields and absolute
paths rooted in the two output directories are canonicalized before comparing
JSON; all other files use streaming SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mouse-behavior-output-equivalence/v1"
REQUIRED_FILES = (
    Path("lightweight_behavior_events.csv"),
    Path("lightweight_contact_events.csv"),
    Path("lightweight_pair_summary.csv"),
    Path("lightweight_top_evidence.csv"),
    Path("lightweight_analysis_metadata.json"),
    Path("annotation_website_export_report.json"),
)
CANONICAL_JSON_FILES = {
    Path("lightweight_analysis_metadata.json"),
    Path("annotation_website_export_report.json"),
}
VOLATILE_JSON_KEYS = frozenset({"elapsed_s", "stage_timings_s"})
PATH_IDENTITY_KEYS = frozenset({"config"})


@dataclass(frozen=True)
class Difference:
    path: str
    kind: str
    detail: str


@dataclass(frozen=True)
class ComparisonReport:
    schema_version: str
    baseline: str
    optimized: str
    compared_files: int
    equivalent: bool
    differences: tuple[Difference, ...]
    baseline_signatures: Mapping[str, str]
    optimized_signatures: Mapping[str, str]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_output_root(value: str, output_root: Path) -> str:
    normalized = value
    variants = {
        str(output_root),
        output_root.as_posix(),
        str(output_root).replace("\\", "/"),
        str(output_root).replace("/", "\\"),
    }
    for variant in sorted(variants, key=len, reverse=True):
        if variant:
            normalized = normalized.replace(variant, "<OUTPUT_ROOT>")
    return normalized


def _canonicalize_json(value: Any, output_root: Path, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            str(child_key): _canonicalize_json(
                child_value,
                output_root,
                key=str(child_key),
            )
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0]))
            if str(child_key) not in VOLATILE_JSON_KEYS
        }
    if isinstance(value, list):
        return [_canonicalize_json(item, output_root, key=key) for item in value]
    if isinstance(value, str):
        if key in PATH_IDENTITY_KEYS:
            return Path(value).name
        return _replace_output_root(value, output_root)
    return value


def _canonical_json_sha256(path: Path, output_root: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    canonical = _canonicalize_json(payload, output_root)
    if path.name == "lightweight_analysis_metadata.json" and isinstance(
        canonical, dict
    ):
        fsm = canonical.get("parallel_behavior_fsm")
        if isinstance(fsm, dict) and "execution_semantics" not in fsm:
            # v1.43 metadata predating the explicit field still has an
            # unambiguous meaning. Infer it so adding provenance does not look
            # like a scientific-output change.
            fsm["execution_semantics"] = (
                "active_temporal_regions"
                if bool(fsm.get("enabled", True))
                else "disabled_no_parallel_events"
            )
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        # The analyzer historically permits JSON NaN for unavailable numeric
        # diagnostics; retain that compatibility while hashing canonically.
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _discover_files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file() and "写入中" not in path.name
    }


def build_signatures(root: Path) -> dict[str, str]:
    """Return stable signatures for every file produced under ``root``."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    signatures: dict[str, str] = {}
    for relative, path in sorted(
        _discover_files(resolved).items(), key=lambda item: item[0].as_posix()
    ):
        digest = (
            _canonical_json_sha256(path, resolved)
            if relative in CANONICAL_JSON_FILES
            else _sha256(path)
        )
        signatures[relative.as_posix()] = digest
    return signatures


def compare_output_directories(
    baseline: Path,
    optimized: Path,
) -> ComparisonReport:
    """Compare complete output trees and return a machine-readable report."""

    baseline_root = baseline.expanduser().resolve()
    optimized_root = optimized.expanduser().resolve()
    baseline_signatures = build_signatures(baseline_root)
    optimized_signatures = build_signatures(optimized_root)
    differences: list[Difference] = []

    for required in REQUIRED_FILES:
        name = required.as_posix()
        if name not in baseline_signatures and name not in optimized_signatures:
            differences.append(Difference(name, "missing", "both runs missing required file"))

    all_paths = sorted(set(baseline_signatures) | set(optimized_signatures))
    for relative in all_paths:
        left = baseline_signatures.get(relative)
        right = optimized_signatures.get(relative)
        if left is None:
            differences.append(Difference(relative, "missing", "baseline missing file"))
        elif right is None:
            differences.append(Difference(relative, "missing", "optimized missing file"))
        elif left != right:
            differences.append(Difference(relative, "content", f"{left} != {right}"))

    return ComparisonReport(
        schema_version=SCHEMA_VERSION,
        baseline=str(baseline_root),
        optimized=str(optimized_root),
        compared_files=len(set(baseline_signatures) & set(optimized_signatures)),
        equivalent=not differences,
        differences=tuple(differences),
        baseline_signatures=baseline_signatures,
        optimized_signatures=optimized_signatures,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较两次轻量行为分析的完整输出；任何缺失或内容差异均返回失败。"
    )
    parser.add_argument("baseline", type=Path, help="基线分析结果目录")
    parser.add_argument("optimized", type=Path, help="待验证分析结果目录")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="可选：将完整签名和差异写入 JSON 报告（建议放在两个结果目录之外）。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = compare_output_directories(args.baseline, args.optimized)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for difference in report.differences:
        print(f"FAIL  {difference.path} | {difference.kind}: {difference.detail}")
    if report.equivalent:
        print(f"PASS  complete output equivalence | files={report.compared_files}")
    else:
        print(
            f"SUMMARY compared={report.compared_files} "
            f"differences={len(report.differences)}"
        )

    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"REPORT {report_path}")
    return 0 if report.equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
