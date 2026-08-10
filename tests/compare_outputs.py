#!/usr/bin/env python3
"""Cell-exact, streaming comparison of scientific CSV outputs from two runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path
from typing import Iterator, Sequence


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.reader(handle)


def compare_csv(left: Path, right: Path) -> tuple[bool, str]:
    if left.stat().st_size == right.stat().st_size and sha256(left) == sha256(right):
        return True, "binary-identical"

    sentinel = object()
    for row_number, (left_row, right_row) in enumerate(
        itertools.zip_longest(csv_rows(left), csv_rows(right), fillvalue=sentinel),
        start=1,
    ):
        if left_row is sentinel:
            return False, f"baseline ended before row {row_number}"
        if right_row is sentinel:
            return False, f"optimized ended before row {row_number}"
        assert isinstance(left_row, list) and isinstance(right_row, list)
        if left_row == right_row:
            continue
        max_columns = max(len(left_row), len(right_row))
        for column in range(max_columns):
            left_value = left_row[column] if column < len(left_row) else "<MISSING>"
            right_value = right_row[column] if column < len(right_row) else "<MISSING>"
            if left_value != right_value:
                return (
                    False,
                    f"row {row_number}, column {column + 1}: "
                    f"{left_value!r} != {right_value!r}",
                )
    return True, "cell-identical (encoding/newline bytes differ only)"


def discover_csvs(root: Path) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for path in root.rglob("*.csv"):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if "写入中" in path.name:
            continue
        result[relative] = path
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐单元格比较基线与优化版输出目录中的全部CSV；大文件采用流式读取。"
    )
    parser.add_argument("baseline", type=Path, help="原版运行结果目录")
    parser.add_argument("optimized", type=Path, help="优化版运行结果目录")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="只比较两侧共同存在的CSV；默认任一侧缺文件即失败",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = args.baseline.expanduser().resolve()
    optimized = args.optimized.expanduser().resolve()
    if not baseline.is_dir() or not optimized.is_dir():
        print("ERROR: 两个参数都必须是已存在的结果目录。", file=sys.stderr)
        return 2

    left_files = discover_csvs(baseline)
    right_files = discover_csvs(optimized)
    all_relative = sorted(set(left_files) | set(right_files), key=lambda path: str(path))
    if not all_relative:
        print("ERROR: 两个目录中均未发现CSV。", file=sys.stderr)
        return 2

    failures = 0
    compared = 0
    for relative in all_relative:
        left = left_files.get(relative)
        right = right_files.get(relative)
        if left is None or right is None:
            message = "baseline missing" if left is None else "optimized missing"
            if args.allow_missing:
                print(f"SKIP  {relative} | {message}")
                continue
            print(f"FAIL  {relative} | {message}")
            failures += 1
            continue
        passed, detail = compare_csv(left, right)
        compared += 1
        print(f"{'PASS' if passed else 'FAIL'}  {relative} | {detail}")
        failures += int(not passed)

    print(f"SUMMARY compared={compared} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
