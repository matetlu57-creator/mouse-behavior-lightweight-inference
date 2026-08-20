#!/usr/bin/env python3
"""Run the repository's canonical quality-gate profile."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".quality-gate.toml"


def _load_config() -> Mapping[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    if int(config.get("version", 0)) != 1:
        raise ValueError("unsupported .quality-gate.toml version")
    return config


def _command_argv(raw: Sequence[Any]) -> list[str]:
    if not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError("quality commands must be non-empty arrays of strings")
    return [str(item).replace("{python}", sys.executable) for item in raw]


def _selected_steps(
    config: Mapping[str, Any],
    *,
    profile: str,
    explicit_steps: Sequence[str],
) -> list[str]:
    commands = config.get("commands", {})
    profiles = config.get("profiles", {})
    if not isinstance(commands, Mapping) or not isinstance(profiles, Mapping):
        raise ValueError("quality gate requires commands and profiles tables")
    if explicit_steps:
        selected = list(explicit_steps)
    else:
        raw_profile = profiles.get(profile)
        if not isinstance(raw_profile, list):
            raise ValueError(f"unknown quality profile: {profile}")
        selected = [str(item) for item in raw_profile]
    unknown = [name for name in selected if name not in commands]
    if unknown:
        raise ValueError(f"unknown quality steps: {', '.join(unknown)}")
    return selected


def run(profile: str, explicit_steps: Sequence[str]) -> int:
    config = _load_config()
    commands = config["commands"]
    selected = _selected_steps(
        config,
        profile=profile,
        explicit_steps=explicit_steps,
    )
    print(f"quality profile={profile} steps={','.join(selected)}", flush=True)
    for step in selected:
        raw_commands = commands[step]
        if not isinstance(raw_commands, list) or not raw_commands:
            raise ValueError(f"quality step has no commands: {step}")
        for raw_command in raw_commands:
            argv = _command_argv(raw_command)
            print(f"\n[{step}] {' '.join(argv)}", flush=True)
            completed = subprocess.run(argv, cwd=ROOT, check=False)
            if completed.returncode != 0:
                print(
                    f"quality gate failed: step={step} exit={completed.returncode}",
                    file=sys.stderr,
                )
                return int(completed.returncode)
    print("\nquality gate: PASS", flush=True)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run the CI profile from .quality-gate.toml.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Named profile; defaults to local, or ci when --ci is present.",
    )
    parser.add_argument(
        "--step",
        action="append",
        default=[],
        help="Run one named step; repeat to run multiple steps in order.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = str(args.profile or ("ci" if args.ci else "local"))
    try:
        return run(profile, args.step)
    except (OSError, TypeError, ValueError) as exc:
        print(f"quality gate configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
