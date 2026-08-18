"""Load YAML configurations with explicit, repository-local inheritance."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings while letting the overlay own scalar/list values."""

    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML mapping and resolve its optional ``extends`` chain.

    ``extends`` may be one path or a list of paths.  Paths are resolved
    relative to the file that declares them, making profiles portable inside
    ``configs/`` and keeping the legacy root YAML usable as a base file.
    """

    config_path = Path(path).expanduser().resolve()
    if config_path in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, config_path))
        raise ValueError(f"配置继承存在循环: {chain}")
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"配置必须是 YAML mapping: {config_path}")

    raw = dict(raw)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    parents = [extends] if isinstance(extends, (str, Path)) else list(extends)
    if not parents or not all(isinstance(item, (str, Path)) for item in parents):
        raise ValueError(f"extends 必须是路径字符串或路径列表: {config_path}")

    merged: dict[str, Any] = {}
    next_seen = (*_seen, config_path)
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = _deep_merge(merged, load_config(parent_path, _seen=next_seen))
    return _deep_merge(merged, raw)
