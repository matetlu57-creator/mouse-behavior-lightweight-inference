"""Gateway-independent persistent local Site Hub for review and error sites.

Runs as a detached loopback HTTP supervisor (``http://127.0.0.1:8788/`` by default)
so review and error-display sites survive agent gateway disconnects/restarts and
can auto-start on Windows logon.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from mouse_behavior.evaluation.review_transaction import safe_path, write_json

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HUB_HOST = "127.0.0.1"
DEFAULT_HUB_PORT = 8788
DEFAULT_HUB_STATE_DIR = _REPO_ROOT / "outputs" / "site_hub"
REGISTRY_SCHEMA = "mouse_behavior_site_hub_v1"
HEALTH_SCHEMA = "site_hub_health_v1"
STATUS_SCHEMA = "site_hub_status_v1"
VALID_CATEGORIES = frozenset({"review", "error_display", "comparison", "other"})
ASSETS_DIR = Path(__file__).with_name("site_hub_assets")
AUTOSTART_TASK_NAME = "MouseBehaviorSiteHub"

_ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_SITE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_POST_BYTES = 256 * 1024
_STATIC_MIME_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}
_LOCAL_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_PREFERRED_ANACONDA_PYTHON = Path(r"D:\Anaconda3\python.exe")
_CLEANED_DATASET_DEFAULT_DIR = Path(r"F:\mouse_behavior_cleaned_dataset_20261121")
LOGGER = logging.getLogger(__name__)


class CorruptRegistryError(RuntimeError):
    """Raised when ``sites.json`` exists on disk but cannot be parsed or validated."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _format_http_url(host: str, port: int, path: str = "/") -> str:
    host_part = f"[{host}]" if ":" in host else host
    norm_path = path if path.startswith("/") else f"/{path}"
    return f"http://{host_part}:{port}{norm_path}"


def resolve_child_python() -> str:
    """Return the console Python interpreter used for supervised child servers."""
    if _PREFERRED_ANACONDA_PYTHON.is_file():
        return str(_PREFERRED_ANACONDA_PYTHON)
    exe = Path(sys.executable)
    if sys.platform == "win32" and exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.is_file():
            return str(sibling)
    return str(exe)


def resolve_detached_python() -> str:
    """Return windowless ``pythonw.exe`` on Windows when available for the daemon."""
    child_py = Path(resolve_child_python())
    if sys.platform == "win32" and child_py.name.lower() == "python.exe":
        windowless = child_py.with_name("pythonw.exe")
        if windowless.is_file():
            return str(windowless)
    return str(child_py)


def validate_site_spec(
    raw: Mapping[str, Any],
    *,
    hub_port: int = DEFAULT_HUB_PORT,
) -> dict[str, Any]:
    """Validate and normalize a registered site specification."""
    if not isinstance(raw, Mapping):
        raise ValueError("site spec must be a JSON object")

    site_id = raw.get("site_id")
    if not isinstance(site_id, str) or not _SITE_ID_RE.fullmatch(site_id):
        raise ValueError(f"invalid site_id {site_id!r}; expected ^[a-z0-9][a-z0-9_-]{{0,63}}$")

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"site {site_id!r} requires a non-empty string title")

    category = raw.get("category", "other")
    if not isinstance(category, str) or category not in VALID_CATEGORIES:
        raise ValueError(f"site {site_id!r} has invalid category {category!r}")

    host = raw.get("host", DEFAULT_HUB_HOST)
    if not isinstance(host, str) or host.strip().lower() not in _ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(f"site {site_id!r} host must be loopback, got {host!r}")
    host_norm = host.strip().lower()

    port = raw.get("port")
    if not _is_strict_int(port) or not (1024 <= port <= 65535):
        raise ValueError(f"site {site_id!r} port must be an integer in 1024..65535, got {port!r}")
    if port == hub_port:
        raise ValueError(f"site {site_id!r} port {port} collides with hub port {hub_port}")

    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and bool(part.strip()) for part in command)
    ):
        raise ValueError(f"site {site_id!r} command must be a non-empty list of non-empty strings")

    health_path = raw.get("health_path", "/api/health")
    if not isinstance(health_path, str) or not health_path.startswith("/"):
        raise ValueError(f"site {site_id!r} health_path must start with '/'")

    cwd = raw.get("cwd", str(_REPO_ROOT))
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError(f"site {site_id!r} cwd must be a non-empty string")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"site {site_id!r} enabled must be a boolean")

    auto_restart = raw.get("auto_restart", True)
    if not isinstance(auto_restart, bool):
        raise ValueError(f"site {site_id!r} auto_restart must be a boolean")

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"site {site_id!r} description must be a string")

    data_summary = raw.get("data_summary", "")
    if not isinstance(data_summary, str):
        raise ValueError(f"site {site_id!r} data_summary must be a string")

    return {
        "site_id": site_id,
        "title": title.strip(),
        "category": category,
        "host": host_norm,
        "port": int(port),
        "health_path": health_path,
        "command": [str(part) for part in command],
        "cwd": cwd,
        "enabled": enabled,
        "auto_restart": auto_restart,
        "description": description.strip(),
        "data_summary": data_summary.strip(),
    }


def build_default_sites(
    repo_root: Path = _REPO_ROOT,
    python_exe: str | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical default site specs for review and error-display sites."""
    root = Path(repo_root).resolve()
    if python_exe is not None:
        chosen = Path(python_exe)
        if sys.platform == "win32" and chosen.name.lower() == "pythonw.exe":
            sibling = chosen.with_name("python.exe")
            py = str(sibling if sibling.is_file() else chosen)
        else:
            py = str(chosen)
    else:
        py = resolve_child_python()

    raw_specs = [
        {
            "site_id": "mouse_clip_review",
            "title": "小鼠行为审核与手动采集工作台",
            "category": "review",
            "host": DEFAULT_HUB_HOST,
            "port": 8795,
            "health_path": "/api/health",
            "cwd": str(root),
            "command": [
                py,
                str(root / "scripts" / "serve_mouse_clip_review_site.py"),
                "--source-dir",
                str(_CLEANED_DATASET_DEFAULT_DIR),
                "--output-dir",
                str(root / "outputs" / "mouse_clip_review_site_cleaned_20261121"),
                "--host",
                DEFAULT_HUB_HOST,
                "--port",
                "8795",
            ],
            "enabled": True,
            "auto_restart": True,
            "description": "逐帧核查已清洗数据集视频，支持通过/拒绝备注与手动行为片段裁剪采集。",
            "data_summary": r"已清洗数据集 F:\mouse_behavior_cleaned_dataset_20261121（764 段 / 14 类）",
        },
        {
            "site_id": "long_video_group_review",
            "title": "长视频群体行为精裁人工审核站",
            "category": "review",
            "host": DEFAULT_HUB_HOST,
            "port": 8790,
            "health_path": "/api/health",
            "cwd": str(root),
            "command": [
                py,
                str(root / "scripts" / "serve_long_video_group_review_site.py"),
                "--site-dir",
                str(root / "outputs" / "long_video_group_review_site"),
                "--host",
                DEFAULT_HUB_HOST,
                "--port",
                "8790",
            ],
            "enabled": True,
            "auto_restart": True,
            "description": "多长视频群体行为边界精裁前后对比、原始/高亮回放与人工通过率复核。",
            "data_summary": "video22 (10min) + 大旷场1 (60min) 群体行为裁剪复核",
        },
        {
            "site_id": "behavior_error_review",
            "title": "全 15 类行为评测错误复核站",
            "category": "error_display",
            "host": DEFAULT_HUB_HOST,
            "port": 8776,
            "health_path": "/api/health",
            "cwd": str(root),
            "command": [
                py,
                str(root / "scripts" / "serve_behavior_errors.py"),
                "--report",
                str(root / "outputs" / "all15_latest_20261116_codex" / "report.json"),
                "--diagnostics",
                str(root / "outputs" / "behavior_error_review_20261116" / "bound_huddle_diagnostics.json"),
                "--port",
                "8776",
            ],
            "enabled": True,
            "auto_restart": True,
            "description": "全 15 类专项评测错误原因归因、指标面板与目标 ID 逐帧视频高亮展示。",
            "data_summary": "A(429) + B(605) 全 15 类评测错误原因与逐帧高亮",
        },
    ]
    return [validate_site_spec(spec) for spec in raw_specs]


def probe_http_health(
    host: str,
    port: int,
    health_path: str = "/api/health",
    timeout_s: float = 1.5,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Probe a loopback JSON health endpoint, bypassing OS HTTP proxies."""
    url = _format_http_url(host, port, health_path)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with _LOCAL_HTTP_OPENER.open(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", resp.getcode())
            if status != 200:
                return False, None, f"HTTP {status}"
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" not in ctype:
                return False, None, f"unexpected content-type: {ctype or 'missing'}"
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
    except urllib.error.HTTPError as exc:
        return False, None, f"HTTP {exc.code}"
    except Exception as exc:
        return False, None, str(exc)

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return False, None, "health payload missing status='ok'"
    return True, payload, None


def _parse_loopback_host(host_header: str | None, expected_port: int) -> bool:
    """Validate loopback ``Host`` header including bracketed IPv6 ``[::1]:port``."""
    if not host_header:
        return False
    raw = host_header.strip().lower()
    if not raw:
        return False
    if raw.startswith("["):
        end_bracket = raw.find("]")
        if end_bracket == -1:
            return False
        host_part = raw[1:end_bracket]
        rest = raw[end_bracket + 1 :]
        if rest:
            if not rest.startswith(":"):
                return False
            port_part: str | None = rest[1:]
        else:
            port_part = None
    else:
        if raw.count(":") > 1:
            return False
        if ":" in raw:
            host_part, port_part = raw.split(":", 1)
        else:
            host_part, port_part = raw, None

    if host_part not in _ALLOWED_LOOPBACK_HOSTS:
        return False
    if port_part is not None:
        if not port_part.isdigit():
            return False
        if expected_port and int(port_part) != expected_port:
            return False
    return True


def _parse_loopback_origin(origin_header: str | None, expected_port: int) -> bool:
    """Allow absent Origin or same-loopback HTTP Origin."""
    if not origin_header:
        return True
    try:
        parsed = urlsplit(origin_header.strip())
    except Exception:
        return False
    if parsed.scheme != "http":
        return False
    if (parsed.hostname or "").lower() not in _ALLOWED_LOOPBACK_HOSTS:
        return False
    try:
        origin_port = parsed.port
    except ValueError:
        return False
    if origin_port is not None and expected_port and origin_port != expected_port:
        return False
    return True


class SiteRegistryStore:
    """Atomic JSON store for ``<state_dir>/sites.json`` preserving corrupt files."""

    def __init__(
        self,
        state_dir: Path = DEFAULT_HUB_STATE_DIR,
        *,
        hub_port: int = DEFAULT_HUB_PORT,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.hub_port = int(hub_port)
        self.registry_path = self.state_dir / "sites.json"
        self._lock = threading.RLock()

    def load(self) -> list[dict[str, Any]]:
        """Read and validate ``sites.json``; raise ``CorruptRegistryError`` on corruption."""
        with self._lock:
            if not self.registry_path.exists():
                return []
            try:
                raw_bytes = self.registry_path.read_bytes()
                doc = json.loads(raw_bytes.decode("utf-8-sig"))
            except Exception as exc:
                raise CorruptRegistryError(
                    f"sites registry is unreadable or malformed JSON ({self.registry_path}): {exc}"
                ) from exc

            if not isinstance(doc, Mapping) or doc.get("schema_version") != REGISTRY_SCHEMA:
                raise CorruptRegistryError(
                    f"sites registry schema mismatch in {self.registry_path}"
                )
            raw_sites = doc.get("sites")
            if not isinstance(raw_sites, list):
                raise CorruptRegistryError(
                    f"sites registry 'sites' field must be a list in {self.registry_path}"
                )

            seen_ids: set[str] = set()
            seen_ports: dict[int, str] = {}
            validated: list[dict[str, Any]] = []
            for item in raw_sites:
                try:
                    spec = validate_site_spec(item, hub_port=self.hub_port)
                except ValueError as exc:
                    raise CorruptRegistryError(str(exc)) from exc
                sid = spec["site_id"]
                port = spec["port"]
                if sid in seen_ids:
                    raise CorruptRegistryError(f"duplicate site_id {sid!r} in registry")
                if port in seen_ports:
                    raise CorruptRegistryError(
                        f"duplicate port {port} shared by {seen_ports[port]!r} and {sid!r}"
                    )
                seen_ids.add(sid)
                seen_ports[port] = sid
                validated.append(spec)
            return validated

    def list_sites(self) -> list[dict[str, Any]]:
        return self.load()

    def _save(self, sites: Sequence[dict[str, Any]]) -> None:
        seen_ids: set[str] = set()
        seen_ports: dict[int, str] = {}
        normalized: list[dict[str, Any]] = []
        for item in sites:
            spec = validate_site_spec(item, hub_port=self.hub_port)
            sid = spec["site_id"]
            port = spec["port"]
            if sid in seen_ids:
                raise ValueError(f"duplicate site_id: {sid!r}")
            if port in seen_ports:
                raise ValueError(f"port {port} already assigned to {seen_ports[port]!r}")
            seen_ids.add(sid)
            seen_ports[port] = sid
            normalized.append(spec)

        payload = {
            "schema_version": REGISTRY_SCHEMA,
            "updated_at": _utc_now_iso(),
            "sites": normalized,
        }
        write_json(self.registry_path, payload)

    def upsert_site(self, raw_spec: Mapping[str, Any]) -> dict[str, Any]:
        """Insert or update a validated site specification atomically."""
        with self._lock:
            spec = validate_site_spec(raw_spec, hub_port=self.hub_port)
            sites = self.load()
            for idx, existing in enumerate(sites):
                if existing["site_id"] != spec["site_id"] and existing["port"] == spec["port"]:
                    raise ValueError(
                        f"port {spec['port']} is already registered to site {existing['site_id']!r}"
                    )
                if existing["site_id"] == spec["site_id"]:
                    sites[idx] = spec
                    self._save(sites)
                    return spec
            sites.append(spec)
            self._save(sites)
            return spec

    def remove_site(self, site_id: str) -> bool:
        """Remove a site specification by ID. Returns True if removed."""
        with self._lock:
            sites = self.load()
            kept = [item for item in sites if item["site_id"] != site_id]
            if len(kept) == len(sites):
                return False
            self._save(kept)
            return True

    def ensure_default_sites(
        self,
        default_sites: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Idempotently add missing canonical sites without overwriting user edits."""
        with self._lock:
            existing = self.load()
            by_id = {item["site_id"]: item for item in existing}
            used_ports = {item["port"]: item["site_id"] for item in existing}
            defaults = list(default_sites) if default_sites is not None else build_default_sites()
            changed = False
            for raw in defaults:
                spec = validate_site_spec(raw, hub_port=self.hub_port)
                if spec["site_id"] in by_id:
                    continue
                if spec["port"] in used_ports:
                    continue
                existing.append(spec)
                by_id[spec["site_id"]] = spec
                used_ports[spec["port"]] = spec["site_id"]
                changed = True
            if changed or not self.registry_path.exists():
                self._save(existing)
            return existing


@dataclass
class _ManagedSiteState:
    spec: dict[str, Any]
    log_path: Path
    process: subprocess.Popen[Any] | None = None
    pid: int | None = None
    status: str = "stopped"
    healthy: bool = False
    last_health: dict[str, Any] | None = None
    last_error: str | None = None
    started_at: str | None = None
    started_monotonic: float | None = None
    restart_timestamps: deque[float] = field(default_factory=deque)
    restart_count: int = 0
    last_exit_code: int | None = None
    manual_stopped: bool = False


def _command_signature(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        spec["host"],
        int(spec["port"]),
        spec["cwd"],
        tuple(spec["command"]),
        spec["health_path"],
    )


class SiteSupervisor:
    """Manages registered site child processes, health probes, and auto-restarts."""

    def __init__(
        self,
        registry: SiteRegistryStore,
        *,
        repo_root: Path = _REPO_ROOT,
        poll_interval_s: float = 3.0,
        max_rapid_restarts: int = 5,
        restart_window_s: float = 60.0,
        health_timeout_s: float = 1.5,
    ) -> None:
        self.registry = registry
        self.repo_root = Path(repo_root).resolve()
        self.logs_dir = self.registry.state_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_s = float(poll_interval_s)
        self.max_rapid_restarts = int(max_rapid_restarts)
        self.restart_window_s = float(restart_window_s)
        self.health_timeout_s = float(health_timeout_s)

        self._lock = threading.RLock()
        self._states: dict[str, _ManagedSiteState] = {}
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._corrupt_error: str | None = None

    def _build_child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        src_dir = str(self.repo_root / "src")
        repo_str = str(self.repo_root)
        existing_pp = env.get("PYTHONPATH", "")
        parts = [p for p in existing_pp.split(os.pathsep) if p]
        for entry in (repo_str, src_dir):
            if entry not in parts:
                parts.insert(0, entry)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _ensure_state(self, spec: dict[str, Any]) -> _ManagedSiteState:
        sid = spec["site_id"]
        log_path = safe_path(self.logs_dir / f"{sid}.log", [self.logs_dir])
        existing = self._states.get(sid)
        if existing is None:
            status = "stopped" if spec["enabled"] else "disabled"
            state = _ManagedSiteState(spec=spec, log_path=log_path, status=status)
            self._states[sid] = state
            return state
        existing.spec = spec
        if not spec["enabled"] and existing.process is None:
            existing.status = "disabled"
        return existing

    def _sync_registry_locked(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Sync in-memory state table with ``sites.json`` under lock.

        Returns ``(specs, sites_to_stop)``.
        """
        specs = self.registry.load()
        self._corrupt_error = None
        desired = {spec["site_id"]: spec for spec in specs}
        to_stop: list[str] = []

        for sid in list(self._states):
            state = self._states[sid]
            if sid not in desired:
                to_stop.append(sid)
                continue
            new_spec = desired[sid]
            if _command_signature(state.spec) != _command_signature(new_spec) or (
                not new_spec["enabled"] and state.process is not None
            ):
                to_stop.append(sid)
            state.spec = new_spec
            if not new_spec["enabled"] and state.process is None:
                state.status = "disabled"
        for spec in specs:
            self._ensure_state(spec)
        return specs, to_stop

    def _terminate_process(self, proc: subprocess.Popen[Any]) -> int | None:
        if proc.poll() is not None:
            return proc.returncode
        try:
            proc.terminate()
            return proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                return proc.wait(timeout=2.0)
            except Exception:
                return proc.poll()
        except Exception:
            return proc.poll()

    def _get_or_load_state(self, site_id: str) -> _ManagedSiteState:
        specs = self.registry.load()
        self._corrupt_error = None
        for spec in specs:
            if spec["site_id"] == site_id:
                return self._ensure_state(spec)
        raise KeyError(f"unknown site_id: {site_id!r}")

    def start_site(self, site_id: str, *, manual: bool = True) -> dict[str, Any]:
        """Start or attach to an already-healthy site process."""
        with self._lock:
            state = self._get_or_load_state(site_id)
            spec = state.spec
            if not spec["enabled"]:
                state.status = "disabled"
                return self._format_site_row(state)
            if manual:
                state.manual_stopped = False
                state.restart_timestamps.clear()

            if state.process is not None and state.process.poll() is None:
                spec_copy = dict(spec)
            else:
                state.process = None
                state.pid = None
                spec_copy = dict(spec)

        # Probe outside the lock so socket I/O cannot block API readers.
        ok, payload, _ = probe_http_health(
            spec_copy["host"],
            spec_copy["port"],
            spec_copy["health_path"],
            timeout_s=min(1.0, self.health_timeout_s),
        )

        proc_to_spawn: subprocess.Popen[Any] | None = None
        spawn_error: str | None = None
        with self._lock:
            state = self._get_or_load_state(site_id)
            if state.process is not None and state.process.poll() is None:
                state.healthy = ok
                state.last_health = payload
                state.status = "running" if ok else "starting"
                return self._format_site_row(state)

            if ok:
                state.process = None
                state.pid = None
                state.healthy = True
                state.last_health = payload
                state.last_error = None
                state.status = "external_running"
                if state.started_at is None:
                    state.started_at = _utc_now_iso()
                return self._format_site_row(state)

            cwd_path = Path(state.spec["cwd"])
            if not cwd_path.is_dir():
                state.status = "error"
                state.healthy = False
                state.last_error = f"working directory does not exist: {cwd_path}"
                return self._format_site_row(state)

            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_handle = state.log_path.open("ab")
            try:
                banner = (
                    f"\n[{_utc_now_iso()}] starting {site_id}: "
                    f"{' '.join(state.spec['command'])}\n"
                ).encode("utf-8", errors="replace")
                log_handle.write(banner)
                log_handle.flush()
                popen_kwargs: dict[str, Any] = {
                    "cwd": str(cwd_path),
                    "env": self._build_child_env(),
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                }
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                proc_to_spawn = subprocess.Popen(state.spec["command"], **popen_kwargs)
            except Exception as exc:
                spawn_error = str(exc)
            finally:
                log_handle.close()

            if spawn_error is not None or proc_to_spawn is None:
                state.process = None
                state.pid = None
                state.healthy = False
                state.status = "error"
                state.last_error = f"failed to spawn process: {spawn_error}"
                return self._format_site_row(state)

            state.process = proc_to_spawn
            state.pid = proc_to_spawn.pid
            state.status = "starting"
            state.healthy = False
            state.last_health = None
            state.last_error = None
            state.last_exit_code = None
            state.started_at = _utc_now_iso()
            state.started_monotonic = time.monotonic()

        # Brief non-blocking readiness check (up to ~0.9s) so fast servers report running promptly.
        deadline = time.monotonic() + 0.9
        while time.monotonic() < deadline:
            if proc_to_spawn.poll() is not None:
                break
            ok_now, payload_now, err_now = probe_http_health(
                spec_copy["host"],
                spec_copy["port"],
                spec_copy["health_path"],
                timeout_s=0.4,
            )
            if ok_now:
                with self._lock:
                    state = self._states.get(site_id)
                    if state and state.process is proc_to_spawn:
                        state.healthy = True
                        state.last_health = payload_now
                        state.last_error = None
                        state.status = "running"
                        return self._format_site_row(state)
            time.sleep(0.1)

        with self._lock:
            state = self._states.get(site_id)
            if state is None:
                raise KeyError(site_id)
            if state.process is proc_to_spawn:
                rc = proc_to_spawn.poll()
                if rc is not None:
                    state.process = None
                    state.pid = None
                    state.healthy = False
                    state.last_exit_code = rc
                    state.status = "error"
                    state.last_error = f"process exited immediately with code {rc}"
            return self._format_site_row(state)

    def stop_site(
        self,
        site_id: str,
        *,
        remove_from_map: bool = False,
        set_manual_stopped: bool = True,
    ) -> dict[str, Any]:
        """Stop a managed child site process cleanly."""
        proc_to_kill: subprocess.Popen[Any] | None = None
        with self._lock:
            state = self._states.get(site_id)
            if state is None:
                state = self._get_or_load_state(site_id)
            if set_manual_stopped:
                state.manual_stopped = True
            proc_to_kill = state.process
            state.process = None
            state.pid = None
            state.healthy = False
            state.last_health = None

        rc = self._terminate_process(proc_to_kill) if proc_to_kill is not None else None

        with self._lock:
            if remove_from_map:
                removed = self._states.pop(site_id, None)
                return self._format_site_row(removed) if removed else {"site_id": site_id, "status": "removed"}
            state = self._states[site_id]
            if rc is not None:
                state.last_exit_code = rc
            state.status = "stopped" if state.spec["enabled"] else "disabled"
            state.last_error = None
            return self._format_site_row(state)

    def restart_site(self, site_id: str) -> dict[str, Any]:
        self.stop_site(site_id)
        return self.start_site(site_id, manual=True)

    def start_all_enabled(self) -> list[dict[str, Any]]:
        """Load registry and start every enabled site."""
        with self._lock:
            specs, to_stop = self._sync_registry_locked()
        for sid in to_stop:
            self.stop_site(sid, set_manual_stopped=False)
        for spec in specs:
            if spec["enabled"]:
                self.start_site(spec["site_id"], manual=True)
        return self.snapshot()["sites"]

    def reload_registry(self) -> dict[str, Any]:
        """Reload ``sites.json``, stop removed/changed sites, and start enabled sites."""
        with self._lock:
            specs, to_stop = self._sync_registry_locked()
            desired_ids = {spec["site_id"] for spec in specs}
        for sid in to_stop:
            self.stop_site(sid, remove_from_map=(sid not in desired_ids), set_manual_stopped=False)
        for spec in specs:
            sid = spec["site_id"]
            if not spec["enabled"]:
                continue
            with self._lock:
                state = self._states.get(sid)
                skip = bool(state and state.manual_stopped)
            if not skip:
                self.start_site(sid, manual=False)
        return self.snapshot()

    def stop_all(self) -> None:
        """Stop watchdog thread and all managed child processes."""
        self._stop_event.set()
        thread = self._watchdog_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            site_ids = list(self._states.keys())
        for sid in site_ids:
            self.stop_site(sid)

    def start_watchdog(self) -> None:
        with self._lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            self._stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="site-hub-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_s):
            try:
                self.tick()
            except Exception as exc:
                LOGGER.warning("site hub watchdog tick failed: %s", exc)

    def tick(self) -> None:
        """Run one monitoring and auto-restart pass across registered sites."""
        probe_targets: list[tuple[str, str, int, str]] = []
        to_restart: list[str] = []
        to_stop_ids: list[str] = []
        desired_ids: set[str] | None = None

        with self._lock:
            try:
                specs, to_stop_ids = self._sync_registry_locked()
                desired_ids = {spec["site_id"] for spec in specs}
            except CorruptRegistryError as exc:
                self._corrupt_error = str(exc)

            stop_set = set(to_stop_ids)
            now = time.monotonic()
            for sid, state in self._states.items():
                if sid in stop_set:
                    continue
                spec = state.spec
                proc = state.process
                if proc is not None:
                    rc = proc.poll()
                    if rc is not None:
                        state.process = None
                        state.pid = None
                        state.healthy = False
                        state.last_health = None
                        state.last_exit_code = rc
                        if spec["enabled"] and spec["auto_restart"] and not state.manual_stopped:
                            cutoff = now - self.restart_window_s
                            while state.restart_timestamps and state.restart_timestamps[0] < cutoff:
                                state.restart_timestamps.popleft()
                            if len(state.restart_timestamps) >= self.max_rapid_restarts:
                                state.status = "error"
                                state.last_error = (
                                    f"{int(self.restart_window_s)}s 内连续退出 {len(state.restart_timestamps)} 次"
                                    f"（退出码 {rc}），已暂停自动拉起"
                                )
                            else:
                                state.restart_timestamps.append(now)
                                state.restart_count += 1
                                state.status = "restarting"
                                state.last_error = f"进程异常退出（退出码 {rc}），正在自动拉起"
                                to_restart.append(sid)
                        else:
                            state.status = "stopped" if spec["enabled"] else "disabled"
                    else:
                        probe_targets.append((sid, spec["host"], spec["port"], spec["health_path"]))
                elif state.status == "external_running" or (
                    spec["enabled"] and spec["auto_restart"] and not state.manual_stopped and state.status != "error"
                ):
                    probe_targets.append((sid, spec["host"], spec["port"], spec["health_path"]))

        for sid in to_stop_ids:
            try:
                remove = desired_ids is not None and sid not in desired_ids
                self.stop_site(sid, remove_from_map=remove, set_manual_stopped=False)
                if not remove:
                    state = self._states.get(sid)
                    if state and state.spec["enabled"]:
                        to_restart.append(sid)
            except Exception as exc:
                LOGGER.warning("failed stopping changed/removed site %s: %s", sid, exc)

        probe_results: dict[str, tuple[bool, dict[str, Any] | None, str | None]] = {}
        for sid, host, port, health_path in probe_targets:
            probe_results[sid] = probe_http_health(host, port, health_path, timeout_s=self.health_timeout_s)

        with self._lock:
            now = time.monotonic()
            for sid, (ok, payload, err) in probe_results.items():
                state = self._states.get(sid)
                if state is None:
                    continue
                if state.process is not None and state.process.poll() is None:
                    state.healthy = ok
                    state.last_health = payload
                    if ok:
                        state.status = "running"
                        state.last_error = None
                    else:
                        elapsed = now - (state.started_monotonic or now)
                        if elapsed < 12.0 and state.status == "starting":
                            state.status = "starting"
                        else:
                            state.status = "unhealthy"
                            state.last_error = err or "health check failed"
                elif state.status == "external_running":
                    if ok:
                        state.healthy = True
                        state.last_health = payload
                        state.last_error = None
                    else:
                        state.healthy = False
                        state.last_health = None
                        state.status = "stopped"
                        if state.spec["enabled"] and state.spec["auto_restart"] and not state.manual_stopped:
                            to_restart.append(sid)
                elif ok:
                    state.status = "external_running"
                    state.healthy = True
                    state.last_health = payload
                    state.last_error = None
                elif (
                    state.spec["enabled"]
                    and state.spec["auto_restart"]
                    and not state.manual_stopped
                    and state.status not in {"error", "disabled", "restarting"}
                ):
                    to_restart.append(sid)

        for sid in dict.fromkeys(to_restart):
            try:
                self.start_site(sid, manual=False)
            except Exception as exc:
                LOGGER.warning("failed auto-restart for %s: %s", sid, exc)

    def _format_site_row(self, state: _ManagedSiteState) -> dict[str, Any]:
        spec = state.spec
        return {
            **spec,
            "url": _format_http_url(spec["host"], spec["port"], "/"),
            "health_url": _format_http_url(spec["host"], spec["port"], spec["health_path"]),
            "status": state.status,
            "healthy": bool(state.healthy),
            "pid": state.pid,
            "started_at": state.started_at,
            "restart_count": state.restart_count,
            "last_exit_code": state.last_exit_code,
            "last_error": state.last_error,
            "last_health": state.last_health,
            "managed": state.process is not None,
            "log_path": str(state.log_path),
        }

    def snapshot(self, *, probe_live: bool = False) -> dict[str, Any]:
        """Return a thread-safe status snapshot of all registered sites."""
        specs = self.registry.load()
        with self._lock:
            self._corrupt_error = None
            for spec in specs:
                self._ensure_state(spec)
            known_ids = {spec["site_id"] for spec in specs}
            probe_targets = [
                (spec["site_id"], spec["host"], spec["port"], spec["health_path"])
                for spec in specs
                if probe_live or self._states[spec["site_id"]].process is not None or self._states[spec["site_id"]].status == "external_running"
            ]

        if probe_targets:
            results = {
                sid: probe_http_health(host, port, health_path, timeout_s=min(1.0, self.health_timeout_s))
                for sid, host, port, health_path in probe_targets
            }
            with self._lock:
                for sid, (ok, payload, err) in results.items():
                    state = self._states.get(sid)
                    if state is None:
                        continue
                    if state.process is not None and state.process.poll() is not None:
                        state.last_exit_code = state.process.returncode
                        state.process = None
                        state.pid = None
                    if state.process is not None:
                        state.healthy = ok
                        state.last_health = payload
                        if ok:
                            state.status = "running"
                            state.last_error = None
                    elif ok:
                        state.status = "external_running"
                        state.healthy = True
                        state.last_health = payload
                        state.last_error = None
                    elif state.status == "external_running":
                        state.healthy = False
                        state.last_health = None
                        state.status = "stopped" if state.spec["enabled"] else "disabled"
                    elif err and not state.healthy and state.status == "unhealthy":
                        state.last_error = err

        with self._lock:
            rows = [
                self._format_site_row(self._states[spec["site_id"]])
                for spec in specs
                if spec["site_id"] in known_ids and spec["site_id"] in self._states
            ]
        summary = {
            "total": len(rows),
            "healthy": sum(1 for row in rows if row["healthy"]),
            "review_count": sum(1 for row in rows if row["category"] == "review"),
            "error_display_count": sum(1 for row in rows if row["category"] == "error_display"),
            "unhealthy": sum(1 for row in rows if row["enabled"] and not row["healthy"]),
        }
        return {"sites": rows, "summary": summary}

    def read_site_logs(self, site_id: str, *, max_lines: int = 80) -> dict[str, Any]:
        """Read tail lines from ``<state_dir>/logs/<site_id>.log`` safely."""
        if not _SITE_ID_RE.fullmatch(site_id):
            raise ValueError(f"invalid site_id: {site_id!r}")
        log_path = safe_path(self.logs_dir / f"{site_id}.log", [self.logs_dir])
        if not log_path.is_file():
            return {"site_id": site_id, "log_path": str(log_path), "lines": []}
        limit = max(1, min(int(max_lines), 500))
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[-limit:]
        return {"site_id": site_id, "log_path": str(log_path), "lines": lines}


class ExclusiveHubHTTPServer(ThreadingHTTPServer):
    """Loopback ThreadingHTTPServer enforcing Windows port exclusivity."""

    allow_reuse_address = False
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        registry: SiteRegistryStore,
        supervisor: SiteSupervisor,
        session_token: str,
    ) -> None:
        self.registry = registry
        self.supervisor = supervisor
        self.session_token = session_token
        self.started_at = _utc_now_iso()
        self.started_monotonic = time.monotonic()
        super().__init__(server_address, handler_cls)

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def server_close(self) -> None:
        try:
            self.supervisor.stop_all()
        finally:
            super().server_close()


def _hub_command_args(
    *,
    repo_root: Path,
    state_dir: Path,
    host: str,
    port: int,
    log_file: Path,
    python_exe: str,
) -> list[str]:
    script = Path(repo_root).resolve() / "scripts" / "manage_site_hub.py"
    return [
        python_exe,
        str(script),
        "daemon",
        "--host",
        host,
        "--port",
        str(port),
        "--state-dir",
        str(Path(state_dir).resolve()),
        "--log-file",
        str(Path(log_file).resolve()),
    ]


def get_windows_autostart_status() -> dict[str, Any]:
    """Check whether Windows logon autostart is configured via Task Scheduler or Startup."""
    if sys.platform != "win32":
        return {"supported": False, "installed": False, "method": None}

    startup_vbs = (
        Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / f"{AUTOSTART_TASK_NAME}.vbs"
    )
    res = subprocess.run(
        ["schtasks", "/Query", "/TN", AUTOSTART_TASK_NAME],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if res.returncode == 0:
        return {
            "supported": True,
            "installed": True,
            "method": "schtasks",
            "task_name": AUTOSTART_TASK_NAME,
            "startup_vbs_exists": startup_vbs.is_file(),
        }
    if startup_vbs.is_file():
        return {
            "supported": True,
            "installed": True,
            "method": "startup_vbs",
            "startup_vbs_path": str(startup_vbs),
        }
    return {"supported": True, "installed": False, "method": None}


def install_windows_autostart(
    *,
    repo_root: Path = _REPO_ROOT,
    state_dir: Path = DEFAULT_HUB_STATE_DIR,
    host: str = DEFAULT_HUB_HOST,
    port: int = DEFAULT_HUB_PORT,
) -> dict[str, Any]:
    """Configure Windows user-logon autostart without requiring elevation."""
    if sys.platform != "win32":
        return {"installed": False, "method": "unsupported", "detail": "Windows only"}

    resolved_state = Path(state_dir).resolve()
    resolved_state.mkdir(parents=True, exist_ok=True)
    log_file = resolved_state / "hub.log"
    py = resolve_detached_python()
    cmd_args = _hub_command_args(
        repo_root=repo_root,
        state_dir=resolved_state,
        host=host,
        port=port,
        log_file=log_file,
        python_exe=py,
    )
    cmd_line = subprocess.list2cmdline(cmd_args)

    res = subprocess.run(
        [
            "schtasks",
            "/Create",
            "/F",
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
            "/TN",
            AUTOSTART_TASK_NAME,
            "/TR",
            cmd_line,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if res.returncode == 0:
        return {
            "installed": True,
            "method": "schtasks",
            "task_name": AUTOSTART_TASK_NAME,
            "command": cmd_line,
        }

    startup_dir = (
        Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )
    startup_dir.mkdir(parents=True, exist_ok=True)
    vbs_path = startup_dir / f"{AUTOSTART_TASK_NAME}.vbs"
    escaped = cmd_line.replace('"', '""')
    vbs_body = (
        'Set shell = CreateObject("WScript.Shell")\r\n'
        f'shell.CurrentDirectory = "{str(Path(repo_root).resolve()).replace(chr(34), chr(34) * 2)}"\r\n'
        f'shell.Run "{escaped}", 0, False\r\n'
    )
    vbs_path.write_text(vbs_body, encoding="utf-8")
    return {
        "installed": True,
        "method": "startup_vbs",
        "path": str(vbs_path),
        "schtasks_fallback_reason": (res.stderr or res.stdout).strip(),
        "command": cmd_line,
    }


def uninstall_windows_autostart() -> dict[str, Any]:
    """Remove Windows logon autostart scheduled task and/or Startup script."""
    if sys.platform != "win32":
        return {"removed": False, "method": "unsupported"}
    removed_methods: list[str] = []
    res = subprocess.run(
        ["schtasks", "/Delete", "/F", "/TN", AUTOSTART_TASK_NAME],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if res.returncode == 0:
        removed_methods.append("schtasks")

    vbs_path = (
        Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / f"{AUTOSTART_TASK_NAME}.vbs"
    )
    if vbs_path.is_file():
        vbs_path.unlink()
        removed_methods.append("startup_vbs")

    return {"removed": bool(removed_methods), "methods": removed_methods}


def spawn_detached_hub(
    *,
    repo_root: Path = _REPO_ROOT,
    state_dir: Path = DEFAULT_HUB_STATE_DIR,
    host: str = DEFAULT_HUB_HOST,
    port: int = DEFAULT_HUB_PORT,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Launch the hub daemon completely detached from any agent gateway or job object."""
    resolved_root = Path(repo_root).resolve()
    resolved_state = Path(state_dir).resolve()
    resolved_state.mkdir(parents=True, exist_ok=True)
    target_log = Path(log_file).resolve() if log_file else resolved_state / "hub.log"
    py = resolve_detached_python()
    cmd_args = _hub_command_args(
        repo_root=resolved_root,
        state_dir=resolved_state,
        host=host,
        port=port,
        log_file=target_log,
        python_exe=py,
    )

    if sys.platform == "win32":
        cmd_line = subprocess.list2cmdline(cmd_args)
        cmd_b64 = base64.b64encode(cmd_line.encode("utf-8")).decode("ascii")
        cwd_b64 = base64.b64encode(str(resolved_root).encode("utf-8")).decode("ascii")
        ps_script = (
            "$ErrorActionPreference = 'Stop'\n"
            f"$cmd = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{cmd_b64}'))\n"
            f"$cwd = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{cwd_b64}'))\n"
            "$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            "-Arguments @{ CommandLine = $cmd; CurrentDirectory = $cwd }\n"
            "if ($res.ReturnValue -ne 0) { throw ('Win32_Process.Create failed: ' + $res.ReturnValue) }\n"
            "Write-Output $res.ProcessId\n"
        )
        encoded = base64.b64encode(ps_script.encode("utf-16-le")).decode("ascii")
        ps_exe = shutil.which("powershell.exe") or "powershell.exe"
        wmi_res = subprocess.run(
            [ps_exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if wmi_res.returncode == 0:
            out_lines = [line.strip() for line in wmi_res.stdout.splitlines() if line.strip().isdigit()]
            pid = int(out_lines[-1]) if out_lines else None
            return {
                "method": "win32_process_wmi",
                "pid": pid,
                "command": cmd_args,
                "log_file": str(target_log),
            }

        base_flags = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        try:
            proc = subprocess.Popen(
                cmd_args,
                cwd=str(resolved_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=base_flags | breakaway,
            )
            return {
                "method": "popen_breakaway",
                "pid": proc.pid,
                "command": cmd_args,
                "log_file": str(target_log),
            }
        except PermissionError:
            proc = subprocess.Popen(
                cmd_args,
                cwd=str(resolved_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=base_flags,
            )
            return {
                "method": "popen_detached",
                "pid": proc.pid,
                "command": cmd_args,
                "log_file": str(target_log),
            }

    proc = subprocess.Popen(
        cmd_args,
        cwd=str(resolved_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    return {
        "method": "posix_setsid",
        "pid": proc.pid,
        "command": cmd_args,
        "log_file": str(target_log),
    }


def _hub_post_json(
    url: str,
    *,
    token: str,
    timeout_s: float = 3.0,
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Hub-Token": token,
        },
    )
    with _LOCAL_HTTP_OPENER.open(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)


def stop_running_hub(
    *,
    host: str = DEFAULT_HUB_HOST,
    port: int = DEFAULT_HUB_PORT,
    state_dir: Path = DEFAULT_HUB_STATE_DIR,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    """Gracefully stop a running Hub daemon and wait for loopback port release."""
    ok, health, _ = probe_http_health(host, port, "/api/health", timeout_s=1.2)
    pid_file = Path(state_dir).resolve() / "hub.pid.json"
    pid: int | None = health.get("pid") if isinstance(health, dict) else None
    if pid is None and pid_file.is_file():
        try:
            pid_doc = json.loads(pid_file.read_text(encoding="utf-8"))
            if _is_strict_int(pid_doc.get("pid")):
                pid = int(pid_doc["pid"])
        except Exception:
            pass

    if ok and isinstance(health, dict) and isinstance(health.get("session_token"), str):
        shutdown_url = _format_http_url(host, port, "/api/hub/shutdown")
        try:
            _hub_post_json(shutdown_url, token=health["session_token"], timeout_s=3.0)
        except Exception:
            pass

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            still_up, _, _ = probe_http_health(host, port, "/api/health", timeout_s=0.4)
            if not still_up:
                return {"stopped": True, "method": "http_shutdown", "pid": pid}
            time.sleep(0.2)

    if pid is not None and sys.platform == "win32":
        res = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if res.returncode == 0:
            return {"stopped": True, "method": "taskkill_tree", "pid": pid}

    if not ok:
        return {"stopped": False, "method": "not_running", "pid": pid}
    return {"stopped": False, "method": "timeout", "pid": pid}


def create_site_hub_server(
    *,
    state_dir: Path = DEFAULT_HUB_STATE_DIR,
    repo_root: Path = _REPO_ROOT,
    host: str = DEFAULT_HUB_HOST,
    port: int = DEFAULT_HUB_PORT,
    assets_dir: Path = ASSETS_DIR,
    poll_interval_s: float = 3.0,
    autostart_sites: bool = False,
) -> ExclusiveHubHTTPServer:
    """Create the loopback HTTP portal & process supervisor server."""
    if host not in _ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(f"hub only binds to loopback hosts {sorted(_ALLOWED_LOOPBACK_HOSTS)}, got {host!r}")

    resolved_state = Path(state_dir).resolve()
    resolved_state.mkdir(parents=True, exist_ok=True)
    resolved_assets = Path(assets_dir).resolve()
    registry = SiteRegistryStore(resolved_state, hub_port=port)
    supervisor = SiteSupervisor(
        registry,
        repo_root=repo_root,
        poll_interval_s=poll_interval_s,
    )
    session_token = secrets.token_urlsafe(32)
    shutdown_lock = threading.Lock()
    shutdown_started = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _check_host(self) -> bool:
            if not _parse_loopback_host(self.headers.get("Host"), self.server.server_port):
                self._send_json(403, {"error": "invalid loopback Host header"})
                return False
            return True

        def _check_mutation_security(self) -> bool:
            if not self._check_host():
                return False
            if not _parse_loopback_origin(self.headers.get("Origin"), self.server.server_port):
                self._send_json(403, {"error": "cross-origin mutation is not allowed"})
                return False
            provided = self.headers.get("X-Hub-Token", "")
            if not secrets.compare_digest(provided, session_token):
                self._send_json(403, {"error": "missing or invalid X-Hub-Token"})
                return False
            return True

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _stream_asset(self, name: str) -> None:
            try:
                asset_path = safe_path(resolved_assets / name, [resolved_assets])
            except ValueError:
                self._send_json(400, {"error": "invalid asset path"})
                return
            if not asset_path.is_file():
                if name == "index.html":
                    fallback = (
                        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
                        "<title>Mouse Behavior Site Hub</title>"
                        "<body><h1>Site Hub assets missing</h1>"
                        "<p>site_hub_assets/index.html was not found.</p></body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(fallback)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(fallback)
                    return
                self._send_json(404, {"error": f"asset not found: {name}"})
                return
            data = asset_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _STATIC_MIME_TYPES[name])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _consume_optional_body(self) -> bool:
            raw_len = self.headers.get("Content-Length")
            if not raw_len:
                return True
            try:
                length = int(raw_len)
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return False
            if length < 0 or length > _MAX_POST_BYTES:
                self._send_json(413, {"error": f"request body too large (max {_MAX_POST_BYTES})"})
                return False
            if length > 0:
                self.rfile.read(length)
            return True

        def do_HEAD(self) -> None:
            self._handle_get()

        def do_GET(self) -> None:
            self._handle_get()

        def _handle_get(self) -> None:
            if not self._check_host():
                return
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            query = parse_qs(parsed.query)

            if route in {"/", "/index.html", "/app.css", "/app.js"}:
                name = "index.html" if route == "/" else route.lstrip("/")
                self._stream_asset(name)
                return

            try:
                if route == "/api/health":
                    try:
                        snap = supervisor.snapshot(probe_live=False)
                        corrupt = False
                        summary = snap["summary"]
                    except CorruptRegistryError:
                        corrupt = True
                        summary = {"total": 0, "healthy": 0, "review_count": 0, "error_display_count": 0, "unhealthy": 0}
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "schema": HEALTH_SCHEMA,
                            "pid": os.getpid(),
                            "host": host,
                            "port": self.server.server_port,
                            "state_dir": str(resolved_state),
                            "started_at": self.server.started_at,
                            "uptime_s": round(max(0.0, time.monotonic() - self.server.started_monotonic), 1),
                            "session_token": session_token,
                            "corrupt_registry": corrupt,
                            "summary": summary,
                        },
                    )
                    return

                if route == "/api/sites":
                    snap = supervisor.snapshot(probe_live=True)
                    self._send_json(
                        200,
                        {
                            "schema": STATUS_SCHEMA,
                            "generated_at": _utc_now_iso(),
                            "hub": {
                                "pid": os.getpid(),
                                "host": host,
                                "port": self.server.server_port,
                                "url": _format_http_url(host, self.server.server_port, "/"),
                                "state_dir": str(resolved_state),
                                "started_at": self.server.started_at,
                                "uptime_s": round(max(0.0, time.monotonic() - self.server.started_monotonic), 1),
                                "autostart": get_windows_autostart_status(),
                            },
                            "summary": snap["summary"],
                            "sites": snap["sites"],
                        },
                    )
                    return

                if route.startswith("/api/sites/") and route.endswith("/logs"):
                    site_id = route[len("/api/sites/") : -len("/logs")].strip("/")
                    if not site_id or "/" in site_id:
                        self._send_json(400, {"error": "invalid site_id"})
                        return
                    try:
                        lines_req = int(query.get("lines", ["80"])[0])
                    except ValueError:
                        lines_req = 80
                    self._send_json(200, supervisor.read_site_logs(site_id, max_lines=lines_req))
                    return

                self._send_json(404, {"error": "not found", "route": route})
            except CorruptRegistryError as exc:
                self._send_json(500, {"error": str(exc), "corrupt_registry": True})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:
            if not self._check_mutation_security():
                return
            if not self._consume_optional_body():
                return

            route = unquote(urlsplit(self.path).path)
            try:
                if route == "/api/sites/reload":
                    snap = supervisor.reload_registry()
                    self._send_json(200, {"status": "reloaded", **snap})
                    return

                if route == "/api/hub/shutdown":
                    nonlocal shutdown_started
                    with shutdown_lock:
                        already = shutdown_started
                        shutdown_started = True
                    self._send_json(200, {"status": "shutting_down", "already_requested": already})
                    if not already:
                        threading.Thread(
                            target=self.server.shutdown,
                            name="site-hub-shutdown",
                            daemon=True,
                        ).start()
                    return

                for action, fn in (
                    ("/start", supervisor.start_site),
                    ("/stop", supervisor.stop_site),
                    ("/restart", supervisor.restart_site),
                ):
                    if route.startswith("/api/sites/") and route.endswith(action):
                        site_id = route[len("/api/sites/") : -len(action)].strip("/")
                        if not site_id or "/" in site_id:
                            self._send_json(400, {"error": "invalid site_id"})
                            return
                        row = fn(site_id)
                        self._send_json(200, {"status": "ok", "site": row})
                        return

                self._send_json(404, {"error": "not found", "route": route})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except CorruptRegistryError as exc:
                self._send_json(409, {"error": str(exc), "corrupt_registry": True})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

    server = ExclusiveHubHTTPServer(
        (host, port),
        Handler,
        registry=registry,
        supervisor=supervisor,
        session_token=session_token,
    )
    if autostart_sites:
        try:
            supervisor.start_all_enabled()
        except CorruptRegistryError as exc:
            LOGGER.error("corrupt sites registry on startup: %s", exc)
    supervisor.start_watchdog()
    return server


def _file_sha256_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


__all__ = [
    "ASSETS_DIR",
    "AUTOSTART_TASK_NAME",
    "CorruptRegistryError",
    "DEFAULT_HUB_HOST",
    "DEFAULT_HUB_PORT",
    "DEFAULT_HUB_STATE_DIR",
    "ExclusiveHubHTTPServer",
    "HEALTH_SCHEMA",
    "REGISTRY_SCHEMA",
    "STATUS_SCHEMA",
    "SiteRegistryStore",
    "SiteSupervisor",
    "VALID_CATEGORIES",
    "build_default_sites",
    "create_site_hub_server",
    "get_windows_autostart_status",
    "install_windows_autostart",
    "probe_http_health",
    "resolve_child_python",
    "resolve_detached_python",
    "spawn_detached_hub",
    "stop_running_hub",
    "uninstall_windows_autostart",
    "validate_site_spec",
]
