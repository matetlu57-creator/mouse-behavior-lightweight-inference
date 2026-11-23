#!/usr/bin/env python3
"""Manage the gateway-independent Persistent Local Site Hub for review and error sites."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.review_transaction import write_json
from mouse_behavior.visualization.site_hub import (
    DEFAULT_HUB_HOST,
    DEFAULT_HUB_PORT,
    DEFAULT_HUB_STATE_DIR,
    VALID_CATEGORIES,
    CorruptRegistryError,
    SiteRegistryStore,
    build_default_sites,
    create_site_hub_server,
    get_windows_autostart_status,
    install_windows_autostart,
    probe_http_health,
    spawn_detached_hub,
    stop_running_hub,
    uninstall_windows_autostart,
)

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _safe_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _hub_url(host: str, port: int, path: str = "/") -> str:
    host_part = f"[{host}]" if ":" in host else host
    norm = path if path.startswith("/") else f"/{path}"
    return f"http://{host_part}:{port}{norm}"


def _fetch_hub_json(host: str, port: int, path: str, timeout_s: float = 3.0) -> dict[str, Any]:
    req = urllib.request.Request(
        _hub_url(host, port, path),
        headers={"Accept": "application/json"},
    )
    with _LOCAL_OPENER.open(req, timeout=timeout_s) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" not in ctype:
            raise ValueError(f"unexpected content-type from {path}: {ctype}")
        return json.loads(resp.read().decode("utf-8"))


def _post_hub_json(
    host: str,
    port: int,
    path: str,
    *,
    token: str,
    timeout_s: float = 12.0,
) -> dict[str, Any]:
    req = urllib.request.Request(
        _hub_url(host, port, path),
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Hub-Token": token,
        },
    )
    with _LOCAL_OPENER.open(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _configure_daemon_io(log_file: Path) -> Any:
    """Redirect logging and stdout/stderr cleanly (especially under pythonw.exe)."""
    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("a", encoding="utf-8", buffering=1, errors="replace")
    if sys.stdout is None:
        sys.stdout = log_stream
    if sys.stderr is None:
        sys.stderr = log_stream
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(log_stream)],
        force=True,
    )
    return log_stream


def _print_status_table(payload: dict[str, Any]) -> None:
    hub = payload["hub"]
    summary = payload["summary"]
    autostart = hub.get("autostart") or {}
    auto_text = f"开启 ({autostart.get('method')})" if autostart.get("installed") else "未开启"
    print("=" * 84)
    print(f"  小鼠行为常驻站点中心: {hub['url']}")
    print(
        f"  PID: {hub['pid']} | 运行时长: {hub['uptime_s']}s | "
        f"在线: {summary['healthy']}/{summary['total']} | 开机自启: {auto_text}"
    )
    print("=" * 84)
    print(f"  {'SITE_ID':<26} {'CATEGORY':<14} {'PORT':<6} {'STATUS':<18} {'HEALTH':<8} URL")
    print("-" * 84)
    for row in payload["sites"]:
        health_txt = "ok" if row["healthy"] else "down"
        print(
            f"  {row['site_id']:<26} {row['category']:<14} {row['port']:<6} "
            f"{row['status']:<18} {health_txt:<8} {row['url']}"
        )
    print("=" * 84)


def _cmd_init_defaults(args: argparse.Namespace) -> int:
    store = SiteRegistryStore(args.state_dir, hub_port=args.port)
    sites = store.ensure_default_sites(build_default_sites(REPO_ROOT))
    print(f"已初始化默认站点注册表 ({store.registry_path}): 共 {len(sites)} 个站点")
    for spec in sites:
        print(f"  - {spec['site_id']} ({spec['category']}) -> {_hub_url(spec['host'], spec['port'])}")
    return 0


def _wait_for_hub_and_sites(host: str, port: int, wait_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + wait_timeout
    last_err: str | None = None
    health_ok = False
    while time.monotonic() < deadline:
        ok, _, err = probe_http_health(host, port, "/api/health", timeout_s=1.0)
        if ok:
            health_ok = True
            break
        last_err = err
        time.sleep(0.25)

    if not health_ok:
        raise RuntimeError(f"站点中心未在 {wait_timeout}s 内响应 /api/health ({last_err})")

    # Wait briefly for enabled sites to complete startup and pass health checks.
    sites_deadline = max(time.monotonic() + 3.0, deadline)
    while time.monotonic() < sites_deadline:
        status_doc = _fetch_hub_json(host, port, "/api/sites", timeout_s=3.0)
        enabled_rows = [row for row in status_doc.get("sites", []) if row.get("enabled")]
        if all(row.get("healthy") for row in enabled_rows):
            return status_doc
        time.sleep(0.4)
    return _fetch_hub_json(host, port, "/api/sites", timeout_s=3.0)


def _cmd_start(args: argparse.Namespace) -> int:
    store = SiteRegistryStore(args.state_dir, hub_port=args.port)
    if not args.no_init_defaults and not store.load():
        store.ensure_default_sites(build_default_sites(REPO_ROOT))

    ok, health, _ = probe_http_health(args.host, args.port, "/api/health", timeout_s=1.2)
    if ok and isinstance(health, dict):
        print(f"站点中心已在运行: {_hub_url(args.host, args.port)} (PID {health.get('pid')})")
        status_doc = _fetch_hub_json(args.host, args.port, "/api/sites", timeout_s=3.0)
        _print_status_table(status_doc)
        if args.open_browser:
            webbrowser.open(_hub_url(args.host, args.port))
        return 0

    launch = spawn_detached_hub(
        repo_root=REPO_ROOT,
        state_dir=args.state_dir,
        host=args.host,
        port=args.port,
    )
    print(
        f"已通过独立进程模式启动站点中心: method={launch['method']} pid={launch['pid']} "
        f"log={launch['log_file']}"
    )
    status_doc = _wait_for_hub_and_sites(args.host, args.port, float(args.wait_timeout))
    _print_status_table(status_doc)
    if args.open_browser:
        webbrowser.open(_hub_url(args.host, args.port))
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    res = stop_running_hub(host=args.host, port=args.port, state_dir=args.state_dir)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["stopped"] or res["method"] == "not_running" else 1


def _cmd_restart(args: argparse.Namespace) -> int:
    stop_running_hub(host=args.host, port=args.port, state_dir=args.state_dir)
    return _cmd_start(args)


def _cmd_status(args: argparse.Namespace) -> int:
    ok, _, err = probe_http_health(args.host, args.port, "/api/health", timeout_s=1.5)
    if not ok:
        payload = {
            "running": False,
            "hub_url": _hub_url(args.host, args.port),
            "error": err,
            "autostart": get_windows_autostart_status(),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"站点中心未运行 ({_hub_url(args.host, args.port)}): {err}")
        return 1

    status_doc = _fetch_hub_json(args.host, args.port, "/api/sites", timeout_s=4.0)
    if args.json:
        print(json.dumps(status_doc, ensure_ascii=False, indent=2))
    else:
        _print_status_table(status_doc)
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("错误：必须通过 `-- <命令参数...>` 提供站点启动命令。", file=sys.stderr)
        return 2

    store = SiteRegistryStore(args.state_dir, hub_port=args.hub_port)
    spec = store.upsert_site(
        {
            "site_id": args.id,
            "title": args.title,
            "category": args.category,
            "host": args.site_host,
            "port": args.site_port,
            "health_path": args.health_path,
            "command": cmd,
            "cwd": str(args.cwd.resolve()),
            "enabled": not args.disabled,
            "auto_restart": not args.no_auto_restart,
            "description": args.description,
            "data_summary": args.data_summary,
        }
    )
    print(f"已注册站点 {spec['site_id']} -> {_hub_url(spec['host'], spec['port'])}")

    ok, health, _ = probe_http_health(args.hub_host, args.hub_port, "/api/health", timeout_s=1.2)
    if ok and isinstance(health, dict) and isinstance(health.get("session_token"), str):
        token = health["session_token"]
        _post_hub_json(args.hub_host, args.hub_port, "/api/sites/reload", token=token)
        if spec["enabled"]:
            _post_hub_json(args.hub_host, args.hub_port, f"/api/sites/{spec['site_id']}/restart", token=token)
        print("已通知在线站点中心重载并拉起新站点。")
    return 0


def _cmd_unregister(args: argparse.Namespace) -> int:
    store = SiteRegistryStore(args.state_dir, hub_port=args.port)
    removed = store.remove_site(args.id)
    if not removed:
        print(f"未找到站点 {args.id!r}", file=sys.stderr)
        return 1
    print(f"已从注册表移除站点: {args.id}")
    ok, health, _ = probe_http_health(args.host, args.port, "/api/health", timeout_s=1.2)
    if ok and isinstance(health, dict) and isinstance(health.get("session_token"), str):
        _post_hub_json(args.host, args.port, "/api/sites/reload", token=health["session_token"])
        print("已通知在线站点中心停止并卸载该站点。")
    return 0


def _cmd_install_autostart(args: argparse.Namespace) -> int:
    res = install_windows_autostart(
        repo_root=REPO_ROOT,
        state_dir=args.state_dir,
        host=args.host,
        port=args.port,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("installed") else 1


def _cmd_uninstall_autostart(_args: argparse.Namespace) -> int:
    res = uninstall_windows_autostart()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    log_file = args.log_file or (Path(args.state_dir).resolve() / "hub.log")
    log_stream = _configure_daemon_io(log_file)
    state_dir = Path(args.state_dir).resolve()
    pid_file = state_dir / "hub.pid.json"
    store = SiteRegistryStore(state_dir, hub_port=args.port)
    try:
        if not store.load():
            store.ensure_default_sites(build_default_sites(REPO_ROOT))
    except CorruptRegistryError as exc:
        logging.error("corrupt sites.json preserved unmodified: %s", exc)

    server = create_site_hub_server(
        state_dir=state_dir,
        repo_root=REPO_ROOT,
        host=args.host,
        port=args.port,
        poll_interval_s=args.poll_interval,
        autostart_sites=True,
    )
    write_json(
        pid_file,
        {
            "pid": os.getpid(),
            "host": args.host,
            "port": server.server_port,
            "url": _hub_url(args.host, server.server_port),
            "started_at": server.started_at,
        },
    )
    logging.info("site hub daemon listening on %s (pid=%s)", _hub_url(args.host, server.server_port), os.getpid())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if pid_file.is_file():
            try:
                pid_file.unlink()
            except OSError:
                pass
        log_stream.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def _add_hub_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default=DEFAULT_HUB_HOST, help="Loopback bind host (default: 127.0.0.1)")
        p.add_argument("--port", type=int, default=DEFAULT_HUB_PORT, help="Hub port (default: 8788)")
        p.add_argument("--state-dir", type=Path, default=DEFAULT_HUB_STATE_DIR, help="Hub state & logs directory")

    p_init = sub.add_parser("init-defaults", help="Initialize default review and error-display sites in sites.json")
    _add_hub_flags(p_init)
    p_init.set_defaults(func=_cmd_init_defaults)

    p_start = sub.add_parser("start", help="Start the detached gateway-independent Site Hub daemon")
    _add_hub_flags(p_start)
    p_start.add_argument("--no-init-defaults", action="store_true", help="Do not populate default sites when empty")
    p_start.add_argument("--wait-timeout", type=float, default=20.0, help="Seconds to wait for Hub readiness")
    p_start.add_argument("--open-browser", action="store_true", help="Open portal URL in default browser")
    p_start.set_defaults(func=_cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the running Site Hub daemon and supervised child sites")
    _add_hub_flags(p_stop)
    p_stop.set_defaults(func=_cmd_stop)

    p_restart = sub.add_parser("restart", help="Restart the Site Hub daemon")
    _add_hub_flags(p_restart)
    p_restart.add_argument("--no-init-defaults", action="store_true")
    p_restart.add_argument("--wait-timeout", type=float, default=20.0)
    p_restart.add_argument("--open-browser", action="store_true")
    p_restart.set_defaults(func=_cmd_restart)

    p_status = sub.add_parser("status", help="Inspect live status of the Site Hub and registered sites")
    _add_hub_flags(p_status)
    p_status.add_argument("--json", action="store_true", help="Print machine-readable JSON status")
    p_status.set_defaults(func=_cmd_status)

    p_reg = sub.add_parser("register", help="Register or update a site in sites.json (and reload if running)")
    p_reg.add_argument("--hub-host", default=DEFAULT_HUB_HOST, help="Hub loopback bind host (default: 127.0.0.1)")
    p_reg.add_argument("--hub-port", type=int, default=DEFAULT_HUB_PORT, help="Hub port (default: 8788)")
    p_reg.add_argument("--state-dir", type=Path, default=DEFAULT_HUB_STATE_DIR, help="Hub state & logs directory")
    p_reg.add_argument("--id", required=True, help="Unique site slug (e.g. my_review_site)")
    p_reg.add_argument("--title", required=True, help="Human-readable Chinese/English title")
    p_reg.add_argument("--category", choices=sorted(VALID_CATEGORIES), default="review")
    p_reg.add_argument(
        "--port",
        "--site-port",
        "--target-port",
        dest="site_port",
        type=int,
        required=True,
        help="Loopback port of the child site",
    )
    p_reg.add_argument(
        "--host",
        "--site-host",
        dest="site_host",
        default=DEFAULT_HUB_HOST,
        help="Loopback host of the child site (default: 127.0.0.1)",
    )
    p_reg.add_argument("--health-path", default="/api/health")
    p_reg.add_argument("--description", default="")
    p_reg.add_argument("--data-summary", default="")
    p_reg.add_argument("--cwd", type=Path, default=REPO_ROOT)
    p_reg.add_argument("--disabled", action="store_true")
    p_reg.add_argument("--no-auto-restart", action="store_true")
    p_reg.add_argument("command", nargs=argparse.REMAINDER, help="Command after `--`, e.g. `-- D:\\Anaconda3\\python.exe scripts/serve_xxx.py --port 8796`")
    p_reg.set_defaults(func=_cmd_register)

    p_unreg = sub.add_parser("unregister", help="Remove a site from sites.json and stop it if running")
    _add_hub_flags(p_unreg)
    p_unreg.add_argument("--id", required=True)
    p_unreg.set_defaults(func=_cmd_unregister)

    p_auto = sub.add_parser("install-autostart", help="Install Windows user-logon autostart for the Site Hub")
    _add_hub_flags(p_auto)
    p_auto.set_defaults(func=_cmd_install_autostart)

    p_unauto = sub.add_parser("uninstall-autostart", help="Remove Windows logon autostart for the Site Hub")
    p_unauto.set_defaults(func=_cmd_uninstall_autostart)

    p_daemon = sub.add_parser("daemon", help="Run the foreground supervisor & portal HTTP server")
    _add_hub_flags(p_daemon)
    p_daemon.add_argument("--log-file", type=Path, default=None)
    p_daemon.add_argument("--poll-interval", type=float, default=3.0)
    p_daemon.set_defaults(func=_cmd_daemon)

    return parser


def main(argv: list[str] | None = None) -> int:
    _safe_stdout_utf8()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CorruptRegistryError as exc:
        print(f"注册表损坏错误（已保留原文件未覆盖）: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
