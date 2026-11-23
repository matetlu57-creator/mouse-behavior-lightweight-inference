"""Unit, security, lifecycle, and API resilience tests for the Persistent Site Hub."""
from __future__ import annotations

import http.client
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from mouse_behavior.visualization.site_hub import (
    ASSETS_DIR,
    DEFAULT_HUB_PORT,
    HEALTH_SCHEMA,
    REGISTRY_SCHEMA,
    STATUS_SCHEMA,
    CorruptRegistryError,
    SiteRegistryStore,
    build_default_sites,
    create_site_hub_server,
    validate_site_spec,
)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_dummy_site_script(path: Path) -> Path:
    script = path / "dummy_site.py"
    script.write_text(
        """
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def do_GET(self):
        if self.path == "/api/health":
            body = json.dumps({"status": "ok", "pid": os.getpid()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"<h1>Dummy Review Site</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

print(f"dummy site listening on {port} pid={os.getpid()}", flush=True)
srv = HTTPServer(("127.0.0.1", port), Handler)
srv.serve_forever()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return script


def _http_request(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, raw
    finally:
        conn.close()


def _wait_until(predicate, *, timeout_s: float = 8.0, interval_s: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout_s
    last_val = None
    while time.monotonic() < deadline:
        last_val = predicate()
        if last_val:
            return last_val
        time.sleep(interval_s)
    raise AssertionError(f"condition not satisfied within {timeout_s}s (last={last_val!r})")


def test_validate_site_spec_rejects_bad_inputs(tmp_path: Path) -> None:
    base = {
        "site_id": "demo_review",
        "title": "Demo Review",
        "category": "review",
        "host": "127.0.0.1",
        "port": 8796,
        "command": [sys.executable, "-c", "pass"],
        "cwd": str(tmp_path),
    }
    valid = validate_site_spec(base)
    assert valid["site_id"] == "demo_review"
    assert valid["health_path"] == "/api/health"

    with pytest.raises(ValueError, match="hub port"):
        validate_site_spec({**base, "port": DEFAULT_HUB_PORT})

    with pytest.raises(ValueError, match="loopback"):
        validate_site_spec({**base, "host": "0.0.0.0"})

    with pytest.raises(ValueError, match="invalid site_id"):
        validate_site_spec({**base, "site_id": "../bad"})

    with pytest.raises(ValueError, match="invalid category"):
        validate_site_spec({**base, "category": "unknown_type"})

    with pytest.raises(ValueError, match="command"):
        validate_site_spec({**base, "command": []})


def test_registry_crud_duplicate_port_and_default_sites(tmp_path: Path) -> None:
    store = SiteRegistryStore(tmp_path / "hub_state")
    assert store.list_sites() == []

    spec_a = store.upsert_site(
        {
            "site_id": "site_a",
            "title": "Site A",
            "category": "review",
            "port": 8801,
            "command": [sys.executable, "-V"],
            "cwd": str(tmp_path),
        }
    )
    assert spec_a["port"] == 8801
    assert len(store.list_sites()) == 1

    with pytest.raises(ValueError, match="already registered"):
        store.upsert_site(
            {
                "site_id": "site_b",
                "title": "Site B",
                "category": "error_display",
                "port": 8801,
                "command": [sys.executable, "-V"],
                "cwd": str(tmp_path),
            }
        )

    defaults = build_default_sites()
    assert [d["site_id"] for d in defaults] == [
        "mouse_clip_review",
        "long_video_group_review",
        "behavior_error_review",
    ]
    assert [d["port"] for d in defaults] == [8795, 8790, 8776]
    clip_cmd = " ".join(defaults[0]["command"])
    assert "mouse_behavior_cleaned_dataset_20261121" in clip_cmd
    assert "mouse_clip_review_site_cleaned_20261121" in clip_cmd
    assert "F:\\mouse " not in clip_cmd

    merged = store.ensure_default_sites(defaults)
    assert {item["site_id"] for item in merged} == {
        "site_a",
        "mouse_clip_review",
        "long_video_group_review",
        "behavior_error_review",
    }
    assert store.remove_site("site_a") is True
    assert store.remove_site("site_a") is False


@pytest.mark.parametrize(
    "broken_bytes",
    [
        b"{not valid json",
        json.dumps({"schema_version": "wrong_schema", "sites": []}).encode("utf-8"),
        json.dumps({"schema_version": REGISTRY_SCHEMA, "sites": [{"site_id": "x"}]}).encode("utf-8"),
    ],
)
def test_corrupt_registry_is_preserved_and_returns_500(tmp_path: Path, broken_bytes: bytes) -> None:
    state_dir = tmp_path / "hub_corrupt"
    state_dir.mkdir(parents=True, exist_ok=True)
    reg_path = state_dir / "sites.json"
    reg_path.write_bytes(broken_bytes)

    store = SiteRegistryStore(state_dir)
    with pytest.raises(CorruptRegistryError):
        store.load()
    with pytest.raises(CorruptRegistryError):
        store.ensure_default_sites()
    assert reg_path.read_bytes() == broken_bytes

    server = create_site_hub_server(state_dir=state_dir, host="127.0.0.1", port=0, autostart_sites=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, raw = _http_request(server.server_port, "GET", "/api/sites")
        assert status == 500
        assert "application/json" in headers["content-type"]
        payload = json.loads(raw.decode("utf-8"))
        assert payload["corrupt_registry"] is True
        assert reg_path.read_bytes() == broken_bytes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_portal_assets_have_zero_cdn_and_resilient_checks() -> None:
    index_html = (ASSETS_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (ASSETS_DIR / "app.js").read_text(encoding="utf-8")
    combined = index_html + "\n" + app_js

    assert "https://" not in combined, "portal assets must not depend on external HTTPS CDNs"
    for match in re.finditer(r"https?://([^\s/'\"`<>]+)", combined):
        authority = match.group(1)
        if authority.startswith("${"):
            continue
        host = authority.split("]", 1)[0] + "]" if authority.startswith("[") else authority.rsplit(":", 1)[0]
        assert host in {"127.0.0.1", "localhost", "[::1]"}, f"external URL {authority!r} is forbidden"
    assert "application/json" in app_js
    assert "OFFLINE_REFERENCE_SITES" in app_js
    assert "retry-connection-btn" in index_html


def test_live_hub_lifecycle_security_reload_and_autorestart(tmp_path: Path) -> None:
    state_dir = tmp_path / "hub_live"
    dummy_script = _write_dummy_site_script(tmp_path)
    port1 = _free_loopback_port()
    port2 = _free_loopback_port()
    while port2 == port1:
        port2 = _free_loopback_port()

    store = SiteRegistryStore(state_dir, hub_port=DEFAULT_HUB_PORT)
    store.upsert_site(
        {
            "site_id": "dummy_review",
            "title": "Dummy Review",
            "category": "review",
            "host": "127.0.0.1",
            "port": port1,
            "health_path": "/api/health",
            "command": [sys.executable, str(dummy_script), str(port1)],
            "cwd": str(tmp_path),
            "enabled": True,
            "auto_restart": True,
            "description": "Synthetic child site",
            "data_summary": "0 clips",
        }
    )

    server = create_site_hub_server(
        state_dir=state_dir,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.25,
        autostart_sites=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    hub_port = server.server_port

    try:
        # 1. Static shell assets work immediately.
        for path, ctype in (("/", "text/html"), ("/app.css", "text/css"), ("/app.js", "application/javascript")):
            st, hdrs, body = _http_request(hub_port, "GET", path)
            assert st == 200
            assert ctype in hdrs["content-type"]
            assert len(body) > 100

        # 2. Health endpoint & token check.
        st, hdrs, raw = _http_request(hub_port, "GET", "/api/health")
        assert st == 200
        assert hdrs["cache-control"] == "no-store"
        assert hdrs["x-content-type-options"] == "nosniff"
        health = json.loads(raw.decode("utf-8"))
        assert health["status"] == "ok"
        assert health["schema"] == HEALTH_SCHEMA
        token = health["session_token"]

        # 3. Child process becomes healthy and logs are readable.
        def _is_dummy_healthy():
            code, _, body_bytes = _http_request(hub_port, "GET", "/api/sites")
            assert code == 200
            doc = json.loads(body_bytes.decode("utf-8"))
            assert doc["schema"] == STATUS_SCHEMA
            row = next((s for s in doc["sites"] if s["site_id"] == "dummy_review"), None)
            return row if row and row["healthy"] and row["status"] == "running" else None

        row1 = _wait_until(_is_dummy_healthy)
        old_pid = row1["pid"]
        assert old_pid is not None

        st, _, raw = _http_request(hub_port, "GET", "/api/sites/dummy_review/logs?lines=20")
        assert st == 200
        log_doc = json.loads(raw.decode("utf-8"))
        assert any("dummy site listening" in line for line in log_doc["lines"])

        # 4. Security guards: bad Host, bad Origin, missing/wrong X-Hub-Token.
        st_host, _, _ = _http_request(hub_port, "GET", "/api/health", headers={"Host": "evil.example.com"})
        assert st_host == 403

        st_origin, _, _ = _http_request(
            hub_port,
            "POST",
            "/api/sites/dummy_review/restart",
            headers={"Origin": "http://evil.example.com", "X-Hub-Token": token},
        )
        assert st_origin == 403

        st_tok, _, _ = _http_request(
            hub_port,
            "POST",
            "/api/sites/dummy_review/restart",
            headers={"X-Hub-Token": "wrong-token"},
        )
        assert st_tok == 403

        # 5. Unexpected child crash triggers automatic watchdog restart.
        with server.supervisor._lock:
            proc = server.supervisor._states["dummy_review"].process
            assert proc is not None
            proc.kill()
            proc.wait(timeout=3)

        def _restarted():
            code, _, body_bytes = _http_request(hub_port, "GET", "/api/sites")
            assert code == 200
            doc = json.loads(body_bytes.decode("utf-8"))
            row = next(s for s in doc["sites"] if s["site_id"] == "dummy_review")
            return row if row["healthy"] and row["restart_count"] >= 1 and row["pid"] != old_pid else None

        restarted_row = _wait_until(_restarted, timeout_s=8.0)
        assert restarted_row["restart_count"] >= 1

        # 6. Manual stop/start cycle via authorized POST.
        st_stop, _, _ = _http_request(
            hub_port,
            "POST",
            "/api/sites/dummy_review/stop",
            headers={"X-Hub-Token": token},
        )
        assert st_stop == 200

        st_after_stop, _, raw_after_stop = _http_request(hub_port, "GET", "/api/sites")
        assert st_after_stop == 200
        stopped_row = json.loads(raw_after_stop.decode("utf-8"))["sites"][0]
        assert stopped_row["status"] == "stopped"
        assert stopped_row["healthy"] is False

        # 7. Reload registry picks up a newly added site and starts it.
        store.upsert_site(
            {
                "site_id": "dummy_error",
                "title": "Dummy Error Site",
                "category": "error_display",
                "host": "127.0.0.1",
                "port": port2,
                "health_path": "/api/health",
                "command": [sys.executable, str(dummy_script), str(port2)],
                "cwd": str(tmp_path),
                "enabled": True,
                "auto_restart": True,
            }
        )
        st_reload, _, _ = _http_request(
            hub_port,
            "POST",
            "/api/sites/reload",
            headers={"X-Hub-Token": token},
        )
        assert st_reload == 200

        def _second_healthy():
            code, _, body_bytes = _http_request(hub_port, "GET", "/api/sites")
            assert code == 200
            doc = json.loads(body_bytes.decode("utf-8"))
            row = next((s for s in doc["sites"] if s["site_id"] == "dummy_error"), None)
            return row if row and row["healthy"] else None

        _wait_until(_second_healthy, timeout_s=8.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=4)
