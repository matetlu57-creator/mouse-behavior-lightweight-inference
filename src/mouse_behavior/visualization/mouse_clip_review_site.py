"""Loopback HTTP server for the Mouse Behavior Review & Clip Collection Website.

Serves static assets from ``clip_review_assets/``, streams read-only MP4 clips
with HTTP 206 Range support, persists human reviews atomically, and exposes
verified frame-accurate H.264 clip collection APIs writing exclusively under
``output_dir``.
"""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from mouse_behavior.evaluation.review_transaction import safe_path
from mouse_behavior.visualization.mouse_clip_cropper import (
    BatchCollectionManager,
    export_collected_clip,
    is_ffmpeg_available,
)
from mouse_behavior.visualization.mouse_clip_review_store import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    CorruptStoreError,
    MouseClipReviewStore,
)

DEFAULT_PORT = 8795
ASSETS_DIR = Path(__file__).with_name("clip_review_assets")
_ALLOWED_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_POST_BYTES = 1024 * 1024
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_STATIC_MIME_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    """Loopback ThreadingHTTPServer enforcing port exclusivity (including Windows)."""

    allow_reuse_address = False
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        store: MouseClipReviewStore,
        batch_manager: BatchCollectionManager,
        session_token: str,
    ) -> None:
        self.store = store
        self.batch_manager = batch_manager
        self.session_token = session_token
        super().__init__(server_address, handler_cls)

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def server_close(self) -> None:
        try:
            self.batch_manager.shutdown()
        finally:
            super().server_close()


def _parse_loopback_host(host_header: str | None, expected_port: int) -> bool:
    """Validate loopback ``Host`` header including bracketed IPv6 ``[::1]:port``."""
    if not host_header:
        return False
    raw = host_header.strip().lower()
    if not raw:
        return False

    host_part: str
    port_part: str | None = None
    if raw.startswith("["):
        end_bracket = raw.find("]")
        if end_bracket == -1:
            return False
        host_part = raw[1:end_bracket]
        rest = raw[end_bracket + 1 :]
        if rest:
            if not rest.startswith(":"):
                return False
            port_part = rest[1:]
    else:
        if raw.count(":") > 1:
            return False
        if ":" in raw:
            host_part, port_part = raw.split(":", 1)
        else:
            host_part = raw

    if host_part not in {"127.0.0.1", "localhost", "::1"}:
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
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        return False
    try:
        origin_port = parsed.port
    except ValueError:
        return False
    if origin_port is not None and expected_port and origin_port != expected_port:
        return False
    return True


def create_mouse_clip_review_server(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    assets_dir: Path = ASSETS_DIR,
) -> ExclusiveThreadingHTTPServer:
    """Create the loopback HTTP server for behavior review & clip collection."""
    if host not in _ALLOWED_BIND_HOSTS:
        raise ValueError(f"server only binds to loopback hosts {sorted(_ALLOWED_BIND_HOSTS)}, got {host!r}")

    store = MouseClipReviewStore(source_dir=source_dir, output_dir=output_dir)
    batch_manager = BatchCollectionManager(store=store)
    session_token = secrets.token_urlsafe(32)
    resolved_assets = Path(assets_dir).resolve()

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
            provided = self.headers.get("X-Review-Token", "")
            if not secrets.compare_digest(provided, session_token):
                self._send_json(403, {"error": "missing or invalid X-Review-Token"})
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

        def _send_download(self, filename: str, content: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        def _send_range_error(self, size: int) -> None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

        def _stream_file(
            self,
            file_path: Path,
            *,
            content_type: str | None = None,
            no_store: bool = False,
        ) -> None:
            try:
                handle = file_path.open("rb")
            except FileNotFoundError:
                self._send_json(404, {"error": "file not found"})
                return
            except OSError as exc:
                self._send_json(503, {"error": f"file unavailable: {exc}"})
                return

            with handle:
                size = file_path.stat().st_size
                start = 0
                end = max(0, size - 1)
                status = 200
                range_hdr = self.headers.get("Range")
                if range_hdr and size > 0:
                    first_range = range_hdr.split(",", 1)[0].strip()
                    match = _RANGE_RE.fullmatch(first_range)
                    if not match or not any(match.groups()):
                        self._send_range_error(size)
                        return
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), size - 1) if right else size - 1
                    else:
                        suffix = int(right)
                        if suffix <= 0:
                            self._send_range_error(size)
                            return
                        start = max(0, size - suffix)
                        end = size - 1
                    if not (0 <= start <= end < size):
                        self._send_range_error(size)
                        return
                    status = 206

                length = (end - start + 1) if size > 0 else 0
                mime = content_type or mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if no_store:
                    self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()

                if self.command == "HEAD" or length == 0:
                    return

                handle.seek(start)
                remaining = length
                try:
                    while remaining > 0:
                        chunk = handle.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        def _read_json_body(self) -> dict[str, Any] | None:
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                self._send_json(415, {"error": "Content-Type application/json is required"})
                return None
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length <= 0 or length > _MAX_POST_BYTES:
                self._send_json(413, {"error": f"request body must be 1..{_MAX_POST_BYTES} bytes"})
                return None
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": f"malformed JSON body: {exc}"})
                return None
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "JSON body must be an object"})
                return None
            return payload

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
                try:
                    asset_path = safe_path(resolved_assets / name, [resolved_assets])
                except ValueError:
                    self._send_json(400, {"error": "invalid asset path"})
                    return
                if not asset_path.is_file():
                    if name == "index.html":
                        fallback = (
                            "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
                            "<title>Mouse Clip Review</title>"
                            "<body><h1>Frontend assets missing</h1>"
                            "<p>clip_review_assets/index.html was not found.</p></body></html>"
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(fallback)))
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        if self.command != "HEAD":
                            self.wfile.write(fallback)
                        return
                    self._send_json(404, {"error": f"static asset not found: {name}"})
                    return
                self._stream_file(asset_path, content_type=_STATIC_MIME_TYPES[name])
                return

            try:
                if route == "/api/health":
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "source_dir": str(store.source_dir),
                            "output_dir": str(store.output_dir),
                            "source_exists": store.source_dir.is_dir(),
                            "read_only_source": True,
                            "ffmpeg_available": is_ffmpeg_available(),
                            "session_token": session_token,
                        },
                    )
                    return

                if route == "/api/catalog":
                    refresh = query.get("refresh", ["0"])[0] in {"1", "true", "yes"}
                    self._send_json(200, store.get_catalog(force_rescan=refresh))
                    return

                if route.startswith("/api/clip/"):
                    clip_uid = route[len("/api/clip/") :]
                    if not clip_uid or "/" in clip_uid:
                        self._send_json(400, {"error": "invalid clip_uid"})
                        return
                    self._send_json(200, store.get_clip_detail(clip_uid))
                    return

                if route.startswith("/api/video/"):
                    clip_uid = route[len("/api/video/") :]
                    if not clip_uid or "/" in clip_uid:
                        self._send_json(400, {"error": "invalid clip_uid"})
                        return
                    video_path = store.get_video_path(clip_uid)
                    self._stream_file(video_path, content_type="video/mp4", no_store=True)
                    return

                if route == "/api/reviews":
                    self._send_json(200, store.load_reviews())
                    return

                if route == "/api/reviews/export":
                    fmt = (query.get("format", ["json"])[0] or "json").lower()
                    if fmt == "json":
                        payload = json.dumps(
                            store.load_reviews(),
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._send_download(
                            "mouse_clip_reviews.json",
                            payload,
                            "application/json; charset=utf-8",
                        )
                        return
                    if fmt == "csv":
                        self._send_download(
                            "mouse_clip_reviews.csv",
                            store.export_reviews_csv(),
                            "text/csv; charset=utf-8",
                        )
                        return
                    self._send_json(400, {"error": "format must be 'json' or 'csv'"})
                    return

                if route in {"/api/collections", "/api/crops"}:
                    self._send_json(200, store.load_collections())
                    return

                if route in {"/api/collections/export", "/api/crops/export"}:
                    fmt = (query.get("format", ["json"])[0] or "json").lower()
                    if fmt == "json":
                        payload = json.dumps(
                            store.load_collections(),
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        ).encode("utf-8")
                        self._send_download(
                            "mouse_clip_collections.json",
                            payload,
                            "application/json; charset=utf-8",
                        )
                        return
                    if fmt == "csv":
                        self._send_download(
                            "mouse_clip_collections.csv",
                            store.export_collections_csv(),
                            "text/csv; charset=utf-8",
                        )
                        return
                    self._send_json(400, {"error": "format must be 'json' or 'csv'"})
                    return

                if route.startswith("/api/exported-video/"):
                    export_id = route[len("/api/exported-video/") :]
                    if not export_id or "/" in export_id:
                        self._send_json(400, {"error": "invalid export_id"})
                        return
                    variant = (query.get("variant", ["raw"])[0] or "raw").lower()
                    video_path = store.get_exported_video_path(export_id, variant=variant)
                    self._stream_file(video_path, content_type="video/mp4", no_store=True)
                    return

                for prefix in ("/api/collect/jobs/", "/api/crop/jobs/"):
                    if route.startswith(prefix):
                        job_id = route[len(prefix) :]
                        if not job_id or "/" in job_id:
                            self._send_json(400, {"error": "invalid job_id"})
                            return
                        self._send_json(200, batch_manager.get_job(job_id))
                        return

                self._send_json(404, {"error": "unknown route", "route": route})
            except CorruptStoreError as exc:
                self._send_json(500, {"error": str(exc), "corrupt_store": True})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except FileNotFoundError as exc:
                if route == "/api/catalog":
                    self._send_json(503, {"error": str(exc)})
                else:
                    self._send_json(404, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except (OSError, RuntimeError) as exc:
                self._send_json(503, {"error": str(exc)})

        def do_POST(self) -> None:
            if not self._check_mutation_security():
                return
            route = unquote(urlsplit(self.path).path)
            if route not in {
                "/api/reviews",
                "/api/collect",
                "/api/crop",
                "/api/collect/batch",
                "/api/crop/batch",
            }:
                self._send_json(404, {"error": "unknown API route", "route": route})
                return

            payload = self._read_json_body()
            if payload is None:
                return

            try:
                if route == "/api/reviews":
                    result = store.save_review(payload)
                    self._send_json(200, result)
                    return

                if route in {"/api/collect", "/api/crop"}:
                    collection = export_collected_clip(store, payload)
                    self._send_json(200, {"status": "ok", "collection": collection})
                    return

                if route in {"/api/collect/batch", "/api/crop/batch"}:
                    accepted = batch_manager.submit(payload.get("items"))
                    self._send_json(202, accepted)
                    return
            except CorruptStoreError as exc:
                self._send_json(409, {"error": str(exc), "corrupt_store": True})
            except FileExistsError as exc:
                self._send_json(409, {"error": str(exc)})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except (OSError, RuntimeError) as exc:
                self._send_json(503, {"error": str(exc)})

        def do_DELETE(self) -> None:
            if not self._check_mutation_security():
                return
            route = unquote(urlsplit(self.path).path)
            export_id: str | None = None
            for prefix in ("/api/collections/", "/api/crops/"):
                if route.startswith(prefix):
                    export_id = route[len(prefix) :]
                    break
            if export_id is None:
                self._send_json(404, {"error": "unknown API route", "route": route})
                return
            if not export_id or "/" in export_id:
                self._send_json(400, {"error": "invalid export_id"})
                return

            try:
                result = store.delete_collection(export_id)
                self._send_json(200, result)
            except CorruptStoreError as exc:
                self._send_json(409, {"error": str(exc), "corrupt_store": True})
            except KeyError as exc:
                self._send_json(404, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except (OSError, RuntimeError) as exc:
                self._send_json(503, {"error": str(exc)})

    return ExclusiveThreadingHTTPServer(
        (host, port),
        Handler,
        store=store,
        batch_manager=batch_manager,
        session_token=session_token,
    )
