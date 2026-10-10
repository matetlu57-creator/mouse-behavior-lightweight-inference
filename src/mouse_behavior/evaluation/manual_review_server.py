"""Loopback-only review API and range-readable static videos."""

from __future__ import annotations

import json
import mimetypes
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from mouse_behavior.evaluation.manual_review import ReviewConflict
from mouse_behavior.evaluation.review_transaction import safe_path


def create_server(store, port=8768):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # POST errors may contain reviewer notes; do not print request bodies.
            return

        def _host_ok(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',
                                                f'localhost:{self.server.server_port}'}

        def _json(self, code, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def _snapshot(self):
            return {**store.snapshot(), 'token': token}

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {'error': 'invalid loopback Host'})
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self._json(403, {'error': 'cross-origin updates are not allowed'})
            if not secrets.compare_digest(self.headers.get('X-Review-Token', ''), token):
                return self._json(403, {'error': 'missing or expired review session token'})
            route = urlsplit(self.path).path
            if route not in {'/api/decision', '/api/recover'}:
                return self._json(404, {'error': 'unknown API route'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self._json(415, {'error': 'JSON body required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    return self._json(413, {'error': 'request must be between 1 and 65536 bytes'})
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('JSON object required')
                result = store.save(payload) if route == '/api/decision' else store.retry_recovery()
                self._json(200, {**result, 'token': token})
            except ReviewConflict as error:
                self._json(409, {'error': str(error)})
            except (ValueError, TypeError, KeyError) as error:
                self._json(400, {'error': str(error)})
            except (OSError, RuntimeError) as error:
                self._json(503, {'error': str(error)})

        def do_GET(self):
            self._get(False)

        def do_HEAD(self):
            self._get(True)

        def _get(self, head):
            if not self._host_ok():
                return self._json(403, {'error': 'invalid loopback Host'})
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/api/state':
                    return self._json(200, self._snapshot())
                if parsed.path == '/api/source-details':
                    key = parse_qs(parsed.query).get('source_key', [''])[0]
                    return self._json(200, store.source_details(key))
                match = re.fullmatch(r'/api/source-tracks/([0-9a-f]{16})\.json', parsed.path)
                if match:
                    return self._json(200, store.source_track_boxes(match[1]))
                if parsed.path == '/api/dataset-manifest':
                    return self._json(200, store.dataset_manifest())
                match = re.fullmatch(r'/api/confirmed/([0-9a-f]{24})/clip\.mp4', parsed.path)
                if match:
                    return self._file(store.confirmed_video(match[1]), head)
                match = re.fullmatch(r'/api/confirmed/([0-9a-f]{24})/tracks\.json', parsed.path)
                if match:
                    return self._file(store.confirmed_tracks(match[1]), head)
                if parsed.path.startswith('/api/'):
                    return self._json(404, {'error': 'unknown API route'})
                relative = unquote(parsed.path).lstrip('/') or 'index.html'
                path = safe_path(store.package / relative, [store.package])
                # Backups, state, journals and editable sources are never static-web roots.
                if (relative not in {'index.html', 'rearing_review.html', 'together_review.html',
                                     'approach_contact_review.html',
                                     'approach_no_contact_review.html',
                                     'approach_no_contact_recrop_review.html',
                                     'huddling_recheck_review.html',
                                     'contact_reclassification_review.html'}
                        and relative.split('/')[0] not in {'dataset', 'review_sources'}):
                    return self._json(404, {'error': 'unknown static resource'})
                return self._file(path, head)
            except (FileNotFoundError, KeyError):
                self._json(404, {'error': 'resource does not exist'})
            except ValueError as error:
                self._json(400, {'error': str(error)})
            except (OSError, RuntimeError) as error:
                self._json(503, {'error': str(error)})

        def _file(self, path: Path, head):
            # Open first: a concurrent approval may relocate a managed sample.
            with path.open('rb') as stream:
                size = path.stat().st_size
                start, end, code = 0, size-1, 200
                range_header = self.headers.get('Range')
                if range_header:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
                    if not match or not any(match.groups()):
                        return self._range_error(size)
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), size-1) if right else size-1
                    else:
                        start = max(0, size-int(right))
                    if not 0 <= start <= end < size:
                        return self._range_error(size)
                    code = 206
                self.send_response(code)
                self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(end-start+1))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if code == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                if head:
                    return
                stream.seek(start)
                remaining = end-start+1
                try:
                    while remaining:
                        block = stream.read(min(256*1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # Browser stopped/seeked playback.

        def _range_error(self, size):
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server
