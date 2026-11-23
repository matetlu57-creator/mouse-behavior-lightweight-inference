# Repository instructions

- Keep visual tracking outside behavior rules. Ultralytics `YOLO.track()` owns `visual_track_id`.
- Treat mTrack/RFID mappings as the only source of stable `animal_id`; never infer a missing identity from position, color, brightness, pose, or motion.
- Preserve the tracked-cache and normalized identity-CSV contracts documented in `docs/index.md`.
- Add regression tests for changes to trajectory loading, identity resolution, or behavior labels.
- In every behavior validation Top-1 report, show per-behavior and overall strict accuracy (target-ID joint). Same-layer dual-label credit was removed on 2026-10-04 (Step 3): Together now allows contact without interaction, so Together-vs-contact compatibility no longer exists and historical dual-label numbers are old-era records only. Never treat the two contact labels as interchangeable.
- Run `python scripts/run_quality.py --ci` before proposing a release or remote delivery.
- Do not commit videos, caches, datasets, generated outputs, secrets, or model files outside the registered Git LFS path.

## Website availability and API resilience

- Every new or updated website must render its application shell and core navigation without requiring an API or external service to succeed.
- Load API-backed content asynchronously. Network errors, timeouts, HTTP errors, invalid JSON/schema, and empty responses must produce a recoverable in-page state, not a blank page, fatal render, or blocked navigation.
- Validate response status, content type, and schema before consuming API data; do not mistake an SPA `index.html` fallback for JSON.
- Provide a clearly labeled local/demo fallback when appropriate. Never present fallback or stale data as current scientific results, and never silently substitute an incomplete run for a completed dataset.
- Add regression or smoke tests for unavailable and malformed APIs, and verify the default local URL after build/startup. Document one reproducible command for starting the site and the expected URL/port.
- Every future **review website** (`category: review`) and **error-display website** (`category: error_display`) must be registered and deployed in the Persistent Local Site Hub (`scripts/manage_site_hub.py`, default portal `http://127.0.0.1:8788/`) as a gateway-independent detached service — never leave it running only as an ephemeral background task bound to an agent session or gateway.
- Standard site deployment workflow: assign a dedicated loopback port (`127.0.0.1`), `/api/health` endpoint, and non-conflicting writable output directory; register and deploy via `D:\Anaconda3\python.exe scripts/manage_site_hub.py register ...` followed by `start` / `status`, and verify both detached execution and HTTP 200 JSON health on the Hub (`8788`) and target site.
- Keep standalone `scripts/serve_*.py` entry points intact for debugging and tests while treating `scripts/manage_site_hub.py` as the canonical deployment path.
