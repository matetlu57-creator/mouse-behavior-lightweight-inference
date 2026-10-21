# Validation video review

This historical viewer reads explicitly selected frozen review JSON and rendered MP4s.
It does not call an inference API or modify datasets, manual decisions, or A1/A2.
Private manifests and videos are not included in Git. A build without an explicit
manifest contains only the application shell and reports missing review data in-page.
When a JSON file or video is unavailable, the page keeps its navigation and
shows an in-page error for the affected data or clip.

## Start locally

From this directory, install the locked dependencies once and start Vite:

```powershell
npm ci --ignore-scripts
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://localhost:5173/`. Local development reads `public/review-data.json`;
explicit historical `?version=v7` through `v11` selections read the corresponding
`public/review-data-v*.json`. No missing version silently falls back to another dataset.
The Vite middleware serves local rendered videos with byte-range support so
the browser can seek without downloading an entire clip.

## Frozen production package

Production builds do not automatically copy `public/` or arbitrary `outputs/` files.
Set `REVIEW_MANIFEST` to an explicit completed manifest, `REVIEW_MANIFEST_SHA256`
to its verified SHA-256, and `REVIEW_OUTPUT_ROOT` to the local video root, then run:

```powershell
npm run build
npm run test:sites
npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
```

The manifest must declare `completed: true`, contain nonempty validated items and
strict behavior-plus-ID results, and use `/video-files/...mp4` URLs. The build
copies every referenced local video to that URL under `dist/client`, checks path
containment and regular-file status, and writes `review-package-receipt.json`
with manifest and media hashes. Missing media or mismatched hashes fail the build.
The preview and Sites worker preserve static byte-range/HEAD requests. Uploading
or publishing the resulting private media requires the user's explicit destination.

Only strict results control correct/error browsing. Historical diagnostic columns
are read-only old measurements, may include retired credit, and do not count as
current correct behavior or update A1/A2. Empty/malformed JSON, HTML fallbacks,
timeouts and missing videos remain recoverable errors within the application shell.
