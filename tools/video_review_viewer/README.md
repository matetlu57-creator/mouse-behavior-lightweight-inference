# Validation video review

This viewer reads frozen review JSON from `public/` and streams rendered MP4s
from the repository `outputs/` directory. It does not call an inference API.
When a JSON file or video is unavailable, the page keeps its navigation and
shows an in-page error for the affected data or clip.

## Start locally

From this directory, install the locked dependencies once and start Vite:

```powershell
pnpm install --frozen-lockfile
pnpm run dev -- --host 0.0.0.0 --port 5173 --strictPort
```

Open `http://localhost:5173/?version=v11` for the compatible validation review.
The Vite middleware serves local rendered videos with byte-range support so
the browser can seek without downloading an entire clip.
