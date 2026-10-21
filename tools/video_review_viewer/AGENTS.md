# Prototype Instructions

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.

## Viewer-specific decisions

- Mirror the supplied dark episode-review reference: left-side success/error clip list, center video playback, and result details beside the video.
- Keep both target-ID hit and strict Top-1 visible; never label a clip as successful when the strict behavior result is wrong.
- Read only the frozen render manifests and validation prediction export. The viewer must not rerun inference or expose validation labels as runtime inputs.
- Keep the review shell and filters usable when a review manifest or video resource is unavailable. Detect non-JSON SPA fallbacks and malformed manifests, show an actionable in-page error, and isolate media failures to the selected clip.
- Keep normal review browsing independent of inference or business APIs: ship frozen review data as static JSON and serve rendered clips as static/range-readable media. A single missing clip must not prevent the page, filters, metrics, or other clips from opening.
- The default route may fall back only to a known completed, frozen review manifest. Never silently point the viewer at partial/incomplete validation output or present fallback data as a new result; test the default URL and missing-manifest state.
