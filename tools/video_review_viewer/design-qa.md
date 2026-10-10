# Video Review Viewer — Design QA

## Source visual truth

- Source screenshot: `C:/Users/28361/AppData/Local/Temp/codex-clipboard-74f43e60-149f-4625-b8c4-254efa06dc11.png`
- Reference state: dark episode-review shell with a left clip list, success/error filtering, a central media player, and a result/config panel.

## Implementation evidence

- Local implementation: `http://localhost:4173/` (Codex in-app browser, tab 2).
- Browser-rendered screenshot: captured from tab 2 during this QA pass; the CUA screenshot surface does not expose a persistent filesystem path.
- QA viewport: 872 × 626 CSS px (the browser surface used for verification); the implementation also has a wide desktop layout and a responsive breakpoint below 1000 px.
- Source pixels: 1714 × 1317. The source is a full browser screenshot, so browser chrome was excluded from the comparison; no density normalization was needed for the structural comparison.
- Final state: 错误筛选，选中 `Together_large_arena_2_part001_002133-002338`; the video is playable and the detail panel shows 标签行为、识别行为、目标 ID 命中、严格 Top‑1 and the mismatch reason.

## Comparison findings and fixes

1. Initial implementation had a functional but non-playable media state because the rendered clips were FMP4. The browser showed “无法播放媒体”. Fixed by creating H.264/yuv420p viewer copies under `outputs/exported_dataset_calibration/video_viewer_media/`; original render files are untouched.
2. Initial narrow browser capture overflowed horizontally because the desktop shell enforced a 1100 px minimum. Added responsive breakpoints: the list remains on the left, the video stays readable, and the result panel moves below the player on narrow viewports.
3. The one-per-behavior example manifest contains a few non-strict examples. The data builder now classifies by strict Top‑1 and deduplicates samples in favor of the complete error export, so a wrong clip is never shown as PASS.

## Required fidelity surfaces

- Fonts/typography: system UI fallback with compact uppercase labels mirrors the reference hierarchy; Chinese labels remain readable at the verified viewport.
- Spacing/layout rhythm: top bar, left navigation, active red tab, rounded episode cards, player frame, and detail card follow the reference proportions; narrow layout was checked after the overflow fix.
- Colors/tokens: navy/purple shell, pink active state, green PASS, and red ERR are mapped to semantic states visible in both list and detail panels.
- Image/video fidelity: real rendered behavior videos are used. H.264 viewer copies preserve the behavior overlays and are generated from the existing clips; no placeholder media is used.
- Copy/content: both 标签行为 and 识别行为 are shown, plus target-ID hit and strict Top‑1. Error clips explicitly state the predicted behavior and label mismatch.

## Interactions checked

- Correct / error / all tabs update the list and current selection.
- Behavior dropdown and search field are wired to the list.
- Selecting an episode updates the video and detail panel.
- Native video controls work; playback speed select supports 0.5×, 1×, 1.5×, and 2×.
- Keyboard navigation is available with Arrow Up/Down or J/K; Space toggles play/pause.
- Browser console check: no error or warning entries in the final state.

## Final result

passed
