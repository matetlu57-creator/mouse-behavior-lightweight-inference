"""Create browser-compatible review copies of the existing rendered clips.

OpenCV's default FMP4 writer is useful for analysis but is not consistently
playable in browser video elements.  The source clips remain untouched; this
script writes compact H.264 copies under the generated viewer-media folder.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "exported_dataset_calibration"
MANIFEST_DIRS = (
    OUTPUT_ROOT / "rendered_validation_with_behavior_v5",
    OUTPUT_ROOT / "rendered_validation_errors_with_behavior_v5",
    OUTPUT_ROOT / "rendered_validation_with_behavior_v6",
    OUTPUT_ROOT / "rendered_validation_errors_with_behavior_v6",
)


def iter_source_paths(manifest_paths: list[Path]):
    seen: set[Path] = set()
    for manifest in manifest_paths:
        if not manifest.exists():
            continue
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            raw = Path(str(item.get("output_video", "")))
            source = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
            if source.exists() and source not in seen:
                seen.add(source)
                yield source


def iter_video_paths(video_dir: Path):
    """Yield rendered MP4s directly when no render manifest was retained."""
    for source in sorted(video_dir.rglob("*.mp4")):
        if source.is_file():
            yield source.resolve()


def convert(source: Path, output_root: Path) -> Path:
    output_root = output_root.resolve()
    relative = source.relative_to(output_root.resolve())
    media_root = output_root / "video_viewer_media"
    target = (media_root / relative).with_suffix(".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        r"scale=w=min(1280\,iw):h=-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="Render manifest to convert; may be repeated. Defaults to the legacy review manifests.",
    )
    source_group.add_argument(
        "--videos-dir",
        type=Path,
        help="Convert every MP4 under this directory when a render manifest is unavailable.",
    )
    args = parser.parse_args()
    if args.videos_dir:
        sources = list(iter_video_paths(args.videos_dir))
    else:
        manifest_paths = args.manifest or [directory / "render_manifest.json" for directory in MANIFEST_DIRS]
        sources = list(iter_source_paths(manifest_paths))
    converted = []
    for index, source in enumerate(sources, start=1):
        target = convert(source, args.output_root)
        converted.append(target)
        print(f"[{index}/{len(sources)}] {target.relative_to(REPO_ROOT)}")
    print(f"prepared={len(converted)} media_root={args.output_root / 'video_viewer_media'}")


if __name__ == "__main__":
    main()
