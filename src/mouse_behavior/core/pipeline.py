"""Thin orchestration API for the current lightweight analysis path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineRun:
    """Inputs and output directory for one reproducible analysis run."""

    video: Path
    yolo_cache: Path
    config: Path
    output: Path


class LightweightPipeline:
    """Coordinate the cache-based lightweight analyzer.

    The implementation is imported lazily so importing the orchestration API
    does not initialize OpenCV, SciPy, or any model runtime.
    """

    def __init__(
        self,
        config: str | Path,
        *,
        expected_mice: int = 20,
        sample_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.config = Path(config)
        self.expected_mice = max(int(expected_mice), 2)
        self.sample_stride = max(int(sample_stride), 1)
        self.max_frames = max_frames

    def run(
        self,
        video: str | Path,
        yolo_cache: str | Path,
        output: str | Path,
        *,
        fps: float | None = None,
    ) -> PipelineRun:
        """Run analysis and return a typed record of the produced location."""

        from ..lightweight_behavior_inference import analyze

        video_path = Path(video)
        cache_path = Path(yolo_cache)
        output_path = Path(output)
        analyze(
            video_path,
            cache_path,
            self.config,
            output_path,
            expected_mice=self.expected_mice,
            max_frames=self.max_frames,
            sample_stride=self.sample_stride,
            fps_override=fps,
        )
        return PipelineRun(video_path, cache_path, self.config, output_path)
