# -*- coding: utf-8 -*-
"""Optional FFmpeg/NVIDIA NVENC writer with deterministic OpenCV fallback."""

from __future__ import annotations
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional
import cv2
import numpy as np


@lru_cache(maxsize=1)
def ffmpeg_nvenc_available() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    if proc.returncode != 0 or "h264_nvenc" not in (proc.stdout or ""):
        return False
    # Encoder presence alone does not prove that the NVIDIA driver/device can
    # actually open NVENC.  Probe one tiny frame once so runtime failures fall
    # back to OpenCV before the real output stream starts.
    try:
        probe = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return probe.returncode == 0


class NVENCWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        cfg: Optional[Mapping[str, Any]] = None,
    ):
        cfg = dict(cfg or {})
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not ffmpeg_nvenc_available():
            raise RuntimeError("ffmpeg h264_nvenc is unavailable")
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.closed = False
        codec = str(cfg.get("codec", "h264_nvenc"))
        preset = str(cfg.get("preset", "p4"))
        cq = str(int(cfg.get("cq", 23)))
        cmd = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{float(fps):.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            codec,
            "-preset",
            preset,
            "-rc",
            "vbr",
            "-cq",
            cq,
            "-pix_fmt",
            "yuv420p",
            self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def isOpened(self) -> bool:
        return not self.closed and self.proc.poll() is None and self.proc.stdin is not None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError("NVENC writer is not open")
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape[:2] != (self.height, self.width) or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"unexpected frame shape {arr.shape}; expected {(self.height, self.width, 3)}"
            )
        try:
            self.proc.stdin.write(np.ascontiguousarray(arr).tobytes())
        except BrokenPipeError as exc:
            err = (
                self.proc.stderr.read().decode("utf-8", errors="replace")
                if self.proc.stderr
                else ""
            )
            raise RuntimeError(f"NVENC ffmpeg pipe failed: {err[-2000:]}") from exc

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        returncode = self.proc.wait()
        err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
        if returncode != 0:
            raise RuntimeError(f"NVENC ffmpeg exited with code {returncode}: {err[-2000:]}")


class OpenCVWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int, fourcc: str = "mp4v"):
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (int(width), int(height))
        )

    def isOpened(self) -> bool:
        return bool(self.writer.isOpened())

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


def create_video_writer(
    path: str | Path, fps: float, width: int, height: int, cfg: Optional[Mapping[str, Any]] = None
):
    cfg = dict(cfg or {})
    prefer_nvenc = bool(cfg.get("prefer_nvenc", True))
    if prefer_nvenc and ffmpeg_nvenc_available():
        try:
            return NVENCWriter(path, fps, width, height, cfg)
        except Exception:
            pass
    return OpenCVWriter(path, fps, width, height, str(cfg.get("fallback_fourcc", "mp4v")))
