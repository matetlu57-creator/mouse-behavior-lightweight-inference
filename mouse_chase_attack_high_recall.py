#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多鼠追逐/攻击高召回启发式粗筛器（YOLO Pose）

功能：
1. 读取Ultralytics兼容的自定义7关键点Pose模型；
2. 支持固定比例、身体长度估算、场地四角单应性三种厘米坐标转换；
3. 同时输出弱候选（高召回）和强候选（较高精度）；
4. 输出四类标签：非追逐非攻击、非攻击性追逐、非追逐攻击、攻击性追逐；
5. 按弱候选自动裁剪固定5秒、最小间隔5秒的待人工复核视频片段；
6. 可与人工标注CSV/XLSX比较，计算事件召回率、候选覆盖率、误报/小时和复核时长比例；
7. 自动导出漏检片段和弱候选误报片段。

固定关键点顺序：
    nose, left ear, right ear, base of neck, left hip, right hip, base of tail

本程序定位为“高召回粗筛”，启发式输出不能直接替代人工真值。

版本：v1.43.0（Standard Behavior Engine：连续证据 + 角色推断 + 时序FSM + 独立Ethogram；追踪/Identity保持v1.42.1路径）
- 白鼠亮斑检测通道（顶帽变换+对比度判别，无需训练模型）解决检测模型
  不认识白鼠的问题；静止反光假斑由静态拒绝+运动门控晋升双重拦截。
- Pose模型回归全图推理（按训练方式使用），7关键点匹配挂回候选框；
  Pose找到而检测器漏掉的鼠保留为独立检测。
- v1.11.0：MemoryIdentityAssigner 短时身份记忆、检测与身份分离、
  未匹配检测即时TMP、冲突冻结身份但保留检测框（修复文档v1.1）。
"""

from __future__ import annotations

import argparse
import copy
import copyreg
import csv
import concurrent.futures
import functools
import gzip
import gc
import hashlib
import sqlite3
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from itertools import combinations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Deque, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment as _hungarian_assignment
except Exception:
    _hungarian_assignment = None

import mouse_chase_attack_extractor_base as base
import pose_quality_recovery as pose_recovery
import mask_cluster_reid as mask_reid
import adaptive_arena_boundary as arena_boundary
import mask_trigger_controller
import nvenc_video_writer
import standard_behavior_engine


_REQUIRED_BASE_API = (
    "Detection", "MouseObservation", "ObservationHistory", "derive_geometry",
    "KeypointSmoother", "ScaleEstimator", "StableIdentityAssigner",
    "PairContactTracker", "parse_yolo_result", "temporal_filter",
    "suppress_duplicate_detections", "OcclusionClusterManager",
    # v1.11.0 新增：短时身份记忆分配器与记忆模块（修复文档v1.1 §12）。
    "MemoryIdentityAssigner", "TemporaryIdentityMemory",
    "KeypointMotionIdentityAssigner",
)
_missing_base_api = [name for name in _REQUIRED_BASE_API if not hasattr(base, name)]
if _missing_base_api:
    raise RuntimeError(
        "主程序与 mouse_chase_attack_extractor_base.py 版本不一致，缺少接口："
        + ", ".join(_missing_base_api)
        + f"。当前加载的底层文件：{getattr(base, '__file__', 'unknown')}。"
        + "请同时替换完整代码包中的主程序和底层模块。"
    )


KEYPOINT_NAMES = base.KEYPOINT_NAMES
KP = base.KP
LABELS = base.LABELS
CLIP_DIRS = base.CLIP_DIRS
VIDEO_EXTENSIONS = base.VIDEO_EXTENSIONS

# v1.39：DISK启发的因果序列一致性、轨迹缺口局部恢复、标注片段和资源计时。
PROGRAM_VERSION = "1.43.0-standard-behavior-engine"
# Server layout: code is uploaded to /NVme1/zhaojun/projects and data lives
# beside it in datasets/, outputs/ and checkpoints/.  The environment variable
# keeps the package relocatable while retaining the requested data-disk default.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("MOUSE_BEHAVIOR_DATA_ROOT", "/NVme1/zhaojun"))
DEFAULT_CONFIG = PROJECT_DIR / "mouse_chase_attack_config.yaml"
DEFAULT_MODEL = Path(
    os.environ.get(
        "MOUSE_BEHAVIOR_MODEL",
        str(DATA_ROOT / "checkpoints" / "best.pt"),
    )
)
DEFAULT_VIDEO = Path(
    os.environ.get(
        "MOUSE_BEHAVIOR_VIDEO",
        str(DATA_ROOT / "datasets" / "3_20mice_w_tag_rgb.mp4"),
    )
)
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "MOUSE_BEHAVIOR_OUTPUT",
        str(DATA_ROOT / "outputs"),
    )
)


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML配置格式错误：{path}")
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_success_artifacts(output_dir: Path, config: Mapping[str, Any]) -> List[str]:
    """成功完成后清理可重建缓存，异常中断时不调用本函数以支持断点续跑。"""
    # 只读取显式的成功清理开关，避免无意间删除用户保留的中间结果。
    performance_cfg = dict(config.get("performance", {}))
    if not bool(performance_cfg.get("cleanup_caches_on_success", True)):
        return []
    # 仅允许删除当前视频结果目录中的两个已知缓存目录。
    cache_names = ("stage3_observation_cache", "yolo_precompute")
    removed: List[str] = []
    for cache_name in cache_names:
        cache_path = output_dir / cache_name
        if not cache_path.exists():
            continue
        try:
            # 缓存目录只包含可由视频和权重重新生成的分块文件。
            if cache_path.is_dir():
                shutil.rmtree(cache_path)
            else:
                cache_path.unlink()
            removed.append(cache_name)
        except OSError:
            # 清理失败不应覆盖已经成功生成的行为结果；保留日志供用户手动处理。
            logging.exception("成功结果已生成，但清理缓存失败：%s", cache_path)
    # 关闭行为标签视频后，顺便删除旧版本遗留的同名视频，避免输出目录保留两个视频。
    output_cfg = dict(config.get("output", {}))
    if not bool(output_cfg.get("save_behavior_label_video", False)):
        behavior_video_path = output_dir / "追踪与行为标签视频.mp4"
        if behavior_video_path.exists():
            try:
                behavior_video_path.unlink()
                removed.append(behavior_video_path.name)
            except OSError:
                logging.exception("成功结果已生成，但清理旧行为标签视频失败：%s", behavior_video_path)
    return removed


def attach_persistent_log(output_root: Path, verbose: bool = False) -> Path:
    """把根日志器同时写入输出盘；重复调用时不会叠加相同文件处理器。"""
    log_path = ensure_dir(output_root) / "运行日志.log"
    resolved_log_path = str(log_path.resolve())
    root_logger = logging.getLogger()
    for existing in root_logger.handlers:
        if getattr(existing, "_mouse_behavior_log_path", None) == resolved_log_path:
            return log_path
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler._mouse_behavior_log_path = resolved_log_path  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    return log_path


class RuntimeProfiler:
    """Collect lightweight stage timings without changing inference decisions.

    The profiler is disabled by default.  When enabled, timings are accumulated
    in a small in-memory window and emitted every ``report_interval_frames``.
    The lock is also used by the optional asynchronous renderer, so render and
    encode timings can safely be added from its worker thread.
    """

    STAGES = (
        "decode_stream",
        "decode",
        "full_frame_inference",
        "yolo_result_transfer",
        "result_parse",
        "candidate_cpu",
        "candidate_filter",
        "dedup",
        "roi_inference",
        "pose_recovery_cpu",
        "mask",
        "cluster_context",
        "identity_cost",
        "identity_assignment",
        "identity",
        "observation_repair",
        "behavior_pair",
        "csv_sqlite",
        "behavior_io",
        "render",
        "encode",
    )

    def __init__(self, config: Mapping[str, Any]) -> None:
        raw = config.get("profiling", False)
        if isinstance(raw, Mapping):
            self.enabled = bool(raw.get("enabled", True))
            interval = raw.get("report_interval_frames", 100)
        else:
            self.enabled = bool(raw)
            interval = 100
        self.interval_frames = max(int(interval), 1)
        self._lock = threading.Lock()
        self._window_frames = 0
        self._total_frames = 0
        self._window_started = time.perf_counter()
        self._total_started = self._window_started
        self._window_seconds: Dict[str, float] = {stage: 0.0 for stage in self.STAGES}
        self._total_seconds: Dict[str, float] = {stage: 0.0 for stage in self.STAGES}

    def add(self, stage: str, seconds: float) -> None:
        """Add one measured interval; unknown stages are intentionally ignored."""
        if not self.enabled or stage not in self._window_seconds:
            return
        value = float(seconds)
        if not np.isfinite(value) or value < 0.0:
            return
        with self._lock:
            self._window_seconds[stage] += value
            self._total_seconds[stage] += value

    def add_result_speed(self, result: Any) -> None:
        """Record Ultralytics' inference timing when a result exposes ``speed``."""
        if not self.enabled:
            return
        speed = getattr(result, "speed", None)
        if not isinstance(speed, Mapping):
            return
        # Ultralytics reports milliseconds.  Keep this separate from the
        # stream-wait measurement because the latter also includes decoding.
        value = speed.get("inference", None)
        try:
            if value is not None:
                self.add("full_frame_inference", float(value) / 1000.0)
        except (TypeError, ValueError):
            return

    def frame_done(self, frame_idx: int) -> None:
        """Count one completed frame and emit a bounded rolling report."""
        if not self.enabled:
            return
        should_report = False
        with self._lock:
            self._window_frames += 1
            self._total_frames += 1
            should_report = self._window_frames >= self.interval_frames
        if should_report:
            self.report(frame_idx)

    def report(self, frame_idx: int, force: bool = False) -> None:
        """Log the current window and reset it; ``force`` flushes short windows."""
        if not self.enabled:
            return
        with self._lock:
            if self._window_frames <= 0 or (not force and self._window_frames < self.interval_frames):
                return
            frames = self._window_frames
            elapsed = max(time.perf_counter() - self._window_started, 1e-9)
            values = dict(self._window_seconds)
            start_frame = int(frame_idx) - frames + 1
            self._window_frames = 0
            self._window_started = time.perf_counter()
            self._window_seconds = {stage: 0.0 for stage in self.STAGES}
        avg_ms = {stage: 1000.0 * value / frames for stage, value in values.items()}
        throughput = frames / elapsed
        logging.info(
            "PROFILE frames %d-%d | %.2f frame/s | decode=%.1f | yolo=%.1f | "
            "transfer=%.1f | parse=%.1f | candidate=%.1f | filter=%.1f | dedup=%.1f | "
            "roi_gpu=%.1f | pose_cpu=%.1f | mask=%.1f | cluster=%.1f | "
            "id_cost=%.1f | id_assign=%.1f | id_total=%.1f | obs=%.1f | "
            "behavior_pair=%.1f | csv_sqlite=%.1f | render=%.1f | encode=%.1f ms",
            start_frame, int(frame_idx), throughput,
            avg_ms["decode"], avg_ms["full_frame_inference"],
            avg_ms["yolo_result_transfer"], avg_ms["result_parse"],
            avg_ms["candidate_cpu"], avg_ms["candidate_filter"], avg_ms["dedup"],
            avg_ms["roi_inference"], avg_ms["pose_recovery_cpu"], avg_ms["mask"],
            avg_ms["cluster_context"], avg_ms["identity_cost"],
            avg_ms["identity_assignment"], avg_ms["identity"],
            avg_ms["observation_repair"], avg_ms["behavior_pair"],
            avg_ms["csv_sqlite"], avg_ms["render"], avg_ms["encode"],
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return totals suitable for run metadata and post-run auditing."""
        if not self.enabled:
            return {"enabled": False}
        with self._lock:
            total_frames = int(self._total_frames)
            total_seconds = max(time.perf_counter() - self._total_started, 1e-9)
            seconds = dict(self._total_seconds)
        return {
            "enabled": True,
            "frames": total_frames,
            "elapsed_seconds": float(total_seconds),
            "average_frame_ms": {
                stage: float(1000.0 * value / max(total_frames, 1))
                for stage, value in seconds.items()
            },
            "stage_seconds": {stage: float(value) for stage, value in seconds.items()},
        }


class ProcessResourceMonitor:
    """Sample this run's CPU core-seconds and GPU activity in the background.

    CPU time is accumulated for the main process and observed child workers.
    GPU busy-equivalent seconds integrate ``nvidia-smi`` utilization and are
    therefore an auditable utilization estimate rather than CUDA kernel time.
    The profiler's inference-stage duration is added to the final report as a
    second, explicitly labelled proxy.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = dict(config.get("resource_monitor", {}))
        self.enabled = bool(cfg.get("enabled", True))
        self.interval = max(float(cfg.get("sample_interval_seconds", 2.0)), 0.5)
        self.gpu_index = max(int(cfg.get("gpu_index", 0)), 0)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_wall = 0.0
        self._last_gpu_wall = 0.0
        self._process_totals: Dict[Tuple[int, float], float] = {}
        self._root_key: Optional[Tuple[int, float]] = None
        self._cpu_core_seconds = 0.0
        self._last_cpu_sample_wall = 0.0
        self._cpu_samples: List[float] = []
        self._cpu_peak_percent = 0.0
        self._gpu_samples: List[float] = []
        self._gpu_memory_samples_mb: List[float] = []
        self._gpu_power_samples_w: List[float] = []
        self._gpu_busy_equivalent_seconds = 0.0
        self._gpu_command = shutil.which("nvidia-smi")
        if self._gpu_command is None:
            windows_candidate = Path(r"C:\Windows\System32\nvidia-smi.exe")
            if windows_candidate.exists():
                self._gpu_command = str(windows_candidate)
        self._psutil: Any = None
        self._process: Any = None
        self._errors: List[str] = []

    @staticmethod
    def _cpu_total(process: Any) -> float:
        times = process.cpu_times()
        return float(times.user + times.system)

    def _sample_cpu(self, *, initialize: bool = False) -> None:
        if self._process is None:
            return
        now = time.perf_counter()
        interval_cpu_seconds = 0.0
        try:
            processes = [self._process] + self._process.children(recursive=True)
        except Exception as exc:
            if len(self._errors) < 8:
                self._errors.append(f"cpu_children:{type(exc).__name__}:{exc}")
            processes = [self._process]
        for process in processes:
            try:
                key = (int(process.pid), float(process.create_time()))
                total = self._cpu_total(process)
            except Exception:
                continue
            previous = self._process_totals.get(key)
            if previous is None:
                # The root existed before monitoring, so its first value is a
                # baseline.  Newly observed children were created by this run;
                # their accumulated time since creation belongs to the run.
                if not initialize and key != self._root_key:
                    increment = max(total, 0.0)
                    self._cpu_core_seconds += increment
                    interval_cpu_seconds += increment
                self._process_totals[key] = total
            else:
                increment = max(total - previous, 0.0)
                self._cpu_core_seconds += increment
                interval_cpu_seconds += increment
                self._process_totals[key] = total
        if not initialize and self._last_cpu_sample_wall > 0.0:
            elapsed = max(now - self._last_cpu_sample_wall, 1.0e-9)
            percent = 100.0 * interval_cpu_seconds / elapsed
            if np.isfinite(percent):
                self._cpu_samples.append(float(percent))
                self._cpu_peak_percent = max(self._cpu_peak_percent, float(percent))
        self._last_cpu_sample_wall = now

    def _sample_gpu(self, *, initialize: bool = False) -> None:
        now = time.perf_counter()
        if not self._gpu_command:
            self._last_gpu_wall = now
            return
        try:
            completed = subprocess.run(
                [
                    self._gpu_command,
                    f"--id={self.gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=max(5.0, self.interval),
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            line = completed.stdout.strip().splitlines()[0]
            fields = [value.strip() for value in line.split(",")]
            utilization = float(fields[0])
            memory_mb = float(fields[1])
            power_w = float(fields[2]) if fields[2].upper() != "N/A" else float("nan")
            if self._last_gpu_wall > 0.0 and not initialize:
                elapsed = max(now - self._last_gpu_wall, 0.0)
                self._gpu_busy_equivalent_seconds += (
                    float(np.clip(utilization, 0.0, 100.0)) / 100.0 * elapsed
                )
            self._gpu_samples.append(utilization)
            self._gpu_memory_samples_mb.append(memory_mb)
            if np.isfinite(power_w):
                self._gpu_power_samples_w.append(power_w)
        except Exception as exc:
            if len(self._errors) < 8:
                self._errors.append(f"gpu_sample:{type(exc).__name__}:{exc}")
        finally:
            self._last_gpu_wall = now

    def _sample(self, *, initialize: bool = False) -> None:
        self._sample_cpu(initialize=initialize)
        self._sample_gpu(initialize=initialize)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._sample()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._started_wall = time.perf_counter()
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._root_key = (
                int(self._process.pid),
                float(self._process.create_time()),
            )
            self._process.cpu_percent(interval=None)
        except Exception as exc:
            self._process = None
            self._errors.append(f"psutil:{type(exc).__name__}:{exc}")
        self._sample(initialize=True)
        self._thread = threading.Thread(
            target=self._run,
            name="mouse-behavior-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 6.0)
        self._sample()
        elapsed = max(time.perf_counter() - self._started_wall, 1.0e-9)
        cpu_average_percent = (
            float(np.mean(self._cpu_samples)) if self._cpu_samples else 0.0
        )
        gpu_average = float(np.mean(self._gpu_samples)) if self._gpu_samples else 0.0
        return {
            "enabled": True,
            "wall_seconds": float(elapsed),
            "cpu": {
                "process_tree_core_seconds": float(self._cpu_core_seconds),
                "average_cores_used": float(self._cpu_core_seconds / elapsed),
                "sampled_process_tree_average_percent": cpu_average_percent,
                "sampled_process_tree_peak_percent": float(self._cpu_peak_percent),
                "logical_cpu_count": int(
                    self._psutil.cpu_count(logical=True) if self._psutil is not None else 0
                ),
                "sample_count": int(len(self._cpu_samples)),
            },
            "gpu": {
                "index": int(self.gpu_index),
                "utilization_busy_equivalent_seconds": float(
                    self._gpu_busy_equivalent_seconds
                ),
                "average_utilization_percent": gpu_average,
                "peak_utilization_percent": float(max(self._gpu_samples, default=0.0)),
                "average_memory_mb": float(np.mean(self._gpu_memory_samples_mb))
                if self._gpu_memory_samples_mb else 0.0,
                "peak_memory_mb": float(max(self._gpu_memory_samples_mb, default=0.0)),
                "average_power_w": float(np.mean(self._gpu_power_samples_w))
                if self._gpu_power_samples_w else 0.0,
                "sample_count": int(len(self._gpu_samples)),
                "measurement": "nvidia-smi utilization integral; not CUDA kernel time",
            },
            "sample_interval_seconds": float(self.interval),
            "errors": list(self._errors),
        }


class AsyncAnnotatedVideoWriter:
    """Preserve frame order while moving drawing and encoding off the main loop."""

    def __init__(
        self,
        writer: Any,
        max_queue_size: int,
        profiler: RuntimeProfiler,
    ) -> None:
        self.writer = writer
        self.profiler = profiler
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(int(max_queue_size), 1))
        self._sentinel = object()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._worker,
            name="mouse-behavior-render-writer",
            daemon=True,
        )
        self._thread.start()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("后台渲染/编码线程失败") from self._error

    def _worker(self) -> None:
        try:
            while True:
                job = self._queue.get()
                try:
                    if job is self._sentinel:
                        return
                    frame, visual_observations, active_raw_records, frame_idx, max_mice, visualization_cfg = job
                    render_started = time.perf_counter()
                    _draw_online_tracking_frame(
                        frame,
                        visual_observations,
                        active_raw_records,
                        frame_idx,
                        max_mice,
                        visualization_cfg,
                    )
                    self.profiler.add("render", time.perf_counter() - render_started)
                    encode_started = time.perf_counter()
                    self.writer.write(frame)
                    self.profiler.add("encode", time.perf_counter() - encode_started)
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # propagate to the producer at the next safe boundary
            self._error = exc

    def submit(
        self,
        frame: np.ndarray,
        visual_observations: Sequence[Any],
        active_raw_records: Sequence[Mapping[str, Any]],
        frame_idx: int,
        max_mice: int,
        visualization_cfg: Mapping[str, Any],
    ) -> None:
        """Queue a private frame copy; block only when the bounded queue is full."""
        job = (
            np.asarray(frame).copy(),
            visual_observations,
            active_raw_records,
            int(frame_idx),
            int(max_mice),
            visualization_cfg,
        )
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(job, timeout=0.25)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        """Flush all queued frames and surface worker failures to the caller."""
        if self._thread is None:
            return
        self._raise_if_failed()
        self._queue.put(self._sentinel)
        self._queue.join()
        self._thread.join()
        thread = self._thread
        self._thread = None
        self._raise_if_failed()
        del thread


class PrefetchedResultStream:
    """在后台生成下一帧检测结果，并在断点边界暂停以保持状态快照精确。"""

    def __init__(
        self,
        source: Iterable[Any],
        max_queue_size: int,
        start_frame: int,
        checkpoint_interval: int,
    ) -> None:
        self._source = iter(source)
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(
            maxsize=max(int(max_queue_size), 1)
        )
        self._stop_event = threading.Event()
        self._checkpoint_gate = threading.Event()
        self._next_frame = int(start_frame)
        self._checkpoint_interval = max(int(checkpoint_interval), 0)
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="mouse-behavior-inference-prefetch",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> "PrefetchedResultStream":
        return self

    def _put(self, kind: str, payload: Any) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put((kind, payload), timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    result = next(self._source)
                except StopIteration:
                    self._put("end", None)
                    return
                if not self._put("result", result):
                    return
                self._next_frame += 1
                if (
                    self._checkpoint_interval > 0
                    and self._next_frame % self._checkpoint_interval == 0
                ):
                    while not self._stop_event.is_set():
                        if self._checkpoint_gate.wait(timeout=0.25):
                            self._checkpoint_gate.clear()
                            break
        except BaseException as exc:
            self._put("error", exc)
        finally:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()

    def __next__(self) -> Any:
        kind, payload = self._queue.get()
        if kind == "result":
            return payload
        if kind == "error":
            raise payload
        raise StopIteration

    def release_checkpoint_barrier(self) -> None:
        """主线程提交完同一帧边界后，才允许生产者处理下一帧。"""
        self._checkpoint_gate.set()

    def close(self) -> None:
        """停止生产者并等待其释放视频和模型流引用。"""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._checkpoint_gate.set()
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            logging.warning("后台推理预取线程未在30秒内退出，将由进程结束时回收。")


class CheckpointSegmentedVideoWriter:
    """把长核查视频写成可独立播放的小段，并在最终成功后合并为原文件名。"""

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        width: int,
        height: int,
        async_render: bool,
        queue_size: int,
        profiler: RuntimeProfiler,
        resume_segments: Sequence[Mapping[str, Any]],
        resume_frame: int,
        fresh_run: bool,
        video_encoding_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.segment_dir = ensure_dir(output_dir / ".追踪视频断点分段")
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.async_render = bool(async_render)
        self.queue_size = max(int(queue_size), 1)
        self.profiler = profiler
        self.video_encoding_cfg = dict(video_encoding_cfg or {})
        self.segments = [dict(item) for item in resume_segments]
        self._writer: Optional[Any] = None
        self._async_writer: Optional[AsyncAnnotatedVideoWriter] = None
        self._segment_start = int(resume_frame)
        self._submitted_frames = 0
        committed_names = {str(item.get("name", "")) for item in self.segments}
        for path in self.segment_dir.glob("tracking_part_*.mp4"):
            if fresh_run or path.name not in committed_names:
                path.unlink(missing_ok=True)
        if not fresh_run:
            self._validate_committed_segments(int(resume_frame))

    def _validate_committed_segments(self, resume_frame: int) -> None:
        """恢复前确认所有已提交视频段连续、存在且帧数与清单一致。"""
        expected_start = 0
        for item in self.segments:
            start = int(item.get("start_frame", -1))
            end = int(item.get("end_frame", -1))
            expected_count = int(item.get("frame_count", -1))
            path = self.segment_dir / str(item.get("name", ""))
            if start != expected_start or end < start or expected_count != end - start + 1:
                raise ValueError(f"追踪视频断点分段清单不连续：{item}")
            if not path.exists() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"追踪视频断点分段缺失：{path}")
            capture = cv2.VideoCapture(str(path))
            actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else -1
            capture.release()
            if actual_count != expected_count:
                raise ValueError(f"追踪视频断点分段帧数不符：{path}，{actual_count}!={expected_count}")
            expected_start = end + 1
        if expected_start != int(resume_frame):
            raise ValueError(f"追踪视频分段只到{expected_start}，但断点要求从{resume_frame}恢复。")

    def _open_segment(self, start_frame: int) -> None:
        """惰性创建当前分段，空恢复任务不会产生零帧MP4。"""
        if self._writer is not None:
            return
        self._segment_start = int(start_frame)
        self._submitted_frames = 0
        path = self.segment_dir / f"tracking_part_{self._segment_start:09d}.mp4"
        path.unlink(missing_ok=True)
        writer = nvenc_video_writer.create_video_writer(
            path, self.fps, self.width, self.height, self.video_encoding_cfg
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建追踪视频断点分段：{path}")
        self._writer = writer
        if self.async_render:
            self._async_writer = AsyncAnnotatedVideoWriter(
                writer,
                max_queue_size=self.queue_size,
                profiler=self.profiler,
            )

    def submit(
        self,
        frame: np.ndarray,
        visual_observations: Sequence[Any],
        active_raw_records: Sequence[Mapping[str, Any]],
        frame_idx: int,
        max_mice: int,
        visualization_cfg: Mapping[str, Any],
    ) -> None:
        """按原渲染规则写入当前段；分段边界不改变画面内容或帧顺序。"""
        self._open_segment(int(frame_idx))
        if self._async_writer is not None:
            self._async_writer.submit(
                frame,
                visual_observations,
                active_raw_records,
                frame_idx,
                max_mice,
                visualization_cfg,
            )
        else:
            render_started = time.perf_counter()
            _draw_online_tracking_frame(
                frame,
                visual_observations,
                active_raw_records,
                frame_idx,
                max_mice,
                visualization_cfg,
            )
            self.profiler.add("render", time.perf_counter() - render_started)
            encode_started = time.perf_counter()
            self._writer.write(frame)
            self.profiler.add("encode", time.perf_counter() - encode_started)
        self._submitted_frames += 1

    def commit_segment(self, end_frame_exclusive: int) -> None:
        """封口当前MP4并把它加入已提交清单；没有新帧时不做任何事。"""
        if self._writer is None:
            return
        if self._async_writer is not None:
            self._async_writer.close()
            self._async_writer = None
        self._writer.release()
        self._writer = None
        expected_count = int(end_frame_exclusive) - int(self._segment_start)
        if expected_count != int(self._submitted_frames):
            raise RuntimeError(
                f"追踪视频断点分段计数不一致：提交{self._submitted_frames}帧，"
                f"边界要求{expected_count}帧。"
            )
        path = self.segment_dir / f"tracking_part_{self._segment_start:09d}.mp4"
        self.segments.append(
            {
                "name": path.name,
                "start_frame": int(self._segment_start),
                "end_frame": int(end_frame_exclusive) - 1,
                "frame_count": int(self._submitted_frames),
                "size_bytes": int(path.stat().st_size),
            }
        )
        self._segment_start = int(end_frame_exclusive)
        self._submitted_frames = 0

    def merge(self, output_path: Path, expected_frames: int) -> None:
        """顺序解码全部已提交分段并原子生成兼容旧流程的单个追踪视频。"""
        if int(expected_frames) <= 0:
            return
        self._validate_committed_segments(int(expected_frames))
        temporary = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.写入中.mp4")
        writer = nvenc_video_writer.create_video_writer(
            temporary, self.fps, self.width, self.height, self.video_encoding_cfg
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建合并后的追踪视频：{temporary}")
        written = 0
        try:
            for item in self.segments:
                capture = cv2.VideoCapture(str(self.segment_dir / str(item["name"])))
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        writer.write(frame)
                        written += 1
                finally:
                    capture.release()
        finally:
            writer.release()
        if written != int(expected_frames):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"合并追踪视频帧数不符：{written}!={expected_frames}")
        os.replace(temporary, output_path)

    def abort_uncommitted_segment(self) -> None:
        """异常清理只丢弃尚未写入断点清单的当前段，已提交分段保持不动。"""
        if self._writer is None:
            return
        try:
            if self._async_writer is not None:
                self._async_writer.close()
                self._async_writer = None
        finally:
            self._writer.release()
            self._writer = None
        path = self.segment_dir / f"tracking_part_{self._segment_start:09d}.mp4"
        path.unlink(missing_ok=True)
        self._submitted_frames = 0

    def cleanup(self) -> None:
        """最终结果完成后只删除本模块创建的已提交分段。"""
        for item in self.segments:
            (self.segment_dir / str(item.get("name", ""))).unlink(missing_ok=True)
        try:
            self.segment_dir.rmdir()
        except OSError:
            pass


def _iter_profiled_results(result_stream: Iterable[Any], profiler: RuntimeProfiler) -> Iterable[Any]:
    """Measure the time spent waiting for decode plus the model stream result."""
    iterator = iter(result_stream)
    try:
        while True:
            started = time.perf_counter()
            try:
                result = next(iterator)
            except StopIteration:
                return
            profiler.add("decode_stream", time.perf_counter() - started)
            profiler.add_result_speed(result)
            hybrid_meta = getattr(result, "hybrid_meta", {})
            if isinstance(hybrid_meta, Mapping):
                for key, stage in (
                    ("profiling_decode_seconds", "decode"),
                    ("profiling_full_frame_inference_seconds", "full_frame_inference"),
                    ("profiling_candidate_cpu_seconds", "candidate_cpu"),
                    ("profiling_candidate_filter_seconds", "candidate_filter"),
                    ("profiling_result_transfer_seconds", "yolo_result_transfer"),
                    ("profiling_result_parse_seconds", "result_parse"),
                    ("profiling_roi_inference_seconds", "roi_inference"),
                ):
                    try:
                        profiler.add(stage, float(hybrid_meta.get(key, 0.0)))
                    except (TypeError, ValueError):
                        continue
            yield result
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def resolve_runtime_path(path: Path) -> Path:
    """Resolve relative CLI paths from the project directory, not an arbitrary cwd."""
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_DIR / expanded).resolve()


def should_enable_long_term_memory(  # 统一决定本次视频是否允许建立和使用长期身份模板。
    total_frames: int,  # 接收视频元数据中的总帧数，避免为判断时长而解码整段视频。
    fps: float,  # 接收视频帧率，用于把总帧数换算为秒。
    config: Mapping[str, Any],  # 接收完整配置，以便保留总开关并集中管理五分钟阈值。
) -> bool:  # 返回True表示启用长期记忆，False表示只保留短时跟踪记忆。
    """仅当视频时长达到配置阈值时启用长期身份记忆；默认阈值为五分钟。"""
    policy = dict(config.get("long_term_memory", {}))  # 复制策略配置，防止运行时修改用户原始配置。
    policy_enabled = bool(policy.get("enabled", True))  # 读取长期记忆总开关，默认允许按时长自动启用。
    threshold_seconds = float(policy.get("min_video_duration_seconds", 300.0))  # 默认阈值固定为五分钟。
    safe_fps = float(fps)  # 把帧率规范成浮点数，保证除法语义一致。
    safe_total_frames = max(int(total_frames), 0)  # 把异常负帧数收敛到零，避免误开启长期记忆。
    if not policy_enabled:  # 用户显式关闭总开关时，不再考虑视频时长。
        return False  # 总开关关闭必须始终禁用长期记忆。
    if not np.isfinite(safe_fps) or safe_fps <= 0.0:  # 无效帧率无法可靠计算视频时长。
        return False  # 元数据不可靠时采取保守策略，不创建长期模板。
    video_duration_seconds = safe_total_frames / safe_fps  # 用总帧数除以帧率得到视频时长。
    return bool(video_duration_seconds >= threshold_seconds)  # “五分钟以上”包含恰好五分钟。


def to_builtin(value: Any) -> Any:
    return base.to_builtin(value)


CHECKPOINT_SCHEMA_VERSION = 1  # 独立版本号用于拒绝读取结构不兼容的旧断点文件。


def _checkpoint_constant_factory(template: Any) -> Any:
    """为原来由lambda创建的defaultdict值返回一个独立副本。"""
    return copy.deepcopy(template)


def _checkpoint_reduce_defaultdict(value: defaultdict) -> Tuple[Any, ...]:
    """把含有不可pickle的lambda工厂的defaultdict转换成可恢复形式。"""
    factory = value.default_factory
    if factory is None:
        safe_factory = None
    elif factory in {int, float, bool, str, list, dict, set, deque}:
        safe_factory = factory
    else:
        try:
            sample = factory()
        except Exception as exc:
            raise TypeError("断点无法保存defaultdict默认工厂") from exc
        safe_factory = functools.partial(_checkpoint_constant_factory, sample)
    return (defaultdict, (safe_factory,), None, None, iter(value.items()))


class _CheckpointPickler(pickle.Pickler):
    """只为断点文件扩展defaultdict序列化，不修改全局pickle行为。"""

    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[defaultdict] = _checkpoint_reduce_defaultdict


def _atomic_pickle_dump(payload: Mapping[str, Any], path: Path) -> None:
    """先完整写临时文件再原子替换，断电时不会留下半个检查点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            _CheckpointPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(dict(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    """原子写入可直接查看的进度JSON，便于不启动Python就检查状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(to_builtin(dict(payload)), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_resource_report(
    output_dir: Path,
    usage: Mapping[str, Any],
    stage_name: Optional[str] = None,
) -> None:
    """Persist resource timing separately and merge it into run metadata."""
    output_dir = Path(output_dir)
    payload = dict(usage)
    metadata_path = output_dir / "运行元数据.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                metadata = dict(loaded)
        except Exception as exc:
            logging.warning("读取运行元数据以合并资源计时失败：%s", exc)
    profiling = dict(metadata.get("profiling", {}))
    stage_seconds = dict(profiling.get("stage_seconds", {}))
    normalized_stage = str(stage_name or "").strip().lower()
    if normalized_stage == "stage2":
        model_stage_proxy = 0.0
        model_stage_note = (
            "stage2 cache-only path: no YOLO model loaded and no model inference executed"
        )
    else:
        model_stage_proxy = float(stage_seconds.get("full_frame_inference", 0.0)) + float(
            stage_seconds.get("roi_inference", 0.0)
        )
        model_stage_note = (
            "Ultralytics full-frame plus ROI inference wall time; "
            "may include CPU transfer/dispatch"
        )
    gpu_payload = dict(payload.get("gpu", {}))
    gpu_payload["model_stage_proxy_seconds"] = float(model_stage_proxy)
    gpu_payload["model_stage_proxy_note"] = model_stage_note
    payload["gpu"] = gpu_payload
    report_name = (
        "资源使用报告_阶段一.json"
        if normalized_stage == "stage1"
        else "资源使用报告_阶段二.json"
        if normalized_stage == "stage2"
        else "资源使用报告.json"
    )
    _atomic_json_dump(payload, output_dir / report_name)
    if metadata:
        if normalized_stage:
            by_stage = dict(metadata.get("resource_usage_by_stage", {}))
            by_stage[normalized_stage] = payload
            metadata["resource_usage_by_stage"] = by_stage
        else:
            metadata["resource_usage"] = payload
        _atomic_json_dump(metadata, metadata_path)


def _checkpoint_config_digest(config: Mapping[str, Any]) -> str:
    """只散列会影响推理结果的配置，允许续跑时调整检查点频率和profiling。"""
    stable = dict(to_builtin(dict(config)))
    stable.pop("checkpoint", None)
    stable.pop("profiling", None)
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file_fingerprint(path: Path) -> Dict[str, Any]:
    """使用路径、大小和纳秒修改时间防止把断点套到另一份视频或权重。"""
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _truncate_csv_at_frame(path: Path, next_frame: int) -> None:
    """恢复前移除未被检查点提交的尾部行，并保留原CSV字段和BOM。"""
    if not path.exists() or path.stat().st_size <= 0:
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.truncate.tmp")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            if "frame" not in fieldnames:
                raise ValueError(f"恢复缓存缺少frame字段：{path}")
            with temporary.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    try:
                        frame = int(float(row.get("frame", "nan")))
                    except (TypeError, ValueError):
                        continue
                    if frame < int(next_frame):
                        writer.writerow(row)
                target.flush()
                os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_frame_cache_boundary(path: Path, next_frame: int) -> None:
    """确认一帧一行的主缓存连续覆盖0到断点前一帧，拒绝带缺口的错误续跑。"""
    required_next_frame = int(next_frame)
    if required_next_frame <= 0:
        return
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"恢复断点缺少逐帧检测缓存：{path}")
    observed_frames: List[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "frame" not in list(reader.fieldnames or []):
            raise ValueError(f"恢复缓存缺少frame字段：{path}")
        for row in reader:
            try:
                observed_frames.append(int(float(row.get("frame", "nan"))))
            except (TypeError, ValueError):
                raise ValueError(f"恢复缓存含非法frame值：{path}") from None
    expected_frames = list(range(required_next_frame))
    if observed_frames != expected_frames:
        first_bad = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_frames, expected_frames)
                )
                if observed != expected
            ),
            min(len(observed_frames), len(expected_frames)),
        )
        raise ValueError(
            "逐帧检测缓存与断点不连续："
            f"断点要求0..{required_next_frame - 1}，"
            f"实际行数{len(observed_frames)}，首个异常位置{first_bad}。"
        )


def _flush_csv_handles(handles: Sequence[Any]) -> None:
    """把所有逐帧CSV从Python缓冲区同步到磁盘后再提交检查点。"""
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())


class InferenceCheckpointManager:
    """验证、保存并清理单视频推理的原子运行时检查点。"""

    def __init__(
        self,
        output_dir: Path,
        video_path: Path,
        model_path: Path,
        config: Mapping[str, Any],
        total_frames: int,
        fps: float,
        width: int,
        height: int,
        enabled: bool,
        interval_frames: int,
        resume_requested: bool,
    ) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "推理断点.pkl"
        self.status_path = output_dir / "推理断点状态.json"
        self.enabled = bool(enabled)
        self.interval_frames = max(int(interval_frames), 1)
        self.resume_requested = bool(resume_requested)
        self.total_frames = max(int(total_frames), 0)
        self.fingerprint = (
            {
                "video": _checkpoint_file_fingerprint(video_path),
                "model": _checkpoint_file_fingerprint(model_path),
                "config_sha256": _checkpoint_config_digest(config),
                "fps": float(fps),
                "width": int(width),
                "height": int(height),
                "total_frames": int(total_frames),
            }
            if self.enabled
            else {}
        )

    def prepare_fresh_run(self) -> None:
        """新任务只清除本视频旧断点；正式结果文件仍由原流程按原规则覆盖。"""
        if self.resume_requested:
            return
        self.path.unlink(missing_ok=True)
        self.status_path.unlink(missing_ok=True)

    def load(self) -> Optional[Dict[str, Any]]:
        """仅在显式--resume时加载，并严格核对视频、权重和分析配置。"""
        if not self.resume_requested:
            return None
        if not self.enabled:
            raise ValueError("--resume需要启用checkpoint配置。")
        if not self.path.exists():
            raise FileNotFoundError(f"没有可恢复的推理断点：{self.path}")
        with self.path.open("rb") as handle:
            payload = pickle.load(handle)
        if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("推理断点结构版本不兼容，请使用生成该断点的程序版本。")
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError("推理断点与当前视频、模型或分析配置不一致，拒绝错误续跑。")
        next_frame = int(payload.get("next_frame", -1))
        if next_frame < 0 or next_frame > self.total_frames:
            raise ValueError(f"推理断点帧号越界：{next_frame}/{self.total_frames}")
        return dict(payload)

    def due(self, next_frame: int) -> bool:
        """按已完成帧数判断是否到达新的安全提交边界。"""
        return bool(
            self.enabled
            and int(next_frame) > 0
            and int(next_frame) % self.interval_frames == 0
        )

    def save(
        self,
        next_frame: int,
        runtime_state: Mapping[str, Any],
        video_segments: Sequence[Mapping[str, Any]],
        inference_complete: bool = False,
    ) -> None:
        """保存下一待处理帧及全部身份/记忆状态，并同步更新人类可读进度。"""
        if not self.enabled:
            return
        saved_at = time.time()
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "program_version": PROGRAM_VERSION,
            "fingerprint": self.fingerprint,
            "next_frame": int(next_frame),
            "runtime_state": dict(runtime_state),
            "video_segments": [dict(item) for item in video_segments],
            "inference_complete": bool(inference_complete),
            "saved_at_unix": float(saved_at),
        }
        _atomic_pickle_dump(payload, self.path)
        _atomic_json_dump(
            {
                "program_version": PROGRAM_VERSION,
                "status": "inference_complete" if inference_complete else "running",
                "next_frame": int(next_frame),
                "processed_frames": int(next_frame),
                "total_frames": int(self.total_frames),
                "progress_percent": float(100.0 * int(next_frame) / max(self.total_frames, 1)),
                "saved_at_unix": float(saved_at),
                "checkpoint_file": str(self.path),
                "video_segment_count": len(video_segments),
            },
            self.status_path,
        )

    def mark_complete(self, processed_frames: Optional[int] = None) -> None:
        """最终结果全部落盘后删除可执行断点，并保留完成状态供用户核查。"""
        completed = self.total_frames if processed_frames is None else max(int(processed_frames), 0)
        self.path.unlink(missing_ok=True)
        _atomic_json_dump(
            {
                "program_version": PROGRAM_VERSION,
                "status": "complete",
                "next_frame": int(completed),
                "processed_frames": int(completed),
                "total_frames": int(self.total_frames),
                "progress_percent": 100.0,
                "checkpoint_file": str(self.path),
            },
            self.status_path,
        )


def finite_point(point: np.ndarray) -> bool:
    return base.finite_point(point)


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return base.point_distance(a, b)


def min_point_distance(point: np.ndarray, points: np.ndarray) -> float:
    return base.min_point_distance(point, points)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return base.cosine_similarity(a, b)


def angle_difference_deg(a: np.ndarray, b: np.ndarray) -> float:
    return base.angle_difference_deg(a, b)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    return base.safe_corr(a, b)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mode_or_default(values: Iterable[Any], default: Any = -1) -> Any:
    filtered = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        filtered.append(value)
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


def interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = max(end_a, end_b) - min(start_a, start_b) + 1
    return float(intersection / union) if union > 0 else 0.0


def intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) <= min(end_a, end_b)


def union_length_frames(intervals: Sequence[Tuple[int, int]]) -> int:
    cleaned = sorted((int(s), int(e)) for s, e in intervals if int(e) >= int(s))
    if not cleaned:
        return 0
    total = 0
    current_start, current_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    total += current_end - current_start + 1
    return total


@dataclass
class ProvisionalDisplayTrack:
    """只负责未获正式身份检测的短时连续显示，不参与正式ID和行为归属。"""

    provisional_id: int
    center_px: np.ndarray
    velocity_px: np.ndarray
    bbox_xyxy: np.ndarray
    body_length_px: float
    last_frame: int
    first_center_px: Optional[np.ndarray] = None
    hits: int = 1
    misses: int = 0


class ProvisionalDisplayTracker:
    """让所有未分配检测仍以稳定Pxx标签出现在核查视频中。

    该追踪器与正式身份分配器完全隔离：
    - 不创建或修改正式逻辑ID；
    - 不参与行为统计；
    - 不写入长期身份、外观或掩码模板；
    - 只在当前检测真实存在时渲染，不画预测框。
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, max_tracks: int = 40) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.max_missing_frames = max(int(cfg.get("max_missing_frames", 4)), 0)
        self.match_gate_body_lengths = float(cfg.get("match_gate_body_lengths", 1.35))
        self.iou_weight = float(cfg.get("iou_weight", 0.22))
        self.size_weight = float(cfg.get("size_weight", 0.10))
        self.max_match_cost = float(cfg.get("max_match_cost", 1.35))
        self.velocity_alpha = float(cfg.get("velocity_alpha", 0.60))
        self.promoted_overlap_iou = float(cfg.get("promoted_overlap_iou", 0.35))
        self.promoted_distance_body_lengths = float(cfg.get("promoted_distance_body_lengths", 0.45))
        self.max_tracks = max(int(cfg.get("max_tracks", max_tracks)), 1)
        self._next_id = 1
        self.tracks: Dict[int, ProvisionalDisplayTrack] = {}

    @staticmethod
    def _body_length(det: base.Detection) -> float:
        value = float(getattr(det, "body_length_px", float("nan")))
        if np.isfinite(value) and value > 3.0:
            return value
        box = np.asarray(det.bbox_xyxy, dtype=np.float64).reshape(-1)
        if box.size >= 4:
            return max(float(max(box[2] - box[0], box[3] - box[1])) * 0.75, 8.0)
        return 20.0

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        return float(base.bbox_iou_xyxy(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))

    def _is_promoted(self, tr: ProvisionalDisplayTrack, assigned: Sequence[base.Detection], frame: int) -> bool:
        if not assigned:
            return False
        dt = max(int(frame) - int(tr.last_frame), 1)
        predicted = np.asarray(tr.center_px, dtype=np.float64) + np.asarray(tr.velocity_px, dtype=np.float64) * dt
        for det in assigned:
            body = max(tr.body_length_px, self._body_length(det), 8.0)
            dist_bl = float(np.linalg.norm(predicted - np.asarray(det.center_px, dtype=np.float64)) / body)
            if dist_bl <= self.promoted_distance_body_lengths:
                return True
            if self._bbox_iou(tr.bbox_xyxy, det.bbox_xyxy) >= self.promoted_overlap_iou:
                return True
        return False

    def _cost(self, tr: ProvisionalDisplayTrack, det: base.Detection, frame: int) -> float:
        dt = max(int(frame) - int(tr.last_frame), 1)
        predicted = np.asarray(tr.center_px, dtype=np.float64) + np.asarray(tr.velocity_px, dtype=np.float64) * dt
        body_det = self._body_length(det)
        body = max(tr.body_length_px, body_det, 8.0)
        dist_bl = float(np.linalg.norm(predicted - np.asarray(det.center_px, dtype=np.float64)) / body)
        if dist_bl > self.match_gate_body_lengths:
            return float("inf")
        iou_cost = 1.0 - self._bbox_iou(tr.bbox_xyxy, det.bbox_xyxy)
        size_cost = abs(math.log(max(body_det, 1.0) / max(tr.body_length_px, 1.0)))
        motion_weight = max(1.0 - self.iou_weight - self.size_weight, 0.05)
        return float(motion_weight * dist_bl + self.iou_weight * iou_cost + self.size_weight * size_cost)

    @staticmethod
    def _solve(cost: np.ndarray) -> List[Tuple[int, int]]:
        if cost.size == 0:
            return []
        finite = np.isfinite(cost)
        if not finite.any():
            return []
        safe = np.where(finite, cost, 1e6)
        if _hungarian_assignment is not None:
            rows, cols = _hungarian_assignment(safe)
            return [(int(r), int(c)) for r, c in zip(rows, cols) if finite[int(r), int(c)]]
        pairs: List[Tuple[int, int]] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        for flat in np.argsort(safe, axis=None):
            r, c = np.unravel_index(int(flat), safe.shape)
            if not finite[r, c] or r in used_r or c in used_c:
                continue
            pairs.append((int(r), int(c)))
            used_r.add(int(r))
            used_c.add(int(c))
        return pairs

    def update(
        self,
        detections: Sequence[base.Detection],
        frame: int,
        assigned_detections: Sequence[base.Detection] = (),
    ) -> List[Tuple[int, base.Detection, str, float]]:
        """返回(provisional_negative_id, detection, label, match_cost)。"""
        if not self.enabled:
            return []

        # 检测晋升正式ID/聚集匿名ID后，清理与其位置重合的旧P轨迹，避免P编号漂到别鼠。
        for tid in list(self.tracks):
            if self._is_promoted(self.tracks[tid], assigned_detections, frame):
                self.tracks.pop(tid, None)

        track_ids = sorted(self.tracks)
        cost = np.full((len(track_ids), len(detections)), np.inf, dtype=np.float64)
        for r, tid in enumerate(track_ids):
            for c, det in enumerate(detections):
                cost[r, c] = self._cost(self.tracks[tid], det, frame)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        outputs: List[Tuple[int, base.Detection, str, float]] = []
        for r, c in self._solve(cost):
            value = float(cost[r, c])
            if not np.isfinite(value) or value > self.max_match_cost:
                continue
            tid = track_ids[r]
            det = detections[c]
            tr = self.tracks[tid]
            dt = max(int(frame) - int(tr.last_frame), 1)
            measured_velocity = (np.asarray(det.center_px, dtype=np.float64) - tr.center_px) / float(dt)
            tr.velocity_px = (
                self.velocity_alpha * measured_velocity
                + (1.0 - self.velocity_alpha) * np.asarray(tr.velocity_px, dtype=np.float64)
            )
            tr.center_px = np.asarray(det.center_px, dtype=np.float64).copy()
            tr.bbox_xyxy = np.asarray(det.bbox_xyxy, dtype=np.float64).copy()
            tr.body_length_px = 0.2 * self._body_length(det) + 0.8 * tr.body_length_px
            tr.last_frame = int(frame)
            tr.hits += 1
            tr.misses = 0
            matched_tracks.add(tid)
            matched_dets.add(c)
            outputs.append((-(2_000_000 + tid), det, f"P{tid:02d}", value))

        for tid in track_ids:
            if tid not in matched_tracks and tid in self.tracks:
                self.tracks[tid].misses += 1
                if self.tracks[tid].misses > self.max_missing_frames:
                    self.tracks.pop(tid, None)

        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            if len(self.tracks) >= self.max_tracks:
                # 容量只限制P标签连续性，不允许丢检测：超容量仍返回单帧标签。
                ephemeral = self._next_id
                self._next_id += 1
                outputs.append((-(2_000_000 + ephemeral), det, f"P{ephemeral:02d}", float("nan")))
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = ProvisionalDisplayTrack(
                provisional_id=tid,
                center_px=np.asarray(det.center_px, dtype=np.float64).copy(),
                velocity_px=np.zeros(2, dtype=np.float64),
                bbox_xyxy=np.asarray(det.bbox_xyxy, dtype=np.float64).copy(),
                body_length_px=self._body_length(det),
                last_frame=int(frame),
                first_center_px=np.asarray(det.center_px, dtype=np.float64).copy(),
            )
            outputs.append((-(2_000_000 + tid), det, f"P{tid:02d}", float("nan")))

        outputs.sort(key=lambda row: row[2])
        return outputs

    def stats(self, provisional_id: int) -> Dict[str, float]:
        """Return immutable bridge evidence for one rendered Pxx track."""
        tid = -int(provisional_id) - 2_000_000
        track = self.tracks.get(int(tid))
        if track is None:
            return {"hits": 0.0, "motion_body_lengths": 0.0}
        first = (
            np.asarray(track.first_center_px, dtype=np.float64)
            if track.first_center_px is not None
            else np.asarray(track.center_px, dtype=np.float64)
        )
        displacement = float(np.linalg.norm(np.asarray(track.center_px) - first))
        body = max(float(track.body_length_px), 8.0)
        return {
            "hits": float(track.hits),
            "motion_body_lengths": float(displacement / body),
        }

    def remove(self, provisional_id: int) -> None:
        """Drop a Pxx track after the identity module has promoted it."""
        tid = -int(provisional_id) - 2_000_000
        self.tracks.pop(int(tid), None)


# -----------------------------------------------------------------------------
# 尺度和四角透视标定
# -----------------------------------------------------------------------------


class CoordinateTransformer:
    """把像素关键点转换为真实厘米坐标。"""

    def __init__(
        self,
        scale_config: Mapping[str, Any],
        video_path: Path,
        calibration_path: Optional[Path] = None,
    ) -> None:
        self.video_path = video_path
        self.mode = str(scale_config.get("mode", "body_length")).lower()
        self.scale_estimator: Optional[base.ScaleEstimator] = None
        self.homography: Optional[np.ndarray] = None
        self.calibration_entry: Optional[Dict[str, Any]] = None
        self.current_cm_per_pixel = float("nan")

        if calibration_path is not None:
            self.mode = "homography"
            self._load_homography(calibration_path)
        elif self.mode == "homography":
            config_path = scale_config.get("calibration_file")
            if not config_path:
                raise ValueError("scale.mode=homography时必须设置calibration_file或使用--calibration。")
            self._load_homography(Path(config_path))
        elif self.mode in {"fixed", "body_length"}:
            self.scale_estimator = base.ScaleEstimator(scale_config)
        else:
            raise ValueError(f"不支持的尺度模式：{self.mode}")

    def _load_homography(self, calibration_path: Path) -> None:
        calibration_path = calibration_path.expanduser().resolve()
        if not calibration_path.exists():
            raise FileNotFoundError(f"标定文件不存在：{calibration_path}")
        with calibration_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        entry: Optional[Mapping[str, Any]] = None
        videos = payload.get("videos") if isinstance(payload, dict) else None
        if isinstance(videos, dict):
            for key in (self.video_path.name, self.video_path.stem):
                if key in videos:
                    entry = videos[key]
                    break
        if entry is None and isinstance(payload, dict) and isinstance(payload.get("default"), dict):
            entry = payload["default"]
        if entry is None and isinstance(payload, dict) and "pixel_corners" in payload:
            entry = payload
        if entry is None:
            raise ValueError(
                f"标定文件中没有找到视频 {self.video_path.name}、{self.video_path.stem} 或 default 配置。"
            )

        corners = np.asarray(entry.get("pixel_corners"), dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError("pixel_corners必须是4×2数组，顺序为左上、右上、右下、左下。")
        width_cm = safe_float(entry.get("width_cm"), -1.0)
        height_cm = safe_float(entry.get("height_cm"), -1.0)
        if width_cm <= 0 or height_cm <= 0:
            raise ValueError("标定中的width_cm和height_cm必须大于0。")

        world = np.asarray(
            [[0.0, 0.0], [width_cm, 0.0], [width_cm, height_cm], [0.0, height_cm]],
            dtype=np.float32,
        )
        self.homography = cv2.getPerspectiveTransform(corners, world)
        self.calibration_entry = {
            "source": str(calibration_path),
            "pixel_corners": corners.tolist(),
            "width_cm": width_cm,
            "height_cm": height_cm,
        }

    def update(self, detections: Sequence[base.Detection]) -> None:
        if self.scale_estimator is not None:
            self.current_cm_per_pixel = float(self.scale_estimator.update(detections))

    def transform_points(self, points_px: np.ndarray) -> np.ndarray:
        points_px = np.asarray(points_px, dtype=np.float64)
        result = np.full_like(points_px, np.nan, dtype=np.float64)
        valid = np.all(np.isfinite(points_px), axis=1)
        if not np.any(valid):
            return result

        if self.mode == "homography":
            assert self.homography is not None
            source = points_px[valid].astype(np.float32).reshape(-1, 1, 2)
            converted = cv2.perspectiveTransform(source, self.homography).reshape(-1, 2)
            result[valid] = converted.astype(np.float64)
        else:
            if not np.isfinite(self.current_cm_per_pixel) or self.current_cm_per_pixel <= 0:
                raise RuntimeError("厘米/像素比例尚未建立。")
            result[valid] = points_px[valid] * self.current_cm_per_pixel
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "cm_per_pixel": (
                float(self.current_cm_per_pixel)
                if np.isfinite(self.current_cm_per_pixel)
                else None
            ),
            "calibration": self.calibration_entry,
        }


# -----------------------------------------------------------------------------
# 观测与时序特征
# -----------------------------------------------------------------------------


def build_observation(
    frame: int,
    fps: float,
    logical_id: int,
    detection: base.Detection,
    smoothed_keypoints_px: np.ndarray,
    effective_conf: np.ndarray,
    transformer: CoordinateTransformer,
    previous: Optional[base.MouseObservation],
    track_state: str = "tracked",
    display_label: str = "",
) -> base.MouseObservation:
    kpts_cm = transformer.transform_points(smoothed_keypoints_px)
    center, head, rear, heading, body_length = base.derive_geometry(kpts_cm)
    # 白鼠或遮挡时可能仍有可靠检测框，但关键点不足。此时使用检测中心维持
    # ID、框和轨迹连续性；不会伪造鼻/耳/尾关键点。
    if not finite_point(center):
        center_px = np.asarray(detection.center_px, dtype=np.float64).reshape(1, 2)
        center_cm_fallback = transformer.transform_points(center_px)[0]
        if finite_point(center_cm_fallback):
            center = center_cm_fallback

    velocity = np.zeros(2, dtype=np.float64)
    speed = acceleration = angular_speed = nose_speed = 0.0
    if previous is not None:
        dt_frames = max(frame - previous.frame, 1)
        dt_seconds = dt_frames / fps
        if finite_point(center) and finite_point(previous.center_cm):
            velocity = (center - previous.center_cm) / dt_seconds
            speed = safe_float(np.linalg.norm(velocity), 0.0)
            acceleration = abs(speed - previous.speed_cm_s) / dt_seconds
        if np.all(np.isfinite(heading)) and np.all(np.isfinite(previous.heading)):
            angular_speed = angle_difference_deg(heading, previous.heading) / dt_seconds
        nose = kpts_cm[KP["nose"]]
        old_nose = previous.keypoints_cm[KP["nose"]]
        if finite_point(nose) and finite_point(old_nose):
            nose_speed = point_distance(nose, old_nose) / dt_seconds

    return base.MouseObservation(
        frame=frame,
        logical_id=logical_id,
        raw_track_id=detection.raw_track_id,
        keypoints_px=smoothed_keypoints_px,
        keypoints_cm=kpts_cm,
        keypoint_conf=effective_conf,
        bbox_xyxy=detection.bbox_xyxy,
        box_conf=detection.box_conf,
        center_cm=center,
        head_cm=head,
        rear_cm=rear,
        heading=heading,
        velocity_cm_s=velocity,
        speed_cm_s=float(speed),
        acceleration_cm_s2=float(acceleration),
        angular_speed_deg_s=float(angular_speed),
        nose_speed_cm_s=float(nose_speed),
        body_length_cm=float(body_length) if np.isfinite(body_length) else float("nan"),
        track_state=track_state,
        display_label=display_label,
    )


@dataclass
class _IndividualBehaviorState:
    last_frame: int = -1
    last_center_px: Optional[np.ndarray] = None
    body_lengths_cm: Deque[float] = None  # type: ignore[assignment]
    hold_until_frame: int = -1

    def __post_init__(self) -> None:
        if self.body_lengths_cm is None:
            self.body_lengths_cm = deque(maxlen=45)


class IndividualBehaviorGate:
    """单鼠排除门：识别靠墙跳跃/攀爬导致的瞬时高速。

    该模块只服务行为分类，不修改检测、ID、掩码记忆或骨架。识别到的
    wall_jump会在短时间内屏蔽该鼠参与追逐/攻击判定，但仍保留其轨迹。
    """

    def __init__(self, fps: float, config: Mapping[str, Any]) -> None:
        self.fps = float(fps)
        self.cfg = dict(config.get("wall_jump", {}))
        self.enabled = bool(self.cfg.get("enabled", True))
        self.states: Dict[int, _IndividualBehaviorState] = {}
        self.hold_frames = max(int(round(float(self.cfg.get("suppress_seconds", 0.35)) * self.fps)), 1)

    @staticmethod
    def _bbox_center_and_body(obs: base.MouseObservation) -> Tuple[np.ndarray, float]:
        box = np.asarray(obs.bbox_xyxy, dtype=np.float64).reshape(-1)
        if len(box) >= 4 and np.all(np.isfinite(box[:4])):
            x1, y1, x2, y2 = box[:4]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
            body_px = max(float(x2 - x1), float(y2 - y1), 1.0)
            return center, body_px
        valid = np.asarray(obs.keypoints_px, dtype=np.float64)
        valid = valid[np.all(np.isfinite(valid), axis=1)]
        if len(valid):
            span = np.ptp(valid, axis=0)
            return np.mean(valid, axis=0), max(float(np.max(span)), 1.0)
        return np.array([np.nan, np.nan], dtype=np.float64), 1.0

    @staticmethod
    def _nearest_outward_vector(center: np.ndarray, width: int, height: int) -> Tuple[float, np.ndarray]:
        x, y = float(center[0]), float(center[1])
        distances = [x, max(width - 1 - x, 0.0), y, max(height - 1 - y, 0.0)]
        vectors = [
            np.array([-1.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, -1.0]), np.array([0.0, 1.0]),
        ]
        idx = int(np.argmin(distances))
        return float(distances[idx]), vectors[idx]

    def update(
        self,
        frame: int,
        observations: Sequence[base.MouseObservation],
        frame_width: int,
        frame_height: int,
    ) -> Dict[int, bool]:
        if not self.enabled:
            return {int(o.logical_id): False for o in observations}

        active: Dict[int, bool] = {}
        seen: set[int] = set()
        for obs in observations:
            lid = int(obs.logical_id)
            seen.add(lid)
            state = self.states.setdefault(lid, _IndividualBehaviorState())
            center_px, body_px = self._bbox_center_and_body(obs)
            if not np.all(np.isfinite(center_px)):
                state.last_frame = int(frame)
                active[lid] = bool(int(frame) <= state.hold_until_frame)
                continue
            edge_distance_px, outward = self._nearest_outward_vector(center_px, frame_width, frame_height)
            edge_distance_bl = edge_distance_px / max(body_px, 1.0)

            pixel_motion = np.zeros(2, dtype=np.float64)
            if state.last_center_px is not None and np.all(np.isfinite(center_px)):
                dt = max(int(frame) - int(state.last_frame), 1)
                pixel_motion = (center_px - state.last_center_px) / dt
            motion_norm = float(np.linalg.norm(pixel_motion))
            outward_alignment = (
                float(np.dot(pixel_motion / motion_norm, outward)) if motion_norm > 1e-6 else -1.0
            )

            body_ratio = 1.0
            if np.isfinite(obs.body_length_cm) and obs.body_length_cm > 0:
                if len(state.body_lengths_cm) >= 5:
                    reference = float(np.median(np.asarray(state.body_lengths_cm, dtype=np.float64)))
                    if reference > 1e-6:
                        body_ratio = float(obs.body_length_cm / reference)
                state.body_lengths_cm.append(float(obs.body_length_cm))

            conf = np.asarray(obs.keypoint_conf, dtype=np.float64)
            valid_conf = conf[np.isfinite(conf)]
            pose_quality = float(np.mean(valid_conf)) if len(valid_conf) else 0.0
            distorted_pose = bool(
                body_ratio < float(self.cfg.get("body_length_ratio_min", 0.62))
                or body_ratio > float(self.cfg.get("body_length_ratio_max", 1.55))
                or pose_quality < float(self.cfg.get("pose_quality_max_for_distortion", 0.35))
            )
            abrupt_motion = bool(
                obs.acceleration_cm_s2 >= float(self.cfg.get("min_acceleration_cm_s2", 80.0))
                or obs.nose_speed_cm_s >= float(self.cfg.get("min_nose_speed_cm_s", 30.0))
                or distorted_pose
            )
            trigger = bool(
                edge_distance_bl <= float(self.cfg.get("max_edge_distance_body_lengths", 0.85))
                and obs.speed_cm_s >= float(self.cfg.get("min_speed_cm_s", 18.0))
                and abrupt_motion
                and (
                    outward_alignment >= float(self.cfg.get("min_outward_alignment", 0.35))
                    or distorted_pose
                )
            )
            if trigger:
                state.hold_until_frame = max(state.hold_until_frame, int(frame) + self.hold_frames)
            state.last_center_px = center_px.copy() if np.all(np.isfinite(center_px)) else state.last_center_px
            state.last_frame = int(frame)
            active[lid] = bool(int(frame) <= state.hold_until_frame)

        # 仅保留近期出现的状态，避免长视频状态表无限增长。
        stale_after = max(int(round(self.fps * 10.0)), 30)
        for lid in list(self.states):
            if lid not in seen and int(frame) - self.states[lid].last_frame > stale_after:
                del self.states[lid]
        return active


@dataclass
class HighRecallPairFeatures:
    actor_id: int
    target_id: int
    center_distance_cm: float
    head_distance_cm: float
    nose_head_distance_cm: float
    nose_body_distance_cm: float
    nose_tail_distance_cm: float
    actor_speed_cm_s: float
    target_speed_cm_s: float
    actor_acceleration_cm_s2: float
    target_acceleration_cm_s2: float
    actor_nose_speed_cm_s: float
    target_nose_speed_cm_s: float
    actor_angular_speed_deg_s: float
    target_angular_speed_deg_s: float
    actor_body_length_cm: float
    target_body_length_cm: float
    actor_pose_deformation_energy: float
    target_pose_deformation_energy: float
    center_distance_body_lengths: float
    closing_speed_cm_s: float
    actor_head_relative_speed_cm_s: float
    target_head_relative_speed_cm_s: float
    direction_similarity: float
    pursuit_alignment: float
    target_escape_alignment: float
    behind_score: float
    actor_behind_target: bool
    actor_wall_jump: bool
    target_wall_jump: bool
    trajectory_correlation: float
    actor_path_window_cm: float
    target_path_window_cm: float
    distance_drop_cm: float
    target_turn_angle_deg: float
    repeated_contact_count: int
    weak_contact: bool
    strong_contact: bool
    weak_potential_attack: bool
    strong_potential_attack: bool
    weak_attack_actor_initiation: bool
    strong_attack_actor_initiation: bool
    weak_attack_target_reaction: bool
    strong_attack_target_reaction: bool
    weak_chase_score: int
    strong_chase_score: int
    weak_chase: bool
    strong_chase: bool
    weak_attack_evidence: int
    strong_attack_evidence: int
    weak_stationary_fight: bool
    strong_stationary_fight: bool
    weak_attack: bool
    strong_attack: bool


class PairFeatureComputer:
    def __init__(self, fps: float, config: Mapping[str, Any]) -> None:
        self.fps = float(fps)
        self.features_cfg = config["features"]
        self.chase_cfg = config["chase"]
        self.attack_cfg = config["attack"]
        self.history_frames = max(
            int(round(self.fps * float(self.features_cfg.get("history_seconds", 1.0)))), 3
        )
        self.lookback_frames = max(
            int(round(self.fps * float(self.features_cfg.get("response_lookback_seconds", 0.3)))), 1
        )

        # Per-frame memoization is safe because history is fully populated
        # before pair enumeration and is not mutated until the next frame.
        self._cache_frame = -1
        self._history_map_cache: Dict[int, Dict[int, base.MouseObservation]] = {}
        self._trajectory_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._target_turn_cache: Dict[int, float] = {}
        self._distance_drop_cache: Dict[Tuple[int, int], float] = {}
        self._pose_deformation_cache: Dict[int, float] = {}

    def _ensure_frame_cache(self, frame: int) -> None:
        frame = int(frame)
        if frame == self._cache_frame:
            return
        self._cache_frame = frame
        self._history_map_cache.clear()
        self._trajectory_cache.clear()
        self._target_turn_cache.clear()
        self._distance_drop_cache.clear()
        self._pose_deformation_cache.clear()

    def _history_map(
        self, logical_id: int, history: base.ObservationHistory
    ) -> Dict[int, base.MouseObservation]:
        logical_id = int(logical_id)
        cached = self._history_map_cache.get(logical_id)
        if cached is not None:
            return cached
        if hasattr(history, "get_window"):
            values = history.get_window(logical_id, self.history_frames)
        else:
            values = history.get(logical_id)[-self.history_frames :]
        cached = {observation.frame: observation for observation in values}
        self._history_map_cache[logical_id] = cached
        return cached

    def _trajectory_features(
        self,
        actor_id: int,
        target_id: int,
        history: base.ObservationHistory,
        frame: int,
    ) -> Tuple[float, float, float]:
        self._ensure_frame_cache(frame)
        key = (int(actor_id), int(target_id))
        cached = self._trajectory_cache.get(key)
        if cached is not None:
            return cached
        actor_by_frame = self._history_map(actor_id, history)
        target_by_frame = self._history_map(target_id, history)
        common = sorted(set(actor_by_frame) & set(target_by_frame))
        a = np.empty((0, 2), dtype=np.float64)
        b = np.empty((0, 2), dtype=np.float64)
        da = np.empty((0, 2), dtype=np.float64)
        db = np.empty((0, 2), dtype=np.float64)
        if len(common) < 4:
            result = (0.0, 0.0, 0.0)
        else:
            a = np.stack([actor_by_frame[f].center_cm for f in common])
            b = np.stack([target_by_frame[f].center_cm for f in common])
            valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
            a, b = a[valid], b[valid]
            if len(a) < 4:
                result = (0.0, 0.0, 0.0)
            else:
                da, db = np.diff(a, axis=0), np.diff(b, axis=0)
                # 对水平/垂直直线运动，单轴可能是常数，逐轴Pearson会错误给0。
                # 展平二维位移后再计算相关性，保持“同轨迹>0.7”的文档含义。
                corr = float(
                    np.clip(safe_corr(da.reshape(-1), db.reshape(-1)), -1.0, 1.0)
                )
                result = (
                    corr,
                    float(np.linalg.norm(da, axis=1).sum()),
                    float(np.linalg.norm(db, axis=1).sum()),
                )
        self._trajectory_cache[key] = result
        if key[0] != key[1]:
            # np.corrcoef can differ by one ULP when its arguments are swapped.
            # Reproduce the historical directed order so CSV values stay exact.
            if len(common) < 4 or len(a) < 4:
                reverse_corr = 0.0
            else:
                reverse_corr = float(
                    np.clip(safe_corr(db.reshape(-1), da.reshape(-1)), -1.0, 1.0)
                )
            self._trajectory_cache[(key[1], key[0])] = (
                reverse_corr, result[2], result[1]
            )
        return result

    def _distance_drop(
        self,
        actor: base.MouseObservation,
        target: base.MouseObservation,
        history: base.ObservationHistory,
    ) -> float:
        self._ensure_frame_cache(actor.frame)
        key = tuple(sorted((int(actor.logical_id), int(target.logical_id))))
        cached = self._distance_drop_cache.get(key)
        if cached is not None:
            return cached
        old_frame = actor.frame - self.lookback_frames
        old_actor = history.near_frame(actor.logical_id, old_frame)
        old_target = history.near_frame(target.logical_id, old_frame)
        current = point_distance(actor.center_cm, target.center_cm)
        if old_actor is None or old_target is None or not np.isfinite(current):
            value = 0.0
        else:
            previous = point_distance(old_actor.center_cm, old_target.center_cm)
            value = float(previous - current) if np.isfinite(previous) else 0.0
        self._distance_drop_cache[key] = float(value)
        return float(value)

    def _target_turn(
        self,
        target: base.MouseObservation,
        history: base.ObservationHistory,
    ) -> float:
        self._ensure_frame_cache(target.frame)
        logical_id = int(target.logical_id)
        cached = self._target_turn_cache.get(logical_id)
        if cached is not None:
            return cached
        old = history.near_frame(logical_id, target.frame - self.lookback_frames)
        value = 0.0 if old is None else angle_difference_deg(old.heading, target.heading)
        self._target_turn_cache[logical_id] = float(value)
        return float(value)

    @staticmethod
    def _body_frame_pose(observation: base.MouseObservation) -> Tuple[np.ndarray, np.ndarray]:
        """Translation/rotation/scale invariant pose for deformation energy."""
        points = np.asarray(observation.keypoints_cm, dtype=np.float64)
        confidence = np.asarray(observation.keypoint_conf, dtype=np.float64).reshape(-1)
        count = min(len(points), len(confidence), len(KEYPOINT_NAMES))
        pose = np.full((len(KEYPOINT_NAMES), 2), np.nan, dtype=np.float64)
        valid = np.zeros(len(KEYPOINT_NAMES), dtype=bool)
        if count <= 0 or not finite_point(observation.center_cm):
            return pose, valid
        body = float(observation.body_length_cm)
        if not np.isfinite(body) or body <= 1e-6:
            return pose, valid
        heading = np.asarray(observation.heading, dtype=np.float64)
        norm = float(np.linalg.norm(heading)) if np.all(np.isfinite(heading)) else 0.0
        if norm <= 1e-9:
            heading = np.array([1.0, 0.0], dtype=np.float64)
        else:
            heading = heading / norm
        lateral = np.array([-heading[1], heading[0]], dtype=np.float64)
        good = (
            np.all(np.isfinite(points[:count]), axis=1)
            & np.isfinite(confidence[:count])
            & (confidence[:count] >= 0.08)
        )
        if not np.any(good):
            return pose, valid
        relative = (points[:count] - np.asarray(observation.center_cm, dtype=np.float64)) / body
        pose[:count, 0] = relative @ heading
        pose[:count, 1] = relative @ lateral
        invalid_indices = np.arange(count, dtype=int)[~good]
        if invalid_indices.size:
            pose[invalid_indices] = np.nan
        valid[:count] = good
        return pose, valid

    def _pose_deformation(
        self, observation: base.MouseObservation, history: base.ObservationHistory
    ) -> float:
        """RMS body-frame keypoint deformation from the previous measured frame.

        Translation, heading rotation and body scale are removed first.  The
        result is dimensionless (body lengths/frame) and is therefore useful
        for stationary wrestling where center speed alone is misleading.
        """
        self._ensure_frame_cache(observation.frame)
        logical_id = int(observation.logical_id)
        cached = self._pose_deformation_cache.get(logical_id)
        if cached is not None:
            return float(cached)
        previous = history.near_frame(logical_id, int(observation.frame) - 1)
        if previous is None:
            value = 0.0
        else:
            current_pose, current_valid = self._body_frame_pose(observation)
            previous_pose, previous_valid = self._body_frame_pose(previous)
            valid = current_valid & previous_valid
            if int(valid.sum()) < 3:
                value = 0.0
            else:
                delta = current_pose[valid] - previous_pose[valid]
                value = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
                if not np.isfinite(value):
                    value = 0.0
        self._pose_deformation_cache[logical_id] = float(value)
        return float(value)

    @staticmethod
    def _evaluate_chase(
        cfg: Mapping[str, Any],
        center_distance: float,
        actor_speed: float,
        target_speed: float,
        direction_similarity: float,
        pursuit_alignment: float,
        target_escape_alignment: float,
        actor_behind: bool,
        trajectory_corr: float,
        actor_wall_jump: bool,
        target_wall_jump: bool,
    ) -> Tuple[int, bool]:
        """按PPT定义做帧级追逐门控；60 cm路程在时序后处理中验证。"""
        conditions = [
            np.isfinite(center_distance) and center_distance <= float(cfg["max_distance_cm"]),
            actor_speed >= float(cfg["actor_min_speed_cm_s"]),
            target_speed >= float(cfg["target_min_speed_cm_s"]),
            direction_similarity >= float(cfg["direction_similarity_min"]),
            pursuit_alignment >= float(cfg["pursuit_alignment_min"]),
            target_escape_alignment >= float(cfg.get("target_escape_alignment_min", 0.35)),
            actor_behind,
            trajectory_corr >= float(cfg["trajectory_correlation_min"]),
            not actor_wall_jump and not target_wall_jump,
        ]
        score = int(sum(bool(v) for v in conditions))
        if bool(cfg.get("require_all_conditions", True)):
            candidate = bool(all(conditions))
        else:
            candidate = score >= int(cfg.get("candidate_score_min", len(conditions)))
        return score, candidate

    @staticmethod
    def _evaluate_attack(
        cfg: Mapping[str, Any],
        center_distance: float,
        nose_body_distance: float,
        actor: base.MouseObservation,
        target: base.MouseObservation,
        distance_drop: float,
        target_turn: float,
        pursuit_alignment: float,
        target_escape_alignment: float,
        repeated_contact_count: int,
        actor_wall_jump: bool,
        target_wall_jump: bool,
    ) -> Tuple[bool, bool, int, bool, bool, bool]:
        """Use nose-to-whole-body contact first, then confirm attack causality."""
        # Potential contact is a nose-to-whole-body gate, not a nose-to-tail gate.
        # The target's seven keypoints cover the head (nose/ears), trunk anchors
        # (neck/hips), and tail; the minimum distance therefore accepts a bite,
        # mount, or grapple regardless of which body part is touched.  Do not add
        # a center-distance fallback here: a nearby but non-touching pair is only
        # a behavioral candidate after dynamic evidence has been confirmed.
        potential_contact = bool(
            np.isfinite(nose_body_distance)
            and nose_body_distance < float(cfg["contact_distance_cm"])
        )
        lunge = actor.speed_cm_s >= float(cfg["actor_lunge_speed_cm_s"])
        rapid_closing = distance_drop >= float(cfg["rapid_closing_distance_cm"])
        target_escape = bool(
            target.speed_cm_s >= float(cfg["target_escape_speed_cm_s"])
            and target_escape_alignment >= float(cfg.get("target_escape_alignment_min", 0.30))
        )
        target_turning = target_turn >= float(cfg["target_turn_angle_deg"])
        repeated_contact = repeated_contact_count >= int(cfg["repeated_contact_count"])
        head_motion = bool(
            actor.nose_speed_cm_s >= float(cfg["head_motion_speed_cm_s"])
            and actor.nose_speed_cm_s
            >= float(cfg["head_to_center_speed_ratio"]) * max(actor.speed_cm_s, 1.0)
        )
        actor_toward_target = pursuit_alignment >= float(cfg.get("attack_pursuit_alignment_min", 0.50))
        actor_initiation = bool(
            actor_toward_target
            and (lunge or rapid_closing or (head_motion and rapid_closing))
        )
        # 仅转身可能是正常接触/嗅探；最终攻击要求目标沿远离攻击者方向逃离。
        target_reaction = bool(target_escape)
        evidence = int(sum([lunge, rapid_closing, target_escape, target_turning, repeated_contact, head_motion]))

        # 静止扭打不再仅凭角速度成立：必须重复接触且双方姿态持续剧烈变化。
        stationary = bool(
            np.isfinite(center_distance)
            and center_distance < float(cfg["stationary_fight_distance_cm"])
            and max(actor.speed_cm_s, target.speed_cm_s)
            < float(cfg["stationary_fight_max_center_speed_cm_s"])
            and min(actor.angular_speed_deg_s, target.angular_speed_deg_s)
            >= float(cfg["stationary_fight_min_angular_speed_deg_s"])
            and repeated_contact
            and potential_contact
        )
        dynamic_confirmed = bool(
            potential_contact
            and evidence >= int(cfg["min_dynamic_evidence"])
            and actor_initiation
            and target_reaction
        )
        attack = bool(
            not actor_wall_jump
            and not target_wall_jump
            and (dynamic_confirmed or stationary)
        )
        return potential_contact, attack, evidence, stationary, actor_initiation, target_reaction

    def compute(
        self,
        actor: base.MouseObservation,
        target: base.MouseObservation,
        history: base.ObservationHistory,
        repeated_contact_count: int,
        actor_wall_jump: bool = False,
        target_wall_jump: bool = False,
        geometry: Optional[Mapping[str, float]] = None,
    ) -> HighRecallPairFeatures:
        # NumPy mode supplies these independent geometric quantities in one
        # broadcasted batch.  The scalar fallback remains byte-compatible with
        # the historical implementation and is used by the inline baseline.
        if geometry is None:
            center_distance = point_distance(actor.center_cm, target.center_cm)
            head_distance = point_distance(actor.head_cm, target.head_cm)
        else:
            center_distance = float(geometry.get("center_distance_cm", np.nan))
            head_distance = float(geometry.get("head_distance_cm", np.nan))
        actor_nose = actor.keypoints_cm[KP["nose"]]
        if geometry is None:
            nose_head = min_point_distance(
                actor_nose,
                target.keypoints_cm[[KP["nose"], KP["left_ear"], KP["right_ear"]]],
            )
            nose_body = min_point_distance(actor_nose, target.keypoints_cm)
            nose_tail = point_distance(actor_nose, target.keypoints_cm[KP["tail"]])
        else:
            nose_head = float(geometry.get("nose_head_distance_cm", np.nan))
            nose_body = float(geometry.get("nose_body_distance_cm", np.nan))
            nose_tail = float(geometry.get("nose_tail_distance_cm", np.nan))
        direction = cosine_similarity(actor.velocity_cm_s, target.velocity_cm_s)
        vector_to_target = target.center_cm - actor.center_cm
        pursuit = cosine_similarity(actor.velocity_cm_s, vector_to_target)
        target_escape = cosine_similarity(target.velocity_cm_s, vector_to_target)
        separation = float(np.linalg.norm(vector_to_target)) if np.all(np.isfinite(vector_to_target)) else float("nan")
        if np.isfinite(separation) and separation > 1e-9:
            radial = vector_to_target / separation
            closing_speed = float(np.dot(actor.velocity_cm_s - target.velocity_cm_s, radial))
        else:
            closing_speed = 0.0
        behind_score = float(-cosine_similarity(actor.center_cm - target.center_cm, target.heading))
        behind = bool(behind_score > 0.0)
        valid_body_lengths = [
            float(value)
            for value in (actor.body_length_cm, target.body_length_cm)
            if np.isfinite(value) and float(value) > 1e-6
        ]
        mean_body_length = (
            float(np.mean(valid_body_lengths)) if valid_body_lengths else float("nan")
        )
        center_distance_bl = (
            float(center_distance) / mean_body_length
            if np.isfinite(center_distance) and np.isfinite(mean_body_length) and mean_body_length > 1e-6
            else float("nan")
        )
        actor_head_relative_speed = max(float(actor.nose_speed_cm_s) - float(actor.speed_cm_s), 0.0)
        target_head_relative_speed = max(float(target.nose_speed_cm_s) - float(target.speed_cm_s), 0.0)
        trajectory_corr, actor_path, target_path = self._trajectory_features(
            actor.logical_id, target.logical_id, history, actor.frame
        )
        distance_drop = self._distance_drop(actor, target, history)
        target_turn = self._target_turn(target, history)
        actor_pose_deformation = self._pose_deformation(actor, history)
        target_pose_deformation = self._pose_deformation(target, history)

        weak_chase_score, weak_chase = self._evaluate_chase(
            self.chase_cfg["weak"], center_distance, actor.speed_cm_s, target.speed_cm_s,
            direction, pursuit, target_escape, behind, trajectory_corr, actor_wall_jump, target_wall_jump
        )
        strong_chase_score, strong_chase = self._evaluate_chase(
            self.chase_cfg["strong"], center_distance, actor.speed_cm_s, target.speed_cm_s,
            direction, pursuit, target_escape, behind, trajectory_corr, actor_wall_jump, target_wall_jump
        )
        (weak_contact, weak_attack, weak_evidence, weak_stationary,
         weak_initiation, weak_reaction) = self._evaluate_attack(
            self.attack_cfg["weak"], center_distance, nose_body, actor, target,
            distance_drop, target_turn, pursuit, target_escape, repeated_contact_count,
            actor_wall_jump, target_wall_jump
        )
        (strong_contact, strong_attack, strong_evidence, strong_stationary,
         strong_initiation, strong_reaction) = self._evaluate_attack(
            self.attack_cfg["strong"], center_distance, nose_body, actor, target,
            distance_drop, target_turn, pursuit, target_escape, repeated_contact_count,
            actor_wall_jump, target_wall_jump
        )

        return HighRecallPairFeatures(
            actor_id=actor.logical_id,
            target_id=target.logical_id,
            center_distance_cm=float(center_distance),
            head_distance_cm=float(head_distance),
            nose_head_distance_cm=float(nose_head),
            nose_body_distance_cm=float(nose_body),
            nose_tail_distance_cm=float(nose_tail),
            actor_speed_cm_s=float(actor.speed_cm_s),
            target_speed_cm_s=float(target.speed_cm_s),
            actor_acceleration_cm_s2=float(actor.acceleration_cm_s2),
            target_acceleration_cm_s2=float(target.acceleration_cm_s2),
            actor_nose_speed_cm_s=float(actor.nose_speed_cm_s),
            target_nose_speed_cm_s=float(target.nose_speed_cm_s),
            actor_angular_speed_deg_s=float(actor.angular_speed_deg_s),
            target_angular_speed_deg_s=float(target.angular_speed_deg_s),
            actor_body_length_cm=float(actor.body_length_cm),
            target_body_length_cm=float(target.body_length_cm),
            actor_pose_deformation_energy=float(actor_pose_deformation),
            target_pose_deformation_energy=float(target_pose_deformation),
            center_distance_body_lengths=float(center_distance_bl),
            closing_speed_cm_s=float(closing_speed),
            actor_head_relative_speed_cm_s=float(actor_head_relative_speed),
            target_head_relative_speed_cm_s=float(target_head_relative_speed),
            direction_similarity=float(direction),
            pursuit_alignment=float(pursuit),
            target_escape_alignment=float(target_escape),
            behind_score=float(behind_score),
            actor_behind_target=behind,
            actor_wall_jump=bool(actor_wall_jump),
            target_wall_jump=bool(target_wall_jump),
            trajectory_correlation=float(trajectory_corr),
            actor_path_window_cm=float(actor_path),
            target_path_window_cm=float(target_path),
            distance_drop_cm=float(distance_drop),
            target_turn_angle_deg=float(target_turn),
            repeated_contact_count=int(repeated_contact_count),
            weak_contact=weak_contact,
            strong_contact=strong_contact,
            weak_potential_attack=weak_contact,
            strong_potential_attack=strong_contact,
            weak_attack_actor_initiation=weak_initiation,
            strong_attack_actor_initiation=strong_initiation,
            weak_attack_target_reaction=weak_reaction,
            strong_attack_target_reaction=strong_reaction,
            weak_chase_score=weak_chase_score,
            strong_chase_score=strong_chase_score,
            weak_chase=weak_chase,
            strong_chase=strong_chase,
            weak_attack_evidence=weak_evidence,
            strong_attack_evidence=strong_evidence,
            weak_stationary_fight=weak_stationary,
            strong_stationary_fight=strong_stationary,
            weak_attack=weak_attack,
            strong_attack=strong_attack,
        )


def choose_direction(a_to_b: HighRecallPairFeatures, b_to_a: HighRecallPairFeatures) -> HighRecallPairFeatures:
    def rank(x: HighRecallPairFeatures) -> Tuple[int, int, int, int, float]:
        return (
            int(x.strong_attack) + int(x.strong_chase),
            int(x.weak_attack) + int(x.weak_chase),
            x.strong_attack_evidence + x.strong_chase_score,
            x.weak_attack_evidence + x.weak_chase_score,
            x.actor_speed_cm_s,
        )
    return a_to_b if rank(a_to_b) >= rank(b_to_a) else b_to_a


# -----------------------------------------------------------------------------
# 帧级时序后处理和事件生成
# -----------------------------------------------------------------------------


def empty_frame_record(frame: int, fps: float, transformer: CoordinateTransformer) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "frame": frame,
        "time_s": frame / fps,
        "valid_pair": False,
        "scale_mode": transformer.mode,
        "cm_per_pixel": transformer.current_cm_per_pixel,
        "mouse_a_id": np.nan,
        "mouse_b_id": np.nan,
        "mouse_a_raw_track_id": np.nan,
        "mouse_b_raw_track_id": np.nan,
        "mouse_a_track_state": "",
        "mouse_b_track_state": "",
        "identity_pair_quality": 0.0,
        "mouse_a_speed_cm_s": 0.0,
        "mouse_b_speed_cm_s": 0.0,
        "center_distance_cm": np.nan,
        "head_distance_cm": np.nan,
        "trajectory_correlation": 0.0,
        "direction_similarity": 0.0,
        "pursuit_alignment": 0.0,
        "target_escape_alignment": 0.0,
        "actor_behind_target": False,
        # v1.43 标准行为引擎：保留A→B和B→A两套连续方向特征，
        # 角色推断不再依赖旧choose_direction提前丢弃另一方向的信息。
        "a_to_b_actor_speed_cm_s": 0.0,
        "a_to_b_target_speed_cm_s": 0.0,
        "a_to_b_actor_acceleration_cm_s2": 0.0,
        "a_to_b_target_acceleration_cm_s2": 0.0,
        "a_to_b_actor_nose_speed_cm_s": 0.0,
        "a_to_b_target_nose_speed_cm_s": 0.0,
        "a_to_b_actor_angular_speed_deg_s": 0.0,
        "a_to_b_target_angular_speed_deg_s": 0.0,
        "a_to_b_actor_body_length_cm": np.nan,
        "a_to_b_target_body_length_cm": np.nan,
        "a_to_b_actor_pose_deformation_energy": 0.0,
        "a_to_b_target_pose_deformation_energy": 0.0,
        "a_to_b_center_distance_body_lengths": np.nan,
        "a_to_b_closing_speed_cm_s": 0.0,
        "a_to_b_actor_head_relative_speed_cm_s": 0.0,
        "a_to_b_target_head_relative_speed_cm_s": 0.0,
        "a_to_b_direction_similarity": 0.0,
        "a_to_b_pursuit_alignment": 0.0,
        "a_to_b_target_escape_alignment": 0.0,
        "a_to_b_behind_score": 0.0,
        "a_to_b_actor_behind_target": False,
        "a_to_b_trajectory_correlation": 0.0,
        "a_to_b_target_turn_angle_deg": 0.0,
        "a_to_b_nose_head_distance_cm": np.nan,
        "a_to_b_nose_body_distance_cm": np.nan,
        "a_to_b_nose_tail_distance_cm": np.nan,
        "b_to_a_actor_speed_cm_s": 0.0,
        "b_to_a_target_speed_cm_s": 0.0,
        "b_to_a_actor_acceleration_cm_s2": 0.0,
        "b_to_a_target_acceleration_cm_s2": 0.0,
        "b_to_a_actor_nose_speed_cm_s": 0.0,
        "b_to_a_target_nose_speed_cm_s": 0.0,
        "b_to_a_actor_angular_speed_deg_s": 0.0,
        "b_to_a_target_angular_speed_deg_s": 0.0,
        "b_to_a_actor_body_length_cm": np.nan,
        "b_to_a_target_body_length_cm": np.nan,
        "b_to_a_actor_pose_deformation_energy": 0.0,
        "b_to_a_target_pose_deformation_energy": 0.0,
        "b_to_a_center_distance_body_lengths": np.nan,
        "b_to_a_closing_speed_cm_s": 0.0,
        "b_to_a_actor_head_relative_speed_cm_s": 0.0,
        "b_to_a_target_head_relative_speed_cm_s": 0.0,
        "b_to_a_direction_similarity": 0.0,
        "b_to_a_pursuit_alignment": 0.0,
        "b_to_a_target_escape_alignment": 0.0,
        "b_to_a_behind_score": 0.0,
        "b_to_a_actor_behind_target": False,
        "b_to_a_trajectory_correlation": 0.0,
        "b_to_a_target_turn_angle_deg": 0.0,
        "b_to_a_nose_head_distance_cm": np.nan,
        "b_to_a_nose_body_distance_cm": np.nan,
        "b_to_a_nose_tail_distance_cm": np.nan,
        "selected_actor_wall_jump": False,
        "selected_target_wall_jump": False,
        "pair_wall_jump_excluded": False,
        "selected_actor_id": np.nan,
        "selected_target_id": np.nan,
        "selected_nose_body_distance_cm": np.nan,
        "selected_target_turn_angle_deg": 0.0,
        "selected_distance_drop_cm": 0.0,
        "selected_actor_speed_cm_s": 0.0,
        "selected_target_speed_cm_s": 0.0,
        "selected_weak_chase_score": 0,
        "selected_strong_chase_score": 0,
        "selected_weak_attack_evidence": 0,
        "selected_strong_attack_evidence": 0,
        "a_to_b_weak_chase": False,
        "b_to_a_weak_chase": False,
        "a_to_b_strong_chase": False,
        "b_to_a_strong_chase": False,
        "a_to_b_weak_attack": False,
        "b_to_a_weak_attack": False,
        "a_to_b_strong_attack": False,
        "b_to_a_strong_attack": False,
        "weak_contact": False,
        "strong_contact": False,
        "weak_potential_attack": False,
        "strong_potential_attack": False,
        "weak_attack_actor_initiation": False,
        "strong_attack_actor_initiation": False,
        "weak_attack_target_reaction": False,
        "strong_attack_target_reaction": False,
        "repeated_contact_count": 0,
        "weak_raw_chase": False,
        "weak_raw_attack": False,
        "weak_raw_label_id": 0,
        "strong_raw_chase": False,
        "strong_raw_attack": False,
        "strong_raw_label_id": 0,
        "pose_pair_quality": 0.0,
        "cluster_attack_hint": False,
        # 以下字段只保存遮挡管理器已经计算好的攻击物证，不反向修改检测或身份结果。
        "cluster_detection_deficit": False,
        "cluster_merged_like": False,
        "cluster_overlap_iou": 0.0,
        "cluster_motion_bl_per_frame": 0.0,
        "cluster_active_frames": 0,
        "cluster_expected_count": 0,
        "cluster_observed_count": 0,
        # v1.22：聚集期间身份未决时保留群体事件候选，但不得把个体归属当真值。
        "identity_ambiguous": False,
        "identity_candidate_set": "",
    }
    return record


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _cluster_attack_evidence_by_pair(
    cluster_context: Optional[Mapping[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """把现有遮挡簇物证展开为鼠对级攻击输入，不改变遮挡或身份状态。"""
    # 没有遮挡上下文时直接返回空映射，保证普通双鼠行为路径完全不变。
    if not cluster_context:
        return {}
    # 每个正式ID鼠对只保留当前帧物证最强的遮挡簇，避免重复区域产生重复记录。
    evidence_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    # 遍历遮挡管理器已经生成的区域；这里不重新计算检测框或轨迹。
    for region in cluster_context.get("regions", []):
        # 只有底层已标记为攻击提示的区域才允许进入攻击候选链路。
        if not bool(region.get("attack_hint", False)):
            continue
        # 过滤匿名和无效ID，并排序去重，使鼠对键在所有帧中保持稳定。
        members = sorted({int(v) for v in region.get("members", []) if int(v) >= 0})
        # 少于两只正式小鼠时不存在可归属的鼠对攻击。
        if len(members) < 2:
            continue
        # 将遮挡簇原始字段复制成只读的攻击证据列，供离线时序门严格复核。
        pair_evidence: Dict[str, Any] = {
            "cluster_attack_hint": True,
            "cluster_detection_deficit": bool(region.get("deficit", False)),
            "cluster_merged_like": bool(region.get("merged_like", False)),
            "cluster_overlap_iou": safe_float(region.get("max_iou"), 0.0),
            "cluster_motion_bl_per_frame": safe_float(
                region.get("motion_bl_per_frame"), 0.0
            ),
            "cluster_active_frames": int(region.get("active_frames", 0)),
            "cluster_expected_count": int(region.get("expected_count", len(members))),
            "cluster_observed_count": int(region.get("observed_count", len(members))),
        }
        # 优先保留“确实少检、出现合并、重叠更深、运动更快、持续更久”的区域。
        score = (
            int(pair_evidence["cluster_detection_deficit"]),
            int(pair_evidence["cluster_merged_like"]),
            float(pair_evidence["cluster_overlap_iou"]),
            float(pair_evidence["cluster_motion_bl_per_frame"]),
            int(pair_evidence["cluster_active_frames"]),
        )
        # 多鼠簇仍按两两组合保存审计证据；最终攻击门会严格要求簇内恰好两只鼠。
        for id_a, id_b in combinations(members, 2):
            # 统一按升序生成键，避免A-B与B-A重复。
            pair = tuple(sorted((int(id_a), int(id_b))))
            # 读取同一鼠对先前区域的强度，若不存在则直接接纳当前区域。
            previous = evidence_by_pair.get(pair)
            # 构造与当前区域相同顺序的比较分数，保持选择规则可重复。
            previous_score = None if previous is None else (
                int(previous["cluster_detection_deficit"]),
                int(previous["cluster_merged_like"]),
                float(previous["cluster_overlap_iou"]),
                float(previous["cluster_motion_bl_per_frame"]),
                int(previous["cluster_active_frames"]),
            )
            # 只替换为证据更强的区域，绝不修改遮挡管理器自己的状态。
            if previous_score is None or score > previous_score:
                evidence_by_pair[pair] = dict(pair_evidence)
    # 返回鼠对级只读快照，供当前帧行为记录写入SQLite。
    return evidence_by_pair


def _apply_document_chase_gate(
    output: pd.DataFrame, mask: np.ndarray, fps: float, cfg: Mapping[str, Any]
) -> np.ndarray:
    """事件级落实PPT：同向、<30 cm、相关性>0.7、双方各移动>=60 cm。"""
    accepted = np.zeros(len(output), dtype=bool)
    for start, end in _true_runs(mask):
        seg = output.iloc[start : end + 1]
        path_a = float(pd.to_numeric(seg["mouse_a_speed_cm_s"], errors="coerce").fillna(0).sum() / fps)
        path_b = float(pd.to_numeric(seg["mouse_b_speed_cm_s"], errors="coerce").fillna(0).sum() / fps)
        corr = pd.to_numeric(seg["trajectory_correlation"], errors="coerce").fillna(-1.0)
        direction = pd.to_numeric(seg["direction_similarity"], errors="coerce").fillna(-1.0)
        distance = pd.to_numeric(seg["center_distance_cm"], errors="coerce").fillna(np.inf)
        wall = seg["pair_wall_jump_excluded"].fillna(False).astype(bool)
        role = pd.to_numeric(seg["selected_actor_id"], errors="coerce").dropna().astype(int)
        role_consistency = float(role.value_counts(normalize=True).max()) if len(role) else 0.0
        ok = bool(
            path_a >= float(cfg.get("min_path_cm", 60.0))
            and path_b >= float(cfg.get("min_path_cm", 60.0))
            and float((corr >= float(cfg["trajectory_correlation_min"])).mean())
                >= float(cfg.get("min_correlation_fraction", 0.70))
            and float((direction >= float(cfg["direction_similarity_min"])).mean())
                >= float(cfg.get("min_same_direction_fraction", 0.70))
            and float((distance <= float(cfg["max_distance_cm"])).mean())
                >= float(cfg.get("min_proximity_fraction", 0.80))
            and role_consistency >= float(cfg.get("min_role_consistency", 0.65))
            and float(wall.mean()) <= float(cfg.get("max_wall_jump_fraction", 0.0))
        )
        if ok:
            accepted[start : end + 1] = True
    return accepted


def _apply_window_chase_gate(
    output: pd.DataFrame,
    fps: float,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Detect short pursuit bouts from window-level evidence.

    Requiring every chase condition on every frame fragments real pursuit into
    a few isolated frames.  This gate integrates the evidence over a short
    window, while requiring asymmetric actor/target motion so that parallel
    locomotion is not promoted to chase.
    """
    accepted = np.zeros(len(output), dtype=bool)
    if not bool(cfg.get("window_gate_enabled", True)) or len(output) == 0:
        return accepted

    window_frames = max(int(round(float(cfg.get("window_seconds", 1.0)) * fps)), 2)
    min_valid = float(cfg.get("window_min_valid_fraction", 0.80))

    valid = output["valid_pair"].fillna(False).astype(bool).to_numpy()
    distance = pd.to_numeric(output["center_distance_cm"], errors="coerce").to_numpy(dtype=float)
    actor_speed = pd.to_numeric(
        output["selected_actor_speed_cm_s"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    target_speed = pd.to_numeric(
        output["selected_target_speed_cm_s"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    direction = pd.to_numeric(
        output["direction_similarity"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    pursuit = pd.to_numeric(
        output["pursuit_alignment"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    escape = pd.to_numeric(
        output["target_escape_alignment"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    wall = output["pair_wall_jump_excluded"].fillna(False).astype(bool).to_numpy()
    actor_id = pd.to_numeric(output["selected_actor_id"], errors="coerce").to_numpy(dtype=float)

    min_speed = float(cfg.get("window_min_speed_cm_s", 5.0))
    max_distance = float(cfg["max_distance_cm"])
    same_threshold = float(cfg.get("window_direction_similarity_min", 0.60))
    pursuit_threshold = float(cfg.get("window_pursuit_alignment_min", 0.35))
    escape_threshold = float(cfg.get("window_target_escape_alignment_min", 0.25))

    for start in range(0, max(len(output) - window_frames + 1, 0)):
        end = start + window_frames
        sl = slice(start, end)
        usable = valid[sl] & np.isfinite(distance[sl])
        if float(usable.mean()) < min_valid:
            continue

        def fraction(mask_values: np.ndarray) -> float:
            return float(np.mean(mask_values[usable])) if np.any(usable) else 0.0

        proximity_fraction = fraction(distance[sl] <= max_distance)
        moving_fraction = fraction(
            (actor_speed[sl] >= min_speed) & (target_speed[sl] >= min_speed)
        )
        same_direction_fraction = fraction(direction[sl] >= same_threshold)
        pursuit_fraction = fraction(pursuit[sl] >= pursuit_threshold)
        escape_fraction = fraction(escape[sl] >= escape_threshold)
        wall_fraction = fraction(wall[sl])

        actor_path = float(np.sum(actor_speed[sl][usable]) / fps)
        target_path = float(np.sum(target_speed[sl][usable]) / fps)
        min_path = min(actor_path, target_path)
        path_ratio = max(actor_path, target_path) / max(min_path, 1e-6)
        distance_values = distance[sl][usable]
        distance_change = (
            float(distance_values[-1] - distance_values[0])
            if len(distance_values) >= 2
            else 0.0
        )

        roles = actor_id[sl][usable]
        roles = roles[np.isfinite(roles)]
        if len(roles):
            _, counts = np.unique(roles.astype(int), return_counts=True)
            role_consistency = float(np.max(counts) / len(roles))
        else:
            role_consistency = 0.0

        asymmetric_motion = bool(
            path_ratio >= float(cfg.get("window_min_path_ratio", 1.20))
            and abs(distance_change) >= float(cfg.get("window_min_distance_change_cm", 1.0))
        )
        sustained_motion = bool(
            min_path >= float(cfg.get("window_sustained_min_path_cm", 30.0))
            and same_direction_fraction
            >= float(cfg.get("window_sustained_same_direction_fraction", 0.70))
            and pursuit_fraction
            >= float(cfg.get("window_sustained_pursuit_fraction", 0.80))
            and escape_fraction
            >= float(cfg.get("window_sustained_escape_fraction", 0.75))
        )

        ok = bool(
            proximity_fraction >= float(cfg.get("window_min_proximity_fraction", 0.80))
            and moving_fraction >= float(cfg.get("window_min_moving_fraction", 0.55))
            and same_direction_fraction
            >= float(cfg.get("window_min_same_direction_fraction", 0.65))
            and pursuit_fraction >= float(cfg.get("window_min_pursuit_fraction", 0.70))
            and escape_fraction >= float(cfg.get("window_min_escape_fraction", 0.70))
            and min_path >= float(cfg.get("window_min_path_cm", 10.0))
            and role_consistency >= float(cfg.get("window_min_role_consistency", 0.60))
            and wall_fraction <= float(cfg.get("max_wall_jump_fraction", 0.0))
            and (asymmetric_motion or sustained_motion)
        )
        if ok:
            # A qualifying window supplies temporal context, but it must not label
            # any current frame whose two mouse centers are already too far apart.
            current_frame_is_near = usable & (distance[sl] <= max_distance)
            # Keep only the close portion of the window so a distant approach is
            # not rendered as chase merely because later frames become close.
            accepted[start:end] |= current_frame_is_near
    return accepted


def _attack_occlusion_overlap_mask(
    output: pd.DataFrame,
    fps: float,
    cfg: Mapping[str, Any],
    level: str,
) -> np.ndarray:
    """确认“框重叠/合并、少检、快速运动、恢复”组成的遮挡型攻击。"""
    # 该分支可按弱/强配置独立关闭；关闭时返回全False以保持旧攻击行为。
    if not bool(cfg.get("occlusion_overlap_gate_enabled", True)) or len(output) == 0:
        return np.zeros(len(output), dtype=bool)

    # 兼容没有新增列的旧行为缓存；缺列必须视为没有物证，不能凭默认值制造攻击。
    def bool_column(name: str, default: bool = False) -> np.ndarray:
        # 旧缓存不存在该列时生成与输入等长的固定布尔数组。
        if name not in output.columns:
            return np.full(len(output), bool(default), dtype=bool)
        # SQLite中的0/1和CSV中的空值统一转换成逐帧布尔值。
        return output[name].fillna(default).astype(bool).to_numpy()

    # 数值列同样提供显式默认值，并把非法文本或空值安全转换为有限浮点数。
    def float_column(name: str, default: float = 0.0) -> np.ndarray:
        # 旧缓存缺列时不能借用其他模块的信息，直接返回配置指定的安全默认值。
        if name not in output.columns:
            return np.full(len(output), float(default), dtype=float)
        # 行为门只使用有限数值；无法解析的值回退到默认值。
        return (
            pd.to_numeric(output[name], errors="coerce")
            .fillna(float(default))
            .to_numpy(dtype=float)
        )

    # 有效鼠对包括真实双框行和遮挡管理器为同一正式ID鼠对补出的候选行。
    valid_pair = bool_column("valid_pair")
    # 只有底层遮挡管理器已经形成攻击提示的帧才进入严格复核。
    hint = bool_column("cluster_attack_hint")
    # 少检必须真实发生，排除两只完整可见时的普通框重叠。
    deficit = bool_column("cluster_detection_deficit")
    # 单个大框覆盖两只鼠是重叠的替代物证，适用于其中一框已经消失的帧。
    merged_like = bool_column("cluster_merged_like")
    # 最大框交并比来自现有遮挡簇，不重新运行检测器或改变追踪结果。
    overlap_iou = float_column("cluster_overlap_iou")
    # 簇内运动按身体长度/帧归一化，避免分辨率不同导致固定像素阈值失真。
    cluster_motion = float_column("cluster_motion_bl_per_frame")
    # 遮挡簇持续帧数用于排除单帧检测抖动。
    active_frames = float_column("cluster_active_frames")
    # 期望与实际检测数明确区分“2变1”和普通框尺寸波动。
    expected_count = float_column("cluster_expected_count")
    # 实际检测数必须小于期望数，避免仅靠布尔标志进入攻击分支。
    observed_count = float_column("cluster_observed_count")
    # 中心距离有限表示当前帧两只正式ID均有可用独立观测。
    center_distance = float_column("center_distance_cm", float("nan"))
    # NaN默认值需要单独保留，不能被float_column的fillna转换为普通数字。
    if "center_distance_cm" in output.columns:
        center_distance = pd.to_numeric(
            output["center_distance_cm"], errors="coerce"
        ).to_numpy(dtype=float)
    # 真实双鼠观测用于确认遮挡前接触和遮挡后恢复。
    observed_pair = valid_pair & np.isfinite(center_distance)
    # 鼻端接触是最直接的遮挡前攻击几何物证。
    potential_contact = bool_column(f"{level}_potential_attack")
    # 施动者主动接近物证用于排除高速交叉经过。
    actor_initiation = bool_column(f"{level}_attack_actor_initiation")
    # 受动者逃逸或反应物证用于排除双方无因果关系的并行移动。
    target_reaction = bool_column(f"{level}_attack_target_reaction")
    # 已有动态攻击证据继续作为上下文约束，不改变原有证据计算方式。
    dynamic_evidence = float_column(f"selected_{level}_attack_evidence")
    # 双方中心速度只用于遮挡前后可见帧的快速运动复核。
    actor_speed = float_column("selected_actor_speed_cm_s")
    # 目标速度与施动者速度相加，形成与方向无关的运动强度。
    target_speed = float_column("selected_target_speed_cm_s")
    # 墙跳标志来自现有个体行为排除门，防止墙边快速动作被提升为攻击。
    wall_jump = bool_column("pair_wall_jump_excluded")
    # 角色一致性只用于决定能否可靠归属发起者，不用ID大小猜测方向。
    actor_id = float_column("selected_actor_id", float("nan"))
    # 与中心距离相同，角色ID必须保留NaN以表示遮挡帧中方向未知。
    if "selected_actor_id" in output.columns:
        actor_id = pd.to_numeric(
            output["selected_actor_id"], errors="coerce"
        ).to_numpy(dtype=float)

    # 框重叠或覆盖两鼠的大框至少满足一项，且阈值按弱/强级别配置。
    overlap_or_merged = merged_like | (
        overlap_iou >= float(cfg.get("occlusion_min_overlap_iou", 0.12))
    )
    # 默认严格限定两鼠簇，防止三鼠以上聚集被两两扩散成多条攻击事件。
    exact_pair_size = expected_count == 2 if bool(
        cfg.get("occlusion_require_exact_pair_size", True)
    ) else expected_count >= 2
    # 原始候选必须同时具备提示、少检、重叠/合并、快速运动和持续帧数。
    raw_candidate = (
        valid_pair
        & hint
        & deficit
        & (observed_count < expected_count)
        & exact_pair_size
        & overlap_or_merged
        & (
            cluster_motion
            >= float(cfg.get("occlusion_min_motion_body_lengths_per_frame", 0.08))
        )
        & (active_frames >= int(cfg.get("occlusion_min_active_frames", 2)))
    )
    # 允许频繁消失/出现之间存在很短的双框恢复帧，但不跨越一次完整分离。
    gap_frames = max(
        int(round(float(cfg.get("occlusion_fill_gap_seconds", 0.20)) * fps)),
        0,
    )
    # 至少两帧严格物证，避免单帧YOLO抖动触发攻击。
    minimum_frames = max(int(cfg.get("occlusion_min_evidence_frames", 2)), 1)
    # 时间滤波只合并同一鼠对的短间歇，不会跨鼠对或跨大段缺失。
    candidate = base.temporal_filter(raw_candidate, minimum_frames, gap_frames)
    # 默认全部拒绝，只有遮挡前后物理顺序完整的段才进入最终攻击。
    accepted = np.zeros(len(output), dtype=bool)
    # 将上下文秒数转换为帧数，适配不同帧率视频。
    pre_context_frames = max(
        int(round(float(cfg.get("occlusion_pre_context_seconds", 0.50)) * fps)),
        1,
    )
    # 恢复窗口允许扭打结束后短暂分离，但不能连接很久以后的再次接触。
    recovery_context_frames = max(
        int(round(float(cfg.get("occlusion_recovery_context_seconds", 1.00)) * fps)),
        1,
    )
    # 过长的单鼠漏检更可能是持续检测失败，限制其直接形成遮挡型攻击。
    max_hidden_frames = max(
        int(round(float(cfg.get("occlusion_max_hidden_seconds", 3.00)) * fps)),
        minimum_frames,
    )
    # 遮挡前中心距离门允许框已经相交但鼻尾关键点暂时不完整。
    pre_contact_distance = float(
        cfg.get("occlusion_pre_contact_center_distance_cm", 8.0)
    )
    # 遮挡后第一帧可稍远，用于覆盖攻击结束后的快速分开。
    recovery_distance = float(
        cfg.get("occlusion_recovery_center_distance_cm", 12.0)
    )
    # 上下文至少包含一帧已有攻击方向/动态物证。
    minimum_dynamic_evidence = float(
        cfg.get("occlusion_min_context_dynamic_evidence", 1)
    )
    # 高速上下文门与簇内运动门同时存在，确保画面内外两种运动证据一致。
    minimum_combined_speed = float(
        cfg.get("occlusion_min_context_combined_speed_cm_s", 20.0)
    )
    # 强候选可要求更多上下文动态帧，弱候选默认一帧即可。
    minimum_dynamic_frames = max(
        int(cfg.get("occlusion_min_context_dynamic_frames", 1)),
        1,
    )

    # 每一段独立验证，防止相隔较远的遮挡被合并成一次攻击。
    for run_start, run_end in _true_runs(candidate):
        # 找到当前时间段内真正同时满足全部物证的帧，而不是时间补洞帧。
        strict_indices = np.flatnonzero(raw_candidate[run_start : run_end + 1]) + run_start
        # 补洞后的长段仍必须包含足够多的真实物证帧。
        if len(strict_indices) < minimum_frames:
            continue
        # 严格段起点用于寻找紧邻其前的最后一帧双鼠观测。
        strict_start = int(strict_indices[0])
        # 严格段终点用于寻找紧邻其后的第一帧双鼠恢复。
        strict_end = int(strict_indices[-1])
        # 单次遮挡超过上限时不由本分支直接定性，避免把长期检测故障当攻击。
        if strict_end - strict_start + 1 > max_hidden_frames:
            continue
        # 只在配置的短前窗中搜索实际双框观测。
        pre_range = np.arange(max(0, strict_start - pre_context_frames), strict_start)
        # 遮挡前必须存在双鼠可见帧，否则无法证明是由接触进入遮挡。
        pre_observed = pre_range[observed_pair[pre_range]] if len(pre_range) else np.array([], dtype=int)
        # 没有遮挡前观测时拒绝，不能从视频开头的单框状态猜测攻击。
        if len(pre_observed) == 0:
            continue
        # 只采用最近的前帧，防止借用更早的一次无关接触。
        pre_index = int(pre_observed[-1])
        # 最近前帧必须鼻端接触或中心足够近，普通远距离漏检不能通过。
        if not (
            bool(potential_contact[pre_index])
            or float(center_distance[pre_index]) <= pre_contact_distance
        ):
            continue
        # 墙跳前帧直接拒绝，避免墙边快速攀爬进入遮挡攻击分支。
        if bool(wall_jump[pre_index]):
            continue
        # 在限定恢复窗中查找遮挡后的实际双框观测。
        post_range = np.arange(
            strict_end + 1,
            min(len(output), strict_end + recovery_context_frames + 1),
        )
        # 第一帧双鼠恢复比稍后再次靠近更能证明是同一次遮挡事件。
        post_observed = post_range[observed_pair[post_range]] if len(post_range) else np.array([], dtype=int)
        # 没有恢复帧时保留为审计提示，但不写入最终攻击标签。
        if len(post_observed) == 0:
            continue
        # 只检查最早恢复帧，禁止跳过一次远距离恢复后连接后续重新接触。
        post_index = int(post_observed[0])
        # 恢复帧必须仍在同一局部交互范围内。
        if float(center_distance[post_index]) > recovery_distance:
            continue
        # 恢复帧若为墙跳同样拒绝，保持现有墙边排除规则。
        if bool(wall_jump[post_index]):
            continue
        # 遮挡前帧、严格物证段和恢复帧共同组成一次待确认攻击上下文。
        context_slice = slice(pre_index, post_index + 1)
        # 动态证据只统计双鼠实际可见帧，不让合成遮挡行贡献速度或方向。
        visible_context = observed_pair[context_slice]
        # 理论上至少包含前后两帧；防御性检查避免异常缓存进入计算。
        if not np.any(visible_context):
            continue
        # 已有攻击证据、主动接近或目标反应任一成立，才属于攻击方向物证。
        attack_oriented = (
            (dynamic_evidence[context_slice] >= minimum_dynamic_evidence)
            | actor_initiation[context_slice]
            | target_reaction[context_slice]
        )
        # 双鼠合计速度验证用户观察到的“重叠同时相对快速移动”。
        rapid_context = (
            actor_speed[context_slice] + target_speed[context_slice]
        ) >= minimum_combined_speed
        # 每一帧上下文动态必须来自真实双鼠观测。
        dynamic_context = visible_context & (attack_oriented | rapid_context)
        # 动态帧不足时视为安静贴靠或普通嗅探，不提升为攻击。
        if int(dynamic_context.sum()) < minimum_dynamic_frames:
            continue
        # 默认至少需要一帧有攻击方向的旧证据；纯高速交叉不能只靠速度通过。
        if bool(cfg.get("occlusion_require_attack_oriented_context", True)) and not bool(
            np.any(visible_context & attack_oriented)
        ):
            continue
        # 墙跳比例只在真实双鼠上下文上计算，合成遮挡行不能稀释比例。
        visible_walls = wall_jump[context_slice][visible_context]
        # 超过允许比例时拒绝，避免墙边跳跃与影子干扰形成假攻击。
        if len(visible_walls) and float(visible_walls.mean()) > float(
            cfg.get("occlusion_max_wall_jump_fraction", 0.50)
        ):
            continue
        # 使用可见帧已有角色判断一致性；遮挡帧绝不按ID大小猜发起者。
        visible_roles = actor_id[context_slice][visible_context]
        # 清除NaN后再计算多数角色占比。
        visible_roles = visible_roles[np.isfinite(visible_roles)]
        # 有角色物证时要求其达到配置的一致性；完全未知时仍可确认“这对鼠发生攻击”。
        if len(visible_roles):
            # 统计上下文中出现次数最多的发起者ID。
            _, role_counts = np.unique(visible_roles.astype(int), return_counts=True)
            # 角色频繁翻转说明方向不可靠，强候选将被更严格地拒绝。
            if float(np.max(role_counts) / len(visible_roles)) < float(
                cfg.get("occlusion_min_role_consistency", 0.50)
            ):
                continue
        # 通过全部门控后回填从最后接触帧到第一恢复帧，保持一次攻击事件连续。
        accepted[pre_index : post_index + 1] = valid_pair[pre_index : post_index + 1]
    # 返回独立攻击掩码，由总攻击结果合并；追逐和其他模块不会读取该列。
    return accepted


def _attack_impulse_mask(
    output: pd.DataFrame,
    cfg: Mapping[str, Any],
    level: str,
) -> np.ndarray:
    """Keep brief, high-energy contacts that cannot satisfy a multi-frame bout."""
    if not bool(cfg.get("impulse_gate_enabled", True)) or len(output) == 0:
        return np.zeros(len(output), dtype=bool)

    potential = output[f"{level}_potential_attack"].fillna(False).astype(bool).to_numpy()
    initiation = output[f"{level}_attack_actor_initiation"].fillna(False).astype(bool).to_numpy()
    reaction = output[f"{level}_attack_target_reaction"].fillna(False).astype(bool).to_numpy()
    evidence = pd.to_numeric(
        output[f"selected_{level}_attack_evidence"], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    actor_speed = pd.to_numeric(
        output["selected_actor_speed_cm_s"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    target_speed = pd.to_numeric(
        output["selected_target_speed_cm_s"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    pose_quality = pd.to_numeric(
        output["pose_pair_quality"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    nose_body = pd.to_numeric(
        output["selected_nose_body_distance_cm"], errors="coerce"
    ).to_numpy(dtype=float)
    center_distance = pd.to_numeric(
        output["center_distance_cm"], errors="coerce"
    ).to_numpy(dtype=float)
    wall = output["pair_wall_jump_excluded"].fillna(False).astype(bool).to_numpy()

    return (
        potential
        & np.isfinite(center_distance)
        & (
            center_distance
            <= float(cfg.get("impulse_center_contact_distance_cm", 6.0))
        )
        & np.isfinite(nose_body)
        & (
            nose_body
            <= float(cfg.get("impulse_body_contact_distance_cm", 4.0))
        )
        & initiation
        & reaction
        & (evidence >= int(cfg.get("impulse_min_dynamic_evidence", cfg["min_dynamic_evidence"])))
        & (
            (actor_speed + target_speed)
            >= float(cfg.get("impulse_min_combined_speed_cm_s", 80.0))
        )
        & (pose_quality >= float(cfg.get("impulse_min_pose_pair_quality", 0.50)))
        & (~wall)
    )


def _near_chase_recovery_mask(
    output: pd.DataFrame,
    fps: float,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Recover close pursuit split by an ID jump without reopening far chase."""
    # This gate is optional because it specifically addresses unstable tracking near contact.
    if not bool(cfg.get("near_recovery_gate_enabled", True)) or len(output) == 0:
        return np.zeros(len(output), dtype=bool)
    # Every accepted frame must retain a valid two-mouse observation.
    valid = output["valid_pair"].fillna(False).astype(bool).to_numpy()
    # Use center distance as the hard physical radius for this recovery path.
    distance = pd.to_numeric(
        output["center_distance_cm"], errors="coerce"
    ).to_numpy(dtype=float)
    # Read both speeds independently so a stationary neighbour cannot be promoted.
    actor_speed = pd.to_numeric(
        output["selected_actor_speed_cm_s"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    # Target motion is required because pursuit is relational, not one-sided locomotion.
    target_speed = pd.to_numeric(
        output["selected_target_speed_cm_s"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    # Direction similarity rejects crossing and opposing trajectories.
    direction = pd.to_numeric(
        output["direction_similarity"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    # Pursuit alignment requires the actor to advance toward the target.
    pursuit = pd.to_numeric(
        output["pursuit_alignment"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    # Escape alignment requires the target to move away from the actor.
    escape = pd.to_numeric(
        output["target_escape_alignment"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    # Correlation requires both tracks to share a sustained route.
    correlation = pd.to_numeric(
        output["trajectory_correlation"], errors="coerce"
    ).fillna(-1.0).to_numpy(dtype=float)
    # Stable actor direction prevents alternating nearest-pair assignments from passing.
    actor_id = pd.to_numeric(
        output["selected_actor_id"], errors="coerce"
    ).to_numpy(dtype=float)
    # Convert the allowed short score gap from seconds to frames.
    gap_frames = max(
        int(round(float(cfg.get("near_recovery_fill_gap_seconds", 0.10)) * fps)),
        0,
    )
    # Require at least four tenths of a second of close directional pursuit.
    minimum_frames = max(
        int(math.ceil(float(cfg.get("near_recovery_min_duration_seconds", 0.40)) * fps)),
        1,
    )
    # Build a strict near-range candidate without trusting the noisy wall-jump flag.
    raw_candidate = (
        valid
        & np.isfinite(distance)
        & (distance <= float(cfg.get("near_recovery_max_distance_cm", 10.0)))
        & (actor_speed >= float(cfg.get("near_recovery_min_speed_cm_s", 5.0)))
        & (target_speed >= float(cfg.get("near_recovery_min_speed_cm_s", 5.0)))
        & (direction >= float(cfg.get("near_recovery_direction_similarity_min", 0.75)))
        & (pursuit >= float(cfg.get("near_recovery_pursuit_alignment_min", 0.55)))
        & (escape >= float(cfg.get("near_recovery_escape_alignment_min", 0.35)))
        & (correlation >= float(cfg.get("near_recovery_correlation_min", 0.75)))
    )
    # Fill only very short pose/identity interruptions before measuring duration.
    candidate = base.temporal_filter(raw_candidate, minimum_frames, gap_frames)
    # Start from rejection and validate each recovered bout independently.
    accepted = np.zeros(len(output), dtype=bool)
    # Independent runs cannot borrow actor consistency or speed from one another.
    for start, end in _true_runs(candidate):
        # Limit aggregate checks to frames with actual directional evidence.
        evidence_rows = raw_candidate[start : end + 1]
        # A filled interval without enough real evidence cannot form a pursuit.
        if int(evidence_rows.sum()) < minimum_frames:
            continue
        # Measure actor stability only on the real evidence frames.
        roles = actor_id[start : end + 1][evidence_rows]
        # Remove missing actor IDs before calculating the majority fraction.
        roles = roles[np.isfinite(roles)]
        # No actor identity means there is no defensible chase direction.
        if len(roles) == 0:
            continue
        # Count the dominant actor within this short physical interaction.
        _, counts = np.unique(roles.astype(int), return_counts=True)
        # Convert the dominant count to a role-consistency fraction.
        role_consistency = float(np.max(counts) / len(roles))
        # Reject windows whose actor/target direction flips repeatedly.
        if role_consistency < float(cfg.get("near_recovery_min_role_consistency", 0.75)):
            continue
        # Require substantial combined locomotion, excluding slow close investigation.
        combined_speed = (
            actor_speed[start : end + 1][evidence_rows]
            + target_speed[start : end + 1][evidence_rows]
        )
        # Slow contact is handled by attack/contact logic rather than chase recovery.
        if float(np.mean(combined_speed)) < float(
            cfg.get("near_recovery_min_mean_combined_speed_cm_s", 25.0)
        ):
            continue
        # Retain only frames that themselves satisfy the stricter recovery range.
        # Gap filling may establish bout continuity, but may not create a distant
        # chase label at a frame without close-range physical evidence.
        accepted[start : end + 1] = raw_candidate[start : end + 1]
    # Return a frame-aligned mask for union with the existing strict chase gates.
    return accepted


def _close_follow_chase_mask(
    output: pd.DataFrame,
    fps: float,
    cfg: Mapping[str, Any],
    level: str,
) -> np.ndarray:
    """识别高速同向且长期保持前后关系的贴近跟随型追逐。"""
    # 该补充门专门覆盖两鼠速度接近、传统路径比条件不成立的追逐。
    if not bool(cfg.get("close_follow_gate_enabled", True)) or len(output) == 0:
        return np.zeros(len(output), dtype=bool)
    # 近距离跟随仍要求鼻端持续接近目标身体，避免平行路过触发。
    contact = (
        output[f"{level}_potential_attack"].fillna(False).astype(bool).to_numpy()
    )
    # 无有效身份或尺度的鼠对不能进入定向追逐判别。
    valid = output["valid_pair"].fillna(False).astype(bool).to_numpy()
    # 允许少量骨架漏检间隙，但不跨越明显分离段连接事件。
    gap_frames = max(
        int(round(float(cfg.get("close_follow_fill_gap_seconds", 0.15)) * fps)),
        0,
    )
    # 持续时间门排除一次擦肩或短促冲击。
    min_frames = max(
        int(
            math.ceil(
                float(cfg.get("close_follow_min_duration_seconds", 0.75)) * fps
            )
        ),
        1,
    )
    # 先建立连续接触候选，再逐段验证方向、速度和施受关系。
    candidate = base.temporal_filter(contact & valid, min_frames, gap_frames)
    # 默认全部拒绝，只有完整通过追逐物证的连续段才会标记。
    accepted = np.zeros(len(output), dtype=bool)
    # 每一段独立计算，避免不同鼠对状态在时间上相互污染。
    for start, end in _true_runs(candidate):
        # 取出当前连续段以及具备原始接近证据的有效行。
        segment = output.iloc[start : end + 1]
        contact_rows = segment[
            segment[f"{level}_potential_attack"].fillna(False).astype(bool)
        ]
        # 时间补洞产生的空段不能单独形成追逐。
        if contact_rows.empty:
            continue
        # 组合速度区分移动追逐与低速骑跨/扭打。
        combined_speed = (
            pd.to_numeric(
                contact_rows["selected_actor_speed_cm_s"], errors="coerce"
            ).fillna(0.0)
            + pd.to_numeric(
                contact_rows["selected_target_speed_cm_s"], errors="coerce"
            ).fillna(0.0)
        )
        # 中心距离要求两鼠贴近但不完全重合。
        center_distance = pd.to_numeric(
            contact_rows["center_distance_cm"], errors="coerce"
        ).fillna(np.inf)
        # 同向比例反映两鼠是否沿近似一致的路线移动。
        same_direction = (
            pd.to_numeric(
                contact_rows["direction_similarity"], errors="coerce"
            ).fillna(-1.0)
            >= float(cfg.get("close_follow_direction_similarity_min", 0.70))
        )
        # 施动者朝向目标是追赶而不是随机并行的必要条件。
        pursuit = (
            pd.to_numeric(
                contact_rows["pursuit_alignment"], errors="coerce"
            ).fillna(-1.0)
            >= float(cfg.get("close_follow_pursuit_alignment_min", 0.35))
        )
        # 受动者远离施动者用于确认被追逐关系。
        escape = (
            pd.to_numeric(
                contact_rows["target_escape_alignment"], errors="coerce"
            ).fillna(-1.0)
            >= float(cfg.get("close_follow_escape_alignment_min", 0.25))
        )
        # 短窗轨迹相关性过滤彼此无关的两条相交路径。
        correlated = (
            pd.to_numeric(
                contact_rows["trajectory_correlation"], errors="coerce"
            ).fillna(-1.0)
            >= float(cfg.get("close_follow_trajectory_correlation_min", 0.55))
        )
        # 施动者ID在窗口内必须稳定，避免方向每帧来回翻转。
        actor_values = pd.to_numeric(
            contact_rows["selected_actor_id"], errors="coerce"
        ).dropna()
        role_consistency = (
            float(actor_values.value_counts(normalize=True).iloc[0])
            if not actor_values.empty
            else 0.0
        )
        # 墙跳比例只作为排除证据，不因贴墙本身否定真实追逐。
        wall_fraction = float(
            contact_rows["pair_wall_jump_excluded"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        # 所有条件同时成立才接受，确保该门只补足贴近跟随而不吞入打斗。
        accepted_segment = bool(
            float(center_distance.mean())
            >= float(cfg.get("close_follow_center_distance_min_cm", 5.0))
            and float(center_distance.mean())
            <= float(cfg.get("close_follow_center_distance_max_cm", 10.0))
            and float(combined_speed.mean())
            >= float(cfg.get("close_follow_min_mean_combined_speed_cm_s", 25.0))
            and float(same_direction.mean())
            >= float(cfg.get("close_follow_min_same_direction_fraction", 0.70))
            and float(pursuit.mean())
            >= float(cfg.get("close_follow_min_pursuit_fraction", 0.55))
            and float(escape.mean())
            >= float(cfg.get("close_follow_min_escape_fraction", 0.55))
            and float(correlated.mean())
            >= float(cfg.get("close_follow_min_correlation_fraction", 0.55))
            and role_consistency
            >= float(cfg.get("close_follow_min_role_consistency", 0.60))
            and wall_fraction
            <= float(cfg.get("close_follow_max_wall_jump_fraction", 0.35))
        )
        # 通过后仍逐帧执行近距离限制，避免片段均值掩盖少数远距帧。
        if accepted_segment:
            segment_distance = pd.to_numeric(
                segment["center_distance_cm"], errors="coerce"
            ).to_numpy(dtype=float)
            segment_is_close = (
                np.isfinite(segment_distance)
                & (
                    segment_distance
                    >= float(cfg.get("close_follow_center_distance_min_cm", 5.0))
                )
                & (
                    segment_distance
                    <= float(cfg.get("close_follow_center_distance_max_cm", 10.0))
                )
            )
            accepted[start : end + 1] = segment_is_close
    # 返回逐帧布尔掩码，与严格追逐和普通窗口追逐统一合并。
    return accepted


def _attack_grapple_mask(
    output: pd.DataFrame,
    fps: float,
    cfg: Mapping[str, Any],
    level: str,
) -> np.ndarray:
    """识别骑跨/扭打造成的持续贴身搏斗，补足短促冲击门无法覆盖的攻击。"""
    # 配置可单独关闭该门，便于在其他场地做消融或回退。
    if not bool(cfg.get("grapple_gate_enabled", True)) or len(output) == 0:
        return np.zeros(len(output), dtype=bool)
    # 潜在接触仍由鼻端与对方身体几何关系产生，不借助视频名或目录标签。
    potential = (
        output[f"{level}_potential_attack"].fillna(False).astype(bool).to_numpy()
    )
    # 只在身份和尺度均可用的有效鼠对上累计接触持续时间。
    valid = output["valid_pair"].fillna(False).astype(bool).to_numpy()
    # 将允许的极短姿态漏检间隙补齐，避免一次扭打被拆成多个不达标短段。
    gap_frames = max(
        int(round(float(cfg.get("grapple_fill_gap_seconds", 0.10)) * fps)),
        0,
    )
    # 持续骑跨/扭打至少保持约四分之三秒，普通擦身不会满足该时长。
    min_frames = max(
        int(math.ceil(float(cfg.get("grapple_min_duration_seconds", 0.75)) * fps)),
        1,
    )
    # 先进行时间连续性过滤，再对每个候选段做动力学与质量复核。
    candidate = base.temporal_filter(potential & valid, min_frames, gap_frames)
    # 默认全部拒绝，只有通过完整搏斗判据的时间段才会写入最终攻击标签。
    accepted = np.zeros(len(output), dtype=bool)
    # 分段处理能够防止不同时间的普通接触被错误累积成一次攻击。
    for start, end in _true_runs(candidate):
        # 取出当前连续接触段及其中真正具备接触物证的帧。
        segment = output.iloc[start : end + 1]
        contact_rows = segment[
            segment[f"{level}_potential_attack"].fillna(False).astype(bool)
        ]
        # 时间滤波补出的间隙不能单独构成攻击，至少需要原始接触帧。
        if contact_rows.empty:
            continue
        # 扭打的框中心整体移动较慢，但内部仍会出现一次以上明显扑动峰值。
        combined_speed = (
            pd.to_numeric(
                contact_rows["selected_actor_speed_cm_s"], errors="coerce"
            ).fillna(0.0)
            + pd.to_numeric(
                contact_rows["selected_target_speed_cm_s"], errors="coerce"
            ).fillna(0.0)
        )
        # 动态证据用于排除两只鼠长时间静止贴靠或嗅探。
        dynamic_evidence = pd.to_numeric(
            contact_rows[f"selected_{level}_attack_evidence"], errors="coerce"
        ).fillna(0.0)
        # 姿态质量门避免由完全失真的交叉骨架单独产生持续攻击。
        pose_quality = pd.to_numeric(
            contact_rows["pose_pair_quality"], errors="coerce"
        ).fillna(0.0)
        # 同向移动更符合贴近跟随；扭打通常产生相反或旋转的中心速度。
        direction_similarity = pd.to_numeric(
            contact_rows["direction_similarity"], errors="coerce"
        ).fillna(1.0)
        # Whole-body contact is only a geometric candidate.  When the complete
        # directional columns are available, require at least one attack-oriented
        # cue before promoting a long contact bout to grappling/attack.  This
        # prevents ordinary nose-to-body investigation from becoming an attack
        # merely because an isolated speed spike or heading jitter was observed.
        orientation_columns = (
            f"{level}_attack_actor_initiation",
            f"{level}_attack_target_reaction",
            "selected_distance_drop_cm",
            "pursuit_alignment",
            "selected_target_turn_angle_deg",
        )
        orientation_available = all(
            column in contact_rows.columns for column in orientation_columns
        )
        orientation_ok = True
        causal_attack_ok = True
        if orientation_available:
            initiation = contact_rows[
                f"{level}_attack_actor_initiation"
            ].fillna(False).astype(bool)
            reaction = contact_rows[
                f"{level}_attack_target_reaction"
            ].fillna(False).astype(bool)
            distance_drop = pd.to_numeric(
                contact_rows["selected_distance_drop_cm"], errors="coerce"
            ).fillna(0.0)
            pursuit = pd.to_numeric(
                contact_rows["pursuit_alignment"], errors="coerce"
            ).fillna(-1.0)
            target_turn = pd.to_numeric(
                contact_rows["selected_target_turn_angle_deg"], errors="coerce"
            ).fillna(0.0)
            dynamic = dynamic_evidence >= int(
                cfg.get("grapple_min_dynamic_evidence", 2)
            )
            approach = (
                (distance_drop >= float(
                    cfg.get("grapple_min_distance_drop_cm", 1.0)
                ))
                & (pursuit >= float(
                    cfg.get("grapple_min_pursuit_alignment", 0.50)
                ))
            )
            target_turning = target_turn >= float(
                cfg.get("grapple_min_target_turn_angle_deg", 40.0)
            )
            orientation_ok = bool(
                (initiation | reaction | approach | (target_turning & dynamic)).any()
            )
            raw_column = f"{level}_raw_attack"
            if raw_column in contact_rows.columns:
                raw_attack_count = int(
                    contact_rows[raw_column].fillna(False).astype(bool).sum()
                )
                same_frame_causal_count = int((initiation & reaction).sum())
                causal_attack_ok = bool(
                    raw_attack_count
                    >= int(cfg.get("grapple_min_raw_attack_frames", 0))
                    or same_frame_causal_count
                    >= int(cfg.get("grapple_min_causal_frames", 0))
                )
        if not orientation_ok:
            continue
        if not causal_attack_ok:
            continue
        # 墙边攻击允许存在墙跳标志，但不能整段只有墙跳证据。
        wall_fraction = float(
            contact_rows["pair_wall_jump_excluded"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        # 同时要求低平均位移、局部运动峰值、动态证据和可用姿态。
        accepted_segment = bool(
            float(combined_speed.mean())
            <= float(cfg.get("grapple_max_mean_combined_speed_cm_s", 24.0))
            and float(combined_speed.max())
            >= float(
                cfg.get(
                    "grapple_min_peak_combined_speed_cm_s",
                    50.0 if level == "weak" else 55.0,
                )
            )
            and float(
                combined_speed.quantile(
                    float(np.clip(cfg.get("grapple_speed_quantile", 0.90), 0.0, 1.0))
                )
            )
            >= float(cfg.get("grapple_min_quantile_combined_speed_cm_s", 0.0))
            and int(
                (
                    dynamic_evidence
                    >= int(cfg.get("grapple_min_dynamic_evidence", 2))
                ).sum()
            )
            >= int(cfg.get("grapple_min_dynamic_frames", 1))
            and float(pose_quality.mean())
            >= float(cfg.get("grapple_min_pose_pair_quality", 0.45))
            and float(direction_similarity.mean())
            <= float(
                cfg.get("grapple_max_mean_direction_similarity", 0.75)
            )
            and wall_fraction
            <= float(cfg.get("grapple_max_wall_jump_fraction", 0.85))
        )
        # 通过后标记完整连续段，渲染视频会在同一段持续显示施动者和受动者。
        if accepted_segment:
            accepted[start : end + 1] = True
    # 返回与输入逐帧表等长的布尔掩码，供弱/强两级统一组合。
    return accepted


def _apply_document_attack_gate(
    output: pd.DataFrame, mask: np.ndarray, fps: float, cfg: Mapping[str, Any], level: str
) -> np.ndarray:
    """PPT潜在攻击为鼻-尾<3 cm且<1 s；最终攻击还要动态确认。"""
    accepted = np.zeros(len(output), dtype=bool)
    max_frames = max(int(math.floor(float(cfg.get("max_duration_seconds", 1.0)) * fps)), 1)
    for start, end in _true_runs(mask):
        seg = output.iloc[start : end + 1]
        length = end - start + 1
        potential = seg[f"{level}_potential_attack"].fillna(False).astype(bool)
        initiation = seg[f"{level}_attack_actor_initiation"].fillna(False).astype(bool)
        reaction = seg[f"{level}_attack_target_reaction"].fillna(False).astype(bool)
        evidence = pd.to_numeric(seg[f"selected_{level}_attack_evidence"], errors="coerce").fillna(0)
        nose_body = pd.to_numeric(
            seg["selected_nose_body_distance_cm"], errors="coerce"
        ).fillna(np.inf)
        center_distance = pd.to_numeric(
            seg["center_distance_cm"], errors="coerce"
        ).fillna(np.inf)
        combined_speed = (
            pd.to_numeric(seg["selected_actor_speed_cm_s"], errors="coerce").fillna(0)
            + pd.to_numeric(seg["selected_target_speed_cm_s"], errors="coerce").fillna(0)
        )
        raw_attack = seg[f"{level}_raw_attack"].fillna(False).astype(bool)
        wall = seg["pair_wall_jump_excluded"].fillna(False).astype(bool)
        close_body = (
            nose_body <= float(cfg.get("body_contact_distance_cm", 4.0))
        )
        close_center = (
            center_distance
            <= float(cfg.get("body_center_contact_distance_cm", 6.0))
        )
        high_energy = bool(
            float(combined_speed.max())
            >= float(cfg.get("min_peak_combined_speed_cm_s", 40.0))
        )
        sustained_raw = bool(
            length >= int(cfg.get("sustained_raw_min_frames", 5))
            and float(raw_attack.mean())
            >= float(cfg.get("sustained_raw_min_fraction", 0.80))
        )
        ok = bool(
            length <= max_frames
            and float(potential.mean()) >= float(cfg.get("min_contact_fraction", 0.50))
            and float(close_body.mean())
                >= float(cfg.get("min_body_contact_fraction", 0.25))
            and float(close_center.mean())
                >= float(cfg.get("min_body_center_contact_fraction", 0.25))
            and float(initiation.mean()) >= float(cfg.get("min_actor_initiation_fraction", 0.20))
            and float(reaction.mean()) >= float(cfg.get("min_target_reaction_fraction", 0.20))
            and float((evidence >= int(cfg["min_dynamic_evidence"])).mean())
                >= float(cfg.get("min_dynamic_evidence_fraction", 0.25))
            and (high_energy or sustained_raw)
            and float(wall.mean()) <= float(cfg.get("max_wall_jump_fraction", 0.0))
        )
        if ok:
            accepted[start : end + 1] = True
    return accepted


def postprocess_frame_labels(df: pd.DataFrame, fps: float, config: Mapping[str, Any]) -> pd.DataFrame:
    output = df.copy()
    valid_pair = output["valid_pair"].fillna(False).astype(bool).to_numpy()

    for level in ("weak", "strong"):
        chase_cfg = config["chase"][level]
        attack_cfg = config["attack"][level]
        raw_chase = output[f"{level}_raw_chase"].fillna(False).astype(bool).to_numpy()
        raw_attack = output[f"{level}_raw_attack"].fillna(False).astype(bool).to_numpy()
        chase_min = max(int(math.ceil(float(chase_cfg["min_duration_seconds"]) * fps)), 1)
        chase_gap = max(int(round(float(chase_cfg["fill_gap_seconds"]) * fps)), 0)
        attack_min = max(int(math.ceil(float(attack_cfg["min_duration_seconds"]) * fps)), 1)
        attack_gap = max(int(round(float(attack_cfg["fill_gap_seconds"]) * fps)), 0)

        chase_temporal = base.temporal_filter(raw_chase, chase_min, chase_gap) & valid_pair
        attack_temporal = base.temporal_filter(raw_attack, attack_min, attack_gap) & valid_pair
        strict_chase = _apply_document_chase_gate(output, chase_temporal, fps, chase_cfg)
        window_chase = _apply_window_chase_gate(output, fps, chase_cfg)
        near_recovery_chase = _near_chase_recovery_mask(
            output, fps, chase_cfg
        )
        close_follow_chase = _close_follow_chase_mask(
            output, fps, chase_cfg, level
        )
        # Apply one final per-frame distance clamp to every chase pathway.
        # This prevents document/window aggregation from assigning chase to a
        # frame in which the two current detections are no longer physically near.
        chase_distance = pd.to_numeric(
            output["center_distance_cm"], errors="coerce"
        ).to_numpy(dtype=float)
        chase_distance_gate = (
            valid_pair
            & np.isfinite(chase_distance)
            & (chase_distance <= float(chase_cfg["max_distance_cm"]))
        )
        strict_chase &= chase_distance_gate
        window_chase &= chase_distance_gate
        near_recovery_chase &= chase_distance_gate
        close_follow_chase &= chase_distance_gate
        final_chase = (
            strict_chase
            | window_chase
            | near_recovery_chase
            | close_follow_chase
        ) & valid_pair
        strict_attack = _apply_document_attack_gate(
            output, attack_temporal, fps, attack_cfg, level
        )
        impulse_attack = _attack_impulse_mask(output, attack_cfg, level)
        grapple_attack = _attack_grapple_mask(
            output, fps, attack_cfg, level
        )
        # 遮挡攻击门只消费现有重叠簇物证，不改变追踪、追逐或普通攻击分支。
        occlusion_overlap_attack = _attack_occlusion_overlap_mask(
            output, fps, attack_cfg, level
        )
        final_attack = (
            strict_attack
            | impulse_attack
            | grapple_attack
            | occlusion_overlap_attack
        ) & valid_pair
        output[f"{level}_strict_chase"] = strict_chase & valid_pair
        output[f"{level}_window_chase"] = window_chase & valid_pair
        output[f"{level}_near_recovery_chase"] = (
            near_recovery_chase & valid_pair
        )
        output[f"{level}_close_follow_chase"] = (
            close_follow_chase & valid_pair
        )
        output[f"{level}_strict_attack"] = strict_attack & valid_pair
        output[f"{level}_impulse_attack"] = impulse_attack & valid_pair
        output[f"{level}_grapple_attack"] = grapple_attack & valid_pair
        output[f"{level}_occlusion_overlap_attack"] = (
            occlusion_overlap_attack & valid_pair
        )
        output[f"{level}_final_chase"] = final_chase
        output[f"{level}_final_attack"] = final_attack
        # v1.43: old gates are retained as auditable evidence providers.
        output[f"{level}_legacy_final_chase"] = final_chase
        output[f"{level}_legacy_final_attack"] = final_attack

    # v1.43 Standard Behavior Engine: continuous evidence -> role inference ->
    # hysteretic chase/attack FSM.  The module can run in shadow/legacy mode
    # for A/B validation, but bundled config makes the standard FSM authoritative.
    output = standard_behavior_engine.apply_standard_behavior_engine(output, fps, config)

    output["strong_final_chase"] &= output["weak_final_chase"]
    output["strong_final_attack"] &= output["weak_final_attack"]

    # Keep standard masks internally consistent with the public final columns
    # after the strong⊆weak safety invariant is enforced.
    if "strong_standard_final_chase" in output.columns:
        output["strong_standard_final_chase"] &= output["weak_final_chase"]
    if "strong_standard_final_attack" in output.columns:
        output["strong_standard_final_attack"] &= output["weak_final_attack"]

    for level in ("weak", "strong"):
        labels = (
            output[f"{level}_final_chase"].astype(np.int8)
            + 2 * output[f"{level}_final_attack"].astype(np.int8)
        )
        output[f"{level}_final_label_id"] = labels
        output[f"{level}_final_label_en"] = [LABELS[int(v)][0] for v in labels]
        output[f"{level}_final_label_zh"] = [LABELS[int(v)][1] for v in labels]
    return output


def classify_video_four_label(
    events: Sequence[Mapping[str, Any]],
    level: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """先独立判断视频是否含追逐/攻击，再组合成四类标签。

    追逐和攻击不要求发生在同一帧：5秒视频中先追逐、后打架仍属于标签3。
    """
    chase_events = [event for event in events if int(event.get("label_id", 0)) in (1, 3)]
    chase_present = bool(chase_events)
    attack_present = any(int(event.get("label_id", 0)) in (2, 3) for event in events)
    standard_cfg = dict(config.get("standard_behavior_engine", {})) if config is not None else {}
    standard_authoritative = bool(standard_cfg.get("enabled", False)) and str(
        standard_cfg.get("decision_mode", "standard")
    ).strip().lower() == "standard"
    # In the authoritative standard engine, chase and attack have already been
    # independently confirmed by their own quality/role/FSM logic.  Four-class
    # classification is therefore a pure composition step and must not let a
    # legacy path-length gate silently overturn the Chase FSM.
    if chase_present and attack_present and config is not None and not standard_authoritative:
        level_cfg = config["chase"][level]
        required_path = float(
            level_cfg.get("cooccurring_attack_min_chase_path_cm", 40.0)
        )
        pair_paths: Dict[Tuple[int, int], List[float]] = {}
        strict = False
        for event in chase_events:
            strict = strict or bool(event.get("strict_chase", False))
            pair = tuple(sorted((
                int(event.get("actor_id", -1)),
                int(event.get("target_id", -1)),
            )))
            paths = pair_paths.setdefault(pair, [0.0, 0.0])
            paths[0] += max(safe_float(event.get("actor_path_cm"), 0.0), 0.0)
            paths[1] += max(safe_float(event.get("target_path_cm"), 0.0), 0.0)
        sustained = any(min(paths) >= required_path for paths in pair_paths.values())
        chase_present = bool(strict or sustained)
    label_id = int(chase_present) + 2 * int(attack_present)
    return {
        "candidate_level": str(level),
        "chase_present": bool(chase_present),
        "attack_present": bool(attack_present),
        "video_label_id": int(label_id),
        "video_label_en": LABELS[label_id][0],
        "video_label_zh": LABELS[label_id][1],
        "positive_event_count": int(sum(int(e.get("label_id", 0)) != 0 for e in events)),
    }


def events_from_frames(
    df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
    level: str,
) -> List[Dict[str, Any]]:
    labels = df[f"{level}_final_label_id"].fillna(0).astype(int).to_numpy()
    events: List[Dict[str, Any]] = []
    index = 1
    start = 0
    while start < len(labels):
        label = int(labels[start])
        end = start
        while end + 1 < len(labels) and int(labels[end + 1]) == label:
            end += 1
        if label != 0:
            segment = df.iloc[start : end + 1]
            actor_path = float(segment["selected_actor_speed_cm_s"].fillna(0).sum() / fps)
            target_path = float(segment["selected_target_speed_cm_s"].fillna(0).sum() / fps)
            strong_corr_threshold = float(config["chase"]["strong"]["trajectory_correlation_min"])
            strict_chase = bool(
                label in (1, 3)
                and actor_path >= float(config["chase"]["strict_min_path_cm"])
                and target_path >= float(config["chase"]["strict_min_path_cm"])
                and safe_float(segment["trajectory_correlation"].mean()) >= strong_corr_threshold
            )
            strong_fraction = float((segment["strong_final_label_id"] != 0).mean())
            window_chase_fraction = float(
                segment.get(
                    f"{level}_window_chase", pd.Series(False, index=segment.index)
                ).fillna(False).astype(bool).mean()
            )
            near_recovery_chase_fraction = float(
                segment.get(
                    f"{level}_near_recovery_chase",
                    pd.Series(False, index=segment.index),
                ).fillna(False).astype(bool).mean()
            )
            close_follow_chase_fraction = float(
                segment.get(
                    f"{level}_close_follow_chase",
                    pd.Series(False, index=segment.index),
                ).fillna(False).astype(bool).mean()
            )
            impulse_attack_fraction = float(
                segment.get(
                    f"{level}_impulse_attack", pd.Series(False, index=segment.index)
                ).fillna(False).astype(bool).mean()
            )
            grapple_attack_fraction = float(
                segment.get(
                    f"{level}_grapple_attack", pd.Series(False, index=segment.index)
                ).fillna(False).astype(bool).mean()
            )
            occlusion_overlap_attack_fraction = float(
                segment.get(
                    f"{level}_occlusion_overlap_attack",
                    pd.Series(False, index=segment.index),
                ).fillna(False).astype(bool).mean()
            )
            # v1.43 keeps chase and attack role inference separate.  For a
            # combined CHASE+ATTACK event, prefer the attack initiator; if that
            # role is ambiguous fall back to the chase direction.
            if label == 2:
                # Pure attack: an occlusion/grapple event may be valid while the
                # initiator is genuinely ambiguous.  Do not invent a chase role.
                primary_actor_col = f"{level}_standard_attack_actor_id"
                primary_target_col = f"{level}_standard_attack_target_id"
                fallback_actor_col = ""
                fallback_target_col = ""
            elif label == 3:
                # Combined chase+attack: prefer the attack initiator, but a clear
                # chase direction is a defensible fallback if attack role is lost
                # during contact/occlusion.
                primary_actor_col = f"{level}_standard_attack_actor_id"
                primary_target_col = f"{level}_standard_attack_target_id"
                fallback_actor_col = f"{level}_standard_chase_actor_id"
                fallback_target_col = f"{level}_standard_chase_target_id"
            else:
                primary_actor_col = f"{level}_standard_chase_actor_id"
                primary_target_col = f"{level}_standard_chase_target_id"
                fallback_actor_col = f"{level}_standard_actor_id"
                fallback_target_col = f"{level}_standard_target_id"
            if primary_actor_col in segment.columns:
                role_pairs = [
                    (int(safe_float(actor, -1)), int(safe_float(target, -1)))
                    for actor, target in zip(
                        segment[primary_actor_col].tolist(),
                        segment[primary_target_col].tolist(),
                    )
                    if safe_float(actor, -1) >= 0 and safe_float(target, -1) >= 0
                ]
                if not role_pairs and fallback_actor_col and fallback_actor_col in segment.columns:
                    role_pairs = [
                        (int(safe_float(actor, -1)), int(safe_float(target, -1)))
                        for actor, target in zip(
                            segment[fallback_actor_col].tolist(),
                            segment[fallback_target_col].tolist(),
                        )
                        if safe_float(actor, -1) >= 0 and safe_float(target, -1) >= 0
                    ]
            else:
                role_pairs = [
                    (int(safe_float(actor, -1)), int(safe_float(target, -1)))
                    for actor, target in zip(
                        segment["selected_actor_id"].tolist(),
                        segment["selected_target_id"].tolist(),
                    )
                    if safe_float(actor, -1) >= 0 and safe_float(target, -1) >= 0
                ]
            event_actor_id, event_target_id = mode_or_default(role_pairs, (-1, -1))
            event_actor_id = int(event_actor_id)
            event_target_id = int(event_target_id)
            standard_conf_col = f"{level}_standard_behavior_confidence"
            standard_role_col = f"{level}_standard_role_confidence"
            subtype_col = f"{level}_standard_attack_subtype"
            if standard_conf_col in segment.columns:
                confidence_series = pd.to_numeric(segment[standard_conf_col], errors="coerce").fillna(0.0)
                peak_local_index = int(np.argmax(confidence_series.to_numpy(dtype=float)))
                peak_frame = int(segment["frame"].iloc[peak_local_index])
                mean_standard_confidence = safe_float(confidence_series.mean())
                peak_standard_confidence = safe_float(confidence_series.max())
            else:
                peak_frame = int(segment["frame"].iloc[0])
                mean_standard_confidence = 0.0
                peak_standard_confidence = 0.0
            attack_subtype = ""
            if subtype_col in segment.columns and label in (2, 3):
                subtype_values = [str(v) for v in segment[subtype_col].tolist() if str(v) not in {"", "nan", "None"}]
                attack_subtype = str(mode_or_default(subtype_values, ""))
            events.append({
                "event_id": f"{level[0].upper()}E{index:05d}",
                "candidate_level": level,
                "label_id": label,
                "label_en": LABELS[label][0],
                "label_zh": LABELS[label][1],
                "actor_id": event_actor_id,
                "target_id": event_target_id,
                "start_frame": int(segment["frame"].iloc[0]),
                "end_frame": int(segment["frame"].iloc[-1]),
                "start_time_s": float(segment["time_s"].iloc[0]),
                "end_time_s": float(segment["time_s"].iloc[-1]),
                "duration_s": float((end - start + 1) / fps),
                "peak_frame": peak_frame,
                "behavior_engine": str(mode_or_default(segment.get("standard_behavior_engine_version", pd.Series(["legacy"])).tolist(), "legacy")),
                "attack_subtype": attack_subtype,
                "mean_standard_behavior_confidence": mean_standard_confidence,
                "peak_standard_behavior_confidence": peak_standard_confidence,
                "mean_standard_role_confidence": (
                    safe_float(pd.to_numeric(segment[standard_role_col], errors="coerce").mean())
                    if standard_role_col in segment.columns else 0.0
                ),
                "max_weak_chase_score": int(segment["selected_weak_chase_score"].max()),
                "max_strong_chase_score": int(segment["selected_strong_chase_score"].max()),
                "max_weak_attack_evidence": int(segment["selected_weak_attack_evidence"].max()),
                "max_strong_attack_evidence": int(segment["selected_strong_attack_evidence"].max()),
                "min_center_distance_cm": safe_float(segment["center_distance_cm"].min(), float("nan")),
                "min_nose_body_distance_cm": safe_float(segment["selected_nose_body_distance_cm"].min(), float("nan")),
                "mean_trajectory_correlation": safe_float(segment["trajectory_correlation"].mean()),
                "actor_path_cm": actor_path,
                "target_path_cm": target_path,
                "actor_max_speed_cm_s": safe_float(segment["selected_actor_speed_cm_s"].max()),
                "target_max_speed_cm_s": safe_float(segment["selected_target_speed_cm_s"].max()),
                "target_max_turn_angle_deg": safe_float(segment["selected_target_turn_angle_deg"].max()),
                "strict_chase": strict_chase,
                "strong_candidate_fraction": strong_fraction,
                "window_chase_fraction": window_chase_fraction,
                "near_recovery_chase_fraction": near_recovery_chase_fraction,
                "close_follow_chase_fraction": close_follow_chase_fraction,
                "impulse_attack_fraction": impulse_attack_fraction,
                "grapple_attack_fraction": grapple_attack_fraction,
                "occlusion_overlap_attack_fraction": occlusion_overlap_attack_fraction,
                "mean_pose_pair_quality": safe_float(segment["pose_pair_quality"].mean()),
                "needs_manual_review": True,
                "is_hard_negative": False,
            })
            index += 1
        start = end + 1
    return events


def hard_negative_events(
    df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
    start_index: int,
) -> List[Dict[str, Any]]:
    cfg = config["clips"]
    if not bool(cfg.get("extract_hard_negatives", True)):
        return []
    label = df["weak_final_label_id"].fillna(0).astype(int).to_numpy()
    valid = df["valid_pair"].fillna(False).astype(bool).to_numpy()
    close = df["center_distance_cm"].fillna(np.inf).to_numpy() < float(cfg["hard_negative_close_distance_cm"])
    fast = np.maximum(
        df["mouse_a_speed_cm_s"].fillna(0).to_numpy(),
        df["mouse_b_speed_cm_s"].fillna(0).to_numpy(),
    ) > float(cfg["hard_negative_min_speed_cm_s"])
    exclusion = label != 0
    pad = int(round(max(float(cfg["pre_padding_seconds"]), float(cfg["post_padding_seconds"])) * fps))
    for idx in np.flatnonzero(exclusion):
        exclusion[max(0, idx - pad) : min(len(exclusion), idx + pad + 1)] = True
    candidate = valid & (label == 0) & (close | fast) & (~exclusion)
    clip_frames = max(int(round(float(cfg["hard_negative_clip_seconds"]) * fps)), 1)
    min_interval = max(int(round(float(cfg["hard_negative_min_interval_seconds"]) * fps)), 1)
    selected: List[int] = []
    last = -10**9
    for idx in np.flatnonzero(candidate):
        if idx - last >= min_interval:
            selected.append(int(idx))
            last = int(idx)
        if len(selected) >= int(cfg.get("max_hard_negative_clips", 200)):
            break
    events = []
    for offset, center in enumerate(selected):
        start = max(0, center - clip_frames // 2)
        end = min(len(df) - 1, start + clip_frames - 1)
        segment = df.iloc[start : end + 1]
        events.append({
            "event_id": f"NE{start_index + offset:05d}",
            "candidate_level": "hard_negative",
            "label_id": 0,
            "label_en": LABELS[0][0],
            "label_zh": LABELS[0][1],
            "actor_id": int(mode_or_default(segment["selected_actor_id"].tolist(), -1)),
            "target_id": int(mode_or_default(segment["selected_target_id"].tolist(), -1)),
            "start_frame": int(segment["frame"].iloc[0]),
            "end_frame": int(segment["frame"].iloc[-1]),
            "start_time_s": float(segment["time_s"].iloc[0]),
            "end_time_s": float(segment["time_s"].iloc[-1]),
            "duration_s": float(len(segment) / fps),
            "strict_chase": False,
            "strong_candidate_fraction": 0.0,
            "needs_manual_review": True,
            "is_hard_negative": True,
        })
    return events


def add_clip_boundaries(events: List[Dict[str, Any]], total_frames: int, fps: float, config: Mapping[str, Any]) -> None:
    """为每个事件生成固定时长的视频窗口。

    默认以事件中心为锚点生成5秒片段；靠近视频开头或结尾时平移窗口，
    只要原视频总长不少于5秒，输出片段就保持精确5秒。
    """
    clips_cfg = config["clips"]
    fixed_seconds = float(clips_cfg.get("fixed_clip_seconds", 0.0))
    fixed_frames = max(int(round(fixed_seconds * fps)), 1) if fixed_seconds > 0 else 0
    pre = int(round(float(clips_cfg.get("pre_padding_seconds", 1.5)) * fps))
    post = int(round(float(clips_cfg.get("post_padding_seconds", 1.5)) * fps))

    for event in events:
        event_start = max(0, int(event["start_frame"]))
        event_end = min(total_frames - 1, int(event["end_frame"]))

        if fixed_frames > 0:
            window_frames = min(fixed_frames, total_frames)
            event_center = int(round((event_start + event_end) / 2.0))
            clip_start = event_center - window_frames // 2
            clip_start = max(0, min(clip_start, total_frames - window_frames))
            clip_end = clip_start + window_frames - 1
        elif event.get("is_hard_negative", False):
            clip_start, clip_end = event_start, event_end
        else:
            clip_start = max(0, event_start - pre)
            clip_end = min(total_frames - 1, event_end + post)

        event.update({
            "clip_start_frame": int(clip_start),
            "clip_end_frame": int(clip_end),
            "clip_start_time_s": float(clip_start / fps),
            "clip_end_time_s": float(clip_end / fps),
            "clip_duration_s": float((clip_end - clip_start + 1) / fps),
            "clip_selected": True,
            "clip_skip_reason": "",
            "clip_path": "",
            "review_status": "待复核",
        })


def enforce_clip_spacing(events: List[Dict[str, Any]], fps: float, config: Mapping[str, Any]) -> None:
    """限制相邻提取片段的起始时间间隔。

    默认按“有方向鼠对+标签”分别限制，因此A追B的片段不会删除C追D的片段。
    将clips.min_interval_scope改为global可对全部片段实施全局5秒间隔。
    未被选中的事件仍保留在CSV中，只是不生成MP4。
    """
    cfg = config["clips"]
    min_seconds = float(cfg.get("min_clip_start_interval_seconds", 0.0))
    min_frames = max(int(round(min_seconds * fps)), 0)
    scope = str(cfg.get("min_interval_scope", "same_actor_target_label")).lower()
    if min_frames <= 0 or not events:
        return

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        if scope == "global":
            key: Tuple[Any, ...] = ("global",)
        elif scope == "same_label":
            key = (int(event.get("label_id", -1)),)
        elif scope == "same_pair_label":
            a = int(event.get("actor_id", -1))
            b = int(event.get("target_id", -1))
            key = (min(a, b), max(a, b), int(event.get("label_id", -1)))
        else:
            # 默认保留追逐/攻击方向：A→B与B→A分别处理。
            key = (
                int(event.get("actor_id", -1)),
                int(event.get("target_id", -1)),
                int(event.get("label_id", -1)),
            )
        groups[key].append(event)

    def quality(event: Mapping[str, Any]) -> Tuple[int, float, float]:
        return (
            int(bool(event.get("clean_for_classifier", False))),
            safe_float(event.get("strong_candidate_fraction"), 0.0),
            safe_float(event.get("duration_s"), 0.0),
        )

    for group_events in groups.values():
        ordered = sorted(group_events, key=lambda e: (int(e["clip_start_frame"]), int(e.get("event_id", 0) if str(e.get("event_id", '')).isdigit() else 0)))
        selected: List[Dict[str, Any]] = []
        for event in ordered:
            event["clip_selected"] = True
            event["clip_skip_reason"] = ""
            if not selected:
                selected.append(event)
                continue

            previous = selected[-1]
            delta = int(event["clip_start_frame"]) - int(previous["clip_start_frame"])
            if delta >= min_frames:
                selected.append(event)
                continue

            # 间隔不足时保留更适合分类器的一个；另一个仍留在事件CSV。
            if quality(event) > quality(previous):
                previous["clip_selected"] = False
                previous["clip_skip_reason"] = f"与事件{event.get('event_id')}起始间隔不足{min_seconds:g}秒，保留质量更高事件"
                selected[-1] = event
            else:
                event["clip_selected"] = False
                event["clip_skip_reason"] = f"与事件{previous.get('event_id')}起始间隔不足{min_seconds:g}秒"


# -----------------------------------------------------------------------------
# 人工标注读取和事件级评估
# -----------------------------------------------------------------------------


LABEL_NAME_TO_ID = {
    "non_chase_non_attack": 0,
    "非追逐非攻击": 0,
    "non_aggressive_chase": 1,
    "非攻击性追逐": 1,
    "non_chase_attack": 2,
    "非追逐攻击": 2,
    "aggressive_chase": 3,
    "攻击性追逐": 3,
}


def load_manual_annotations(path: Path, video_path: Path, fps: float) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]

    if "video_name" in df.columns:
        names = {video_path.name.lower(), video_path.stem.lower()}
        df = df[df["video_name"].astype(str).str.lower().isin(names)].copy()
    if df.empty:
        return df

    if "start_frame" not in df.columns:
        if "start_time_s" not in df.columns:
            raise ValueError("人工标注必须包含start_frame或start_time_s。")
        df["start_frame"] = np.round(pd.to_numeric(df["start_time_s"], errors="coerce") * fps)
    if "end_frame" not in df.columns:
        if "end_time_s" not in df.columns:
            raise ValueError("人工标注必须包含end_frame或end_time_s。")
        df["end_frame"] = np.round(pd.to_numeric(df["end_time_s"], errors="coerce") * fps)

    if "label_id" not in df.columns:
        label_column = next((c for c in ("label_name", "label_zh", "label_en", "behavior") if c in df.columns), None)
        if label_column is not None:
            df["label_id"] = df[label_column].astype(str).str.strip().map(LABEL_NAME_TO_ID)
        elif "chase" in df.columns and "attack" in df.columns:
            chase = pd.to_numeric(df["chase"], errors="coerce").fillna(0).astype(bool)
            attack = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(bool)
            df["label_id"] = chase.astype(int) + 2 * attack.astype(int)
        else:
            raise ValueError("人工标注必须包含label_id、标签名称，或者chase和attack两列。")

    if "ignore" in df.columns:
        ignore = df["ignore"].astype(str).str.lower().isin({"1", "true", "yes", "y", "是"})
        df = df[~ignore].copy()

    df["start_frame"] = pd.to_numeric(df["start_frame"], errors="coerce")
    df["end_frame"] = pd.to_numeric(df["end_frame"], errors="coerce")
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["start_frame", "end_frame", "label_id"]).copy()
    df["start_frame"] = df["start_frame"].astype(int).clip(lower=0)
    df["end_frame"] = df["end_frame"].astype(int).clip(lower=0)
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].isin([0, 1, 2, 3]) & (df["end_frame"] >= df["start_frame"])].copy()
    df["manual_event_id"] = [f"M{i + 1:05d}" for i in range(len(df))]
    df["label_en"] = [LABELS[int(x)][0] for x in df["label_id"]]
    df["label_zh"] = [LABELS[int(x)][1] for x in df["label_id"]]
    df["start_time_s"] = df["start_frame"] / fps
    df["end_time_s"] = df["end_frame"] / fps
    return df.reset_index(drop=True)


def _category_mask(df: pd.DataFrame, category: str) -> pd.Series:
    labels = df["label_id"].astype(int)
    if category == "all_positive":
        return labels.isin([1, 2, 3])
    if category == "chase_binary":
        return labels.isin([1, 3])
    if category == "attack_binary":
        return labels.isin([2, 3])
    if category.startswith("label_"):
        return labels == int(category.split("_", 1)[1])
    raise ValueError(category)


def greedy_match_events(
    manual: pd.DataFrame,
    predictions: pd.DataFrame,
    tolerance_frames: int,
    iou_threshold: float,
) -> Tuple[Dict[int, Tuple[int, float]], Dict[int, Tuple[int, float]]]:
    candidates: List[Tuple[float, int, int]] = []
    for mi, m in manual.iterrows():
        expanded_start = max(0, int(m["start_frame"]) - tolerance_frames)
        expanded_end = int(m["end_frame"]) + tolerance_frames
        for pi, p in predictions.iterrows():
            iou = interval_iou(expanded_start, expanded_end, int(p["start_frame"]), int(p["end_frame"]))
            if iou >= iou_threshold:
                candidates.append((iou, int(mi), int(pi)))
    candidates.sort(reverse=True)
    manual_matches: Dict[int, Tuple[int, float]] = {}
    pred_matches: Dict[int, Tuple[int, float]] = {}
    for iou, mi, pi in candidates:
        if mi in manual_matches or pi in pred_matches:
            continue
        manual_matches[mi] = (pi, iou)
        pred_matches[pi] = (mi, iou)
    return manual_matches, pred_matches


def evaluate_predictions(
    manual: pd.DataFrame,
    weak_events: List[Dict[str, Any]],
    strong_events: List[Dict[str, Any]],
    fps: float,
    total_frames: int,
    config: Mapping[str, Any],
    output_dir: Path,
    video_path: Path,
    width: int,
    height: int,
    export_error_clips: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_cfg = config["evaluation"]
    tolerance_frames = int(round(float(eval_cfg["boundary_tolerance_seconds"]) * fps))
    iou_threshold = float(eval_cfg["temporal_iou_threshold"])
    categories = ["all_positive", "chase_binary", "attack_binary", "label_1", "label_2", "label_3"]
    category_zh = {
        "all_positive": "全部阳性行为",
        "chase_binary": "所有追逐",
        "attack_binary": "所有攻击",
        "label_1": "非攻击性追逐",
        "label_2": "非追逐攻击",
        "label_3": "攻击性追逐",
    }

    summary_rows: List[Dict[str, Any]] = []
    manual_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    details_for_export: Dict[str, Any] = {}

    for level, event_list in (("weak", weak_events), ("strong", strong_events)):
        pred_df = pd.DataFrame([e for e in event_list if int(e["label_id"]) != 0])
        if pred_df.empty:
            pred_df = pd.DataFrame(columns=["event_id", "label_id", "start_frame", "end_frame", "clip_start_frame", "clip_end_frame"])

        candidate_core_frames = union_length_frames([
            (int(e["start_frame"]), int(e["end_frame"])) for e in event_list if int(e["label_id"]) != 0
        ])
        candidate_clip_frames = union_length_frames([
            (int(e["clip_start_frame"]), int(e["clip_end_frame"]))
            for e in event_list
            if int(e["label_id"]) != 0 and bool(e.get("clip_selected", True))
        ])

        for category in categories:
            m = manual[_category_mask(manual, category)].copy().reset_index(drop=True)
            p = pred_df[_category_mask(pred_df, category)].copy().reset_index(drop=True)
            manual_matches, pred_matches = greedy_match_events(m, p, tolerance_frames, iou_threshold)

            clip_covered = 0
            for mi, manual_row in m.iterrows():
                covered = False
                for _, pred_row in p.iterrows():
                    if not bool(pred_row.get("clip_selected", True)):
                        continue
                    if intervals_overlap(
                        int(manual_row["start_frame"]), int(manual_row["end_frame"]),
                        int(pred_row.get("clip_start_frame", pred_row["start_frame"])),
                        int(pred_row.get("clip_end_frame", pred_row["end_frame"])),
                    ):
                        covered = True
                        break
                clip_covered += int(covered)
                match = manual_matches.get(int(mi))
                manual_rows.append({
                    "candidate_level": level,
                    "category": category,
                    "category_zh": category_zh[category],
                    "manual_event_id": manual_row["manual_event_id"],
                    "manual_label_id": int(manual_row["label_id"]),
                    "manual_label_zh": manual_row["label_zh"],
                    "manual_start_frame": int(manual_row["start_frame"]),
                    "manual_end_frame": int(manual_row["end_frame"]),
                    "matched": match is not None,
                    "matched_prediction_id": p.iloc[match[0]]["event_id"] if match else "",
                    "temporal_iou": match[1] if match else 0.0,
                    "clip_covered": covered,
                })

            for pi, pred_row in p.iterrows():
                match = pred_matches.get(int(pi))
                pred_rows.append({
                    "candidate_level": level,
                    "category": category,
                    "category_zh": category_zh[category],
                    "prediction_event_id": pred_row["event_id"],
                    "prediction_label_id": int(pred_row["label_id"]),
                    "prediction_label_zh": pred_row["label_zh"],
                    "prediction_start_frame": int(pred_row["start_frame"]),
                    "prediction_end_frame": int(pred_row["end_frame"]),
                    "matched": match is not None,
                    "matched_manual_id": m.iloc[match[0]]["manual_event_id"] if match else "",
                    "temporal_iou": match[1] if match else 0.0,
                })

            manual_count = len(m)
            matched_manual = len(manual_matches)
            pred_count = len(p)
            matched_pred = len(pred_matches)
            video_hours = total_frames / fps / 3600.0 if fps > 0 else 0.0
            summary_rows.append({
                "candidate_level": level,
                "category": category,
                "category_zh": category_zh[category],
                "manual_events": manual_count,
                "matched_manual_events": matched_manual,
                "event_recall": matched_manual / manual_count if manual_count else np.nan,
                "clip_covered_manual_events": clip_covered,
                "clip_coverage_recall": clip_covered / manual_count if manual_count else np.nan,
                "predicted_events": pred_count,
                "matched_prediction_events": matched_pred,
                "event_precision_reference": matched_pred / pred_count if pred_count else np.nan,
                "false_positive_events": pred_count - matched_pred,
                "false_positive_events_per_hour": (pred_count - matched_pred) / video_hours if video_hours > 0 else np.nan,
                "candidate_core_seconds_all_labels": candidate_core_frames / fps,
                "candidate_clip_seconds_all_labels": candidate_clip_frames / fps,
                "review_ratio_all_labels": candidate_clip_frames / total_frames if total_frames > 0 else np.nan,
            })

            if level == "weak" and category in {"label_1", "label_2", "label_3"}:
                details_for_export[category] = (m, p, manual_matches, pred_matches)

    summary_df = pd.DataFrame(summary_rows)
    manual_detail_df = pd.DataFrame(manual_rows)
    prediction_detail_df = pd.DataFrame(pred_rows)
    summary_df.to_csv(output_dir / "评估汇总.csv", index=False, encoding="utf-8-sig")
    manual_detail_df.to_csv(output_dir / "评估人工标注明细.csv", index=False, encoding="utf-8-sig")
    prediction_detail_df.to_csv(output_dir / "评估预测明细.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(output_dir / "评估报告.xlsx", engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            manual_detail_df.to_excel(writer, sheet_name="manual_details", index=False)
            prediction_detail_df.to_excel(writer, sheet_name="prediction_details", index=False)
    except Exception as exc:
        logging.warning("无法输出评估报告.xlsx，仅保留CSV：%s", exc)

    if export_error_clips:
        missed: List[Dict[str, Any]] = []
        false_positives: List[Dict[str, Any]] = []
        padding = int(round(float(eval_cfg.get("error_clip_padding_seconds", 1.5)) * fps))
        seen_manual: set[str] = set()
        seen_pred: set[str] = set()
        for category, (m, p, manual_matches, pred_matches) in details_for_export.items():
            for mi, row in m.iterrows():
                if int(mi) in manual_matches or row["manual_event_id"] in seen_manual:
                    continue
                seen_manual.add(row["manual_event_id"])
                missed.append({
                    "name": f"漏检_{row['manual_event_id']}_{row['label_zh']}",
                    "start_frame": max(0, int(row["start_frame"]) - padding),
                    "end_frame": min(total_frames - 1, int(row["end_frame"]) + padding),
                })
            for pi, row in p.iterrows():
                if int(pi) in pred_matches or row["event_id"] in seen_pred:
                    continue
                seen_pred.add(row["event_id"])
                false_positives.append({
                    "name": f"误报_{row['event_id']}_{row['label_zh']}",
                    "start_frame": int(row.get("clip_start_frame", row["start_frame"])),
                    "end_frame": int(row.get("clip_end_frame", row["end_frame"])),
                })
        extract_named_intervals(video_path, missed, output_dir / "评估复核_漏检片段", fps, width, height)
        extract_named_intervals(video_path, false_positives, output_dir / "评估复核_误报片段", fps, width, height)

    return summary_df, manual_detail_df, prediction_detail_df


def extract_named_intervals(
    video_path: Path,
    intervals: List[Dict[str, Any]],
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    if not intervals:
        return
    ensure_dir(output_dir)
    start_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in intervals:
        path = output_dir / f"{base.sanitize_filename(item['name'])}.mp4"
        item["path"] = path
        start_map[int(item["start_frame"])].append(item)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频导出评估片段：{video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    active: Dict[str, Tuple[cv2.VideoWriter, Dict[str, Any]]] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for item in start_map.get(frame_idx, []):
            writer = cv2.VideoWriter(str(item["path"]), fourcc, fps, (width, height))
            if writer.isOpened():
                active[item["name"]] = (writer, item)
        finished = []
        for key, (writer, item) in active.items():
            if frame_idx <= int(item["end_frame"]):
                writer.write(frame)
            if frame_idx >= int(item["end_frame"]):
                writer.release()
                finished.append(key)
        for key in finished:
            active.pop(key, None)
        frame_idx += 1
    for writer, _ in active.values():
        writer.release()
    cap.release()


# -----------------------------------------------------------------------------
# 主推理流程
# -----------------------------------------------------------------------------



def _pair_key(id_a: int, id_b: int) -> str:
    a, b = sorted((int(id_a), int(id_b)))
    return f"{a}_{b}"


def _contiguous_chunks(
    df: pd.DataFrame,
    max_fill_gap_frames: int = 5,
) -> List[pd.DataFrame]:
    """按真实帧号补齐短缺口，避免先切段后让时间滤波的fill_gap失效。"""
    if df.empty:
        return []
    # 按帧排序并移除同一鼠对同一帧的重复行。
    ordered = df.sort_values("frame").reset_index(drop=True)
    ordered = ordered.drop_duplicates(subset=["frame"], keep="last")
    # 只有超过允许补洞范围的大缺口才真正切成独立片段。
    breaks = (
        ordered["frame"]
        .diff()
        .fillna(1)
        .astype(int)
        .gt(max(int(max_fill_gap_frames), 0) + 1)
    )
    # 为每个大段生成稳定分组编号。
    groups = breaks.cumsum()
    chunks: List[pd.DataFrame] = []
    # 每个大段内部按真实帧号补空行，让后续布尔时间滤波看见缺失帧。
    for _, chunk in ordered.groupby(groups, sort=False):
        # 建立从首帧到末帧的完整整数帧索引。
        frame_index = np.arange(
            int(chunk["frame"].iloc[0]),
            int(chunk["frame"].iloc[-1]) + 1,
            dtype=int,
        )
        # 重索引后缺失观测保持NaN，行为门会把相应布尔证据视为False。
        expanded = (
            chunk.set_index("frame")
            .reindex(frame_index)
            .rename_axis("frame")
            .reset_index()
        )
        # 鼠对身份字段属于整个片段，可安全前后填充到短缺口。
        for column in ("pair_key", "mouse_a_id", "mouse_b_id"):
            if column in expanded.columns:
                expanded[column] = expanded[column].ffill().bfill()
        # 时间戳只按相邻真实帧线性补齐，不对运动特征做插值。
        if "time_s" in expanded.columns:
            expanded["time_s"] = pd.to_numeric(
                expanded["time_s"], errors="coerce"
            ).interpolate(limit_direction="both")
        # 保存补齐后的独立时间段。
        chunks.append(expanded.reset_index(drop=True))
    # 返回全部大段；短缺口已保留为显式无效帧。
    return chunks


def postprocess_all_pairs(
    pair_df_raw: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """每个鼠对独立做时序滤波，避免不同鼠对的标签被错误连接。"""
    if pair_df_raw.empty:
        return pair_df_raw.copy(), [], []

    processed_parts: List[pd.DataFrame] = []
    weak_events: List[Dict[str, Any]] = []
    strong_events: List[Dict[str, Any]] = []

    for pair_key, pair_group in pair_df_raw.groupby("pair_key", sort=False):
        mouse_a_id = int(pair_group["mouse_a_id"].dropna().iloc[0])
        mouse_b_id = int(pair_group["mouse_b_id"].dropna().iloc[0])
        for chunk in _contiguous_chunks(pair_group):
            processed = postprocess_frame_labels(chunk, fps, config)
            processed_parts.append(processed)
            for level, target in (("weak", weak_events), ("strong", strong_events)):
                chunk_events = events_from_frames(processed, fps, config, level)
                for event in chunk_events:
                    event["pair_key"] = str(pair_key)
                    event["mouse_a_id"] = mouse_a_id
                    event["mouse_b_id"] = mouse_b_id
                target.extend(chunk_events)

    pair_df = pd.concat(processed_parts, ignore_index=True)
    pair_df = pair_df.sort_values(["frame", "mouse_a_id", "mouse_b_id"]).reset_index(drop=True)

    for prefix, events in (("WE", weak_events), ("SE", strong_events)):
        events.sort(key=lambda e: (int(e["start_frame"]), str(e.get("pair_key", ""))))
        for idx, event in enumerate(events, start=1):
            event["event_id"] = f"{prefix}{idx:05d}"

    return pair_df, weak_events, strong_events


def mark_event_cleanliness(
    events: List[Dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """标记适合分类器训练的单一鼠对片段，但不删除任何候选。"""
    clean_cfg = config.get("clean_dataset", {})
    max_other_overlap = float(clean_cfg.get("max_other_event_overlap_fraction", 0.10))
    min_pose_quality = float(clean_cfg.get("min_pose_pair_quality", 0.30))

    positives = [e for e in events if int(e.get("label_id", 0)) != 0]
    for event in events:
        if int(event.get("label_id", 0)) == 0:
            event.update({
                "other_positive_event_count": 0,
                "max_other_event_overlap_fraction": 0.0,
                "clean_for_classifier": True,
                "clean_status": "clean_negative",
            })
            continue

        start = int(event["start_frame"])
        end = int(event["end_frame"])
        length = max(end - start + 1, 1)
        overlaps: List[float] = []
        count = 0
        for other in positives:
            if other is event or other.get("pair_key") == event.get("pair_key"):
                continue
            inter = max(0, min(end, int(other["end_frame"])) - max(start, int(other["start_frame"])) + 1)
            if inter > 0:
                count += 1
                overlaps.append(inter / length)
        max_fraction = max(overlaps) if overlaps else 0.0
        pose_quality = safe_float(event.get("mean_pose_pair_quality", 0.0))
        clean = max_fraction <= max_other_overlap and pose_quality >= min_pose_quality
        if max_fraction > max_other_overlap:
            status = "mixed_with_other_pair"
        elif pose_quality < min_pose_quality:
            status = "low_pose_quality"
        else:
            status = "clean_train"
        event.update({
            "other_positive_event_count": count,
            "max_other_event_overlap_fraction": float(max_fraction),
            "clean_for_classifier": bool(clean),
            "clean_status": status,
        })


def hard_negative_events_multimouse(
    pair_df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
    start_index: int,
) -> List[Dict[str, Any]]:
    cfg = config["clips"]
    if pair_df.empty or not bool(cfg.get("extract_hard_negatives", True)):
        return []

    positive_frames = set(
        pair_df.loc[pair_df["weak_final_label_id"].fillna(0).astype(int) != 0, "frame"]
        .astype(int).tolist()
    )
    candidates = pair_df[
        pair_df["valid_pair"].fillna(False).astype(bool)
        & (pair_df["weak_final_label_id"].fillna(0).astype(int) == 0)
        & (
            (pair_df["center_distance_cm"].fillna(np.inf) < float(cfg["hard_negative_close_distance_cm"]))
            | (
                np.maximum(
                    pair_df["mouse_a_speed_cm_s"].fillna(0).to_numpy(),
                    pair_df["mouse_b_speed_cm_s"].fillna(0).to_numpy(),
                ) > float(cfg["hard_negative_min_speed_cm_s"])
            )
        )
    ].copy()
    if candidates.empty:
        return []

    candidates = candidates[~candidates["frame"].astype(int).isin(positive_frames)]
    candidates = candidates.sort_values(["frame", "center_distance_cm"])
    clip_frames = max(int(round(float(cfg["hard_negative_clip_seconds"]) * fps)), 1)
    min_interval = max(int(round(float(cfg["hard_negative_min_interval_seconds"]) * fps)), 1)
    max_clips = int(cfg.get("max_hard_negative_clips", 200))

    selected_rows: List[pd.Series] = []
    last_frame = -10**9
    for _, row in candidates.iterrows():
        frame = int(row["frame"])
        if frame - last_frame < min_interval:
            continue
        selected_rows.append(row)
        last_frame = frame
        if len(selected_rows) >= max_clips:
            break

    events: List[Dict[str, Any]] = []
    half = clip_frames // 2
    for offset, row in enumerate(selected_rows):
        center = int(row["frame"])
        start = max(0, center - half)
        end = start + clip_frames - 1
        events.append({
            "event_id": f"NE{start_index + offset:05d}",
            "candidate_level": "hard_negative",
            "pair_key": str(row["pair_key"]),
            "mouse_a_id": int(row["mouse_a_id"]),
            "mouse_b_id": int(row["mouse_b_id"]),
            "label_id": 0,
            "label_en": LABELS[0][0],
            "label_zh": LABELS[0][1],
            "actor_id": int(row["selected_actor_id"]) if pd.notna(row["selected_actor_id"]) else int(row["mouse_a_id"]),
            "target_id": int(row["selected_target_id"]) if pd.notna(row["selected_target_id"]) else int(row["mouse_b_id"]),
            "start_frame": start,
            "end_frame": end,
            "start_time_s": start / fps,
            "end_time_s": end / fps,
            "duration_s": clip_frames / fps,
            "strict_chase": False,
            "strong_candidate_fraction": 0.0,
            "mean_pose_pair_quality": safe_float(row.get("pose_pair_quality", 0.0)),
            "needs_manual_review": True,
            "is_hard_negative": True,
            "other_positive_event_count": 0,
            "max_other_event_overlap_fraction": 0.0,
            "clean_for_classifier": True,
            "clean_status": "clean_negative",
        })
    return events


def build_frame_summary(
    frame_detection_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = frame_detection_df.copy()
    if summary.empty:
        return summary

    active = pair_df[pair_df.get("weak_final_label_id", pd.Series(dtype=int)).fillna(0).astype(int) != 0].copy()
    if active.empty:
        summary["active_pair_count"] = 0
        summary["frame_label_id"] = 0
        summary["selected_actor_id"] = np.nan
        summary["selected_target_id"] = np.nan
        summary["selected_pair_key"] = ""
        return summary

    active["_priority"] = (
        active["weak_final_label_id"].astype(int) * 100
        + active["selected_weak_attack_evidence"].fillna(0).astype(int) * 10
        + active["selected_weak_chase_score"].fillna(0).astype(int)
    )
    counts = active.groupby("frame").size().rename("active_pair_count")
    top = active.sort_values(["frame", "_priority"], ascending=[True, False]).drop_duplicates("frame")
    top = top.set_index("frame")
    summary = summary.set_index("frame")
    summary["active_pair_count"] = counts
    summary["frame_label_id"] = top["weak_final_label_id"]
    summary["selected_actor_id"] = top["selected_actor_id"]
    summary["selected_target_id"] = top["selected_target_id"]
    summary["selected_pair_key"] = top["pair_key"]
    summary["selected_chase_score"] = top["selected_weak_chase_score"]
    summary["selected_attack_evidence"] = top["selected_weak_attack_evidence"]
    summary["center_distance_cm"] = top["center_distance_cm"]
    summary = summary.fillna({"active_pair_count": 0, "frame_label_id": 0, "selected_pair_key": ""})
    return summary.reset_index()


def _observation_center_px(obs: base.MouseObservation) -> Optional[Tuple[int, int]]:
    valid = obs.keypoints_px[np.all(np.isfinite(obs.keypoints_px), axis=1)]
    if len(valid) == 0:
        return None
    center = np.mean(valid, axis=0)
    return int(round(center[0])), int(round(center[1]))


def _mouse_id_color(logical_id: int) -> Tuple[int, int, int]:
    """为0~19号小鼠生成稳定且尽量不重复的BGR颜色。"""
    hue = int((int(logical_id) * 137) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _clean_keypoint_mask(
    obs: base.MouseObservation,
    min_confidence: float,
    constrain_to_bbox: bool,
    bbox_margin_ratio: float,
) -> np.ndarray:
    """只允许当前框附近且置信度合格的点进入渲染，避免错误点把骨架拉飞。"""
    points = np.asarray(obs.keypoints_px, dtype=np.float32)
    conf = np.asarray(obs.keypoint_conf, dtype=np.float32).reshape(-1)
    valid = np.all(np.isfinite(points), axis=1)
    if len(conf) >= len(points):
        valid &= np.isfinite(conf[:len(points)]) & (conf[:len(points)] >= float(min_confidence))

    if constrain_to_bbox and len(obs.bbox_xyxy) >= 4:
        x1, y1, x2, y2 = np.asarray(obs.bbox_xyxy, dtype=np.float32)[:4]
        if np.all(np.isfinite([x1, y1, x2, y2])) and x2 > x1 and y2 > y1:
            margin_x = max((x2 - x1) * float(bbox_margin_ratio), 2.0)
            margin_y = max((y2 - y1) * float(bbox_margin_ratio), 2.0)
            valid &= (
                (points[:, 0] >= x1 - margin_x)
                & (points[:, 0] <= x2 + margin_x)
                & (points[:, 1] >= y1 - margin_y)
                & (points[:, 1] <= y2 + margin_y)
            )
    return valid


# 关键点调色板（BGR，nose/双耳/neck/双髋/tail 各一色，类YOLO可视化风格）。
_KEYPOINT_PALETTE = [
    (56, 56, 255), (51, 157, 255), (52, 225, 255), (52, 225, 71),
    (255, 194, 71), (225, 134, 52), (148, 52, 203),
]


def _draw_clean_mouse_overlay(
    frame: np.ndarray,
    obs: base.MouseObservation,
    visualization_cfg: Optional[Mapping[str, Any]] = None,
) -> None:
    """只画7点骨架+关键点彩点+干净ID；绝不画鼠对连线、轨迹线或行为箭头。

    v1.12.2：`skeleton_id_only: true` 时不画检测框，只保留骨架与ID标签
    （用户要求：渲染视频里只要骨架和干净的ID）。
    """
    cfg = visualization_cfg or {}
    state_for_color = str(getattr(obs, "track_state", "tracked") or "tracked")
    if state_for_color == "provisional":
        color = (185, 185, 185)  # Pxx：灰色，明确表示检测存在但身份未提交
    elif state_for_color in {"cluster_anonymous", "post_split_anonymous", "reid_ambiguous"}:
        color = (0, 165, 255)  # 聚集匿名：橙色
    else:
        color = _mouse_id_color(int(obs.logical_id))
    height, width = frame.shape[:2]
    skeleton_id_only = bool(cfg.get("skeleton_id_only", False))
    draw_bbox = bool(cfg.get("draw_bbox", True)) and not skeleton_id_only

    bbox = np.asarray(obs.bbox_xyxy, dtype=np.float32).reshape(-1)
    if len(bbox) >= 4 and np.all(np.isfinite(bbox[:4])):
        x1, y1, x2, y2 = np.round(bbox[:4]).astype(int)
        x1 = int(np.clip(x1, 0, max(width - 1, 0)))
        x2 = int(np.clip(x2, 0, max(width - 1, 0)))
        y1 = int(np.clip(y1, 0, max(height - 1, 0)))
        y2 = int(np.clip(y2, 0, max(height - 1, 0)))
        if x2 > x1 and y2 > y1:
            if draw_bbox:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color,
                    int(cfg.get("box_thickness", 2)), cv2.LINE_AA,
                )
        else:
            x1 = y1 = x2 = y2 = 0
    else:
        x1 = y1 = x2 = y2 = 0

    min_conf = float(cfg.get("render_min_keypoint_confidence", cfg.get("raw_min_confidence", 0.10)))
    valid = _clean_keypoint_mask(
        obs,
        min_confidence=min_conf,
        constrain_to_bbox=bool(cfg.get("constrain_keypoints_to_bbox", True)),
        bbox_margin_ratio=float(cfg.get("bbox_margin_ratio", 0.12)),
    )
    points = np.asarray(obs.keypoints_px, dtype=np.float32)

    if x2 > x1 and y2 > y1:
        bbox_diag = float(np.hypot(x2 - x1, y2 - y1))
    else:
        bbox_diag = float("inf")
    max_bone = bbox_diag * float(cfg.get("max_bone_length_bbox_diagonal_ratio", 0.72))

    # 文档§3.3：每个当前检测必须可见。无关键点检测（白鼠亮斑/检测器补缺框）
    # 在"仅骨架"模式下没有骨架可画，必须回退画框，否则该鼠在视频里消失。
    if skeleton_id_only and not valid.any() and x2 > x1 and y2 > y1:
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), color,
            int(cfg.get("box_thickness", 2)), cv2.LINE_AA,
        )
        draw_bbox = True  # 标签锚定到框

    line_thickness = int(cfg.get("skeleton_thickness", 2))
    point_radius = int(cfg.get("point_radius", 4 if skeleton_id_only else 3))
    nose_radius = int(cfg.get("nose_point_radius", max(point_radius + 1, 4)))
    use_palette = bool(cfg.get("keypoint_palette", True))

    # 与用户参考代码一致：仅遍历固定SKEL边，端点都有效时才连线。
    for a, b in base.SKELETON_EDGES:
        if a >= len(points) or b >= len(points) or not (valid[a] and valid[b]):
            continue
        pa = points[a]
        pb = points[b]
        if np.linalg.norm(pa - pb) > max_bone:
            continue
        p1 = tuple(np.round(pa).astype(int))
        p2 = tuple(np.round(pb).astype(int))
        cv2.line(frame, p1, p2, color, line_thickness, cv2.LINE_AA)

    # 关键点彩点（调色板逐点一色，接近用户目标图的可视化风格）。
    for idx, point in enumerate(points):
        if idx >= len(valid) or not valid[idx]:
            continue
        center = tuple(np.round(point).astype(int))
        dot_color = _KEYPOINT_PALETTE[idx % len(_KEYPOINT_PALETTE)] if use_palette else color
        cv2.circle(
            frame, center,
            nose_radius if idx == KP["nose"] else point_radius,
            dot_color, -1, cv2.LINE_AA,
        )

    if bool(cfg.get("show_mouse_id", True)):
        # 文档§3.2：confirmed→"ID n"；tentative→"TMP n"；suspicious→"ID? n"。
        state = str(getattr(obs, "track_state", "tracked") or "tracked")
        label = str(getattr(obs, "display_label", "") or "")
        if not label:
            if state == "tentative":
                label = f"TMP {int(obs.logical_id)}"
            elif state == "suspicious":
                label = f"ID? {int(obs.logical_id)}"
            else:
                label = f"ID {int(obs.logical_id)}"
        if x2 > x1 and draw_bbox:
            tx, ty = x1, max(18, y1 - 6)
        elif valid.any():
            # 无框模式：标签锚定在骨架最上方关键点的头顶，保持画面干净。
            top_idx = int(np.nanargmin(points[valid, 1]))
            anchor = points[valid][top_idx]
            tx = int(anchor[0] - 12)
            ty = max(18, int(anchor[1]) - 12)
        elif x2 > x1:
            tx, ty = x1, max(18, y1 - 6)
        else:
            tx, ty = 0, 18
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.rectangle(frame, (tx, max(0, ty - th - baseline - 3)), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def save_annotated_video_multimouse(
    video_path: Path,
    output_path: Path,
    frame_summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    visual_records: Mapping[int, Sequence[base.MouseObservation]],
    fps: float,
    width: int,
    height: int,
    max_mice: int,
    visualization_cfg: Optional[Mapping[str, Any]] = None,
) -> None:
    """兼容旧的二次渲染入口；输出同样只含框、骨架和ID。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频用于绘制：{video_path}")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建核查视频：{output_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    with tqdm(total=total, desc="生成纯框+骨架核查视频", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            observations = list(visual_records.get(frame_idx, []))
            for obs in observations:
                _draw_clean_mouse_overlay(frame, obs, visualization_cfg)
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    writer.release()
    cap.release()


def save_behavior_label_video(
    tracking_video_path: Path,
    output_path: Path,
    frame_summary_df: pd.DataFrame,
    detection_map_path: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    """在纯追踪视频上二次叠加行为角色标签，不依赖事件片段导出开关。"""
    cap = cv2.VideoCapture(str(tracking_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开追踪视频用于行为标签渲染：{tracking_video_path}")
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建行为标签视频：{output_path}")

    summaries = {
        int(row["frame"]): row
        for row in frame_summary_df.to_dict("records")
    }
    map_handle = detection_map_path.open("r", encoding="utf-8-sig", newline="")
    map_reader = csv.DictReader(map_handle)
    pending_row = next(map_reader, None)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    try:
        with tqdm(total=total, desc="生成追踪+行为标签视频", unit="frame") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_boxes: Dict[int, np.ndarray] = {}
                while pending_row is not None and int(pending_row["frame"]) < frame_idx:
                    pending_row = next(map_reader, None)
                while pending_row is not None and int(pending_row["frame"]) == frame_idx:
                    logical_id = int(float(pending_row["logical_id"]))
                    frame_boxes[logical_id] = np.array([
                        float(pending_row["bbox_x1"]),
                        float(pending_row["bbox_y1"]),
                        float(pending_row["bbox_x2"]),
                        float(pending_row["bbox_y2"]),
                    ], dtype=np.float64)
                    pending_row = next(map_reader, None)

                summary = summaries.get(frame_idx, {})
                label_id = int(safe_float(summary.get("frame_label_id"), 0))
                actor_value = safe_float(summary.get("selected_actor_id"), float("nan"))
                target_value = safe_float(summary.get("selected_target_id"), float("nan"))
                actor_id = int(actor_value) if np.isfinite(actor_value) else None
                target_id = int(target_value) if np.isfinite(target_value) else None
                if label_id in (1, 2, 3):
                    behavior = {1: "CHASE", 2: "ATTACK", 3: "CHASE+ATTACK"}[label_id]
                    color = {1: (0, 220, 255), 2: (0, 64, 255), 3: (255, 64, 255)}[label_id]
                    pair_text = (
                        f"ID {actor_id} {behavior} -> ID {target_id}"
                        if actor_id is not None and target_id is not None
                        else f"{behavior} | IDENTITY AMBIGUOUS"
                    )
                    panel_width = min(max(420, 13 * len(pair_text)), width - 20)
                    cv2.rectangle(frame, (10, 10), (10 + panel_width, 54), (0, 0, 0), -1)
                    cv2.putText(
                        frame, pair_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                        0.78, color, 2, cv2.LINE_AA,
                    )
                    if actor_id in frame_boxes:
                        box = frame_boxes[actor_id]
                        x = int(np.clip(box[0], 0, max(width - 1, 0)))
                        y = int(np.clip(box[3] + 24, 22, max(height - 4, 22)))
                        cv2.putText(
                            frame, f"{behavior} -> ID {target_id}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA,
                        )
                    if target_id in frame_boxes:
                        box = frame_boxes[target_id]
                        x = int(np.clip(box[0], 0, max(width - 1, 0)))
                        y = int(np.clip(box[3] + 24, 22, max(height - 4, 22)))
                        cv2.putText(
                            frame, f"TARGET <- ID {actor_id}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA,
                        )
                    if actor_id in frame_boxes and target_id in frame_boxes:
                        actor_box = frame_boxes[actor_id]
                        target_box = frame_boxes[target_id]
                        actor_center = (
                            int((actor_box[0] + actor_box[2]) / 2),
                            int((actor_box[1] + actor_box[3]) / 2),
                        )
                        target_center = (
                            int((target_box[0] + target_box[2]) / 2),
                            int((target_box[1] + target_box[3]) / 2),
                        )
                        cv2.arrowedLine(
                            frame, actor_center, target_center, color, 2, cv2.LINE_AA,
                            tipLength=0.18,
                        )
                writer.write(frame)
                frame_idx += 1
                pbar.update(1)
    finally:
        map_handle.close()
        writer.release()
        cap.release()



class PairSQLiteStore:
    """将逐帧鼠对结果分批写入SQLite，避免把整个视频放进Python列表。"""

    def __init__(
        self,
        path: Path,
        batch_size: int = 1500,
        resume: bool = False,
        resume_frame: int = 0,
        expected_initialized: Optional[bool] = None,
    ) -> None:
        self.path = path
        if self.path.exists() and not resume:
            self.path.unlink()
        self.conn = sqlite3.connect(str(self.path))
        self.batch_size = max(int(batch_size), 100)
        self.buffer: List[Dict[str, Any]] = []
        table_exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pair_raw'"
        ).fetchone()
        self.initialized = bool(table_exists)
        if resume and expected_initialized is True and not self.initialized:
            self.conn.close()
            raise FileNotFoundError(f"恢复断点缺少鼠对SQLite缓存表：{self.path}")
        if resume and expected_initialized is False and self.initialized:
            self.conn.execute("DROP TABLE pair_raw")
            self.conn.commit()
            self.initialized = False
        elif resume and self.initialized:
            self.conn.execute("DELETE FROM pair_raw WHERE frame >= ?", (int(resume_frame),))
            self.conn.commit()

    @staticmethod
    def _clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in row.items():
            value = to_builtin(value)
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            elif isinstance(value, bool):
                value = int(value)
            clean[str(key)] = value
        return clean

    def add(self, row: Mapping[str, Any]) -> None:
        self.buffer.append(self._clean_row(row))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def add_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append one deterministic frame/chunk with fewer call boundaries."""
        clean_row = self._clean_row
        for row in rows:
            self.buffer.append(clean_row(row))
            if len(self.buffer) >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        df.to_sql(
            "pair_raw",
            self.conn,
            if_exists="append" if self.initialized else "replace",
            index=False,
            chunksize=200,
        )
        self.initialized = True
        self.buffer.clear()
        self.conn.commit()

    def finalize(self) -> None:
        self.flush()
        if self.initialized:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pair_raw_key_frame ON pair_raw(pair_key, frame)")
            self.conn.commit()

    def pair_keys(self) -> List[str]:
        if not self.initialized:
            return []
        rows = self.conn.execute("SELECT DISTINCT pair_key FROM pair_raw ORDER BY pair_key").fetchall()
        return [str(row[0]) for row in rows]

    def read_pair(self, pair_key: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM pair_raw WHERE pair_key = ? ORDER BY frame",
            self.conn,
            params=(pair_key,),
        )

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.conn.close()


class Stage3ObservationCache:
    """Persist the output of steps 1--3 before any pair behavior work starts.

    Each bounded pickle part contains only the final stable-ID observations and
    the occlusion context for a short frame range.  Keeping parts instead of one
    growing object makes the detector stage resumable and prevents the second
    behavior stage from holding the complete video in memory.
    """

    PREFIX = "stage3_observations"
    MANIFEST_NAME = "stage3_observations_manifest.json"
    SCHEMA_VERSION = 2

    def __init__(
        self,
        output_dir: Path,
        chunk_frames: int = 300,
        resume: bool = False,
        resume_frame: int = 0,
        fingerprint: Optional[Mapping[str, Any]] = None,
        total_frames: Optional[int] = None,
        read_only: bool = False,
    ) -> None:
        self.directory = Path(output_dir) / "stage3_observation_cache"
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.directory.is_dir():
                raise FileNotFoundError(f"阶段一身份缓存目录不存在：{self.directory}")
        else:
            self.directory = ensure_dir(self.directory)
        self.chunk_frames = max(int(chunk_frames), 1)
        self.buffer: List[Dict[str, Any]] = []
        self.parts: List[Dict[str, Any]] = []
        self.fingerprint = to_builtin(dict(fingerprint or {}))
        self.total_frames = int(total_frames) if total_frames is not None else None
        self.complete = False
        if self.read_only:
            payload = self._read_manifest_payload(required=True)
            self.chunk_frames = max(int(payload.get("chunk_frames", self.chunk_frames)), 1)
            self.parts = [
                dict(item) for item in payload.get("parts", []) if isinstance(item, Mapping)
            ]
            self.fingerprint = dict(payload.get("fingerprint", {}))
            manifest_total = payload.get("total_frames")
            self.total_frames = int(manifest_total) if manifest_total is not None else None
            self.complete = bool(payload.get("complete", False))
        elif not resume:
            for path in self.directory.glob(f"{self.PREFIX}.*.pkl"):
                path.unlink(missing_ok=True)
            (self.directory / self.MANIFEST_NAME).unlink(missing_ok=True)
        else:
            # A checkpoint is flushed before it is committed, so no valid part
            # straddles the resume boundary.  Remove only this cache's own parts.
            for path in self.directory.glob(f"{self.PREFIX}.*.pkl"):
                parsed = self._parse_part_name(path)
                if parsed is not None and parsed[1] >= int(resume_frame):
                    path.unlink(missing_ok=True)
            self.parts = self._read_manifest_parts()
            self.parts = [
                part for part in self.parts
                if int(part.get("end_frame", -1)) < int(resume_frame)
            ]

    @classmethod
    def open_existing(
        cls,
        output_dir: Path,
        require_complete: bool = True,
    ) -> "Stage3ObservationCache":
        cache = cls(output_dir, read_only=True)
        if require_complete and not cache.is_complete():
            raise ValueError(
                f"阶段一身份缓存未完成或存在帧缺口：{cache.directory}"
            )
        return cache

    @classmethod
    def _parse_part_name(cls, path: Path) -> Optional[Tuple[int, int]]:
        pieces = path.stem.split(".")
        if len(pieces) != 3:
            return None
        try:
            return int(pieces[1]), int(pieces[2])
        except ValueError:
            return None

    def _read_manifest_payload(self, required: bool = False) -> Dict[str, Any]:
        path = self.directory / self.MANIFEST_NAME
        if not path.exists():
            if required:
                raise FileNotFoundError(f"阶段一身份缓存清单不存在：{path}")
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, Mapping):
                raise ValueError("manifest root is not a mapping")
            schema = int(payload.get("schema_version", 1))
            if schema not in {1, self.SCHEMA_VERSION}:
                raise ValueError(f"unsupported schema version {schema}")
            return dict(payload)
        except (OSError, ValueError, TypeError) as exc:
            if required:
                raise ValueError(f"阶段一身份缓存清单损坏：{path}：{exc}") from exc
            return {}

    def _read_manifest_parts(self) -> List[Dict[str, Any]]:
        payload = self._read_manifest_payload(required=False)
        if self.fingerprint and payload.get("fingerprint") not in ({}, self.fingerprint):
            return []
        parts = payload.get("parts", [])
        return [dict(item) for item in parts if isinstance(item, Mapping)]

    def add(
        self,
        frame: int,
        observations: Sequence[base.MouseObservation],
        cluster_context: Mapping[str, Any],
        scale_mode: str,
        cm_per_pixel: float,
    ) -> None:
        if self.read_only:
            raise RuntimeError("只读阶段一身份缓存不能追加帧。")
        # MouseObservation and its NumPy arrays are intentionally pickled as-is;
        # this is lossless and avoids converting every keypoint to CSV text.
        self.buffer.append({
            "frame": int(frame),
            "observations": list(observations),
            "cluster_context": copy.deepcopy(dict(cluster_context)),
            "scale_mode": str(scale_mode),
            "cm_per_pixel": float(cm_per_pixel)
            if np.isfinite(cm_per_pixel)
            else None,
        })
        if len(self.buffer) >= self.chunk_frames:
            self.flush()

    def flush(self) -> None:
        if self.read_only:
            if self.buffer:
                raise RuntimeError("只读阶段一身份缓存存在意外写缓冲。")
            return
        if not self.buffer:
            return
        start_frame = int(self.buffer[0]["frame"])
        end_frame = int(self.buffer[-1]["frame"])
        final_path = self.directory / f"{self.PREFIX}.{start_frame:09d}.{end_frame:09d}.pkl"
        temporary = final_path.with_suffix(final_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(final_path)
        self.parts = [
            part for part in self.parts
            if str(part.get("path", "")) != final_path.name
        ]
        self.parts.append({
            "path": final_path.name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": len(self.buffer),
        })
        self.parts.sort(key=lambda part: int(part.get("start_frame", 0)))
        self.buffer.clear()
        self._write_manifest()

    def _write_manifest(self) -> None:
        if self.read_only:
            raise RuntimeError("只读阶段一身份缓存不能写清单。")
        _atomic_json_dump(
            {
                "schema_version": self.SCHEMA_VERSION,
                "program_version": PROGRAM_VERSION,
                "chunk_frames": self.chunk_frames,
                "fingerprint": self.fingerprint,
                "total_frames": self.total_frames,
                "complete": bool(self.complete),
                "parts": self.parts,
            },
            self.directory / self.MANIFEST_NAME,
        )

    def close(self) -> None:
        if self.read_only:
            return
        self.flush()
        self._write_manifest()

    def mark_complete(self, total_frames: int) -> None:
        """Commit the Stage-1/Stage-2 contract only after every frame is present."""
        if self.read_only:
            raise RuntimeError("只读阶段一身份缓存不能标记完成。")
        self.flush()
        self.total_frames = int(total_frames)
        expected = 0
        for part in sorted(self.parts, key=lambda item: int(item.get("start_frame", -1))):
            start = int(part.get("start_frame", -1))
            end = int(part.get("end_frame", -1))
            frame_count = int(part.get("frame_count", -1))
            path = self.directory / str(part.get("path", ""))
            if start != expected or end < start or frame_count != end - start + 1 or not path.exists():
                raise ValueError(
                    f"阶段一身份缓存不连续：期望从{expected}开始，实际分片={part}"
                )
            expected = end + 1
        if expected != self.total_frames:
            raise ValueError(
                f"阶段一身份缓存帧数不完整：缓存{expected}帧，视频{self.total_frames}帧。"
            )
        self.complete = True
        self._write_manifest()

    def is_complete(self) -> bool:
        if not self.complete or self.total_frames is None:
            return False
        expected = 0
        for part in sorted(self.parts, key=lambda item: int(item.get("start_frame", -1))):
            start = int(part.get("start_frame", -1))
            end = int(part.get("end_frame", -1))
            path = self.directory / str(part.get("path", ""))
            if start != expected or end < start or not path.exists():
                return False
            expected = end + 1
        return expected == int(self.total_frames)

    def iter_frames(self) -> Iterable[Dict[str, Any]]:
        parts = sorted(
            self.parts or self._read_manifest_parts(),
            key=lambda part: int(part.get("start_frame", 0)),
        )
        expected: Optional[int] = 0 if self.complete else None
        for part in parts:
            path = self.directory / str(part["path"])
            if not path.exists():
                raise FileNotFoundError(f"阶段3缓存分片不存在：{path}")
            with path.open("rb") as handle:
                entries = pickle.load(handle)
            if not isinstance(entries, list):
                raise ValueError(f"阶段3缓存分片格式错误：{path}")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"阶段3缓存帧记录格式错误：{path}")
                frame = int(entry.get("frame", -1))
                if expected is not None and frame != expected:
                    raise ValueError(
                        f"阶段一身份缓存存在帧缺口：期望{expected}，实际{frame}。"
                    )
                if expected is not None:
                    expected += 1
                yield dict(entry)
        if expected is not None and expected != int(self.total_frames or 0):
            raise ValueError(
                f"阶段一身份缓存读取帧数不完整：读取{expected}帧，清单{self.total_frames}帧。"
            )


def _serialize_yolo_detection(detection: base.Detection) -> Dict[str, Any]:
    """把第一阶段Pose结果压缩成与底层模型解耦的数值记录。"""
    return {
        "keypoints_px": np.asarray(detection.keypoints_px, dtype=np.float64).copy(),
        "keypoint_conf": np.asarray(detection.keypoint_conf, dtype=np.float64).copy(),
        "bbox_xyxy": np.asarray(detection.bbox_xyxy, dtype=np.float64).copy(),
        "box_conf": float(detection.box_conf),
        "pose_quality": float(getattr(detection, "pose_quality", 0.0)),
    }


def _deserialize_yolo_detection(
    payload: Mapping[str, Any],
    expected_keypoints: int,
) -> base.Detection:
    """从第一阶段缓存恢复Detection，保持第二阶段的候选语义不变。"""
    points = np.asarray(payload.get("keypoints_px", []), dtype=np.float64)
    confidence = np.asarray(payload.get("keypoint_conf", []), dtype=np.float64).reshape(-1)
    if points.shape != (int(expected_keypoints), 2):
        raise ValueError(
            "YOLO预推理缓存关键点形状错误："
            f"{points.shape}，期望({int(expected_keypoints)}, 2)。"
        )
    if confidence.shape != (int(expected_keypoints),):
        raise ValueError(
            "YOLO预推理缓存关键点置信度形状错误："
            f"{confidence.shape}，期望({int(expected_keypoints)},)。"
        )
    bbox = np.asarray(payload.get("bbox_xyxy", []), dtype=np.float64).reshape(-1)
    if bbox.shape != (4,):
        raise ValueError(f"YOLO预推理缓存边界框形状错误：{bbox.shape}。")
    return base.Detection(
        raw_track_id=None,
        keypoints_px=points.copy(),
        keypoint_conf=confidence.copy(),
        bbox_xyxy=bbox.copy(),
        box_conf=float(payload.get("box_conf", 0.0)),
        pose_quality=float(payload.get("pose_quality", 0.0)),
    )


class YOLOPrecomputeCache:
    """分块持久化YOLO结果，作为全图检测和后续CPU计算之间的边界。

    第一阶段只保存Pose关键点和可选检测器框，不保存视频帧，因此不会把长视频
    全部载入内存；第二阶段重新顺序读取原视频，并从这里恢复同一帧的YOLO结果。
    分块文件使用gzip+pickle，写入采用临时文件替换，进程中断后已完成分块仍可复用。
    """

    PREFIX = "yolo_results"
    MANIFEST_NAME = "yolo_results_manifest.json"
    STATUS_NAME = "yolo_results_status.json"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_dir: Path,
        video_path: Path,
        model_path: Path,
        config: Mapping[str, Any],
        total_frames: int,
        fps: float,
        width: int,
        height: int,
        chunk_frames: int = 300,
    ) -> None:
        self.directory = ensure_dir(Path(output_dir) / "yolo_precompute")
        self.chunk_frames = max(int(chunk_frames), 1)
        self.total_frames = max(int(total_frames), 0)
        self.fingerprint = {
            "video": _checkpoint_file_fingerprint(video_path),
            "model": _checkpoint_file_fingerprint(model_path),
            "config_sha256": _checkpoint_config_digest(config),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "total_frames": int(total_frames),
        }
        self.buffer: List[Dict[str, Any]] = []
        self.parts: List[Dict[str, Any]] = self._read_manifest_parts()

    def _read_manifest_parts(self) -> List[Dict[str, Any]]:
        """读取分块清单；清单损坏时安全回退为空，随后重新预推理。"""
        path = self.directory / self.MANIFEST_NAME
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if int(payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
                return []
            if payload.get("fingerprint") != self.fingerprint:
                return []
            return [dict(part) for part in payload.get("parts", []) if isinstance(part, Mapping)]
        except (OSError, ValueError, TypeError):
            return []

    def _read_status(self) -> Dict[str, Any]:
        """读取人类可读进度；状态仅用于恢复预推理，不参与行为判定。"""
        path = self.directory / self.STATUS_NAME
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return dict(payload) if isinstance(payload, Mapping) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def reset(self) -> None:
        """清除不完整或不匹配的YOLO分块，避免混用不同模型的结果。"""
        for path in self.directory.glob(f"{self.PREFIX}.*.pkl.gz"):
            path.unlink(missing_ok=True)
        (self.directory / self.MANIFEST_NAME).unlink(missing_ok=True)
        (self.directory / self.STATUS_NAME).unlink(missing_ok=True)
        self.parts = []
        self.buffer.clear()

    def _write_manifest(self) -> None:
        """原子写入带输入指纹的分块清单。"""
        _atomic_json_dump(
            {
                "schema_version": self.SCHEMA_VERSION,
                "chunk_frames": self.chunk_frames,
                "fingerprint": self.fingerprint,
                "parts": self.parts,
            },
            self.directory / self.MANIFEST_NAME,
        )

    def _write_status(self, next_frame: int, complete: bool) -> None:
        """原子写入预推理进度，允许只完成YOLO阶段后继续运行。"""
        _atomic_json_dump(
            {
                "program_version": PROGRAM_VERSION,
                "status": "complete" if complete else "running",
                "next_frame": int(next_frame),
                "processed_frames": int(next_frame),
                "total_frames": int(self.total_frames),
                "progress_percent": 100.0 * int(next_frame) / max(self.total_frames, 1),
                "fingerprint": self.fingerprint,
                "cache_directory": str(self.directory),
            },
            self.directory / self.STATUS_NAME,
        )

    def is_complete(self) -> bool:
        """只在清单、分块文件和完成状态都一致时复用缓存。"""
        status = self._read_status()
        if status.get("status") != "complete":
            return False
        if status.get("fingerprint") != self.fingerprint:
            return False
        if int(status.get("next_frame", -1)) != self.total_frames:
            return False
        parts = sorted(self.parts, key=lambda part: int(part.get("start_frame", 0)))
        expected = 0
        for part in parts:
            start = int(part.get("start_frame", -1))
            end = int(part.get("end_frame", -1))
            path = self.directory / str(part.get("path", ""))
            if start != expected or end < start or not path.exists():
                return False
            expected = end + 1
        return expected == self.total_frames

    def next_frame(self, resume_requested: bool = False) -> int:
        """返回安全恢复边界；新任务也会复用同一指纹下的完整分块。"""
        if self.is_complete():
            return self.total_frames
        status = self._read_status()
        if status.get("fingerprint") != self.fingerprint:
            self.reset()
            return 0
        candidate = int(status.get("next_frame", 0)) if status else 0
        candidate = max(0, min(candidate, self.total_frames))
        valid_parts: List[Dict[str, Any]] = []
        expected = 0
        for part in sorted(self.parts, key=lambda item: int(item.get("start_frame", 0))):
            start = int(part.get("start_frame", -1))
            end = int(part.get("end_frame", -1))
            path = self.directory / str(part.get("path", ""))
            if start != expected or end < start or not path.exists() or end >= candidate:
                break
            valid_parts.append(part)
            expected = end + 1
        if expected != candidate:
            candidate = expected
        self.parts = valid_parts
        for path in self.directory.glob(f"{self.PREFIX}.*.pkl.gz"):
            pieces = path.name.split(".")
            try:
                start = int(pieces[1])
            except (IndexError, ValueError):
                continue
            if start >= candidate:
                path.unlink(missing_ok=True)
        self._write_manifest()
        self._write_status(candidate, complete=False)
        logging.info(
            "YOLO预推理缓存%s：从第%d帧继续，已保留%d个完整分块。",
            "续跑" if resume_requested else "恢复",
            candidate,
            len(self.parts),
        )
        return candidate

    def add(
        self,
        frame: int,
        pose_detections: Sequence[base.Detection],
        detector_boxes: Sequence[Tuple[np.ndarray, float]],
    ) -> None:
        """追加单帧原始YOLO结果；达到块大小后立即落盘。"""
        self.buffer.append({
            "frame": int(frame),
            "pose_detections": [_serialize_yolo_detection(det) for det in pose_detections],
            "detector_boxes": [
                (np.asarray(box, dtype=np.float64).copy(), float(conf))
                for box, conf in detector_boxes
            ],
        })
        if len(self.buffer) >= self.chunk_frames:
            self.flush()

    def flush(self) -> None:
        """原子写入当前分块，并推进可恢复边界。"""
        if not self.buffer:
            return
        start_frame = int(self.buffer[0]["frame"])
        end_frame = int(self.buffer[-1]["frame"])
        final_path = self.directory / f"{self.PREFIX}.{start_frame:09d}.{end_frame:09d}.pkl.gz"
        temporary = final_path.with_suffix(final_path.suffix + ".tmp")
        with gzip.open(temporary, "wb", compresslevel=3) as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        os.replace(temporary, final_path)
        self.parts = [
            part for part in self.parts if str(part.get("path", "")) != final_path.name
        ]
        self.parts.append({
            "path": final_path.name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": len(self.buffer),
        })
        self.parts.sort(key=lambda part: int(part.get("start_frame", 0)))
        self.buffer.clear()
        self._write_manifest()
        self._write_status(end_frame + 1, complete=False)

    def close(self) -> None:
        """完成最后不足一个块的结果，并更新最终状态。"""
        self.flush()
        self._write_manifest()
        expected = 0
        complete = bool(self.parts)
        for part in sorted(self.parts, key=lambda item: int(item.get("start_frame", 0))):
            start = int(part.get("start_frame", -1))
            end = int(part.get("end_frame", -1))
            path = self.directory / str(part.get("path", ""))
            if start != expected or end < start or not path.exists():
                complete = False
                break
            expected = end + 1
        complete = bool(complete and expected == self.total_frames)
        next_frame = self.total_frames if complete else (
            int(self.parts[-1].get("end_frame", -1)) + 1 if self.parts else 0
        )
        self._write_status(next_frame, complete=complete)

    def iter_frames(self, start_frame: int = 0) -> Iterable[Dict[str, Any]]:
        """按帧顺序读取缓存，第二阶段只保留当前分块在内存中。"""
        expected = max(int(start_frame), 0)
        for part in sorted(self.parts, key=lambda item: int(item.get("start_frame", 0))):
            path = self.directory / str(part.get("path", ""))
            if not path.exists():
                raise FileNotFoundError(f"YOLO预推理缓存分块不存在：{path}")
            with gzip.open(path, "rb") as handle:
                entries = pickle.load(handle)
            if not isinstance(entries, list):
                raise ValueError(f"YOLO预推理缓存分块格式错误：{path}")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"YOLO预推理帧记录格式错误：{path}")
                frame = int(entry.get("frame", -1))
                if frame < expected:
                    continue
                if frame != expected:
                    raise ValueError(
                        f"YOLO预推理缓存存在帧缺口：期望{expected}，实际{frame}。"
                    )
                expected += 1
                yield dict(entry)


def _extract_detector_boxes_from_result(
    result: Any,
    max_det: int,
    profiling: Optional[MutableMapping[str, float]] = None,
) -> List[Tuple[np.ndarray, float]]:
    """从普通YOLO检测结果提取补充框，并可拆分GPU传输/CPU解析计时。"""
    if result is None or getattr(result, "boxes", None) is None:
        return []
    boxes = getattr(result.boxes, "xyxy", None)
    conf = getattr(result.boxes, "conf", None)
    if boxes is None or conf is None or len(boxes) == 0:
        return []
    transfer_started = time.perf_counter()
    xyxy = boxes.detach().cpu().numpy().astype(np.float64)
    scores = conf.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if profiling is not None:
        profiling["yolo_result_transfer_seconds"] = float(
            profiling.get("yolo_result_transfer_seconds", 0.0)
        ) + (time.perf_counter() - transfer_started)
    parse_started = time.perf_counter()
    order = np.argsort(-scores)[: max(int(max_det), 1)]
    output = [(xyxy[int(index)].copy(), float(scores[int(index)])) for index in order]
    if profiling is not None:
        profiling["result_parse_seconds"] = float(
            profiling.get("result_parse_seconds", 0.0)
        ) + (time.perf_counter() - parse_started)
    return output


def _predict_yolo_batch(
    model: Any,
    frames: Sequence[np.ndarray],
    kwargs: Mapping[str, Any],
) -> List[Any]:
    """批量调用Ultralytics；遇到旧版本或测试替身返回数量异常时逐帧回退。"""
    if not frames:
        return []
    batch_kwargs = dict(kwargs)
    batch_kwargs.update({"source": list(frames), "stream": False, "batch": len(frames)})
    try:
        results = list(model.predict(**batch_kwargs))
        if len(results) == len(frames):
            return results
    except (TypeError, RuntimeError, ValueError) as exc:
        logging.warning("YOLO批量预推理不可用，回退逐帧调用：%s", exc)
    results = []
    for frame in frames:
        single_kwargs = dict(kwargs)
        single_kwargs["source"] = frame
        single_kwargs["stream"] = False
        single = model.predict(**single_kwargs)
        single_results = list(single)
        results.append(single_results[0] if single_results else None)
    return results


def run_yolo_first_pass(
    video_path: Path,
    model_path: Path,
    pose_model: Any,
    detector_model: Any,
    detector_cfg: Mapping[str, Any],
    performance_cfg: Mapping[str, Any],
    output_dir: Path,
    config: Mapping[str, Any],
    device: Any,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    expected_keypoints: int,
    resume_requested: bool = False,
    profiler: Optional[RuntimeProfiler] = None,
) -> Optional[YOLOPrecomputeCache]:
    """先完成整段视频的YOLO批量推理，再返回第二阶段可顺序读取的缓存。

    该函数只负责视频解码、GPU批量推理和结果落盘；检测过滤、身份追踪、遮挡恢复、
    行为规则和渲染仍由原有第二阶段执行，从而不改变步骤1--7的业务顺序。
    """
    cfg = dict(performance_cfg.get("yolo_first_pass", {}))
    if not bool(cfg.get("enabled", False)):
        return None
    if str(detector_cfg.get("pose_mode", "full_frame")).lower() != "full_frame":
        logging.warning("YOLO-first仅支持full_frame，当前pose_mode=roi，回退原逐帧ROI流。")
        return None
    cache = YOLOPrecomputeCache(
        output_dir=output_dir,
        video_path=video_path,
        model_path=model_path,
        config=config,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        chunk_frames=int(cfg.get("cache_chunk_frames", 300)),
    )
    start_frame = cache.next_frame(resume_requested=resume_requested)
    if cache.is_complete():
        logging.info("YOLO预推理缓存已完整：%s", cache.directory)
        return cache

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"YOLO预推理无法打开视频：{video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        positioned = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if positioned > start_frame:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            positioned = 0
        while positioned < start_frame:
            if not cap.grab():
                cap.release()
                raise RuntimeError(f"YOLO预推理无法定位恢复帧：{start_frame}")
            positioned += 1

    batch_size = max(int(cfg.get("batch_size", 8)), 1)
    pose_kwargs: Dict[str, Any] = {
        "imgsz": int(detector_cfg.get("pose_full_imgsz", 960)),
        "conf": float(detector_cfg.get("pose_full_conf", 0.08)),
        "iou": float(detector_cfg.get("pose_full_iou", 0.50)),
        "max_det": int(detector_cfg.get("pose_full_max_det", 40)),
        "device": device,
        "verbose": False,
    }
    if bool(detector_cfg.get("pose_half", True)) and str(device).lower() != "cpu":
        pose_kwargs["half"] = True
    detector_kwargs: Dict[str, Any] = {
        "imgsz": int(detector_cfg.get("detector_imgsz", 1280)),
        "conf": float(detector_cfg.get("detector_conf", 0.14)),
        "iou": float(detector_cfg.get("detector_iou", 0.50)),
        "max_det": int(detector_cfg.get("detector_max_det", 40)),
        "device": device,
        "verbose": False,
    }
    if bool(detector_cfg.get("half", True)) and str(device).lower() != "cpu":
        detector_kwargs["half"] = True
    frame_idx = int(start_frame)
    logging.info(
        "YOLO预推理第一阶段开始：batch=%d，起始帧=%d/%d，结果目录=%s。",
        batch_size,
        frame_idx,
        total_frames,
        cache.directory,
    )
    try:
        with tqdm(total=total_frames, initial=frame_idx, desc="YOLO整段预推理", unit="frame") as pbar:
            while True:
                frames: List[np.ndarray] = []
                indices: List[int] = []
                decode_started = time.perf_counter()
                for _ in range(batch_size):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                    indices.append(frame_idx + len(indices))
                if profiler is not None:
                    profiler.add("decode", time.perf_counter() - decode_started)
                if not frames:
                    break
                pose_inference_started = time.perf_counter()
                pose_results = _predict_yolo_batch(pose_model, frames, pose_kwargs)
                if profiler is not None:
                    profiler.add("full_frame_inference", time.perf_counter() - pose_inference_started)
                if len(pose_results) != len(frames):
                    raise RuntimeError(
                        f"YOLO Pose批量结果数量不一致：帧{len(frames)}，结果{len(pose_results)}。"
                    )
                if detector_model is not None:
                    detector_inference_started = time.perf_counter()
                    detector_results = _predict_yolo_batch(detector_model, frames, detector_kwargs)
                    if profiler is not None:
                        profiler.add("full_frame_inference", time.perf_counter() - detector_inference_started)
                    if len(detector_results) != len(frames):
                        raise RuntimeError(
                            "YOLO检测器批量结果数量不一致："
                            f"帧{len(frames)}，结果{len(detector_results)}。"
                        )
                else:
                    detector_results = [None] * len(frames)
                for local_idx, (frame_no, pose_result, detector_result) in enumerate(
                    zip(indices, pose_results, detector_results)
                ):
                    parse_profile: Dict[str, float] = {}
                    pose_detections = base.parse_yolo_result(
                        pose_result,
                        expected_keypoints,
                        int(detector_cfg.get("pose_full_max_det", 40)),
                        profiling=parse_profile,
                    )
                    detector_boxes = _extract_detector_boxes_from_result(
                        detector_result,
                        int(detector_cfg.get("detector_max_det", 40)),
                        profiling=parse_profile,
                    )
                    if profiler is not None:
                        profiler.add(
                            "yolo_result_transfer",
                            float(parse_profile.get("yolo_result_transfer_seconds", 0.0)),
                        )
                        profiler.add(
                            "result_parse",
                            float(parse_profile.get("result_parse_seconds", 0.0)),
                        )
                    cache.add(frame_no, pose_detections, detector_boxes)
                    frame_idx = int(frame_no) + 1
                pbar.update(len(frames))
        cache.close()
    except BaseException:
        cache.flush()
        cache._write_manifest()
        logging.exception("YOLO预推理第一阶段中断：已保存到第%d帧。", cache.next_frame())
        raise
    finally:
        cap.release()
    if not cache.is_complete():
        raise RuntimeError("YOLO预推理结束但缓存未覆盖完整视频，拒绝进入第二阶段。")
    logging.info("YOLO预推理第一阶段完成：%d帧；开始进入原有检测/追踪/行为步骤。", total_frames)
    return cache


class _Stage3TransformerStub:
    """Provide the two fields used by ``empty_frame_record`` during stage 4--7."""

    def __init__(self, mode: str, cm_per_pixel: Optional[float]) -> None:
        self.mode = str(mode)
        self.current_cm_per_pixel = (
            float(cm_per_pixel) if cm_per_pixel is not None else float("nan")
        )


def _behavior_eligible_observations(
    observations: Sequence[base.MouseObservation],
) -> List[base.MouseObservation]:
    """Return only formal IDs that are safe inputs to steps 4--7.

    Stage 3 also stores provisional/anonymous render observations so that the
    rendered video remains complete.  Those observations must not silently
    become behavior actors, otherwise an occlusion label can create a false
    pair or a duplicate logical ID.
    """
    excluded_states = {
        "cluster_anonymous",
        "post_split_anonymous",
        "reid_ambiguous",
    }
    return [
        observation
        for observation in observations
        if str(getattr(observation, "track_state", "tracked")) not in excluded_states
        and int(observation.logical_id) >= 0
    ]


def _pair_record_from_features(
    frame_idx: int,
    fps: float,
    transformer: Any,
    actor: base.MouseObservation,
    target: base.MouseObservation,
    actor_to_target: HighRecallPairFeatures,
    target_to_actor: HighRecallPairFeatures,
    selected: HighRecallPairFeatures,
    repeated_contact_count: int,
    cluster_attack_hint: bool,
    pair_cluster_evidence: Mapping[str, Any],
    record_template: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize one directional feature computation into the legacy CSV row.

    Keeping this serializer in one place is important: inline and staged
    pipelines must produce byte-for-byte equivalent behavior evidence before
    temporal post-processing is applied.
    """
    weak_chase = bool(actor_to_target.weak_chase or target_to_actor.weak_chase)
    weak_attack = bool(actor_to_target.weak_attack or target_to_actor.weak_attack)
    strong_chase = bool(actor_to_target.strong_chase or target_to_actor.strong_chase)
    strong_attack = bool(actor_to_target.strong_attack or target_to_actor.strong_attack)
    conf_values = np.concatenate([actor.keypoint_conf, target.keypoint_conf])
    valid_conf = conf_values[np.isfinite(conf_values)]
    pose_quality = float(np.mean(valid_conf)) if len(valid_conf) else 0.0
    identity_quality_map = {
        "tracked": 1.0,
        "tentative": 0.65,
        "suspicious": 0.35,
        "lost": 0.0,
        "cluster_anonymous": 0.0,
        "post_split_anonymous": 0.0,
        "reid_ambiguous": 0.20,
    }
    actor_track_state = str(getattr(actor, "track_state", "tracked"))
    target_track_state = str(getattr(target, "track_state", "tracked"))
    actor_identity_quality = float(identity_quality_map.get(actor_track_state, 0.50))
    target_identity_quality = float(identity_quality_map.get(target_track_state, 0.50))
    identity_pair_quality = float(math.sqrt(actor_identity_quality * target_identity_quality))
    record = (
        dict(record_template)
        if record_template is not None
        else empty_frame_record(frame_idx, fps, transformer)
    )
    record.update({
        "pair_key": _pair_key(actor.logical_id, target.logical_id),
        "valid_pair": True,
        "mouse_a_id": actor.logical_id,
        "mouse_b_id": target.logical_id,
        "mouse_a_raw_track_id": actor.raw_track_id if actor.raw_track_id is not None else np.nan,
        "mouse_b_raw_track_id": target.raw_track_id if target.raw_track_id is not None else np.nan,
        "mouse_a_track_state": actor_track_state,
        "mouse_b_track_state": target_track_state,
        "identity_pair_quality": identity_pair_quality,
        "mouse_a_speed_cm_s": actor.speed_cm_s,
        "mouse_b_speed_cm_s": target.speed_cm_s,
        "center_distance_cm": point_distance(actor.center_cm, target.center_cm),
        "head_distance_cm": point_distance(actor.head_cm, target.head_cm),
        "trajectory_correlation": selected.trajectory_correlation,
        "direction_similarity": selected.direction_similarity,
        "pursuit_alignment": selected.pursuit_alignment,
        "target_escape_alignment": selected.target_escape_alignment,
        "actor_behind_target": selected.actor_behind_target,
        "a_to_b_actor_speed_cm_s": actor_to_target.actor_speed_cm_s,
        "a_to_b_target_speed_cm_s": actor_to_target.target_speed_cm_s,
        "a_to_b_actor_acceleration_cm_s2": actor_to_target.actor_acceleration_cm_s2,
        "a_to_b_target_acceleration_cm_s2": actor_to_target.target_acceleration_cm_s2,
        "a_to_b_actor_nose_speed_cm_s": actor_to_target.actor_nose_speed_cm_s,
        "a_to_b_target_nose_speed_cm_s": actor_to_target.target_nose_speed_cm_s,
        "a_to_b_actor_angular_speed_deg_s": actor_to_target.actor_angular_speed_deg_s,
        "a_to_b_target_angular_speed_deg_s": actor_to_target.target_angular_speed_deg_s,
        "a_to_b_actor_body_length_cm": actor_to_target.actor_body_length_cm,
        "a_to_b_target_body_length_cm": actor_to_target.target_body_length_cm,
        "a_to_b_actor_pose_deformation_energy": actor_to_target.actor_pose_deformation_energy,
        "a_to_b_target_pose_deformation_energy": actor_to_target.target_pose_deformation_energy,
        "a_to_b_center_distance_body_lengths": actor_to_target.center_distance_body_lengths,
        "a_to_b_closing_speed_cm_s": actor_to_target.closing_speed_cm_s,
        "a_to_b_actor_head_relative_speed_cm_s": actor_to_target.actor_head_relative_speed_cm_s,
        "a_to_b_target_head_relative_speed_cm_s": actor_to_target.target_head_relative_speed_cm_s,
        "a_to_b_direction_similarity": actor_to_target.direction_similarity,
        "a_to_b_pursuit_alignment": actor_to_target.pursuit_alignment,
        "a_to_b_target_escape_alignment": actor_to_target.target_escape_alignment,
        "a_to_b_behind_score": actor_to_target.behind_score,
        "a_to_b_actor_behind_target": actor_to_target.actor_behind_target,
        "a_to_b_trajectory_correlation": actor_to_target.trajectory_correlation,
        "a_to_b_target_turn_angle_deg": actor_to_target.target_turn_angle_deg,
        "a_to_b_nose_head_distance_cm": actor_to_target.nose_head_distance_cm,
        "a_to_b_nose_body_distance_cm": actor_to_target.nose_body_distance_cm,
        "a_to_b_nose_tail_distance_cm": actor_to_target.nose_tail_distance_cm,
        "b_to_a_actor_speed_cm_s": target_to_actor.actor_speed_cm_s,
        "b_to_a_target_speed_cm_s": target_to_actor.target_speed_cm_s,
        "b_to_a_actor_acceleration_cm_s2": target_to_actor.actor_acceleration_cm_s2,
        "b_to_a_target_acceleration_cm_s2": target_to_actor.target_acceleration_cm_s2,
        "b_to_a_actor_nose_speed_cm_s": target_to_actor.actor_nose_speed_cm_s,
        "b_to_a_target_nose_speed_cm_s": target_to_actor.target_nose_speed_cm_s,
        "b_to_a_actor_angular_speed_deg_s": target_to_actor.actor_angular_speed_deg_s,
        "b_to_a_target_angular_speed_deg_s": target_to_actor.target_angular_speed_deg_s,
        "b_to_a_actor_body_length_cm": target_to_actor.actor_body_length_cm,
        "b_to_a_target_body_length_cm": target_to_actor.target_body_length_cm,
        "b_to_a_actor_pose_deformation_energy": target_to_actor.actor_pose_deformation_energy,
        "b_to_a_target_pose_deformation_energy": target_to_actor.target_pose_deformation_energy,
        "b_to_a_center_distance_body_lengths": target_to_actor.center_distance_body_lengths,
        "b_to_a_closing_speed_cm_s": target_to_actor.closing_speed_cm_s,
        "b_to_a_actor_head_relative_speed_cm_s": target_to_actor.actor_head_relative_speed_cm_s,
        "b_to_a_target_head_relative_speed_cm_s": target_to_actor.target_head_relative_speed_cm_s,
        "b_to_a_direction_similarity": target_to_actor.direction_similarity,
        "b_to_a_pursuit_alignment": target_to_actor.pursuit_alignment,
        "b_to_a_target_escape_alignment": target_to_actor.target_escape_alignment,
        "b_to_a_behind_score": target_to_actor.behind_score,
        "b_to_a_actor_behind_target": target_to_actor.actor_behind_target,
        "b_to_a_trajectory_correlation": target_to_actor.trajectory_correlation,
        "b_to_a_target_turn_angle_deg": target_to_actor.target_turn_angle_deg,
        "b_to_a_nose_head_distance_cm": target_to_actor.nose_head_distance_cm,
        "b_to_a_nose_body_distance_cm": target_to_actor.nose_body_distance_cm,
        "b_to_a_nose_tail_distance_cm": target_to_actor.nose_tail_distance_cm,
        "selected_actor_wall_jump": selected.actor_wall_jump,
        "selected_target_wall_jump": selected.target_wall_jump,
        "pair_wall_jump_excluded": bool(selected.actor_wall_jump or selected.target_wall_jump),
        "selected_actor_id": selected.actor_id,
        "selected_target_id": selected.target_id,
        "selected_nose_body_distance_cm": selected.nose_body_distance_cm,
        "selected_target_turn_angle_deg": selected.target_turn_angle_deg,
        "selected_distance_drop_cm": selected.distance_drop_cm,
        "selected_actor_speed_cm_s": selected.actor_speed_cm_s,
        "selected_target_speed_cm_s": selected.target_speed_cm_s,
        "selected_weak_chase_score": selected.weak_chase_score,
        "selected_strong_chase_score": selected.strong_chase_score,
        "selected_weak_attack_evidence": int(selected.weak_attack_evidence),
        "selected_strong_attack_evidence": selected.strong_attack_evidence,
        "a_to_b_weak_chase": actor_to_target.weak_chase,
        "b_to_a_weak_chase": target_to_actor.weak_chase,
        "a_to_b_strong_chase": actor_to_target.strong_chase,
        "b_to_a_strong_chase": target_to_actor.strong_chase,
        "a_to_b_weak_attack": actor_to_target.weak_attack,
        "b_to_a_weak_attack": target_to_actor.weak_attack,
        "a_to_b_strong_attack": actor_to_target.strong_attack,
        "b_to_a_strong_attack": target_to_actor.strong_attack,
        "weak_contact": bool(actor_to_target.weak_contact or target_to_actor.weak_contact),
        "strong_contact": bool(actor_to_target.strong_contact or target_to_actor.strong_contact),
        "weak_potential_attack": bool(actor_to_target.weak_potential_attack or target_to_actor.weak_potential_attack),
        "strong_potential_attack": bool(actor_to_target.strong_potential_attack or target_to_actor.strong_potential_attack),
        "weak_attack_actor_initiation": bool(selected.weak_attack_actor_initiation),
        "strong_attack_actor_initiation": bool(selected.strong_attack_actor_initiation),
        "weak_attack_target_reaction": bool(selected.weak_attack_target_reaction),
        "strong_attack_target_reaction": bool(selected.strong_attack_target_reaction),
        "repeated_contact_count": int(repeated_contact_count),
        "weak_raw_chase": weak_chase,
        "weak_raw_attack": weak_attack,
        "weak_raw_label_id": int(weak_chase) + 2 * int(weak_attack),
        "strong_raw_chase": strong_chase,
        "strong_raw_attack": strong_attack,
        "strong_raw_label_id": int(strong_chase) + 2 * int(strong_attack),
        "pose_pair_quality": pose_quality,
        "cluster_attack_hint": bool(cluster_attack_hint),
        **dict(pair_cluster_evidence),
    })
    return record


def _cluster_fallback_record(
    frame_idx: int,
    fps: float,
    transformer: Any,
    id_a: int,
    id_b: int,
    cluster_attack_evidence: Mapping[str, Any],
    cluster_reid_active_count: int,
    record_template: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the legacy missing-detection cluster fallback row."""
    record = (
        dict(record_template)
        if record_template is not None
        else empty_frame_record(frame_idx, fps, transformer)
    )
    pair_key = _pair_key(id_a, id_b)
    record.update({
        "pair_key": pair_key,
        "valid_pair": True,
        "mouse_a_id": int(id_a),
        "mouse_b_id": int(id_b),
        "selected_actor_id": np.nan,
        "selected_target_id": np.nan,
        "selected_weak_attack_evidence": 0,
        "weak_contact": False,
        "weak_potential_attack": False,
        "weak_raw_attack": False,
        "weak_raw_label_id": 0,
        "pose_pair_quality": 0.0,
        "cluster_attack_hint": True,
        **dict(cluster_attack_evidence),
        "identity_ambiguous": bool(cluster_reid_active_count > 0),
        "identity_candidate_set": f"{id_a},{id_b}" if cluster_reid_active_count > 0 else "",
    })
    return record


def _numpy_pair_geometry(
    observations: Sequence[base.MouseObservation],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute all unordered-pair geometry with NumPy broadcasting.

    The returned arrays use the same upper-triangular order as
    ``np.triu_indices``.  Invalid keypoints are ignored exactly like
    ``min_point_distance``; a pair with no valid point is represented by NaN.
    """
    count = len(observations)
    pair_i, pair_j = np.triu_indices(count, k=1)
    if len(pair_i) == 0:
        return np.empty((0, 2), dtype=np.int64), {
            name: np.empty(0, dtype=np.float64)
            for name in (
                "center_distance_cm",
                "head_distance_cm",
                "nose_head_ab_cm",
                "nose_head_ba_cm",
                "nose_body_ab_cm",
                "nose_body_ba_cm",
                "nose_tail_ab_cm",
                "nose_tail_ba_cm",
            )
        }
    centers = np.asarray([observation.center_cm for observation in observations], dtype=np.float64)
    heads = np.asarray([observation.head_cm for observation in observations], dtype=np.float64)
    keypoints = np.asarray([observation.keypoints_cm for observation in observations], dtype=np.float64)
    nose = keypoints[:, KP["nose"]]
    tail = keypoints[:, KP["tail"]]

    def safe_norm(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(values, axis=-1)
        return np.where(valid, distances, np.nan).astype(np.float64)

    center_delta = centers[pair_j] - centers[pair_i]
    head_delta = heads[pair_j] - heads[pair_i]
    center_valid = np.all(np.isfinite(centers[pair_i]), axis=1) & np.all(
        np.isfinite(centers[pair_j]), axis=1
    )
    head_valid = np.all(np.isfinite(heads[pair_i]), axis=1) & np.all(
        np.isfinite(heads[pair_j]), axis=1
    )
    nose_i_valid = np.all(np.isfinite(nose[pair_i]), axis=1)
    nose_j_valid = np.all(np.isfinite(nose[pair_j]), axis=1)
    target_j_valid = np.all(np.isfinite(keypoints[pair_j]), axis=2)
    target_i_valid = np.all(np.isfinite(keypoints[pair_i]), axis=2)
    head_indices = np.asarray([KP["nose"], KP["left_ear"], KP["right_ear"]], dtype=np.int64)
    head_ab_raw = np.linalg.norm(
        keypoints[pair_j][:, head_indices, :] - nose[pair_i, None, :], axis=2
    )
    head_ba_raw = np.linalg.norm(
        keypoints[pair_i][:, head_indices, :] - nose[pair_j, None, :], axis=2
    )
    head_ab_valid = nose_i_valid[:, None] & target_j_valid[:, head_indices]
    head_ba_valid = nose_j_valid[:, None] & target_i_valid[:, head_indices]
    nose_head_ab = np.min(np.where(head_ab_valid, head_ab_raw, np.inf), axis=1)
    nose_head_ba = np.min(np.where(head_ba_valid, head_ba_raw, np.inf), axis=1)
    nose_head_ab = np.where(np.isfinite(nose_head_ab), nose_head_ab, np.nan)
    nose_head_ba = np.where(np.isfinite(nose_head_ba), nose_head_ba, np.nan)
    body_ab_raw = np.linalg.norm(keypoints[pair_j] - nose[pair_i, None, :], axis=2)
    body_ba_raw = np.linalg.norm(keypoints[pair_i] - nose[pair_j, None, :], axis=2)
    body_ab_valid = nose_i_valid[:, None] & target_j_valid
    body_ba_valid = nose_j_valid[:, None] & target_i_valid
    body_ab = np.min(np.where(body_ab_valid, body_ab_raw, np.inf), axis=1)
    body_ba = np.min(np.where(body_ba_valid, body_ba_raw, np.inf), axis=1)
    body_ab = np.where(np.isfinite(body_ab), body_ab, np.nan)
    body_ba = np.where(np.isfinite(body_ba), body_ba, np.nan)
    tail_ab = safe_norm(tail[pair_j] - nose[pair_i], nose_i_valid & np.all(np.isfinite(tail[pair_j]), axis=1))
    tail_ba = safe_norm(tail[pair_i] - nose[pair_j], nose_j_valid & np.all(np.isfinite(tail[pair_i]), axis=1))
    return np.column_stack((pair_i, pair_j)).astype(np.int64), {
        "center_distance_cm": safe_norm(center_delta, center_valid),
        "head_distance_cm": safe_norm(head_delta, head_valid),
        "nose_head_ab_cm": nose_head_ab.astype(np.float64),
        "nose_head_ba_cm": nose_head_ba.astype(np.float64),
        "nose_body_ab_cm": body_ab.astype(np.float64),
        "nose_body_ba_cm": body_ba.astype(np.float64),
        "nose_tail_ab_cm": tail_ab,
        "nose_tail_ba_cm": tail_ba,
    }


def _compute_behavior_frame_records(
    frame_idx: int,
    fps: float,
    width: int,
    height: int,
    observations: Sequence[base.MouseObservation],
    cluster_context: Mapping[str, Any],
    transformer: Any,
    config: Mapping[str, Any],
    history: base.ObservationHistory,
    feature_computer: PairFeatureComputer,
    individual_behavior_gate: IndividualBehaviorGate,
    contact_tracker: base.PairContactTracker,
    tracking_only: bool = False,
    pair_compute_mode: str = "python",
    cluster_reid_active_count: int = 0,
) -> List[Dict[str, Any]]:
    """Run the established steps 4--7 for one cached frame.

    ``pair_compute_mode`` controls only pair enumeration.  The default Python
    path is the compatibility implementation.  The NumPy backend computes the
    unordered index pairs without nested Python ``combinations`` calls; feature
    objects are still serialized in deterministic ID order.  Multiprocess
    chunking is handled outside this frame-level function so stateful history
    and contact windows remain ordered.
    """
    if tracking_only:
        return []
    # Every pair row starts from the same frame metadata. Copying one
    # prepared template preserves key order and values while avoiding
    # construction of the large record dictionary up to 190 times/frame.
    record_template = empty_frame_record(frame_idx, fps, transformer)
    behavior_observations = _behavior_eligible_observations(observations)
    wall_jump_flags = individual_behavior_gate.update(
        frame_idx, behavior_observations, width, height
    )
    cluster_attack_evidence = _cluster_attack_evidence_by_pair(cluster_context)
    cluster_attack_pairs = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in cluster_context.get("attack_pairs", set())
    }
    pair_geometry = None
    graph_cfg = dict(config.get("standard_behavior_engine", {}).get("interaction_graph", {}))
    graph_prune = bool(
        config.get("standard_behavior_engine", {}).get("enabled", True)
        and graph_cfg.get("enabled", True)
        and graph_cfg.get("prune_pair_computation", False)
    )
    graph_radius_cm = float(
        graph_cfg.get(
            "radius_cm",
            float(config["chase"]["weak"]["max_distance_cm"])
            + float(graph_cfg.get("buffer_cm", 5.0)),
        )
    )
    if pair_compute_mode in {"numpy", "multiprocess"}:
        pair_indices, pair_geometry = _numpy_pair_geometry(behavior_observations)
        if graph_prune and len(pair_indices):
            centers = pair_geometry["center_distance_cm"]
            keep = np.isfinite(centers) & (centers <= graph_radius_cm)
            # Occlusion evidence always survives spatial pruning because a merged
            # pair may temporarily have no trustworthy independent centers.
            for index, (i, j) in enumerate(pair_indices):
                pair_ids = tuple(sorted((
                    int(behavior_observations[int(i)].logical_id),
                    int(behavior_observations[int(j)].logical_id),
                )))
                if pair_ids in cluster_attack_pairs:
                    keep[index] = True
            pair_indices = pair_indices[keep]
            pair_geometry = {name: values[keep] for name, values in pair_geometry.items()}
        pair_iter: Iterable[
            Tuple[base.MouseObservation, base.MouseObservation, Optional[Mapping[str, float]], Optional[Mapping[str, float]]]
        ] = (
            (
                behavior_observations[int(i)],
                behavior_observations[int(j)],
                {
                    "center_distance_cm": pair_geometry["center_distance_cm"][index],
                    "head_distance_cm": pair_geometry["head_distance_cm"][index],
                    "nose_head_distance_cm": pair_geometry["nose_head_ab_cm"][index],
                    "nose_body_distance_cm": pair_geometry["nose_body_ab_cm"][index],
                    "nose_tail_distance_cm": pair_geometry["nose_tail_ab_cm"][index],
                },
                {
                    "center_distance_cm": pair_geometry["center_distance_cm"][index],
                    "head_distance_cm": pair_geometry["head_distance_cm"][index],
                    "nose_head_distance_cm": pair_geometry["nose_head_ba_cm"][index],
                    "nose_body_distance_cm": pair_geometry["nose_body_ba_cm"][index],
                    "nose_tail_distance_cm": pair_geometry["nose_tail_ba_cm"][index],
                },
            )
            for index, (i, j) in enumerate(pair_indices)
        )
    else:
        base_pairs = combinations(behavior_observations, 2)
        if graph_prune:
            def _graph_candidate(pair: Tuple[base.MouseObservation, base.MouseObservation]) -> bool:
                actor, target = pair
                pair_ids = tuple(sorted((int(actor.logical_id), int(target.logical_id))))
                if pair_ids in cluster_attack_pairs:
                    return True
                distance = point_distance(actor.center_cm, target.center_cm)
                return bool(np.isfinite(distance) and distance <= graph_radius_cm)

            pair_iter = (
                (actor, target, None, None)
                for actor, target in filter(_graph_candidate, base_pairs)
            )
        else:
            pair_iter = ((actor, target, None, None) for actor, target in base_pairs)

    records: List[Dict[str, Any]] = []
    existing_pair_keys: set[str] = set()
    for actor, target, geometry_ab, geometry_ba in pair_iter:
        d_ab = (
            float(geometry_ab["nose_body_distance_cm"])
            if geometry_ab is not None
            else min_point_distance(actor.keypoints_cm[KP["nose"]], target.keypoints_cm)
        )
        d_ba = (
            float(geometry_ba["nose_body_distance_cm"])
            if geometry_ba is not None
            else min_point_distance(target.keypoints_cm[KP["nose"]], actor.keypoints_cm)
        )
        weak_contact_threshold = float(config["attack"]["weak"]["contact_distance_cm"])
        symmetric_weak_contact = bool(
            (np.isfinite(d_ab) and d_ab < weak_contact_threshold)
            or (np.isfinite(d_ba) and d_ba < weak_contact_threshold)
        )
        repeated = contact_tracker.update(
            actor.logical_id, target.logical_id, frame_idx, symmetric_weak_contact
        )
        actor_wall_jump = bool(wall_jump_flags.get(int(actor.logical_id), False))
        target_wall_jump = bool(wall_jump_flags.get(int(target.logical_id), False))
        actor_to_target = feature_computer.compute(
            actor, target, history, repeated, actor_wall_jump, target_wall_jump, geometry_ab
        )
        target_to_actor = feature_computer.compute(
            target, actor, history, repeated, target_wall_jump, actor_wall_jump, geometry_ba
        )
        selected = choose_direction(actor_to_target, target_to_actor)
        pair_tuple = tuple(sorted((int(actor.logical_id), int(target.logical_id))))
        pair_cluster_evidence = cluster_attack_evidence.get(pair_tuple, {})
        record = _pair_record_from_features(
            frame_idx,
            fps,
            transformer,
            actor,
            target,
            actor_to_target,
            target_to_actor,
            selected,
            repeated,
            pair_tuple in cluster_attack_pairs,
            pair_cluster_evidence,
            record_template=record_template,
        )
        records.append(record)
        existing_pair_keys.add(str(record["pair_key"]))

    # Preserve the existing occlusion-cluster fallback rows exactly.  These rows
    # are evidence-only and never assign an actor/target direction.
    for id_a, id_b in sorted(cluster_attack_pairs):
        pair_key = _pair_key(id_a, id_b)
        if pair_key in existing_pair_keys:
            continue
        records.append(
            _cluster_fallback_record(
                frame_idx,
                fps,
                transformer,
                id_a,
                id_b,
                cluster_attack_evidence.get(tuple(sorted((int(id_a), int(id_b)))), {}),
                cluster_reid_active_count,
                record_template=record_template,
            )
        )
        existing_pair_keys.add(pair_key)
    return records


def populate_pair_store_from_stage3_cache(
    cache: Stage3ObservationCache,
    raw_store: PairSQLiteStore,
    fps: float,
    width: int,
    height: int,
    config: Mapping[str, Any],
    pair_compute_mode: str = "python",
    pair_workers: int = 0,
    profiler: Optional[RuntimeProfiler] = None,
) -> int:
    """Run steps 4--7 over the durable stage-3 cache.

    The stateful history, wall-jump gate, and repeated-contact tracker are
    rebuilt in frame order.  Thus the staged path uses exactly the same
    temporal windows as the inline path while keeping YOLO/identity inference
    completely out of this CPU pass.  Multiprocess workers receive overlapping
    temporal warm-up prefixes and return only their core frames, preserving
    deterministic row order at chunk boundaries.
    """
    if pair_compute_mode == "multiprocess":
        requested_workers = int(pair_workers)
        worker_count = requested_workers if requested_workers > 1 else max(
            2, min(int(os.cpu_count() or 2), 4)
        )
        configured_chunk_frames = max(
            int(config.get("performance", {}).get("stage3_cache_chunk_frames", 300)),
            1,
        )
        chunk_frames = configured_chunk_frames
        # A short cache used to collapse into one 300-frame task, which meant a
        # nominal four-worker Stage 2 actually used only one CPU process.  Keep
        # the configured upper bound for long videos, but expose at least one
        # core chunk per worker when the complete frame count is known.
        if cache.total_frames is not None and int(cache.total_frames) > 0:
            chunk_frames = min(
                configured_chunk_frames,
                max(int(math.ceil(int(cache.total_frames) / worker_count)), 1),
            )
        logging.info(
            "阶段4-7多进程：%d个worker，核心块%d帧（配置上限%d）；每块带历史预热并按原帧序写入SQLite。",
            worker_count,
            chunk_frames,
            configured_chunk_frames,
        )
        payloads = _iter_stage3_chunk_payloads(
            cache=cache,
            fps=fps,
            width=width,
            height=height,
            config=config,
            pair_mode="numpy",
            chunk_frames=chunk_frames,
        )
        processed_frames = 0
        multiprocess_started = time.perf_counter()
        sqlite_seconds = 0.0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_stage3_worker_initializer,
        ) as executor:
            for core_count, records in executor.map(_stage3_chunk_worker, payloads):
                write_started = time.perf_counter()
                raw_store.add_many(records)
                sqlite_seconds += time.perf_counter() - write_started
                processed_frames += int(core_count)
        if profiler is not None:
            total_seconds = time.perf_counter() - multiprocess_started
            profiler.add("csv_sqlite", sqlite_seconds)
            profiler.add("behavior_pair", max(total_seconds - sqlite_seconds, 0.0))
        return processed_frames

    history_seconds = max(float(config["features"].get("history_seconds", 1.0)), 1.0)
    history = base.ObservationHistory(
        max_frames=max(int(round(fps * history_seconds)) + 5, 10)
    )
    feature_computer = PairFeatureComputer(fps, config)
    individual_behavior_gate = IndividualBehaviorGate(fps, config)
    contact_tracker = base.PairContactTracker(
        fps=fps,
        window_seconds=float(config["attack"]["weak"]["repeated_contact_window_seconds"]),
    )
    if pair_compute_mode == "multiprocess":
        logging.warning(
            "阶段4-7多进程暂以有序NumPy枚举执行（workers=%d）；" 
            "在保留跨帧历史/接触状态的前提下避免错误切断行为事件。",
            max(int(pair_workers), 0),
        )
        pair_compute_mode = "numpy"
    processed_frames = 0
    for entry in cache.iter_frames():
        frame_started = time.perf_counter()
        frame_idx = int(entry["frame"])
        observations = sorted(
            list(entry.get("observations", [])),
            key=lambda observation: int(observation.logical_id),
        )
        for observation in observations:
            history.add(observation)
        transformer = _Stage3TransformerStub(
            str(entry.get("scale_mode", "unknown")),
            entry.get("cm_per_pixel"),
        )
        pair_started = time.perf_counter()
        records = _compute_behavior_frame_records(
            frame_idx=frame_idx,
            fps=fps,
            width=width,
            height=height,
            observations=observations,
            cluster_context=dict(entry.get("cluster_context", {})),
            transformer=transformer,
            config=config,
            history=history,
            feature_computer=feature_computer,
            individual_behavior_gate=individual_behavior_gate,
            contact_tracker=contact_tracker,
            tracking_only=False,
            pair_compute_mode=pair_compute_mode,
            cluster_reid_active_count=0,
        )
        if profiler is not None:
            profiler.add("behavior_pair", time.perf_counter() - pair_started)
        write_started = time.perf_counter()
        raw_store.add_many(records)
        if profiler is not None:
            profiler.add("csv_sqlite", time.perf_counter() - write_started)
        processed_frames += 1
        if profiler is not None:
            profiler.add("behavior_io", time.perf_counter() - frame_started)
    if pair_workers and int(pair_workers) > 1:
        logging.info(
            "阶段4-7已完成有序后端；pair_workers=%d仅用于保留兼容配置，不改变结果。",
            int(pair_workers),
        )
    return processed_frames


def _pair_geometry_python_kernel(
    centers: np.ndarray,
    keypoints: np.ndarray,
    repeats: int,
) -> float:
    """Reference nested-loop geometry kernel used only for benchmarking."""
    total = 0.0
    count = int(len(centers))
    for _ in range(max(int(repeats), 1)):
        for i in range(count):
            for j in range(i + 1, count):
                total += float(np.linalg.norm(centers[j] - centers[i]))
                nose_i = keypoints[i, KP["nose"]]
                nose_j = keypoints[j, KP["nose"]]
                total += float(np.nanmin(np.linalg.norm(keypoints[j] - nose_i, axis=1)))
                total += float(np.nanmin(np.linalg.norm(keypoints[i] - nose_j, axis=1)))
    return total


def _pair_geometry_numpy_kernel(
    centers: np.ndarray,
    keypoints: np.ndarray,
    repeats: int,
) -> float:
    """Vectorized unordered-pair geometry kernel used for benchmarking."""
    pair_i, pair_j = np.triu_indices(int(len(centers)), k=1)
    total = 0.0
    for _ in range(max(int(repeats), 1)):
        center_delta = centers[pair_j] - centers[pair_i]
        total += float(np.linalg.norm(center_delta, axis=1).sum())
        nose_i = keypoints[pair_i, KP["nose"]]
        nose_j = keypoints[pair_j, KP["nose"]]
        body_j = np.linalg.norm(keypoints[pair_j] - nose_i[:, None, :], axis=2)
        body_i = np.linalg.norm(keypoints[pair_i] - nose_j[:, None, :], axis=2)
        total += float(np.nanmin(body_j, axis=1).sum())
        total += float(np.nanmin(body_i, axis=1).sum())
    return total


def _pair_geometry_chunk_worker(payload: Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]) -> float:
    """Pickle-safe worker for the optional multiprocess benchmark."""
    centers, keypoints, repeats, pair_i, pair_j = payload
    total = 0.0
    for _ in range(max(int(repeats), 1)):
        center_delta = centers[pair_j] - centers[pair_i]
        total += float(np.linalg.norm(center_delta, axis=1).sum())
        nose_i = keypoints[pair_i, KP["nose"]]
        nose_j = keypoints[pair_j, KP["nose"]]
        total += float(np.nanmin(np.linalg.norm(keypoints[pair_j] - nose_i[:, None, :], axis=2), axis=1).sum())
        total += float(np.nanmin(np.linalg.norm(keypoints[pair_i] - nose_j[:, None, :], axis=2), axis=1).sum())
    return total


_STAGE3_THREADPOOL_LIMITER: Any = None


def _stage3_worker_initializer() -> None:
    """限制行为worker内部BLAS/OpenCV线程，避免4进程再各自开多线程超卖CPU。"""
    global _STAGE3_THREADPOOL_LIMITER
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    # Windows spawn 会先导入NumPy再调用initializer，单改环境变量可能过晚；
    # threadpoolctl可对已经加载的BLAS/OpenMP runtime立即限为1线程。
    try:
        from threadpoolctl import threadpool_limits
        _STAGE3_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _STAGE3_THREADPOOL_LIMITER = None
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _stage3_chunk_worker(
    payload: Tuple[
        List[Dict[str, Any]],
        int,
        float,
        int,
        int,
        Mapping[str, Any],
        str,
    ]
) -> Tuple[int, List[Dict[str, Any]]]:
    """Process one stage-3 chunk in a spawned worker.

    The payload begins with a temporal warm-up prefix.  The worker rebuilds all
    stateful windows from that prefix and returns only the chunk's core frames,
    so parallel boundaries cannot duplicate rows or reset contact onsets.
    """
    entries, core_start_frame, fps, width, height, config, pair_mode = payload
    history_seconds = max(float(config["features"].get("history_seconds", 1.0)), 1.0)
    history = base.ObservationHistory(
        max_frames=max(int(round(fps * history_seconds)) + 5, 10)
    )
    feature_computer = PairFeatureComputer(fps, config)
    individual_behavior_gate = IndividualBehaviorGate(fps, config)
    contact_tracker = base.PairContactTracker(
        fps=fps,
        window_seconds=float(config["attack"]["weak"]["repeated_contact_window_seconds"]),
    )
    records: List[Dict[str, Any]] = []
    core_count = 0
    for entry in entries:
        frame_idx = int(entry["frame"])
        observations = sorted(
            list(entry.get("observations", [])),
            key=lambda observation: int(observation.logical_id),
        )
        for observation in observations:
            history.add(observation)
        frame_records = _compute_behavior_frame_records(
            frame_idx=frame_idx,
            fps=fps,
            width=width,
            height=height,
            observations=observations,
            cluster_context=dict(entry.get("cluster_context", {})),
            transformer=_Stage3TransformerStub(
                str(entry.get("scale_mode", "unknown")),
                entry.get("cm_per_pixel"),
            ),
            config=config,
            history=history,
            feature_computer=feature_computer,
            individual_behavior_gate=individual_behavior_gate,
            contact_tracker=contact_tracker,
            tracking_only=False,
            pair_compute_mode=pair_mode,
            cluster_reid_active_count=0,
        )
        if frame_idx >= int(core_start_frame):
            records.extend(frame_records)
            core_count += 1
    return core_count, records


def _iter_stage3_chunk_payloads(
    cache: Stage3ObservationCache,
    fps: float,
    width: int,
    height: int,
    config: Mapping[str, Any],
    pair_mode: str,
    chunk_frames: int,
) -> Iterable[
    Tuple[List[Dict[str, Any]], int, float, int, int, Mapping[str, Any], str]
]:
    """Yield bounded worker payloads with enough prefix for temporal state."""
    feature_frames = max(
        int(round(fps * float(config["features"].get("history_seconds", 1.0)))) + 5,
        10,
    )
    lookback_frames = max(
        int(round(fps * float(config["features"].get("response_lookback_seconds", 0.3)))) + 5,
        5,
    )
    contact_frames = max(
        int(round(fps * float(config["attack"]["weak"].get("repeated_contact_window_seconds", 1.0)))) + 5,
        5,
    )
    warmup_limit = max(feature_frames, lookback_frames, contact_frames)
    warmup: Deque[Dict[str, Any]] = deque(maxlen=warmup_limit)
    core: List[Dict[str, Any]] = []
    for entry in cache.iter_frames():
        core.append(entry)
        if len(core) < max(int(chunk_frames), 1):
            continue
        core_start = int(core[0]["frame"])
        yield (list(warmup) + core, core_start, fps, width, height, config, pair_mode)
        warmup.extend(core)
        core = []
    if core:
        core_start = int(core[0]["frame"])
        yield (list(warmup) + core, core_start, fps, width, height, config, pair_mode)


def benchmark_pair_backends(
    mouse_count: int = 20,
    repeats: int = 200,
    workers: int = 0,
) -> Dict[str, Any]:
    """Compare Python, NumPy and optional multiprocessing pair kernels.

    This benchmark isolates the CPU pair geometry that is moved after stage 3;
    it does not load YOLO or alter any behavior thresholds.  The checksum is
    included so an optimizer cannot accidentally benchmark an empty result.
    """
    count = max(int(mouse_count), 2)
    rng = np.random.default_rng(20260806)
    centers = rng.normal(size=(count, 2)).astype(np.float64)
    keypoints = rng.normal(size=(count, len(KEYPOINT_NAMES), 2)).astype(np.float64)
    timings: Dict[str, Any] = {"mouse_count": count, "repeats": int(repeats)}
    started = time.perf_counter()
    python_checksum = _pair_geometry_python_kernel(centers, keypoints, repeats)
    python_seconds = time.perf_counter() - started
    started = time.perf_counter()
    numpy_checksum = _pair_geometry_numpy_kernel(centers, keypoints, repeats)
    numpy_seconds = time.perf_counter() - started
    timings.update({
        "python_seconds": float(python_seconds),
        "numpy_seconds": float(numpy_seconds),
        "numpy_speedup": float(python_seconds / max(numpy_seconds, 1e-12)),
        "python_checksum": float(python_checksum),
        "numpy_checksum": float(numpy_checksum),
        "numpy_checksum_match": bool(np.isclose(python_checksum, numpy_checksum, rtol=1e-10, atol=1e-10)),
    })
    if int(workers) > 1:
        pair_i, pair_j = np.triu_indices(count, k=1)
        chunks = np.array_split(np.arange(len(pair_i), dtype=np.int64), int(workers))
        payloads = [
            (centers, keypoints, int(repeats), pair_i[chunk], pair_j[chunk])
            for chunk in chunks if len(chunk)
        ]
        started = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=mp.get_context("spawn"),
            initializer=_stage3_worker_initializer,
        ) as executor:
            values = list(executor.map(_pair_geometry_chunk_worker, payloads))
        multiprocess_seconds = time.perf_counter() - started
        multiprocess_checksum = float(sum(values))
        timings.update({
            "multiprocess_seconds": float(multiprocess_seconds),
            "multiprocess_speedup_vs_python": float(python_seconds / max(multiprocess_seconds, 1e-12)),
            "multiprocess_checksum": multiprocess_checksum,
            "multiprocess_checksum_match": bool(np.isclose(python_checksum, multiprocess_checksum, rtol=1e-10, atol=1e-10)),
            "multiprocess_workers": int(workers),
        })
    return timings


class PairDataFrameStore:
    """从已导出的鼠对CSV重放行为后处理，避免重新执行YOLO姿态推理。"""

    def __init__(self, path: Path) -> None:
        # 一次性载入当前短视频的鼠对特征；这里只用于快速行为规则复算。
        self.table = pd.read_csv(path, encoding="utf-8-sig")
        # 缓存文件必须保留原始鼠对键，否则无法按独立鼠对执行时序滤波。
        if "pair_key" not in self.table.columns:
            raise ValueError(f"行为缓存缺少pair_key字段：{path}")
        # Build the row index once. The previous implementation converted and
        # scanned the complete pair_key column for every pair: O(K × N).
        key_series = self.table["pair_key"].astype("string")
        grouped = key_series.groupby(key_series, sort=False).groups
        self._pair_indices: Dict[str, np.ndarray] = {
            str(key): np.asarray(indices, dtype=np.int64)
            for key, indices in grouped.items()
            if not pd.isna(key)
        }
        self._pair_keys = sorted(self._pair_indices)

    def pair_keys(self) -> List[str]:
        # 返回副本，避免调用方修改内部稳定顺序。
        return list(self._pair_keys)

    def read_pair(self, pair_key: str) -> pd.DataFrame:
        indices = self._pair_indices.get(str(pair_key))
        if indices is None:
            return self.table.iloc[0:0].copy()
        # 返回副本，避免后处理新增列时污染其他鼠对或后续重复读取。
        pair = self.table.iloc[indices].copy()
        # 帧号排序保持时间滤波输入严格单调。
        return pair.sort_values("frame", kind="stable").reset_index(drop=True)


def _open_dict_writer(
    path: Path,
    fieldnames: Sequence[str],
    append: bool = False,
) -> Tuple[Any, csv.DictWriter]:
    existing = bool(append and path.exists() and path.stat().st_size > 0)
    handle = path.open("a" if existing else "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
    if not existing:
        writer.writeheader()
    return handle, writer


def _draw_online_tracking_frame(
    frame: np.ndarray,
    observations: Sequence[base.MouseObservation],
    active_records: Sequence[Mapping[str, Any]],
    frame_idx: int,
    max_mice: int,
    visualization_cfg: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    当前帧立即绘制。

    重要：active_records仅用于行为CSV，不参与渲染。画面中只保留每只小鼠的
    检测框、7点骨架和小号ID，彻底禁止鼠对箭头、中心连线、轨迹线和行为文字面板。
    """
    del active_records, frame_idx, max_mice
    for obs in observations:
        _draw_clean_mouse_overlay(frame, obs, visualization_cfg)


def _positive_near(sorted_positive: np.ndarray, frame: int, padding: int) -> bool:
    if sorted_positive.size == 0:
        return False
    left = int(np.searchsorted(sorted_positive, frame - padding, side="left"))
    return left < len(sorted_positive) and int(sorted_positive[left]) <= frame + padding


def postprocess_pair_store(
    store: PairSQLiteStore,
    fps: float,
    config: Mapping[str, Any],
    output_dir: Path,
    total_frames: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Counter, Dict[int, Dict[str, Any]]]:
    """逐鼠对读取SQLite并处理；内存规模约等于单个鼠对，而非整段视频所有鼠对。"""
    pair_output = output_dir / "成对行为标签.csv"
    # 每次使用唯一临时文件，避免两个复算进程互相删除或覆盖对方缓存。
    pair_write_output = output_dir / (
        f".成对行为标签.{uuid.uuid4().hex}.写入中.csv"
    )
    first_write = True
    weak_events: List[Dict[str, Any]] = []
    strong_events: List[Dict[str, Any]] = []
    standard_ethogram_events: List[Dict[str, Any]] = []
    frame_active_count: Counter = Counter()
    frame_top: Dict[int, Dict[str, Any]] = {}

    clips_cfg = config["clips"]
    negative_enabled = bool(clips_cfg.get("extract_hard_negatives", True))
    neg_interval = max(int(round(float(clips_cfg.get("hard_negative_min_interval_seconds", 5.0)) * fps)), 1)
    negative_bins: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    pair_keys = store.pair_keys()
    for pair_key in tqdm(pair_keys, desc="逐鼠对时序后处理", unit="pair"):
        pair_group = store.read_pair(pair_key)
        if pair_group.empty:
            continue
        mouse_a_id = int(pd.to_numeric(pair_group["mouse_a_id"], errors="coerce").dropna().iloc[0])
        mouse_b_id = int(pd.to_numeric(pair_group["mouse_b_id"], errors="coerce").dropna().iloc[0])
        for chunk in _contiguous_chunks(pair_group):
            processed = postprocess_frame_labels(chunk, fps, config)
            processed.to_csv(
                pair_write_output,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig" if first_write else "utf-8",
            )
            first_write = False

            for level, target in (("weak", weak_events), ("strong", strong_events)):
                chunk_events = events_from_frames(processed, fps, config, level)
                for event in chunk_events:
                    event["pair_key"] = str(pair_key)
                    event["mouse_a_id"] = mouse_a_id
                    event["mouse_b_id"] = mouse_b_id
                target.extend(chunk_events)
                standard_ethogram_events.extend(
                    standard_behavior_engine.extract_standard_behavior_events(
                        processed, fps, level, pair_key=str(pair_key)
                    )
                )

            active = processed[processed["weak_final_label_id"].fillna(0).astype(int) != 0]
            for row in active.to_dict("records"):
                frame = int(row["frame"])
                frame_active_count[frame] += 1
                priority = (
                    int(row.get("weak_final_label_id", 0)) * 100
                    + int(safe_float(row.get("selected_weak_attack_evidence"), 0)) * 10
                    + int(safe_float(row.get("selected_weak_chase_score"), 0))
                )
                previous = frame_top.get(frame)
                if previous is None or priority > int(previous["_priority"]):
                    saved = dict(row)
                    saved["_priority"] = priority
                    frame_top[frame] = saved

            if negative_enabled:
                valid = processed["valid_pair"].fillna(False).astype(bool)
                zero = processed["weak_final_label_id"].fillna(0).astype(int).eq(0)
                close = processed["center_distance_cm"].fillna(np.inf) < float(clips_cfg["hard_negative_close_distance_cm"])
                max_speed = np.maximum(
                    processed["mouse_a_speed_cm_s"].fillna(0).to_numpy(),
                    processed["mouse_b_speed_cm_s"].fillna(0).to_numpy(),
                )
                fast = max_speed > float(clips_cfg["hard_negative_min_speed_cm_s"])
                candidates = processed[valid & zero & (close | fast)]
                for row in candidates.to_dict("records"):
                    frame = int(row["frame"])
                    bin_id = frame // neg_interval
                    item = dict(row)
                    distance = safe_float(item.get("center_distance_cm"), 1e9)
                    speed = max(safe_float(item.get("mouse_a_speed_cm_s"), 0), safe_float(item.get("mouse_b_speed_cm_s"), 0))
                    item["_negative_rank"] = (distance, -speed)
                    bucket = negative_bins[bin_id]
                    bucket.append(item)
                    bucket.sort(key=lambda x: x["_negative_rank"])
                    del bucket[5:]

        del pair_group
        gc.collect()

    # 没有有效鼠对时仍生成可识别的空CSV，而不是遗留旧结果。
    if first_write:
        pd.DataFrame().to_csv(
            pair_write_output,
            index=False,
            encoding="utf-8-sig",
        )
    # 所有鼠对成功写完后再原子替换正式CSV；异常中断不会破坏旧缓存。
    pair_write_output.replace(pair_output)
    # 独立ethogram：追逐与攻击分别聚合，不因四分类标签切换而把持续追逐拆段。
    standard_ethogram_path = output_dir / "标准行为事件_时序引擎.csv"
    if standard_ethogram_events:
        standard_ethogram_events.sort(
            key=lambda event: (
                str(event.get("candidate_level", "")),
                int(event.get("start_frame", -1)),
                str(event.get("pair_key", "")),
                str(event.get("behavior", "")),
            )
        )
        ethogram_counters: Counter = Counter()
        for event in standard_ethogram_events:
            level_key = str(event.get("candidate_level", "weak"))
            ethogram_counters[level_key] += 1
            event["standard_event_id"] = (
                f"{level_key[:1].upper()}SB{ethogram_counters[level_key]:06d}"
            )
        pd.DataFrame(standard_ethogram_events).to_csv(
            standard_ethogram_path, index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=[
            "standard_event_id", "behavior_engine", "candidate_level", "behavior",
            "subtype", "pair_key", "actor_id", "target_id", "role_ambiguous",
            "start_frame", "peak_frame", "end_frame", "start_time_s", "end_time_s",
            "duration_s", "mean_score", "peak_score", "mean_behavior_confidence",
            "mean_role_confidence",
        ]).to_csv(standard_ethogram_path, index=False, encoding="utf-8-sig")

    for prefix, events in (("WE", weak_events), ("SE", strong_events)):
        events.sort(key=lambda e: (int(e["start_frame"]), str(e.get("pair_key", ""))))
        for idx, event in enumerate(events, start=1):
            event["event_id"] = f"{prefix}{idx:05d}"

    negative_events: List[Dict[str, Any]] = []
    if negative_enabled:
        positive_sorted = np.asarray(sorted(frame_active_count.keys()), dtype=np.int64)
        pad = int(round(max(float(clips_cfg.get("pre_padding_seconds", 1.5)), float(clips_cfg.get("post_padding_seconds", 1.5))) * fps))
        last_frame = -10**12
        max_clips = int(clips_cfg.get("max_hard_negative_clips", 200))
        for bin_id in sorted(negative_bins):
            selected = None
            for candidate in negative_bins[bin_id]:
                frame = int(candidate["frame"])
                if frame - last_frame < neg_interval:
                    continue
                if _positive_near(positive_sorted, frame, pad):
                    continue
                selected = candidate
                break
            if selected is None:
                continue
            frame = int(selected["frame"])
            negative_events.append({
                "event_id": f"NE{len(weak_events) + len(negative_events) + 1:05d}",
                "candidate_level": "hard_negative",
                "pair_key": str(selected.get("pair_key", "")),
                "mouse_a_id": int(selected.get("mouse_a_id", -1)),
                "mouse_b_id": int(selected.get("mouse_b_id", -1)),
                "label_id": 0,
                "label_en": LABELS[0][0],
                "label_zh": LABELS[0][1],
                "actor_id": int(selected.get("selected_actor_id", selected.get("mouse_a_id", -1))),
                "target_id": int(selected.get("selected_target_id", selected.get("mouse_b_id", -1))),
                "start_frame": frame,
                "end_frame": frame,
                "start_time_s": frame / fps,
                "end_time_s": frame / fps,
                "duration_s": 1.0 / fps,
                "strict_chase": False,
                "strong_candidate_fraction": 0.0,
                "mean_pose_pair_quality": safe_float(selected.get("pose_pair_quality"), 0.0),
                "needs_manual_review": True,
                "is_hard_negative": True,
                "other_positive_event_count": 0,
                "max_other_event_overlap_fraction": 0.0,
                "clean_for_classifier": True,
                "clean_status": "clean_negative",
            })
            last_frame = frame
            if len(negative_events) >= max_clips:
                break

    return weak_events, strong_events, negative_events, frame_active_count, frame_top


def _load_pair_reconciliation_table(pair_path: Path) -> pd.DataFrame:
    """Load only the columns needed for event-level pair and role reconciliation."""
    # Keeping this list narrow avoids loading the large pose/debug portion of the pair cache.
    required_columns = [
        "frame",
        "mouse_a_id",
        "mouse_b_id",
        "center_distance_cm",
        "selected_actor_id",
        "selected_target_id",
        "selected_weak_chase_score",
        "selected_weak_attack_evidence",
    ]
    # Read the compact evidence table from the already postprocessed pair cache.
    table = pd.read_csv(
        pair_path,
        encoding="utf-8-sig",
        usecols=lambda column: column in required_columns,
        low_memory=False,
    )
    # An incomplete cache cannot support a safe event correction, so leave events unchanged.
    if not set(required_columns).issubset(table.columns):
        return pd.DataFrame(columns=required_columns)
    # Normalize historical caches that stored logical IDs as floating-point CSV values.
    for column in ("mouse_a_id", "mouse_b_id", "selected_actor_id", "selected_target_id"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    # Normalize all remaining numeric evidence before ranking candidate pairs.
    for column in (
        "frame",
        "center_distance_cm",
        "selected_weak_chase_score",
        "selected_weak_attack_evidence",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    # Rows without a complete formal pair cannot participate in event reconciliation.
    table = table.dropna(subset=["frame", "mouse_a_id", "mouse_b_id"]).copy()
    # Integer IDs make grouping stable across old and new cache formats.
    table["frame"] = table["frame"].astype(int)
    table["mouse_a_id"] = table["mouse_a_id"].astype(int)
    table["mouse_b_id"] = table["mouse_b_id"].astype(int)
    # Canonical pair keys prevent the same pair being split by A/B ordering.
    table["pair_low"] = table[["mouse_a_id", "mouse_b_id"]].min(axis=1).astype(int)
    table["pair_high"] = table[["mouse_a_id", "mouse_b_id"]].max(axis=1).astype(int)
    return table


def _dominant_pair_for_event_kind(
    table: pd.DataFrame,
    event_kind: str,
    total_frames: int,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Choose the dominant close, directionally consistent pair in the supplied window."""
    # No pair evidence means the original detector result must be preserved.
    if table.empty:
        return None
    # Read conservative thresholds from one dedicated configuration section.
    cfg = dict(config.get("behavior_reconciliation", {}))
    # Chase and attack use different evidence scales and proximity limits.
    if event_kind == "chase":
        evidence_column = "selected_weak_chase_score"
        high_threshold = float(cfg.get("chase_high_score", 7.0))
        support_threshold = float(cfg.get("chase_support_score", 6.0))
        distance_limit = float(cfg.get("chase_max_support_distance_cm", 10.0))
        consistency_limit = float(cfg.get("chase_min_role_consistency", 0.65))
        minimum_count = max(
            int(cfg.get("chase_min_high_frames", 8)),
            int(round(total_frames * float(cfg.get("chase_min_high_fraction", 0.04)))),
        )
    else:
        evidence_column = "selected_weak_attack_evidence"
        high_threshold = float(cfg.get("attack_high_evidence", 4.0))
        support_threshold = float(cfg.get("attack_support_evidence", 3.0))
        distance_limit = float(cfg.get("attack_max_support_distance_cm", 8.0))
        consistency_limit = float(cfg.get("attack_min_role_consistency", 0.52))
        minimum_count = max(
            int(cfg.get("attack_min_high_frames", 5)),
            int(round(total_frames * float(cfg.get("attack_min_high_fraction", 0.025)))),
        )
    # Rank every formal pair using sustained evidence, close distance, and stable role direction.
    candidates: List[Dict[str, Any]] = []
    for (mouse_a_id, mouse_b_id), group in table.groupby(
        ["pair_low", "pair_high"],
        sort=False,
    ):
        # High-evidence rows measure persistence instead of a single noisy peak.
        high_rows = group[group[evidence_column].fillna(0.0) >= high_threshold]
        if len(high_rows) < minimum_count:
            continue
        # Supporting rows provide a robust distance estimate around the interaction.
        support_rows = group[group[evidence_column].fillna(0.0) >= support_threshold]
        support_distance = pd.to_numeric(
            support_rows["center_distance_cm"],
            errors="coerce",
        ).dropna()
        if support_distance.empty:
            continue
        median_distance = float(support_distance.median())
        if not np.isfinite(median_distance) or median_distance > distance_limit:
            continue
        # The initiating role must be substantially more stable than a coin flip.
        roles = pd.to_numeric(
            high_rows["selected_actor_id"],
            errors="coerce",
        ).dropna().astype(int)
        if roles.empty:
            continue
        role_counts = roles.value_counts()
        role_consistency = float(role_counts.iloc[0] / len(roles))
        if role_consistency < consistency_limit:
            continue
        # More sustained, more consistent, and closer interactions receive higher rank.
        rank = float(
            len(high_rows)
            * role_consistency
            / max(1.0 + median_distance, 1e-6)
        )
        candidates.append({
            "mouse_a_id": int(mouse_a_id),
            "mouse_b_id": int(mouse_b_id),
            "high_frame_count": int(len(high_rows)),
            "median_support_distance_cm": median_distance,
            "role_consistency": role_consistency,
            "rank": rank,
            "evidence_column": evidence_column,
            "high_threshold": high_threshold,
        })
    # Returning no candidate is safer than forcing a weak alternative pair.
    if not candidates:
        return None
    # Deterministic tie-breaking keeps repeated cache reprocessing reproducible.
    candidates.sort(
        key=lambda item: (
            -float(item["rank"]),
            -int(item["high_frame_count"]),
            int(item["mouse_a_id"]),
            int(item["mouse_b_id"]),
        )
    )
    return candidates[0]


def _event_window_role(
    table: pd.DataFrame,
    pair: Mapping[str, Any],
    event: Mapping[str, Any],
) -> Tuple[int, int, float]:
    """Resolve actor direction inside the event window instead of across the whole video."""
    # Limit evidence to the chosen pair and the exact event interval.
    window = table[
        (table["pair_low"] == int(pair["mouse_a_id"]))
        & (table["pair_high"] == int(pair["mouse_b_id"]))
        & (table["frame"] >= int(event["start_frame"]))
        & (table["frame"] <= int(event["end_frame"]))
    ].copy()
    # Prefer the same high-evidence rows that selected the focal pair.
    support = window[
        window[str(pair["evidence_column"])].fillna(0.0)
        >= float(pair["high_threshold"])
    ]
    # Sparse event windows fall back to every valid directional row for that pair.
    if support.empty:
        support = window
    # Count actor votes after dropping missing or unrelated IDs.
    roles = pd.to_numeric(
        support["selected_actor_id"],
        errors="coerce",
    ).dropna().astype(int)
    valid_ids = {int(pair["mouse_a_id"]), int(pair["mouse_b_id"])}
    roles = roles[roles.isin(valid_ids)]
    # The supplied local context is the last fallback when the exact event has a gap.
    if roles.empty:
        global_pair = table[
            (table["pair_low"] == int(pair["mouse_a_id"]))
            & (table["pair_high"] == int(pair["mouse_b_id"]))
        ]
        roles = pd.to_numeric(
            global_pair["selected_actor_id"],
            errors="coerce",
        ).dropna().astype(int)
        roles = roles[roles.isin(valid_ids)]
    # Canonical order is only used if no directional evidence exists at all.
    if roles.empty:
        actor_id = int(pair["mouse_a_id"])
        consistency = 0.0
    else:
        counts = roles.value_counts()
        actor_id = int(counts.index[0])
        consistency = float(counts.iloc[0] / len(roles))
    # The other member of the pair is necessarily the target.
    target_id = (
        int(pair["mouse_b_id"])
        if actor_id == int(pair["mouse_a_id"])
        else int(pair["mouse_a_id"])
    )
    return actor_id, target_id, consistency


def _merge_reconciled_events(
    events: List[Dict[str, Any]],
    max_gap_frames: int,
    attack_max_gap_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Merge same-pair fragments, allowing attacks a longer detector-dropout gap."""
    # Preserve the former single-gap API for callers that do not request attack bridging.
    attack_gap_frames = (
        max_gap_frames
        if attack_max_gap_frames is None
        else max(0, int(attack_max_gap_frames))
    )
    # Sort by class, pair, and time so adjacent duplicates become consecutive.
    ordered = sorted(
        events,
        key=lambda event: (
            int(event.get("label_id", 0)),
            str(event.get("pair_key", "")),
            int(event.get("start_frame", 0)),
        ),
    )
    merged: List[Dict[str, Any]] = []
    for event in ordered:
        # The first event starts a new merged sequence.
        if not merged:
            merged.append(dict(event))
            continue
        previous = merged[-1]
        # Fighting can temporarily hide one animal, so attack fragments use their own gap.
        label_id = int(event.get("label_id", 0))
        allowed_gap_frames = (
            attack_gap_frames if label_id in (2, 3) else max_gap_frames
        )
        # Only identical class, pair, and direction may be merged.
        compatible = bool(
            int(previous.get("label_id", 0)) == label_id
            and str(previous.get("pair_key", "")) == str(event.get("pair_key", ""))
            and int(previous.get("actor_id", -1)) == int(event.get("actor_id", -2))
            and int(previous.get("target_id", -1)) == int(event.get("target_id", -2))
            and int(event.get("start_frame", 0))
            <= int(previous.get("end_frame", 0)) + allowed_gap_frames + 1
        )
        if not compatible:
            merged.append(dict(event))
            continue
        # Expand the temporal boundary to cover both overlapping candidates.
        previous["start_frame"] = min(
            int(previous["start_frame"]),
            int(event["start_frame"]),
        )
        previous["end_frame"] = max(
            int(previous["end_frame"]),
            int(event["end_frame"]),
        )
        previous["start_time_s"] = min(
            float(previous["start_time_s"]),
            float(event["start_time_s"]),
        )
        previous["end_time_s"] = max(
            float(previous["end_time_s"]),
            float(event["end_time_s"]),
        )
        # Preserve the strongest evidence from all merged fragments.
        for key in (
            "max_weak_chase_score",
            "max_strong_chase_score",
            "max_weak_attack_evidence",
            "max_strong_attack_evidence",
        ):
            previous[key] = max(
                safe_float(previous.get(key), 0.0),
                safe_float(event.get(key), 0.0),
            )
        # Record how many raw fragments were consolidated for auditing.
        previous["reconciled_fragment_count"] = int(
            previous.get("reconciled_fragment_count", 1)
        ) + int(event.get("reconciled_fragment_count", 1))
    return merged


def reconcile_detected_event_pairs(
    events: List[Dict[str, Any]],
    pair_path: Path,
    total_frames: int,
    fps: float,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Replace nearest-pair mistakes using event-level evidence consensus."""
    # This stage only refines already detected behavior classes; it does not infer from names.
    if not events or not bool(
        config.get("behavior_reconciliation", {}).get("enabled", True)
    ):
        return events
    # Load the normalized pair evidence once for all events in this video.
    table = _load_pair_reconciliation_table(pair_path)
    if table.empty:
        return events
    # A short context lets adjacent fragments share evidence after a detector dropout.
    reconciliation_cfg = config.get("behavior_reconciliation", {})
    # Preserve the generic context key as a backward-compatible fallback.
    default_context_seconds = float(
        reconciliation_cfg.get("pair_context_seconds", 1.00)
    )
    # Chase needs a wider context because fast ID swaps can precede the detected window.
    context_frames_by_kind = {
        "chase": max(
            0,
            int(
                round(
                    fps
                    * float(
                        reconciliation_cfg.get(
                            "chase_pair_context_seconds",
                            2.00,
                        )
                    )
                )
            ),
        ),
        # Attack uses a tighter context so other nearby contacts cannot steal the bout.
        "attack": max(
            0,
            int(
                round(
                    fps
                    * float(
                        reconciliation_cfg.get(
                            "attack_pair_context_seconds",
                            default_context_seconds,
                        )
                    )
                )
            ),
        ),
    }
    # Nearby fragments belong to one physical bout even when per-frame IDs change.
    bout_gap_frames = max(
        0,
        int(
            round(
                fps
                * float(reconciliation_cfg.get("pair_bout_gap_seconds", 0.75))
            )
        ),
    )
    # Store one pair and one direction for every raw fragment in a temporal bout.
    reconciliation_by_index: Dict[
        int,
        Tuple[Dict[str, Any], int, int, float],
    ] = {}
    # Build bouts independently for chase and attack so class semantics never mix.
    for event_kind, label_id in (("chase", 1), ("attack", 2)):
        # Use the class-specific context selected above for this entire behavior kind.
        context_frames = context_frames_by_kind[event_kind]
        # Sort only single-class events of this kind by their actual start frame.
        indexed_events = sorted(
            (
                (index, event)
                for index, event in enumerate(events)
                if int(event.get("label_id", 0)) == label_id
            ),
            key=lambda item: int(item[1].get("start_frame", 0)),
        )
        # Accumulate one connected temporal bout at a time.
        current_bout: List[Tuple[int, Dict[str, Any]]] = []
        # The active boundary grows when overlapping fragments extend the bout.
        current_end = -1
        # A sentinel iteration flushes the final accumulated bout.
        for indexed_event in indexed_events + [(None, None)]:
            # Read the next start only for a real event, not the sentinel.
            next_start = (
                int(indexed_event[1].get("start_frame", 0))
                if indexed_event[1] is not None
                else None
            )
            # Flush when the next fragment is outside the allowed dropout interval.
            should_flush = bool(
                current_bout
                and (
                    next_start is None
                    or next_start > current_end + bout_gap_frames + 1
                )
            )
            if should_flush:
                # Use the earliest and latest raw boundaries as the physical bout span.
                bout_start = min(
                    int(item[1].get("start_frame", 0)) for item in current_bout
                )
                # Include every overlapping fragment when determining the bout end.
                bout_end = max(
                    int(item[1].get("end_frame", 0)) for item in current_bout
                )
                # Clamp the contextual evidence window to valid video frames.
                context_start = max(bout_start - context_frames, 0)
                # The inclusive end retains evidence on the final bout frame.
                context_end = min(
                    bout_end + context_frames,
                    max(total_frames - 1, 0),
                )
                # Rank mouse pairs only within this bout and its short context.
                bout_table = table[
                    (table["frame"] >= context_start)
                    & (table["frame"] <= context_end)
                ].copy()
                # Fractional thresholds scale to the local evidence duration.
                local_total_frames = max(context_end - context_start + 1, 1)
                # Choose one physical pair for every fragment in the bout.
                pair = _dominant_pair_for_event_kind(
                    bout_table,
                    event_kind,
                    local_total_frames,
                    config,
                )
                if pair is not None:
                    # Resolve one shared role direction across the full bout.
                    bout_event = {
                        "start_frame": bout_start,
                        "end_frame": bout_end,
                    }
                    # Majority high-evidence roles suppress per-fragment ID flips.
                    actor_id, target_id, role_consistency = _event_window_role(
                        bout_table,
                        pair,
                        bout_event,
                    )
                    # Assign the shared reconciliation result to every raw fragment.
                    for original_index, _ in current_bout:
                        reconciliation_by_index[original_index] = (
                            pair,
                            actor_id,
                            target_id,
                            role_consistency,
                        )
                # Reset the accumulator before considering the next real event.
                current_bout = []
                # Reset the temporal boundary together with the bout members.
                current_end = -1
            # The sentinel exists only to flush and must not enter a new bout.
            if indexed_event[1] is None:
                continue
            # Add the current real fragment to the active temporal bout.
            current_bout.append(indexed_event)
            # Extend the active boundary through overlapping or adjacent fragments.
            current_end = max(
                current_end,
                int(indexed_event[1].get("end_frame", 0)),
            )
    # Rewrite only single-class events; combined chase+attack events remain untouched.
    reconciled: List[Dict[str, Any]] = []
    for event_index, original_event in enumerate(events):
        event = dict(original_event)
        label_id = int(event.get("label_id", 0))
        event_kind = "chase" if label_id == 1 else "attack" if label_id == 2 else ""
        # Combined events keep their original pair because role semantics are mixed.
        if not event_kind:
            reconciled.append(event)
            continue
        # Retrieve the shared physical pair and role direction for this bout.
        reconciliation = reconciliation_by_index.get(event_index)
        # Insufficient local evidence is safer than forcing a pair from another time.
        if reconciliation is None:
            reconciled.append(event)
            continue
        # Unpack the immutable bout-level identity decision.
        pair, actor_id, target_id, role_consistency = reconciliation
        original_pair = {
            int(event.get("mouse_a_id", -1)),
            int(event.get("mouse_b_id", -1)),
        }
        proposed_pair = {
            int(pair.get("mouse_a_id", -1)),
            int(pair.get("mouse_b_id", -1)),
        }
        # v1.40: event-level consensus may repair actor/target direction within
        # one physical pair, but it must not teleport an event to another pair
        # elsewhere in the frame.  The old behavior changed the selected false
        # clip from pair 2/3 to unrelated pair 17/18 and produced a cross-screen
        # attack arrow.  Keep the old mode only behind an explicit opt-in for
        # backwards-compatible experiments.
        allow_cross_pair = bool(
            reconciliation_cfg.get("allow_cross_pair_reassignment", True)
        )
        if proposed_pair != original_pair and not allow_cross_pair:
            event["pair_reconciled"] = False
            event["pair_reconciliation_reason"] = "cross_pair_reassignment_rejected"
            reconciled.append(event)
            continue
        # Preserve the former pair and direction for review traceability.
        event["original_pair_key"] = str(event.get("pair_key", ""))
        event["original_actor_id"] = int(event.get("actor_id", -1))
        event["original_target_id"] = int(event.get("target_id", -1))
        standard_cfg = config.get("standard_behavior_engine", {})
        standard_authoritative = bool(standard_cfg.get("enabled", False)) and str(
            standard_cfg.get("decision_mode", "standard")
        ).strip().lower() == "standard"
        preserve_standard_role = bool(
            reconciliation_cfg.get("preserve_standard_role_direction", True)
        )
        same_physical_pair = proposed_pair == original_pair
        # Pair reconciliation may repair which two mice own an event.  In the
        # authoritative standard engine, however, actor/target is a behavior
        # inference output (and -1 can intentionally mean ambiguous).  Do not
        # overwrite that result merely because the legacy evidence consensus
        # prefers one direction inside the same unordered pair.
        if standard_authoritative and preserve_standard_role and same_physical_pair:
            actor_id = int(event.get("actor_id", -1))
            target_id = int(event.get("target_id", -1))
            role_reason = "standard_behavior_role_preserved"
        else:
            role_reason = "event_level_evidence_consensus"
        # Replace pair fields consistently; role replacement obeys the rule above.
        event["mouse_a_id"] = int(pair["mouse_a_id"])
        event["mouse_b_id"] = int(pair["mouse_b_id"])
        event["pair_key"] = (
            f"{int(pair['mouse_a_id'])}_{int(pair['mouse_b_id'])}"
        )
        event["actor_id"] = actor_id
        event["target_id"] = target_id
        # Store the evidence explaining the automatic correction.
        event["pair_reconciled"] = True
        event["pair_reconciliation_reason"] = role_reason
        event["pair_reconciliation_rank"] = float(pair["rank"])
        event["pair_reconciliation_high_frames"] = int(
            pair["high_frame_count"]
        )
        event["pair_reconciliation_role_consistency"] = role_consistency
        reconciled.append(event)
    # Consolidate duplicate fragments that originally came from several wrong nearby pairs.
    # Read the reconciliation settings once so chase and attack use independent gaps.
    # Chase keeps a short gap because interrupted following is not one pursuit.
    chase_merge_gap_frames = max(
        0,
        int(
            round(
                fps
                * float(
                    reconciliation_cfg.get(
                        "chase_merge_gap_seconds",
                        reconciliation_cfg.get("merge_gap_seconds", 0.20),
                    )
                )
            )
        ),
    )
    # Attack gets a bounded longer gap for temporary disappearance during wrestling.
    attack_merge_gap_frames = max(
        chase_merge_gap_frames,
        int(
            round(
                fps
                * float(
                    reconciliation_cfg.get(
                        "attack_merge_gap_seconds",
                        0.75,
                    )
                )
            )
        ),
    )
    # Merge only after pair and direction correction, so an ID swap cannot create a new bout.
    merged = _merge_reconciled_events(
        reconciled,
        max_gap_frames=chase_merge_gap_frames,
        attack_max_gap_frames=attack_merge_gap_frames,
    )
    # Rebuild deterministic event IDs after merging.
    prefix = "SE" if any(
        str(event.get("candidate_level", "")) == "strong"
        for event in merged
    ) else "WE"
    for index, event in enumerate(
        sorted(merged, key=lambda item: int(item["start_frame"])),
        start=1,
    ):
        event["event_id"] = f"{prefix}{index:05d}"
        event["duration_s"] = (
            int(event["end_frame"]) - int(event["start_frame"]) + 1
        ) / fps
    return sorted(merged, key=lambda item: int(item["start_frame"]))


def _bbox_long_side_from_row(row: pd.Series) -> float:
    """Return a robust pixel body scale from one rendered tracking row."""
    # Width and height are clipped to one pixel to keep ratios finite.
    width = max(safe_float(row.get("bbox_x2")) - safe_float(row.get("bbox_x1")), 1.0)
    height = max(safe_float(row.get("bbox_y2")) - safe_float(row.get("bbox_y1")), 1.0)
    return float(max(width, height))


def reconcile_mount_occlusion_events(
    events: List[Dict[str, Any]],
    detection_map_path: Path,
    total_frames: int,
    fps: float,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Backfill a hidden mounted target that becomes visible only after separation."""
    # Mount backfill is a pair correction for an already detected attack, not a new class detector.
    attack_events = [
        event for event in events if int(event.get("label_id", 0)) == 2
    ]
    cfg = dict(config.get("behavior_reconciliation", {}))
    if (
        not attack_events
        or not bool(cfg.get("mount_backfill_enabled", True))
        or not detection_map_path.exists()
    ):
        return events
    # Load formal rendered tracks; provisional P/C labels are intentionally excluded.
    tracks = pd.read_csv(detection_map_path, encoding="utf-8-sig", low_memory=False)
    required = {
        "frame",
        "logical_id",
        "center_x_px",
        "center_y_px",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "render_class",
    }
    if not required.issubset(tracks.columns):
        return events
    tracks = tracks[tracks["render_class"].astype(str) == "formal_id"].copy()
    for column in (
        "frame",
        "logical_id",
        "center_x_px",
        "center_y_px",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ):
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    tracks = tracks.dropna(subset=["frame", "logical_id", "center_x_px", "center_y_px"])
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["logical_id"] = tracks["logical_id"].astype(int)
    # A hidden target must remain absent for a substantial portion of the detected attack.
    minimum_hidden_frames = max(
        int(round(fps * float(cfg.get("mount_min_hidden_seconds", 0.80)))),
        1,
    )
    # The separated pair must remain close for enough frames to reject one-frame debris.
    minimum_close_frames = max(
        int(round(fps * float(cfg.get("mount_min_post_close_seconds", 0.40)))),
        2,
    )
    post_window_frames = max(
        int(round(fps * float(cfg.get("mount_post_window_seconds", 1.00)))),
        minimum_close_frames,
    )
    maximum_distance_ratio = float(cfg.get("mount_max_center_distance_body_lengths", 0.65))
    # Evaluate late-appearing formal IDs against tracks already visible during the attack.
    candidates: List[Dict[str, Any]] = []
    first_frame_by_id = tracks.groupby("logical_id")["frame"].min().to_dict()
    for newcomer_id, first_frame in first_frame_by_id.items():
        first_frame = int(first_frame)
        # A startup ID is not evidence of release from a mount occlusion.
        if first_frame < minimum_hidden_frames:
            continue
        for attack in attack_events:
            attack_start = int(attack["start_frame"])
            attack_end = int(attack["end_frame"])
            # The newcomer must appear late enough to have been hidden during this attack.
            if first_frame - attack_start < minimum_hidden_frames:
                continue
            # A release well outside the detected attack cannot explain that event.
            if first_frame > attack_end + int(round(0.25 * fps)):
                continue
            newcomer_rows = tracks[
                (tracks["logical_id"] == int(newcomer_id))
                & (tracks["frame"] >= first_frame)
                & (tracks["frame"] <= first_frame + post_window_frames)
            ]
            if newcomer_rows.empty:
                continue
            # Existing tracks are possible visible riders/attackers.
            anchor_ids = [
                int(logical_id)
                for logical_id, anchor_first in first_frame_by_id.items()
                if int(logical_id) != int(newcomer_id)
                and int(anchor_first) <= attack_start
            ]
            for anchor_id in anchor_ids:
                anchor_rows = tracks[
                    (tracks["logical_id"] == anchor_id)
                    & (tracks["frame"] >= first_frame)
                    & (tracks["frame"] <= first_frame + post_window_frames)
                ]
                joined = anchor_rows.merge(
                    newcomer_rows,
                    on="frame",
                    suffixes=("_anchor", "_new"),
                )
                if len(joined) < minimum_close_frames:
                    continue
                # Measure separation relative to the visible bodies rather than fixed pixels.
                close_count = 0
                ratios: List[float] = []
                for _, row in joined.iterrows():
                    distance = float(
                        np.hypot(
                            safe_float(row["center_x_px_anchor"])
                            - safe_float(row["center_x_px_new"]),
                            safe_float(row["center_y_px_anchor"])
                            - safe_float(row["center_y_px_new"]),
                        )
                    )
                    anchor_scale = _bbox_long_side_from_row(
                        pd.Series({
                            "bbox_x1": row["bbox_x1_anchor"],
                            "bbox_y1": row["bbox_y1_anchor"],
                            "bbox_x2": row["bbox_x2_anchor"],
                            "bbox_y2": row["bbox_y2_anchor"],
                        })
                    )
                    newcomer_scale = _bbox_long_side_from_row(
                        pd.Series({
                            "bbox_x1": row["bbox_x1_new"],
                            "bbox_y1": row["bbox_y1_new"],
                            "bbox_x2": row["bbox_x2_new"],
                            "bbox_y2": row["bbox_y2_new"],
                        })
                    )
                    ratio = distance / max(anchor_scale, newcomer_scale, 1.0)
                    ratios.append(ratio)
                    close_count += int(ratio <= maximum_distance_ratio)
                if close_count < minimum_close_frames:
                    continue
                # Prefer longer post-release co-visibility and tighter separation.
                median_ratio = float(np.median(ratios))
                score = float(close_count / max(median_ratio, 0.05))
                candidates.append({
                    "score": score,
                    "anchor_id": anchor_id,
                    "newcomer_id": int(newcomer_id),
                    "first_frame": first_frame,
                    "attack": attack,
                    "close_frame_count": close_count,
                    "median_distance_ratio": median_ratio,
                })
    # Preserve the event detector result when no strong release pattern exists.
    if not candidates:
        return events
    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["first_frame"]),
            int(item["anchor_id"]),
            int(item["newcomer_id"]),
        )
    )
    selected = candidates[0]
    source_event = dict(selected["attack"])
    # The target first becomes independently visible on the frame after the hidden bout.
    source_event["end_frame"] = max(
        int(source_event["start_frame"]),
        int(selected["first_frame"]) - 1,
    )
    source_event["end_time_s"] = source_event["end_frame"] / fps
    source_event["duration_s"] = (
        int(source_event["end_frame"]) - int(source_event["start_frame"]) + 1
    ) / fps
    # The continuously visible mounted mouse is the initiator; the released newcomer is target.
    source_event["original_pair_key"] = str(source_event.get("pair_key", ""))
    source_event["original_actor_id"] = int(source_event.get("actor_id", -1))
    source_event["original_target_id"] = int(source_event.get("target_id", -1))
    source_event["mouse_a_id"] = min(
        int(selected["anchor_id"]),
        int(selected["newcomer_id"]),
    )
    source_event["mouse_b_id"] = max(
        int(selected["anchor_id"]),
        int(selected["newcomer_id"]),
    )
    source_event["pair_key"] = (
        f"{source_event['mouse_a_id']}_{source_event['mouse_b_id']}"
    )
    source_event["actor_id"] = int(selected["anchor_id"])
    source_event["target_id"] = int(selected["newcomer_id"])
    source_event["pair_reconciled"] = True
    source_event["pair_reconciliation_reason"] = (
        "mount_hidden_target_post_separation_backfill"
    )
    source_event["mount_release_frame"] = int(selected["first_frame"])
    source_event["mount_post_close_frames"] = int(selected["close_frame_count"])
    source_event["mount_median_distance_body_lengths"] = float(
        selected["median_distance_ratio"]
    )
    # Replace only attack events; chase and negative events are left untouched.
    output = [
        dict(event)
        for event in events
        if int(event.get("label_id", 0)) != 2
    ]
    output.append(source_event)
    output.sort(key=lambda event: int(event["start_frame"]))
    prefix = "SE" if str(source_event.get("candidate_level", "")) == "strong" else "WE"
    for index, event in enumerate(output, start=1):
        event["event_id"] = f"{prefix}{index:05d}"
    return output


def rebuild_frame_event_maps(
    events: Sequence[Mapping[str, Any]],
) -> Tuple[DefaultDict[int, int], Dict[int, Dict[str, Any]]]:
    """Rebuild rendered per-frame labels exclusively from reconciled events."""
    # Start from empty maps so discarded wrong-pair frames cannot leak into the video.
    active_count: DefaultDict[int, int] = defaultdict(int)
    frame_top: Dict[int, Dict[str, Any]] = {}
    # Every surviving event paints its full inclusive time interval.
    for event in events:
        label_id = int(event.get("label_id", 0))
        if label_id == 0:
            continue
        for frame in range(int(event["start_frame"]), int(event["end_frame"]) + 1):
            active_count[frame] += 1
            # Attack has higher display priority than chase when intervals overlap.
            priority = label_id * 100
            previous = frame_top.get(frame)
            if previous is not None and int(previous.get("_priority", 0)) >= priority:
                continue
            frame_top[frame] = {
                "_priority": priority,
                "weak_final_label_id": label_id,
                "selected_actor_id": int(event.get("actor_id", -1)),
                "selected_target_id": int(event.get("target_id", -1)),
                "pair_key": str(event.get("pair_key", "")),
                "selected_weak_chase_score": safe_float(
                    event.get("max_weak_chase_score"),
                    0.0,
                ),
                "selected_weak_attack_evidence": safe_float(
                    event.get("max_weak_attack_evidence"),
                    0.0,
                ),
                "center_distance_cm": safe_float(
                    event.get("min_center_distance_cm"),
                    float("nan"),
                ),
            }
    return active_count, frame_top


def reprocess_behavior_from_cache(
    output_dir: Path,
    config: Mapping[str, Any],
) -> Path:
    """利用既有逐帧鼠对特征重新生成事件、四分类和行为标签视频。"""
    # 规范化目录，确保后续所有文件操作都针对一个明确的视频结果目录。
    output_dir = Path(output_dir).resolve()
    # 元数据提供原视频、帧率和画幅，不依赖视频或目录名称推断行为类别。
    metadata_path = output_dir / "运行元数据.json"
    # 鼠对CSV保留检测阶段已经计算好的几何、速度和姿态质量特征。
    pair_path = output_dir / "成对行为标签.csv"
    # 逐帧检测流用于重建每帧最高优先级行为摘要。
    frame_detection_path = output_dir / "逐帧检测流缓存.csv"
    # 纯追踪视频作为行为文字和箭头的无损身份底图。
    tracking_video_path = output_dir / "追踪标注视频_仅框与骨架.mp4"
    # 身份框坐标表用于把行为角色文字贴到对应小鼠旁边。
    detection_map_path = output_dir / "检测轨迹对应表.csv"
    # 缺少任一核心缓存都不能安全复算，直接给出具体目录错误。
    required_paths = (
        metadata_path,
        pair_path,
        frame_detection_path,
        tracking_video_path,
        detection_map_path,
    )
    # 逐个验证路径，避免处理到一半才留下不完整覆盖文件。
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"行为缓存文件不存在：{required_path}")
    # 先把完整鼠对CSV载入内存，因为后处理会原位重写同名文件。
    store = PairDataFrameStore(pair_path)
    # 读取正常推理保存的权威视频参数。
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    # 帧率必须为正，时间阈值才能换算为帧数。
    fps = float(metadata["fps"])
    # 总帧数用于事件片段边界和逐帧摘要完整性检查。
    total_frames = int(metadata["total_frames"])
    # 行为标签视频必须沿用原追踪视频宽度。
    width = int(metadata["width"])
    # 行为标签视频必须沿用原追踪视频高度。
    height = int(metadata["height"])
    # 原视频路径只用于写回视频名称，不用于规则或标签判定。
    source_video_path = Path(str(metadata["video"]))
    # 复用与正常推理完全相同的逐鼠对时序后处理。
    weak_events, strong_events, negative_events, frame_active_count, frame_top = (
        postprocess_pair_store(store, fps, config, output_dir, total_frames)
    )
    # Correct nearest-pair mistakes with whole-event evidence before any CSV or video is written.
    weak_events = reconcile_detected_event_pairs(
        weak_events,
        pair_path,
        total_frames,
        fps,
        config,
    )
    # Strong candidates use the same focal-pair logic but remain independent of weak thresholds.
    strong_events = reconcile_detected_event_pairs(
        strong_events,
        pair_path,
        total_frames,
        fps,
        config,
    )
    # Recover a mounted target whose formal ID becomes visible only after the attack separates.
    weak_events = reconcile_mount_occlusion_events(
        weak_events,
        detection_map_path,
        total_frames,
        fps,
        config,
    )
    # Apply mount backfill to strong events only when a strong attack was already detected.
    strong_events = reconcile_mount_occlusion_events(
        strong_events,
        detection_map_path,
        total_frames,
        fps,
        config,
    )
    # Rebuild rendered frame maps so old wrong-pair labels cannot remain after reconciliation.
    frame_active_count, frame_top = rebuild_frame_event_maps(weak_events)
    # 重新计算事件清洁度，保持缓存复算和正常推理输出字段一致。
    mark_event_cleanliness(weak_events, config)
    # 强候选同样需要清洁度标志供后续人工复核使用。
    mark_event_cleanliness(strong_events, config)
    # 为弱候选生成固定时长片段边界，但不重复复制视频片段。
    add_clip_boundaries(weak_events, total_frames, fps, config)
    # 强候选使用同一边界策略。
    add_clip_boundaries(strong_events, total_frames, fps, config)
    # 困难负样本也保留可复核时间范围。
    add_clip_boundaries(negative_events, total_frames, fps, config)
    # 按配置去除弱候选与困难负样本之间过密的片段起点。
    enforce_clip_spacing(weak_events + negative_events, fps, config)
    # 强候选单独执行间隔限制，避免被弱候选影响。
    enforce_clip_spacing(strong_events, fps, config)
    # 载入检测阶段逐帧统计，保留所有身份和检测质量列。
    frame_detection_df = pd.read_csv(
        frame_detection_path, encoding="utf-8-sig"
    )
    # 基于检测统计副本附加行为列，不修改原始检测缓存。
    frame_summary_df = frame_detection_df.copy()
    # 写入每帧活跃行为鼠对数量。
    frame_summary_df["active_pair_count"] = (
        frame_summary_df["frame"].map(frame_active_count).fillna(0).astype(int)
    )
    # 写入每帧最高优先级四分类标签。
    frame_summary_df["frame_label_id"] = frame_summary_df["frame"].map(
        lambda frame: int(
            frame_top.get(int(frame), {}).get("weak_final_label_id", 0)
        )
    )
    # 写入渲染使用的施动者ID。
    frame_summary_df["selected_actor_id"] = frame_summary_df["frame"].map(
        lambda frame: frame_top.get(int(frame), {}).get(
            "selected_actor_id", np.nan
        )
    )
    # 写入渲染使用的受动者ID。
    frame_summary_df["selected_target_id"] = frame_summary_df["frame"].map(
        lambda frame: frame_top.get(int(frame), {}).get(
            "selected_target_id", np.nan
        )
    )
    # 写入当前被选中的稳定鼠对键。
    frame_summary_df["selected_pair_key"] = frame_summary_df["frame"].map(
        lambda frame: frame_top.get(int(frame), {}).get("pair_key", "")
    )
    # 写入追逐分数供人工复核。
    frame_summary_df["selected_chase_score"] = frame_summary_df["frame"].map(
        lambda frame: frame_top.get(int(frame), {}).get(
            "selected_weak_chase_score", 0
        )
    )
    # 写入攻击动态证据计数供人工复核。
    frame_summary_df["selected_attack_evidence"] = frame_summary_df[
        "frame"
    ].map(
        lambda frame: frame_top.get(int(frame), {}).get(
            "selected_weak_attack_evidence", 0
        )
    )
    # 写入被选鼠对的中心距离。
    frame_summary_df["center_distance_cm"] = frame_summary_df["frame"].map(
        lambda frame: frame_top.get(int(frame), {}).get(
            "center_distance_cm", np.nan
        )
    )
    # 覆盖逐帧行为汇总，使其与当前规则完全一致。
    frame_summary_df.to_csv(
        output_dir / "逐帧行为汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 覆盖弱候选事件表。
    pd.DataFrame(weak_events).to_csv(
        output_dir / "行为事件_弱候选.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 覆盖强候选事件表。
    pd.DataFrame(strong_events).to_csv(
        output_dir / "行为事件_强候选.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 待复核表继续包含弱候选和困难负样本。
    pd.DataFrame(weak_events + negative_events).to_csv(
        output_dir / "行为事件_待复核.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 只有显式打开行为标签视频时才二次渲染；默认只保留纯追踪框与骨架视频。
    if bool(config.get("output", {}).get("save_behavior_label_video", False)):
        # 行为视频先写临时文件，成功关闭后再替换旧结果，防止中断留下损坏MP4。
        temporary_behavior_video = output_dir / (
            f".追踪与行为标签视频.{uuid.uuid4().hex}.写入中.mp4"
        )
        # 用纯追踪视频、稳定ID框和新逐帧标签生成最终可视化。
        save_behavior_label_video(
            tracking_video_path,
            temporary_behavior_video,
            frame_summary_df,
            detection_map_path,
            fps,
            width,
            height,
        )
        # 原子替换最终行为视频，旧文件在新文件完成前始终可用。
        temporary_behavior_video.replace(output_dir / "追踪与行为标签视频.mp4")
    else:
        # 兼容旧结果目录，关闭该输出后清掉旧版本留下的同名视频。
        (output_dir / "追踪与行为标签视频.mp4").unlink(missing_ok=True)
    # 基于重新生成的弱候选事件计算视频级四分类。
    weak_video_class = classify_video_four_label(
        weak_events, "weak", config
    )
    # 基于重新生成的强候选事件计算视频级四分类。
    strong_video_class = classify_video_four_label(
        strong_events, "strong", config
    )
    # 强追逐必须同时得到弱追逐支持。
    strong_video_class["chase_present"] = bool(
        strong_video_class["chase_present"]
        and weak_video_class["chase_present"]
    )
    # 强攻击必须同时得到弱攻击支持。
    strong_video_class["attack_present"] = bool(
        strong_video_class["attack_present"]
        and weak_video_class["attack_present"]
    )
    # 重新组合强候选四分类编号。
    strong_label_id = (
        int(strong_video_class["chase_present"])
        + 2 * int(strong_video_class["attack_present"])
    )
    # 写回强候选数字标签。
    strong_video_class["video_label_id"] = strong_label_id
    # 写回强候选英文标签。
    strong_video_class["video_label_en"] = LABELS[strong_label_id][0]
    # 写回强候选中文标签。
    strong_video_class["video_label_zh"] = LABELS[strong_label_id][1]
    # 强候选最终为阴性时同步清零阳性事件数。
    if strong_label_id == 0:
        strong_video_class["positive_event_count"] = 0
    # 生成与正常推理格式一致的视频级结果表。
    video_class_df = pd.DataFrame(
        [weak_video_class, strong_video_class]
    )
    # 视频名仅作为输出索引，不参与前面的分类计算。
    video_class_df.insert(0, "video_name", source_video_path.name)
    # 覆盖视频四分类结果。
    video_class_df.to_csv(
        output_dir / "视频四分类结果.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 在元数据中记录缓存复算版本，便于审计结果和代码是否对应。
    metadata["behavior_reprocessed_from_cache"] = True
    # 保存当前程序版本。
    metadata["behavior_program_version"] = PROGRAM_VERSION
    # 保存弱候选视频级标签编号。
    metadata["behavior_weak_video_label_id"] = int(
        weak_video_class["video_label_id"]
    )
    # 将更新后的元数据安全写回UTF-8 JSON。
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    # 释放完整鼠对表，批量处理多个视频时及时回收内存。
    del store
    # 返回处理完成的目录，供批处理汇总记录。
    return output_dir



# -----------------------------------------------------------------------------
# v1.9 检测器优先：检测框负责“鼠是否存在”，Pose只负责ROI内关键点
# -----------------------------------------------------------------------------


def _resolve_local_resource(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _bbox_area(box: np.ndarray) -> float:
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    if box.size < 4:
        return 0.0
    return float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


def _bbox_long_side(box: np.ndarray) -> float:
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    if box.size < 4:
        return 0.0
    return float(max(abs(box[2] - box[0]), abs(box[3] - box[1])))


def _roi_white_score(crop: np.ndarray) -> float:
    """仅用于决定是否增加白鼠增强分支，不用于ID身份判定。"""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape[:2]
    y1, y2 = int(h * 0.12), max(int(h * 0.88), int(h * 0.12) + 1)
    x1, x2 = int(w * 0.12), max(int(w * 0.88), int(w * 0.12) + 1)
    core = gray[y1:y2, x1:x2]
    if core.size == 0:
        core = gray
    q25 = float(np.quantile(core, 0.25))
    q50 = float(np.quantile(core, 0.50))
    bright = float(np.mean(core > 0.72))
    return float(np.clip(0.45 * q25 + 0.35 * bright + 0.20 * q50, 0.0, 1.0))


def _white_invert_clahe(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _crop_box(frame: np.ndarray, box: np.ndarray, padding_ratio: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64).reshape(-1)[:4]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = bw * padding_ratio, bh * padding_ratio
    ix1 = int(np.clip(math.floor(x1 - px), 0, max(w - 1, 0)))
    iy1 = int(np.clip(math.floor(y1 - py), 0, max(h - 1, 0)))
    ix2 = int(np.clip(math.ceil(x2 + px), ix1 + 1, w))
    iy2 = int(np.clip(math.ceil(y2 + py), iy1 + 1, h))
    return frame[iy1:iy2, ix1:ix2].copy(), (ix1, iy1, ix2, iy2)


def _pose_candidate_score(det: base.Detection, crop_shape: Tuple[int, int]) -> float:
    conf = np.asarray(det.keypoint_conf, dtype=np.float64)
    valid = np.isfinite(conf) & (conf >= 0.08)
    pose_quality = float(np.mean(valid)) if conf.size else 0.0
    mean_conf = float(np.mean(conf[valid])) if np.any(valid) else 0.0
    h, w = crop_shape
    crop_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    diag = max(float(np.hypot(w, h)), 1.0)
    center_penalty = float(np.linalg.norm(det.center_px - crop_center) / diag)
    return float(0.52 * pose_quality + 0.28 * mean_conf + 0.20 * det.box_conf - 0.18 * center_penalty)


def _bbox_only_detection(
    box: np.ndarray,
    conf: float,
    expected_keypoints: int,
    white_score: float,
    source: str = "detector_bbox_only",
) -> base.Detection:
    return base.Detection(
        raw_track_id=None,
        keypoints_px=np.full((expected_keypoints, 2), np.nan, dtype=np.float64),
        keypoint_conf=np.zeros(expected_keypoints, dtype=np.float64),
        bbox_xyxy=np.asarray(box, dtype=np.float64).copy(),
        box_conf=float(conf),
        white_score=float(white_score),
        is_white_candidate=bool(white_score >= 0.52),
        appearance_mode="detector_bbox_only" if source == "detector_bbox_only" else source,
        appearance_reliable=False,
        detection_source=source,
    )


class DetectorCandidateFilter:
    """过滤明显碎屑假阳性，同时允许白鼠以bbox-only状态维持追踪。"""

    def __init__(self, config: Mapping[str, Any]):
        self.cfg = dict(config or {})
        self.long_history: Deque[float] = deque(maxlen=int(self.cfg.get("reference_history", 600)))
        self.area_history: Deque[float] = deque(maxlen=int(self.cfg.get("reference_history", 600)))
        self.static_cells: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # v1.24.1：无姿态white_blob必须先表现出真实位移，才允许进入身份分配。
        # 这阻止脱落标签/纸片在3帧后由Pxx晋升为正式ID；有Pose的真实小鼠
        # 不经过此门，移动中的bbox-only白鼠确认后也会继续保留。
        self.pose_free_blob_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_pose_free_blob_track_id = 1
        # v1.28：Pose模型也会给墙面水渍画出七点“伪骨架”。这类候选通常显著
        # 小于正常小鼠、置信度偏低且长期固定；单独维护运动确认轨迹，避免它们
        # 因“有七个关键点”而绕过原有的静态碎屑过滤。
        self.small_pose_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_small_pose_track_id = 1

    @staticmethod
    def _pose_quality(det: base.Detection) -> float:
        c = np.asarray(det.keypoint_conf, dtype=np.float64)
        return float(np.mean(np.isfinite(c) & (c >= 0.08))) if c.size else 0.0

    def _pose_free_blob_has_motion(
        self,
        det: base.Detection,
        frame_idx: int,
        reference_long: float,
        used_track_ids: set[int],
    ) -> bool:
        """无姿态亮斑只有形成连续、真实移动轨迹后才可进入正式检测流。"""
        if not bool(self.cfg.get("pose_free_blob_require_motion", True)):
            return True

        ttl = max(int(self.cfg.get("pose_free_blob_track_ttl_frames", 10)), 1)
        min_hits = max(int(self.cfg.get("pose_free_blob_motion_min_hits", 3)), 2)
        match_ratio = float(self.cfg.get("pose_free_blob_match_distance_reference", 0.90))
        motion_ratio = float(self.cfg.get("pose_free_blob_motion_min_reference", 0.25))
        min_motion_px = float(self.cfg.get("pose_free_blob_motion_min_px", 24.0))

        for track_id in list(self.pose_free_blob_tracks):
            state = self.pose_free_blob_tracks[track_id]
            if frame_idx - int(state["last_frame"]) > ttl:
                self.pose_free_blob_tracks.pop(track_id, None)

        center = np.asarray(det.center_px, dtype=np.float64)
        box = np.asarray(det.bbox_xyxy, dtype=np.float64)
        long_side = max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
        match_gate = max(float(reference_long) * match_ratio, long_side * 0.75, 24.0)

        candidates: List[Tuple[float, int]] = []
        for track_id, state in self.pose_free_blob_tracks.items():
            if track_id in used_track_ids:
                continue
            distance = float(np.linalg.norm(center - np.asarray(state["last_center"], dtype=np.float64)))
            old_long = max(float(state.get("long_side", long_side)), 1.0)
            size_ratio = max(long_side / old_long, old_long / long_side)
            if distance <= match_gate and size_ratio <= 1.8:
                candidates.append((distance, int(track_id)))

        if candidates:
            _, track_id = min(candidates)
            state = self.pose_free_blob_tracks[track_id]
            state["last_center"] = center.copy()
            state["last_frame"] = int(frame_idx)
            state["long_side"] = 0.8 * float(state.get("long_side", long_side)) + 0.2 * long_side
            state["hits"] = int(state.get("hits", 1)) + 1
            displacement = float(
                np.linalg.norm(center - np.asarray(state["first_center"], dtype=np.float64))
            )
            state["max_displacement"] = max(float(state.get("max_displacement", 0.0)), displacement)
        else:
            track_id = int(self.next_pose_free_blob_track_id)
            self.next_pose_free_blob_track_id += 1
            self.pose_free_blob_tracks[track_id] = {
                "first_center": center.copy(),
                "last_center": center.copy(),
                "last_frame": int(frame_idx),
                "long_side": long_side,
                "hits": 1,
                "max_displacement": 0.0,
                "confirmed": False,
            }
            state = self.pose_free_blob_tracks[track_id]

        used_track_ids.add(track_id)
        motion_gate = max(float(reference_long) * motion_ratio, min_motion_px)
        if int(state["hits"]) >= min_hits and float(state["max_displacement"]) >= motion_gate:
            state["confirmed"] = True
        return bool(state.get("confirmed", False))

    def _small_pose_has_motion(
        self,
        det: base.Detection,
        frame_idx: int,
        reference_long: float,
        used_track_ids: set[int],
    ) -> bool:
        """小型低置信Pose候选必须先表现出位移，才能进入身份分配。"""
        ttl = max(int(self.cfg.get("small_pose_track_ttl_frames", 12)), 1)
        min_hits = max(int(self.cfg.get("small_pose_motion_min_hits", 3)), 2)
        match_ratio = float(self.cfg.get("small_pose_match_distance_reference", 0.55))
        motion_ratio = float(self.cfg.get("small_pose_motion_min_reference", 0.12))
        min_motion_px = float(self.cfg.get("small_pose_motion_min_px", 12.0))

        for track_id in list(self.small_pose_tracks):
            state = self.small_pose_tracks[track_id]
            if frame_idx - int(state["last_frame"]) > ttl:
                self.small_pose_tracks.pop(track_id, None)

        center = np.asarray(det.center_px, dtype=np.float64)
        box = np.asarray(det.bbox_xyxy, dtype=np.float64)
        long_side = max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
        match_gate = max(float(reference_long) * match_ratio, long_side * 0.65, 18.0)
        candidates: List[Tuple[float, int]] = []
        for track_id, state in self.small_pose_tracks.items():
            if track_id in used_track_ids:
                continue
            distance = float(np.linalg.norm(center - np.asarray(state["last_center"], dtype=np.float64)))
            old_long = max(float(state.get("long_side", long_side)), 1.0)
            size_ratio = max(long_side / old_long, old_long / long_side)
            if distance <= match_gate and size_ratio <= 1.65:
                candidates.append((distance, int(track_id)))

        if candidates:
            _, track_id = min(candidates)
            state = self.small_pose_tracks[track_id]
            state["last_center"] = center.copy()
            state["last_frame"] = int(frame_idx)
            state["long_side"] = 0.8 * float(state.get("long_side", long_side)) + 0.2 * long_side
            state["hits"] = int(state.get("hits", 1)) + 1
            displacement = float(
                np.linalg.norm(center - np.asarray(state["first_center"], dtype=np.float64))
            )
            state["max_displacement"] = max(float(state.get("max_displacement", 0.0)), displacement)
        else:
            track_id = int(self.next_small_pose_track_id)
            self.next_small_pose_track_id += 1
            self.small_pose_tracks[track_id] = {
                "first_center": center.copy(),
                "last_center": center.copy(),
                "last_frame": int(frame_idx),
                "long_side": long_side,
                "hits": 1,
                "max_displacement": 0.0,
                "confirmed": False,
            }
            state = self.small_pose_tracks[track_id]

        used_track_ids.add(track_id)
        motion_gate = max(float(reference_long) * motion_ratio, min_motion_px)
        if int(state["hits"]) >= min_hits and float(state["max_displacement"]) >= motion_gate:
            state["confirmed"] = True
        return bool(state.get("confirmed", False))

    def filter(self, detections: Sequence[base.Detection], frame_idx: int, frame_shape: Tuple[int, int]) -> List[base.Detection]:
        if not detections:
            return []
        h, w = frame_shape
        frame_area = max(float(h * w), 1.0)
        min_area_abs = frame_area * float(self.cfg.get("min_box_area_ratio", 0.00045))
        max_area_abs = frame_area * float(self.cfg.get("max_box_area_ratio", 0.10))

        current_longs = np.array([_bbox_long_side(d.bbox_xyxy) for d in detections], dtype=np.float64)
        current_areas = np.array([_bbox_area(d.bbox_xyxy) for d in detections], dtype=np.float64)
        reliable_mask = np.array([
            (d.box_conf >= float(self.cfg.get("reference_min_conf", 0.24)))
            or (self._pose_quality(d) >= float(self.cfg.get("reference_min_pose_quality", 0.28)))
            or bool(d.is_white_candidate)
            for d in detections
        ])
        if len(self.long_history) >= 20:
            ref_long = float(np.median(np.asarray(self.long_history, dtype=np.float64)))
            ref_area = float(np.median(np.asarray(self.area_history, dtype=np.float64)))
        else:
            sample_l = current_longs[reliable_mask]
            sample_a = current_areas[reliable_mask]
            ref_long = float(np.median(sample_l)) if sample_l.size else float(np.median(current_longs))
            ref_area = float(np.median(sample_a)) if sample_a.size else float(np.median(current_areas))

        min_long_ratio = float(self.cfg.get("min_long_side_ratio", 0.38))
        max_long_ratio = float(self.cfg.get("max_long_side_ratio", 2.8))
        min_area_ratio = float(self.cfg.get("min_area_ratio_to_reference", 0.16))
        max_area_ratio = float(self.cfg.get("max_area_ratio_to_reference", 5.5))
        bbox_only_min_conf = float(self.cfg.get("bbox_only_min_conf", 0.16))
        static_cell_px = max(int(self.cfg.get("static_cell_px", 28)), 8)
        static_reject_frames = max(int(self.cfg.get("static_reject_frames", 45)), 1)
        static_max_conf = float(self.cfg.get("static_max_conf", 0.30))
        # v1.11.3：墙上水痕等静止目标也可能被Pose误画骨架（带弱姿态）。
        # 带弱姿态但长期固定不动的候选用更长窗口拒绝（真鼠哪怕伏地，
        # 全图Pose给出的姿态通常完整 pose_q≥0.5，不受该层影响）。
        static_reject_frames_pose = max(int(self.cfg.get("static_reject_frames_with_pose", 150)), 1)
        static_pose_q_max2 = float(self.cfg.get("static_pose_quality_max_with_pose", 0.50))

        accepted: List[base.Detection] = []
        used_pose_free_blob_track_ids: set[int] = set()
        used_small_pose_track_ids: set[int] = set()
        for det, long_side, area in zip(detections, current_longs, current_areas):
            if area < min_area_abs or area > max_area_abs or long_side <= 2:
                continue
            box = np.asarray(det.bbox_xyxy, dtype=np.float64)
            bw, bh = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
            aspect = max(bw / bh, bh / bw)
            if aspect > float(self.cfg.get("max_aspect_ratio", 5.5)):
                continue
            if ref_long > 1 and (long_side < ref_long * min_long_ratio or long_side > ref_long * max_long_ratio):
                continue
            if ref_area > 1 and (area < ref_area * min_area_ratio or area > ref_area * max_area_ratio):
                continue

            pose_q = self._pose_quality(det)
            if pose_q <= 0 and det.box_conf < bbox_only_min_conf:
                continue
            if (
                pose_q <= 0
                and str(getattr(det, "detection_source", "")).startswith("white_blob")
                and not self._pose_free_blob_has_motion(
                    det, frame_idx, ref_long, used_pose_free_blob_track_ids
                )
            ):
                continue

            # 七点齐全不等于真实小鼠。墙上水渍在本数据中约为正常鼠面积的
            # 20%~30%，框近似正方形且置信度低，但Pose会稳定输出七个点。
            # 仅对“同时小、弱、非白鼠”的候选加运动门，不影响正常尺寸小鼠，
            # 也不影响接触区中尺寸仍正常的低置信真实小鼠。
            small_pose = bool(
                pose_q > 0
                and ref_long > 1
                and ref_area > 1
                and long_side < ref_long * float(self.cfg.get("small_pose_max_long_ratio", 0.72))
                and area < ref_area * float(self.cfg.get("small_pose_max_area_ratio", 0.42))
                and float(det.box_conf) < float(self.cfg.get("small_pose_max_conf", 0.45))
                and not bool(det.is_white_candidate)
            )
            if (
                small_pose
                and bool(self.cfg.get("small_pose_require_motion", True))
                and not self._small_pose_has_motion(
                    det, frame_idx, ref_long, used_small_pose_track_ids
                )
            ):
                continue

            # 无姿态且长期固定在同一小格的目标，通常是地面碎屑/水印。
            # v1.11.2：置信度不再豁免——水印等误检可能以较高置信度反复出现；
            # 真鼠（含伏地不动的）经全图Pose匹配有姿态证据，不受影响。
            center = det.center_px
            cell = (int(center[0] // static_cell_px), int(center[1] // static_cell_px))
            last_frame, count = self.static_cells.get(cell, (-999999, 0))
            count = count + 1 if frame_idx - last_frame <= 2 else 1
            self.static_cells[cell] = (frame_idx, count)
            if not bool(det.is_white_candidate) and (
                (
                    pose_q < float(self.cfg.get("static_pose_quality_max", 0.12))
                    and count >= static_reject_frames
                )
                or (
                    pose_q < static_pose_q_max2
                    and count >= static_reject_frames_pose
                )
            ):
                continue

            accepted.append(det)
            if (
                pose_q >= float(self.cfg.get("reference_min_pose_quality", 0.28))
                or det.box_conf >= float(self.cfg.get("reference_min_conf", 0.24))
                or bool(det.is_white_candidate)
            ):
                self.long_history.append(float(long_side))
                self.area_history.append(float(area))
        return accepted


def _resume_model_track_stream(
    video_path: Path,
    model: Any,
    track_kwargs: Mapping[str, Any],
    start_frame: int,
) -> Iterable[Any]:
    """非detector-first模式恢复时从指定帧逐帧调用持久化YOLO跟踪器。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    requested_start = max(int(start_frame), 0)
    if requested_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(requested_start))
        positioned = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if positioned > requested_start:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            positioned = 0
        while positioned < requested_start:
            if not cap.grab():
                cap.release()
                raise RuntimeError(f"无法定位到恢复帧：{requested_start}")
            positioned += 1
    kwargs = dict(track_kwargs)
    kwargs.pop("source", None)
    kwargs.pop("stream", None)
    kwargs.pop("stream_buffer", None)
    kwargs.pop("vid_stride", None)
    kwargs["persist"] = True
    kwargs["stream"] = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.track(source=frame, **kwargs)
            if results:
                yield results[0]
    finally:
        cap.release()


def _detector_first_stream(
    video_path: Path,
    detector_model: Any,
    pose_model: Any,
    config: Mapping[str, Any],
    device: Any,
    expected_keypoints: int,
    start_frame: int = 0,
    candidate_filter: Optional[DetectorCandidateFilter] = None,
    precomputed_records: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Iterable[Any]:
    """逐帧Pose主通道融合（v1.12.0，按文档§5重构）：

    1. Pose模型全图推理输出所有检测（框+关键点）——文档第1步，唯一权威检测流；
    2. 普通检测器框与白鼠亮斑仅作补缺：Pose已覆盖的鼠不重复出框，
       Pose超大框（覆盖多只紧挨的鼠）内的独立个体（如白鼠）由补充框拆出；
    3. 场地掩码/镜像对/静态碎屑防线 → 记忆身份分配 → 全量渲染。

    pose_mode='roi' 时回退为旧的逐ROI姿态推理（适配ROI训练的模型）。
    """
    cfg = dict(config or {})
    pose_mode = str(cfg.get("pose_mode", "full_frame")).lower()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    requested_start = max(int(start_frame), 0)
    if requested_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(requested_start))
        positioned = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if positioned > requested_start:
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            positioned = 0
        while positioned < requested_start:
            if not cap.grab():
                cap.release()
                raise RuntimeError(f"无法定位到恢复帧：{requested_start}")
            positioned += 1
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != requested_start:
            cap.release()
            raise RuntimeError(f"视频恢复定位不精确：要求{requested_start}，实际{positioned}")

    sanity = candidate_filter or DetectorCandidateFilter(cfg.get("candidate_filter", {}))
    arena_mask = base.ArenaFloorMask(cfg.get("arena_mask", {}))
    reflection_cfg = dict(cfg.get("reflection_pair", {}))
    reflection_enabled = bool(reflection_cfg.get("enabled", True))
    detector_half = bool(cfg.get("half", True)) and str(device).lower() != "cpu"
    pose_half = bool(cfg.get("pose_half", True)) and str(device).lower() != "cpu"
    padding = float(cfg.get("roi_padding_ratio", 0.18))
    white_threshold = float(cfg.get("white_threshold", 0.50))
    use_white_invert = bool(cfg.get("use_white_invert", True))
    allow_bbox_only = bool(cfg.get("accept_bbox_without_pose", True))
    blob_cfg = dict(cfg.get("white_blob", {}))
    blob_enabled = bool(blob_cfg.get("enabled", True))
    frame_idx = requested_start
    precomputed_iterator = iter(precomputed_records) if precomputed_records is not None else None

    try:
        while True:
            frame_cycle_started = time.perf_counter()
            decode_started = time.perf_counter()
            ok, frame = cap.read()
            decode_seconds = time.perf_counter() - decode_started
            if not ok:
                break
            precomputed_record: Optional[Mapping[str, Any]] = None
            if precomputed_iterator is not None:
                try:
                    precomputed_record = next(precomputed_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"YOLO预推理缓存在第{frame_idx}帧提前结束。"
                    ) from exc
                cached_frame = int(precomputed_record.get("frame", -1))
                if cached_frame != frame_idx:
                    raise RuntimeError(
                        f"YOLO预推理缓存帧号不连续：当前{frame_idx}，缓存{cached_frame}。"
                    )
            boxes: List[Tuple[np.ndarray, float]] = []
            full_frame_inference_seconds = 0.0
            result_transfer_seconds = 0.0
            result_parse_seconds = 0.0
            candidate_filter_seconds = 0.0
            if precomputed_record is not None:
                boxes = [
                    (np.asarray(box, dtype=np.float64).copy(), float(conf))
                    for box, conf in precomputed_record.get("detector_boxes", [])
                ]
            elif detector_model is not None:
                detector_kwargs: Dict[str, Any] = {
                    "source": frame,
                    "imgsz": int(cfg.get("detector_imgsz", 1280)),
                    "conf": float(cfg.get("detector_conf", 0.14)),
                    "iou": float(cfg.get("detector_iou", 0.50)),
                    "max_det": int(cfg.get("detector_max_det", 40)),
                    "device": device,
                    "verbose": False,
                }
                if detector_half:
                    detector_kwargs["half"] = True
                detector_started = time.perf_counter()
                detector_results = detector_model.predict(**detector_kwargs)
                full_frame_inference_seconds += time.perf_counter() - detector_started
                if detector_results and detector_results[0].boxes is not None and len(detector_results[0].boxes):
                    transfer_started = time.perf_counter()
                    b = detector_results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float64)
                    c = detector_results[0].boxes.conf.detach().cpu().numpy().astype(np.float64)
                    result_transfer_seconds += time.perf_counter() - transfer_started
                    detector_parse_started = time.perf_counter()
                    order = np.argsort(-c)[: int(cfg.get("detector_max_det", 40))]
                    boxes = [(b[i], float(c[i])) for i in order]
                    result_parse_seconds += time.perf_counter() - detector_parse_started
            # use_detector_model: false 时 boxes 恒空——Pose主通道+白鼠亮斑照常工作，
            # 检测器补充框只是可选补缺来源，不是必需品。

            # ================= Pose主通道模式（默认，v1.12.0按文档§5重构） =================
            # 文档流程：第1步 YOLO-Pose输出当前帧所有检测（框+关键点）→
            # 第2步必要过滤 → 后续匹配/渲染。普通检测器框与白鼠亮斑只作"补缺补充框"：
            # Pose已覆盖的鼠不重复出框；Pose漏掉的鼠（含超大框里的白鼠）由补充框补上。
            if pose_mode == "full_frame":
                pose_dets: List[base.Detection] = []
                if precomputed_record is not None:
                    pose_parse_started = time.perf_counter()
                    pose_dets = [
                        _deserialize_yolo_detection(payload, expected_keypoints)
                        for payload in precomputed_record.get("pose_detections", [])
                    ]
                    result_parse_seconds += time.perf_counter() - pose_parse_started
                else:
                    pose_kwargs: Dict[str, Any] = {
                        "source": frame,
                        "imgsz": int(cfg.get("pose_full_imgsz", 960)),
                        "conf": float(cfg.get("pose_full_conf", 0.08)),
                        "iou": float(cfg.get("pose_full_iou", 0.50)),
                        "max_det": int(cfg.get("pose_full_max_det", 40)),
                        "device": device,
                        "verbose": False,
                    }
                    if pose_half:
                        pose_kwargs["half"] = True
                    pose_started = time.perf_counter()
                    pose_results = pose_model.predict(**pose_kwargs)
                    full_frame_inference_seconds += time.perf_counter() - pose_started
                    if pose_results:
                        parse_profile: Dict[str, float] = {}
                        pose_dets = base.parse_yolo_result(
                            pose_results[0], expected_keypoints,
                            int(cfg.get("pose_full_max_det", 40)),
                            profiling=parse_profile,
                        )
                        result_transfer_seconds += float(parse_profile.get("yolo_result_transfer_seconds", 0.0))
                        result_parse_seconds += float(parse_profile.get("result_parse_seconds", 0.0))
                for pdet in pose_dets:
                    pdet.raw_track_id = None
                    pdet.detection_source = "pose_full"
                    pdet.appearance_mode = "pose_full_frame"

                supplement_dets: List[base.Detection] = []
                for box, conf in boxes:
                    supplement_dets.append(_bbox_only_detection(
                        box, conf, expected_keypoints, 0.0, source="detector_box"
                    ))
                white_blob_count = 0
                # The bright-blob channel is a recall fallback, not a mandatory
                # full-frame pass.  When Pose already supplies the configured
                # population, the expensive morphology scan only redetects the
                # same white mice and wastes CPU.  It is still run immediately
                # on under-count frames, where it can actually recover a miss.
                blob_skip_pose_count = max(
                    int(blob_cfg.get("skip_when_pose_count_at_least", 0)), 0
                )
                run_blob_fallback = bool(
                    blob_enabled
                    and (
                        blob_skip_pose_count <= 0
                        or len(pose_dets) < blob_skip_pose_count
                    )
                )
                if run_blob_fallback:
                    for blob_det in base.detect_bright_blob_candidates(frame, blob_cfg, expected_keypoints):
                        supplement_dets.append(blob_det)
                        white_blob_count += 1

                # Pose主通道融合：超大框拆分、白鼠独立、叠框消除。
                merge_cfg = dict(cfg.get("cross_channel_merge", {}))
                if bool(merge_cfg.get("enabled", True)):
                    candidate_dets = base.fuse_pose_primary_detections(
                        pose_dets, supplement_dets,
                        same_mouse_center_bl=float(merge_cfg.get("same_mouse_center_body_lengths", 0.45)),
                        same_mouse_min_iou=float(merge_cfg.get("same_mouse_min_iou", 0.15)),
                        mega_size_ratio=float(merge_cfg.get("mega_box_size_ratio", 1.6)),
                        kp_same_mouse_bl=float(merge_cfg.get("kp_same_mouse_body_lengths", 0.30)),
                        kp_same_mouse_bl_unkeyed=float(merge_cfg.get("kp_same_mouse_unkeyed_body_lengths", 0.20)),
                        kp_min_conf=float(merge_cfg.get("kp_dedup_min_conf", 0.10)),
                    )
                else:
                    candidate_dets = list(pose_dets) + list(supplement_dets)

                # 文档§8：逐阶段计数（raw→融合→掩码→镜像→静态），用于定位丢鼠环节。
                stage_raw = len(pose_dets) + len(supplement_dets)
                stage_after_fusion = len(candidate_dets)

                # 场地掩码：墙外/场外候选直接拒绝；镜像对：溢进地面边缘的反光。
                candidate_dets = arena_mask.filter(candidate_dets, frame.shape[:2])
                stage_after_mask = len(candidate_dets)
                if reflection_enabled:
                    candidate_dets = base.suppress_reflection_pairs(
                        candidate_dets, arena_mask,
                        max_distance_body_lengths=float(reflection_cfg.get("max_distance_body_lengths", 1.3)),
                        boundary_band_px=float(reflection_cfg.get("boundary_band_px", 90)),
                        max_size_log_ratio=float(reflection_cfg.get("max_size_log_ratio", 0.35)),
                        max_size_log_ratio_with_contrast=float(
                            reflection_cfg.get("max_size_log_ratio_with_contrast", 1.0)
                        ),
                        frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        pose_quality_ratio=float(reflection_cfg.get("pose_quality_ratio", 0.9)),
                        contrast_margin=float(reflection_cfg.get("contrast_margin", 0.15)),
                        bright_object_factor=float(reflection_cfg.get("bright_object_factor", 1.3)),
                    )
                stage_after_reflection = len(candidate_dets)
                candidate_filter_started = time.perf_counter()
                detections = sanity.filter(candidate_dets, frame_idx, frame.shape[:2])
                candidate_filter_seconds += time.perf_counter() - candidate_filter_started
                # v1.12.7：骨架解剖学不可信的检测，跟踪/几何中心回退到框中心。
                # 模型在域外场地（裸板）上关键点系统性错位但框可靠，
                # 关键点加权中心被拉偏会污染匹配代价与行为几何。
                if bool(cfg.get("tracking_center_fallback", True)):
                    an_min_conf = float(cfg.get("tracking_center_anatomy_min_conf", 0.20))
                    an_span = float(cfg.get("tracking_center_anatomy_span_ratio", 0.45))
                    for d in detections:
                        d.prefer_bbox_center = not base.skeleton_anatomy_ok(
                            d, min_conf=an_min_conf, span_ratio=an_span,
                        )
                bbox_only_count = sum(1 for d in detections if d.pose_quality <= 0.0)
                candidate_cpu_seconds = max(
                    time.perf_counter() - frame_cycle_started
                    - decode_seconds
                    - full_frame_inference_seconds,
                    0.0,
                )
                yield SimpleNamespace(
                    orig_img=frame,
                    hybrid_detections=detections,
                    hybrid_meta={
                        "detector_candidate_count": len(candidate_dets),
                        "bbox_only_count": bbox_only_count,
                        "white_roi_count": white_blob_count,
                        # 文档§8逐阶段计数（raw→Pose/补充→融合→掩码→镜像→静态）
                        "stage_raw_count": stage_raw,
                        "stage_pose_count": len(pose_dets),
                        "stage_supplement_count": len(supplement_dets),
                        "stage_after_fusion_count": stage_after_fusion,
                        "stage_after_mask_count": stage_after_mask,
                        "stage_after_reflection_count": stage_after_reflection,
                        "stage_after_static_count": len(detections),
                        "profiling_decode_seconds": decode_seconds,
                        "profiling_full_frame_inference_seconds": full_frame_inference_seconds,
                        "profiling_candidate_cpu_seconds": candidate_cpu_seconds,
                        "profiling_candidate_filter_seconds": candidate_filter_seconds,
                        "profiling_result_transfer_seconds": result_transfer_seconds,
                        "profiling_result_parse_seconds": result_parse_seconds,
                    },
                )
                frame_idx += 1
                continue

            # ================= 逐ROI姿态模式（旧行为，适配ROI训练的模型） =================
            variants: List[np.ndarray] = []
            variant_meta: List[Tuple[int, str, Tuple[int, int, int, int], float]] = []
            proposal_meta: Dict[int, Tuple[np.ndarray, float, float]] = {}
            for proposal_idx, (box, conf) in enumerate(boxes):
                crop, crop_box = _crop_box(frame, box, padding)
                if crop.size == 0:
                    continue
                white_score = _roi_white_score(crop)
                proposal_meta[proposal_idx] = (np.asarray(box, dtype=np.float64), conf, white_score)
                variants.append(crop)
                variant_meta.append((proposal_idx, "original", crop_box, white_score))
                if use_white_invert and white_score >= white_threshold:
                    variants.append(_white_invert_clahe(crop))
                    variant_meta.append((proposal_idx, "white_invert", crop_box, white_score))

            pose_results: Sequence[Any] = []
            roi_inference_seconds = 0.0
            if variants:
                pose_kwargs = {
                    "source": variants,
                    "imgsz": int(cfg.get("pose_roi_imgsz", 384)),
                    "conf": float(cfg.get("pose_conf", 0.025)),
                    "iou": float(cfg.get("pose_iou", 0.40)),
                    "max_det": int(cfg.get("pose_max_det_per_roi", 3)),
                    "device": device,
                    "verbose": False,
                }
                if pose_half:
                    pose_kwargs["half"] = True
                pose_started = time.perf_counter()
                pose_results = pose_model.predict(**pose_kwargs)
                roi_inference_seconds = time.perf_counter() - pose_started

            best_by_proposal: Dict[int, Tuple[float, base.Detection, str]] = {}
            for result, (proposal_idx, variant_name, crop_box, white_score) in zip(pose_results, variant_meta):
                parse_profile: Dict[str, float] = {}
                local = base.parse_yolo_result(
                    result,
                    expected_keypoints,
                    int(cfg.get("pose_max_det_per_roi", 3)),
                    profiling=parse_profile,
                )
                result_transfer_seconds += float(parse_profile.get("yolo_result_transfer_seconds", 0.0))
                result_parse_seconds += float(parse_profile.get("result_parse_seconds", 0.0))
                if not local:
                    continue
                crop_h = max(crop_box[3] - crop_box[1], 1)
                crop_w = max(crop_box[2] - crop_box[0], 1)
                for det in local:
                    score = _pose_candidate_score(det, (crop_h, crop_w))
                    translated = _translate_detection(det, crop_box[0], crop_box[1])
                    if proposal_idx not in proposal_meta:
                        continue
                    detector_box, detector_conf, _ = proposal_meta[proposal_idx]
                    translated = replace(
                        translated,
                        raw_track_id=None,
                        bbox_xyxy=np.asarray(detector_box, dtype=np.float64).copy(),
                        box_conf=float(detector_conf),
                        white_score=float(white_score),
                        is_white_candidate=bool(white_score >= white_threshold),
                        appearance_mode=f"detector_roi_{variant_name}",
                        detection_source=f"detector_roi_pose_{variant_name}",
                    )
                    old = best_by_proposal.get(proposal_idx)
                    if old is None or score > old[0]:
                        best_by_proposal[proposal_idx] = (score, translated, variant_name)

            detections = []
            bbox_only_count = 0
            white_count = 0
            for proposal_idx, (box, conf, white_score) in proposal_meta.items():
                if white_score >= white_threshold:
                    white_count += 1
                best = best_by_proposal.get(proposal_idx)
                if best is not None:
                    detections.append(best[1])
                elif allow_bbox_only:
                    detections.append(_bbox_only_detection(box, conf, expected_keypoints, white_score))
                    bbox_only_count += 1

            # ROI分支同样应用场地掩码与镜像对判别。
            detections = arena_mask.filter(detections, frame.shape[:2])
            if reflection_enabled:
                detections = base.suppress_reflection_pairs(
                    detections, arena_mask,
                    max_distance_body_lengths=float(reflection_cfg.get("max_distance_body_lengths", 1.3)),
                    boundary_band_px=float(reflection_cfg.get("boundary_band_px", 90)),
                    max_size_log_ratio=float(reflection_cfg.get("max_size_log_ratio", 0.35)),
                    max_size_log_ratio_with_contrast=float(
                        reflection_cfg.get("max_size_log_ratio_with_contrast", 1.0)
                    ),
                    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    pose_quality_ratio=float(reflection_cfg.get("pose_quality_ratio", 0.9)),
                    contrast_margin=float(reflection_cfg.get("contrast_margin", 0.15)),
                    bright_object_factor=float(reflection_cfg.get("bright_object_factor", 1.3)),
                )
            candidate_filter_started = time.perf_counter()
            detections = sanity.filter(detections, frame_idx, frame.shape[:2])
            candidate_filter_seconds += time.perf_counter() - candidate_filter_started
            candidate_cpu_seconds = max(
                time.perf_counter() - frame_cycle_started
                - decode_seconds
                - full_frame_inference_seconds
                - roi_inference_seconds,
                0.0,
            )
            yield SimpleNamespace(
                orig_img=frame,
                hybrid_detections=detections,
                hybrid_meta={
                    "detector_candidate_count": len(boxes),
                    "bbox_only_count": bbox_only_count,
                    "white_roi_count": white_count,
                    "profiling_decode_seconds": decode_seconds,
                    "profiling_full_frame_inference_seconds": full_frame_inference_seconds,
                    "profiling_candidate_cpu_seconds": candidate_cpu_seconds,
                    "profiling_candidate_filter_seconds": candidate_filter_seconds,
                    "profiling_result_transfer_seconds": result_transfer_seconds,
                    "profiling_result_parse_seconds": result_parse_seconds,
                    "profiling_roi_inference_seconds": roi_inference_seconds,
                },
            )
            frame_idx += 1
    finally:
        cap.release()

def _translate_detection(
    det: base.Detection,
    offset_x: int,
    offset_y: int,
    detection_source: str = "local_recovery_cluster",
    recovery_target_logical_id: int = -1,
) -> base.Detection:
    points = np.asarray(det.keypoints_px, dtype=np.float64).copy()
    points[:, 0] += float(offset_x)
    points[:, 1] += float(offset_y)
    bbox = np.asarray(det.bbox_xyxy, dtype=np.float64).copy()
    bbox[[0, 2]] += float(offset_x)
    bbox[[1, 3]] += float(offset_y)
    return replace(
        det,
        raw_track_id=None,
        keypoints_px=points,
        bbox_xyxy=bbox,
        synthetic_recovery=False,
        detection_source=str(detection_source),
        appearance_mode=f"{detection_source}_pending",
        recovery_target_logical_id=int(recovery_target_logical_id),
    )


def _build_track_gap_recovery_regions(
    identity: Any,
    detections: Sequence[base.Detection],
    frame: int,
    frame_shape: Sequence[int],
    config: Mapping[str, Any],
    *,
    frozen_ids: Sequence[int] = (),
) -> List[Dict[str, Any]]:
    """Build targeted ROIs for confirmed tracks that have just gone missing."""
    cfg = dict(config.get("track_gap", {}))
    if not bool(cfg.get("enabled", True)):
        return []
    start_missing = max(int(cfg.get("start_after_missing_frames", 1)), 1)
    max_missing = max(int(cfg.get("max_missing_frames", 12)), start_missing)
    retry_interval = max(int(cfg.get("retry_interval_frames", 2)), 1)
    max_regions = max(int(cfg.get("max_regions_per_frame", 2)), 0)
    if max_regions <= 0:
        return []
    height, width = int(frame_shape[0]), int(frame_shape[1])
    blocked_ids = {int(value) for value in frozen_ids}
    candidates: List[Tuple[int, float, Dict[str, Any]]] = []
    tracks = dict(getattr(identity, "tracks", {}) or {})
    missing_by_id = dict(getattr(identity, "kpt_missing", {}) or {})
    for logical_id, track in tracks.items():
        lid = int(logical_id)
        missing = int(missing_by_id.get(lid, max(int(frame - track.last_frame), 0)))
        if lid in blocked_ids or missing < start_missing or missing > max_missing:
            continue
        if (missing - start_missing) % retry_interval != 0:
            continue
        predicted_box: Optional[np.ndarray] = None
        try:
            predicted_box = identity._predicted_bbox(track, int(frame))
        except Exception:
            predicted_box = getattr(track, "last_bbox_xyxy", None)
        if predicted_box is None:
            continue
        box = np.asarray(predicted_box, dtype=np.float64).reshape(-1)
        if box.size < 4 or not np.all(np.isfinite(box[:4])):
            continue
        box = box[:4].copy()
        box_width = max(float(box[2] - box[0]), 2.0)
        box_height = max(float(box[3] - box[1]), 2.0)
        expand = min(
            float(cfg.get("bbox_expand_ratio", 0.55))
            + float(cfg.get("missing_growth_ratio", 0.06)) * max(missing - start_missing, 0),
            float(cfg.get("max_expand_ratio", 1.10)),
        )
        expanded = np.asarray(
            [
                box[0] - expand * box_width,
                box[1] - expand * box_height,
                box[2] + expand * box_width,
                box[3] + expand * box_height,
            ],
            dtype=np.float64,
        )
        expanded[[0, 2]] = np.clip(expanded[[0, 2]], 0.0, max(width - 1, 0))
        expanded[[1, 3]] = np.clip(expanded[[1, 3]], 0.0, max(height - 1, 0))
        predicted_center = np.asarray(
            [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5],
            dtype=np.float64,
        )
        body = max(float(getattr(track, "body_length_px", 8.0)), 8.0)
        block_distance = float(cfg.get("block_distance_body_lengths", 0.55))
        block_iou = float(cfg.get("block_iou", 0.10))
        already_covered = False
        if bool(cfg.get("skip_when_detection_near_prediction", True)):
            for det in detections:
                det_center = np.asarray(det.center_px, dtype=np.float64)
                distance_bl = float(np.linalg.norm(det_center - predicted_center)) / body
                overlap = base.bbox_iou_xyxy(box, det.bbox_xyxy)
                if distance_bl <= block_distance or overlap >= block_iou:
                    already_covered = True
                    break
        if already_covered:
            continue
        priority_conf = float(getattr(track, "last_box_conf", 0.0))
        candidates.append(
            (
                missing,
                priority_conf,
                {
                    "bbox": expanded.tolist(),
                    "expected_count": 1,
                    "kind": "track_gap",
                    "logical_id": lid,
                    "missing_frames": missing,
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1], -item[2]["logical_id"]), reverse=True)
    return [item[2] for item in candidates[:max_regions]]


def _run_local_occlusion_recovery(
    recovery_model: Any,
    frame: np.ndarray,
    recovery_regions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: Any,
) -> List[base.Detection]:
    """在接触簇ROI内做一次低阈值高分辨率Pose推理。

    局部结果没有底层Track ID，只能作为已有逻辑ID的观测候选；
    身份分配器会禁止它们在接触簇中创建新ID。
    """
    cfg = dict(config or {})
    if recovery_model is None or not bool(cfg.get("enabled", True)):
        return []
    max_regions = max(int(cfg.get("max_regions_per_frame", 1)), 0)
    if max_regions <= 0:
        return []
    results_out: List[base.Detection] = []
    h, w = frame.shape[:2]
    # Collect all valid recovery crops before calling Ultralytics.  The previous
    # implementation called predict once per region, which serialized many
    # small GPU launches and left the accelerator idle between calls.
    roi_jobs: List[Tuple[np.ndarray, int, int, int, str, int]] = []
    for region in list(recovery_regions)[:max_regions]:
        box = np.asarray(region.get("bbox", []), dtype=np.float64).reshape(-1)
        if box.size < 4:
            continue
        x1 = int(np.clip(math.floor(box[0]), 0, max(w - 1, 0)))
        y1 = int(np.clip(math.floor(box[1]), 0, max(h - 1, 0)))
        x2 = int(np.clip(math.ceil(box[2]), x1 + 1, w))
        y2 = int(np.clip(math.ceil(box[3]), y1 + 1, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < int(cfg.get("min_roi_size_px", 64)):
            continue
        expected = int(region.get("expected_count", 2))
        per_region_max_det = max(expected + int(cfg.get("extra_detections", 2)), 2)
        region_kind = str(region.get("kind", "cluster")).strip().lower()
        source = (
            "local_recovery_track_gap"
            if region_kind == "track_gap"
            else "local_recovery_cluster"
        )
        target_logical_id = (
            int(region.get("logical_id", -1)) if region_kind == "track_gap" else -1
        )
        roi_jobs.append(
            (crop, x1, y1, per_region_max_det, source, target_logical_id)
        )

    if not roi_jobs:
        return results_out

    common_kwargs: Dict[str, Any] = {
        "imgsz": int(cfg.get("imgsz", 1024)),
        "conf": float(cfg.get("conf", 0.025)),
        "iou": float(cfg.get("iou", 0.35)),
        # Use the largest original per-ROI limit and apply each crop's original
        # limit again when parsing, preserving downstream candidate semantics.
        "max_det": max(job[3] for job in roi_jobs),
        "device": device,
        "verbose": False,
    }
    if bool(cfg.get("half", False)) and str(device).lower() != "cpu":
        common_kwargs["half"] = True

    batch_results: Optional[List[Any]] = None
    if bool(cfg.get("batch_inference", True)):
        try:
            batch_results = list(
                recovery_model.predict(
                    source=[job[0] for job in roi_jobs],
                    **common_kwargs,
                )
            )
        except Exception as exc:
            logging.warning("局部恢复批量推理失败，回退逐ROI：%s", exc)

    # Compatibility fallback for older Ultralytics versions and OOM errors.
    if batch_results is None:
        batch_results = []
        for crop, _x0, _y0, _per_region_max_det, _source, _target_id in roi_jobs:
            single_kwargs = dict(common_kwargs)
            single_kwargs["source"] = crop
            try:
                local_results = recovery_model.predict(**single_kwargs)
                batch_results.append(local_results[0] if local_results else None)
            except Exception as exc:
                logging.warning("局部二次推理失败，本次ROI跳过：%s", exc)
                batch_results.append(None)

    for result, (
        _crop,
        x0,
        y0,
        per_region_max_det,
        source,
        target_logical_id,
    ) in zip(batch_results, roi_jobs):
        if result is None:
            continue
        local = base.parse_yolo_result(
            result,
            len(KEYPOINT_NAMES),
            per_region_max_det,
        )
        for det in local:
            results_out.append(
                _translate_detection(
                    det,
                    x0,
                    y0,
                    source,
                    recovery_target_logical_id=target_logical_id,
                )
            )
    return results_out

def process_video(
    video_path: Path,
    model_path: Path,
    output_root: Path,
    config: MutableMapping[str, Any],
    calibration_path: Optional[Path],
    manual_annotations: Optional[Path],
    save_clips: bool,
    export_error_clips: bool,
    resume: bool = False,
    checkpoint_every_frames: Optional[int] = None,
    stage_mode: str = "all",
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装ultralytics，请先执行pip install -U ultralytics") from exc

    stage_mode = str(stage_mode).strip().lower()
    if stage_mode not in {"all", "stage1"}:
        raise ValueError(f"process_video只接受all/stage1，实际为：{stage_mode}")
    stage1_only = stage_mode == "stage1"

    # 只读取元数据，随后立即释放VideoCapture。真正推理由Ultralytics单一流式生成器完成。
    meta_cap = cv2.VideoCapture(str(video_path))
    if not meta_cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = float(meta_cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(meta_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(meta_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(meta_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_cap.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"视频元数据异常：fps={fps}, size={width}×{height}")

    video_duration_seconds = total_frames / fps  # 使用已校验的元数据计算本次视频总时长。
    long_term_memory_enabled = should_enable_long_term_memory(  # 在任何身份模块初始化前确定长期记忆状态。
        total_frames,  # 传入总帧数以覆盖短片、五分钟边界和长片三种情况。
        fps,  # 传入实际帧率以支持非30 FPS视频。
        config,  # 传入配置以读取总开关和默认五分钟阈值。
    )  # 完成本视频唯一一次长期记忆策略判定。
    long_term_threshold_seconds = float(  # 读取阈值仅用于日志和结果元数据审计。
        config.get("long_term_memory", {}).get("min_video_duration_seconds", 300.0)  # 未配置时仍固定使用五分钟。
    )  # 得到本次运行所采用的长期记忆阈值。
    output_dir = ensure_dir(output_root / video_path.stem)
    checkpoint_cfg = dict(config.get("checkpoint", {}))
    checkpoint_min_seconds = max(float(checkpoint_cfg.get("min_video_duration_seconds", 300.0)), 0.0)
    checkpoint_stream_safe = bool(config.get("detector_first", {}).get("enabled", False))
    if resume and not checkpoint_stream_safe:
        raise ValueError(
            "--resume当前要求detector_first.enabled=true；普通model.track内部ByteTrack状态"
            "无法可靠序列化，拒绝产生可能变更raw track ID的续跑结果。"
        )
    checkpoint_enabled = bool(
        checkpoint_cfg.get("enabled", True)
        and video_duration_seconds >= checkpoint_min_seconds
        and checkpoint_stream_safe
    )
    if resume:
        checkpoint_enabled = True
    checkpoint_interval = int(
        checkpoint_every_frames
        if checkpoint_every_frames is not None
        else checkpoint_cfg.get("interval_frames", 300)
    )
    checkpoint_manager = InferenceCheckpointManager(
        output_dir=output_dir,
        video_path=video_path,
        model_path=model_path,
        config=config,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        enabled=checkpoint_enabled,
        interval_frames=checkpoint_interval,
        resume_requested=resume,
    )
    checkpoint_manager.prepare_fresh_run()
    checkpoint_payload = checkpoint_manager.load()
    resume_frame = int(checkpoint_payload.get("next_frame", 0)) if checkpoint_payload else 0
    restored_runtime_state = dict(checkpoint_payload.get("runtime_state", {})) if checkpoint_payload else {}
    restored_video_segments = list(checkpoint_payload.get("video_segments", [])) if checkpoint_payload else []
    profiler = RuntimeProfiler(config)
    performance_cfg = dict(config.get("performance", {}))
    # 外层已有Mask ROI线程池和阶段4-7多进程时，限制OpenCV内部线程可避免CPU超卖。
    try:
        cv2.setNumThreads(max(1, int(performance_cfg.get("opencv_threads", 1))))
    except Exception:
        logging.warning("无法设置OpenCV线程数，继续使用当前OpenCV默认值。")
    behavior_pipeline = str(
        performance_cfg.get("behavior_pipeline", "inline")
    ).strip().lower()
    if stage1_only:
        behavior_pipeline = "staged"
    if behavior_pipeline not in {"inline", "staged"}:
        raise ValueError(
            "performance.behavior_pipeline must be 'inline' or 'staged'"
        )
    pair_compute_mode = str(
        performance_cfg.get("pair_compute_mode", "python")
    ).strip().lower()
    if pair_compute_mode not in {"python", "numpy", "multiprocess"}:
        raise ValueError(
            "performance.pair_compute_mode must be 'python', 'numpy', or 'multiprocess'"
        )
    stage3_cache: Optional[Stage3ObservationCache] = None
    if behavior_pipeline == "staged" and (
        stage1_only
        or not bool(config.get("output", {}).get("tracking_only", False))
    ):
        stage3_fingerprint = {
            "video": _checkpoint_file_fingerprint(video_path),
            "model": _checkpoint_file_fingerprint(model_path),
            "stage1_config_sha256": _checkpoint_config_digest(config),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "total_frames": int(total_frames),
        }
        stage3_cache = Stage3ObservationCache(
            output_dir,
            chunk_frames=int(performance_cfg.get("stage3_cache_chunk_frames", 300)),
            resume=bool(checkpoint_payload),
            resume_frame=resume_frame,
            fingerprint=stage3_fingerprint,
            total_frames=total_frames,
        )
        logging.info(
            "阶段化行为管线：步骤1-3写入分块缓存，步骤4-7在推理完成后运行；缓存目录=%s",
            stage3_cache.directory,
        )
    if pair_compute_mode == "multiprocess":
        logging.info(
            "鼠对计算后端：multiprocess（当前阶段化入口使用有序分块；无阶段缓存时回退到numpy枚举）"
        )
    elif pair_compute_mode == "numpy":
        logging.info("鼠对计算后端：numpy索引枚举（行为阈值仍使用既有PairFeatureComputer）")
    if profiler.enabled:
        logging.info(
            "性能 profiling：开启（每%d帧汇总）；异步渲染：%s；ROI批处理：%s；"
            "Identity cost=%s；OpenCV线程=%d",
            profiler.interval_frames,
            bool(performance_cfg.get("async_render", False)),
            bool(performance_cfg.get("batch_roi_inference", True)),
            str(performance_cfg.get("identity_cost_backend", "auto")),
            max(1, int(performance_cfg.get("opencv_threads", 1))),
        )
    logging.info("处理：%s | %d帧 | %.3f FPS | %d×%d", video_path.name, total_frames, fps, width, height)
    if checkpoint_enabled:
        logging.info(
            "断点续跑：%s | 每%d帧提交 | 下一待处理帧%d | 状态文件：%s",
            "恢复" if checkpoint_payload else "新任务",
            checkpoint_manager.interval_frames,
            resume_frame,
            checkpoint_manager.status_path,
        )
    elif bool(checkpoint_cfg.get("enabled", True)) and video_duration_seconds >= checkpoint_min_seconds:
        logging.warning("断点续跑未开启：当前配置没有启用detector_first安全流。")
    logging.info(  # 明确记录为什么本次开启或关闭长期记忆，方便复核运行行为。
        "长期身份记忆：%s | 视频时长：%.3f秒 | 开启条件：达到或超过%.3f秒",  # 日志同时给出状态、时长和边界。
        "开启" if long_term_memory_enabled else "关闭",  # 把布尔状态转换为直观中文。
        video_duration_seconds,  # 输出由视频元数据计算得到的实际时长。
        long_term_threshold_seconds,  # 输出本次采用的五分钟阈值。
    )  # 完成长期记忆策略日志。
    tracking_only = bool(config.get("output", {}).get("tracking_only", False))
    if stage1_only:
        tracking_only = False
        logging.info("阶段一：只执行批量推理、候选过滤、稳定ID和完整缓存；不生成视频、不计算行为。")
    if tracking_only:
        logging.info("仅追踪模式：开启——输出检测/ID/骨架视频和轨迹CSV，跳过成对行为计算。")

    model_cfg = config["model"]
    max_mice = int(model_cfg.get("max_mice", 20))
    logging.info(
        "程序版本：%s | 鼠数模式：全自适应（安全上限%d） | 流式推理：开启 | "
        "检测器优先：%s | 全视频帧缓存：关闭",
        PROGRAM_VERSION, max_mice,
        bool(config.get("detector_first", {}).get("enabled", False)),
    )

    model = YOLO(str(model_path))
    detector_cfg = dict(config.get("detector_first", {}))
    detector_enabled = bool(detector_cfg.get("enabled", False))
    # v1.12.4：检测器只是可选补充通道（补缺/超大框拆分证据），Pose模型才是
    # 主通道。use_detector_model: false 时不再需要 mouse_1280_200_yolo11n.pt。
    use_detector_model = bool(detector_cfg.get("use_detector_model", True))
    # v1.13.0 简洁运动身份模式：不计算外观直方图（身份不靠它，省每帧每检测
    # 一遍直方图的开销）；颜色、OKS同样不参与ID判定。
    pre_identity_mode = str(config.get("identity", {}).get("mode", "hybrid")).lower()
    keypoint_motion_requested = pre_identity_mode in {
        "keypoint_motion", "pure_keypoint", "keypoint_hungarian", "motion_keypoint"
    }
    simple_motion_mode = bool(
        keypoint_motion_requested
        or config.get("identity", {}).get("memory", {}).get("simple_motion_mode", True)
    )
    if keypoint_motion_requested:
        logging.info("逐关键点运动身份模式：开（v1.14.0）——跳过外观直方图，保留姿态/朝向轻量字段")
    elif simple_motion_mode:
        logging.info("简洁运动身份模式：开（v1.13.0）——轨迹段+位置+朝向+速度判ID，跳过外观/OKS计算")
    detector_model = None
    if detector_enabled and use_detector_model:
        detector_path = _resolve_local_resource(detector_cfg.get("model", "mouse_1280_200_yolo11n.pt"))
        if not detector_path.exists():
            # v1.14.0：检测器是召回补充，不应因为可选权重缺失导致整套程序无法运行。
            # 找不到时自动回退Pose主通道+白鼠亮斑，并明确写日志。
            logging.warning(
                "检测器补充通道请求已开启，但找不到模型：%s；"
                "本次自动回退为Pose主通道+白鼠亮斑。要补回孤立漏检，请把"
                "mouse_1280_200_yolo11n.pt放到代码目录或修改detector_first.model。",
                detector_path,
            )
            detector_model = None
        else:
            detector_model = YOLO(str(detector_path))
            logging.info("检测器补充通道：%s | Pose主通道：%s", detector_path, model_path)
    elif detector_enabled:
        if str(detector_cfg.get("pose_mode", "full_frame")).lower() == "roi":
            raise RuntimeError(
                "pose_mode: roi 必须靠检测器框裁ROI，不能关闭检测器模型；"
                "请改用 pose_mode: full_frame，或设 use_detector_model: true。"
            )
        logging.info(
            "检测器补充通道：关闭（use_detector_model: false）| "
            "仅Pose主通道+白鼠亮斑补缺：%s", model_path
        )

    # YOLO第一阶段只在检测器优先的全图Pose路径启用；tracking_only保留原断点语义，
    # 其余完整行为分析默认采用“整段批量YOLO→原有步骤1-7”的新路径。
    device = model_cfg.get("device", 0)
    yolo_precompute_cache: Optional[YOLOPrecomputeCache] = None
    yolo_first_cfg = dict(performance_cfg.get("yolo_first_pass", {}))
    yolo_first_requested = bool(yolo_first_cfg.get("enabled", False))
    if yolo_first_requested and tracking_only:
        logging.info("YOLO-first：仅追踪模式保留原逐帧断点流，跳过全视频预推理。")
    elif yolo_first_requested and not detector_enabled:
        logging.warning("YOLO-first要求detector_first.enabled=true，当前关闭，回退原model.track流。")
    elif yolo_first_requested:
        yolo_precompute_cache = run_yolo_first_pass(
            video_path=video_path,
            model_path=model_path,
            pose_model=model,
            detector_model=detector_model,
            detector_cfg=detector_cfg,
            performance_cfg=performance_cfg,
            output_dir=output_dir,
            config=config,
            device=device,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            expected_keypoints=len(KEYPOINT_NAMES),
            resume_requested=resume,
            profiler=profiler,
        )
    adaptive_arena_result: Optional[arena_boundary.ArenaBoundaryResult] = None
    adaptive_arena_cfg = dict(config.get("adaptive_arena", {}))
    if (
        bool(adaptive_arena_cfg.get("enabled", True))
        and detector_enabled
        and yolo_precompute_cache is not None
    ):
        configured_polygon = detector_cfg.get("arena_mask", {}).get("polygon", [])
        reuse_value = str(adaptive_arena_cfg.get("reuse_boundary_json", "") or "").strip()
        if reuse_value:
            reuse_path = Path(reuse_value)
            if not reuse_path.is_absolute():
                reuse_path = Path(__file__).resolve().parent / reuse_path
            adaptive_arena_result = arena_boundary.load_boundary_json(
                reuse_path,
                width=width,
                height=height,
                source_video=video_path,
                require_video_match=bool(adaptive_arena_cfg.get("reuse_require_video_match", True)),
            )
            heatmap = np.zeros((max(height // 20, 2), max(width // 20, 2)), dtype=np.float32)
        else:
            adaptive_arena_result, heatmap = arena_boundary.learn_from_yolo_records(
                yolo_precompute_cache.iter_frames(start_frame=0),
                width=width,
                height=height,
                config=adaptive_arena_cfg,
                configured_polygon=configured_polygon,
                source_video=video_path,
            )
        arena_boundary.save_boundary_artifacts(
            adaptive_arena_result,
            heatmap,
            output_dir / "阶段一_自适应笼界.json",
            output_dir / "阶段一_运动热力图与笼界.png",
            output_dir / "阶段一_原视频帧叠加笼界.png",
        )
        learned_mask = dict(detector_cfg.get("arena_mask", {}))
        learned_mask["enabled"] = True
        learned_mask["polygon"] = adaptive_arena_result.polygon
        learned_mask["tolerance_px"] = float(adaptive_arena_cfg.get("hard_gate_tolerance_px", 2.0))
        detector_cfg["arena_mask"] = learned_mask
        config.setdefault("detector_first", {})["arena_mask"] = copy.deepcopy(learned_mask)
        logging.info(
            "自适应笼界：%s | 运动样本%d/%d | 面积比%.3f | 边界尺寸调整%+.1f%%；边界外候选不进入ID分配。",
            adaptive_arena_result.source,
            adaptive_arena_result.motion_sample_count,
            adaptive_arena_result.sample_count,
            adaptive_arena_result.occupied_area_ratio,
            100.0 * (adaptive_arena_result.expansion_ratio - 1.0),
        )
    transformer = CoordinateTransformer(config["scale"], video_path, calibration_path)
    identity_cfg = dict(config["identity"])  # 复制身份配置，确保不同视频之间不会共享运行时开关。
    identity_runtime_cfg = dict(identity_cfg)  # 创建仅供当前视频使用的身份配置副本。
    identity_runtime_cfg["long_term_memory_enabled"] = long_term_memory_enabled  # 把时长判定传给长期外观模板模块。
    # Identity类原本只接收identity子配置；显式注入性能后端开关，确保python回退真正可用。
    identity_runtime_cfg["performance"] = {
        "identity_cost_backend": str(performance_cfg.get("identity_cost_backend", "auto")),
        "identity_cost_tie_fallback_epsilon": float(
            performance_cfg.get("identity_cost_tie_fallback_epsilon", 1.0e-10)
        ),
        "identity_cpp_threads": max(
            1, int(performance_cfg.get("identity_cpp_threads", 1))
        ),
        "identity_cpp_auto_threads": bool(
            performance_cfg.get("identity_cpp_auto_threads", True)
        ),
        "identity_cpp_max_threads": max(
            1, int(performance_cfg.get("identity_cpp_max_threads", 4))
        ),
        "identity_cpp_parallel_min_cells": max(
            1, int(performance_cfg.get("identity_cpp_parallel_min_cells", 16384))
        ),
        "identity_cpp_selftest": bool(
            performance_cfg.get("identity_cpp_selftest", True)
        ),
        "identity_cpp_fallback_on_tie": bool(
            performance_cfg.get("identity_cpp_fallback_on_tie", True)
        ),
        # v1.42.1: pass the cascade policy through to the active
        # KeypointMotionIdentityAssigner.  Identity receives a per-video copy,
        # so the global YAML is never mutated at runtime.
        "identity_cascade": dict(performance_cfg.get("identity_cascade", {})),
    }
    instance_mask_runtime_cfg = dict(config.get("instance_mask_memory", {}))  # 复制实例掩码记忆配置以隔离本视频状态。
    instance_mask_runtime_cfg["long_term_memory_enabled"] = long_term_memory_enabled  # 把同一开关传给长期掩码模板模块。
    instance_mask_runtime_cfg["workers"] = max(
        1,
        int(performance_cfg.get("mask_workers", instance_mask_runtime_cfg.get("workers", 1))),
    )
    instance_mask_runtime_cfg["parallel_min_detections"] = max(
        2,
        int(
            performance_cfg.get(
                "mask_parallel_min_detections",
                instance_mask_runtime_cfg.get("parallel_min_detections", 2),
            )
        ),
    )
    identity_runtime_cfg["instance_mask_memory"] = instance_mask_runtime_cfg  # 让身份分配器读取已应用时长策略的掩码配置。
    identity_mode = str(identity_cfg.get("mode", "hybrid")).lower()
    memory_cfg = dict(identity_cfg.get("memory", {}))
    if identity_mode in {"keypoint_motion", "pure_keypoint", "keypoint_hungarian", "motion_keypoint"}:
        # v1.14.0：把旧双鼠PureKeypointTracker扩展为多鼠版本。
        # v1.22额外把顶层实例掩码记忆配置注入身份分配器。
        identity = base.KeypointMotionIdentityAssigner(identity_runtime_cfg, max_mice=max_mice)
        logging.info(
            "身份模式：Adaptive KeypointMotionIdentityAssigner（无需填写鼠数；"
            "候选连续确认后动态建ID；逐关键点运动预测、匈牙利匹配和漏检保持）"
        )
    elif identity_mode in {"memory", "memory_slot", "memory_stable_slot"} or bool(memory_cfg.get("enabled", False)):
        # 修复文档v1.1：短时身份记忆 + 检测与身份分离。
        # 当前帧每个检测都渲染；未匹配检测立即TMP；冲突冻结身份但保留检测。
        identity = base.MemoryIdentityAssigner(identity_runtime_cfg, max_mice=max_mice)
        logging.info(
            "身份模式：MemoryIdentityAssigner（短时身份记忆、检测与身份分离、"
            "未匹配检测即时TMP、冲突保留检测框、体长归一化双门限）"
        )
    elif identity_mode in {"legacy_slot", "legacy_stable_slot", "fixed_slot"}:
        identity = base.LegacyStableSlotAssigner(identity_runtime_cfg, max_mice=max_mice)
        logging.info("身份模式：LegacyStableSlotAssigner（固定槽、忽略raw ID、漏检不删除）")
    else:
        identity = base.StableIdentityAssigner(identity_runtime_cfg, max_mice=max_mice)
    occlusion_manager = base.OcclusionClusterManager(config.get("occlusion_cluster", {}))
    # v1.22：Pose框约束伪实例掩码 + 聚集后延迟身份恢复。
    mask_extractor = mask_reid.PseudoInstanceMaskExtractor(instance_mask_runtime_cfg)  # 掩码提取保留，但长期模板服从视频时长开关。
    cluster_reid = mask_reid.ClusterReIDResolver(config.get("cluster_reid", {}))
    mask_trigger = mask_trigger_controller.MaskTriggerController(
        performance_cfg.get("mask_trigger", {}),
        instance_mask_runtime_cfg,
        config.get("cluster_reid", {}),
    )
    provisional_tracker = ProvisionalDisplayTracker(
        config.get("provisional_render", {}), max_tracks=max(max_mice * 2, 20)
    )
    logging.info(
        "实例掩码短时记忆：%s（Pose框+关键点种子+GrabCut伪掩码，ROI线程=%d）；长期身份模板：%s；"
        "聚集后延迟ReID：%s；未确认检测完整渲染：%s",
        "开启" if mask_extractor.enabled else "关闭",
        int(getattr(mask_extractor, "workers", 1)),
        "开启" if long_term_memory_enabled else "关闭",  # 单独显示按五分钟策略生效的长期模板状态。
        "开启" if cluster_reid.enabled else "关闭",
        "开启" if provisional_tracker.enabled else "关闭",
    )
    recovery_cfg = dict(config.get("occlusion_recovery", {}))
    recovery_cfg.setdefault(
        "batch_inference",
        bool(performance_cfg.get("batch_roi_inference", True)),
    )
    recovery_model = None

    # v1.20：异常骨架不再直接渲染。先做解剖学质量门控，坏骨架仅对对应
    # ROI使用同一Pose权重二次推理；身份分配后再按逻辑ID做逐关键点时序门控。
    pose_recovery_cfg = dict(config.get("pose_recovery", {}))
    pose_recovery_enabled = bool(pose_recovery_cfg.get("enabled", True))
    pose_roi_recoverer = pose_recovery.BadPoseROIRecoverer(
        pose_recovery_cfg, expected_keypoints=len(KEYPOINT_NAMES)
    )
    temporal_pose_cfg = dict(pose_recovery_cfg)
    temporal_pose_cfg.update(dict(pose_recovery_cfg.get("temporal", {})))
    temporal_pose_repairer = pose_recovery.TemporalKeypointRepairer(
        temporal_pose_cfg, num_keypoints=len(KEYPOINT_NAMES)
    )
    bad_pose_recovery_model = None
    if pose_recovery_enabled:
        logging.info(
            "Pose修复：开启（质量门控→坏骨架ROI二次推理→逐关键点时序门控→DISK序列补点→模板补点）；"
            "DISK/PREDICTED/TEMPLATE点仅用于可视化，不作为行为强证据"
        )

    smoother = base.KeypointSmoother(
        alpha=float(config["keypoints"]["smoothing_alpha"]),
        # v1.11.4：平滑阈值独立于渲染阈值（0.1），保住0.10~0.20置信度的
        # nose/tail端点——旧值0.2会把它们抹除导致骨架缺端点呈"蝴蝶结"。
        min_conf=float(config["keypoints"].get(
            "smoothing_min_confidence", config["keypoints"].get("min_confidence", 0.1)
        )),
        max_missing=int(config["keypoints"]["interpolation_max_frames"]),
        # v1.12.4：插值点置信度衰减系数；地板=min_conf保证窗口内可渲染。
        interp_decay=float(config["keypoints"].get("interpolation_conf_decay", 0.85)),
    )
    history_seconds = max(float(config["features"]["history_seconds"]), 1.0)
    history = base.ObservationHistory(max_frames=max(int(round(fps * history_seconds)) + 5, 10))
    feature_computer = PairFeatureComputer(fps, config)
    individual_behavior_gate = IndividualBehaviorGate(fps, config)
    contact_tracker = base.PairContactTracker(
        fps=fps,
        window_seconds=float(config["attack"]["weak"]["repeated_contact_window_seconds"]),
    )
    detector_candidate_filter = DetectorCandidateFilter(
        detector_cfg.get("candidate_filter", {})
    )

    if restored_runtime_state:
        required_runtime_keys = {
            "identity",
            "occlusion_manager",
            "mask_extractor",
            "cluster_reid",
            "provisional_tracker",
            "temporal_pose_repairer",
            "smoother",
            "history",
            "individual_behavior_gate",
            "contact_tracker",
            "transformer",
            "detector_candidate_filter",
            "pair_store_initialized",
        }
        missing_runtime_keys = sorted(required_runtime_keys - set(restored_runtime_state))
        if missing_runtime_keys:
            raise ValueError("推理断点缺少运行时状态：" + ", ".join(missing_runtime_keys))
        identity = restored_runtime_state["identity"]
        occlusion_manager = restored_runtime_state["occlusion_manager"]
        mask_extractor = restored_runtime_state["mask_extractor"]
        cluster_reid = restored_runtime_state["cluster_reid"]
        provisional_tracker = restored_runtime_state["provisional_tracker"]
        temporal_pose_repairer = restored_runtime_state["temporal_pose_repairer"]
        smoother = restored_runtime_state["smoother"]
        history = restored_runtime_state["history"]
        individual_behavior_gate = restored_runtime_state["individual_behavior_gate"]
        contact_tracker = restored_runtime_state["contact_tracker"]
        transformer = restored_runtime_state["transformer"]
        detector_candidate_filter = restored_runtime_state["detector_candidate_filter"]
        logging.info(
            "已恢复身份、长期模板、遮挡ReID、关键点历史和行为接触状态；从第%d帧继续。",
            resume_frame,
        )

    def checkpoint_runtime_state() -> Dict[str, Any]:
        """收集影响后续ID和行为判定的有界状态；模型和文件句柄不进入断点。"""
        return {
            "identity": identity,
            "occlusion_manager": occlusion_manager,
            "mask_extractor": mask_extractor,
            "cluster_reid": cluster_reid,
            "provisional_tracker": provisional_tracker,
            "temporal_pose_repairer": temporal_pose_repairer,
            "smoother": smoother,
            "history": history,
            "individual_behavior_gate": individual_behavior_gate,
            "contact_tracker": contact_tracker,
            "transformer": transformer,
            "detector_candidate_filter": detector_candidate_filter,
            "pair_store_initialized": bool(raw_store.initialized),
        }

    cache_csv_paths = [
        output_dir / "逐帧检测流缓存.csv",
        output_dir / "身份分配调试记录.csv",
        output_dir / "检测轨迹对应表.csv",
        output_dir / "遮挡簇调试记录.csv",
        output_dir / "实例掩码质量记录.csv",
        output_dir / "聚集后身份恢复记录.csv",
    ]
    if checkpoint_payload:
        for cache_path in cache_csv_paths:
            _truncate_csv_at_frame(cache_path, resume_frame)
        _validate_frame_cache_boundary(output_dir / "逐帧检测流缓存.csv", resume_frame)

    raw_store = PairSQLiteStore(
        output_dir / "成对原始流缓存.sqlite3",
        batch_size=int(config.get("memory", {}).get("sqlite_batch_rows", 1500)),
        resume=bool(checkpoint_payload),
        resume_frame=resume_frame,
        expected_initialized=(
            bool(restored_runtime_state.get("pair_store_initialized"))
            if checkpoint_payload
            else None
        ),
    )
    frame_fields = [
        "frame", "time_s", "detected_mice_count", "detected_logical_ids", "raw_track_ids",
        "scale_mode", "cm_per_pixel", "active_occlusion_cluster_count",
        "missing_in_cluster_count", "local_recovery_detection_count",
        "track_gap_recovery_region_count", "track_gap_recovery_detection_count",
        "detector_candidate_count", "bbox_only_count", "white_roi_count",
        # v1.11.0 身份阶段统计（文档§8日志格式）
        "identity_matched_count", "identity_low_rescued_count", "identity_lost_recovered_count",
        "identity_new_tentative_count", "identity_tentative_count", "identity_suspicious_count",
        "identity_lost_count", "identity_rendered_count",
        "adaptive_estimated_mouse_count", "adaptive_visible_mouse_count",
        "disk_sequence_active_tracks", "disk_sequence_mean_reliability",
        "disk_sequence_rejected_updates", "disk_contact_order_veto_count",
        "disk_contact_order_veto_total", "contact_order_regret_veto_count",
        "contact_order_regret_veto_total",
        # v1.12.1 文档§8逐阶段检测计数（定位丢鼠环节）
        "stage_raw_count", "stage_pose_count", "stage_supplement_count",
        "stage_after_fusion_count", "stage_after_mask_count",
        "stage_after_reflection_count", "stage_after_static_count",
        # v1.20 Pose异常骨架恢复统计
        "pose_bad_count", "pose_roi_attempts", "pose_roi_accepted", "pose_roi_rejected",
        "temporal_rejected_outliers", "temporal_disk_points",
        "temporal_predicted_points", "temporal_template_points",
        # v1.22 掩码与聚集后ReID统计
        "mask_attempted_count", "mask_reliable_count", "mask_fallback_count", "mask_failed_count",
        "cluster_reid_active_count", "cluster_reid_ambiguous_count",
        "cluster_frozen_id_count", "cluster_reserved_detection_count",
        # v1.23 检测完整渲染守恒统计
        "post_dedup_renderable_detection_count", "formal_id_render_count",
        "cluster_anonymous_render_count", "provisional_render_count",
        "final_detection_render_count", "unrendered_detection_count",
        "tracklet_bridge_count", "predicted_hold_render_count",
    ]
    frame_handle, frame_writer = _open_dict_writer(
        output_dir / "逐帧检测流缓存.csv",
        frame_fields,
        append=bool(checkpoint_payload),
    )
    debug_fields = [
        "frame", "logical_id", "raw_track_id", "assignment_cost", "proposed_logical_id",
        "assignment_gain", "dwell_count", "dwell_required", "cooldown_remaining",
        "commit_status", "switch_rejected_reason", "appearance_mode", "detection_source",
        "occlusion_cluster_id", "cluster_expected_count", "cluster_observed_count",
        "track_state",
    ]
    debug_handle, debug_writer = _open_dict_writer(
        output_dir / "身份分配调试记录.csv",
        debug_fields,
        append=bool(checkpoint_payload),
    )
    # 文档§9：每个当前检测一行，含detection/logical_id、display_label、track_state。
    det_map_fields = [
        "frame", "time_s", "detection_index", "logical_id", "display_label", "track_state",
        "detection_source", "appearance_mode", "box_conf", "pose_valid_keypoints",
        "center_x_px", "center_y_px", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "assignment_cost", "match_method",
        "pose_recovery_score", "pose_recovery_reason", "keypoint_sources",
        "mask_quality", "mask_reliable", "mask_source", "mask_area_ratio",
        "candidate_identity_set", "cluster_reid_id",
        "render_class", "is_behavior_eligible",
    ]
    det_map_handle, det_map_writer = _open_dict_writer(
        output_dir / "检测轨迹对应表.csv",
        det_map_fields,
        append=bool(checkpoint_payload),
    )
    cluster_fields = [
        "frame", "cluster_id", "members", "expected_count", "observed_count", "deficit",
        "merged_like", "merged_member_count", "max_det_area_ratio", "locally_visible_count",
        "ambiguity_frames",
        "max_iou", "motion_bl_per_frame", "active_frames", "attack_hint",
        "recovery_requested", "local_recovery_added",
    ]
    cluster_handle, cluster_writer = _open_dict_writer(
        output_dir / "遮挡簇调试记录.csv",
        cluster_fields,
        append=bool(checkpoint_payload),
    )
    mask_fields = [
        "frame", "detection_index", "source", "quality", "reliable", "area_ratio",
        "keypoint_coverage", "border_fraction", "error",
    ]
    mask_handle, mask_writer = _open_dict_writer(
        output_dir / "实例掩码质量记录.csv",
        mask_fields,
        append=bool(checkpoint_payload),
    )
    reid_fields = [
        "frame", "cluster_id", "members", "stage", "tracklet_count", "best_cost",
        "second_cost", "margin", "reason",
    ]
    reid_handle, reid_writer = _open_dict_writer(
        output_dir / "聚集后身份恢复记录.csv",
        reid_fields,
        append=bool(checkpoint_payload),
    )

    annotated_writer = None
    annotated_async_writer: Optional[AsyncAnnotatedVideoWriter] = None
    checkpoint_video_writer: Optional[CheckpointSegmentedVideoWriter] = None
    annotated_path: Optional[Path] = None
    if (not stage1_only) and bool(config["output"].get("save_annotated_video", True)):
        annotated_path = output_dir / "追踪标注视频_仅框与骨架.mp4"
        if checkpoint_enabled:
            checkpoint_video_writer = CheckpointSegmentedVideoWriter(
                output_dir=output_dir,
                fps=fps,
                width=width,
                height=height,
                async_render=bool(performance_cfg.get("async_render", False)),
                queue_size=int(performance_cfg.get("render_queue_size", 16)),
                profiler=profiler,
                resume_segments=restored_video_segments,
                resume_frame=resume_frame,
                fresh_run=not bool(checkpoint_payload),
                video_encoding_cfg=config.get("video_encoding", {}),
            )
            logging.info(
                "断点视频分段：开启（已提交%d段；异步渲染=%s；队列%d）",
                len(checkpoint_video_writer.segments),
                bool(performance_cfg.get("async_render", False)),
                max(int(performance_cfg.get("render_queue_size", 16)), 1),
            )
        else:
            annotated_writer = nvenc_video_writer.create_video_writer(
                annotated_path, fps, width, height, config.get("video_encoding", {})
            )
            if not annotated_writer.isOpened():
                annotated_writer = None
                logging.warning("无法创建核查视频，将继续输出CSV和事件片段：%s", annotated_path)
            elif bool(performance_cfg.get("async_render", False)):
                annotated_async_writer = AsyncAnnotatedVideoWriter(
                    annotated_writer,
                    max_queue_size=int(performance_cfg.get("render_queue_size", 16)),
                    profiler=profiler,
                )
                logging.info(
                    "异步渲染/编码：开启（队列%d；队列满时按顺序回压）",
                    max(int(performance_cfg.get("render_queue_size", 16)), 1),
                )

    tracker_value = str(model_cfg.get("tracker", "bytetrack_mouse20.yaml"))
    tracker_candidate = Path(tracker_value)
    if not tracker_candidate.is_absolute():
        local_tracker = Path(__file__).resolve().parent / tracker_candidate
        tracker_path = str(local_tracker if local_tracker.exists() else tracker_value)
    else:
        tracker_path = str(tracker_candidate)
    device = model_cfg.get("device", 0)
    track_kwargs: Dict[str, Any] = {
        "source": str(video_path),
        "stream": True,
        "persist": True,
        "tracker": tracker_path,
        "imgsz": int(model_cfg.get("imgsz", 768)),
        "conf": float(model_cfg.get("conf", 0.15)),
        "iou": float(model_cfg.get("iou", 0.5)),
        "max_det": int(model_cfg.get("max_det", max_mice + 10)),
        "device": device,
        "verbose": False,
        "stream_buffer": False,
        "vid_stride": 1,
    }
    if bool(model_cfg.get("half", False)) and str(device).lower() != "cpu":
        track_kwargs["half"] = True

    frame_idx = int(resume_frame)
    csv_handles = (
        frame_handle,
        debug_handle,
        det_map_handle,
        cluster_handle,
        mask_handle,
        reid_handle,
    )

    def commit_inference_checkpoint(inference_complete: bool) -> None:
        """在视频段、SQLite和CSV都持久化后原子提交同一帧边界。"""
        if not checkpoint_enabled:
            return
        if checkpoint_video_writer is not None:
            checkpoint_video_writer.commit_segment(frame_idx)
        if stage3_cache is not None:
            # Persist the same frame boundary as the CSV/video checkpoint so a
            # resume never replays a half-written stage-3 chunk.
            stage3_cache.flush()
        raw_store.flush()
        _flush_csv_handles(csv_handles)
        segments = checkpoint_video_writer.segments if checkpoint_video_writer is not None else []
        checkpoint_manager.save(
            next_frame=frame_idx,
            runtime_state=checkpoint_runtime_state(),
            video_segments=segments,
            inference_complete=inference_complete,
        )
        logging.info(
            "断点已提交：下一帧%d/%d（%.2f%%），视频分段%d个。",
            frame_idx,
            total_frames,
            100.0 * frame_idx / max(total_frames, 1),
            len(segments),
        )

    if checkpoint_enabled and checkpoint_payload is None:
        commit_inference_checkpoint(inference_complete=False)

    prefetched_result_stream: Optional[PrefetchedResultStream] = None
    try:
        if detector_enabled:
            yolo_records = (
                yolo_precompute_cache.iter_frames(start_frame=resume_frame)
                if yolo_precompute_cache is not None
                else None
            )
            result_stream = _detector_first_stream(
                video_path=video_path,
                detector_model=detector_model,
                pose_model=model,
                config=detector_cfg,
                device=device,
                expected_keypoints=len(KEYPOINT_NAMES),
                start_frame=resume_frame,
                candidate_filter=detector_candidate_filter,
                precomputed_records=yolo_records,
            )
            if bool(performance_cfg.get("prefetch_inference", False)):
                prefetched_result_stream = PrefetchedResultStream(
                    source=result_stream,
                    max_queue_size=int(performance_cfg.get("prefetch_queue_size", 1)),
                    start_frame=resume_frame,
                    checkpoint_interval=(
                        checkpoint_manager.interval_frames if checkpoint_enabled else 0
                    ),
                )
                result_stream = prefetched_result_stream
                logging.info(
                    "下一帧解码/全图推理预取：开启（队列%d；断点边界同步=%s）",
                    max(int(performance_cfg.get("prefetch_queue_size", 1)), 1),
                    "开启" if checkpoint_enabled else "无需",
                )
        elif resume_frame > 0:
            result_stream = _resume_model_track_stream(
                video_path,
                model,
                track_kwargs,
                start_frame=resume_frame,
            )
        else:
            result_stream = model.track(**track_kwargs)
        result_stream = _iter_profiled_results(result_stream, profiler)
        with tqdm(
            total=total_frames if total_frames > 0 else None,
            initial=resume_frame,
            desc=f"流式多鼠粗筛 {video_path.name}",
            unit="frame",
        ) as pbar:
            for result in result_stream:
                frame = result.orig_img
                if frame is None:
                    continue
                hybrid_meta = dict(getattr(result, "hybrid_meta", {}) or {})
                if hasattr(result, "hybrid_detections"):
                    # Detailed parsing/transfer timing was measured inside _detector_first_stream.
                    detections = list(result.hybrid_detections)
                else:
                    candidate_limit = min(
                        int(model_cfg.get("max_det", max_mice + 10)),
                        max_mice + int(config.get("identity", {}).get("candidate_extra", 10)),
                    )
                    parse_profile: Dict[str, float] = {}
                    detections = base.parse_yolo_result(
                        result, len(KEYPOINT_NAMES), candidate_limit, profiling=parse_profile
                    )
                    profiler.add(
                        "yolo_result_transfer",
                        float(parse_profile.get("yolo_result_transfer_seconds", 0.0)),
                    )
                    profiler.add(
                        "result_parse",
                        float(parse_profile.get("result_parse_seconds", 0.0)),
                    )
                # 先做保守实例去重：只有框、中心和关键点都近乎相同才删除，
                # 避免打斗区域的一只鼠被重复检测后凭空创建新逻辑ID。
                dedup_started = time.perf_counter()
                detections = base.suppress_duplicate_detections(
                    detections, config.get("duplicate_suppression", {})
                )
                profiler.add("dedup", time.perf_counter() - dedup_started)

                pose_cpu_extra = 0.0
                pose_recovery_stats = pose_recovery.FrameRecoveryStats()
                if pose_recovery_enabled and detections:
                    pose_eval_started = time.perf_counter()
                    has_bad_pose = any(
                        not pose_recovery.evaluate_detection(det, pose_recovery_cfg).is_good
                        for det in detections
                    )
                    pose_cpu_extra += time.perf_counter() - pose_eval_started
                    if has_bad_pose and bad_pose_recovery_model is None:
                        logging.info(
                            "首次发现异常骨架，加载独立Pose ROI恢复模型：%s", model_path
                        )
                        bad_pose_recovery_model = YOLO(str(model_path))
                    recovery_pose_model = bad_pose_recovery_model if has_bad_pose else model
                    detections, pose_recovery_stats = pose_roi_recoverer.recover_frame(
                        recovery_pose_model, frame, detections, device=device,
                        half=bool(model_cfg.get("half", False)),
                    )
                    profiler.add(
                        "roi_inference",
                        float(getattr(pose_roi_recoverer, "last_inference_seconds", 0.0)),
                    )
                    profiler.add(
                        "pose_recovery_cpu",
                        pose_cpu_extra + float(getattr(pose_roi_recoverer, "last_cpu_seconds", 0.0)),
                    )

                detections = base.enrich_detections_with_appearance(
                    frame, detections, config.get("identity", {}),
                    skip_appearance=simple_motion_mode,
                )

                # 基于上一帧稳定逻辑轨迹识别接触簇。簇内冻结ID集合、禁止新ID，
                # 当“预期数量 > 当前检测数量”时触发局部二次推理。
                cluster_started = time.perf_counter()
                cluster_state_snapshot = (
                    copy.deepcopy(occlusion_manager.states),
                    int(occlusion_manager.next_cluster_id),
                    copy.deepcopy(occlusion_manager.last_recovery_frame),
                    len(occlusion_manager.debug_rows),
                )
                cluster_context = occlusion_manager.build_context(
                    identity, detections, frame_idx, frame.shape[:2]
                )
                profiler.add("cluster_context", time.perf_counter() - cluster_started)
                track_gap_regions = _build_track_gap_recovery_regions(
                    identity,
                    detections,
                    frame_idx,
                    frame.shape[:2],
                    recovery_cfg,
                    frozen_ids=cluster_context.get("frozen_ids", set()),
                )
                recovery_regions = list(cluster_context.get("recovery_regions", []))
                recovery_regions.extend(track_gap_regions)
                local_recovery_detections: List[base.Detection] = []
                if (
                    bool(recovery_cfg.get("enabled", True))
                    and recovery_regions
                ):
                    configured_recovery_device = recovery_cfg.get("device", "same")
                    recovery_device = (
                        device
                        if str(configured_recovery_device).strip().lower() == "same"
                        else configured_recovery_device
                    )
                    if recovery_model is None:
                        logging.info(
                            "首次触发局部二次推理（接触簇/轨迹缺口），加载恢复模型到设备：%s",
                            recovery_device,
                        )
                        recovery_model = YOLO(str(model_path))
                    local_recovery_started = time.perf_counter()
                    local_recovery_detections = _run_local_occlusion_recovery(
                        recovery_model,
                        frame,
                        recovery_regions,
                        recovery_cfg,
                        recovery_device,
                    )
                    profiler.add(
                        "roi_inference",
                        time.perf_counter() - local_recovery_started,
                    )
                    if local_recovery_detections:
                        local_dedup_started = time.perf_counter()
                        detections = base.suppress_duplicate_detections(
                            list(detections) + local_recovery_detections,
                            config.get("duplicate_suppression", {}),
                        )
                        profiler.add("dedup", time.perf_counter() - local_dedup_started)
                        detections = base.enrich_detections_with_appearance(
                            frame, detections, config.get("identity", {}),
                            skip_appearance=simple_motion_mode,
                        )
                        # 重新计算检测索引与冻结上下文，但先恢复管理器状态，
                        # 避免同一帧调用两次导致active_frames和恢复冷却重复递增。
                        (
                            occlusion_manager.states,
                            occlusion_manager.next_cluster_id,
                            occlusion_manager.last_recovery_frame,
                            previous_debug_length,
                        ) = cluster_state_snapshot
                        del occlusion_manager.debug_rows[previous_debug_length:]
                        rebuilt_started = time.perf_counter()
                        cluster_context = occlusion_manager.build_context(
                            identity, detections, frame_idx, frame.shape[:2]
                        )
                        profiler.add(
                            "cluster_context",
                            time.perf_counter() - rebuilt_started,
                        )
                # v1.22：对全部最终候选生成Pose框约束伪实例掩码。真实掩码质量
                # 不足时不会进入身份代价；聚集/高IoU期间也不会更新长期模板。
                mask_started = time.perf_counter()
                mask_decision = mask_trigger.decide(
                    frame_idx, detections, cluster_context, recovery_regions, identity
                )
                if mask_decision.run_mask:
                    mask_stats = mask_extractor.enrich_frame(
                        frame, detections, frame_idx=frame_idx
                    )
                else:
                    mask_stats = SimpleNamespace(debug_rows=[])
                profiler.add("mask", time.perf_counter() - mask_started)
                if mask_stats.debug_rows:
                    mask_writer.writerows(
                        {**dict(row), "frame": int(frame_idx)}
                        for row in mask_stats.debug_rows
                    )

                identity_started = time.perf_counter()
                transformer.update(detections)
                guarded_context = cluster_reid.prepare(
                    frame_idx, detections, cluster_context, identity, frame.shape[:2]
                )
                assigned = identity.assign(
                    detections, frame_idx, occlusion_context=guarded_context
                )
                profiler.add(
                    "identity_cost",
                    float(getattr(identity, "last_cost_build_seconds", 0.0)),
                )
                profiler.add(
                    "identity_assignment",
                    float(getattr(identity, "last_assignment_seconds", 0.0)),
                )
                cluster_assigned = cluster_reid.resolve(
                    frame_idx, detections, identity, guarded_context
                )
                profiler.add("identity", time.perf_counter() - identity_started)
                # 普通区域用正式ID；聚集区域用Cxx-A匿名tracklet，只有延迟ReID
                # 全局匹配达到成本和边际门槛后才恢复原ID。
                assigned_map: Dict[int, base.Detection] = {int(lid): det for lid, det in assigned}
                for lid, det in cluster_assigned:
                    assigned_map[int(lid)] = det
                assigned = sorted(assigned_map.items(), key=lambda x: x[0])

                # v1.11.0：身份分配器输出的每检测标签/状态与逐帧阶段统计（文档§3.2/§8）。
                output_info: Dict[int, Dict[str, Any]] = dict(getattr(identity, "output_info", {}) or {})
                output_info.update(cluster_reid.output_info)
                id_stats: Dict[str, int] = dict(getattr(identity, "frame_stats", {}) or {})
                id_stats["cluster_reid_active"] = int(cluster_reid.active_count)
                id_stats["cluster_reid_ambiguous"] = int(cluster_reid.ambiguous_count)

                # 运动预测仅用于内部ID保持，不能画假骨架，也不能参与行为距离计算。
                # 同一检测若意外被多个逻辑ID引用，只保留优先级最高的一项：
                # 正式ID优先于聚集匿名ID，避免一只鼠在最终视频中重复绘制。
                candidate_assignments = [
                    (int(logical_id), detection)
                    for logical_id, detection in assigned
                    if not bool(detection.synthetic_recovery)
                    and str(getattr(detection, "detection_source", "global")) != "predicted_hold"
                ]
                real_assigned: List[Tuple[int, base.Detection]] = []
                used_detection_objects: set[int] = set()
                for logical_id, detection in sorted(
                    candidate_assignments,
                    key=lambda row: (0 if int(row[0]) >= 0 else 1, int(row[0])),
                ):
                    object_key = id(detection)
                    if object_key in used_detection_objects:
                        logging.warning(
                            "Frame %d: 同一检测被多个身份引用，已保留优先身份并删除重复绘制：logical_id=%s",
                            frame_idx, logical_id,
                        )
                        continue
                    used_detection_objects.add(object_key)
                    real_assigned.append((logical_id, detection))
                real_assigned.sort(key=lambda row: int(row[0]))

                # v1.23：检测完整性与身份确定性分离。所有真实YOLO候选都必须出现在
                # 最终核查视频中；未获正式ID/聚集匿名ID的候选交给独立Pxx短时显示轨迹器。
                renderable_detections = [
                    det for det in detections
                    if not bool(getattr(det, "synthetic_recovery", False))
                    and str(getattr(det, "detection_source", "global")) != "predicted_hold"
                ]
                assigned_object_ids = {id(det) for _, det in real_assigned}
                unassigned_detections = [
                    det for det in renderable_detections if id(det) not in assigned_object_ids
                ]
                provisional_assigned = provisional_tracker.update(
                    unassigned_detections,
                    frame_idx,
                    assigned_detections=[det for _, det in real_assigned],
                )
                # v1.29: 临时轨迹连续出现时，优先接回仍在保留窗内的旧正式 ID。
                # 这一步发生在Pxx最终渲染前，因而不会先把同一只鼠画成Pxx再突然跳成新ID。
                bridge_assigned: List[Tuple[int, base.Detection]] = []
                bridge_count = 0
                if hasattr(identity, "bridge_provisional_track"):
                    occupied_ids = {int(logical_id) for logical_id, _ in real_assigned}
                    remaining_provisional = []
                    for provisional_id, detection, provisional_label, provisional_cost in provisional_assigned:
                        provisional_stats = provisional_tracker.stats(int(provisional_id))
                        bridged_id = identity.bridge_provisional_track(
                            detection,
                            frame_idx,
                            occupied_ids=occupied_ids,
                            provisional_hits=int(provisional_stats.get("hits", 0.0)),
                            provisional_motion_bl=float(provisional_stats.get("motion_body_lengths", 0.0)),
                        )
                        if bridged_id is None:
                            remaining_provisional.append(
                                (provisional_id, detection, provisional_label, provisional_cost)
                            )
                            continue
                        bridge_assigned.append((int(bridged_id), detection))
                        occupied_ids.add(int(bridged_id))
                        provisional_tracker.remove(int(provisional_id))
                        bridge_count += 1
                    provisional_assigned = remaining_provisional
                    if bridge_assigned:
                        real_assigned.extend(bridge_assigned)
                        real_assigned.sort(key=lambda row: int(row[0]))
                        output_info.update(getattr(identity, "output_info", {}) or {})
                        id_stats["tracklet_bridged"] = int(bridge_count)
                provisional_info: Dict[int, Dict[str, Any]] = {
                    int(pid): {
                        "state": "provisional",
                        "label": str(label),
                        "cost": float(cost),
                        "method": "provisional_display_only",
                        "cluster_id": -1,
                        "candidate_ids": "",
                    }
                    for pid, _det, label, cost in provisional_assigned
                }

                # 只对真实检测做完整性计数；预测保持和Pxx仍然属于可视化附加层。
                formal_render_count = sum(1 for lid, _ in real_assigned if int(lid) >= 0)
                cluster_anonymous_render_count = sum(1 for lid, _ in real_assigned if int(lid) < 0)
                final_detection_render_count = len(real_assigned) + len(provisional_assigned)
                unrendered_detection_count = len(renderable_detections) - final_detection_render_count
                if unrendered_detection_count != 0:
                    raise RuntimeError(
                        f"Frame {frame_idx}: 检测完整渲染不守恒：去重后检测={len(renderable_detections)}，"
                        f"正式/聚集={len(real_assigned)}，临时={len(provisional_assigned)}，"
                        f"未渲染={unrendered_detection_count}。"
                    )

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    pending_now = int(id_stats.get("tentative", 0))
                    estimated_now = int(id_stats.get("adaptive_estimated_count", 0))
                    logging.debug(
                        "Frame %d: dedup_detections=%d, formal=%d, cluster_anon=%d, provisional=%d, "
                        "final_render=%d, pending_pool=%d, estimated_total=%d",
                        frame_idx, len(renderable_detections), formal_render_count,
                        cluster_anonymous_render_count, len(provisional_assigned),
                        final_detection_render_count, pending_now, estimated_now,
                    )

                observations: List[base.MouseObservation] = []
                visual_by_id: Dict[int, base.MouseObservation] = {}
                predicted_hold_assigned = (
                    identity.render_predictions(frame_idx, blocked_detections=renderable_detections)
                    if hasattr(identity, "render_predictions") else []
                )
                temporal_rejected_outliers = 0
                temporal_disk_points = 0
                temporal_predicted_points = 0
                temporal_template_points = 0
                observation_started = time.perf_counter()
                for logical_id, detection in real_assigned:
                    info = output_info.get(int(logical_id), {})
                    state_now = str(info.get("state", "tracked"))
                    prev_obs = history.previous(logical_id)
                    # 聚集后刚恢复原ID时不跨越整个遮挡时段计算瞬时速度，避免伪高速。
                    if state_now == "reid_resolved":
                        prev_obs = None
                    ref_pts = None if prev_obs is None else np.asarray(prev_obs.keypoints_px, dtype=np.float64)
                    # v1.12.7：头尾方向校正（速度一致性守卫）在先——模型在域外
                    # 场地可能整套骨架180°颠倒，先按运动方向/帧间连续性翻正；
                    # v1.11.3手性校正在后处理左右镜像。
                    vel_px = None
                    if prev_obs is not None and frame_idx > int(prev_obs.frame):
                        dt = frame_idx - int(prev_obs.frame)
                        if dt <= 5:
                            prev_c = np.asarray(prev_obs.bbox_xyxy, dtype=np.float64)
                            prev_center = np.array([(prev_c[0] + prev_c[2]) / 2.0,
                                                    (prev_c[1] + prev_c[3]) / 2.0])
                            vel_px = (np.asarray(detection.center_px, dtype=np.float64)
                                      - prev_center) / float(dt)
                    ht_px, ht_conf, _ht_flipped = base.correct_head_tail_orientation(
                        detection.keypoints_px, detection.keypoint_conf,
                        reference_px=ref_pts, velocity_px=vel_px,
                    )
                    # v1.11.3：左右耳/髋单帧镜像翻转校正（"蝴蝶结"骨架），
                    # 以轨迹上一帧为基准做帧间一致性选择。
                    fixed_px, fixed_conf, _flipped = base.stabilize_keypoint_chirality(
                        ht_px, ht_conf, ref_pts
                    )
                    if pose_recovery_enabled:
                        input_sources = getattr(detection, "keypoint_sources", None)
                        repair = temporal_pose_repairer.update(
                            logical_id=int(logical_id),
                            frame_idx=frame_idx,
                            points=fixed_px,
                            confidence=fixed_conf,
                            bbox=detection.bbox_xyxy,
                            input_sources=input_sources,
                            suspicious=(state_now == "suspicious"),
                        )
                        # 行为分析只用RAW/ROI可靠点；预测/模板点仅给核查视频补骨架。
                        smoothed_px = repair.analysis_points
                        effective_conf = repair.analysis_confidence
                        visual_px = repair.visual_points
                        visual_conf = repair.visual_confidence
                        visual_sources = repair.sources
                        temporal_rejected_outliers += int(repair.rejected_outliers)
                        temporal_disk_points += int(repair.disk_count)
                        temporal_predicted_points += int(repair.predicted_count)
                        temporal_template_points += int(repair.template_count)
                    elif state_now == "suspicious":
                        # 旧模式兼容：身份不确定期间冻结普通EMA平滑器。
                        smoothed_px = np.asarray(fixed_px, dtype=np.float64).copy()
                        effective_conf = np.asarray(fixed_conf, dtype=np.float64).copy()
                        visual_px = smoothed_px.copy()
                        visual_conf = effective_conf.copy()
                        visual_sources = np.full(len(KEYPOINT_NAMES), pose_recovery.SOURCE_RAW, dtype=object)
                        smoother.points.pop(logical_id, None)
                        smoother.missing_counts.pop(logical_id, None)
                        smoother.last_raw_conf.pop(logical_id, None)
                    else:
                        smoothed_px, effective_conf = smoother.update(logical_id, fixed_px, fixed_conf)
                        visual_px = smoothed_px.copy()
                        visual_conf = effective_conf.copy()
                        visual_sources = np.full(len(KEYPOINT_NAMES), pose_recovery.SOURCE_RAW, dtype=object)
                    obs = build_observation(
                        frame=frame_idx,
                        fps=fps,
                        logical_id=logical_id,
                        detection=detection,
                        smoothed_keypoints_px=smoothed_px,
                        effective_conf=effective_conf,
                        transformer=transformer,
                        previous=prev_obs,
                        track_state=state_now,
                        display_label=str(info.get("label", "")),
                    )
                    obs.keypoint_sources = np.asarray(visual_sources, dtype=object).copy()
                    if finite_point(obs.center_cm):
                        observations.append(obs)
                        visual_by_id[int(logical_id)] = replace(
                            obs,
                            keypoints_px=np.asarray(visual_px, dtype=np.float32).copy(),
                            keypoint_conf=np.asarray(visual_conf, dtype=np.float32).copy(),
                            keypoint_sources=np.asarray(visual_sources, dtype=object).copy(),
                        )
                profiler.add("observation_repair", time.perf_counter() - observation_started)

                # v1.23：检测轨迹对应表覆盖全部最终渲染检测，包括正式ID、
                # 聚集匿名ID和仅用于显示的Pxx临时轨迹；Pxx不会进入行为分析。
                detection_index_by_object = {
                    id(det): int(idx) for idx, det in enumerate(renderable_detections)
                }
                render_assignment_rows: List[Tuple[int, base.Detection, Dict[str, Any], str, int]] = []
                for logical_id, detection in real_assigned:
                    info = dict(output_info.get(int(logical_id), {}))
                    render_class = "formal_id" if int(logical_id) >= 0 else "cluster_anonymous"
                    render_assignment_rows.append(
                        (int(logical_id), detection, info, render_class, int(int(logical_id) >= 0))
                    )
                for provisional_id, detection, _label, _cost in provisional_assigned:
                    render_assignment_rows.append(
                        (
                            int(provisional_id), detection,
                            dict(provisional_info.get(int(provisional_id), {})),
                            "provisional", 0,
                        )
                    )
                render_assignment_rows.sort(
                    key=lambda row: detection_index_by_object.get(id(row[1]), 10**9)
                )

                for logical_id, detection, info, render_class, behavior_eligible in render_assignment_rows:
                    det_conf = np.asarray(detection.keypoint_conf, dtype=np.float64).reshape(-1)
                    bbox = np.asarray(detection.bbox_xyxy, dtype=np.float64).reshape(-1)
                    det_map_writer.writerow({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "detection_index": detection_index_by_object.get(id(detection), -1),
                        "logical_id": int(logical_id),
                        "display_label": str(info.get("label", f"ID {int(logical_id)}")),
                        "track_state": str(info.get("state", "tracked")),
                        "detection_source": str(getattr(detection, "detection_source", "global")),
                        "appearance_mode": str(getattr(detection, "appearance_mode", "")),
                        "box_conf": float(detection.box_conf),
                        "pose_valid_keypoints": int(np.sum(np.isfinite(det_conf) & (det_conf >= 0.10))),
                        "center_x_px": float(detection.center_px[0]),
                        "center_y_px": float(detection.center_px[1]),
                        "bbox_x1": float(bbox[0]) if bbox.size >= 4 else "",
                        "bbox_y1": float(bbox[1]) if bbox.size >= 4 else "",
                        "bbox_x2": float(bbox[2]) if bbox.size >= 4 else "",
                        "bbox_y2": float(bbox[3]) if bbox.size >= 4 else "",
                        "assignment_cost": (
                            float(info["cost"])
                            if np.isfinite(float(info.get("cost", float("nan")))) else ""
                        ),
                        "match_method": str(info.get("method", "")),
                        "pose_recovery_score": (
                            float(getattr(detection, "pose_recovery_score", float("nan")))
                            if np.isfinite(float(getattr(detection, "pose_recovery_score", float("nan")))) else ""
                        ),
                        "pose_recovery_reason": str(getattr(detection, "pose_recovery_reason", "")),
                        "keypoint_sources": ",".join(
                            str(x) for x in np.asarray(
                                getattr(detection, "keypoint_sources", None)
                                if getattr(detection, "keypoint_sources", None) is not None
                                else np.full(len(KEYPOINT_NAMES), "RAW", dtype=object),
                                dtype=object,
                            ).reshape(-1)
                        ),
                        "mask_quality": float(getattr(detection, "mask_quality", 0.0)),
                        "mask_reliable": int(bool(getattr(detection, "mask_reliable", False))),
                        "mask_source": str(getattr(detection, "mask_source", "none")),
                        "mask_area_ratio": (
                            float(getattr(detection, "mask_area_ratio", float("nan")))
                            if np.isfinite(float(getattr(detection, "mask_area_ratio", float("nan")))) else ""
                        ),
                        "candidate_identity_set": str(info.get("candidate_ids", "")),
                        "cluster_reid_id": int(info.get("cluster_id", -1)),
                        "render_class": render_class,
                        "is_behavior_eligible": int(behavior_eligible),
                    })

                # v1.29: 将正式ID的短时预测保持单独写入轨迹表，便于审计；
                # is_behavior_eligible固定为0，确保预测框不会污染追逐/攻击几何。
                for logical_id, detection, info in predicted_hold_assigned:
                    det_conf = np.asarray(detection.keypoint_conf, dtype=np.float64).reshape(-1)
                    bbox = np.asarray(detection.bbox_xyxy, dtype=np.float64).reshape(-1)
                    det_map_writer.writerow({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "detection_index": -1,
                        "logical_id": int(logical_id),
                        "display_label": str(info.get("label", f"ID {int(logical_id)}")),
                        "track_state": "predicted_hold",
                        "detection_source": "predicted_hold",
                        "appearance_mode": "predicted_hold",
                        "box_conf": float(detection.box_conf),
                        "pose_valid_keypoints": int(np.sum(np.isfinite(det_conf) & (det_conf >= 0.10))),
                        "center_x_px": float(detection.center_px[0]),
                        "center_y_px": float(detection.center_px[1]),
                        "bbox_x1": float(bbox[0]) if bbox.size >= 4 else "",
                        "bbox_y1": float(bbox[1]) if bbox.size >= 4 else "",
                        "bbox_x2": float(bbox[2]) if bbox.size >= 4 else "",
                        "bbox_y2": float(bbox[3]) if bbox.size >= 4 else "",
                        "assignment_cost": "",
                        "match_method": str(info.get("method", "keypoint_render_hold")),
                        "pose_recovery_score": "",
                        "pose_recovery_reason": "short_identity_render_hold",
                        "keypoint_sources": ",".join(["PREDICTED"] * len(KEYPOINT_NAMES)),
                        "mask_quality": 0.0,
                        "mask_reliable": 0,
                        "mask_source": "none",
                        "mask_area_ratio": "",
                        "candidate_identity_set": "",
                        "cluster_reid_id": -1,
                        "render_class": "predicted_hold",
                        "is_behavior_eligible": 0,
                    })

                observations = sorted(observations, key=lambda x: x.logical_id)
                for obs in observations:
                    history.add(obs)
                if stage3_cache is not None:
                    # Stage 3 is the boundary: stable IDs, repaired keypoints,
                    # scale and occlusion evidence are persisted before any
                    # pair-level behavior calculation is allowed to run.
                    stage3_cache.add(
                        frame_idx,
                        observations,
                        cluster_context,
                        transformer.mode,
                        transformer.current_cm_per_pixel,
                    )

                raw_visual_min_conf = float(config.get("visualization", {}).get("raw_min_confidence", 0.05))
                render_raw = bool(config.get("visualization", {}).get("render_raw_keypoints", True))
                if render_raw:
                    assigned_by_id = {int(logical_id): det for logical_id, det in real_assigned}
                    visual_observations: List[base.MouseObservation] = []
                    for obs in observations:
                        det = assigned_by_id.get(int(obs.logical_id))
                        if det is None:
                            visual_observations.append(obs)
                            continue
                        raw_points = det.keypoints_px.astype(np.float32, copy=True)
                        raw_conf = det.keypoint_conf.astype(np.float32, copy=True)
                        invalid = (~np.isfinite(raw_conf)) | (raw_conf < raw_visual_min_conf)
                        raw_points[invalid] = np.nan
                        visual_observations.append(replace(
                            obs,
                            keypoints_px=raw_points,
                            keypoint_conf=raw_conf,
                            bbox_xyxy=det.bbox_xyxy.astype(np.float32, copy=True),
                            box_conf=float(det.box_conf),
                        ))
                elif pose_recovery_enabled:
                    visual_observations = [
                        visual_by_id.get(int(obs.logical_id), obs) for obs in observations
                    ]
                else:
                    visual_observations = list(observations)

                # v1.23：未确认检测只进入可视化，不进入history、行为分析或正式ID记忆。
                # 即使关键点质量不足，也会通过bbox回退显示，确保YOLO已检出的鼠不消失。
                for provisional_id, detection, provisional_label, _provisional_cost in provisional_assigned:
                    raw_points = np.asarray(detection.keypoints_px, dtype=np.float32).copy()
                    raw_conf = np.asarray(detection.keypoint_conf, dtype=np.float32).copy()
                    invalid = (~np.isfinite(raw_conf)) | (raw_conf < raw_visual_min_conf)
                    raw_points[invalid] = np.nan
                    provisional_obs = build_observation(
                        frame=frame_idx,
                        fps=fps,
                        logical_id=int(provisional_id),
                        detection=detection,
                        smoothed_keypoints_px=raw_points,
                        effective_conf=raw_conf,
                        transformer=transformer,
                        previous=None,
                        track_state="provisional",
                        display_label=str(provisional_label),
                    )
                    provisional_obs.keypoint_sources = np.full(
                        len(KEYPOINT_NAMES), pose_recovery.SOURCE_RAW, dtype=object
                    )
                    visual_observations.append(provisional_obs)

                # 预测保持仅用于覆盖短时标签空窗；不加入observations/history，
                # 因而不会触发行为事件或改变正式轨迹的速度统计。
                for logical_id, detection, info in predicted_hold_assigned:
                    predicted_points = np.asarray(detection.keypoints_px, dtype=np.float32).copy()
                    predicted_conf = np.asarray(detection.keypoint_conf, dtype=np.float32).copy()
                    predicted_obs = build_observation(
                        frame=frame_idx,
                        fps=fps,
                        logical_id=int(logical_id),
                        detection=detection,
                        smoothed_keypoints_px=predicted_points,
                        effective_conf=predicted_conf,
                        transformer=transformer,
                        previous=history.previous(int(logical_id)),
                        track_state="predicted_hold",
                        display_label=str(info.get("label", f"ID {int(logical_id)}")),
                    )
                    predicted_obs.keypoint_sources = np.full(
                        len(KEYPOINT_NAMES), "PREDICTED", dtype=object
                    )
                    visual_observations.append(predicted_obs)

                # 最终渲染必须与去重后的真实检测一一对应。
                expected_visual_count = len(renderable_detections) + len(predicted_hold_assigned)
                if len(visual_observations) != expected_visual_count:
                    raise RuntimeError(
                        f"Frame {frame_idx}: visual_observations={len(visual_observations)} 与 "
                        f"renderable_detections+predicted_holds={expected_visual_count} 不一致。"
                    )

                frame_writer.writerow({
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,
                    "detected_mice_count": len(observations),
                    "detected_logical_ids": ",".join(str(o.logical_id) for o in observations),
                    "raw_track_ids": ",".join(str(o.raw_track_id) for o in observations if o.raw_track_id is not None),
                    "scale_mode": transformer.mode,
                    "cm_per_pixel": transformer.current_cm_per_pixel,
                    "active_occlusion_cluster_count": len(cluster_context.get("regions", [])),
                    "missing_in_cluster_count": int(sum(
                        max(int(r.get("expected_count", 0)) - int(r.get("observed_count", 0)), 0)
                        for r in cluster_context.get("regions", [])
                    )),
                    "local_recovery_detection_count": len(local_recovery_detections),
                    "track_gap_recovery_region_count": int(len(track_gap_regions)),
                    "track_gap_recovery_detection_count": int(sum(
                        str(getattr(det, "detection_source", "")) == "local_recovery_track_gap"
                        for det in local_recovery_detections
                    )),
                    "detector_candidate_count": int(hybrid_meta.get("detector_candidate_count", len(detections))),
                    "bbox_only_count": int(hybrid_meta.get("bbox_only_count", 0)),
                    "white_roi_count": int(hybrid_meta.get("white_roi_count", 0)),
                    "identity_matched_count": int(id_stats.get("matched", 0)),
                    "identity_low_rescued_count": int(id_stats.get("low_rescued", 0)),
                    "identity_lost_recovered_count": int(id_stats.get("lost_recovered", 0)),
                    "identity_new_tentative_count": int(id_stats.get("new_tentative", 0)),
                    "identity_tentative_count": int(id_stats.get("tentative", 0)),
                    "identity_suspicious_count": int(id_stats.get("suspicious", 0)),
                    "identity_lost_count": int(id_stats.get("lost", 0)),
                    "identity_rendered_count": int(id_stats.get("rendered", len(real_assigned))),
                    "adaptive_estimated_mouse_count": int(id_stats.get("adaptive_estimated_count", len(getattr(identity, "tracks", {})))),
                    "adaptive_visible_mouse_count": int(id_stats.get("adaptive_visible_count", len(real_assigned))),
                    "disk_sequence_active_tracks": int(id_stats.get("disk_sequence_active_tracks", 0)),
                    "disk_sequence_mean_reliability": float(id_stats.get("disk_sequence_mean_reliability", 0.0)),
                    "disk_sequence_rejected_updates": int(id_stats.get("disk_sequence_rejected_updates", 0)),
                    "disk_contact_order_veto_count": int(id_stats.get("disk_contact_order_veto_count", 0)),
                    "disk_contact_order_veto_total": int(id_stats.get("disk_contact_order_veto_total", 0)),
                    "contact_order_regret_veto_count": int(id_stats.get("contact_order_regret_veto_count", 0)),
                    "contact_order_regret_veto_total": int(id_stats.get("contact_order_regret_veto_total", 0)),
                    "stage_raw_count": int(hybrid_meta.get("stage_raw_count", -1)),
                    "stage_pose_count": int(hybrid_meta.get("stage_pose_count", -1)),
                    "stage_supplement_count": int(hybrid_meta.get("stage_supplement_count", -1)),
                    "stage_after_fusion_count": int(hybrid_meta.get("stage_after_fusion_count", -1)),
                    "stage_after_mask_count": int(hybrid_meta.get("stage_after_mask_count", -1)),
                    "stage_after_reflection_count": int(hybrid_meta.get("stage_after_reflection_count", -1)),
                    "stage_after_static_count": int(hybrid_meta.get("stage_after_static_count", -1)),
                    "pose_bad_count": int(pose_recovery_stats.bad_count),
                    "pose_roi_attempts": int(pose_recovery_stats.roi_attempts),
                    "pose_roi_accepted": int(pose_recovery_stats.roi_accepted),
                    "pose_roi_rejected": int(pose_recovery_stats.roi_rejected),
                    "temporal_rejected_outliers": int(temporal_rejected_outliers),
                    "temporal_disk_points": int(temporal_disk_points),
                    "temporal_predicted_points": int(temporal_predicted_points),
                    "temporal_template_points": int(temporal_template_points),
                    "mask_attempted_count": int(mask_stats.attempted),
                    "mask_reliable_count": int(mask_stats.reliable),
                    "mask_fallback_count": int(mask_stats.fallback),
                    "mask_failed_count": int(mask_stats.failed),
                    "cluster_reid_active_count": int(cluster_reid.active_count),
                    "cluster_reid_ambiguous_count": int(cluster_reid.ambiguous_count),
                    "cluster_frozen_id_count": int(id_stats.get("cluster_frozen_count", 0)),
                    "cluster_reserved_detection_count": int(id_stats.get("cluster_reserved_detection_count", 0)),
                    "post_dedup_renderable_detection_count": int(len(renderable_detections)),
                    "formal_id_render_count": int(formal_render_count),
                    "cluster_anonymous_render_count": int(cluster_anonymous_render_count),
                    "provisional_render_count": int(len(provisional_assigned)),
                    "final_detection_render_count": int(final_detection_render_count),
                    "unrendered_detection_count": int(unrendered_detection_count),
                    "tracklet_bridge_count": int(bridge_count),
                    "predicted_hold_render_count": int(len(predicted_hold_assigned)),
                })

                behavior_started = time.perf_counter()
                active_raw_records: List[Dict[str, Any]] = []
                existing_pair_keys: set[str] = set()
                # 将遮挡管理器已经计算好的簇物证展开为鼠对字段；只供攻击模块读取。
                cluster_attack_evidence = (
                    {} if tracking_only or stage3_cache is not None
                    else _cluster_attack_evidence_by_pair(cluster_context)
                )
                cluster_attack_pairs = set() if tracking_only or stage3_cache is not None else {
                    tuple(sorted((int(p[0]), int(p[1]))))
                    for p in cluster_context.get("attack_pairs", set())
                }
                # 聚集匿名/身份未决轨迹只用于可视化与重识别证据，不参与个体级
                # 攻击/追逐归属，防止把C01-A猜成某个正式ID。
                behavior_observations = [] if tracking_only or stage3_cache is not None else [
                    o for o in observations
                    if str(getattr(o, "track_state", "tracked")) not in {
                        "cluster_anonymous", "post_split_anonymous", "reid_ambiguous"
                    }
                    and int(o.logical_id) >= 0
                ]
                wall_jump_flags = individual_behavior_gate.update(
                    frame_idx, behavior_observations, width, height
                ) if not tracking_only else {}
                for a, b in combinations(behavior_observations, 2):
                    # Repeated contact uses the same nose-to-whole-body evidence
                    # as the attack gate; a head, trunk, or tail touch can start
                    # a repeated-contact bout, not only a nose-to-tail touch.
                    d_ab = min_point_distance(
                        a.keypoints_cm[KP["nose"]],
                        b.keypoints_cm,
                    )
                    d_ba = min_point_distance(
                        b.keypoints_cm[KP["nose"]],
                        a.keypoints_cm,
                    )
                    weak_contact_threshold = float(config["attack"]["weak"]["contact_distance_cm"])
                    symmetric_weak_contact = bool(
                        (np.isfinite(d_ab) and d_ab < weak_contact_threshold)
                        or (np.isfinite(d_ba) and d_ba < weak_contact_threshold)
                    )
                    repeated = contact_tracker.update(a.logical_id, b.logical_id, frame_idx, symmetric_weak_contact)
                    a_wall_jump = bool(wall_jump_flags.get(int(a.logical_id), False))
                    b_wall_jump = bool(wall_jump_flags.get(int(b.logical_id), False))
                    ab = feature_computer.compute(
                        a, b, history, repeated, a_wall_jump, b_wall_jump
                    )
                    ba = feature_computer.compute(
                        b, a, history, repeated, b_wall_jump, a_wall_jump
                    )
                    selected = choose_direction(ab, ba)

                    pair_tuple = tuple(sorted((int(a.logical_id), int(b.logical_id))))
                    cluster_attack_hint = pair_tuple in cluster_attack_pairs
                    # 当前鼠对若处于攻击提示簇，附带重叠、少检和运动物证供离线攻击门复核。
                    pair_cluster_evidence = cluster_attack_evidence.get(pair_tuple, {})
                    weak_chase = bool(ab.weak_chase or ba.weak_chase)
                    # 普通逐帧攻击仍只用原几何证据；簇提示在离线严格恢复门中单独复核。
                    weak_attack = bool(ab.weak_attack or ba.weak_attack)
                    strong_chase = bool(ab.strong_chase or ba.strong_chase)
                    strong_attack = bool(ab.strong_attack or ba.strong_attack)
                    conf_values = np.concatenate([a.keypoint_conf, b.keypoint_conf])
                    valid_conf = conf_values[np.isfinite(conf_values)]
                    pose_quality = float(np.mean(valid_conf)) if len(valid_conf) else 0.0

                    record = empty_frame_record(frame_idx, fps, transformer)
                    record.update({
                        "pair_key": _pair_key(a.logical_id, b.logical_id),
                        "valid_pair": True,
                        "mouse_a_id": a.logical_id,
                        "mouse_b_id": b.logical_id,
                        "mouse_a_raw_track_id": a.raw_track_id if a.raw_track_id is not None else np.nan,
                        "mouse_b_raw_track_id": b.raw_track_id if b.raw_track_id is not None else np.nan,
                        "mouse_a_speed_cm_s": a.speed_cm_s,
                        "mouse_b_speed_cm_s": b.speed_cm_s,
                        "center_distance_cm": point_distance(a.center_cm, b.center_cm),
                        "head_distance_cm": point_distance(a.head_cm, b.head_cm),
                        "trajectory_correlation": selected.trajectory_correlation,
                        "direction_similarity": selected.direction_similarity,
                        "pursuit_alignment": selected.pursuit_alignment,
                        "target_escape_alignment": selected.target_escape_alignment,
                        "actor_behind_target": selected.actor_behind_target,
                        "selected_actor_wall_jump": selected.actor_wall_jump,
                        "selected_target_wall_jump": selected.target_wall_jump,
                        "pair_wall_jump_excluded": bool(selected.actor_wall_jump or selected.target_wall_jump),
                        "selected_actor_id": selected.actor_id,
                        "selected_target_id": selected.target_id,
                        "selected_nose_body_distance_cm": selected.nose_body_distance_cm,
                        "selected_target_turn_angle_deg": selected.target_turn_angle_deg,
                        "selected_distance_drop_cm": selected.distance_drop_cm,
                        "selected_actor_speed_cm_s": selected.actor_speed_cm_s,
                        "selected_target_speed_cm_s": selected.target_speed_cm_s,
                        "selected_weak_chase_score": selected.weak_chase_score,
                        "selected_strong_chase_score": selected.strong_chase_score,
                        "selected_weak_attack_evidence": int(selected.weak_attack_evidence),
                        "selected_strong_attack_evidence": selected.strong_attack_evidence,
                        "a_to_b_weak_chase": ab.weak_chase,
                        "b_to_a_weak_chase": ba.weak_chase,
                        "a_to_b_strong_chase": ab.strong_chase,
                        "b_to_a_strong_chase": ba.strong_chase,
                        "a_to_b_weak_attack": ab.weak_attack,
                        "b_to_a_weak_attack": ba.weak_attack,
                        "a_to_b_strong_attack": ab.strong_attack,
                        "b_to_a_strong_attack": ba.strong_attack,
                        "weak_contact": bool(ab.weak_contact or ba.weak_contact),
                        "strong_contact": bool(ab.strong_contact or ba.strong_contact),
                        "weak_potential_attack": bool(ab.weak_potential_attack or ba.weak_potential_attack),
                        "strong_potential_attack": bool(ab.strong_potential_attack or ba.strong_potential_attack),
                        "weak_attack_actor_initiation": bool(selected.weak_attack_actor_initiation),
                        "strong_attack_actor_initiation": bool(selected.strong_attack_actor_initiation),
                        "weak_attack_target_reaction": bool(selected.weak_attack_target_reaction),
                        "strong_attack_target_reaction": bool(selected.strong_attack_target_reaction),
                        "repeated_contact_count": repeated,
                        "weak_raw_chase": weak_chase,
                        "weak_raw_attack": weak_attack,
                        "weak_raw_label_id": int(weak_chase) + 2 * int(weak_attack),
                        "strong_raw_chase": strong_chase,
                        "strong_raw_attack": strong_attack,
                        "strong_raw_label_id": int(strong_chase) + 2 * int(strong_attack),
                        "pose_pair_quality": pose_quality,
                        "cluster_attack_hint": cluster_attack_hint,
                        # 字段只进入行为记录，不回写检测框、轨迹ID或遮挡簇状态。
                        **pair_cluster_evidence,
                    })
                    existing_pair_keys.add(str(record["pair_key"]))
                    raw_store.add(record)
                    if int(record["weak_raw_label_id"]) != 0:
                        active_raw_records.append(record)

                # 当打斗导致其中一只鼠没有独立检测时，普通鼠对循环不会生成该对记录。
                # 接触簇状态仍保留遮挡前ID，因此补一条“簇攻击兜底候选”，保证5秒片段被提取。
                for id_a, id_b in sorted(cluster_attack_pairs):
                    pair_key = _pair_key(id_a, id_b)
                    if pair_key in existing_pair_keys:
                        continue
                    # 缺失帧使用同一鼠对的簇证据，不根据ID大小猜测施动者方向。
                    pair_cluster_evidence = cluster_attack_evidence.get(
                        tuple(sorted((int(id_a), int(id_b)))), {}
                    )
                    record = empty_frame_record(frame_idx, fps, transformer)
                    record.update({
                        "pair_key": pair_key,
                        "valid_pair": True,
                        "mouse_a_id": id_a,
                        "mouse_b_id": id_b,
                        # 聚集身份未决时不声称“ID a攻击ID b”；a/b只表示
                        # 进入聚集簇前的候选身份集合，具体施动者/受动者留空。
                        "selected_actor_id": np.nan,
                        "selected_target_id": np.nan,
                        "selected_weak_attack_evidence": 0,
                        "weak_contact": False,
                        "weak_potential_attack": False,
                        "weak_raw_attack": False,
                        "weak_raw_label_id": 0,
                        "pose_pair_quality": 0.0,
                        "cluster_attack_hint": True,
                        # 遮挡帧保留原始物证，最终是否攻击由前后恢复上下文统一确认。
                        **pair_cluster_evidence,
                        "identity_ambiguous": bool(cluster_reid.active_count > 0),
                        "identity_candidate_set": f"{id_a},{id_b}" if cluster_reid.active_count > 0 else "",
                    })
                    raw_store.add(record)
                    active_raw_records.append(record)
                    existing_pair_keys.add(pair_key)

                profiler.add("behavior_io", time.perf_counter() - behavior_started)
                local_added = len(local_recovery_detections)
                if occlusion_manager.debug_rows:
                    cluster_writer.writerows(
                        {**dict(row), "local_recovery_added": local_added}
                        for row in occlusion_manager.debug_rows
                    )
                occlusion_manager.debug_rows.clear()
                if cluster_reid.debug_rows:
                    reid_writer.writerows(dict(row) for row in cluster_reid.debug_rows)
                cluster_reid.debug_rows.clear()

                if identity.debug_records:
                    debug_writer.writerows(asdict(debug) for debug in identity.debug_records)
                identity.debug_records.clear()

                if checkpoint_video_writer is not None:
                    checkpoint_video_writer.submit(
                        frame,
                        visual_observations,
                        active_raw_records,
                        frame_idx,
                        max_mice,
                        config.get("visualization", {}),
                    )
                elif annotated_writer is not None:
                    if annotated_async_writer is not None:
                        annotated_async_writer.submit(
                            frame,
                            visual_observations,
                            active_raw_records,
                            frame_idx,
                            max_mice,
                            config.get("visualization", {}),
                        )
                    else:
                        render_started = time.perf_counter()
                        _draw_online_tracking_frame(
                            frame, visual_observations, active_raw_records, frame_idx, max_mice,
                            config.get("visualization", {}),
                        )
                        profiler.add("render", time.perf_counter() - render_started)
                        encode_started = time.perf_counter()
                        annotated_writer.write(frame)
                        profiler.add("encode", time.perf_counter() - encode_started)

                profiler.frame_done(frame_idx)
                frame_idx += 1
                pbar.update(1)

                if checkpoint_manager.due(frame_idx):
                    commit_inference_checkpoint(inference_complete=False)
                    if prefetched_result_stream is not None:
                        prefetched_result_stream.release_checkpoint_barrier()

                # 释放本帧引用；流式生成器不会保留整段视频。
                del detections, assigned, candidate_assignments, real_assigned, observations, visual_observations, visual_by_id
                del renderable_detections, unassigned_detections, provisional_assigned, provisional_info, render_assignment_rows
                del predicted_hold_assigned, bridge_assigned
                del active_raw_records, local_recovery_detections, cluster_context, guarded_context, result, frame, pose_recovery_stats, mask_stats
                if frame_idx % int(config.get("memory", {}).get("gc_interval_frames", 60)) == 0:
                    gc.collect()
        commit_inference_checkpoint(inference_complete=True)
        if prefetched_result_stream is not None:
            prefetched_result_stream.release_checkpoint_barrier()
    except BaseException:
        if checkpoint_enabled and frame_idx >= resume_frame and prefetched_result_stream is None:
            try:
                commit_inference_checkpoint(inference_complete=False)
            except Exception:
                logging.exception("异常退出时提交紧急断点失败，将保留上一个完整断点。")
        elif checkpoint_enabled and prefetched_result_stream is not None:
            logging.warning("预取模式异常退出：保留上一个同步断点，未提交的尾部将在续跑时回滚。")
        raise
    finally:
        if prefetched_result_stream is not None:
            prefetched_result_stream.close()
        if stage3_cache is not None:
            # Flush an incomplete chunk so --resume can safely continue from
            # the last committed detector frame without touching other files.
            stage3_cache.close()
        if hasattr(mask_extractor, "close"):
            mask_extractor.close()
        raw_store.finalize()
        frame_handle.close()
        debug_handle.close()
        det_map_handle.close()
        cluster_handle.close()
        mask_handle.close()
        reid_handle.close()
        try:
            if annotated_async_writer is not None:
                annotated_async_writer.close()
        finally:
            if annotated_writer is not None:
                annotated_writer.release()
            if checkpoint_video_writer is not None:
                checkpoint_video_writer.abort_uncommitted_segment()
            if frame_idx > 0:
                profiler.report(frame_idx - 1, force=True)

    if frame_idx == 0:
        raw_store.close()
        raise RuntimeError("视频未读取到任何帧。")

    # 推理结束后先释放主模型和可选局部恢复模型，再进行CPU时序后处理。
    del model
    if detector_model is not None:
        del detector_model
    if recovery_model is not None:
        del recovery_model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if checkpoint_video_writer is not None and annotated_path is not None:
        checkpoint_video_writer.merge(annotated_path, expected_frames=frame_idx)

    if stage1_only:
        if stage3_cache is None:
            raw_store.close()
            raise RuntimeError("阶段一要求performance.behavior_pipeline=staged并生成身份缓存。")
        stage3_cache.mark_complete(frame_idx)
        stage1_manifest = {
            "schema_version": 1,
            "program_version": PROGRAM_VERSION,
            "stage": "stage1_complete",
            "video": str(video_path.resolve()),
            "model": str(model_path.resolve()),
            "video_fingerprint": _checkpoint_file_fingerprint(video_path),
            "model_fingerprint": _checkpoint_file_fingerprint(model_path),
            "stage1_config_sha256": _checkpoint_config_digest(config),
            "fps": float(fps),
            "total_frames": int(frame_idx),
            "width": int(width),
            "height": int(height),
            "max_mice": int(max_mice),
            "stage3_cache": str(stage3_cache.directory.resolve()),
            "stage3_manifest": str((stage3_cache.directory / stage3_cache.MANIFEST_NAME).resolve()),
            "track_table": str((output_dir / "检测轨迹对应表.csv").resolve()),
            "frame_table": str((output_dir / "逐帧检测流缓存.csv").resolve()),
            "adaptive_arena": asdict(adaptive_arena_result) if adaptive_arena_result is not None else None,
            "profiling": profiler.snapshot(),
            "config": to_builtin(config),
        }
        _atomic_json_dump(stage1_manifest, output_dir / "阶段一清单.json")
        _atomic_json_dump(
            {
                **stage1_manifest,
                "architecture": "stage1_batch_yolo_then_identity_cache",
                "behavior_computed": False,
                "rendered_video_count": 0,
            },
            output_dir / "运行元数据.json",
        )
        raw_store.close()
        (output_dir / "成对原始流缓存.sqlite3").unlink(missing_ok=True)
        if checkpoint_enabled:
            checkpoint_manager.mark_complete(processed_frames=frame_idx)
        logging.info(
            "阶段一完成：%d帧；稳定ID缓存和表格已落盘，未生成视频、未计算行为。",
            frame_idx,
        )
        return output_dir

    if stage3_cache is not None:
        stage2_started = time.perf_counter()
        stage2_frames = populate_pair_store_from_stage3_cache(
            cache=stage3_cache,
            raw_store=raw_store,
            fps=fps,
            width=width,
            height=height,
            config=config,
            pair_compute_mode=pair_compute_mode,
            pair_workers=int(performance_cfg.get("pair_workers", 0)),
            profiler=profiler,
        )
        raw_store.finalize()
        logging.info(
            "阶段4-7完成：按缓存处理%d帧，耗时%.3fs；未重新执行YOLO/身份追踪。",
            stage2_frames,
            time.perf_counter() - stage2_started,
        )

    pair_benchmark_result: Optional[Dict[str, Any]] = None
    if bool(performance_cfg.get("pair_benchmark", False)):
        benchmark_started = time.perf_counter()
        pair_benchmark_result = benchmark_pair_backends(
            mouse_count=int(max_mice),
            repeats=int(performance_cfg.get("pair_benchmark_repeats", 200)),
            workers=int(performance_cfg.get("pair_workers", 0)),
        )
        logging.info(
            "鼠对后端基准：Python %.4fs，NumPy %.4fs（加速%.2fx），总耗时%.4fs；校验=%s",
            pair_benchmark_result["python_seconds"],
            pair_benchmark_result["numpy_seconds"],
            pair_benchmark_result["numpy_speedup"],
            time.perf_counter() - benchmark_started,
            pair_benchmark_result["numpy_checksum_match"],
        )

    weak_events, strong_events, negative_events, frame_active_count, frame_top = postprocess_pair_store(
        raw_store, fps, config, output_dir, frame_idx
    )
    # Select the interaction pair from sustained event evidence instead of nearest distance alone.
    pair_cache_path = output_dir / "成对行为标签.csv"
    weak_events = reconcile_detected_event_pairs(
        weak_events,
        pair_cache_path,
        frame_idx,
        fps,
        config,
    )
    # Keep strong candidate reconciliation separate for auditability.
    strong_events = reconcile_detected_event_pairs(
        strong_events,
        pair_cache_path,
        frame_idx,
        fps,
        config,
    )
    # Backfill mounted targets that become independently visible after separation.
    track_map_path = output_dir / "检测轨迹对应表.csv"
    weak_events = reconcile_mount_occlusion_events(
        weak_events,
        track_map_path,
        frame_idx,
        fps,
        config,
    )
    # A strong mount correction is permitted only when the strong detector already fired.
    strong_events = reconcile_mount_occlusion_events(
        strong_events,
        track_map_path,
        frame_idx,
        fps,
        config,
    )
    # Regenerate per-frame render labels from the corrected event list.
    frame_active_count, frame_top = rebuild_frame_event_maps(weak_events)
    mark_event_cleanliness(weak_events, config)
    mark_event_cleanliness(strong_events, config)

    add_clip_boundaries(weak_events, frame_idx, fps, config)
    add_clip_boundaries(strong_events, frame_idx, fps, config)
    add_clip_boundaries(negative_events, frame_idx, fps, config)
    enforce_clip_spacing(weak_events + negative_events, fps, config)
    enforce_clip_spacing(strong_events, fps, config)

    frame_detection_df = pd.read_csv(output_dir / "逐帧检测流缓存.csv", encoding="utf-8-sig")
    frame_summary_df = frame_detection_df.copy()
    frame_summary_df["active_pair_count"] = frame_summary_df["frame"].map(frame_active_count).fillna(0).astype(int)
    frame_summary_df["frame_label_id"] = frame_summary_df["frame"].map(lambda f: int(frame_top.get(int(f), {}).get("weak_final_label_id", 0)))
    frame_summary_df["selected_actor_id"] = frame_summary_df["frame"].map(lambda f: frame_top.get(int(f), {}).get("selected_actor_id", np.nan))
    frame_summary_df["selected_target_id"] = frame_summary_df["frame"].map(lambda f: frame_top.get(int(f), {}).get("selected_target_id", np.nan))
    frame_summary_df["selected_pair_key"] = frame_summary_df["frame"].map(lambda f: frame_top.get(int(f), {}).get("pair_key", ""))
    frame_summary_df["selected_chase_score"] = frame_summary_df["frame"].map(lambda f: frame_top.get(int(f), {}).get("selected_weak_chase_score", 0))
    frame_summary_df["selected_attack_evidence"] = frame_summary_df["frame"].map(lambda f: frame_top.get(int(f), {}).get("selected_weak_attack_evidence", 0))
    frame_summary_df["center_distance_cm"] = frame_summary_df["frame"].map(lambda f: frame_top.get(int(f), {}).get("center_distance_cm", np.nan))

    clip_source_video = video_path
    clip_source_mode = "original"
    requested_clip_source = str(
        config.get("clips", {}).get("source_video", "original")
    ).strip().lower()
    if (
        requested_clip_source == "annotated"
        and annotated_path is not None
        and annotated_path.exists()
    ):
        clip_source_video = annotated_path
        clip_source_mode = "annotated"
    selected_clip_count = 0
    if save_clips:
        selected_clip_events = [event for event in (weak_events + negative_events) if bool(event.get("clip_selected", True))]
        selected_clip_count = len(selected_clip_events)
        base.extract_event_clips(
            clip_source_video,
            selected_clip_events,
            output_dir,
            fps,
            width,
            height,
            filename_stem=video_path.stem,
        )

    frame_summary_df.to_csv(output_dir / "逐帧行为汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weak_events).to_csv(output_dir / "行为事件_弱候选.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strong_events).to_csv(output_dir / "行为事件_强候选.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weak_events + negative_events).to_csv(output_dir / "行为事件_待复核.csv", index=False, encoding="utf-8-sig")
    if (
        not tracking_only
        and bool(config["output"].get("save_behavior_label_video", False))
        and annotated_path is not None
        and annotated_path.exists()
    ):
        save_behavior_label_video(
            annotated_path,
            output_dir / "追踪与行为标签视频.mp4",
            frame_summary_df,
            output_dir / "检测轨迹对应表.csv",
            fps,
            width,
            height,
        )

    weak_video_class = classify_video_four_label(weak_events, "weak", config)
    strong_video_class = classify_video_four_label(strong_events, "strong", config)
    strong_video_class["chase_present"] = bool(
        strong_video_class["chase_present"] and weak_video_class["chase_present"]
    )
    strong_video_class["attack_present"] = bool(
        strong_video_class["attack_present"] and weak_video_class["attack_present"]
    )
    strong_label_id = (
        int(strong_video_class["chase_present"])
        + 2 * int(strong_video_class["attack_present"])
    )
    strong_video_class["video_label_id"] = strong_label_id
    strong_video_class["video_label_en"] = LABELS[strong_label_id][0]
    strong_video_class["video_label_zh"] = LABELS[strong_label_id][1]
    if strong_label_id == 0:
        strong_video_class["positive_event_count"] = 0
    video_class_df = pd.DataFrame([weak_video_class, strong_video_class])
    video_class_df.insert(0, "video_name", video_path.name)
    video_class_df.to_csv(output_dir / "视频四分类结果.csv", index=False, encoding="utf-8-sig")

    evaluation_summary = None
    if manual_annotations is not None:
        manual = load_manual_annotations(manual_annotations, video_path, fps)
        manual.to_csv(output_dir / "人工标注_规范化.csv", index=False, encoding="utf-8-sig")
        if manual.empty:
            logging.warning("人工标注中没有当前视频%s的有效事件。", video_path.name)
        else:
            evaluation_summary, _, _ = evaluate_predictions(
                manual, weak_events, strong_events, fps, frame_idx, config,
                output_dir, video_path, width, height, export_error_clips
            )

    max_detected = int(frame_detection_df["detected_mice_count"].max()) if not frame_detection_df.empty else 0
    profiling_summary = profiler.snapshot()
    metadata = {
        "program_version": PROGRAM_VERSION,
        "architecture": (
            "yolo_first_batch_cache_then_stable_id_behavior"
            if yolo_precompute_cache is not None
            else "streaming_adaptive_count_mask_memory_cluster_delayed_reid_pose_recovery"
        ),
        "video": str(video_path.resolve()),
        "model": str(model_path.resolve()),
        "fps": fps,
        "total_frames": frame_idx,
        "width": width,
        "height": height,
        "max_mice": max_mice,
        "mouse_count_mode": "adaptive",
        "tracking_only": bool(tracking_only),
        "event_clips_enabled": bool(save_clips),
        "event_clip_count": int(selected_clip_count),
        "event_clip_source": clip_source_mode,
        "event_clip_source_video": str(Path(clip_source_video).resolve()),
        "profiling": profiling_summary,
        "behavior_pipeline": behavior_pipeline,
        "pair_compute_mode": pair_compute_mode,
        "pair_workers": int(performance_cfg.get("pair_workers", 0)),
        "cleanup_caches_on_success": bool(
            performance_cfg.get("cleanup_caches_on_success", True)
        ),
        "stage3_cache": str(stage3_cache.directory) if stage3_cache is not None else None,
        "yolo_precompute_cache": (
            str(yolo_precompute_cache.directory) if yolo_precompute_cache is not None else None
        ),
        "yolo_first_pass_enabled": bool(yolo_precompute_cache is not None),
        "pair_benchmark": pair_benchmark_result,
        "video_duration_seconds": float(video_duration_seconds),  # 保存实际时长，便于核对开关是否符合五分钟边界。
        "long_term_memory_threshold_seconds": float(long_term_threshold_seconds),  # 保存本次运行采用的阈值。
        "long_term_memory_enabled": bool(long_term_memory_enabled),  # 保存最终生效状态而不是只保存静态配置。
        "adaptive_estimated_mouse_count": int(getattr(identity, "estimated_mouse_count", len(getattr(identity, "tracks", {})))),
        "adaptive_confirmed_track_count": int(len(getattr(identity, "tracks", {}))),
        "adaptive_pending_candidate_count": int(len(getattr(identity, "pending_candidates", {}))),
        "max_detected_mice_in_one_frame": max_detected,
        "occlusion_cluster_lock_enabled": bool(config.get("occlusion_cluster", {}).get("enabled", True)),
        "local_occlusion_recovery_enabled": bool(config.get("occlusion_recovery", {}).get("enabled", True)),
        "track_gap_recovery_enabled": bool(
            config.get("occlusion_recovery", {}).get("track_gap", {}).get("enabled", True)
        ),
        "disk_sequence_identity_guard": (
            identity.disk_sequence_guard.summary()
            if hasattr(identity, "disk_sequence_guard")
            else {"enabled": False}
        ),
        "disk_contact_order_veto_total": int(
            getattr(identity, "disk_contact_order_veto_count", 0)
        ),
        "contact_order_regret_veto_total": int(
            getattr(identity, "contact_order_regret_veto_count", 0)
        ),
        "disk_sequence_pose_repair_visual_only": bool(
            config.get("pose_recovery", {}).get("temporal", {}).get("disk_sequence", {}).get("enabled", False)
        ),
        "duplicate_suppression_enabled": bool(config.get("duplicate_suppression", {}).get("enabled", True)),
        "instance_mask_memory_enabled": bool(config.get("instance_mask_memory", {}).get("enabled", True)),
        "mask_trigger": mask_trigger.summary(),
        "identity_fast_gate": {
            "candidate_count_last_frame": int(getattr(identity, "last_fast_gate_candidate_count", 0)),
            "total_count_last_frame": int(getattr(identity, "last_fast_gate_total_count", 0)),
            "density_last_frame": float(getattr(identity, "last_fast_gate_density", 1.0)),
            "base_cost_mode_last_frame": str(getattr(identity, "last_base_cost_mode", "n/a")),
            "backend_last_frame": str(getattr(identity, "last_cost_backend_used", "n/a")),
        },
        "video_encoding": {
            "prefer_nvenc": bool(config.get("video_encoding", {}).get("prefer_nvenc", True)),
            "nvenc_runtime_available": bool(nvenc_video_writer.ffmpeg_nvenc_available()),
        },
        "mask_type": "pose_bbox_constrained_pseudo_instance_mask_not_trained_segmentation",
        "cluster_delayed_reid_enabled": bool(config.get("cluster_reid", {}).get("enabled", True)),
        "cluster_reid_unresolved_count": int(cluster_reid.active_count),
        "cluster_reid_ambiguous_count": int(cluster_reid.ambiguous_count),
        "behavior_definition_source": "standard_temporal_behavior_engine_plus_legacy_evidence_providers",
        "behavior_classifier_version": standard_behavior_engine.ENGINE_VERSION,
        "behavior_engine_decision_mode": str(
            config.get("standard_behavior_engine", {}).get("decision_mode", "standard")
        ),
        "keypoints": KEYPOINT_NAMES,
        "coordinate_transform": transformer.metadata(),
        "weak_event_counts": {LABELS[i][1]: sum(int(e["label_id"]) == i for e in weak_events) for i in LABELS},
        "strong_event_counts": {LABELS[i][1]: sum(int(e["label_id"]) == i for e in strong_events) for i in LABELS},
        "weak_video_classification": weak_video_class,
        "strong_video_classification": strong_video_class,
        "clean_weak_event_count": sum(bool(e.get("clean_for_classifier")) for e in weak_events),
        "manual_evaluation_performed": evaluation_summary is not None,
        "config": to_builtin(config),
    }
    with (output_dir / "运行元数据.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    raw_store.close()
    if checkpoint_enabled:
        checkpoint_manager.mark_complete(processed_frames=frame_idx)
    if checkpoint_video_writer is not None:
        try:
            checkpoint_video_writer.cleanup()
        except Exception:
            logging.exception("最终结果已完成，但清理追踪视频断点分段失败；可稍后手动删除分段目录。")
    keep_sqlite = bool(config.get("memory", {}).get("keep_raw_sqlite", False))
    if not keep_sqlite:
        try:
            (output_dir / "成对原始流缓存.sqlite3").unlink(missing_ok=True)
        except Exception:
            pass

    # 只有所有结果、断点完成状态和视频文件都成功落盘后才删除可重建缓存。
    # 若推理中断或抛出异常不会执行到这里，因此缓存仍可用于--resume续跑。
    removed_success_artifacts = cleanup_success_artifacts(output_dir, config)
    if removed_success_artifacts:
        logging.info("成功结果缓存清理完成：%s", ", ".join(removed_success_artifacts))

    logging.info("完成：%s", output_dir)
    return output_dir


def _build_stabilized_render_arrays(
    cache: Stage3ObservationCache,
    total_frames: int,
    max_mice: int,
    config: Mapping[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build bounded numeric render tracks from formal Stage-1 observations.

    Behavior is always computed from actual Stage-1 observations.  These arrays
    are visualization-only: isolated one-frame outliers are removed, short
    gaps are interpolated only between two plausible measured endpoints, and a
    short zero-velocity hold is allowed only at the absolute beginning/end of
    the video.  There is intentionally no free-running motion prediction or
    provisional Pxx rendering.
    """
    render_cfg = dict(config.get("stage2_render", {}))
    frame_count = max(int(total_frames), 0)
    mouse_count = max(int(max_mice), 1)
    bbox = np.full((frame_count, mouse_count, 4), np.nan, dtype=np.float32)
    keypoints = np.full(
        (frame_count, mouse_count, len(KEYPOINT_NAMES), 2),
        np.nan,
        dtype=np.float32,
    )
    confidence = np.zeros(
        (frame_count, mouse_count, len(KEYPOINT_NAMES)),
        dtype=np.float32,
    )
    box_confidence = np.zeros((frame_count, mouse_count), dtype=np.float32)
    actual = np.zeros((frame_count, mouse_count), dtype=bool)
    excluded_states = {
        "provisional",
        "tentative",
        "suspicious",
        "predicted_hold",
        "cluster_anonymous",
        "post_split_anonymous",
        "reid_ambiguous",
    }
    ignored_invalid_ids = 0
    ignored_nonformal_states = 0
    for entry in cache.iter_frames():
        frame = int(entry.get("frame", -1))
        if frame < 0 or frame >= frame_count:
            continue
        for observation in entry.get("observations", []) or []:
            logical_id = int(getattr(observation, "logical_id", -1))
            if logical_id < 0 or logical_id >= mouse_count:
                ignored_invalid_ids += 1
                continue
            state = str(getattr(observation, "track_state", "tracked") or "tracked").lower()
            if state in excluded_states:
                ignored_nonformal_states += 1
                continue
            current_bbox = np.asarray(
                getattr(observation, "bbox_xyxy", []), dtype=np.float32
            ).reshape(-1)
            if (
                current_bbox.shape != (4,)
                or not np.all(np.isfinite(current_bbox))
                or current_bbox[2] <= current_bbox[0]
                or current_bbox[3] <= current_bbox[1]
            ):
                continue
            current_points = np.asarray(
                getattr(observation, "keypoints_px", []), dtype=np.float32
            )
            current_conf = np.asarray(
                getattr(observation, "keypoint_conf", []), dtype=np.float32
            ).reshape(-1)
            bbox[frame, logical_id] = current_bbox
            if current_points.shape == (len(KEYPOINT_NAMES), 2):
                keypoints[frame, logical_id] = current_points
            if current_conf.shape == (len(KEYPOINT_NAMES),):
                confidence[frame, logical_id] = np.nan_to_num(
                    current_conf,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            box_confidence[frame, logical_id] = float(
                getattr(observation, "box_conf", 0.0)
            )
            actual[frame, logical_id] = True

    rejected_outliers = 0
    outlier_threshold_bl = max(
        float(render_cfg.get("single_frame_outlier_body_lengths", 0.80)),
        0.10,
    )
    # Reject only a strict one-frame spike bracketed by the same ID on adjacent
    # frames.  Longer motion is left untouched because it may be real.
    for logical_id in range(mouse_count):
        frames = np.flatnonzero(actual[:, logical_id])
        for position in range(1, len(frames) - 1):
            previous_frame = int(frames[position - 1])
            frame = int(frames[position])
            next_frame = int(frames[position + 1])
            if previous_frame != frame - 1 or next_frame != frame + 1:
                continue
            previous_center = 0.5 * (
                bbox[previous_frame, logical_id, :2]
                + bbox[previous_frame, logical_id, 2:]
            )
            center = 0.5 * (
                bbox[frame, logical_id, :2] + bbox[frame, logical_id, 2:]
            )
            next_center = 0.5 * (
                bbox[next_frame, logical_id, :2]
                + bbox[next_frame, logical_id, 2:]
            )
            expected = 0.5 * (previous_center + next_center)
            previous_size = bbox[previous_frame, logical_id, 2:] - bbox[previous_frame, logical_id, :2]
            next_size = bbox[next_frame, logical_id, 2:] - bbox[next_frame, logical_id, :2]
            body = max(
                float(np.max(previous_size)),
                float(np.max(next_size)),
                8.0,
            )
            endpoint_motion = float(np.linalg.norm(next_center - previous_center)) / body
            spike = float(np.linalg.norm(center - expected)) / body
            if endpoint_motion <= outlier_threshold_bl and spike > outlier_threshold_bl:
                actual[frame, logical_id] = False
                bbox[frame, logical_id] = np.nan
                keypoints[frame, logical_id] = np.nan
                confidence[frame, logical_id] = 0.0
                box_confidence[frame, logical_id] = 0.0
                rejected_outliers += 1

    rendered = actual.copy()
    interpolated = np.zeros_like(actual)
    endpoint_held = np.zeros_like(actual)
    maximum_gap = max(int(render_cfg.get("interpolate_max_gap_frames", 12)), 0)
    maximum_speed_bl = max(
        float(render_cfg.get("interpolate_max_speed_body_lengths_per_frame", 0.65)),
        0.05,
    )
    confidence_decay = float(
        np.clip(render_cfg.get("interpolate_confidence_decay", 0.96), 0.0, 1.0)
    )
    for logical_id in range(mouse_count):
        frames = np.flatnonzero(actual[:, logical_id])
        for previous_frame, next_frame in zip(frames[:-1], frames[1:]):
            previous_frame = int(previous_frame)
            next_frame = int(next_frame)
            missing = next_frame - previous_frame - 1
            if missing <= 0 or missing > maximum_gap:
                continue
            previous_center = 0.5 * (
                bbox[previous_frame, logical_id, :2]
                + bbox[previous_frame, logical_id, 2:]
            )
            next_center = 0.5 * (
                bbox[next_frame, logical_id, :2]
                + bbox[next_frame, logical_id, 2:]
            )
            previous_size = bbox[previous_frame, logical_id, 2:] - bbox[previous_frame, logical_id, :2]
            next_size = bbox[next_frame, logical_id, 2:] - bbox[next_frame, logical_id, :2]
            body = max(
                float(np.max(previous_size)),
                float(np.max(next_size)),
                8.0,
            )
            endpoint_jump_bl = float(np.linalg.norm(next_center - previous_center)) / body
            if endpoint_jump_bl > maximum_speed_bl * float(next_frame - previous_frame):
                continue
            for offset in range(1, missing + 1):
                frame = previous_frame + offset
                alpha = float(offset) / float(next_frame - previous_frame)
                bbox[frame, logical_id] = (
                    (1.0 - alpha) * bbox[previous_frame, logical_id]
                    + alpha * bbox[next_frame, logical_id]
                )
                previous_points = keypoints[previous_frame, logical_id]
                next_points = keypoints[next_frame, logical_id]
                both_valid = np.all(np.isfinite(previous_points), axis=1) & np.all(
                    np.isfinite(next_points), axis=1
                )
                keypoints[frame, logical_id, both_valid] = (
                    (1.0 - alpha) * previous_points[both_valid]
                    + alpha * next_points[both_valid]
                )
                confidence[frame, logical_id, both_valid] = (
                    np.minimum(
                        confidence[previous_frame, logical_id, both_valid],
                        confidence[next_frame, logical_id, both_valid],
                    )
                    * (confidence_decay ** min(offset, missing + 1 - offset))
                )
                box_confidence[frame, logical_id] = min(
                    float(box_confidence[previous_frame, logical_id]),
                    float(box_confidence[next_frame, logical_id]),
                ) * (confidence_decay ** min(offset, missing + 1 - offset))
                rendered[frame, logical_id] = True
                interpolated[frame, logical_id] = True

    # A cropped validation clip can end in the middle of a real occlusion.  In
    # that case ordinary interpolation cannot run because there is no measured
    # endpoint beyond the video boundary.  Copying the nearest measured pose
    # for a small, configured number of boundary frames avoids visible ID
    # flashing without extrapolating velocity/acceleration.  These observations
    # remain visualization-only and are never added to the behavior cache.
    endpoint_hold_max_frames = max(
        int(render_cfg.get("endpoint_hold_max_frames", 0)),
        0,
    )
    endpoint_hold_confidence_decay = float(
        np.clip(render_cfg.get("endpoint_hold_confidence_decay", 0.92), 0.0, 1.0)
    )
    if endpoint_hold_max_frames > 0:
        for logical_id in range(mouse_count):
            frames = np.flatnonzero(actual[:, logical_id])
            if len(frames) == 0:
                continue
            first_frame = int(frames[0])
            leading_start = max(0, first_frame - endpoint_hold_max_frames)
            for frame in range(leading_start, first_frame):
                distance = first_frame - frame
                bbox[frame, logical_id] = bbox[first_frame, logical_id]
                keypoints[frame, logical_id] = keypoints[first_frame, logical_id]
                confidence[frame, logical_id] = (
                    confidence[first_frame, logical_id]
                    * (endpoint_hold_confidence_decay ** distance)
                )
                box_confidence[frame, logical_id] = float(
                    box_confidence[first_frame, logical_id]
                ) * (endpoint_hold_confidence_decay ** distance)
                rendered[frame, logical_id] = True
                endpoint_held[frame, logical_id] = True

            last_frame = int(frames[-1])
            trailing_end = min(
                frame_count,
                last_frame + endpoint_hold_max_frames + 1,
            )
            for frame in range(last_frame + 1, trailing_end):
                distance = frame - last_frame
                bbox[frame, logical_id] = bbox[last_frame, logical_id]
                keypoints[frame, logical_id] = keypoints[last_frame, logical_id]
                confidence[frame, logical_id] = (
                    confidence[last_frame, logical_id]
                    * (endpoint_hold_confidence_decay ** distance)
                )
                box_confidence[frame, logical_id] = float(
                    box_confidence[last_frame, logical_id]
                ) * (endpoint_hold_confidence_decay ** distance)
                rendered[frame, logical_id] = True
                endpoint_held[frame, logical_id] = True

    actual_count = int(np.count_nonzero(actual))
    interpolated_count = int(np.count_nonzero(interpolated))
    endpoint_held_count = int(np.count_nonzero(endpoint_held))
    report = {
        "actual_formal_observation_count": actual_count,
        "interpolated_visual_observation_count": interpolated_count,
        "bounded_endpoint_hold_observation_count": endpoint_held_count,
        "rejected_single_frame_outlier_count": int(rejected_outliers),
        "ignored_out_of_range_id_count": int(ignored_invalid_ids),
        "ignored_nonformal_state_count": int(ignored_nonformal_states),
        "rendered_observation_count": int(np.count_nonzero(rendered)),
        "interpolate_max_gap_frames": int(maximum_gap),
        "interpolate_max_speed_body_lengths_per_frame": float(maximum_speed_bl),
        "endpoint_hold_max_frames": int(endpoint_hold_max_frames),
        "endpoint_hold_confidence_decay": float(endpoint_hold_confidence_decay),
        "visual_only_interpolation": True,
        "behavior_uses_interpolation": False,
        "zero_velocity_endpoint_hold_visual_only": True,
        "behavior_uses_endpoint_hold": False,
        "provisional_ids_rendered": False,
        "free_running_predictions_rendered": False,
    }
    return {
        "bbox": bbox,
        "keypoints": keypoints,
        "confidence": confidence,
        "box_confidence": box_confidence,
        "actual": actual,
        "rendered": rendered,
        "interpolated": interpolated,
        "endpoint_held": endpoint_held,
    }, report


def _draw_behavior_summary_overlay(
    frame: np.ndarray,
    summary: Mapping[str, Any],
    frame_boxes: Mapping[int, np.ndarray],
) -> None:
    height, width = frame.shape[:2]
    label_id = int(safe_float(summary.get("frame_label_id"), 0))
    if label_id not in (1, 2, 3):
        return
    actor_value = safe_float(summary.get("selected_actor_id"), float("nan"))
    target_value = safe_float(summary.get("selected_target_id"), float("nan"))
    actor_id = int(actor_value) if np.isfinite(actor_value) else None
    target_id = int(target_value) if np.isfinite(target_value) else None
    behavior = {1: "CHASE", 2: "ATTACK", 3: "CHASE+ATTACK"}[label_id]
    color = {1: (0, 220, 255), 2: (0, 64, 255), 3: (255, 64, 255)}[label_id]
    pair_text = (
        f"ID {actor_id} {behavior} -> ID {target_id}"
        if actor_id is not None and target_id is not None
        else f"{behavior} | IDENTITY AMBIGUOUS"
    )
    panel_width = min(max(420, 13 * len(pair_text)), max(width - 20, 20))
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 54), (0, 0, 0), -1)
    cv2.putText(
        frame,
        pair_text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )
    if actor_id in frame_boxes and target_id in frame_boxes:
        actor_box = frame_boxes[int(actor_id)]
        target_box = frame_boxes[int(target_id)]
        actor_center = (
            int((actor_box[0] + actor_box[2]) * 0.5),
            int((actor_box[1] + actor_box[3]) * 0.5),
        )
        target_center = (
            int((target_box[0] + target_box[2]) * 0.5),
            int((target_box[1] + target_box[3]) * 0.5),
        )
        cv2.arrowedLine(
            frame,
            actor_center,
            target_center,
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )


def save_final_analysis_video_from_stage1(
    video_path: Path,
    output_path: Path,
    cache: Stage3ObservationCache,
    frame_summary_df: pd.DataFrame,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    max_mice: int,
    config: Mapping[str, Any],
    arena_polygon: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    arrays, report = _build_stabilized_render_arrays(
        cache,
        total_frames=total_frames,
        max_mice=max_mice,
        config=config,
    )
    summaries = {
        int(row["frame"]): row for row in frame_summary_df.to_dict("records")
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"阶段二无法打开原始视频：{video_path}")
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    temporary.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"阶段二无法创建最终渲染视频：{temporary}")
    polygon = None
    if arena_polygon is not None:
        polygon_array = np.asarray(arena_polygon, dtype=np.float64)
        if polygon_array.ndim == 2 and polygon_array.shape[0] >= 3 and polygon_array.shape[1] == 2:
            polygon = np.round(polygon_array).astype(np.int32).reshape(-1, 1, 2)
    show_boundary = bool(config.get("stage2_render", {}).get("show_arena_boundary", True))
    frame_idx = 0
    try:
        with tqdm(total=total_frames, desc="阶段二：生成唯一最终渲染视频", unit="frame") as progress:
            while frame_idx < total_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_boxes: Dict[int, np.ndarray] = {}
                for logical_id in np.flatnonzero(arrays["rendered"][frame_idx]):
                    logical_id = int(logical_id)
                    current_bbox = arrays["bbox"][frame_idx, logical_id].astype(np.float64)
                    current_points = arrays["keypoints"][frame_idx, logical_id].astype(np.float64)
                    current_conf = arrays["confidence"][frame_idx, logical_id].astype(np.float64)
                    observation = base.MouseObservation(
                        frame=int(frame_idx),
                        logical_id=logical_id,
                        raw_track_id=None,
                        keypoints_px=current_points,
                        keypoints_cm=np.full_like(current_points, np.nan),
                        keypoint_conf=current_conf,
                        bbox_xyxy=current_bbox,
                        box_conf=float(arrays["box_confidence"][frame_idx, logical_id]),
                        center_cm=np.full(2, np.nan, dtype=np.float64),
                        head_cm=np.full(2, np.nan, dtype=np.float64),
                        rear_cm=np.full(2, np.nan, dtype=np.float64),
                        heading=np.full(2, np.nan, dtype=np.float64),
                        velocity_cm_s=np.zeros(2, dtype=np.float64),
                        speed_cm_s=0.0,
                        acceleration_cm_s2=0.0,
                        angular_speed_deg_s=0.0,
                        nose_speed_cm_s=0.0,
                        body_length_cm=float("nan"),
                        track_state=(
                            "interpolated_short_gap"
                            if bool(arrays["interpolated"][frame_idx, logical_id])
                            else (
                                "bounded_endpoint_hold"
                                if bool(arrays["endpoint_held"][frame_idx, logical_id])
                                else "tracked"
                            )
                        ),
                        display_label=f"ID {logical_id}",
                    )
                    _draw_clean_mouse_overlay(
                        frame,
                        observation,
                        config.get("visualization", {}),
                    )
                    frame_boxes[logical_id] = current_bbox
                if polygon is not None and show_boundary:
                    cv2.polylines(frame, [polygon], True, (60, 220, 60), 2, cv2.LINE_AA)
                _draw_behavior_summary_overlay(
                    frame,
                    summaries.get(frame_idx, {}),
                    frame_boxes,
                )
                writer.write(frame)
                frame_idx += 1
                progress.update(1)
    finally:
        writer.release()
        capture.release()
    if frame_idx != int(total_frames):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"最终渲染帧数不完整：写入{frame_idx}，阶段一清单{total_frames}。"
        )
    os.replace(temporary, output_path)
    report["video_frame_count"] = int(frame_idx)
    report["output_video"] = str(output_path.resolve())
    return report


def run_stage2_from_stage1(
    output_dir: Path,
    config: MutableMapping[str, Any],
    manual_annotations: Optional[Path] = None,
    save_clips: bool = True,
    export_error_clips: bool = True,
) -> Path:
    """Run behavior/classification/clips/rendering without loading YOLO."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "阶段一清单.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"阶段二找不到阶段一清单：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        stage1 = json.load(handle)
    if stage1.get("stage") != "stage1_complete":
        raise ValueError(f"阶段一清单未完成：{manifest_path}")
    video_path = Path(str(stage1["video"]))
    if not video_path.exists():
        raise FileNotFoundError(f"阶段二所需原始视频不存在：{video_path}")
    if stage1.get("video_fingerprint") != _checkpoint_file_fingerprint(video_path):
        raise ValueError("原始视频已变化，拒绝把阶段一身份缓存用于不同视频。")
    fps = float(stage1["fps"])
    total_frames = int(stage1["total_frames"])
    width = int(stage1["width"])
    height = int(stage1["height"])
    max_mice = int(stage1.get("max_mice", config.get("model", {}).get("max_mice", 20)))
    cache = Stage3ObservationCache.open_existing(output_dir, require_complete=True)
    if int(cache.total_frames or -1) != total_frames:
        raise ValueError(
            f"阶段一清单与身份缓存帧数不一致：{total_frames} vs {cache.total_frames}"
        )

    started = time.perf_counter()
    performance_cfg = dict(config.get("performance", {}))
    profiler = RuntimeProfiler(config)
    raw_store = PairSQLiteStore(
        output_dir / "成对原始流缓存.sqlite3",
        batch_size=int(config.get("memory", {}).get("sqlite_batch_rows", 1500)),
        resume=False,
    )
    try:
        stage2_frames = populate_pair_store_from_stage3_cache(
            cache=cache,
            raw_store=raw_store,
            fps=fps,
            width=width,
            height=height,
            config=config,
            pair_compute_mode=str(performance_cfg.get("pair_compute_mode", "multiprocess")),
            pair_workers=int(performance_cfg.get("pair_workers", 0)),
            profiler=profiler,
        )
        raw_store.finalize()
        if stage2_frames != total_frames:
            raise RuntimeError(
                f"阶段二行为重放帧数不一致：{stage2_frames} vs {total_frames}"
            )
        weak_events, strong_events, negative_events, frame_active_count, frame_top = postprocess_pair_store(
            raw_store,
            fps,
            config,
            output_dir,
            total_frames,
        )
        pair_cache_path = output_dir / "成对行为标签.csv"
        weak_events = reconcile_detected_event_pairs(
            weak_events, pair_cache_path, total_frames, fps, config
        )
        strong_events = reconcile_detected_event_pairs(
            strong_events, pair_cache_path, total_frames, fps, config
        )
        track_map_path = output_dir / "检测轨迹对应表.csv"
        weak_events = reconcile_mount_occlusion_events(
            weak_events, track_map_path, total_frames, fps, config
        )
        strong_events = reconcile_mount_occlusion_events(
            strong_events, track_map_path, total_frames, fps, config
        )
        frame_active_count, frame_top = rebuild_frame_event_maps(weak_events)
        mark_event_cleanliness(weak_events, config)
        mark_event_cleanliness(strong_events, config)
        add_clip_boundaries(weak_events, total_frames, fps, config)
        add_clip_boundaries(strong_events, total_frames, fps, config)
        add_clip_boundaries(negative_events, total_frames, fps, config)
        enforce_clip_spacing(weak_events + negative_events, fps, config)
        enforce_clip_spacing(strong_events, fps, config)

        frame_detection_df = pd.read_csv(
            output_dir / "逐帧检测流缓存.csv",
            encoding="utf-8-sig",
        )
        frame_summary_df = frame_detection_df.copy()
        frame_summary_df["active_pair_count"] = (
            frame_summary_df["frame"].map(frame_active_count).fillna(0).astype(int)
        )
        frame_summary_df["frame_label_id"] = frame_summary_df["frame"].map(
            lambda frame: int(frame_top.get(int(frame), {}).get("weak_final_label_id", 0))
        )
        frame_summary_df["selected_actor_id"] = frame_summary_df["frame"].map(
            lambda frame: frame_top.get(int(frame), {}).get("selected_actor_id", np.nan)
        )
        frame_summary_df["selected_target_id"] = frame_summary_df["frame"].map(
            lambda frame: frame_top.get(int(frame), {}).get("selected_target_id", np.nan)
        )
        frame_summary_df["selected_pair_key"] = frame_summary_df["frame"].map(
            lambda frame: frame_top.get(int(frame), {}).get("pair_key", "")
        )
        frame_summary_df["selected_chase_score"] = frame_summary_df["frame"].map(
            lambda frame: frame_top.get(int(frame), {}).get("selected_weak_chase_score", 0)
        )
        frame_summary_df["selected_attack_evidence"] = frame_summary_df["frame"].map(
            lambda frame: frame_top.get(int(frame), {}).get("selected_weak_attack_evidence", 0)
        )
        frame_summary_df["center_distance_cm"] = frame_summary_df["frame"].map(
            lambda frame: frame_top.get(int(frame), {}).get("center_distance_cm", np.nan)
        )
        frame_summary_df.to_csv(
            output_dir / "逐帧行为汇总.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(weak_events).to_csv(
            output_dir / "行为事件_弱候选.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(strong_events).to_csv(
            output_dir / "行为事件_强候选.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(weak_events + negative_events).to_csv(
            output_dir / "行为事件_待复核.csv", index=False, encoding="utf-8-sig"
        )

        # Remove only the two legacy root-level render products.  Event clips are
        # regenerated below from the single final video.
        for legacy_name in ("追踪标注视频_仅框与骨架.mp4", "追踪与行为标签视频.mp4"):
            (output_dir / legacy_name).unlink(missing_ok=True)
        final_video_path = output_dir / "最终分析渲染视频.mp4"
        adaptive_payload = stage1.get("adaptive_arena") or {}
        arena_polygon = adaptive_payload.get("polygon") if isinstance(adaptive_payload, Mapping) else None
        render_report = save_final_analysis_video_from_stage1(
            video_path=video_path,
            output_path=final_video_path,
            cache=cache,
            frame_summary_df=frame_summary_df,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            max_mice=max_mice,
            config=config,
            arena_polygon=arena_polygon,
        )
        _atomic_json_dump(render_report, output_dir / "渲染稳定性报告.json")

        selected_clip_events = [
            event
            for event in weak_events + negative_events
            if bool(event.get("clip_selected", True))
        ]
        if save_clips:
            clips_directory = output_dir / "事件片段"
            if clips_directory.exists() and bool(
                config.get("clips", {}).get("replace_on_stage2", True)
            ):
                shutil.rmtree(clips_directory)
            base.extract_event_clips(
                final_video_path,
                selected_clip_events,
                output_dir,
                fps,
                width,
                height,
                filename_stem=video_path.stem,
            )

        weak_video_class = classify_video_four_label(weak_events, "weak", config)
        strong_video_class = classify_video_four_label(strong_events, "strong", config)
        strong_video_class["chase_present"] = bool(
            strong_video_class["chase_present"] and weak_video_class["chase_present"]
        )
        strong_video_class["attack_present"] = bool(
            strong_video_class["attack_present"] and weak_video_class["attack_present"]
        )
        strong_label_id = int(strong_video_class["chase_present"]) + 2 * int(
            strong_video_class["attack_present"]
        )
        strong_video_class["video_label_id"] = strong_label_id
        strong_video_class["video_label_en"] = LABELS[strong_label_id][0]
        strong_video_class["video_label_zh"] = LABELS[strong_label_id][1]
        if strong_label_id == 0:
            strong_video_class["positive_event_count"] = 0
        video_class_df = pd.DataFrame([weak_video_class, strong_video_class])
        video_class_df.insert(0, "video_name", video_path.name)
        video_class_df.to_csv(
            output_dir / "视频四分类结果.csv", index=False, encoding="utf-8-sig"
        )

        evaluation_summary = None
        if manual_annotations is not None:
            manual = load_manual_annotations(manual_annotations, video_path, fps)
            manual.to_csv(
                output_dir / "人工标注_规范化.csv", index=False, encoding="utf-8-sig"
            )
            if not manual.empty:
                evaluation_summary, _, _ = evaluate_predictions(
                    manual,
                    weak_events,
                    strong_events,
                    fps,
                    total_frames,
                    config,
                    output_dir,
                    video_path,
                    width,
                    height,
                    export_error_clips,
                )

        elapsed = time.perf_counter() - started
        metadata_path = output_dir / "运行元数据.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                metadata = dict(loaded)
        metadata.update({
            "program_version": PROGRAM_VERSION,
            "architecture": "stage1_identity_cache_then_stage2_behavior_single_render",
            "stage": "stage2_complete",
            "behavior_computed": True,
            "stage2_no_model_loaded": True,
            "stage2_frames": int(stage2_frames),
            "stage2_elapsed_seconds": float(elapsed),
            "rendered_video_count": 1,
            "final_rendered_video": str(final_video_path.resolve()),
            "event_clips_enabled": bool(save_clips),
            "event_clip_count": int(len(selected_clip_events) if save_clips else 0),
            "event_clip_source": "final_single_render",
            "event_clip_source_video": str(final_video_path.resolve()),
            "weak_event_counts": {
                LABELS[index][1]: sum(int(event["label_id"]) == index for event in weak_events)
                for index in LABELS
            },
            "strong_event_counts": {
                LABELS[index][1]: sum(int(event["label_id"]) == index for event in strong_events)
                for index in LABELS
            },
            "weak_video_classification": weak_video_class,
            "strong_video_classification": strong_video_class,
            "manual_evaluation_performed": evaluation_summary is not None,
            "render_stabilization": render_report,
            "stage2_config": to_builtin(config),
        })
        _atomic_json_dump(metadata, metadata_path)
        _atomic_json_dump(
            {
                "schema_version": 1,
                "program_version": PROGRAM_VERSION,
                "stage": "stage2_complete",
                "source_stage1_manifest": str(manifest_path.resolve()),
                "stage2_no_model_loaded": True,
                "frames": int(stage2_frames),
                "elapsed_seconds": float(elapsed),
                "final_video": str(final_video_path.resolve()),
                "root_rendered_video_count": 1,
                "event_clip_count": int(len(selected_clip_events) if save_clips else 0),
            },
            output_dir / "阶段二清单.json",
        )
        logging.info(
            "阶段二完成：%d帧，耗时%.3fs；未加载YOLO，只生成一个最终渲染视频：%s",
            stage2_frames,
            elapsed,
            final_video_path,
        )
    finally:
        raw_store.close()
    if not bool(config.get("memory", {}).get("keep_raw_sqlite", False)):
        (output_dir / "成对原始流缓存.sqlite3").unlink(missing_ok=True)
    return output_dir


def _lightweight_cache_is_ready(
    cache_dir: Path,
    video_path: Path,
    model_path: Path,
    total_frames: int,
) -> bool:
    """Check whether a completed YOLO cache is safe to reuse for light mode."""
    status_path = Path(cache_dir) / "yolo_results_status.json"
    if not status_path.exists():
        return False
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        if str(status.get("status", "")).strip().lower() != "complete":
            return False
        if int(status.get("next_frame", -1)) != int(total_frames):
            return False
        if int(status.get("total_frames", -1)) != int(total_frames):
            return False
        fingerprint = status.get("fingerprint", {})
        if isinstance(fingerprint, Mapping):
            for key, expected_path in (("video", video_path), ("model", model_path)):
                observed = fingerprint.get(key, {})
                if not isinstance(observed, Mapping) or not observed.get("path"):
                    continue
                observed_path = os.path.normcase(str(Path(str(observed["path"])).resolve()))
                required_path = os.path.normcase(str(Path(expected_path).resolve()))
                if observed_path != required_path:
                    return False
        return any(Path(cache_dir).glob("yolo_results.*.*.pkl.gz"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_lightweight_behavior_inference(
    video_path: Path,
    model_path: Path,
    output_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    *,
    allow_precompute: bool = True,
    no_clips: bool = False,
    no_render: bool = False,
) -> Path:
    """Run only the bounded light behavior path for one video.

    The full identity, occlusion, mask/ReID, ROI recovery, staged behavior,
    annotated rendering, and error-clip code remains available in the normal
    pipeline.  This entry point simply does not construct or call those
    components while ``lightweight_behavior_inference.enabled`` is true.
    """
    import lightweight_behavior_inference as lightweight_behavior

    video_path = Path(video_path).resolve()
    model_path = Path(model_path).resolve()
    output_dir = ensure_dir(Path(output_root) / video_path.stem)
    light_cfg = dict(config.get("lightweight_behavior_inference", {}))

    meta_cap = cv2.VideoCapture(str(video_path))
    if not meta_cap.isOpened():
        raise RuntimeError(f"轻量行为推理无法打开视频：{video_path}")
    fps = float(meta_cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(meta_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(meta_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(meta_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_cap.release()
    if fps <= 0.0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"轻量行为推理视频元数据异常：fps={fps}, frames={total_frames}, size={width}x{height}"
        )

    cache_value = str(light_cfg.get("yolo_cache_dir", "") or "").strip()
    if cache_value:
        try:
            cache_value = cache_value.format(
                video_stem=video_path.stem,
                video_name=video_path.name,
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "lightweight_behavior_inference.yolo_cache_dir只支持{video_stem}和{video_name}占位符"
            ) from exc
        cache_dir = Path(cache_value)
        if not cache_dir.is_absolute():
            cache_dir = output_dir / cache_dir
    else:
        cache_dir = output_dir / "yolo_precompute"
    cache_dir = cache_dir.resolve()
    default_cache_dir = (output_dir / "yolo_precompute").resolve()
    cache_ready = _lightweight_cache_is_ready(
        cache_dir,
        video_path,
        model_path,
        total_frames,
    )
    if not cache_ready and cache_dir != default_cache_dir:
        logging.warning(
            "自定义轻量YOLO缓存未完成，将回退到当前视频输出目录：%s",
            default_cache_dir,
        )
        cache_dir = default_cache_dir
        cache_ready = _lightweight_cache_is_ready(
            cache_dir,
            video_path,
            model_path,
            total_frames,
        )

    if not cache_ready:
        if not allow_precompute:
            raise FileNotFoundError(
                "轻量行为阶段找不到完整YOLO缓存；请先用--stage all生成缓存，"
                f"或检查：{cache_dir}"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"轻量行为推理需要模型，但模型不存在：{model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "轻量行为推理生成YOLO缓存需要安装ultralytics。"
            ) from exc

        # The cache pass keeps only the Pose channel.  The detector supplement
        # is deliberately bypassed for this mode; the original config is not
        # mutated and the full detector path remains available when light mode
        # is disabled.
        runtime_config = copy.deepcopy(dict(config))
        runtime_performance = runtime_config.setdefault("performance", {})
        runtime_yolo_first = runtime_performance.setdefault("yolo_first_pass", {})
        runtime_yolo_first["enabled"] = True
        runtime_detector = copy.deepcopy(dict(runtime_config.get("detector_first", {})))
        runtime_detector.update(
            {
                "enabled": True,
                "pose_mode": "full_frame",
                "use_detector_model": False,
            }
        )
        runtime_config["detector_first"] = runtime_detector
        runtime_profiler = RuntimeProfiler(runtime_config)
        model_cfg = dict(runtime_config.get("model", {}))
        pose_model = YOLO(str(model_path))
        generated_cache = run_yolo_first_pass(
            video_path=video_path,
            model_path=model_path,
            pose_model=pose_model,
            detector_model=None,
            detector_cfg=runtime_detector,
            performance_cfg=runtime_performance,
            output_dir=output_dir,
            config=runtime_config,
            device=model_cfg.get("device", 0),
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            expected_keypoints=len(KEYPOINT_NAMES),
            resume_requested=False,
            profiler=runtime_profiler,
        )
        if generated_cache is None:
            raise RuntimeError("轻量行为推理未能生成YOLO缓存。")
        cache_dir = Path(generated_cache.directory).resolve()

    expected_mice = max(
        int(light_cfg.get("expected_mice", config.get("model", {}).get("max_mice", 20))),
        2,
    )
    sample_stride = max(int(light_cfg.get("sample_stride", 1)), 1)
    max_frames_value = light_cfg.get("max_frames")
    max_frames = None if max_frames_value in (None, "", 0) else max(int(max_frames_value), 1)
    analysis_dir = lightweight_behavior.analyze(
        video_path=video_path,
        cache_dir=cache_dir,
        config_path=Path(config_path).resolve(),
        output_dir=output_dir,
        expected_mice=expected_mice,
        max_frames=max_frames,
        sample_stride=sample_stride,
        fps_override=fps,
    )
    events_path = analysis_dir / "lightweight_behavior_events.csv"

    clips_enabled = bool(light_cfg.get("extract_four_class_clips", False)) and not bool(no_clips)
    clip_dir: Optional[Path] = None
    if clips_enabled:
        configured_clip_dir = Path(str(light_cfg.get("clips_output_dir", "四类视频")))
        clip_dir = configured_clip_dir if configured_clip_dir.is_absolute() else analysis_dir / configured_clip_dir
        lightweight_behavior.extract_four_class_clips(
            video_path=video_path,
            events_path=events_path,
            output_dir=clip_dir,
            expected_level=str(light_cfg.get("clip_level", "strong")),
            clip_seconds=max(float(light_cfg.get("clip_seconds", 5.0)), 0.1),
            min_start_interval_seconds=max(
                float(light_cfg.get("clip_min_start_interval_seconds", 5.0)),
                0.0,
            ),
            max_clips_per_class=max(int(light_cfg.get("max_clips_per_class", 200)), 1),
        )

    render_enabled = bool(light_cfg.get("render_video", False)) and not bool(no_render)
    rendered_video: Optional[Path] = None
    if render_enabled:
        configured_render = Path(
            str(light_cfg.get("render_output", "轻量行为推理_渲染.mp4"))
        )
        rendered_video = (
            configured_render
            if configured_render.is_absolute()
            else analysis_dir / configured_render
        )
        lightweight_behavior.render_behavior_video(
            video_path=video_path,
            cache_dir=cache_dir,
            events_path=events_path,
            output_path=rendered_video,
            expected_mice=expected_mice,
            max_frames=max_frames,
        )

    analysis_metadata_path = analysis_dir / "lightweight_analysis_metadata.json"
    analysis_metadata: Dict[str, Any] = {}
    if analysis_metadata_path.exists():
        try:
            with analysis_metadata_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                analysis_metadata = dict(loaded)
        except (OSError, ValueError, TypeError):
            logging.warning("读取轻量行为元数据失败，将重新写入：%s", analysis_metadata_path)
    analysis_metadata.update(
        {
            "lightweight_behavior_enabled": True,
            "full_pipeline_not_run": True,
            "yolo_cache_reused": bool(cache_ready),
            "four_class_clips_enabled": bool(clips_enabled),
            "four_class_clips_output": str(clip_dir.resolve()) if clip_dir else None,
            "render_video_enabled": bool(render_enabled),
            "rendered_video": str(rendered_video.resolve()) if rendered_video else None,
        }
    )
    _atomic_json_dump(analysis_metadata, analysis_metadata_path)

    # Keep the common metadata filename so the existing CPU/GPU resource
    # reporter can merge timing data without touching the legacy pipeline.
    run_metadata_path = analysis_dir / "运行元数据.json"
    run_metadata: Dict[str, Any] = {}
    if run_metadata_path.exists():
        try:
            with run_metadata_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                run_metadata = dict(loaded)
        except (OSError, ValueError, TypeError):
            logging.warning("读取既有运行元数据失败，将保留可恢复的轻量结果：%s", run_metadata_path)
    run_metadata.update(
        {
            "program_version": PROGRAM_VERSION,
            "architecture": "lightweight_behavior_inference",
            "stage": "lightweight_complete",
            "source_video": str(video_path),
            "yolo_cache": str(cache_dir),
            "behavior_computed": True,
            "full_pipeline_not_run": True,
            "rendered_video_count": 1 if rendered_video else 0,
            "final_rendered_video": str(rendered_video.resolve()) if rendered_video else None,
            "four_class_clips_enabled": bool(clips_enabled),
            "four_class_clips_output": str(clip_dir.resolve()) if clip_dir else None,
            "lightweight_config": to_builtin(light_cfg),
        }
    )
    _atomic_json_dump(run_metadata, run_metadata_path)
    logging.info(
        "轻量行为推理完成：%s | YOLO缓存%s复用 | 四类片段=%s | 渲染=%s",
        video_path.name,
        "已" if cache_ready else "未",
        "开" if clips_enabled else "关",
        "开" if rendered_video else "关",
    )
    return analysis_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def discover_videos(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式：{path.suffix}")
        return [path]
    if path.is_dir():
        videos = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise FileNotFoundError(f"目录中没有视频：{path}")
        return videos
    raise FileNotFoundError(f"视频路径不存在：{path}")


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("model", "keypoints", "identity", "scale", "features", "chase", "attack", "clips", "evaluation", "output"):
        if key not in config:
            raise ValueError(f"配置缺少字段：{key}")
    if list(config["keypoints"].get("names", [])) != KEYPOINT_NAMES:
        raise ValueError("关键点顺序必须是：" + ", ".join(KEYPOINT_NAMES))
    for parent in ("chase", "attack"):
        for level in ("weak", "strong"):
            if level not in config[parent]:
                raise ValueError(f"配置缺少{parent}.{level}")
    for key in ("instance_mask_memory", "cluster_reid", "pose_recovery", "provisional_render", "wall_jump"):
        if key not in config:
            raise ValueError(f"v1.24配置缺少字段：{key}")
    long_term_policy = dict(config.get("long_term_memory", {}))  # 读取长期记忆策略以审查五分钟阈值是否合法。
    long_term_threshold = float(long_term_policy.get("min_video_duration_seconds", 300.0))  # 未配置时沿用五分钟默认值。
    if not np.isfinite(long_term_threshold) or long_term_threshold < 0.0:  # 拒绝NaN、无穷大和负数造成的不可预测开关。
        raise ValueError("long_term_memory.min_video_duration_seconds必须是非负有限秒数。")  # 给出可直接修正的配置错误。
    checkpoint_policy = dict(config.get("checkpoint", {}))
    checkpoint_threshold = float(checkpoint_policy.get("min_video_duration_seconds", 300.0))
    checkpoint_interval = int(checkpoint_policy.get("interval_frames", 300))
    if not np.isfinite(checkpoint_threshold) or checkpoint_threshold < 0.0:
        raise ValueError("checkpoint.min_video_duration_seconds必须是非负有限秒数。")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint.interval_frames必须是正整数。")
    yolo_first_policy = dict(config.get("performance", {}).get("yolo_first_pass", {}))
    yolo_batch_size = int(yolo_first_policy.get("batch_size", 8))
    yolo_chunk_frames = int(yolo_first_policy.get("cache_chunk_frames", 300))
    if yolo_batch_size <= 0:
        raise ValueError("performance.yolo_first_pass.batch_size必须是正整数。")
    if yolo_chunk_frames <= 0:
        raise ValueError("performance.yolo_first_pass.cache_chunk_frames必须是正整数。")
    cascade_policy = dict(config.get("performance", {}).get("identity_cascade", {}))
    cascade_density = float(cascade_policy.get("sparse_density_threshold", 0.35))
    cascade_min_cells = int(cascade_policy.get("min_cells", 64))
    if not np.isfinite(cascade_density) or not (0.0 <= cascade_density <= 1.0):
        raise ValueError("performance.identity_cascade.sparse_density_threshold必须位于[0,1]。")
    if cascade_min_cells <= 0:
        raise ValueError("performance.identity_cascade.min_cells必须是正整数。")
    trigger_policy = dict(config.get("performance", {}).get("mask_trigger", {}))
    trigger_iou = float(trigger_policy.get("overlap_iou_threshold", 0.02))
    if not np.isfinite(trigger_iou) or not (0.0 <= trigger_iou <= 1.0):
        raise ValueError("performance.mask_trigger.overlap_iou_threshold必须位于[0,1]。")
    if int(trigger_policy.get("force_refresh_interval_frames", 15)) < 0:
        raise ValueError("performance.mask_trigger.force_refresh_interval_frames不能为负数。")
    km = config.get("identity", {}).get("keypoint_motion", {})
    if int(km.get("expected_mice_count", 0)) != 0:
        raise ValueError("v1.24沿用自适应鼠数版，identity.keypoint_motion.expected_mice_count必须为0。")
    if not bool(km.get("adaptive_count", {}).get("enabled", True)):
        raise ValueError("v1.24要求identity.keypoint_motion.adaptive_count.enabled=true。")


def apply_overrides(config: MutableMapping[str, Any], args: argparse.Namespace) -> None:
    if args.lightweight_behavior and args.full_behavior:
        raise ValueError("--lightweight-behavior与--full-behavior不能同时使用。")
    if args.device is not None:
        config["model"]["device"] = args.device
    if args.tracker is not None:
        config["model"]["tracker"] = args.tracker
    if args.cm_per_pixel is not None:
        if args.cm_per_pixel <= 0:
            raise ValueError("--cm-per-pixel必须大于0。")
        config["scale"]["mode"] = "fixed"
        config["scale"]["cm_per_pixel"] = float(args.cm_per_pixel)
    if args.no_annotated:
        config["output"]["save_annotated_video"] = False
    if args.no_hard_negatives:
        config["clips"]["extract_hard_negatives"] = False
    if args.tracking_only:
        config.setdefault("output", {})["tracking_only"] = True
    lightweight_cfg = config.setdefault("lightweight_behavior_inference", {})
    if args.lightweight_behavior:
        lightweight_cfg["enabled"] = True
    if args.full_behavior:
        lightweight_cfg["enabled"] = False
    performance = config.setdefault("performance", {})
    if args.behavior_pipeline is not None:
        performance["behavior_pipeline"] = str(args.behavior_pipeline)
    if args.pair_compute_mode is not None:
        performance["pair_compute_mode"] = str(args.pair_compute_mode)
    if args.pair_workers is not None:
        if int(args.pair_workers) < 0:
            raise ValueError("--pair-workers must be >= 0")
        performance["pair_workers"] = int(args.pair_workers)
    if args.benchmark_pairs:
        performance["pair_benchmark"] = True
    if args.checkpoint_every_frames is not None and args.checkpoint_every_frames <= 0:
        raise ValueError("--checkpoint-every-frames必须是正整数。")
    if args.arena_boundary is not None:
        config.setdefault("adaptive_arena", {})["reuse_boundary_json"] = str(
            resolve_runtime_path(args.arena_boundary)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多鼠追逐/攻击高召回四分类粗筛、片段提取及人工标注对照评估")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"自定义7关键点YOLO Pose模型best.pt，默认：{DEFAULT_MODEL}")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help=f"单个视频或视频目录，默认：{DEFAULT_VIDEO}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"输出根目录，默认：{DEFAULT_OUTPUT}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("all", "stage1", "stage2"),
        default="all",
        help="all兼容旧单程序；stage1只做推理定ID并缓存；stage2只读阶段一缓存做行为、片段和唯一视频",
    )
    parser.add_argument(
        "--arena-boundary",
        type=Path,
        default=None,
        help="复用已有阶段一_自适应笼界.json；未提供时从本视频运动热力图自动学习并外扩5%%",
    )
    parser.add_argument("--calibration", type=Path, default=None, help="四角透视标定JSON")
    parser.add_argument("--manual-annotations", type=Path, default=None, help="人工标注CSV或XLSX")
    parser.add_argument("--device", default=None, help="0、1或cpu")
    parser.add_argument("--tracker", default=None, help="botsort.yaml或bytetrack.yaml")
    parser.add_argument("--cm-per-pixel", type=float, default=None, help="固定厘米/像素，优先级低于--calibration")
    parser.add_argument("--no-clips", action="store_true", help="不提取弱候选及困难负样本片段")
    parser.add_argument("--no-annotated", action="store_true", help="不输出带标签核查视频")
    parser.add_argument("--no-hard-negatives", action="store_true", help="不抽取困难负样本")
    parser.add_argument("--no-error-clips", action="store_true", help="评估时不导出漏检和误报片段")
    parser.add_argument("--tracking-only", action="store_true", help="只输出检测、稳定ID、骨架视频和轨迹CSV，跳过行为分析")
    parser.add_argument(
        "--lightweight-behavior",
        action="store_true",
        help="只运行YOLO Pose缓存+轻量追踪+追逐/攻击行为推理，旁路完整遮挡/ReID/Mask/ROI/渲染流程",
    )
    parser.add_argument(
        "--full-behavior",
        action="store_true",
        help="临时关闭配置中的轻量行为模式，恢复完整行为流水线；不删除任何轻量代码",
    )
    parser.add_argument(
        "--behavior-pipeline",
        choices=("inline", "staged"),
        default=None,
        help="行为计算路径：inline保持旧版逐帧流程；staged先写步骤1-3缓存再运行步骤4-7",
    )
    parser.add_argument(
        "--pair-compute-mode",
        choices=("python", "numpy", "multiprocess"),
        default=None,
        help="鼠对计算后端；python为结果一致性基线，numpy/multiprocess用于性能比较",
    )
    parser.add_argument(
        "--pair-workers",
        type=int,
        default=None,
        help="阶段化鼠对计算预留的进程数；0表示自动，必须为非负整数",
    )
    parser.add_argument(
        "--benchmark-pairs",
        action="store_true",
        help="记录Python/NumPy鼠对枚举基准，结果写入运行元数据",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从输出目录内最后一个完整推理断点继续；视频、权重或分析配置不一致时拒绝恢复",
    )
    parser.add_argument(
        "--checkpoint-every-frames",
        type=int,
        default=None,
        help="覆盖YAML中的断点间隔帧数；仅影响保存频率，不改变追踪或行为判定",
    )
    parser.add_argument(
        "--behavior-from-cache",
        action="store_true",
        help="读取输出目录中已有逐帧鼠对缓存，只重算行为事件、四分类和标签视频",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(forced_stage: Optional[str] = None) -> int:
    args = build_parser().parse_args()
    if forced_stage is not None:
        args.stage = str(forced_stage)
    base.setup_logging(args.verbose)
    print(f"程序版本：{PROGRAM_VERSION}")
    print(f"底层模块：{getattr(base, 'BASE_MODULE_VERSION', 'unknown')} | {Path(base.__file__).resolve()}")
    print("关键点顺序：" + " -> ".join(KEYPOINT_NAMES))
    print(f"使用模型：{args.model}")
    print(f"输入视频：{args.video}")
    print(f"输出目录：{args.output}")
    print(f"运行阶段：{args.stage}")
    try:
        config_path = resolve_runtime_path(args.config)
        config = load_yaml(config_path)
        validate_config(config)
        apply_overrides(config, args)
        model_path = resolve_runtime_path(args.model)
        video_input = resolve_runtime_path(args.video)
        output_root = ensure_dir(resolve_runtime_path(args.output))
        persistent_log_path = attach_persistent_log(output_root, args.verbose)
        logging.info("持久运行日志：%s", persistent_log_path)
        calibration = resolve_runtime_path(args.calibration) if args.calibration else None
        manual = resolve_runtime_path(args.manual_annotations) if args.manual_annotations else None
        # 缓存复算不加载YOLO模型，也不读取视频名或目录标签参与分类。
        if args.behavior_from_cache:
            # 只选择同时具备元数据和鼠对缓存的单视频结果目录。
            cache_directories = sorted(
                directory
                for directory in output_root.iterdir()
                if directory.is_dir()
                and (directory / "运行元数据.json").exists()
                and (directory / "成对行为标签.csv").exists()
            )
            # 空目录通常表示--output写错，立即报错而不是静默成功。
            if not cache_directories:
                raise FileNotFoundError(
                    f"输出根目录中没有可复算的行为缓存：{output_root}"
                )
            # 分别记录成功目录和失败原因，保持批处理可审计。
            completed, failed = [], []
            # 每个视频独立处理，一个失败不阻止其他视频完成。
            for directory in cache_directories:
                try:
                    completed.append(
                        str(reprocess_behavior_from_cache(directory, config))
                    )
                except Exception as exc:
                    logging.exception("行为缓存复算失败：%s", directory)
                    failed.append(
                        {"video": str(directory), "error": str(exc)}
                    )
            # 写回统一批处理汇总。
            with (output_root / "批处理汇总.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {"completed": completed, "failed": failed},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            # 有失败时沿用普通推理的退出码2。
            return 2 if failed else 0
        lightweight_enabled = bool(
            config.get("lightweight_behavior_inference", {}).get("enabled", False)
        )
        if lightweight_enabled and args.stage in {"all", "stage2"}:
            # Stage 1 remains available for the legacy two-stage workflow.  In
            # all/stage2 mode the active behavior path is explicitly light;
            # the full pipeline below is left untouched for --full-behavior.
            if args.stage == "all" and not model_path.exists():
                raise FileNotFoundError(f"轻量行为推理生成YOLO缓存需要模型：{model_path}")
            completed, failed = [], []
            for video in discover_videos(video_input):
                resource_monitor = ProcessResourceMonitor(config)
                resource_monitor.start()
                completed_directory: Optional[Path] = None
                try:
                    completed_directory = run_lightweight_behavior_inference(
                        video,
                        model_path,
                        output_root,
                        config,
                        config_path,
                        allow_precompute=args.stage == "all",
                        no_clips=args.no_clips,
                        no_render=args.no_annotated,
                    )
                    completed.append(str(completed_directory))
                except Exception as exc:
                    logging.exception("轻量行为推理失败：%s", video)
                    failed.append({"video": str(video), "error": str(exc)})
                finally:
                    usage = resource_monitor.stop()
                    report_directory = completed_directory or (output_root / video.stem)
                    if report_directory.exists():
                        try:
                            _write_resource_report(report_directory, usage)
                            logging.info(
                                "轻量行为CPU/GPU资源计时已写入：%s",
                                report_directory / "资源使用报告.json",
                            )
                        except Exception:
                            logging.exception("写入轻量行为CPU/GPU资源计时失败：%s", report_directory)
            _atomic_json_dump(
                {
                    "mode": "lightweight_behavior_inference",
                    "stage": "lightweight_complete",
                    "completed": completed,
                    "failed": failed,
                },
                output_root / "批处理汇总.json",
            )
            return 2 if failed else 0
        if args.stage == "stage2":
            if output_root.joinpath("阶段一清单.json").exists():
                cache_directories = [output_root]
            else:
                cache_directories = sorted(
                    directory
                    for directory in output_root.iterdir()
                    if directory.is_dir() and (directory / "阶段一清单.json").exists()
                )
            if not cache_directories:
                raise FileNotFoundError(
                    f"输出目录中没有阶段一清单：{output_root}"
                )
            completed, failed = [], []
            for directory in cache_directories:
                resource_monitor = ProcessResourceMonitor(config)
                resource_monitor.start()
                try:
                    completed_directory = run_stage2_from_stage1(
                        directory,
                        config,
                        manual_annotations=manual,
                        save_clips=not args.no_clips,
                        export_error_clips=not args.no_error_clips,
                    )
                    completed.append(str(completed_directory))
                except Exception as exc:
                    logging.exception("阶段二处理失败：%s", directory)
                    failed.append({"video": str(directory), "error": str(exc)})
                finally:
                    usage = resource_monitor.stop()
                    try:
                        _write_resource_report(directory, usage, stage_name="stage2")
                    except Exception:
                        logging.exception("写入阶段二CPU/GPU资源计时失败：%s", directory)
            _atomic_json_dump(
                {"stage": "stage2", "completed": completed, "failed": failed},
                output_root / "批处理汇总_阶段二.json",
            )
            return 2 if failed else 0
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在：{model_path}")
        if manual is not None and not manual.exists():
            raise FileNotFoundError(f"人工标注不存在：{manual}")

        completed, failed = [], []
        for video in discover_videos(video_input):
            resource_monitor = ProcessResourceMonitor(config)
            resource_monitor.start()
            completed_directory: Optional[Path] = None
            try:
                completed_directory = process_video(
                    video, model_path, output_root, config, calibration, manual,
                    save_clips=not args.no_clips,
                    export_error_clips=not args.no_error_clips,
                    resume=args.resume,
                    checkpoint_every_frames=args.checkpoint_every_frames,
                    stage_mode=args.stage,
                )
                completed.append(str(completed_directory))
            except Exception as exc:
                logging.exception("处理失败：%s", video)
                failed.append({"video": str(video), "error": str(exc)})
            finally:
                usage = resource_monitor.stop()
                report_directory = completed_directory or (output_root / video.stem)
                if report_directory.exists():
                    try:
                        _write_resource_report(
                            report_directory,
                            usage,
                            stage_name="stage1" if args.stage == "stage1" else None,
                        )
                        logging.info(
                            "CPU/GPU资源计时已写入：%s",
                            report_directory / "资源使用报告.json",
                        )
                    except Exception:
                        logging.exception("写入CPU/GPU资源计时失败：%s", report_directory)
        with (output_root / "批处理汇总.json").open("w", encoding="utf-8") as f:
            json.dump({"completed": completed, "failed": failed}, f, ensure_ascii=False, indent=2)
        return 2 if failed else 0
    except Exception as exc:
        logging.exception("程序终止：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
