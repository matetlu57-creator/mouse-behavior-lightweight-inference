# -*- coding: utf-8 -*-
"""v1.42.1 mask trigger controller.

This module is intentionally separate from mask_cluster_reid.py because the
uploaded optimization bundle does not contain that project-side source file.
It can therefore be overlaid without shadowing/replacing the user's existing
mask_cluster_reid implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class MaskTriggerDecision:
    run_mask: bool
    reason: str


class MaskTriggerController:
    def __init__(
        self,
        config: Mapping[str, Any],
        instance_mask_cfg: Mapping[str, Any],
        cluster_reid_cfg: Mapping[str, Any],
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.result_preserving_only = bool(cfg.get("result_preserving_only", True))
        self.overlap_iou_threshold = float(cfg.get("overlap_iou_threshold", 0.02))
        self.force_refresh_interval_frames = max(0, int(cfg.get("force_refresh_interval_frames", 15)))
        self.instance_identity_weight = float(instance_mask_cfg.get("identity_cost_weight", 0.0))
        self.cluster_mask_weight = float(cluster_reid_cfg.get("mask_weight", 0.0))
        self.run_count = 0
        self.skip_count = 0
        self.reason_counts: dict[str, int] = {}

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float64).reshape(-1)
        bb = np.asarray(b, dtype=np.float64).reshape(-1)
        if aa.size < 4 or bb.size < 4 or not np.all(np.isfinite(aa[:4])) or not np.all(np.isfinite(bb[:4])):
            return 0.0
        x1, y1 = max(aa[0], bb[0]), max(aa[1], bb[1])
        x2, y2 = min(aa[2], bb[2]), min(aa[3], bb[3])
        inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        area_a = max(aa[2] - aa[0], 0.0) * max(aa[3] - aa[1], 0.0)
        area_b = max(bb[2] - bb[0], 0.0) * max(bb[3] - bb[1], 0.0)
        union = area_a + area_b - inter
        return float(inter / union) if union > 1e-9 else 0.0

    def _finish(self, run: bool, reason: str) -> MaskTriggerDecision:
        if run:
            self.run_count += 1
        else:
            self.skip_count += 1
        self.reason_counts[reason] = int(self.reason_counts.get(reason, 0)) + 1
        return MaskTriggerDecision(bool(run), str(reason))

    def decide(
        self,
        frame_idx: int,
        detections: Sequence[Any],
        cluster_context: Mapping[str, Any],
        recovery_regions: Sequence[Mapping[str, Any]],
        identity: Any,
    ) -> MaskTriggerDecision:
        # Disabled controller means legacy behavior: always run mask extraction.
        if not self.enabled:
            return self._finish(True, "controller_disabled_legacy")

        # Strict mode is the default for a FINAL merge: if mask descriptors have
        # any downstream identity/ReID weight, skipping them can alter matching.
        if self.result_preserving_only and (
            self.instance_identity_weight > 0.0 or self.cluster_mask_weight > 0.0
        ):
            return self._finish(True, "strict_downstream_mask_dependency")

        if recovery_regions:
            return self._finish(True, "recovery_requested")
        for region in (cluster_context or {}).get("regions", []):
            if any(bool(region.get(k, False)) for k in ("deficit", "merged_like", "recovery_requested", "attack_hint")):
                return self._finish(True, "occlusion_cluster_risk")

        dets = list(detections)
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                if self._bbox_iou(dets[i].bbox_xyxy, dets[j].bbox_xyxy) > self.overlap_iou_threshold:
                    return self._finish(True, "bbox_overlap")

        frame_stats = dict(getattr(identity, "frame_stats", {}) or {})
        if int(frame_stats.get("suspicious", 0)) > 0 or int(frame_stats.get("lost", 0)) > 0:
            return self._finish(True, "identity_uncertainty")

        if self.force_refresh_interval_frames > 0 and int(frame_idx) % self.force_refresh_interval_frames == 0:
            return self._finish(True, "periodic_refresh")
        return self._finish(False, "low_risk")

    def summary(self) -> dict[str, Any]:
        total = self.run_count + self.skip_count
        return {
            "run_count": int(self.run_count),
            "skip_count": int(self.skip_count),
            "skip_fraction": float(self.skip_count / total) if total else 0.0,
            "reason_counts": dict(self.reason_counts),
            "result_preserving_only": bool(self.result_preserving_only),
        }
