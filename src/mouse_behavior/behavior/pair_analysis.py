"""Candidate-pair orchestration and event finalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import standard_behavior_engine as behavior_engine
from ..parallel_behavior_fsm import ParallelBehaviorFSM
from ..preprocessing.pair_features import _PairWorkset, _pair_dataframe
from .ethogram import (
    _extended_pair_events,
    _extended_short_clip_pair_events,
    _extract_contact_events,
)
from ..utils.timer import Timer

LOGGER = logging.getLogger("mouse_behavior.lightweight_behavior_inference")


@dataclass
class _PairAnalysisResult:
    """In-memory outputs produced by the candidate-pair analysis stage."""

    events: list[dict[str, Any]]
    contact_events: list[dict[str, Any]]
    extended_events: list[dict[str, Any]]
    pair_summaries: list[dict[str, Any]]
    top_evidence: list[dict[str, Any]]
    fsm_coordinator: ParallelBehaviorFSM


def _analyze_candidate_pairs(
    workset: _PairWorkset,
    kin: Mapping[str, Any],
    *,
    fps: float,
    source_fps: float,
    sample_stride: int,
    video_path: Path,
    config: Mapping[str, Any],
    stage_timings: dict[str, float] | None = None,
) -> _PairAnalysisResult:
    """Analyze retained pairs while preserving the complete pair timeline."""

    events: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    extended_events: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    top_evidence: list[dict[str, Any]] = []
    pair_analysis_timer = Timer(
        "candidate_pair_analysis",
        logger=LOGGER,
        sink=stage_timings,
    ).start()
    pair_fsm_coordinator = ParallelBehaviorFSM(dict(config.get("parallel_behavior_fsm", {})))
    candidate_ordinal = 0
    for pair_index, (mouse_a, mouse_b) in enumerate(zip(workset.all_pair_i, workset.all_pair_j)):
        metric_index = workset.candidate_metric_index.get(pair_index)
        base_summary: dict[str, Any] = {
            "pair_key": f"{int(mouse_a)}_{int(mouse_b)}",
            "mouse_a_id": int(mouse_a),
            "mouse_b_id": int(mouse_b),
            "valid_frames": int(
                np.asarray(
                    workset.prefilter["valid_pair"][:, pair_index],
                    dtype=bool,
                ).sum()
            ),
            "min_distance_cm": float(np.nanmin(workset.prefilter["distance"][:, pair_index]))
            if np.isfinite(workset.prefilter["distance"][:, pair_index]).any()
            else float("nan"),
            "max_speed_cm_s": float(
                max(
                    float(np.asarray(workset.metrics["speed"][:, int(mouse_a)]).max(initial=0.0)),
                    float(np.asarray(workset.metrics["speed"][:, int(mouse_b)]).max(initial=0.0)),
                )
            ),
            "engine_evaluated": metric_index is not None,
            "fsm_evaluated_frames": 0,
            "fsm_skipped_frames": 0,
            "fsm_evaluated_fraction": 0.0,
            "nose_head_contact_event_count": 0,
            "nose_tail_contact_event_count": 0,
            "combined_nose_head_nose_tail_event_count": 0,
            "contact_sample_count": 0,
        }
        if metric_index is None:
            for level in ("weak", "strong"):
                for behavior in ("chase", "attack"):
                    base_summary[f"{level}_{behavior}_frames"] = 0
                    base_summary[f"{level}_{behavior}_max_score"] = 0.0
                    base_summary[f"{level}_{behavior}_role_known_rate"] = None
            pair_summaries.append(base_summary)
            continue

        candidate_ordinal += 1
        if pair_index == 0 or pair_index % 20 == 0 or pair_index == len(workset.all_pair_i) - 1:
            LOGGER.info(
                "[pair analysis] pair %d/%d (%d/%d candidates)",
                pair_index + 1,
                len(workset.all_pair_i),
                candidate_ordinal,
                len(workset.candidate_pair_indices),
            )
        pair_df = _pair_dataframe(
            workset.metrics,
            metric_index,
            int(mouse_a),
            int(mouse_b),
            workset.pair_i,
            workset.pair_j,
            fps,
            float(kin["cm_per_pixel"]),
        )
        pair_contact_events = _extract_contact_events(
            pair_df,
            pair_key=f"{int(mouse_a)}_{int(mouse_b)}",
            source_video=video_path,
            source_fps=source_fps,
            sample_stride=sample_stride,
            contact_config=dict(config.get("contact_detection", {})),
            fsm_coordinator=pair_fsm_coordinator,
        )
        contact_events.extend(pair_contact_events)
        enriched = behavior_engine.apply_standard_behavior_engine(
            pair_df,
            fps,
            config,
            copy_input=False,
        )
        extended_events.extend(
            _extended_pair_events(
                pair_df,
                metrics=workset.metrics,
                pair_index=metric_index,
                enriched=enriched,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
                fsm_coordinator=pair_fsm_coordinator,
            )
        )
        extended_events.extend(
            _extended_short_clip_pair_events(
                pair_df,
                enriched,
                source_video=video_path,
                source_fps=source_fps,
                sample_stride=sample_stride,
                config=config,
                fsm_coordinator=pair_fsm_coordinator,
            )
        )
        summary = base_summary
        fsm_compute = (
            enriched.get(
                "standard_behavior_compute_row",
                pair_df.get("valid_pair", pd.Series(True, index=pair_df.index)),
            )
            .fillna(False)
            .astype(bool)
        )
        summary["fsm_evaluated_frames"] = int(fsm_compute.sum())
        summary["fsm_skipped_frames"] = int(len(fsm_compute) - fsm_compute.sum())
        summary["fsm_evaluated_fraction"] = float(fsm_compute.mean()) if len(fsm_compute) else 0.0
        summary["nose_head_contact_event_count"] = int(
            sum(
                "nose_head" in str(event.get("contact_type_components", "")).split(";")
                for event in pair_contact_events
            )
        )
        summary["nose_tail_contact_event_count"] = int(
            sum(
                "nose_tail" in str(event.get("contact_type_components", "")).split(";")
                for event in pair_contact_events
            )
        )
        summary["combined_nose_head_nose_tail_event_count"] = int(
            sum(
                event.get("contact_type") == "nose_head_and_nose_tail"
                for event in pair_contact_events
            )
        )
        summary["contact_sample_count"] = int(
            sum(int(event.get("sample_count", 0)) for event in pair_contact_events)
        )
        for level in ("weak", "strong"):
            for behavior in ("chase", "attack"):
                mask_col = f"{level}_standard_final_{behavior}"
                score_col = f"{level}_standard_{behavior}_score"
                actor_col = f"{level}_standard_{behavior}_actor_id"
                target_col = f"{level}_standard_{behavior}_target_id"
                active = (
                    enriched[mask_col].fillna(False).astype(bool)
                    if mask_col in enriched
                    else pd.Series(False, index=enriched.index)
                )
                summary[f"{level}_{behavior}_frames"] = int(active.sum())
                summary[f"{level}_{behavior}_max_score"] = (
                    float(enriched[score_col].max()) if score_col in enriched else 0.0
                )
                if active.any() and actor_col in enriched and target_col in enriched:
                    known = (
                        pd.to_numeric(enriched.loc[active, actor_col], errors="coerce") >= 0
                    ) & (pd.to_numeric(enriched.loc[active, target_col], errors="coerce") >= 0)
                    summary[f"{level}_{behavior}_role_known_rate"] = float(known.mean())
                else:
                    summary[f"{level}_{behavior}_role_known_rate"] = None
            for event in behavior_engine.extract_standard_behavior_events(
                enriched,
                fps,
                level,
                pair_key=f"{int(mouse_a)}_{int(mouse_b)}",
            ):
                event = dict(event)
                event["source_video"] = str(video_path)
                event["analysis_mode"] = "lightweight_cache_tracking"
                event["analysis_start_frame"] = int(event.get("start_frame", 0))
                event["analysis_peak_frame"] = int(event.get("peak_frame", 0))
                event["analysis_end_frame"] = int(event.get("end_frame", 0))
                event["start_frame"] = int(event.get("start_frame", 0)) * sample_stride
                event["peak_frame"] = int(event.get("peak_frame", 0)) * sample_stride
                event["end_frame"] = int(event.get("end_frame", 0)) * sample_stride
                events.append(event)
        for behavior in ("chase", "attack"):
            score_column = f"strong_standard_{behavior}_score"
            if score_column not in enriched:
                continue
            candidate = enriched[
                ["frame", "time_s", "pair_key", "center_distance_cm", score_column]
            ].copy()
            candidate["behavior"] = behavior
            candidate["level"] = "strong"
            candidate = candidate.rename(columns={score_column: "score"})
            evidence_rows = candidate.nlargest(5, "score").to_dict("records")
            for row in evidence_rows:
                row["analysis_frame"] = int(row["frame"])
                row["source_frame"] = int(row["frame"]) * sample_stride
                row["source_time_s"] = float(row["frame"]) * sample_stride / source_fps
            top_evidence.extend(evidence_rows)
        pair_summaries.append(summary)
    pair_analysis_timer.stop()
    return _PairAnalysisResult(
        events=events,
        contact_events=contact_events,
        extended_events=extended_events,
        pair_summaries=pair_summaries,
        top_evidence=top_evidence,
        fsm_coordinator=pair_fsm_coordinator,
    )


def _finalize_event_records_in_place(
    events: list[dict[str, Any]],
    contact_events: list[dict[str, Any]],
    source_fps: float,
) -> None:
    """Assign stable IDs and source-time fields without changing event order."""

    for index, event in enumerate(
        sorted(
            events,
            key=lambda item: (
                int(item.get("start_frame", 0)),
                str(item.get("pair_key", "")),
                str(item.get("level", "")),
            ),
        ),
        start=1,
    ):
        event["light_event_id"] = f"LWE{index:05d}"
        event["start_time_s"] = float(event.get("start_frame", 0)) / source_fps
        event["end_time_s"] = float(event.get("end_frame", 0)) / source_fps
        event["duration_s"] = (
            float(event.get("end_frame", 0) - event.get("start_frame", 0) + 1) / source_fps
        )

    for index, event in enumerate(
        sorted(
            contact_events,
            key=lambda item: (
                int(item.get("start_frame", 0)),
                str(item.get("pair_key", "")),
                str(item.get("contact_type", "")),
            ),
        ),
        start=1,
    ):
        event["contact_event_id"] = f"LCE{index:05d}"
