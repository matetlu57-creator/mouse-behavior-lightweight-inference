#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v1.43 Standard Behavior Engine.

This module turns frame-wise pair geometry/kinematics into auditable temporal
behavior states.  It deliberately separates four layers:

    measured features -> continuous evidence -> role inference -> FSM/events

The legacy chase/attack gates are accepted only as *evidence providers*.  They
never directly OR a frame into the final label when decision_mode=standard.
This keeps the existing domain knowledge while removing the growing collection
of mutually independent final-decision branches.

The engine is pure post-processing: it does not modify YOLO detections,
identity assignment, keypoint smoothing or observation history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .behavior.standard_evidence import (
    _PROVIDER_SUFFIXES,
    _EVIDENCE_BASE_COLUMNS,
    _EVIDENCE_DIRECTION_SUFFIXES,
    _evidence_record_columns,
    _clip01,
    _num,
    _slice_max,
    _bool,
    _ramp_high,
    _ramp_low,
    _threshold_membership,
    _distance_membership,
    _speed_membership,
    _identity_quality_from_state,
    _pair_quality,
    _prefix_value,
    _prefix_bool,
    DirectionEvidence,
    ChaseFSMResult,
    AttackFSMResult,
    _chase_evidence,
    _occlusion_score,
    _provider_floor,
    _direction_ids,
)
from .behavior.standard_fsm import _run_attack_fsm, _run_chase_fsm

ENGINE_VERSION = "1.43.0-standard-behavior-engine"
LOGGER = logging.getLogger(__name__)

__all__ = [
    "ENGINE_VERSION",
    "apply_standard_behavior_engine",
    "extract_standard_behavior_events",
    "_PROVIDER_SUFFIXES",
    "_EVIDENCE_BASE_COLUMNS",
    "_EVIDENCE_DIRECTION_SUFFIXES",
    "_evidence_record_columns",
    "_clip01",
    "_num",
    "_slice_max",
    "_bool",
    "_ramp_high",
    "_ramp_low",
    "_threshold_membership",
    "_distance_membership",
    "_speed_membership",
    "_identity_quality_from_state",
    "_pair_quality",
    "_prefix_value",
    "_prefix_bool",
    "DirectionEvidence",
    "ChaseFSMResult",
    "AttackFSMResult",
    "_chase_evidence",
    "_occlusion_score",
    "_provider_floor",
    "_direction_ids",
    "_run_chase_fsm",
    "_run_attack_fsm",
]


def apply_standard_behavior_engine(
    df: pd.DataFrame,
    fps: float,
    config: Mapping[str, Any],
    *,
    copy_input: bool = True,
) -> pd.DataFrame:
    """Add v1.43 evidence/state columns and optionally replace final labels.

    ``decision_mode``:
      - ``standard``: standard FSM owns weak/strong final_chase/final_attack.
      - ``shadow``: compute every standard column but retain legacy final labels.
      - ``legacy``: no replacement; useful for emergency rollback.

    ``copy_input`` keeps the historical non-mutating public behavior by
    default. Internal callers that own a temporary pair DataFrame may opt into
    in-place enrichment to avoid a full-width timeline copy.
    """
    if df.empty:
        return df.copy() if copy_input else df
    LOGGER.debug("Applying %s to %d pair rows at %.3f FPS", ENGINE_VERSION, len(df), fps)
    engine_cfg = dict(config.get("standard_behavior_engine", {}))
    if not bool(engine_cfg.get("enabled", True)):
        return df.copy() if copy_input else df

    # Public callers keep the historical non-mutating default. The lightweight
    # analyzer owns its per-pair DataFrame and can enrich it in place, avoiding
    # one full-width, full-timeline copy for every retained pair.
    output = df.copy() if copy_input else df
    decision_mode = str(engine_cfg.get("decision_mode", "standard")).strip().lower()
    if decision_mode not in {"standard", "shadow", "legacy"}:
        raise ValueError("standard_behavior_engine.decision_mode must be standard/shadow/legacy")

    response_seconds = float(config.get("features", {}).get("response_lookback_seconds", 0.30))
    common_engine = dict(engine_cfg)
    common_engine["response_lookback_seconds"] = response_seconds
    common_engine["reference_body_length_cm"] = float(
        config.get("scale", {}).get("assumed_mouse_body_length_cm", 8.0)
    )

    row_count = len(output)
    pose_quality = np.zeros(row_count, dtype=float)
    identity_quality = np.zeros(row_count, dtype=float)
    behavior_quality = np.zeros(row_count, dtype=float)
    interaction_cfg = engine_cfg.get("interaction_graph", {})
    interaction_radius = float(
        interaction_cfg.get(
            "radius_cm",
            max(
                float(config["chase"]["weak"]["max_distance_cm"]),
                float(config["attack"]["weak"].get("body_center_contact_distance_cm", 6.0)),
            )
            + float(interaction_cfg.get("buffer_cm", 5.0)),
        )
    )

    # The lightweight analyzer keeps the pair's complete timeline for audit,
    # but marks frames outside its padded distance/heading windows invalid.
    # Turning every one of those rows into a Python dictionary and evaluating
    # four directional evidence objects was the dominant long-video cost.  A
    # hard-vetoed row cannot open or hold an FSM state, so it is safe to skip
    # its expensive evidence calculation.  Explicit provider/occlusion hints
    # remain computable for full-pipeline callers even when valid_pair is false.
    valid_pair_input = (
        output.get("valid_pair", pd.Series(True, index=output.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    provider_hint = np.zeros(row_count, dtype=bool)
    for level in ("weak", "strong"):
        for suffix in _PROVIDER_SUFFIXES:
            column = f"{level}_{suffix}"
            if column in output:
                provider_hint |= output[column].fillna(False).astype(bool).to_numpy()
    if "cluster_attack_hint" in output:
        provider_hint |= output["cluster_attack_hint"].fillna(False).astype(bool).to_numpy()

    compute_candidate = valid_pair_input | provider_hint
    skip_inactive_rows = bool(engine_cfg.get("skip_inactive_rows", True))
    compute_row_mask = (
        compute_candidate.copy() if skip_inactive_rows else np.ones(row_count, dtype=bool)
    )
    compute_indices = np.flatnonzero(compute_row_mask)
    evidence_columns = _evidence_record_columns(output.columns.tolist())
    compute_rows = output.loc[:, evidence_columns].iloc[compute_indices].to_dict("records")

    distance_all = pd.to_numeric(
        output.get("center_distance_cm", pd.Series(np.nan, index=output.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    interaction_candidate = np.isfinite(distance_all) & (distance_all <= interaction_radius)
    a_ids = pd.to_numeric(output["mouse_a_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    b_ids = pd.to_numeric(output["mouse_b_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    selected_actor = (
        pd.to_numeric(
            output.get("selected_actor_id", pd.Series(-1, index=output.index)),
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    selected_target = (
        pd.to_numeric(
            output.get("selected_target_id", pd.Series(-1, index=output.index)),
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    selected_role_cfg = dict(engine_cfg.get("selected_role_fallback", {}))
    valid_pair = output["valid_pair"].fillna(False).astype(bool).to_numpy()
    wall_veto = (
        output.get("pair_wall_jump_excluded", pd.Series(False, index=output.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    physics_veto = (~valid_pair) | wall_veto | (~np.isfinite(distance_all))
    physics_veto |= distance_all > interaction_radius

    def finite_numeric_column(column: str) -> np.ndarray:
        values = output.get(column, pd.Series(np.inf, index=output.index))
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        numeric[~np.isfinite(numeric)] = np.inf
        return numeric

    head_distance = np.minimum(
        finite_numeric_column("a_to_b_nose_head_distance_cm"),
        finite_numeric_column("b_to_a_nose_head_distance_cm"),
    )
    tail_distance = np.minimum(
        finite_numeric_column("a_to_b_nose_tail_distance_cm"),
        finite_numeric_column("b_to_a_nose_tail_distance_cm"),
    )
    body_distance = np.minimum(
        finite_numeric_column("a_to_b_nose_body_distance_cm"),
        finite_numeric_column("b_to_a_nose_body_distance_cm"),
    )
    for i, row in zip(compute_indices, compute_rows):
        pq, iq, bq = _pair_quality(row, engine_cfg)
        pose_quality[i], identity_quality[i], behavior_quality[i] = pq, iq, bq
    output["standard_pose_quality"] = pose_quality
    output["standard_identity_quality"] = identity_quality
    output["standard_behavior_quality"] = behavior_quality
    output["standard_interaction_candidate"] = interaction_candidate
    output["standard_behavior_compute_candidate"] = compute_candidate
    output["standard_behavior_compute_row"] = compute_row_mask

    for level in ("weak", "strong"):
        chase_cfg = config["chase"][level]
        attack_cfg = config["attack"][level]
        level_cfg = dict(engine_cfg.get(level, {}))
        chase_fsm_cfg = dict(level_cfg.get("chase_fsm", {}))
        attack_fsm_cfg = dict(level_cfg.get("attack_fsm", {}))
        provider_cfg = dict(level_cfg.get("provider_floors", {}))

        ab_list = []
        ba_list = []
        occ = np.zeros(row_count, dtype=float)
        for i, row in zip(compute_indices, compute_rows):
            ab = _chase_evidence(row, "a_to_b", chase_cfg, attack_cfg, common_engine)
            ba = _chase_evidence(row, "b_to_a", chase_cfg, attack_cfg, common_engine)
            ab_list.append(ab)
            ba_list.append(ba)
            occ[i] = _occlusion_score(row, attack_cfg)

        def scatter_evidence(
            values: Sequence[DirectionEvidence],
            attribute: str,
            dtype: Any = float,
        ) -> np.ndarray:
            scattered = np.zeros(row_count, dtype=dtype)
            if len(compute_indices):
                scattered[compute_indices] = np.asarray(
                    [getattr(value, attribute) for value in values], dtype=dtype
                )
            return scattered

        ab_chase = scatter_evidence(ab_list, "chase")
        ba_chase = scatter_evidence(ba_list, "chase")
        ab_approach = scatter_evidence(ab_list, "approach")
        ba_approach = scatter_evidence(ba_list, "approach")
        ab_contact = scatter_evidence(ab_list, "contact")
        ba_contact = scatter_evidence(ba_list, "contact")
        ab_initiation = scatter_evidence(ab_list, "initiation")
        ba_initiation = scatter_evidence(ba_list, "initiation")
        ab_reaction = scatter_evidence(ab_list, "reaction")
        ba_reaction = scatter_evidence(ba_list, "reaction")
        ab_dynamic = scatter_evidence(ab_list, "dynamic_attack")
        ba_dynamic = scatter_evidence(ba_list, "dynamic_attack")
        ab_grapple = scatter_evidence(ab_list, "grapple")
        ba_grapple = scatter_evidence(ba_list, "grapple")
        ab_dynamic_gate = scatter_evidence(ab_list, "dynamic_attack_gate", bool)
        ba_dynamic_gate = scatter_evidence(ba_list, "dynamic_attack_gate", bool)
        ab_impulse_gate = scatter_evidence(ab_list, "impulse_attack_gate", bool)
        ba_impulse_gate = scatter_evidence(ba_list, "impulse_attack_gate", bool)
        ab_dynamic_context_gate = scatter_evidence(ab_list, "dynamic_attack_context_gate", bool)
        ba_dynamic_context_gate = scatter_evidence(ba_list, "dynamic_attack_context_gate", bool)
        ab_stationary_gate = scatter_evidence(ab_list, "stationary_fight_gate", bool)
        ba_stationary_gate = scatter_evidence(ba_list, "stationary_fight_gate", bool)
        ab_potential_contact = scatter_evidence(ab_list, "potential_contact", bool)
        ba_potential_contact = scatter_evidence(ba_list, "potential_contact", bool)
        ab_evidence_count = scatter_evidence(ab_list, "attack_evidence_count", int)
        ba_evidence_count = scatter_evidence(ba_list, "attack_evidence_count", int)
        pair_deformation = np.maximum(
            scatter_evidence(ab_list, "pose_deformation"),
            scatter_evidence(ba_list, "pose_deformation"),
        )

        # Existing specialized gates become lower bounds on evidence only.
        def scatter_provider(columns: Sequence[str], floor: float) -> np.ndarray:
            scattered = np.zeros(row_count, dtype=float)
            if len(compute_indices):
                scattered[compute_indices] = np.asarray(
                    [_provider_floor(row, columns, floor) for row in compute_rows],
                    dtype=float,
                )
            return scattered

        chase_provider = scatter_provider(
            [
                f"{level}_strict_chase",
                f"{level}_window_chase",
                f"{level}_near_recovery_chase",
                f"{level}_close_follow_chase",
            ],
            float(provider_cfg.get("chase", 0.70 if level == "weak" else 0.78)),
        )
        dynamic_provider = scatter_provider(
            [f"{level}_strict_attack", f"{level}_impulse_attack"],
            float(provider_cfg.get("dynamic_attack", 0.70 if level == "weak" else 0.80)),
        )
        grapple_provider = scatter_provider(
            [f"{level}_grapple_attack"],
            float(provider_cfg.get("grapple", 0.72 if level == "weak" else 0.82)),
        )
        occlusion_provider = scatter_provider(
            [f"{level}_occlusion_overlap_attack"],
            float(provider_cfg.get("occlusion", 0.78 if level == "weak" else 0.86)),
        )
        occ = np.maximum(occ, occlusion_provider)

        # Role inference is behavior-specific.  Chase and attack can disagree
        # during close contact, so never force both through one role score.
        chase_role_conf = np.abs(ab_chase - ba_chase)
        chase_ab_is_actor = ab_chase >= ba_chase
        selected_role_fallback = np.zeros(row_count, dtype=bool)
        if bool(selected_role_cfg.get("enabled", False)):
            selected_valid = (
                ((selected_actor == a_ids) | (selected_actor == b_ids))
                & ((selected_target == a_ids) | (selected_target == b_ids))
                & (selected_actor != selected_target)
            )
            # This is deliberately limited to exact directional ties.  A
            # real directional score wins whenever the pair features provide
            # one, so the fallback cannot overwrite a resolved role.
            selected_role_fallback = selected_valid & (chase_role_conf <= 1e-9)
            fallback_conf = float(selected_role_cfg.get("confidence", 0.20))
            chase_role_conf = np.where(
                selected_role_fallback,
                np.maximum(chase_role_conf, fallback_conf),
                chase_role_conf,
            )
            chase_ab_is_actor = np.where(
                selected_role_fallback,
                selected_actor == a_ids,
                chase_ab_is_actor,
            )
        ab_attack_role_score = 0.58 * ab_initiation + 0.42 * ab_reaction
        ba_attack_role_score = 0.58 * ba_initiation + 0.42 * ba_reaction
        attack_role_conf = np.abs(ab_attack_role_score - ba_attack_role_score)
        attack_ab_is_actor = ab_attack_role_score >= ba_attack_role_score

        chase_score = np.maximum(np.maximum(ab_chase, ba_chase), chase_provider)
        approach_score = np.maximum(ab_approach, ba_approach)
        contact_score = np.maximum(ab_contact, ba_contact)
        initiation_score = np.where(attack_ab_is_actor, ab_initiation, ba_initiation)
        reaction_score = np.where(attack_ab_is_actor, ab_reaction, ba_reaction)
        dynamic_gate = np.where(attack_ab_is_actor, ab_dynamic_gate, ba_dynamic_gate)
        impulse_gate = np.where(attack_ab_is_actor, ab_impulse_gate, ba_impulse_gate)
        dynamic_context_gate = np.where(
            attack_ab_is_actor,
            ab_dynamic_context_gate,
            ba_dynamic_context_gate,
        )
        stationary_gate = np.where(attack_ab_is_actor, ab_stationary_gate, ba_stationary_gate)
        potential_contact = np.where(attack_ab_is_actor, ab_potential_contact, ba_potential_contact)
        attack_evidence_count = np.where(attack_ab_is_actor, ab_evidence_count, ba_evidence_count)
        causal_distance_max = float(
            attack_fsm_cfg.get("causal_max_center_distance_cm", float("inf"))
        )
        if np.isfinite(causal_distance_max):
            causal_near = np.isfinite(distance_all) & (distance_all <= causal_distance_max)
            dynamic_gate &= causal_near
            dynamic_context_gate &= causal_near
            impulse_gate &= causal_near
            stationary_gate &= causal_near
        dynamic_score = np.maximum(
            np.where(attack_ab_is_actor, ab_dynamic, ba_dynamic), dynamic_provider
        )
        grapple_score = np.maximum(np.maximum(ab_grapple, ba_grapple), grapple_provider)
        attack_score = np.maximum.reduce([dynamic_score, grapple_score, occ])

        chase_fsm = _run_chase_fsm(
            chase_score,
            approach_score,
            behavior_quality,
            chase_role_conf,
            physics_veto,
            fps,
            chase_fsm_cfg,
        )
        attack_fsm = _run_attack_fsm(
            initiation_score,
            contact_score,
            reaction_score,
            dynamic_score,
            grapple_score,
            occ,
            behavior_quality,
            attack_role_conf,
            physics_veto,
            dynamic_gate,
            stationary_gate,
            dynamic_context_gate,
            impulse_gate,
            fps,
            attack_fsm_cfg,
        )

        chase_role_min = float(chase_fsm_cfg.get("min_role_confidence", 0.08))
        attack_role_min = float(attack_fsm_cfg.get("min_role_confidence", 0.06))
        chase_role_known = chase_role_conf >= chase_role_min
        attack_role_known = attack_role_conf >= attack_role_min
        chase_actor_ids = np.where(chase_ab_is_actor, a_ids, b_ids)
        chase_target_ids = np.where(chase_ab_is_actor, b_ids, a_ids)
        attack_actor_ids = np.where(attack_ab_is_actor, a_ids, b_ids)
        attack_target_ids = np.where(attack_ab_is_actor, b_ids, a_ids)
        chase_actor_ids = np.where(chase_role_known, chase_actor_ids, -1)
        chase_target_ids = np.where(chase_role_known, chase_target_ids, -1)
        attack_actor_ids = np.where(attack_role_known, attack_actor_ids, -1)
        attack_target_ids = np.where(attack_role_known, attack_target_ids, -1)
        attack_dominates = attack_score > chase_score
        actor_ids = np.where(attack_dominates, attack_actor_ids, chase_actor_ids)
        target_ids = np.where(attack_dominates, attack_target_ids, chase_target_ids)
        role_conf = np.where(attack_dominates, attack_role_conf, chase_role_conf)

        contact_types = np.full(row_count, "", dtype=object)
        contact_threshold = float(attack_cfg.get("contact_distance_cm", 3.0))
        head_contact = compute_row_mask & (head_distance <= contact_threshold)
        tail_contact = compute_row_mask & (tail_distance <= contact_threshold)
        body_contact = compute_row_mask & (body_distance <= contact_threshold)
        contact_types[head_contact & tail_contact] = "nose_head_and_nose_tail"
        contact_types[head_contact & ~tail_contact] = "nose_head"
        contact_types[~head_contact & tail_contact] = "nose_tail"
        contact_types[~head_contact & ~tail_contact & body_contact] = "nose_body"

        output[f"{level}_standard_a_to_b_chase_score"] = ab_chase
        output[f"{level}_standard_b_to_a_chase_score"] = ba_chase
        output[f"{level}_standard_chase_score"] = chase_score
        output[f"{level}_standard_approach_score"] = approach_score
        output[f"{level}_standard_contact_score"] = contact_score
        output[f"{level}_standard_initiation_score"] = initiation_score
        output[f"{level}_standard_reaction_score"] = reaction_score
        output[f"{level}_standard_dynamic_attack_score"] = dynamic_score
        output[f"{level}_standard_attack_dynamic_gate"] = dynamic_gate
        output[f"{level}_standard_attack_context_gate"] = dynamic_context_gate
        output[f"{level}_standard_attack_impulse_gate"] = impulse_gate
        output[f"{level}_standard_attack_stationary_gate"] = stationary_gate
        output[f"{level}_standard_attack_potential_contact"] = potential_contact
        output[f"{level}_standard_attack_evidence_count"] = attack_evidence_count
        output[f"{level}_standard_grapple_score"] = grapple_score
        output[f"{level}_standard_pose_deformation_score"] = pair_deformation
        output[f"{level}_standard_contact_type"] = contact_types
        output[f"{level}_standard_occlusion_score"] = occ
        output[f"{level}_standard_attack_score"] = attack_score
        output[f"{level}_standard_chase_role_confidence"] = chase_role_conf
        output[f"{level}_standard_chase_role_fallback"] = selected_role_fallback
        output[f"{level}_standard_attack_role_confidence"] = attack_role_conf
        output[f"{level}_standard_role_confidence"] = role_conf
        output[f"{level}_standard_chase_actor_id"] = chase_actor_ids
        output[f"{level}_standard_chase_target_id"] = chase_target_ids
        output[f"{level}_standard_attack_actor_id"] = attack_actor_ids
        output[f"{level}_standard_attack_target_id"] = attack_target_ids
        output[f"{level}_standard_actor_id"] = actor_ids
        output[f"{level}_standard_target_id"] = target_ids
        output[f"{level}_standard_chase_state"] = chase_fsm.state
        output[f"{level}_standard_attack_state"] = attack_fsm.state
        output[f"{level}_standard_attack_subtype"] = attack_fsm.subtype
        output[f"{level}_standard_final_chase"] = chase_fsm.mask & valid_pair
        output[f"{level}_standard_final_attack"] = attack_fsm.mask & valid_pair
        output[f"{level}_standard_behavior_confidence"] = np.where(
            chase_fsm.mask | attack_fsm.mask,
            np.maximum(chase_score, attack_score) * behavior_quality,
            0.0,
        )

        if decision_mode == "standard":
            output[f"{level}_final_chase"] = output[f"{level}_standard_final_chase"].astype(bool)
            output[f"{level}_final_attack"] = output[f"{level}_standard_final_attack"].astype(bool)

    output["standard_behavior_engine_version"] = ENGINE_VERSION
    output["standard_behavior_decision_mode"] = decision_mode
    return output


def extract_standard_behavior_events(
    df: pd.DataFrame,
    fps: float,
    level: str,
    pair_key: str = "",
) -> list[dict[str, Any]]:
    """Extract independent chase and attack ethogram events from FSM masks.

    Unlike the legacy four-class segmenter, chase and attack are aggregated
    independently, so an attack beginning during an ongoing chase does not
    split the underlying chase bout.  The four-class API can therefore remain
    backward compatible while this CSV provides the scientific ethogram.
    """
    if df.empty:
        return []
    fps = max(float(fps), 1e-9)
    events: list[dict[str, Any]] = []
    frame_values = (
        pd.to_numeric(df.get("frame", pd.Series(range(len(df)))), errors="coerce")
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    for behavior in ("chase", "attack"):
        mask_col = f"{level}_standard_final_{behavior}"
        if mask_col not in df.columns:
            continue
        mask = df[mask_col].fillna(False).astype(bool).to_numpy()
        if behavior == "chase":
            actor_col = f"{level}_standard_chase_actor_id"
            target_col = f"{level}_standard_chase_target_id"
            score_col = f"{level}_standard_chase_score"
            role_col = f"{level}_standard_chase_role_confidence"
            subtype_col = None
        else:
            actor_col = f"{level}_standard_attack_actor_id"
            target_col = f"{level}_standard_attack_target_id"
            score_col = f"{level}_standard_attack_score"
            role_col = f"{level}_standard_attack_role_confidence"
            subtype_col = f"{level}_standard_attack_subtype"
        quality_col = f"{level}_standard_behavior_confidence"
        i = 0
        while i < len(mask):
            if not mask[i]:
                i += 1
                continue
            start = i
            while i + 1 < len(mask) and mask[i + 1]:
                i += 1
            end = i
            segment = df.iloc[start : end + 1]
            actors = (
                pd.to_numeric(
                    segment.get(actor_col, pd.Series(-1, index=segment.index)), errors="coerce"
                )
                .fillna(-1)
                .astype(int)
                .to_numpy()
            )
            targets = (
                pd.to_numeric(
                    segment.get(target_col, pd.Series(-1, index=segment.index)), errors="coerce"
                )
                .fillna(-1)
                .astype(int)
                .to_numpy()
            )
            role_pairs = [
                (int(actor), int(target))
                for actor, target in zip(actors, targets)
                if int(actor) >= 0 and int(target) >= 0
            ]
            if role_pairs:
                counts: Dict[Tuple[int, int], int] = {}
                for pair in role_pairs:
                    counts[pair] = counts.get(pair, 0) + 1
                actor_id, target_id = max(
                    counts, key=lambda pair: (counts[pair], -pair[0], -pair[1])
                )
            else:
                actor_id, target_id = -1, -1
            scores = pd.to_numeric(
                segment.get(score_col, pd.Series(0.0, index=segment.index)), errors="coerce"
            ).fillna(0.0)
            confidence = pd.to_numeric(
                segment.get(quality_col, pd.Series(0.0, index=segment.index)), errors="coerce"
            ).fillna(0.0)
            roles = pd.to_numeric(
                segment.get(role_col, pd.Series(0.0, index=segment.index)), errors="coerce"
            ).fillna(0.0)
            peak_offset = int(np.argmax(scores.to_numpy(dtype=float))) if len(scores) else 0
            subtype = ""
            if subtype_col and subtype_col in segment.columns:
                values = [
                    str(v)
                    for v in segment[subtype_col].tolist()
                    if str(v) not in {"", "none", "nan"}
                ]
                if values:
                    subtype = max(set(values), key=values.count)
            start_frame = int(frame_values[start])
            end_frame = int(frame_values[end])
            events.append(
                {
                    "behavior_engine": ENGINE_VERSION,
                    "candidate_level": str(level),
                    "behavior": behavior,
                    "subtype": subtype,
                    "pair_key": str(pair_key),
                    "actor_id": actor_id,
                    "target_id": target_id,
                    "role_ambiguous": bool(actor_id < 0 or target_id < 0),
                    "start_frame": start_frame,
                    "peak_frame": int(frame_values[start + peak_offset]),
                    "end_frame": end_frame,
                    "start_time_s": start_frame / fps,
                    "end_time_s": end_frame / fps,
                    "duration_s": (end_frame - start_frame + 1) / fps,
                    "mean_score": float(scores.mean()) if len(scores) else 0.0,
                    "peak_score": float(scores.max()) if len(scores) else 0.0,
                    "mean_behavior_confidence": float(confidence.mean())
                    if len(confidence)
                    else 0.0,
                    "mean_role_confidence": float(roles.mean()) if len(roles) else 0.0,
                }
            )
            i += 1
    events.sort(key=lambda item: (int(item["start_frame"]), str(item["behavior"])))
    for index, event in enumerate(events, start=1):
        event["standard_event_id"] = f"{level[:1].upper()}SB{index:06d}"
    return events
