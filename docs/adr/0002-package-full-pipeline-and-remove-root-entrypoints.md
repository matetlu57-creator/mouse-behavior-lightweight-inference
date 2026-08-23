# ADR 0002: Package the full pipeline and remove root Python entry points

- Status: accepted
- Date: 2026-08-23

## Context

The repository root still contained seven import shims and two complete
full-pipeline modules. Tests, the Windows installer and the stage runner loaded
those paths directly. This preserved old commands, but also kept `sys.path`
workarounds, allowed new root Python files, and made the package/script boundary
ambiguous.

## Decision

- Move the complete pipeline to `src/mouse_behavior/full_pipeline/`.
- Keep reusable algorithms under `mouse_behavior` and all repository-run CLI
  wrappers under `scripts/`.
- Provide `python -m mouse_behavior.full_pipeline` and the installed
  `mouse-behavior-full` console command.
- Keep the renamed lightweight module only as the package-local alias
  `mouse_behavior.lightweight_cache_behavior_analysis`.
- Reject every repository-root `.py` file in repository validation.
- Use an external Git worktree for parallel branch development instead of
  copied source directories.

## Migration

| Previous usage | Maintained usage |
| --- | --- |
| `python mouse_chase_attack_high_recall.py ...` | `python scripts/run_full_behavior_pipeline.py ...` |
| `import standard_behavior_engine` | `from mouse_behavior import standard_behavior_engine` |
| `import adaptive_arena_boundary` | `from mouse_behavior import adaptive_arena_boundary` |
| `import annotation_website_export` | `from mouse_behavior import annotation_website_export` |
| `import mask_trigger_controller` | `from mouse_behavior import mask_trigger_controller` |
| `import nvenc_video_writer` | `from mouse_behavior import nvenc_video_writer` |
| `import lightweight_cache_behavior_analysis` | `from mouse_behavior import lightweight_cache_behavior_analysis` |

## Consequences

The repository root has one enforceable meaning: project metadata,
configuration and documentation only. The full pipeline remains available, but
its external `disk_sequence_guard`, `pose_quality_recovery` and
`mask_cluster_reid` extensions are still required when the corresponding full
pipeline features execute. CLI help remains available without those extensions,
and first use reports an actionable dependency error. Moving the two large
pipeline files is intentionally mechanical;
their behavioral redesign remains a separate, regression-gated task.
