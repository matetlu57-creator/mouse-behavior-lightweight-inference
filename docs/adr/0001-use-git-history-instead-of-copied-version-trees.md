# ADR 0001: Use Git history instead of copied version trees

- Status: accepted
- Date: 2026-08-20

## Context

The repository previously kept `historical_v1.40_v1.41`, `historical_v1.42.1`,
and `original` trees beside the maintained source. These copies mixed reports,
patches, bytecode-era manifests, and source snapshots with the current project.
They made the repository root harder to understand and encouraged file-copy
version management instead of branches, tags, and reproducible tests.

Two original source files are still required by the deterministic performance
regression suite. The current full-pipeline root modules also remain supported
compatibility entry points.

## Decision

- Use Git commits, tags, branches, and release notes as the source of truth for
  historical versions.
- Remove copied historical report/source trees from the maintained branch.
- Retain only the two executable baseline files required by regression tests,
  under `tests/regression/fixtures/legacy_v138`.
- Keep current root compatibility entry points until a separately tested
  migration removes their downstream usage.
- Keep scientific validation evidence under `docs/validation`.

## Consequences

The current tree becomes smaller and responsibility boundaries become clearer.
Historical files remain recoverable from Git. Performance regression remains
reproducible, while adding another copied `v2`, `final`, or `original` tree is
now a repository-validation failure.
