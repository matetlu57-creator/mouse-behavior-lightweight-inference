# Behavior boundaries and media provenance audit repair

This change repairs reproduced rule and delivery defects without retuning
numeric thresholds, reconstructing missing poses, or changing reviewed datasets.
Historical human-reviewed A1/A2 reports remain immutable. Fresh evaluation is
written to a separate run and must be described as a new machine-only result.

## Behavior contracts

- When Chasing takes frames from Approach, each remaining Approach span must
  independently satisfy its onset, endpoint, visibility and duration conditions.
  Lagged measured motion can provide context; excluded endpoints cannot qualify
  the remaining event.
- Independently qualified pursuit directions retain their measured role support.
  Ambiguous rows remain unknown. Sustained role changes are split into separate
  events instead of losing one direction to whole-event majority voting.
- Dispersal discovers groups formed after a video window begins. Each observed
  formation has a fixed origin and its own first-observation admission clock;
  later arrivals outside that origin cannot acquire membership through reseeding.
  Raw pair/triple reconnections cannot truncate an ongoing departure. A new
  formation follows a measured departure-attempt boundary, including a short
  attempt that does not itself qualify as Dispersal. Published events still
  require the unchanged strict greater-than-five-second rule. Missing observations
  alone do not establish a completed attempt. Reformations may occur at a new
  location; a regrouped subset does not cancel other members' departure.
  Reformations have separate origins and cannot borrow an earlier departure.
- Core-window clipping must not remove the required endpoint or confirmation of
  an Approach or Dispersal event. Retained complete qualified parents are marked
  `retained_complete_parent_for_semantics`; their frames remain real observations
  or explicitly recorded original process intervals, never fabricated padding.
- Pair feature construction preserves the validated kinematic
  `reference_body_cm`. Missing or invalid scale is rejected rather than silently
  replaced with a constant.

## Video and package contracts

The lightweight cache builder seals source bytes and trajectory chunks. Analysis
seals that binding and its main/contact event CSVs. Rendering verifies the entire
chain before writing and checks it again before publishing. Paths, FPS and frame
counts are checked in addition to hashes. A deliberate preview records its
source interval and actual decoded frame count.

Digest-free legacy lightweight results cannot establish this chain. Rebuild the
cache and rerun analysis with the same authorized source/model inputs. Do not
manufacture a historical receipt by merely adding a current hash. The separate
canonical pipeline can still consume an existing cache with its independently
verified source/model receipt.

Canonical run receipts contain portable module-key source hashes, their local
dependency closure, actual configuration/default values and effective duration
policies. Render-only/export operations reject changed or incomplete historical
rule snapshots and request fresh inference. This deliberately avoids silently
requalifying old events under a new policy. Model inference need not be repeated
when an existing observed cache still passes source/model binding.

Recropping reconstructs current annotation fields, intersects and rebases known
support intervals, and recomputes counts and decoded media facts. Relabeling
withdraws old behavior/role evidence. Changing the participant ID set also
withdraws inherited roles, even when every requested ID exists in the source.
The same ID set in a different order remains compatible. Old event IDs and unknown extensions remain
in provenance instead of becoming current claims. Visual IDs must come from the
source tracks.

Fixed-window showcase exporters skip a source shorter than the requested window.
They encode into temporary files, decode and verify all clips, then publish.
Early EOF leaves no completed clips or manifest. These showcase windows remain
distinct from duration-qualified canonical behavior event packages.

## Verification and interpretation

Reproductions include real seven-point trajectories through feature loading and
video-window aggregation, actual SQLite stores, and real encoded/decoded MP4s.
Tests cover both failures and valid short-prefix/control cases, identity changes,
late group formation, repeated episodes, short departures, source replacement,
CSV/cache mutation, recropping, missing roles and early EOF.

Run the repository checks from the checkout:

```powershell
python scripts/validate_repository.py
python scripts/run_quality.py --ci
```

Use `scripts/evaluate_active_dataset_v17.py` with an explicit dataset pointer and
provenance root for before/after labeled-clip evaluation. Keep the approved 755
cohort as the paired denominator. Report the lineage-verified 30-clip Isolation
subset separately. Preserve the approved acceptance thresholds: Attack 30%
coverage and other behaviors 50%, with exact annotated target sets for strict
acceptance. This is positive-clip event-coverage acceptance, not population
precision, a false-positive rate, or full-video frame accuracy.

For end-to-end video validation use `scripts/run_canonical_behavior_video.py`
with `--video`, `--pose-cache`, `--source-receipt`, a new `--output` directory and
`--mode all`. Verify the complete run receipt, all source frames, per-behavior
event counts, decoded event package durations and hashes. A full video without
complete human labels is an unreviewed demonstration, not an accuracy test.
Grooming/Rearing remain disabled by default; `--posture-mode v3_review` is an
explicit separate review-candidate lane.

## Remaining semantic limits

Isolation intentionally uses the minimum nearest-neighbor distance across the
input clip, as previously specified. Long-video segmentation of internal isolated
episodes needs a separate agreed contract; this repair does not reinterpret the
original sample criterion. Group locomotion still permits pauses with measured
synchronous support recorded separately. A pause termination policy requires
behavioral evidence and is not introduced as an arbitrary new threshold here.
Finite window context and fragmented visual IDs remain limitations. Visual IDs
must never be represented as verified biological identities without mTrack/RFID.

The retained v16 audit suite includes historical assertions and deferred
calibration/configuration questions. A green configured quality gate does not
mean those findings have all been resolved. In particular, direct construction
of `HeuristicParameters` and YAML-seeded configuration can have different legacy
defaults; consumers must use the sealed effective parameters for a run. This
change does not establish a new physical calibration or enable legacy posture
prediction. The historical IND-7 fixture currently fails its reachability guard
(no Locomotion is produced), before its missing-observation assertion; that
XFAIL must not be presented as evidence of fabricated movement in this run.

## Frozen-source validation

The local CI-equivalent quality gate passed: Ruff, compilation, repository
validation, package build, and **1257 tests passed, 20 skipped, 9 expected
failures**. The skipped tests require optional historical private artifacts;
the expected failures are retained legacy-audit assertions, not silently removed
or counted as passing tests. No test was disabled for this change.

An independent installation of the fresh wheel outside the checkout, using
`pip --no-deps --no-index --target`, matched all 110 package Python modules
byte-for-byte against the frozen source. Default and posture-review execution
closures contain 42 and 45 modules, respectively. Role withdrawal after changing
crop participants, the strict Dispersal duration boundary (150 versus 151 frames
at 30 fps), and the installed CLI help command passed. This check reused existing
dependencies; it is not a new GPU/model-environment installation.

The paired labeled-clip comparison uses the same 755 IDs and immutable input
hashes before and after this repair (baseline commit `588c7b1`). Machine-only
behavior coverage acceptance increased from **587/755 to 591/755**; all four
gains were Dispersal (**26/41 to 30/41**) with no lost sample in any class.
Exact complete-target-set acceptance remained **520/755**, including Dispersal
**13/41**. Every other class retained its prior hit count. The lineage-verified
30-clip Isolation subset remained **27/30**. Both formal runs completed with
zero sample errors and unchanged protected inputs. These counts do not modify
the historical reviewed A1/A2 report, establish role-direction accuracy, or
measure false positives on unannotated video.

The complete 16,614-frame, 1920-by-1080 source video was inferred and rendered
from its verified existing tracked cache. It produced 4,499 unreviewed
candidates (4,392 before repair). At the user's request, exhaustive clip export
was stopped and replaced with a fixed sample: shortest duration, median duration,
and latest endpoint for each emitted behavior, deduplicated to **35 clips across
12 behaviors**. All sampled videos decoded fully, satisfied their actual-FPS
duration policy, and matched the source trajectories and event annotations.
The complete rendered video also passed full decode and source-sidecar checks.
The interrupted exhaustive run is marked as a partial export, not complete;
1,312 previously completed packages were retained, six interrupted packages
excluded, and 24 additional selected clips exported. No new YOLO/SAM inference
was performed. The source video has no complete manual annotations, so these
are delivery checks, not a full-video accuracy measurement. Visual inspection
also retains crowded-label overlap and possible inherited background detections
as review limitations.
