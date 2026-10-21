# ADR 0002: Train-only physical ranges before layered behavior ranking

- Status: experimental; not accepted as a 90% classifier
- Date: 2026-09-24

## Context

The previous clip-level hierarchy promoted every positive group candidate above
pair and individual behavior. On the existing 274-clip validation split, its
strict behavior-plus-target-ID Top-1 was 20.80%. The train split showed the
failure was not only score calibration: the original group proposal list
contained a correct Huddling target set in just 15/53 training clips and a
correct Social clustering target set in 15/22. A ranker cannot recover a
missing target set.

The 679 training target trajectories show different physical regimes. Median
nearest-neighbor distance was 74.6 cm for Isolation and 44.4 cm for Static;
both can have low speed, so motion alone does not distinguish them. Median
speed was about 4 cm/s for both Huddling and Social clustering, so a speed-only
gate is likewise insufficient. Target-member visibility and displacement carry
additional information. Measured Huddling center-to-center distances exceeded
the old 3–5 cm contact proposal cap; a center-distance gate is not the same as
an anatomical body-width measurement.

## Decision

1. Keep visual tracking and RFID identity resolution outside behavior rules.
   Candidate member IDs are observed visual track IDs only.
2. Offer a second, train-calibrated Huddling proposal at up to 10 cm
   center distance (approximately 1.25 of the assumed 8 cm body length).
   A tracklet absent for much of a group episode can enter the observed union
   only if it is spatially connected while visible. Keep the stable core as a
   separate proposal; do not relabel or infer a missing ID.
3. Fit one-dimensional, inspectable physical ranges from labeled **training
   targets only**: nearest distance for Isolation versus Static; median speed
   for Static versus Walking; group distance, member visibility, and
   displacement for Huddling versus Social clustering. Weak contrasts do not
   become hard exclusion rules.
4. Select the strength of range penalties and a conditional layer bonus with
   leave-one-recording-session-out training folds. Group candidate membership
   weighting is searched but may remain zero if held-out training clips do not
   support it. A group candidate is never automatically the clip's primary
   label just because it occupies the highest layer.
5. Freeze parameters before validation inference. The validation command
   projects the existing manifest to trajectory path/FPS and opaque row ID,
   writes and hashes predictions, then reads labels for scoring. The original
   validation set has been used in prior experiments, so its result is a
   reused-set diagnostic, not a fresh blind test.

## Alternatives considered

- **Unconditional group > social > individual resolution:** rejected for
  clip-level Top-1 because legitimate secondary group context frequently
  masks the annotated target pair or individual behavior.
- **Increasing all group distances without candidate/ID audit:** rejected;
  this creates broad false groups and does not explain member coverage.
- **Deep induced decision tree on all training snippets:** rejected after
  earlier experiments showed a large training/recording-held-out gap.
- **Rewarding ever-larger group member lists:** searched with a bound, but
  selection also tracks exact-ID accuracy to expose over-inclusion.
- **Treating every matching target group as a positive in the listwise ranker:**
  tested on the same cached training candidates. Its best session-held-out
  joint Top-1 was 43.2%, below the 43.6% baseline; full-training accuracy was
  about 41.4%, below 57.0%. The experiment was not sent to validation and its
  extra optimizer was removed from the production path.

## Consequences

The rule system gains auditable, physical-unit thresholds and separate
candidate-coverage versus ranking diagnostics. It does not guarantee 90%
accuracy: fragmented visual IDs, incomplete group proposals, overlapping
behaviors in a single clip, and imperfect annotation scope remain limiting.
Generated caches, predictions, and reports stay under ignored `outputs/`.

## 2026-09-24 rule refinement

The user clarified that Together means two resting mice, while Huddling means a
resting group of at least three. Therefore the pair rule now requires both
members to remain within the calibrated Static speed band and uses an 8 cm
maximum pair distance. The training Together pair-distance median was 6.33 cm
and its 75th percentile was 7.52 cm. Huddling now requires at least three
spatially close members, group mean speed at most 5 cm/s, high-motion members
at most 20%, and at least 1 second of evidence; short identity dropouts remain
bridgeable. A qualified social/group event remains the primary rendered layer
for its own members while the lower individual Static event is retained.

After regenerating the training candidate cache and refitting on the original
679 training clips, the fitted training joint Top-1 was 49.48% and the
recording-session holdout was 45.10%. The existing 274-clip validation set was
then inferred from trajectories and visual IDs only; after the prediction
receipt was written, its strict joint Top-1 was 40.51% (macro 39.53%). Huddling
was 5/25 (20.00%), Together 3/26 (11.54%), Static 4/27 (14.81%), and Social
clustering 13/34 (38.24%). This modest overall change does not meet the 90%
goal. The validation set has been used in earlier experiments, so this remains
a reused-set diagnostic rather than a fresh blind test.

Reproduce with `scripts/audit_training_ranges.py`,
`scripts/calibrate_existing_split.py train`, and
`scripts/calibrate_empirical_ranges.py train` followed by `validate` using the
registered 679/274 manifests. Do not use validation labels to revise the
parameters and still call the same validation run blind.

## 2026-09-25 Isolation and Together/Huddling distance alignment

Isolation now measures, per frame, the target's mean distance to every other
currently visible mouse, rather than only its nearest neighbor. The threshold
is calibrated from training-only per-clip medians for Isolation versus Static:
77.16 cm maximizes balanced accuracy in the full training set (78.72%
sensitivity, 78.95% specificity). Leave-one-recording-session-out balanced
accuracy is only about 63.8%, so this boundary is weak and must not be described
as a reliable separator. The runtime rule still requires a per-frame mean at
or above 77 cm for at least 10 seconds and excludes members of the main group.
Ten of 47 training Isolation clips are shorter than that minimum duration.

Together no longer has an independent distance scale: its pair-distance gate
uses the same effective cutoff as Huddling. In the offline trajectory adapter
this is `min(mouse_width_cm * huddle_width_multiplier, huddle_distance_cm)`;
the video output layer uses `group.huddle_distance_cm` (5 cm by default). The
existing 8 cm Together threshold was removed because it described a looser
state than Huddling. Both rules retain their distinct participant-count and
motion requirements.

The next validation run must be regenerated only after the training candidate
cache, physical-range model, and ranker have been rebuilt from the train split.
Validation inference remains trajectory-and-track-ID-only; its labels are read
only after the prediction receipt is written. The validation set has already
been examined in prior experiments and remains a reused-set diagnostic.

The frozen run produced training session-holdout joint Top-1 of 47.57% (macro
41.50%). On the reused 274-clip validation set, strict behavior-plus-target-ID
Top-1 was 39.05% (macro 37.17%), down from the prior 40.51% (macro 39.53%).
Static rules emitted individual Static evidence for 26/27 annotated targets,
but strict Top-1 was only 2/27; the ranker selected Social clustering 15 times
and Isolation 7 times for Static labels. Together strict Top-1 was 1/26, with
18/26 labeled clips ranked as Social clustering. This is evidence that the
current weak classes are dominated by candidate ranking/overlap, not simply
missing individual Static detections. Do not tune these values against the
validation labels and reuse the same score as blind validation.

## Superseding decision: restore the better-scoring baseline

The mean-distance Isolation and shared Together/Huddling cutoff trial above
reduced reused-validation strict joint Top-1 from 40.51% to 39.05%, with a
particularly large drop for Together (11.54% to 3.85%) and Static (14.81% to
7.41%). The active heuristic therefore returns to nearest-neighbor Isolation
(20 cm seed, 10 s minimum; train profile bank 15–50 cm) and an independent
Together pair-distance rule (8 cm seed), while Huddling retains its own 5 cm
group-distance threshold. The lower-scoring experiment artifacts remain
available for comparison; neither validation result is a fresh blind estimate.
