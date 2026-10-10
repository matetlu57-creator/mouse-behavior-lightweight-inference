import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { resolveValidationVideoPath } from "../src/video-paths.js";

const outputRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../outputs");

test("serves legacy v7 viewer media from the exported calibration root", () => {
  const { videoPath, allowed } = resolveValidationVideoPath(
    "video_viewer_media/validation_review_20260928_v1/rendered/clip.mp4",
    outputRoot,
  );

  assert.equal(allowed, true);
  assert.equal(
    videoPath,
    path.join(
      outputRoot,
      "exported_dataset_calibration",
      "video_viewer_media",
      "validation_review_20260928_v1",
      "rendered",
      "clip.mp4",
    ),
  );
});

test("keeps current output-root video paths available", () => {
  const { videoPath, allowed } = resolveValidationVideoPath(
    "collective_state_features_20260928_v7/rendered_validation_v10/clip.mp4",
    outputRoot,
  );

  assert.equal(allowed, true);
  assert.equal(
    videoPath,
    path.join(
      outputRoot,
      "collective_state_features_20260928_v7",
      "rendered_validation_v10",
      "clip.mp4",
    ),
  );
});

test("rejects a legacy media path that escapes its media root", () => {
  const { allowed } = resolveValidationVideoPath(
    "video_viewer_media/../../outside/clip.mp4",
    outputRoot,
  );

  assert.equal(allowed, false);
});
