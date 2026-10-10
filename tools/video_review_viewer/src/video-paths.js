import path from "node:path";

const LEGACY_MEDIA_PREFIX = "video_viewer_media/";

export function resolveValidationVideoPath(requested, outputRoot) {
  const isLegacyMedia = requested.startsWith(LEGACY_MEDIA_PREFIX);
  const root = isLegacyMedia
    ? path.resolve(outputRoot, "exported_dataset_calibration")
    : path.resolve(outputRoot);
  const videoPath = path.resolve(root, requested);

  return {
    videoPath,
    allowed: videoPath.startsWith(`${root}${path.sep}`),
  };
}
