import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, realpathSync, writeFileSync, createReadStream, statSync } from 'node:fs';
import path from 'node:path';
import { validateReviewData } from '../src/review-data.js';
import { resolveValidationVideoPath } from '../src/video-paths.js';

async function sha256(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

export async function packageReviewMedia({ manifestPath, expectedSha256, outputRoot, destination }) {
  const raw = readFileSync(manifestPath);
  const digest = createHash('sha256').update(raw).digest('hex');
  if (!expectedSha256 || expectedSha256.toLowerCase() !== digest) {
    throw new Error('Explicit frozen review manifest SHA-256 does not match');
  }
  const data = validateReviewData(JSON.parse(raw.toString('utf8').replace(/^\uFEFF/, '')));
  if (data.completed !== true) throw new Error('Review manifest must explicitly declare completed: true');
  const root = realpathSync(outputRoot);
  const targetRoot = path.resolve(destination);
  const files = new Map();
  // Validate every source and destination before copying any private media.
  for (const item of data.items) {
    if (!item.videoUrl.startsWith('/video-files/') || !item.videoUrl.toLowerCase().endsWith('.mp4')) {
      throw new Error('Packaged review videos must use local /video-files/*.mp4 URLs');
    }
    const requested = decodeURIComponent(item.videoUrl.slice('/video-files/'.length));
    const { videoPath, allowed } = resolveValidationVideoPath(requested, root);
    if (!allowed) throw new Error('Video path escapes outputs root');
    const source = realpathSync(videoPath);
    if (!statSync(source).isFile()) throw new Error('Review video source must be a regular file');
    const target = path.resolve(targetRoot, 'video-files', requested);
    if (!source.startsWith(root + path.sep) || !target.startsWith(path.join(targetRoot, 'video-files') + path.sep)) {
      throw new Error('Video path escapes package boundary');
    }
    files.set(target, source);
  }
  const media = [];
  for (const [target, source] of files) {
    mkdirSync(path.dirname(target), { recursive: true });
    copyFileSync(source, target);
    media.push({ path: path.relative(targetRoot, target).split(path.sep).join('/'), sha256: await sha256(target) });
  }
  mkdirSync(targetRoot, { recursive: true });
  writeFileSync(path.join(targetRoot, 'review-data.json'), raw);
  const receipt = { mode: 'explicit_frozen_review', manifest_sha256: digest, item_count: data.items.length, media };
  writeFileSync(path.join(targetRoot, 'review-package-receipt.json'), JSON.stringify(receipt, null, 2));
  return receipt;
}
