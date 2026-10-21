import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { packageReviewMedia } from '../scripts/package-review-media.mjs';

const manifest = () => ({ completed: true, metrics: { total: 1 }, behaviors: [], items: [{
  id: 'one', sampleId: 'session/one', truth: 'stationary', truthName: '静止',
  videoUrl: '/video-files/example/clip.mp4', targetIds: [42], strictTop1: true,
}] });

test('packages every referenced video at its manifest URL and seals the copied bytes', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'review-package-'));
  try {
    const outputRoot = path.join(root, 'outputs');
    const destination = path.join(root, 'dist');
    mkdirSync(path.join(outputRoot, 'example'), { recursive: true });
    const media = Buffer.from('packaging fixture: not a playback validation');
    writeFileSync(path.join(outputRoot, 'example', 'clip.mp4'), media);
    const manifestPath = path.join(root, 'frozen.json');
    const save = (payload) => {
      writeFileSync(manifestPath, JSON.stringify(payload));
      return createHash('sha256').update(readFileSync(manifestPath)).digest('hex');
    };
    let digest = save(manifest());
    const receipt = await packageReviewMedia({ manifestPath, expectedSha256: digest, outputRoot, destination });
    assert.deepEqual(readFileSync(path.join(destination, 'video-files/example/clip.mp4')), media);
    assert.equal(receipt.media[0].sha256, createHash('sha256').update(media).digest('hex'));
    assert.deepEqual(JSON.parse(readFileSync(path.join(destination, 'review-data.json'))), manifest());
    await assert.rejects(packageReviewMedia({ manifestPath, expectedSha256: 'wrong', outputRoot,
      destination: path.join(root, 'bad-hash') }), /SHA-256/);
    const incomplete = manifest(); incomplete.completed = false; digest = save(incomplete);
    await assert.rejects(packageReviewMedia({ manifestPath, expectedSha256: digest, outputRoot,
      destination: path.join(root, 'incomplete') }), /completed/);
    const missing = manifest(); missing.items[0].videoUrl = '/video-files/missing.mp4'; digest = save(missing);
    await assert.rejects(packageReviewMedia({ manifestPath, expectedSha256: digest, outputRoot,
      destination: path.join(root, 'missing') }));
    assert.equal(existsSync(path.join(root, 'missing/review-data.json')), false);
    mkdirSync(path.join(outputRoot, 'directory.mp4'));
    const directory = manifest(); directory.items.push({ ...directory.items[0], id: 'two',
      videoUrl: '/video-files/directory.mp4' }); digest = save(directory);
    const invalidDestination = path.join(root, 'non-file');
    await assert.rejects(packageReviewMedia({ manifestPath, expectedSha256: digest, outputRoot,
      destination: invalidDestination }), /regular file/);
    assert.equal(existsSync(invalidDestination), false);
    const escape = manifest(); escape.items[0].videoUrl = '/video-files/../../outside.mp4'; digest = save(escape);
    await assert.rejects(packageReviewMedia({ manifestPath, expectedSha256: digest, outputRoot,
      destination: path.join(root, 'escape') }), /escapes/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
