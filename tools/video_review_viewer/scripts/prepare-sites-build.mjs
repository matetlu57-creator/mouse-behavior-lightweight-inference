#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { packageReviewMedia } from './package-review-media.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const index = path.join(dist, "client", "index.html");
const worker = path.join(root, "worker", "index.js");
const hosting = path.join(root, ".openai", "hosting.json");

for (const file of [index, worker, hosting]) {
  if (!existsSync(file)) throw new Error("Missing Sites build input: " + file);
}

mkdirSync(path.join(dist, "server"), { recursive: true });
mkdirSync(path.join(dist, ".openai"), { recursive: true });
copyFileSync(worker, path.join(dist, "server", "index.js"));
copyFileSync(hosting, path.join(dist, ".openai", "hosting.json"));

if (process.env.REVIEW_MANIFEST) {
  await packageReviewMedia({
    manifestPath: process.env.REVIEW_MANIFEST,
    expectedSha256: process.env.REVIEW_MANIFEST_SHA256,
    outputRoot: process.env.REVIEW_OUTPUT_ROOT || path.resolve(root, '../../outputs'),
    destination: path.join(dist, 'client'),
  });
} else {
  writeFileSync(path.join(dist, 'client', 'review-package-receipt.json'), JSON.stringify({
    mode: 'shell_only', scientific_results_included: false,
  }));
}

console.log("Prepared Sites build: dist/server/index.js and dist/.openai/hosting.json");
