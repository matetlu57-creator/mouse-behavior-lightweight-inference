import assert from 'node:assert/strict';
import test from 'node:test';
import { loadReviewData, validateReviewData } from '../src/review-data.js';

export const fixture = () => ({
  completed: true, metrics: { total: 1, strictTop1Rate: 0 },
  behaviors: [{ key: 'stationary', name: '静止' }],
  items: [{ id: 'one', sampleId: 'session/one', truth: 'stationary', truthName: '静止',
    videoUrl: '/video-files/example/clip.mp4', targetIds: [42], strictTop1: false,
    status: 'success', compatibleTop1: true }],
});

test('validates frozen data without rewriting historical status or metrics', () => {
  const payload = fixture();
  assert.equal(validateReviewData(payload), payload);
  assert.equal(payload.items[0].strictTop1, false);
  assert.equal(payload.items[0].status, 'success');
});

test('rejects empty and malformed nested manifests before rendering', () => {
  const changes = [
    (p) => { p.items = []; }, (p) => { delete p.items[0].sampleId; },
    (p) => { p.items[0].targetIds = '42'; }, (p) => { p.items[0].auxiliaryEvents = [{ ids: null }]; },
    (p) => { p.items[0].candidateScores = { attack: 'bad' }; },
    (p) => { p.metrics.perBehavior = 'bad'; }, (p) => { p.items.push(p.items[0]); },
    (p) => { p.items[0].failedConditions = 'bad'; },
    (p) => { p.items[0].predictedName = {}; }, (p) => { p.metrics.total = {}; },
    (p) => { p.metrics.perBehavior = [{ key: 'x', name: {} }]; },
    (p) => { p.metrics.errorConfusion = [{ pair: {}, count: 1 }]; },
  ];
  for (const change of changes) {
    const payload = fixture(); change(payload);
    assert.throws(() => validateReviewData(payload), /清单无效/);
  }
});

test('load errors remain recoverable for unavailable, HTML, and invalid JSON responses', async () => {
  for (const response of [
    new Response('missing', { status: 404 }),
    new Response('<html/>', { headers: { 'content-type': 'text/html' } }),
    new Response('{broken', { headers: { 'content-type': 'application/json' } }),
  ]) {
    await assert.rejects(loadReviewData('/review-data.json', undefined, async () => response));
  }
  const controller = new AbortController(); controller.abort();
  await assert.rejects(loadReviewData('http://127.0.0.1:1/', controller.signal), { name: 'AbortError' });
});
