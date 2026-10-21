const record = (value) => value && typeof value === 'object' && !Array.isArray(value);
const text = (value) => typeof value === 'string' && value.length > 0;
const ids = (value) => Array.isArray(value) && value.every((id) => Number.isInteger(id) && id >= 0)
  && new Set(value).size === value.length;

export function validateReviewData(payload) {
  const invalid = () => { throw new Error('审核清单无效或为空；请选择已完成的冻结清单'); };
  if (!record(payload) || !record(payload.metrics) || !Array.isArray(payload.behaviors)
      || !Array.isArray(payload.items) || payload.items.length === 0) invalid();
  if (!payload.behaviors.every((row) => record(row) && text(row.key) && text(row.name))) invalid();
  for (const key of ['total', 'strictTop1Rate', 'strictCorrect', 'success', 'errors',
    'strictOverallPrecision', 'overallPrecision', 'compatibleOverallPrecision', 'compatibleCorrect',
    'predictionCount', 'microPredictedCount', 'strictOverallRecall', 'overallRecall',
    'compatibleOverallRecall', 'compatibleTop1Rate']) {
    if (payload.metrics[key] != null && !Number.isFinite(payload.metrics[key])) invalid();
  }
  const seen = new Set();
  for (const item of payload.items) {
    if (!record(item) || !['id', 'sampleId', 'truth', 'truthName', 'videoUrl'].every((key) => text(item[key]))
        || !ids(item.targetIds) || typeof item.strictTop1 !== 'boolean' || seen.has(item.id)) invalid();
    seen.add(item.id);
    for (const key of ['predicted', 'predictedName', 'errorReason', 'compatibilityReason',
      'ruleSummary', 'ruleCue', 'ruleTitle']) {
      if (item[key] != null && typeof item[key] !== 'string') invalid();
    }
    for (const key of ['width', 'height', 'fps', 'frames', 'selectedTrackCount']) {
      if (item[key] != null && !Number.isFinite(item[key])) invalid();
    }
    if (item.predictedTargetIds !== undefined && !ids(item.predictedTargetIds)) invalid();
    if (item.candidateScores !== undefined && (!record(item.candidateScores)
        || !Object.values(item.candidateScores).every(Number.isFinite))) invalid();
    if (item.failedConditions !== undefined && (!Array.isArray(item.failedConditions)
        || !item.failedConditions.every(text))) invalid();
    if (item.auxiliaryEvents !== undefined && (!Array.isArray(item.auxiliaryEvents)
        || !item.auxiliaryEvents.every((event) => record(event) && ids(event.ids)
          && Number.isFinite(event.score) && text(event.behavior)
          && (event.behaviorName == null || typeof event.behaviorName === 'string')))) invalid();
  }
  for (const key of ['perBehavior', 'errorConfusion']) {
    if (payload.metrics[key] !== undefined && (!Array.isArray(payload.metrics[key])
        || !payload.metrics[key].every(record))) invalid();
  }
  for (const row of payload.metrics.perBehavior || []) {
    if (!text(row.key) || !text(row.name)) invalid();
    for (const [key, value] of Object.entries(row)) {
      if (!['key', 'name'].includes(key) && value != null && !Number.isFinite(value)) invalid();
    }
  }
  if ((payload.metrics.errorConfusion || []).some((row) => !text(row.pair) || !Number.isFinite(row.count))) invalid();
  return payload;
}

export async function loadReviewData(url, signal, fetcher = fetch) {
  const response = await fetcher(url, { signal });
  if (!response.ok) throw new Error(`${url} HTTP ${response.status}`);
  if (!response.headers.get('content-type')?.includes('application/json')) {
    throw new Error(`审核数据不存在或不是 JSON：${url}`);
  }
  return validateReviewData(await response.json());
}
