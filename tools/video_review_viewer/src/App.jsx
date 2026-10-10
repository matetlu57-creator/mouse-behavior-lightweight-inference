import { useEffect, useMemo, useRef, useState } from 'react';

const EMPTY_DATA = { metrics: { total: 0, success: 0, errors: 0, strictTop1Rate: 0 }, behaviors: [], items: [] };
const REVIEW_DATA_FILES = {
  legacy: '/review-data.json',
  v6: '/review-data-v6.json',
  v7: '/review-data-v7.json',
  v8: '/review-data-v8.json',
  v9: '/review-data-v9.json',
  v10: '/review-data-v10.json',
  v11: '/review-data-v11.json',
};
const REVIEW_TITLES = {
  legacy: '验证集渲染视频 · 旧版',
  v6: '验证集渲染视频 · 层级版 v6',
  v7: '验证集渲染视频 · 目标 ID 复核版 v7',
  v8: '验证集渲染视频 · 群体手工特征版 v8',
  v9: '验证集渲染视频 · 预测目标 ID 修复版 v9',
  v10: '验证集渲染视频 · 预测 ID 与辅助轨迹修复版 v10',
  v11: '验证集渲染视频 · 兼容口径复核版 v11',
};

function formatClip(item) {
  const parts = item.sampleId.split('/');
  return parts[parts.length - 1] || item.sampleId;
}

function formatDuration(item) {
  if (!item.fps || !item.frames) return '—';
  const seconds = item.frames / item.fps;
  return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
}

function scoreRows(scores) {
  return Object.entries(scores || {})
    .sort(([, left], [, right]) => right - left)
    .slice(0, 5);
}

function scoreLabel(value) {
  return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
}

function percentLabel(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)}%`;
}

function metricWithCount(value, numerator, denominator) {
  if (value == null) return '—';
  if (numerator == null || denominator == null) return percentLabel(value);
  return `${percentLabel(value)} (${numerator}/${denominator})`;
}

function StatusBadge({ item, compact = false }) {
  const success = item.status === 'success';
  const compatibleOnly = success && !item.strictTop1 && item.compatibleTop1;
  const label = !success ? 'ERR' : compatibleOnly ? 'COMPAT' : item.strictTop1 ? 'STRICT' : 'PASS';
  return (
    <span className={`status-badge ${!success ? 'is-error' : compatibleOnly ? 'is-compatible' : 'is-success'} ${compact ? 'is-compact' : ''}`}>
      {label}
    </span>
  );
}

function Metric({ label, value, tone = '' }) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function App() {
  const [data, setData] = useState(EMPTY_DATA);
  const [loadError, setLoadError] = useState('');
  const [filter, setFilter] = useState('all');
  const [behavior, setBehavior] = useState('all');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [speed, setSpeed] = useState('1');
  const [videoError, setVideoError] = useState('');
  const videoRef = useRef(null);
  const reviewVersion = new URLSearchParams(window.location.search).get('version');
  const dataFile = REVIEW_DATA_FILES[reviewVersion] || REVIEW_DATA_FILES.v7;
  const reviewTitle = REVIEW_TITLES[reviewVersion] || REVIEW_TITLES.v7;

  useEffect(() => {
    fetch(dataFile)
      .then((response) => {
        if (!response.ok) throw new Error(`${dataFile} ${response.status}`);
        if (!response.headers.get('content-type')?.includes('application/json')) {
          throw new Error(`审核数据不存在或不是 JSON：${dataFile}`);
        }
        return response.json();
      })
      .then((payload) => {
        if (!payload || !Array.isArray(payload.items) || !Array.isArray(payload.behaviors) || !payload.metrics) {
          throw new Error(`审核数据结构无效：${dataFile}`);
        }
        setData(payload);
        setSelectedId(payload.items?.[0]?.id || '');
        setVideoError('');
      })
      .catch((error) => setLoadError(error.message));
  }, [dataFile]);

  const filteredItems = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return data.items.filter((item) => {
      if (filter !== 'all' && item.status !== filter) return false;
      if (behavior !== 'all' && item.truth !== behavior) return false;
      if (!normalizedQuery) return true;
      return [item.sampleId, item.truthName, item.predictedName, item.errorReason, ...(item.targetIds || [])]
        .join(' ')
        .toLowerCase()
        .includes(normalizedQuery);
    });
  }, [behavior, data.items, filter, query]);

  const selected = data.items.find((item) => item.id === selectedId) || filteredItems[0] || null;

  useEffect(() => {
    if (!filteredItems.some((item) => item.id === selectedId)) {
      setSelectedId(filteredItems[0]?.id || '');
    }
  }, [filteredItems, selectedId]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = Number(speed);
  }, [selected, speed]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const currentIndex = filteredItems.findIndex((item) => item.id === selected?.id);
      if (event.key === 'ArrowDown' || event.key === 'j') {
        event.preventDefault();
        const next = filteredItems[Math.min(filteredItems.length - 1, currentIndex + 1)];
        if (next) setSelectedId(next.id);
      }
      if (event.key === 'ArrowUp' || event.key === 'k') {
        event.preventDefault();
        const next = filteredItems[Math.max(0, currentIndex - 1)];
        if (next) setSelectedId(next.id);
      }
      if (event.key === ' ') {
        event.preventDefault();
        if (!videoRef.current) return;
        if (videoRef.current.paused) videoRef.current.play();
        else videoRef.current.pause();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [filteredItems, selected]);

  const counts = {
    all: data.items.length,
    success: data.items.filter((item) => item.status === 'success').length,
    error: data.items.filter((item) => item.status === 'error').length,
  };
  const compatibleReview = data.reviewVersion === 'v11-compatible';

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">MB</div>
          <div>
            <div className="brand-name">Mouse Behavior</div>
            <div className="brand-subtitle">Validation Review</div>
          </div>
        </div>
        <div className="topbar-title">
          <span>{reviewTitle}</span>
          <small>验证集 · 行为识别 · 目标 ID 复核</small>
        </div>
        <div className="topbar-stats">
          <span className="topbar-count"><b>{data.metrics.total}</b> clips · {compatibleReview ? '兼容正确' : '严格正确'} {data.metrics.success} / 错误 {data.metrics.errors}</span>
          <span className="topbar-rate"><b>{compatibleReview ? data.metrics.compatibleTop1Rate : data.metrics.strictTop1Rate}%</b> {compatibleReview ? '兼容' : '严格'} Top‑1</span>
        </div>
      </header>

      <div className="layout">
        <aside className="library-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">EPISODES</span>
              <h1>行为片段</h1>
            </div>
            <span className="result-count">{filteredItems.length}</span>
          </div>

          <div className="filter-tabs" role="tablist" aria-label="结果筛选">
            {[
              ['all', '全部'],
              ['success', compatibleReview ? '兼容正确' : '严格正确'],
              ['error', '错误'],
            ].map(([key, label]) => (
              <button
                key={key}
                className={filter === key ? 'tab-button is-active' : 'tab-button'}
                onClick={() => setFilter(key)}
                role="tab"
                aria-selected={filter === key}
              >
                {label}<span>{counts[key]}</span>
              </button>
            ))}
          </div>

          <label className="search-field">
            <span>搜索</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="行为、ID 或片段名" />
          </label>

          <label className="select-field">
            <span>标签行为</span>
            <select value={behavior} onChange={(event) => setBehavior(event.target.value)}>
              <option value="all">全部行为</option>
              {data.behaviors.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}
            </select>
          </label>

          <div className="list-hint">上下键 / J K 切换 · 空格播放暂停</div>
          <div className="episode-list">
            {filteredItems.map((item) => (
              <button
                className={`episode-card ${selected?.id === item.id ? 'is-selected' : ''}`}
                key={item.id}
                onClick={() => setSelectedId(item.id)}
              >
                <div className="episode-card-topline">
                  <StatusBadge item={item} compact />
                  <span className="episode-behavior">{item.truthName}</span>
                  <span className="episode-duration">{formatDuration(item)}</span>
                </div>
                <div className="episode-title">{formatClip(item)}</div>
                <div className="episode-meta">
                  <span>{item.recordingSession || 'validation'}</span>
                  <span>标签 {item.truthName}</span>
                  <span className={item.status === 'success' ? 'meta-good' : 'meta-bad'}>
                    {item.strictTop1 ? '严格正确' : item.status === 'success' ? '兼容正确' : `识别 ${item.predictedName || '未识别'}`}
                  </span>
                </div>
              </button>
            ))}
            {!filteredItems.length && <div className="empty-state">没有符合筛选条件的片段。</div>}
          </div>
        </aside>

        <main className="review-panel">
          {loadError && <div className="load-error">无法读取审核数据：{loadError}</div>}
          {!selected && !loadError && <div className="empty-review">正在读取渲染清单…</div>}
          {selected && (
            <>
              <div className="review-header">
                <div>
                  <div className="breadcrumb">VALIDATION / {selected.sourceKind === 'error' ? 'ERROR' : 'BEHAVIOR EXAMPLE'}</div>
                  <h2>{selected.truthName} <span>·</span> {formatClip(selected)}</h2>
                  <p>{selected.sampleId}</p>
                </div>
                <StatusBadge item={selected} />
              </div>

              {data.metrics.perBehavior?.length > 0 && (
                <section className="metrics-section">
                  <div className="metrics-heading">
                    <div>
                      <span className="eyebrow">VALIDATION METRICS</span>
                      <h3>各行为精确率、召回率与 Top‑1</h3>
                    </div>
                    <span>严格口径要求行为与 ID 集合完全匹配；兼容口径另计目标 ID 包含与允许的同层双标签</span>
                  </div>
                  <div className="metrics-table-wrap">
                    <table className="metrics-table">
                      <thead><tr><th>行为</th><th>P 严格</th><th>P 兼容</th><th>R 严格</th><th>R 兼容</th><th>严格 Top‑1</th><th>兼容 Top‑1</th><th>严格 TP / 预测数 / 标签数</th></tr></thead>
                      <tbody>
                        {data.metrics.perBehavior.map((row) => (
                          <tr key={row.key} className={row.key === selected.truth ? 'is-current' : ''}>
                            <th scope="row">{row.name}</th>
                            <td>{metricWithCount(row.precisionStrict ?? row.precision, row.strictPrecisionCorrect ?? row.truePositives, row.predictedCount)}</td>
                            <td>{metricWithCount(row.precisionCompatible ?? row.precision, row.compatiblePrecisionCorrect ?? row.truePositives, row.predictedCount)}</td>
                            <td>{metricWithCount(row.recallStrict ?? row.recall, row.strictRecallCorrect ?? row.truePositives, row.support)}</td>
                            <td>{metricWithCount(row.recallCompatible ?? row.recall, row.compatibleRecallCorrect ?? row.truePositives, row.support)}</td>
                            <td>{percentLabel(row.strictTop1)}</td>
                            <td>{percentLabel(row.dualTop1 ?? row.compatibleTop1)}</td>
                            <td>{row.strictPrecisionCorrect ?? row.truePositives}/{row.predictedCount}/{row.support}</td>
                          </tr>
                        ))}
                      </tbody>
                      <tfoot><tr><th>总体</th><td>{metricWithCount(data.metrics.strictOverallPrecision ?? data.metrics.overallPrecision, data.metrics.strictCorrect ?? data.metrics.success, data.metrics.predictionCount ?? data.metrics.microPredictedCount)}</td><td>{metricWithCount(data.metrics.compatibleOverallPrecision ?? data.metrics.overallPrecision, data.metrics.compatibleCorrect ?? data.metrics.success, data.metrics.predictionCount ?? data.metrics.microPredictedCount)}</td><td>{metricWithCount(data.metrics.strictOverallRecall ?? data.metrics.overallRecall, data.metrics.strictCorrect ?? data.metrics.success, data.metrics.total)}</td><td>{metricWithCount(data.metrics.compatibleOverallRecall ?? data.metrics.overallRecall, data.metrics.compatibleCorrect ?? data.metrics.success, data.metrics.total)}</td><td>{metricWithCount(data.metrics.strictTop1Rate, data.metrics.strictCorrect ?? data.metrics.success, data.metrics.total)}</td><td>{metricWithCount(data.metrics.compatibleTop1Rate, data.metrics.compatibleCorrect ?? data.metrics.success, data.metrics.total)}</td><td>{data.metrics.strictCorrect ?? data.metrics.success}/{data.metrics.predictionCount ?? data.metrics.microPredictedCount}/{data.metrics.total}</td></tr></tfoot>
                    </table>
                  </div>
                  <div className="metric-note">当前验证集 {data.metrics.total} 条：严格正确 {data.metrics.strictCorrect ?? data.metrics.success} 条；兼容口径正确 {data.metrics.compatibleCorrect ?? data.metrics.success} 条。P/R 均分别展示严格与兼容口径；兼容口径包含目标 ID 集合包含关系及允许的同层双标签。{compatibleReview ? '接近↔鼻头/鼻尾接触要求同一 ID 对且接触累计时长 >0.5 秒；Together↔接触要求同一 ID 对。' : ''}该集合此前已用于诊断，不是新的盲测。</div>
                  {data.metrics.errorConfusion?.length > 0 && (
                    <div className="confusion-summary"><strong>错误去向：</strong>{data.metrics.errorConfusion.map((item) => `${item.pair} ${item.count} 条`).join('；')}</div>
                  )}
                </section>
              )}

              <div className="review-grid">
                <section className="video-section">
                  <div className="video-frame">
                    <video
                      ref={videoRef}
                      key={selected.videoUrl}
                      src={selected.videoUrl}
                      controls
                      preload="metadata"
                      onError={() => setVideoError(`视频无法加载：${selected.videoUrl}`)}
                      onLoadStart={() => setVideoError('')}
                    >
                      您的浏览器不支持视频播放。
                    </video>
                    <div className="video-caption">
                      <span>{selected.truthName} · {selected.predictedName || '未识别'}</span>
                      <span>{selected.width}×{selected.height} · {selected.fps} fps · {selected.frames} frames</span>
                    </div>
                  </div>
                  {videoError && <div className="load-error media-error">{videoError}</div>}
                  <div className="video-toolbar">
                    <span>播放速度</span>
                    <select value={speed} onChange={(event) => setSpeed(event.target.value)}>
                      <option value="0.5">0.5×</option>
                      <option value="1">1×</option>
                      <option value="1.5">1.5×</option>
                      <option value="2">2×</option>
                    </select>
                    <span className="toolbar-note">当前片段使用已有行为叠加渲染</span>
                  </div>
                </section>

                <aside className="detail-panel">
                  <div className="detail-heading">
                    <span className="eyebrow">RESULT</span>
                    <h3>识别结果</h3>
                  </div>
                  <div className="result-pair">
                    <div className="result-block truth">
                      <span>标签行为</span>
                      <strong>{selected.truthName}</strong>
                      <small>{selected.truth}</small>
                    </div>
                    <div className="result-block prediction">
                      <span>识别行为</span>
                      <strong>{selected.predictedName || '未识别'}</strong>
                      <small>{selected.predicted || '—'}</small>
                    </div>
                  </div>

                  {(selected.ruleSummary || selected.ruleCue) && (
                    <div className="rule-box">
                      <span className="eyebrow">BEHAVIOR RULE</span>
                      <strong>{selected.ruleTitle || '当前行为口径'}</strong>
                      {selected.ruleSummary && <p>{selected.ruleSummary}</p>}
                      {selected.ruleCue && <small>{selected.ruleCue}</small>}
                    </div>
                  )}

                  {selected.auxiliaryEvents?.length > 0 && (
                    <div className="detail-section auxiliary-section">
                      <div className="score-heading">
                        <span className="eyebrow">OTHER TRACKS</span>
                        <span>按层级显示</span>
                      </div>
                      <div className="auxiliary-list">
                        {selected.auxiliaryEvents.slice(0, 12).map((event, index) => (
                          <div className="auxiliary-row" key={`${event.behavior}-${event.ids.join('-')}-${index}`}>
                            <span>{event.behaviorName || event.behavior}</span>
                            <small>{event.ids.length ? event.ids.map((id) => `ID ${id}`).join(' · ') : '群体'} · {event.score.toFixed(2)} s</small>
                          </div>
                        ))}
                        {selected.auxiliaryEvents.length > 12 && (
                          <small className="auxiliary-more">另有 {selected.auxiliaryEvents.length - 12} 个候选；视频框按“社交行为 → 群体行为 → 个体行为”优先级显示。</small>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="status-grid">
                    <Metric label="目标 ID 包含命中" value={selected.targetHit ? '是' : '否'} tone={selected.targetHit ? 'good' : 'bad'} />
                    <Metric label="ID 集合完全一致" value={selected.exactTargetHit ? '是' : '否'} tone={selected.exactTargetHit ? 'good' : 'bad'} />
                    <Metric label="严格 Top‑1" value={selected.strictTop1 ? '是' : '否'} tone={selected.strictTop1 ? 'good' : 'bad'} />
                    <Metric label="兼容 Top‑1" value={selected.compatibleTop1 ? '是' : '否'} tone={selected.compatibleTop1 ? 'good' : 'bad'} />
                  </div>

                  <div className="detail-section">
                    <span className="eyebrow">标签目标 ID</span>
                    <div className="id-pills">
                      {selected.targetIds.length ? selected.targetIds.map((id) => <span key={id}>ID {id}</span>) : <span className="muted">未提供标签 ID</span>}
                    </div>
                  </div>

                  <div className="detail-section">
                    <span className="eyebrow">预测目标 ID</span>
                    <div className="id-pills">
                      {selected.predictedTargetIds?.length ? selected.predictedTargetIds.map((id) => <span key={id}>ID {id}</span>) : <span className="muted">未选择目标 ID</span>}
                    </div>
                  </div>

                  <div className={`reason-box ${selected.status === 'success' ? 'is-success' : 'is-error'}`}>
                    <span className="eyebrow">REVIEW NOTE</span>
                    <strong>{selected.strictTop1 ? '严格正确：行为与 ID 完全匹配' : selected.compatibleTop1 ? '兼容口径命中（原始预测未改写）' : `${selected.truthName} 被识别为 ${selected.predictedName || '未识别'}`}</strong>
                    <p>{selected.compatibilityReason || selected.errorReason || '—'}</p>
                  </div>

                  {selected.status === 'error' && selected.failedConditions?.length > 0 && (
                    <div className="condition-box">
                      <span className="eyebrow">FEATURE CONDITION REVIEW</span>
                      <strong>未满足的特征条件 / 错误归因</strong>
                      <ul>{selected.failedConditions.map((condition, index) => <li key={`${index}-${condition}`}>{condition}</li>)}</ul>
                      <small>数值来自目标 ID 轨迹的事后特征核对；没有作为推理输入。复合时序规则无法由单一汇总值确认时已标明。</small>
                    </div>
                  )}

                  <div className="detail-section score-section">
                    <div className="score-heading"><span className="eyebrow">CANDIDATE SCORES</span><span>Top 5</span></div>
                    {(() => {
                      const rows = scoreRows(selected.candidateScores);
                      const maxScore = Math.max(...rows.map(([, value]) => value), 1);
                      return rows.map(([key, value]) => (
                        <div className="score-row" key={key}>
                          <span>{key}</span><span className="score-bar"><i style={{ width: `${Math.min(100, (value / maxScore) * 100)}%` }} /></span><b>{scoreLabel(value)}</b>
                        </div>
                      ));
                    })()}
                  </div>

                  <div className="detail-footer">
                    <span>轨迹数 {selected.selectedTrackCount || '—'}</span>
                    <span>片段时长 {formatDuration(selected)}</span>
                  </div>
                </aside>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
