"use strict";

const $ = (id) => document.getElementById(id);
const reasons = {
  passed: "判定正确", not_detected: "未检出目标行为", other_ids: "检出其他 ID",
  short_coverage: "目标行为覆盖不足", other_behavior: "目标被判为其他行为",
  incomplete_evidence: "证据不完整", unavailable: "记录不可用"
};
const state = { catalogue: null, cohort: "B", behavior: "Huddling", status: "failed", reason: "all", query: "", rows: [], selected: null, detail: null, controller: null, catalogueController: null, frames: new Map(), targetIds: new Set(), frameCallback: null };
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const percent = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
const idsText = (ids) => ids.length ? ids.map((id) => `ID ${id}`).join("、") : "无";
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const numericIds = (value) => Array.isArray(value) && value.every((id) => Number.isInteger(id));
const node = (tag, text, className) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; };

function validClip(clip) {
  return record(clip) && typeof clip.id === "string" && typeof clip.clip_id === "string" && ["A", "B", "C"].includes(clip.dataset)
    && typeof clip.behavior === "string" && typeof clip.label === "string" && numericIds(clip.target_ids)
    && typeof clip.correct === "boolean" && typeof clip.reason_code === "string" && typeof clip.reason_label === "string"
    && typeof clip.summary === "string" && typeof clip.prediction_text === "string" && typeof clip.in_union === "boolean"
    && finite(clip.duration_s) && clip.duration_s >= 0 && finite(clip.fps) && clip.fps > 0
    && Number.isInteger(clip.frame_count) && clip.frame_count > 0 && typeof clip.source_video === "string";
}

function validateCatalogue(data) {
  if (!record(data) || data.schema !== "behavior_error_catalogue_v1" || !finite(data.generated_at)
    || typeof data.source_hash !== "string" || typeof data.source_code_matches !== "boolean" || !Array.isArray(data.changed_sources)
    || !record(data.cohorts) || !Array.isArray(data.clips) || !data.clips.every(validClip)) throw new Error("评测目录格式不符合约定。");
  for (const key of Object.keys(data.cohorts)) {
    const cohort = data.cohorts[key];
    if (!record(cohort) || typeof cohort.label !== "string" || ![cohort.total, cohort.correct, cohort.failed].every(Number.isInteger)
      || !(cohort.accuracy === null || finite(cohort.accuracy)) || !Array.isArray(cohort.behaviors)
      || !cohort.behaviors.every((item) => record(item) && typeof item.behavior === "string" && typeof item.label === "string"
        && [item.total, item.correct, item.failed].every(Number.isInteger) && (item.accuracy === null || finite(item.accuracy)) && record(item.error_counts))) throw new Error("数据集统计格式不符合约定。");
  }
  return data;
}

function validateDetail(data) {
  if (!record(data) || data.schema !== "behavior_error_clip_v1" || !validClip(data.clip)
    || typeof data.reason_detail !== "string" || typeof data.classification_note !== "string"
    || !(data.true_target_coverage === null || finite(data.true_target_coverage)) || !numericIds(data.best_group_ids)
    || !(data.best_group_coverage === null || finite(data.best_group_coverage))
    || !record(data.video) || typeof data.video.url !== "string" || ![data.video.width, data.video.height, data.video.fps].every((v) => finite(v) && v > 0)
    || !Number.isInteger(data.video.frame_count) || data.video.frame_count < 1 || !Array.isArray(data.frames)
    || !data.frames.every((frame) => record(frame) && Number.isInteger(frame.frame) && finite(frame.time) && Array.isArray(frame.detections)
      && frame.detections.every((d) => record(d) && Number.isInteger(d.id) && Array.isArray(d.box) && d.box.length === 4 && d.box.every(finite) && typeof d.target === "boolean" && typeof d.candidate === "boolean"))
    || !Array.isArray(data.observations) || !data.observations.every((item) => record(item) && Number.isInteger(item.id)
      && Number.isInteger(item.observed_frames) && finite(item.observed_fraction)
      && (item.first_frame === null || Number.isInteger(item.first_frame)) && (item.last_frame === null || Number.isInteger(item.last_frame)))
    || !record(data.target_candidate_scores) || !Object.values(data.target_candidate_scores).every(finite)) throw new Error("样本证据格式不符合约定。");
  if (data.huddle_gate !== null) {
    const gate = data.huddle_gate;
    if (!record(gate) || ![gate.common_endpoint_span_s, gate.common_endpoint_coverage_upper_bound, gate.compact_total_s].every(finite)
      || !Array.isArray(gate.members) || !gate.members.every((m) => record(m) && Number.isInteger(m.id) && finite(m.observed_fraction) && finite(m.held_fraction))) throw new Error("聚集证据格式不符合约定。");
  }
  const mediaUrl = new URL(data.video.url, window.location.href);
  if (mediaUrl.origin !== window.location.origin || !["http:", "https:"].includes(mediaUrl.protocol)) throw new Error("视频地址不符合本地服务约定。");
  return data;
}

async function fetchJson(url, controller, validate) {
  let timedOut = false;
  const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, 20000);
  try {
    const response = await fetch(url, { signal: controller.signal, headers: { Accept: "application/json" }, cache: "no-store" });
    if (!response.ok) throw new Error(`服务返回 HTTP ${response.status}，请重试。`);
    if (!/\bapplication\/json\b/i.test(response.headers.get("content-type") || "")) throw new Error("服务未返回 JSON 数据，请检查本地服务地址。");
    return validate(await response.json());
  } catch (error) {
    if (timedOut) throw new Error("请求超过 20 秒，请重试。");
    if (error instanceof SyntaxError) throw new Error("服务返回的 JSON 无法解析。");
    throw error;
  } finally { window.clearTimeout(timeout); }
}

function showCatalogueState(message, error = false) {
  $("catalogue-message").textContent = message;
  $("catalogue-state").hidden = !message;
  $("catalogue-state").classList.toggle("error", error);
  $("retry-catalogue").hidden = !error;
}

async function loadCatalogue() {
  state.catalogueController?.abort();
  const controller = new AbortController();
  state.catalogueController = controller;
  showCatalogueState("正在读取评测记录…");
  try {
    const catalogue = await fetchJson("/api/catalogue", controller, validateCatalogue);
    if (state.catalogueController !== controller) return;
    state.catalogue = catalogue;
    showCatalogueState("");
    const cohortSelect = $("cohort");
    cohortSelect.replaceChildren();
    for (const [key, c] of Object.entries(catalogue.cohorts)) {
      const opt = node("option", `${c.label} · ${c.full_total ?? c.total} 个样本`);
      opt.value = key;
      cohortSelect.append(opt);
    }
    if (catalogue.default_cohort && catalogue.cohorts[catalogue.default_cohort]) state.cohort = catalogue.default_cohort;
    else if (!catalogue.cohorts[state.cohort]) state.cohort = Object.keys(catalogue.cohorts)[0] || "B";
    cohortSelect.value = state.cohort;
    if (catalogue.default_status) { state.status = catalogue.default_status; $("status-filter").value = state.status; }
    $("source-warning").hidden = catalogue.source_code_matches;
    $("source-warning").textContent = catalogue.source_code_matches ? "" : "当前源码与评测时的源码不一致。本页准确率属于保存的评测记录，不能视为当前源码的验证结果。";
    const stamp = new Date(catalogue.generated_at * 1000);
    $("provenance").textContent = `评测目录生成：${stamp.toLocaleString("zh-CN")} · 来源指纹：${catalogue.source_hash}`;
    renderCohort();
  } catch (error) {
    if (state.catalogueController !== controller || error.name === "AbortError") return;
    showCatalogueState(`无法加载评测记录：${error.message}`, true);
    $("behavior-summary").textContent = "评测记录暂不可用；可在上方重新加载。";
  }
}

function behaviorRows() {
  const isBcUnion = state.cohort === "union" && Boolean(state.catalogue.cohorts.union?.is_bc_union);
  return state.catalogue.clips.filter((clip) => (isBcUnion ? (clip.dataset === "B" || clip.dataset === "C") : (state.cohort === "union" ? clip.in_union : clip.dataset === state.cohort)) && clip.behavior === state.behavior);
}

function renderCohort() {
  if (!state.catalogue) return;
  const cohort = state.catalogue.cohorts[state.cohort];
  if (!cohort.behaviors.some((item) => item.behavior === state.behavior)) state.behavior = cohort.behaviors[0]?.behavior || "Huddling";
  $("overall-accuracy").textContent = percent(cohort.full_accuracy ?? cohort.accuracy);
  $("overall-counts").textContent = cohort.full_total
    ? `全集 ${cohort.full_total} · P ${percent(cohort.full_precision)} · R ${percent(cohort.full_recall)} · 展示 ${cohort.total} 段`
    : `${cohort.correct} / ${cohort.total} 正确 · ${cohort.failed} 错误`;
  $("behavior-count").textContent = `${cohort.behaviors.length} 类`;
  const navigation = $("behavior-nav");
  navigation.replaceChildren();
  for (const item of cohort.behaviors) {
    const button = node("button", undefined, `behavior-button${item.behavior === state.behavior ? " active" : ""}`);
    button.type = "button";
    button.setAttribute("aria-current", item.behavior === state.behavior ? "true" : "false");
    const row = node("span", undefined, "behavior-row");
    const name = node("span", item.label, "behavior-name");
    name.append(node("span", item.behavior, "behavior-en"));
    row.append(name, node("span", `${item.failed} 错 / ${item.total} 选`, `error-count${item.failed === 0 ? " zero" : ""}`));
    const statText = item.full_total !== undefined
      ? `准确 ${percent(item.full_accuracy)} · P ${percent(item.full_precision)} · R ${percent(item.full_recall)}`
      : `${item.correct}/${item.total} 正确 · ${percent(item.accuracy)}`;
    button.append(row, node("span", statText, "behavior-stat"));
    button.addEventListener("click", () => { state.behavior = item.behavior; state.status = state.catalogue.default_status || "failed"; state.reason = "all"; state.query = ""; $("status-filter").value = state.status; $("search").value = ""; renderCohort(); });
    navigation.append(button);
  }
  const current = cohort.behaviors.find((item) => item.behavior === state.behavior);
  $("current-behavior").textContent = current ? `${current.label} · ${current.behavior}` : "暂无行为记录";
  $("behavior-summary").textContent = current
    ? (current.full_total !== undefined
      ? `全集 ${current.full_total} 段（P=${percent(current.full_precision)}, R=${percent(current.full_recall)}）· 本行为精选 ${current.total} 个实例（${current.correct} 对 / ${current.failed} 错）`
      : `${current.total} 个样本 · ${current.correct} 个正确 · ${current.failed} 个错误`)
    : "数据集中没有行为分类。";
  $("behavior-accuracy").textContent = percent(current?.full_accuracy ?? current?.accuracy);
  renderReasons();
  renderList();
}

function renderReasons() {
  const select = $("reason-filter");
  select.replaceChildren(node("option", "全部原因"));
  select.firstChild.value = "all";
  const counts = new Map();
  for (const row of behaviorRows()) {
    if (state.status === "failed" && row.correct || state.status === "passed" && !row.correct) continue;
    const item = counts.get(row.reason_code) || { count: 0, label: row.reason_label };
    item.count += 1; counts.set(row.reason_code, item);
  }
  for (const [code, item] of counts) {
    const option = node("option", `${item.label || reasons[code] || code} · ${item.count}`);
    option.value = code; select.append(option);
  }
  if (!counts.has(state.reason)) state.reason = "all";
  select.value = state.reason;
}

function renderList() {
  const query = state.query.trim().toLowerCase();
  state.rows = behaviorRows().filter((clip) => (state.status === "all" || (state.status === "passed") === clip.correct)
    && (state.reason === "all" || clip.reason_code === state.reason)
    && (!query || `${clip.clip_id} ${clip.id} ${clip.target_ids.map((id) => `id ${id}`).join(" ")}`.toLowerCase().includes(query)));
  $("result-count").textContent = `${state.rows.length} 个`;
  const list = $("sample-list"); list.replaceChildren();
  $("list-empty").hidden = state.rows.length > 0;
  const unfiltered = behaviorRows();
  $("empty-explanation").textContent = unfiltered.length && unfiltered.every((row) => row.correct) && state.status === "failed"
    ? "该行为没有错误样本，可查看全部样本及视频。" : "可查看全部样本，或调整原因与搜索条件。";
  for (const clip of state.rows) {
    const button = node("button", undefined, "sample-button"); button.type = "button"; button.dataset.recordId = clip.id;
    button.append(node("span", `${clip.dataset} · ${clip.clip_id}`, "sample-title"), node("span", `标注：${idsText(clip.target_ids)}`, "sample-targets"), node("span", clip.reason_label || reasons[clip.reason_code] || clip.reason_code, `sample-reason${clip.correct ? " passed" : ""}`));
    button.addEventListener("click", () => selectClip(clip.id)); list.append(button);
  }
  if (state.rows.length) selectClip(state.rows[0].id);
  else { state.selected = null; clearDetail(); $("clip-title").textContent = "暂无符合条件的样本"; $("clip-subtitle").textContent = "调整左侧筛选后可继续复核"; showDetailState("当前筛选没有样本。", false); updatePagination(); }
}

function updatePagination() {
  const index = state.rows.findIndex((clip) => clip.id === state.selected);
  $("previous").disabled = index <= 0;
  $("next").disabled = index < 0 || index >= state.rows.length - 1;
  $("sample-position").textContent = index >= 0 ? `${index + 1} / ${state.rows.length}` : "—";
  for (const button of $("sample-list").children) { const active = button.dataset.recordId === state.selected; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); }
}

function showDetailState(message, error = false) {
  $("detail-state").hidden = !message;
  $("detail-message").textContent = message;
  $("detail-state").classList.toggle("error", error);
  $("retry-detail").hidden = !error;
}

function clearDetail() {
  state.controller?.abort();
  if (state.frameCallback !== null && typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(state.frameCallback);
  state.frameCallback = null; state.detail = null; state.frames.clear(); state.targetIds.clear();
  video.pause(); video.removeAttribute("src"); video.load(); ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("video-placeholder").hidden = false; $("clip-evidence").hidden = true;
  $("frame-position").textContent = "帧 — / —"; $("visible-targets").textContent = "当前帧可见目标：—"; $("missing-targets").textContent = "缺失目标：—";
  for (const id of ["frame-back", "frame-forward", "fullscreen"]) $(id).disabled = true;
}

async function selectClip(id) {
  clearDetail(); state.selected = id; updatePagination();
  const clip = state.rows.find((item) => item.id === id);
  $("clip-title").textContent = `${clip.dataset} · ${clip.clip_id}`;
  $("clip-subtitle").textContent = `${clip.label} · 标注目标 ${idsText(clip.target_ids)}`;
  showDetailState("正在读取视频与逐帧证据…");
  const controller = new AbortController(); state.controller = controller;
  try {
    const detail = await fetchJson(`/api/clip/${encodeURIComponent(id)}`, controller, validateDetail);
    if (state.controller !== controller || state.selected !== id) return;
    if (detail.clip.id !== id) throw new Error("服务返回了其他样本的证据，请重试。");
    state.detail = detail; state.targetIds = new Set(detail.clip.target_ids);
    state.frames = new Map(detail.frames.map((frame) => [frame.frame, frame]));
    canvas.width = detail.video.width; canvas.height = detail.video.height;
    $("video-stage").style.setProperty("--video-ratio", String(detail.video.width / detail.video.height));
    renderEvidence(detail);
    video.src = detail.video.url; video.playbackRate = Number($("playback-rate").value);
    $("video-placeholder").hidden = true;
    $("fullscreen").disabled = typeof $("video-shell").requestFullscreen !== "function";
    showDetailState(""); drawFrame(0);
  } catch (error) {
    if (state.controller !== controller || error.name === "AbortError") return;
    showDetailState(`无法加载此样本：${error.message}`, true);
  }
}

function renderEvidence(detail) {
  const clip = detail.clip;
  $("clip-evidence").hidden = false;
  $("result-badge").textContent = clip.correct ? "正确" : "错误";
  $("result-badge").classList.toggle("passed", clip.correct);
  $("reason-title").textContent = clip.reason_label || reasons[clip.reason_code] || clip.reason_code;
  $("reason-detail").textContent = detail.reason_detail || clip.summary;
  $("target-ids").textContent = idsText(clip.target_ids);
  $("candidate-ids").textContent = idsText(detail.best_group_ids);
  $("target-coverage").textContent = percent(detail.true_target_coverage);
  $("candidate-coverage").textContent = percent(detail.best_group_coverage);
  $("classification-note").textContent = detail.classification_note;
  const rows = $("observation-rows"); rows.replaceChildren();
  for (const id of clip.target_ids) {
    const item = detail.observations.find((row) => row.id === id);
    const tr = node("tr");
    for (const value of [String(id), item ? String(item.observed_frames) : "无记录", item ? percent(item.observed_fraction) : "—", item && item.first_frame !== null ? `${item.first_frame}—${item.last_frame}` : "—"]) tr.append(node("td", value));
    rows.append(tr);
  }
  $("huddle-section").hidden = detail.huddle_gate === null;
  $("huddle-rows").replaceChildren();
  if (detail.huddle_gate) {
    const gate = detail.huddle_gate;
    $("huddle-evidence").textContent = `共同端点跨度 ${gate.common_endpoint_span_s.toFixed(2)} 秒 · 覆盖上限 ${percent(gate.common_endpoint_coverage_upper_bound)} · 紧密聚集累计 ${gate.compact_total_s.toFixed(2)} 秒`;
    for (const member of gate.members) {
      const tr = node("tr");
      for (const value of [String(member.id), percent(member.observed_fraction), percent(member.held_fraction), `${member.first ?? "—"}—${member.last ?? "—"}`]) tr.append(node("td", value));
      $("huddle-rows").append(tr);
    }
  }
  const scores = $("candidate-scores"); scores.replaceChildren();
  const labels = new Map(state.catalogue.cohorts[state.cohort].behaviors.map((item) => [item.behavior, item.label]));
  const entries = Object.entries(detail.target_candidate_scores).sort((a, b) => b[1] - a[1]);
  for (const [behavior, seconds] of entries) scores.append(node("span", `${labels.get(behavior) || behavior} · ${seconds.toFixed(2)} 秒`, "score-item"));
  if (!entries.length) scores.append(node("span", "无候选行为记录", "muted"));
  $("prediction-text").textContent = clip.prediction_text || "无记录";
  $("source-video").textContent = clip.source_video;
  $("record-id").textContent = clip.id;
}

function drawFrame(time) {
  const detail = state.detail;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!detail) return;
  const frameIndex = Math.min(detail.video.frame_count - 1, Math.max(0, Math.round(time * detail.video.fps)));
  const frame = state.frames.get(frameIndex);
  const detections = frame ? frame.detections : [];
  const visible = new Set(detections.filter((d) => state.targetIds.has(d.id)).map((d) => d.id));
  $("frame-position").textContent = `帧 ${frameIndex} / ${detail.video.frame_count - 1} · ${time.toFixed(2)} 秒`;
  $("visible-targets").textContent = `当前帧可见目标：${idsText([...visible])}`;
  $("missing-targets").textContent = `缺失目标：${idsText([...state.targetIds].filter((id) => !visible.has(id)))}`;
  const lineWidth = Math.max(2, canvas.width / 300);
  const fontSize = Math.max(14, Math.round(canvas.width / 58));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textBaseline = "top";
  const sorted = [...detections].sort((a, b) => Number(state.targetIds.has(b.id)) - Number(state.targetIds.has(a.id)));
  const labels = [];
  const occupied = [];
  for (const detection of sorted) {
    const target = state.targetIds.has(detection.id);
    if (target ? !$("show-targets").checked : detection.candidate ? !$("show-candidates").checked : !$("show-others").checked) continue;
    const [x1, y1, x2, y2] = detection.box;
    const color = target ? "#ffe52c" : detection.candidate ? "#ff922e" : "#a7b3be";
    ctx.save();
    if (target) {
      ctx.fillStyle = "rgba(255, 229, 44, 0.16)";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.shadowColor = "rgba(255, 229, 44, 0.9)";
      ctx.shadowBlur = 12;
    }
    ctx.strokeStyle = color; ctx.lineWidth = target ? lineWidth * 1.8 : lineWidth;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.restore();
    const label = `${target ? "★ 目标 " : detection.candidate ? "候选 " : ""}ID ${detection.id}`;
    const labelWidth = ctx.measureText(label).width + 10;
    const labelHeight = fontSize + 8;
    const labelX = Math.max(0, Math.min(x1, canvas.width - labelWidth));
    let labelY = Math.max(0, Math.min(y1 - labelHeight - 4, canvas.height - labelHeight));
    for (let attempt = 0; attempt < detections.length * 2 + 8; attempt++) {
      const shift = Math.ceil(attempt / 2) * (labelHeight + 5) * (attempt % 2 ? -1 : 1);
      const candidateY = Math.max(0, Math.min(y1 - labelHeight - 4 + shift, canvas.height - labelHeight));
      if (!occupied.some((r) => labelX < r.x + r.width + 3 && labelX + labelWidth + 3 > r.x && candidateY < r.y + r.height + 3 && candidateY + labelHeight + 3 > r.y)) {
        labelY = candidateY;
        break;
      }
    }
    occupied.push({x: labelX, y: labelY, width: labelWidth, height: labelHeight});
    labels.push({label, color, x: labelX, y: labelY, width: labelWidth, height: labelHeight, boxX: (x1 + x2) / 2, boxY: y1});
  }
  // Pile labels need separate lanes; leader lines preserve the measured box association.
  for (const {label, color, x, y, width, height, boxX, boxY} of labels) {
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, lineWidth / 2);
    ctx.beginPath(); ctx.moveTo(boxX, boxY); ctx.lineTo(x + width / 2, y + height); ctx.stroke();
  }
  for (const {label, color, x, y, width, height} of labels) {
    ctx.fillStyle = color; ctx.fillRect(x, y, width, height);
    ctx.fillStyle = "#18262e"; ctx.fillText(label, x + 5, y + 4);
  }
}

function queueVideoFrame() {
  if (!state.detail || video.paused || typeof video.requestVideoFrameCallback !== "function" || state.frameCallback !== null) return;
  state.frameCallback = video.requestVideoFrameCallback((_now, metadata) => {
    state.frameCallback = null;
    drawFrame(metadata.mediaTime);
    queueVideoFrame();
  });
}

function stepFrame(direction) {
  if (!state.detail || video.readyState < 1) return;
  video.pause();
  const { fps, frame_count: count } = state.detail.video;
  const index = Math.max(0, Math.min(count - 1, Math.round(video.currentTime * fps) + direction));
  video.currentTime = index / fps;
  drawFrame(video.currentTime);
}

$("cohort").addEventListener("change", () => { state.cohort = $("cohort").value; state.reason = "all"; renderCohort(); });
$("status-filter").addEventListener("change", () => { state.status = $("status-filter").value; state.reason = "all"; if (state.catalogue) { renderReasons(); renderList(); } });
$("reason-filter").addEventListener("change", () => { state.reason = $("reason-filter").value; if (state.catalogue) renderList(); });
$("search").addEventListener("input", () => { state.query = $("search").value; if (state.catalogue) renderList(); });
$("show-all").addEventListener("click", () => { state.status = "all"; state.reason = "all"; state.query = ""; $("status-filter").value = "all"; $("search").value = ""; renderReasons(); renderList(); });
$("retry-catalogue").addEventListener("click", loadCatalogue);
$("retry-detail").addEventListener("click", () => { if (state.selected) selectClip(state.selected); });
$("previous").addEventListener("click", () => { const index = state.rows.findIndex((row) => row.id === state.selected); if (index > 0) selectClip(state.rows[index - 1].id); });
$("next").addEventListener("click", () => { const index = state.rows.findIndex((row) => row.id === state.selected); if (index >= 0 && index + 1 < state.rows.length) selectClip(state.rows[index + 1].id); });
$("frame-back").addEventListener("click", () => stepFrame(-1));
$("frame-forward").addEventListener("click", () => stepFrame(1));
$("playback-rate").addEventListener("change", () => { video.playbackRate = Number($("playback-rate").value); });
for (const id of ["show-targets", "show-candidates", "show-others"]) $(id).addEventListener("change", () => drawFrame(video.currentTime));
$("fullscreen").addEventListener("click", async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("video-shell").requestFullscreen(); }
  catch { showDetailState("浏览器未能进入全屏；可继续在页面内播放。", false); }
});
video.addEventListener("loadedmetadata", () => { if (!state.detail) return; for (const id of ["frame-back", "frame-forward"]) $(id).disabled = false; video.playbackRate = Number($("playback-rate").value); drawFrame(video.currentTime); });
video.addEventListener("play", queueVideoFrame);
video.addEventListener("pause", () => { if (state.frameCallback !== null && typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(state.frameCallback); state.frameCallback = null; drawFrame(video.currentTime); });
video.addEventListener("seeked", () => { drawFrame(video.currentTime); queueVideoFrame(); });
video.addEventListener("timeupdate", () => { if (typeof video.requestVideoFrameCallback !== "function" || video.paused) drawFrame(video.currentTime); });
video.addEventListener("error", () => { if (state.detail && video.getAttribute("src")) showDetailState("视频加载失败，请重试此样本。评测文字证据仍可查看。", true); });
loadCatalogue();
