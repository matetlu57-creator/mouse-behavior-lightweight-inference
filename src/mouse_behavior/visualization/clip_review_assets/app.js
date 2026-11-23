"use strict";

(() => {
  /* =========================================================================
   * Constants & Domain Metadata
   * ========================================================================= */
  const BEHAVIOR_ORDER = [
    "Approach",
    "Chasing",
    "Dispersal",
    "Grooming",
    "Group locomotion",
    "Huddling",
    "Isolation",
    "Moving",
    "Rearing",
    "Snout-head_contact",
    "Snout-rear_contact",
    "Social clustering",
    "Static",
    "Together",
  ];

  const EXTRA_COLLECT_BEHAVIORS = ["Attack", "Locomotion"];

  const BEHAVIOR_CN = {
    Approach: "接近",
    Chasing: "追逐",
    Dispersal: "散开",
    Grooming: "理毛",
    "Group locomotion": "群体同步运动",
    Huddling: "扎堆",
    Isolation: "孤立",
    Moving: "移动",
    Rearing: "站立",
    "Snout-head_contact": "嗅闻头部",
    "Snout-rear_contact": "嗅闻尾部",
    "Social clustering": "社会聚集",
    Static: "静止",
    Together: "共处",
    Attack: "攻击",
    Locomotion: "运动 (Locomotion)",
  };

  const DIRECTIONAL_ROLES = {
    Approach: [
      { role_key: "approaching", role_name: "Approaching", label: "接近方 (Approaching)" },
      { role_key: "approached", role_name: "Approached", label: "被接近方 (Approached)" },
    ],
    Chasing: [
      { role_key: "chasing", role_name: "Chasing", label: "追逐方 (Chasing)" },
      { role_key: "chased", role_name: "Chased", label: "被追逐方 (Chased)" },
    ],
    "Snout-head_contact": [
      { role_key: "snout_contacting", role_name: "Snout_contacting", label: "嗅闻方 (Snout_contacting)" },
      { role_key: "head_contacted", role_name: "Head_contacted", label: "被嗅头方 (Head_contacted)" },
    ],
    "Snout-rear_contact": [
      { role_key: "snout_contacting", role_name: "Snout_contacting", label: "嗅闻方 (Snout_contacting)" },
      { role_key: "rear_contacted", role_name: "Rear_contacted", label: "被嗅尾方 (Rear_contacted)" },
    ],
    Attack: [
      { role_key: "attacker", role_name: "Attacker", label: "攻击方 (Attacker)" },
      { role_key: "attacked", role_name: "Attacked", label: "被攻击方 (Attacked)" },
    ],
  };

  const ROLE_CN = {
    Approaching: "接近方",
    Approached: "被接近方",
    Chasing: "追逐方",
    Chased: "被追逐方",
    Snout_contacting: "嗅闻方",
    Head_contacted: "被嗅头方",
    Rear_contacted: "被嗅尾方",
    Attacker: "攻击方",
    Attacked: "被攻击方",
  };

  const STATUS_META = {
    pending: { label: "待审核", cls: "pending" },
    pass: { label: "通过", cls: "pass" },
    fail: { label: "驳回", cls: "fail" },
    needs_recrop: { label: "需重裁", cls: "needs_recrop" },
  };

  const DEFAULT_SKELETON_EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
    [3, 5],
    [4, 6],
    [5, 6],
  ];

  const PENDING_STORAGE_KEY = "mouse_clip_review_pending_v1";
  const PAGE_SIZE = 50;

  /* =========================================================================
   * DOM Helpers & Security Escaping
   * ========================================================================= */
  const $ = (id) => document.getElementById(id);

  function escHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function isFiniteNum(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function isIntList(value) {
    return Array.isArray(value) && value.every((v) => Number.isInteger(v));
  }

  function clamp(num, min, max) {
    return Math.min(max, Math.max(min, num));
  }

  function behaviorCn(name) {
    return BEHAVIOR_CN[name] || name || "未知行为";
  }

  function formatDuration(frames, fps) {
    if (!fps || fps <= 0 || !Number.isFinite(frames)) return "0.00s";
    return `${(Math.max(0, frames) / fps).toFixed(2)}s`;
  }

  function formatTimeStamp(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    if (Number.isNaN(d.getTime())) return String(isoStr);
    return d.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function assertSafeUrl(rawUrl) {
    if (typeof rawUrl !== "string" || !rawUrl.trim()) {
      throw new Error("媒体 URL 缺失或格式无效。");
    }
    const parsed = new URL(rawUrl, window.location.origin);
    if (parsed.origin !== window.location.origin || !["http:", "https:"].includes(parsed.protocol)) {
      throw new Error("媒体 URL 不符合同源本地服务安全策略。");
    }
    return parsed.pathname + parsed.search;
  }

  function isDemoUid(clipUid) {
    return typeof clipUid === "string" && clipUid.startsWith("DEMO__");
  }

  /* =========================================================================
   * Application State
   * ========================================================================= */
  const state = {
    connected: false,
    offlineMode: false,
    offlineReason: "",
    sessionToken: "",
    health: null,

    catalog: null,
    clips: [],
    clipsByUid: new Map(),
    behaviors: [],

    reviews: {},
    pendingReviews: {},

    allCollections: [],
    collectionsOutputDir: "",

    // Browser Filters & Pagination
    selectedBehavior: "__ALL__",
    statusFilter: "all",
    searchQuery: "",
    filteredClips: [],
    page: 1,

    // Active Clip & Video Sync
    activeUid: null,
    detail: null,
    framesMap: new Map(),
    skeletonEdges: DEFAULT_SKELETON_EDGES,
    currentFrame: 0,
    videoError: false,
    rvfcId: null,
    demoPlayRaf: null,
    demoLastTick: 0,
    demoPlaying: false,
    loopPreview: false,
    hoveredTrackId: null,
    lastHitBoxPos: null,

    // Crop Range [cropStart, cropEnd)  (start inclusive, end exclusive)
    cropStart: 0,
    cropEnd: 1,
    draggingMode: null, // "start" | "end" | "scrub" | null

    // Right Panel State
    activeTab: "review",
    reviewDraft: {
      status: "pending",
      tags: new Set(),
      note: "",
      attachCrop: false,
    },
    collectDraft: {
      behavior: "",
      customBehavior: "",
      mouseIds: [],
      roleAssignments: {}, // role_name -> track_id | null
      paddingFrames: 0,
      includeOverlay: true,
      overwrite: false,
      note: "",
      submitting: false,
    },

    // Expanded inline preview states keyed by export_id -> "raw" | "overlay"
    expandedPreviews: new Map(),

    // Active Batch Job
    batchJobId: null,
    batchPollTimer: null,

    // In-flight AbortControllers
    bootController: null,
    detailController: null,
  };

  const videoEl = $("clip-video");
  const canvasEl = $("clip-canvas");
  const canvasCtx = canvasEl.getContext("2d");
  const visCanvasEl = $("visibility-canvas");
  const visCtx = visCanvasEl.getContext("2d");

  /* =========================================================================
   * Toast Notifications
   * ========================================================================= */
  function showToast(message, type = "info", durationMs = 3600) {
    const region = $("toast-region");
    if (!region) return;
    const item = document.createElement("div");
    item.className = `toast ${type}`;
    item.textContent = message;
    region.appendChild(item);
    window.setTimeout(() => {
      item.remove();
    }, durationMs);
  }

  /* =========================================================================
   * Strict JSON Fetch & Schema Validation (AGENTS.md API Resilience)
   * ========================================================================= */
  async function fetchJson(url, opts = {}) {
    const {
      method = "GET",
      body = undefined,
      timeoutMs = 15000,
      signal = null,
      validate = null,
      retryOn403 = true,
    } = opts;

    const controller = new AbortController();
    let timedOut = false;

    const onExternalAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) {
        const abortErr = new DOMException("Aborted", "AbortError");
        throw abortErr;
      }
      signal.addEventListener("abort", onExternalAbort, { once: true });
    }

    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    const headers = {
      Accept: "application/json",
    };

    const upperMethod = method.toUpperCase();
    if (upperMethod === "POST" || upperMethod === "DELETE") {
      if (state.sessionToken) {
        headers["X-Review-Token"] = state.sessionToken;
      }
    }
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }

    try {
      const response = await fetch(url, {
        method: upperMethod,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        cache: "no-store",
        signal: controller.signal,
      });

      const contentType = response.headers.get("content-type") || "";
      const isJson = /\bapplication\/json\b/i.test(contentType);

      if (!response.ok) {
        let serverMsg = "";
        let parsedErrBody = null;
        if (isJson) {
          try {
            parsedErrBody = await response.json();
            serverMsg =
              parsedErrBody?.error ||
              parsedErrBody?.message ||
              parsedErrBody?.detail ||
              "";
          } catch {
            // ignore JSON parse failure on error body
          }
        } else if (/\btext\/html\b/i.test(contentType)) {
          serverMsg = "返回了 HTML 页面而非 JSON API 响应";
        }

        // If backend restarted with a fresh session_token, refresh once on 403
        if (response.status === 403 && retryOn403 && (upperMethod === "POST" || upperMethod === "DELETE")) {
          try {
            const refreshed = await fetchJson("/api/health", {
              timeoutMs: 6000,
              validate: validateHealth,
              retryOn403: false,
            });
            if (refreshed.session_token && refreshed.session_token !== state.sessionToken) {
              state.sessionToken = refreshed.session_token;
              return await fetchJson(url, { ...opts, retryOn403: false });
            }
          } catch {
            // fall through to throw original 403
          }
        }

        const err = new Error(
          serverMsg
            ? `HTTP ${response.status}：${serverMsg}`
            : `后端接口返回 HTTP ${response.status}`
        );
        err.status = response.status;
        err.body = parsedErrBody;
        const explicitCorrupt = Boolean(parsedErrBody && parsedErrBody.corrupt_store === true);
        const alreadyExists409 =
          response.status === 409 &&
          !explicitCorrupt &&
          /exist|overwrite|已存在|覆盖/i.test(serverMsg);
        err.isAlreadyExists = alreadyExists409;
        err.isCorrupt =
          explicitCorrupt ||
          response.status === 500 ||
          (response.status === 409 && !alreadyExists409);
        throw err;
      }

      if (!isJson) {
        const isHtml = /\btext\/html\b/i.test(contentType);
        const err = new Error(
          isHtml
            ? "接口返回了 HTML（可能命中 SPA 回退页），已拒绝当作 JSON 解析。"
            : `接口 Content-Type 异常（${contentType || "空"}），预期 application/json。`
        );
        err.status = response.status;
        err.isMalformed = true;
        throw err;
      }

      let data;
      try {
        data = await response.json();
      } catch {
        const err = new Error("接口返回的 JSON 语法无效，无法解析。");
        err.isMalformed = true;
        throw err;
      }

      if (typeof validate === "function") {
        return validate(data);
      }
      return data;
    } catch (error) {
      if (timedOut) {
        const err = new Error(`请求超时（>${Math.round(timeoutMs / 1000)}s），请检查本地后端服务。`);
        err.isTimeout = true;
        err.isNetwork = true;
        throw err;
      }
      if (error && error.name === "AbortError") {
        throw error;
      }
      if (error instanceof TypeError) {
        const err = new Error("无法连接本地后端服务（网络不可达或服务未启动）。");
        err.isNetwork = true;
        throw err;
      }
      throw error;
    } finally {
      window.clearTimeout(timer);
      if (signal) {
        signal.removeEventListener("abort", onExternalAbort);
      }
    }
  }

  /* =========================================================================
   * Response Schema Validators (strictly matching locked contract #45)
   * ========================================================================= */
  function validateHealth(data) {
    if (!isRecord(data) || typeof data.status !== "string" || typeof data.session_token !== "string") {
      throw new Error("/api/health 响应格式不符合契约。");
    }
    return data;
  }

  function validateCatalog(data) {
    if (
      !isRecord(data) ||
      data.schema !== "mouse_clip_catalog_v1" ||
      !Array.isArray(data.behaviors) ||
      !Array.isArray(data.clips)
    ) {
      throw new Error("/api/catalog 响应缺少 mouse_clip_catalog_v1 契约字段。");
    }
    for (const clip of data.clips) {
      if (
        !isRecord(clip) ||
        typeof clip.clip_uid !== "string" ||
        typeof clip.behavior !== "string" ||
        typeof clip.clip_id !== "string" ||
        !isFiniteNum(clip.fps) ||
        clip.fps <= 0 ||
        !Number.isInteger(clip.frame_count) ||
        clip.frame_count <= 0
      ) {
        throw new Error("/api/catalog 片段条目字段无效或缺失。");
      }
    }
    return data;
  }

  function validateClipDetail(data) {
    if (
      !isRecord(data) ||
      data.schema !== "mouse_clip_detail_v1" ||
      !isRecord(data.clip) ||
      !isRecord(data.video) ||
      typeof data.video.url !== "string" ||
      !isFiniteNum(data.video.fps) ||
      data.video.fps <= 0 ||
      !Number.isInteger(data.video.frame_count) ||
      data.video.frame_count <= 0 ||
      !Array.isArray(data.frames) ||
      !Array.isArray(data.visible_tracks)
    ) {
      throw new Error("/api/clip 详情格式不符合 mouse_clip_detail_v1 契约。");
    }
    if (!isDemoUid(data.clip.clip_uid)) {
      data.video.url = assertSafeUrl(data.video.url);
    }
    if (!Array.isArray(data.collections)) {
      data.collections = [];
    } else {
      for (const item of data.collections) {
        if (isRecord(item)) {
          if (item.raw_video_url) item.raw_video_url = assertSafeUrl(item.raw_video_url);
          if (item.overlay_video_url) item.overlay_video_url = assertSafeUrl(item.overlay_video_url);
        }
      }
    }
    return data;
  }

  function validateReviews(data) {
    if (!isRecord(data) || data.schema !== "mouse_clip_reviews_v1" || !isRecord(data.reviews)) {
      throw new Error("/api/reviews 响应不符合 mouse_clip_reviews_v1 契约。");
    }
    return data;
  }

  function validateReviewSave(data) {
    if (!isRecord(data) || data.status !== "ok" || !isRecord(data.review)) {
      throw new Error("POST /api/reviews 响应格式不符合契约。");
    }
    return data;
  }

  function validateCollections(data) {
    if (
      !isRecord(data) ||
      data.schema !== "mouse_clip_collections_v1" ||
      !Array.isArray(data.collections)
    ) {
      throw new Error("/api/collections 响应不符合 mouse_clip_collections_v1 契约。");
    }
    for (const item of data.collections) {
      if (isRecord(item)) {
        if (item.raw_video_url) item.raw_video_url = assertSafeUrl(item.raw_video_url);
        if (item.overlay_video_url) item.overlay_video_url = assertSafeUrl(item.overlay_video_url);
      }
    }
    return data;
  }

  function validateCollectResponse(data) {
    if (!isRecord(data) || data.status !== "ok" || !isRecord(data.collection)) {
      throw new Error("POST /api/collect 响应格式不符合契约。");
    }
    const c = data.collection;
    if (c.raw_video_url) c.raw_video_url = assertSafeUrl(c.raw_video_url);
    if (c.overlay_video_url) c.overlay_video_url = assertSafeUrl(c.overlay_video_url);
    return data;
  }

  function validateDeleteResponse(data) {
    if (!isRecord(data) || data.status !== "ok") {
      throw new Error("DELETE /api/collections 响应格式不符合契约。");
    }
    return data;
  }

  function validateBatchAccept(data) {
    if (!isRecord(data) || data.status !== "accepted" || typeof data.job_id !== "string") {
      throw new Error("POST /api/collect/batch 响应格式不符合契约。");
    }
    return data;
  }

  function validateBatchJob(data) {
    if (!isRecord(data) || typeof data.job_id !== "string" || typeof data.status !== "string") {
      throw new Error("GET /api/collect/jobs 响应格式不符合契约。");
    }
    return data;
  }

  /* =========================================================================
   * Local Pending Reviews Queue (localStorage resilience)
   * ========================================================================= */
  function loadPendingReviewsFromStorage() {
    try {
      const raw = window.localStorage.getItem(PENDING_STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return isRecord(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  function persistPendingReviews() {
    try {
      const keys = Object.keys(state.pendingReviews);
      if (!keys.length) {
        window.localStorage.removeItem(PENDING_STORAGE_KEY);
      } else {
        window.localStorage.setItem(PENDING_STORAGE_KEY, JSON.stringify(state.pendingReviews));
      }
    } catch {
      // Ignore storage quota errors cleanly
    }
    updateHeaderStats();
  }

  function queuePendingReview(payload) {
    state.pendingReviews[payload.clip_uid] = {
      ...payload,
      updated_at: new Date().toISOString(),
    };
    persistPendingReviews();
  }

  function realPendingKeys() {
    return Object.keys(state.pendingReviews).filter((uid) => !isDemoUid(uid));
  }

  async function flushPendingReviews() {
    const keys = realPendingKeys();
    if (!keys.length || !state.connected || state.offlineMode) return { flushed: 0, failed: 0 };

    let flushed = 0;
    let failed = 0;

    for (const uid of keys) {
      const entry = state.pendingReviews[uid];
      if (!entry) continue;
      const body = {
        clip_uid: entry.clip_uid,
        status: entry.status,
        tags: Array.isArray(entry.tags) ? entry.tags : [],
        note: typeof entry.note === "string" ? entry.note : "",
        crop_range: Array.isArray(entry.crop_range) && entry.crop_range.length === 2 ? entry.crop_range : null,
      };
      try {
        const res = await fetchJson("/api/reviews", {
          method: "POST",
          body,
          timeoutMs: 12000,
          validate: validateReviewSave,
        });
        state.reviews[uid] = res.review;
        delete state.pendingReviews[uid];
        const clip = state.clipsByUid.get(uid);
        if (clip) clip.review_status = res.review.status;
        flushed += 1;
      } catch (error) {
        failed += 1;
        if (error.isCorrupt) {
          showCorruptBanner(
            `同步本地审核时后端返回 HTTP ${error.status}（持久化文件异常）`,
            `${error.message}。系统已原样保留后端损坏文件，并保留您剩余的 ${keys.length - flushed} 条本地待同步审核不丢失。`
          );
          break;
        }
        if (error.isNetwork) {
          break;
        }
      }
    }

    persistPendingReviews();
    renderBehaviorNav();
    renderClipList();
    if (state.activeUid) {
      loadReviewFormFromState(state.activeUid);
    }
    updateBatchSummary();

    if (flushed > 0 && failed === 0) {
      hideCorruptBanner();
      showToast(`已自动同步 ${flushed} 条本地暂存审核至后端。`, "ok");
    } else if (flushed > 0 && failed > 0) {
      showToast(`已同步 ${flushed} 条，仍有 ${failed} 条保留在本地待重试。`, "warn");
    }
    return { flushed, failed };
  }

  /* =========================================================================
   * Offline Demo Fallback Dataset (clearly marked non-scientific demo data)
   * ========================================================================= */
  function buildDemoBundle() {
    const fps = 30.013;
    const width = 1920;
    const height = 1080;
    const frameCount = 120;
    const clips = [];
    const details = new Map();

    const behaviors = BEHAVIOR_ORDER.map((name) => ({ name, count: 1 }));

    BEHAVIOR_ORDER.forEach((beh, idx) => {
      const clipId = `DEMO_${beh.replace(/\s+/g, "_")}_synthetic_${String(idx + 1).padStart(3, "0")}`;
      const clipUid = `DEMO__${beh.replace(/\s+/g, "_")}__${idx + 1}`;
      const isDyadic = Boolean(DIRECTIONAL_ROLES[beh]) || beh === "Together";
      const isGroup = ["Dispersal", "Group locomotion", "Huddling", "Social clustering"].includes(beh);
      const targetIds = isGroup ? [10, 21, 34] : isDyadic ? [10, 62] : [10];
      const allIds = isGroup ? [10, 21, 34, 48] : [10, 62, 48];

      const participants = DIRECTIONAL_ROLES[beh]
        ? [
            {
              role_key: DIRECTIONAL_ROLES[beh][0].role_key,
              role_name: DIRECTIONAL_ROLES[beh][0].role_name,
              track_ids: [targetIds[targetIds.length - 1]],
            },
            {
              role_key: DIRECTIONAL_ROLES[beh][1].role_key,
              role_name: DIRECTIONAL_ROLES[beh][1].role_name,
              track_ids: [targetIds[0]],
            },
          ]
        : [];

      const clipSummary = {
        clip_uid: clipUid,
        behavior: beh,
        clip_id: clipId,
        fps,
        width,
        height,
        frame_count: frameCount,
        duration_s: Number((frameCount / fps).toFixed(3)),
        annotation_range: [30, 89],
        target_mouse_ids: targetIds,
        source_video: "DEMO_synthetic_arena_part001.mp4",
        has_tracks: true,
        has_annotation: true,
        has_metadata: true,
        review_status: "pending",
        collection_count: 0,
      };
      clips.push(clipSummary);

      // Build synthetic measured frames with realistic partial dropout on frames 52..60 for demo
      const frames = [];
      const trackStats = new Map();
      for (let f = 0; f < frameCount; f++) {
        const t = f / fps;
        const detections = [];
        allIds.forEach((tid, mIdx) => {
          // Simulate partial dropout on target mouse #2 (or #1) for frames 52..60
          if (mIdx === (targetIds.length > 1 ? 1 : 0) && f >= 52 && f <= 60) {
            return;
          }
          const angle = t * (0.7 + mIdx * 0.22) + idx * 0.6 + mIdx * 2.1;
          const cx = 620 + mIdx * 260 + Math.cos(angle) * 140;
          const cy = 460 + Math.sin(angle * 1.15) * 130 + (mIdx % 2) * 90;
          const heading = angle + Math.PI / 2;
          const cosH = Math.cos(heading);
          const sinH = Math.sin(heading);

          const kp = [
            [cx + cosH * 62, cy + sinH * 62, 0.94], // nose
            [cx + cosH * 36 - sinH * 18, cy + sinH * 36 + cosH * 18, 0.91], // right_ear
            [cx + cosH * 36 + sinH * 18, cy + sinH * 36 - cosH * 18, 0.91], // left_ear
            [cx + cosH * 18, cy + sinH * 18, 0.95], // base_of_neck
            [cx - cosH * 24 + sinH * 24, cy - sinH * 24 - cosH * 24, 0.89], // left_hip
            [cx - cosH * 24 - sinH * 24, cy - sinH * 24 + cosH * 24, 0.89], // right_hip
            [cx - cosH * 56, cy - sinH * 56, 0.87], // base_of_tail
          ];
          const xs = kp.map((p) => p[0]);
          const ys = kp.map((p) => p[1]);
          const pad = 26;
          const box = [
            Math.max(10, Math.min(...xs) - pad),
            Math.max(10, Math.min(...ys) - pad),
            Math.min(width - 10, Math.max(...xs) + pad),
            Math.min(height - 10, Math.max(...ys) + pad),
          ];

          detections.push({
            track_id: tid,
            box,
            keypoints: kp,
            detection_confidence: 0.92,
          });

          const stat = trackStats.get(tid) || { track_id: tid, frame_count: 0, first_frame: f, last_frame: f };
          stat.frame_count += 1;
          stat.last_frame = f;
          trackStats.set(tid, stat);
        });
        frames.push({ frame: f, time: t, detections });
      }

      details.set(clipUid, {
        schema: "mouse_clip_detail_v1",
        clip: clipSummary,
        annotation: {
          behavior: beh,
          mouse_ids: targetIds,
          participants,
          confidence: "certain",
          frame_range: { start: 30, end: 89 },
          time_range: { start: 30 / fps, end: 89 / fps },
        },
        metadata: {
          schema_version: "2.0",
          clip: { filename: "clip.mp4", fps, width, height, frame_count: frameCount },
          tracks: { frame_origin: 0, time_origin: 0.0, coordinate_system: "clip_pixels" },
          pose: {
            keypoint_names: ["nose", "right_ear", "left_ear", "base_of_neck", "left_hip", "right_hip", "base_of_tail"],
            skeleton_edges: DEFAULT_SKELETON_EDGES,
          },
          source: {
            video_filename: "DEMO_synthetic_arena_part001.mp4",
            clip: { start_frame: 1000, end_frame: 1119, start_time: 33.32, end_time: 37.28 },
            annotation: { start_frame: 1030, end_frame: 1089, start_time: 34.32, end_time: 36.28 },
            crop_region: { x: 0, y: 0, w: width, h: height },
          },
          distance_scale: { cm_per_pixel: 0.099, unit: "cm" },
        },
        video: { url: "", fps, frame_count: frameCount, width, height },
        frames,
        visible_tracks: [...trackStats.values()].sort((a, b) => a.track_id - b.track_id),
        review: null,
        collections: [],
      });
    });

    return {
      catalog: {
        schema: "mouse_clip_catalog_v1",
        source_dir: "演示回退模式（未连接 F:\\mouse）",
        output_dir: "本地浏览器暂存",
        total_clips: clips.length,
        behaviors,
        clips,
      },
      details,
    };
  }

  let demoBundleCache = null;
  function getDemoBundle() {
    if (!demoBundleCache) demoBundleCache = buildDemoBundle();
    return demoBundleCache;
  }

  /* =========================================================================
   * Effective Review & Collection State Helpers
   * ========================================================================= */
  function getEffectiveReview(clipUid) {
    if (state.pendingReviews[clipUid]) return state.pendingReviews[clipUid];
    if (state.reviews[clipUid]) return state.reviews[clipUid];
    return null;
  }

  function getEffectiveStatus(clip) {
    const rev = getEffectiveReview(clip.clip_uid);
    if (rev && rev.status && STATUS_META[rev.status]) return rev.status;
    if (clip.review_status && STATUS_META[clip.review_status]) return clip.review_status;
    return "pending";
  }

  function getCollectionCount(clip) {
    return Math.max(0, Number(clip.collection_count || 0));
  }

  function syncCatalogCollectionCounts() {
    const counts = new Map();
    for (const item of state.allCollections) {
      counts.set(item.clip_uid, (counts.get(item.clip_uid) || 0) + 1);
    }
    for (const clip of state.clips) {
      clip.collection_count = counts.get(clip.clip_uid) || 0;
    }
  }

  /* =========================================================================
   * Connection, Banners & Bootstrap
   * ========================================================================= */
  function setConnectionState(mode, text) {
    const pill = $("connection-status");
    pill.className = `conn-pill ${mode}`;
    $("connection-text").textContent = text;
  }

  function showCatalogNotice(message, isError = false) {
    const el = $("catalog-notice");
    $("catalog-notice-text").textContent = message;
    el.hidden = !message;
    el.classList.toggle("error", isError);
    $("catalog-retry-btn").hidden = !isError;
  }

  function showOfflineBanner(reasonText) {
    state.offlineMode = true;
    state.offlineReason = reasonText;
    $("offline-banner").hidden = false;
    $("offline-reason").textContent = `${reasonText} 已切换至内置合成演示数据以保持工作台完整可用；审核编辑会自动暂存于本地浏览器，重连后自动同步。`;
    $("readonly-pill").textContent = "演示回退模式";
    $("readonly-pill").className = "meta-pill offline";
    setConnectionState("offline", "离线演示模式");
    updateHeaderStats();
  }

  function hideOfflineBanner() {
    state.offlineMode = false;
    state.offlineReason = "";
    $("offline-banner").hidden = true;
    $("readonly-pill").textContent = "只读源目录";
    $("readonly-pill").className = "meta-pill readonly";
  }

  function showCorruptBanner(title, detail) {
    $("corrupt-title").textContent = title;
    $("corrupt-detail").textContent = detail;
    $("corrupt-banner").hidden = false;
  }

  function hideCorruptBanner() {
    $("corrupt-banner").hidden = true;
  }

  function updateHeaderStats() {
    const total = state.clips.length;
    let reviewed = 0;
    for (const clip of state.clips) {
      if (getEffectiveStatus(clip) !== "pending") reviewed += 1;
    }
    const collectedTotal = state.offlineMode ? 0 : state.allCollections.length;
    const pendingReal = realPendingKeys().length;

    $("global-reviewed-count").textContent = `${reviewed} / ${total}`;
    $("global-collected-count").textContent = String(collectedTotal);
    $("drawer-count-badge").textContent = String(collectedTotal);

    $("pending-sync-chip").hidden = pendingReal === 0;
    $("global-pending-sync-count").textContent = String(pendingReal);
    $("banner-flush-btn").hidden = pendingReal === 0 || state.offlineMode;
  }

  function applyCatalogData(catalog, isDemo = false) {
    state.catalog = catalog;
    state.clips = catalog.clips || [];
    state.clipsByUid = new Map(state.clips.map((c) => [c.clip_uid, c]));

    // Keep canonical order for known behaviors, plus any extra behaviors discovered
    const countMap = new Map();
    for (const b of catalog.behaviors || []) {
      countMap.set(b.name, Number(b.count || 0));
    }
    for (const clip of state.clips) {
      if (!countMap.has(clip.behavior)) {
        countMap.set(clip.behavior, 0);
      }
    }
    const orderedNames = [];
    for (const name of BEHAVIOR_ORDER) {
      if (countMap.has(name)) orderedNames.push(name);
    }
    for (const name of countMap.keys()) {
      if (!orderedNames.includes(name)) orderedNames.push(name);
    }
    state.behaviors = orderedNames.map((name) => ({
      name,
      count: countMap.get(name) || 0,
    }));

    if (
      state.selectedBehavior !== "__ALL__" &&
      !state.behaviors.some((b) => b.name === state.selectedBehavior)
    ) {
      state.selectedBehavior = "__ALL__";
    }

    $("behavior-total-badge").textContent = `${state.behaviors.length} 类`;
    $("source-path-line").textContent = isDemo
      ? "演示回退数据 · 非实时 F:\\mouse 评测结果（等待重新连接后端）"
      : `源目录(只读)：${catalog.source_dir || "F:\\mouse"} · 输出目录：${catalog.output_dir || "—"}`;
    $("drawer-output-dir").textContent = `输出目录：${catalog.output_dir || state.collectionsOutputDir || "—"}`;

    renderBehaviorNav();
    applyFiltersAndRender({ preserveSelection: true });
    updateHeaderStats();
    updateBatchSummary();
  }

  async function connectAndLoad({ silentToast = false } = {}) {
    state.bootController?.abort();
    const controller = new AbortController();
    state.bootController = controller;

    setConnectionState("connecting", "正在连接…");
    showCatalogNotice("正在连接本地后端并读取片段目录…", false);

    try {
      const health = await fetchJson("/api/health", {
        signal: controller.signal,
        timeoutMs: 10000,
        validate: validateHealth,
      });
      if (state.bootController !== controller) return;

      state.health = health;
      state.sessionToken = health.session_token || "";
      state.connected = true;

      const [catalogRes, reviewsRes, collectionsRes] = await Promise.allSettled([
        fetchJson("/api/catalog", {
          signal: controller.signal,
          timeoutMs: 25000,
          validate: validateCatalog,
        }),
        fetchJson("/api/reviews", {
          signal: controller.signal,
          timeoutMs: 15000,
          validate: validateReviews,
        }),
        fetchJson("/api/collections", {
          signal: controller.signal,
          timeoutMs: 15000,
          validate: validateCollections,
        }),
      ]);

      if (state.bootController !== controller) return;

      if (catalogRes.status !== "fulfilled") {
        throw catalogRes.reason;
      }

      hideOfflineBanner();
      setConnectionState("connected", "后端已连接");

      // Handle reviews result (preserve corrupt store handling cleanly)
      if (reviewsRes.status === "fulfilled") {
        state.reviews = reviewsRes.value.reviews || {};
        hideCorruptBanner();
      } else {
        state.reviews = {};
        const revErr = reviewsRes.reason;
        if (revErr && revErr.isCorrupt) {
          showCorruptBanner(
            `读取审核台账失败（HTTP ${revErr.status}）`,
            `${revErr.message}。后端已完整保留损坏文件未覆盖；您仍可正常浏览与标注，新审核将安全暂存至本地浏览器。`
          );
        } else {
          showToast(`读取历史审核记录失败：${revErr?.message || "未知错误"}`, "warn");
        }
      }

      // Handle collections result
      if (collectionsRes.status === "fulfilled") {
        state.allCollections = collectionsRes.value.collections || [];
        state.collectionsOutputDir = collectionsRes.value.output_dir || "";
      } else {
        state.allCollections = [];
        const colErr = collectionsRes.reason;
        if (colErr && colErr.isCorrupt) {
          showCorruptBanner(
            `读取已采集清单失败（HTTP ${colErr.status}）`,
            `${colErr.message}。后端已保留原清单文件未覆盖，请检查输出目录中的 JSON 文件。`
          );
        }
      }

      applyCatalogData(catalogRes.value, false);
      if (collectionsRes.status === "fulfilled") {
        syncCatalogCollectionCounts();
      }
      renderBehaviorNav();
      renderClipList();
      renderAllCollectionsDrawer();
      updateHeaderStats();
      updateBatchSummary();
      showCatalogNotice("");

      if (!health.source_exists) {
        showCatalogNotice(`提示：源目录 ${health.source_dir || ""} 当前不存在或为空，请检查启动参数。`, true);
      }

      if (!health.ffmpeg_available) {
        showToast("提示：后端未检测到可用 FFmpeg，视频裁剪导出将受限。", "warn", 5000);
      }

      // Auto-flush pending real reviews stored while offline
      if (realPendingKeys().length > 0) {
        await flushPendingReviews();
      }

      if (!silentToast) {
        showToast(`已连接后端，共加载 ${state.clips.length} 个行为片段。`, "ok", 2600);
      }
    } catch (error) {
      if (state.bootController !== controller || (error && error.name === "AbortError")) return;
      state.connected = false;
      state.sessionToken = "";
      state.health = null;

      const reason = error?.message || "后端接口不可达或返回格式异常。";
      showOfflineBanner(reason);
      showCatalogNotice("");

      // Fall back to clearly labeled synthetic demo dataset so the workspace stays usable
      const demo = getDemoBundle();
      state.allCollections = [];
      applyCatalogData(demo.catalog, true);
      renderAllCollectionsDrawer();
    }
  }

  /* =========================================================================
   * Left Panel: Behaviors & Paginated Clip Browser
   * ========================================================================= */
  function renderBehaviorNav() {
    const nav = $("behavior-nav");
    const totalClips = state.clips.length;
    let totalReviewed = 0;
    let totalCollected = 0;

    const perBeh = new Map();
    for (const b of state.behaviors) {
      perBeh.set(b.name, { total: 0, reviewed: 0, collected: 0 });
    }
    for (const clip of state.clips) {
      const bucket = perBeh.get(clip.behavior) || { total: 0, reviewed: 0, collected: 0 };
      bucket.total += 1;
      const isRev = getEffectiveStatus(clip) !== "pending";
      const isCol = getCollectionCount(clip) > 0;
      if (isRev) {
        bucket.reviewed += 1;
        totalReviewed += 1;
      }
      if (isCol) {
        bucket.collected += 1;
        totalCollected += 1;
      }
      perBeh.set(clip.behavior, bucket);
    }

    const parts = [];
    const allPct = totalClips > 0 ? Math.round((totalReviewed / totalClips) * 100) : 0;
    parts.push(`
      <button type="button" class="behavior-item${state.selectedBehavior === "__ALL__" ? " active" : ""}" data-behavior="__ALL__">
        <div class="behavior-top">
          <span><span class="behavior-cn">全部行为</span><span class="behavior-en">All</span></span>
          <span class="behavior-count-pill mono">${totalClips}</span>
        </div>
        <div class="behavior-sub mono">
          <span>已审 ${totalReviewed}/${totalClips}</span>
          <span>已采 ${totalCollected}</span>
        </div>
        <div class="behavior-mini-bar"><i style="width:${allPct}%"></i></div>
      </button>
    `);

    for (const b of state.behaviors) {
      const stats = perBeh.get(b.name) || { total: b.count, reviewed: 0, collected: 0 };
      const pct = stats.total > 0 ? Math.round((stats.reviewed / stats.total) * 100) : 0;
      const isActive = state.selectedBehavior === b.name;
      parts.push(`
        <button type="button" class="behavior-item${isActive ? " active" : ""}" data-behavior="${escHtml(b.name)}">
          <div class="behavior-top">
            <span><span class="behavior-cn">${escHtml(behaviorCn(b.name))}</span><span class="behavior-en">${escHtml(b.name)}</span></span>
            <span class="behavior-count-pill mono">${stats.total}</span>
          </div>
          <div class="behavior-sub mono">
            <span>已审 ${stats.reviewed}/${stats.total}</span>
            <span>已采 ${stats.collected}</span>
          </div>
          <div class="behavior-mini-bar"><i style="width:${pct}%"></i></div>
        </button>
      `);
    }

    nav.innerHTML = parts.join("");
  }

  function applyFiltersAndRender({ preserveSelection = false } = {}) {
    const q = state.searchQuery.trim().toLowerCase();
    const stFilter = state.statusFilter;
    const behFilter = state.selectedBehavior;

    state.filteredClips = state.clips.filter((clip) => {
      if (behFilter !== "__ALL__" && clip.behavior !== behFilter) return false;
      const effStatus = getEffectiveStatus(clip);
      const colCount = getCollectionCount(clip);

      if (stFilter === "collected") {
        if (colCount <= 0) return false;
      } else if (stFilter !== "all") {
        if (effStatus !== stFilter) return false;
      }

      if (q) {
        const trackStr = Array.isArray(clip.target_mouse_ids)
          ? clip.target_mouse_ids.map((id) => `${id} id${id} id ${id}`).join(" ")
          : "";
        const hay = `${clip.clip_id} ${clip.clip_uid} ${clip.source_video || ""} ${clip.behavior} ${behaviorCn(clip.behavior)} ${trackStr}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    const totalPages = Math.max(1, Math.ceil(state.filteredClips.length / PAGE_SIZE));

    if (preserveSelection && state.activeUid) {
      const activeIdx = state.filteredClips.findIndex((c) => c.clip_uid === state.activeUid);
      if (activeIdx >= 0) {
        state.page = Math.floor(activeIdx / PAGE_SIZE) + 1;
      } else {
        state.page = clamp(state.page, 1, totalPages);
      }
    } else {
      state.page = 1;
    }

    renderClipList();

    // Auto-select first matching clip if nothing selected or active clip no longer in dataset
    if (state.filteredClips.length > 0) {
      const stillVisible = state.activeUid && state.filteredClips.some((c) => c.clip_uid === state.activeUid);
      if (!stillVisible) {
        selectClip(state.filteredClips[0].clip_uid);
      } else {
        updateClipNavigationButtons();
      }
    } else {
      state.activeUid = null;
      resetPlayerAndDetail();
      $("active-clip-title").textContent = "当前筛选下无片段";
      $("active-clip-meta").textContent = "可调整左侧行为、状态或搜索关键词";
      $("active-behavior-tag").hidden = true;
      $("active-demo-tag").hidden = true;
      updateClipNavigationButtons();
    }
  }

  function renderClipList() {
    const total = state.filteredClips.length;
    $("filtered-count").textContent = `${total} 条`;

    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    state.page = clamp(state.page, 1, totalPages);

    $("page-info").textContent = `第 ${state.page} / ${totalPages} 页`;
    $("page-prev-btn").disabled = state.page <= 1;
    $("page-next-btn").disabled = state.page >= totalPages;

    const listEl = $("clip-list");
    const emptyEl = $("clip-list-empty");

    if (total === 0) {
      listEl.innerHTML = "";
      emptyEl.hidden = false;
      return;
    }
    emptyEl.hidden = true;

    const startIdx = (state.page - 1) * PAGE_SIZE;
    const slice = state.filteredClips.slice(startIdx, startIdx + PAGE_SIZE);

    const html = slice.map((clip) => {
      const effStatus = getEffectiveStatus(clip);
      const stInfo = STATUS_META[effStatus] || STATUS_META.pending;
      const colCount = getCollectionCount(clip);
      const isUnsynced = Boolean(state.pendingReviews[clip.clip_uid]);
      const isActive = clip.clip_uid === state.activeUid;
      const idsLabel = Array.isArray(clip.target_mouse_ids) && clip.target_mouse_ids.length
        ? clip.target_mouse_ids.map((id) => `#${id}`).join(", ")
        : "无标注ID";

      const missingTags = [];
      if (clip.has_tracks === false) missingTags.push("缺轨迹");
      if (clip.has_annotation === false) missingTags.push("缺标注");

      return `
        <button type="button" class="clip-card${isActive ? " active" : ""}" role="option" aria-selected="${isActive ? "true" : "false"}" data-clip-uid="${escHtml(clip.clip_uid)}" title="${escHtml(clip.clip_id)}">
          <div class="clip-card-head">
            <span class="clip-card-id mono">${escHtml(clip.clip_id)}</span>
            <span class="clip-card-badges">
              ${isUnsynced ? '<span class="status-badge unsynced">待同步</span>' : ""}
              ${colCount > 0 ? `<span class="status-badge collected mono">采 ${colCount}</span>` : ""}
              <span class="status-badge ${escHtml(stInfo.cls)}">${escHtml(stInfo.label)}</span>
            </span>
          </div>
          <div class="clip-card-meta mono">
            <span>${escHtml(behaviorCn(clip.behavior))} · 目标 ${escHtml(idsLabel)}</span>
            <span>${clip.frame_count}帧 · ${formatDuration(clip.frame_count, clip.fps)}</span>
          </div>
          <div class="clip-card-source mono">${escHtml(clip.source_video || "未记录原视频")}${missingTags.length ? ` · ${escHtml(missingTags.join("/"))}` : ""}</div>
        </button>
      `;
    });

    listEl.innerHTML = html.join("");
  }

  function updateClipNavigationButtons() {
    const idx = state.filteredClips.findIndex((c) => c.clip_uid === state.activeUid);
    $("clip-prev-btn").disabled = idx <= 0;
    $("clip-next-btn").disabled = idx < 0 || idx >= state.filteredClips.length - 1;
    $("clip-position").textContent = idx >= 0 ? `${idx + 1} / ${state.filteredClips.length}` : "— / —";
  }

  function stepClip(delta) {
    const idx = state.filteredClips.findIndex((c) => c.clip_uid === state.activeUid);
    if (idx < 0) return;
    const nextIdx = idx + delta;
    if (nextIdx < 0 || nextIdx >= state.filteredClips.length) return;
    const target = state.filteredClips[nextIdx];
    const targetPage = Math.floor(nextIdx / PAGE_SIZE) + 1;
    if (targetPage !== state.page) {
      state.page = targetPage;
      renderClipList();
    }
    selectClip(target.clip_uid);
  }

  /* =========================================================================
   * Clean Video Reset & Clip Detail Loading
   * ========================================================================= */
  function cancelVideoCallbacks() {
    if (state.rvfcId !== null && typeof videoEl.cancelVideoFrameCallback === "function") {
      try {
        videoEl.cancelVideoFrameCallback(state.rvfcId);
      } catch {
        // ignore
      }
    }
    state.rvfcId = null;

    if (state.demoPlayRaf !== null) {
      window.cancelAnimationFrame(state.demoPlayRaf);
      state.demoPlayRaf = null;
    }
    state.demoPlaying = false;
  }

  function resetPlayerAndDetail() {
    state.detailController?.abort();
    cancelVideoCallbacks();

    videoEl.pause();
    videoEl.removeAttribute("src");
    videoEl.load();

    state.detail = null;
    state.framesMap.clear();
    state.skeletonEdges = DEFAULT_SKELETON_EDGES;
    state.currentFrame = 0;
    state.videoError = false;
    state.hoveredTrackId = null;
    state.cropStart = 0;
    state.cropEnd = 1;

    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    visCtx.clearRect(0, 0, visCanvasEl.width, visCanvasEl.height);

    $("video-placeholder").hidden = false;
    $("video-error-overlay").hidden = true;
    $("canvas-hover-tip").hidden = true;
    $("missing-target-badge").hidden = true;
    $("visible-mice-readout").textContent = "当前帧可见鼠：—";
    $("frame-clock").textContent = "帧 0 / 0 · 0.00s / 0.00s";
    $("play-pause-label").textContent = "播放";

    const controlIds = [
      "play-pause-btn",
      "step-back-10",
      "step-back-1",
      "step-fwd-1",
      "step-fwd-10",
      "set-in-btn",
      "set-out-btn",
      "loop-preview-btn",
      "align-annotation-btn",
      "fullscreen-btn",
      "save-review-btn",
      "submit-collect-btn",
    ];
    for (const id of controlIds) {
      $(id).disabled = true;
    }
  }

  function showDetailStatus(message, isError = false) {
    const box = $("detail-status");
    $("detail-status-text").textContent = message;
    box.hidden = !message;
    box.classList.toggle("error", isError);
    $("retry-detail-btn").hidden = !isError;
  }

  async function selectClip(clipUid) {
    resetPlayerAndDetail();
    state.activeUid = clipUid;
    renderClipList();
    updateClipNavigationButtons();

    const summary = state.clipsByUid.get(clipUid);
    if (!summary) return;

    $("active-clip-title").textContent = summary.clip_id;
    $("active-behavior-tag").textContent = `${behaviorCn(summary.behavior)} · ${summary.behavior}`;
    $("active-behavior-tag").hidden = false;
    $("active-demo-tag").hidden = !isDemoUid(clipUid);
    $("active-clip-meta").textContent = `${summary.width || 1920}×${summary.height || 1080} · ${(summary.fps || 30).toFixed(2)} fps · ${summary.frame_count} 帧 · 来源：${summary.source_video || "—"}`;

    showDetailStatus("正在读取逐帧轨迹与片段元信息…", false);

    const controller = new AbortController();
    state.detailController = controller;

    try {
      let detail;
      if (state.offlineMode || isDemoUid(clipUid)) {
        detail = getDemoBundle().details.get(clipUid);
        if (!detail) throw new Error("未找到对应的离线演示片段。");
      } else {
        detail = await fetchJson(`/api/clip/${encodeURIComponent(clipUid)}`, {
          signal: controller.signal,
          timeoutMs: 20000,
          validate: validateClipDetail,
        });
      }

      if (state.detailController !== controller || state.activeUid !== clipUid) return;

      applyClipDetail(detail);
      showDetailStatus("");
    } catch (error) {
      if (state.detailController !== controller || (error && error.name === "AbortError")) return;
      showDetailStatus(`加载片段详情失败：${error.message}`, true);
    }
  }

  function getDefaultCropRangeFromDetail(detail) {
    const total = detail.video.frame_count;
    const rev = getEffectiveReview(detail.clip.clip_uid);
    if (
      rev &&
      Array.isArray(rev.crop_range) &&
      rev.crop_range.length === 2 &&
      Number.isInteger(rev.crop_range[0]) &&
      Number.isInteger(rev.crop_range[1]) &&
      rev.crop_range[0] >= 0 &&
      rev.crop_range[1] > rev.crop_range[0] &&
      rev.crop_range[1] <= total
    ) {
      return [rev.crop_range[0], rev.crop_range[1]];
    }
    return getOriginalAnnotationHalfOpen(detail);
  }

  function getOriginalAnnotationHalfOpen(detail) {
    const total = detail.video.frame_count;
    const annRange = detail.annotation?.frame_range;
    if (annRange && Number.isInteger(annRange.start) && Number.isInteger(annRange.end)) {
      const s = clamp(annRange.start, 0, Math.max(0, total - 1));
      const e = clamp(annRange.end + 1, s + 1, total);
      return [s, e];
    }
    const catRange = detail.clip?.annotation_range;
    if (Array.isArray(catRange) && catRange.length === 2 && Number.isInteger(catRange[0]) && Number.isInteger(catRange[1])) {
      const s = clamp(catRange[0], 0, Math.max(0, total - 1));
      const e = clamp(catRange[1] + 1, s + 1, total);
      return [s, e];
    }
    return [0, total];
  }

  function applyClipDetail(detail) {
    state.detail = detail;
    state.framesMap = new Map();
    for (const f of detail.frames || []) {
      if (Number.isInteger(f.frame)) {
        state.framesMap.set(f.frame, f);
      }
    }

    const edges = detail.metadata?.pose?.skeleton_edges;
    state.skeletonEdges =
      Array.isArray(edges) && edges.length > 0 ? edges : DEFAULT_SKELETON_EDGES;

    if (detail.review && !state.pendingReviews[detail.clip.clip_uid]) {
      state.reviews[detail.clip.clip_uid] = detail.review;
    }

    const vw = detail.video.width || 1920;
    const vh = detail.video.height || 1080;
    $("video-stage").style.setProperty("--video-ratio", String(vw / vh));

    const scaleInfo = detail.metadata?.distance_scale;
    const scaleText =
      scaleInfo && isFiniteNum(scaleInfo.cm_per_pixel)
        ? ` · 标尺 ${scaleInfo.cm_per_pixel.toFixed(4)} ${scaleInfo.unit || "cm"}/px`
        : "";
    $("active-clip-meta").textContent = `${vw}×${vh} · ${detail.video.fps.toFixed(2)} fps · ${detail.video.frame_count} 帧 (${formatDuration(detail.video.frame_count, detail.video.fps)}) · 来源：${detail.clip.source_video || detail.metadata?.source?.video_filename || "—"}${scaleText}`;

    // Initialize crop range [start, end)
    const [initStart, initEnd] = getDefaultCropRangeFromDetail(detail);
    state.cropStart = initStart;
    state.cropEnd = initEnd;
    state.currentFrame = initStart;

    // Initialize Review Form & Manual Collect Form
    initCollectFormFromDetail(detail);
    loadReviewFormFromState(detail.clip.clip_uid);

    // Enable Controls
    const enableIds = [
      "play-pause-btn",
      "step-back-10",
      "step-back-1",
      "step-fwd-1",
      "step-fwd-10",
      "set-in-btn",
      "set-out-btn",
      "loop-preview-btn",
      "align-annotation-btn",
      "save-review-btn",
      "submit-collect-btn",
    ];
    for (const id of enableIds) {
      $(id).disabled = false;
    }
    $("fullscreen-btn").disabled = typeof $("video-shell").requestFullscreen !== "function";

    // Bind Video Source or Demo Canvas
    $("video-placeholder").hidden = true;
    if (detail.video.url && !state.offlineMode) {
      state.videoError = false;
      $("video-error-overlay").hidden = true;
      videoEl.src = detail.video.url;
      videoEl.playbackRate = Number($("playback-speed").value) || 1;
    } else {
      state.videoError = true;
      $("video-error-overlay").hidden = false;
      $("video-error-text").textContent = state.offlineMode
        ? "当前处于离线演示模式：使用合成轨迹画布模拟播放与点选（非实时 F:\\mouse 视频）。"
        : "视频地址缺失，当前展示实测轨迹画布。";
    }

    renderMetadataSection();
    renderClipCollections();
    syncTimelineUI();
    renderCollectMouseChips();
    renderVisibilityStrip();
    resizeOverlayCanvas();
    drawCurrentFrame();
  }

  /* =========================================================================
   * Accurate object-fit: contain Mapping, Canvas Overlay & Click-to-Pick
   * ========================================================================= */
  function getIntrinsicVideoSize() {
    const vw = videoEl.videoWidth || state.detail?.video?.width || 1920;
    const vh = videoEl.videoHeight || state.detail?.video?.height || 1080;
    return { vw, vh };
  }

  function getVideoContentRect() {
    const stage = $("video-stage");
    const cw = stage.clientWidth;
    const ch = stage.clientHeight;
    const { vw, vh } = getIntrinsicVideoSize();

    if (!cw || !ch || !vw || !vh) {
      return { cw, ch, vw, vh, offsetX: 0, offsetY: 0, drawW: cw, drawH: ch, scale: 1 };
    }
    const containerRatio = cw / ch;
    const videoRatio = vw / vh;
    let drawW;
    let drawH;
    let offsetX;
    let offsetY;
    if (containerRatio > videoRatio) {
      drawH = ch;
      drawW = ch * videoRatio;
      offsetX = (cw - drawW) / 2;
      offsetY = 0;
    } else {
      drawW = cw;
      drawH = cw / videoRatio;
      offsetX = 0;
      offsetY = (ch - drawH) / 2;
    }
    return { cw, ch, vw, vh, offsetX, offsetY, drawW, drawH, scale: drawW / vw };
  }

  function resizeOverlayCanvas() {
    const stage = $("video-stage");
    const cw = stage.clientWidth;
    const ch = stage.clientHeight;
    if (!cw || !ch) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetW = Math.round(cw * dpr);
    const targetH = Math.round(ch * dpr);
    if (canvasEl.width !== targetW || canvasEl.height !== targetH) {
      canvasEl.width = targetW;
      canvasEl.height = targetH;
    }
    canvasCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function getRoleForTrackId(trackId) {
    const roles = DIRECTIONAL_ROLES[getEffectiveCollectBehavior()];
    if (roles) {
      for (const r of roles) {
        if (state.collectDraft.roleAssignments[r.role_name] === trackId) {
          return ROLE_CN[r.role_name] || r.role_name;
        }
      }
    }
    const annParts = state.detail?.annotation?.participants;
    if (Array.isArray(annParts) && getEffectiveCollectBehavior() === state.detail?.clip?.behavior) {
      for (const p of annParts) {
        if (Array.isArray(p.track_ids) && p.track_ids.includes(trackId)) {
          return ROLE_CN[p.role_name] || p.role_name;
        }
      }
    }
    return "";
  }

  function getActiveTargetSet() {
    if (state.collectDraft.mouseIds.length > 0) {
      return new Set(state.collectDraft.mouseIds);
    }
    return new Set(state.detail?.clip?.target_mouse_ids || []);
  }

  function drawDemoArenaBackdrop(rect) {
    const { offsetX, offsetY, drawW, drawH } = rect;
    canvasCtx.save();
    canvasCtx.fillStyle = "#121d2b";
    canvasCtx.fillRect(offsetX, offsetY, drawW, drawH);

    // Arena boundary & subtle lab grid
    canvasCtx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    canvasCtx.lineWidth = 1;
    const cols = 12;
    const rows = 7;
    for (let i = 1; i < cols; i++) {
      const x = offsetX + (drawW * i) / cols;
      canvasCtx.beginPath();
      canvasCtx.moveTo(x, offsetY);
      canvasCtx.lineTo(x, offsetY + drawH);
      canvasCtx.stroke();
    }
    for (let j = 1; j < rows; j++) {
      const y = offsetY + (drawH * j) / rows;
      canvasCtx.beginPath();
      canvasCtx.moveTo(offsetX, y);
      canvasCtx.lineTo(offsetX + drawW, y);
      canvasCtx.stroke();
    }

    canvasCtx.strokeStyle = "rgba(56, 189, 248, 0.25)";
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeRect(offsetX + 18, offsetY + 18, drawW - 36, drawH - 36);

    canvasCtx.fillStyle = "rgba(253, 230, 138, 0.55)";
    canvasCtx.font = "600 12px sans-serif";
    canvasCtx.textBaseline = "top";
    canvasCtx.fillText("演示回退模式 · 合成实验箱轨迹（非实时 F:\\mouse 数据）", offsetX + 26, offsetY + 26);
    canvasCtx.restore();
  }

  function drawCurrentFrame() {
    const detail = state.detail;
    const rect = getVideoContentRect();
    resizeOverlayCanvas();
    canvasCtx.clearRect(0, 0, rect.cw, rect.ch);

    if (!detail) return;

    const total = detail.video.frame_count;
    const fps = detail.video.fps;
    const fIdx = clamp(state.currentFrame, 0, total - 1);
    const frameObj = state.framesMap.get(fIdx);
    const detections = Array.isArray(frameObj?.detections) ? frameObj.detections : [];

    // Update Frame Clock & Playheads
    const sec = fIdx / fps;
    const totalSec = total / fps;
    $("frame-clock").textContent = `帧 ${fIdx} / ${total - 1} · ${sec.toFixed(2)}s / ${totalSec.toFixed(2)}s`;
    updatePlayheadPositions(fIdx, total);

    // Check Missing Target Mice (explicit warning, NEVER fabricate or interpolate!)
    const selectedSet = new Set(state.collectDraft.mouseIds);
    const checkTargets = selectedSet.size > 0
      ? [...selectedSet]
      : (detail.clip.target_mouse_ids || []);
    const visibleIds = detections.map((d) => d.track_id).filter(Number.isInteger);
    const visibleSet = new Set(visibleIds);
    const missingIds = checkTargets.filter((id) => !visibleSet.has(id));

    $("visible-mice-readout").textContent = visibleIds.length
      ? `当前帧可见鼠：${visibleIds.map((id) => `ID ${id}`).join("、")}`
      : "当前帧可见鼠：无实测检测";

    const missingBadge = $("missing-target-badge");
    if (missingIds.length > 0) {
      missingBadge.hidden = false;
      missingBadge.textContent = `⚠ 缺失目标鼠：${missingIds.map((id) => `ID ${id}`).join("、")}（仅展示实测轨迹，不插值补框）`;
    } else {
      missingBadge.hidden = true;
      missingBadge.textContent = "";
    }

    if (state.offlineMode || state.videoError) {
      drawDemoArenaBackdrop(rect);
    }

    const showTargets = $("toggle-targets").checked;
    const showOthers = $("toggle-others").checked;
    const showSkeleton = $("toggle-skeleton").checked;
    const showLabels = $("toggle-labels").checked;

    const origTargets = new Set(detail.clip.target_mouse_ids || []);
    const { offsetX, offsetY, scale } = rect;

    // Draw non-targets first, targets on top
    const ordered = [...detections].sort((a, b) => {
      const at = selectedSet.has(a.track_id) ? 1 : 0;
      const bt = selectedSet.has(b.track_id) ? 1 : 0;
      return at - bt;
    });

    const pendingLabels = [];
    const occupiedRects = [];

    for (const det of ordered) {
      const tid = det.track_id;
      const box = det.box;
      if (!Array.isArray(box) || box.length !== 4 || !box.every(isFiniteNum)) continue;

      const isSelected = selectedSet.has(tid);
      const isHovered = state.hoveredTrackId === tid;
      if (isSelected && !showTargets) continue;
      if (!isSelected && !showOthers) continue;

      const x1 = offsetX + box[0] * scale;
      const y1 = offsetY + box[1] * scale;
      const x2 = offsetX + box[2] * scale;
      const y2 = offsetY + box[3] * scale;
      const bw = Math.max(2, x2 - x1);
      const bh = Math.max(2, y2 - y1);

      canvasCtx.save();
      if (isSelected) {
        canvasCtx.strokeStyle = "#22c55e";
        canvasCtx.lineWidth = isHovered ? 3.5 : 2.5;
        canvasCtx.setLineDash([]);
        if (isHovered) {
          canvasCtx.fillStyle = "rgba(34, 197, 94, 0.16)";
          canvasCtx.fillRect(x1, y1, bw, bh);
        }
      } else {
        canvasCtx.strokeStyle = isHovered ? "#7dd3fc" : "rgba(56, 189, 248, 0.82)";
        canvasCtx.lineWidth = isHovered ? 2.5 : 1.6;
        canvasCtx.setLineDash([6, 4]);
        if (isHovered) {
          canvasCtx.fillStyle = "rgba(56, 189, 248, 0.14)";
          canvasCtx.fillRect(x1, y1, bw, bh);
        }
      }
      canvasCtx.strokeRect(x1, y1, bw, bh);
      canvasCtx.restore();

      // Draw 7-keypoint skeleton if toggled on
      if (showSkeleton && Array.isArray(det.keypoints) && det.keypoints.length > 0) {
        const kps = det.keypoints;
        const validPt = (idx) => {
          const p = kps[idx];
          if (!Array.isArray(p) || p.length < 2 || !isFiniteNum(p[0]) || !isFiniteNum(p[1])) return null;
          const conf = p.length >= 3 && isFiniteNum(p[2]) ? p[2] : 1;
          if (conf < 0.05 || (p[0] === 0 && p[1] === 0)) return null;
          return { x: offsetX + p[0] * scale, y: offsetY + p[1] * scale, conf };
        };

        canvasCtx.save();
        canvasCtx.strokeStyle = isSelected ? "rgba(134, 239, 172, 0.92)" : "rgba(125, 211, 252, 0.65)";
        canvasCtx.lineWidth = isSelected ? 2 : 1.3;
        for (const edge of state.skeletonEdges) {
          if (!Array.isArray(edge) || edge.length !== 2) continue;
          const pA = validPt(edge[0]);
          const pB = validPt(edge[1]);
          if (!pA || !pB) continue;
          canvasCtx.beginPath();
          canvasCtx.moveTo(pA.x, pA.y);
          canvasCtx.lineTo(pB.x, pB.y);
          canvasCtx.stroke();
        }
        for (let k = 0; k < kps.length; k++) {
          const pt = validPt(k);
          if (!pt) continue;
          canvasCtx.beginPath();
          canvasCtx.arc(pt.x, pt.y, k === 0 ? 3.8 : 2.6, 0, Math.PI * 2);
          canvasCtx.fillStyle = k === 0 ? "#facc15" : isSelected ? "#4ade80" : "#38bdf8";
          canvasCtx.fill();
          canvasCtx.strokeStyle = "rgba(15, 23, 42, 0.85)";
          canvasCtx.lineWidth = 1;
          canvasCtx.stroke();
        }
        canvasCtx.restore();
      }

      if (showLabels) {
        const roleTag = isSelected ? getRoleForTrackId(tid) : "";
        const origHint = !isSelected && origTargets.has(tid) ? " [原]" : "";
        const text = roleTag ? `ID ${tid} · ${roleTag}` : `ID ${tid}${origHint}`;

        canvasCtx.font = "700 11.5px 'Segoe UI', 'Microsoft YaHei', sans-serif";
        const tw = canvasCtx.measureText(text).width + 10;
        const th = 19;
        const lx = clamp(x1, offsetX + 2, Math.max(offsetX + 2, offsetX + rect.drawW - tw - 2));
        let ly = clamp(y1 - th - 4, offsetY + 2, Math.max(offsetY + 2, offsetY + rect.drawH - th - 2));

        for (let attempt = 0; attempt < ordered.length * 2 + 6; attempt++) {
          const shift = Math.ceil(attempt / 2) * (th + 3) * (attempt % 2 ? -1 : 1);
          const candY = clamp(y1 - th - 4 + shift, offsetY + 2, Math.max(offsetY + 2, offsetY + rect.drawH - th - 2));
          const overlaps = occupiedRects.some(
            (r) => lx < r.x + r.w + 3 && lx + tw + 3 > r.x && candY < r.y + r.h + 3 && candY + th + 3 > r.y
          );
          if (!overlaps) {
            ly = candY;
            break;
          }
        }
        occupiedRects.push({ x: lx, y: ly, w: tw, h: th });
        pendingLabels.push({
          text,
          lx,
          ly,
          tw,
          th,
          boxCx: (x1 + x2) / 2,
          boxTop: y1,
          isSelected,
        });
      }
    }

    // Draw label leader lines and high-contrast pills
    for (const lbl of pendingLabels) {
      if (Math.abs(lbl.ly + lbl.th - lbl.boxTop) > 8) {
        canvasCtx.save();
        canvasCtx.strokeStyle = lbl.isSelected ? "rgba(34, 197, 94, 0.85)" : "rgba(56, 189, 248, 0.7)";
        canvasCtx.lineWidth = 1.2;
        canvasCtx.beginPath();
        canvasCtx.moveTo(lbl.boxCx, lbl.boxTop);
        canvasCtx.lineTo(lbl.lx + lbl.tw / 2, lbl.ly + lbl.th);
        canvasCtx.stroke();
        canvasCtx.restore();
      }
    }
    for (const lbl of pendingLabels) {
      canvasCtx.save();
      canvasCtx.fillStyle = lbl.isSelected ? "#16a34a" : "rgba(15, 23, 42, 0.84)";
      canvasCtx.strokeStyle = lbl.isSelected ? "#bbf7d0" : "rgba(56, 189, 248, 0.75)";
      canvasCtx.lineWidth = 1;
      canvasCtx.beginPath();
      if (typeof canvasCtx.roundRect === "function") {
        canvasCtx.roundRect(lbl.lx, lbl.ly, lbl.tw, lbl.th, 4);
      } else {
        canvasCtx.rect(lbl.lx, lbl.ly, lbl.tw, lbl.th);
      }
      canvasCtx.fill();
      canvasCtx.stroke();

      canvasCtx.fillStyle = "#ffffff";
      canvasCtx.font = "700 11.5px 'Segoe UI', 'Microsoft YaHei', sans-serif";
      canvasCtx.textBaseline = "middle";
      canvasCtx.fillText(lbl.text, lbl.lx + 5, lbl.ly + lbl.th / 2 + 0.5);
      canvasCtx.restore();
    }
  }

  function hitTestDetectionAtClientPos(clientX, clientY) {
    if (!state.detail) return null;
    const frameObj = state.framesMap.get(state.currentFrame);
    const detections = Array.isArray(frameObj?.detections) ? frameObj.detections : [];
    if (!detections.length) return null;

    const bounds = canvasEl.getBoundingClientRect();
    const cssX = clientX - bounds.left;
    const cssY = clientY - bounds.top;
    const rect = getVideoContentRect();
    if (!rect.scale) return null;

    const vx = (cssX - rect.offsetX) / rect.scale;
    const vy = (cssY - rect.offsetY) / rect.scale;
    const margin = 6 / rect.scale;

    let best = null;
    let bestArea = Infinity;
    for (const det of detections) {
      const box = det.box;
      if (!Array.isArray(box) || box.length !== 4 || !box.every(isFiniteNum)) continue;
      if (vx >= box[0] - margin && vx <= box[2] + margin && vy >= box[1] - margin && vy <= box[3] + margin) {
        const area = Math.max(1, (box[2] - box[0]) * (box[3] - box[1]));
        if (area < bestArea) {
          bestArea = area;
          best = det;
        }
      }
    }
    return best ? { detection: best, cssX, cssY } : null;
  }

  function handleCanvasPointerMove(ev) {
    const hit = hitTestDetectionAtClientPos(ev.clientX, ev.clientY);
    const tip = $("canvas-hover-tip");
    if (hit) {
      const tid = hit.detection.track_id;
      const isSelected = state.collectDraft.mouseIds.includes(tid);
      canvasEl.classList.add("pick-hover");
      tip.hidden = false;
      tip.textContent = isSelected ? `ID ${tid}（点击取消目标）` : `ID ${tid}（点击设为目标）`;
      tip.style.left = `${clamp(hit.cssX + 12, 6, Math.max(6, canvasEl.clientWidth - 160))}px`;
      tip.style.top = `${clamp(hit.cssY - 28, 6, Math.max(6, canvasEl.clientHeight - 30))}px`;
      if (state.hoveredTrackId !== tid) {
        state.hoveredTrackId = tid;
        drawCurrentFrame();
      }
    } else {
      canvasEl.classList.remove("pick-hover");
      tip.hidden = true;
      if (state.hoveredTrackId !== null) {
        state.hoveredTrackId = null;
        drawCurrentFrame();
      }
    }
  }

  function handleCanvasClick(ev) {
    const hit = hitTestDetectionAtClientPos(ev.clientX, ev.clientY);
    if (!hit) return;
    const tid = hit.detection.track_id;
    toggleCollectMouseId(tid, { fromCanvas: true });
  }

  /* =========================================================================
   * Video Playback, Frame Stepping & Loop Preview
   * ========================================================================= */
  function isPlaying() {
    if (state.offlineMode || state.videoError) return state.demoPlaying;
    return !videoEl.paused && !videoEl.ended;
  }

  function seekToFrame(frameIndex, { keepPlaying = false } = {}) {
    if (!state.detail) return;
    const total = state.detail.video.frame_count;
    const fps = state.detail.video.fps;
    const target = clamp(Math.round(frameIndex), 0, total - 1);
    state.currentFrame = target;

    if (!state.offlineMode && !state.videoError && videoEl.getAttribute("src")) {
      if (!keepPlaying && !videoEl.paused) {
        videoEl.pause();
      }
      const targetTime = Math.min((target + 0.08) / fps, Math.max(0, (total - 0.5) / fps));
      if (videoEl.readyState >= 1) {
        videoEl.currentTime = targetTime;
      }
    }
    drawCurrentFrame();
    updatePlayButtonLabel();
  }

  function stepFrames(delta) {
    if (!state.detail) return;
    pausePlayback();
    seekToFrame(state.currentFrame + delta);
  }

  function updatePlayButtonLabel() {
    $("play-pause-label").textContent = isPlaying() ? "暂停" : "播放";
  }

  function pausePlayback() {
    if (state.offlineMode || state.videoError) {
      state.demoPlaying = false;
      if (state.demoPlayRaf !== null) {
        window.cancelAnimationFrame(state.demoPlayRaf);
        state.demoPlayRaf = null;
      }
    } else if (!videoEl.paused) {
      videoEl.pause();
    }
    updatePlayButtonLabel();
  }

  function startDemoLoop() {
    state.demoPlaying = true;
    state.demoLastTick = performance.now();
    let accumFrames = 0;

    const tick = (now) => {
      if (!state.demoPlaying || !state.detail) return;
      const dt = Math.min(0.25, (now - state.demoLastTick) / 1000);
      state.demoLastTick = now;
      const speed = Number($("playback-speed").value) || 1;
      accumFrames += dt * state.detail.video.fps * speed;

      if (accumFrames >= 1) {
        const adv = Math.floor(accumFrames);
        accumFrames -= adv;
        let nextF = state.currentFrame + adv;
        if (state.loopPreview) {
          if (nextF >= state.cropEnd || nextF < state.cropStart) {
            nextF = state.cropStart;
          }
        } else if (nextF >= state.detail.video.frame_count) {
          nextF = state.detail.video.frame_count - 1;
          state.currentFrame = nextF;
          drawCurrentFrame();
          pausePlayback();
          return;
        }
        state.currentFrame = nextF;
        drawCurrentFrame();
      }
      state.demoPlayRaf = window.requestAnimationFrame(tick);
    };
    state.demoPlayRaf = window.requestAnimationFrame(tick);
    updatePlayButtonLabel();
  }

  async function togglePlayPause() {
    if (!state.detail) return;
    if (isPlaying()) {
      pausePlayback();
      return;
    }

    if (state.loopPreview && (state.currentFrame < state.cropStart || state.currentFrame >= state.cropEnd)) {
      seekToFrame(state.cropStart, { keepPlaying: true });
    } else if (state.currentFrame >= state.detail.video.frame_count - 1) {
      seekToFrame(0, { keepPlaying: true });
    }

    if (state.offlineMode || state.videoError) {
      startDemoLoop();
      return;
    }

    try {
      videoEl.playbackRate = Number($("playback-speed").value) || 1;
      await videoEl.play();
      scheduleVideoFrameSync();
      updatePlayButtonLabel();
    } catch {
      showToast("浏览器阻止了播放或视频尚未就绪，可手动步进帧查看。", "warn");
    }
  }

  function syncFrameFromMediaTime(mediaTime) {
    if (!state.detail) return;
    const { fps, frame_count: total } = state.detail.video;
    let fIdx = clamp(Math.round(mediaTime * fps), 0, total - 1);

    if (state.loopPreview && !videoEl.paused) {
      if (fIdx >= state.cropEnd || fIdx < state.cropStart) {
        seekToFrame(state.cropStart, { keepPlaying: true });
        return;
      }
    }
    state.currentFrame = fIdx;
    drawCurrentFrame();
  }

  function scheduleVideoFrameSync() {
    if (
      !state.detail ||
      videoEl.paused ||
      typeof videoEl.requestVideoFrameCallback !== "function" ||
      state.rvfcId !== null
    ) {
      return;
    }
    state.rvfcId = videoEl.requestVideoFrameCallback((_now, metadata) => {
      state.rvfcId = null;
      syncFrameFromMediaTime(metadata.mediaTime);
      scheduleVideoFrameSync();
    });
  }

  /* =========================================================================
   * Dual-Handle Crop Timeline & Target Visibility Strip
   * ========================================================================= */
  function setCropRange(newStart, newEnd, { seekTo = null } = {}) {
    if (!state.detail) return;
    const total = state.detail.video.frame_count;
    const s = clamp(Math.round(newStart), 0, Math.max(0, total - 1));
    const e = clamp(Math.round(newEnd), s + 1, total);

    state.cropStart = s;
    state.cropEnd = e;

    syncTimelineUI();
    renderCollectMouseChips();
    renderVisibilityStrip();
    renderMetadataSection();

    if (seekTo !== null) {
      seekToFrame(seekTo);
    }
  }

  function syncTimelineUI() {
    const detail = state.detail;
    if (!detail) return;
    const total = Math.max(1, detail.video.frame_count);
    const fps = detail.video.fps;

    const startPct = (state.cropStart / total) * 100;
    const endPct = (state.cropEnd / total) * 100;

    $("crop-fill").style.left = `${startPct}%`;
    $("crop-fill").style.width = `${Math.max(0.5, endPct - startPct)}%`;

    const hStart = $("handle-start");
    const hEnd = $("handle-end");
    hStart.style.left = `${startPct}%`;
    hEnd.style.left = `${endPct}%`;
    hStart.setAttribute("aria-valuemin", "0");
    hStart.setAttribute("aria-valuemax", String(Math.max(0, state.cropEnd - 1)));
    hStart.setAttribute("aria-valuenow", String(state.cropStart));
    hEnd.setAttribute("aria-valuemin", String(state.cropStart + 1));
    hEnd.setAttribute("aria-valuemax", String(total));
    hEnd.setAttribute("aria-valuenow", String(state.cropEnd));
    $("handle-start-tip").textContent = String(state.cropStart);
    $("handle-end-tip").textContent = String(state.cropEnd);

    const spanFrames = Math.max(0, state.cropEnd - state.cropStart);
    $("range-summary").textContent = `裁剪区间：[${state.cropStart}, ${state.cropEnd}) · ${spanFrames} 帧 (${formatDuration(spanFrames, fps)})`;
    $("review-crop-preview").textContent = `[${state.cropStart}, ${state.cropEnd})`;

    // Original Annotation Band
    const annFill = $("ann-band-fill");
    const annLabel = $("ann-band-label");
    const annRange = detail.annotation?.frame_range || (
      Array.isArray(detail.clip.annotation_range) && detail.clip.annotation_range.length === 2
        ? { start: detail.clip.annotation_range[0], end: detail.clip.annotation_range[1] }
        : null
    );
    if (annRange && Number.isInteger(annRange.start) && Number.isInteger(annRange.end)) {
      const aLeft = (clamp(annRange.start, 0, total) / total) * 100;
      const aRight = (clamp(annRange.end + 1, 0, total) / total) * 100;
      annFill.hidden = false;
      annFill.style.left = `${aLeft}%`;
      annFill.style.width = `${Math.max(0.5, aRight - aLeft)}%`;
      annLabel.textContent = `原标注区间 [${annRange.start}..${annRange.end}]（含两端，共 ${annRange.end - annRange.start + 1} 帧）`;
    } else {
      annFill.hidden = true;
      annLabel.textContent = "无原标注区间";
    }

    // Sync right-panel numeric inputs & duration preview
    const startInp = $("collect-start-input");
    const endInp = $("collect-end-input");
    startInp.max = String(Math.max(0, total - 1));
    endInp.max = String(total);
    if (document.activeElement !== startInp) startInp.value = String(state.cropStart);
    if (document.activeElement !== endInp) endInp.value = String(state.cropEnd);

    updateCollectDurationReadout();
    updatePlayheadPositions(state.currentFrame, total);
  }

  function updatePlayheadPositions(frameIdx, totalFrames) {
    const denom = Math.max(1, totalFrames);
    const pct = clamp(((frameIdx + 0.5) / denom) * 100, 0, 100);
    $("playhead-line").style.left = `${pct}%`;
    $("visibility-playhead").style.left = `${pct}%`;
  }

  function renderVisibilityStrip() {
    const detail = state.detail;
    const labelsEl = $("visibility-labels");
    if (!detail) {
      labelsEl.innerHTML = "";
      visCtx.clearRect(0, 0, visCanvasEl.width, visCanvasEl.height);
      return;
    }

    const total = Math.max(1, detail.video.frame_count);
    const targetIds = state.collectDraft.mouseIds.length > 0
      ? [...state.collectDraft.mouseIds]
      : (detail.clip.target_mouse_ids || []);

    const displayIds = targetIds.slice(0, 6);
    if (!displayIds.length) {
      labelsEl.innerHTML = "<span>无目标</span>";
      visCanvasEl.height = 20;
      visCtx.clearRect(0, 0, visCanvasEl.width, visCanvasEl.height);
      return;
    }

    const laneH = 12;
    const gap = 3;
    const cssH = displayIds.length * laneH + Math.max(0, displayIds.length - 1) * gap + 6;
    const boxW = Math.max(120, visCanvasEl.parentElement.clientWidth || 400);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    visCanvasEl.style.height = `${cssH}px`;
    visCanvasEl.width = Math.round(boxW * dpr);
    visCanvasEl.height = Math.round(cssH * dpr);
    visCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    visCtx.clearRect(0, 0, boxW, cssH);

    labelsEl.innerHTML = displayIds.map((id) => `<span>ID ${id}</span>`).join("");

    // Precompute frame presence sets for selected IDs
    const presence = new Map(displayIds.map((id) => [id, new Uint8Array(total)]));
    for (let f = 0; f < total; f++) {
      const fr = state.framesMap.get(f);
      if (!fr || !Array.isArray(fr.detections)) continue;
      for (const det of fr.detections) {
        const arr = presence.get(det.track_id);
        if (arr) arr[f] = 1;
      }
    }

    const pxPerFrame = boxW / total;
    displayIds.forEach((id, laneIdx) => {
      const y = 3 + laneIdx * (laneH + gap);
      const arr = presence.get(id);

      // Base dropout background
      visCtx.fillStyle = "#fee2e2";
      visCtx.fillRect(0, y, boxW, laneH);

      // Highlight measured visible spans
      let runStart = -1;
      for (let f = 0; f <= total; f++) {
        const present = f < total && arr[f] === 1;
        if (present && runStart < 0) {
          runStart = f;
        } else if (!present && runStart >= 0) {
          const x = runStart * pxPerFrame;
          const w = Math.max(1, (f - runStart) * pxPerFrame);
          visCtx.fillStyle = "#22c55e";
          visCtx.fillRect(x, y, w, laneH);
          runStart = -1;
        }
      }

      // Mark red stripes on missing frames inside [cropStart, cropEnd) for clarity
      for (let f = state.cropStart; f < state.cropEnd; f++) {
        if (arr[f] === 0) {
          visCtx.fillStyle = "#ef4444";
          visCtx.fillRect(f * pxPerFrame, y, Math.max(1.5, pxPerFrame), laneH);
        }
      }
    });

    // Subtle veil outside [cropStart, cropEnd)
    const cropLeft = (state.cropStart / total) * boxW;
    const cropRight = (state.cropEnd / total) * boxW;
    visCtx.fillStyle = "rgba(15, 23, 42, 0.22)";
    if (cropLeft > 0) visCtx.fillRect(0, 0, cropLeft, cssH);
    if (cropRight < boxW) visCtx.fillRect(cropRight, 0, boxW - cropRight, cssH);
  }

  function pointerEventToFrame(ev, allowExclusiveEnd = false) {
    if (!state.detail) return 0;
    const rail = $("timeline-rail");
    const bounds = rail.getBoundingClientRect();
    const frac = clamp((ev.clientX - bounds.left) / Math.max(1, bounds.width), 0, 1);
    const total = state.detail.video.frame_count;
    if (allowExclusiveEnd) {
      return clamp(Math.round(frac * total), 1, total);
    }
    return clamp(Math.floor(frac * total), 0, total - 1);
  }

  /* =========================================================================
   * Metadata & Per-Track Visibility Summary Table
   * ========================================================================= */
  function computeTrackVisibilityInRange(startF, endF) {
    const detail = state.detail;
    if (!detail) return [];
    const spanLen = Math.max(1, endF - startF);
    const byId = new Map();

    for (const vt of detail.visible_tracks || []) {
      byId.set(vt.track_id, {
        track_id: vt.track_id,
        total_frames: vt.frame_count,
        first_frame: vt.first_frame,
        last_frame: vt.last_frame,
        in_range_frames: 0,
      });
    }

    if (state.framesMap.size > 0) {
      for (let f = startF; f < endF; f++) {
        const fr = state.framesMap.get(f);
        if (!fr || !Array.isArray(fr.detections)) continue;
        const seen = new Set();
        for (const det of fr.detections) {
          if (!Number.isInteger(det.track_id) || seen.has(det.track_id)) continue;
          seen.add(det.track_id);
          if (!byId.has(det.track_id)) {
            byId.set(det.track_id, {
              track_id: det.track_id,
              total_frames: 1,
              first_frame: f,
              last_frame: f,
              in_range_frames: 0,
            });
          }
          byId.get(det.track_id).in_range_frames += 1;
        }
      }
    } else {
      for (const item of byId.values()) {
        const overlap = Math.max(0, Math.min(item.last_frame + 1, endF) - Math.max(item.first_frame, startF));
        item.in_range_frames = overlap;
      }
    }

    return [...byId.values()]
      .map((item) => ({
        ...item,
        span_frames: spanLen,
        coverage: item.in_range_frames / spanLen,
      }))
      .sort((a, b) => a.track_id - b.track_id);
  }

  function renderMetadataSection() {
    const detail = state.detail;
    if (!detail) return;

    const kv = [];
    kv.push(["片段 UID", detail.clip.clip_uid]);
    kv.push(["原标注行为", `${behaviorCn(detail.clip.behavior)} (${detail.clip.behavior})`]);
    kv.push(["原始视频", detail.metadata?.source?.video_filename || detail.clip.source_video || "—"]);
    if (detail.metadata?.source?.clip) {
      const sc = detail.metadata.source.clip;
      kv.push(["源视频片段帧", `${sc.start_frame} — ${sc.end_frame}`]);
    }
    if (detail.metadata?.distance_scale && isFiniteNum(detail.metadata.distance_scale.cm_per_pixel)) {
      kv.push(["空间比例尺", `${detail.metadata.distance_scale.cm_per_pixel.toFixed(5)} ${detail.metadata.distance_scale.unit || "cm"}/px`]);
    }
    kv.push(["文件完整性", `轨迹:${detail.clip.has_tracks ? "有" : "缺"} · 标注:${detail.clip.has_annotation ? "有" : "缺"} · 元数据:${detail.clip.has_metadata ? "有" : "缺"}`]);

    $("meta-kv-grid").innerHTML = kv
      .map(
        ([k, v]) => `
        <div class="meta-kv-item">
          <dt>${escHtml(k)}</dt>
          <dd class="mono">${escHtml(v)}</dd>
        </div>`
      )
      .join("");

    const rows = computeTrackVisibilityInRange(state.cropStart, state.cropEnd);
    const selectedSet = new Set(state.collectDraft.mouseIds);
    const origSet = new Set(detail.clip.target_mouse_ids || []);
    const tbody = $("track-coverage-tbody");

    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="muted">该片段暂无实测轨迹记录</td></tr>';
      return;
    }

    tbody.innerHTML = rows
      .map((row) => {
        const badges = [];
        if (selectedSet.has(row.track_id)) badges.push("当前目标");
        if (origSet.has(row.track_id)) badges.push("原标注");
        const roleName = getRoleForTrackId(row.track_id);
        if (roleName) badges.push(roleName);
        const pct = Math.round(row.coverage * 100);
        return `
          <tr>
            <td class="mono"><strong>ID ${row.track_id}</strong></td>
            <td>${escHtml(badges.join(" / ") || "可见轨迹")}</td>
            <td class="mono">${row.total_frames} 帧</td>
            <td class="mono">${row.in_range_frames} / ${row.span_frames} (${pct}%)</td>
            <td class="mono">${row.first_frame} — ${row.last_frame}</td>
          </tr>
        `;
      })
      .join("");
  }

  /* =========================================================================
   * Right Panel — Tab 1: Review Form & Persistence
   * ========================================================================= */
  function loadReviewFormFromState(clipUid) {
    const rev = getEffectiveReview(clipUid);
    const status = rev?.status && STATUS_META[rev.status] ? rev.status : "pending";
    const tags = Array.isArray(rev?.tags) ? rev.tags : [];
    const note = typeof rev?.note === "string" ? rev.note : "";
    const hasSavedCrop = Array.isArray(rev?.crop_range) && rev.crop_range.length === 2;

    state.reviewDraft = {
      status,
      tags: new Set(tags),
      note,
      attachCrop: hasSavedCrop || status === "needs_recrop",
    };

    syncReviewFormUI(clipUid);
  }

  function syncReviewFormUI(clipUid) {
    const st = state.reviewDraft.status;
    const stInfo = STATUS_META[st] || STATUS_META.pending;

    $("review-current-badge").textContent = stInfo.label;
    $("review-current-badge").className = `status-badge ${stInfo.cls}`;
    $("tab-review-dot").className = `status-dot ${stInfo.cls}`;

    for (const btn of document.querySelectorAll(".verdict-btn")) {
      btn.classList.toggle("selected", btn.dataset.verdict === st);
    }

    for (const chip of document.querySelectorAll("#reason-tags-group .tag-chip")) {
      chip.classList.toggle("selected", state.reviewDraft.tags.has(chip.dataset.tag));
    }

    $("review-attach-crop").checked = state.reviewDraft.attachCrop;
    $("review-note").value = state.reviewDraft.note;
    $("review-crop-preview").textContent = `[${state.cropStart}, ${state.cropEnd})`;

    const syncEl = $("review-sync-status");
    const pending = clipUid ? state.pendingReviews[clipUid] : null;
    const saved = clipUid ? state.reviews[clipUid] : null;

    if (pending) {
      syncEl.textContent = isDemoUid(clipUid)
        ? `演示审核已暂存浏览器 (${formatTimeStamp(pending.updated_at)})`
        : `待同步：已保存在本地队列 (${formatTimeStamp(pending.updated_at)})`;
      syncEl.className = "sync-status warn";
    } else if (saved && saved.updated_at) {
      syncEl.textContent = `已保存至后端 · ${formatTimeStamp(saved.updated_at)}`;
      syncEl.className = "sync-status ok";
    } else {
      syncEl.textContent = "尚未提交审核记录";
      syncEl.className = "sync-status muted";
    }
  }

  function selectReviewVerdict(verdict) {
    if (!STATUS_META[verdict]) return;
    state.reviewDraft.status = verdict;
    if (verdict === "needs_recrop") {
      state.reviewDraft.attachCrop = true;
    }
    syncReviewFormUI(state.activeUid);
  }

  async function saveCurrentReview() {
    if (!state.activeUid || !state.detail) return;
    const clipUid = state.activeUid;
    const cropRange = state.reviewDraft.attachCrop ? [state.cropStart, state.cropEnd] : null;

    const payload = {
      clip_uid: clipUid,
      status: state.reviewDraft.status,
      tags: [...state.reviewDraft.tags],
      note: $("review-note").value.trim(),
      crop_range: cropRange,
    };
    state.reviewDraft.note = payload.note;

    const saveBtn = $("save-review-btn");
    saveBtn.disabled = true;

    const finishAdvance = () => {
      if ($("auto-advance-check").checked) {
        stepClip(1);
      }
    };

    // Offline or Demo clip -> store locally without calling backend
    if (state.offlineMode || isDemoUid(clipUid)) {
      queuePendingReview(payload);
      const clip = state.clipsByUid.get(clipUid);
      if (clip) clip.review_status = payload.status;
      syncReviewFormUI(clipUid);
      renderBehaviorNav();
      renderClipList();
      updateBatchSummary();
      showToast(
        isDemoUid(clipUid) ? "演示审核已暂存至本地浏览器。" : "当前离线，审核已暂存本地队列，重连后自动同步。",
        "warn"
      );
      saveBtn.disabled = false;
      finishAdvance();
      return;
    }

    try {
      const res = await fetchJson("/api/reviews", {
        method: "POST",
        body: payload,
        timeoutMs: 12000,
        validate: validateReviewSave,
      });

      state.reviews[clipUid] = res.review;
      if (state.pendingReviews[clipUid]) {
        delete state.pendingReviews[clipUid];
        persistPendingReviews();
      }
      const clip = state.clipsByUid.get(clipUid);
      if (clip) clip.review_status = res.review.status;

      hideCorruptBanner();
      syncReviewFormUI(clipUid);
      renderBehaviorNav();
      renderClipList();
      updateHeaderStats();
      updateBatchSummary();
      showToast(`审核已保存：${STATUS_META[res.review.status]?.label || res.review.status}`, "ok", 2200);
      finishAdvance();
    } catch (error) {
      // NEVER lose user's pending edit!
      queuePendingReview(payload);
      const clip = state.clipsByUid.get(clipUid);
      if (clip) clip.review_status = payload.status;
      syncReviewFormUI(clipUid);
      renderBehaviorNav();
      renderClipList();
      updateBatchSummary();

      if (error.isCorrupt) {
        showCorruptBanner(
          `审核写入被后端拒绝（HTTP ${error.status}：存储文件损坏）`,
          `${error.message}。后端已保留损坏文件未覆盖，您的本次审核已安全保存在本地浏览器队列中。`
        );
        showToast("后端审核文件异常 (409/500)，已自动为您保留到本地队列。", "error", 5000);
      } else {
        showToast(`提交失败（${error.message}），已暂存本地队列防止丢失。`, "warn", 4500);
      }
    } finally {
      saveBtn.disabled = false;
    }
  }

  /* =========================================================================
   * Right Panel — Tab 2: Manual Behavior Clip Collection
   * ========================================================================= */
  function populateCollectBehaviorOptions() {
    const select = $("collect-behavior-select");
    const opts = [];
    for (const name of BEHAVIOR_ORDER) {
      opts.push(`<option value="${escHtml(name)}">${escHtml(behaviorCn(name))} · ${escHtml(name)}</option>`);
    }
    for (const name of EXTRA_COLLECT_BEHAVIORS) {
      opts.push(`<option value="${escHtml(name)}">${escHtml(behaviorCn(name))} · ${escHtml(name)}</option>`);
    }
    opts.push('<option value="__CUSTOM__">自定义行为名称…</option>');
    select.innerHTML = opts.join("");
  }

  function getEffectiveCollectBehavior() {
    if (state.collectDraft.behavior === "__CUSTOM__") {
      return state.collectDraft.customBehavior.trim();
    }
    return state.collectDraft.behavior;
  }

  function initCollectFormFromDetail(detail) {
    const clipBeh = detail.clip.behavior || BEHAVIOR_ORDER[0];
    const known = BEHAVIOR_ORDER.includes(clipBeh) || EXTRA_COLLECT_BEHAVIORS.includes(clipBeh);

    state.collectDraft.behavior = known ? clipBeh : "__CUSTOM__";
    state.collectDraft.customBehavior = known ? "" : clipBeh;
    $("collect-behavior-select").value = state.collectDraft.behavior;
    $("collect-behavior-custom").hidden = state.collectDraft.behavior !== "__CUSTOM__";
    $("collect-behavior-custom").value = state.collectDraft.customBehavior;

    const origIds = Array.isArray(detail.annotation?.mouse_ids)
      ? [...detail.annotation.mouse_ids]
      : Array.isArray(detail.clip.target_mouse_ids)
      ? [...detail.clip.target_mouse_ids]
      : [];
    state.collectDraft.mouseIds = [...new Set(origIds.filter(Number.isInteger))].sort((a, b) => a - b);

    // Prepopulate directional roles from annotation.participants if available
    state.collectDraft.roleAssignments = {};
    if (Array.isArray(detail.annotation?.participants)) {
      for (const p of detail.annotation.participants) {
        if (p && typeof p.role_name === "string" && Array.isArray(p.track_ids) && p.track_ids.length > 0) {
          state.collectDraft.roleAssignments[p.role_name] = p.track_ids[0];
        }
      }
    }

    state.collectDraft.note = "";
    $("collect-note").value = "";
    $("collect-result-hint").textContent = "";
    $("collect-result-hint").className = "sync-status muted";
    $("collect-warning-box").hidden = true;

    renderDirectionalRolesUI();
    updateCollectDurationReadout();
    updateCollectTabCount();
  }

  function toggleCollectMouseId(trackId, { fromCanvas = false } = {}) {
    if (!Number.isInteger(trackId)) return;
    const list = state.collectDraft.mouseIds;
    const idx = list.indexOf(trackId);
    let added = false;

    if (idx >= 0) {
      list.splice(idx, 1);
      // Remove from role assignments if assigned
      for (const [roleName, assigned] of Object.entries(state.collectDraft.roleAssignments)) {
        if (assigned === trackId) {
          state.collectDraft.roleAssignments[roleName] = null;
        }
      }
    } else {
      list.push(trackId);
      list.sort((a, b) => a - b);
      added = true;
      // Auto-assign into first empty directional role slot if directional behavior active
      const roles = DIRECTIONAL_ROLES[getEffectiveCollectBehavior()];
      if (roles) {
        for (const r of roles) {
          if (state.collectDraft.roleAssignments[r.role_name] == null) {
            state.collectDraft.roleAssignments[r.role_name] = trackId;
            break;
          }
        }
      }
    }

    renderCollectMouseChips();
    renderDirectionalRolesUI();
    renderVisibilityStrip();
    renderMetadataSection();
    drawCurrentFrame();

    if (fromCanvas) {
      showToast(added ? `已将 ID ${trackId} 加入采集目标` : `已将 ID ${trackId} 移出采集目标`, "info", 1600);
    }
  }

  function renderCollectMouseChips() {
    const container = $("collect-mouse-chips");
    if (!state.detail) {
      container.innerHTML = '<p class="empty-hint">载入片段后展示区间内可见轨迹</p>';
      $("collect-mice-count").textContent = "(0 只)";
      return;
    }

    const stats = computeTrackVisibilityInRange(state.cropStart, state.cropEnd);
    const byId = new Map(stats.map((s) => [s.track_id, s]));
    const spanFrames = Math.max(1, state.cropEnd - state.cropStart);

    // Show tracks overlapping [start, end) PLUS any currently selected IDs even if 0 overlap
    const visibleIds = new Set();
    for (const s of stats) {
      if (s.in_range_frames > 0) visibleIds.add(s.track_id);
    }
    for (const id of state.collectDraft.mouseIds) {
      visibleIds.add(id);
    }
    const sortedIds = [...visibleIds].sort((a, b) => a - b);
    $("collect-mice-count").textContent = `(${state.collectDraft.mouseIds.length} 只)`;

    if (!sortedIds.length) {
      container.innerHTML = '<p class="empty-hint">当前裁剪区间内未观测到小鼠轨迹</p>';
      return;
    }

    container.innerHTML = sortedIds
      .map((tid) => {
        const info = byId.get(tid);
        const vis = info ? info.in_range_frames : 0;
        const isSel = state.collectDraft.mouseIds.includes(tid);
        const dotCls = vis === 0 ? "absent" : vis < spanFrames ? "partial" : "";
        return `
          <button type="button" class="mouse-chip${isSel ? " selected" : ""}" data-track-id="${tid}" aria-pressed="${isSel ? "true" : "false"}" title="区间内可见 ${vis} / ${spanFrames} 帧">
            <span class="mouse-chip-dot ${dotCls}" aria-hidden="true"></span>
            <span class="mouse-chip-id mono">ID ${tid}</span>
            <span class="mouse-chip-cov mono">${vis}/${spanFrames}帧</span>
          </button>
        `;
      })
      .join("");
  }

  function renderDirectionalRolesUI() {
    const beh = getEffectiveCollectBehavior();
    const roles = DIRECTIONAL_ROLES[beh];
    const card = $("participants-card");
    const fields = $("participants-fields");

    if (!roles) {
      card.hidden = true;
      fields.innerHTML = "";
      return;
    }

    card.hidden = false;
    const candidates = [...state.collectDraft.mouseIds];

    // Auto-populate if exactly 2 candidates selected and no roles assigned yet
    if (candidates.length === 2) {
      const r0 = state.collectDraft.roleAssignments[roles[0].role_name];
      const r1 = state.collectDraft.roleAssignments[roles[1].role_name];
      if (r0 == null && r1 == null) {
        state.collectDraft.roleAssignments[roles[0].role_name] = candidates[0];
        state.collectDraft.roleAssignments[roles[1].role_name] = candidates[1];
      }
    }

    fields.innerHTML = roles
      .map((r) => {
        const currentVal = state.collectDraft.roleAssignments[r.role_name];
        const options = ['<option value="">未指定</option>'];
        for (const id of candidates) {
          options.push(
            `<option value="${id}"${currentVal === id ? " selected" : ""}>ID ${id}</option>`
          );
        }
        return `
          <div class="role-field">
            <label for="role-select-${escHtml(r.role_name)}">${escHtml(r.label)}</label>
            <select id="role-select-${escHtml(r.role_name)}" data-role-name="${escHtml(r.role_name)}">
              ${options.join("")}
            </select>
          </div>
        `;
      })
      .join("");
  }

  function buildParticipantsPayload() {
    const beh = getEffectiveCollectBehavior();
    const roles = DIRECTIONAL_ROLES[beh];
    if (!roles) return [];

    // Preserve original role_key from annotation.participants when role_name matches
    const origKeyMap = new Map();
    if (Array.isArray(state.detail?.annotation?.participants)) {
      for (const p of state.detail.annotation.participants) {
        if (p && typeof p.role_name === "string" && typeof p.role_key === "string") {
          origKeyMap.set(p.role_name, p.role_key);
        }
      }
    }

    const result = [];
    for (const r of roles) {
      const assigned = state.collectDraft.roleAssignments[r.role_name];
      if (Number.isInteger(assigned) && state.collectDraft.mouseIds.includes(assigned)) {
        result.push({
          role_key: origKeyMap.get(r.role_name) || r.role_key,
          role_name: r.role_name,
          track_ids: [assigned],
        });
      }
    }
    return result;
  }

  function updateCollectDurationReadout() {
    if (!state.detail) return;
    const total = state.detail.video.frame_count;
    const fps = state.detail.video.fps;
    const pad = Number($("collect-padding-select").value) || 0;
    state.collectDraft.paddingFrames = pad;

    const netFrames = Math.max(0, state.cropEnd - state.cropStart);
    $("collect-duration-preview").textContent = `净行为区间：[${state.cropStart}, ${state.cropEnd}) · ${netFrames} 帧 (${formatDuration(netFrames, fps)})`;

    if (pad > 0) {
      const clipS = Math.max(0, state.cropStart - pad);
      const clipE = Math.min(total, state.cropEnd + pad);
      const clipLen = clipE - clipS;
      $("collect-padded-preview").textContent = `含填充导出视频帧：[${clipS}, ${clipE}) · 共 ${clipLen} 帧 (${formatDuration(clipLen, fps)})，包内 annotation.frame_range 自动平移至 [${state.cropStart - clipS}..${state.cropEnd - 1 - clipS}]`;
    } else {
      $("collect-padded-preview").textContent = "0 帧精准裁剪：导出视频帧与行为区间完全一致。";
    }
  }

  function updateCollectTabCount() {
    const count = state.detail?.collections?.length || 0;
    $("tab-clip-collect-count").textContent = String(count);
  }

  function renderCollectionCardHtml(item, { showJumpBtn = false } = {}) {
    const expId = item.export_id;
    const expandedVariant = state.expandedPreviews.get(expId) || null;
    const hasOverlay = Boolean(item.overlay_video_url);
    const idsStr = Array.isArray(item.mouse_ids) && item.mouse_ids.length
      ? item.mouse_ids.map((id) => `ID ${id}`).join("、")
      : "无";

    const rolesStr =
      Array.isArray(item.participants) && item.participants.length > 0
        ? item.participants
            .map((p) => `${ROLE_CN[p.role_name] || p.role_name}: ${(p.track_ids || []).join(",")}`)
            .join(" · ")
        : "";

    const verifOk = item.verification && (item.verification.ok === true || item.verification.verified === true || item.verification.frame_count > 0);
    const padStr = Number(item.padding_frames || 0) > 0
      ? `（含 ±${item.padding_frames} 帧填充 → [${item.clip_start_frame}, ${item.clip_end_frame})）`
      : "（0 帧精准裁剪）";

    let previewHtml = "";
    if (expandedVariant) {
      const rawUrl = item.raw_video_url || "";
      const ovUrl = item.overlay_video_url || "";
      const activeUrl = expandedVariant === "overlay" && ovUrl ? ovUrl : rawUrl;
      previewHtml = `
        <div class="preview-box" data-preview-export-id="${escHtml(expId)}">
          <div class="preview-tabs">
            <button type="button" class="preview-tab-btn${expandedVariant === "raw" ? " active" : ""}" data-preview-switch="${escHtml(expId)}" data-variant="raw">原始裁剪 clip.mp4</button>
            <button type="button" class="preview-tab-btn${expandedVariant === "overlay" ? " active" : ""}" data-preview-switch="${escHtml(expId)}" data-variant="overlay" ${hasOverlay ? "" : "disabled"}>目标高亮 clip_overlay.mp4</button>
          </div>
          ${
            activeUrl
              ? `<video class="preview-video" controls playsinline preload="metadata" data-video-export-id="${escHtml(expId)}" src="${escHtml(activeUrl)}"></video>`
              : '<p class="muted">暂无可预览视频地址</p>'
          }
        </div>
      `;
    }

    return `
      <article class="collection-card" data-export-id="${escHtml(expId)}">
        <div class="collection-card-top">
          <div>
            <div class="collection-card-title">${escHtml(behaviorCn(item.behavior))} · <span class="mono">${escHtml(item.behavior)}</span></div>
            <div class="collection-card-meta mono">${escHtml(expId)}</div>
          </div>
          <span class="status-badge ${verifOk ? "pass" : "collected"}">${verifOk ? "已校验 H.264" : "已导出"}</span>
        </div>
        <div class="collection-card-meta">
          <div class="mono">区间：[${item.start_frame}, ${item.end_frame}) · ${item.frame_count} 帧 (${formatDuration(item.frame_count, item.fps || 30)}) ${escHtml(padStr)}</div>
          <div>目标鼠：${escHtml(idsStr)}${rolesStr ? ` · ${escHtml(rolesStr)}` : ""}</div>
          ${item.note ? `<div>备注：${escHtml(item.note)}</div>` : ""}
          ${item.output_dir ? `<div class="mono" title="${escHtml(item.output_dir)}">目录：${escHtml(item.output_dir)}</div>` : ""}
        </div>
        <div class="collection-card-actions">
          <div class="btn-row">
            <button type="button" class="btn btn-quiet btn-xs" data-toggle-preview="${escHtml(expId)}">
              ${expandedVariant ? "收起预览" : "页内预览视频"}
            </button>
            ${
              showJumpBtn && item.clip_uid
                ? `<button type="button" class="btn btn-quiet btn-xs" data-jump-clip="${escHtml(item.clip_uid)}">定位原片段</button>`
                : ""
            }
          </div>
          <button type="button" class="btn btn-danger-outline btn-xs" data-delete-export="${escHtml(expId)}">删除</button>
        </div>
        ${previewHtml}
      </article>
    `;
  }

  function renderClipCollections() {
    const container = $("clip-collections-list");
    const list = state.detail?.collections || [];
    updateCollectTabCount();

    if (!list.length) {
      container.innerHTML = '<p class="empty-hint">当前片段尚未手动采集导出。</p>';
      return;
    }
    container.innerHTML = list.map((item) => renderCollectionCardHtml(item, { showJumpBtn: false })).join("");
  }

  function detachPreviewVideos(exportId = null) {
    for (const v of document.querySelectorAll("video.preview-video")) {
      if (exportId !== null && v.dataset.videoExportId !== exportId) continue;
      v.pause();
      v.removeAttribute("src");
      v.load();
    }
  }

  async function submitManualCollect() {
    if (!state.detail || !state.activeUid || state.collectDraft.submitting) return;
    const warnBox = $("collect-warning-box");
    const hint = $("collect-result-hint");
    warnBox.hidden = true;

    if (state.offlineMode || isDemoUid(state.activeUid)) {
      warnBox.textContent = "当前处于离线演示模式，无法调用后端 FFmpeg 导出视频包。请点击顶部「重新连接后端 API」后再执行采集。";
      warnBox.hidden = false;
      return;
    }

    const behavior = getEffectiveCollectBehavior();
    if (!behavior) {
      warnBox.textContent = "请选择或输入有效的行为类别名称。";
      warnBox.hidden = false;
      return;
    }

    if (state.collectDraft.mouseIds.length === 0) {
      warnBox.textContent = "请至少选择 1 只目标鼠 ID（可点击上方芯片或直接点击左侧视频画面中的小鼠框）。";
      warnBox.hidden = false;
      return;
    }

    const total = state.detail.video.frame_count;
    if (!(state.cropStart >= 0 && state.cropEnd > state.cropStart && state.cropEnd <= total)) {
      warnBox.textContent = `帧区间 [${state.cropStart}, ${state.cropEnd}) 超出片段有效范围 [0, ${total}]。`;
      warnBox.hidden = false;
      return;
    }

    const payload = {
      clip_uid: state.activeUid,
      start_frame: state.cropStart,
      end_frame: state.cropEnd,
      behavior,
      mouse_ids: [...state.collectDraft.mouseIds],
      participants: buildParticipantsPayload(),
      padding_frames: Number($("collect-padding-select").value) || 0,
      include_overlay: $("collect-include-overlay").checked,
      confidence: "certain",
      note: $("collect-note").value.trim(),
      overwrite: $("collect-overwrite").checked,
    };

    state.collectDraft.submitting = true;
    const btn = $("submit-collect-btn");
    btn.disabled = true;
    $("submit-collect-label").textContent = "正在编码与校验导出包…";
    hint.textContent = "正在调用 H.264 编码器裁剪并写入四文件行为包，请稍候…";
    hint.className = "sync-status muted";

    try {
      const res = await fetchJson("/api/collect", {
        method: "POST",
        body: payload,
        timeoutMs: 120000,
        validate: validateCollectResponse,
      });

      const created = res.collection;
      // Update current clip collections (replace if overwritten)
      state.detail.collections = [
        created,
        ...(state.detail.collections || []).filter((c) => c.export_id !== created.export_id),
      ];
      state.allCollections = [
        created,
        ...state.allCollections.filter((c) => c.export_id !== created.export_id),
      ];

      // Auto-expand preview on newly exported package for instant verification
      state.expandedPreviews.set(
        created.export_id,
        created.overlay_video_url ? "overlay" : "raw"
      );

      syncCatalogCollectionCounts();
      renderClipCollections();
      renderAllCollectionsDrawer();
      renderBehaviorNav();
      renderClipList();
      updateHeaderStats();

      hint.textContent = `导出成功：${created.export_id}（${created.frame_count} 帧）`;
      hint.className = "sync-status ok";
      showToast(`已导出片段包：${created.export_id}`, "ok", 3500);
    } catch (error) {
      if (error.isAlreadyExists) {
        warnBox.textContent = "已存在同名导出包：如需重新生成并覆盖，请勾选上方「覆盖已有同名导出包 (overwrite)」后再点击采集。";
        warnBox.hidden = false;
      } else if (error.isCorrupt) {
        showCorruptBanner(
          `采集清单写入异常（HTTP ${error.status}）`,
          `${error.message}。后端已保留损坏的清单文件未覆盖，请先检查输出目录。`
        );
      }
      hint.textContent = `导出失败：${error.message}`;
      hint.className = "sync-status error";
      showToast(`采集导出失败：${error.message}`, "error", 5000);
    } finally {
      state.collectDraft.submitting = false;
      btn.disabled = false;
      $("submit-collect-label").textContent = "采集并导出片段";
    }
  }

  async function deleteCollectionExport(exportId) {
    if (!exportId) return;
    if (state.offlineMode) {
      showToast("离线模式下无法删除后端导出包。", "warn");
      return;
    }
    if (!window.confirm(`确定删除已采集包「${exportId}」吗？此操作将删除输出目录中该导出子文件夹，源目录 F:\\mouse 不会受任何影响。`)) {
      return;
    }

    // Release browser video handles before requesting backend file deletion (prevents Windows lock errors)
    detachPreviewVideos(exportId);
    state.expandedPreviews.delete(exportId);

    try {
      await fetchJson(`/api/collections/${encodeURIComponent(exportId)}`, {
        method: "DELETE",
        timeoutMs: 15000,
        validate: validateDeleteResponse,
      });

      state.allCollections = state.allCollections.filter((c) => c.export_id !== exportId);
      if (state.detail && Array.isArray(state.detail.collections)) {
        state.detail.collections = state.detail.collections.filter((c) => c.export_id !== exportId);
      }

      syncCatalogCollectionCounts();
      renderClipCollections();
      renderAllCollectionsDrawer();
      renderBehaviorNav();
      renderClipList();
      updateHeaderStats();
      showToast(`已删除采集包：${exportId}`, "ok");
    } catch (error) {
      if (error.isCorrupt) {
        showCorruptBanner(
          `删除采集记录失败（HTTP ${error.status}）`,
          `${error.message}。后端已保留损坏的清单文件未覆盖。`
        );
      }
      showToast(`删除失败：${error.message}`, "error", 4500);
    }
  }

  /* =========================================================================
   * Global Collections Drawer & Batch Export (needs_recrop clips)
   * ========================================================================= */
  function getBatchEligibleItems() {
    const pad = Number($("batch-padding-select")?.value) || 0;
    const includeOverlay = $("batch-include-overlay")?.checked !== false;
    const items = [];

    for (const clip of state.clips) {
      if (isDemoUid(clip.clip_uid)) continue;
      const rev = getEffectiveReview(clip.clip_uid);
      if (!rev || rev.status !== "needs_recrop") continue;
      const cr = rev.crop_range;
      if (
        !Array.isArray(cr) ||
        cr.length !== 2 ||
        !Number.isInteger(cr[0]) ||
        !Number.isInteger(cr[1]) ||
        cr[0] < 0 ||
        cr[1] <= cr[0] ||
        cr[1] > clip.frame_count
      ) {
        continue;
      }
      items.push({
        clip_uid: clip.clip_uid,
        start_frame: cr[0],
        end_frame: cr[1],
        behavior: clip.behavior,
        mouse_ids: Array.isArray(clip.target_mouse_ids) ? clip.target_mouse_ids : [],
        participants: [],
        padding_frames: pad,
        include_overlay: includeOverlay,
        confidence: "certain",
        note: rev.note || "批量重裁导出",
        overwrite: false,
      });
    }
    return items;
  }

  function updateBatchSummary() {
    const items = getBatchEligibleItems();
    const count = items.length;
    $("batch-eligible-summary").textContent =
      count > 0
        ? `检测到 ${count} 条标记为「需重裁」且已保存裁剪区间的片段，可一键批量导出。`
        : "当前暂无同时满足「需重裁」且已保存裁剪区间的片段。";
    const btn = $("run-batch-collect-btn");
    btn.textContent = `批量导出 (${count})`;
    btn.disabled = count === 0 || state.offlineMode || Boolean(state.batchJobId);
  }

  async function runBatchCollect() {
    const items = getBatchEligibleItems();
    if (!items.length || state.offlineMode || state.batchJobId) return;

    const btn = $("run-batch-collect-btn");
    btn.disabled = true;

    const progBox = $("batch-progress-box");
    const progStatus = $("batch-progress-status");
    const progCounts = $("batch-progress-counts");
    const progBar = $("batch-progress-bar");
    const errList = $("batch-errors-list");

    progBox.hidden = false;
    errList.hidden = true;
    errList.innerHTML = "";
    progStatus.textContent = "正在提交批量导出任务…";
    progCounts.textContent = `0 / ${items.length}`;
    progBar.style.width = "0%";

    try {
      const accepted = await fetchJson("/api/collect/batch", {
        method: "POST",
        body: { items },
        timeoutMs: 20000,
        validate: validateBatchAccept,
      });

      state.batchJobId = accepted.job_id;
      progStatus.textContent = `任务 ${accepted.job_id} 执行中…`;
      pollBatchJob(accepted.job_id);
    } catch (error) {
      progStatus.textContent = `批量提交失败：${error.message}`;
      showToast(`批量导出启动失败：${error.message}`, "error", 4500);
      updateBatchSummary();
    }
  }

  async function pollBatchJob(jobId) {
    if (state.batchPollTimer) {
      window.clearTimeout(state.batchPollTimer);
      state.batchPollTimer = null;
    }
    try {
      const job = await fetchJson(`/api/collect/jobs/${encodeURIComponent(jobId)}`, {
        timeoutMs: 12000,
        validate: validateBatchJob,
      });

      const total = Math.max(1, Number(job.total || 0));
      const completed = Number(job.completed || 0);
      const failed = Number(job.failed || 0);
      const doneCount = completed + failed;
      const pct = clamp(Math.round((doneCount / total) * 100), 0, 100);

      $("batch-progress-bar").style.width = `${pct}%`;
      $("batch-progress-counts").textContent = `完成 ${completed} · 失败 ${failed} / 共 ${total}`;

      const errs = Array.isArray(job.errors) ? job.errors : [];
      if (errs.length > 0) {
        $("batch-errors-list").hidden = false;
        $("batch-errors-list").innerHTML = errs
          .map((e) => {
            const msg = typeof e === "string" ? e : `${e.clip_uid || ""}: ${e.error || JSON.stringify(e)}`;
            return `<li>${escHtml(msg)}</li>`;
          })
          .join("");
      }

      const stLower = String(job.status || "").toLowerCase();
      const isRunning = ["pending", "queued", "running", "in_progress"].includes(stLower);

      if (isRunning && doneCount < total) {
        $("batch-progress-status").textContent = `批量导出执行中（${pct}%）…`;
        state.batchPollTimer = window.setTimeout(() => pollBatchJob(jobId), 800);
      } else {
        state.batchJobId = null;
        if (failed > 0) {
          $("batch-progress-status").textContent = `批量导出结束（存在部分失败：成功 ${completed}，失败 ${failed}，请勿视为完整导出）`;
          showToast(`批量导出完成：成功 ${completed}，失败 ${failed}`, "warn", 5000);
        } else {
          $("batch-progress-status").textContent = `批量导出全部完成（共 ${completed} 个片段包）`;
          showToast(`批量导出全部完成：共 ${completed} 个片段包。`, "ok", 3600);
        }
        await refreshCollectionsFromBackend();
        updateBatchSummary();
      }
    } catch (error) {
      state.batchJobId = null;
      $("batch-progress-status").textContent = `查询批量进度失败：${error.message}`;
      showToast(`批量进度查询失败：${error.message}`, "error");
      updateBatchSummary();
    }
  }

  async function refreshCollectionsFromBackend() {
    if (state.offlineMode) return;
    try {
      const res = await fetchJson("/api/collections", {
        timeoutMs: 15000,
        validate: validateCollections,
      });
      state.allCollections = res.collections || [];
      state.collectionsOutputDir = res.output_dir || state.collectionsOutputDir;
      $("drawer-output-dir").textContent = `输出目录：${state.collectionsOutputDir || "—"}`;
      syncCatalogCollectionCounts();

      if (state.detail && state.activeUid) {
        state.detail.collections = state.allCollections.filter((c) => c.clip_uid === state.activeUid);
        renderClipCollections();
      }
      renderAllCollectionsDrawer();
      renderBehaviorNav();
      renderClipList();
      updateHeaderStats();
    } catch (error) {
      showToast(`刷新采集清单失败：${error.message}`, "error");
    }
  }

  function renderAllCollectionsDrawer() {
    const query = ($("drawer-search")?.value || "").trim().toLowerCase();
    const filtered = state.allCollections.filter((item) => {
      if (!query) return true;
      const hay = `${item.export_id || ""} ${item.behavior || ""} ${behaviorCn(item.behavior)} ${item.clip_uid || ""} ${item.note || ""}`.toLowerCase();
      return hay.includes(query);
    });

    $("drawer-total-count").textContent = `共 ${filtered.length} 条`;
    const body = $("all-collections-list");
    if (!filtered.length) {
      body.innerHTML = `<p class="empty-hint">${state.allCollections.length ? "没有匹配的已采集片段。" : "暂无手动采集导出的片段。"}</p>`;
      return;
    }

    body.innerHTML = filtered
      .map((item) => renderCollectionCardHtml(item, { showJumpBtn: true }))
      .join("");
  }

  function openDrawer() {
    $("collections-drawer-backdrop").hidden = false;
    $("collections-drawer").hidden = false;
    updateBatchSummary();
    renderAllCollectionsDrawer();
    if (!state.offlineMode) {
      refreshCollectionsFromBackend();
    }
  }

  function closeDrawer() {
    detachPreviewVideos();
    $("collections-drawer-backdrop").hidden = true;
    $("collections-drawer").hidden = true;
  }

  /* =========================================================================
   * Safe Verified File Download (JSON / CSV Exports)
   * ========================================================================= */
  async function downloadExportEndpoint(url, fallbackFilename, offlineExportType = null) {
    if (state.offlineMode) {
      if (offlineExportType === "reviews") {
        const localDump = {
          notice: "离线本地草稿导出 · 非后端持久化数据",
          exported_at: new Date().toISOString(),
          pending_reviews: state.pendingReviews,
        };
        const blob = new Blob([JSON.stringify(localDump, null, 2)], { type: "application/json;charset=utf-8" });
        triggerBlobDownload(blob, "offline_pending_reviews_draft.json");
        showToast("已导出本地浏览器暂存审核草稿（非后端数据）。", "warn");
        return;
      }
      showToast("离线模式下无法导出后端文件，请先重新连接后端 API。", "warn");
      return;
    }

    try {
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 20000);
      const response = await fetch(url, { cache: "no-store", signal: controller.signal });
      window.clearTimeout(timer);

      if (!response.ok) {
        const ct = response.headers.get("content-type") || "";
        let msg = `HTTP ${response.status}`;
        if (/\bapplication\/json\b/i.test(ct)) {
          try {
            const errJson = await response.json();
            if (errJson?.error) msg += `：${errJson.error}`;
          } catch {
            // ignore
          }
        }
        throw new Error(msg);
      }

      const ct = response.headers.get("content-type") || "";
      if (/\btext\/html\b/i.test(ct)) {
        throw new Error("导出接口返回了 HTML 页面而非下载文件。");
      }

      const blob = await response.blob();
      const dispo = response.headers.get("content-disposition") || "";
      const match = dispo.match(/filename\*?=(?:UTF-8''|")?([^";\n]+)/i);
      const filename = match ? decodeURIComponent(match[1].replace(/"/g, "").trim()) : fallbackFilename;
      triggerBlobDownload(blob, filename);
      showToast(`已开始下载：${filename}`, "ok", 2200);
    } catch (error) {
      showToast(`导出失败：${error.message}`, "error", 4200);
    }
  }

  function triggerBlobDownload(blob, filename) {
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(objUrl), 4000);
  }

  /* =========================================================================
   * Tab Switching & Modals
   * ========================================================================= */
  function switchRightTab(tabName) {
    state.activeTab = tabName;
    const isReview = tabName === "review";
    $("tab-btn-review").classList.toggle("active", isReview);
    $("tab-btn-review").setAttribute("aria-selected", String(isReview));
    $("tab-panel-review").hidden = !isReview;
    $("tab-panel-review").classList.toggle("active", isReview);

    $("tab-btn-collect").classList.toggle("active", !isReview);
    $("tab-btn-collect").setAttribute("aria-selected", String(!isReview));
    $("tab-panel-collect").hidden = isReview;
    $("tab-panel-collect").classList.toggle("active", !isReview);
  }

  function openShortcutsModal() {
    $("shortcuts-modal-backdrop").hidden = false;
  }

  function closeShortcutsModal() {
    $("shortcuts-modal-backdrop").hidden = true;
  }

  /* =========================================================================
   * Event Delegation & Keyboard Shortcuts
   * ========================================================================= */
  function bindEvents() {
    // Reconnect & Banner Buttons
    $("header-reconnect-btn").addEventListener("click", () => connectAndLoad());
    $("banner-reconnect-btn").addEventListener("click", () => connectAndLoad());
    $("catalog-retry-btn").addEventListener("click", () => connectAndLoad());
    $("banner-flush-btn").addEventListener("click", () => flushPendingReviews());
    $("corrupt-retry-btn").addEventListener("click", () => flushPendingReviews());
    $("corrupt-dismiss-btn").addEventListener("click", () => hideCorruptBanner());
    $("retry-detail-btn").addEventListener("click", () => {
      if (state.activeUid) selectClip(state.activeUid);
    });

    // Drawer & Modal Controls
    $("open-collections-btn").addEventListener("click", openDrawer);
    $("open-drawer-inline-btn").addEventListener("click", openDrawer);
    $("close-collections-btn").addEventListener("click", closeDrawer);
    $("collections-drawer-backdrop").addEventListener("click", closeDrawer);
    $("refresh-collections-btn").addEventListener("click", refreshCollectionsFromBackend);
    $("drawer-search").addEventListener("input", renderAllCollectionsDrawer);

    $("open-shortcuts-btn").addEventListener("click", openShortcutsModal);
    $("close-shortcuts-btn").addEventListener("click", closeShortcutsModal);
    $("shortcuts-modal-backdrop").addEventListener("click", (ev) => {
      if (ev.target === $("shortcuts-modal-backdrop")) closeShortcutsModal();
    });

    // Left Panel Behavior Navigation
    $("behavior-nav").addEventListener("click", (ev) => {
      const btn = ev.target.closest(".behavior-item");
      if (!btn) return;
      state.selectedBehavior = btn.dataset.behavior;
      renderBehaviorNav();
      applyFiltersAndRender({ preserveSelection: false });
    });

    // Search & Status Filter
    let searchTimer = null;
    $("clip-search").addEventListener("input", () => {
      window.clearTimeout(searchTimer);
      searchTimer = window.setTimeout(() => {
        state.searchQuery = $("clip-search").value;
        applyFiltersAndRender({ preserveSelection: true });
      }, 120);
    });
    $("status-filter").addEventListener("change", () => {
      state.statusFilter = $("status-filter").value;
      applyFiltersAndRender({ preserveSelection: true });
    });
    $("reset-filters-btn").addEventListener("click", () => {
      state.selectedBehavior = "__ALL__";
      state.statusFilter = "all";
      state.searchQuery = "";
      $("status-filter").value = "all";
      $("clip-search").value = "";
      renderBehaviorNav();
      applyFiltersAndRender({ preserveSelection: false });
    });

    // Paginated Clip List & Clip Selection
    $("clip-list").addEventListener("click", (ev) => {
      const card = ev.target.closest(".clip-card");
      if (!card) return;
      selectClip(card.dataset.clipUid);
    });
    $("page-prev-btn").addEventListener("click", () => {
      if (state.page > 1) {
        state.page -= 1;
        renderClipList();
      }
    });
    $("page-next-btn").addEventListener("click", () => {
      const totalPages = Math.max(1, Math.ceil(state.filteredClips.length / PAGE_SIZE));
      if (state.page < totalPages) {
        state.page += 1;
        renderClipList();
      }
    });

    // Prev / Next Clip
    $("clip-prev-btn").addEventListener("click", () => stepClip(-1));
    $("clip-next-btn").addEventListener("click", () => stepClip(1));

    // Overlay Toggle Switches
    for (const id of ["toggle-targets", "toggle-others", "toggle-skeleton", "toggle-labels"]) {
      $(id).addEventListener("change", drawCurrentFrame);
    }

    // Canvas Click-to-Pick & Hover
    canvasEl.addEventListener("pointermove", handleCanvasPointerMove);
    canvasEl.addEventListener("pointerleave", () => {
      canvasEl.classList.remove("pick-hover");
      $("canvas-hover-tip").hidden = true;
      if (state.hoveredTrackId !== null) {
        state.hoveredTrackId = null;
        drawCurrentFrame();
      }
    });
    canvasEl.addEventListener("click", handleCanvasClick);

    // Transport Controls
    $("play-pause-btn").addEventListener("click", togglePlayPause);
    $("step-back-10").addEventListener("click", () => stepFrames(-10));
    $("step-back-1").addEventListener("click", () => stepFrames(-1));
    $("step-fwd-1").addEventListener("click", () => stepFrames(1));
    $("step-fwd-10").addEventListener("click", () => stepFrames(10));
    $("playback-speed").addEventListener("change", () => {
      videoEl.playbackRate = Number($("playback-speed").value) || 1;
    });

    $("set-in-btn").addEventListener("click", () => {
      if (!state.detail) return;
      const total = state.detail.video.frame_count;
      const newStart = clamp(state.currentFrame, 0, total - 1);
      const newEnd = Math.max(state.cropEnd, newStart + 1);
      setCropRange(newStart, newEnd);
    });
    $("set-out-btn").addEventListener("click", () => {
      if (!state.detail) return;
      const total = state.detail.video.frame_count;
      const newEnd = clamp(state.currentFrame + 1, 1, total);
      const newStart = Math.min(state.cropStart, newEnd - 1);
      setCropRange(newStart, newEnd);
    });
    $("loop-preview-btn").addEventListener("click", () => {
      state.loopPreview = !state.loopPreview;
      $("loop-preview-btn").setAttribute("aria-pressed", String(state.loopPreview));
      if (state.loopPreview && state.detail && (state.currentFrame < state.cropStart || state.currentFrame >= state.cropEnd)) {
        seekToFrame(state.cropStart, { keepPlaying: isPlaying() });
      }
    });
    $("align-annotation-btn").addEventListener("click", () => {
      if (!state.detail) return;
      const [s, e] = getOriginalAnnotationHalfOpen(state.detail);
      setCropRange(s, e, { seekTo: s });
      showToast(`已对齐原标注区间：[${s}, ${e})`, "info", 1800);
    });
    $("fullscreen-btn").addEventListener("click", async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await $("video-shell").requestFullscreen();
      } catch {
        showToast("当前浏览器窗口未能进入全屏模式。", "warn");
      }
    });

    // Video Element Lifecycle Events
    videoEl.addEventListener("loadedmetadata", () => {
      if (!state.detail) return;
      videoEl.playbackRate = Number($("playback-speed").value) || 1;
      resizeOverlayCanvas();
      seekToFrame(state.currentFrame);
    });
    videoEl.addEventListener("play", () => {
      scheduleVideoFrameSync();
      updatePlayButtonLabel();
    });
    videoEl.addEventListener("pause", () => {
      if (state.rvfcId !== null && typeof videoEl.cancelVideoFrameCallback === "function") {
        videoEl.cancelVideoFrameCallback(state.rvfcId);
        state.rvfcId = null;
      }
      if (state.detail) syncFrameFromMediaTime(videoEl.currentTime);
      updatePlayButtonLabel();
    });
    videoEl.addEventListener("seeked", () => {
      if (state.detail) {
        syncFrameFromMediaTime(videoEl.currentTime);
        scheduleVideoFrameSync();
      }
    });
    videoEl.addEventListener("timeupdate", () => {
      if (state.detail && (typeof videoEl.requestVideoFrameCallback !== "function" || videoEl.paused)) {
        syncFrameFromMediaTime(videoEl.currentTime);
      }
    });
    videoEl.addEventListener("ended", () => {
      if (state.loopPreview && state.detail) {
        seekToFrame(state.cropStart, { keepPlaying: true });
        videoEl.play().catch(() => {});
      } else {
        updatePlayButtonLabel();
      }
    });
    videoEl.addEventListener("error", () => {
      if (state.detail && videoEl.getAttribute("src")) {
        state.videoError = true;
        $("video-error-overlay").hidden = false;
        $("video-error-text").textContent = "视频流加载失败，已保留实测轨迹画布供帧级核查，可尝试重新连接或重试此片段。";
        drawCurrentFrame();
      }
    });

    // Dual-Handle Timeline Pointer Interaction
    const rail = $("timeline-rail");
    const hStart = $("handle-start");
    const hEnd = $("handle-end");

    const startDrag = (mode, targetEl, ev) => {
      if (!state.detail) return;
      ev.preventDefault();
      ev.stopPropagation();
      pausePlayback();
      state.draggingMode = mode;
      targetEl.classList.add("dragging");
      if (typeof targetEl.setPointerCapture === "function") {
        try {
          targetEl.setPointerCapture(ev.pointerId);
        } catch {
          // ignore
        }
      }
      if (mode === "scrub") {
        seekToFrame(pointerEventToFrame(ev, false));
      }
    };

    const moveDrag = (ev) => {
      if (!state.draggingMode || !state.detail) return;
      const total = state.detail.video.frame_count;
      if (state.draggingMode === "start") {
        const f = clamp(pointerEventToFrame(ev, false), 0, Math.max(0, state.cropEnd - 1));
        setCropRange(f, state.cropEnd, { seekTo: f });
      } else if (state.draggingMode === "end") {
        const f = clamp(pointerEventToFrame(ev, true), state.cropStart + 1, total);
        setCropRange(state.cropStart, f, { seekTo: f - 1 });
      } else if (state.draggingMode === "scrub") {
        seekToFrame(pointerEventToFrame(ev, false));
      }
    };

    const endDrag = () => {
      state.draggingMode = null;
      hStart.classList.remove("dragging");
      hEnd.classList.remove("dragging");
    };

    hStart.addEventListener("pointerdown", (ev) => startDrag("start", hStart, ev));
    hEnd.addEventListener("pointerdown", (ev) => startDrag("end", hEnd, ev));
    rail.addEventListener("pointerdown", (ev) => {
      if (ev.target === hStart || ev.target === hEnd) return;
      startDrag("scrub", rail, ev);
    });
    window.addEventListener("pointermove", moveDrag);
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);

    // Keyboard Accessibility for Slider Handles
    hStart.addEventListener("keydown", (ev) => {
      if (!state.detail) return;
      const step = ev.shiftKey ? 10 : 1;
      if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        setCropRange(state.cropStart - step, state.cropEnd, { seekTo: Math.max(0, state.cropStart - step) });
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        const next = Math.min(state.cropEnd - 1, state.cropStart + step);
        setCropRange(next, state.cropEnd, { seekTo: next });
      }
    });
    hEnd.addEventListener("keydown", (ev) => {
      if (!state.detail) return;
      const step = ev.shiftKey ? 10 : 1;
      const total = state.detail.video.frame_count;
      if (ev.key === "ArrowLeft") {
        ev.preventDefault();
        const next = Math.max(state.cropStart + 1, state.cropEnd - step);
        setCropRange(state.cropStart, next, { seekTo: next - 1 });
      } else if (ev.key === "ArrowRight") {
        ev.preventDefault();
        const next = Math.min(total, state.cropEnd + step);
        setCropRange(state.cropStart, next, { seekTo: next - 1 });
      }
    });

    // ResizeObserver for crisp Canvas & Visibility Strip alignment
    if (typeof ResizeObserver === "function") {
      const ro = new ResizeObserver(() => {
        resizeOverlayCanvas();
        drawCurrentFrame();
        renderVisibilityStrip();
      });
      ro.observe($("video-stage"));
      ro.observe(visCanvasEl.parentElement);
    } else {
      window.addEventListener("resize", () => {
        resizeOverlayCanvas();
        drawCurrentFrame();
        renderVisibilityStrip();
      });
    }

    // Right Panel Tabs
    $("tab-btn-review").addEventListener("click", () => switchRightTab("review"));
    $("tab-btn-collect").addEventListener("click", () => switchRightTab("collect"));

    // Tab 1: Review Events
    document.querySelector(".verdict-grid").addEventListener("click", (ev) => {
      const btn = ev.target.closest(".verdict-btn");
      if (!btn) return;
      selectReviewVerdict(btn.dataset.verdict);
    });
    $("reason-tags-group").addEventListener("click", (ev) => {
      const chip = ev.target.closest(".tag-chip");
      if (!chip) return;
      const tag = chip.dataset.tag;
      if (state.reviewDraft.tags.has(tag)) state.reviewDraft.tags.delete(tag);
      else state.reviewDraft.tags.add(tag);
      syncReviewFormUI(state.activeUid);
    });
    $("review-attach-crop").addEventListener("change", () => {
      state.reviewDraft.attachCrop = $("review-attach-crop").checked;
    });
    $("review-note").addEventListener("input", () => {
      state.reviewDraft.note = $("review-note").value;
    });
    $("save-review-btn").addEventListener("click", saveCurrentReview);
    $("export-reviews-json-btn").addEventListener("click", () =>
      downloadExportEndpoint("/api/reviews/export?format=json", "mouse_clip_reviews.json", "reviews")
    );
    $("export-reviews-csv-btn").addEventListener("click", () =>
      downloadExportEndpoint("/api/reviews/export?format=csv", "mouse_clip_reviews.csv", "reviews")
    );

    // Tab 2: Manual Collect Events
    $("collect-behavior-select").addEventListener("change", () => {
      const val = $("collect-behavior-select").value;
      state.collectDraft.behavior = val;
      $("collect-behavior-custom").hidden = val !== "__CUSTOM__";
      if (val === "__CUSTOM__") $("collect-behavior-custom").focus();
      state.collectDraft.roleAssignments = {};
      renderDirectionalRolesUI();
      drawCurrentFrame();
    });
    $("collect-behavior-custom").addEventListener("input", () => {
      state.collectDraft.customBehavior = $("collect-behavior-custom").value;
    });

    $("collect-mouse-chips").addEventListener("click", (ev) => {
      const chip = ev.target.closest(".mouse-chip");
      if (!chip) return;
      const tid = Number.parseInt(chip.dataset.trackId, 10);
      toggleCollectMouseId(tid);
    });
    $("sync-orig-targets-btn").addEventListener("click", () => {
      if (!state.detail) return;
      const orig = Array.isArray(state.detail.annotation?.mouse_ids)
        ? state.detail.annotation.mouse_ids
        : state.detail.clip.target_mouse_ids || [];
      state.collectDraft.mouseIds = [...new Set(orig.filter(Number.isInteger))].sort((a, b) => a - b);
      state.collectDraft.roleAssignments = {};
      if (Array.isArray(state.detail.annotation?.participants)) {
        for (const p of state.detail.annotation.participants) {
          if (p && p.role_name && Array.isArray(p.track_ids) && p.track_ids.length) {
            state.collectDraft.roleAssignments[p.role_name] = p.track_ids[0];
          }
        }
      }
      renderCollectMouseChips();
      renderDirectionalRolesUI();
      renderVisibilityStrip();
      renderMetadataSection();
      drawCurrentFrame();
    });
    $("clear-targets-btn").addEventListener("click", () => {
      state.collectDraft.mouseIds = [];
      state.collectDraft.roleAssignments = {};
      renderCollectMouseChips();
      renderDirectionalRolesUI();
      renderVisibilityStrip();
      renderMetadataSection();
      drawCurrentFrame();
    });

    $("participants-fields").addEventListener("change", (ev) => {
      const sel = ev.target.closest("select[data-role-name]");
      if (!sel) return;
      const roleName = sel.dataset.roleName;
      const val = sel.value === "" ? null : Number.parseInt(sel.value, 10);
      state.collectDraft.roleAssignments[roleName] = Number.isInteger(val) ? val : null;
      drawCurrentFrame();
      renderMetadataSection();
    });
    $("swap-roles-btn").addEventListener("click", () => {
      const roles = DIRECTIONAL_ROLES[getEffectiveCollectBehavior()];
      if (!roles || roles.length < 2) return;
      const a = state.collectDraft.roleAssignments[roles[0].role_name] ?? null;
      const b = state.collectDraft.roleAssignments[roles[1].role_name] ?? null;
      state.collectDraft.roleAssignments[roles[0].role_name] = b;
      state.collectDraft.roleAssignments[roles[1].role_name] = a;
      renderDirectionalRolesUI();
      drawCurrentFrame();
      renderMetadataSection();
    });

    $("collect-start-input").addEventListener("change", () => {
      if (!state.detail) return;
      const v = Number.parseInt($("collect-start-input").value, 10);
      if (Number.isInteger(v)) {
        const total = state.detail.video.frame_count;
        const s = clamp(v, 0, total - 1);
        const e = Math.max(state.cropEnd, s + 1);
        setCropRange(s, e, { seekTo: s });
      } else {
        syncTimelineUI();
      }
    });
    $("collect-end-input").addEventListener("change", () => {
      if (!state.detail) return;
      const v = Number.parseInt($("collect-end-input").value, 10);
      if (Number.isInteger(v)) {
        const total = state.detail.video.frame_count;
        const e = clamp(v, 1, total);
        const s = Math.min(state.cropStart, e - 1);
        setCropRange(s, e, { seekTo: e - 1 });
      } else {
        syncTimelineUI();
      }
    });
    $("collect-padding-select").addEventListener("change", updateCollectDurationReadout);
    $("submit-collect-btn").addEventListener("click", submitManualCollect);

    // Exported Package Card Actions (shared delegation for Right Panel & Drawer)
    const handleCollectionCardClick = (ev) => {
      const previewToggle = ev.target.closest("[data-toggle-preview]");
      if (previewToggle) {
        const expId = previewToggle.dataset.togglePreview;
        if (state.expandedPreviews.has(expId)) {
          detachPreviewVideos(expId);
          state.expandedPreviews.delete(expId);
        } else {
          const item =
            state.allCollections.find((c) => c.export_id === expId) ||
            (state.detail?.collections || []).find((c) => c.export_id === expId);
          state.expandedPreviews.set(expId, item?.overlay_video_url ? "overlay" : "raw");
        }
        renderClipCollections();
        renderAllCollectionsDrawer();
        return;
      }

      const previewSwitch = ev.target.closest("[data-preview-switch]");
      if (previewSwitch) {
        const expId = previewSwitch.dataset.previewSwitch;
        const variant = previewSwitch.dataset.variant;
        detachPreviewVideos(expId);
        state.expandedPreviews.set(expId, variant);
        renderClipCollections();
        renderAllCollectionsDrawer();
        return;
      }

      const delBtn = ev.target.closest("[data-delete-export]");
      if (delBtn) {
        deleteCollectionExport(delBtn.dataset.deleteExport);
        return;
      }

      const jumpBtn = ev.target.closest("[data-jump-clip]");
      if (jumpBtn) {
        const targetUid = jumpBtn.dataset.jumpClip;
        closeDrawer();
        if (state.clipsByUid.has(targetUid)) {
          // Ensure target clip is visible in filters if currently filtered out
          if (!state.filteredClips.some((c) => c.clip_uid === targetUid)) {
            state.selectedBehavior = "__ALL__";
            state.statusFilter = "all";
            state.searchQuery = "";
            $("status-filter").value = "all";
            $("clip-search").value = "";
            renderBehaviorNav();
            applyFiltersAndRender({ preserveSelection: false });
          }
          selectClip(targetUid);
        } else {
          showToast("当前目录中未找到该原始片段。", "warn");
        }
      }
    };

    $("clip-collections-list").addEventListener("click", handleCollectionCardClick);
    $("all-collections-list").addEventListener("click", handleCollectionCardClick);

    // Drawer Export & Batch Export
    $("export-collections-json-btn").addEventListener("click", () =>
      downloadExportEndpoint("/api/collections/export?format=json", "mouse_clip_collections.json")
    );
    $("export-collections-csv-btn").addEventListener("click", () =>
      downloadExportEndpoint("/api/collections/export?format=csv", "mouse_clip_collections.csv")
    );
    $("run-batch-collect-btn").addEventListener("click", runBatchCollect);
    $("batch-padding-select").addEventListener("change", updateBatchSummary);

    // Global Keyboard Shortcuts
    window.addEventListener("keydown", (ev) => {
      const target = ev.target;
      const inEditable =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);

      if (ev.key === "Escape") {
        if (!$("shortcuts-modal-backdrop").hidden) {
          closeShortcutsModal();
          return;
        }
        if (!$("collections-drawer").hidden) {
          closeDrawer();
          return;
        }
      }

      // Allow Ctrl+S / Cmd+S to save review even from textarea
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
        ev.preventDefault();
        saveCurrentReview();
        return;
      }

      if (inEditable || ev.ctrlKey || ev.metaKey || ev.altKey) return;

      // Do not trigger global playback shortcuts while a modal/drawer is open
      if (!$("shortcuts-modal-backdrop").hidden || !$("collections-drawer").hidden) {
        if (ev.key === "?") {
          ev.preventDefault();
          closeShortcutsModal();
        }
        return;
      }

      const key = ev.key;
      if (key === " " || ev.code === "Space") {
        ev.preventDefault();
        togglePlayPause();
      } else if (key === "ArrowLeft" || key === ",") {
        ev.preventDefault();
        stepFrames(ev.shiftKey ? -10 : -1);
      } else if (key === "ArrowRight" || key === ".") {
        ev.preventDefault();
        stepFrames(ev.shiftKey ? 10 : 1);
      } else if (key === "[") {
        ev.preventDefault();
        if (state.detail) {
          const s = clamp(state.currentFrame, 0, state.detail.video.frame_count - 1);
          setCropRange(s, Math.max(state.cropEnd, s + 1));
        }
      } else if (key === "]") {
        ev.preventDefault();
        if (state.detail) {
          const e = clamp(state.currentFrame + 1, 1, state.detail.video.frame_count);
          setCropRange(Math.min(state.cropStart, e - 1), e);
        }
      } else if (key.toLowerCase() === "l") {
        ev.preventDefault();
        $("loop-preview-btn").click();
      } else if (key.toLowerCase() === "j") {
        ev.preventDefault();
        stepClip(-1);
      } else if (key.toLowerCase() === "k") {
        ev.preventDefault();
        stepClip(1);
      } else if (key === "1") {
        ev.preventDefault();
        switchRightTab("review");
        selectReviewVerdict("pass");
      } else if (key === "2") {
        ev.preventDefault();
        switchRightTab("review");
        selectReviewVerdict("fail");
      } else if (key === "3") {
        ev.preventDefault();
        switchRightTab("review");
        selectReviewVerdict("needs_recrop");
      } else if (key.toLowerCase() === "s") {
        ev.preventDefault();
        saveCurrentReview();
      } else if (key === "?") {
        ev.preventDefault();
        openShortcutsModal();
      }
    });
  }

  /* =========================================================================
   * Immediate App Shell Boot (No API required to render full shell!)
   * ========================================================================= */
  function bootstrap() {
    state.pendingReviews = loadPendingReviewsFromStorage();
    populateCollectBehaviorOptions();
    bindEvents();
    updateHeaderStats();
    connectAndLoad({ silentToast: true });
  }

  bootstrap();
})();
