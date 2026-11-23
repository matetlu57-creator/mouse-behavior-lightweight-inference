"use strict";

const CATEGORY_LABELS = {
  review: "人工审核网站",
  error_display: "错误显示网站",
  comparison: "对比复核网站",
  other: "其他站点",
};

const STATUS_LABELS = {
  running: "运行中（健康）",
  external_running: "外部运行中（健康）",
  starting: "启动中…",
  restarting: "自动拉起中…",
  stopped: "已停止",
  disabled: "已禁用",
  unhealthy: "健康检查异常",
  error: "进程异常停止",
  offline: "离线参考（非实时）",
};

const OFFLINE_REFERENCE_SITES = [
  {
    site_id: "mouse_clip_review",
    title: "小鼠行为审核与手动采集工作台",
    category: "review",
    host: "127.0.0.1",
    port: 8795,
    url: "http://127.0.0.1:8795/",
    status: "offline",
    healthy: false,
    description: "逐帧核查已清洗数据集视频，支持通过/拒绝备注与手动行为片段裁剪采集。",
    data_summary: "已清洗数据集 F:\\mouse_behavior_cleaned_dataset_20261121（764 段 / 14 类）",
    offlineReference: true,
  },
  {
    site_id: "long_video_group_review",
    title: "长视频群体行为精裁人工审核站",
    category: "review",
    host: "127.0.0.1",
    port: 8790,
    url: "http://127.0.0.1:8790/",
    status: "offline",
    healthy: false,
    description: "多长视频群体行为边界精裁前后对比、原始/高亮回放与人工通过率复核。",
    data_summary: "video22 (10min) + 大旷场1 (60min) 群体行为裁剪复核",
    offlineReference: true,
  },
  {
    site_id: "behavior_error_review",
    title: "全 15 类行为评测错误复核站",
    category: "error_display",
    host: "127.0.0.1",
    port: 8776,
    url: "http://127.0.0.1:8776/",
    status: "offline",
    healthy: false,
    description: "全 15 类专项评测错误原因归因、指标面板与目标 ID 逐帧视频高亮展示。",
    data_summary: "A(429) + B(605) 全 15 类评测错误原因与逐帧高亮",
    offlineReference: true,
  },
];

const state = {
  connected: false,
  corruptRegistry: false,
  sessionToken: "",
  category: "all",
  search: "",
  sites: [],
  summary: null,
  hub: null,
  busyIds: new Set(),
  activeLogSiteId: null,
  refreshTimer: null,
};

function escHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m ${sec}s`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

async function fetchJson(url, options = {}) {
  const timeoutMs = typeof options.timeoutMs === "number" ? options.timeoutMs : 6000;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: options.method || "GET",
      headers: options.headers || { Accept: "application/json" },
      body: options.body,
      cache: "no-store",
      signal: controller.signal,
    });
    const ctype = (res.headers.get("content-type") || "").toLowerCase();
    if (!ctype.includes("application/json")) {
      throw new Error(`接口 ${url} 未返回 JSON（收到 ${ctype || "未知类型"}），拒绝误解析 HTML`);
    }
    const data = await res.json();
    if (!res.ok) {
      const err = new Error(data && data.error ? String(data.error) : `HTTP ${res.status}`);
      err.status = res.status;
      err.payload = data;
      throw err;
    }
    if (!data || typeof data !== "object") {
      throw new Error(`接口 ${url} 返回非对象 JSON`);
    }
    if (options.validate && !options.validate(data)) {
      throw new Error(`接口 ${url} 响应 schema 校验失败`);
    }
    return data;
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new Error(`请求 ${url} 超时（>${timeoutMs}ms）`);
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

function isStatusPayload(data) {
  return (
    data &&
    data.schema === "site_hub_status_v1" &&
    Array.isArray(data.sites) &&
    data.hub &&
    typeof data.hub === "object"
  );
}

function setBanner(tone, title, detail, showRetry) {
  const el = document.getElementById("connection-banner");
  el.className = `notice-banner notice-${tone}`;
  document.getElementById("banner-title").textContent = title;
  document.getElementById("banner-detail").textContent = detail;
  document.getElementById("retry-connection-btn").hidden = !showRetry;
}

function renderSummary() {
  const totalEl = document.getElementById("kpi-total");
  const healthyEl = document.getElementById("kpi-healthy");
  const reviewEl = document.getElementById("kpi-review");
  const errEl = document.getElementById("kpi-error-site");
  const autoEl = document.getElementById("kpi-autostart");
  const autoSubEl = document.getElementById("kpi-autostart-sub");
  const pillEl = document.getElementById("hub-status-pill");
  const metaEl = document.getElementById("hub-meta");

  if (!state.connected) {
    totalEl.textContent = String(OFFLINE_REFERENCE_SITES.length);
    healthyEl.textContent = "—";
    reviewEl.textContent = "2";
    errEl.textContent = "1";
    autoEl.textContent = "未探测";
    autoSubEl.textContent = "守护中心未连接";
    pillEl.className = "status-pill status-error";
    pillEl.textContent = state.corruptRegistry ? "注册表异常" : "守护中心离线";
    metaEl.textContent = "PID — · 请执行 manage_site_hub.py start";
    return;
  }

  const s = state.summary || {};
  totalEl.textContent = String(s.total ?? state.sites.length);
  healthyEl.textContent = String(s.healthy ?? 0);
  reviewEl.textContent = String(s.review_count ?? 0);
  errEl.textContent = String(s.error_display_count ?? 0);

  const autostart = state.hub && state.hub.autostart ? state.hub.autostart : {};
  if (autostart.installed) {
    autoEl.textContent = "已开启";
    autoSubEl.textContent = autostart.method === "schtasks" ? "Windows 计划任务（登录自启）" : "Windows 启动文件夹";
  } else {
    autoEl.textContent = "未开启";
    autoSubEl.textContent = "可执行 install-autostart 启用";
  }

  pillEl.className = "status-pill status-running";
  pillEl.textContent = "站点中心运行中";
  metaEl.textContent = `PID ${state.hub.pid ?? "—"} · 已运行 ${formatDuration(state.hub.uptime_s)}`;
}

function filterSites(items) {
  const q = state.search.trim().toLowerCase();
  return items.filter((site) => {
    if (state.category === "other") {
      if (site.category === "review" || site.category === "error_display") {
        return false;
      }
    } else if (state.category !== "all" && site.category !== state.category) {
      return false;
    }
    if (!q) return true;
    const hay = [
      site.site_id,
      site.title,
      site.description,
      site.data_summary,
      String(site.port),
      site.url,
    ]
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
}

function renderSites() {
  const grid = document.getElementById("site-grid");
  const empty = document.getElementById("site-empty");
  const fallbackBanner = document.getElementById("offline-fallback-header");

  const list = state.connected ? state.sites : OFFLINE_REFERENCE_SITES;
  fallbackBanner.hidden = state.connected;

  const filtered = filterSites(list);
  if (filtered.length === 0) {
    grid.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  grid.innerHTML = filtered
    .map((site) => {
      const status = site.status || "stopped";
      const statusLabel = STATUS_LABELS[status] || status;
      const catLabel = CATEGORY_LABELS[site.category] || site.category;
      const busy = state.busyIds.has(site.site_id);
      const isOnline = Boolean(site.healthy);
      const isRunningProc = status === "running" || status === "starting" || status === "restarting";
      const url = site.url || `http://${site.host}:${site.port}/`;
      const errorBox = site.last_error
        ? `<div class="site-error-note">${escHtml(site.last_error)}</div>`
        : "";
      const opsHtml = site.offlineReference
        ? `<span class="muted">离线参考卡片 · 启动守护中心后可直达并控制启停</span>`
        : `<div class="site-ops">
            <button type="button" class="btn btn-quiet" data-action="start" data-site="${escHtml(site.site_id)}" ${busy || isRunningProc ? "disabled" : ""}>启动</button>
            <button type="button" class="btn btn-quiet" data-action="restart" data-site="${escHtml(site.site_id)}" ${busy ? "disabled" : ""}>重启</button>
            <button type="button" class="btn btn-danger" data-action="stop" data-site="${escHtml(site.site_id)}" ${busy || (!site.managed && !isRunningProc) ? "disabled" : ""}>停止</button>
            <button type="button" class="btn btn-quiet" data-action="logs" data-site="${escHtml(site.site_id)}">日志</button>
          </div>`;

      return `
        <article class="site-card">
          <div class="site-card-top">
            <div>
              <div class="site-tags">
                <span class="cat-badge cat-${escHtml(site.category)}">${escHtml(catLabel)}</span>
                <span class="mono muted">${escHtml(site.site_id)}</span>
              </div>
              <h3 class="site-title">${escHtml(site.title)}</h3>
            </div>
            <span class="status-pill status-${escHtml(status)}">${escHtml(statusLabel)}</span>
          </div>

          <p class="site-desc">${escHtml(site.description || "无补充说明")}</p>

          <div class="site-meta-box">
            <div class="meta-row">
              <span class="meta-label">访问地址</span>
              <span class="mono">${escHtml(url)}</span>
            </div>
            <div class="meta-row">
              <span class="meta-label">数据范围</span>
              <span>${escHtml(site.data_summary || "—")}</span>
            </div>
            <div class="meta-row">
              <span class="meta-label">进程状态</span>
              <span class="mono">${
                site.offlineReference
                  ? "未探测"
                  : `PID ${escHtml(site.pid ?? "—")} · 重启 ${escHtml(site.restart_count ?? 0)} 次`
              }</span>
            </div>
          </div>

          ${errorBox}

          <div class="site-actions">
            ${opsHtml}
            <a class="btn ${isOnline ? "btn-primary" : "btn-quiet"}" href="${escHtml(url)}" target="_blank" rel="noopener noreferrer">
              打开网站 ↗
            </a>
          </div>
        </article>
      `;
    })
    .join("");
}

async function refreshAll() {
  try {
    const health = await fetchJson("/api/health");
    state.sessionToken = typeof health.session_token === "string" ? health.session_token : "";

    const status = await fetchJson("/api/sites", { validate: isStatusPayload });
    state.connected = true;
    state.corruptRegistry = false;
    state.hub = status.hub;
    state.summary = status.summary;
    state.sites = status.sites;

    const s = status.summary || {};
    const allOk = s.unhealthy === 0 && s.total > 0;
    if (allOk) {
      setBanner(
        "ok",
        `全部 ${s.total} 个站点均已健康在线`,
        `站点中心独立运行于 ${status.hub.url}（PID ${status.hub.pid}），会话网关关闭后站点仍持续服务。`,
        false
      );
    } else {
      setBanner(
        "warn",
        `已连接站点中心：${s.healthy}/${s.total} 个站点在线`,
        "部分站点仍在启动或处于停止/异常状态，可直接点击卡片上的“重启”或“日志”排查。",
        false
      );
    }
    document.getElementById("last-updated").textContent = `最后探测：${new Date().toLocaleTimeString()}`;
  } catch (err) {
    state.connected = false;
    const isCorrupt = Boolean(err && err.payload && err.payload.corrupt_registry);
    state.corruptRegistry = isCorrupt;
    if (isCorrupt) {
      setBanner(
        "danger",
        "站点注册表 sites.json 损坏（已保护原文件，未静默覆盖）",
        err.message || "请检查 outputs/site_hub/sites.json 格式。",
        true
      );
    } else {
      setBanner(
        "danger",
        "无法连接常驻站点中心，当前显示离线参考视图",
        `${err.message || "网络异常"}。请运行 D:\\Anaconda3\\python.exe scripts/manage_site_hub.py start 后点击重试。`,
        true
      );
    }
    document.getElementById("last-updated").textContent = `探测失败：${new Date().toLocaleTimeString()}`;
  }

  renderSummary();
  renderSites();
}

async function runSiteAction(siteId, action) {
  if (!state.sessionToken) {
    await refreshAll();
  }
  state.busyIds.add(siteId);
  renderSites();
  try {
    await fetchJson(`/api/sites/${encodeURIComponent(siteId)}/${action}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Hub-Token": state.sessionToken,
      },
      body: "{}",
      timeoutMs: 12000,
    });
  } catch (err) {
    setBanner("danger", `操作 ${siteId} (${action}) 失败`, err.message || "请求失败", true);
  } finally {
    state.busyIds.delete(siteId);
    await refreshAll();
  }
}

async function openLogs(siteId) {
  state.activeLogSiteId = siteId;
  const dialog = document.getElementById("logs-dialog");
  const titleEl = document.getElementById("logs-dialog-title");
  const pathEl = document.getElementById("logs-dialog-path");
  const preEl = document.getElementById("logs-content");
  const errEl = document.getElementById("logs-error");

  titleEl.textContent = `站点运行日志 · ${siteId}`;
  pathEl.textContent = "加载中…";
  preEl.textContent = "正在读取日志…";
  errEl.hidden = true;

  if (!dialog.open && typeof dialog.showModal === "function") {
    dialog.showModal();
  }

  try {
    const data = await fetchJson(`/api/sites/${encodeURIComponent(siteId)}/logs?lines=120`);
    pathEl.textContent = data.log_path || "";
    const lines = Array.isArray(data.lines) ? data.lines : [];
    preEl.textContent = lines.length > 0 ? lines.join("\n") : "（日志文件暂无输出）";
    preEl.scrollTop = preEl.scrollHeight;
  } catch (err) {
    errEl.textContent = `读取日志失败：${err.message || "未知错误"}`;
    errEl.hidden = false;
    preEl.textContent = "";
  }
}

function initTheme() {
  const saved = window.localStorage.getItem("site_hub_theme");
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDarkNow = current ? current === "dark" : prefersDark;
    const next = isDarkNow ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    window.localStorage.setItem("site_hub_theme", next);
  });
}

function initListeners() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.category = btn.dataset.category || "all";
      document.querySelectorAll(".tab-btn").forEach((item) => {
        const active = item === btn;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      renderSites();
    });
  });

  document.getElementById("site-search").addEventListener("input", (evt) => {
    state.search = evt.target.value || "";
    renderSites();
  });

  document.getElementById("refresh-now-btn").addEventListener("click", () => refreshAll());
  document.getElementById("retry-connection-btn").addEventListener("click", () => refreshAll());

  document.getElementById("reload-registry-btn").addEventListener("click", async () => {
    try {
      await fetchJson("/api/sites/reload", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Hub-Token": state.sessionToken,
        },
        body: "{}",
      });
      await refreshAll();
    } catch (err) {
      setBanner("danger", "重载注册表失败", err.message || "未知错误", true);
    }
  });

  document.getElementById("site-grid").addEventListener("click", (evt) => {
    const target = evt.target.closest("button[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    const siteId = target.dataset.site;
    if (!siteId || !action) return;
    if (action === "logs") {
      openLogs(siteId);
    } else if (action === "start" || action === "stop" || action === "restart") {
      runSiteAction(siteId, action);
    }
  });

  const dialog = document.getElementById("logs-dialog");
  document.getElementById("logs-close-btn").addEventListener("click", () => dialog.close());
  document.getElementById("logs-refresh-btn").addEventListener("click", () => {
    if (state.activeLogSiteId) openLogs(state.activeLogSiteId);
  });
  document.getElementById("logs-copy-btn").addEventListener("click", async () => {
    const text = document.getElementById("logs-content").textContent || "";
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
      } catch (_err) {
        /* ignore clipboard permission denial */
      }
    }
  });
  dialog.addEventListener("click", (evt) => {
    if (evt.target === dialog) dialog.close();
  });

  const autoCheckbox = document.getElementById("auto-refresh-toggle");
  const schedule = () => {
    if (state.refreshTimer) window.clearInterval(state.refreshTimer);
    if (autoCheckbox.checked) {
      state.refreshTimer = window.setInterval(() => {
        if (document.visibilityState === "visible") {
          refreshAll();
        }
      }, 8000);
    }
  };
  autoCheckbox.addEventListener("change", schedule);
  schedule();
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initListeners();
  renderSummary();
  renderSites();
  refreshAll();
});
