# 持久化本地统一站点中心（Persistent Local Site Hub）

默认门户地址：<http://127.0.0.1:8788/>

## 设计目的：脱离会话与网关生命周期持续在线

过去通过终端或 Agent 会话临时启动 `scripts/serve_*.py` 时，HTTP 服务往往挂在会话进程的 Windows Job Object 下。当网关断开、会话重启或超时清理发生时，站点会被连带终止。

**统一站点中心**（`src/mouse_behavior/visualization/site_hub.py` + `scripts/manage_site_hub.py`）解决该问题：

1. **独立进程树拉起**：在 Windows 上优先通过 WMI `Win32_Process.Create` 配合无窗口 `pythonw.exe`（回退为 `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`）在独立进程上下文启动守护进程，完全脱离调用方终端与 Agent 网关。
2. **后台健康巡检与异常自愈**：守护进程每 3 秒探测各已启用站点的回环 `/api/health` 接口；若托管子站点异常退出，自动在 60 秒有界退避策略内重启恢复，日志写入 `outputs/site_hub/logs/<site_id>.log`。
3. **Windows 登录自动启动**：通过 `install-autostart` 注册当前用户 `ONLOGON` 计划任务 `MouseBehaviorSiteHub`（无管理员权限要求，失败时自动降级到启动目录 VBS），重启电脑登录后自动上线。
4. **安全与容错边界**：
   - 仅绑定本机回环地址（`127.0.0.1`、`localhost`、`::1`），启用 Windows `SO_EXCLUSIVEADDRUSE` 端口独占保护。
   - 写控制接口（`/start`、`/stop`、`/restart`、`/reload`、`/shutdown`）强校验 `Host`、`Origin` 与会话 `X-Hub-Token`。
   - 门户页（`site_hub_assets/`）零外部 CDN 依赖；若后台未启动或 `sites.json` 损坏，门户仍完整渲染并提示明确的重试按钮与离线参考卡片，且绝不覆盖损坏的 `sites.json`。

---

## 常驻站点与固定端口分配

| 站点 ID (`site_id`) | 类型 (`category`) | 固定端口 | 默认地址 | 用途与当前数据源 |
|---|---|---:|---|---|
| `site_hub`（门户本身） | 统一导航/守护中心 | `8788` | <http://127.0.0.1:8788/> | 统一站点导航、在线探测、日志查看与启停控制台 |
| `mouse_clip_review` | `review`（人工审核） | `8795` | <http://127.0.0.1:8795/> | 小鼠行为审核与手动采集工作台；当前指向已清洗数据集 `F:\mouse_behavior_cleaned_dataset_20261121`（764 段 / 14 类），输出目录 `outputs/mouse_clip_review_site_cleaned_20261121` |
| `long_video_group_review` | `review`（人工审核） | `8790` | <http://127.0.0.1:8790/> | 长视频群体行为精裁人工审核站（`outputs/long_video_group_review_site`，涵盖 `video22` 与 `dakuangchang1`） |
| `behavior_error_review` | `error_display`（错误显示） | `8776` | <http://127.0.0.1:8776/> | 全 15 类行为评测错误复核站（`outputs/all15_latest_20261116_codex/report.json` 与 `bound_huddle_diagnostics.json`） |

> 历史静态/脚本默认端口参考：`8768`（`serve_behavior_review.py`）、`8769`（`serve_comparison_site.py`）、`8780`（`serve_group_behaviors_site.py`）。新增站点请避开上述已占用端口。

---

## 可复现运维命令

在仓库根目录 `f:\mouse_behavior_new_project` 下使用 `D:\Anaconda3\python.exe` 执行：

```powershell
# 1. 初始化默认站点注册表（幂等，保留已有自定义配置）
D:\Anaconda3\python.exe scripts/manage_site_hub.py init-defaults

# 2. 脱离网关在后台独立启动站点中心（默认 http://127.0.0.1:8788/）
D:\Anaconda3\python.exe scripts/manage_site_hub.py start

# 3. 查看站点中心及全部托管站点的健康状态
D:\Anaconda3\python.exe scripts/manage_site_hub.py status

# 4. 安装或移除 Windows 登录自启
D:\Anaconda3\python.exe scripts/manage_site_hub.py install-autostart
D:\Anaconda3\python.exe scripts/manage_site_hub.py uninstall-autostart

# 5. 重启或安全停止整个站点中心（同时优雅退出托管子站）
D:\Anaconda3\python.exe scripts/manage_site_hub.py restart
D:\Anaconda3\python.exe scripts/manage_site_hub.py stop
```

### 将后续新建的审核站或错误显示站注册部署到中心

按照 `AGENTS.md` 规范，之后所有新开发的审核站（`--category review`）或错误展示站（`--category error_display`）都应分配独立端口与输出目录，并通过以下命令接入统一站点中心：

```powershell
D:\Anaconda3\python.exe scripts/manage_site_hub.py register `
  --id my_new_review `
  --title "新专项行为审核工作台" `
  --category review `
  --port 8796 `
  --data-summary "数据源说明" `
  --description "站点功能简述" `
  -- D:\Anaconda3\python.exe scripts/serve_my_new_review.py --port 8796
```

若站点中心已在运行，`register` 会自动通知其重载配置并拉起该站点。

---

## 验证命令

```powershell
D:\Anaconda3\python.exe -m pytest -q tests/unit/test_site_hub.py
D:\Anaconda3\python.exe scripts/run_quality.py --ci
```
