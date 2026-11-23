# 小鼠行为审核与手动片段采集工作台

> [!NOTE]
> 原待清洗数据集 `F:\mouse` 已完成审核、修剪并整理至 `F:\mouse_behavior_cleaned_dataset_20261121`（764 段 / 14 类）；原 `F:\mouse` 目录已清理。当前工作台已长期部署在 [持久化统一站点中心](persistent-site-hub.md)（统一门户 <http://127.0.0.1:8788/> → 工作台直达 <http://127.0.0.1:8795/>），直接服务于已清洗数据集并写入独立输出目录 `outputs/mouse_clip_review_site_cleaned_20261121`。

本地默认地址：<http://127.0.0.1:8795/>

## 可复现启动命令

在项目根目录 `f:\mouse_behavior_new_project` 运行（推荐通过站点中心常驻托管，或按需单站调试）：

```powershell
D:\Anaconda3\python.exe scripts/manage_site_hub.py start
# 独立调试启动命令：
D:\Anaconda3\python.exe scripts/serve_mouse_clip_review_site.py --source-dir F:\mouse_behavior_cleaned_dataset_20261121 --output-dir outputs/mouse_clip_review_site_cleaned_20261121 --host 127.0.0.1 --port 8795
```

说明：

- 可附加 `--open-browser` 自动打开默认浏览器。
- 服务仅允许绑定本机回环地址（`127.0.0.1`、`localhost`、`::1`），并启用 Windows 独占端口保护（`allow_reuse_address = False` + `SO_EXCLUSIVEADDRUSE`）。
- 导出 H.264 视频使用 `imageio-ffmpeg` 提供的 `ffmpeg`（`libx264` + `yuv420p` + `+faststart`），无需在系统 `PATH` 中另行安装 `ffmpeg`。

## 只读 `F:\mouse` 契约

- `F:\mouse` 作为**严格只读**源素材目录（当前审计为 14 个行为目录、790 个 4 文件片段包，总计约 1.43 GiB；全部为 `1920x1080`、约 `30.01 fps`、`H.264 High L4.0 / yuv420p / faststart` 浏览器直放 MP4）。
- 服务端与存储层在启动时强制校验 `--output-dir` **不得等于或位于 `--source-dir` 内部**，否则立即拒绝启动（`ValueError`）。
- 所有目录读取（`annotation.json`、`metadata.json`、`tracks.json`、`clip.mp4`）均使用 `safe_path` 边界约束与只读文件句柄；所有审核记录与导出片段仅写入独立的 `--output-dir`：
  - `<output-dir>/reviews.json`：人工审核账本（原子写入 `tmp + fsync + os.replace`）。
  - `<output-dir>/collections_manifest.json`：手动采集片段索引账本（原子写入）。
  - `<output-dir>/collected_clips/<safe_behavior>/<export_id>/`：导出的独立片段包目录。
- 叠加层与目标可见性仅基于 `tracks.json` 中的实测检测框和 7 关键点骨架；当部分帧存在目标 `mouse_id` 缺失（实测 `F:\mouse` 中 361/790 片段存在局部丢帧）时，页面显式提示缺失帧，**绝不根据位置、颜色、姿态或运动补框或推断身份**。
- URL 标识 `clip_uid` 采用确定性无空格编码（如 `Group locomotion` 规范化为 `Group_locomotion__...`），安全兼容带空格或连字符的行为目录名。

## 手动片段采集与 4 文件导出规范

工作台支持逐帧定位（`[start_frame, end_frame)` 左闭右开区间）、双滑块范围选择、画布点击选鼠、可选方向角色分配（`Approach`、`Chasing`、`Snout-head_contact`、`Snout-rear_contact`、`Attack`），以及单条/批量导出：

- 导出接口：`POST /api/collect`（兼容别名 `POST /api/crop`）及批量接口 `POST /api/collect/batch`（返回 `202`，通过 `GET /api/collect/jobs/<job_id>` 轮询进度）。
- 当 `padding_frames = p` 时，实际导出视频帧范围为 `[clip_s, clip_e) = [max(0, start_frame - p), min(frame_count, end_frame + p))`，标注区间为相对于导出片段的闭区间 `[start_frame - clip_s, end_frame - 1 - clip_s]`。
- 导出目录通过临时暂存目录完成编码与解码帧数/FPS/尺寸校验后原子发布，包含与 `F:\mouse` 兼容的 4 文件包（加可选高亮视频）：
  1. `clip.mp4`：浏览器原生可播 H.264（`libx264`、`yuv420p`、`+faststart`），与源视频逐帧对齐并通过回读验证。
  2. `tracks.json`：重编号为 `0..N-1` 的帧数组，每帧包含 `frame`、`time`、`source_frame` 及原样保留的实测 `detections`（不伪造缺失轨迹）。
  3. `metadata.json`：`schema_version: "2.0"`，保留 `pose`、`tracks` 及可选 `distance_scale`，更新 `clip`、`source`，并附加 `manual_collection` 溯源信息。
  4. `annotation.json`：完整保留全部 6 个规范字段（`behavior`、`mouse_ids`、`participants`、`confidence`、`frame_range`、`time_range`）并附加 `collection_provenance`。
  5. `clip_overlay.mp4`（可选，`include_overlay: true`）：叠加目标高亮框、角色标签与 HUD 的独立 H.264 审阅视频。
- 重名保护：当相同目标导出包已存在且 `overwrite: false` 时返回 `HTTP 409`，不会静默覆盖；显式传入 `overwrite: true` 时通过备份切换完成原子替换。
- 删除接口：`DELETE /api/collections/<export_id>` 严格校验目标路径位于 `<output-dir>/collected_clips/<behavior>/<export_id>` 内，绝不触碰只读 `source_dir`。
- 视频流接口：`GET /api/video/<clip_uid>` 与 `GET /api/exported-video/<export_id>?variant=raw|overlay` 均支持 HTTP `206 Partial Content` Range 分段拖动播放。

## 安全策略与离线/容错弹性

- **零外部 CDN**：所有静态资源位于 `src/mouse_behavior/visualization/clip_review_assets/{index.html,app.css,app.js}`，随 Python 包一起打包分发。
- **本机防 CSRF / DNS Rebinding**：校验回环 `Host`（含 IPv6 `[::1]:port`）、`Origin` 以及会话 `X-Review-Token`（所有 `POST`/`DELETE` 必须携带）；所有 API 响应附加 `Cache-Control: no-store` 与 `X-Content-Type-Options: nosniff`。
- **页面壳独立渲染与 API 容错**：前端应用壳、行为导航与工作台结构不依赖任何后端接口即可立即渲染。`fetchJson()` 严格校验 HTTP 状态码、`application/json` 响应类型（拒绝将 SPA `index.html` 误判为 JSON）、请求超时与响应 schema；动态文本统一经 `escHtml()` 转义。
- **清晰标记的离线回退**：后端不可达或响应异常时自动进入离线模式，顶部醒目提示 **`演示回退数据 · 非实时 F:\mouse 评测结果`**，绝不将演示数据冒充实时科研结果；离线期间的审核编辑暂存于 `localStorage`，恢复连接后通过 `flushPendingReviews()` 自动同步。
- **损坏账本保护**：若 `reviews.json` 或 `collections_manifest.json` 发生损坏或 schema 异常，后端不会静默覆写原文件，而是完整保留原始损坏字节，读取返回 `HTTP 500`、写入/删除返回 `HTTP 409`（带 `corrupt_store: true`），前端弹出明确告警并保留本地未提交队列。

## 验证命令

```powershell
D:\Anaconda3\python.exe -m pytest -q tests/unit/test_mouse_clip_review_site.py
D:\Anaconda3\python.exe scripts/run_quality.py --ci
```
