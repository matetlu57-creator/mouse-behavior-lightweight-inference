# Step 5-A：理毛 / 站立攀爬人工审计

2026-10-05 续作只准备审计材料。无监督筛查的 185 个候选不是人工标签；
主链路的 grooming/rearing 证据门仍关闭，尚未固定检测阈值或重建 Step 15 数据集。

## 当前批次

产物：`outputs/grooming_rearing_review_v1/index.html`、`manifest.json`、
`review_decisions.csv` 以及 30 段 H.264 MP4 和 30 张预览图。

- 站立攀爬、理毛各 15 个窗口。按持续时间优先，先覆盖不同片段/visual track，
  再补同轨窗口；理毛先纳入全部 4 个后肢均速 >5 cm/s 的疑似假阳。
- 编号 001–015 为站立攀爬，016–030 为理毛；016–019 为疑似假阳对照。
- 每段保留前后各 1 秒上下文，遇片段边界截断。帧区间为 clip-local
  `[start_frame, end_frame)`；FPS 从各片段 metadata 读取。
- 左侧场景标记目标视觉 ID；中间原始放大，右侧同一画面的骨架放大。
  红点为鼻头，黄色标题标记候选时间。ID 缺失时明确显示，不补推身份。

## 操作

从仓库根目录启动，打开 `http://127.0.0.1:8767/`：

```powershell
& 'D:\Anaconda3\python.exe' -m http.server 8767 --bind 127.0.0.1 --directory outputs\grooming_rearing_review_v1
```

每段选择确认 / 否决 / 无法判定，并填写实际行为和理由。
按用户最新口径：头部前后反复运动、扭头背部理毛，以及部分站立伴头部前后理毛，
均归 grooming；墙边攀爬站立和空地站立归 rearing，后者的头部前后运动幅度较小。
仅头部左右探索仍可属于 Static。遮挡、关键点塌缩或动作不清晰可标无法判定。
旧批次的候选名不决定实际标签，旧“站立理毛归 rearing”口径已废止。

浏览器仅暂存结论，不回写仓库 CSV。完成后点击“导出审计 CSV”，把导出文件
单独留存供后续校准读取。不要覆盖初始 pending CSV 或 manifest；后者是候选证据记录。
不同批次用候选清单与窗口顺序的 SHA-256 隔离暂存，避免编号相同而标签串用。

## 复现和边界

审计额外依赖 `imageio-ffmpeg>=0.6,<0.7`，已声明在 `review` extra 及开发依赖中。
当前环境已安装 0.6.0，本次未安装或升级包。它提供浏览器可播放 H.264 的编码器。

```powershell
& 'D:\Anaconda3\python.exe' scripts\prepare_grooming_rearing_review.py `
  --candidates outputs\grooming_rearing_screen_v1\candidates.csv `
  --output-dir outputs\grooming_rearing_review_v2 `
  --per-behavior 15 --context-seconds 1
```

输出目录必须不存在；失败批次保留用于诊断，重跑使用新版本目录。
脚本不运行行为预测、不改变尺度或阈值、不读取 annotation.json。
这批选样用于审查候选与假阳模式，不能单独估计全数据集召回率或泛化精度。
人工结论齐备后，再决定是否需要负样本、阈值固化及主链路接线。

## 验证

针对性测试覆盖去重、顺序稳定、轨迹多样性、全部疑似假阳纳入、非法窗口拒绝、
批次标签隔离、真实视频帧对齐、缺 ID 统计、拒绝覆盖和 HTML 转义。
合成视频检查原始放大区确实来自请求帧；真实 30 段逐段核验输出帧数。
质量门沿用交接中的 Ruff、compileall、全量 pytest 与 validate_repository。

2026-10-05 续作核验：Ruff、compileall、validate_repository 均 PASS，pytest 为 321 passed。
既有 30 段共 2500 帧逐帧完整解码，输出帧数、FPS、1632×512 尺寸、预览图、
原视频指纹及候选/批次 SHA-256 均通过核验；浏览器实际播放无错误。
初始 CSV 和当前浏览器显示的结论仍全部 pending（0/30），审计材料齐备，人工标签尚未取得。
