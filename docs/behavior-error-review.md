# 行为错误样本复核网站

本地地址：<http://127.0.0.1:8776/>。

在项目根目录启动：

```powershell
python scripts/serve_behavior_errors.py --port 8776
```

运行期间保留这个终端。当前地址已有服务时直接访问即可。默认读取 `outputs/all15_latest_20261116_codex/report.json`；可用 `--report` 指定同格式的报告，`--diagnostics` 指定与该报告哈希绑定的补充诊断。

## 查看方法

默认显示 B 数据集的扎堆失败样本。左侧切换全部 15 类行为，上方可切换 A、B、去重并集；“评测结果”可选择失败、正确或全部。没有失败的行为可点击“查看全部样本”播放视频。

主准确率只使用用户选定的分项评测结果，不使用全笼第一候选来改判。错误原因依据各行为自己的专项输出、规则证据、目标集合和时间要求归类。视频中的颜色含义：

- 黄色：人工标注的目标视觉 ID。
- 橙色：最佳扎堆候选中的非标注目标 ID。
- 灰色：其他轨迹，默认隐藏。

可逐帧、慢放、全屏或开关高亮。标签发生遮挡时分开排放，并用引线指向对应检测框。没有观测的帧不补框，不依据位置推断身份。video 与 canvas 一起全屏；浏览器自带的独立视频全屏可能不包含叠加层，使用页面“全屏”按钮。

## 当前数据与扎堆原因

报告包含 A 429 条、B 605 条，共 1,034 条评测记录；去重并集按 B 优先为 764 段。默认 B 的专项正确数 530/605，失败 75 段。所有 1,034 个源视频已检查容器可读、H.264 编码且帧数与报告一致。

扎堆 25 段失败的具体归因：

| 情况 | 段数 | 含义 |
|---|---:|---|
| 未检测到 | 5 | 没有生成扎堆候选 |
| 目标 ID 不符 | 19 | 有扎堆候选，但没有候选包含与标注完全相同的成员集合 |
| 覆盖时间不足 | 1 | 正确成员集合的候选只覆盖约 21.48%，未达 50% |

这 25 段的专项输出均不是另一个行为标签，因此不将它们归为“错判成其他行为”。其他行为的确存在不同情况，例如鼻头接触被共处结果压过；网页提供各自的输出和规则数值。

原报告把部分扎堆成员不匹配概括为“未识别到”。网站使用报告中的扎堆候选诊断细分原因，保留原始逐段原因供对照，不改写原始评测成绩。部分目标只短暂出现或支持时间不重叠，是成员集合不能通过的重要轨迹证据；仅凭这些数据不能区分跟踪断裂和人工标注成员范围问题，需结合视频复核。

## 数据绑定与故障恢复

报告中的每条记录只绑定自己的 `clip.mp4`、`tracks.json` 与 `metadata.json`。逐帧高亮核对轨迹 SHA-256、视频尺寸和帧数；变化会显示可重试错误，不静默显示旧轨迹。补充扎堆诊断只有在报告 SHA-256 一致时才显示。当前源码与评测指纹不符时，页面明确标记报告属于保存的旧评测。

页面框架、数据集选择与筛选器不依赖接口加载成功。接口不可用、非 JSON、格式损坏或空报告都有页面内提示和重试；不生成演示成绩。服务仅绑定本机地址，媒体路径由评测记录索引，不开放任意文件读取。

## 验证

```powershell
python -m pytest -q -p no:cacheprovider tests/unit/test_behavior_error_site.py tests/unit/test_comparison_site.py
python -m ruff check src/mouse_behavior/visualization/behavior_error_site.py scripts/serve_behavior_errors.py tests/unit/test_behavior_error_site.py
node --check src/mouse_behavior/visualization/error_review_assets/app.js
python scripts/validate_repository.py
```

本轮证据保存在 `outputs/behavior_error_review_20261116/`：视频清单、浏览器检查记录与截图、报告绑定诊断、包内静态资源检查。浏览器检查覆盖三种扎堆原因、全部 15 类的代表视频实际播放、黄色目标框、逐帧/慢放、移动端，以及 HTTP/网络/非 JSON/坏 JSON/错误 schema 后的恢复。

视频高亮使用浏览器原生逐帧回调，在不支持时使用播放时间事件。参考 [MDN requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)。本轮 Modern Web Guidance 的包下载失败，未取得或采用其专门指南。
