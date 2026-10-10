# 行为片段初筛与裁剪（2026-10-06 v7）

当前目录：`outputs/behavior_segment_screen_v3/`，冻结基础版本 v6，接近候选修订 v7。
审阅地址：`http://127.0.0.1:8768/?manual=1`。原始 `$env:DATASET_ROOT` 的
人工确认前 1,142 个来源的视频、轨迹与标注保持原样；人工确认后按最新授权同步四类文件并保留完整备份。替换前 v5 的 602 段整包存入
`outputs/behavior_segment_revision_v6_task/previous_review_package_v5/`。

## 2026-10-06 人工确认网站

已升级为本地保存服务；每张审核卡可改行为、视觉 ID 和帧范围，确认通过后进入
`outputs/behavior_confirmed_dataset/dataset/`，同步回写 E 原视频、轨迹、标注与 metadata。
未通过/待审核不进入新数据集；撤回最后一段可恢复完整原片。刷新/重启保留审核状态。
操作、备份、异常恢复和启动方式见 [人工确认与原文件同步](behavior-manual-confirmation.md)。
冻结原片副本和基础清单不变，原数据集在人工审核期间已不再是只读数据源。v7 通过不可变候选修订替换 1 段接近、撤下 3 段不合规自动候选；网站有效候选 594 个、无自动候选 560 卡，审核 ID 和已有结论保留。已通过片段的修正只在再次人工确认后同步。

## 最新行为边界

- Approach：导出**实际首帧**鼻头到对方关键点最近距离必须 **≥6.6875 cm**，不能用前 0.3 s 最大距离替代。允许向前恢复最多 0.3 s 连续有效、运动中、无接触且距离单调减小的真实帧；停顿、离开或缺失帧阻断恢复，无合格起点则不输出自动候选。用户明确要求鼻头到对方身体表面的间距约 **3.5 cm** 时结束。
  用真实关键点凸包计算表面距离，在首次跨过 3.5 cm 的前后两帧中选择更接近 3.5 cm、
  未实际接触的真实帧。最终端点必须在 3.5±0.5 cm 内，否则保留复核；取消旧的
  0.15 s 提前截断和 0.15 体宽余量，剔除后续离开。
  没有实际到达该边界的原片留待复核，不输出“完成接近”。视频是离散帧，
  不补造恰好 3.500 cm 的插值帧；起点与朝向、运动门仍要求成立。
- Chasing：允许追逐中的鼻尾接触，鼻头接触排除。双鼠各 ≥26.75 cm/s 的高速种子、
  中心 ≤16.05 cm、连续至少 1 s 保留。领先鼠转弯与追鼠到达弯道可相差至多 0.5 s；
  必须保持前方追随关系，或位于领先鼠历史路径半个体宽内，方向 cos≥0.707。
  高速种子间最多 0.4 s 的下降只在双鼠各 ≥10.7 cm/s 且追随证据持续成立时连接；
  缺失、停顿、相向和真正分离不能补成追逐。旧“全程零接触”裁剪规则废止。
- Together：有间隔无互动，或挨着静止；排除接近、追逐、攻击、活跃局部头部互动及相对运动。
- 接触、攻击、追逐从接近结束后挖掘。纯接触不能借用另一段接触的累计时长。
  攻击保留动作种子间实际持续的事件，去掉安静首尾；独立输出至少 0.5 s。
- Social clustering：至少一鼠向至少两只静止鼠持续移动，到扎堆形成时截断。
  独立输出至少 0.5 s；真实背离、群体断连和已形成扎堆不能桥成聚集。
- Static 允许左右转头。Huddling、Dispersal、Group locomotion、Isolation 保留既有语义及时长门；
  Walking/Running 合并为 Locomotion；Following/Avoiding 暂停产出。

尺度仍为统一体长 10.7 cm、缺失观测体宽时髋宽 4.1467 cm（约 4.15 cm）。
每原片使用完整片段确定尺度；不从裁剪后缩短的姿态重新估体长，不重复放大已有 cm 轨迹。
原片/行为/确切 ID 集合默认取最长连续核心，不拼接实际分离的事件。

## 理毛与站立的导出资格

理毛包含头部前后、扭头背部与部分站立理毛；站立包括爬墙、空地站立，小幅头部动作。
自动探针已改为头部相对颈部的重复动作，排除整体前身起伏；站立增加相邻伸展变化，
防止恒定蜷缩直接成为站立。然而人工审计回放仍有真站立漏检、真理毛误判站立，
二维鼻/耳/颈/髋/尾缺乏前爪和离地高度，不能据此宣称可靠自动识别。

因此正式输出默认 `ScreenPolicy.posture_export_mode=human_confirmed_only`，仅包含既有
人工明确确认的 **4 个 Grooming、8 个 Rearing**，严格使用已审计的来源、视觉 ID 和帧范围。
审计可来自 Static、Together、Huddling 等任意原标签，层级标签不互相替代。
uncertain、rejected、非鼠结论不导出；人工 CSV 保留原文，不重写用户结论。
自动姿态探针存于 `plan.jsonl.posture_probe_windows`，作为待复核线索，不导为正式姿态视频。
原 Static 视频及 v4 自动姿态视频均保留供后续重新审阅。

## 本批结果

597 段：Approach65、Snout-head_contact76、Attack20、Chasing1、Dispersal47、Grooming4、Rearing8、Group locomotion8、Huddling62、Isolation53、Locomotion129、Snout-rear_contact30、Social clustering16、Static60、Together18。
585 个原片有输出、557 个无合格输出，全部保留原片复核。自动社交/群体片段仍标记 `requires_review=true`，本轮不报告模型准确率。

42 个原 Chasing 中，旧同步方向与速度门把弯道追随截碎；放开鼻尾本身仍不足 1 s。
本轮示例保留一段约 1.37 s 的追逐裁剪；片段依据同向追随与速度证据确定，允许出现鼻尾接触。其他未满足证据门的原片保留复核，不补造追逐标签。

## 导出与核验

`dataset/` 存视频、annotation、metadata、tracks、目标 ID 封面；`review_sources/` 含全部
1,142 个原视频副本。`plan.jsonl` 保存选择及备选，`provenance.json` 保存参数/尺度/代码哈希，
`render_receipts.jsonl` 保存源与输出哈希、帧范围、录制偏移、完整解码回执。
`combined_review_manifest.json`、CSV、`package_summary.json` 与 `validation_report.json` 提供核验清单。

本轮重放全部 642 个双鼠来源；553 个帧范围/ID/标签/状态完全相同的输出核对哈希后复用；其余 44 段重新导出。
复用 metadata 追加当前选择核验，保留历史生成 provenance。原 v5 的 602 段整包存入
`outputs/behavior_segment_revision_v6_task/previous_review_package_v5/`，不加载到当前页。
原 clip 范围用 `[start,end)`；输出 annotation 为 `0..n-1`、时间 `[0,n/fps)`；
metadata.source 保存绝对录制偏移。使用原片真实 FPS，保留所有检测和视觉 track ID，不推断 RFID。
新视频 H.264/yuv420p；每个新视频完整解码检查帧数/FPS/尺寸，复用视频核对既有完整解码回执和现文件哈希。

384 pytest、Ruff、compileall、repository validation PASS。独立审查发现配置域缺口后已修复，
v6 最小修改审查无阻塞项；新增默认 3.5 cm 与旧 3 cm 的参数回归，覆盖离散帧跨界、增距后的截止提前和双鼠阶段不重叠。
接近实际末帧距离 3.020–3.977 cm（全部在 3.5±0.5 内）。其他回归覆盖 3/3.5 cm 终点、未到边界拒绝、转弯时间差、鼻尾接触、不补真实停顿、
恒定蜷缩、前身摆动、人工姿态资格及非 Static 实际导出路径。技术核验不能代替逐片人工行为确认。
主 Top-1 历史模型尚未按这批初筛标签重新训练或迁移；本轮范围为初筛与视频裁剪。

## 重建方式

```powershell
python scripts\screen_behavior_segments.py --dataset-root $env:DATASET_ROOT --output-dir outputs\NEW_BUILD --human-audit outputs\grooming_rearing_human_audit_v1\review_decisions.csv --plan-only
python scripts\screen_behavior_segments.py --dataset-root $env:DATASET_ROOT --output-dir outputs\NEW_BUILD --render-only
python scripts\finalize_behavior_review_package.py --output-dir outputs\NEW_BUILD
python scripts\serve_behavior_review.py --package-root outputs\behavior_segment_screen_v3 --confirmed-root outputs\behavior_confirmed_dataset --source-root $env:DATASET_ROOT --port 8768
```

先在新目录生成、核验，再替换指定生成目录；不覆盖原数据源。
本轮用户要求截止距离再增加 0.5 cm，默认由 3.0 调为 3.5 cm；重算全部 642 个双鼠源，保留未变化的 500 个其他源计划。
缓存理由、代码哈希、源指纹、重规划与替换核验存于 `outputs/behavior_segment_revision_v6_task/`。
无 commit/push，既有 dirty 改动保留。
