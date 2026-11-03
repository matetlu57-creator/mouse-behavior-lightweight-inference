# V34：源视频完整过程边界与单事件目标高亮

V34 修复过程边界和导出语义，不把人工精裁数据集重新裁剪成所谓的新识别结果。当前运行输入是 Video Project 22 的既有 YOLO 观测；605 段人工数据集不作为本轮示例输入，也不进行写入、清理或重新标注。

## 最短时长与完整边界

最短时长只决定一个独立事件是否成立。导出长度始终取规则事件的 `[start_frame, end_frame)`，不能达到最低秒数便停止，也不能通过播放上下文使事件达标。

- Huddling 发布同组视觉 ID 的完整稳定过程。短暂未知内部保留为过程间隔，紧凑状态支持独立记录，未知不充当紧凑支持；过程跨度及紧凑支持均须达到配置最小时长（当前为 5 秒）。
- 实测空间解散是过程的终止证据；超限未知间隔分开处理。一个 proposal 的多个合格过程分别输出，不能只输出得分最高的一段。
- 驻留筛选改变参与 ID 集合后，必须使用最终集合重新计算连通性、支持区间和过程边界，不能借用已排除成员的桥接作用。回归反例中，原代码将三鼠真正成立的 5 秒错误延长为 11 秒；V34.1 重新计算后只保留这组目标自身成立的 5 秒。
- 计算 core 只负责分配工作。任何与 core 相交的合格父事件保留完整推断区间；同标签、同视觉 ID、同角色的重叠结果才可合并。
- `inference_parent_contexts` 保留各父过程和上下文边界。`process_boundary_status` 标明端点是否在计算上下文内闭合，或触及源视频边界。碰到计算窗口边缘不代表真实行为已经结束。
- 播放上下文默认 0。原始视频包与高亮包的事件、播放帧数和 JSON 范围必须一致；若显式启用上下文，额外帧不参与资格验证。

8 秒过程内有 0.2 秒未知时，可以保留 8 秒过程以及 7.8 秒紧凑支持；不能拆成若干片段再强行裁 5 秒。反过来，8 秒跨度但只有 2 秒紧凑支持，不足以发布 Huddling。

## 高亮与四文件契约

`export_packages(..., target_overlay=True)` 直接从绑定哈希的原视频渲染当前事件，只高亮其 `mouse_ids`，避免全视频叠加结果中其他事件的参与鼠混入片段。

每个叶目录仍只有：

1. `clip.mp4`
2. `annotation.json`
3. `metadata.json`
4. `tracks.json`

`original_video=True` 的变体保留源画面；`target_overlay=True` 的变体为 `frame_exact_event_target_overlay`，与前者互斥。高亮包中的轨迹仍是原始检测观测。两个变体不能共用一个含糊的“原始视频”名称。

绿色框只来自该源帧实际存在且坐标有限的检测。缺失目标显示 `not detected`，不画虚构位置、不替换视觉 ID。事件采用半开区间，因此真正的首末播放帧分别是 `start`、`end - 1`；`end` 已不属于事件。

## 命令与使用

```powershell
python scripts/run_canonical_behavior_video.py `
  --video <source.mp4> --pose-cache <verified-yolo-cache> `
  --source-receipt <source-receipt.json> --output <new-inference-run> `
  --mode infer --workers 2 --posture-mode v3_review

python scripts/run_canonical_behavior_video.py `
  --video <source.mp4> --events-from <new-inference-run> `
  --output <new-highlight-run> --mode clips --target-overlay `
  --clip-context-seconds 0
```

第二条命令使用完整预测清单。局部审核抽样可以调用同一 `export_packages` 接口，但应另写抽样清单，记录每类上限、未展示原因，并保留完整预测；不能改动原数据集或把抽样数当行为识别准确率。

本轮完整视频先运行冻结的 V34 引擎。发现最终成员集合问题后，使用生产 `group_rule_candidates(..., behaviors=("huddle",))` 在相同源轨迹、标定、配置及 30 个上下文窗口重新生成整个 Huddle 类，其余行为事件保持逐条不变。最终派生回执明确记录 `fresh_full_inference=false`、重算类别、上游回执及哈希；不能把这种单类更新描述为全部类别均由 V34.1 重新推理。

V3 姿态推理保留原始候选、支持和事件 ledgers，整视频运行关闭冗长的逐窗口诊断细节，避免将短视频 receipt 扩大到 GB 级；分类计算本身不因此改动。

## 验证范围

新增回归覆盖完整 8 秒过程、两次扎堆都保留、实测解散、跨 core 父事件、未知不充支持、真实 12 秒媒体导出、首末帧高亮、重叠其他事件不混入、四文件和源轨迹一致性。测试通过证明这些工程约束；真实行为的生物学起止和分类正确性仍需视频人工审核。源视频没有人工逐段真值时，不报告识别准确率，也不更新 A1/A2。
