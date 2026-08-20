# 2026-08-10 阈值校准记录

## 结论

本次使用同一 `best.pt`、同一 v1.24 特征提取流程和 v1.43 标准行为引擎，对 12 个视频做了离线重放：

- 追逐：3 个；
- 攻击：3 个；
- 鼻头接触：3 个，作为普通接触负样本；
- 鼻尾接触：3 个，作为普通接触负样本。

没有发现单独命名为“扭打”或“遮挡”的标注目录；这两类只作为攻击视频中的标准引擎子类型诊断，不当作独立人工真值。由于提供的是视频分类文件夹，而不是逐帧事件标注，本次 Precision、Recall、F1 是视频级指标；actor/target accuracy 没有人工角色真值，因此必须记为 N/A。

## 已写入的配置变更

文件：`mouse_chase_attack_config.yaml`

| 层级 | 参数 | 原值 | 校准值 | 依据 |
|---|---|---:|---:|---|
| weak chase | `enter_score` | 0.66 | 0.70 | 与 0.80 秒连续确认组合时，减少接触误报 |
| weak chase | `confirm_seconds` | 0.30 | 0.80 | 抑制持续普通接触造成的追逐误报 |
| strong attack | `occlusion_confirm_score` | 0.80 | 0.64 | 攻击1的遮挡证据峰值约 0.75，原值漏检 |
| compatibility | `selected_role_fallback.confidence` | 未配置 | 0.20 | 旧版 CSV 没有方向运动列时，使用已有 selected actor/target 作为低置信度 role hint；只在两个方向证据完全打平时生效 |

strong chase 没有降阈值：搜索中能召回 2/3 追逐视频的候选会引入 5 个误报，不能称为安全校准结果。weak attack 也暂不下调/上调：3 个攻击视频全部召回，但 3 个普通接触视频会产生 `grapple_fight` 误报，当前阈值网格无法在保持 3/3 召回的同时消除它们。

## 改前与改后

| 层级 | 行为 | 改前 Precision | 改前 Recall | 改前 F1 | 改后 Precision | 改后 Recall | 改后 F1 | 改后 TP/FP/FN/TN |
|---|---|---:|---:|---:|---:|---:|---:|---|
| weak | chase | 0.200 | 0.667 | 0.308 | 0.667 | 0.667 | 0.667 | 2 / 1 / 1 / 8 |
| strong | chase | 0.000 | 0.000 | N/A | 0.000 | 0.000 | N/A | 0 / 1 / 3 / 8 |
| weak | attack | 0.500 | 1.000 | 0.667 | 0.500 | 1.000 | 0.667 | 3 / 3 / 0 / 6 |
| strong | attack | 1.000 | 0.667 | 0.800 | 1.000 | 1.000 | 1.000 | 3 / 0 / 0 / 9 |

改后 strong attack 的 3 个正样本分别由 `occlusion_fight`（攻击1）和 `grapple_fight`（攻击2、攻击3）触发；这些事件的 actor/target 仍为 `-1/-1`，因为当前输入没有可验证的角色标注，不能把 selected role 当作 accuracy 真值。

## 可复现文件

- `calibrate_standard_behavior.py`：对已有成对特征 CSV 按鼠对重放 FSM，并输出视频级指标。
- `sweep_standard_behavior.py`：对追逐进入/确认时长以及攻击动态、扭打、遮挡确认阈值做网格搜索。
- 外部复测报告保存在未纳入 Git 的 `<external-report-dir>/threshold_calibration_20260810/`，包括 `calibration_summary.md`、`calibration_report.json`、`video_results.csv` 和 `standard_events.csv`。

下一轮如果要真正校准 event-level Precision/Recall/F1 和 actor/target accuracy，需要为每个视频补充逐帧事件区间及 actor/target ID；仅靠文件夹分类标签无法可靠计算这两个层面的指标。
