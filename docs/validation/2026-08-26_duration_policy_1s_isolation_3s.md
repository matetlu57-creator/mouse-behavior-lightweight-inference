# 北医行为持续时间规则验证

日期：2026-08-26

## 本次规则

本次只调整持续时间可靠性门，没有修改北医 profile 的速度、距离、笼界或模型
参数。规则如下：

| 行为 | 核心证据最短时间 | 说明 |
| --- | ---: | --- |
| running | 0.5 秒 | 按规范明确时长 |
| walking、stationary、together、huddle | 1 秒 | 按规范明确时长 |
| approach、avoidance、attack、nose_head | 1 秒 | 规范未规定固定时长，使用项目统一下限 |
| chase | 2 秒 | 按规范明确时长 |
| nose_tail | 0.5 秒 | 按规范明确时长 |
| isolation | 3 秒 | 当前项目规则；覆盖此前的 10 秒设置 |

渲染和网站导出的 `duration_s` 可能包含有限复核前后文，正式行为判断和本报告
优先使用 `core_duration_s`。不足核心时长的攻击候选不会因为前后文 padding 被
升级为正式攻击。鼻头和鼻尾组合接触同时满足两个成分的规则，使用较严格的
1 秒鼻头门。

## 实现位置

- `configs/profiles/beiyi.yaml`：记录北医 profile 的持续时间配置；
- `src/mouse_behavior/behavior/ethogram.py`：扩展行为和接触事件的核心时长门；
- `src/mouse_behavior/behavior/pair_analysis.py`：对标准 FSM 事件执行最终行为级门控，
  防止标准攻击事件绕过 1 秒规则；
- `src/mouse_behavior/lightweight_behavior_inference.py`：把 profile 中的接近、回避、
  攻击门传入最终汇总；
- `scripts/validate_beiyi_extended_ethogram.py` 和
  `scripts/rerun_beiyi_lightweight_rules.py`：输出逐视频持续时间审计；
- `tests/unit/test_analysis_orchestration.py`、`tests/unit/test_config_loader.py`、
  `tests/unit/test_lightweight_contact_detection.py`：覆盖配置、标准事件、鼻头、
  鼻尾和组合接触的边界。

## 验证方式

北医示例是视频或文件夹级分类目录，不是逐帧人工真值。因此本次报告只表示：

1. 目录对应的视频中是否出现了达到核心持续时间门的同名事件；
2. 规则是否在 CSV、渲染输入和网站导出前统一生效。

使用上一轮已经生成的 39 个 YOLO Pose 缓存，只重新执行轻量跟踪、行为规则、事件
导出和时长审计，没有重复 YOLO 推理。最终输出目录为：

`<validation-output-root>/duration_policy_1s_isolation_3s_v2`

验证共处理 39 个视频，原始视频总时长约 193.77 秒，规则复跑耗时约 232.03 秒；
39 个缓存均可读取，缓存来源环境为 `yolo26`。结果文件为：

- `beiyi_validation_summary.json`；
- `beiyi_video_validation.csv`；
- `beiyi_behavior_coverage.csv`。

## 结果

| 目标行为 | 视频数 | 达到持续时间门的视频数 | 视频级覆盖 |
| --- | ---: | ---: | ---: |
| running | 3 | 3 | 3/3 |
| walking | 3 | 3 | 3/3 |
| stationary | 3 | 3 | 3/3 |
| together | 4 | 4 | 4/4 |
| approach | 3 | 0 | 0/3 |
| chase | 3 | 0 | 0/3 |
| avoidance | 3 | 0 | 0/3 |
| attack | 3 | 0 | 0/3 |
| nose_head | 3 | 0 | 0/3 |
| nose_tail | 3 | 3 | 3/3 |
| huddle | 4 | 4 | 4/4 |
| isolation | 4 | 3 | 3/4 |

总计 23/39 个视频出现达到对应核心持续时间门的目标事件；所有视频本身都足够
长，可以验证相应时长，因此没有 `clip_too_short_to_verify`。孤立样例 1、2、4
分别保留了约 3.93 秒、6.47 秒和 4.43 秒核心事件；孤立样例 3 没有形成可导出
的孤立事件，属于行为证据未命中，不是 3 秒上下文不足。

接近、回避、攻击和鼻头接触目录中的现有几何候选均没有形成至少 1 秒的正式核心
事件，故本次结果没有把短暂候选伪装成满足规范的行为。这也说明下一步若要提高
这些类别的召回率，应改进跨帧证据连接、身份连续性或标注对齐，而不能降低本次
持续时间门。

## 测试证据

本次修改后已通过：

```text
25 passed in 1.23s  （持续时间和时序定向测试）
102 passed in 27.21s （全量 pytest）
```

本报告的覆盖数字不能替代带逐帧区间和 actor/target ID 的 Precision、Recall、
F1 及角色准确率评估。
