# 导出行为数据集划分与校准

> **历史诊断流程，不是盲测或联合 Top-1。** 本文中的 `target_id_accuracy`
> 是已知标注目标 ID 后的规则命中率；`strict_target_id_accuracy` 也仍以
> 标注 ID 选择候选。当前训练集调参及真正盲测协议见
> [train-only-blind-calibration.md](train-only-blind-calibration.md)。
> 本脚本会扫描整份带标签导出数据，不得用于新的盲测流程。

`scripts/calibrate_exported_dataset.py` 用于处理 `annotation.json`、
`metadata.json` 和 `tracks.json` 组成的行为片段导出数据。它不重新训练
YOLO，也不实现第二套视觉追踪器；脚本只从已导出的视觉轨迹计算行为学特征，
用于离线调节距离、速度、接触和持续时间阈值。

## 运行

```powershell
$env:PYTHONPATH = "src"
python scripts/calibrate_exported_dataset.py `
  --dataset-root $env:DATASET_ROOT `
  --config configs\default.yaml `
  --output-dir outputs\exported_dataset_calibration_95_target `
  --initial-parameters outputs\exported_dataset_calibration\tuned_parameters.json `
  --target-train-accuracy 0.95 `
  --target-validation-accuracy 0.90 `
  --validation-was-previously-observed
```

默认流程会：

1. 统计每个行为目录的样本数，并把少于 50 个样本的行为排除在实验清单之外；
2. 按原始录制会话划分训练集和验证集，防止同一视频片段跨集合泄漏；
3. 缓存标签无关的轨迹特征，在训练集上先从保守到高召回搜索候选参数；
4. 要求训练集总体以及每个保留行为的目标 ID 命中率都达到
   `--target-train-accuracy`；达到门槛后，在约束内收紧速度、距离、接触和持续时间参数，
   以宏平均 F1 和严格 Top-1 作为次级排序依据；
5. 固定最佳参数后，仅在验证集上运行一次评估。训练门槛未通过时，程序返回状态码 2，
   且不会读取或评估验证集。

`--initial-parameters` 可以继续上一轮训练参数；`--max-passes` 控制训练集坐标优化轮数。
如果某个验证划分过去已经被检查过，应传入 `--validation-was-previously-observed`，
报告会把它标记为诊断集而非全新盲测集。

若要从已保存的 v7 参数仅调整 Huddling 与 Social clustering，可使用
`--two-group-only --initial-parameters <v7 tuned_parameters.json>`。
此模式只搜索两个群体行为的参数；训练集上其他八类的逐片预测必须
与起点一致。使用
`--two-group-parameters` 可指定本轮搜索的参数名，例如
`clustering_max_distance_cm clustering_max_gap_s`。每次运行使用新的输出目录，
保留起点与候选结果以便逐片比较。旧验证集只用于参数冻结后的诊断，不用于选参。

输出目录包含：

- `dataset_split/train.csv`：训练样本清单；
- `dataset_split/val.csv`：验证样本清单；
- `dataset_split/split_manifest.json`：筛选、会话和逐类数量；
- `tuned_parameters.json`：冻结后的参数；
- `tuning_history.csv`：训练集候选参数与目标函数；
- `training_metrics.json`、`validation_metrics.json`：指标；
- `train_predictions.csv`、`validation_predictions.csv`：逐样本预测；
- `calibration_report.md`：人类可读报告。

## 指标解释

导出数据允许不同标签的片段来自同一源视频时间段，因此未标注行为不能直接视为负样本。
这里的主指标是“给定标注目标 ID 后，该行为规则是否命中”的 recall-like 指标，
不能直接当作互斥多分类准确率。报告同时给出：

- `target_id_accuracy`：精确到标注单鼠 ID、无序双鼠 ID 对或完整群体 ID 集合的行为命中率；
- `macro_target_id_accuracy`：按行为类别宏平均的目标 ID 命中率；
- `minimum behavior accuracy`：所有有样本行为中的最低目标 ID 命中率，用于防止大类掩盖小类失败；
- `strict_target_id_accuracy`：只取同一行为族中证据最高行为时的严格 Top-1 准确率；
- `precision` 和 `macro_f1`：揭示高召回阈值造成的额外误报，训练门槛通过后用于选择更有区分力的参数。

若总体命中率达到目标、但某个行为低于目标，报告按逐类验收判为失败；不得继续读取验证标签反向调参。
需要继续改进时，应回到训练集做交叉验证，最后使用新的、从未检查过的测试会话确认泛化能力。

在没有 mTrack/RFID 映射的导出数据上，`track_id` 只能作为片段内的视觉轨迹编号，不能解释为跨视频的生物学 `animal_id`。生产分析仍应使用 Ultralytics 追踪结果和 mTrack/RFID 身份映射。
