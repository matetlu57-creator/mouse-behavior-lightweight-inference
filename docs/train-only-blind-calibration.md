# 导出行为数据的训练调参和独立盲测

当前实验的主要指标是片段级联合 Top-1：算法只看轨迹及其视觉 `track_id`，
独立输出一个 `(行为, 目标 ID 集合)`；行为和完整 ID 集合都与标注相同才计为正确。
单鼠 ID、双鼠无序 ID 对和群体完整成员集合分别按对应基数比较。
`track_id` 是片段内视觉编号，不是 mTrack/RFID 确认的跨片段动物身份。

## 当前要求：沿用原训练集和验证集

现在使用原有 `dataset_split/train.csv`（679 条）和 `val.csv`（274 条），不重新
划分、不额外要求测试集，也不以训练是否达到 95% 作为运行验证的前提。
下面的新入口将训练和验证拆为两个命令。训练命令没有验证集参数，验证命令没有
调参功能；验证预测落盘并生成 SHA-256 回执之后，才连接标签计算准确率。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python scripts/calibrate_existing_split.py train `
  --train-manifest outputs/exported_dataset_calibration_95_target/dataset_split/train.csv `
  --output-dir outputs/existing_split_ranker_v1 `
  --steps 500 --learning-rate 0.03 --objective listwise `
  --regularization 0.001 0.01 0.1

python scripts/calibrate_existing_split.py validate `
  --val-manifest outputs/exported_dataset_calibration_95_target/dataset_split/val.csv `
  --model-dir outputs/existing_split_ranker_v1 `
  --output-dir outputs/existing_split_validation_v1
```

每次运行使用新输出目录，避免覆盖已有实验。`report.md`、
`validation_metrics.json`、`validation_predictions_scored.csv` 提供逐行为计数、
准确率和逐样本错误；不带真值的预测保存在 `predictions.csv`。

### 参数搜索方法

1. 在同一组可解释的距离、速度、接触和时长规则的多个阈值配置上生成候选。
   群体候选包含稳定三成员核心和不同空间支持率的完整事件参与集合；这不建立
   新的动物身份映射，不替代 RFID。社会聚集须先在 2 秒形成窗口内有同一群体
   内部平均最近邻距离下降至少 2 cm，随后保持空间连通至少 5 秒；可跨过最长约
   0.5 秒的漏检间隙，但至少 60% 的时段必须有实际连通三鼠的轨迹证据。
2. 对每个候选提取持续比例、相对证据强度、成员可见性、速度分布、距离分布、
   接触比例和运动方向等轨迹特征。文件名、类别目录、标签 ID、录制名称不作为特征。
3. 只用训练标签，以 `--objective` 指定的类别均衡损失优化每类行为的线性加权
   评分参数。本次既有划分的可复现基线使用 `listwise`；`hard_margin` 会针对当前
   最强错误候选更新，直接对应“行为＋完整 ID 集合同时正确”的接受标准，可作为
   训练内对照。两种方式都属于**有监督的启发式评分校准**，并非只修改单个硬阈值。
   系数、目标函数和学习率都保存在 `frozen_model.json` / `run_provenance.json` 中，
   可以检查、复现。
4. 正则化强度在训练集内部按录像会话轮流留出选择；保存训练内逐类结果及
   搜索历史。优先比较最差类别准确率，再比较宏平均和总体准确率，避免以大类
   成绩掩盖小类失败。候选中没有正确完整 ID 集的样本仍计错，不能注入答案。
5. 冻结模型、代码哈希与训练数据来源，再运行原验证集。验证结果不得用于本轮
   参数选择或回调。原始样本名包含行为名称，因此预测文件改用 `v-000001`
   一类匿名行号；仅在预测校验回执锁定后按行号连接标签。每个类别 90% 是验收
   目标，不是程序保证会达到的输出。

当前实现仅接入导出轨迹的离线评估入口，不自动修改生产行为引擎或渲染默认配置。
既有验证集曾经被查看，因此结果称为“原验证集评估”，不包装成全新的独立盲测。

## 历史入口：只用训练集调参并设置盲测门槛

已有 `train.csv` 是训练清单，包含训练标签与轨迹路径。使用新的入口时必须显式
提供这个文件；程序不会扫描导出数据根目录，也不会读取验证或测试清单：

```powershell
python scripts/tune_exported_training.py `
  --train-manifest outputs\exported_dataset_calibration_95_target\dataset_split\train.csv `
  --output-dir outputs\train_only_top1 `
  --max-rule-trials 24 `
  --target-train-accuracy 0.95
```

训练流程按原始录制会话逐一留出：每轮只用其他会话的标签拟合规则阈值及
跨行为证据权重，再对该会话生成一次不带标签的预测。随后使用全部训练样本
重新拟合，分别报告训练集内和会话留出的联合 Top-1、行为单项准确率、
ID 集合单项准确率、逐行为结果与错误类型。候选诊断还会报告：正确答案是否
存在于当前规则生成的候选中；若不存在，单靠调整排名权重不可能命中。

`outputs\train_only_top1\report.md` 和 JSON/CSV 文件只包含训练数据。
此历史入口只有训练集内及会话留出联合 Top-1 **都**达到目标，才生成
`frozen_model.json` 并在 `run_provenance.json` 标记可进入盲测；否则只生成
`provisional_model.json`，盲测推理入口会拒绝运行。门槛没有达到时不得
把旧验证集拿来继续调参或报告为全新盲测准确率。这个门槛不适用于上面当前要求的
原验证集评估入口。当前仅有三个训练会话，
内部会话留出估计仍不够稳健；尤其缺少某些行为的独立会话覆盖。
导出适配器也遵守 Social clustering 最短 5 秒的配置；现有训练标注中
有 3/22 条该行为片段不足 5 秒，需要单独复核标注与定义的一致性。

## 真正盲测：先封存，再预测，最后揭盲

参数和代码冻结前不接触全新测试集，连轨迹也不用于设计规则。独立数据管理员
在冻结后提供不含标签、文件名和样本编号不透露行为的 CSV，列严格为：

```text
sample_id,tracks_path,fps
```

完成预测后，`prediction_receipt.json` 锁定预测 CSV 的 SHA-256；此时才由
独立评分进程读取标签文件。标签 CSV 的列严格为
`sample_id,canonical_behavior,mouse_ids`。两个阶段分别运行：

```powershell
python scripts/predict_exported_blind.py `
  --blind-manifest <仅含轨迹的盲测清单.csv> `
  --model-dir outputs\train_only_top1 `
  --output-dir outputs\blind_prediction_run

python scripts/score_exported_blind.py `
  --predictions-dir outputs\blind_prediction_run `
  --labels <揭盲后才提供的标签.csv> `
  --output outputs\blind_prediction_run\score.json
```

预测入口严格拒绝带标注列的输入、未达训练门槛的模型、修改过的参数或源代码；
评分入口先校验预测文件哈希，再按不透明 `sample_id` 一一匹配真值。
若盲测结果低于期望，不应拿同一份测试集反复改参数并再次称其为盲测。

## 历史结果的边界

`scripts/calibrate_exported_dataset.py` 是历史诊断入口；它的
`target_id_accuracy` 是给定标注目标 ID 后规则是否命中的召回式指标，
`strict_target_id_accuracy` 也仍然以标注 ID 选择候选，**都不是上述联合 Top-1**。
该脚本会扫描整份带标签导出数据，不可用于新的盲测流程。
此前使用过的验证划分已经被查看，不能重新封装为真正未见过的测试集。
