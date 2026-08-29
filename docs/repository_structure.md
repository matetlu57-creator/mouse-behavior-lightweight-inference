# 仓库结构、模块边界与运行链路

本文说明 mouse-behavior-lightweight-inference 的 main 分支当前结构、主要
Python 模块、输入输出、行为分析链路、调试方式和 Git 协作规则。它面向三类
读者：

1. 想运行一次分析的人；
2. 想修改行为算法或渲染逻辑的开发者；
3. 需要检查结果、定位问题或维护 GitHub 仓库的人。

本文只描述仓库中已经存在的职责边界，不把视频、YOLO 缓存、模型权重或本地
分析结果当作源码的一部分。未来代码发生变化时，应同时更新本文、对应模块的
测试和相关算法文档。

## 一、项目定位

项目是一个面向多鼠视频的 Python 行为分析仓库，主要包含两条运行路径：

轻量缓存路径：读取已经完成的 YOLO Pose 七关键点缓存，进行轨迹匹配、笼界
范围学习、候选鼠对筛选、几何与运动学分析、行为 FSM、事件导出和渲染。

完整管线：保留较完整的检测、遮挡处理、身份恢复、ROI 推理和视频输出流程，
用于需要更高召回率或旧版本兼容的场景。完整管线的代码已经放进
src/mouse_behavior/full_pipeline/，不再作为根目录散落脚本维护。

两条路径共用仓库的配置、日志、测试和部分数据契约，但不能把轻量缓存路径的
结果解释成完整遮挡恢复管线的结果。

## 二、仓库总览

当前仓库采用 src layout。顶层目录按职责划分如下：

~~~text
mouse-behavior-lightweight-inference/
├── src/mouse_behavior/       可复用 Python 包和算法模块
├── scripts/                  命令行、批处理、校准和验证入口
├── configs/                  默认配置、运行 profile 和实验覆盖
├── tests/                    unit、integration、regression、e2e 测试
├── docs/                     用户、架构、算法、数据和开发文档
├── examples/                 最小可运行示例
├── tools/                    仓库检查、构建检查和结果比较工具
├── data/                     数据目录说明，不保存真实数据集
├── weights/                  模型权重放置说明，不提交权重文件
├── outputs/                  本地运行结果占位目录，不提交生成物
├── .github/                  GitHub Actions、Issue 和 PR 模板
├── pyproject.toml            Python 包、入口、pytest、ruff、mypy 配置
├── .quality-gate.toml        本地和 CI 共用的质量门命令
├── requirements.txt          运行依赖
├── requirements-dev.txt      开发、测试和质量检查依赖
├── environment.yml           Conda 环境参考
├── CONTRIBUTING.md           开发、日志、测试和 Git workflow
├── SECURITY.md               数据、凭据和安全问题处理规则
├── CHANGELOG.md              版本变化记录
└── README.md                 项目概览和最短运行路径
~~~

根目录只保留项目级配置、文档和入口说明。新的生产 Python 代码不能放在根目
录，版本之间不能通过复制 v2、final2 或 new 文件夹管理。需要并行开发
时使用 Git branch 和仓库外 worktree。

## 三、生产代码的位置

生产代码全部位于 src/mouse_behavior/。目录名称表示职责，而不是个人、日期
或实验编号。

### 3.1 包入口和运行门面

位置：

~~~text
src/mouse_behavior/
├── __init__.py
├── __main__.py
├── lightweight_behavior_inference.py
├── standard_behavior_engine.py
├── parallel_behavior_fsm.py
├── pose_cache.py
├── annotation_website_export.py
├── config.py
├── logging_config.py
├── mask_trigger_controller.py
└── nvenc_video_writer.py
~~~

职责：

- __init__.py：包级公开版本和稳定入口，不在导入时启动任务；
- __main__.py：支持 python -m mouse_behavior 的包入口；
- lightweight_behavior_inference.py：轻量行为分析的应用门面，负责解析输入、
  编排各模块、写入 CSV 和元数据；核心算法不应继续堆回这个文件；
- standard_behavior_engine.py：标准追逐和攻击引擎的兼容入口与对外编排；
- parallel_behavior_fsm.py：个体、鼠对、接触和群体行为的并行状态区域；
- pose_cache.py：使用 YOLO Pose 模型生成七关键点缓存；
- annotation_website_export.py：将内部事件和轨迹转换为标注网站导入包；
- config.py：读取 YAML、解析 extends 继承并进行深度合并；
- logging_config.py：命令行进程的统一日志配置；
- mask_trigger_controller.py、nvenc_video_writer.py：完整管线或视频输出所需
  的辅助能力。

lightweight_behavior_inference.py 是运行编排层，不是所有逻辑的唯一实现处。
当需要修改几何、候选鼠对、FSM 或渲染时，应优先进入对应子包，而不是继续在
门面中增加分支。

### 3.2 core：应用流程编排

位置：src/mouse_behavior/core/

- pipeline.py：轻量分析的高层流程和运行对象；
- __init__.py：公开 LightweightPipeline、PipelineRun 等稳定对象。

该层只负责组织阶段顺序、传递配置和收集运行结果，不应直接复制某个算法的
阈值实现。算法细节进入 preprocessing/ 或 behavior/。

### 3.3 models：模型和推理接口

位置：src/mouse_behavior/models/

- pose.py：Pose 检测结果、关键点和模型接口的轻量表示。

该层描述模型输出如何进入后续代码，不负责完整视频分析、不负责行为分类，也不
负责命令行参数解析。生成缓存时使用外部 YOLO Pose 权重；分析已有缓存时不会
重新加载 YOLO 权重。

### 3.4 tracking：缓存规范化和身份轨迹

位置：src/mouse_behavior/tracking/

- cache.py：读取 yolo_results.*.*.pkl.gz 分块，检查缓存状态，规范化七点
  Pose、框、置信度和中心点，并使用位置与关键点信息进行轻量逻辑 ID 匹配；
- README.md：说明缓存与追踪职责边界。

输入是 YOLO Pose 缓存中的检测记录，输出是按视频帧和逻辑 ID 组织的中心点、
关键点、框、身体长度、姿态质量和有效性数组。该层不根据视频文件名判断行为，
也不把行为标签写入轨迹匹配逻辑。

### 3.5 preprocessing：笼界、几何、运动学和候选鼠对

位置：src/mouse_behavior/preprocessing/

- arena_learning.py：从运动和检测证据学习小鼠笼子范围，保存边界、视频指纹、
  像素到厘米的尺度和审计信息；
- geometry.py：中心点、头部方向、鼻头、鼻尾、身体轴和距离等几何计算；
- kinematics.py：速度、加速度、位移方向和滚动运动特征；
- pair_features.py：基于距离、朝向、速度和时间窗口构造候选鼠对及 Pair 特征；
- constants.py：七个关键点的索引和基础常量；
- README.md：预处理层说明。

候选鼠对筛选是性能优化的关键边界。系统先使用距离、朝向、有效性和笼界范围
生成 valuable_frame，再给候选鼠对建立上下文时间窗；昂贵的鼻体几何、rolling
features 和标准行为证据只在候选窗口内计算。窗口外仍保留完整时间轴，用于状态
重置和硬否决，不能把“未计算”误解成“发生了行为”。

### 3.6 behavior：行为证据和状态机

位置：src/mouse_behavior/behavior/

- standard_evidence.py：为追逐和攻击计算距离、速度、朝向、接触、角色、反应、
  事件分数等连续证据；
- standard_fsm.py：标准追逐和攻击的状态转移、持续时间和角色确定；
- ethogram.py：扩展行为分类、短事件恢复、个体行为、社交行为和群体行为；
- pair_analysis.py：候选鼠对的窗口化分析编排和事件汇总；
- social_fsm.py：扩展社交行为的相对距离和方向 FSM；北医 profile 在这里使用
  框接触、框运动和有限遮挡桥接作为攻击的主要证据；
- engine.py：对外组织标准行为引擎相关接口；
- __init__.py：公开必要的行为分析对象。

当前行为层按语义分为多个区域：

~~~text
个体区域       running、walking、stationary
鼠对区域       together、approach、chase、avoidance、attack
接触区域       nose_head、nose_tail、nose_head_and_nose_tail
群体区域       huddle、isolation
~~~

标准追逐和攻击 FSM 与扩展行为区域并行工作。普通鼻头或鼻尾接触是独立几何事
件，不会仅因为达到接触距离就自动升级为攻击。群体事件必须保存实际参与者
member_ids 和必要时的 member_ids_at_peak，不能把全体小鼠统一标记为群体
行为。

短暂事件恢复只使用轨迹和几何证据，例如距离变化、朝向、速度、接触、发起方和
目标反应。恢复事件会通过 event_recovery 标记，不能使用视频名称、文件夹名称
或人工预设的文件名类别来分类。

### 3.7 data：数据契约

位置：src/mouse_behavior/data/

- schema.py：事件 CSV 的核心文件名和必需字段校验；
- README.md：数据层使用说明。

该层不负责计算行为，只定义输入输出之间的最低契约。例如行为事件至少要包含
行为名称、候选级别、事件范围和起止时间。新增字段时应同步更新数据文档、网站
导出适配器和回归测试。

### 3.8 io：文件和运行目录

位置：src/mouse_behavior/io/

- paths.py：定义运行目录对象，区分 tracking、behavior、visualization、report
  和 logs；
- csv.py：统一 CSV 写入；
- arena_boundary.py：笼界 JSON、PNG 和审计覆盖图的读写；
- __init__.py：I/O 层公开接口。

I/O 层使用 pathlib.Path，不应依赖开发者个人电脑上的绝对路径。配置中的路径
应由运行命令传入，生成物应落在输出目录或被 .gitignore 排除的位置。

### 3.9 visualization：渲染和行为显示

位置：src/mouse_behavior/visualization/

- overlay.py：为每只小鼠构造 ID、行为、框、骨架和参与者显示上下文；
- rendering.py：读取视频、Pose 缓存和事件 CSV，写出带框、ID、骨架、中文行为
  标签和右侧行为面板的 MP4；
- README.md：渲染器和显示约定。

显示上下文与核心行为证据分开。渲染用 start_frame、end_frame 和有限前后文，
统计核心行为优先使用 core_start_frame、core_end_frame 和 core_duration_s。
没有事件的小鼠显示“仅追踪”；群体和社交标签只显示在真实参与者上，不会用一个
全局标签覆盖整帧。

### 3.10 网站导出适配器

位置：src/mouse_behavior/annotation_website_export.py

该模块把内部事件、轨迹、视频尺寸、FPS、关键点名称和骨架连接转换为网站 schema
version 1.0 的完整视频导入包。它负责：

- 生成 annotations.json；
- 生成逐帧 tracks.jsonl；
- 生成 metadata.json；
- 复制或链接完整视频；
- 记录导出数量和跳过原因到报告。

内部事件仍保存 actor_id、target_id 和角色信息。网站文件中的 mouse_ids
会进行排序去重，但不能依赖 ID 的排序表达主动方和被动方。

### 3.11 full_pipeline：完整管线和兼容边界

位置：src/mouse_behavior/full_pipeline/

- high_recall.py：完整管线应用入口；
- extractor_base.py：完整检测、身份恢复、行为提取和片段输出实现；
- dependencies.py：对可选依赖进行延迟检查和友好错误提示；
- __main__.py：支持模块方式启动完整管线。

完整管线属于高依赖、较重的兼容区域。轻量行为分析不应直接导入它来完成普通
缓存分析，否则会把完整检测、模型加载和重型依赖带回轻量路径。修改完整管线时
应优先补充回归夹具和端到端冒烟测试。

### 3.12 utils、日志和计时

位置：src/mouse_behavior/utils/

- logger.py：公开日志辅助接口；
- timer.py：阶段计时器，在日志中输出阶段耗时；
- rolling.py：滚动统计辅助函数；
- __init__.py：公开稳定工具。

可复用模块使用 logging.getLogger(__name__) 或仓库统一 logger。命令行入口通过
logging_config.configure_logging() 配置级别和 handler。运行状态、阶段耗时和可
恢复降级使用 logging，不在高频循环里使用 print() 输出调试信息。

## 四、脚本和模块的区别

scripts/ 只放命令行和批处理入口，算法实现应放在 src/mouse_behavior/。

### 分析和缓存脚本

- scripts/build_lightweight_pose_cache.py：使用 YOLO Pose 生成七关键点缓存；
- scripts/run_lightweight_behavior_inference.py：轻量缓存分析、已有结果渲染和
  行为片段裁剪；
- scripts/run_full_behavior_pipeline.py：完整管线的批处理入口；
- scripts/run_lightweight_behavior_inference.ps1、scripts/run_stage1_stage2.ps1：
  Windows 参数转发和批处理辅助。

### 北医样例、验证和校准脚本

- scripts/render_beiyi_behavior_videos.py：按清单批量渲染北医案例；
- scripts/rerun_beiyi_lightweight_rules.py：重新运行轻量规则并生成验证结果；
- scripts/validate_beiyi_extended_ethogram.py：检查扩展行为标签在样例中的出现；
- scripts/calibrate_standard_behavior.py：使用标注事件做阈值或指标校准；
- scripts/sweep_standard_behavior.py：参数扫描；
- scripts/compare_parallel_fsm_validation.py：比较并行 FSM 验证输出。

### 仓库质量和输出检查脚本

- scripts/validate_repository.py：目录边界、敏感文件和大文件检查；
- scripts/run_quality.py：从 .quality-gate.toml 读取并执行质量门；
- scripts/run_quality.ps1、scripts/run_pytest.ps1：Windows 下固定优先使用项目
  `.venv` 的测试/质量门入口，避免误用全局 Anaconda base；
- tools/check_repository.py：仓库检查的实际实现；
- tools/compare_analysis_outputs.py：规范化比较两次分析输出；
- tools/inspect_distribution.py：检查打包产物内容；
- tools/README.md：维护工具说明。

脚本可以解析参数、配置日志、调用模块和报告结果，但不应复制
geometry.py、pair_features.py、standard_fsm.py 等核心逻辑。

## 五、配置、依赖和模型边界

### 5.1 配置层次

~~~text
configs/default.yaml
          ↓
configs/profiles/fast.yaml
configs/profiles/balanced.yaml
configs/profiles/high_accuracy.yaml
configs/profiles/beiyi.yaml
          ↓
configs/experiments/<experiment>.yaml
~~~

mouse_behavior.config.load_config() 负责解析 extends，路径相对于声明继承
关系的 YAML 文件。推荐新分析使用 configs/default.yaml 或 profile，实验参数
使用 configs/experiments/ 中独立文件记录，不直接改 Python 源码中的阈值。

根目录的 mouse_chase_attack_config.yaml 是完整管线和旧调用方的兼容配置，不是
新实验的首选入口。

### 5.2 运行依赖和开发依赖

- requirements.txt：NumPy、Pandas、PyYAML、SciPy、OpenCV、Pillow、tqdm 等
  运行依赖；
- requirements-dev.txt：pytest、ruff、mypy、coverage、build 等开发和质量工具；
- environment.yml：Conda 环境参考；
- pyproject.toml：包名、Python 版本、console entry point、pytest、ruff、mypy
  和 coverage 配置。

项目要求 Python 3.10 或更高版本。生成 YOLO Pose 缓存时还需要已经安装
Ultralytics 和 PyTorch 的推理环境；缓存分析本身只读取缓存，不重新加载模型。

### 5.3 模型权重

模型权重不进入普通 Git 历史。公开说明位于 weights/README.md，本地约定路径
是：

~~~text
weights/pose/best.pt
~~~

轻量路径使用 YOLO Pose 七关键点模型，不使用 OBB 模型。模型的实际绝对路径可由
缓存生成命令传入，不能写死为某位开发者电脑上的路径。

## 六、输入数据和前置条件

### 6.1 轻量行为分析输入

普通轻量分析需要以下输入：

1. 原始视频，例如 MP4、MOV、AVI、MKV、WMV 或 M4V；
2. 与该视频一一对应且已经完成的 YOLO Pose 缓存目录；
3. YAML 配置或 profile；
4. 视频 FPS，除非从调用方明确提供其他有效值；
5. 预期小鼠数量：通用 20 鼠视频传入 20；北医 RFID-CV 10 鼠示例使用北医脚本默认值 10。

缓存目录至少应包含：

~~~text
yolo_precompute/
├── yolo_results.000000.000000.pkl.gz
├── yolo_results.000000.000001.pkl.gz
├── ...
└── yolo_results_status.json
~~~

yolo_results_status.json 必须标记完成，next_frame 必须与视频总帧数一致。
每条 Pose 记录包含七个关键点、关键点置信度、bbox_xyxy、框置信度和姿态质量。
轻量跟踪输出还会记录 `bbox_observed` 和 `bbox_imputed`。后者只表示短时框级
遮挡保持，不等于检测器在该帧真实检测到了小鼠。

### 6.2 缓存生成输入

缓存生成脚本需要：

- 视频文件；
- YOLO Pose 权重；
- 推理设备；
- batch size、图像尺寸、置信度、IoU 和最大检测数等参数。

缓存生成阶段只负责模型推理和缓存写入，不负责行为分析、行为标签或视频切片。

### 6.3 渲染输入

只渲染已有结果时需要：

- 原始视频；
- 对应 YOLO Pose 缓存；
- lightweight_behavior_events.csv；
- 输出 MP4 路径；
- 可选中文字体路径。

渲染不会重新推理行为，也不会根据视频名称猜测行为类别。

## 七、输出文件和字段含义

一次轻量分析的典型输出目录如下：

~~~text
outputs/<run-id>/
├── lightweight_behavior_events.csv
├── lightweight_contact_events.csv
├── lightweight_pair_summary.csv
├── lightweight_top_evidence.csv
├── lightweight_analysis_metadata.json
├── annotation_website_export_report.json
├── annotation_website_import/
│   └── <video-name>/
│       ├── video.mp4 或 video.mov
│       ├── annotations.json
│       ├── tracks.jsonl
│       └── metadata.json
└── 轻量行为推理_渲染.mp4
~~~

渲染视频只在显式请求渲染时生成；行为片段只在显式请求切片时生成。所有视频、
缓存、模型、日志和结果都应留在本地输出位置，不提交到普通 Git。

### 7.1 主要 CSV

- lightweight_behavior_events.csv：追逐、攻击、接近、回避、一起、个体和群体
  行为事件；
- lightweight_contact_events.csv：鼻头、鼻尾和组合接触事件；
- lightweight_pair_summary.csv：候选鼠对、距离、朝向、速度、FSM 评估和诊断；
- lightweight_top_evidence.csv：高价值证据帧摘要。

### 7.2 事件时间字段

事件包含两套不同目的的时间范围：

- analysis_start_frame、analysis_peak_frame、analysis_end_frame：算法证据
  的分析帧范围；
- core_start_frame、core_end_frame、core_duration_s：核心证据范围和核心
  持续时间；
- start_frame、end_frame、start_time_s、end_time_s、duration_s：用于
  渲染、切片和网站导出的公开范围；
- temporal_padding_frames：公开范围相对于核心证据增加的前后文；
- event_recovery：记录普通 FSM、短攻击恢复或短回避恢复的来源。

公开范围可能比核心范围长。统计行为持续时间时优先使用 core_duration_s；
制作复核视频时使用公开范围。

### 7.3 参与者和角色字段

- behavior：稳定的英文内部行为名；
- behavior_name_zh：渲染用中文行为名；
- actor_id、target_id：主动方和被动方；
- pair_key：鼠对标识；
- member_ids、member_ids_at_peak：群体事件的实际参与者；
- candidate_level、mean_score、peak_score：候选级别和证据分数；
- analysis_mode、behavior_engine：生成路径和引擎来源。

群体事件没有可靠成员证据时不会虚构 ID。旧格式缺少 member_ids 时，网站导出
适配器可以使用兼容的几何回退，但这不等于原始行为事件已经补回了真实成员。

### 7.4 运行元数据和耗时

lightweight_analysis_metadata.json 用于复现和性能分析，通常包含：

- 输入视频、YOLO 缓存和配置路径；
- 视频帧数、FPS、分析帧数、采样步长和预期小鼠数量；
- 笼界和像素到厘米尺度；
- 轨迹有效率和检测统计；
- 候选鼠对数量、总鼠对数量、valuable frame 数量；
- 并行 FSM 的 enabled、mode、版本和区域；
- stage_timings_s 各阶段耗时；
- 总耗时 elapsed_s；
- 行为、接触和网站导出数量。

这份元数据是判断“是否真的减少了鼠对计算”和“哪一个阶段变慢”的主要证据，
不能只根据终端上一次总耗时判断优化是否有效。

## 八、行为判定和显示层级

行为识别不是根据视频名称分类，而是根据轨迹、几何、运动学、接触和状态机证据
进行分类。

### 8.1 标准因果行为

- 追逐：需要持续的距离、速度、朝向和追随关系；
- 攻击：需要发起、接触、角色、目标反应或冲击等组合证据；
- 短攻击和短回避：只有在短时间内仍有足够证据时进入窄门槛恢复分支；
- 鼻头和鼻尾接触：独立接触事件，不自动变成攻击。

### 8.2 并行行为区域

- 个体行为区域：每只小鼠独立判断 running、walking、stationary；
- 鼠对行为区域：判断 together、approach、chase、avoidance、attack；
- 接触行为区域：判断 nose_head、nose_tail 和组合接触；
- 群体行为区域：判断 huddle、isolation，并保存真实成员。

这些区域并行运行，事件 CSV 保留各区域的核心证据。渲染器根据实际参与者和固定
层级“群体行为 > 社交行为 > 个体行为”决定每个 ID 的文字：参与 huddle 或
isolation 的小鼠不再显示较低层级的鼠对或个体行为，未参与群体事件的小鼠仍可
显示自己的社交或个体行为；不会因为当前帧有群体行为，就把整帧所有小鼠都标成
群体行为。

## 九、调试和性能定位

### 9.1 调试原则

采用以下顺序：

~~~text
复现问题
    ↓
记录视频、缓存、配置、模型和解释器
    ↓
查看日志和 lightweight_analysis_metadata.json
    ↓
缩小到缓存、追踪、候选鼠对、FSM、导出或渲染阶段
    ↓
补最小 pytest 回归
    ↓
修改最小模块并重新验证
~~~

不要先凭终端的一条错误信息大范围重写算法，也不要用“感觉更快”替代计时证据。

### 9.2 日志级别

命令行入口提供 --log-level：

- DEBUG：候选窗口、分支决策、详细诊断和阶段细节；
- INFO：模型或缓存加载、视频信息、阶段进度、输出位置和耗时；
- WARNING：回退、缺失字段、质量问题但程序仍可继续；
- ERROR：当前操作失败；
- CRITICAL：进程无法安全继续。

可复用库模块不调用 logging.basicConfig()，由应用入口统一配置 handler。高频
帧循环不能每帧输出 INFO；需要定位时使用 DEBUG、抽样日志或结果摘要。

### 9.3 常见问题定位

缓存找不到或不完整：

1. 检查缓存目录是否存在 yolo_results_status.json；
2. 检查状态是否为 complete；
3. 检查 next_frame 是否等于视频帧数；
4. 检查缓存与视频的帧数、FPS 和文件指纹是否对应。

候选鼠对过多或运行变慢：

1. 查看元数据中的 total_pair_count 和 candidate_pair_count；
2. 查看 pair_prefilter.valuable_frame_count 和比例；
3. 查看 pair_window.candidate_active_frame_count；
4. 查看 standard_behavior_engine.evaluated_pair_frame_count；
5. 比较 stage_timings_s 中的候选筛选、Pair 特征和事件汇总耗时。

攻击或短暂回避没有显示：

1. 先查看 lightweight_contact_events.csv 是否有接触证据；
2. 查看行为事件中的 peak_score、角色字段和 event_recovery；
3. 区分核心证据只有一到两帧与公开渲染区间的差别；
4. 使用同一缓存和配置运行回归样例，不用视频名人为补标签。

渲染视频只有“仅追踪”：

1. 确认使用的是同一分析输出的事件 CSV；
2. 检查事件的 start_frame 和 end_frame 是否覆盖当前视频帧范围；
3. 检查 actor_id、target_id、member_ids 是否有效；
4. 使用 --log-level DEBUG 检查渲染器是否读取到了事件；
5. 检查中文字体路径和视频编码器是否可用。

### 9.4 性能验证

性能修改必须固定同一视频、缓存、配置、采样步长和硬件环境，至少记录：

- 总耗时；
- 各阶段耗时；
- 总鼠对数量和候选鼠对数量；
- valuable frame 比例；
- FSM 实际评估帧数；
- 输出事件数量和回归比较结果。

只有在行为结果、事件数量或允许的差异范围通过回归后，才能把运行时间下降称为
有效优化。

## 十、测试、质量门和 CI

测试按风险分层：

~~~text
tests/unit/          几何、笼界、配置、FSM、日志、计时和单模块行为
tests/integration/   CSV、网站导出、渲染、仓库边界和跨模块契约
tests/regression/    历史输出、性能、旧版本夹具和行为回归
tests/e2e/           CLI、模块入口和完整路径冒烟
~~~

已有旧实现仅作为 tests/regression/fixtures/legacy_v138/ 的回归夹具保存，不能
作为新的生产代码入口。

常用验证命令：

~~~powershell
python -m pytest -q
python scripts/validate_repository.py
python scripts/run_quality.py
python scripts/run_quality.py --ci
git diff --check
~~~

.quality-gate.toml 统一定义格式检查、lint、类型检查、仓库检查、分层测试、
coverage 和构建步骤。GitHub Actions 在 Python 3.10 和 3.11 上执行 CI，仓库权限
默认是 contents: read。修改代码时，先补能反驳问题的最小测试，再运行相邻层和
完整质量门。

日志测试使用 pytest 的 caplog 检查级别和关键上下文，不把完整格式化字符串
当作稳定业务接口。文件 I/O 测试优先使用 tmp_path，外部模型和视频边界只在
必要处 mock，核心行为逻辑仍需要真实输入结构的单元或集成证据。

## 十一、Git branch、worktree 和 GitHub 交付

### 11.1 本地协作规则

一个功能或一个逻辑修复使用一个 branch。需要同时开发两个互不影响的方向时，
使用仓库外 worktree：

~~~powershell
git fetch origin main
git worktree add -b feat/another-direction ..\mouse_behavior_another_direction origin/main
~~~

不要在当前仓库根目录复制一份完整源码，也不要把视频、缓存、权重或结果加入
commit。

提交前检查：

~~~powershell
git status --short --branch
git diff --check
git diff --staged
git log --oneline --decorate -8
~~~

每个 commit 保持一个清晰主题。模块调整、行为修复、文档更新和大规模生成物
清理不要混成无法审查的单个提交。

### 11.2 GitHub main

GitHub 远端仓库是：

[mouse-behavior-lightweight-inference](https://github.com/matetlu57-creator/mouse-behavior-lightweight-inference)

main 是受保护的默认分支。推荐流程是：

~~~text
本地 branch
    ↓
本地测试和质量门
    ↓
推送 branch
    ↓
Pull Request
    ↓
CI、代码审查和安全检查
    ↓
合并 main
~~~

普通 HTTPS Git 连接不可用时，不能把“本地 commit 成功”误报成“已经上传”。应
通过 GitHub API 或恢复网络后使用正常的 fetch/push 验证远端 commit、文件清单和
Actions 结果。远端 main 的文件树应与待交付版本逐项比对。

## 十二、数据安全和发布边界

以下内容不进入普通 Git 历史：

- 原始视频和渲染视频；
- YOLO Pose 缓存；
- 模型权重；
- 私人标注、SQLite 数据库和真实数据集；
- API token、密码、cookie、私钥和本机凭据；
- 临时日志、调试 dump、缓存和分析输出。

data/README.md、weights/README.md 和 SECURITY.md 只提供放置、授权和安全
边界说明。公开仓库当前没有擅自选择一个开源许可证；使用或发布源码前应确认
版权持有人和第三方模型、数据的许可。

## 十三、扩展代码时放在哪里

新增功能可以按以下规则选择位置：

1. 会被多个入口复用的算法或数据处理，放入 src/mouse_behavior/ 对应职责包；
2. 只负责参数解析、批处理或一次性校准的代码，放入 scripts/；
3. 新输入输出契约放入 data/ 说明和 src/mouse_behavior/data/ 校验；
4. 新配置放入 configs/，不要把机器路径写进源代码；
5. 新行为判定放入 behavior/，并补充时间边界、角色和参与者测试；
6. 新的显示效果放入 visualization/，不要把绘图逻辑写进 FSM；
7. 新的架构选择记录在 docs/adr/；
8. 修复一个实际问题时，先在 tests/regression/ 增加能失败的回归测试；
9. 任何日志状态使用 logging，不要在库模块增加调试 print()；
10. 修改公共字段、配置或网站 schema 时，同时更新迁移说明和导出测试。

推荐的最小开发任务描述应写清楚：输入、输出、不变条件、目标模块、日志级别、
测试样例、性能指标和验证命令。这样无论由人还是 AI 协助实现，都能保持代码可
复用、可审查和可回滚。

## 十四、当前验证边界

仓库的单元、集成、回归和端到端测试用于保护代码契约。北医样例验证可以检查某
些行为类别是否在视频或案例层面出现，但视频级覆盖不等于逐帧 Precision、Recall、
F1，也不等于 actor 或 target 角色准确率。

若要声称识别准确度提升，需要额外准备带帧级行为起止和参与者真值的验证集，并
报告：

- 行为类别的 Precision、Recall、F1；
- 主动方和被动方角色准确率；
- 短暂行为的召回率；
- 群体成员识别准确率；
- 渲染与 CSV 的一致性；
- 相同 workload 下的耗时、内存和 GPU 使用情况。

因此，本文说明的是当前仓库如何组织和运行，不把结构化、可测试和可复现误写成
已经完成科学准确性验证。

## 十五、相关文档

- [项目 README](../README.md)
- [快速开始](quickstart.md)
- [安装与环境](installation.md)
- [仓库架构](architecture.md)
- [算法说明](algorithms.md)
- [输出格式](data_format.md)
- [配置说明](configuration.md)
- [测试与开发](development/testing.md)
- [贡献规范](../CONTRIBUTING.md)
- [安全策略](../SECURITY.md)
- [架构决策 0001](adr/0001-use-git-history-instead-of-copied-version-trees.md)
- [架构决策 0002](adr/0002-package-full-pipeline-and-remove-root-entrypoints.md)
