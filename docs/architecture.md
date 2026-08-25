# 仓库架构

项目使用 src layout。生产代码、脚本、配置、测试和生成结果具有明确边界。

## 数据流

~~~text
视频与 YOLO Pose 缓存
          ↓
core / pipeline facade
          ↓
轻量缓存轨迹和笼界范围
          ↓
候选鼠对、运动学和接触几何
          ↓
行为、接触、个体和群体并行 FSM
          ↓
CSV、JSON、网站导入包和渲染视频
~~~

## 目录边界

- src/mouse_behavior/core/：协调运行流程，不保存检测器实现和阈值细节；
- src/mouse_behavior/models/：Pose 和缓存模型接口；
- src/mouse_behavior/tracking/：缓存规范化、有效检测和轻量身份匹配；
- src/mouse_behavior/preprocessing/：笼界学习、几何、运动学和候选鼠对筛选；
- src/mouse_behavior/behavior/：标准证据、追逐/攻击 FSM、扩展行为和鼠对编排；
- src/mouse_behavior/data/：CSV、JSON 和网站导出数据契约；
- src/mouse_behavior/io/：运行目录、文件写入和笼界审计输出；
- src/mouse_behavior/visualization/：行为覆盖层、渲染视频和片段；
- src/mouse_behavior/full_pipeline/：完整检测、身份恢复和视频管线；
- scripts/：薄 CLI、批处理、校准和验证入口；
- configs/：默认配置、profile 和实验覆盖；
- tests/：分层测试和最小回归夹具；
- tools/：仓库边界、构建产物和结果比较工具。

## 主要模块关系

轻量分析入口是编排门面，可复用逻辑分布在以下模块：

- tracking/cache.py：缓存归一化和轻量轨迹；
- preprocessing/geometry.py：中心点、朝向和关键点几何；
- preprocessing/kinematics.py：个体速度、加速度和方向；
- preprocessing/pair_features.py：候选鼠对和 Pair 特征；
- behavior/standard_evidence.py：追逐、攻击所需的连续证据；
- behavior/standard_fsm.py：标准追逐/攻击状态转移；
- behavior/ethogram.py：个体、社交、群体和短事件恢复；
- behavior/pair_analysis.py：鼠对分析编排；
- preprocessing/arena_learning.py：小鼠笼子范围学习；
- io/arena_boundary.py：笼界 JSON、PNG 和视频审计输出；
- visualization/rendering.py：结果视频和行为片段渲染。

## 设计约束

1. 可复用逻辑不能依赖命令行参数解析；
2. 库模块使用 logging.getLogger(__name__)，不使用调试 print()；
3. 行为核心证据和渲染上下文必须分开记录；
4. 群体事件必须携带实际参与者，不能为了填满整帧而虚构 ID；
5. 旧版本通过 Git 历史和回归夹具维护，不能复制到根目录；
6. 修改模块接口时必须补充单元或集成回归测试。

完整管线可以通过 scripts/run_full_behavior_pipeline.py、python -m
mouse_behavior.full_pipeline 或安装后的 mouse-behavior-full 启动。
