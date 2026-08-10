# Changelog — v1.40.1-performance-preserving

## Added

- `Detection` 一帧内 `center_px` / `body_length_px` 派生缓存；
- Detection 缓存失效、pickle/旧缓存兼容逻辑，且 dataclass 字段结构保持不变；
- `ObservationHistory.get_window()` 与 `near_frame()` 读取缓存；
- 主程序和底层鼠对特征的一帧历史、轨迹、转角、距离收缩缓存；
- `PairSQLiteStore.add_many()`；
- Stage 2 `pair_key` 一次性行索引；
- 可重复补丁脚本、pytest 回归套件、CSV 逐单元格比较工具；
- PowerShell 安装、Stage 1/Stage 2 运行脚本。

## Changed

- CLAHE 改为线程内对象复用；
- 遮挡簇检测证据在单帧 `build_context()` 内复用；
- 每帧鼠对记录复用同一默认模板；
- debug CSV 改为批量 `writerows()`；
- 主程序和底层版本标识更新为 `1.40.1-performance-preserving`。

## Unchanged

- YAML 文件字节级不变；
- 所有模型、检测、身份、mask、ReID、行为阈值和时间窗口不变；
- 帧数、鼠对数、方向数和输出字段不变；
- Stage 2 仍为 cache-only，不加载 YOLO。
