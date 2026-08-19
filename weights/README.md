# 模型权重

本目录只提供本地放置说明，源码仓库不包含模型权重。只有在版权和发布许可确认后，才应把批准发布的这一份 YOLO Pose 权重作为 GitHub Release 附件提供：

- `pose/best.pt`：YOLO Pose 小鼠关键点模型。

文件校验信息（SHA-256）：

- `pose/best.pt`：`AB2F2FBE7A52980DF993FAD1914B630D9004254A9547FA48F245244662A1BED8`。

轻量行为分析入口读取已经生成的 `yolo_precompute` 缓存，不会在缓存分析阶段重新加载权重。需要生成缓存时，模型路径应指向这份 Pose 权重。OBB 权重不属于当前轻量路径，不在本仓库发布范围内。

由于权重是大二进制文件，它不进入 Git 提交。若对应 Release 已经公开发布，下载 `pose_best.pt` 后，放置为：

```text
weights/pose/best.pt
```

不需要安装 Git LFS。正常克隆源码后，只有在对应 Release 已经公开且具备下载权限时，才从仓库的 Releases 页面下载该附件。

权重文件不包含视频、缓存、账号信息或个人凭据。
