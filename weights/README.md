# 模型权重

本仓库通过 Git LFS 发布已经确认可公开使用的 YOLO Pose 权重。克隆仓库前请先安装并启用 Git LFS：

```powershell
git lfs install
git lfs pull
```

如果当前环境不能使用 Git LFS，仓库中可能只会检出权重指针文件；这时不要把指针文件当作模型使用，应在具备 LFS 访问权限的环境重新执行 `git lfs pull`。

- `pose/best.pt`：YOLO Pose 小鼠关键点模型。

当前源码仓库的默认 Pose 推理模型是 `weights/pose/best.pt`。运行
`scripts/build_lightweight_pose_cache.py` 时如果不显式传入 `--model`，就会使用这份
权重；服务器专用的 TensorRT engine 不属于当前默认流程。

文件校验信息（SHA-256）：

- `pose/best.pt`：`AB2F2FBE7A52980DF993FAD1914B630D9004254A9547FA48F245244662A1BED8`。

轻量行为分析入口读取已经生成的 `yolo_precompute` 缓存，不会在缓存分析阶段重新加载权重。需要生成缓存时，模型路径应指向这份 Pose 权重。OBB 权重不属于当前轻量路径，不在本仓库发布范围内。

由于权重是大二进制文件，当前仓库通过 Git LFS 管理，不作为普通 Git blob 保存。权重检出后的路径是：

```text
weights/pose/best.pt
```

正常克隆源码后，具备仓库 LFS 访问权限时即可通过 `git lfs pull` 获取该文件。视频、缓存、训练数据、推理结果和服务器专用 TensorRT engine 仍不进入仓库。

权重文件不包含视频、缓存、账号信息或个人凭据。
