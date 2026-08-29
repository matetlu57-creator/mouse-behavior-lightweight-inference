# 安装与环境

## 系统要求

- Python 3.10 或更高版本；
- 运行轻量分析需要 NumPy、Pandas、SciPy、OpenCV、Pillow、PyYAML 和 tqdm；
- 运行测试、质量门和构建检查需要开发依赖；
- 从视频生成 YOLO Pose 缓存时，需要已经可用的 PyTorch 和 Ultralytics 环境；
- NVENC 只影响视频编码，不是轻量缓存分析的必需条件。

## 推荐安装方式

~~~powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-dev.txt
~~~

Windows 下测试和质量门固定使用这个项目本地 `.venv`。不要直接使用包含多个
深度学习环境的全局 Anaconda base 环境，因为它可能加载到不匹配的 CUDA 或
Microsoft C/C++ DLL。安装完成后使用：

~~~powershell
.\scripts\run_pytest.ps1
.\scripts\run_quality.ps1 -CI
~~~

PowerShell 如果阻止当前脚本执行，可以只对当前进程放开策略：

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
~~~

这不会永久修改系统执行策略。

## 已有 Conda 环境

如果本机已经有能运行 PyTorch 和 YOLO 的环境，不需要为了仓库安装过程删除或重建
它。可以在项目环境中安装轻量分析依赖，在 PyTorch 环境中运行缓存生成脚本：

~~~powershell
python -m pip install -e .
python -m pip install -r requirements-dev.txt
~~~

运行时明确使用对应解释器路径，避免把两个环境混用。项目测试/质量门使用
`.venv`；从视频生成 YOLO Pose 缓存使用已经验证可用的 `yolo26`：

~~~powershell
D:\Anaconda3\envs\yolo26\python.exe .\scripts\build_lightweight_pose_cache.py --help
~~~

## 模型和数据

仓库不提交视频、Pose 缓存、模型权重、私人标注或生成结果。权重应放在本地
weights/，数据应放在仓库外；公开发布前需要确认模型、数据和标注的许可证。

## 安装验证

~~~powershell
python -c "import mouse_behavior; print(mouse_behavior.__file__)"
python .\scripts\run_lightweight_behavior_inference.py --help
.\scripts\run_pytest.ps1 tests/unit
~~~

如果 import mouse_behavior 指向其他目录，先确认已经使用 pip install -e .，并检查
当前 PowerShell 中的 python 路径。
