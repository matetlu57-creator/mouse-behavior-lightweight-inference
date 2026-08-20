# Installation

The lightweight path requires Python 3.10 or newer and the dependencies in
`requirements.txt`. Development checks additionally use
`requirements-dev.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

For an existing environment:

```powershell
python -m pip install -r requirements-dev.txt
```

Pose-cache generation may require the user's existing PyTorch and Ultralytics
Conda environment. Keep that environment separate if it is already working;
the public repository does not commit model binaries.

NVENC acceleration is optional and applies only to video encoding. Verify the
local FFmpeg/OpenCV build and NVIDIA driver before selecting it; analysis from
an existing Pose cache remains CPU-oriented and does not require NVENC.
