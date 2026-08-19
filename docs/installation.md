# Installation

The lightweight path requires Python 3.10 or newer and the dependencies in
`requirements.txt`. Development checks additionally use
`requirements-dev.txt`.

```powershell
conda env create -f environment.yml
conda activate mouse-behavior-lightweight
python -m pip install -e .
```

For an existing environment:

```powershell
python -m pip install -r requirements-dev.txt
```

Pose-cache generation may require the user's existing PyTorch and Ultralytics
environment. The public repository does not commit model binaries.
