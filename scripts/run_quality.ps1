[CmdletBinding()]
param(
    [switch]$CI,
    [string]$Profile = "",
    [string[]]$Step = @(),
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
# Keep third-party diagnostics (including Chinese/Unicode messages) from being
# encoded with the legacy Windows code page in the child Python process.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Resolve-ProjectPython {
    param([string]$RequestedPython)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPython)) {
        $candidate = (Resolve-Path -LiteralPath $RequestedPython -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "指定的 Python 不存在: $candidate"
        }
        return $candidate
    }

    $localPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython -PathType Leaf) {
        return (Resolve-Path -LiteralPath $localPython).Path
    }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython -PathType Leaf) {
            return (Resolve-Path -LiteralPath $condaPython).Path
        }
    }

    $pathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pathPython -and (Test-Path -LiteralPath $pathPython.Source -PathType Leaf)) {
        Write-Warning "未找到项目 .venv，质量门将使用 PATH 中的 Python: $($pathPython.Source)"
        return $pathPython.Source
    }

    throw "找不到可用的 Python。请先创建 .venv，或使用 -Python 指定 python.exe。"
}

$PythonExe = Resolve-ProjectPython $Python
$QualityArgs = @()
if ($CI) { $QualityArgs += "--ci" }
if (-not [string]::IsNullOrWhiteSpace($Profile)) { $QualityArgs += @("--profile", $Profile) }
foreach ($Name in $Step) { $QualityArgs += @("--step", $Name) }

Write-Host "Project Python: $PythonExe"
& $PythonExe (Join-Path $RepoRoot "scripts\run_quality.py") @QualityArgs
if ($LASTEXITCODE -ne 0) {
    throw "质量门失败，退出码: $LASTEXITCODE"
}
