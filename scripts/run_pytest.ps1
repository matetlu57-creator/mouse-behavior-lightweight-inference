[CmdletBinding()]
param(
    [string]$Python = "",
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @()
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
        Write-Warning "未找到项目 .venv，pytest 将使用 PATH 中的 Python: $($pathPython.Source)"
        return $pathPython.Source
    }

    throw "找不到可用的 Python。请先创建 .venv，或使用 -Python 指定 python.exe。"
}

$PythonExe = Resolve-ProjectPython $Python
if ($PytestArgs.Count -eq 0) { $PytestArgs = @("-q") }

Write-Host "Project Python: $PythonExe"
& $PythonExe -m pytest @PytestArgs
if ($LASTEXITCODE -ne 0) {
    throw "pytest 失败，退出码: $LASTEXITCODE"
}
