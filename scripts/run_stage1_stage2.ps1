[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$ProjectDir = "",
    [string]$Video = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectDir)) { $ProjectDir = Split-Path -Parent $ScriptRoot }
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path $ProjectDir "outputs\full_pipeline" }
if ([string]::IsNullOrWhiteSpace($Video)) { throw "Pass -Video with the input video path." }
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python was not found: $Python" }

$Script = Join-Path $ProjectDir "mouse_chase_attack_high_recall.py"
$Model = Join-Path $ProjectDir "weights\best.pt"
$Config = Join-Path $ProjectDir "mouse_chase_attack_config.yaml"

foreach ($Path in @($Script, $Model, $Config, $Video)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required path does not exist: $Path" }
}

$Common = @(
    $Script,
    "--video", $Video,
    "--model", $Model,
    "--config", $Config,
    "--output", $Output
)

Write-Host "=== Stage 1: inference, identity, and lossless cache ==="
& $Python @Common "--stage" "stage1"
if ($LASTEXITCODE -ne 0) { throw ("Stage 1 failed; exit code: " + $LASTEXITCODE) }

Write-Host "=== Stage 2: read Stage 1 cache without loading YOLO ==="
& $Python @Common "--stage" "stage2"
if ($LASTEXITCODE -ne 0) { throw ("Stage 2 failed; exit code: " + $LASTEXITCODE) }

Write-Host ("Completed: " + $Output)
