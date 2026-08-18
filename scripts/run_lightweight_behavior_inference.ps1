[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$ProjectDir = "",
    [string]$Video = "",
    [string]$YoloCache = "",
    [string]$Output = ".\outputs\lightweight_behavior",
    [double]$Fps = 29.329,
    [int]$ExpectedMice = 20,
    [int]$SampleStride = 1,
    [switch]$RenderVideo,
    [switch]$ExtractBehaviorClips
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = Split-Path -Parent $ScriptRoot
}
$ProjectDir = (Resolve-Path $ProjectDir).Path
$Script = Join-Path $ProjectDir "scripts\run_lightweight_behavior_inference.py"
$Config = Join-Path $ProjectDir "configs\profiles\balanced.yaml"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python was not found: $Python. Activate the project environment or pass -Python with the full path to python.exe."
}

if ([string]::IsNullOrWhiteSpace($Video) -or [string]::IsNullOrWhiteSpace($YoloCache)) {
    throw "Pass both -Video and -YoloCache. The cache must already contain yolo_precompute records."
}

foreach ($Path in @($Script, $Config, $Video, $YoloCache)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path does not exist: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$RenderOutput = Join-Path $Output "轻量行为推理_渲染.mp4"
$BehaviorClipsOutput = Join-Path $Output "behavior_clips"
$AnalysisArguments = @(
    $Script,
    "--video", $Video,
    "--yolo-cache", $YoloCache,
    "--config", $Config,
    "--output-dir", $Output,
    "--fps", $Fps,
    "--expected-mice", $ExpectedMice,
    "--sample-stride", $SampleStride
)

$BehaviorClipArguments = @(
    $Script,
    "--video", $Video,
    "--yolo-cache", $YoloCache,
    "--output-dir", $Output,
    "--extract-behavior-clips",
    "--behavior-level", "all",
    "--behavior-clip-seconds", "5",
    "--max-clips-per-behavior", "200",
    "--behavior-clips-output", $BehaviorClipsOutput
)

Write-Host "=== Lightweight behavior inference ==="
Write-Host ("Video: " + $Video)
Write-Host ("YOLO cache: " + $YoloCache)
Write-Host ("Output: " + $Output)
Write-Host ("Sample stride: " + $SampleStride)
Write-Host ("Render: " + $(if ($RenderVideo) { "enabled" } else { "disabled" }))
Write-Host ("Behavior clips: " + $(if ($ExtractBehaviorClips) { "enabled" } else { "disabled" }))
& $Python @AnalysisArguments
if ($LASTEXITCODE -ne 0) {
    throw ("Lightweight behavior inference failed; exit code: " + $LASTEXITCODE)
}

if ($RenderVideo) {
    Write-Host "=== Rendering annotated behavior video ==="
    & $Python $Script --video $Video --yolo-cache $YoloCache --output-dir $Output --render-only --events (Join-Path $Output "lightweight_behavior_events.csv") --render-output $RenderOutput
    if ($LASTEXITCODE -ne 0) {
        throw ("Behavior video rendering failed; exit code: " + $LASTEXITCODE)
    }
}

if ($ExtractBehaviorClips) {
    Write-Host "=== Extracting behavior-specific raw clips ==="
    & $Python @BehaviorClipArguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Behavior clip extraction failed; exit code: " + $LASTEXITCODE)
    }
}

Write-Host ("Lightweight behavior inference completed: " + $Output)
