[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Video = "",
    [string]$YoloCache = "",
    [string]$Output = ".\outputs\lightweight_behavior",
    [double]$Fps = 29.329,
    [int]$ExpectedMice = 20,
    [int]$SampleStride = 3,
    [switch]$NoClips
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Script = Join-Path $ProjectDir "lightweight_behavior_inference.py"
$Config = Join-Path $ProjectDir "mouse_chase_attack_config.yaml"

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
$ClipsOutput = Join-Path $Output "four_class_clips"
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

$ClipArguments = @(
    $Script,
    "--video", $Video,
    "--yolo-cache", $YoloCache,
    "--output-dir", $Output,
    "--extract-four-class-clips",
    "--clip-level", "strong",
    "--clip-seconds", "5",
    "--max-clips-per-class", "200",
    "--clips-output", $ClipsOutput
)

Write-Host "=== Lightweight behavior inference ==="
Write-Host ("Video: " + $Video)
Write-Host ("YOLO cache: " + $YoloCache)
Write-Host ("Output: " + $Output)
Write-Host "Render: disabled"
& $Python @AnalysisArguments
if ($LASTEXITCODE -ne 0) {
    throw ("Lightweight behavior inference failed; exit code: " + $LASTEXITCODE)
}

if (-not $NoClips) {
    Write-Host "=== Extracting four raw video classes ==="
    & $Python @ClipArguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Four-class clip extraction failed; exit code: " + $LASTEXITCODE)
    }
}

Write-Host ("Lightweight behavior inference completed: " + $Output)
