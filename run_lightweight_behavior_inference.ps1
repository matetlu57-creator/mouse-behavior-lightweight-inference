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
$Script = Join-Path $PSScriptRoot "scripts\run_lightweight_behavior_inference.ps1"
& $Script @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
