[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [switch]$InstallBundledConfig
)

$ErrorActionPreference = "Stop"
$PackageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectDir)) { $ProjectDir = $PackageDir }
$ProjectDir = (Resolve-Path $ProjectDir).Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $ProjectDir "backup_before_v1.43_$Timestamp"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

$Files = @(
    "mouse_chase_attack_high_recall.py",
    "mouse_chase_attack_extractor_base.py",
    "mask_trigger_controller.py",
    "nvenc_video_writer.py",
    "standard_behavior_engine.py"
)
if ($InstallBundledConfig) {
    $Files += "mouse_chase_attack_config.yaml"
}

foreach ($Name in $Files) {
    $Source = Join-Path $PackageDir $Name
    $Target = Join-Path $ProjectDir $Name
    if (-not (Test-Path -LiteralPath $Source)) { throw "Package file is missing: $Source" }
    if (Test-Path -LiteralPath $Target) { Copy-Item -LiteralPath $Target -Destination (Join-Path $BackupDir $Name) -Force }
    Copy-Item -LiteralPath $Source -Destination $Target -Force
}

$CompileFiles = @(
    (Join-Path $ProjectDir "mouse_chase_attack_high_recall.py"),
    (Join-Path $ProjectDir "mouse_chase_attack_extractor_base.py"),
    (Join-Path $ProjectDir "mask_trigger_controller.py"),
    (Join-Path $ProjectDir "nvenc_video_writer.py"),
    (Join-Path $ProjectDir "standard_behavior_engine.py")
)
& python -m py_compile @CompileFiles
if ($LASTEXITCODE -ne 0) { throw "Python syntax validation failed; backup: $BackupDir" }

Write-Host ("v1.43 Standard Behavior Engine installation completed. Backup: " + $BackupDir)
Write-Host "The existing project YAML is preserved by default."
Write-Host "Use -InstallBundledConfig to also install the bundled YAML configuration."
