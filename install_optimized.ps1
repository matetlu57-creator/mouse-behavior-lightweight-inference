[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [string]$Python = "python",
    [switch]$Editable,
    [switch]$InstallBundledConfig
)

$ErrorActionPreference = "Stop"
$PackageDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectDir)) { $ProjectDir = $PackageDir }
$ProjectDir = (Resolve-Path $ProjectDir).Path
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "Python was not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectDir "pyproject.toml"))) {
    throw "Not a package source tree: $ProjectDir"
}

$InstallArgs = @("-m", "pip", "install")
if ($Editable) { $InstallArgs += "-e" }
$InstallArgs += $ProjectDir
& $Python @InstallArgs
if ($LASTEXITCODE -ne 0) { throw "Package installation failed with exit code $LASTEXITCODE" }

& $Python -c "import mouse_behavior; import mouse_behavior.full_pipeline"
if ($LASTEXITCODE -ne 0) { throw "Installed package import verification failed" }

if ($InstallBundledConfig) {
    Write-Warning "Configuration is no longer copied into another source tree; use configs/ or pass --config explicitly."
}

Write-Host "Mouse behavior package installation completed."
Write-Host "Lightweight CLI: mouse-behavior-lightweight"
Write-Host "Full pipeline CLI: mouse-behavior-full"
