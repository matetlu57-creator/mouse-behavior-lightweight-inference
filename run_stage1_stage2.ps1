[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$ProjectDir = "",
    [string]$Video = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$Script = Join-Path $PSScriptRoot "scripts\run_stage1_stage2.ps1"
& $Script @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
