param(
    [ValidateSet("all","prepare","speed","quality","recovery","report")][string]$Stage="all",
    [string]$Results="",
    [int]$PhysicalCores=0
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Run setup.ps1 first" }
$Options = @("--stage", $Stage)
if ($Results) { $Options += @("--resume", $Results) }
if ($PhysicalCores -gt 0) { $Options += @("--physical-cores", "$PhysicalCores") }
$env:USE_LIBUV = "0"
& $Py -u experiment.py @Options
if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
