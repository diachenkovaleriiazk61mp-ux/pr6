$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10 or newer is required'"
if ($LASTEXITCODE -ne 0) { throw "Python check failed" }
if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
}
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip setup failed" }
& $Py -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }
& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependencies installation failed" }
& $Py -c "import torch; assert torch.version.cuda is None; assert torch.distributed.is_gloo_available(); print('CPU PyTorch and gloo: OK')"
if ($LASTEXITCODE -ne 0) { throw "CPU/gloo check failed" }
