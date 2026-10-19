$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { python -m venv .venv }
if (Test-Path -LiteralPath 'requirements.lock.txt') {
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
} else {
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
& '.\.venv\Scripts\python.exe' -c "from pathlib import Path; from plotly.offline import get_plotlyjs; Path('static/plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')"
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed' }
Write-Output 'Setup complete. On a new machine, run .\setup-models.ps1 once while online, then .\launch.ps1. Models already cached on this machine need no download.'
