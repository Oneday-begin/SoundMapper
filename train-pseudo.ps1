$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m soundmap.pseudo_model
if ($LASTEXITCODE -ne 0) { throw 'Machine-supervised training failed. See data/runtime/pseudo-training.json.' }
