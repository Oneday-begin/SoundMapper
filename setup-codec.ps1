$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m soundmap.codec --setup
if ($LASTEXITCODE -ne 0) { throw 'EnCodec setup failed; cached files are safe to resume.' }
