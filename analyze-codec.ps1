$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m soundmap.codec --samples
if ($LASTEXITCODE -ne 0) { throw 'EnCodec analysis failed; retry resumes saved excerpts.' }
