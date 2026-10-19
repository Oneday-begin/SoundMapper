$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -m soundmap.genre_model
if ($LASTEXITCODE -ne 0) { throw 'Genre training failed; inspect the output.' }
