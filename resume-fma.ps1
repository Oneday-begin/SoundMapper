param([int]$PilotSize = 100, [switch]$Expand)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -m soundmap.cli extract
if ($LASTEXITCODE -eq 3) { Write-Output 'FMA download pending. Run this same command when fma_small.zip is complete.'; return }
if ($LASTEXITCODE -ne 0) { throw 'Archive verification/extraction failed; originals preserved' }
if ($Expand) { $PilotSize = 0 }
& $python -m soundmap.cli fma --limit $PilotSize --seed 2026
if ($LASTEXITCODE -ne 0) { throw 'FMA import failed' }
& $python -m soundmap.cli analyze --source fma
if ($LASTEXITCODE -ne 0) { throw 'FMA analysis failed' }
& $python -m soundmap.cli genre
