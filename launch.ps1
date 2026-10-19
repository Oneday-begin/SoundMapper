$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first' }
try {
    $soundMapStatus = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 2
    if ($soundMapStatus.configuration_key) {
        Write-Output 'SoundMap is already running: http://127.0.0.1:8765'
        return
    }
} catch { }
Write-Output 'SoundMap: http://127.0.0.1:8765 (Ctrl+C to stop)'
& '.\.venv\Scripts\python.exe' -m soundmap.serve
