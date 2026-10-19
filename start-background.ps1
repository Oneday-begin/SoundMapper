$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$runtime = Join-Path $PSScriptRoot 'data\runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$legacy = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'run-fma-pilot.ps1' }
if ($legacy) { throw 'The older exclusive FMA launcher is active. Stop that run before starting background mode; checkpoints are preserved.' }
$web = $null
try { $web = Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 2 } catch { }
if (-not $web) {
    $server = Start-Process -FilePath $python -ArgumentList @('-m','soundmap.serve') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'server.stdout.log') -RedirectStandardError (Join-Path $runtime 'server.stderr.log') -PassThru
    Set-Content -LiteralPath (Join-Path $runtime 'server.pid') -Value $server.Id
}
$batch = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match ' -m soundmap.background_fma' }
if (-not $batch) {
    Start-Process -FilePath $python -ArgumentList @('-u','-m','soundmap.background_fma') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'fma-background.stdout.log') -RedirectStandardError (Join-Path $runtime 'fma-background.stderr.log') | Out-Null
}
Write-Host 'Website: http://127.0.0.1:8765'
Write-Host 'FMA resumes in a separate background process with two CPU threads and below-normal priority.'
Write-Host 'Check progress with scripts\status-fma.ps1. Keep the computer awake and plugged in.'
