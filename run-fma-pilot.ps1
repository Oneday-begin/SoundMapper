param([int]$PilotSize = 100, [switch]$Expand, [switch]$Online, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
if ($PilotSize -lt 0) { throw 'PilotSize must be zero (all tracks) or a positive count.' }
if ($Expand) { $PilotSize = 0 }
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Python environment missing. Run setup.ps1 first.' }
$env:PYTHONIOENCODING = 'utf-8'
if (-not $Online) {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
} else {
    $env:HF_HUB_OFFLINE = '0'
    $env:TRANSFORMERS_OFFLINE = '0'
}
New-Item -ItemType Directory -Force -Path 'reports' | Out-Null
$soundMapRuntime = Join-Path $PSScriptRoot 'data\runtime'
New-Item -ItemType Directory -Force -Path $soundMapRuntime | Out-Null
function Invoke-SoundMapStep {
    param([string[]]$Arguments)
    $soundMapLabel = ($Arguments -join ' ')
    $soundMapSafeLabel = ($soundMapLabel -replace '[^A-Za-z0-9_-]+','-').Trim('-')
    $soundMapOut = Join-Path $soundMapRuntime ($soundMapSafeLabel + '.stdout.log')
    Write-Host "Starting: $soundMapLabel"
    $soundMapPreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $python -X utf8 @Arguments 2>&1 | Out-File -LiteralPath $soundMapOut -Append -Encoding utf8
        $soundMapStepExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $soundMapPreviousPreference }
    if ($soundMapStepExit -ne 0) { throw "Step failed (exit $soundMapStepExit). Completed windows are saved; rerun this launcher to resume. Logs are in data\runtime." }
    Write-Host "Finished: $soundMapLabel"
}

$soundMapMutex = [System.Threading.Mutex]::new($false,'Local\SoundMapFmaPilot')
$soundMapLock = $false
$soundMapStoppedServer = $false
$soundMapStartServerOnExit = $false
try {
    try { $soundMapLock = $soundMapMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $soundMapLock = $true }
    if (-not $soundMapLock) { throw 'A SoundMap FMA run is already active. Leave that window running.' }
    Add-Content -LiteralPath (Join-Path $soundMapRuntime 'fma-run.log') -Value ("`nRun started: " + (Get-Date -Format o)) -Encoding UTF8
    Write-Host 'SoundMap runs on this computer. ChatGPT and Wi-Fi can be closed.'
    Write-Host 'Keep this window open and the computer awake. Ctrl+C interrupts; rerun to resume.'
    if ($Online) { & (Join-Path $PSScriptRoot 'setup-models.ps1') }
    Invoke-SoundMapStep -Arguments @('-c',"from soundmap.encoder import load,status; load(); print('Cached CPU model ready:',status['revision'])")
    if ($CheckOnly) {
        Invoke-SoundMapStep -Arguments @('-m','soundmap.cli','metadata')
        Write-Host 'Offline model and metadata check passed. No pilot analysis or training started.'
        return
    }
    $soundMapRunning = $null
    try { $soundMapRunning = Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 2 } catch { }
    if ($soundMapRunning) {
        $soundMapJobs = @(Invoke-RestMethod 'http://127.0.0.1:8765/api/jobs' -TimeoutSec 5)
        if (@($soundMapJobs | Where-Object { $_.status -in @('running','queued') }).Count) { throw 'Wait for the app analysis queue to finish, then run this launcher again.' }
        & (Join-Path $PSScriptRoot 'stop.ps1')
        $soundMapStoppedServer = $true
    } else { $soundMapStartServerOnExit = $true }
    Invoke-SoundMapStep -Arguments @('-m','soundmap.cli','extract')
    $soundMapImportedCount = 0
    if (Test-Path -LiteralPath 'data\soundmap.sqlite') {
        $soundMapImportedCount = [int](& $python -c "import sqlite3; c=sqlite3.connect(r'data/soundmap.sqlite'); print(c.execute('SELECT COUNT(*) FROM import_events').fetchone()[0])")
    }
    if ($Expand -and $soundMapImportedCount -ge 8000) {
        Write-Host "FMA import already has $soundMapImportedCount events; skipping completed import."
    } else {
        Invoke-SoundMapStep -Arguments @('-m','soundmap.cli','fma','--limit',"$PilotSize",'--seed','2026')
    }
    Invoke-SoundMapStep -Arguments @('-m','soundmap.cli','analyze','--source','fma')
    Invoke-SoundMapStep -Arguments @('-m','soundmap.cli','dimensions','--source','fma')
    Invoke-SoundMapStep -Arguments @('-m','soundmap.cli','genre')
    Write-Host 'Finished. Read reports\analysis-fma.json, reports\dimensions-fma.json, and reports\genre.json.'
    Write-Host 'Genre classification evaluates representations, not human similarity. Frozen MERT was not fine-tuned.'
} finally {
    if ($soundMapLock) { $soundMapMutex.ReleaseMutex() }
    $soundMapMutex.Dispose()
    if ($soundMapStoppedServer -or $soundMapStartServerOnExit) {
        $soundMapServer = Start-Process -FilePath $python -ArgumentList @('-m','soundmap.serve') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $soundMapRuntime 'server.stdout.log') -RedirectStandardError (Join-Path $soundMapRuntime 'server.stderr.log') -PassThru
        Set-Content -LiteralPath (Join-Path $soundMapRuntime 'server.pid') -Value $soundMapServer.Id
        Write-Host 'SoundMap restarted at http://127.0.0.1:8765. Refresh its Library view.'
    }
}
