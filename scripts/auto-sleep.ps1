param([switch]$Cancel, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $projectRoot 'data\runtime'
$cancelPath = Join-Path $runtime 'auto-sleep.cancel'
$statusPath = Join-Path $runtime 'fma-background.json'
$logPath = Join-Path $runtime 'auto-sleep.log'
if ($Cancel) {
    Set-Content -LiteralPath $cancelPath -Value 'cancelled'
    Write-Output 'Automatic sleep cancelled. FMA processing continues.'
    return
}
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class SoundMapPower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
    [DllImport("powrprof.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool SetSuspendState(bool hibernate, bool forceCritical, bool disableWakeEvent);
}
'@
function Read-Batch {
    try { return Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json } catch { return $null }
}
function Write-SleepLog([string]$message) {
    Add-Content -LiteralPath $logPath -Value ((Get-Date -Format o) + ' ' + $message) -Encoding utf8
}
$initial = Read-Batch
if (-not $initial -or $initial.state -ne 'running') { throw 'No running FMA batch found; automatic sleep was not armed.' }
$batchProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($initial.pid)"
if (-not $batchProcess -or $batchProcess.CommandLine -notmatch 'soundmap.background_fma') { throw 'FMA process is not alive; automatic sleep was not armed.' }
if ($CheckOnly) { Write-Output 'Verified running FMA process and Windows power API. No power action taken.'; return }
$watcherMutex = [System.Threading.Mutex]::new($false,'Local\SoundMapAutoSleep')
$held = $false
try {
    $held = $watcherMutex.WaitOne(0)
    if (-not $held) { throw 'Automatic sleep watcher is already running.' }
    Remove-Item -LiteralPath $cancelPath -ErrorAction SilentlyContinue
    Set-Content -LiteralPath (Join-Path $runtime 'auto-sleep.pid') -Value $PID
    # Prevent automatic idle sleep during this run, without changing Windows power settings.
    if ([SoundMapPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Could not request the computer stay awake.' }
    Write-SleepLog "Armed for FMA process $($initial.pid). Waiting for analyze, dimensions, and genre success."
    while (-not (Test-Path -LiteralPath $cancelPath)) {
        $batch = Read-Batch
        if (-not $batch) { Start-Sleep -Seconds 5; continue }
        if ($batch.pid -ne $initial.pid) { Write-SleepLog 'Batch identity changed; disarmed.'; break }
        if ($batch.state -eq 'complete' -and @($batch.failures).Count -eq 0) {
            Write-SleepLog 'All batch stages succeeded. Sleep countdown: 120 seconds.'
            try { (New-Object -ComObject WScript.Shell).Popup('SoundMap FMA processing finished successfully. The PC will sleep in two minutes. Run scripts\auto-sleep.ps1 -Cancel to cancel.',5,'SoundMap automatic sleep',64) | Out-Null } catch { }
            for ($second=0; $second -lt 120; $second++) {
                if (Test-Path -LiteralPath $cancelPath) { break }
                Start-Sleep -Seconds 1
            }
            $final = Read-Batch
            if ((Test-Path -LiteralPath $cancelPath) -or -not $final -or $final.state -ne 'complete' -or @($final.failures).Count -ne 0) { Write-SleepLog 'Sleep cancelled or completion state changed.'; break }
            # Website work also has to finish before sleeping.
            $query = @'
import sqlite3
c=sqlite3.connect('file:data/soundmap.sqlite?mode=ro',uri=True)
print(c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0])
'@
            Push-Location $projectRoot
            try { $busy = & (Join-Path $projectRoot '.venv\Scripts\python.exe') -c $query } finally { Pop-Location }
            if ($LASTEXITCODE -ne 0 -or [int]$busy -ne 0) { Write-SleepLog 'Website jobs are pending; waiting before another countdown.'; Start-Sleep -Seconds 30; continue }
            Write-SleepLog 'Requesting ordinary sleep, without forcing applications to close.'
            [SoundMapPower]::SetThreadExecutionState([uint32]2147483648) | Out-Null
            if (-not [SoundMapPower]::SetSuspendState($false,$false,$false)) { Write-SleepLog 'Windows rejected the sleep request.' }
            break
        }
        if ($batch.state -ne 'running') { Write-SleepLog "Batch ended with state $($batch.state); no sleep requested."; break }
        $live = Get-CimInstance Win32_Process -Filter "ProcessId=$($initial.pid)"
        if (-not $live -or $live.CommandLine -notmatch 'soundmap.background_fma') { Write-SleepLog 'Batch stopped unexpectedly; no sleep requested.'; break }
        Start-Sleep -Seconds 15
    }
} finally {
    [SoundMapPower]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    if ($held) {
        Remove-Item -LiteralPath (Join-Path $runtime 'auto-sleep.pid') -ErrorAction SilentlyContinue
        $watcherMutex.ReleaseMutex()
        Write-SleepLog 'Watcher ended.'
    }
    $watcherMutex.Dispose()
}
