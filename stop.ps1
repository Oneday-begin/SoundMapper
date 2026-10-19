$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath 'data\runtime\server.pid')) { throw 'No managed background server recorded. Stop an interactive launch with Ctrl+C.' }
$soundMapManagedId = [int](Get-Content -LiteralPath 'data\runtime\server.pid')
$soundMapParent = Get-CimInstance Win32_Process -Filter "ProcessId = $soundMapManagedId"
if (-not $soundMapParent) { Write-Output 'Managed server already stopped.'; return }
if (-not $soundMapParent.CommandLine.Contains('soundmap.serve')) { throw 'Recorded PID is not the SoundMap server; no process stopped.' }
$soundMapListener = Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $soundMapListener) { throw 'SoundMap is not listening on its managed local port; no process stopped.' }
if ($soundMapListener.OwningProcess -ne $soundMapManagedId) {
    $soundMapListenerProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($soundMapListener.OwningProcess)"
    if (-not $soundMapListenerProcess -or $soundMapListenerProcess.ParentProcessId -ne $soundMapManagedId -or -not $soundMapListenerProcess.CommandLine.Contains('soundmap.serve')) {
        throw 'Recorded process does not own the SoundMap local port; no process stopped.'
    }
}
$soundMapChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $soundMapManagedId"
foreach ($soundMapChild in $soundMapChildren) {
    if ($soundMapChild.CommandLine.Contains('soundmap.serve')) { Stop-Process -Id $soundMapChild.ProcessId -ErrorAction SilentlyContinue }
}
Stop-Process -Id $soundMapManagedId -ErrorAction SilentlyContinue
Write-Output 'SoundMap server stopped. Audio and data preserved.'
