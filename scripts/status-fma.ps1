$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$database = Join-Path $projectRoot 'data\soundmap.sqlite'
if (-not (Test-Path -LiteralPath $database)) { throw 'SoundMap database not found. Run the local app or FMA launcher first.' }
$query = @"
import sqlite3
c=sqlite3.connect('file:data/soundmap.sqlite?mode=ro', uri=True)
tracks=c.execute(\"SELECT COUNT(*) FROM tracks WHERE source='fma'\").fetchone()[0]
analyses=c.execute(\"SELECT COUNT(DISTINCT a.track_id) FROM analyses a JOIN tracks t ON t.id=a.track_id WHERE t.source='fma'\").fetchone()[0]
print(f'FMA tracks imported: {tracks:,} / 8,000')
print(f'FMA tracks with cached analysis: {analyses:,} / {tracks:,}')
print('Import outcomes:')
for status,n in c.execute('SELECT status,COUNT(*) FROM import_events GROUP BY status ORDER BY status'):
    print(f'  {status}: {n:,}')
print('Skipped audio:')
for reason,n in c.execute(\"SELECT reason,COUNT(*) FROM import_events WHERE status='skipped' GROUP BY reason ORDER BY 2 DESC\"):
    print(f'  {n:,} — {reason}')
"@
Push-Location $projectRoot
try {
    & $python -c $query
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
