"""Resume registered FMA analyses while the web server remains available."""
import os
os.environ['SOUNDMAP_BATCH'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
import json
import datetime
import psutil
from .config import CONFIG
CONFIG['threads'] = 2
from .compute import RUNTIME
from filelock import FileLock, Timeout

def main():
    lock = FileLock(str(RUNTIME / 'fma-background.lock'))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise SystemExit('FMA background processing is already running')
    try:
        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
        (RUNTIME / 'fma-background.pid').write_text(str(process.pid))
        from .cli import main as cli
        failures = []
        for stage in ('analyze', 'dimensions', 'genre'):
            status = {'pid': process.pid, 'stage': stage, 'state': 'running',
                      'updated': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (RUNTIME / 'fma-background.json').write_text(json.dumps(status))
            print('Starting stage:', stage, flush=True)
            sys.argv = ['soundmap', stage] + (['--source','fma'] if stage != 'genre' else [])
            try:
                cli()
            except SystemExit as error:
                if error.code:
                    failures.append({'stage': stage, 'error': str(error)})
            except Exception as error:
                failures.append({'stage': stage, 'error': str(error)})
        status.update(state='completed_with_errors' if failures else 'complete', failures=failures)
        (RUNTIME / 'fma-background.json').write_text(json.dumps(status))
    finally:
        (RUNTIME / 'fma-background.pid').unlink(missing_ok=True)
        lock.release()

if __name__ == '__main__':
    main()
