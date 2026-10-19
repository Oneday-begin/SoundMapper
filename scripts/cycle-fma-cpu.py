"""Alternate FMA CPU allocation locally; never restart or suspend the batch."""
import argparse
import datetime as dt
import json
import time
from pathlib import Path
import psutil
from filelock import FileLock, Timeout

RUNTIME = Path(__file__).resolve().parents[1] / 'data' / 'runtime'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    job = json.loads((RUNTIME / 'fma-background.json').read_text())
    process = psutil.Process(job['pid'])
    if job['state'] != 'running' or not any('soundmap.background_fma' in a for a in process.cmdline()):
        raise RuntimeError('No verified running FMA batch')
    created = process.create_time()
    available = process.cpu_affinity()
    if args.check_only:
        print(json.dumps({'pid':process.pid, 'normal_minutes':30, 'reduced_minutes':15, 'normal_affinity':available,
                          'reduced_affinity':[available[-1]]}))
        return
    lock = FileLock(str(RUNTIME / 'cpu-cycle.lock'))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise SystemExit('CPU cycling is already active')
    cancel = RUNTIME / 'cpu-cycle.cancel'
    cancel.unlink(missing_ok=True)
    (RUNTIME / 'cpu-cycle.pid').write_text(str(psutil.Process().pid))
    mode = 'normal'
    next_switch = time.time()
    state = {}
    try:
        while not cancel.exists():
            try:
                job = json.loads((RUNTIME / 'fma-background.json').read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(5)
                continue
            if job['pid'] != process.pid or job['state'] != 'running':
                break
            if not process.is_running() or process.create_time() != created:
                break
            if time.time() >= next_switch:
                process.cpu_affinity(available if mode == 'normal' else [available[-1]])
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if mode == 'normal' else psutil.IDLE_PRIORITY_CLASS)
                duration_minutes = 30 if mode == 'normal' else 15
                next_switch = time.time() + duration_minutes * 60
                state = {'state':'active', 'batch_pid':process.pid, 'mode':mode,
                         'next_switch':dt.datetime.fromtimestamp(next_switch,dt.timezone.utc).astimezone().isoformat(),
                         'normal_minutes':30, 'reduced_minutes':15, 'phase_minutes':duration_minutes}
                (RUNTIME / 'cpu-cycle.json').write_text(json.dumps(state,indent=2))
                with (RUNTIME / 'cpu-cycle.log').open('a',encoding='utf-8') as log:
                    log.write(dt.datetime.now().astimezone().isoformat()+' '+mode+'; next switch '+state['next_switch']+'\n')
                mode = 'reduced' if mode == 'normal' else 'normal'
            time.sleep(5)
    except psutil.NoSuchProcess:
        pass
    finally:
        if cancel.exists():
            try:
                if process.create_time() == created:
                    process.cpu_affinity(available)
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except psutil.Error:
                pass
        state['state'] = 'cancelled' if cancel.exists() else 'batch-ended'
        (RUNTIME / 'cpu-cycle.json').write_text(json.dumps(state,indent=2))
        (RUNTIME / 'cpu-cycle.pid').unlink(missing_ok=True)
        lock.release()

if __name__ == '__main__':
    main()
