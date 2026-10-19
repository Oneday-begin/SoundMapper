"""Coordinate CPU/cache writers across the website and offline batch."""
import os
import time
from functools import wraps
from filelock import FileLock
from .config import STATE

RUNTIME = STATE / 'runtime'
RUNTIME.mkdir(parents=True, exist_ok=True)
LOCK = FileLock(str(RUNTIME / 'compute.lock'))

def serialized(function):
    @wraps(function)
    def call(*args, **kwargs):
        if os.environ.get('SOUNDMAP_BATCH') == '1' and not LOCK.is_locked:
            from . import store
            import psutil
            # Yield only to a live web server; stale queued rows cannot stall a batch.
            while True:
                try:
                    pid = int((RUNTIME / 'server.pid').read_text().strip())
                    server_live = psutil.pid_exists(pid)
                except (OSError, ValueError):
                    server_live = False
                with store.connect() as db:
                    waiting = db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
                if not server_live or not waiting:
                    break
                time.sleep(.5)
        with LOCK:
            return function(*args, **kwargs)
    return call
