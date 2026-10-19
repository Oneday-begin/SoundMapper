import subprocess
import sys
import pytest
from filelock import FileLock, Timeout
from soundmap.compute import LOCK, serialized

def test_process_lock_blocks_another_writer_and_releases():
    code = "from filelock import FileLock; import sys; lock=FileLock(sys.argv[1]); lock.acquire(); print('locked',flush=True); sys.stdin.readline()"
    child = subprocess.Popen([sys.executable, '-c', code, LOCK.lock_file],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'locked'
        with pytest.raises(Timeout):
            FileLock(LOCK.lock_file).acquire(timeout=.1)
    finally:
        child.communicate('\n', timeout=10)
    with LOCK.acquire(timeout=1):
        assert serialized(lambda: 'nested')() == 'nested'
