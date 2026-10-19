import queue
import threading
import uuid
from . import store
from .pipeline import analyze

_queue=queue.Queue(maxsize=32)
_lock=threading.Lock()
_started=False

def worker():
    while True:
        job,id_,kind=_queue.get()
        try:
            def progress(p,message):
                with store.connect() as db: db.execute('UPDATE jobs SET status=?,progress=?,message=? WHERE id=?',('running',p,message,job))
            if kind=='codec':
                from .codec import analyze as codec_analyze
                codec_analyze(id_,progress)
            elif kind=='dimensions':
                from .dimensions import analyze as measure
                measure(id_,progress,deep=True)
            elif kind=='tagging':
                from .tagger import analyze as tag
                tag(id_,progress)
            elif kind=='ranking':
                import json
                from .experiments import ranking
                from .config import ROOT
                progress(.1,'Checking genuine listener ratings and disjoint splits')
                result=ranking()
                (ROOT/'reports'/'ranking.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                progress(1,'Listener model '+result['status']+': '+result.get('reason','evaluation saved'))
            else:
                analyze(id_,progress)
                from .dimensions import analyze as measure
                measure(id_,progress,deep=False)
            with store.connect() as db: db.execute('UPDATE jobs SET status=?,progress=1 WHERE id=?',('complete',job))
        except Exception as e:
            with store.connect() as db: db.execute('UPDATE jobs SET status=?,message=? WHERE id=?',('failed',str(e),job))
        finally: _queue.task_done()

def start():
    global _started
    with _lock:
        if _started: return
        _started=True
        with store.connect() as db:
            # Interrupted windows remain on disk. Restarted jobs explicitly invite resuming.
            db.execute("UPDATE jobs SET status='interrupted',message='Server restarted; retry resumes checkpoints' WHERE status IN ('running','queued')")
        threading.Thread(target=worker,daemon=True,name='soundmap-analysis').start()

def submit(id_,kind='analysis'):
    if kind not in ('analysis','tagging','ranking','dimensions','codec'):raise ValueError('Unknown job kind')
    if id_ is not None:store.track(id_)
    elif kind!='ranking':raise ValueError('Track ID required')
    with _lock:
        with store.connect() as db:
            prior=db.execute("SELECT id FROM jobs WHERE track_id IS ? AND kind=? AND status IN ('queued','running')",(id_,kind)).fetchone()
            if prior: return prior[0]
            if _queue.full(): raise RuntimeError('Analysis queue full; try again shortly')
            job=uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,track_id,status,message,kind) VALUES(?,?,?,?,?)',(job,id_,'queued','Waiting for CPU worker',kind))
        _queue.put_nowait((job,id_,kind))
        return job
