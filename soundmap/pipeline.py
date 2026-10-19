import hashlib
import json
import time
import threading
import psutil
from . import store, encoder
from .config import CONFIG, STATE
from .audio import decode, features
from .compute import serialized

ANALYSIS_CONFIG = {'version':'soundmap-1.0', 'sr':24000, 'hop':512, 'fft':2048,
                   'window_seconds':CONFIG['window_seconds'], 'mert':CONFIG['mert_enabled'],
                   'revision':CONFIG['model_revision'], 'layers':[4,8,12], 'pooling':'duration-weighted-frame-mean',
                   'weight_loader':'strict-weight-norm-key-map-v1'}
KEY = hashlib.sha256(json.dumps(ANALYSIS_CONFIG,sort_keys=True).encode()).hexdigest()[:20]

class MemoryMeter:
    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.05): self.peak=max(self.peak,psutil.Process().memory_info().rss)
        self.thread = threading.Thread(target=sample,daemon=True); self.thread.start()
        return self
    def __exit__(self,*args): self.stop.set(); self.thread.join()

def cached(id_):
    with store.connect() as db:
        row=db.execute('SELECT result FROM analyses WHERE track_id=? AND config_key=?',(id_,KEY)).fetchone()
    return json.loads(row[0]) if row else None

@serialized
def analyze(id_, progress=lambda p,m:None, force=False):
    t = store.track(id_)
    existing = cached(id_)
    if existing and not force and (existing['silent'] or existing.get('mert') or not CONFIG['mert_enabled']):
        progress(1,'Cache reused')
        return {**existing,'cache_hit':True}
    start=time.perf_counter()
    with MemoryMeter() as meter:
        progress(.01,'Decoding and resampling')
        y, info=decode(t['path'])
        result=features(y) if not existing else {k:v for k,v in existing.items() if k not in ('mert','measurements','mert_error')}
        result.update(info)
        for w in result['windows']: w['end']=min(w['end'],info['duration'])
        result['decoded_hash']=hashlib.sha256(y.tobytes()).hexdigest()
        progress(.2,'Acoustic analysis complete')
        result['mert']=None
        if CONFIG['mert_enabled'] and not result['silent']:
            try: result['mert']=encoder.extract(y,result['windows'],STATE/'cache'/t['hash']/KEY,progress)
            except Exception as e: result['mert_error']=str(e)
        result['model_predictions']={'instruments':None,'reason':'MERT does not identify instruments; separate AST annotations are joined by the analysis API.'}
        result['structure']={'labels':None,'reason':'Verse/chorus detection has not been validated'}
        result['analysis_config']=ANALYSIS_CONFIG
    result['measurements']={'processing_seconds':time.perf_counter()-start,'peak_rss_mb':meter.peak/1024**2,
                             'rss_sampling_interval_seconds':.05,'cache_hit':False}
    result['cache_hit']=False
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO analyses VALUES(?,?,?)',(id_,KEY,json.dumps(result,allow_nan=False)))
    progress(1,'Complete' if not result.get('mert_error') else 'Acoustic ready; MERT unavailable')
    return result
