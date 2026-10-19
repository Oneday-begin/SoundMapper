"""Reproducible real-audio perturbation checks; no human-quality targets."""
import json,time
import numpy as np
from . import store,encoder
from .audio import decode,unit
from .config import CONFIG,ROOT
from .pipeline import MemoryMeter
from .genre_model import affinity

def run():
    CONFIG['threads']=2
    t=next(t for t in store.tracks() if t['source']=='sample' and 'Parvati' in t['title'])
    y,_=decode(t['path']);y=y[:120000]
    def analysis(v):return {'mert':{'layers':{str(k):unit(v[i]).tolist() for i,k in enumerate(encoder.LAYERS)}}}
    start=time.perf_counter()
    with MemoryMeter() as meter:
        base=encoder.embed(y);gain=encoder.embed(y*.5);reverse=encoder.embed(y[::-1].copy())
    r={'source':t['title'],'source_timestamp':[0,len(y)/24000],
       'purpose':'Controlled five-second encoder sanity checks, not recommendation-quality labels or similarity training',
       'half_gain':{'embedding_max_abs_difference':float(np.max(np.abs(base-gain))),
                    'genre_affinity':affinity(analysis(base),analysis(gain))['index']},
       'reversed_audio':{'genre_affinity':affinity(analysis(base),analysis(reverse))['index'],
                         'note':'Chronology changed; no assumed human-similarity target'},
       'processing_seconds_including_model_load':time.perf_counter()-start,'peak_rss_mb':meter.peak/1024**2}
    (ROOT/'reports/controlled-transforms.json').write_text(json.dumps(r,indent=2))
    return r

if __name__=='__main__':print(json.dumps(run(),indent=2))
