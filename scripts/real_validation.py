"""Actual seven-song regression checks, checkpoint reuse, and evidence audit."""
import copy
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
import httpx
import numpy as np
from soundmap import store,encoder
from soundmap.audio import decode
from soundmap.pipeline import analyze,cached,KEY
from soundmap.config import ROOT,STATE
from soundmap.compare import cosine

rows=[t for t in store.tracks() if t['source']=='sample']
assert len(rows)==7
results=[];pair_results=[]
with httpx.Client(base_url='http://127.0.0.1:8765',timeout=60) as client:
    ratings_before=client.get('/api/status').json()['rating_count']
    for t in rows:
        start=time.perf_counter();a=analyze(t['id']);cache_seconds=time.perf_counter()-start
        assert a['cache_hit'] and a['mert'] and np.isfinite(a['mert']['vector']).all()
        assert len(a['mert']['vector'])==768
        assert store.digest(t['path'])==t['hash']
        for w in a['windows']:assert 0<=w['start']<w['end']<=a['duration']
        n=client.get('/api/tracks/'+t['id']+'/neighbors').json()
        assert 1<=len(n)<=10 and t['id'] not in [r['track']['id'] for r in n]
        assert all(store.track(r['track']['id'])['hash']!=t['hash'] and cached(r['track']['id'])['decoded_hash']!=a['decoded_hash'] for r in n)
        audio=client.get('/api/tracks/'+t['id']+'/audio',headers={'Range':'bytes=0-63'})
        assert audio.status_code==206 and len(audio.content)==64
        self_=client.post('/api/compare',params={'generation':False},json={'a':t['id'],'b':t['id']}).json()
        assert abs(self_['overall']-100)<.001
        results.append({'title':t['title'],'finite_embedding':True,'all_timestamps_valid':True,
                        'original_sha256_unchanged':True,'neighbors_exclude_self':True,
                        'identical_index':self_['overall'],'range_audio_status':audio.status_code,'cache_seconds':cache_seconds})
    import itertools
    for a,b in itertools.combinations(rows,2):
        r=client.post('/api/compare',params={'generation':False},json={'a':a['id'],'b':b['id']}).json()
        e=r['explanation'];cited=set(re.findall(r'\[([a-z]+)\]',' '.join(e['measurements']+e['predictions']+e['concepts'])))
        assert cited<={ref['id'] for ref in e['references']}
        assert not e['generation']['enabled']
        for p in r['passages']:
            assert p['a']['end']<=cached(a['id'])['duration'] and p['b']['end']<=cached(b['id'])['duration']
        pair_results.append({'a':a['title'],'b':b['title'],'all_citations_resolve':True,'passage_bounds_valid':True})
    t=rows[0]
    with open(t['path'],'rb') as f:
        uploaded=client.post('/api/upload',files={'file':(Path(t['path']).name,f,'audio/mpeg')}).json()
    assert uploaded['duplicate'] and uploaded['track']['id']==t['id']
    assert client.get('/api/status').json()['rating_count']==ratings_before

t=rows[0];y,_=decode(t['path']);windows=copy.deepcopy(cached(t['id'])['windows'][:2])
with tempfile.TemporaryDirectory(prefix='soundmap-real-resume-') as directory:
    p=Path(directory)
    shutil.copy2(STATE/'cache'/t['hash']/KEY/'window-00000.npz',p/'window-00000.npz')
    first=encoder.extract(y,windows,p,lambda p,m:None)
    second=encoder.extract(y,windows,p,lambda p,m:None)
    assert first['reused_windows']==1 and second['reused_windows']==2
    assert np.max(np.abs(np.asarray(first['vector'])-second['vector']))==0
layers={}
for layer in ('4','8','12'):
    scores=[100*(cosine(cached(a['id'])['mert']['layers'][layer],cached(b['id'])['mert']['layers'][layer])+1)/2 for a,b in itertools.combinations(rows,2)]
    layers[layer]={'min':min(scores),'max':max(scores),'mean':float(np.mean(scores)),'std':float(np.std(scores))}
report={'status':'passed','samples':results,'pair_evidence_audits':pair_results,'actual_duplicate_upload':True,
        'actual_mert_checkpoint_resume':{'first_reused_windows':1,'second_reused_windows':2,'repeat_max_abs_difference':0},
        'layer_score_distributions':layers,'layer_selection':'Layer 12 unchanged. Distributions are descriptive, not validated quality comparisons.',
        'human_ratings_created':0,'ratings_in_database':ratings_before,'reference_support_scope':'Template measurements match returned values and every reference id resolves; no human support-quality ratings.'}
(ROOT/'reports'/'real-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
