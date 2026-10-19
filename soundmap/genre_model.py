"""Frozen-MERT genre classifier and explicitly weak genre-affinity proxy."""
import json
import time
from functools import lru_cache
import numpy as np
from .config import STATE, ROOT, CONFIG
from .pipeline import KEY

ARTIFACT = STATE / 'genre-model.json'
NOTE = ('Genre affinity compares predicted FMA genre profiles. It is not listener similarity, '
        'instrument recognition, or a calibrated musical-similarity probability. '
        'FMA coverage may not transfer to Hindi film songs.')

def vector(analysis, representation):
    layers = analysis['mert']['layers']
    return np.mean([layers[k] for k in ('4','8','12')], axis=0) if representation == 'mean_4_8_12' else np.asarray(layers[representation])

@lru_cache(maxsize=2)
def load(stamp):
    return json.loads(ARTIFACT.read_text(encoding='utf-8'))

def profile(analysis, model):
    x = (vector(analysis, model['representation']) - model['mean']) / model['scale']
    logits = np.asarray(model['coefficient']) @ x + model['intercept']
    logits -= np.max(logits)
    p = np.exp(logits)
    return p / p.sum()

def affinity(a, b):
    if not ARTIFACT.exists(): return {'available':False,'reason':'Genre classifier artifact has not been trained.'}
    model = load(ARTIFACT.stat().st_mtime_ns)
    if model['analysis_key'] != KEY: return {'available':False,'reason':'Model configuration differs; retrain.'}
    if not a.get('mert') or not b.get('mert'): return {'available':False,'reason':'Both recordings need MERT embeddings.'}
    pa, pb = profile(a, model), profile(b, model)
    def top(p):
        return [{'genre':model['classes'][i], 'activation':float(p[i])} for i in np.argsort(-p)[:3]]
    return {'available':True,'index':100*float(np.sum(np.sqrt(pa*pb))),
            'method':'Bhattacharyya coefficient of learned genre profiles',
            'a':top(pa),'b':top(pb),'note':NOTE,'test':model['test'],
            'representation':model['representation'],'target':'FMA genre labels; no human similarity targets'}

def _train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    from threadpoolctl import threadpool_limits
    from . import store
    from .pipeline import MemoryMeter
    report = json.loads((ROOT/'reports/genre.json').read_text())
    chosen = report['selected_on_validation']
    # Read only pooled vectors, avoiding gigabytes of waveform/window JSON.
    with store.connect() as db:
        raw=db.execute("SELECT t.id,t.hash,t.metadata,json_extract(a.result,'$.mert.layers') AS layers,json_extract(a.result,'$.decoded_hash') AS decoded_hash FROM tracks t JOIN analyses a ON a.track_id=t.id AND a.config_key=? WHERE t.source='fma' AND json_extract(a.result,'$.mert.layers') IS NOT NULL",(KEY,)).fetchall()
    rows=[{'id':r['id'],'hash':r['hash'],'metadata':json.loads(r['metadata']),
           '_analysis':{'mert':{'layers':json.loads(r['layers'])},'decoded_hash':r['decoded_hash']}} for r in raw]
    del raw
    order={'training':0,'validation':1,'test':2}
    groups={}
    for t in rows: groups.setdefault(t['metadata']['artist_id'],set()).add(t['metadata']['split'])
    rows=[t for t in rows if order[t['metadata']['split']]==max(order[s] for s in groups[t['metadata']['artist_id']])]
    # Duplicate content cannot cross splits, even if an artist attribution differs.
    hashes={}
    for t in rows:
        h=t['_analysis'].get('decoded_hash',t['hash'])
        hashes.setdefault(h,set()).add(t['metadata']['split'])
    rows=[t for t in rows if len(hashes[t['_analysis'].get('decoded_hash',t['hash'])])==1]
    split={s:[t for t in rows if t['metadata']['split']==s] for s in order}
    if any(not ts for ts in split.values()): raise ValueError('All three supplied splits are required')
    X={s:np.stack([vector(t['_analysis'],chosen['representation']) for t in ts]) for s,ts in split.items()}
    y={s:[t['metadata']['genre'] for t in ts] for s,ts in split.items()}
    start=time.perf_counter()
    with MemoryMeter() as meter, threadpool_limits(limits=2):
        scaler=StandardScaler().fit(X['training'])
        model=LogisticRegression(C=chosen['C'], max_iter=2000,class_weight='balanced',random_state=2026)
        model.fit(scaler.transform(X['training']),y['training'])
        pred=model.predict(scaler.transform(X['test']))
    metrics={'accuracy':accuracy_score(y['test'],pred),'balanced_accuracy':balanced_accuracy_score(y['test'],pred),
             'macro_f1':f1_score(y['test'],pred,average='macro',zero_division=0)}
    artifact={'version':1,'analysis_key':KEY,'revision':CONFIG['model_revision'],
              'representation':chosen['representation'],'C':chosen['C'], 'classes':model.classes_.tolist(),
              'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
              'coefficient':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'test':metrics,
              'target':'FMA genre labels','human_similarity_trained':False}
    temp=ARTIFACT.with_suffix('.tmp');temp.write_text(json.dumps(artifact));temp.replace(ARTIFACT)
    result={'status':'trained','task':'Auxiliary genre classifier and weak genre-affinity proxy','note':NOTE,
            'selected_from_existing_validation':chosen,'test':metrics,'split_counts':{s:len(ts) for s,ts in split.items()},
            'artist_overlap':{k:sorted(v) for k,v in groups.items() if len(v)>1},
            'cross_split_duplicate_hashes':sum(len(v)>1 for v in hashes.values()),
            'confusion_matrix':confusion_matrix(y['test'],pred,labels=model.classes_).tolist(),'classes':model.classes_.tolist(),
            'processing_seconds':time.perf_counter()-start,'peak_rss_mb':meter.peak/1024**2,
            'human_similarity_trained':False,'encoder_finetuned':False,
            'controlled_transform_report':'reports/controlled-transforms.json when run separately; no recommendation-quality claim.'}
    (ROOT/'reports/genre-model.json').write_text(json.dumps(result,indent=2))
    return result

def train():
    from .pipeline import MemoryMeter
    start=time.perf_counter()
    with MemoryMeter() as meter:
        result=_train()
    result['classifier_fit_and_test_seconds']=result.pop('processing_seconds')
    result['total_seconds_including_data_preparation']=time.perf_counter()-start
    result['peak_rss_mb']=meter.peak/1024**2
    result['memory_note']='Sampled whole-process RSS every 50 ms, including data preparation; not incremental model memory.'
    (ROOT/'reports/genre-model.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__': print(json.dumps(train(),indent=2))
