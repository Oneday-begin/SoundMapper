"""Actual CPU tagger, local Ollama, cached library and API controls; no listener labels created."""
import json
import time
import numpy as np
import librosa
import httpx
from unittest.mock import patch
from soundmap import store,tagger
from soundmap.config import ROOT,CONFIG
from soundmap.audio import decode
from soundmap.pipeline import cached,MemoryMeter
from soundmap.compare import compare
from soundmap.explain import explain

rows=[t for t in store.tracks() if t['source']=='sample'];checks=[]
for t in rows:
    a=tagger.cached(t['id']);assert a and a['state']=='ready'
    assert store.digest(t['path'])==t['hash']
    assert all(0<=w['start']<w['end']<=a['source_duration'] for w in a['sampled_windows'])
    assert all(np.isfinite(p['score']) and 0<=p['score']<=1 for p in a['labels']+a['passage_predictions']+a['low_score_candidates'])
    assert tagger.analyze(t['id'])['cache_hit']
    checks.append({'title':t['title'],'windows':a['windows'],'whole_excerpt_mean_tags':len(a['labels']),
                   'local_suggestions':a['passage_predictions'],'timestamps_valid':True,'hash_preserved':True,'cache_reused':True})
y,_=decode(rows[0]['path']);y=librosa.resample(y[:240000],orig_sr=24000,target_sr=16000,res_type='soxr_hq')
with MemoryMeter() as memory:
    start=time.perf_counter();first=tagger.infer(y);load_seconds=time.perf_counter()-start
    start=time.perf_counter();second=tagger.infer(y);repeat_seconds=time.perf_counter()-start
assert np.isfinite(first).all() and np.max(np.abs(first-second))==0
r=compare(cached(rows[0]['id']),cached(rows[1]['id']))
r.update(instrument_a=tagger.cached(rows[0]['id']),instrument_b=tagger.cached(rows[1]['id']))
out=explain(r);generation=out['generation'];assert generation['enabled'],generation
assert all(f['text'] in out[f['kind']] for f in generation['selected_evidence'])
start=time.perf_counter();again=explain(r);cache_seconds=time.perf_counter()-start
assert again['generation']['cache_hit'] and again['generation']['text']==generation['text']
with patch.dict(CONFIG,{'ollama_model':'soundmap-deliberately-absent-local-model'}):
    fallback=explain(r)
assert not fallback['generation']['enabled'] and 'fallback' in fallback['generation']['reason']
with httpx.Client(base_url='http://127.0.0.1:8765',timeout=120) as client:
    before=client.get('/api/status').json()
    assert before['tracks']==107 and before['instrument_tagging']['cached_tracks']==7
    for t in rows:
        api=client.get('/api/tracks/'+t['id']+'/analysis').json()
        assert api['instrument_predictions']['state']=='ready' and api['mert']
        assert api['model_predictions']['instruments']==api['instrument_predictions']
        recurrence=api['recurrence']
        assert len(recurrence['matrix'])==recurrence['sampled_windows'] and recurrence['sampled_windows']<=160
        assert all(0<=p['a']['start']<p['a']['end']<=api['duration'] and
                   0<=p['b']['start']<p['b']['end']<=api['duration'] for p in recurrence['candidates'])
    comparison=client.post('/api/compare',params={'generation':False},json={'a':rows[0]['id'],'b':rows[1]['id']}).json()
    assert comparison['instrument_a']['state']=='ready'
    assert any('[ast]' in p for p in comparison['explanation']['predictions'])
    assert not comparison['explanation']['generation']['enabled']
    assert client.get('/api/evaluation/status').json()['training_optional']
    assert client.post('/api/evaluation/train').status_code==409
    pair=client.get('/api/ratings/next',params={'mode':'training','session':'read-only-infrastructure-check'}).json()
    assert pair['collection_mode']=='training' and pair['split'] in ('training','validation','test')
    job=client.post('/api/tracks/'+rows[0]['id']+'/instruments').json()['job']
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        finished=next(j for j in client.get('/api/jobs').json() if j['id']==job)
        if finished['status'] in ('complete','failed'):break
        time.sleep(.2)
    assert finished['status']=='complete' and 'cache reused' in finished['message'].lower()
    assert client.get('/api/status').json()['rating_count']==before['rating_count']
report={'status':'passed','date':'2026-10-09','sample_tags':checks,
        'tagger_cpu_verification':{'finite':True,'shape':list(first.shape),'repeat_max_abs_difference':float(np.max(np.abs(first-second))),
           'first_seconds_including_load':load_seconds,'repeat_seconds':repeat_seconds,'peak_process_rss_mb':memory.peak/1024**2,'status':tagger.status},
        'ollama':generation,'ollama_cached_lookup_seconds':cache_seconds,'absent_model_fallback':True,
        'api_cached_tagging_job':'complete','optional_training_api_blocks_insufficient_labels':True,
        'guided_pair_defined_before_training':True,'sample_recurrence_matrix_and_timestamps_valid':True,
        'human_ratings_created':0,'human_ratings_in_database':before['rating_count'],
        'remaining_fma_tracks_processed':0,'instrument_precision_recall':None,
        'instrument_evaluation_limit':'No instrument ground-truth annotations supplied. Scores and local hypotheses are not verified identities.',
        'human_similarity_model':'untrained; optional training deferred by user'}
(ROOT/'reports'/'extensions-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='sample_tags'},indent=2))
