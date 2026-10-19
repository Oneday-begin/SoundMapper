"""Controlled fixtures only: no test labels enter the real listener database."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from soundmap import tagger,store,listener
from soundmap.config import STATE,CONFIG
from soundmap.pipeline import KEY
from soundmap.explain import generate
from soundmap.experiments import ranking

def recording(name,seconds=12,silent=False,freq=220,metadata=None):
    p=STATE/'uploads'/(name+'.wav')
    t=np.arange(16000*seconds)/16000
    sf.write(p,np.zeros(len(t)) if silent else .2*np.sin(2*np.pi*freq*t),16000)
    return store.register(p,'upload',metadata)[0]

def test_instrument_checkpoints_scores_and_silence():
    id_=recording('ast-resume');blank=recording('ast-blank',silent=True);short=recording('ast-short',seconds=1)
    model=SimpleNamespace(config=SimpleNamespace(id2label={i:('Piano' if i==0 else 'Music') for i in range(527)}))
    vector=np.zeros(527,dtype='float32');vector[0]=.7
    with patch.dict(CONFIG,{'tagging_enabled':True}),patch('soundmap.tagger.load',return_value=(model,None)),patch('soundmap.tagger.infer',return_value=vector) as inference:
        first=tagger.analyze(id_);repeat=tagger.analyze(id_,force=True)
        assert first['labels'][0]['label']=='Piano' and first['labels'][0]['score']==pytest.approx(.7)
        assert repeat['reused_windows']==1 and inference.call_count==1
        assert tagger.analyze(id_)['cache_hit']
        assert tagger.analyze(blank)['state']=='insufficient_audio'
        assert tagger.analyze(short)['state']=='insufficient_audio'
        assert inference.call_count==1
        assert first['analyzed_seconds']<=first['source_duration']
    assert all(0<=w['start']<w['end']<=first['source_duration'] for w in first['sampled_windows'])
    from soundmap.app import app
    with TestClient(app) as client:
        assert client.get('/api/tracks/'+id_+'/instruments').json()['labels'][0]['label']=='Piano'
        with patch('soundmap.jobs.submit',return_value='tag-job') as submit:
            with patch.dict(CONFIG,{'tagging_enabled':True}):
                assert client.post('/api/tracks/'+id_+'/instruments').status_code==200
                submit.assert_called_once_with(id_,'tagging')
        assert client.delete('/api/tracks/'+id_+'/cache').status_code==200
        assert tagger.cached(id_) is None

def evidence(marker):
    return {'measurements':['Calculated '+marker+' 123.4 BPM [tempo]'],
            'predictions':['Model representation [mert]'],'concepts':['General concept [tempo]'],
            'references':[{'id':'tempo','url':'https://librosa.org','text':'Reference'}]}

def test_ollama_accepted_evidence_and_cache():
    fake_tags=Mock();fake_tags.json.return_value={'models':[{'name':'test-local','digest':'test-digest'}]}
    fake_output=Mock();fake_output.json.return_value={'response':json.dumps({'measurements':['measurements:0'],'predictions':['predictions:0'],'concepts':['concepts:0']}),'eval_count':20,'total_duration':100}
    with patch.dict(CONFIG,{'ollama_model':'test-local','ollama_digest':'test-digest'}),patch('soundmap.explain.httpx.get',return_value=fake_tags),patch('soundmap.explain.httpx.post',return_value=fake_output) as call:
        r=generate(evidence('unique-pass'));second=generate(evidence('unique-pass'))
    assert r['enabled'] and second['cache_hit'] and call.call_count==1
    assert all(f['text'] in evidence('unique-pass')[f['kind']] for f in r['selected_evidence'])

def test_ollama_rejects_claims_unknown_ids_and_changed_model():
    fake_tags=Mock();fake_tags.json.return_value={'models':[{'name':'test-local','digest':'test-digest'}]}
    fake_output=Mock();fake_output.json.return_value={'response':json.dumps({'measurements':['measurements:0'],'predictions':['predictions:999'],'concepts':['concepts:0'],'biography':'invented'})}
    with patch.dict(CONFIG,{'ollama_model':'test-local','ollama_digest':'test-digest'}),patch('soundmap.explain.httpx.get',return_value=fake_tags),patch('soundmap.explain.httpx.post',return_value=fake_output):
        r=generate(evidence('unique-rejection'))
    assert not r['enabled'] and 'fallback' in r['reason'] and 'text' not in r
    with patch.dict(CONFIG,{'ollama_model':'test-local','ollama_digest':'changed-digest'}),patch('soundmap.explain.httpx.get',return_value=fake_tags),patch('soundmap.explain.httpx.post') as call:
        r=generate(evidence('unique-digest'))
    assert not r['enabled'] and 'digest' in r['reason'] and not call.called

def test_listener_pipeline_grouping_repeat_aggregation_training_and_leak_guard():
    ids=[];pairs=[];groups={s:[] for s in ('training','validation','test')}
    try:
        for s,count in [('training',16),('validation',8),('test',8)]:
            k=0
            while len(groups[s])<count:
                artist='controlled-'+s+'-'+str(k);k+=1
                if listener.split_for('artist:'+artist)!=s:continue
                i=recording('rank-'+str(len(ids)),seconds=2,freq=200+len(ids)*7,metadata={'artist_id':artist})
                ids.append(i);groups[s].append(i)
                index=len(ids);vec=np.zeros(768);vec[index]=1
                a={'silent':False,'decoded_hash':store.track(i)['hash'],'mert':{'vector':vec.tolist()},
                   'rhythm':[1,index/40,.1],'harmonic':[[1.,0.,0.],[.7,.7,0.]],'timbre':[index/20,.3],
                   'tempo':100+index,'windows':[]}
                with store.connect() as db:db.execute('INSERT INTO analyses VALUES(?,?,?)',(i,KEY,json.dumps(a)))
            pairs.extend(list(itertools.combinations(groups[s],2))[:40 if s=='training' else 10])
        from soundmap.app import app
        with TestClient(app) as client:
            pair=client.get('/api/ratings/next',params={'session':'controlled-plan','mode':'training'}).json()
            assert listener.split_for('artist:'+store.track(pair['a'])['metadata']['artist_id'])==listener.split_for('artist:'+store.track(pair['b'])['metadata']['artist_id'])
            assert client.get('/api/evaluation/status').json()['training_optional']
            assert client.post('/api/evaluation/train').status_code==409
        with store.connect() as db:
            for index,(a,b) in enumerate(pairs):
                db.execute('INSERT INTO ratings(id,a,b,rater,session,overall) VALUES(?,?,?,?,?,?)',('controlled-'+str(index),a,b,'fixture-only','fixture-only',1+index%5))
            # Repeats do not create new examples.
            db.execute('INSERT INTO ratings(id,a,b,rater,session,overall) VALUES(?,?,?,?,?,?)',('controlled-repeat',*pairs[0],'fixture-only','fixture-only',3))
        report=listener.readiness()
        assert report['ready'] and report['eligible_distinct_pairs']==60 and report['distinct_recordings']==32
        assert report['split_pair_counts']=={'training':40,'validation':10,'test':10}
        r=ranking();assert r['status']=='trained'
        with TestClient(app) as client,patch('soundmap.jobs.submit',return_value='listener-test-job') as submit:
            assert client.post('/api/evaluation/train').json()['job']=='listener-test-job'
            submit.assert_called_once_with(None,'ranking')
        assert all(np.isfinite(m['mae']) for m in r['test'].values())
        assert (STATE/'ranking-model.json').exists()
        # The same decoded recording in two different artist splits blocks leakage.
        a,b=groups['training'][0],groups['test'][0]
        with store.connect() as db:
            row=db.execute('SELECT result FROM analyses WHERE track_id=? AND config_key=?',(b,KEY)).fetchone()
            data=json.loads(row[0]);data['decoded_hash']=store.track(a)['hash']
            db.execute('UPDATE analyses SET result=? WHERE track_id=? AND config_key=?',(json.dumps(data),b,KEY))
        report=listener.readiness()
        assert report['duplicate_recording_split_conflicts']==1 and not report['ready']
        assert ranking()['status']=='untrained'
    finally:
        with store.connect() as db:
            for i in ids:db.execute('DELETE FROM tracks WHERE id=?',(i,))
        (STATE/'ranking-model.json').unlink(missing_ok=True)
