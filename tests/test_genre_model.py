import json
import numpy as np
from soundmap import genre_model as g, encoder

def test_profile_affinity_and_stale_artifact(tmp_path,monkeypatch):
    p=tmp_path/'model.json';monkeypatch.setattr(g,'ARTIFACT',p)
    assert not g.affinity({}, {})['available']
    m={'analysis_key':g.KEY,'representation':'8','classes':['A','B'],
       'mean':[0,0],'scale':[1,1],'coefficient':[[1,0],[-1,0]],'intercept':[0,0],'test':{'accuracy':.5}}
    p.write_text(json.dumps(m));a={'mert':{'layers':{'8':[1,0]}}};b={'mert':{'layers':{'8':[-1,0]}}}
    assert abs(g.affinity(a,a)['index']-100)<1e-6
    assert g.affinity(a,b)['index']<g.affinity(a,a)['index']
    assert np.allclose(g.affinity(a,b)['index'],g.affinity(b,a)['index'])
    assert not g.affinity(a,{})['available']
    m['analysis_key']='stale';p.write_text(json.dumps(m));g.load.cache_clear()
    assert not g.affinity(a,a)['available']

def test_corrupt_embedding_checkpoint_recovers(tmp_path,monkeypatch):
    monkeypatch.setattr(encoder,'load',lambda:None)
    monkeypatch.setattr(encoder,'embed',lambda y:np.ones((3,768),dtype=np.float32))
    (tmp_path/'window-00000.npz').write_bytes(b'broken archive')
    windows=[{'start':0,'end':1,'rms':.1}]
    r=encoder.extract(np.ones(24000),windows,tmp_path,lambda p,m:None)
    assert r['windows']==1
    assert len(list(tmp_path.glob('*.corrupt-*')))==1
    with np.load(tmp_path/'window-00000.npz') as d:assert np.isfinite(d['vectors']).all()
    assert encoder.extract(np.ones(24000),windows,tmp_path,lambda p,m:None)['reused_windows']==1


def test_personal_library_excludes_training_audio(tmp_path):
    from soundmap import store
    from soundmap.app import app
    from fastapi.testclient import TestClient
    p=tmp_path/'training.mp3';p.write_bytes(b'training fixture')
    training,_=store.register(p,'fma',title='training fixture')
    p=tmp_path/'personal.mp3';p.write_bytes(b'personal fixture')
    personal,_=store.register(p,'sample',title='personal fixture')
    rows=TestClient(app).get('/api/tracks?source=personal').json()
    assert personal in {r['id'] for r in rows}
    assert training not in {r['id'] for r in rows}
    assert all(r['source']!='fma' for r in rows)
