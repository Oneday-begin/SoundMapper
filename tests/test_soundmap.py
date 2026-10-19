import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from soundmap import store,encoder,importer
from soundmap.audio import decode,features,SR
from soundmap.pipeline import analyze,cached,KEY
from soundmap.compare import compare
from soundmap.app import app
from soundmap.config import STATE,CONFIG
from soundmap.explain import explain
from soundmap.experiments import ranking,agreement

def wav(seconds=3,sr=16000,freq=440,silent=False):
    t=np.arange(int(sr*seconds))/sr
    y=np.zeros(len(t)) if silent else .2*np.sin(2*np.pi*freq*t)*(1+.5*np.sin(2*np.pi*2*t))
    b=io.BytesIO();sf.write(b,y,sr,format='WAV');return b.getvalue()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
        from soundmap.jobs import _queue
        _queue.join()

def add(name='tone',**kw):
    p=STATE/'uploads'/(name+'.wav');p.write_bytes(wav(**kw));id_,_=store.register(p,'upload',title=name);return id_

def test_resample_stereo_and_unequal_duration():
    a=add('a',sr=16000);b=add('b',seconds=4,sr=44100,freq=660)
    aa=analyze(a);bb=analyze(b);r=compare(aa,bb)
    assert aa['analysis_sr']==24000 and aa['original_sr']==16000
    assert r['available'] and 0<=r['overall']<=100
    for p in r['passages']:
        assert 0<=p['a']['start']<p['a']['end']<=aa['duration']
        assert 0<=p['b']['start']<p['b']['end']<=bb['duration']
    stereo=STATE/'uploads'/'stereo.wav';y=np.sin(np.arange(44100)/44100*2*np.pi*440).astype('float32');sf.write(stereo,np.stack([y,y],axis=1),44100)
    data,info=decode(stereo);assert info['channels']==2 and len(data)==24000

def test_identical_and_cache():
    id_=add('repeat');a=analyze(id_);b=analyze(id_)
    assert b['cache_hit'] and compare(a,a)['overall']==pytest.approx(100,abs=1e-5)
    assert cached(id_)['analysis_config']['version']=='soundmap-1.0'

def test_bulk_library_query_joins_analysis_in_one_result():
    id_=add('bulk-library');analyze(id_)
    assert next(t for t in store.tracks(KEY) if t['id']==id_)['_analysis']['duration']==pytest.approx(3)
    assert next(t for t in store.tracks('not-the-active-key') if t['id']==id_)['_analysis'] is None

def test_identical_with_silent_tail():
    y=np.r_[.2*np.sin(np.arange(SR*2)/SR*2*np.pi*440),np.zeros(SR*2)].astype('float32')
    a=features(y)
    assert compare(a,a)['overall']==pytest.approx(100,abs=1e-5)

def test_silence_short_and_corrupt():
    id_=add('silence',silent=True);a=analyze(id_)
    assert a['silent'] and a['tempo'] is None and not compare(a,a)['available']
    short=STATE/'uploads'/'short.wav';short.write_bytes(wav(seconds=.2))
    with pytest.raises(ValueError,match='Insufficient'):decode(short)
    bad=STATE/'uploads'/'bad.mp3';bad.write_bytes(b'not an audio file')
    with pytest.raises(ValueError,match='Unreadable'):decode(bad)

def test_upload_limits_malformed_and_duplicate(client):
    assert client.post('/api/upload',files={'file':('bad.mp3',b'broken','audio/mpeg')}).status_code==422
    assert client.post('/api/upload',files={'file':('bad.txt',b'broken','text/plain')}).status_code==415
    assert client.post('/api/upload',files={'file':('short.wav',wav(seconds=.2),'audio/wav')}).status_code==422
    assert client.post('/api/upload',files={'file':('large.wav',b'0'*(1024**2+1),'audio/wav')}).status_code==413
    with patch('soundmap.jobs.submit',return_value='test-job'):
        first=client.post('/api/upload',files={'file':('../../safe.wav',wav(freq=330),'audio/wav')})
        second=client.post('/api/upload',files={'file':('same.wav',wav(freq=330),'audio/wav')})
    assert first.status_code==200 and second.json()['duplicate']
    id_=first.json()['track']['id'];analyze(id_)
    assert all(r['track']['id']!=id_ for r in client.get('/api/tracks/'+id_+'/neighbors').json())
    audio=client.get('/api/tracks/'+id_+'/audio',headers={'Range':'bytes=0-31'})
    assert audio.status_code==206 and len(audio.content)==32
    assert client.get('/api/tracks/'+id_+'/analysis').status_code==200

def test_api_security_and_missing(client):
    assert client.get('/api/tracks/not-found/audio').status_code==404
    assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==403
    assert client.post('/api/compare',json={'a':'x','b':'y'},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/api/tracks/../../Windows/audio').status_code==404
    pending=add('pending',freq=999)
    assert client.get('/api/tracks/'+pending+'/analysis').status_code==409
    assert client.get('/api/tracks/'+pending+'/neighbors?limit=200').status_code==422

def test_ratings_skips_export_repeats_and_untrained(client):
    a=add('ratedA',freq=230);b=add('ratedB',freq=350);analyze(a);analyze(b)
    assert client.post('/api/ratings',json={'a':a,'b':b}).status_code==422
    assert client.post('/api/ratings',json={'a':a,'b':b,'rhythm':6}).status_code==422
    for i in range(5):
        r=client.post('/api/ratings',json={'a':a,'b':b,'rhythm':3,'overall':4,'rater':'test-only','session':'repeat-test'})
        assert r.status_code==200
    next_=client.get('/api/ratings/next?session=repeat-test&rater=test-only').json()
    assert next_['repeat'] and {next_['a'],next_['b']}=={a,b}
    assert 'test-only' in client.get('/api/ratings/export').text
    assert ranking()['status']=='untrained'
    assert agreement()['repeat_consistency']['rhythm']['mean_absolute_difference']==0
    # These test-fixture ratings live exclusively in the temporary test database.

def test_explanations_optional_service():
    a=features(.2*np.sin(np.arange(SR*2)/SR*2*np.pi*440).astype('float32'))
    r=compare(a,a)
    out=explain(r)
    assert not out['generation']['enabled'] and out['references']
    with patch.dict(CONFIG,{'ollama_model':'unavailable-test'}),patch('soundmap.explain.httpx.post',side_effect=ConnectionError('offline')):
        out=explain(r)
    assert not out['generation']['enabled'] and 'fallback' in out['generation']['reason']
    assert not any('guitar' in s for s in out['predictions'])

def test_resumable_window_extraction(tmp_path):
    y=np.ones(SR*10,dtype='float32')*.1
    windows=[{'start':0,'end':5,'rms':.1},{'start':5,'end':10,'rms':.1}]
    count=[0]
    def fake_embed(y):count[0]+=1;return np.ones((3,768),dtype='float32')
    with patch('soundmap.encoder.load'),patch('soundmap.encoder.embed',side_effect=fake_embed):
        a=encoder.extract(y,windows,tmp_path,lambda p,m:None)
        b=encoder.extract(y,windows,tmp_path,lambda p,m:None)
    assert count[0]==2 and b['reused_windows']==2
    assert np.allclose(a['vector'],b['vector'])

def test_metadata_multilevel_join(tmp_path):
    (tmp_path/'tracks.csv').write_text(',set,set,artist,artist,track,track,track,track\n,subset,split,id,name,genre_top,title,license,language_code\ntrack_id,,,,,,,,\n2,small,training,1,Person,Rock,Song,CC,en\n3,large,test,2,Other,Jazz,Else,CC,en\n')
    rows=list(importer.metadata(tmp_path));assert len(rows)==1 and rows[0]['track_id']==2 and rows[0]['artist_id']=='1'
    assert rows[0]['split']=='training'

def test_archive_guard_and_pilot():
    importer.safe_archive_paths(['fma_small/000/000002.mp3'])
    for path in ['../a','fma_small/../a','C:/Windows/a','other/a']:
        with pytest.raises(ValueError):importer.safe_archive_paths([path])
    with patch.dict(CONFIG,{'archive':'nonexistent/fma_small.zip'}):assert importer.extract_archive()['status']=='pending'
    rows=[{'split':s,'genre':g,'track_id':i} for i,(s,g) in enumerate(itertools_product())]
    assert importer.pilot_rows(rows,4)==importer.pilot_rows(rows,4)

def itertools_product():
    import itertools
    return itertools.product(['training','validation','test'],['Rock','Jazz'])

def test_delete_uploaded_and_cache(client):
    id_=add('deletion',freq=123);analyze(id_);p=Path(store.track(id_)['path'])
    assert client.delete('/api/tracks/'+id_+'/cache').status_code==200 and cached(id_) is None
    assert client.delete('/api/tracks/'+id_).status_code==200 and not p.exists()

def test_queue_status_deduplicates(client):
    id_=add('queued',freq=117)
    first=client.post('/api/tracks/'+id_+'/analyze');second=client.post('/api/tracks/'+id_+'/analyze')
    assert first.status_code==200 and second.status_code==200
    assert len(client.get('/api/jobs').json())>=1

def test_full_queue_preserves_upload(client):
    with patch('soundmap.jobs.submit',side_effect=RuntimeError('Analysis queue full')):
        r=client.post('/api/upload',files={'file':('queuefull.wav',wav(freq=913),'audio/wav')})
    assert r.status_code==200 and r.json()['job'] is None and 'saved in the library' in r.json()['queue_error']
    assert Path(store.track(r.json()['track']['id'])['path']).is_file()
