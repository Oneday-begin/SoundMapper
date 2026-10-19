"""Read-only validation of the completed real FMA pilot and running local API."""
import json
from collections import Counter,defaultdict
import httpx
import numpy as np
from soundmap import store
from soundmap.config import ROOT
from soundmap.pipeline import cached,analyze
from soundmap.importer import metadata,pilot_rows

rows=[t for t in store.tracks() if t['source']=='fma']
assert len(rows)==100, 'This validation expects the initial 100-track pilot.'
assert {t['metadata']['track_id'] for t in rows}=={r['track_id'] for r in pilot_rows(list(metadata()),100,2026)}
recordings=defaultdict(set);artists=defaultdict(set)
checks=[]
for t in rows:
    a=cached(t['id']);assert a and a['mert'] and not a.get('mert_error')
    for layer in ('4','8','12'):
        v=np.asarray(a['mert']['layers'][layer]);assert v.shape==(768,) and np.isfinite(v).all()
        assert abs(np.linalg.norm(v)-1)<1e-5
    assert all(0<=w['start']<w['end']<=a['duration'] for w in a['windows'])
    assert store.digest(t['path'])==t['hash']
    recordings[a['decoded_hash']].add(t['metadata']['split'])
    artists[t['metadata']['artist_id']].add(t['metadata']['split'])
    checks.append({'fma_track_id':t['metadata']['track_id'],'finite_normalized_layers':True,'timestamps_valid':True,'content_hash_unchanged':True})
assert not any(len(s)>1 for s in recordings.values()), 'Duplicate recordings span splits'
assert not any(len(s)>1 for s in artists.values()), 'Artists span splits'
assert analyze(rows[0]['id'])['cache_hit']
pairs=[]
with httpx.Client(base_url='http://127.0.0.1:8765',timeout=120) as client:
    status=client.get('/api/status').json()
    library=client.get('/api/tracks',params={'source':'fma'}).json()
    assert len(library)==100 and all(t['mert_ready'] for t in library)
    m=client.get('/api/map',params={'source':'fma'}).json()
    assert len(m['points'])==100 and all(np.isfinite([p['x'],p['y']]).all() for p in m['points'])
    neighbors=client.get('/api/tracks/'+rows[0]['id']+'/neighbors',params={'source':'fma'}).json()
    assert len(neighbors)==10 and all(t['track']['id']!=rows[0]['id'] for t in neighbors)
    assert len({cached(t['track']['id'])['decoded_hash'] for t in neighbors})==10
    assert all(cached(t['track']['id'])['decoded_hash']!=cached(rows[0]['id'])['decoded_hash'] for t in neighbors)
    for a,b in [(rows[0],rows[0]),(rows[0],rows[1]),(rows[2],rows[3])]:
        r=client.post('/api/compare',params={'generation':False},json={'a':a['id'],'b':b['id']}).json()
        assert r['available'] and np.isfinite([r['overall'],r['acoustic'],r['mert']]).all()
        if a==b: assert abs(r['overall']-100)<.001
        for p in r['passages']:
            assert 0<=p['a']['start']<p['a']['end']<=cached(a['id'])['duration']
            assert 0<=p['b']['start']<p['b']['end']<=cached(b['id'])['duration']
        pairs.append({'a':a['metadata']['track_id'],'b':b['metadata']['track_id'],'overall':r['overall'],'acoustic':r['acoustic'],'mert':r['mert'],'passage_bounds_valid':True})
    response=client.get('/api/tracks/'+rows[0]['id']+'/audio',headers={'Range':'bytes=0-63'})
    assert response.status_code==206 and len(response.content)==64
report={'status':'passed','pilot_tracks':100,'track_checks':checks,'artist_overlap_across_splits':0,
        'decoded_duplicate_overlap_across_splits':0,'split_counts':dict(Counter(t['metadata']['split'] for t in rows)),
        'genre_counts':dict(Counter(t['metadata']['genre'] for t in rows)),'cache_reused':True,
        'api_library_ready':100,'map_points_finite':100,'self_excluding_neighbors':10,'audio_range_status':206,
        'pair_controls':pairs,'library_total':status['tracks'],'human_ratings_in_database':status['rating_count'],
        'human_ratings_created_by_test':0,'scope':'Integrity and application controls; no human recommendation-quality claim.'}
(ROOT/'reports'/'fma-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='track_checks'},indent=2))
