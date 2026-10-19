import numpy as np
from soundmap import pseudo_model as pm

def test_artist_and_recording_leakage():
    rows=[{'id':'a','decoded_hash':'one','metadata':{'artist_id':1,'split':'training'}}, {'id':'b','decoded_hash':'two','metadata':{'artist_id':1,'split':'test'}}, {'id':'c','decoded_hash':'two','metadata':{'artist_id':2,'split':'validation'}}]
    splits,audit=pm.clean_splits(rows)
    assert not any(splits.values())
    assert audit['artist_split_conflicts']==1 and audit['duplicate_split_conflicts']==1

def test_symmetric_prediction_features():
    rng=np.random.default_rng(3)
    a={k:rng.normal(size=8) for k in ('4','8','12')};b={k:rng.normal(size=8) for k in a}
    m={'pca_mean':np.zeros(8),'pca_components':np.eye(8)[:4]}
    assert np.allclose(pm.pair_features(a,b,m),pm.pair_features(b,a,m))

def test_missing_model_is_explicit(tmp_path,monkeypatch):
    monkeypatch.setattr(pm,'ARTIFACT',tmp_path/'absent.json')
    assert pm.predict({}, {})['available'] is False

def test_pairs_exclude_duplicate_audio():
    rows=[{'id':str(i),'decoded_hash':str(i//2),'layers':{'8':np.eye(12)[i]}} for i in range(12)]
    pairs=pm.pairs(rows,12,8,7)
    assert pairs==pm.pairs(rows,12,8,7)
    assert all(a!=b and rows[a]['decoded_hash']!=rows[b]['decoded_hash'] for a,b in pairs)
    assert len(pairs)==len(set(tuple(sorted(p)) for p in pairs))
