import json
import numpy as np
import pytest
from soundmap import codec, tonal_detail, external_validation, source_detail


def test_codec_tokens_are_categorical_and_symmetric():
    a=codec.histogram(np.array([[0,0,7,7],[9,9,10,10]],dtype=np.int16))
    b=codec.histogram(np.array([[1,1,8,8],[9,9,10,10]],dtype=np.int16))
    assert codec.token_affinity(a,a)==pytest.approx(100)
    assert codec.token_affinity(a,b)==pytest.approx(50)
    assert codec.token_affinity(a,b)==codec.token_affinity(b,a)
    with pytest.raises(ValueError):codec.histogram(np.array([[1024]]))
    with pytest.raises(ValueError):codec.histogram(np.array([[.5]]))


def test_codec_corruption_is_quarantined_and_handles_closed(tmp_path):
    p=tmp_path/'window.npz';p.write_bytes(b'broken zip file')
    assert codec.checkpoint(p) is None
    assert not p.exists() and len(list(tmp_path.glob('*.corrupt-*')))==1
    np.savez_compressed(p,codes=np.array([[1,2,3]],dtype=np.int16),input_rms_dbfs=-20.,reconstruction_rms_error_db=.2)
    assert codec.checkpoint(p)['input_rms_dbfs']==-20


def test_codec_silence_short_and_timestamps():
    assert codec.encode(np.zeros(24000,dtype=np.float32))['silent']
    with pytest.raises(ValueError):codec.encode(np.zeros(10))
    with pytest.raises(ValueError):codec.encode(np.full(24000,np.nan))
    for duration in (1,3.1,8,15.9,33.2,300):
        windows=codec.positions(duration)
        assert len(windows)<=5
        assert all(0<=s<e<=duration and e-s<=8.00001 for s,e in windows)


def test_tonal_hypotheses_preserve_timeline_and_unknowns():
    x=np.zeros((12,12));x[:6,[0,4,7]]=1/np.sqrt(3);x[6:,[2,5,9]]=1/np.sqrt(3)
    r=tonal_detail.progression(x,np.arange(12),12)
    labels={s['label'] for s in r['segments']}
    assert 'C major' in labels and 'D minor' in labels
    assert r['segments'][0]['start']==0 and r['segments'][-1]['end']==12
    assert all(s['start']<s['end'] for s in r['segments'])
    uncertain=tonal_detail.progression(np.ones((4,12)),np.arange(4),4)
    assert uncertain['supported_fraction']==0
    with pytest.raises(ValueError):tonal_detail.progression(x,np.arange(12),2)


def test_external_labels_are_not_fabricated_or_silently_mapped():
    rows=[{'audio_pair':str(i),'overall_music_similarity':str(i/2),'overall_music_similarity_annotations':'[1, 5]'} for i in range(3)]
    assert external_validation.agreement(rows)['mean_absolute_disagreement_0_to_1']==1
    predictions=[{'audio_pair':str(i),'score':i/2} for i in range(3)]
    r=external_validation.evaluate_predictions(rows,predictions)
    assert r['mae']==0 and r['spearman']==1
    with pytest.raises(ValueError):external_validation.evaluate_predictions(rows,predictions+[predictions[0]])
    with pytest.raises(ValueError):external_validation.evaluate_predictions(rows,[{'audio_pair':'unknown','score':.5}])
    with pytest.raises(ValueError):external_validation.evaluate_predictions(rows,[{'audio_pair':'0','score':float('nan')}])


def test_source_detail_missing_evidence_is_explicit(monkeypatch):
    monkeypatch.setattr(source_detail,'cached',lambda _:None)
    assert source_detail.compare('a','b')['available'] is False
    assert source_detail.summarize({'windows':[]}) is None


def test_tonal_playback_clamps_resampling_roundoff(monkeypatch):
    from soundmap import audio,store
    monkeypatch.setattr(tonal_detail,'cached',lambda _: {'silent':False,'duration':2.00002,
        'harmony':{'sequence':np.ones((2,12)).tolist(),'time':[0.,1.]}})
    monkeypatch.setattr(store,'track',lambda _: {'path':'original.mp3'})
    monkeypatch.setattr(audio,'probe',lambda _:2.)
    r=tonal_detail.get('a')
    assert r['segments'][-1]['end']==2.
