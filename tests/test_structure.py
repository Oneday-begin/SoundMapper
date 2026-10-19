import numpy as np
from soundmap.structure import recurrence

def window(start,kind=0,rms=.1):
    return {'start':start,'end':start+5,'rms':rms,'rhythm':[1,kind/2],'chroma':[1,kind/2],
            'timbre':[kind*3,kind+1],'embedding':None}

def test_recurrence_identical_control_silence_and_candidate_bounds():
    r=recurrence({'windows':[window(0),window(10,1),window(30),window(40,rms=0)]})
    m=np.asarray([[v if v is not None else np.nan for v in row] for row in r['matrix']])
    assert np.isclose(m[0,0],100,atol=1e-5) and np.isclose(m[0,2],100,atol=1e-5) and m[0,1]<100
    assert np.isnan(m[3]).all() and np.allclose(m[:3,:3],m[:3,:3].T)
    assert r['candidates'][0]['a']=={'start':0,'end':5} and r['candidates'][0]['b']=={'start':30,'end':35}
    assert all(p['b']['start']-p['a']['start']>=20 for p in r['candidates'])

def test_recurrence_downsampling_preserves_source_times():
    r=recurrence({'windows':[window(i*5) for i in range(240)]},max_windows=160)
    assert len(r['starts'])==160 and r['starts'][0]==0 and r['starts'][-1]==1195
    assert r['total_windows']==240 and r['sampled_windows']==160
    assert all(p['b']['end']<=1200 for p in r['candidates'])
