"""Conservative chronological chord hypotheses and local tonal-profile estimates."""
import numpy as np
from .dimensions import cached, tonal

NAMES=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B']
NOTE='Major/minor triad template hypotheses from mixed-audio CQT; not validated chord transcription. Other chords, ragas and modes are unsupported.'


def progression(sequence, times, duration):
    x=np.asarray(sequence,dtype=float);times=np.asarray(times,dtype=float)
    if x.ndim!=2 or x.shape[1]!=12 or len(x)!=len(times) or not len(x):raise ValueError('Invalid CQT sequence')
    if not np.isfinite(x).all() or not np.isfinite(times).all() or np.any(np.diff(times)<0):raise ValueError('Invalid CQT timeline')
    if duration<=0 or times[0]<0 or times[-1]>=duration:raise ValueError('CQT timestamps out of range')
    # Local three-frame mean stabilizes isolated noisy pitch-class peaks.
    smooth=np.array([x[max(0,i-1):min(len(x),i+2)].mean(axis=0) for i in range(len(x))])
    smooth/=np.maximum(np.linalg.norm(smooth,axis=1,keepdims=True),1e-10)
    templates=[];labels=[]
    for root in range(12):
        for quality,intervals in [('major',(0,4,7)),('minor',(0,3,7))]:
            t=np.zeros(12);t[[(root+i)%12 for i in intervals]]=1/np.sqrt(3)
            templates.append(t);labels.append(NAMES[root]+' '+quality)
    scores=smooth@np.asarray(templates).T;order=np.argsort(scores,axis=1)
    segments=[]
    for i,row in enumerate(scores):
        best=int(order[i,-1]);margin=float(row[best]-row[order[i,-2]])
        label=labels[best] if row[best]>=.72 and margin>=.06 else 'uncertain'
        end=float(times[i+1]) if i+1<len(times) else float(duration)
        if segments and segments[-1]['label']==label:
            segments[-1]['end']=end;segments[-1]['frames']+=1
        else:segments.append({'start':float(times[i]),'end':end,'label':label,'frames':1})
    local=[]
    for start in np.arange(0,duration,30):
        selected=x[(times>=start)&(times<start+30)]
        if len(selected):local.append({'start':float(start),'end':float(min(duration,start+30)),**tonal(selected.mean(axis=0))})
    return {'segments':segments,'local_keys':local,'note':NOTE,'grid_frames':len(x),
            'supported_fraction':sum(s['end']-s['start'] for s in segments if s['label']!='uncertain')/duration,
            'resolution_seconds':float(np.median(np.diff(times))) if len(times)>1 else duration,
            'threshold_note':'Fixed cosine >= 0.72 and winner margin >= 0.06; heuristic gates, not calibrated confidence.',
            'source':'https://librosa.org/doc/0.10.2/generated/librosa.feature.chroma_cqt.html'}


def get(id_):
    r=cached(id_)
    if not r or r['silent'] or not r.get('harmony'):return {'available':False,'reason':'No nonsilent CQT evidence available.'}
    h=r['harmony']
    from .audio import probe
    from . import store
    # Resampled lengths can round upward by one sample. Playback uses the source duration.
    duration=min(r['duration'],probe(store.track(id_)['path']))
    keep=np.asarray(h['time'])<duration
    return {'available':True,**progression(np.asarray(h['sequence'])[keep],np.asarray(h['time'])[keep],duration)}
