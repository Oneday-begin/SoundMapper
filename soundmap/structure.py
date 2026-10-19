"""Measured window recurrence; no verse, chorus, melody, or boundary classifier."""
import numpy as np
from .audio import unit

def recurrence(analysis,max_windows=160):
    windows=analysis['windows']
    if not windows:return {'starts':[],'matrix':[],'candidates':[],'method':'No retained audio windows','note':'Insufficient audio'}
    indices=np.unique(np.rint(np.linspace(0,len(windows)-1,min(max_windows,len(windows)))).astype(int))
    ws=[windows[i] for i in indices];n=len(ws)
    rhythm=np.asarray([unit(w['rhythm']) for w in ws]);chroma=np.asarray([unit(w['chroma']) for w in ws])
    timbre=np.asarray([w['timbre'] for w in ws])
    distances=np.linalg.norm(timbre[:,None,:]-timbre[None,:,:],axis=-1)
    acoustic=.35*np.clip(rhythm@rhythm.T,0,1)+.30*np.clip(chroma@chroma.T,0,1)+.35*np.exp(-distances/np.sqrt(timbre.shape[1]))
    matrix=acoustic.copy()
    if any(w.get('embedding') for w in ws):
        d=len(next(w['embedding'] for w in ws if w.get('embedding')))
        vectors=np.asarray([w.get('embedding') or [0]*d for w in ws]);valid=np.asarray([bool(w.get('embedding')) for w in ws])
        pairs=np.outer(valid,valid);embedding=(np.clip(vectors@vectors.T,-1,1)+1)/2
        matrix[pairs]=.5*acoustic[pairs]+.5*embedding[pairs]
    nonsilent=np.asarray([w['rms']>=1e-6 for w in ws]);mask=np.outer(nonsilent,nonsilent)
    matrix=np.clip(matrix*100,0,100)
    # A window matches itself; silence is omitted rather than treated as music.
    for i in range(n):
        if nonsilent[i]:matrix[i,i]=100
    options=[]
    for i in range(n):
        for j in range(i+1,n):
            if mask[i,j] and ws[j]['start']-ws[i]['start']>=20:
                options.append({'score':float(matrix[i,j]),'a':{'start':ws[i]['start'],'end':ws[i]['end']},'b':{'start':ws[j]['start'],'end':ws[j]['end']}})
    selected=[]
    for p in sorted(options,key=lambda p:p['score'],reverse=True):
        if all(abs(p['a']['start']-q['a']['start'])>=10 and abs(p['b']['start']-q['b']['start'])>=10 for q in selected):selected.append(p)
        if len(selected)==3:break
    return {'starts':[w['start'] for w in ws],'matrix':[[float(matrix[i,j]) if mask[i,j] else None for j in range(n)] for i in range(n)],
            'candidates':selected,'sampled_windows':n,'total_windows':len(windows),
            'method':'Five-second acoustic-window recurrence; 50/50 with frozen MERT when both window embeddings exist',
            'note':'Similar window textures and pitch-class patterns are candidates for repeated material. These are uncalibrated indices, not verified structural boundaries, verse/chorus labels, or melody recognition. Strongest candidates are selected from a broad search.'}
