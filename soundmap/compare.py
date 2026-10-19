import numpy as np
import librosa
from .audio import unit

def cosine(a,b):
    return float(np.clip(np.dot(unit(a),unit(b)), -1, 1))

def timbre(a,b):
    return float(np.exp(-np.linalg.norm(np.asarray(a)-np.asarray(b))/np.sqrt(len(a))))

def harmonic(a,b):
    # Bound quadratic alignment; retain order, no transposition invariance.
    a=np.asarray(a); b=np.asarray(b)
    a=a[::max(1,int(np.ceil(len(a)/200)))]; b=b[::max(1,int(np.ceil(len(b)/200)))]
    cost=1-np.clip(a@b.T,0,1)
    # Two empty pitch-class frames match each other; an empty/nonempty pair does not.
    empty_a=np.linalg.norm(a,axis=1)<1e-8; empty_b=np.linalg.norm(b,axis=1)<1e-8
    cost[np.ix_(empty_a,empty_b)]=0
    D,path=librosa.sequence.dtw(C=cost)
    return float(np.clip(1-D[-1,-1]/len(path),0,1)), float(D[-1,-1]/len(path))

def compare(a,b,include_passages=True):
    if a['silent'] or b['silent']:
        return {'available':False,'reason':'Silence has no meaningful musical similarity','passages':[]}
    rhythm=max(0,cosine(a['rhythm'],b['rhythm']))
    tune, dtw_cost=harmonic(a['harmonic'],b['harmonic'])
    color=timbre(a['timbre'],b['timbre'])
    acoustic=.35*rhythm+.30*tune+.35*color
    embedding=None
    if a.get('mert') and b.get('mert'):
        embedding=(cosine(a['mert']['vector'],b['mert']['vector'])+1)/2
    overall=acoustic if embedding is None else .5*acoustic+.5*embedding
    passages=[]
    valid_a=[w for w in a['windows'] if w['rms']>=1e-6]
    valid_b=[w for w in b['windows'] if w['rms']>=1e-6]
    for wa in valid_a if include_passages else []:
        for wb in valid_b:
            local=.35*max(0,cosine(wa['rhythm'],wb['rhythm']))+.30*max(0,cosine(wa['chroma'],wb['chroma']))+.35*timbre(wa['timbre'],wb['timbre'])
            if wa.get('embedding') and wb.get('embedding'): local=.5*local+.5*(cosine(wa['embedding'],wb['embedding'])+1)/2
            passages.append({'a':{'start':wa['start'],'end':wa['end']},'b':{'start':wb['start'],'end':wb['end']},'score':100*local})
    passages.sort(key=lambda p:p['score'],reverse=True)
    selected=[]
    for p in passages:
        if all(abs(p['a']['start']-q['a']['start'])>=5 and abs(p['b']['start']-q['b']['start'])>=5 for q in selected): selected.append(p)
        if len(selected)==3: break
    result={'available':True,'overall':100*overall,'acoustic':100*acoustic,
            'rhythm':100*rhythm,'tune':100*tune,'timbre':100*color,
            'mert':None if embedding is None else 100*embedding, 'harmonic_dtw_cost':dtw_cost,
            'tempo_a':a['tempo'],'tempo_b':b['tempo'],'passages':selected,
            'aggregation':{'acoustic':{'rhythm':.35,'chroma_dtw':.30,'timbre':.35},'overall':{'acoustic':1 if embedding is None else .5,'mert':0 if embedding is None else .5}},
            'score_note':'Uncalibrated 0–100 similarity indices. Not probabilities, identity percentages, or evidence of copying.',
            'ranking_model':'Optional listener model untrained; these acoustic/MERT scores need no ratings.'}
    from .config import STATE
    import json
    learned=STATE/'ranking-model.json'
    if learned.exists() and embedding is not None:
        model=json.loads(learned.read_text())
        from .pipeline import KEY
        if model.get('analysis_key')!=KEY:
            result['ranking_model']='listener model needs retraining for the current analysis configuration'
        else:
            x=np.array([rhythm,tune,color,embedding])
            value=float(np.dot((x-model['mean'])/model['scale'],model['coefficient'])+model['intercept'])
            result['learned_index']=100*float(np.clip(value,0,1))
            result['ranking_model']=model['scope']+' regularized listener model; evaluated separately from fixed index'
    return result

class ExactSearch:
    """Search interface: replace candidate/vector dot products with a FAISS adapter later."""
    def search(self,query,candidates,limit=10):
        result=[]
        for id_,analysis in candidates:
            score=compare(query,analysis,include_passages=False)
            if score['available']: result.append({'id':id_,**score})
        return sorted(result,key=lambda r:r.get('learned_index',r['overall']),reverse=True)[:limit]
