"""Reproducible evaluation with explicit unavailable-label outcomes."""
import itertools
import hashlib
import json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,mean_absolute_error,ndcg_score
from . import store
from .config import STATE
from .pipeline import cached,KEY
from .compare import compare

def benchmark():
    rows=[t for t in store.tracks() if t['source']=='sample' and cached(t['id'])]
    pairs=[]
    for a,b in itertools.combinations(rows,2):
        r=compare(cached(a['id']),cached(b['id']))
        pairs.append({'a':a['title'],'b':b['title'],**r})
    self_checks=[]
    for t in rows:
        r=compare(cached(t['id']),cached(t['id']))
        self_checks.append({'title':t['title'],'overall':r.get('overall'),'acoustic':r.get('acoustic'),'mert':r.get('mert')})
    return {'pairs':pairs,'identical_inputs':self_checks,'human_retrieval_quality':None,
            'reason':'Insufficient listener relevance labels. These are measured similarities, not recommendation-quality evidence.'}

def genre():
    rows=[t for t in store.tracks(KEY) if t['source']=='fma' and t['_analysis'] and t['_analysis'].get('mert')]
    if not rows: return {'status':'pending','reason':'FMA audio/MERT pilot absent','task':'Auxiliary genre classification; not similarity ground truth'}
    # Preserve supplied splits; conservatively drop earlier-split artists that occur later.
    order={'training':0,'validation':1,'test':2}
    seen=defaultdict(set)
    for t in rows: seen[t['metadata']['artist_id']].add(t['metadata']['split'])
    clean=[t for t in rows if order[t['metadata']['split']]==max(order[s] for s in seen[t['metadata']['artist_id']])]
    overlap={k:sorted(v) for k,v in seen.items() if len(v)>1}
    split={s:[t for t in clean if t['metadata']['split']==s] for s in order}
    if any(len(v)<8 for v in split.values()) or len(set(t['metadata']['genre'] for t in split['training']))<2:
        return {'status':'insufficient_data','split_counts':{s:len(v) for s,v in split.items()},'artist_overlap':overlap}
    y={s:[t['metadata']['genre'] for t in ts] for s,ts in split.items()}
    trials=[]; best=None
    for representation in ('4','8','12','mean_4_8_12'):
        def vec(t):
            layers=t['_analysis']['mert']['layers']
            return np.mean([layers[l] for l in ('4','8','12')],axis=0) if representation.startswith('mean') else layers[representation]
        X={s:np.array([vec(t) for t in ts]) for s,ts in split.items()}
        for C in (.01,.1,1,10):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,class_weight='balanced',random_state=2026))
            model.fit(X['training'],y['training'])
            score=balanced_accuracy_score(y['validation'],model.predict(X['validation']))
            trials.append({'representation':representation,'C':C,'validation_balanced_accuracy':score})
            if best is None or score>best[0]: best=(score,representation,C,model,X)
    _,representation,C,model,X=best
    pred=model.predict(X['test'])
    return {'status':'complete','task':'Auxiliary genre classification; not musical-similarity ground truth',
            'split_counts':{s:len(v) for s,v in split.items()},'dropped_artist_overlap_tracks':len(rows)-len(clean),
            'overlap_artist_splits':overlap,'trials':trials,'selected_on_validation':{'representation':representation,'C':C},
            'test':{'accuracy':accuracy_score(y['test'],pred),'balanced_accuracy':balanced_accuracy_score(y['test'],pred),
                    'macro_f1':f1_score(y['test'],pred,average='macro',zero_division=0)},
            'test_predictions':[{'fma_track_id':t['metadata']['track_id'],'actual':truth,'prediction':p} for t,truth,p in zip(split['test'],y['test'],pred)],
            'note':'Pilot estimates have high uncertainty. Existing encoder pretraining overlap is not audited.'}

def ratings():
    with store.connect() as db: return [dict(r) for r in db.execute('SELECT * FROM ratings')]

def agreement():
    rows=ratings(); grouped=defaultdict(list)
    for r in rows: grouped[(tuple(sorted((r['a'],r['b']))),r['rater'] or r['session'])].append(r)
    repeat={d:[] for d in ('rhythm','tune','instruments','overall')}
    for rs in grouped.values():
        for a,b in itertools.combinations(rs,2):
            for d in repeat:
                if a[d] is not None and b[d] is not None: repeat[d].append(abs(a[d]-b[d]))
    by_rater=defaultdict(dict)
    for (pair,rater),rs in grouped.items():
        for d in repeat:
            values=[r[d] for r in rs if r[d] is not None]
            if values: by_rater[rater][(pair,d)]=float(np.mean(values))
    correlations=[]
    for a,b in itertools.combinations(by_rater,2):
        for d in repeat:
            common=[k for k in by_rater[a].keys() & by_rater[b].keys() if k[1]==d]
            if len(common)>=3:
                x=[by_rater[a][k] for k in common]; y=[by_rater[b][k] for k in common]
                rho=float(spearmanr(x,y).statistic) if len(set(x))>1 and len(set(y))>1 else None
                correlations.append({'raters':[a,b],'dimension':d,'common_pairs':len(common),'spearman':rho})
    return {'ratings':len(rows),'repeat_consistency':{d:{'repeat_comparisons':len(v),'mean_absolute_difference':float(np.mean(v)) if v else None} for d,v in repeat.items()},
            'listener_correlations':correlations,'status':'available' if correlations or any(repeat.values()) else 'insufficient: no shared or repeated judgments' if rows else 'unavailable: no human ratings'}

def group_for(t):
    # Never split windows or tracks; group FMA artist IDs, unknown artists by recording hash.
    return 'artist:'+t['metadata']['artist_id'] if t['metadata'].get('artist_id') else 'recording:'+t['hash']

def split_for(group):
    n=int(hashlib.sha256(('soundmap-2026:'+group).encode()).hexdigest()[:8],16)%10
    return 'training' if n<6 else 'validation' if n<8 else 'test'

def rank_features(a,b):
    r=compare(cached(a),cached(b),include_passages=False)
    if not r['available'] or r['mert'] is None: return None
    return np.array([r['rhythm'],r['tune'],r['timbre'],r['mert']])/100

def ranking():
    from .listener import prepare
    from .pipeline import KEY
    readiness,ts,assignment,pairs,_=prepare()
    if not readiness['ready']:
        return {'status':'untrained',**readiness,'ratings':readiness['rating_rows'],'genre_agreement_used_as_target':False}
    splits={s:[] for s in ('training','validation','test')}
    for r in pairs:
        x=rank_features(r['a'],r['b'])
        if x is None: raise RuntimeError('An eligible embedding became unavailable; retry after processing.')
        splits[r['split']].append((x,(r['overall']-1)/4,r))
    counts={s:len(v) for s,v in splits.items()}
    scope=readiness['scope']
    data={s:(np.stack([x for x,y,r in v]),np.array([y for x,y,r in v])) for s,v in splits.items()}
    trials=[]; best=None
    for alpha in (.1,1,10,100):
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(*data['training']); error=mean_absolute_error(data['validation'][1],model.predict(data['validation'][0]))
        trials.append({'alpha':alpha,'validation_mae':error})
        if best is None or error<best[0]: best=(error,alpha,model)
    model=best[2]; X,y=data['test']; learned=np.clip(model.predict(X),0,1)
    fixed=(X@np.array([.35,.30,.35,0])+X[:,3])/2
    acoustic=X@np.array([.35,.30,.35,0])
    def metrics(pred):
        queries=defaultdict(list)
        for i,(_,truth,r) in enumerate(splits['test']):
            queries[r['a']].append((truth,pred[i])); queries[r['b']].append((truth,pred[i]))
        values=[ndcg_score([[v[0] for v in vs]],[[v[1] for v in vs]]) for vs in queries.values() if len(vs)>=3]
        rho=float(spearmanr(y,pred).statistic) if len(set(y))>1 and len(set(pred))>1 else None
        return {'mae':mean_absolute_error(y,pred),'spearman':rho,'mean_ndcg':float(np.mean(values)) if values else None,'ndcg_queries':len(values)}
    scaler=model[0]; ridge=model[1]
    artifact={'features':['rhythm','tune','timbre','mert'],'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
              'coefficient':ridge.coef_.tolist(),'intercept':float(ridge.intercept_),'scope':scope,'analysis_key':KEY,'distinct_pairs':len(pairs),'training_track_ids':sorted(ts)}
    (STATE/'ranking-model.json').write_text(json.dumps(artifact,indent=2))
    return {'status':'trained','scope':artifact['scope'],'counts':counts,'readiness':readiness,'trials':trials,
            'selected_alpha':best[1],'test':{'ridge':metrics(learned),'fixed_combined':metrics(fixed),'acoustic':metrics(acoustic),'mert':metrics(X[:,3])},
            'group_assignment':assignment,'unknown_artists':'Excluded from training; ratings preserved for export.', 'repeat_aggregation':'Average repeats per identifier, then identifiers per unordered pair.'}
