"""Distill acoustic measurements into a small symmetric frozen-MERT pair model.
Machine targets only: never writes human ratings or claims perceptual validation.
"""
import hashlib,json,sqlite3,time
from pathlib import Path
from functools import lru_cache
import numpy as np
from .config import STATE,ROOT,CONFIG
from .pipeline import KEY,MemoryMeter
from .dimensions import KEY as DIM_KEY

TARGETS=['rhythm_proxy','harmonic_proxy','timbral_proxy','dynamic_range_proxy']
WEIGHTS=np.array([.30,.30,.25,.15])
VERSION='pseudo-distillation-1'
ARTIFACT=STATE/'pseudo-model.json'
NOTE='Machine-supervised approximation of acoustic rules; no human listening ratings. Scores are not validated perceptual similarity or instrument recognition.'


def clean_splits(rows):
    order={'training':0,'validation':1,'test':2};artists={};hashes={}
    valid=[r for r in rows if r['metadata'].get('artist_id') and r['metadata'].get('split') in order]
    for r in valid:
        artists.setdefault(str(r['metadata']['artist_id']),set()).add(r['metadata']['split'])
        hashes.setdefault(r['decoded_hash'],set()).add(r['metadata']['split'])
    # Conservatively keep artists in their latest supplied split; exclude hash conflicts.
    clean=[r for r in valid if order[r['metadata']['split']]==max(order[s] for s in artists[str(r['metadata']['artist_id'])]) and len(hashes[r['decoded_hash']])==1]
    return {s:[r for r in clean if r['metadata']['split']==s] for s in order}, {'excluded':len(rows)-len(clean),'artist_split_conflicts':sum(len(v)>1 for v in artists.values()),'duplicate_split_conflicts':sum(len(v)>1 for v in hashes.values())}


def teacher(a,b):
    from .compare import harmonic
    rhythm=max(0,float(np.dot(a['rhythm'],b['rhythm'])))
    tempo=np.exp(-abs(np.log2(a['tempo']/b['tempo']))/.25)
    harmonic_score=harmonic(a['harmonic'],b['harmonic'])[0]
    timbre=np.exp(-np.linalg.norm(np.asarray(a['timbre'])-b['timbre'])/np.sqrt(len(a['timbre'])))
    dynamics=np.exp(-np.linalg.norm((np.asarray(a['dynamics'])-b['dynamics'])/6)/np.sqrt(2))
    return np.array([.6*rhythm+.4*tempo,harmonic_score,timbre,dynamics])


def pair_features(a,b,model):
    def project(x):
        z=(np.asarray(x['8'])-model['pca_mean']) @ np.asarray(model['pca_components']).T
        return z/max(np.linalg.norm(z),1e-10)
    za,zb=project(a),project(b)
    cos=np.array([np.clip(np.dot(a[k],b[k]),-1,1) for k in ('4','8','12')])
    return np.r_[np.abs(za-zb),za*zb,cos,cos*cos]


@lru_cache(maxsize=2)
def load(stamp):return json.loads(ARTIFACT.read_text())


def predict(a,b):
    if not ARTIFACT.exists():return {'available':False,'reason':'Machine-supervised model has not been trained.'}
    m=load(ARTIFACT.stat().st_mtime_ns)
    if m['analysis_key']!=KEY or m['dimension_key']!=DIM_KEY:return {'available':False,'reason':'Analysis settings changed; retraining required.'}
    if a.get('silent') or b.get('silent') or not a.get('mert') or not b.get('mert'):return {'available':False,'reason':'Both nonsilent recordings require MERT embeddings.'}
    x=pair_features(a['mert']['layers'],b['mert']['layers'],m)
    y=np.clip(np.asarray(m['coefficient']) @ ((x-m['mean'])/m['scale'])+m['intercept'],0,1)
    return {'available':True,'index':100*float(y @ WEIGHTS), 'dimensions':{k:100*float(v) for k,v in zip(TARGETS,y)},
            'note':NOTE,'training_target':'Calculated acoustic pseudo-labels, not listener judgments',
            'aggregation':dict(zip(TARGETS,WEIGHTS.tolist())),'model':'Standardized multi-output ridge above frozen MERT; PCA fitted on training tracks only',
            'unsupported':['instrument identities','vocal delivery','isolated melody','key/mode','human overall feel'],'validation':m['validation']}


def _rows():
    from . import store
    with sqlite3.connect('file:'+str(STATE/'soundmap.sqlite').replace('\\','/')+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        raw=db.execute("""SELECT t.id,t.hash,t.metadata,
        json_extract(a.result,'$.decoded_hash') AS decoded_hash,
        json_extract(a.result,'$.mert.layers') AS layers,
        json_extract(a.result,'$.rhythm') AS rhythm,
        json_extract(a.result,'$.timbre') AS timbre,
        json_extract(a.result,'$.tempo') AS tempo,
        json_extract(d.result,'$.harmony.sequence') AS harmonic,
        json_extract(d.result,'$.dynamics.crest_db') AS crest,
        json_extract(d.result,'$.dynamics.level_spread_db') AS spread
        FROM tracks t JOIN analyses a ON a.track_id=t.id AND a.config_key=?
        JOIN annotations d ON d.track_id=t.id AND d.kind='dimensions' AND d.config_key=?
        WHERE t.source='fma' AND json_extract(a.result,'$.silent')=0 AND json_type(a.result,'$.mert.layers')='object'
        AND json_extract(a.result,'$.tempo')>0 AND json_type(d.result,'$.harmony.sequence')='array'""",(KEY,DIM_KEY)).fetchall()
    rows=[]
    for r in raw:
        row=dict(r);row['metadata']=json.loads(row['metadata'])
        for k in ('layers','rhythm','timbre','harmonic'):row[k]=json.loads(row[k])
        row['dynamics']=[row.pop('crest'),row.pop('spread')]
        if all(np.isfinite(np.asarray(row[k])).all() for k in ('rhythm','timbre','harmonic','dynamics')) and all(np.isfinite(v).all() for v in row['layers'].values()):rows.append(row)
    return sorted(rows,key=lambda r:r['id'])


def pairs(rows,anchors,per_anchor,seed):
    rng=np.random.default_rng(seed);vectors=np.stack([r['layers']['8'] for r in rows]);seen=set();result=[]
    for i in rng.permutation(len(rows))[:min(anchors,len(rows))]:
        choices=np.array([j for j,r in enumerate(rows) if j!=i and r['decoded_hash']!=rows[i]['decoded_hash']])
        if not len(choices):continue
        pool=rng.choice(choices,min(200,len(choices)),replace=False);scores=vectors[pool] @ vectors[i]
        order=pool[np.argsort(scores)];n=per_anchor//4
        selected=list(order[:n])+list(order[-n:])+list(rng.permutation(pool))
        used=set();count=0
        for j in selected:
            pair=tuple(sorted((int(i),int(j))))
            if j in used or pair in seen:continue
            used.add(j);seen.add(pair);result.append((int(i),int(j)));count+=1
            if count>=per_anchor:break
    return result


def train():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import ndcg_score
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    from filelock import FileLock
    runtime=STATE/'runtime';runtime.mkdir(exist_ok=True)
    def progress(stage,**extra):
        temp=runtime/'pseudo-training.tmp';temp.write_text(json.dumps({'state':'running','stage':stage,**extra}));temp.replace(runtime/'pseudo-training.json');print(stage,extra,flush=True)
    start=time.perf_counter()
    with FileLock(str(runtime/'pseudo-training.lock'),timeout=0), MemoryMeter() as meter, threadpool_limits(limits=2):
        progress('Loading cached pooled embeddings and measurement teachers')
        rows=_rows();split,audit=clean_splits(rows)
        if any(len(rs)<50 for rs in split.values()):raise ValueError('At least 50 eligible recordings per supplied split required.')
        fingerprint=hashlib.sha256(json.dumps([(r['id'],r['hash'],r['decoded_hash'],r['metadata']['split']) for r in rows]).encode()).hexdigest()
        progress('Fitting training-only PCA',tracks={s:len(rs) for s,rs in split.items()})
        pca=PCA(n_components=32,svd_solver='randomized',random_state=2026).fit(np.array([r['layers']['8'] for r in split['training']]))
        m={'version':VERSION,'analysis_key':KEY,'dimension_key':DIM_KEY,'revision':CONFIG['model_revision'],
           'pca_mean':pca.mean_.tolist(),'pca_components':pca.components_.tolist(),'human_similarity_trained':False,'dataset_fingerprint':fingerprint}
        cache=sqlite3.connect(STATE/'pseudo-labels.sqlite')
        cache.execute('CREATE TABLE IF NOT EXISTS labels (key TEXT,a TEXT,b TEXT,target TEXT,PRIMARY KEY(key,a,b))')
        cachekey=hashlib.sha256((VERSION+KEY+DIM_KEY+fingerprint).encode()).hexdigest();data={};manifest=[];reused=0
        try:
            for index,(s,rs) in enumerate(split.items()):
                ps=pairs(rs,150 if s=='training' else 30,40,2026+index);X=[];Y=[]
                for n,(i,j) in enumerate(ps):
                    a,b=rs[i],rs[j];ids=sorted((a['id'],b['id']))
                    old=cache.execute('SELECT target FROM labels WHERE key=? AND a=? AND b=?',(cachekey,*ids)).fetchone()
                    if old:y=np.array(json.loads(old[0]));reused+=1
                    else:
                        y=teacher(a,b);cache.execute('INSERT INTO labels VALUES(?,?,?,?)',(cachekey,*ids,json.dumps(y.tolist())))
                    X.append(pair_features(a['layers'],b['layers'],m));Y.append(y)
                    manifest.append({'split':s,'a':a['id'],'b':b['id'],'label_source':'calculated_acoustic_rules','targets':dict(zip(TARGETS,y.tolist()))})
                    if n%100==0:cache.commit();progress('Generating checkpointed machine labels',split=s,done=n,total=len(ps))
                cache.commit();data[s]=(np.array(X),np.array(Y),ps)
        finally:cache.close()
        progress('Training regularized pair models')
        X,Y,_=data['training'];scaler=StandardScaler().fit(X);trials=[];best=None
        for alpha in (.1,1,10,100,1000):
            ridge=Ridge(alpha=alpha).fit(scaler.transform(X),Y)
            prediction=np.clip(ridge.predict(scaler.transform(data['validation'][0])),0,1)
            error=float(np.mean(np.abs(prediction-data['validation'][1])))
            trials.append({'alpha':alpha,'validation_mae':error})
            if best is None or error<best[0]:best=(error,alpha,ridge)
        _,alpha,ridge=best;m.update(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficient=ridge.coef_.tolist(),intercept=ridge.intercept_.tolist(),alpha=alpha,validation={'machine_target_mae':best[0]})
        def evaluate(pred,Y,ps):
            summary={k:float(np.mean(np.abs(pred[:,i]-Y[:,i]))) for i,k in enumerate(TARGETS)}
            truth=Y@WEIGHTS;estimate=pred@WEIGHTS;groups={}
            for n,(a,b) in enumerate(ps):groups.setdefault(a,[]).append(n);groups.setdefault(b,[]).append(n)
            scores=[ndcg_score([truth[v]],[estimate[v]],k=10) for v in groups.values() if len(v)>=5]
            summary.update(overall_mae=float(np.mean(np.abs(truth-estimate))),overall_spearman=float(spearmanr(truth,estimate).statistic) if np.ptp(estimate)>1e-12 and np.ptp(truth)>1e-12 else None,
                           machine_target_ndcg_at_10=float(np.mean(scores)) if scores else None,ranking_queries=len(scores))
            return summary
        Xt,Yt,ps=data['test'];learned=np.clip(ridge.predict(scaler.transform(Xt)),0,1)
        mean=np.tile(Y.mean(axis=0),(len(Yt),1))
        rawcos=Xt[:,-6:-3];mert=np.tile(((rawcos[:,2]+1)/2)[:,None],(1,4))
        results={'ridge':evaluate(learned,Yt,ps),'training_mean':evaluate(mean,Yt,ps),'frozen_mert_layer12':evaluate(mert,Yt,ps),
                 'exact_rule_teacher':evaluate(Yt,Yt,ps)}
        # Teacher perfect agreement is tautological, not perceptual evidence.
        report={'status':'trained','task':'Machine-supervised acoustic-rule distillation','note':NOTE,'targets':TARGETS,'aggregation':dict(zip(TARGETS,WEIGHTS.tolist())),
                'teacher_rules':{'rhythm':'0.6 onset-autocorrelation cosine + 0.4 exp(-abs(log2(BPM ratio))/0.25)',
                'harmonic':'chronological CQT chroma DTW, bounded at 200 frames','timbral':'exp(-L2 distance of fixed-scaled MFCC/spectral features / sqrt(feature count))',
                'dynamic_range':'exp(-L2 distance of crest and RMS spread divided by 6 dB / sqrt(2))'},
                'tracks':{s:len(rs) for s,rs in split.items()},'pair_counts':{s:len(d[0]) for s,d in data.items()},'split_audit':audit,
                'trials':trials,'selected_alpha':alpha,'test_against_machine_targets':results,'checkpoint_reused_pairs':reused,
                'human_ratings_created':0,'human_similarity_trained':False,'encoder_finetuned':False,'dataset_fingerprint':fingerprint,
                'pair_selection':'150 training / 30 validation / 30 test anchors; up to 40 candidates each, uniform and nearest/farthest from a deterministic 200-track pool; no test-target tuning',
                'leakage_control':'Supplied splits, artist and recording conflicts removed before pairs; PCA/scaling fit only on training tracks/pairs; all windows remain with source recording',
                'limitations':['Machine-target accuracy is not listener agreement.','Rule teacher scores remain available and are exact by construction.','No instrument/vocal/melody labels synthesized.','30-second FMA to full Hindi songs is an unevaluated domain shift.','Random/stratified candidates do not establish catalogue-wide retrieval quality.']}
        temp=ARTIFACT.with_suffix('.tmp');temp.write_text(json.dumps(m));temp.replace(ARTIFACT)
        (STATE/'pseudo-pairs.jsonl').write_text('\n'.join(json.dumps(r) for r in manifest))
    report.update(processing_seconds=time.perf_counter()-start,peak_rss_mb=meter.peak/1024**2)
    (ROOT/'reports/pseudo-model.json').write_text(json.dumps(report,indent=2))
    (runtime/'pseudo-training.json').write_text(json.dumps({'state':'complete','stage':'evaluated','pair_counts':report['pair_counts']}))
    return report

if __name__=='__main__':
    try:print(json.dumps(train(),indent=2))
    except Exception as e:
        (STATE/'runtime'/'pseudo-training.json').write_text(json.dumps({'state':'failed','error':str(e)}));raise
