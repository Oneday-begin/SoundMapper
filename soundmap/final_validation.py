"""CPU-only real-audio codec checks and held-out genre coverage analysis."""
import json
import time
from itertools import combinations
import numpy as np
from . import store, codec, pseudo_model, dimensions, tonal_detail, source_detail
from .audio import decode
from .pipeline import MemoryMeter, KEY
from .config import ROOT, STATE


def coverage():
    with store.connect() as db:
        rows=db.execute("SELECT t.id,t.metadata,json_extract(a.result,'$.mert.layers') layers FROM tracks t JOIN analyses a ON a.track_id=t.id AND a.config_key=? WHERE t.source='fma' AND json_extract(t.metadata,'$.split')='test' AND json_type(a.result,'$.mert.layers')='object'",(KEY,)).fetchall()
    tracks={r['id']:{'metadata':json.loads(r['metadata']),'layers':json.loads(r['layers'])} for r in rows}
    pairs=[json.loads(line) for line in (STATE/'pseudo-pairs.jsonl').read_text().splitlines() if line]
    groups={};examples=[]
    for p in pairs:
        if p['split']!='test':continue
        a,b=tracks[p['a']],tracks[p['b']]
        prediction=pseudo_model.predict({'mert':{'layers':a['layers']}},{'mert':{'layers':b['layers']}})
        truth=np.asarray([p['targets'][k] for k in pseudo_model.TARGETS])*100
        pred=np.asarray([prediction['dimensions'][k] for k in pseudo_model.TARGETS]);errors=np.abs(pred-truth)
        genres=sorted({a['metadata'].get('genre','unknown'),b['metadata'].get('genre','unknown')})
        for g in genres:groups.setdefault(g,[]).append(errors.tolist())
        examples.append({'a':p['a'],'b':p['b'],'genres':genres,'mean_dimension_error':float(errors.mean()),
                         'errors':dict(zip(pseudo_model.TARGETS,errors.tolist()))})
    return {'test_recordings':len(tracks),'test_pairs':len(examples),
            'by_genre':{k:{'pairs_with_either_endpoint':len(v),'mae_points_by_dimension':dict(zip(pseudo_model.TARGETS,np.mean(v,axis=0).tolist()))} for k,v in sorted(groups.items())},
            'largest_machine_target_errors':sorted(examples,key=lambda p:p['mean_dimension_error'],reverse=True)[:5],
            'language_coverage':'No verified language labels in this experiment; language accuracy cannot be calculated.',
            'note':'Post-hoc descriptive breakdown of the fixed model; no retuning. Groups overlap when endpoints have different genres. Machine-target errors do not measure human judgment.'}


def run():
    started=time.perf_counter();personal=[t for t in store.tracks() if t['source']!='fma']
    with store.connect() as db:rating_count=db.execute('SELECT count(*) FROM ratings').fetchone()[0]
    hashes={t['id']:store.digest(t['path']) for t in personal}
    samples=[];transforms=[];pair_results=[]
    with MemoryMeter() as meter:
        for t in personal:
            r=codec.analyze(t['id']);resumed=codec.analyze(t['id'],force=True)
            assert resumed['reused_windows']==len(r['windows'])
            assert np.allclose(r['histogram'],resumed['histogram'])
            tonal=tonal_detail.get(t['id'])
            assert all(0<=s['start']<s['end']<=r['source_duration'] for s in tonal['segments'])
            identity=codec.compare(t['id'],t['id'])['score'];assert abs(identity-100)<1e-6
            samples.append({'id':t['id'],'title':t['title'],'identity_score':identity,
                'excerpt_count':len(r['windows']),'sampled_seconds':r['sampled_seconds'],
                'source_seconds':r['source_duration'],'initial_processing_seconds':r['processing_seconds'],
                'cache_reused_windows':resumed['reused_windows'],'tonal_hypothesis_supported_fraction':tonal['supported_fraction']})
            y,_=decode(t['path']);start=max(0,(len(y)-8*24000)//2);clip=y[start:start+8*24000]
            original=codec.encode(clip)
            for name,changed in [('half_gain',clip*.5),('time_reversal',clip[::-1].copy())]:
                other=codec.encode(changed)
                transforms.append({'id':t['id'],'transform':name,'codec_token_affinity':codec.token_affinity(original['histogram'],other['histogram']),
                    'input_rms_change_db':other['input_rms_dbfs']-original['input_rms_dbfs'],
                    'source_start':start/24000,'source_end':(start+len(clip))/24000})
            print('Verified',t['title'],flush=True)
        for a,b in combinations(personal,2):
            forward=codec.compare(a['id'],b['id']);reverse=codec.compare(b['id'],a['id'])
            assert abs(forward['score']-reverse['score'])<1e-8
            pair_results.append({'a':a['title'],'b':b['title'],'codec_texture':forward['score'],
                                 'source_detail':source_detail.compare(a['id'],b['id'])})
        cov=coverage()
    with store.connect() as db:after=db.execute('SELECT count(*) FROM ratings').fetchone()[0]
    assert after==rating_count
    assert all(store.digest(t['path'])==hashes[t['id']] for t in personal)
    result={'status':'complete','samples':samples,'pairs':pair_results,'real_excerpt_transforms':transforms,
            'coverage':cov,'human_ratings_unchanged':True,'original_hashes_unchanged':True,
            'processing_seconds':time.perf_counter()-started,'peak_rss_mb':meter.peak/1024**2,
            'note':'Real-audio engineering checks and calculated-target coverage only. Reversal/gain scores are observations, not recommendation-quality labels. EnCodec histogram pooling discards token order.',
            'model_revision':codec.REVISION}
    path=ROOT/'reports'/'final-validation.json';path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Report saved:',path,flush=True)
    return result


if __name__=='__main__':run()
