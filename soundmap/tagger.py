"""Separate, frozen AudioSet AST tagger. Tag scores never change similarity indices."""
import hashlib
import json
import math
import threading
import time
import numpy as np
import librosa
from . import store
from .config import CONFIG,ROOT,STATE
from .audio import decode

MODEL_ID='MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION=CONFIG.get('tagging_revision','f826b80d28226b62986cc218e5cec390b1096902')
ONTOLOGY=json.loads((ROOT/'instrument-ontology.json').read_text(encoding='utf-8'))
SETTINGS={'version':2,'model':MODEL_ID,'revision':REVISION,'sr':16000,'seconds':10,
          'max_windows':int(CONFIG.get('tagging_max_windows',9)),'aggregation':'mean sigmoid across non-silent sampled windows',
          'display_threshold':.2,'ontology_revision':ONTOLOGY['revision'],'preprocessing':'official ASTFeatureExtractor torchaudio 2.6 CPU Kaldi fbank; 24k mono decode resampled to 16k'}
KEY=hashlib.sha256(json.dumps(SETTINGS,sort_keys=True).encode()).hexdigest()[:20]
_model=None;_processor=None;_lock=threading.Lock()
status={'state':'not_loaded','model':MODEL_ID,'revision':REVISION,'error':None}

def load():
    global _model,_processor
    if not CONFIG.get('tagging_enabled',False): raise RuntimeError('Instrument tagging disabled')
    with _lock:
        if _model is not None: return _model,_processor
        import torch
        import torchaudio
        from transformers import ASTForAudioClassification,ASTFeatureExtractor
        torch.set_num_threads(CONFIG['threads']);torch.manual_seed(0)
        try:
            status.update(state='loading',error=None)
            _processor=ASTFeatureExtractor.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True)
            model,info=ASTForAudioClassification.from_pretrained(MODEL_ID,revision=REVISION,
                local_files_only=True,use_safetensors=True,output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise RuntimeError('Incomplete AST weight load: '+str(info))
            assert _processor.sampling_rate==16000 and len(model.config.id2label)==527
            _model=model.cpu().eval()
            for p in _model.parameters():p.requires_grad_(False)
            status.update(state='ready',strict_weight_load=True,classes=527)
        except Exception as e:
            _model=None;status.update(state='unavailable',error=str(e));raise
        return _model,_processor

def infer(y):
    import torch
    model,processor=load()
    inputs=processor(np.asarray(y,dtype=np.float32),sampling_rate=16000,return_tensors='pt')
    with torch.inference_mode(): scores=torch.sigmoid(model(**inputs).logits[0]).cpu().numpy()
    if scores.shape!=(527,) or not np.isfinite(scores).all():raise ValueError('Invalid AST scores')
    return scores

def windows(duration):
    if duration<2: return []
    n=min(SETTINGS['max_windows'],max(1,int(duration//10)))
    return [{'start':float(s),'end':float(min(duration,s+10))} for s in np.linspace(0,max(0,duration-10),n)]

def cached(id_):
    with store.connect() as db:
        row=db.execute('SELECT result FROM annotations WHERE track_id=? AND kind=? AND config_key=?',(id_,'instruments',KEY)).fetchone()
    return json.loads(row[0]) if row else None

from .compute import serialized

@serialized
def analyze(id_,progress=lambda p,m:None,force=False):
    prior=cached(id_)
    if prior and not force:
        progress(1,'Instrument prediction cache reused')
        return {**prior,'cache_hit':True}
    from .pipeline import MemoryMeter
    t=store.track(id_);start=time.perf_counter()
    with MemoryMeter() as meter:
        progress(.02,'Decoding for separate instrument tagger')
        y,info=decode(t['path']);positions=windows(info['duration'])
        y=librosa.resample(y,orig_sr=24000,target_sr=16000,res_type='soxr_hq')
        folder=STATE/'cache'/t['hash']/('ast-'+KEY);folder.mkdir(parents=True,exist_ok=True)
        scores=[];used=[];reused=0
        for i,w in enumerate(positions):
            segment=y[int(round(w['start']*16000)):int(round(w['end']*16000))]
            if np.sqrt(np.mean(segment**2))<1e-6:continue
            path=folder/f'window-{i:03d}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as data:v=data['scores']
                if v.shape!=(527,) or not np.isfinite(v).all():raise ValueError('Invalid cached AST scores')
                reused+=1
            else:
                v=infer(segment);temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,scores=v);temp.replace(path)
            scores.append(v);used.append(w)
            progress(.1+.85*(i+1)/len(positions),f'Instrument model excerpt {i+1}/{len(positions)}')
        labels=[];candidates=[];passage_predictions=[]
        if scores:
            model,_=load();means=np.mean(scores,axis=0)
            allowed={r['name'] for r in ONTOLOGY['instrument_classes']}
            for index,name in model.config.id2label.items():
                if name in allowed:
                    best=int(np.argmax([v[index] for v in scores]))
                    candidates.append({'label':name,'score':float(means[index]),'strongest_excerpt':used[best],
                                   'strongest_excerpt_score':float(scores[best][index])})
            candidates=sorted(candidates,key=lambda r:r['score'],reverse=True)
            labels=[r for r in candidates if r['score']>=SETTINGS['display_threshold']][:8]
            passage_predictions=sorted([{'label':r['label'],'score':r['strongest_excerpt_score'],
                'excerpt':r['strongest_excerpt']} for r in candidates if r['strongest_excerpt_score']>=.2],key=lambda r:r['score'],reverse=True)[:8]
        result={'state':'ready' if scores else 'insufficient_audio','labels':labels,'passage_predictions':passage_predictions,'low_score_candidates':[r for r in candidates if r['score']<.2][:5],'sampled_windows':used,
                'analyzed_seconds':sum(w['end']-w['start'] for w in used),'source_duration':info['duration'],
                'coverage':'Sampled excerpts only; intervening audio may contain different instruments.',
                'confidence_note':'Uncalibrated AudioSet sigmoid tag scores; not similarity or verified instrument identities. No local instrument ground-truth labels.',
                'no_labels_note':'No instrument class reached the fixed 0.20 display threshold; absence of a tag does not establish absence of an instrument.',
                'model':MODEL_ID,'revision':REVISION,'settings':SETTINGS,'cache_hit':False,
                'windows':len(scores),'reused_windows':reused}
    result['processing_seconds']=time.perf_counter()-start;result['peak_rss_mb']=meter.peak/1024**2
    with store.connect() as db:db.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?,?)',(id_,'instruments',KEY,json.dumps(result)))
    progress(1,'Instrument predictions cached; similarity indices unchanged')
    return result
