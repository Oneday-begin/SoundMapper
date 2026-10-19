"""Independent, versioned measurements; no inferred genre or structural labels."""
import hashlib,json,time
import numpy as np
import librosa
import soundfile as sf
from . import store
from .config import STATE,CONFIG
from .audio import decode,unit
from .pipeline import cached as base_cached,MemoryMeter

SETTINGS={'version':'dimensions-2','sr':24000,'cqt_hop':1024,'key_profiles':'Krumhansl major/minor; correlation gap >= .08',
 'dynamics':'non-silent 400ms RMS p95-p10 dB; crest factor; original amplitude',
 'groove':'beat-normalized onset phase histogram, 16 bins','deep_excerpts':3,'deep_seconds':8,
 'melody':'chronological 64-frame vocal-stem pYIN contour with voiced-mask alignment'}
KEY=hashlib.sha256(json.dumps(SETTINGS,sort_keys=True).encode()).hexdigest()[:20]
NAMES={'instrumentation':'Instrumentation','texture':'Sonic texture & production','vocals':'Vocal delivery',
 'dynamics':'Dynamic range','melody':'Melodic contour','harmony':'Harmonic progression','key':'Key and mode',
 'tempo':'Tempo (BPM)','groove':'Rhythm and groove'}
DEFAULT_WEIGHTS={'instrumentation':1,'texture':1,'vocals':1,'dynamics':1,'melody':1,'harmony':1,'key':1,'tempo':1,'groove':1}

def cached(id_):
    with store.connect() as db:r=db.execute('SELECT result FROM annotations WHERE track_id=? AND kind=? AND config_key=?',(id_,'dimensions',KEY)).fetchone()
    return json.loads(r[0]) if r else None

def tonal(chroma):
    profiles=[np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]),
              np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])]
    names=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'];candidates=[]
    for mode,p in zip(('major','minor'),profiles):
        for root in range(12):
            c=float(np.corrcoef(chroma,np.roll(p,root))[0,1]) if np.std(chroma)>1e-8 else 0
            candidates.append({'root':root,'label':names[root]+' '+mode,'mode':mode,'correlation':c})
    candidates.sort(key=lambda r:r['correlation'],reverse=True)
    gap=candidates[0]['correlation']-candidates[1]['correlation']
    return {'estimate':candidates[0] if gap>=.08 and candidates[0]['correlation']>=.5 else None,
            'candidates':candidates[:3],'correlation_gap':gap,'note':'Major/minor profile estimate only; gap is not a probability. Modulations and other modal systems are not identified.'}

def core(y,stereo=None,sr=24000):
    y=np.asarray(y,dtype=np.float32)
    if len(y)<sr:raise ValueError('At least one second required')
    if not np.isfinite(y).all():raise ValueError('Non-finite samples')
    duration=len(y)/sr
    rms=np.sqrt(np.mean(y*y));silent=rms<1e-6
    frame=int(.4*sr);levels=np.array([np.sqrt(np.mean(v*v)) for v in np.array_split(y,max(1,int(np.ceil(len(y)/frame))))])
    active=levels[levels>max(1e-6,float(levels.max())*.001)]
    db=20*np.log10(np.maximum(active,1e-10))
    dynamics={'rms_dbfs':float(20*np.log10(max(rms,1e-10))),
       'crest_db':float(20*np.log10(max(float(np.max(np.abs(y))),1e-10)/max(rms,1e-10))),
       'level_spread_db':float(np.percentile(db,95)-np.percentile(db,10)) if len(db)>1 else 0,
       'level_time':np.linspace(0,duration,len(levels),endpoint=False).tolist(),
       'level_dbfs':(20*np.log10(np.maximum(levels,1e-10))).tolist(),
       'note':'RMS level-spread proxy and crest factor; not standardized EBU loudness range. Silence gated; original amplitude preserved.'}
    texture={'stereo_width':None,'channel_correlation':None,'note':'Spectral descriptors plus original stereo side/mid energy; not named production-style recognition.'}
    if stereo is not None and stereo.shape[1]==2:
        mid=(stereo[:,0]+stereo[:,1])*.5;side=(stereo[:,0]-stereo[:,1])*.5
        texture['stereo_width']=float(np.sqrt(np.mean(side*side))/max(np.sqrt(np.mean(mid*mid)),1e-8))
        texture['channel_correlation']=float(np.clip(np.corrcoef(stereo.T)[0,1],-1,1)) if np.std(stereo[:,0])>1e-8 and np.std(stereo[:,1])>1e-8 else None
    if silent:return {'silent':True,'duration':duration,'dynamics':dynamics,'texture':texture,'tempo':None,'groove':None,'harmony':None,'key':None}
    # CQT retains chronological pitch-class evidence; it is not an isolated melody.
    chroma=librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=1024,n_octaves=6,tuning=0)
    onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=512)
    bpm,beats=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=512)
    tempo=float(np.asarray(bpm).ravel()[0]) if len(beats)>=3 else None
    times=librosa.frames_to_time(beats,sr=sr,hop_length=512)
    phases=np.zeros(16);beat_strength=[]
    for a,b in zip(beats[:-1],beats[1:]):
        if b<=a:continue
        part=onset[a:b];bins=np.minimum(15,(np.arange(len(part))*16/len(part)).astype(int))
        np.add.at(phases,bins,part);beat_strength.append(float(np.sum(part)))
    # Bound DTW while retaining ordered CQT samples and their exact grid times.
    idx=np.unique(np.rint(np.linspace(0,chroma.shape[1]-1,min(180,chroma.shape[1]))).astype(int))
    sequence=[unit(chroma[:,i]).tolist() for i in idx]
    return {'silent':False,'duration':duration,'dynamics':dynamics,'texture':texture,'tempo':tempo,
       'beat_times':times.tolist(),'groove':{'phase_histogram':unit(phases).tolist(),'beats':len(beats),
         'note':'Beat-normalized onset phase proxy; no validated swing, meter or syncopation labels.'} if len(beats)>=3 else None,
       'harmony':{'sequence':sequence,'time':(idx*1024/sr).tolist(),'profile':unit(chroma.mean(axis=1)).tolist(),
         'note':'Chronological CQT chroma; combines melody and accompaniment. No chord labels.'},'key':tonal(chroma.mean(axis=1))}

from .compute import serialized

@serialized
def analyze(id_,progress=lambda p,m:None,deep=False,force=False):
    prior=cached(id_)
    if prior and not force and (not deep or prior.get('deep')):return {**prior,'cache_hit':True}
    t=store.track(id_);start=time.perf_counter()
    with MemoryMeter() as meter:
        if prior and not force:r=prior
        else:
            progress(.05,'Measuring dynamics, stereo, CQT and beat patterns')
            y,_=decode(t['path'])
            with sf.SoundFile(t['path']) as f:stereo=f.read(dtype='float32',always_2d=True) if f.channels==2 else None
            r=core(y,stereo);r.update(settings=SETTINGS,config_key=KEY,deep=None)
        if deep and not r['silent']:
            from .separation import analyze as separate
            progress(.45,'Estimating vocal stems and chronological pitch contours')
            try:r['deep']=separate(t,progress)
            except Exception as e:r['deep_error']=str(e)
        r.update(cache_hit=False,processing_seconds=time.perf_counter()-start,peak_rss_mb=meter.peak/1024**2)
    with store.connect() as db:db.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?,?)',(id_,'dimensions',KEY,json.dumps(r,allow_nan=False)))
    progress(1,'Dimension measurements cached'+('; vocal/melody model unavailable' if r.get('deep_error') else ''))
    return r

def exp_distance(a,b,scales):
    return float(100*np.exp(-np.linalg.norm((np.asarray(a)-np.asarray(b))/scales)/np.sqrt(len(scales))))

def tag_profile(id_,category):
    from . import tagger
    t=store.track(id_);summary=tagger.cached(id_)
    if not summary or summary['state']!='ready':return None
    folder=STATE/'cache'/t['hash']/('ast-'+tagger.KEY)
    values=[]
    for p in sorted(folder.glob('window-*.npz')):
        with np.load(p,allow_pickle=False) as d:values.append(d['scores'])
    if not values:return None
    _,processor=tagger.load()
    labels=tagger._model.config.id2label
    allowed={x['name'] for x in tagger.ONTOLOGY['instrument_classes']} if category=='instrumentation' else {'Singing','Male singing','Female singing','Child singing','Rapping','Speech','Humming','Chant','Choir','Yodeling','Whistling'}
    means=np.mean(values,axis=0);indices=[i for i,name in labels.items() if name in allowed]
    scores=np.array([means[i] for i in indices]);top=sorted([{'label':labels[i],'score':float(means[i])} for i in indices],key=lambda r:r['score'],reverse=True)[:5]
    return {'vector':scores.tolist(),'top':top,'supported':bool(np.max(scores)>=.2),'windows':len(summary['sampled_windows']),
      'note':'Frozen AST sampled-excerpt hypotheses; 0.20 support gate, uncalibrated and not ground-truth labels.'}

def dtw_index(a,b):
    from .compare import harmonic
    return harmonic(a,b)[0]*100

def compare_dimensions(id_a,id_b,weights=None):
    from .compare import cosine,timbre
    a,b=cached(id_a),cached(id_b);aa,bb=base_cached(id_a),base_cached(id_b)
    rows=[]
    def add(k,score,evidence,method,quality='measurement',passages=None,aggregate=True):
        rows.append({'id':k,'name':NAMES[k],'score':None if score is None else float(np.clip(score,0,100)),
         'available':score is not None,'aggregate':bool(aggregate),'evidence':evidence,'method':method,'quality':quality,'passages':passages or []})
    if not a or not b:return {'ready':False,'missing':[i for i,r in ((id_a,a),(id_b,b)) if not r],'rows':[],'overall':None,'version':SETTINGS['version']}
    if a['silent'] or b['silent']:return {'ready':False,'missing':[],'rows':[],'overall':None,'reason':'Silence has no meaningful musical comparison.'}
    profiles={}
    for k in ('instrumentation','vocals'):
        pa,pb=tag_profile(id_a,k),tag_profile(id_b,k);profiles[k]=(pa,pb)
        available=pa is not None and pb is not None
        score=100*max(0,cosine(pa['vector'],pb['vector'])) if available else None
        reason=None if available and pa['supported'] and pb['supported'] else (
            'AST profiles are available, but one or both tracks have no class above the 0.20 display threshold; treat this profile similarity as weak evidence.'
            if available else 'AST profile is missing for one or both tracks; no score is fabricated.')
        def compact(p):
            return None if p is None else {q:p[q] for q in ('top','supported','windows','note')}
        add(k,score,{'a':compact(pa),'b':compact(pb),'reason':reason},
            'Cosine of sampled AST class activation profiles; independent uncalibrated tag scores shown as hypotheses',
            'model hypothesis',aggregate=bool(available and pa['supported'] and pb['supported']))
    ta,tb=a['texture'],b['texture'];score=timbre(aa['timbre'],bb['timbre'])*100
    if ta['stereo_width'] is not None and tb['stereo_width'] is not None:
        score=.7*score+.3*100*np.exp(-abs(np.log1p(ta['stereo_width'])-np.log1p(tb['stereo_width'])))
    add('texture',score,{'a':ta,'b':tb},'70% spectral/MFCC timbre, 30% log stereo-width similarity when both stereo; otherwise timbre','texture proxy')
    da,db=a['dynamics'],b['dynamics']
    add('dynamics',exp_distance([da['crest_db'],da['level_spread_db']],[db['crest_db'],db['level_spread_db']],np.array([6,6])),
        {'a':da,'b':db},'Exponential distance of crest factor and RMS spread, fixed 6 dB scales; loudness difference reported separately','dynamic-range proxy')
    add('harmony',dtw_index(a['harmony']['sequence'],b['harmony']['sequence']),
        {'a':{'profile':a['harmony']['profile'],'frames':len(a['harmony']['sequence']),'duration':a['duration']},
         'b':{'profile':b['harmony']['profile'],'frames':len(b['harmony']['sequence']),'duration':b['duration']},
         'note':a['harmony']['note']},
        'Ordered CQT chroma DTW; pitch-class patterns, not chord recognition','harmonic proxy')
    ka,kb=a['key']['estimate'],b['key']['estimate']
    keyscore=100 if ka and kb and ka['root']==kb['root'] and ka['mode']==kb['mode'] else 0 if ka and kb else None
    add('key',keyscore,{'a':a['key'],'b':b['key']},'Exact agreement of supported major/minor profile estimates; uncertain keys unavailable','tonal estimate')
    ba,bc=a['tempo'],b['tempo']
    tempo=100*np.exp(-abs(np.log2(ba/bc))/.25) if ba and bc else None
    compatible=max(100*np.exp(-abs(np.log2(ba/(bc*k)))/.25) for k in (.5,1,2)) if ba and bc else None
    add('tempo',tempo,{'a_bpm':ba,'b_bpm':bc,'difference_bpm':abs(ba-bc) if ba and bc else None,'half_double_compatible_index':compatible},
        'Absolute log2 BPM distance, fixed quarter-octave scale. Half/double alternative shown separately','estimated beat rate')
    ga,gb=a['groove'],b['groove']
    add('groove',100*max(0,cosine(ga['phase_histogram'],gb['phase_histogram'])) if ga and gb else None,
        {'a':ga,'b':gb},'Cosine of 16-bin beat-normalized onset phase histograms','groove proxy')
    deepa,deepb=a.get('deep'),b.get('deep');candidates=[]
    if deepa and deepb:
        for wa in deepa['windows']:
            for wb in deepb['windows']:
                if wa['usable_pitch'] and wb['usable_pitch']:
                    # Median-relative semitones retain contour while ignoring tonic.
                    xa=np.array(wa['pitch_semitones']);xb=np.array(wb['pitch_semitones'])
                    va=np.array(wa['pitch_voiced'],dtype=bool);vb=np.array(wb['pitch_voiced'],dtype=bool)
                    cost=np.minimum(1,abs(xa[:,None]-xb[None,:])/6)+.15*(va[:,None]!=vb[None,:])
                    D,path=librosa.sequence.dtw(C=cost)
                    candidates.append({'score':float(max(0,100*(1-D[-1,-1]/len(path)))),
                        'a':{'start':wa['start'],'end':wa['end']},'b':{'start':wb['start'],'end':wb['end']}})
    melody=float(np.median([p['score'] for p in candidates])) if candidates else None
    def melody_summary(d):
        return None if not d else {'model':d.get('model'),'windows':[{'start':w['start'],'end':w['end'],
            'usable_pitch':w['usable_pitch'],'voiced_fraction':w['voiced_fraction'],
            'median_pitch_midi':w['median_pitch_midi']} for w in d.get('windows',[])],
            'sampled_seconds':d.get('sampled_seconds'),'note':d.get('note')}
    add('melody',melody,{'a':melody_summary(deepa),'b':melody_summary(deepb),
        'reason':None if candidates else 'No usable separated-vocal pitch evidence for both inputs.'},
        'Median pairwise DTW of median-relative pYIN vocal pitch; sampled excerpts only. Not validated predominant-melody recognition.',
        'estimated vocal-contour proxy',sorted(candidates,key=lambda p:p['score'],reverse=True)[:3])
    # Explicit user weights; unsupported dimensions remain absent. MERT never silently overrides them.
    weights={**DEFAULT_WEIGHTS,**(weights or {})};total=sum(weights[r['id']] for r in rows)
    active=[r for r in rows if r['available'] and r['aggregate'] and weights[r['id']]>0];available_weight=sum(weights[r['id']] for r in active)
    overall=sum(r['score']*weights[r['id']] for r in active)/available_weight if available_weight else None
    return {'ready':True,'deep_ready':bool(a.get('deep') and b.get('deep')),
       'tagging_ready':all(profiles[k][0] is not None and profiles[k][1] is not None for k in ('instrumentation','vocals')),
       'version':SETTINGS['version'],'rows':sorted(rows,key=lambda r:list(NAMES).index(r['id'])),
       'overall':overall,'weights':weights,'available_dimensions':len(active),
       'weak_dimensions':[r['id'] for r in rows if r['available'] and not r['aggregate']],
       'selected_dimensions':sum(w>0 for w in weights.values()),
       'coverage':available_weight/total if total else 0,'aggregation':'Weighted mean of available selected dimensions; missing dimensions omitted and coverage explicit.',
       'note':'Optional descriptive index, not calibrated perceptual similarity. Compare individual evidence first. No listener model was trained.'}
