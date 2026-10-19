import csv
import io
import json
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI,HTTPException,UploadFile,File,Request,Query
from fastapi.responses import FileResponse,Response,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from sklearn.decomposition import PCA
from . import store,jobs,encoder,tagger,listener
from .config import ROOT,STATE,CONFIG
from .audio import probe,unit
from .pipeline import cached,KEY
from .compare import compare,ExactSearch
from .explain import explain
from .structure import recurrence
from . import dimensions
from .compute import serialized

@asynccontextmanager
async def lifespan(app):
    jobs.start()
    yield

app=FastAPI(title='SoundMap',lifespan=lifespan)

@app.middleware('http')
async def local_guard(request:Request,call_next):
    host=request.headers.get('host','').split(':')[0]
    if host not in ('127.0.0.1','localhost','testserver'):
        return JSONResponse({'detail':'Local host only'},status_code=403)
    if request.method in ('POST','DELETE','PUT','PATCH'):
        origin=request.headers.get('origin')
        if origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'):
            return JSONResponse({'detail':'Cross-origin writes are disabled'},status_code=403)
        try: size=int(request.headers.get('content-length','0'))
        except ValueError: return JSONResponse({'detail':'Invalid content length'},status_code=400)
        if size>(CONFIG['upload_mb']+1)*1024**2:
            return JSONResponse({'detail':'Upload exceeds limit'},status_code=413)
        if request.url.path=='/api/upload' and not request.headers.get('content-length'):
            return JSONResponse({'detail':'Uploads require Content-Length'},status_code=411)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    return response

@app.exception_handler(KeyError)
async def missing(request,error): return JSONResponse({'detail':'Track not found'},status_code=404)

def public(t):
    a=t['_analysis'] if '_analysis' in t else cached(t['id'])
    return {k:t[k] for k in ('id','title','source','metadata')} | {'ready':a is not None,
        'duration':a.get('duration') if a else None,'mert_ready':bool(a and a.get('mert')),
        'silent':bool(a and a['silent'])}

def ready(id_):
    store.track(id_)
    a=cached(id_)
    if a is None: raise HTTPException(409,'Track needs analysis')
    return a

@app.get('/api/status')
def status():
    with store.connect() as db:
        ratings=db.execute('SELECT COUNT(*) FROM ratings').fetchone()[0]
        events=[dict(r) for r in db.execute('SELECT * FROM import_events WHERE status != ? LIMIT 30',('imported',))]
        tagged=db.execute('SELECT COUNT(*) FROM annotations WHERE kind=? AND config_key=?',('instruments',tagger.KEY)).fetchone()[0]
    return {'encoder':encoder.status,'tracks':len(store.tracks()),'rating_count':ratings,
            'ranking_model':'untrained' if not (STATE/'ranking-model.json').exists() else 'trained; see experiment report',
            'instrument_tagging':{**tagger.status,'cached_tracks':tagged} if CONFIG.get('tagging_enabled') else {'state':'disabled','cached_tracks':tagged},'ollama_configured':bool(CONFIG.get('ollama_model')),
            'fma_archive_complete':Path(CONFIG['archive']).is_file(), 'import_events':events,
            'upload_mb':CONFIG['upload_mb'],'configuration_key':KEY}

@app.get('/api/tracks')
def library(genre:str='',source:str='',q:str=''):
    with store.connect() as db:
        rows=db.execute("SELECT t.id,t.title,t.source,t.metadata,json_extract(a.result,'$.duration') AS duration,json_extract(a.result,'$.silent') AS silent,json_type(a.result,'$.mert.layers') IS NOT NULL AS mert_ready,a.track_id IS NOT NULL AS ready FROM tracks t LEFT JOIN analyses a ON a.track_id=t.id AND a.config_key=? WHERE (?='' OR (?='personal' AND t.source!='fma') OR t.source=?)",(KEY,source,source,source)).fetchall()
    result=[]
    for row in rows:
        t=dict(row);t['metadata']=json.loads(t['metadata'])
        if source and (t['source']=='fma' if source=='personal' else t['source']!=source):continue
        if genre and t['metadata'].get('genre')!=genre:continue
        if q.casefold() not in t['title'].casefold():continue
        for key in ('silent','mert_ready','ready'):t[key]=bool(t[key])
        result.append(t)
    return result


@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    ext=Path(file.filename or '').suffix.lower()
    if ext not in ('.mp3','.wav'): raise HTTPException(415,'Use MP3 or WAV')
    path=STATE/'uploads'/(uuid.uuid4().hex+ext)
    size=0
    try:
        with path.open('wb') as f:
            while block:=await file.read(1024**2):
                size+=len(block)
                if size>CONFIG['upload_mb']*1024**2: raise HTTPException(413,'Upload exceeds limit')
                f.write(block)
        probe(path)
        id_,new=store.register(path,'upload',title=Path(file.filename).stem[:200])
        if not new: path.unlink()
        try: job=jobs.submit(id_)
        except RuntimeError as e:
            return {'track':public(store.track(id_)),'job':None,'duplicate':not new,
                    'queue_error':str(e)+'; recording saved in the library. Retry Process recording.'}
        return {'track':public(store.track(id_)),'job':job,'duplicate':not new}
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422,str(e))
    except Exception:
        # A registered upload is kept if scheduling fails and can be retried.
        with store.connect() as db: registered=db.execute('SELECT 1 FROM tracks WHERE path=?',(str(path.resolve()),)).fetchone()
        if not registered: path.unlink(missing_ok=True)
        raise
    finally: await file.close()

@app.post('/api/tracks/{id_}/analyze')
def analyze_track(id_:str):
    try: return {'job':jobs.submit(id_)}
    except RuntimeError as e: raise HTTPException(429,str(e))

@app.get('/api/jobs')
def job_list():
    with store.connect() as db: return [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 100')]

@app.get('/api/tracks/{id_}/analysis')
def analysis(id_:str):
    a=ready(id_)
    tags=tagger.cached(id_)
    return {**a,'instrument_predictions':tags,'recurrence':recurrence(a),
            'dimensions':dimensions.cached(id_),
            'model_predictions':{'instruments':tags,'reason':'Separate AST hypotheses; MERT does not identify instruments.' if tags else
                                'Separate instrument predictions have not been calculated for this recording.'}}

@app.get('/api/tracks/{id_}/instruments')
def instruments(id_:str):
    store.track(id_)
    return tagger.cached(id_) or {'state':'not_analyzed','labels':[],
        'reason':'Separate instrument model has not processed this recording; use Predict instruments.'}

@app.post('/api/tracks/{id_}/instruments')
def predict_instruments(id_:str):
    if not CONFIG.get('tagging_enabled'):raise HTTPException(409,'Instrument model disabled')
    try:return {'job':jobs.submit(id_,'tagging')}
    except RuntimeError as e:raise HTTPException(429,str(e))

@app.get('/api/tracks/{id_}/audio')
def audio(id_:str):
    t=store.track(id_)
    if not Path(t['path']).is_file(): raise HTTPException(404,'Audio file no longer available')
    return FileResponse(t['path'],media_type='audio/mpeg' if t['path'].lower().endswith('.mp3') else 'audio/wav')

class Pair(BaseModel):
    a:str
    b:str

class ComparisonPair(Pair):
    weights:Optional[dict[str,float]]=None

@app.post('/api/tracks/{id_}/dimensions')
def measure_dimensions(id_:str):
    ready(id_)
    try:return {'job':jobs.submit(id_,'dimensions')}
    except RuntimeError as e:raise HTTPException(429,str(e))

@app.get('/api/tracks/{id_}/codec')
def codec_result(id_:str):
    from .codec import cached as codec_cached
    store.track(id_)
    result=codec_cached(id_)
    return {k:v for k,v in result.items() if k!='histogram'} if result else {'state':'unavailable','reason':'EnCodec analysis has not run.'}

@app.post('/api/tracks/{id_}/codec')
def codec_job(id_:str):
    if not (STATE/'models'/'encodec-manifest.json').exists():raise HTTPException(409,'EnCodec unavailable; run setup-codec.ps1 once.')
    try:return {'job':jobs.submit(id_,'codec')}
    except RuntimeError as e:raise HTTPException(429,str(e))

@app.post('/api/compare')
def comparison(pair:ComparisonPair,generation:bool=True):
    if pair.weights is not None:
        if any(k not in dimensions.NAMES or not np.isfinite(v) or not 0<=v<=5 for k,v in pair.weights.items()):
            raise HTTPException(422,'Dimension weights must be known names and finite numbers from 0 to 5.')
    result=compare(ready(pair.a),ready(pair.b))
    from .genre_model import affinity
    result['genre_affinity']=affinity(ready(pair.a),ready(pair.b))
    from .pseudo_model import predict
    result['machine_proxy']=predict(ready(pair.a),ready(pair.b))
    from .codec import compare as codec_compare
    result['codec_comparison']=codec_compare(pair.a,pair.b)
    from .tonal_detail import get as tonal_detail
    result['tonal_detail']={'a':tonal_detail(pair.a),'b':tonal_detail(pair.b)}
    from .source_detail import compare as source_compare
    result['source_detail']=source_compare(pair.a,pair.b)
    result.update(instrument_a=tagger.cached(pair.a),instrument_b=tagger.cached(pair.b))
    result['dimension_comparison']=dimensions.compare_dimensions(pair.a,pair.b,pair.weights)
    return {**result,'a':public(store.track(pair.a)),'b':public(store.track(pair.b)),'explanation':explain(result,generation)}

@app.get('/api/evaluation/status')
def evaluation_status():
    from .experiments import agreement
    genre_path=ROOT/'reports'/'genre.json'
    genre=json.loads(genre_path.read_text(encoding='utf-8')) if genre_path.exists() else None
    proxy_path=ROOT/'reports'/'genre-model.json'
    proxy=json.loads(proxy_path.read_text()) if proxy_path.exists() else None
    machine_path=ROOT/'reports'/'pseudo-model.json'
    machine=json.loads(machine_path.read_text()) if machine_path.exists() else None
    external_path=ROOT/'reports'/'external-validation.json'
    external=json.loads(external_path.read_text(encoding='utf-8')) if external_path.exists() else None
    return {'external_validation':external,'machine_model':machine,'genre_model':proxy,'training':listener.readiness(),'agreement':agreement(),
            'genre':genre,
            'training_optional':True,'message':'Acoustic + pretrained MERT similarity works without listener training.'}

@app.post('/api/evaluation/train')
def train_listener():
    report=listener.readiness()
    if not report['ready']:raise HTTPException(409,report['reason'])
    try:return {'job':jobs.submit(None,'ranking')}
    except RuntimeError as e:raise HTTPException(429,str(e))

@app.get('/api/tracks/{id_}/neighbors')
def neighbors(id_:str,limit:int=Query(default=10,ge=1,le=50),genre:str='',source:str=''):
    query=store.track(id_)
    query_analysis=ready(id_)
    seen={query['hash']}; decoded_seen={query_analysis.get('decoded_hash')}; candidates=[]
    for t in store.tracks(KEY,source or None):
        if t['hash'] in seen or (genre and t['metadata'].get('genre')!=genre) or (source and (t['source']=='fma' if source=='personal' else t['source']!=source)): continue
        a=t['_analysis']
        if a and not a['silent'] and a.get('decoded_hash') not in decoded_seen:
            seen.add(t['hash']); candidates.append((t['id'],a))
            decoded_seen.add(a.get('decoded_hash'))
    found=ExactSearch().search(query_analysis,candidates,int(limit))
    return [{'track':public(store.track(r['id'])),'overall':r['overall'],'acoustic':r['acoustic'],'mert':r['mert'],
             'learned_index':r.get('learned_index')} for r in found]

@app.get('/api/map')
def music_map(genre:str='',source:str=''):
    rows=[t for t in store.tracks(KEY,source or None) if t['_analysis'] and not t['_analysis']['silent']
          and (not genre or t['metadata'].get('genre')==genre) and (not source or (t['source']!='fma' if source=='personal' else source==t['source']))]
    all_mert=bool(rows) and all(t['_analysis'].get('mert') for t in rows)
    vectors=[t['_analysis']['mert']['vector'] if all_mert else unit(t['_analysis']['timbre']).tolist() for t in rows]
    if len(rows)<2: points=np.zeros((len(rows),2)); variance=[0,0]
    else:
        pca=PCA(n_components=2,svd_solver='full'); points=pca.fit_transform(vectors)
        variance=np.nan_to_num(pca.explained_variance_ratio_).tolist()
    return {'points':[{'track':public(t),'x':float(p[0]),'y':float(p[1])} for t,p in zip(rows,points)],
            'method':'PCA of frozen MERT embeddings' if all_mert else 'PCA of normalized acoustic timbre descriptors',
            'explained_variance':variance,'note':'Two-dimensional projection discards information. Distances can distort sound similarity; this projection does not establish stable genre clusters.'}

class Rating(Pair):
    rater:Optional[str]=Field(default=None,max_length=100)
    session:str=Field(default='default',max_length=100)
    rhythm:Optional[int]=Field(default=None,ge=1,le=5)
    tune:Optional[int]=Field(default=None,ge=1,le=5)
    instruments:Optional[int]=Field(default=None,ge=1,le=5)
    overall:Optional[int]=Field(default=None,ge=1,le=5)

@app.post('/api/ratings')
def rate(r:Rating):
    store.track(r.a); store.track(r.b)
    if r.a==r.b: raise HTTPException(422,'Choose distinct tracks for listener judgments')
    if all(getattr(r,k) is None for k in ('rhythm','tune','instruments','overall')): raise HTTPException(422,'Rate at least one dimension')
    id_=uuid.uuid4().hex
    with store.connect() as db: db.execute('INSERT INTO ratings(id,a,b,rater,session,rhythm,tune,instruments,overall) VALUES(?,?,?,?,?,?,?,?,?)',
        (id_,r.a,r.b,r.rater,r.session,r.rhythm,r.tune,r.instruments,r.overall))
    return {'id':id_}

@app.get('/api/ratings/next')
def next_pair(session:str=Query(default='default',max_length=100),mode:str=Query(default='general',pattern='^(general|training)$'),rater:Optional[str]=Query(default=None,max_length=100)):
    rows=[t for t in store.tracks(KEY) if t['_analysis'] and not t['_analysis']['silent']]
    if len(rows)<2: raise HTTPException(409,'Analyze at least two non-silent tracks')
    with store.connect() as db:
        prior=[dict(r) for r in db.execute('SELECT a,b FROM ratings WHERE rater=? ORDER BY created,id',(rater,))] if rater else [dict(r) for r in db.execute("SELECT a,b FROM ratings WHERE session=? AND (rater IS NULL OR rater='') ORDER BY created,id",(session,))]
    if prior and len(prior)%6==5:
        a,b=prior[0]['a'],prior[0]['b']
        return {'a':a,'b':b,'repeat':True}
    if mode=='training':
        try:return listener.next_training_pair(session,prior)
        except ValueError as e:raise HTTPException(409,str(e))
    a,b=random.Random((rater or session)+str(len(prior))).sample(rows,2)
    return {'a':a['id'],'b':b['id'],'repeat':False}

@app.get('/api/ratings/export')
def export():
    with store.connect() as db: rows=[dict(r) for r in db.execute('SELECT * FROM ratings ORDER BY created,id')]
    output=io.StringIO(); writer=csv.DictWriter(output,fieldnames=['id','a','b','rater','session','rhythm','tune','instruments','overall','created'])
    writer.writeheader(); writer.writerows(rows)
    return Response(output.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="soundmap-ratings.csv"'})

@app.delete('/api/tracks/{id_}/cache')
@serialized
def delete_cache(id_:str):
    import shutil
    t=store.track(id_)
    with store.connect() as db:
        busy=db.execute("SELECT 1 FROM jobs WHERE (track_id=? OR kind='ranking') AND status IN ('running','queued')",(id_,)).fetchone()
        if busy: raise HTTPException(409,'Wait for active analysis before deletion')
        db.execute('DELETE FROM analyses WHERE track_id=?',(id_,))
        db.execute('DELETE FROM annotations WHERE track_id=?',(id_,))
    path=(STATE/'cache'/t['hash']).resolve()
    if path.parent != (STATE/'cache').resolve(): raise HTTPException(400,'Invalid cache path')
    if path.exists(): shutil.rmtree(path)
    explanation_cache=(STATE/'explanations').resolve()
    if explanation_cache.parent==STATE.resolve() and explanation_cache.exists():shutil.rmtree(explanation_cache)
    (STATE/'ranking-model.json').unlink(missing_ok=True)
    return {'deleted':True}

@app.delete('/api/tracks/{id_}')
@serialized
def delete_track(id_:str):
    t=store.track(id_); delete_cache(id_)
    with store.connect() as db: db.execute('DELETE FROM tracks WHERE id=?',(id_,))
    path=Path(t['path']).resolve()
    if t['source']=='upload' and path.parent==(STATE/'uploads').resolve(): path.unlink(missing_ok=True)
    return {'deleted':True,'original_preserved':t['source']!='upload'}

@app.get('/')
def index(): return FileResponse(ROOT/'static'/'index.html')

app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
