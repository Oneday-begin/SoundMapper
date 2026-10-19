import json
import hashlib
import re
import threading
import time
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import ROOT,CONFIG,STATE

REFERENCES=json.loads((ROOT/'references.json').read_text())
_vectorizer=TfidfVectorizer()
_matrix=_vectorizer.fit_transform([r['text'] for r in REFERENCES])
_generation_lock=threading.BoundedSemaphore(1)

def generate(evidence):
    """Ollama selects/orders cited evidence; unsupported free-form claims cannot enter output."""
    facts={f'{kind}:{i}':{'kind':kind,'text':s} for kind in ('measurements','predictions','concepts') for i,s in enumerate(evidence[kind])}
    kinds=('measurements','predictions','concepts')
    schema={'type':'object','properties':{kind:{'type':'array','items':{'type':'string','enum':[i for i,f in facts.items() if f['kind']==kind]},'minItems':1,'maxItems':2} for kind in kinds},'required':list(kinds),'additionalProperties':False}
    model=CONFIG['ollama_model'];expected=CONFIG.get('ollama_digest')
    key=hashlib.sha256(json.dumps({'version':2,'model':model,'digest':expected,'facts':facts,'references':evidence['references']},sort_keys=True).encode()).hexdigest()
    folder=STATE/'explanations';path=folder/(key+'.json')
    if path.exists():return {**json.loads(path.read_text(encoding='utf-8')),'cache_hit':True}
    if not _generation_lock.acquire(blocking=False):return {'enabled':False,'reason':'Ollama is busy; deterministic evidence is available now.'}
    start=time.perf_counter()
    try:
        tags=httpx.get('http://127.0.0.1:11434/api/tags',timeout=3);tags.raise_for_status()
        entry=next((m for m in tags.json()['models'] if m['name']==model),None)
        if entry is None:raise RuntimeError('Configured local model is absent; no download is started')
        if expected and entry['digest']!=expected:raise RuntimeError('Ollama model digest changed; review/update configuration before using it')
        prompt='Arrange a concise music comparison using only these supplied facts and retrieved references. Return JSON with measurements, predictions, and concepts arrays. Select 1 or 2 unique fact IDs per category. Prefer relevant cited evidence. Do not add prose, instruments, identities, biographies or listener judgments.\n'+json.dumps({'facts':facts,'references':evidence['references']})
        r=httpx.post('http://127.0.0.1:11434/api/generate',json={'model':model,'prompt':prompt,'format':schema,
            'think':False,'stream':False,'keep_alive':'5m','options':{'temperature':0,'seed':2026,'num_predict':192,'num_ctx':2048,'num_thread':4,'num_gpu':0}},timeout=60)
        r.raise_for_status();raw=r.json();answer=json.loads(raw['response'])
        if set(answer)!=set(kinds) or any(not isinstance(answer[k],list) or not 1<=len(answer[k])<=2 for k in kinds):raise ValueError('Generation omitted a required evidence category')
        order=[i for k in kinds for i in answer[k]]
        if any(not isinstance(i,str) or i not in facts or facts[i]['kind']!=k for k in kinds for i in answer[k]) or len(set(order))!=len(order):raise ValueError('Unsupported generation structure or fact IDs')
        selected=[facts[i] for i in order]
        result={'enabled':True,'model':model,'digest':entry['digest'],'selected_evidence':selected,
            'text':'\n'.join(f"{f['kind'].capitalize()}: {f['text']}" for f in selected),'seconds':time.perf_counter()-start,
            'ollama_total_seconds':raw.get('total_duration',0)/1e9,'eval_tokens':raw.get('eval_count'),
            'cache_hit':False,'support_check':'Every sentence is copied verbatim from displayed evidence; category and source IDs validated.',
            'note':'Local Ollama arranged the cited evidence. This constrained summary adds no independent claims.'}
        folder.mkdir(exist_ok=True);temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2),encoding='utf-8');temp.replace(path)
        return result
    except Exception as e:return {'enabled':False,'reason':'Ollama unavailable or output rejected; deterministic fallback. '+str(e),'seconds':time.perf_counter()-start}
    finally:_generation_lock.release()

def retrieve(query,limit=4):
    score=cosine_similarity(_vectorizer.transform([query]),_matrix)[0]
    return [{**REFERENCES[i],'retrieval_score':float(score[i])} for i in score.argsort()[::-1][:limit]]

def explain(result,generate_summary=True):
    refs=retrieve('rhythm tempo chroma harmonic pitch pattern timbre spectral MERT embedding similarity AST instrument tagging',6)
    if not result['available']:
        return {'measurements':[result['reason']],'predictions':[], 'concepts':[], 'references':refs,
                'generation':{'enabled':False,'reason':'No meaningful audio evidence'}}
    def tempo(value): return 'unavailable' if value is None else f'{value:.1f}'
    measurements=[f"Whole-song sound index: {result['overall']:.1f}/100. Acoustic baseline: {result['acoustic']:.1f}/100.",
        f"Onset-pattern rhythm index: {result['rhythm']:.1f}/100; estimated tempos: {tempo(result['tempo_a'])} and {tempo(result['tempo_b'])} BPM. [tempo]",
        f"Ordered chroma alignment index: {result['tune']:.1f}/100; mean DTW cost {result['harmonic_dtw_cost']:.4f}. This is a harmonic/pitch-pattern proxy. [chroma] [dtw]",
        f"MFCC/spectral timbre index: {result['timbre']:.1f}/100. Acoustic features do not establish instrument identities. [timbre]"]
    if result['passages']: measurements.append(f"Strongest local window index: {result['passages'][0]['score']:.1f}/100, compared separately from the whole-song index.")
    detailed=result.get('dimension_comparison',{})
    if detailed.get('overall') is not None:
        measurements.insert(0,f"Selected-feature descriptive index: {detailed['overall']:.1f}/100; {detailed['coverage']:.0%} selected-weight coverage. The older acoustic/MERT index below uses a different aggregation.")
    predictions=[]
    machine=result.get('machine_proxy',{})
    if machine.get('available'):
        predictions.append(f"Machine-trained acoustic-rule estimate: {machine['index']:.1f}/100. It approximates four calculated features; human agreement is untested.")
    codec=result.get('codec_comparison',{})
    if codec.get('available'):
        predictions.append(f"EnCodec sampled texture affinity: {codec['score']:.1f}/100. Based on audio-token distributions, not instrument identities or verified vocal style. [encodec]")
    proxy=result.get('genre_affinity',{})
    if proxy.get('available'):
        predictions.append(f"Learned FMA genre-profile affinity: {proxy['index']:.1f}/100. This compares predicted genre profiles, not human judgments or individual feature matches. Genre held-out accuracy: {proxy['test']['accuracy']:.1%}. [mert]")
    if result['mert'] is not None: predictions.append(f"Frozen MERT representation index: {result['mert']:.1f}/100, from rescaled cosine similarity. MERT itself does not identify instruments. [mert]")
    else: predictions.append('MERT unavailable for at least one input; overall uses acoustic features only.')
    for side in ('a','b'):
        tags=result.get('instrument_'+side)
        if tags and tags.get('state')=='ready':
            names=', '.join(f"{r['label']} (tag score {r['score']:.3f})" for r in tags['labels'][:3])
            predictions.append(f"Separate AST model, recording {side.upper()}: "+(names or 'no instrument tag above the fixed display threshold')+'. Predictions cover sampled excerpts and are uncalibrated; not similarity scores or verified identities. [ast]')
            if tags.get('passage_predictions'):
                p=tags['passage_predictions'][0]
                predictions.append(f"AST excerpt suggestion for {side.upper()}: {p['label']}, tag score {p['score']:.3f}, at {p['excerpt']['start']:.1f}–{p['excerpt']['end']:.1f} seconds. This is a local model hypothesis, not a verified instrument identity. [ast]")
    out={'measurements':measurements,'predictions':predictions,
         'concepts':['Tempo can have half/double-time ambiguity. [tempo]',
                     'Similar pitch-class patterns can occur in different melodies. [chroma]',
                     'Timbre is affected by recording, mixing, and mastering. These features do not identify instruments. [timbre]'],
         'references':refs,'generation':{'enabled':False,'reason':'Generative RAG disabled; deterministic evidence and retrieved sources are shown.'}}
    cited=set(re.findall(r'\[([a-z]+)\]',' '.join(measurements+predictions+out['concepts'])))
    have={r['id'] for r in refs}
    refs.extend(r for r in REFERENCES if r['id'] in cited-have)
    if CONFIG.get('ollama_model') and generate_summary:out['generation']=generate(out)
    elif not generate_summary:out['generation']['reason']='Deterministic evidence requested; local generation skipped.'
    return out
