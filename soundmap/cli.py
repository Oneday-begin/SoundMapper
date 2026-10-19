import argparse
import json
import time
import sys
from pathlib import Path
from .config import ROOT
from . import store

def save(name,result):
    (ROOT/'reports').mkdir(exist_ok=True)
    path=ROOT/'reports'/name
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='SoundMap trusted local tools')
    parser.add_argument('command',choices=['samples','metadata','extract','fma','analyze','dimensions','tag','verify-mert','benchmark','genre','ranking','agreement','annotate'])
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--source',choices=['sample','fma','upload'])
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--deep',action='store_true',help='Sample vocal separation/pitch; CPU intensive')
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    from . import importer
    if args.command=='annotate':
        if not args.manifest: parser.error('--manifest required')
        manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
        changes=[]
        for t in store.tracks():
            if t['hash'] not in manifest: continue
            item=manifest[t['hash']]
            if not isinstance(item.get('artist_id'),str) or not item['artist_id']: raise ValueError('Nonempty string artist_id required')
            t['metadata'].update({k:item[k] for k in ('artist_id','artist','genre','language') if k in item})
            with store.connect() as db: db.execute('UPDATE tracks SET metadata=? WHERE id=?',(json.dumps(t['metadata']),t['id']))
            changes.append({'id':t['id'],'metadata':t['metadata']})
        save('annotations.json',changes)
    elif args.command=='samples': save('sample-import.json',importer.import_samples())
    elif args.command=='metadata': save('metadata-audit.json',importer.audit_metadata())
    elif args.command=='extract':
        result=importer.extract_archive();save('fma-extraction.json',result)
        if result['status']=='pending': raise SystemExit(3)
    elif args.command=='fma': save('fma-import.json',importer.import_fma(args.limit,args.seed))
    elif args.command=='tag':
        from .tagger import analyze
        results=[]
        tracks=[t for t in store.tracks() if not args.source or t['source']==args.source]
        for index,t in enumerate(tracks,1):
            try:
                r=analyze(t['id'],lambda p,m:print(f"[{index}/{len(tracks)}] {p:.0%} | {m}",flush=True),args.force)
                results.append({'id':t['id'],'title':t['title'],**r})
            except Exception as e:results.append({'id':t['id'],'error':str(e)})
        save('tagging-'+(args.source or 'all')+'.json',results)
        if any(r.get('error') for r in results):raise SystemExit(1)
    elif args.command=='dimensions':
        from .dimensions import analyze
        results=[]
        for t in store.tracks():
            if args.source and t['source']!=args.source:continue
            try:
                a=analyze(t['id'],lambda p,m:print(t['title'],m,flush=True),deep=args.deep,force=args.force)
                results.append({'id':t['id'],'title':t['title'],'processing_seconds':a['processing_seconds'],
                    'peak_rss_mb':a['peak_rss_mb'],'cache_hit':a['cache_hit'],'deep':bool(a.get('deep')),'error':a.get('deep_error')})
            except Exception as e:results.append({'id':t['id'],'error':str(e)})
        save('dimensions-'+(args.source or 'all')+'.json',results)
        if any(r.get('error') for r in results):raise SystemExit(1)
    elif args.command=='analyze':
        from .pipeline import analyze
        results=[]
        tracks=[t for t in store.tracks() if not args.source or t['source']==args.source]
        for index,t in enumerate(tracks,1):
            last=[0]
            def progress(p,m):
                if time.monotonic()-last[0]>10 or p==1:
                    print(f"[{index}/{len(tracks)}] {t['title']} | {p:.0%} | {m}",flush=True); last[0]=time.monotonic()
            try:
                a=analyze(t['id'],progress,args.force)
                results.append({'id':t['id'],'title':t['title'],'duration':a['duration'],
                    'mert_ready':bool(a['mert']),'mert_error':a.get('mert_error'),
                    'mert_seconds':a['mert']['seconds'] if a['mert'] else None,
                    'windows':a['mert']['windows'] if a['mert'] else 0,'tempo':a['tempo'],
                    **a['measurements'],'cache_hit':a['cache_hit']})
            except Exception as e: results.append({'id':t['id'],'error':str(e)})
        save('analysis-'+(args.source or 'all')+'.json',results)
        if any(r.get('error') or r.get('mert_error') for r in results): raise SystemExit(1)
    elif args.command=='verify-mert':
        from .encoder import embed,status
        from .audio import decode
        from .pipeline import MemoryMeter
        import numpy as np
        tracks=store.tracks()
        if not tracks: raise RuntimeError('Import samples first')
        y,info=decode(tracks[0]['path']); y=y[:120000]
        with MemoryMeter() as meter:
            start=time.perf_counter(); a=embed(y); first=time.perf_counter()-start
            start=time.perf_counter(); b=embed(y); repeat=time.perf_counter()-start
        save('mert-verification.json',{'input_title':tracks[0]['title'],'device':'cpu','shape':list(a.shape),
            'layers':[4,8,12],'finite':bool(np.isfinite(a).all()),'repeat_max_abs_difference':float(np.max(np.abs(a-b))),
            'repeat_allclose_1e-6':bool(np.allclose(a,b,atol=1e-6,rtol=1e-6)),
            'first_seconds_including_load':first,'repeat_inference_seconds':repeat,'peak_rss_mb':meter.peak/1024**2,
            'model_status':status,'window_seconds':len(y)/24000})
    else:
        from . import experiments
        save(args.command+'.json',getattr(experiments,args.command)())

if __name__=='__main__': main()
