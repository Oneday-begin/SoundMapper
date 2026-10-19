"""Trusted CLI data import; filesystem paths are never accepted by public APIs."""
import csv
import hashlib
import json
import random
import re
import subprocess
import zipfile
from pathlib import Path,PurePosixPath
from .config import CONFIG,ROOT
from . import store
from .audio import probe

SHA1='ade154f733639d52e35e32f5593efe5be76c6d70'

def event(path,status,reason=''):
    with store.connect() as db: db.execute('INSERT OR REPLACE INTO import_events VALUES(?,?,?)',(str(path),status,reason))

def import_samples():
    results=[]
    for p in sorted(Path(CONFIG['samples']).iterdir()):
        if p.suffix.lower() not in ('.mp3','.wav'): continue
        try:
            probe(p); id_,new=store.register(p,'sample')
            event(p,'imported'); results.append({'id':id_,'new':new,'filename':p.name})
        except Exception as e: event(p,'skipped',str(e)); results.append({'filename':p.name,'error':str(e)})
    return results

def metadata(path=None):
    path=Path(path or CONFIG['metadata'])/'tracks.csv'
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f)
        top=next(reader); sub=next(reader); index=next(reader)
        if index[0]!='track_id': raise ValueError('Expected FMA track_id third header row')
        keys=list(zip(top,sub))
        wanted={k:keys.index(k) for k in [('set','subset'),('set','split'),('artist','id'),('artist','name'),
                 ('track','genre_top'),('track','title'),('track','license'),('track','language_code')]}
        for row in reader:
            if row[wanted[('set','subset')]]!='small': continue
            yield {'track_id':int(row[0]),'split':row[wanted[('set','split')]],
                   'artist_id':row[wanted[('artist','id')]],'artist':row[wanted[('artist','name')]],
                   'genre':row[wanted[('track','genre_top')]],'title':row[wanted[('track','title')]],
                   'license':row[wanted[('track','license')]],'language':row[wanted[('track','language_code')]]}

def audit_metadata():
    rows=list(metadata()); groups={s:set() for s in ('training','validation','test')}
    counts={}
    for r in rows:
        groups[r['split']].add(r['artist_id'])
        key=r['split']+' / '+r['genre']; counts[key]=counts.get(key,0)+1
    overlap={f'{a}:{b}':sorted(groups[a]&groups[b]) for a,b in [('training','validation'),('training','test'),('validation','test')]}
    return {'small_tracks':len(rows),'split_genre_counts':counts,'artist_overlap_ids':overlap,
            'features_csv_exists':(Path(CONFIG['metadata'])/'features.csv').exists(),
            'genres_csv_exists':(Path(CONFIG['metadata'])/'genres.csv').exists(),
            'metadata_root':CONFIG['metadata']}

def safe_archive_paths(names):
    for name in names:
        p=PurePosixPath(name.replace('\\','/'))
        if p.is_absolute() or '..' in p.parts or ':' in name or not p.parts or p.parts[0]!='fma_small':
            raise ValueError('Unsafe/unexpected archive path: '+name)

def extract_archive():
    archive=Path(CONFIG['archive'])
    if archive.suffix.lower()!='.zip' or not archive.is_file():
        return {'status':'pending','reason':'Completed fma_small.zip absent; browser partial file is not read.'}
    # Reject a file still being written by checking size/mtime across complete hashing.
    before=archive.stat()
    h=hashlib.sha1()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    after=archive.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('Archive changed during verification')
    if h.hexdigest()!=SHA1: raise ValueError('FMA SHA-1 mismatch; do not extract')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist(); safe_archive_paths([i.filename for i in infos])
        audio=[i for i in infos if re.fullmatch(r'fma_small/\d{3}/\d{6}\.mp3',i.filename)]
        if len(audio)!=8000: raise ValueError(f'Expected 8000 numbered audio tracks, found {len(audio)}')
    target=Path(CONFIG['fma_audio']).resolve()
    if target.name!='fma_small': raise ValueError('Extraction target must end in fma_small')
    data_root=target.parent
    with zipfile.ZipFile(archive) as z:
        # Prevent collisions, except an already complete extraction verified by CRC below.
        destinations=[data_root.joinpath(*PurePosixPath(i.filename).parts) for i in infos if not i.is_dir()]
        for dst in destinations:
            if not dst.resolve().is_relative_to(target): raise ValueError('Escaping extraction target')
        if target.exists():
            import zlib
            for i,dst in zip([i for i in infos if not i.is_dir()],destinations):
                if not dst.exists() or dst.stat().st_size!=i.file_size: raise ValueError('Existing FMA directory differs; move it manually before extraction')
                crc=0
                with dst.open('rb') as f:
                    for block in iter(lambda:f.read(1024**2),b''): crc=zlib.crc32(block,crc)
                if crc!=i.CRC: raise ValueError('Existing extracted file CRC differs: '+i.filename)
            return {'status':'existing','sha1':h.hexdigest(),'audio_tracks':len(audio),
                    'note':'Existing sizes and CRCs match archive; no overwrite performed. Run importer to decode.'}
    seven=CONFIG['seven_zip']
    test=subprocess.run([seven,'t',str(archive)],capture_output=True,text=True)
    if test.returncode!=0: raise ValueError('7-Zip integrity test failed: '+test.stdout[-1000:])
    listing=subprocess.run([seven,'l','-slt',str(archive)],capture_output=True,text=True,check=True)
    reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
    (reports/'archive-paths.txt').write_text(listing.stdout,encoding='utf-8')
    # A dedicated staging location protects all original paths. Failed extraction can resume here.
    staging=data_root/'.soundmap-fma-staging'; staging.mkdir(parents=True,exist_ok=True)
    subprocess.run([seven,'x','-aos',str(archive),'-o'+str(staging)],check=True)
    extracted=staging/'fma_small'
    import zlib
    for i in infos:
        if i.is_dir(): continue
        p=staging.joinpath(*PurePosixPath(i.filename).parts)
        if not p.is_file() or p.stat().st_size!=i.file_size: raise ValueError('Incomplete staged file: '+i.filename)
        crc=0
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''): crc=zlib.crc32(block,crc)
        if crc!=i.CRC: raise ValueError('Staged CRC mismatch: '+i.filename)
    extracted.rename(target)
    return {'status':'extracted','sha1':h.hexdigest(),'audio_tracks':len(audio),'seven_zip_test':'passed','crc':'all files verified'}

def pilot_rows(rows,n=100,seed=2026):
    # Round robin over split/genre cells; immutable seeded track selection.
    groups={}
    for r in rows: groups.setdefault((r['split'],r['genre']),[]).append(r)
    rng=random.Random(seed)
    for g in groups.values(): rng.shuffle(g)
    out=[]
    while len(out)<n:
        changed=False
        for k in sorted(groups):
            if groups[k] and len(out)<n: out.append(groups[k].pop()); changed=True
        if not changed: break
    return sorted(out,key=lambda r:r['track_id'])

def import_fma(limit=100,seed=2026):
    root=Path(CONFIG['fma_audio'])
    if not root.is_dir(): return {'status':'pending','reason':'FMA audio directory absent; metadata audit available'}
    rows=list(metadata())
    if limit: rows=pilot_rows(rows,limit,seed)
    results=[]
    for r in rows:
        tid=f"{r['track_id']:06d}"; p=root/tid[:3]/(tid+'.mp3')
        try:
            probe(p)
            id_,new=store.register(p,'fma',r,r['title'] or tid)
            event(p,'imported'); results.append({'id':id_,'track_id':r['track_id'],'new':new})
        except Exception as e: event(p,'skipped',str(e)); results.append({'track_id':r['track_id'],'error':str(e)})
    return {'status':'imported','seed':seed,'requested':limit,'results':results}
