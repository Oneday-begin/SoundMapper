"""Reproducible audit of external human labels; never writes local listener ratings."""
import csv
import hashlib
import json
from pathlib import Path
from itertools import combinations
import numpy as np
from .config import ROOT, STATE, CONFIG

REPOSITORY='https://github.com/pxaris/CCMSim'
REVISION='eff8ab428dd12f1973548d657928b5211dd290bb'
FILES=('README.md','LICENSE','LICENSE-DATA','data/annotations/audio_annotations_mapping.csv','data/annotations/human_similarity.csv')
FOLDER=STATE/'benchmarks'/'ccmsim'


def fetch():
    import httpx
    FOLDER.mkdir(parents=True,exist_ok=True)
    hashes={}
    with httpx.Client(timeout=45,follow_redirects=True) as client:
        for name in FILES:
            target=FOLDER/Path(name).name
            if not target.exists():
                r=client.get(f'https://raw.githubusercontent.com/pxaris/CCMSim/{REVISION}/{name}');r.raise_for_status()
                temp=target.with_suffix('.tmp');temp.write_bytes(r.content);temp.replace(target)
            hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    result={'repository':REPOSITORY,'revision':REVISION,'sha256':hashes,
            'license_note':'Repository code MIT; README identifies annotation dataset as CC BY 4.0. Source audio has separate licensing.'}
    (FOLDER/'provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def agreement(rows):
    gaps=[];repeated=0
    for row in rows:
        values=np.asarray(json.loads(row['overall_music_similarity_annotations']),dtype=float)
        if not np.isfinite(values).all() or np.any(values<1) or np.any(values>5):raise ValueError('Unexpected human rating scale')
        if len(values)>1:
            repeated+=1;gaps.extend(abs(a-b)/4 for a,b in combinations(values,2))
    return {'repeated_pairs':repeated,'within_pair_rating_comparisons':len(gaps),
            'mean_absolute_disagreement_0_to_1':float(np.mean(gaps)) if gaps else None,
            'note':'External published within-pair rating dispersion; not agreement with SoundMap and not a chance-corrected reliability statistic.'}


def evaluate_predictions(rows, predictions):
    """Prediction CSV: audio_pair,score [0,1]; only matched, unique known pairs accepted."""
    from scipy.stats import spearmanr
    labels={r['audio_pair']:float(r['overall_music_similarity']) for r in rows}
    seen=set();truth=[];estimate=[]
    for p in predictions:
        pair=p['audio_pair'];score=float(p['score'])
        if pair not in labels or pair in seen:raise ValueError('Unknown or duplicate benchmark pair')
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Prediction scores must be finite in [0,1]')
        seen.add(pair);truth.append(labels[pair]);estimate.append(score)
    if len(truth)<3:raise ValueError('At least three independently predicted pairs required')
    corr=float(spearmanr(truth,estimate).statistic) if np.ptp(truth)>1e-12 and np.ptp(estimate)>1e-12 else None
    return {'pairs':len(truth),'mae':float(np.mean(np.abs(np.asarray(truth)-estimate))), 'spearman':corr,
            'note':'Evaluation only; predictions must come from the exact rated excerpts. Report audio and training-overlap provenance separately.'}


def audit(download=False,predictions_path=None):
    if download:fetch()
    if not (FOLDER/'human_similarity.csv').exists():raise RuntimeError('Run evaluate-external.ps1 -Download once to acquire public annotations.')
    with (FOLDER/'audio_annotations_mapping.csv').open(encoding='utf-8',newline='') as f:mapping=list(csv.DictReader(f))
    with (FOLDER/'human_similarity.csv').open(encoding='utf-8',newline='') as f:ratings=list(csv.DictReader(f))
    available=[];missing=[];root=Path(CONFIG['fma_audio']).resolve()
    for r in mapping:
        candidate=(root/r['audio_file_path']).resolve()
        ok=r['dataset']=='fma' and candidate.is_relative_to(root) and candidate.is_file()
        (available if ok else missing).append(r['survey_audio_id'])
    pair_count=sum(all(i in available for i in r['audio_pair'].split('-')) for r in ratings)
    result={'status':'blocked_missing_rated_audio' if not pair_count else 'audio_matching_required',
            'source':REPOSITORY,'revision':REVISION,'mapped_recordings':len(mapping),'rated_pairs':len(ratings),
            'available_recordings':len(available),'fully_available_rated_pairs':pair_count,
            'missing_recordings':missing,'external_listener_dispersion':agreement(ratings),
            'soundmap_human_similarity_metrics':None,'local_human_ratings_created':0,
            'reason':'No rated pair has both source recordings in the available FMA Small audio. Full-song cached scores cannot replace the rated 20-second excerpts.',
            'next_requirement':'Obtain licensed source audio matching the published clip mapping. No manual ratings are needed if a suitable external benchmark is used.',
            'coverage_note':'Nine dataset groups are mapped. Their names are not verified language labels. Model pretraining overlap is unknown.'}
    if predictions_path:
        with Path(predictions_path).open(encoding='utf-8',newline='') as f:result['supplied_prediction_metrics']=evaluate_predictions(ratings,list(csv.DictReader(f)))
    (ROOT/'reports'/'external-validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');parser.add_argument('--predictions')
    args=parser.parse_args();r=audit(args.download,args.predictions)
    print(json.dumps({k:v for k,v in r.items() if k!='missing_recordings'},indent=2))
