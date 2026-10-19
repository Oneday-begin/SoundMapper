"""Listener collection plans and eligibility, defined before pair construction."""
import itertools
import random
from collections import defaultdict,Counter
from . import store
from .pipeline import KEY

REQUIREMENTS={'eligible_distinct_pairs':60,'distinct_recordings':30,'training_pairs':20,'validation_pairs':10,'test_pairs':10}

def split_for(group):
    import hashlib
    n=int(hashlib.sha256(('soundmap-2026:'+group).encode()).hexdigest()[:8],16)%10
    return 'training' if n<6 else 'validation' if n<8 else 'test'

def prepare():
    tracks={t['id']:t for t in store.tracks()}
    with store.connect() as db:
        rows=[dict(r) for r in db.execute('SELECT * FROM ratings ORDER BY created,id')]
        decoded={r[0]:r[1] for r in db.execute("SELECT track_id,json_extract(result,'$.decoded_hash') FROM analyses WHERE config_key=? AND json_type(result,'$.mert')='object' AND json_extract(result,'$.silent')=0",(KEY,))}
    eligible={i:t for i,t in tracks.items() if t['metadata'].get('artist_id') and i in decoded}
    assignment={i:split_for('artist:'+t['metadata']['artist_id']) for i,t in eligible.items()}
    duplicates=defaultdict(set)
    for i in eligible:duplicates[decoded[i]].add(assignment[i])
    conflicts={h for h,s in duplicates.items() if len(s)>1}
    eligible={i:t for i,t in eligible.items() if decoded[i] not in conflicts}
    assignment={i:s for i,s in assignment.items() if i in eligible}
    grouped=defaultdict(lambda:defaultdict(list));excluded=Counter();scopes=set()
    for r in rows:
        if r['overall'] is None:excluded['overall_skipped']+=1;continue
        if r['a'] not in eligible or r['b'] not in eligible:excluded['missing_artist_or_embedding_or_duplicate_conflict']+=1;continue
        if assignment[r['a']]!=assignment[r['b']]:excluded['cross_split_pair']+=1;continue
        if decoded[r['a']]==decoded[r['b']]:excluded['duplicate_recording_pair']+=1;continue
        identity=r['rater'] or r['session'];pair=tuple(sorted((r['a'],r['b'])))
        grouped[pair][identity].append(r['overall']);scopes.add(identity)
    # Average repeats per identifier first, then identifiers per unordered pair.
    # Repeated pairs are not independent examples and cannot inflate sample counts.
    pairs=[]
    for (a,b),by_rater in grouped.items():
        means=[sum(v)/len(v) for v in by_rater.values()]
        pairs.append({'a':a,'b':b,'overall':sum(means)/len(means),'split':assignment[a],'raters':len(means)})
    counts={s:sum(p['split']==s for p in pairs) for s in ('training','validation','test')}
    ntracks=len({p['a'] for p in pairs}|{p['b'] for p in pairs})
    ready=len(pairs)>=60 and ntracks>=30 and counts['training']>=20 and counts['validation']>=10 and counts['test']>=10
    report={'ready':ready,'rating_rows':len(rows),'eligible_distinct_pairs':len(pairs),'distinct_recordings':ntracks,
            'split_pair_counts':counts,'requirements':REQUIREMENTS,'eligible_library_tracks':len(eligible),
            'eligible_library_splits':dict(Counter(assignment.values())),'excluded_rows':dict(excluded),
            'duplicate_recording_split_conflicts':len(conflicts),'listener_identifiers':len(scopes),
            'scope':'personal preference' if len(scopes)==1 else 'multi-listener similarity' if scopes else 'none',
            'reason':None if ready else 'Need 60 distinct eligible overall judgments across 30 recordings, including 20 training, 10 validation and 10 test pairs.',
            'note':'Artist groups are assigned before pairs. Unknown artists and cross-split pairs remain saved but are excluded from training. Multiple identifiers are not verified distinct people.'}
    return report,eligible,assignment,pairs,decoded

def readiness():return prepare()[0]

def next_training_pair(session,prior):
    report,tracks,assignment,_,decoded=prepare()
    groups={s:sorted([t for i,t in tracks.items() if assignment[i]==s],key=lambda t:(t['metadata'].get('track_id',0),t['hash'])) for s in ('training','validation','test')}
    plan={}
    for s,ts in groups.items():
        rng=random.Random('soundmap-plan-2026:'+s)
        shuffled=list(ts);rng.shuffle(shuffled)
        if len(shuffled)<2:raise ValueError('Training collection needs analyzed tracks with known artists in all three splits.')
        candidates=list(itertools.combinations([t['id'] for t in shuffled],2))
        candidates=[p for p in candidates if decoded[p[0]]!=decoded[p[1]]]
        rng.shuffle(candidates)
        # Held-out anchor queries each acquire >=3 candidates for ranking metrics.
        if s!='training':
            anchors={t['id'] for t in shuffled[:3]}
            preferred=[p for p in candidates if (p[0] in anchors) != (p[1] in anchors)]
            balanced=[]
            for anchor in [t['id'] for t in shuffled[:3]]:
                balanced.extend([p for p in preferred if anchor in p][:4])
            candidates=balanced+[p for p in candidates if p not in balanced]
        plan[s]=candidates
    seen={tuple(sorted((r['a'],r['b']))) for r in prior}
    # Interleave 4 train, 1 validation and 1 test; no similarity score guides selection.
    for i in range(max(len(p) for p in plan.values())):
        block=[]
        block.extend(('training',p) for p in plan['training'][i*4:i*4+4])
        for s in ('validation','test'):
            if i<len(plan[s]):block.append((s,plan[s][i]))
        for s,(a,b) in block:
            if tuple(sorted((a,b))) not in seen:return {'a':a,'b':b,'repeat':False,'collection_mode':'training','split':s,'readiness':report}
    raise ValueError('All eligible collection pairs have been rated in this session; start another listener identifier/session for independent agreement checks.')
