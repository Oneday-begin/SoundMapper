"""Measured descriptors of cached estimated stems; no semantic delivery labels."""
import numpy as np
from .dimensions import cached, exp_distance


def summarize(deep):
    if not deep or not deep.get('windows'):return None
    windows=deep['windows'];valid=[w for w in windows if w['usable_pitch']]
    notes=[];steps=[]
    for w in valid:
        x=np.asarray(w['pitch_semitones']);voiced=np.asarray(w['pitch_voiced'],dtype=bool)
        notes.extend(x[voiced]);steps.extend(np.abs(np.diff(x))[voiced[1:]&voiced[:-1]])
    spread=float(np.percentile(notes,90)-np.percentile(notes,10)) if len(notes)>1 else None
    return {'sampled_seconds':deep.get('sampled_seconds'), 'usable_pitch_excerpts':len(valid),
            'total_excerpts':len(windows),'voiced_fraction':float(np.mean([w['voiced_fraction'] for w in windows])),
            'relative_pitch_spread_semitones':spread,'median_pitch_step_semitones':float(np.median(steps)) if steps else None,
            'stem_energy_shares':{k:float(np.mean([w['stem_energy_shares'][k] for w in windows])) for k in ('drums','bass','other','vocals')},
            'note':'Demucs source estimates and gated pYIN pitch measurements. Stem leakage and sampling affect these values; no verified instrument or singing-technique labels.'}


def compare(a,b):
    ra,rb=cached(a),cached(b);sa=summarize(ra.get('deep')) if ra else None;sb=summarize(rb.get('deep')) if rb else None
    if not sa or not sb:return {'available':False,'reason':'Estimated source excerpts are unavailable.'}
    # Require multiple voiced excerpts and adjacent voiced frames in both recordings.
    supported=all(s['usable_pitch_excerpts']>=2 and s['relative_pitch_spread_semitones'] is not None and s['median_pitch_step_semitones'] is not None for s in (sa,sb))
    keys=('voiced_fraction','relative_pitch_spread_semitones','median_pitch_step_semitones')
    score=exp_distance([sa[k] for k in keys],[sb[k] for k in keys],np.array([.25,6,2])) if supported else None
    return {'available':True,'a':sa,'b':sb,'vocal_measurement_index':score,
            'reason':None if supported else 'At least two usable voiced excerpts in each recording are required; a missing score does not mean dissimilar voices.',
            'note':'Comparison of measured voicing and pitch movement in sampled estimated vocals. Not a classifier for rap, breathiness, emotion, singer identity or vocal delivery.',
            'excluded_from_descriptive_index':True}
