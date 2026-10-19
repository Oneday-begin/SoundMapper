"""Reproducible descriptive comparison over the seven supplied sample songs."""
import itertools
import json
from pathlib import Path

from soundmap import dimensions, store
from soundmap.config import ROOT
from soundmap.pipeline import cached as analysis


def main():
    tracks = [t for t in store.tracks() if t['source'] == 'sample']
    if len(tracks) != 7:
        raise SystemExit(f'Expected seven registered sample songs; found {len(tracks)}')
    missing = [t['title'] for t in tracks if not analysis(t['id']) or not dimensions.cached(t['id'])]
    if missing:
        raise SystemExit('Run the sample analysis first; missing: ' + ', '.join(missing))
    pairs = []
    for left, right in itertools.combinations(tracks, 2):
        result = dimensions.compare_dimensions(left['id'], right['id'])
        pairs.append({'a': left['title'], 'b': right['title'], 'overall': result['overall'],
                      'coverage': result['coverage'], 'available_dimensions': result['available_dimensions'],
                      'rows': [{'id': r['id'], 'score': r['score'], 'quality': r['quality'],
                                'aggregate': r['aggregate'],
                                'method': r['method'], 'reason': r['evidence'].get('reason')}
                               for r in result['rows']]})
    selected = next(p for p in pairs if 'Parvati' in p['a'] and 'Gehra Hua' in p['b'] or
                    'Gehra Hua' in p['a'] and 'Parvati' in p['b'])
    report = {'status': 'complete', 'dataset': 'seven supplied sample recordings', 'pair_count': len(pairs),
              'config': dimensions.SETTINGS, 'tracks': [
                  {'title': t['title'], 'duration_seconds': analysis(t['id'])['duration'],
                   'dimension_processing_seconds': dimensions.cached(t['id'])['processing_seconds'],
                   'process_peak_rss_mb': dimensions.cached(t['id'])['peak_rss_mb'],
                   'deep_vocal_pitch': bool(dimensions.cached(t['id']).get('deep')),
                   'analysis_cached': True} for t in tracks],
              'parvati_gehra_hua': selected, 'pairs': pairs,
              'interpretation': 'Descriptive algorithm output only; no listener similarity ground truth or calibrated score.'}
    path = ROOT / 'reports' / 'nine-feature-evaluation.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'report': str(path), 'tracks': len(tracks), 'pairs': len(pairs),
                      'pair_of_interest': selected}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
