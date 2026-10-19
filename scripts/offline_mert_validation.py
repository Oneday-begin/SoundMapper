"""Fresh local-only CPU inference against the first original sample checkpoint."""
import json
import numpy as np
from soundmap.config import CONFIG,STATE,ROOT
# Force local-only loading before encoder.load resolves any cached code or weights.
CONFIG['model_downloads_allowed']=False
from soundmap import store,encoder
from soundmap.audio import decode
from soundmap.pipeline import KEY,cached

track=next(t for t in store.tracks() if t['source']=='sample')
analysis=cached(track['id'])
assert analysis and analysis['mert'],'Analyze a sample first.'
i,w=next((i,w) for i,w in enumerate(analysis['windows']) if w.get('embedding'))
y,_=decode(track['path'])
actual=encoder.embed(y[int(w['start']*24000):int(w['end']*24000)])
with np.load(STATE/'cache'/track['hash']/KEY/f'window-{i:05d}.npz',allow_pickle=False) as saved:
    difference=float(np.max(np.abs(actual-saved['vectors'])))
assert np.isfinite(actual).all() and difference<=1e-6
report={'status':'passed','offline':True,'finite':True,
        'max_abs_difference_vs_original_cached_embedding':difference,'model_status':encoder.status}
(ROOT/'reports/offline-mert-extension-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
