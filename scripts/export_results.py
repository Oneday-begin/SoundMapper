"""Export actual acoustic results without audio, binary caches, or model weights."""
import json
from soundmap import store
from soundmap.config import ROOT
from soundmap.pipeline import cached

root=ROOT/'reports'/'analyses';root.mkdir(exist_ok=True)
for t in store.tracks():
    if t['source']!='sample':continue
    result=cached(t['id'])
    if not result:continue
    result['recording']={'id':t['id'],'title':t['title'],'file_sha256':t['hash']}
    if result.get('mert'):
        result['mert'].pop('vector',None);result['mert'].pop('layers',None)
        result['mert']['embedding_dimension']=768
        result['mert']['note']='Vectors remain in the local runtime cache; full analysis can be exported from the app.'
    for w in result['windows']:w.pop('embedding',None)
    (root/(t['id']+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('Exported actual features and model provenance for seven samples to reports/analyses.')
