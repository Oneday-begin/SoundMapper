$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
@'
from soundmap.config import STATE,CONFIG
import torch,hashlib
path=STATE/'models'/'separation'/'hdemucs_high_musdbhq_only.pt'
path.parent.mkdir(parents=True,exist_ok=True)
if not path.exists():
    temp=path.with_suffix('.download')
    torch.hub.download_url_to_file('https://download.pytorch.org/torchaudio/models/hdemucs_high_musdbhq_only.pt',str(temp))
    with temp.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=CONFIG['separation_sha256']: raise RuntimeError('Unexpected separation weights hash')
    temp.replace(path)
with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==CONFIG['separation_sha256']
print('Pinned source-separation weights ready; subsequent inference is offline.')
'@ | & '.\.venv\Scripts\python.exe' -X utf8 -
if ($LASTEXITCODE -ne 0) { throw 'Separation model setup failed; original audio preserved.' }
