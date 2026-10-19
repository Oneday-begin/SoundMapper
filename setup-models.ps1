$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# Explicit model provisioning only. Normal app inference defaults to local cached files.
@'
from soundmap.config import CONFIG
from huggingface_hub import snapshot_download
models = [
    ('m-a-p/MERT-v1-95M', CONFIG['model_revision'], ['*.json','*.py','pytorch_model.bin','README.md']),
    ('MIT/ast-finetuned-audioset-10-10-0.4593', CONFIG['tagging_revision'], ['config.json','preprocessor_config.json','model.safetensors','README.md'])
]
for name,revision,patterns in models:
    print('Provisioning pinned model:',name,revision,flush=True)
    snapshot_download(name,revision=revision,allow_patterns=patterns,max_workers=2)
print('Model cache ready. Local app inference does not need Wi-Fi.')
'@ | & '.\.venv\Scripts\python.exe' -X utf8 -
if ($LASTEXITCODE -ne 0) { throw 'Model provisioning failed; rerun online to resume downloads. Existing audio and caches preserved.' }
