import json
import os
import tempfile
from pathlib import Path
_directory=tempfile.TemporaryDirectory(prefix='soundmap-tests-')
_root=Path(_directory.name)
(_root/'config.json').write_text(json.dumps({'mert_enabled':False,'tagging_enabled':False,'ollama_model':None,'upload_mb':1,'max_duration_seconds':20}))
os.environ['SOUNDMAP_STATE']=str(_root/'state')
os.environ['SOUNDMAP_CONFIG']=str(_root/'config.json')
