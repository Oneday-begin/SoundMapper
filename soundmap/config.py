import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config.example.json').read_text())
config_path = Path(os.environ.get('SOUNDMAP_CONFIG', ROOT / 'config.json'))
if config_path.exists():
    CONFIG.update(json.loads(config_path.read_text()))
STATE = Path(os.environ.get('SOUNDMAP_STATE', CONFIG['state']))
if not STATE.is_absolute():
    STATE = ROOT / STATE
for name in ('uploads', 'cache', 'models'):
    (STATE / name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', str(STATE / 'models'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('OMP_NUM_THREADS', str(CONFIG['threads']))
os.environ.setdefault('MKL_NUM_THREADS', str(CONFIG['threads']))
