import threading
import time
import zipfile
import numpy as np
from .config import CONFIG, STATE
from .audio import SR, unit

MODEL_ID = 'm-a-p/MERT-v1-95M'
LAYERS = (4, 8, 12)
_model = None
_processor = None
_lock = threading.Lock()
status = {'state': 'not_loaded', 'revision': CONFIG['model_revision'], 'error': None}

def load():
    global _model, _processor
    if not CONFIG['mert_enabled']: raise RuntimeError('MERT disabled in configuration')
    with _lock:
        if _model is not None: return _model, _processor
        status.update(state='loading', error=None)
        start = time.perf_counter()
        try:
            import torch
            from transformers import AutoConfig, Wav2Vec2FeatureExtractor
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from huggingface_hub import hf_hub_download
            torch.set_num_threads(CONFIG['threads'])
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True)
            rev = CONFIG['model_revision']
            local=not CONFIG.get('model_downloads_allowed',False)
            _processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, revision=rev,local_files_only=local)
            assert _processor.sampling_rate == SR
            # Custom code reviewed at this exact immutable revision. CPU weights-only loader.
            config=AutoConfig.from_pretrained(MODEL_ID,revision=rev,code_revision=rev,trust_remote_code=True,local_files_only=local)
            # 4.38 AutoModel.from_config forwards local_files_only to __init__, so
            # resolve reviewed custom code explicitly before constructing the model.
            model_class=get_class_from_dynamic_module(config.auto_map['AutoModel'],MODEL_ID,
                revision=rev,code_revision=rev,local_files_only=local)
            _model=model_class._from_config(config).cpu()
            path=hf_hub_download(MODEL_ID,'pytorch_model.bin',revision=rev,local_files_only=local)
            weights=torch.load(path,map_location='cpu',weights_only=True)
            # PyTorch's modern parametrized weight_norm uses new state-dict key names.
            # Transformers 4.38's permissive loader silently misses the old conv keys.
            # Rename only those mathematically equivalent tensors and enforce strict loading.
            old='encoder.pos_conv_embed.conv.'
            expected=_model.state_dict()
            remapped=[]
            for suffix,target in [('weight_g','parametrizations.weight.original0'),('weight_v','parametrizations.weight.original1')]:
                if old+suffix in weights and old+target in expected:
                    weights[old+target]=weights.pop(old+suffix);remapped.append(old+suffix+' -> '+old+target)
            _model.load_state_dict(weights,strict=True)
            del weights,expected
            _model.eval()
            status.update(strict_weight_load=True,remapped_weight_norm_keys=remapped)
            for p in _model.parameters(): p.requires_grad_(False)
            status.update(state='ready', load_seconds=time.perf_counter()-start)
        except Exception as e:
            _model = None
            status.update(state='unavailable', error=str(e))
            raise
        return _model, _processor

def embed(y):
    import torch
    model, processor = load()
    # Batch size one: no padded window bias; final window retains its own length.
    inputs = processor(y, sampling_rate=SR, return_tensors='pt')
    with torch.inference_mode():
        result = model(**inputs, output_hidden_states=True)
        vectors = np.stack([result.hidden_states[l][0].mean(dim=0).cpu().numpy() for l in LAYERS])
    if not np.isfinite(vectors).all(): raise ValueError('Non-finite MERT embedding')
    return vectors

def extract(y, windows, cache_dir, progress):
    start = time.perf_counter()
    load()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pooled, durations = [], []
    reused = 0
    for i,w in enumerate(windows):
        path = cache_dir / f'window-{i:05d}.npz'
        if w['rms'] < 1e-6:
            w['embedding'] = None
            continue
        v = None
        if path.exists():
            try:
                with path.open('rb') as handle:
                    with np.load(handle, allow_pickle=False) as data: v = data['vectors']
                if v.shape != (len(LAYERS), 768) or not np.isfinite(v).all():
                    raise ValueError('Invalid cached embedding')
                reused += 1
            except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile):
                path.replace(path.with_suffix('.corrupt-' + str(time.time_ns())))
                v = None
        if v is None:
            v = embed(y[int(w['start']*SR):int(w['end']*SR)])
            # Individual atomic checkpoints survive process interruption.
            temp = path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, vectors=v)
            temp.replace(path)
        pooled.append(v); durations.append(w['end']-w['start'])
        w['embedding'] = unit(v[-1]).tolist()
        progress(0.2 + 0.78*(i+1)/len(windows), f'MERT window {i+1}/{len(windows)}')
    if not pooled: return None
    means = np.average(np.stack(pooled), axis=0, weights=durations)
    return {'vector':unit(means[-1]).tolist(), 'layers':{str(l):unit(means[j]).tolist() for j,l in enumerate(LAYERS)},
            'seconds':time.perf_counter()-start, 'windows':len(pooled), 'reused_windows':reused,
            'model':MODEL_ID, 'revision':CONFIG['model_revision'], 'hidden_layer':12,
            'pooling':'frame mean per window; duration-weighted raw window means; L2 track normalization',
            'preprocessing':{'mono':'channel mean', 'sr':SR, 'resample':'soxr_hq',
                             'normalize':'official Wav2Vec2FeatureExtractor', 'window_seconds':CONFIG['window_seconds'], 'batch':1}}
