"""Pinned, frozen EnCodec excerpts; codec representations are not semantic labels."""
import hashlib
import json
import threading
import time
import zipfile
from pathlib import Path
import numpy as np
from .config import STATE, CONFIG
from . import store
from .audio import decode
from .compute import serialized

MODEL = 'facebook/encodec_24khz'
REVISION = 'c1dbe2ae3f1de713481a3b3e7c47f357092ee040'
SETTINGS = {'version': 'codec-2', 'model': MODEL, 'revision': REVISION,
            'sr': 24000, 'bandwidth': 6.0, 'excerpt_seconds': 8, 'max_excerpts': 5,
            'pooling': 'per-codebook token frequency; mean Hellinger affinity',
            'token_preprocessing': 'mono excerpt RMS normalized to 0.1; original RMS retained separately',
            'amplitude': 'original input measured; reconstruction rescaled to original level for RMS diagnostic'}
KEY = hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:20]
NOTE = ('EnCodec token distributions describe sampled sound texture. They do not identify instruments, '
        'production techniques or vocal delivery. No perceptual calibration.')
_model = None
_lock = threading.Lock()


def setup():
    from huggingface_hub import snapshot_download
    folder = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md'], max_workers=2))
    manifest = {'model': MODEL, 'revision': REVISION, 'files': {}}
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md'):
        with (folder/name).open('rb') as f: manifest['files'][name] = hashlib.file_digest(f, 'sha256').hexdigest()
    target = STATE/'models'/'encodec-manifest.json'
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load():
    global _model
    with _lock:
        if _model is not None: return _model
        import torch
        from transformers import EncodecConfig, EncodecModel
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        manifest_path = STATE/'models'/'encodec-manifest.json'
        if not manifest_path.exists(): raise RuntimeError('EnCodec is unavailable. Run setup-codec.ps1 once while online.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['revision'] != REVISION: raise RuntimeError('EnCodec manifest revision mismatch')
        folder = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
        for name in ('config.json', 'model.safetensors'):
            with (folder/name).open('rb') as f: actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if actual != manifest['files'][name]: raise RuntimeError('EnCodec checkpoint checksum mismatch: '+name)
        torch.set_num_threads(min(2, CONFIG['threads']))
        model = EncodecModel(EncodecConfig.from_pretrained(folder, local_files_only=True))
        model.load_state_dict(load_file(str(folder/'model.safetensors')), strict=True)
        model.cpu().eval()
        for p in model.parameters(): p.requires_grad_(False)
        _model = model
        return model


def positions(duration):
    if duration < 1: return []
    count = min(SETTINGS['max_excerpts'], max(1, int(duration//SETTINGS['excerpt_seconds'])))
    return [(float(s), min(float(s)+SETTINGS['excerpt_seconds'], duration))
            for s in np.linspace(0, max(0, duration-SETTINGS['excerpt_seconds']), count)]


def histogram(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or not codes.shape[1] or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('Invalid codec token dimensions/type')
    if codes.min() < 0 or codes.max() >= 1024: raise ValueError('Invalid codec token ID')
    counts = np.stack([np.bincount(row, minlength=1024) for row in codes]).astype(float)
    return counts/counts.sum(axis=1, keepdims=True)


def token_affinity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2: raise ValueError('Incompatible codec codebooks')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError('Invalid token distributions')
    return float(np.clip(100*np.mean(np.sqrt(a*b).sum(axis=1)), 0, 100))


def encode(y):
    import torch
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or len(y) < 24000 or len(y) > 24000*8 or not np.isfinite(y).all():
        raise ValueError('EnCodec expects 1–8 seconds of finite mono 24 kHz audio')
    if np.sqrt(np.mean(y*y)) < 1e-6: return {'silent': True}
    model = load()
    original_rms = float(np.sqrt(np.mean(y*y)))
    input_scale = original_rms/.1
    x = torch.from_numpy((y/input_scale).copy())[None, None, :]
    with torch.inference_mode():
        encoded = model.encode(x, bandwidth=SETTINGS['bandwidth'])
        codes = encoded.audio_codes[0, 0].numpy()
        reconstruction = model.decode(encoded.audio_codes, encoded.audio_scales).audio_values[0, 0, :len(y)].numpy()*input_scale
    reconstructed_rms = float(np.sqrt(np.mean(reconstruction**2)))
    return {'silent': False, 'codes': codes.astype(np.int16), 'histogram': histogram(codes),
            'input_rms_dbfs': float(20*np.log10(max(original_rms, 1e-10))),
            'reconstruction_rms_error_db': float(20*np.log10(max(reconstructed_rms, 1e-10)/original_rms)),
            'input_scale':input_scale,'codec_audio_scale': None, 'scale_note': '24 kHz checkpoint has normalize=false. Application normalizes token input; original amplitude is retained separately.'}


def cached(id_):
    with store.connect() as db:
        row = db.execute('SELECT result FROM annotations WHERE track_id=? AND kind=? AND config_key=?',
                         (id_, 'codec', KEY)).fetchone()
    return json.loads(row[0]) if row else None


def checkpoint(path):
    try:
        with path.open('rb') as f, np.load(f, allow_pickle=False) as d:
            codes = d['codes']; h = histogram(codes)
            level, error = float(d['input_rms_dbfs']), float(d['reconstruction_rms_error_db'])
            if not np.isfinite([level, error]).all(): raise ValueError('Invalid amplitude diagnostic')
            return {'histogram': h, 'input_rms_dbfs': level, 'reconstruction_rms_error_db': error}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        if path.exists(): path.replace(path.with_suffix('.corrupt-'+str(time.time_ns())))
        return None


@serialized
def analyze(id_, progress=lambda p, m: None, force=False):
    prior = cached(id_)
    if prior and not force:
        progress(1, 'EnCodec cache reused')
        return {**prior, 'cache_hit': True}
    from .pipeline import MemoryMeter
    track = store.track(id_); started = time.perf_counter()
    folder = STATE/'cache'/track['hash']/('codec-'+KEY); folder.mkdir(parents=True, exist_ok=True)
    with MemoryMeter() as meter:
        y, info = decode(track['path']); windows = []; distributions = []; reused = 0
        for i, (start, end) in enumerate(positions(info['duration'])):
            path = folder/f'window-{i}.npz'
            result = checkpoint(path) if path.exists() else None
            if result is not None: reused += 1
            else:
                result = encode(y[round(start*24000):round(end*24000)])
                if result['silent']: continue
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, codes=result['codes'], input_rms_dbfs=result['input_rms_dbfs'],
                                    reconstruction_rms_error_db=result['reconstruction_rms_error_db'])
                temp.replace(path)
            distributions.append(result['histogram'])
            windows.append({'start': start, 'end': end, 'input_rms_dbfs': result['input_rms_dbfs'],
                            'reconstruction_rms_error_db': result['reconstruction_rms_error_db']})
            progress((i+1)/len(positions(info['duration'])), f'EnCodec excerpt {i+1}')
        result = {'state': 'ready' if windows else 'silent', 'settings': SETTINGS, 'note': NOTE,
                  'source_duration': info['duration'], 'sampled_seconds': sum(w['end']-w['start'] for w in windows),
                  'windows': windows, 'histogram': np.average(distributions, axis=0, weights=[w['end']-w['start'] for w in windows]).tolist() if distributions else None,
                  'cache_hit': False, 'reused_windows': reused,
                  'amplitude_note': 'Tokens use RMS-normalized excerpts for texture. Input excerpt RMS is measured before normalization. Reconstruction RMS error is a fidelity diagnostic; full-song dynamics use original audio.',
                  'processing_seconds': time.perf_counter()-started, 'peak_rss_mb': meter.peak/1024**2}
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?,?)', (id_, 'codec', KEY, json.dumps(result, allow_nan=False)))
    return result


def compare(id_a, id_b):
    a, b = cached(id_a), cached(id_b)
    if not a or not b: return {'available': False, 'reason': 'EnCodec excerpts have not been analyzed for both recordings.', 'missing': [i for i, r in ((id_a,a),(id_b,b)) if not r]}
    if a['state'] != 'ready' or b['state'] != 'ready': return {'available': False, 'reason': 'Silence has no meaningful codec texture comparison.', 'missing': []}
    def public(r): return {k:v for k,v in r.items() if k != 'histogram'}
    return {'available': True, 'score': token_affinity(a['histogram'], b['histogram']), 'a': public(a), 'b': public(b),
            'note': NOTE, 'method': SETTINGS['pooling'], 'source': 'https://arxiv.org/abs/2210.13438',
            'excluded_from_descriptive_index': True}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--setup', action='store_true'); parser.add_argument('--samples', action='store_true')
    args=parser.parse_args()
    if args.setup: print(json.dumps(setup(), indent=2))
    if args.samples:
        for track in store.tracks():
            if track['source'] != 'fma':
                result=analyze(track['id']);print(track['title'], result['state'], round(result['processing_seconds'],2), flush=True)
