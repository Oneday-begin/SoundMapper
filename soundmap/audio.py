"""Bounded decoding and interpretable acoustic features; no instrument/melody labels."""
import numpy as np
import soundfile as sf
import librosa
from .config import CONFIG

SR = 24000
HOP = 512

def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-10)

def decode(path):
    try:
        with sf.SoundFile(path) as f:
            duration = len(f) / f.samplerate
            if duration < 1: raise ValueError('Insufficient audio: at least one second required')
            if duration > CONFIG['max_duration_seconds']: raise ValueError('Audio duration exceeds configured limit')
            if f.channels > 8: raise ValueError('Too many audio channels')
            original_sr, channels = f.samplerate, f.channels
            y = f.read(dtype='float32', always_2d=True).mean(axis=1)
    except ValueError: raise
    except Exception as e: raise ValueError('Unreadable audio: ' + str(e)) from e
    if not np.isfinite(y).all(): raise ValueError('Non-finite audio samples')
    if original_sr != SR: y = librosa.resample(y, orig_sr=original_sr, target_sr=SR, res_type='soxr_hq')
    # Resamplers can round upward by one sample; match the source duration grid.
    target_length=int(round(duration*SR))
    y=y[:target_length]
    if len(y)<target_length: y=np.pad(y,(0,target_length-len(y)))
    return y, {'duration': float(duration), 'original_sr': original_sr, 'channels': channels, 'analysis_sr': SR}

def probe(path):
    try:
        with sf.SoundFile(path) as f:
            d = len(f) / f.samplerate
            if d < 1: raise ValueError('Insufficient audio: at least one second required')
            if d > CONFIG['max_duration_seconds']: raise ValueError('Audio duration exceeds configured limit')
            if f.channels > 8: raise ValueError('Too many audio channels')
            return d
    except ValueError: raise
    except Exception as e: raise ValueError('Unreadable audio') from e

def features(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    rms = librosa.feature.rms(S=S)[0]
    chroma = librosa.feature.chroma_stft(S=S**2, sr=SR, hop_length=HOP, tuning=0)
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SR)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=SR, hop_length=HOP)
    # Normalized onset autocorrelation at 24 log-spaced tempo bins, 40..240 BPM.
    bpms = np.geomspace(40, 240, 24)
    lags = np.round(60 * SR / HOP / bpms).astype(int)
    def rhythm_profile(v):
        return unit([np.dot(v[k:], v[:-k]) / max(1, len(v)-k) if len(v)>k else 0 for k in lags])
    rhythm = rhythm_profile(onset)
    silent = bool(np.max(np.abs(y)) < 1e-5 or np.mean(rms) < 1e-6)
    tempo = None
    beats = []
    if not silent and np.max(onset) > 1e-6:
        t, b = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
        tempo = float(np.asarray(t).ravel()[0])
        beats = librosa.frames_to_time(b, sr=SR, hop_length=HOP).tolist()
    # A fixed scale baseline avoids fitting test-library normalization statistics.
    def timbre(i, j):
        m = mfcc[1:, i:j].mean(axis=1) / 50
        return np.r_[m, np.log1p([centroid[i:j].mean(), bandwidth[i:j].mean(), rolloff[i:j].mean()])/10,
                     flatness[i:j].mean(), np.std(rms[i:j]) / max(np.mean(rms[i:j]),1e-8)]
    seconds = len(y) / SR
    windows = []
    ws = CONFIG['window_seconds']
    for start in np.arange(0, seconds, ws):
        end = min(start + ws, seconds)
        if end-start < 1: continue
        i = int(start * SR / HOP); j = min(len(rms), max(i+1, int(end*SR/HOP)))
        windows.append({'start': float(start), 'end': float(end), 'rms': float(rms[i:j].mean()),
                        'chroma': unit(chroma[:,i:j].mean(axis=1)).tolist(),
                        'rhythm': rhythm_profile(onset[i:j]).tolist(), 'timbre': timbre(i,j).tolist()})
    # Average roughly one-second chroma blocks for constrained sequence comparison.
    step = max(1, int(SR/HOP))
    harmonic = np.array([unit(chroma[:,i:i+step].mean(axis=1)) for i in range(0,chroma.shape[1],step)])
    wave = []
    for s in np.array_split(y, min(800,len(y))): wave.append([float(s.min()),float(s.max())])
    energy_step = max(1,len(rms)//600)
    return {'silent':silent, 'tempo':tempo, 'beats':beats, 'waveform':wave,
            'energy':{'time':(np.arange(0,len(rms),energy_step)*HOP/SR).tolist(), 'rms':rms[::energy_step].tolist()},
            'spectral':{'centroid_hz':float(centroid.mean()), 'bandwidth_hz':float(bandwidth.mean()),
                        'rolloff_hz':float(rolloff.mean()), 'flatness':float(flatness.mean()), 'rms':float(rms.mean())},
            'chroma': unit(chroma.mean(axis=1)).tolist(), 'harmonic':harmonic.tolist(),
            'rhythm':rhythm.tolist(), 'timbre':timbre(0,len(rms)).tolist(), 'windows':windows}
