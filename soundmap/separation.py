"""Bounded CPU Hybrid Demucs excerpts and pYIN pitch, with immutable checkpoints."""
import hashlib,json,time,threading
import numpy as np
import soundfile as sf
import librosa
from .config import STATE,CONFIG

MODEL='torchaudio 2.6 HDEMUCS_HIGH_MUSDB'
PATH=STATE/'models'/'separation'/'hdemucs_high_musdbhq_only.pt'
_model=None;_lock=threading.Lock()
def load():
    global _model
    import torch,torchaudio
    with _lock:
        if _model is not None:return _model
        if not PATH.exists():raise RuntimeError('Separation weights absent; run setup-separation.ps1 while online.')
        expected=CONFIG.get('separation_sha256')
        with PATH.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if not expected or actual!=expected:raise RuntimeError('Separation weights hash not pinned or mismatched.')
        torch.set_num_threads(CONFIG['threads']);torch.manual_seed(0)
        _model=torchaudio.models.hdemucs_high(sources=['drums','bass','other','vocals'])
        _model.load_state_dict(torch.load(PATH,map_location='cpu',weights_only=True),strict=True)
        _model.cpu().eval()
        for p in _model.parameters():p.requires_grad_(False)
        return _model

def separate(audio):
    import torch
    model=load();scale=max(float(np.std(audio)),1e-8)
    with torch.inference_mode():stems=model(torch.from_numpy((audio.T/scale).copy()).unsqueeze(0))[0].numpy()*scale
    if not np.isfinite(stems).all():raise ValueError('Non-finite separated audio')
    return stems

def pitch(vocal,sr=8000):
    f0,flag,prob=librosa.pyin(vocal,sr=sr,fmin=65,fmax=1000,frame_length=1024,hop_length=256,n_thresholds=32)
    valid=flag & np.isfinite(f0) & (prob>=.5)
    notes=librosa.hz_to_midi(f0[valid])
    hop_times=librosa.frames_to_time(np.arange(len(f0)),sr=sr,hop_length=256)
    duration=len(vocal)/sr;grid=np.linspace(0,duration,64,endpoint=False);contour=np.zeros(64,dtype=float);voiced=np.zeros(64,dtype=bool)
    if len(notes):
        center=float(np.median(notes));grid_step=duration/64
        for i,t in enumerate(grid):
            ix=np.flatnonzero(valid & (hop_times>=t-grid_step/2) & (hop_times<t+grid_step/2))
            if len(ix):contour[i]=float(np.median(librosa.hz_to_midi(f0[ix]))-center);voiced[i]=True
    return {'pitch_semitones':contour.tolist(),'pitch_times':grid.tolist(),'pitch_voiced':voiced.tolist(),'voiced_fraction':float(np.mean(valid)),
        'median_pitch_midi':float(np.median(notes)) if len(notes) else None,
        'usable_pitch':bool(len(notes)>=20 and np.mean(valid)>=.25),'pitch_note':'pYIN on estimated vocal stem; confidence gate does not establish melody ground truth.'}

def analyze(track,progress=lambda p,m:None):
    from .dimensions import KEY,SETTINGS
    folder=STATE/'cache'/track['hash']/('stems-'+KEY);folder.mkdir(parents=True,exist_ok=True)
    with sf.SoundFile(track['path']) as f:
        duration=len(f)/f.samplerate;sr=f.samplerate
        starts=np.clip(np.linspace(duration*.25-4,duration*.75-4,min(3,max(1,int(duration//8)))),0,max(0,duration-8))
        windows=[];start=time.perf_counter();reused=0
        for i,s in enumerate(starts):
            p=folder/f'window-{i}.json'
            if p.exists():windows.append(json.loads(p.read_text()));reused+=1;continue
            f.seek(int(round(s*sr)));raw=f.read(min(int(8*sr),len(f)-f.tell()),dtype='float32',always_2d=True)
            if raw.shape[1]==1:raw=np.repeat(raw,2,axis=1)
            raw=raw[:,:2];end=min(duration,float(s)+len(raw)/sr)
            audio=librosa.resample(raw.T,orig_sr=sr,target_sr=44100,res_type='soxr_hq').T
            stems=separate(audio);energy=np.mean(stems**2,axis=(1,2));shares=energy/max(float(energy.sum()),1e-10)
            vocal=librosa.resample(stems[3].mean(axis=0),orig_sr=44100,target_sr=8000,res_type='soxr_hq')
            row={'start':float(s),'end':end,'stem_energy_shares':dict(zip(('drums','bass','other','vocals'),shares.tolist())),**pitch(vocal)}
            if shares[3]<.01:row['usable_pitch']=False
            row['voice_note']='Estimated source energies can contain leakage; not instrument identities or calibrated vocal presence.'
            temp=p.with_suffix('.tmp.json');temp.write_text(json.dumps(row,allow_nan=False),encoding='utf-8');temp.replace(p);windows.append(row)
            progress(.45+.5*(i+1)/len(starts),f'Vocal/pitch excerpt {i+1}/{len(starts)}')
    return {'model':MODEL,'weights_sha256':CONFIG['separation_sha256'],'sr':44100,'pitch_sr':8000,'windows':windows,
       'processing_seconds':time.perf_counter()-start,'reused_windows':reused,'sampled_seconds':sum(w['end']-w['start'] for w in windows),
       'source_duration':duration,'note':'Three sampled excerpts at most; no separation accuracy ground truth. Pitch is a vocal-contour proxy, not full-song melody transcription.'}
