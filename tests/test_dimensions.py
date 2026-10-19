import numpy as np

from soundmap import dimensions
from soundmap.separation import pitch


def test_core_silence_and_stereo_measurements():
    sr = 24000
    silence = dimensions.core(np.zeros(sr * 2, dtype=np.float32), np.zeros((sr * 2, 2), dtype=np.float32), sr)
    assert silence['silent']
    assert silence['tempo'] is None
    assert silence['harmony'] is None
    assert silence['texture']['stereo_width'] == 0
    assert np.isfinite(silence['dynamics']['crest_db'])


def test_core_tonal_audio_is_finite_and_keeps_source_timeline():
    sr = 24000
    t = np.arange(sr * 4, dtype=np.float32) / sr
    y = (.15 * np.sin(2 * np.pi * 440 * t) + .08 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    result = dimensions.core(y, np.column_stack([y, y]), sr)
    assert not result['silent']
    assert result['duration'] == 4
    assert result['harmony']['time'][0] == 0
    assert result['harmony']['time'][-1] < result['duration']
    assert result['key']['estimate'] is None or result['key']['estimate']['mode'] in ('major', 'minor')
    assert np.isfinite(np.asarray(result['harmony']['sequence'])).all()
    assert result['texture']['channel_correlation'] > .999


def test_vocal_pitch_proxy_retains_chronological_frames():
    sr = 8000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    y = (.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    result = pitch(y, sr)
    assert len(result['pitch_times']) == len(result['pitch_semitones']) == len(result['pitch_voiced']) == 64
    assert result['pitch_times'][0] == 0
    assert result['pitch_times'][-1] < 2
    assert result['usable_pitch']
    assert all(np.isfinite(result['pitch_semitones']))


def test_dimension_comparison_weights_and_missing_evidence(monkeypatch):
    base = {'timbre': [1., 0., 0.]}
    key = {'estimate': {'root': 0, 'mode': 'major', 'label': 'C major'}, 'candidates': [], 'correlation_gap': .2}

    def recording(tempo, groove):
        return {'silent': False, 'duration': 8., 'texture': {'stereo_width': .1},
                'dynamics': {'crest_db': 8., 'level_spread_db': 12., 'rms_dbfs': -10.},
                    'harmony': {'sequence': [[1., 0.]] * 8, 'time': list(range(8)), 'profile': [1.], 'note': 'test proxy'},
                'key': key, 'tempo': tempo,
                'groove': {'phase_histogram': groove, 'beats': 40}, 'deep': None}

    monkeypatch.setattr(dimensions, 'cached', lambda _id: recording(100., [1., 0.]) if _id == 'a' else recording(100., [1., 0.]))
    monkeypatch.setattr(dimensions, 'base_cached', lambda _id: base)
    monkeypatch.setattr(dimensions, 'tag_profile', lambda _id, category: {
        'vector': [0.1, .2], 'top': [], 'supported': False, 'windows': 2,
        'note': 'uncalibrated'} if category == 'instrumentation' else None)
    same = dimensions.compare_dimensions('a', 'b', {'tempo': 0, 'groove': 3})
    assert same['ready'] and not same['deep_ready']
    assert abs(same['overall'] - 100) < 1e-5
    assert same['coverage'] == .7
    rows = {r['id']: r for r in same['rows']}
    assert abs(rows['instrumentation']['score'] - 100) < 1e-5
    assert not rows['instrumentation']['aggregate']
    assert rows['instrumentation']['evidence']['reason']
    assert rows['vocals']['score'] is None
    assert rows['key']['score'] == 100
    assert rows['tempo']['score'] == 100
    assert rows['tempo']['evidence']['a_bpm'] == rows['tempo']['evidence']['b_bpm'] == 100
    assert np.isfinite([r['score'] for r in same['rows'] if r['score'] is not None]).all()
