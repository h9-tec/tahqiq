import io

import numpy as np
import soundfile as sf

from harfiyyan.io import decode_audio_bytes, load_audio_16k_mono, write_wav_16k


def _wav_bytes(sr: int, seconds: float, channels: int = 1) -> bytes:
    n = int(sr * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_bytes_mono_and_stereo():
    a, sr = decode_audio_bytes(_wav_bytes(48000, 0.5))
    assert sr == 48000 and a.ndim == 1 and len(a) == 24000 and a.dtype == np.float32
    b, sr2 = decode_audio_bytes(_wav_bytes(16000, 0.25, channels=2))
    assert sr2 == 16000 and b.ndim == 1 and len(b) == 4000


def test_write_then_load_resamples(tmp_path):
    a, sr = decode_audio_bytes(_wav_bytes(48000, 0.5))
    p = tmp_path / "x.wav"
    write_wav_16k(p, a, sr)
    y, dur = load_audio_16k_mono(p)
    assert abs(len(y) - 8000) <= 2 and abs(dur - 0.5) < 0.01
    assert sf.info(str(p)).samplerate == 16000
