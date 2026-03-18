from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from utils import mono_resample_to_fs, to_mono


def read_audio_mono(path):
    x, fs = sf.read(str(path), dtype="float64")
    x = to_mono(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, int(round(float(fs)))


def resample_mono(x, fs_in, fs_out, allow_upsample=False):
    return mono_resample_to_fs(x, fs_in, fs_out, allow_upsample=allow_upsample)


def convolve_dry_rir(dry, rir):
    dry = np.asarray(dry, dtype=np.float64).reshape(-1)
    r = np.asarray(rir, dtype=np.float64)
    if r.ndim == 1:
        wet = fftconvolve(dry, r)[: len(dry)]
        return wet.astype(np.float64)
    if r.ndim == 2:
        # Channel-first layout: convolve one RIR per output channel.
        out = np.zeros((r.shape[0], len(dry)), dtype=np.float64)
        for ch in range(r.shape[0]):
            out[ch] = fftconvolve(dry, r[ch])[: len(dry)]
        return out
    raise ValueError(f"Unsupported RIR shape: {r.shape}")


def save_wav(path, x, fs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        # Internal layout is [ch, n], soundfile expects [n, ch].
        arr = arr.T
    sf.write(str(p), arr, int(fs))
