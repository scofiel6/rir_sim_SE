from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def read_audio_mono(path):
    x, fs = sf.read(str(path), dtype="float64")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if x.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, int(round(float(fs)))


def resample_mono(x, fs_in, fs_out, allow_upsample=False):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs_in = int(round(float(fs_in)))
    fs_out = int(round(float(fs_out)))
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"Invalid sample rates: fs_in={fs_in}, fs_out={fs_out}")
    if fs_in == fs_out:
        return x
    if fs_out > fs_in and not allow_upsample:
        raise ValueError(
            f"Upsampling disabled: {fs_in} -> {fs_out}. "
            "Please use dry source with fs >= target fs."
        )
    g = gcd(fs_out, fs_in)
    up = fs_out // g
    down = fs_in // g
    y = resample_poly(x, up, down).astype(np.float64)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def convolve_dry_rir(dry, rir):
    dry = np.asarray(dry, dtype=np.float64).reshape(-1)
    rir = np.asarray(rir, dtype=np.float64).reshape(-1)
    wet = fftconvolve(dry, rir)[: len(dry)]
    return wet.astype(np.float64)


def save_wav(path, x, fs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.asarray(x, dtype=np.float32), int(fs))
