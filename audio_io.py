from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from engine.sound_field_sim.utils import mono_resample_to_fs, to_mono


def read_audio_mono(path):
    x, fs = sf.read(str(path), dtype="float64")
    x = to_mono(x)
    return x, int(round(float(fs)))


def resample_mono(x, fs_in, fs_out, allow_upsample=False):
    return mono_resample_to_fs(x, fs_in, fs_out, allow_upsample=allow_upsample)


def convolve_dry_rir(dry, rir):
    dry = np.asarray(dry, dtype=np.float64).reshape(-1)
    r = np.asarray(rir, dtype=np.float64)
    if r.ndim == 1:
        wet = fftconvolve(dry, r)[: len(dry)]
        return wet.astype(np.float64)

    # Multi-channel RIRs use internal layout [ch, n]. Each channel gets its own
    # convolution against the same dry signal and the result keeps the same layout.
    out = np.zeros((r.shape[0], len(dry)), dtype=np.float64)
    for ch in range(r.shape[0]):
        out[ch] = fftconvolve(dry, r[ch])[: len(dry)]
    return out


def save_wav(path, x, fs):
    p = Path(path)
    if p.suffix == "":
        # Batch scripts often pass bare stem names like `rir_000123`.
        # Adding `.wav` here keeps the call site short and always gives soundfile
        # an explicit format.
        p = p.with_suffix(".wav")
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        # The generator keeps audio as [ch, n]. soundfile expects [n, ch], so
        # file writing is the one place where the layout flips.
        arr = arr.T
    sf.write(str(p), arr, int(fs))
