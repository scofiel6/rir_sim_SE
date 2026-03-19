from math import gcd

import numpy as np
from scipy.signal import resample_poly


def to_mono(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.mean(arr, axis=1)
    raise ValueError(f"Unsupported audio shape: {arr.shape}")


def resample_poly_1d(x, fs_in, fs_out, allow_upsample=False):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    fs_in = int(round(float(fs_in)))
    fs_out = int(round(float(fs_out)))
    if fs_in == fs_out:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if fs_out > fs_in and not allow_upsample:
        raise ValueError(
            f"Upsampling disabled: {fs_in} -> {fs_out}. "
            "Please use audio with fs >= target fs."
        )
    g = gcd(fs_out, fs_in)
    up = fs_out // g
    down = fs_in // g
    out = resample_poly(arr, up, down).astype(np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def mono_resample_to_fs(x, fs_in, fs_out, allow_upsample=False):
    arr = to_mono(x)
    return resample_poly_1d(arr, fs_in, fs_out, allow_upsample=allow_upsample)
