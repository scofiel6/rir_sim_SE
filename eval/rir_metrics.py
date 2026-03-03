from __future__ import annotations

import numpy as np
from scipy import stats


def _safe_energy(x: np.ndarray) -> float:
    return float(np.sum(np.square(x)) + 1e-12)


def locate_direct_peak(rir: np.ndarray, fs: int, search_ms: float = 120.0) -> int:
    r = np.asarray(rir, dtype=np.float64).reshape(-1)
    n = r.size
    if n == 0:
        return 0
    n_search = min(n, max(8, int(round(search_ms * 1e-3 * fs))))
    return int(np.argmax(np.abs(r[:n_search])))


def schroeder_decay_db(rir: np.ndarray) -> np.ndarray:
    r = np.asarray(rir, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    e_rev = np.cumsum((r[::-1] ** 2))[::-1]
    e_rev /= max(float(e_rev[0]), 1e-12)
    return 10.0 * np.log10(np.maximum(e_rev, 1e-12))


def _fit_decay_time(decay_db: np.ndarray, fs: int, lo_db: float, hi_db: float) -> float | None:
    idx = np.where((decay_db <= lo_db) & (decay_db >= hi_db))[0]
    if idx.size < 8:
        return None
    t = idx.astype(np.float64) / float(fs)
    y = decay_db[idx]
    slope, intercept = np.polyfit(t, y, 1)
    if slope >= 0.0:
        return None
    # Extrapolate to 60 dB decay.
    return float(-60.0 / slope)


def estimate_rt60_bundle(rir: np.ndarray, fs: int) -> dict[str, float | None]:
    decay_db = schroeder_decay_db(rir)
    t20 = _fit_decay_time(decay_db, fs, lo_db=-5.0, hi_db=-25.0)
    t30 = _fit_decay_time(decay_db, fs, lo_db=-5.0, hi_db=-35.0)
    edt = _fit_decay_time(decay_db, fs, lo_db=0.0, hi_db=-10.0)
    return {"t20": t20, "t30": t30, "edt": edt}


def compute_drr_c50_c80(
    rir: np.ndarray,
    fs: int,
    direct_ms: float = 2.5,
    c50_ms: float = 50.0,
    c80_ms: float = 80.0,
) -> dict[str, float]:
    r = np.asarray(rir, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return {"drr": 0.0, "c50": 0.0, "c80": 0.0}

    peak = locate_direct_peak(r, fs=fs, search_ms=120.0)
    d0 = max(0, peak - 1)
    d1 = min(r.size, peak + max(1, int(round(direct_ms * 1e-3 * fs))))
    c50 = min(r.size, peak + max(1, int(round(c50_ms * 1e-3 * fs))))
    c80 = min(r.size, peak + max(1, int(round(c80_ms * 1e-3 * fs))))

    e_d = _safe_energy(r[d0:d1])
    e_e50 = _safe_energy(r[d1:c50])
    e_l50 = _safe_energy(r[c50:])
    e_e80 = _safe_energy(r[d1:c80])
    e_l80 = _safe_energy(r[c80:])

    drr = 10.0 * np.log10(e_d / (e_e50 + e_l50))
    c50v = 10.0 * np.log10((e_d + e_e50) / e_l50)
    c80v = 10.0 * np.log10((e_d + e_e80) / e_l80)
    return {"drr": float(drr), "c50": float(c50v), "c80": float(c80v)}


def direct_peak_ratio(rir_a: np.ndarray, rir_b: np.ndarray, fs: int) -> float:
    a = np.asarray(rir_a, dtype=np.float64).reshape(-1)
    b = np.asarray(rir_b, dtype=np.float64).reshape(-1)
    ia = locate_direct_peak(a, fs=fs)
    ib = locate_direct_peak(b, fs=fs)
    pa = float(abs(a[ia])) if a.size else 0.0
    pb = float(abs(b[ib])) if b.size else 0.0
    return float(pa / max(pb, 1e-12))


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 1.0
    return float(stats.ks_2samp(a, b, alternative="two-sided", method="auto").statistic)
