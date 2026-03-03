import numpy as np

from eval.rir_metrics import compute_drr_c50_c80, estimate_rt60_bundle


def test_metrics_return_finite_values():
    fs = 16000
    n = int(1.2 * fs)
    t = np.arange(n, dtype=np.float64) / fs
    rir = np.zeros(n, dtype=np.float64)
    rir[80] = 1.0
    # Exponential tail.
    rir[81:] += 0.2 * np.exp(-t[: n - 81] / 0.25)

    rt = estimate_rt60_bundle(rir, fs=fs)
    dc = compute_drr_c50_c80(rir, fs=fs)

    assert rt["t20"] is None or np.isfinite(rt["t20"])
    assert rt["t30"] is None or np.isfinite(rt["t30"])
    assert rt["edt"] is None or np.isfinite(rt["edt"])
    assert np.isfinite(dc["drr"])
    assert np.isfinite(dc["c50"])
    assert np.isfinite(dc["c80"])
