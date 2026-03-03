import numpy as np

from rir_generation import _build_ref_rir_from_full_rir


def test_ref_builder_keeps_early_and_suppresses_late():
    fs = 16000
    rir = np.zeros(8000, dtype=np.float64)
    rir[120] = 1.0
    rir[200] = 0.4
    rir[3500] = 0.25

    ref = _build_ref_rir_from_full_rir(rir, fs=fs, ref_early_ms=20.0, ref_late_tail_db=-26.0)

    assert np.isclose(ref[120], 1.0)
    assert np.isclose(ref[200], 0.4)
    assert abs(ref[3500]) < abs(rir[3500])
