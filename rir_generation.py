import numpy as np


def generate_single_rir(gen, seed, use_drr_c50=True, rir_seconds=2.5):
    fs = int(gen.fs)
    rir_len = int(max(512, round(float(rir_seconds) * fs)))
    dry_delta = np.zeros(rir_len, dtype=np.float64)
    dry_delta[0] = 1.0

    y, _, meta = gen.generate(
        dry_delta,
        seed=int(seed),
        return_ref=False,
        ref_direct=True,
        branch="custom",
        normalize_output=False,
        apply_drr_c50=bool(use_drr_c50),
    )
    rir = np.asarray(y[0], dtype=np.float64)
    return rir, meta
