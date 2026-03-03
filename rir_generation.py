import numpy as np


def generate_single_rir(gen, seed, use_drr_c50=True, rir_seconds=2.5):
    fs = int(gen.fs)
    rir_len = int(max(512, round(float(rir_seconds) * fs)))
    dry_delta = np.zeros(rir_len, dtype=np.float64)
    dry_delta[0] = 1.0

    # Delta excitation turns output into impulse response.
    # `y` is full RIR, `ref` is direct-path-only reference.
    y, ref, meta = gen.generate(
        dry_delta,
        seed=int(seed),
        return_ref=True,
        ref_direct=True,
        branch="custom",
        normalize_output=False,
        apply_drr_c50=bool(use_drr_c50),
    )
    rir = np.asarray(y[0], dtype=np.float64)
    rir_direct = np.asarray(ref[0], dtype=np.float64)
    return rir, rir_direct, meta
