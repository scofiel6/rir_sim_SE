import numpy as np


def _build_ref_rir_from_full_rir(rir, fs, ref_early_ms=20.0, ref_late_tail_db=-26.0):
    """
    Build SE reference RIR from full RIR:
    1) keep direct + early reflections,
    2) keep a small attenuated late tail (not hard-zero) to avoid over-clean target.
    """
    r = np.asarray(rir, dtype=np.float64).reshape(-1)
    n = r.size
    if n == 0:
        return r

    fs = int(fs)
    # Direct arrival search window should cover practical source delays.
    # 120 ms is conservative for far source cases.
    search_n = min(n, max(16, int(0.12 * fs)))
    idx_direct = int(np.argmax(np.abs(r[:search_n])))
    early_n = max(1, int(round(float(ref_early_ms) * 1e-3 * fs)))
    cut = min(n, idx_direct + early_n)

    ref = np.zeros_like(r)
    ref[:cut] = r[:cut]

    if cut < n:
        # Preserve a little late energy with smooth decay to reduce target mismatch.
        tail_gain0 = float(10.0 ** (float(ref_late_tail_db) / 20.0))
        tail_len = n - cut
        t = np.linspace(0.0, 1.0, tail_len, endpoint=False)
        env = tail_gain0 * np.exp(-4.0 * t)
        ref[cut:] = r[cut:] * env
    return ref


def generate_single_rir(
    gen,
    seed,
    use_drr_c50=True,
    rir_seconds=2.0,
    ref_early_ms=20.0,
    ref_late_tail_db=-26.0,
):
    fs = int(gen.fs)
    rir_len = int(max(512, round(float(rir_seconds) * fs)))
    dry_delta = np.zeros(rir_len, dtype=np.float64)
    dry_delta[0] = 1.0

    # We only ask for full RIR here (return_ref=False) for speed.
    # Ref target is then built from full RIR with early-dominant shaping.
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
    rir_ref = _build_ref_rir_from_full_rir(
        rir,
        fs=fs,
        ref_early_ms=float(ref_early_ms),
        ref_late_tail_db=float(ref_late_tail_db),
    )
    return rir, rir_ref, meta
