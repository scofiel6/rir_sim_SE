import numpy as np

from acoustic_inversion import invert_acoustic_params
from config import RIRSimSEConfig
from rir_generation import generate_single_rir


def _as_2d_ch_first(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _direct_index_from_rir(r, fs, search_ms=120.0):
    x = np.asarray(r, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0
    n_search = min(x.size, max(16, int(round(float(search_ms) * 1e-3 * float(fs)))))
    return int(np.argmax(np.abs(x[:n_search])))


def _select_sparse_peak_indices(x_abs, n_taps, min_gap):
    arr = np.asarray(x_abs, dtype=np.float64).reshape(-1)
    n_taps = int(max(0, n_taps))
    min_gap = int(max(1, min_gap))
    if n_taps <= 0 or arr.size == 0:
        return []

    order = np.argsort(arr)[::-1]
    chosen = []
    for idx in order:
        if float(arr[idx]) <= 0.0:
            break
        i = int(idx)
        keep = True
        for c in chosen:
            if abs(i - c) < min_gap:
                keep = False
                break
        if keep:
            chosen.append(i)
        if len(chosen) >= n_taps:
            break
    return sorted(chosen)


def _build_ref2_from_rir(
    rirs,
    ref1_rirs,
    fs,
    src_dist,
    early_ms,
    early_taps,
    min_tap_ms,
):
    """
    ref2:
    - full-band (no air-tilt transfer),
    - remove late reverb (direct + sparse early),
    - attenuation strength matched to ref1 early energy per channel.
    """
    r = _as_2d_ch_first(rirs)
    ref1 = _as_2d_ch_first(ref1_rirs)
    n_ch, n = r.shape
    fs = int(fs)

    early_n = int(max(1, round(float(early_ms) * 1e-3 * fs)))
    min_gap = int(max(1, round(float(min_tap_ms) * 1e-3 * fs)))
    out = np.zeros_like(r)

    direct_idx = []
    for ch in range(n_ch):
        rc = r[ch]
        i0 = _direct_index_from_rir(rc, fs=fs, search_ms=120.0)
        direct_idx.append(int(i0))

        h = np.zeros(n, dtype=np.float64)
        if 0 <= i0 < n:
            h[i0] = 1.0
            direct_amp = float(abs(rc[i0]))
            if direct_amp < 1e-9:
                direct_amp = 1.0

            st = int(i0 + 1)
            ed = int(min(n, i0 + early_n))
            if ed > st and int(early_taps) > 0:
                seg = rc[st:ed]
                peaks = _select_sparse_peak_indices(np.abs(seg), n_taps=int(early_taps), min_gap=min_gap)
                for ridx in peaks:
                    ii = int(st + ridx)
                    rel = float(np.clip(abs(rc[ii]) / direct_amp, 0.02, 0.65))
                    h[ii] += float(np.sign(rc[ii])) * rel

            # Match ref1 attenuation strength in the same early window.
            rms_ref1 = float(np.sqrt(np.mean(ref1[ch, st:ed] ** 2) + 1e-12)) if ed > st else 0.0
            rms_h = float(np.sqrt(np.mean(h[st:ed] ** 2) + 1e-12)) if ed > st else 0.0
            if rms_h > 0.0:
                h *= float(rms_ref1 / rms_h)

        out[ch] = h

    trace = {
        "src_dist": None if src_dist is None else float(src_dist),
        "direct_indices": direct_idx,
        "early_ms": float(early_ms),
        "early_taps": int(early_taps),
    }
    return out, trace


def invert_acoustic_state(cfg: RIRSimSEConfig, pulse_recording):
    """
    Step-1: invert acoustic state from impulse recordings.
    """
    gen, fit = invert_acoustic_params(cfg, pulse_recording)
    return {
        "gen": gen,
        "fit": fit,
        "pulse_recording": str(pulse_recording),
    }


def generate_rir_from_state(cfg: RIRSimSEConfig, state, seed=None):
    """
    Step-2: generate one full RIR + two refs from inversion state.
    No convolution here.
    """
    if seed is None:
        seed = int(cfg.seed) + 1

    gen = state["gen"]
    rirs, ref1_rirs, meta = generate_single_rir(
        gen=gen,
        seed=int(seed),
        use_drr_c50=bool(cfg.use_drr_c50),
        rir_seconds=float(cfg.rir_seconds),
        ref_early_ms=float(cfg.ref_early_ms),
        ref_late_tail_db=float(cfg.ref_late_tail_db),
    )
    rirs = _as_2d_ch_first(rirs)
    ref1_rirs = _as_2d_ch_first(ref1_rirs)

    ref2_rirs, ref2_trace = _build_ref2_from_rir(
        rirs=rirs,
        ref1_rirs=ref1_rirs,
        fs=int(cfg.fs),
        src_dist=meta.get("src_dist"),
        early_ms=float(cfg.ref_early_ms if cfg.ref2_early_ms is None else cfg.ref2_early_ms),
        early_taps=int(cfg.ref2_early_taps),
        min_tap_ms=float(cfg.ref2_min_tap_ms),
    )

    return {
        "rir": rirs,
        "ref1": ref1_rirs,
        "ref2": ref2_rirs,
        "meta": meta,
        "fit": state.get("fit"),
        "ref2_trace": ref2_trace,
    }


# Backward-compatible names.
def prepare_rir_sim_se_state(cfg: RIRSimSEConfig, pulse_recording):
    return invert_acoustic_state(cfg, pulse_recording)


def run_rir_sim_se(cfg: RIRSimSEConfig, pulse_recording=None, state=None, seed=None):
    if state is None:
        if pulse_recording is None:
            raise ValueError("Either `state` or `pulse_recording` must be provided.")
        state = invert_acoustic_state(cfg, pulse_recording)
    return generate_rir_from_state(cfg, state=state, seed=seed)


def clear_rir_sim_se_state_cache():
    # Cache removed for simpler pipeline.
    return None

