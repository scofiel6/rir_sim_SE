import json
from pathlib import Path

import numpy as np

from acoustic_inversion import create_generator_from_fit, invert_acoustic_params
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


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _build_eq_magnitude_curve(fs, n_fft, centers_hz, gains_db):
    fs = int(fs)
    n_fft = int(n_fft)
    if fs <= 0 or n_fft <= 0:
        return np.ones(max(1, n_fft // 2 + 1), dtype=np.float64)

    c = np.asarray(list(centers_hz), dtype=np.float64).reshape(-1)
    g = np.asarray(list(gains_db), dtype=np.float64).reshape(-1)
    m = min(c.size, g.size)
    if m < 2:
        return np.ones(n_fft // 2 + 1, dtype=np.float64)
    c = c[:m]
    g = g[:m]

    order = np.argsort(c)
    c = c[order]
    g = g[order]

    ny = 0.5 * float(fs)
    keep = np.logical_and(c > 1.0, c < 0.999 * ny)
    c = c[keep]
    g = g[keep]
    if c.size < 2:
        return np.ones(n_fft // 2 + 1, dtype=np.float64)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    f0 = float(max(1.0, freqs[1] if freqs.size > 1 else 1.0))
    c_ext = np.concatenate(([f0], c, [0.999 * ny]))
    g_ext = np.concatenate(([g[0]], g, [g[-1]]))

    log_f = np.log(np.clip(freqs, f0, 0.999 * ny))
    log_c = np.log(np.clip(c_ext, f0, 0.999 * ny))
    gain_i = np.interp(log_f, log_c, g_ext)
    return np.power(10.0, gain_i / 20.0).astype(np.float64)


def _apply_device_eq_multich(x, fs, centers_hz, gains_db):
    r = _as_2d_ch_first(x)
    n = int(r.shape[1])
    g = np.asarray(list(gains_db), dtype=np.float64).reshape(-1)
    if g.size == 0 or np.max(np.abs(g)) < 1e-12:
        return r

    mag = _build_eq_magnitude_curve(fs=fs, n_fft=n, centers_hz=centers_hz, gains_db=gains_db)
    out = np.zeros_like(r)
    for ch in range(r.shape[0]):
        X = np.fft.rfft(r[ch], n=n)
        out[ch] = np.fft.irfft(X * mag, n=n)
    return out


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


def save_acoustic_state_json(state, json_path):
    """
    Save inversion result to json so it can be reused without re-fitting.
    """
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    fit = state.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("state['fit'] must be a dict")
    payload = {
        "schema_version": 1,
        "fit": _to_jsonable(fit),
        "pulse_recording": state.get("pulse_recording"),
    }
    p = Path(json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def load_acoustic_state_json(cfg: RIRSimSEConfig, json_path):
    """
    Load inversion result from json and rebuild generator from fitted params.
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"State json not found: {json_path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise ValueError(f"Invalid state json (missing fit dict): {json_path}")
    gen = create_generator_from_fit(cfg, fit)
    return {
        "gen": gen,
        "fit": fit,
        "pulse_recording": payload.get("pulse_recording"),
        "state_json_path": str(p),
    }


def prepare_state_from_cfg(cfg: RIRSimSEConfig):
    """
    Baseline-(1): prepare acoustic state by:
    - inverting from IR folder, or
    - loading existing state json.
    """
    src = str(cfg.acoustic_param_source).strip().lower()
    if src not in ("invert", "json"):
        raise ValueError(f"Unsupported acoustic_param_source: {cfg.acoustic_param_source!r}")

    if src == "json":
        if not cfg.acoustic_state_json:
            raise ValueError("cfg.acoustic_state_json is required when acoustic_param_source='json'")
        return load_acoustic_state_json(cfg, cfg.acoustic_state_json)

    if not cfg.pulse_recording:
        raise ValueError("cfg.pulse_recording is required when acoustic_param_source='invert'")
    state = invert_acoustic_state(cfg, pulse_recording=cfg.pulse_recording)
    if bool(cfg.save_state_json_after_invert) and cfg.acoustic_state_json:
        save_acoustic_state_json(state, cfg.acoustic_state_json)
    return state


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

    eq_applied = bool(cfg.device_eq_enable)
    eq_gain_db = np.asarray(list(cfg.device_eq_gains_db), dtype=np.float64).reshape(-1)
    if eq_applied and eq_gain_db.size > 0:
        rirs = _apply_device_eq_multich(
            rirs,
            fs=int(cfg.fs),
            centers_hz=cfg.device_eq_centers_hz,
            gains_db=cfg.device_eq_gains_db,
        )
        ref1_rirs = _apply_device_eq_multich(
            ref1_rirs,
            fs=int(cfg.fs),
            centers_hz=cfg.device_eq_centers_hz,
            gains_db=cfg.device_eq_gains_db,
        )
        ref2_rirs = _apply_device_eq_multich(
            ref2_rirs,
            fs=int(cfg.fs),
            centers_hz=cfg.device_eq_centers_hz,
            gains_db=cfg.device_eq_gains_db,
        )

    if isinstance(meta, dict):
        meta["device_eq_enable"] = bool(eq_applied)
        meta["device_eq_centers_hz"] = [float(v) for v in list(cfg.device_eq_centers_hz)]
        meta["device_eq_gains_db"] = [float(v) for v in list(cfg.device_eq_gains_db)]

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
        if pulse_recording is not None:
            state = invert_acoustic_state(cfg, pulse_recording)
        else:
            state = prepare_state_from_cfg(cfg)
    return generate_rir_from_state(cfg, state=state, seed=seed)


def clear_rir_sim_se_state_cache():
    # Cache removed for simpler pipeline.
    return None
