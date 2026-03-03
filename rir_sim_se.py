import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_inversion import create_generator_from_fit, invert_acoustic_params
from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import RIRSimSEConfig
from rir_generation import generate_single_rir


SOUND_SPEED_MPS = 343.0
_STATE_MEM_CACHE = {}


def _load_engine_manifest():
    p = Path(__file__).resolve().parent / "reproducibility" / "engine_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_fit_recordings(pulse_recording, max_fit_files, seed):
    if max_fit_files is None:
        return pulse_recording
    max_fit_files = int(max_fit_files)
    if max_fit_files <= 0:
        return pulse_recording

    p = Path(pulse_recording)
    if not p.is_dir():
        return pulse_recording

    exts = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
    files = []
    for ext in exts:
        files.extend([str(x) for x in p.rglob(f"*{ext}")])
    files = sorted(set(files))
    if len(files) <= max_fit_files:
        return files

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(files), size=max_fit_files, replace=False)
    idx = np.sort(idx)
    return [files[int(i)] for i in idx]


def _resolve_recording_items_for_fingerprint(recordings):
    if isinstance(recordings, list):
        return [str(x) for x in recordings]
    if isinstance(recordings, tuple):
        return [str(x) for x in list(recordings)]
    p = Path(recordings)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        exts = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
        files = []
        for ext in exts:
            files.extend([str(x) for x in p.rglob(f"*{ext}")])
        return sorted(set(files))
    return [str(recordings)]


def _recordings_fingerprint(recordings):
    items = _resolve_recording_items_for_fingerprint(recordings)
    info = []
    for it in items:
        p = Path(it)
        if p.exists() and p.is_file():
            st = p.stat()
            info.append(
                {
                    "path": str(p.resolve()) if p.exists() else str(p),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
        else:
            info.append({"path": str(p), "missing": True})
    raw = json.dumps(info, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _fit_cfg_signature(cfg: RIRSimSEConfig):
    sig = {
        "fs": int(cfg.fs),
        "use_drr_c50": bool(cfg.use_drr_c50),
        "room_size_hint": [float(v) for v in cfg.room_size_hint],
        "room_jitter_ratio": float(cfg.room_jitter_ratio),
        "mic_array_type": str(cfg.mic_array_type),
        "mic_num": int(cfg.mic_num),
        "mic_spacing": float(cfg.mic_spacing),
        "mic_radius": float(cfg.mic_radius),
        "mic_position_jitter_m": float(cfg.mic_position_jitter_m),
    }
    return sig


def _fit_cache_path(cfg: RIRSimSEConfig):
    if cfg.fit_cache_path:
        return Path(cfg.fit_cache_path)
    return Path(cfg.out_dir) / "fit_cache.json"


def _load_fit_cache(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_fit_cache(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_2d_ch_first(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _compute_array_aperture_m(cfg: RIRSimSEConfig):
    n = int(max(1, cfg.mic_num))
    j = float(max(0.0, cfg.mic_position_jitter_m))
    array_type = str(cfg.mic_array_type).lower().strip()
    if array_type == "circular":
        return float(max(0.0, 2.0 * cfg.mic_radius + 2.0 * j))
    return float(max(0.0, (n - 1) * cfg.mic_spacing + 2.0 * j))


def _compute_max_delay_samples(cfg: RIRSimSEConfig, fs: int):
    aperture = _compute_array_aperture_m(cfg)
    tdoa_max_s = aperture / SOUND_SPEED_MPS
    return int(np.ceil(tdoa_max_s * float(fs)))


def _sample_channel_mismatch_params(cfg: RIRSimSEConfig, fs: int, n_ch: int, seed: int):
    """
    Sample per-channel device mismatch with geometry-bounded delay.
    Delay bounds are tied to array aperture and sampling rate:
      |delay| <= ceil((aperture / c) * fs)
    """
    if (not cfg.enable_channel_mismatch) or n_ch <= 1:
        return {
            "enabled": False,
            "white_noise_enabled": bool(cfg.enable_channel_white_noise),
            "gain_db": [0.0] * n_ch,
            "delay_samples": [0] * n_ch,
            "noise_dbfs": [None] * n_ch,
            "max_delay_samples_physical": 0,
        }

    rng = np.random.default_rng(int(seed) + 333)
    max_delay_samples = _compute_max_delay_samples(cfg, fs)
    delay_std_samples = float(max(0.0, cfg.channel_delay_us_std)) * 1e-6 * float(fs)
    noise_lo = float(min(cfg.channel_noise_dbfs_range))
    noise_hi = float(max(cfg.channel_noise_dbfs_range))
    white_noise_enabled = bool(cfg.enable_channel_white_noise)

    gain_db = []
    delay_samples = []
    noise_dbfs = []
    for _ in range(n_ch):
        g_db = float(rng.normal(0.0, float(max(0.0, cfg.channel_gain_db_std))))
        d = int(np.round(rng.normal(0.0, delay_std_samples)))
        d = int(np.clip(d, -max_delay_samples, max_delay_samples))

        if white_noise_enabled:
            n_db = float(rng.uniform(noise_lo, noise_hi))
        else:
            n_db = None

        gain_db.append(g_db)
        delay_samples.append(d)
        noise_dbfs.append(n_db)

    return {
        "enabled": True,
        "white_noise_enabled": white_noise_enabled,
        "gain_db": gain_db,
        "delay_samples": delay_samples,
        "noise_dbfs": noise_dbfs,
        "max_delay_samples_physical": int(max_delay_samples),
    }


def _apply_integer_delay_zero_pad(sig_1d, shift_samples):
    """
    Zero-padding delay, no circular wrap-around for both positive/negative shift.
    """
    x = np.asarray(sig_1d, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    out = np.zeros_like(x)
    shift = int(shift_samples)
    if shift == 0:
        out[:] = x
        return out

    if shift > 0:
        # Delay to the right: prefix is zero, tail is truncated.
        if shift < n:
            out[shift:] = x[: n - shift]
        return out

    # Advance to the left: suffix is zero, head is truncated.
    k = -shift
    if k < n:
        out[: n - k] = x[k:]
    return out


def _apply_channel_mismatch(arr, fs, params, noise_seed):
    x = _as_2d_ch_first(arr)
    if x.shape[0] == 0:
        return x

    out = np.zeros_like(x)
    rng_noise = np.random.default_rng(int(noise_seed) + 991)
    for ch in range(x.shape[0]):
        g_db = float(params["gain_db"][ch]) if ch < len(params["gain_db"]) else 0.0
        g = float(10.0 ** (g_db / 20.0))
        d = int(params["delay_samples"][ch]) if ch < len(params["delay_samples"]) else 0

        y = _apply_integer_delay_zero_pad(x[ch], d) * g

        if bool(params.get("white_noise_enabled", False)):
            n_db = params["noise_dbfs"][ch]
            if n_db is not None:
                n_rms = float(10.0 ** (float(n_db) / 20.0))
                noise = rng_noise.standard_normal(y.shape[0]).astype(np.float64)
                noise = noise / max(float(np.sqrt(np.mean(noise**2) + 1e-12)), 1e-12) * n_rms
                y = y + noise

        out[ch] = y
    return out


def clear_rir_sim_se_state_cache():
    _STATE_MEM_CACHE.clear()


def prepare_rir_sim_se_state(cfg: RIRSimSEConfig, pulse_recording):
    """
    Prepare reusable state for fast repeated generation.

    Typical usage in training loop:
    1) state = prepare_rir_sim_se_state(cfg, pulse_recording)
    2) call run_rir_sim_se(cfg, state=state, dry_wav=...) many times
    """
    fit_items = _resolve_fit_recordings(pulse_recording, cfg.max_fit_files, cfg.seed)
    rec_fp = _recordings_fingerprint(fit_items)
    cfg_sig = _fit_cfg_signature(cfg)
    mem_key = json.dumps({"rec_fp": rec_fp, "cfg_sig": cfg_sig}, sort_keys=True, ensure_ascii=False)

    if (not cfg.fit_cache_force_refit) and (mem_key in _STATE_MEM_CACHE):
        return _STATE_MEM_CACHE[mem_key]

    fit_cache_path = _fit_cache_path(cfg)
    fit_source = "fresh_fit"
    fit = None
    gen = None

    if cfg.enable_fit_cache and (not cfg.fit_cache_force_refit):
        payload = _load_fit_cache(fit_cache_path)
        if isinstance(payload, dict):
            ok_fp = payload.get("recordings_fingerprint") == rec_fp
            ok_cfg = payload.get("cfg_signature") == cfg_sig
            fit_cached = payload.get("fit")
            if ok_fp and ok_cfg and isinstance(fit_cached, dict):
                fit = fit_cached
                gen = create_generator_from_fit(cfg, fit)
                fit_source = "disk_cache"

    if gen is None or fit is None:
        gen, fit = invert_acoustic_params(cfg, fit_items)
        if cfg.enable_fit_cache:
            payload = {
                "version": 1,
                "recordings_fingerprint": rec_fp,
                "cfg_signature": cfg_sig,
                "fit": fit,
            }
            _save_fit_cache(fit_cache_path, payload)

    state = {
        "gen": gen,
        "fit": fit,
        "fit_items": fit_items if isinstance(fit_items, list) else [str(fit_items)],
        "recordings_fingerprint": rec_fp,
        "cfg_signature": cfg_sig,
        "fit_cache_path": str(fit_cache_path),
        "fit_source": fit_source,
    }
    _STATE_MEM_CACHE[mem_key] = state
    return state


def run_rir_sim_se(
    cfg: RIRSimSEConfig,
    pulse_recording=None,
    dry_wav=None,
    dry_audio=None,
    dry_fs=None,
    state=None,
    save_outputs=None,
    return_audio_arrays=None,
):
    if save_outputs is None:
        save_outputs = bool(cfg.save_outputs)
    if return_audio_arrays is None:
        return_audio_arrays = bool(cfg.return_audio_arrays)

    out_dir = Path(cfg.out_dir)
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    if state is None:
        if pulse_recording is None:
            raise ValueError("Either `state` or `pulse_recording` must be provided.")
        state = prepare_rir_sim_se_state(cfg, pulse_recording)

    gen = state["gen"]
    fit = state["fit"]
    fit_items = state["fit_items"]

    rirs, rirs_ref, meta = generate_single_rir(
        gen,
        seed=cfg.seed + 1,
        use_drr_c50=cfg.use_drr_c50,
        rir_seconds=cfg.rir_seconds,
        ref_early_ms=cfg.ref_early_ms,
        ref_late_tail_db=cfg.ref_late_tail_db,
    )

    rirs = _as_2d_ch_first(rirs)
    rirs_ref = _as_2d_ch_first(rirs_ref)

    if dry_audio is not None:
        dry = np.asarray(dry_audio, dtype=np.float64)
        if dry.ndim != 1:
            raise ValueError("`dry_audio` must be 1-D mono waveform.")
        if dry_fs is None:
            raise ValueError("`dry_fs` must be provided when `dry_audio` is given.")
        dry = resample_mono(dry, int(dry_fs), cfg.fs, allow_upsample=False)
        dry_id = f"<array:{int(dry.shape[0])}@{int(dry_fs)}>"
    elif dry_wav is not None and Path(dry_wav).exists():
        dry, dry_fs = read_audio_mono(dry_wav)
        dry = resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)
        dry_id = str(dry_wav)
    else:
        dry_id = "<synthetic>"
        t = np.arange(int(4.0 * cfg.fs), dtype=np.float64) / float(cfg.fs)
        dry = 0.15 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)

    wet = _as_2d_ch_first(convolve_dry_rir(dry, rirs))
    ref = _as_2d_ch_first(convolve_dry_rir(dry, rirs_ref))

    mismatch = _sample_channel_mismatch_params(
        cfg=cfg,
        fs=int(cfg.fs),
        n_ch=int(wet.shape[0]),
        seed=int(cfg.seed),
    )
    wet = _apply_channel_mismatch(wet, cfg.fs, mismatch, noise_seed=cfg.seed + 101)
    ref = _apply_channel_mismatch(ref, cfg.fs, mismatch, noise_seed=cfg.seed + 202)

    peak = max(
        float(np.max(np.abs(wet))) if wet.size > 0 else 0.0,
        float(np.max(np.abs(ref))) if ref.size > 0 else 0.0,
    )
    if peak > 0.99:
        gain = 0.99 / peak
        wet = wet * gain
        ref = ref * gain

    if save_outputs:
        rir_path = out_dir / "rir.wav"
        rir_ref_path = out_dir / "rir_ref.wav"
        dry_path = out_dir / "dry.wav"
        wet_path = out_dir / "wet.wav"
        ref_path = out_dir / "ref.wav"
        save_wav(rir_path, rirs, cfg.fs)
        save_wav(rir_ref_path, rirs_ref, cfg.fs)
        save_wav(dry_path, dry, cfg.fs)
        save_wav(wet_path, wet, cfg.fs)
        save_wav(ref_path, ref, cfg.fs)
    else:
        rir_path = None
        rir_ref_path = None
        dry_path = None
        wet_path = None
        ref_path = None

    summary = {
        "fs": int(cfg.fs),
        "seed": int(cfg.seed),
        "pulse_recording": None if pulse_recording is None else str(pulse_recording),
        "fit_items_used": fit_items,
        "fit_source": state.get("fit_source"),
        "fit_cache_path": state.get("fit_cache_path"),
        "recordings_fingerprint": state.get("recordings_fingerprint"),
        "dry_source": dry_id,
        "n_channels": int(rirs.shape[0]),
        "use_drr_c50": bool(cfg.use_drr_c50),
        "rir_seconds": float(cfg.rir_seconds),
        "ref_early_ms": float(cfg.ref_early_ms),
        "ref_late_tail_db": float(cfg.ref_late_tail_db),
        "rir_path": None if rir_path is None else str(rir_path),
        "rir_ref_path": None if rir_ref_path is None else str(rir_ref_path),
        "dry_path": None if dry_path is None else str(dry_path),
        "wet_path": None if wet_path is None else str(wet_path),
        "ref_path": None if ref_path is None else str(ref_path),
        "mismatch": mismatch,
        "channel_white_noise_enabled": bool(cfg.enable_channel_white_noise),
        "fit": fit,
        "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
        "engine_manifest": _load_engine_manifest(),
    }

    if save_outputs:
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["summary_path"] = str(summary_path)
    else:
        summary["summary_path"] = None

    if return_audio_arrays:
        summary["dry"] = dry
        summary["wet"] = wet
        summary["ref"] = ref
        summary["rir"] = rirs
        summary["rir_ref"] = rirs_ref

    return summary
