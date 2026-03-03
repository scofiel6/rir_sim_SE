import json
from pathlib import Path

import numpy as np

from acoustic_inversion import invert_acoustic_params
from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import RIRSimSEConfig
from rir_generation import generate_single_rir


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


def _as_2d_ch_first(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _apply_integer_delay(sig, shift):
    y = np.zeros_like(sig)
    if shift == 0:
        y[:] = sig
        return y
    if shift > 0:
        y[shift:] = sig[:-shift]
        return y
    y[:shift] = sig[-shift:]
    return y


def _apply_channel_mismatch(arr, fs, cfg, seed):
    x = _as_2d_ch_first(arr)
    if (not cfg.enable_channel_mismatch) or x.shape[0] <= 1:
        return x, {
            "enabled": False,
            "white_noise_enabled": bool(getattr(cfg, "enable_channel_white_noise", False)),
        }

    rng = np.random.default_rng(int(seed) + 333)
    out = np.zeros_like(x)
    gains_db = []
    delays_samp = []
    noise_dbfs = []

    delay_std_samp = float(max(0.0, cfg.channel_delay_us_std)) * 1e-6 * float(fs)
    white_noise_enabled = bool(getattr(cfg, "enable_channel_white_noise", False))
    n_lo, n_hi = float(min(cfg.channel_noise_dbfs_range)), float(max(cfg.channel_noise_dbfs_range))
    for ch in range(x.shape[0]):
        g_db = float(rng.normal(0.0, float(max(0.0, cfg.channel_gain_db_std))))
        g = float(10.0 ** (g_db / 20.0))
        d = int(np.round(rng.normal(0.0, delay_std_samp)))
        x_ch = _apply_integer_delay(x[ch], d) * g

        if white_noise_enabled:
            n_db = float(rng.uniform(n_lo, n_hi))
            n_rms = float(10.0 ** (n_db / 20.0))
            noise = rng.standard_normal(x_ch.shape[0]).astype(np.float64)
            noise = noise / max(float(np.sqrt(np.mean(noise**2) + 1e-12)), 1e-12) * n_rms
            out[ch] = x_ch + noise
        else:
            n_db = None
            out[ch] = x_ch

        gains_db.append(g_db)
        delays_samp.append(int(d))
        noise_dbfs.append(n_db)

    info = {
        "enabled": True,
        "white_noise_enabled": bool(white_noise_enabled),
        "gain_db": gains_db,
        "delay_samples": delays_samp,
        "noise_dbfs": noise_dbfs,
    }
    return out, info


def run_rir_sim_se(cfg: RIRSimSEConfig, pulse_recording, dry_wav=None):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_items = _resolve_fit_recordings(pulse_recording, cfg.max_fit_files, cfg.seed)
    gen, fit = invert_acoustic_params(cfg, fit_items)
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

    rir_path = out_dir / "rir.wav"
    rir_ref_path = out_dir / "rir_ref.wav"
    save_wav(rir_path, rirs, cfg.fs)
    save_wav(rir_ref_path, rirs_ref, cfg.fs)

    if dry_wav is not None and Path(dry_wav).exists():
        dry, dry_fs = read_audio_mono(dry_wav)
        dry = resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)
        dry_id = str(dry_wav)
    else:
        dry_id = "<synthetic>"
        t = np.arange(int(4.0 * cfg.fs), dtype=np.float64) / float(cfg.fs)
        dry = 0.15 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)

    wet = _as_2d_ch_first(convolve_dry_rir(dry, rirs))
    ref = _as_2d_ch_first(convolve_dry_rir(dry, rirs_ref))

    wet, mm_wet = _apply_channel_mismatch(wet, cfg.fs, cfg, cfg.seed + 101)
    ref, mm_ref = _apply_channel_mismatch(ref, cfg.fs, cfg, cfg.seed + 202)

    peak = max(
        float(np.max(np.abs(wet))) if wet.size > 0 else 0.0,
        float(np.max(np.abs(ref))) if ref.size > 0 else 0.0,
    )
    if peak > 0.99:
        gain = 0.99 / peak
        wet = wet * gain
        ref = ref * gain

    dry_path = out_dir / "dry.wav"
    wet_path = out_dir / "wet.wav"
    ref_path = out_dir / "ref.wav"
    save_wav(dry_path, dry, cfg.fs)
    save_wav(wet_path, wet, cfg.fs)
    save_wav(ref_path, ref, cfg.fs)

    summary = {
        "fs": int(cfg.fs),
        "seed": int(cfg.seed),
        "pulse_recording": str(pulse_recording),
        "fit_items_used": fit_items if isinstance(fit_items, list) else [str(fit_items)],
        "dry_source": dry_id,
        "n_channels": int(rirs.shape[0]),
        "use_drr_c50": bool(cfg.use_drr_c50),
        "rir_seconds": float(cfg.rir_seconds),
        "ref_early_ms": float(cfg.ref_early_ms),
        "ref_late_tail_db": float(cfg.ref_late_tail_db),
        "rir_path": str(rir_path),
        "rir_ref_path": str(rir_ref_path),
        "dry_path": str(dry_path),
        "wet_path": str(wet_path),
        "ref_path": str(ref_path),
        "mismatch_wet": mm_wet,
        "mismatch_ref": mm_ref,
        "channel_white_noise_enabled": bool(cfg.enable_channel_white_noise),
        "fit": fit,
        "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
        "engine_manifest": _load_engine_manifest(),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
