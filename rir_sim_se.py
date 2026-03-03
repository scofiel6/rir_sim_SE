import json
from pathlib import Path

import numpy as np

from acoustic_inversion import invert_acoustic_params
from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import RIRSimSEConfig
from rir_generation import generate_single_rir


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


def run_rir_sim_se(cfg: RIRSimSEConfig, pulse_recording, dry_wav=None):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_items = _resolve_fit_recordings(pulse_recording, cfg.max_fit_files, cfg.seed)
    gen, fit = invert_acoustic_params(cfg, fit_items)
    rir, rir_ref, meta = generate_single_rir(
        gen,
        seed=cfg.seed + 1,
        use_drr_c50=cfg.use_drr_c50,
        rir_seconds=cfg.rir_seconds,
        ref_early_ms=cfg.ref_early_ms,
        ref_late_tail_db=cfg.ref_late_tail_db,
    )

    rir_path = out_dir / "rir.wav"
    rir_ref_path = out_dir / "rir_ref.wav"
    save_wav(rir_path, rir, cfg.fs)
    save_wav(rir_ref_path, rir_ref, cfg.fs)

    if dry_wav is not None and Path(dry_wav).exists():
        dry, dry_fs = read_audio_mono(dry_wav)
        dry = resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)
        dry_id = str(dry_wav)
    else:
        dry_id = "<synthetic>"
        t = np.arange(int(4.0 * cfg.fs), dtype=np.float64) / float(cfg.fs)
        dry = 0.15 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)

    wet = convolve_dry_rir(dry, rir)
    # SE reference target: early-dominant, late-suppressed.
    ref = convolve_dry_rir(dry, rir_ref)

    peak_wet = float(np.max(np.abs(wet))) if wet.size > 0 else 0.0
    peak_ref = float(np.max(np.abs(ref))) if ref.size > 0 else 0.0
    peak = max(peak_wet, peak_ref)
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
        "pulse_recording": str(pulse_recording),
        "fit_items_used": fit_items if isinstance(fit_items, list) else [str(fit_items)],
        "dry_source": dry_id,
        "use_drr_c50": bool(cfg.use_drr_c50),
        "rir_seconds": float(cfg.rir_seconds),
        "ref_early_ms": float(cfg.ref_early_ms),
        "ref_late_tail_db": float(cfg.ref_late_tail_db),
        "rir_path": str(rir_path),
        "rir_ref_path": str(rir_ref_path),
        "dry_path": str(dry_path),
        "wet_path": str(wet_path),
        "ref_path": str(ref_path),
        "fit": fit,
        "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
