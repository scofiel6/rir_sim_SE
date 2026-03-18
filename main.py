from pathlib import Path
import numpy as np

from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import load_rir_sim_se_config
from rir_sim_se import (
    generate_rir_from_state,
    invert_acoustic_state,
    load_acoustic_state_json,
    save_acoustic_state_json,
)


def _load_state(cfg, state_choice):
    # The runtime has only two modes: reuse fitted state or refit from recordings.
    if state_choice == "invert":
        state = invert_acoustic_state(cfg, pulse_recording=cfg.pulse_recording)
        save_acoustic_state_json(state, cfg.acoustic_state_json)
        return state, f"from_recording: {cfg.pulse_recording}"
    state = load_acoustic_state_json(cfg, cfg.acoustic_state_json)
    return state, f"from_acoustic_state_json: {cfg.acoustic_state_json}"


def _load_dry_signal(cfg):
    dry_path = Path(cfg.dry_wav)
    if dry_path.exists():
        dry, dry_fs = read_audio_mono(dry_path)
        return resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)

    # Keep the demo runnable even when the config points to a workstation-only path.
    t = np.arange(int(3.0 * cfg.fs), dtype=np.float64) / float(cfg.fs)
    return 0.15 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)


def _write_outputs(result_dir, fs, dry, rir, ref1, ref2):
    # Keep file writing in one place so the demo path stays easy to scan.
    wet = convolve_dry_rir(dry, rir)
    wet_ref1 = convolve_dry_rir(dry, ref1)
    wet_ref2 = convolve_dry_rir(dry, ref2)

    save_wav(result_dir / "rir.wav", rir, fs)
    save_wav(result_dir / "rir_ref1.wav", ref1, fs)
    save_wav(result_dir / "rir_ref2.wav", ref2, fs)
    save_wav(result_dir / "dry.wav", dry, fs)
    save_wav(result_dir / "wet.wav", wet, fs)
    save_wav(result_dir / "wet_ref1.wav", wet_ref1, fs)
    save_wav(result_dir / "wet_ref2.wav", wet_ref2, fs)


def main():
    base_dir = Path(__file__).resolve().parent
    cfg = load_rir_sim_se_config(base_dir / "configs" / "rir_sim_se_config.json")
    result_dir = base_dir / "outputs"
    # Default to the checked-in acoustic state so the demo runs out of the box.
    state_choice = "json"  # "invert" or "json"

    result_dir.mkdir(parents=True, exist_ok=True)
    state, state_source = _load_state(cfg, state_choice)

    out = generate_rir_from_state(cfg, state=state)
    rir = out["rir"]
    ref1 = out["ref1"]
    ref2 = out["ref2"]
    dry = _load_dry_signal(cfg)

    _write_outputs(result_dir, cfg.fs, dry, rir, ref1, ref2)
    print("state_source:", state_source)


if __name__ == "__main__":
    main()
