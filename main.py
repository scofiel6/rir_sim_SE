from pathlib import Path

from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import load_rir_sim_se_config
from rir_sim_se import (
    generate_rir_from_state,
    invert_acoustic_state,
    load_acoustic_state_json,
    save_acoustic_state_json,
)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "configs" / "rir_sim_se_config.json"
    result_dir = base_dir / "tests" / "re"
    state_choice = "json"  # "invert" or "json"

    cfg = load_rir_sim_se_config(cfg_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    if state_choice == "invert":
        state = invert_acoustic_state(cfg, pulse_recording=cfg.pulse_recording)
        save_acoustic_state_json(state, cfg.acoustic_state_json)
        state_source = f"from_recording: {cfg.pulse_recording}"
    else:
        state = load_acoustic_state_json(cfg, cfg.acoustic_state_json)
        state_source = f"from_acoustic_state_json: {cfg.acoustic_state_json}"

    out = generate_rir_from_state(cfg, state=state)
    rir = out["rir"]
    ref1 = out["ref1"]
    ref2 = out["ref2"]

    save_wav(result_dir / "rir.wav", rir, cfg.fs)
    save_wav(result_dir / "rir_ref1.wav", ref1, cfg.fs)
    save_wav(result_dir / "rir_ref2.wav", ref2, cfg.fs)

    dry, dry_fs = read_audio_mono(cfg.dry_wav)
    dry = resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)
    wet = convolve_dry_rir(dry, rir)
    wet_ref1 = convolve_dry_rir(dry, ref1)
    wet_ref2 = convolve_dry_rir(dry, ref2)

    save_wav(result_dir / "dry.wav", dry, cfg.fs)
    save_wav(result_dir / "wet.wav", wet, cfg.fs)
    save_wav(result_dir / "wet_ref1.wav", wet_ref1, cfg.fs)
    save_wav(result_dir / "wet_ref2.wav", wet_ref2, cfg.fs)

    print("state_source:", state_source)
