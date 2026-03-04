from pathlib import Path

from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import RIRSimSEConfig
from rir_sim_se import generate_rir_from_state, prepare_state_from_cfg


if __name__ == "__main__":
    cfg = RIRSimSEConfig(
        fs=32000,
        seed=2026,
        use_drr_c50=True,
        # Keep shorter tail for small-room speech tasks.
        rir_seconds=1.4,
        # Baseline-(1): choose "invert" or "json".
        acoustic_param_source="invert",
        pulse_recording="/home/xukj/dataset_comsolTest/room_test",
        acoustic_state_json="./_out_rir_sim_se/acoustic_state.json",
        save_state_json_after_invert=True,
        # All inversion recordings are IR captures.
        inversion_drr_c50_mode="from_recording",
        inversion_rt60_min=0.12,
        inversion_rt60_max=0.70,
        # Increase low-frequency absorption to reduce boomy low band.
        low_freq_absorption_boost=1.45,
        low_freq_absorption_cut_hz=500.0,
        out_dir="./_out_rir_sim_se",
    )

    dry_wav = "/home/xukj/dataset_rir/sound_field_sim/test.wav"

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step-(1): inversion from IR folder OR load json (chosen by cfg.acoustic_param_source).
    state = prepare_state_from_cfg(cfg)

    # Step-(2): generate RIRs only (no convolution inside generator module).
    out = generate_rir_from_state(cfg, state=state)
    rir = out["rir"]
    ref1 = out["ref1"]
    ref2 = out["ref2"]

    save_wav(out_dir / "rir.wav", rir, cfg.fs)
    save_wav(out_dir / "rir_ref1.wav", ref1, cfg.fs)
    save_wav(out_dir / "rir_ref2.wav", ref2, cfg.fs)

    # Convolution is handled in main (outside RIR generation module).
    dry, dry_fs = read_audio_mono(dry_wav)
    dry = resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)
    wet = convolve_dry_rir(dry, rir)
    wet_ref1 = convolve_dry_rir(dry, ref1)
    wet_ref2 = convolve_dry_rir(dry, ref2)

    save_wav(out_dir / "dry.wav", dry, cfg.fs)
    save_wav(out_dir / "wet.wav", wet, cfg.fs)
    save_wav(out_dir / "wet_ref1.wav", wet_ref1, cfg.fs)
    save_wav(out_dir / "wet_ref2.wav", wet_ref2, cfg.fs)

    fit = out.get("fit", {})
    print("=== rir_sim_SE done ===")
    print("acoustic_param_source:", cfg.acoustic_param_source)
    print("state_json:", cfg.acoustic_state_json)
    print("rt60:", fit.get("rt60_median"), "range:", fit.get("rt60_p20"), fit.get("rt60_p80"))
    print("drr range:", fit.get("drr_db_p20_p80"), "c50 range:", fit.get("c50_db_p20_p80"))
    print("rir:", out_dir / "rir.wav")
    print("ref1:", out_dir / "rir_ref1.wav")
    print("ref2:", out_dir / "rir_ref2.wav")
    print("wet:", out_dir / "wet.wav")
