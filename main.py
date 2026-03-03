from config import RIRSimSEConfig
from rir_sim_se import prepare_rir_sim_se_state, run_rir_sim_se


if __name__ == "__main__":
    cfg = RIRSimSEConfig(
        fs=32000,
        seed=2026,
        use_drr_c50=True,
        room_size_hint=(3.6, 3.8, 2.7),
        room_jitter_ratio=0.04,
        rir_seconds=2.0,
        ref_early_ms=20.0,
        ref_late_tail_db=-26.0,
        max_fit_files=12,
        ref_target_mode="dry_distance",
        enable_channel_mismatch=False,
        enable_channel_white_noise=False,
        out_dir="./_out_rir_sim_se",
    )

    pulse_recording = "/home/xukj/dataset_comsolTest/room_test"
    dry_wav = "/home/xukj/dataset_rir/sound_field_sim/test.wav"

    # Fit once and reuse state for fast repeated generation.
    state = prepare_rir_sim_se_state(cfg, pulse_recording=pulse_recording)
    out = run_rir_sim_se(cfg, state=state, dry_wav=dry_wav)

    print("=== rir_sim_SE done ===")
    print("rir:", out["rir_path"])
    print("rir_ref:", out["rir_ref_path"])
    print("dry:", out["dry_path"])
    print("wet:", out["wet_path"])
    print("ref:", out["ref_path"])
    print("n_channels:", out["n_channels"])
    print("fit_source:", out.get("fit_source"))
    print("fit_cache_path:", out.get("fit_cache_path"))
    print("ref_target_mode:", out.get("ref_target_mode"))
    print("ref_build_trace:", out.get("ref_build_trace"))
    print("drr/c50 strategy:", out["fit"].get("drr_c50_strategy"))
    print("rt60 median:", out["fit"].get("rt60_median"))
