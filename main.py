import copy
from dataclasses import replace

from acoustic_inversion import create_generator_from_fit
from config import RIRSimSEConfig
from rir_sim_se import prepare_rir_sim_se_state, run_rir_sim_se


if __name__ == "__main__":
    cfg_a = RIRSimSEConfig(
        fs=32000,
        seed=2026,
        use_drr_c50=True,
        room_size_hint=(3.6, 3.8, 2.7),
        room_jitter_ratio=0.04,
        rir_seconds=2.0,
        generation_profile="fit_aligned",
        ref_early_ms=20.0,
        ref_late_tail_db=-26.0,
        max_fit_files=12,
        ref2_enabled=True,
        enable_channel_mismatch=False,
        enable_channel_white_noise=False,
        out_dir="./_out_rir_sim_se/A_fit_aligned",
    )
    cfg_b = replace(
        cfg_a,
        generation_profile="smallroom_conservative",
        out_dir="./_out_rir_sim_se/B_smallroom_conservative",
    )

    pulse_recording = "/home/xukj/dataset_comsolTest/room_test"
    dry_wav = "/home/xukj/dataset_rir/sound_field_sim/test.wav"

    # Fit once (A), then re-profile same fit for B without re-inversion.
    state_a = prepare_rir_sim_se_state(cfg_a, pulse_recording=pulse_recording)
    out_a = run_rir_sim_se(cfg_a, state=state_a, dry_wav=dry_wav)

    state_b = dict(state_a)
    state_b["fit"] = copy.deepcopy(state_a["fit"])
    state_b["gen"] = create_generator_from_fit(cfg_b, state_b["fit"])
    state_b["fit_source"] = f"{state_a.get('fit_source', 'unknown')}+reprofile"
    out_b = run_rir_sim_se(cfg_b, state=state_b, dry_wav=dry_wav)

    print("=== rir_sim_SE A/B done ===")
    print("[A] profile:", out_a.get("generation_profile"))
    print("[A] rir_seconds used/cfg:", out_a.get("rir_seconds"), out_a.get("rir_seconds_cfg"))
    print("[A] wet/ref/ref2:", out_a.get("wet_path"), out_a.get("ref_path"), out_a.get("ref2_path"))
    print("[A] applied ranges:", out_a["fit"].get("applied_ranges"))

    print("[B] profile:", out_b.get("generation_profile"))
    print("[B] rir_seconds used/cfg:", out_b.get("rir_seconds"), out_b.get("rir_seconds_cfg"))
    print("[B] wet/ref/ref2:", out_b.get("wet_path"), out_b.get("ref_path"), out_b.get("ref2_path"))
    print("[B] applied ranges:", out_b["fit"].get("applied_ranges"))
