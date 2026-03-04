from pathlib import Path

from audio_io import convolve_dry_rir, read_audio_mono, resample_mono, save_wav
from config import load_rir_sim_se_config
from rir_sim_se import (
    generate_rir_from_state,
    invert_acoustic_state,
    load_acoustic_state_json,
    save_acoustic_state_json,
)


def refreshState(state, state_json_path):
    save_acoustic_state_json(state, state_json_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / "configs" / "rir_sim_se_config.json"
    result_dir = base_dir / "testes" / "re"

    # 1) Read cfg (preset room/material/generation settings).
    cfg = load_rir_sim_se_config(cfg_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 2) Build state explicitly from recording or acoustic_state.json.
    # invert from recording
    state = invert_acoustic_state(cfg, pulse_recording=cfg.pulse_recording)
    refreshState(state, cfg.acoustic_state_json)
    state_source = f"from_recording: {cfg.pulse_recording}"
    # json
    # state = load_acoustic_state_json(cfg, cfg.acoustic_state_json)
    # state_source = f"from_acoustic_state_json: {cfg.acoustic_state_json}"

    # 3) Generate RIR outputs from cfg + state.
    out = generate_rir_from_state(cfg, state=state)
    rir = out["rir"]
    ref1 = out["ref1"]
    ref2 = out["ref2"]

    save_wav(result_dir / "rir.wav", rir, cfg.fs)
    save_wav(result_dir / "rir_ref1.wav", ref1, cfg.fs)
    save_wav(result_dir / "rir_ref2.wav", ref2, cfg.fs)

    # 4) Convolve dry with RIR/ref and save all outputs to ./testes/re.
    dry, dry_fs = read_audio_mono(cfg.dry_wav)
    dry = resample_mono(dry, dry_fs, cfg.fs, allow_upsample=False)
    wet = convolve_dry_rir(dry, rir)
    wet_ref1 = convolve_dry_rir(dry, ref1)
    wet_ref2 = convolve_dry_rir(dry, ref2)

    save_wav(result_dir / "dry.wav", dry, cfg.fs)
    save_wav(result_dir / "wet.wav", wet, cfg.fs)
    save_wav(result_dir / "wet_ref1.wav", wet_ref1, cfg.fs)
    save_wav(result_dir / "wet_ref2.wav", wet_ref2, cfg.fs)

    fit = out.get("fit", {})
    print("=== rir_sim_SE done ===")
    print("cfg:", str(cfg_path))
    print("state_source:", state_source)
    print("state_json:", cfg.acoustic_state_json)
    print("dry_wav:", cfg.dry_wav)
    print("rt60:", fit.get("rt60_median"), "range:", fit.get("rt60_p20"), fit.get("rt60_p80"))
    print("drr range:", fit.get("drr_db_p20_p80"), "c50 range:", fit.get("c50_db_p20_p80"))
    print("device_eq_enable:", out.get("meta", {}).get("device_eq_enable"))
    print("device_eq_gains_db:", out.get("meta", {}).get("device_eq_gains_db"))
    print("result_dir:", result_dir)
    print("rir:", result_dir / "rir.wav")
    print("ref1:", result_dir / "rir_ref1.wav")
    print("ref2:", result_dir / "rir_ref2.wav")
    print("wet:", result_dir / "wet.wav")
