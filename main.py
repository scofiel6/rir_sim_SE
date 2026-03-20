from pathlib import Path

from audio_io import save_wav
from config import load_rir_sim_se_config
from rir_sim_se import (
    generate_rir_from_state,
    invert_acoustic_state,
    load_acoustic_state_json,
    save_acoustic_state_json,
)


def _load_state(cfg, state_choice):
    # `invert` refits room priors from measured impulse responses and overwrites
    # the saved state json. `json` skips inversion and reuses the checked-in fit.
    if state_choice == "invert":
        state = invert_acoustic_state(cfg, pulse_recording=cfg.pulse_recording)
        save_acoustic_state_json(state, cfg.acoustic_state_json)
        return state
    return load_acoustic_state_json(cfg, cfg.acoustic_state_json)


def main():
    base_dir = Path(__file__).resolve().parent
    save_dir = Path("/home/xukj/dataset_comsolTest/data_se/0319_")
    cfg = load_rir_sim_se_config(base_dir / "configs" / "rir_sim_se_config.json")
    state_choice = "json"  # "invert" or "json"
    state = _load_state(cfg, state_choice)

    # This batch export keeps one acoustic state fixed and only changes the seed.
    # The saved state still controls room priors and fitted decay statistics.
    # The seed changes the actual sample draw inside the engine: room sampling
    # within the configured range, mic/src placement, band perturbation, late
    # reverb realization, and low-frequency modal details.
    for ii in range(300000):
        save_name = f"rir_{ii:06d}.wav"
        out = generate_rir_from_state(cfg, state=state, seed=cfg.seed + ii)
        rir = out["rir"]
        save_wav(save_dir / save_name, rir, cfg.fs)
        print(f"Saved {save_name}")


if __name__ == "__main__":
    main()
