# rir_sim_SE

Small-room RIR simulation for SE with two stages:
1. Invert acoustic state from measured impulse recordings.
2. Generate `rir/ref1/ref2` from `cfg + acoustic_state`.

Convolution stays outside the generator and is done in `main.py`.

## Minimal Runtime API

```python
from config import load_rir_sim_se_config
from rir_sim_se import load_acoustic_state_json, generate_rir_from_state

cfg = load_rir_sim_se_config("./configs/rir_sim_se_config.json")
state = load_acoustic_state_json(cfg, cfg.acoustic_state_json)
out = generate_rir_from_state(cfg, state=state)
```

## Main Demo

`main.py` uses one switch:
- `state_choice = "invert"`: invert from `cfg.pulse_recording`, then save/update `acoustic_state.json`.
- `state_choice = "json"`: load from `cfg.acoustic_state_json` directly.

Outputs are written to `tests/re/`:
- `rir.wav`, `rir_ref1.wav`, `rir_ref2.wav`
- `dry.wav`, `wet.wav`, `wet_ref1.wav`, `wet_ref2.wav`

## Config

Main config file: `configs/rir_sim_se_config.json`

Core physical controls include:
- `material_center_freqs_hz`
- `material_absorption_curve`
- `material_scattering_curve`
- `material_face_absorption_scale`
- `material_face_scattering_scale`
- `mode_fmin_hz`, `mode_fmax_hz`
- `mode_n_min`, `mode_n_max`
- `mode_rel_db_min`, `mode_rel_db_max`
