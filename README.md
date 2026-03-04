# rir_sim_SE

Two-baseline small-room pipeline:
1. Baseline-(1): acoustic inversion from recorded IR folder.
2. Baseline-(2): RIR generation from json/inverted acoustic params.

Convolution is intentionally outside generator and done in `main.py`.

## Baseline-(1) and Baseline-(2)

`cfg` and `acoustic_state` can both be file-driven.

## State JSON

- Save: `save_state_json_after_invert=True` and set `acoustic_state_json`.
- Load: set `acoustic_param_source="json"` and set `acoustic_state_json`.
- Path rule: when `acoustic_state_json` is relative (or empty), it is resolved near cfg json file.
  Default target is `<cfg_dir>/acoustic_state.json`.

## Config JSON

Use `configs/rir_sim_se_config.json` as the main config file.

Key physical fields:
- `material_center_freqs_hz`
- `material_absorption_curve`
- `material_scattering_curve`
- `material_face_absorption_scale`
- `material_face_scattering_scale`
- `mode_fmin_hz`
- `mode_fmax_hz`
- `mode_n_min` / `mode_n_max`
- `mode_rel_db_min` / `mode_rel_db_max`

These are applied inside the RIR generation strategy (absorption/scattering coefficients),
not as post-hoc low/high-band gain fixes.

## Device EQ

Optional EQ can be applied to generated `rir/ref1/ref2`:
- `device_eq_enable`
- `device_eq_centers_hz`
- `device_eq_gains_db`

Default EQ is flat (`0 dB` on every band), so it keeps output unchanged.

## Demo

```bash
python main.py --cfg ./configs/rir_sim_se_config.json --dry-wav /path/to/dry.wav
```

## From Files API

```python
from rir_sim_se import generate_rir_from_files

out = generate_rir_from_files(
    cfg_json_path="./configs/rir_sim_se_config.json",
    state_json_path="./_out_rir_sim_se/acoustic_state.json",  # optional
)
```
