# rir_sim_SE

Two-baseline small-room pipeline:
1. Baseline-(1): acoustic inversion from recorded IR folder.
2. Baseline-(2): RIR generation from json/inverted acoustic params.

Convolution is intentionally outside generator and done in `main.py`.

## Baseline-(1) and Baseline-(2)

```python
from config import RIRSimSEConfig
from rir_sim_se import prepare_state_from_cfg, generate_rir_from_state

cfg = RIRSimSEConfig(
    acoustic_param_source="invert",  # or "json"
    pulse_recording="/path/to/ir_folder",
    acoustic_state_json="./_out_rir_sim_se/acoustic_state.json",
)

state = prepare_state_from_cfg(cfg)   # invert or json-load (by cfg)
out = generate_rir_from_state(cfg, state)

rir = out["rir"]      # full RIR
ref1 = out["ref1"]    # early-dominant reference
ref2 = out["ref2"]    # full-band early reference
```

## State JSON

- Save: `save_state_json_after_invert=True` and set `acoustic_state_json`.
- Load: set `acoustic_param_source="json"` and set `acoustic_state_json`.

## Frequency-dependent absorption tuning

To reduce overly strong low-band reverberation:
- `low_freq_absorption_boost` (default `1.75`)
- `low_freq_absorption_cut_hz` (default `1500`)
- `high_freq_absorption_boost` (default `1.25`)
- `high_freq_absorption_start_hz` (default `9000`)

## Device EQ

Optional EQ can be applied to generated `rir/ref1/ref2`:
- `device_eq_enable`
- `device_eq_centers_hz`
- `device_eq_gains_db`

Default EQ is flat (`0 dB` on every band), so it keeps output unchanged.

## Demo

```bash
python main.py
```
