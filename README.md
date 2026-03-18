# rir_sim_SE

Small-room RIR simulation pipeline for speech enhancement.

The project has one runtime path:

1. Load config from `configs/rir_sim_se_config.json`.
2. Build or load an acoustic state.
3. Generate `rir`, `ref1`, and `ref2`.
4. Optionally convolve a dry signal for demo output.

The physical engine lives in `engine/sound_field_sim/base_engine.py`.

## Runtime Flow

`main.py` is the runnable demo entry.

It does three things:

1. Resolve config.
2. Choose one acoustic-state source:
   - `state_choice = "invert"`: fit room priors from `cfg.pulse_recording`, then save `cfg.acoustic_state_json`
   - `state_choice = "json"`: load `cfg.acoustic_state_json` directly
3. Generate RIR outputs and write wav files to `outputs/`

`main.py` also loads `cfg.dry_wav` and convolves it with the generated RIRs.
If `cfg.dry_wav` does not exist locally, it falls back to a short synthetic dry signal so the demo still runs.

## Main Files

- `main.py`: demo runner
- `rir_sim_se.py`: acoustic-state I/O, `ref1/ref2` construction, top-level generation API
- `acoustic_inversion.py`: build and tune `BaseEngine` from config and fitted priors
- `engine/sound_field_sim/base_engine.py`: physical RIR engine
- `audio_io.py`: wav I/O and dry/RIR convolution
- `utils.py`: shared audio helpers
- `config.py`: config schema and config loading

## Outputs

Running `python main.py` writes these files to `outputs/`:

- `rir.wav`
- `rir_ref1.wav`
- `rir_ref2.wav`
- `dry.wav`
- `wet.wav`
- `wet_ref1.wav`
- `wet_ref2.wav`

## Acoustic State

The acoustic state is a compact JSON snapshot of fitted room priors.

It stores the parameters needed for later synthesis, such as:

- fitted RT60 range
- fitted band RT60 profile
- fitted DRR/C50 ranges
- fitted room-size range
- basic recording metadata

This lets you invert once and regenerate many times without re-running fitting.

## Reference Signals

The generator produces three RIR-style outputs:

- `rir`: full simulated RIR
- `ref1`: direct + early part, with a weak late tail kept on purpose
- `ref2`: sparse direct/early reference derived from `rir` and shaped against `ref1`

`rir_sim_se.py` handles `ref1` and `ref2` construction.
`base_engine.py` is only responsible for the physical RIR generation side.

## Minimal API

Load a fitted state and generate one sample:

```python
from config import load_rir_sim_se_config
from rir_sim_se import load_acoustic_state_json, generate_rir_from_state

cfg = load_rir_sim_se_config("./configs/rir_sim_se_config.json")
state = load_acoustic_state_json(cfg, cfg.acoustic_state_json)
out = generate_rir_from_state(cfg, state)

rir = out["rir"]
ref1 = out["ref1"]
ref2 = out["ref2"]
meta = out["meta"]
```

Fit a new acoustic state from measured recordings:

```python
from config import load_rir_sim_se_config
from rir_sim_se import invert_acoustic_state, save_acoustic_state_json

cfg = load_rir_sim_se_config("./configs/rir_sim_se_config.json")
state = invert_acoustic_state(cfg, pulse_recording=cfg.pulse_recording)
save_acoustic_state_json(state, cfg.acoustic_state_json)
```

## Config Notes

Important config groups:

- Room prior:
  `room_size_hint`, `room_jitter_ratio`, `custom_room_range`, `generic_room_range`
- Inversion:
  `pulse_recording`, `inversion_rt60_min`, `inversion_rt60_max`, `inversion_drr_c50_mode`
- Reference shaping:
  `ref_early_ms`, `ref_late_tail_db`, `ref2_early_ms`, `ref2_early_taps`, `ref2_min_tap_ms`
- Engine material profile:
  `material_center_freqs_hz`, `material_absorption_curve`, `material_scattering_curve`
- Low-frequency modal tail:
  `mode_fmin_hz`, `mode_fmax_hz`, `mode_n_min`, `mode_n_max`, `mode_rel_db_min`, `mode_rel_db_max`
- Output device EQ:
  `device_eq_enable`, `device_eq_centers_hz`, `device_eq_gains_db`
- Array geometry:
  `mic_array_type`, `mic_positions_m`, `mic_num`, `mic_spacing`, `mic_radius`, `mic_position_jitter_m`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Before running:

- check `configs/rir_sim_se_config.json`
- make sure `pulse_recording` or `acoustic_state_json` points to valid local data
- set `state_choice` in `main.py` to `"invert"` or `"json"`
