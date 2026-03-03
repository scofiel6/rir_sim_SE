# rir_sim_SE

Small-room SE data project:
1. real pulse recordings -> acoustic inversion,
2. inversion priors -> `im_rir_v2` physical simulation,
3. full RIR + SE reference RIR generation,
4. dry -> wet/ref synthesis for SE training.

`ref` is early-dominant (direct + early reflections) with an attenuated late tail.
Multi-channel arrays are supported (linear/circular), with optional channel mismatch.

## Core files
- `config.py`: `RIRSimSEConfig`
- `acoustic_inversion.py`: inversion flow
- `rir_generation.py`: full-RIR/ref-RIR builder
- `rir_sim_se.py`: main API `run_rir_sim_se(...)`
- `engine/sound_field_sim/`: vendored physical engine sources
- `eval/`: metric and distribution evaluation utilities

## Main API
```python
out = run_rir_sim_se(cfg, pulse_recording=pulse_recording, dry_wav=dry_wav)
```

Fast integration pattern (fit once, generate many):
```python
state = prepare_rir_sim_se_state(cfg, pulse_recording=pulse_recording)
out = run_rir_sim_se(cfg, state=state, dry_wav=dry_wav)
```

Training-time in-memory fast path (no per-step disk I/O):
```python
cfg.save_outputs = False
cfg.return_audio_arrays = True
state = prepare_rir_sim_se_state(cfg, pulse_recording=pulse_recording)
out = run_rir_sim_se(cfg, state=state, dry_audio=dry_np, dry_fs=dry_fs)
wet = out["wet"]
ref = out["ref"]
```

Output keys:
- `rir_path`
- `rir_ref_path`
- `wet_path`
- `ref_path`
- `fit`
- `meta`
- `engine_manifest`
- `mismatch`
- `fit_source`
- `fit_cache_path`
- `recordings_fingerprint`
- `ref1_build_trace`
- `ref2_build_trace`
- `ref2_path`
- `rir_ref2_path`

## Reproducibility
- Engine vendoring + hash manifest: `reproducibility/engine_manifest.json`
- RNG policy checker: `tools/check_rng_policy.py`
- Rebuild manifest: `python tools/build_engine_manifest.py`
- Details: `REPRODUCIBILITY.md`
- Fit cache controls in `RIRSimSEConfig`:
  - `enable_fit_cache`
  - `fit_cache_path`
  - `fit_cache_force_refit`
- Online speed controls in `RIRSimSEConfig`:
  - `save_outputs`
  - `return_audio_arrays`

## Ref targets (SE supervision)
- `ref` (ref1):
  - early-dominant RIR reference (keeps frequency-dependent air/band behavior, suppresses late reverberation).
- `ref2`:
  - broadband distance-attenuated reference with sparse early taps.
  - suppresses late reverberation while avoiding over-attenuated high-frequency target.
  - key knobs:
    - `ref2_enabled`
    - `ref2_early_ms`
    - `ref2_distance_ref_m`
    - `ref2_distance_power`
    - `ref2_distance_gain_min` / `ref2_distance_gain_max`
    - `ref2_early_taps`
    - `ref2_min_tap_ms`

## Array and device realism
- `RIRSimSEConfig.mic_array_type`: `linear` or `circular`
- `RIRSimSEConfig.mic_num`: number of channels
- `RIRSimSEConfig.mic_spacing` / `mic_radius`: array geometry
- `RIRSimSEConfig.enable_channel_mismatch`: enable per-channel gain/delay/noise perturbation
- `RIRSimSEConfig.enable_channel_white_noise`: additive white-noise switch in post mismatch stage
- Delay mismatch is physically bounded by aperture and fs:
  - `|delay_samples| <= ceil((array_aperture / c) * fs)`
- Delay is applied by zero-padding shift, never circular wrap-around.

## Multi-channel DRR/C50 consistency
- DRR/C50 shaping uses shared direct/early/late gains across channels.
- This avoids per-channel independent shaping that can distort ITD/ILD cues.

## Material/scattering model
- Material sampling uses a controllable material library (wall/floor/ceiling categories).
- Each material has frequency-dependent absorption and scattering curves.
- Per-face sampling applies bounded perturbation and logs trace info in metadata.

For pure measured-room inversion -> RIR -> convolution (no additive white noise), use:
- `enable_channel_mismatch=False`
- `enable_channel_white_noise=False`

## Quality evaluation
- Compare real/sim RIR distributions:
```bash
python eval/evaluate_rir_sets.py --real /path/to/real --sim /path/to/sim --out eval_report.json
```
- Metrics include: `T20/T30/EDT`, `DRR`, `C50`, `C80`, and KS distance.

## Dev baseline
- Install:
```bash
pip install -r requirements-dev.txt
```
- Run checks:
```bash
ruff check .
black --check .
python tools/check_rng_policy.py
pytest
```

## Run demo
```bash
python main.py
```
