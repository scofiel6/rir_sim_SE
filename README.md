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

Output keys:
- `rir_path`
- `rir_ref_path`
- `wet_path`
- `ref_path`
- `fit`
- `meta`
- `engine_manifest`
- `mismatch_wet`
- `mismatch_ref`

## Reproducibility
- Engine vendoring + hash manifest: `reproducibility/engine_manifest.json`
- RNG policy checker: `tools/check_rng_policy.py`
- Rebuild manifest: `python tools/build_engine_manifest.py`
- Details: `REPRODUCIBILITY.md`

## Array and device realism
- `RIRSimSEConfig.mic_array_type`: `linear` or `circular`
- `RIRSimSEConfig.mic_num`: number of channels
- `RIRSimSEConfig.mic_spacing` / `mic_radius`: array geometry
- `RIRSimSEConfig.enable_channel_mismatch`: enable per-channel gain/delay/noise perturbation
- `RIRSimSEConfig.enable_channel_white_noise`: additive white-noise switch in post mismatch stage

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
