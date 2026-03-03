# rir_sim_SE

This project targets a small-room workflow:
1. real recorded pulse audio input,
2. acoustic parameter inversion,
3. RIR simulation via `im_rir_v2`,
4. SE-ready `wet/ref` data generation.

The `ref` signal is early-dominant, not pure direct-path:
- direct and early reflections are kept,
- late tail is strongly attenuated but still partially preserved.

## Files
- `config.py`: `RIRSimSEConfig` runtime config
- `audio_io.py`: audio I/O, resampling, convolution
- `imrir_adapter.py`: `im_rir_v2` compatibility adapter
- `acoustic_inversion.py`: acoustic inversion module
- `rir_generation.py`: single-RIR generation (full RIR + early-dominant ref RIR)
- `rir_sim_se.py`: end-to-end entry `run_rir_sim_se(...)`
- `main.py`: runnable demo

## Main API
```python
out = run_rir_sim_se(cfg, pulse_recording=pulse_recording, dry_wav=dry_wav)
```

Returned fields include:
- `rir_path`: full RIR
- `rir_ref_path`: early-dominant reference RIR
- `wet_path`: `dry * rir`
- `ref_path`: `dry * rir_ref` (SE supervision target)
- `fit`: inverted acoustic parameter summary

Speed knobs in `RIRSimSEConfig`:
- `max_fit_files`: limit number of recordings used in inversion.
- `rir_seconds`: cap generated RIR length.
- `ref_early_ms` and `ref_late_tail_db`: control ref strictness.

## Run
```bash
python main.py
```
