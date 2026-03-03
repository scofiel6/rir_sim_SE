# rir_sim_SE

This project targets a small-room workflow:
1. real recorded pulse audio input,
2. acoustic parameter inversion,
3. RIR simulation via `im_rir_v2`,
4. SE-ready `wet/ref` data generation.

The `ref` signal is direct-path only. It does not include late reverberation or reflection tail.

## Files
- `config.py`: `RIRSimSEConfig` runtime config
- `audio_io.py`: audio I/O, resampling, convolution
- `imrir_adapter.py`: `im_rir_v2` compatibility adapter
- `acoustic_inversion.py`: acoustic inversion module
- `rir_generation.py`: single-RIR generation (full RIR + direct-path RIR)
- `rir_sim_se.py`: end-to-end entry `run_rir_sim_se(...)`
- `main.py`: runnable demo

## Main API
```python
out = run_rir_sim_se(cfg, pulse_recording=pulse_recording, dry_wav=dry_wav)
```

Returned fields include:
- `rir_path`: full RIR
- `rir_direct_path`: direct-path-only RIR
- `wet_path`: `dry * rir`
- `ref_path`: `dry * rir_direct` (SE supervision target)
- `fit`: inverted acoustic parameter summary

## Run
```bash
python main.py
```
