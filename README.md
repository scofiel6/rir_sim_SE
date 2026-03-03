# rir_sim_SE

Minimal modular pipeline for:
1) acoustic parameter inversion from recorded pulse audio,
2) RIR generation via `im_rir_v2` backend wrappers in `sound_field_sim/rirGen_baseSE.py`,
3) dry->wet convolution demo.

## Files
- `config.py`: runtime config dataclass
- `audio_io.py`: mono I/O, resample_poly, convolution helpers
- `imrir_adapter.py`: im_rir_v2 compatibility adapter re-export
- `acoustic_inversion.py`: inversion module
- `rir_generation.py`: single-RIR generation module
- `pipeline.py`: end-to-end orchestration
- `main.py`: runnable entry

## Run
```bash
python main.py
```
