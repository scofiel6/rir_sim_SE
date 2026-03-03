# rir_sim_SE

Minimal small-room pipeline (IR inversion -> RIR generation):
1. `cfg + impulse recordings -> state`
2. `state -> rir + ref1 + ref2`
3. convolution is done in `main.py` (outside generation module)

## Core API

```python
from config import RIRSimSEConfig
from rir_sim_se import invert_acoustic_state, generate_rir_from_state

cfg = RIRSimSEConfig()
state = invert_acoustic_state(cfg, pulse_recording="/path/to/ir_recordings")
out = generate_rir_from_state(cfg, state)

rir = out["rir"]     # full RIR
ref1 = out["ref1"]   # early-dominant ref (air/band behavior kept)
ref2 = out["ref2"]   # full-band ref, late removed, attenuation matched to ref1
```

## Notes

- This project assumes inversion recordings are impulse responses.
- Default config is conservative for small-room SE:
  - shorter RIR
  - bounded RT60
  - DRR/C50 inversion from recordings

## Demo

```bash
python main.py
```

`main.py` performs:
- inversion
- RIR/ref generation
- dry convolution (`wet`, `wet_ref1`, `wet_ref2`)
- wav export
