# rir_sim_SE

Small-room RIR simulation for speech-enhancement data.

## What It Does

The project runs one simple chain:

1. Load priors from `configs/rir_sim_se_config.json`.
2. Load or fit an acoustic state.
3. Use `engine/sound_field_sim/base_engine.py` to sample one physical RIR.
4. Derive `ref1` and `ref2` from that same RIR.
5. Optionally convolve a dry signal.

`base_engine.py` is the only physical engine.  
`rir_sim_se.py` handles state I/O and `ref1`/`ref2`.  
`main.py` is the batch/export entry.

## Main Files

- `main.py`: batch generation script
- `config.py`: config schema and loader
- `acoustic_inversion.py`: build engine from config and apply fitted priors
- `rir_sim_se.py`: state save/load and top-level generation
- `audio_io.py`: wav I/O and dry/RIR convolution
- `engine/sound_field_sim/base_engine.py`: physical RIR engine

## Run

```bash
pip install -r requirements.txt
python main.py
```

Before running, check:

- `configs/rir_sim_se_config.json`
- `state_choice` in `main.py`
- output directory in `main.py`

## State

`acoustic_state.json` stores fitted acoustic priors such as RT60, DRR/C50, and
early-reflection summaries.

Room-size prior still comes from config unless a later inversion stage writes
`estimated_room_range`.

## Test

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```
