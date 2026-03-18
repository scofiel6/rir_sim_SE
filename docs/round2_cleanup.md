# Round 2 Cleanup

This repository has one physical engine and one runtime path.

## Result

- The only physical RIR engine is `engine/sound_field_sim/base_engine.py`.
- `acoustic_inversion.py` only builds and tunes `BaseEngine` from config or fitted state.
- `rir_sim_se.py` only handles acoustic-state I/O and `rir/ref1/ref2` construction.
- `main.py` is only a runnable demo entry.

## What Was Removed

- Old compatibility layers and adapter-style wrappers are gone.
- Generated demo artifacts are no longer part of the repository layout.
- The old fake `tests/re` asset folder is no longer treated as a test suite.
- Broken CI references to non-existent tooling were removed.

## Interface Tightening

- Config remains the single source of truth for physical priors.
- Acoustic state stores fitted inversion outputs, not duplicated config priors.
- Linear array geometry can now come from explicit `mic_positions_m`.
- Demo defaults to `state_choice = "json"` so the repository runs without workstation-only input paths.

## Tests

The repository now uses real automated smoke tests under `tests/`:

- config loading
- `BaseEngine` smoke generation
- state-driven `rir/ref1/ref2` generation

These tests verify that:

- `base_engine` is the runtime engine
- late reverb and source-radiation traces are present
- the checked-in acoustic state still produces valid outputs
