# Reproducibility and Audit

## Engine pinning
- RIR engine sources are vendored in `engine/sound_field_sim/`.
- File-level SHA256 hashes are stored in `reproducibility/engine_manifest.json`.
- Rebuild the manifest with:
  - `python tools/build_engine_manifest.py`

## RNG policy
- Project code uses `np.random.default_rng(seed)` and `numpy.random.Generator`.
- `random.*` and unseeded `default_rng()` are blocked in project code.
- Validate policy with:
  - `python tools/check_rng_policy.py`

## CI baseline
- Lint: `ruff`
- Format: `black --check`
- Tests: `pytest`
- Workflow file: `.github/workflows/ci.yml`
