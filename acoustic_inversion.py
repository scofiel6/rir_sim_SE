from pathlib import Path
import sys

from config import RIRSimSEConfig
from imrir_adapter import call_sample_room_params, call_simulate_rir

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sound_field_sim.rirGen_baseSE import BaseSERIRGenerator, _room_range_from_hint


def _build_single_mic_info():
    return {
        "device_id": "single_mic_api",
        "device_height": 1.2,
        "array_type": "linear",
        "mic_pos": [0.0],
    }


def create_generator(cfg: RIRSimSEConfig):
    # Ensure im_rir_v2 adapter is loaded and callable.
    if (not callable(call_sample_room_params)) or (not callable(call_simulate_rir)):
        raise RuntimeError("im_rir_v2 adapter is not available.")

    custom_room_range = cfg.custom_room_range
    if custom_room_range is None:
        custom_room_range = _room_range_from_hint(cfg.room_size_hint, cfg.room_jitter_ratio)

    generic_room_range = cfg.generic_room_range or {
        "lx": (2.8, 6.5),
        "ly": (2.8, 6.5),
        "lz": (2.4, 3.6),
    }

    gen = BaseSERIRGenerator(
        fs=cfg.fs,
        mic_info=_build_single_mic_info(),
        custom_room_range=custom_room_range,
        generic_room_range=generic_room_range,
        custom_rt60_range=(0.2, 1.2),
        generic_rt60_range=(0.12, 1.3),
        generic_mix_prob=0.0,
        center_jitter_oct=1.0 / 6.0,
        band_rt60_jitter_oct=1.0 / 8.0,
        band_smoothing_passes=2,
        source_dist_range=(0.7, 4.2),
        drr_range_db=(-5.0, 12.0),
        c50_range_db=(-2.0, 16.0),
        snr_range_db=(0.0, 25.0),
        enable_physical_calibration=True,
        enable_final_output_norm=False,
    )
    return gen


def invert_acoustic_params(cfg: RIRSimSEConfig, pulse_recording):
    gen = create_generator(cfg)
    fit = gen.fit_from_recordings(
        recordings=pulse_recording,
        room_size_hint=cfg.room_size_hint,
        room_jitter_ratio=cfg.room_jitter_ratio,
        rt60_min_max=(0.12, 1.4),
        drr_prior_range_db=(-3.0, 8.0),
        c50_prior_range_db=(0.0, 14.0),
        drr_c50_jitter_db=0.6,
        drr_c50_mode=("auto" if bool(cfg.use_drr_c50) else "fixed"),
        drr_c50_from_recording_jitter_db=0.2,
        fit_seed=cfg.seed,
        update_generator=True,
    )
    return gen, fit
