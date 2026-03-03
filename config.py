from dataclasses import dataclass
from typing import Dict, Optional, Tuple

RoomRange = Dict[str, Tuple[float, float]]


@dataclass
class RIRSimSEConfig:
    fs: int = 32000
    seed: int = 2026
    use_drr_c50: bool = True
    room_size_hint: Tuple[float, float, float] = (3.6, 3.8, 2.7)
    room_jitter_ratio: float = 0.04
    custom_room_range: Optional[RoomRange] = None
    generic_room_range: Optional[RoomRange] = None
    out_dir: str = "./_out_rir_sim_se"
    # If False, skip writing wav/json outputs to disk (recommended for online training).
    save_outputs: bool = True
    # If True, include in-memory arrays (`dry`, `wet`, `ref`, `rir`, `rir_ref`) in return dict.
    return_audio_arrays: bool = False
    # RIR length cap (seconds). Smaller values speed up convolution.
    rir_seconds: float = 2.0
    # Keep direct arrival + early reflections within this window for ref target.
    ref_early_ms: float = 20.0
    # Keep a little late tail in ref (dB relative scale at early/late boundary).
    # Example: -26 dB means late part is strongly suppressed but not fully removed.
    ref_late_tail_db: float = -26.0
    # Reference target mode:
    # - "early_rir": ref from early-dominant RIR (current classic behavior).
    # - "dry_distance": dry-like ref with broadband distance attenuation + early sparse taps.
    ref_target_mode: str = "dry_distance"
    # Broadband distance attenuation for "dry_distance" ref mode:
    #   gain_dist = clip((ref_distance_ref_m / src_dist) ** ref_distance_power, min, max)
    ref_distance_ref_m: float = 1.0
    ref_distance_power: float = 1.0
    ref_distance_gain_min: float = 0.2
    ref_distance_gain_max: float = 1.2
    # Number of early sparse reflection taps kept in "dry_distance" ref mode.
    ref_distance_early_taps: int = 8
    # Minimum spacing between selected sparse early taps in milliseconds.
    ref_distance_min_tap_ms: float = 0.4
    # If pulse_recording is a directory, use at most this many files for fitting.
    # This is the highest-impact speed knob for large recording folders.
    max_fit_files: Optional[int] = 12
    # Fit cache control for speeding up repeated generation.
    enable_fit_cache: bool = True
    fit_cache_path: Optional[str] = None
    fit_cache_force_refit: bool = False
    # Array settings for multi-channel consistent simulation.
    mic_array_type: str = "linear"  # linear or circular
    mic_num: int = 4
    mic_spacing: float = 0.04
    mic_radius: float = 0.04
    mic_position_jitter_m: float = 0.001
    # Device mismatch model.
    # Set False for pure "RIR + convolution + ref" generation with no device perturbation.
    enable_channel_mismatch: bool = False
    # Extra additive white-noise switch (independent from RIR reverberation synthesis).
    enable_channel_white_noise: bool = False
    channel_gain_db_std: float = 0.8
    channel_delay_us_std: float = 20.0
    channel_noise_dbfs_range: Tuple[float, float] = (-65.0, -50.0)
