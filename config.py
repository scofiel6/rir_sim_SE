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

    # Baseline selection:
    # - "invert": estimate acoustic params from recorded IR folder.
    # - "json": read acoustic params from an existing json file.
    acoustic_param_source: str = "invert"
    pulse_recording: Optional[str] = None
    acoustic_state_json: Optional[str] = None
    save_state_json_after_invert: bool = True
    # Keep short tails for small-room SE.
    rir_seconds: float = 1.4

    # Inversion knobs (IR recordings -> acoustic priors).
    inversion_rt60_min: float = 0.12
    inversion_rt60_max: float = 0.70
    # IR recordings are assumed, so default to from_recording.
    inversion_drr_c50_mode: str = "from_recording"
    inversion_drr_c50_jitter_db: float = 0.5
    inversion_drr_c50_from_recording_jitter_db: float = 0.2

    # ref1 parameters.
    ref_early_ms: float = 20.0
    ref_late_tail_db: float = -26.0

    # ref2 parameters (full-band early reference).
    ref2_early_ms: Optional[float] = None
    ref2_early_taps: int = 8
    ref2_min_tap_ms: float = 0.4

    # Increase low-frequency absorption to reduce boomy low-band reverberation.
    low_freq_absorption_boost: float = 1.35
    low_freq_absorption_cut_hz: float = 500.0

    # Array settings.
    mic_array_type: str = "linear"
    mic_num: int = 4
    mic_spacing: float = 0.04
    mic_radius: float = 0.04
    mic_position_jitter_m: float = 0.001
