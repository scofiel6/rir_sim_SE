import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

RoomRange = Dict[str, Tuple[float, float]]
DEFAULT_SOUND_SPEED_M_S = 343.0
DEFAULT_LATE_TAIL_HIGHPASS_HZ = 40.0
DEFAULT_MODE_FMIN_HZ = 40.0
DEFAULT_MODE_FMAX_HZ = 800.0
DEFAULT_MODE_N_MIN = 3
DEFAULT_MODE_N_MAX = 8
DEFAULT_MODE_REL_DB_MIN = -38.0
DEFAULT_MODE_REL_DB_MAX = -30.0

DEFAULT_MATERIAL_LIBRARY = {
    "painted_wall": {
        "absorption": (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18),
        "scattering": (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22),
    },
    "gypsum_board": {
        "absorption": (0.10, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05),
        "scattering": (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    },
    "concrete": {
        "absorption": (0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02),
        "scattering": (0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16),
    },
    "glass": {
        "absorption": (0.03, 0.03, 0.03, 0.04, 0.05, 0.06, 0.06),
        "scattering": (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18),
    },
    "curtain_heavy": {
        "absorption": (0.15, 0.25, 0.40, 0.55, 0.65, 0.70, 0.72),
        "scattering": (0.12, 0.14, 0.16, 0.20, 0.24, 0.30, 0.36),
    },
    "carpet_floor": {
        "absorption": (0.08, 0.12, 0.20, 0.30, 0.40, 0.50, 0.55),
        "scattering": (0.10, 0.14, 0.18, 0.22, 0.26, 0.32, 0.36),
    },
    "wood_floor": {
        "absorption": (0.05, 0.06, 0.08, 0.10, 0.11, 0.12, 0.12),
        "scattering": (0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28),
    },
    "acoustic_tile_ceiling": {
        "absorption": (0.30, 0.45, 0.65, 0.75, 0.75, 0.70, 0.65),
        "scattering": (0.18, 0.22, 0.28, 0.34, 0.38, 0.40, 0.42),
    },
    "plaster_ceiling": {
        "absorption": (0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16),
        "scattering": (0.10, 0.12, 0.14, 0.18, 0.22, 0.26, 0.30),
    },
}

DEFAULT_MATERIAL_FACE_CATEGORY_GROUPS = {
    "wall": ("painted_wall", "gypsum_board", "concrete", "glass", "curtain_heavy"),
    "floor": ("carpet_floor", "wood_floor"),
    "ceiling": ("acoustic_tile_ceiling", "plaster_ceiling"),
}


def _to_float_tuple(x, fallback):
    if x is None:
        return tuple(float(v) for v in fallback)
    return tuple(float(v) for v in list(x))


def _normalize_room_range(x):
    if x is None:
        return None
    out = {}
    for k, v in dict(x).items():
        vv = list(v)
        if len(vv) != 2:
            continue
        out[str(k)] = (float(vv[0]), float(vv[1]))
    return out


def _normalize_scale_dict(x):
    if x is None:
        return {}
    return {str(k): float(v) for k, v in dict(x).items()}


def _normalize_material_library(x):
    if x is None:
        x = DEFAULT_MATERIAL_LIBRARY
    out = {}
    for name, spec in dict(x).items():
        spec_d = dict(spec)
        absorption = tuple(float(v) for v in list(spec_d.get("absorption", [])))
        scattering = tuple(float(v) for v in list(spec_d.get("scattering", [])))
        if len(absorption) == 0 or len(scattering) == 0:
            continue
        out[str(name)] = {
            "absorption": absorption,
            "scattering": scattering,
        }
    return out


def _normalize_material_face_category_groups(x):
    if x is None:
        x = DEFAULT_MATERIAL_FACE_CATEGORY_GROUPS
    out = {}
    for face_type, names in dict(x).items():
        vals = tuple(str(v) for v in list(names) if str(v).strip())
        if len(vals) == 0:
            continue
        out[str(face_type)] = vals
    return out


def _resolve_path_near(base_dir, p):
    if p is None:
        return None
    pp = Path(str(p))
    if pp.is_absolute():
        return str(pp)
    return str((Path(base_dir) / pp).resolve())


def _load_json_with_line_comments(path: Path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    return json.loads(text)


@dataclass
class RIRSimSEConfig:
    fs: int = 32000
    seed: int = 2026
    use_drr_c50: bool = True
    room_size_hint: Tuple[float, float, float] = (3.6, 3.8, 2.7)
    room_jitter_ratio: float = 0.04
    custom_room_range: Optional[RoomRange] = None
    generic_room_range: Optional[RoomRange] = None
    dry_wav: str = "/home/xukj/dataset_rir/sound_field_sim/test.wav"
    pulse_recording: Optional[str] = None
    acoustic_state_json: Optional[str] = None
    # Keep short tails for small-room SE.
    rir_seconds: float = 1.4

    # Inversion knobs (IR recordings -> acoustic priors).
    inversion_rt60_min: float = 0.12
    inversion_rt60_max: float = 0.70
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

    # Material profile for RIR generation:
    # these are physical shaping coefficients applied to absorption/scattering,
    # instead of post-hoc low/high band gain tweaks.
    material_center_freqs_hz: Tuple[float, ...] = (
        125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0
    )
    # Relative absorption shape (mean-normalized internally).
    # >1 means stronger absorption at that band.
    material_absorption_curve: Tuple[float, ...] = (
        1.30, 1.22, 1.12, 1.00, 0.94, 0.92, 0.95, 1.10, 1.20
    )
    # Frequency-dependent scattering profile [0..1].
    material_scattering_curve: Tuple[float, ...] = (
        0.22, 0.24, 0.26, 0.30, 0.34, 0.40, 0.48, 0.54, 0.58
    )
    material_face_absorption_scale: Dict[str, float] = field(default_factory=lambda: {
        "west": 1.00, "east": 1.00, "south": 1.00, "north": 1.00, "floor": 1.00, "ceiling": 1.00
    })
    material_face_scattering_scale: Dict[str, float] = field(default_factory=lambda: {
        "west": 1.00, "east": 1.00, "south": 1.00, "north": 1.00, "floor": 1.00, "ceiling": 1.00
    })
    material_library: Dict[str, Dict[str, Tuple[float, ...]]] = field(default_factory=lambda: _normalize_material_library(None))
    material_face_category_groups: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: _normalize_material_face_category_groups(None))

    # Physical propagation prior used by delay, modal frequency, and ISM order heuristics.
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED_M_S
    # High-pass used on the synthetic late tail before modal augmentation.
    late_tail_highpass_hz: float = DEFAULT_LATE_TAIL_HIGHPASS_HZ
    # Low-frequency modal tail priors used by the engine.
    # These are the single source of truth for modal augmentation defaults.
    mode_fmin_hz: float = DEFAULT_MODE_FMIN_HZ
    mode_fmax_hz: float = DEFAULT_MODE_FMAX_HZ
    mode_n_min: int = DEFAULT_MODE_N_MIN
    mode_n_max: int = DEFAULT_MODE_N_MAX
    mode_rel_db_min: float = DEFAULT_MODE_REL_DB_MIN
    mode_rel_db_max: float = DEFAULT_MODE_REL_DB_MAX

    # Optional device EQ on generated RIR/ref outputs.
    # Default is flat (all 0 dB), so it does not alter signals.
    device_eq_enable: bool = True
    device_eq_centers_hz: Tuple[float, ...] = (
        63.0, 125.0, 250.0, 500.0, 1000.0,
        2000.0, 4000.0, 8000.0, 12000.0, 16000.0,
    )
    device_eq_gains_db: Tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )

    # Array settings.
    mic_array_type: str = "linear"
    mic_num: int = 4
    mic_spacing: float = 0.04
    mic_radius: float = 0.04
    mic_position_jitter_m: float = 0.001

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        d = dict(data)
        if "room_size_hint" in d:
            d["room_size_hint"] = _to_float_tuple(d["room_size_hint"], (3.6, 3.8, 2.7))
        if "custom_room_range" in d:
            d["custom_room_range"] = _normalize_room_range(d["custom_room_range"])
        if "generic_room_range" in d:
            d["generic_room_range"] = _normalize_room_range(d["generic_room_range"])

        for k in (
            "material_center_freqs_hz",
            "material_absorption_curve",
            "material_scattering_curve",
            "device_eq_centers_hz",
            "device_eq_gains_db",
        ):
            if k in d:
                d[k] = tuple(float(v) for v in list(d[k]))

        for k in ("sound_speed_m_s", "late_tail_highpass_hz", "mode_fmin_hz", "mode_fmax_hz", "mode_rel_db_min", "mode_rel_db_max"):
            if k in d:
                d[k] = float(d[k])
        for k in ("mode_n_min", "mode_n_max"):
            if k in d:
                d[k] = int(float(d[k]))

        if "material_face_absorption_scale" in d:
            d["material_face_absorption_scale"] = _normalize_scale_dict(d["material_face_absorption_scale"])
        if "material_face_scattering_scale" in d:
            d["material_face_scattering_scale"] = _normalize_scale_dict(d["material_face_scattering_scale"])
        if "material_library" in d:
            d["material_library"] = _normalize_material_library(d["material_library"])
        if "material_face_category_groups" in d:
            d["material_face_category_groups"] = _normalize_material_face_category_groups(d["material_face_category_groups"])
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    def to_dict(self):
        return asdict(self)


def load_rir_sim_se_config(config_path):
    p = Path(config_path).resolve()
    payload = _load_json_with_line_comments(p)
    cfg = RIRSimSEConfig.from_dict(payload)

    base_dir = p.parent
    # Keep runtime clean: all path-like fields are normalized here.
    cfg.dry_wav = _resolve_path_near(base_dir, cfg.dry_wav)
    cfg.pulse_recording = _resolve_path_near(base_dir, cfg.pulse_recording)
    if cfg.acoustic_state_json:
        cfg.acoustic_state_json = _resolve_path_near(base_dir, cfg.acoustic_state_json)
    else:
        # Default state file is stored beside cfg json.
        cfg.acoustic_state_json = str((base_dir / "acoustic_state.json").resolve())
    return cfg
