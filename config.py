import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

RoomRange = Dict[str, Tuple[float, float]]


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


def _resolve_path_near(base_dir, p):
    if p is None:
        return None
    pp = Path(str(p))
    if pp.is_absolute():
        return str(pp)
    return str((Path(base_dir) / pp).resolve())


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

    # Low-frequency modal tail controls (fed into im_rir_v2 core).
    mode_fmin_hz: float = 40.0
    mode_fmax_hz: float = 800.0
    mode_n_min: int = 3
    mode_n_max: int = 8
    mode_rel_db_min: float = -38.0
    mode_rel_db_max: float = -30.0

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

        for k in ("mode_fmin_hz", "mode_fmax_hz", "mode_rel_db_min", "mode_rel_db_max"):
            if k in d:
                d[k] = float(d[k])
        for k in ("mode_n_min", "mode_n_max"):
            if k in d:
                d[k] = int(float(d[k]))

        if "material_face_absorption_scale" in d:
            d["material_face_absorption_scale"] = _normalize_scale_dict(d["material_face_absorption_scale"])
        if "material_face_scattering_scale" in d:
            d["material_face_scattering_scale"] = _normalize_scale_dict(d["material_face_scattering_scale"])
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    def to_dict(self):
        return asdict(self)


def load_rir_sim_se_config(config_path):
    p = Path(config_path).resolve()
    payload = json.loads(p.read_text(encoding="utf-8"))
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


def save_rir_sim_se_config(cfg: RIRSimSEConfig, config_path):
    p = Path(config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)
