import numpy as np

from config import RIRSimSEConfig
from imrir_adapter import base, call_sample_room_params, call_simulate_rir


BaseSERIRGenerator = base.BaseSERIRGenerator
_room_range_from_hint = base._room_range_from_hint


def _build_mic_info(cfg: RIRSimSEConfig):
    arr = str(cfg.mic_array_type).lower().strip()
    n = int(max(1, cfg.mic_num))
    if arr == "circular":
        return {
            "device_id": f"circular_{n}ch",
            "device_height": 1.2,
            "array_type": "circular",
            "mic_num": n,
            "mic_radius": float(max(0.01, cfg.mic_radius)),
        }

    spacing = float(max(0.005, cfg.mic_spacing))
    pos = np.arange(n, dtype=np.float64) * spacing
    pos = pos - float(np.mean(pos))
    jitter = float(max(0.0, cfg.mic_position_jitter_m))
    if jitter > 0.0:
        rng = np.random.default_rng(int(cfg.seed) + 11)
        pos = pos + rng.normal(0.0, jitter, size=pos.shape[0])
    pos = np.sort(pos)
    return {
        "device_id": f"linear_{n}ch",
        "device_height": 1.2,
        "array_type": "linear",
        "mic_pos": pos.tolist(),
    }


def _clip_range(rng, lo, hi, min_width=0.05):
    a, b = float(rng[0]), float(rng[1])
    if b < a:
        a, b = b, a
    a = float(np.clip(a, lo, hi))
    b = float(np.clip(b, lo, hi))
    if b - a < float(min_width):
        c = 0.5 * (a + b)
        h = 0.5 * float(min_width)
        a = max(float(lo), c - h)
        b = min(float(hi), c + h)
    return (float(a), float(b))


def create_generator(cfg: RIRSimSEConfig):
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

    # Keep baseline conservative for small-room SE.
    gen = BaseSERIRGenerator(
        fs=cfg.fs,
        mic_info=_build_mic_info(cfg),
        custom_room_range=custom_room_range,
        generic_room_range=generic_room_range,
        custom_rt60_range=(0.18, 0.65),
        generic_rt60_range=(0.15, 0.80),
        generic_mix_prob=0.0,
        center_jitter_oct=1.0 / 8.0,
        band_rt60_jitter_oct=1.0 / 10.0,
        band_smoothing_passes=2,
        source_dist_range=(0.6, 1.8),
        drr_range_db=(2.0, 10.0),
        c50_range_db=(6.0, 16.0),
        snr_range_db=(0.0, 25.0),
        enable_physical_calibration=True,
        enable_final_output_norm=False,
    )
    # Physical material profile (frequency-dependent) used inside RIR generation.
    gen.material_center_freqs = np.asarray(cfg.material_center_freqs_hz, dtype=np.float64)
    gen.material_absorption_curve = np.asarray(cfg.material_absorption_curve, dtype=np.float64)
    gen.material_scattering_curve = np.asarray(cfg.material_scattering_curve, dtype=np.float64)
    gen.material_face_absorption_scale = dict(cfg.material_face_absorption_scale)
    gen.material_face_scattering_scale = dict(cfg.material_face_scattering_scale)
    gen.mode_fmin_hz = float(cfg.mode_fmin_hz)
    gen.mode_fmax_hz = float(cfg.mode_fmax_hz)
    gen.mode_n_min = int(cfg.mode_n_min)
    gen.mode_n_max = int(cfg.mode_n_max)
    gen.mode_rel_db_min = float(cfg.mode_rel_db_min)
    gen.mode_rel_db_max = float(cfg.mode_rel_db_max)
    return gen


def apply_fit_to_generator(gen, fit):
    if not isinstance(fit, dict):
        return gen

    room = fit.get("fitted_custom_room_range")
    if isinstance(room, dict):
        gen.custom_room_range = room

    rt20 = fit.get("rt60_p20")
    rt80 = fit.get("rt60_p80")
    if rt20 is not None and rt80 is not None:
        gen.custom_rt60_range = (float(rt20), float(rt80))

    rt50 = fit.get("rt60_median")
    if rt50 is not None:
        gen.custom_rt60_center = float(rt50)

    band = fit.get("rt60_band_median")
    if isinstance(band, list) and len(band) > 0:
        gen.custom_band_rt60_prior = np.asarray(band, dtype=np.float64)

    drr = fit.get("drr_db_p20_p80")
    if isinstance(drr, list) and len(drr) == 2:
        gen.drr_range_db = (float(drr[0]), float(drr[1]))

    c50 = fit.get("c50_db_p20_p80")
    if isinstance(c50, list) and len(c50) == 2:
        gen.c50_range_db = (float(c50[0]), float(c50[1]))

    gen.fitted = fit
    return gen


def _apply_conservative_postfit_clamp(gen, cfg: RIRSimSEConfig):
    rt_lo = float(min(cfg.inversion_rt60_min, cfg.inversion_rt60_max))
    rt_hi = float(max(cfg.inversion_rt60_min, cfg.inversion_rt60_max))
    gen.custom_rt60_range = _clip_range(gen.custom_rt60_range, rt_lo, rt_hi, min_width=0.04)
    gen.drr_range_db = _clip_range(gen.drr_range_db, 2.0, 12.0, min_width=1.0)
    gen.c50_range_db = _clip_range(gen.c50_range_db, 4.0, 18.0, min_width=1.0)
    gen.source_dist_range = (0.6, 1.8)
    gen.band_rt60_jitter_oct = float(min(float(gen.band_rt60_jitter_oct), 1.0 / 10.0))
    return gen


def create_generator_from_fit(cfg: RIRSimSEConfig, fit):
    gen = create_generator(cfg)
    gen = apply_fit_to_generator(gen, fit)
    return _apply_conservative_postfit_clamp(gen, cfg)


def invert_acoustic_params(cfg: RIRSimSEConfig, pulse_recording):
    gen = create_generator(cfg)
    rt60_lo = float(min(cfg.inversion_rt60_min, cfg.inversion_rt60_max))
    rt60_hi = float(max(cfg.inversion_rt60_min, cfg.inversion_rt60_max))
    mode = str(cfg.inversion_drr_c50_mode).lower().strip()
    if mode not in ("fixed", "auto", "from_recording"):
        mode = "from_recording"

    def _fit_with_mode(mode_use: str):
        return gen.fit_from_recordings(
            recordings=pulse_recording,
            room_size_hint=cfg.room_size_hint,
            room_jitter_ratio=cfg.room_jitter_ratio,
            rt60_min_max=(rt60_lo, rt60_hi),
            drr_prior_range_db=(2.0, 10.0),
            c50_prior_range_db=(6.0, 16.0),
            drr_c50_jitter_db=float(max(0.0, cfg.inversion_drr_c50_jitter_db)),
            drr_c50_mode=str(mode_use),
            drr_c50_from_recording_jitter_db=float(max(0.0, cfg.inversion_drr_c50_from_recording_jitter_db)),
            fit_seed=cfg.seed,
            update_generator=True,
        )

    try:
        fit = _fit_with_mode(mode)
        effective_mode = mode
    except ValueError as e:
        msg = str(e)
        # Some measured IR files may fail strict impulse-like check.
        # Fallback to auto mode so inversion continues instead of hard-failing.
        if mode == "from_recording" and "requires impulse-like recordings" in msg:
            fit = _fit_with_mode("auto")
            effective_mode = "auto"
            if isinstance(fit, dict):
                warnings = fit.get("warnings")
                if not isinstance(warnings, list):
                    warnings = []
                warnings.append(
                    "Requested drr_c50_mode='from_recording' but at least one file "
                    "failed impulse-like check; fallback to drr_c50_mode='auto'."
                )
                fit["warnings"] = warnings
        else:
            raise

    if isinstance(fit, dict):
        fit["drr_c50_mode_requested"] = str(mode)
        fit["drr_c50_mode_effective"] = str(effective_mode)

    gen = _apply_conservative_postfit_clamp(gen, cfg)
    return gen, fit
