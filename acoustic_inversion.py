from config import RIRSimSEConfig
from imrir_adapter import base, call_sample_room_params, call_simulate_rir

import numpy as np

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

    # Linear array in x-axis using explicit mic positions.
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


def _clip_range(rng, lo, hi, min_width):
    a, b = float(rng[0]), float(rng[1])
    if b < a:
        a, b = b, a
    lo = float(lo)
    hi = float(hi)
    mw = float(max(0.0, min_width))

    a = float(np.clip(a, lo, hi))
    b = float(np.clip(b, lo, hi))
    if b < a:
        a, b = b, a

    if (b - a) < mw:
        c = 0.5 * (a + b)
        h = 0.5 * mw
        a = max(lo, c - h)
        b = min(hi, c + h)
        if (b - a) < mw:
            if a <= lo + 1e-9:
                b = min(hi, a + mw)
            elif b >= hi - 1e-9:
                a = max(lo, b - mw)
    return (float(a), float(b))


def _profile_source_dist_range(cfg: RIRSimSEConfig, profile_name: str):
    lx, ly, lz = [float(v) for v in cfg.room_size_hint]
    diag = float(np.sqrt(lx * lx + ly * ly + lz * lz))
    min_d = 0.6
    if profile_name == "smallroom_conservative":
        max_d = min(1.8, 0.35 * diag)
    else:
        max_d = min(2.8, 0.42 * diag)
    max_d = max(min_d + 0.2, float(max_d))
    return (float(min_d), float(max_d))


def _apply_generation_profile(gen, fit, cfg: RIRSimSEConfig):
    """
    Apply profile-level clamps so generated RIRs stay close to measured-room realism.
    This avoids over-reverberant tails when inversion has noisy estimates.
    """
    profile = str(getattr(cfg, "generation_profile", "fit_aligned")).strip().lower()
    if profile not in ("fit_aligned", "smallroom_conservative", "legacy"):
        raise ValueError(f"Unsupported generation_profile: {cfg.generation_profile!r}")

    if profile == "legacy":
        return gen

    if profile == "smallroom_conservative":
        rt_clip = (0.14, 0.65)
        drr_clip = (2.0, 12.0)
        c50_clip = (4.0, 18.0)
        rt_min_w = 0.05
        drr_min_w = 1.5
        c50_min_w = 1.5
        band_jitter = 1.0 / 12.0
    else:
        # fit_aligned
        rt_clip = (0.14, 0.90)
        drr_clip = (-1.0, 12.0)
        c50_clip = (1.0, 18.0)
        rt_min_w = 0.06
        drr_min_w = 1.5
        c50_min_w = 1.5
        band_jitter = 1.0 / 10.0

    rt_rng = _clip_range(gen.custom_rt60_range, rt_clip[0], rt_clip[1], min_width=rt_min_w)
    drr_rng = _clip_range(gen.drr_range_db, drr_clip[0], drr_clip[1], min_width=drr_min_w)
    c50_rng = _clip_range(gen.c50_range_db, c50_clip[0], c50_clip[1], min_width=c50_min_w)
    src_rng = _profile_source_dist_range(cfg, profile_name=profile)

    gen.custom_rt60_range = rt_rng
    gen.drr_range_db = drr_rng
    gen.c50_range_db = c50_rng
    gen.source_dist_range = src_rng
    gen.band_rt60_jitter_oct = float(max(0.0, band_jitter))

    # Keep this trace in fit metadata for reproducibility and A/B auditing.
    if isinstance(fit, dict):
        fit["generation_profile"] = profile
        fit["applied_ranges"] = {
            "custom_rt60_range": [float(rt_rng[0]), float(rt_rng[1])],
            "drr_range_db": [float(drr_rng[0]), float(drr_rng[1])],
            "c50_range_db": [float(c50_rng[0]), float(c50_rng[1])],
            "source_dist_range_m": [float(src_rng[0]), float(src_rng[1])],
            "band_rt60_jitter_oct": float(gen.band_rt60_jitter_oct),
        }
    return gen


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
        mic_info=_build_mic_info(cfg),
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


def apply_fit_to_generator(gen, fit):
    """
    Write cached fit parameters back to generator so we can skip expensive inversion.
    """
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

    nr = fit.get("noise_rms_median")
    if nr is not None:
        gen.custom_noise_rms = float(nr)

    nt = fit.get("noise_tilt_db_per_oct_median")
    if nt is not None:
        gen.custom_noise_tilt_db_oct = float(nt)

    gen.fitted = fit
    return gen


def create_generator_from_fit(cfg: RIRSimSEConfig, fit):
    gen = create_generator(cfg)
    gen = apply_fit_to_generator(gen, fit)
    return _apply_generation_profile(gen, fit, cfg)


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
    gen = _apply_generation_profile(gen, fit, cfg)
    return gen, fit
