import numpy as np

from config import RIRSimSEConfig
from engine.sound_field_sim.base_engine import BaseEngine, _room_range_from_hint


def _build_mic_info(cfg: RIRSimSEConfig):
    arr = str(cfg.mic_array_type).lower().strip()
    n = int(cfg.mic_num)
    if arr == "circular":
        return {
            "device_id": f"circular_{n}ch",
            "device_height": 1.2,
            "array_type": "circular",
            "mic_num": n,
            "mic_radius": float(cfg.mic_radius),
        }

    if cfg.mic_positions_m is not None and len(cfg.mic_positions_m) > 0:
        # Explicit linear geometry from cfg takes priority over uniform spacing.
        pos = np.asarray(cfg.mic_positions_m, dtype=np.float64).reshape(-1)
        n = int(pos.shape[0])
    else:
        spacing = float(cfg.mic_spacing)
        pos = np.arange(n, dtype=np.float64) * spacing
        pos = pos - float(np.mean(pos))
    jitter = float(cfg.mic_position_jitter_m)
    if jitter > 0.0:
        rng = np.random.default_rng(int(cfg.seed) + 11)
        pos = pos + rng.normal(0.0, jitter, size=pos.shape[0])
    return {
        "device_id": f"linear_{n}ch",
        "device_height": 1.2,
        "array_type": "linear",
        "mic_pos": pos.tolist(),
    }


def _merge_fit_dicts(stage1_fit, stage2_fit):
    fit = dict(stage1_fit)
    fit.update({k: v for k, v in stage2_fit.items() if k != "warnings"})
    warnings = list(stage1_fit.get("warnings", [])) + list(stage2_fit.get("warnings", []))
    if warnings:
        fit["warnings"] = list(dict.fromkeys(warnings))
    return fit


def create_generator(cfg: RIRSimSEConfig):
    # This is the one place where config priors become a concrete engine object.
    # Everything below is still just engine state; the actual sample draw happens
    # later inside BaseEngine.generate(...).
    custom_room_range = cfg.custom_room_range
    if custom_room_range is None:
        custom_room_range = _room_range_from_hint(cfg.room_size_hint, cfg.room_jitter_ratio)

    generic_room_range = cfg.generic_room_range or {
        "lx": (2.8, 6.5),
        "ly": (2.8, 6.5),
        "lz": (2.4, 3.6),
    }

    gen = BaseEngine(
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
    gen.material_library = {
        name: {
            "absorption": tuple(float(v) for v in spec["absorption"]),
            "scattering": tuple(float(v) for v in spec["scattering"]),
        }
        for name, spec in cfg.material_library.items()
    }
    gen.material_face_category_groups = {
        face_type: tuple(str(v) for v in names)
        for face_type, names in cfg.material_face_category_groups.items()
    }
    gen.material_face_absorption_scale = dict(cfg.material_face_absorption_scale)
    gen.material_face_scattering_scale = dict(cfg.material_face_scattering_scale)
    gen.sound_speed_m_s = float(cfg.sound_speed_m_s)
    gen.late_tail_highpass_hz = float(cfg.late_tail_highpass_hz)
    gen.late_reverb_bandwidth_oct = float(cfg.late_reverb_bandwidth_oct)
    gen.late_reverb_break_fractions = tuple(float(v) for v in cfg.late_reverb_break_fractions)
    gen.late_reverb_density_scale = float(cfg.late_reverb_density_scale)
    gen.late_reverb_slope_scales = tuple(float(v) for v in cfg.late_reverb_slope_scales)
    gen.source_directivity_strength = float(cfg.source_directivity_strength)
    gen.source_head_shadow_strength = float(cfg.source_head_shadow_strength)
    gen.source_torso_scattering_strength = float(cfg.source_torso_scattering_strength)
    gen.source_head_radius_m = float(cfg.source_head_radius_m)
    gen.source_torso_radius_m = float(cfg.source_torso_radius_m)
    gen.source_directivity_bandwidth_oct = float(cfg.source_directivity_bandwidth_oct)
    gen.mode_fmin_hz = float(cfg.mode_fmin_hz)
    gen.mode_fmax_hz = float(cfg.mode_fmax_hz)
    gen.mode_n_min = int(cfg.mode_n_min)
    gen.mode_n_max = int(cfg.mode_n_max)
    gen.mode_rel_db_min = float(cfg.mode_rel_db_min)
    gen.mode_rel_db_max = float(cfg.mode_rel_db_max)
    return gen


def apply_fit_to_generator(gen, fit):
    room = fit.get("estimated_room_range")
    if room is not None:
        gen.custom_room_range = room

    rt20 = fit.get("rt60_p20")
    rt80 = fit.get("rt60_p80")
    if rt20 is not None and rt80 is not None:
        gen.custom_rt60_range = (float(rt20), float(rt80))

    rt50 = fit.get("rt60_median")
    if rt50 is not None:
        gen.custom_rt60_center = float(rt50)

    band = fit.get("rt60_band_median")
    if band is not None:
        gen.custom_band_rt60_prior = np.asarray(band, dtype=np.float64)

    drr = fit.get("drr_db_p20_p80")
    if drr is not None:
        gen.drr_range_db = (float(drr[0]), float(drr[1]))

    c50 = fit.get("c50_db_p20_p80")
    if c50 is not None:
        gen.c50_range_db = (float(c50[0]), float(c50[1]))

    gen.fitted = fit
    return gen


def create_generator_from_fit(cfg: RIRSimSEConfig, fit):
    # Rebuild a fresh engine from config, then overwrite the fields that came from
    # inversion. This keeps the saved state small and avoids storing a pickled engine.
    return apply_fit_to_generator(create_generator(cfg), fit)


def _run_stage1_statistical_inversion(gen, cfg: RIRSimSEConfig, pulse_recording):
    rt60_lo = float(min(cfg.inversion_rt60_min, cfg.inversion_rt60_max))
    rt60_hi = float(max(cfg.inversion_rt60_min, cfg.inversion_rt60_max))
    mode = str(cfg.inversion_drr_c50_mode).lower().strip()
    # Stage-1 solves the coarse room statistics used later by the generator:
    # RT60 range, band RT60 profile, DRR/C50 range, and noise summary.
    fit = gen.fit_from_recordings(
        recordings=pulse_recording,
        room_size_hint=cfg.room_size_hint,
        room_jitter_ratio=cfg.room_jitter_ratio,
        rt60_min_max=(rt60_lo, rt60_hi),
        drr_prior_range_db=(2.0, 10.0),
        c50_prior_range_db=(6.0, 16.0),
        drr_c50_jitter_db=float(cfg.inversion_drr_c50_jitter_db),
        drr_c50_mode=mode,
        drr_c50_from_recording_jitter_db=float(cfg.inversion_drr_c50_from_recording_jitter_db),
        fit_seed=cfg.seed,
        update_generator=True,
    )
    fit["drr_c50_mode_requested"] = mode
    fit["drr_c50_mode_effective"] = mode
    fit["inversion_stage1"] = "statistical_priors"
    return fit


def _run_stage2_echo_structure_inversion(gen, pulse_recording):
    # Stage-2 keeps the same recordings but extracts early-echo structure and EDT.
    # These values are stored in the state for later analysis and future calibration.
    fit = gen.analyze_reflection_structure_from_recordings(
        recordings=pulse_recording,
        max_early_ms=80.0,
        n_echoes=6,
    )
    fit["inversion_stage2"] = "echo_structure"
    return fit


def invert_acoustic_params(cfg: RIRSimSEConfig, pulse_recording):
    gen = create_generator(cfg)
    fit_stage1 = _run_stage1_statistical_inversion(gen, cfg, pulse_recording)
    fit_stage2 = _run_stage2_echo_structure_inversion(gen, pulse_recording)
    fit = _merge_fit_dicts(fit_stage1, fit_stage2)
    gen.fitted = fit
    return gen, fit
