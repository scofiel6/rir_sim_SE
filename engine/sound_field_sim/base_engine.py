import json
from pathlib import Path
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import butter, fftconvolve, sosfilt

from config import (
    DEFAULT_LATE_REVERB_BANDWIDTH_OCT,
    DEFAULT_LATE_REVERB_BREAK_FRACTIONS,
    DEFAULT_LATE_REVERB_DENSITY_SCALE,
    DEFAULT_LATE_REVERB_SLOPE_SCALES,
    DEFAULT_LATE_TAIL_HIGHPASS_HZ,
    DEFAULT_MATERIAL_FACE_CATEGORY_GROUPS,
    DEFAULT_MATERIAL_LIBRARY,
    DEFAULT_MODE_FMAX_HZ,
    DEFAULT_MODE_FMIN_HZ,
    DEFAULT_MODE_N_MAX,
    DEFAULT_MODE_N_MIN,
    DEFAULT_MODE_REL_DB_MAX,
    DEFAULT_MODE_REL_DB_MIN,
    DEFAULT_SOUND_SPEED_M_S,
    DEFAULT_SOURCE_DIRECTIVITY_BANDWIDTH_OCT,
    DEFAULT_SOURCE_DIRECTIVITY_STRENGTH,
    DEFAULT_SOURCE_HEAD_RADIUS_M,
    DEFAULT_SOURCE_HEAD_SHADOW_STRENGTH,
    DEFAULT_SOURCE_TORSO_RADIUS_M,
    DEFAULT_SOURCE_TORSO_SCATTERING_STRENGTH,
)
from utils import to_mono


def add_low_freq_modes(
    tail,
    fs,
    room_dim,
    rt60,
    fmin,
    fmax,
    n_modes_range,
    rel_db_range,
    sound_speed_m_s,
    rng=None,
    return_meta=False,
):
    lx, ly, lz = room_dim
    n_samples = len(tail)
    t = np.arange(n_samples) / fs

    cand = []
    for room_len in (lx, ly, lz):
        n_max = int(np.floor(2 * fmax * room_len / sound_speed_m_s))
        for n in range(1, max(2, n_max + 1)):
            freq = (sound_speed_m_s / 2.0) * (n / room_len)
            if fmin <= freq <= fmax:
                cand.append(freq)

    cand = np.array(sorted(set(cand)))
    if cand.size == 0:
        meta = {"mode_freqs_hz": [], "mode_taus_s": [], "mode_rel_db": None}
        return (tail, meta) if return_meta else tail

    rng = np.random.default_rng(0) if rng is None else rng
    k = min(int(rng.integers(n_modes_range[0], n_modes_range[1] + 1)), cand.size)
    fk = rng.choice(cand, size=k, replace=False)
    fk = fk * rng.uniform(0.98, 1.02, size=k)

    modes = np.zeros_like(tail, dtype=np.float64)
    mode_taus = []
    for freq in fk:
        phi = rng.uniform(0, 2 * np.pi)
        tau = rt60 * rng.uniform(0.4, 1.2)
        tau *= (120.0 / max(freq, 60.0)) ** 0.2
        mode_taus.append(float(tau))
        modes += np.exp(-t / max(tau, 1e-3)) * np.sin(2 * np.pi * freq * t + phi)

    rms_tail = np.sqrt(np.mean(tail**2) + 1e-12)
    modes /= np.sqrt(np.mean(modes**2) + 1e-12)
    rel_db = rng.uniform(rel_db_range[0], rel_db_range[1])
    modes *= rms_tail * (10.0 ** (rel_db / 20.0))

    out = tail + modes
    meta = {
        "mode_freqs_hz": [float(v) for v in np.asarray(fk, dtype=np.float64).tolist()],
        "mode_taus_s": mode_taus,
        "mode_rel_db": float(rel_db),
    }
    return (out, meta) if return_meta else out


def generate_velvet_noise(length, fs, density=2000, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    length = int(max(0, length))
    density = float(max(1e-6, density))
    velvet = np.zeros(length, dtype=np.float64)
    grid_size = max(1, int(fs / density))
    n_pulses = length // grid_size
    for i in range(n_pulses):
        pos = i * grid_size + rng.integers(0, grid_size)
        if pos < length:
            velvet[pos] = rng.choice([-1, 1])
    return velvet


def apply_highpass(sig, fs, cutoff=40):
    sos = butter(4, cutoff, "hp", fs=fs, output="sos")
    return sosfilt(sos, sig)


def _log_gaussian_band_weights(freqs, center_hz, bandwidth_oct=0.9):
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if f.size == 0:
        return np.zeros(0, dtype=np.float64)
    fc = float(max(1.0, center_hz))
    bw = float(max(0.15, bandwidth_oct))
    sigma = bw / 2.355
    out = np.exp(-0.5 * ((np.log2(np.clip(f, 1.0, None)) - np.log2(fc)) / sigma) ** 2)
    out[0] = 0.0
    return out


def _piecewise_multislope_envelope(t, rt_a, rt_b, rt_c, break_a_s, break_b_s):
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    if tt.size == 0:
        return np.zeros(0, dtype=np.float64)

    ra = float(max(0.03, rt_a))
    rb = float(max(0.03, rt_b))
    rc = float(max(0.03, rt_c))
    ta = float(max(0.0, break_a_s))
    tb = float(max(ta + 1e-4, break_b_s))

    tau_a = ra / 6.9
    tau_b = rb / 6.9
    tau_c = rc / 6.9

    out = np.empty_like(tt)
    m0 = tt < ta
    m1 = np.logical_and(tt >= ta, tt < tb)
    m2 = tt >= tb

    out[m0] = np.exp(-tt[m0] / tau_a)
    e_a = float(np.exp(-ta / tau_a))
    out[m1] = e_a * np.exp(-(tt[m1] - ta) / tau_b)
    e_b = float(e_a * np.exp(-(tb - ta) / tau_b))
    out[m2] = e_b * np.exp(-(tt[m2] - tb) / tau_c)
    return out


def _synthesize_multiband_late_reverb(fs, tail_len, room_dim, rt60_target, params, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    n = int(max(1, tail_len))
    t = np.arange(n, dtype=np.float64) / float(fs)
    lx, ly, lz = [float(v) for v in room_dim]
    volume = float(max(1e-6, lx * ly * lz))
    surface = float(max(1e-6, 2.0 * (lx * ly + lx * lz + ly * lz)))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    fc = np.asarray(params.get("center_freqs", [125, 250, 500, 1000, 2000, 4000, 8000]), dtype=np.float64).reshape(-1)
    if fc.size == 0:
        fc = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float64)

    band_rt60 = params.get("band_rt60")
    if band_rt60 is None:
        band_rt60 = np.full(fc.shape, float(rt60_target), dtype=np.float64)
    else:
        band_rt60 = np.asarray(band_rt60, dtype=np.float64).reshape(-1)
        if band_rt60.size != fc.size:
            rt_fill = float(np.nanmedian(band_rt60)) if band_rt60.size > 0 else float(rt60_target)
            band_rt60 = np.full(fc.shape, rt_fill, dtype=np.float64)

    band_alpha = params.get("band_alpha")
    if band_alpha is None:
        alpha_fn = params.get("alpha_continuous")
        if alpha_fn is None:
            band_alpha = np.clip(0.161 * volume / (surface * np.maximum(band_rt60, 1e-4)), 0.02, 0.98)
        else:
            band_alpha = np.clip(alpha_fn(np.log(np.clip(fc, 50.0, float(fs) / 2.0))), 0.02, 0.98)
    else:
        band_alpha = np.asarray(band_alpha, dtype=np.float64).reshape(-1)
        if band_alpha.size != fc.size:
            alpha_fill = float(np.nanmedian(band_alpha)) if band_alpha.size > 0 else 0.25
            band_alpha = np.full(fc.shape, alpha_fill, dtype=np.float64)

    band_scat = params.get("band_scattering_curve")
    if band_scat is None:
        band_scat = np.full(fc.shape, 0.35, dtype=np.float64)
    else:
        band_scat = np.asarray(band_scat, dtype=np.float64).reshape(-1)
        if band_scat.size != fc.size:
            scat_fill = float(np.nanmedian(band_scat)) if band_scat.size > 0 else 0.35
            band_scat = np.full(fc.shape, scat_fill, dtype=np.float64)
    band_scat = np.clip(band_scat, 0.05, 0.95)

    bandwidth_oct = float(params.get("late_reverb_bandwidth_oct", DEFAULT_LATE_REVERB_BANDWIDTH_OCT))
    break_fracs = params.get("late_reverb_break_fractions", DEFAULT_LATE_REVERB_BREAK_FRACTIONS)
    break_fracs = [float(v) for v in list(break_fracs)[:2]]
    if len(break_fracs) < 2:
        break_fracs = [float(DEFAULT_LATE_REVERB_BREAK_FRACTIONS[0]), float(DEFAULT_LATE_REVERB_BREAK_FRACTIONS[1])]
    b0, b1 = sorted((break_fracs[0], break_fracs[1]))
    density_scale = float(params.get("late_reverb_density_scale", DEFAULT_LATE_REVERB_DENSITY_SCALE))
    slope_scales = params.get("late_reverb_slope_scales", DEFAULT_LATE_REVERB_SLOPE_SCALES)
    slope_scales = [float(v) for v in list(slope_scales)[:3]]
    if len(slope_scales) < 3:
        slope_scales = [float(v) for v in DEFAULT_LATE_REVERB_SLOPE_SCALES]

    reflection_rate = float(params.get("sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S) * surface / max(4.0 * volume, 1e-6))
    geom_density = float(np.clip(reflection_rate / 140.0, 0.7, 1.8))
    tail = np.zeros(n, dtype=np.float64)
    band_traces = []

    for i, center_hz in enumerate(fc):
        rt_band = float(np.clip(band_rt60[i], 0.06, 3.0))
        alpha_i = float(np.clip(band_alpha[i], 0.02, 0.98))
        scat_i = float(np.clip(band_scat[i], 0.05, 0.95))
        freq_oct = float(max(0.0, np.log2(max(center_hz, 125.0) / 125.0)))
        hf_air = float(max(0.0, np.log2(max(center_hz, 1000.0) / 1000.0)))

        rt_a = rt_band * slope_scales[0] * (0.92 + 0.32 * scat_i)
        rt_b = rt_band * slope_scales[1] * (0.95 + 0.16 * (1.0 - alpha_i) + 0.10 * scat_i)
        rt_c = rt_band * slope_scales[2] * (0.86 + 0.22 * (1.0 - alpha_i) + 0.14 * scat_i) / (1.0 + 0.12 * hf_air)
        rt_a = float(np.clip(rt_a, 0.04, 3.0))
        rt_b = float(np.clip(rt_b, 0.05, 3.2))
        rt_c = float(np.clip(rt_c, 0.04, 2.6))

        frac_a = float(np.clip(b0 * (1.08 - 0.18 * scat_i), 0.08, 0.36))
        frac_b = float(np.clip(b1 * (0.95 + 0.08 * scat_i - 0.05 * alpha_i), frac_a + 0.10, 0.92))
        break_a_s = frac_a * (n / float(fs))
        break_b_s = frac_b * (n / float(fs))

        density = density_scale * geom_density * (520.0 + 2600.0 * scat_i + 180.0 * freq_oct)
        noise = generate_velvet_noise(n, fs, density=density, rng=rng)
        weight = _log_gaussian_band_weights(freqs, center_hz, bandwidth_oct=bandwidth_oct)
        band = np.fft.irfft(np.fft.rfft(noise) * weight, n=n)
        band /= np.sqrt(np.mean(band**2) + 1e-12)

        diffuse_gain = (0.35 + 0.75 * scat_i) * np.sqrt(max(1.0 - alpha_i, 1e-4)) / (1.0 + 0.10 * hf_air)
        envelope = _piecewise_multislope_envelope(t, rt_a, rt_b, rt_c, break_a_s, break_b_s)
        band = band * diffuse_gain * envelope
        tail += band

        band_traces.append({
            "center_hz": float(center_hz),
            "rt60_band_s": float(rt_band),
            "alpha": float(alpha_i),
            "scattering": float(scat_i),
            "density_hz": float(density),
            "break_a_ms": float(1000.0 * break_a_s),
            "break_b_ms": float(1000.0 * break_b_s),
            "rt_a_s": float(rt_a),
            "rt_b_s": float(rt_b),
            "rt_c_s": float(rt_c),
            "diffuse_gain": float(diffuse_gain),
        })

    tail /= np.sqrt(np.mean(tail**2) + 1e-12)
    trace = {
        "variant": "multiband_multi_slope_diffuse",
        "reflection_rate_hz": float(reflection_rate),
        "geom_density_scale": float(geom_density),
        "bandwidth_oct": float(bandwidth_oct),
        "break_fractions": [float(b0), float(b1)],
        "density_scale": float(density_scale),
        "slope_scales": [float(v) for v in slope_scales],
        "bands": band_traces,
    }
    return tail, trace


def _safe_unit(vec):
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        out = np.zeros_like(v, dtype=np.float64)
        if out.size > 0:
            out[0] = 1.0
        return out
    return v / n


def _angular_log_interp_gain(freqs, center_freqs, band_gains, bandwidth_oct=1.0):
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)
    fc = np.asarray(center_freqs, dtype=np.float64).reshape(-1)
    g = np.asarray(band_gains, dtype=np.float64).reshape(-1)
    if f.size == 0 or fc.size == 0 or g.size == 0:
        return np.ones_like(f, dtype=np.float64)

    n = min(fc.size, g.size)
    fc = fc[:n]
    g = np.clip(g[:n], 1e-4, 10.0)
    out = np.zeros_like(f, dtype=np.float64)
    wsum = np.zeros_like(f, dtype=np.float64)
    for i in range(n):
        w = _log_gaussian_band_weights(f, fc[i], bandwidth_oct=bandwidth_oct)
        out += np.log(g[i]) * w
        wsum += w
    fill = float(np.mean(np.log(g)))
    mask = wsum > 1e-8
    resp = np.empty_like(f, dtype=np.float64)
    resp[mask] = np.exp(out[mask] / wsum[mask])
    resp[~mask] = np.exp(fill)
    resp[0] = 1.0
    return resp


def _apply_frequency_response(sig, fs, center_freqs, band_gains, bandwidth_oct=1.0):
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(fs))
    resp = _angular_log_interp_gain(freqs, center_freqs, band_gains, bandwidth_oct=bandwidth_oct)
    y = np.fft.irfft(np.fft.rfft(x) * resp, n=x.size)
    return np.asarray(y, dtype=np.float64)


def _compute_voice_radiation_band_gains(center_freqs, src_xyz, mic_xyz, source_forward, params):
    fc = np.asarray(center_freqs, dtype=np.float64).reshape(-1)
    if fc.size == 0:
        return np.ones(0, dtype=np.float64), {}

    ray = _safe_unit(np.asarray(mic_xyz, dtype=np.float64) - np.asarray(src_xyz, dtype=np.float64))
    fwd = _safe_unit(source_forward)
    cos_theta = float(np.clip(np.dot(fwd, ray), -1.0, 1.0))
    frontness = 0.5 * (1.0 + cos_theta)
    backness = 1.0 - frontness
    elevation = float(np.clip(ray[2], -1.0, 1.0)) if ray.size >= 3 else 0.0

    directivity_strength = float(params.get("source_directivity_strength", DEFAULT_SOURCE_DIRECTIVITY_STRENGTH))
    head_shadow_strength = float(params.get("source_head_shadow_strength", DEFAULT_SOURCE_HEAD_SHADOW_STRENGTH))
    torso_strength = float(params.get("source_torso_scattering_strength", DEFAULT_SOURCE_TORSO_SCATTERING_STRENGTH))
    head_radius = float(max(0.04, params.get("source_head_radius_m", DEFAULT_SOURCE_HEAD_RADIUS_M)))
    torso_radius = float(max(0.08, params.get("source_torso_radius_m", DEFAULT_SOURCE_TORSO_RADIUS_M)))

    head_transition_hz = float(max(350.0, params.get("sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S) / (2.0 * np.pi * head_radius)))
    torso_transition_hz = float(max(180.0, params.get("sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S) / (2.0 * np.pi * torso_radius)))
    freq_weight = 1.0 - np.exp(-np.maximum(fc, 1.0) / head_transition_hz)

    # Spherical-harmonic style radiation model: low frequencies stay close to omni,
    # high frequencies become forward-biased with stronger rear attenuation.
    p2 = 0.5 * (3.0 * cos_theta * cos_theta - 1.0)
    dir_gain = 1.0 + directivity_strength * (0.55 * freq_weight * cos_theta + 0.18 * (freq_weight ** 1.3) * p2)
    dir_gain = np.clip(dir_gain, 0.18, 2.4)

    # Source-side head shadow: mainly high-frequency rear attenuation.
    shadow_db = head_shadow_strength * (5.5 + 6.5 * freq_weight) * backness * (1.0 - 0.25 * abs(elevation))
    shadow_gain = np.power(10.0, -np.clip(shadow_db, 0.0, 18.0) / 20.0)

    # Torso scattering: low-mid presence in front, upper-mid loss toward the back.
    low_mid = np.exp(-0.5 * (np.log2(np.maximum(fc, 1.0) / max(350.0, torso_transition_hz)) / 0.80) ** 2)
    presence = np.exp(-0.5 * (np.log2(np.maximum(fc, 1.0) / max(900.0, 1.8 * torso_transition_hz)) / 0.70) ** 2)
    rear_notch = np.exp(-0.5 * (np.log2(np.maximum(fc, 1.0) / 2600.0) / 0.80) ** 2)
    torso_db = torso_strength * (
        (1.2 * frontness + 0.4 * (1.0 - abs(elevation))) * low_mid
        + (2.2 * frontness) * presence
        - (3.0 * backness) * rear_notch
    )
    torso_gain = np.power(10.0, np.clip(torso_db, -8.0, 6.0) / 20.0)

    total_gain = np.clip(dir_gain * shadow_gain * torso_gain, 0.08, 3.0)
    trace = {
        "cos_theta": float(cos_theta),
        "frontness": float(frontness),
        "backness": float(backness),
        "elevation_component": float(elevation),
        "head_transition_hz": float(head_transition_hz),
        "torso_transition_hz": float(torso_transition_hz),
        "band_gains": [float(v) for v in total_gain.tolist()],
    }
    return total_gain, trace


def _apply_source_radiation_and_scattering(rir, fs, split_idx, src_xyz, mic_xyz, source_forward, params):
    r = np.asarray(rir, dtype=np.float64).reshape(-1)
    if r.size < 16:
        return r, {"enabled": False}

    fc = np.asarray(params.get("center_freqs", [125, 250, 500, 1000, 2000, 4000, 8000]), dtype=np.float64).reshape(-1)
    if fc.size == 0:
        return r, {"enabled": False}

    direct_gains, ang_trace = _compute_voice_radiation_band_gains(
        fc,
        src_xyz=src_xyz,
        mic_xyz=mic_xyz,
        source_forward=source_forward,
        params=params,
    )
    scat = np.asarray(params.get("band_scattering_curve", np.full(fc.shape, 0.35)), dtype=np.float64).reshape(-1)
    if scat.size != fc.size:
        scat = np.full(fc.shape, float(np.nanmean(scat)) if scat.size > 0 else 0.35, dtype=np.float64)
    scat = np.clip(scat, 0.05, 0.95)

    direct_idx = int(np.argmax(np.abs(r[: min(r.size, max(16, int(0.03 * fs)))])))
    direct_end = min(r.size, direct_idx + max(1, int(0.003 * fs)))
    early_end = min(r.size, max(direct_end + 1, int(split_idx)))
    bw = float(params.get("source_directivity_bandwidth_oct", DEFAULT_SOURCE_DIRECTIVITY_BANDWIDTH_OCT))

    early_mix = np.clip(0.22 + 0.55 * (1.0 - scat), 0.10, 0.85)
    late_mix = np.clip(0.04 + 0.16 * (1.0 - scat), 0.03, 0.28)
    early_gains = np.exp(early_mix * np.log(np.clip(direct_gains, 1e-4, 10.0)))
    late_gains = np.exp(late_mix * np.log(np.clip(direct_gains, 1e-4, 10.0)))

    r_dir = _apply_frequency_response(r, fs, fc, direct_gains, bandwidth_oct=bw)
    r_early = _apply_frequency_response(r, fs, fc, early_gains, bandwidth_oct=bw)
    r_late = _apply_frequency_response(r, fs, fc, late_gains, bandwidth_oct=bw)

    out = np.zeros_like(r)
    out[:direct_end] = r_dir[:direct_end]
    out[direct_end:early_end] = r_early[direct_end:early_end]
    out[early_end:] = r_late[early_end:]
    trace = {
        "enabled": True,
        "variant": "voice_radiation_head_torso",
        "direct_index": int(direct_idx),
        "direct_end": int(direct_end),
        "early_end": int(early_end),
        "bandwidth_oct": float(bw),
        "direct_band_gains": [float(v) for v in direct_gains.tolist()],
        "early_band_gains": [float(v) for v in early_gains.tolist()],
        "late_band_gains": [float(v) for v in late_gains.tolist()],
        "angular": ang_trace,
    }
    return out, trace


def _weighted_scattering_scalar(scattering_curve, center_freqs):
    s = np.asarray(scattering_curve, dtype=np.float64)
    f = np.asarray(center_freqs, dtype=np.float64)
    w = np.sqrt(np.maximum(f, 1.0) / np.maximum(np.min(f), 1.0))
    return float(np.clip(np.average(s, weights=w), 0.05, 0.95))


def _sample_face_categories(rng, face_category_groups=None):
    groups = face_category_groups or DEFAULT_MATERIAL_FACE_CATEGORY_GROUPS
    wall_candidates = list(groups.get("wall", ()))
    floor_candidates = list(groups.get("floor", ()))
    ceil_candidates = list(groups.get("ceiling", ()))
    if len(wall_candidates) == 0 or len(floor_candidates) == 0 or len(ceil_candidates) == 0:
        raise ValueError("material_face_category_groups must define non-empty wall/floor/ceiling candidates")
    return {
        "west": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "east": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "south": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "north": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "floor": floor_candidates[int(rng.integers(0, len(floor_candidates)))],
        "ceiling": ceil_candidates[int(rng.integers(0, len(ceil_candidates)))],
    }


def _build_materials_from_library(center_freqs, alpha_mean, face_categories, rng, material_library=None):
    library = material_library or DEFAULT_MATERIAL_LIBRARY
    materials = {}
    trace = {}
    coeff_stack = []
    for face, cat in face_categories.items():
        if cat not in library:
            raise KeyError(f"Unknown material category: {cat}")
        base = library[cat]
        abs_base = np.asarray(base["absorption"], dtype=np.float64)
        scat_base = np.asarray(base["scattering"], dtype=np.float64)

        shape = abs_base / max(float(np.mean(abs_base)), 1e-6)
        coeffs = np.clip(
            alpha_mean * shape * rng.uniform(0.92, 1.08, size=abs_base.shape[0]) * float(rng.uniform(0.90, 1.10)),
            0.01,
            0.98,
        )
        scat_curve = np.clip(scat_base * rng.uniform(0.90, 1.10, size=scat_base.shape[0]), 0.05, 0.98)
        scat_scalar = _weighted_scattering_scalar(scat_curve, center_freqs)

        materials[face] = pra.Material(
            {"coeffs": coeffs, "scattering": scat_scalar, "center_freqs": center_freqs}
        )
        trace[face] = {
            "category": cat,
            "absorption_coeffs": coeffs.tolist(),
            "scattering_curve": scat_curve.tolist(),
            "scattering_scalar": float(scat_scalar),
        }
        coeff_stack.append(coeffs)

    return materials, trace, np.mean(np.stack(coeff_stack, axis=0), axis=0)


def _sample_common_room_params(
    lx,
    ly,
    lz,
    fs,
    rng,
    rt60_target,
    material_library=None,
    face_category_groups=None,
    sound_speed_m_s=None,
):
    c = float(DEFAULT_SOUND_SPEED_M_S if sound_speed_m_s is None else sound_speed_m_s)
    center_freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float64)
    room_dim = [float(lx), float(ly), float(lz)]
    rt60_value = float(rng.uniform(0.1, 1.0) if rt60_target is None else rt60_target)

    volume = float(lx * ly * lz)
    surface = float(2.0 * (lx * ly + lx * lz + ly * lz))
    alpha_mean = float(np.clip(0.161 * volume / max(surface * rt60_value, 1e-6), 0.03, 0.75))

    face_categories = _sample_face_categories(rng, face_category_groups=face_category_groups)
    materials, material_trace, alpha_bar = _build_materials_from_library(
        center_freqs=center_freqs,
        alpha_mean=alpha_mean,
        face_categories=face_categories,
        rng=rng,
        material_library=material_library,
    )
    alpha_continuous = interp1d(np.log(center_freqs), alpha_bar, kind="linear", fill_value="extrapolate")
    max_order = int(np.clip(np.ceil(c * float(rng.uniform(0.06, 0.12)) / max(min(room_dim), 1e-6)), 5, 40))

    return {
        "room_dim": room_dim,
        "RT60_target": rt60_value,
        "center_freqs": center_freqs,
        "alpha_continuous": alpha_continuous,
        "materials": materials,
        "material_trace": material_trace,
        "face_categories": face_categories,
        "max_order": max_order,
    }


def sample_room_params(
    lx,
    ly,
    lz,
    fs=32000,
    rng=None,
    rt60_target=None,
    material_library=None,
    face_category_groups=None,
    sound_speed_m_s=None,
):
    rng = np.random.default_rng(0) if rng is None else rng
    return _sample_common_room_params(
        lx=lx,
        ly=ly,
        lz=lz,
        fs=fs,
        rng=rng,
        rt60_target=rt60_target,
        material_library=material_library,
        face_category_groups=face_category_groups,
        sound_speed_m_s=sound_speed_m_s,
    )


def simulate_rir_with_params(
    mic_xyz,
    src_xyz,
    angle_offset,
    lx,
    ly,
    lz,
    fs,
    params,
    rng=None,
    sound_speed_m_s=None,
):
    rng = np.random.default_rng(0) if rng is None else rng
    rt60_target = params["RT60_target"]
    alpha_continuous = params["alpha_continuous"]
    c = float(params.get("sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S if sound_speed_m_s is None else sound_speed_m_s))

    room = pra.ShoeBox(
        [lx, ly, lz],
        fs=fs,
        materials=params["materials"],
        max_order=int(params["max_order"]),
        use_rand_ism=True,
        air_absorption=True,
    )
    azimuth = np.deg2rad(angle_offset)
    source_forward = np.array([np.cos(azimuth), np.sin(azimuth), 0.0], dtype=np.float64)
    room.add_source(list(src_xyz), signal=None)
    room.add_microphone_array(np.array(mic_xyz, dtype=np.float64).reshape(3, 1))
    room.compute_rir()
    rir_ism = np.asarray(room.rir[0][0], dtype=np.float64)

    f_sch = 2000.0 * np.sqrt(max(rt60_target, 1e-3) / max(lx * ly * lz, 1e-3))
    t_split = np.clip(3.0 / max(f_sch, 50.0), 0.05, 0.12)
    fade_len = int(0.02 * fs)
    split_idx = int(np.clip(int(t_split * fs), fade_len + 1, len(rir_ism)))
    early = rir_ism[:split_idx]

    jitter = int(rng.uniform(0.2e-3, 0.8e-3) * fs)
    if jitter >= 2:
        early = np.convolve(early, rng.standard_normal(jitter) * 0.05, mode="same")
    early_f = np.fft.rfft(early)
    early = np.fft.irfft(
        early_f * np.exp(1j * rng.uniform(-0.1, 0.1, size=early_f.shape)),
        n=len(early),
    )

    tail_len = max(int(rt60_target * fs * 1.1), len(rir_ism) - split_idx)
    tail, late_reverb_trace = _synthesize_multiband_late_reverb(
        fs=fs,
        tail_len=tail_len,
        room_dim=(lx, ly, lz),
        rt60_target=rt60_target,
        params=params,
        rng=rng,
    )
    tail = apply_highpass(tail, fs, cutoff=float(params.get("late_tail_highpass_hz", DEFAULT_LATE_TAIL_HIGHPASS_HZ)))

    mode_n_range = params.get("mode_n_range", [DEFAULT_MODE_N_MIN, DEFAULT_MODE_N_MAX])
    mode_rel_db_range = params.get("mode_rel_db_range", [DEFAULT_MODE_REL_DB_MIN, DEFAULT_MODE_REL_DB_MAX])
    n0, n1 = sorted((int(mode_n_range[0]), int(mode_n_range[1])))
    r0, r1 = sorted((float(mode_rel_db_range[0]), float(mode_rel_db_range[1])))
    tail, mode_meta = add_low_freq_modes(
        tail,
        fs,
        room_dim=(lx, ly, lz),
        rt60=rt60_target,
        fmin=float(params.get("mode_fmin_hz", DEFAULT_MODE_FMIN_HZ)),
        fmax=float(params.get("mode_fmax_hz", DEFAULT_MODE_FMAX_HZ)),
        n_modes_range=(n0, n1),
        rel_db_range=(r0, r1),
        sound_speed_m_s=c,
        rng=rng,
        return_meta=True,
    )

    w = np.linspace(0, 1, fade_len, endpoint=False)
    a0 = split_idx - fade_len
    tail *= np.sqrt(np.mean(early[a0:split_idx] ** 2) + 1e-12) / np.sqrt(np.mean(tail[:fade_len] ** 2) + 1e-12)
    rir = np.concatenate([early[:a0], early[a0:split_idx] * (1 - w) + tail[:fade_len] * w, tail[fade_len:]])
    rir, source_radiation_trace = _apply_source_radiation_and_scattering(
        rir,
        fs=fs,
        split_idx=split_idx,
        src_xyz=src_xyz,
        mic_xyz=mic_xyz,
        source_forward=source_forward,
        params=params,
    )

    params["_trace_last"] = {
        "engine_variant": "vector",
        "split_idx": int(split_idx),
        "split_time_ms": float(1000.0 * split_idx / fs),
        "fade_len": int(fade_len),
        "tail_len": int(tail_len),
        "late_reverb": late_reverb_trace,
        "source_radiation": source_radiation_trace,
        "max_order": int(params.get("max_order", -1)),
        "mode_meta": mode_meta,
    }
    return rir, rt60_target


class BaseEngine:
    """
    SE-oriented RIR generator with:
    1) room-custom + generic mixed sampling,
    2) band-wise RT60 perturbation (with smooth constraint),
    3) DRR/C50 post-control for direct/early/late balance.
    """

    def __init__(
        self,
        fs,
        mic_info,
        custom_room_range,
        generic_room_range=None,
        custom_rt60_range=(0.2, 0.8),
        generic_rt60_range=(0.15, 1.1),
        generic_mix_prob=0.3,
        center_jitter_oct=1.0 / 6.0,
        band_rt60_jitter_oct=1.0 / 8.0,
        band_smoothing_passes=2,
        source_dist_range=(0.7, 4.5),
        doa_range=None,
        drr_range_db=(-4.0, 12.0),
        c50_range_db=(-2.0, 16.0),
        snr_range_db=(0.0, 25.0),
        enable_physical_calibration=True,
        direct_peak_at_1m=0.10,
        physical_scale_clip=(0.05, 20.0),
        enable_final_output_norm=True,
        final_peak_dbfs=-3.0,
        final_norm_attenuate_only=True,
        final_norm_gain_clip=(0.05, 20.0),
    ):
        self.fs = int(fs)
        self.mic_info = dict(mic_info)

        self.custom_room_range = dict(custom_room_range)
        self.generic_room_range = dict(generic_room_range or {
            "lx": (2.5, 8.5),
            "ly": (2.5, 8.5),
            "lz": (2.3, 4.0),
        })
        self.custom_rt60_range = tuple(custom_rt60_range)
        self.generic_rt60_range = tuple(generic_rt60_range)
        self.generic_mix_prob = float(np.clip(generic_mix_prob, 0.0, 1.0))

        self.center_jitter_oct = float(max(0.0, center_jitter_oct))
        self.band_rt60_jitter_oct = float(max(0.0, band_rt60_jitter_oct))
        self.band_smoothing_passes = int(max(0, band_smoothing_passes))

        self.source_dist_range = tuple(source_dist_range)
        self.drr_range_db = tuple(drr_range_db)
        self.c50_range_db = tuple(c50_range_db)
        self.snr_range_db = tuple(snr_range_db)
        self.enable_physical_calibration = bool(enable_physical_calibration)
        self.direct_peak_at_1m = float(max(1e-5, direct_peak_at_1m))
        self.physical_scale_clip = tuple(physical_scale_clip)
        self.enable_final_output_norm = bool(enable_final_output_norm)
        self.final_peak_dbfs = float(final_peak_dbfs)
        self.final_norm_attenuate_only = bool(final_norm_attenuate_only)
        self.final_norm_gain_clip = tuple(final_norm_gain_clip)

        if doa_range is None:
            if self.mic_info.get("array_type") == "linear":
                self.doa_range = (0.0, 180.0)
            else:
                self.doa_range = (0.0, 360.0)
        else:
            self.doa_range = tuple(doa_range)

        # Include >8k anchors so high-frequency absorption can be explicitly shaped.
        self.band_centers_ref = np.array(
            [125, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000],
            dtype=np.float64,
        )
        # Physical material profile defaults (can be overridden by config file).
        self.material_center_freqs = self.band_centers_ref.copy()
        self.material_absorption_curve = np.array(
            [1.30, 1.22, 1.12, 1.00, 0.94, 0.92, 0.95, 1.10, 1.20],
            dtype=np.float64,
        )
        self.material_scattering_curve = np.array(
            [0.22, 0.24, 0.26, 0.30, 0.34, 0.40, 0.48, 0.54, 0.58],
            dtype=np.float64,
        )
        self.material_library = {
            name: {
                "absorption": tuple(float(v) for v in spec["absorption"]),
                "scattering": tuple(float(v) for v in spec["scattering"]),
            }
            for name, spec in DEFAULT_MATERIAL_LIBRARY.items()
        }
        self.material_face_category_groups = {
            face_type: tuple(str(v) for v in names)
            for face_type, names in DEFAULT_MATERIAL_FACE_CATEGORY_GROUPS.items()
        }
        self.material_face_absorption_scale = {
            "west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0, "floor": 1.0, "ceiling": 1.0
        }
        self.material_face_scattering_scale = {
            "west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0, "floor": 1.0, "ceiling": 1.0
        }
        self.sound_speed_m_s = float(DEFAULT_SOUND_SPEED_M_S)
        self.late_tail_highpass_hz = float(DEFAULT_LATE_TAIL_HIGHPASS_HZ)
        self.late_reverb_bandwidth_oct = float(DEFAULT_LATE_REVERB_BANDWIDTH_OCT)
        self.late_reverb_break_fractions = tuple(float(v) for v in DEFAULT_LATE_REVERB_BREAK_FRACTIONS)
        self.late_reverb_density_scale = float(DEFAULT_LATE_REVERB_DENSITY_SCALE)
        self.late_reverb_slope_scales = tuple(float(v) for v in DEFAULT_LATE_REVERB_SLOPE_SCALES)
        self.source_directivity_strength = float(DEFAULT_SOURCE_DIRECTIVITY_STRENGTH)
        self.source_head_shadow_strength = float(DEFAULT_SOURCE_HEAD_SHADOW_STRENGTH)
        self.source_torso_scattering_strength = float(DEFAULT_SOURCE_TORSO_SCATTERING_STRENGTH)
        self.source_head_radius_m = float(DEFAULT_SOURCE_HEAD_RADIUS_M)
        self.source_torso_radius_m = float(DEFAULT_SOURCE_TORSO_RADIUS_M)
        self.source_directivity_bandwidth_oct = float(DEFAULT_SOURCE_DIRECTIVITY_BANDWIDTH_OCT)
        # Low-frequency modal controls (overridden from cfg).
        self.mode_fmin_hz = float(DEFAULT_MODE_FMIN_HZ)
        self.mode_fmax_hz = float(DEFAULT_MODE_FMAX_HZ)
        self.mode_n_min = int(DEFAULT_MODE_N_MIN)
        self.mode_n_max = int(DEFAULT_MODE_N_MAX)
        self.mode_rel_db_min = float(DEFAULT_MODE_REL_DB_MIN)
        self.mode_rel_db_max = float(DEFAULT_MODE_REL_DB_MAX)

        # Updated by fit_from_recordings(...)
        self.fitted = None
        self.custom_band_rt60_prior = None
        self.custom_rt60_center = None
        self.custom_noise_rms = None
        self.custom_noise_tilt_db_oct = None

    def _resolve_audio_items(self, items, name):
        if items is None:
            return []
        if isinstance(items, (str, Path)):
            p = Path(items)
            if p.is_dir():
                exts = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
                files = []
                for ext in exts:
                    files.extend([str(x) for x in p.rglob(f"*{ext}")])
                files = sorted(set(files))
                if not files:
                    raise ValueError(f"No audio files found under {p} for {name}")
                return files
            if p.is_file():
                return [str(p)]
            raise FileNotFoundError(f"{name} path not found: {p}")
        if isinstance(items, (list, tuple)):
            if len(items) == 0:
                raise ValueError(f"{name} is empty")
            return list(items)
        raise TypeError(f"{name} must be path or list/tuple, got {type(items)}")

    def _load_audio_mono_keep_fs(self, item):
        """
        Load mono audio but keep original sampling rate.

        Used by acoustic-parameter inversion so we do not distort decay/band
        characteristics by resampling before estimation.
        """
        if isinstance(item, np.ndarray):
            x = to_mono(item)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, int(self.fs), "<ndarray>"
        if isinstance(item, dict):
            x = to_mono(item["audio"])
            fs_in = int(item.get("fs", self.fs))
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, fs_in, str(item.get("id", item.get("path", "<dict-audio>")))
        if isinstance(item, (str, Path)):
            path = str(item)
            x, fs_in = sf.read(path, dtype="float64")
            x = to_mono(x)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, int(round(float(fs_in))), path
        raise TypeError(f"Unsupported audio item type: {type(item)}")

    @staticmethod
    def _max_dist_to_walls(center_xy, dir_xy, room_size, margin):
        cx, cy = float(center_xy[0]), float(center_xy[1])
        dx, dy = float(dir_xy[0]), float(dir_xy[1])
        lx, ly, _ = room_size
        m = float(margin)
        eps = 1e-9

        ts = []
        if abs(dx) > eps:
            t1 = (m - cx) / dx
            t2 = (lx - m - cx) / dx
            if t1 > 0:
                ts.append(t1)
            if t2 > 0:
                ts.append(t2)
        if abs(dy) > eps:
            t3 = (m - cy) / dy
            t4 = (ly - m - cy) / dy
            if t3 > 0:
                ts.append(t3)
            if t4 > 0:
                ts.append(t4)
        return 0.0 if len(ts) == 0 else float(min(ts))

    def _sample_room_size(self, room_range, rng):
        lx = float(rng.uniform(*room_range["lx"]))
        ly = float(rng.uniform(*room_range["ly"]))
        lz = float(rng.uniform(*room_range["lz"]))
        return [round(lx, 3), round(ly, 3), round(lz, 3)]

    def _get_mic_array_loc(self, room_size, rng, min_dis_to_wall=0.6):
        lx, ly, lz = room_size
        h = self.mic_info.get("device_height")
        if h in (None, [], 0):
            h = float(rng.uniform(0.9, max(1.0, lz - 0.3)))
        h = float(np.clip(float(h), 0.6, lz - 0.15))

        arr = self.mic_info.get("array_type", "linear")
        if arr == "circular":
            m = int(self.mic_info.get("mic_num", 4))
            rad = float(self.mic_info.get("mic_radius", 0.05))
            max_rad = max(0.02, min(lx, ly) / 2.0 - min_dis_to_wall - 0.02)
            rad = min(rad, max_rad)
            p = pra.bf.circular_2D_array(center=[lx / 2.0, ly / 2.0], M=m, phi0=np.pi, radius=rad)
            z = np.full(m, h, dtype=np.float64)
            return np.asarray([p[0], p[1], z], dtype=np.float64)

        if "mic_pos" in self.mic_info:
            mp = np.asarray(self.mic_info["mic_pos"], dtype=np.float64).reshape(-1)
            mp = mp - np.min(mp)
        else:
            m = int(self.mic_info.get("mic_num", 4))
            d = float(self.mic_info.get("mic_spacing", 0.05))
            mp = np.arange(m, dtype=np.float64) * d

        span = float(np.max(mp) - np.min(mp)) if mp.size > 1 else 0.0
        usable = max(0.12, lx - 2.0 * min_dis_to_wall)
        if span > usable:
            mp = mp * (usable / max(span, 1e-9))
            span = float(np.max(mp) - np.min(mp)) if mp.size > 1 else 0.0

        sx = (lx - span) / 2.0
        x = sx + mp
        y = np.full_like(x, ly - min_dis_to_wall, dtype=np.float64)
        z = np.full(x.shape[0], h, dtype=np.float64)
        return np.asarray([x, y, z], dtype=np.float64)

    def _get_source_loc(self, room_size, mic_loc, doa_deg, rng, min_dis_to_wall=0.5):
        lx, ly, lz = room_size
        cx = float(np.mean(mic_loc[0]))
        cy = float(np.mean(mic_loc[1]))
        cz = float(np.mean(mic_loc[2]))

        rad = np.deg2rad(float(doa_deg))
        dir_xy = (-np.cos(rad), -np.sin(rad))
        max_d = self._max_dist_to_walls((cx, cy), dir_xy, room_size, min_dis_to_wall)
        # If current direction cannot satisfy minimal source distance, flip direction.
        # Why: keep DOA semantics while avoiding invalid geometry near walls.
        if max_d <= self.source_dist_range[0]:
            dir_xy = (-dir_xy[0], -dir_xy[1])
            max_d = self._max_dist_to_walls((cx, cy), dir_xy, room_size, min_dis_to_wall)

        min_d = float(self.source_dist_range[0])
        max_req = float(self.source_dist_range[1])
        max_d = min(max_req, max_d)
        if max_d <= min_d:
            d = min_d
        else:
            d = float(rng.uniform(min_d, max_d))

        sx = float(np.clip(cx + d * dir_xy[0], min_dis_to_wall, lx - min_dis_to_wall))
        sy = float(np.clip(cy + d * dir_xy[1], min_dis_to_wall, ly - min_dis_to_wall))
        z_hi = min(1.8, lz - min_dis_to_wall)
        if z_hi <= 1.0:
            sz = float(np.clip(cz, 0.6, lz - 0.15))
        else:
            sz = float(rng.uniform(1.0, z_hi))
        return np.asarray([sx, sy, sz], dtype=np.float64), float(d)

    @staticmethod
    def _smooth_curve(vals, passes=2):
        x = np.asarray(vals, dtype=np.float64).copy()
        if x.size <= 2:
            return x
        k = np.asarray([0.25, 0.5, 0.25], dtype=np.float64)
        for _ in range(max(0, int(passes))):
            xp = np.pad(x, (1, 1), mode="edge")
            x = np.convolve(xp, k, mode="valid")
        return x

    def _jitter_band_centers(self, rng):
        # Randomly perturb band centers per-sample to avoid "fixed comb" artifacts.
        # Why: fixed bands can be memorized by the model and may show stripe-like
        # spectral artifacts in enhanced outputs.
        f0 = self.band_centers_ref.astype(np.float64)
        u = rng.uniform(-self.center_jitter_oct, self.center_jitter_oct, size=f0.shape[0])
        fc = f0 * (2.0 ** u)
        fc = np.sort(fc)
        ny = 0.48 * self.fs
        fc = np.clip(fc, 63.0, max(200.0, ny))
        for i in range(1, len(fc)):
            if fc[i] <= fc[i - 1] * 1.04:
                fc[i] = fc[i - 1] * 1.04
        if fc[-1] > ny:
            scale = ny / fc[-1]
            fc *= scale
        return fc

    def _sample_band_rt60(self, base_rt60, rng, band_prior=None):
        if band_prior is not None:
            rt = np.asarray(band_prior, dtype=np.float64).copy()
            if rt.shape[0] != self.band_centers_ref.shape[0]:
                # Backward compatibility for old state json with fewer bands:
                # interpolate existing priors onto current band grid.
                if rt.shape[0] >= 2:
                    src = np.linspace(0.0, 1.0, rt.shape[0])
                    dst = np.linspace(0.0, 1.0, self.band_centers_ref.shape[0])
                    rt = np.interp(dst, src, rt).astype(np.float64)
                else:
                    rt = np.full_like(self.band_centers_ref, float(base_rt60), dtype=np.float64)
        else:
            rt = np.full_like(self.band_centers_ref, float(base_rt60), dtype=np.float64)

        # Log-domain jitter keeps multiplicative behavior physically meaningful
        # (e.g., +1/6 octave equivalent scaling) instead of additive offset.
        u = rng.uniform(-self.band_rt60_jitter_oct, self.band_rt60_jitter_oct, size=rt.shape[0])
        rt = rt * (2.0 ** u)
        # Smooth in log-domain to avoid sawtooth band profile, which is often non-physical
        # and can over-regularize the model toward unrealistic band discontinuities.
        rt = self._smooth_curve(np.log(np.clip(rt, 0.08, 2.8)), passes=self.band_smoothing_passes)
        rt = np.exp(rt)
        return np.clip(rt, 0.08, 2.8)

    @staticmethod
    def _interp_profile(src_freqs, src_vals, target_freqs, default_val):
        sf = np.asarray(src_freqs, dtype=np.float64).reshape(-1)
        sv = np.asarray(src_vals, dtype=np.float64).reshape(-1)
        tf = np.asarray(target_freqs, dtype=np.float64).reshape(-1)
        if tf.size == 0:
            return tf
        if sf.size < 2 or sv.size < 2:
            return np.full(tf.shape, float(default_val), dtype=np.float64)
        n = min(sf.size, sv.size)
        sf = sf[:n]
        sv = sv[:n]

        order = np.argsort(sf)
        sf = sf[order]
        sv = sv[order]
        sf, uniq = np.unique(sf, return_index=True)
        sv = sv[uniq]
        if sf.size < 2:
            return np.full(tf.shape, float(default_val), dtype=np.float64)

        f_min = float(max(20.0, sf[0]))
        f_max = float(max(f_min * 1.05, sf[-1]))
        tf_c = np.clip(tf, f_min, f_max)
        return np.interp(np.log(tf_c), np.log(sf), sv).astype(np.float64)

    @staticmethod
    def _weighted_scattering_scalar(scattering_curve, center_freqs):
        s = np.asarray(scattering_curve, dtype=np.float64).reshape(-1)
        f = np.asarray(center_freqs, dtype=np.float64).reshape(-1)
        n = min(s.size, f.size)
        if n == 0:
            return 0.35
        s = s[:n]
        f = f[:n]
        f0 = max(float(np.min(f)), 1.0)
        w = np.sqrt(np.maximum(f, 1.0) / f0)
        return float(np.clip(np.average(s, weights=w), 0.05, 0.95))

    @staticmethod
    def _to_material_dict(
        keys,
        coeffs,
        center_freqs,
        rng,
        scattering_curve=None,
        face_abs_scale=None,
        face_scat_scale=None,
    ):
        out = {}
        c0 = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        fc = np.asarray(center_freqs, dtype=np.float64).reshape(-1)
        if c0.size == 0:
            return out

        if scattering_curve is None:
            s_curve = np.full(c0.shape, 0.35, dtype=np.float64)
        else:
            s_raw = np.asarray(scattering_curve, dtype=np.float64).reshape(-1)
            if s_raw.size != c0.size:
                s_curve = np.full(c0.shape, float(np.mean(s_raw)) if s_raw.size > 0 else 0.35, dtype=np.float64)
            else:
                s_curve = s_raw
        s_base = BaseEngine._weighted_scattering_scalar(s_curve, fc)

        abs_scale_map = dict(face_abs_scale or {})
        scat_scale_map = dict(face_scat_scale or {})
        for k in keys:
            abs_scale = float(abs_scale_map.get(k, 1.0))
            scat_scale = float(scat_scale_map.get(k, 1.0))

            c = c0 * abs_scale
            c = np.clip(c * rng.uniform(0.96, 1.04, size=c.shape[0]), 0.01, 0.99)
            s = float(np.clip(s_base * scat_scale * rng.uniform(0.95, 1.05), 0.05, 0.95))
            out[k] = pra.Material({
                "coeffs": c,
                "scattering": s,
                "center_freqs": fc,
            })
        return out

    def _apply_band_profile_to_params(self, params, room_size, band_centers, band_rt60, rng):
        lx, ly, lz = room_size
        V = float(lx * ly * lz)
        S = float(2.0 * (lx * ly + lx * lz + ly * lz))
        # Map target RT60(f) -> equivalent absorption alpha(f) via Sabine-style inversion.
        # Why: this bridges data-driven target profile and physical simulator parameters.
        alpha = np.clip(0.161 * V / (S * np.maximum(band_rt60, 1e-4)), 0.02, 0.95)
        alpha = self._smooth_curve(alpha, passes=1)

        # Apply physical material absorption shape directly in coefficient domain.
        # This is not a post-hoc gain trick: it reshapes absorption alpha(f) used by
        # the room simulator and therefore changes decay physically.
        fc = np.asarray(band_centers, dtype=np.float64)
        prof_f = np.asarray(getattr(self, "material_center_freqs", fc), dtype=np.float64)
        prof_a = np.asarray(getattr(self, "material_absorption_curve", np.ones_like(prof_f)), dtype=np.float64)
        abs_shape = self._interp_profile(prof_f, prof_a, fc, default_val=1.0)
        abs_shape = np.clip(abs_shape, 0.35, 2.5)
        abs_shape = abs_shape / max(float(np.mean(abs_shape)), 1e-8)
        alpha = np.clip(alpha * abs_shape, 0.02, 0.98)
        alpha = self._smooth_curve(alpha, passes=1)

        # Frequency-dependent scattering profile (converted to pyroom scalar per face).
        prof_s = np.asarray(getattr(self, "material_scattering_curve", np.full_like(prof_f, 0.35)), dtype=np.float64)
        scat_curve = self._interp_profile(prof_f, prof_s, fc, default_val=0.35)
        scat_curve = np.clip(scat_curve, 0.05, 0.95)

        out = dict(params) if isinstance(params, dict) else params
        if not isinstance(out, dict):
            return out

        out["center_freqs"] = np.asarray(band_centers, dtype=np.float64)
        log_fc = np.log(np.asarray(band_centers, dtype=np.float64))
        out["alpha_continuous"] = interp1d(log_fc, np.asarray(alpha, dtype=np.float64), kind="linear", fill_value="extrapolate")

        face_abs_scale = getattr(self, "material_face_absorption_scale", None)
        face_scat_scale = getattr(self, "material_face_scattering_scale", None)
        abs_scale_vals = np.asarray(list(dict(face_abs_scale or {}).values()), dtype=np.float64)
        scat_scale_vals = np.asarray(list(dict(face_scat_scale or {}).values()), dtype=np.float64)
        mean_abs_scale = float(np.mean(abs_scale_vals)) if abs_scale_vals.size > 0 else 1.0
        mean_scat_scale = float(np.mean(scat_scale_vals)) if scat_scale_vals.size > 0 else 1.0
        out["band_rt60"] = np.asarray(band_rt60, dtype=np.float64)
        out["band_alpha"] = np.clip(np.asarray(alpha, dtype=np.float64) * mean_abs_scale, 0.02, 0.98)
        out["band_scattering_curve"] = np.clip(np.asarray(scat_curve, dtype=np.float64) * mean_scat_scale, 0.05, 0.95)
        if "materials" in out:
            keys = list(out["materials"].keys())
            out["materials"] = self._to_material_dict(
                keys,
                alpha,
                out["center_freqs"],
                rng,
                scattering_curve=scat_curve,
                face_abs_scale=face_abs_scale,
                face_scat_scale=face_scat_scale,
            )
        if "material" in out:
            keys = list(out["material"].keys())
            out["material"] = self._to_material_dict(
                keys,
                alpha,
                out["center_freqs"],
                rng,
                scattering_curve=scat_curve,
                face_abs_scale=face_abs_scale,
                face_scat_scale=face_scat_scale,
            )
        return out

    def _direct_ref(self, src_xyz, mic_xyz, n_samples):
        dist = float(np.linalg.norm(np.asarray(src_xyz) - np.asarray(mic_xyz)))
        delay_samp = int(round(dist / float(self.sound_speed_m_s) * self.fs))
        ref = np.zeros(n_samples, dtype=np.float64)
        if delay_samp < n_samples:
            ref[delay_samp] = 1.0 / max(dist, 1e-3)
        return ref

    def _rir_windows(self, rir, c50_ms=50.0, direct_ms=2.5, fs_hz=None):
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        r = np.asarray(rir, dtype=np.float64).reshape(-1)
        n = len(r)
        if n == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), 0

        # Direct-arrival anchor: search peak in first 30ms to robustly locate onset.
        # Why: keeps DRR/C50 windows stable even if absolute delay changes by geometry.
        search_n = min(n, max(16, int(0.03 * fs_use)))
        idx = int(np.argmax(np.abs(r[:search_n])))
        d_len = max(1, int(direct_ms * 1e-3 * fs_use))
        c_len = max(d_len + 1, int(c50_ms * 1e-3 * fs_use))

        m_dir = np.zeros(n, dtype=bool)
        m_early = np.zeros(n, dtype=bool)
        m_late = np.zeros(n, dtype=bool)

        d0 = max(0, idx - 1)
        d1 = min(n, idx + d_len)
        e1 = min(n, idx + c_len)
        m_dir[d0:d1] = True
        m_early[idx:e1] = True
        m_late[e1:] = True
        return m_dir, m_early, m_late, idx

    def _compute_drr_c50(self, rir, fs_hz=None):
        r = np.asarray(rir, dtype=np.float64).reshape(-1)
        m_dir, m_early, m_late, _ = self._rir_windows(r, fs_hz=fs_hz)
        e_d = float(np.sum(r[m_dir] ** 2) + 1e-12)
        e_er = float(np.sum(r[np.logical_and(m_early, ~m_dir)] ** 2) + 1e-12)
        e_l = float(np.sum(r[m_late] ** 2) + 1e-12)
        drr = 10.0 * np.log10(e_d / (e_er + e_l))
        c50 = 10.0 * np.log10((e_d + e_er) / e_l)
        return float(drr), float(c50)

    @staticmethod
    def _extract_impulse_segment(x, fs_hz, pre_ms=3.0, tail_s=1.2):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x, 0
        fs_use = int(fs_hz)
        p = int(np.argmax(np.abs(x)))
        pre_n = int(max(1, round(pre_ms * 1e-3 * fs_use)))
        tail_n = int(max(pre_n + 1, round(tail_s * fs_use)))
        s = max(0, p - pre_n)
        e = min(x.size, s + tail_n)
        return x[s:e], p

    def _is_impulse_like(self, x, fs_hz=None):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < 64:
            return False
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        crest_db = 20.0 * np.log10(peak / (rms + 1e-12))
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        p = int(np.argmax(np.abs(x)))
        d_n = max(1, int(round(0.003 * fs_use)))
        late_s = min(x.size, p + max(d_n, int(round(0.05 * fs_use))))
        e_d = float(np.sum(x[p:min(x.size, p + d_n)] ** 2) + 1e-12)
        e_l = float(np.sum(x[late_s:] ** 2) + 1e-12)
        dlr_db = 10.0 * np.log10(e_d / e_l)
        return bool((crest_db >= 14.0) and (dlr_db >= 3.0))

    def _apply_drr_c50_target(self, rir, drr_tgt_db, c50_tgt_db):
        r = np.asarray(rir, dtype=np.float64).copy().reshape(-1)
        if r.size < 16:
            return r

        m_dir, m_early, m_late, _ = self._rir_windows(r)
        m_er = np.logical_and(m_early, ~m_dir)
        eps = 1e-12

        # Two-step iterative scaling:
        # 1) adjust late energy for C50,
        # 2) adjust direct energy for DRR.
        # Iterate twice because these two metrics are coupled.
        for _ in range(2):
            e_d = float(np.sum(r[m_dir] ** 2) + eps)
            e_er = float(np.sum(r[m_er] ** 2) + eps)
            e_l = float(np.sum(r[m_late] ** 2) + eps)

            C = 10.0 ** (float(c50_tgt_db) / 10.0)
            D = 10.0 ** (float(drr_tgt_db) / 10.0)

            # Step-1: late gain for C50 target.
            y = (e_d + e_er) / (C * e_l + eps)
            y = float(np.clip(y, 0.05, 30.0))
            r[m_late] *= np.sqrt(y)

            # Step-2: direct gain for DRR target (with updated late energy).
            e_d = float(np.sum(r[m_dir] ** 2) + eps)
            e_er = float(np.sum(r[m_er] ** 2) + eps)
            e_l = float(np.sum(r[m_late] ** 2) + eps)
            x = D * (e_er + e_l) / (e_d + eps)
            x = float(np.clip(x, 0.05, 30.0))
            r[m_dir] *= np.sqrt(x)
        return r

    def _solve_shared_drr_c50_segment_gains(self, rir_ref, drr_tgt_db, c50_tgt_db):
        """
        Solve one shared set of segment gains for all channels.

        Why:
        Applying independent DRR/C50 shaping per channel can distort inter-channel
        spatial cues (ITD/ILD). Shared gains preserve multi-channel consistency.
        """
        r = np.asarray(rir_ref, dtype=np.float64).reshape(-1)
        if r.size < 16:
            return {"dir": 1.0, "early": 1.0, "late": 1.0}

        m_dir, m_early, m_late, _ = self._rir_windows(r)
        m_er = np.logical_and(m_early, ~m_dir)
        eps = 1e-12

        e_d = float(np.sum(r[m_dir] ** 2) + eps)
        e_er = float(np.sum(r[m_er] ** 2) + eps)
        e_l = float(np.sum(r[m_late] ** 2) + eps)

        D = 10.0 ** (float(drr_tgt_db) / 10.0)
        C = 10.0 ** (float(c50_tgt_db) / 10.0)

        # Constrained solution: keep early gain as anchor (1.0), solve direct/late.
        # If C<=D, exact positive solution may not exist; we fall back to clipped values.
        g_early_e = 1.0
        denom = max((C - D), 1e-3) * e_l
        g_late_e = ((D + 1.0) * e_er) / max(denom, eps)
        g_late_e = float(np.clip(g_late_e, 0.05, 30.0))

        g_dir_e = D * (g_early_e * e_er + g_late_e * e_l) / max(e_d, eps)
        g_dir_e = float(np.clip(g_dir_e, 0.05, 30.0))

        return {
            "dir": float(np.sqrt(g_dir_e)),
            "early": float(np.sqrt(g_early_e)),
            "late": float(np.sqrt(g_late_e)),
        }

    def _apply_segment_gains(self, rir, gains):
        r = np.asarray(rir, dtype=np.float64).copy().reshape(-1)
        if r.size < 16:
            return r
        m_dir, m_early, m_late, _ = self._rir_windows(r)
        m_er = np.logical_and(m_early, ~m_dir)
        r[m_dir] *= float(gains["dir"])
        r[m_er] *= float(gains["early"])
        r[m_late] *= float(gains["late"])
        return r

    def _apply_drr_c50_target_multich(self, rirs, drr_tgt_db, c50_tgt_db, ref_ch=0):
        if len(rirs) == 0:
            return [], {"enabled": False}
        ref_i = int(np.clip(int(ref_ch), 0, len(rirs) - 1))
        gains = self._solve_shared_drr_c50_segment_gains(
            rirs[ref_i],
            drr_tgt_db=drr_tgt_db,
            c50_tgt_db=c50_tgt_db,
        )
        out = [self._apply_segment_gains(r, gains) for r in rirs]
        trace = {
            "enabled": True,
            "shared_segment_gains": {
                "dir": float(gains["dir"]),
                "early": float(gains["early"]),
                "late": float(gains["late"]),
            },
            "reference_channel": int(ref_i),
        }
        return out, trace

    def _apply_physical_calibration(self, rir, src_xyz, mic_xyz):
        """
        Calibrate RIR amplitude using distance-based direct-path anchor.

        Why:
        Peak-normalizing each RIR destroys distance attenuation cues.
        Here we anchor direct-path level roughly to 1/r, preserving relative
        loudness across source distances.
        """
        r = np.asarray(rir, dtype=np.float64).copy().reshape(-1)
        if r.size == 0:
            return r, 1.0, 0.0

        m_dir, _, _, idx = self._rir_windows(r)
        direct_peak = float(np.max(np.abs(r[m_dir]))) if np.any(m_dir) else float(np.abs(r[idx]))
        if not np.isfinite(direct_peak) or direct_peak <= 1e-12:
            return r, 1.0, 0.0

        dist = float(np.linalg.norm(np.asarray(src_xyz, dtype=np.float64) - np.asarray(mic_xyz, dtype=np.float64)))
        dist = max(0.1, dist)
        target_direct_peak = float(self.direct_peak_at_1m / dist)

        g = target_direct_peak / direct_peak
        lo, hi = float(min(self.physical_scale_clip)), float(max(self.physical_scale_clip))
        g = float(np.clip(g, lo, hi))
        return r * g, g, target_direct_peak

    def _final_peak_normalize_triplet(self, mix, clean=None, ref=None):
        """
        Final output normalization for file writing.

        Why:
        Keeps saved waveform peak in a stable range while applying the same gain
        to mix/clean/ref, so supervision alignment and SNR consistency are preserved.
        """
        y = np.asarray(mix, dtype=np.float64)
        c = None if clean is None else np.asarray(clean, dtype=np.float64)
        r = None if ref is None else np.asarray(ref, dtype=np.float64)

        if not self.enable_final_output_norm or y.size == 0:
            return y, c, r, 1.0, 0.0

        peak = float(np.max(np.abs(y)))
        if not np.isfinite(peak) or peak <= 1e-12:
            return y, c, r, 1.0, peak

        target_peak = float(10.0 ** (self.final_peak_dbfs / 20.0))
        raw_gain = target_peak / peak
        if self.final_norm_attenuate_only:
            raw_gain = min(1.0, raw_gain)

        lo, hi = float(min(self.final_norm_gain_clip)), float(max(self.final_norm_gain_clip))
        gain = float(np.clip(raw_gain, lo, hi))

        y = y * gain
        if c is not None:
            c = c * gain
        if r is not None:
            r = r * gain
        return y, c, r, gain, peak

    def _estimate_rt60_schroeder(self, x, fs_hz=None, noise_comp=True):
        """
        Estimate RT60 from a decay segment using Schroeder integration.

        Why:
        This is the classic robust approach for reverberation decay fitting.
        """
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < max(64, int(0.12 * fs_use)):
            return None
        x = x - np.mean(x)
        e = x * x
        if bool(noise_comp):
            # Remove stationary noise floor before EDC integration; otherwise
            # tail flattening tends to overestimate RT60 in real recordings.
            tail_n = max(64, int(0.1 * e.size))
            noise_p = float(np.median(e[-tail_n:]))
            if np.isfinite(noise_p) and noise_p > 0.0:
                e = np.maximum(e - noise_p, 0.0)
        if np.max(e) <= 1e-12:
            return None

        edc = np.cumsum(e[::-1])[::-1]
        edc = edc / (np.max(edc) + 1e-12)
        db = 10.0 * np.log10(np.maximum(edc, 1e-12))
        t = np.arange(db.size, dtype=np.float64) / float(fs_use)

        # Fit only before first crossing of lower bound to avoid noise-floor bias.
        idx = np.arange(db.size)
        hit_35 = np.where(db <= -35.0)[0]
        if hit_35.size > 0:
            lo_db = -35.0
            end_i = int(hit_35[0])
        else:
            hit_25 = np.where(db <= -25.0)[0]
            if hit_25.size == 0:
                return None
            lo_db = -25.0
            end_i = int(hit_25[0])

        m = (db <= -5.0) & (db >= lo_db) & (idx <= end_i)
        if np.count_nonzero(m) < 20:
            return None
        slope, _ = np.polyfit(t[m], db[m], 1)
        if slope >= -1e-3:
            return None
        rt60 = -60.0 / slope
        if not np.isfinite(rt60):
            return None
        return float(np.clip(rt60, 0.08, 3.0))

    def _estimate_rt60_from_impulse(self, x, fs_hz=None):
        """
        RT60 estimate specialized for impulse-like recordings.
        """
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        seg, _ = self._extract_impulse_segment(x, fs_hz=fs_use, pre_ms=2.0, tail_s=1.0)
        if seg.size < max(64, int(0.12 * fs_use)):
            return None
        p = int(np.argmax(np.abs(seg)))
        decay = seg[p:]
        return self._estimate_rt60_schroeder(decay, fs_hz=fs_use, noise_comp=True)

    def _estimate_rt60_from_recording(self, x, fs_hz=None):
        """
        RT60 estimate from generic recording (speech/noise allowed).

        Why:
        Real recordings are not ideal impulse responses. We first detect strong
        transient-like peaks and estimate tail decay around them; if unavailable,
        we fallback to full-signal Schroeder estimate.
        """
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < max(128, int(0.2 * fs_use)):
            return None

        # Prefer impulse-tail estimator when signal looks like an IR capture.
        if self._is_impulse_like(x, fs_hz=fs_use):
            est_imp = self._estimate_rt60_from_impulse(x, fs_hz=fs_use)
            if est_imp is not None:
                return float(est_imp)

        env = np.abs(x - np.mean(x))
        ma_n = max(1, int(0.01 * fs_use))
        env = np.convolve(env, np.ones(ma_n, dtype=np.float64) / ma_n, mode="same")
        # High percentile threshold selects transient candidates with enough decay tail.
        h = np.percentile(env, 85.0)
        d = max(1, int(0.15 * fs_use))
        peaks = []
        i = d
        while i < len(env) - d:
            if env[i] >= h and env[i] == np.max(env[i - d:i + d + 1]):
                peaks.append(i)
                i += d
            else:
                i += 1

        rt = []
        # Tail length should cover enough decay for regression.
        tail_n = int(0.8 * fs_use)
        for p in peaks[-24:]:
            seg = x[p:min(len(x), p + tail_n)]
            est = self._estimate_rt60_schroeder(seg, fs_hz=fs_use, noise_comp=True)
            if est is not None:
                rt.append(float(est))
        if len(rt) > 0:
            return float(np.median(rt))
        return self._estimate_rt60_schroeder(x, fs_hz=fs_use, noise_comp=True)

    def _estimate_edt_from_impulse(self, x, fs_hz=None):
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        seg, _ = self._extract_impulse_segment(x, fs_hz=fs_use, pre_ms=2.0, tail_s=0.8)
        if seg.size < max(64, int(0.08 * fs_use)):
            return None

        peak_i = int(np.argmax(np.abs(seg)))
        decay = seg[peak_i:]
        if decay.size < max(64, int(0.06 * fs_use)):
            return None

        decay = decay - np.mean(decay[-max(16, int(0.05 * decay.size)):])
        e = decay * decay
        if np.max(e) <= 1e-12:
            return None

        edc = np.cumsum(e[::-1])[::-1]
        edc = edc / (np.max(edc) + 1e-12)
        db = 10.0 * np.log10(np.maximum(edc, 1e-12))
        t = np.arange(db.size, dtype=np.float64) / float(fs_use)

        m = (db <= -0.5) & (db >= -10.0)
        if np.count_nonzero(m) < 12:
            return None
        slope, _ = np.polyfit(t[m], db[m], 1)
        if slope >= -1e-3:
            return None

        edt = -60.0 / slope
        if not np.isfinite(edt):
            return None
        return float(np.clip(edt, 0.05, 3.0))

    def _bandpass(self, x, f1, f2, fs_hz=None):
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        x = np.asarray(x, dtype=np.float64)
        ny = 0.5 * fs_use
        lo = max(20.0, float(f1))
        hi = min(float(f2), ny * 0.98)
        if hi <= lo * 1.05:
            return x.copy()
        sos = butter(4, [lo, hi], btype="band", fs=fs_use, output="sos")
        return sosfilt(sos, x)

    def _estimate_band_rt60_from_recording(self, x, band_centers=None, fs_hz=None):
        # Estimate octave-band RT60 prior from real recording.
        # Why: SE models benefit from frequency-dependent decay realism.
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        centers = self.band_centers_ref if band_centers is None else np.asarray(band_centers, dtype=np.float64)
        out = []
        for fc in centers:
            f1 = fc / np.sqrt(2.0)
            f2 = fc * np.sqrt(2.0)
            xb = self._bandpass(x, f1, f2, fs_hz=fs_use)
            est = self._estimate_rt60_from_recording(xb, fs_hz=fs_use)
            out.append(np.nan if est is None else float(est))
        out = np.asarray(out, dtype=np.float64)
        if np.all(~np.isfinite(out)):
            return None
        med = np.nanmedian(out)
        out = np.where(np.isfinite(out), out, med)
        return np.clip(out, 0.08, 3.0)

    def _estimate_band_edt_from_recording(self, x, band_centers=None, fs_hz=None):
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        centers = self.band_centers_ref if band_centers is None else np.asarray(band_centers, dtype=np.float64)
        out = []
        for fc in centers:
            f1 = fc / np.sqrt(2.0)
            f2 = fc * np.sqrt(2.0)
            xb = self._bandpass(x, f1, f2, fs_hz=fs_use)
            est = self._estimate_edt_from_impulse(xb, fs_hz=fs_use)
            out.append(np.nan if est is None else float(est))
        out = np.asarray(out, dtype=np.float64)
        if np.all(~np.isfinite(out)):
            return None
        finite = out[np.isfinite(out)]
        med = float(np.median(finite))
        out = np.where(np.isfinite(out), out, med)
        return np.clip(out, 0.05, 3.0)

    def _find_direct_path_idx(self, x, fs_hz=None, search_ms=30.0):
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return 0
        n_search = min(arr.size, max(16, int(round(float(search_ms) * 1e-3 * fs_use))))
        return int(np.argmax(np.abs(arr[:n_search])))

    def _estimate_early_echoes_from_impulse(
        self,
        x,
        fs_hz=None,
        max_early_ms=80.0,
        n_echoes=6,
        min_gap_ms=1.0,
        min_rel_db=-35.0,
        guard_ms=0.4,
    ):
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return None

        direct_idx = self._find_direct_path_idx(arr, fs_hz=fs_use)
        direct_amp = float(np.abs(arr[direct_idx]))
        if direct_amp <= 1e-12:
            return None

        start = min(arr.size, direct_idx + int(round(float(guard_ms) * 1e-3 * fs_use)))
        stop = min(arr.size, direct_idx + int(round(float(max_early_ms) * 1e-3 * fs_use)))
        if stop <= start:
            return {
                "direct_path_time_ms": float(1000.0 * direct_idx / fs_use),
                "early_echo_count_50ms": 0,
                "early_echo_count_80ms": 0,
                "echo_density_50ms": 0.0,
                "echoes": [],
            }

        region_abs = np.abs(arr)
        min_gap = max(1, int(round(float(min_gap_ms) * 1e-3 * fs_use)))
        chosen = []
        order = np.argsort(region_abs[start:stop])[::-1]
        for idx_rel in order:
            amp = float(region_abs[start + idx_rel])
            if amp <= 0.0:
                break
            rel_db = float(20.0 * np.log10(max(amp, 1e-12) / direct_amp))
            if rel_db < float(min_rel_db):
                break

            idx = int(start + idx_rel)
            if idx > 0 and amp < float(region_abs[idx - 1]):
                continue
            if idx + 1 < arr.size and amp < float(region_abs[idx + 1]):
                continue
            if any(abs(idx - p["sample_idx"]) < min_gap for p in chosen):
                continue

            chosen.append({
                "sample_idx": idx,
                "toa_ms": float(1000.0 * (idx - direct_idx) / fs_use),
                "rel_db": rel_db,
            })
            if len(chosen) >= int(n_echoes):
                break

        chosen = sorted(chosen, key=lambda p: p["sample_idx"])
        echoes = [{"rank": i + 1, "toa_ms": p["toa_ms"], "rel_db": p["rel_db"]} for i, p in enumerate(chosen)]
        count_50 = int(np.count_nonzero([p["toa_ms"] <= 50.0 for p in chosen]))
        count_80 = int(np.count_nonzero([p["toa_ms"] <= 80.0 for p in chosen]))
        return {
            "direct_path_time_ms": float(1000.0 * direct_idx / fs_use),
            "early_echo_count_50ms": count_50,
            "early_echo_count_80ms": count_80,
            "echo_density_50ms": float(count_50 / 0.05),
            "echoes": echoes,
        }

    @staticmethod
    def _aggregate_ranked_echoes(echo_lists, n_echoes):
        toa_med = []
        rel_db_med = []
        for rank in range(int(n_echoes)):
            rank_toa = []
            rank_rel = []
            for echoes in echo_lists:
                if len(echoes) <= rank:
                    continue
                rank_toa.append(float(echoes[rank]["toa_ms"]))
                rank_rel.append(float(echoes[rank]["rel_db"]))
            if len(rank_toa) == 0:
                break
            toa_med.append(float(np.median(np.asarray(rank_toa, dtype=np.float64))))
            rel_db_med.append(float(np.median(np.asarray(rank_rel, dtype=np.float64))))
        return toa_med, rel_db_med

    def analyze_reflection_structure_from_recordings(
        self,
        recordings,
        max_early_ms=80.0,
        n_echoes=6,
    ):
        items = self._resolve_audio_items(recordings, "recordings")
        ny_target = 0.48 * float(self.fs)
        fit_band_mask = self.band_centers_ref <= ny_target
        fit_band_centers = self.band_centers_ref[fit_band_mask]

        direct_times = []
        echo_count_50 = []
        echo_count_80 = []
        echo_density_50 = []
        echo_lists = []
        edt_band_vals = []
        n_used = 0
        warnings = []

        for item in items:
            x, fs_item, _ = self._load_audio_mono_keep_fs(item)
            if x.size < max(128, int(0.2 * fs_item)):
                continue

            ir_seg, _ = self._extract_impulse_segment(x, fs_hz=fs_item, pre_ms=3.0, tail_s=1.2)
            if not self._is_impulse_like(ir_seg, fs_hz=fs_item):
                continue

            echo_fit = self._estimate_early_echoes_from_impulse(
                ir_seg,
                fs_hz=fs_item,
                max_early_ms=float(max_early_ms),
                n_echoes=int(n_echoes),
            )
            if echo_fit is None:
                continue

            edt_partial = self._estimate_band_edt_from_recording(
                ir_seg,
                band_centers=fit_band_centers,
                fs_hz=fs_item,
            )
            if edt_partial is not None:
                edt_full = np.full_like(self.band_centers_ref, np.nan, dtype=np.float64)
                edt_full[fit_band_mask] = np.asarray(edt_partial, dtype=np.float64)
                edt_band_vals.append(edt_full)

            n_used += 1
            direct_times.append(float(echo_fit["direct_path_time_ms"]))
            echo_count_50.append(int(echo_fit["early_echo_count_50ms"]))
            echo_count_80.append(int(echo_fit["early_echo_count_80ms"]))
            echo_density_50.append(float(echo_fit["echo_density_50ms"]))
            echo_lists.append(list(echo_fit["echoes"]))

        if n_used == 0:
            warnings.append("No impulse-like recordings available for early-echo structure analysis.")
            return {
                "echo_analysis_version": 1,
                "echo_analysis_n_used": 0,
                "echo_analysis_max_early_ms": float(max_early_ms),
                "echo_analysis_n_echoes": int(n_echoes),
                "direct_path_time_ms_median": None,
                "early_echo_count_50ms_median": None,
                "early_echo_count_80ms_median": None,
                "echo_density_50ms_median": None,
                "early_echo_toa_ms_median": [],
                "early_echo_rel_db_median": [],
                "edt_band_median": None,
                "warnings": warnings,
            }

        edt_band_median = None
        if len(edt_band_vals) > 0:
            edt_mat = np.asarray(edt_band_vals, dtype=np.float64)
            med = np.full(edt_mat.shape[1], np.nan, dtype=np.float64)
            for i in range(edt_mat.shape[1]):
                col = edt_mat[:, i]
                col = col[np.isfinite(col)]
                if col.size > 0:
                    med[i] = float(np.median(col))
            valid_idx = np.where(np.isfinite(med))[0]
            if valid_idx.size > 0:
                all_idx = np.arange(med.size)
                med = np.interp(all_idx, valid_idx, med[valid_idx]).astype(np.float64)
                edt_band_median = med.tolist()

        toa_med, rel_db_med = self._aggregate_ranked_echoes(echo_lists, n_echoes=n_echoes)
        return {
            "echo_analysis_version": 1,
            "echo_analysis_n_used": int(n_used),
            "echo_analysis_max_early_ms": float(max_early_ms),
            "echo_analysis_n_echoes": int(n_echoes),
            "direct_path_time_ms_median": float(np.median(np.asarray(direct_times, dtype=np.float64))),
            "early_echo_count_50ms_median": int(np.median(np.asarray(echo_count_50, dtype=np.float64))),
            "early_echo_count_80ms_median": int(np.median(np.asarray(echo_count_80, dtype=np.float64))),
            "echo_density_50ms_median": float(np.median(np.asarray(echo_density_50, dtype=np.float64))),
            "early_echo_toa_ms_median": toa_med,
            "early_echo_rel_db_median": rel_db_med,
            "edt_band_median": edt_band_median,
            "warnings": warnings,
        }

    def _estimate_noise_stats(self, x, fs_hz=None):
        """
        Estimate noise statistics from low-energy frames:
        - noise_rms: amplitude level prior for additive noise
        - noise_tilt_db_per_oct: rough spectral tilt in dB/oct

        Why:
        During dataset synthesis, these priors help keep noise profile close to
        target room/device recordings instead of purely white synthetic noise.
        """
        fs_use = int(self.fs if fs_hz is None else fs_hz)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < max(64, int(0.1 * fs_use)):
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            return {"rms": rms, "tilt_db_per_oct": 0.0}

        frame = max(128, int(0.032 * fs_use))
        hop = max(64, int(0.016 * fs_use))
        if x.size < frame:
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            return {"rms": rms, "tilt_db_per_oct": 0.0}

        rms_list = []
        idxs = []
        for s in range(0, x.size - frame + 1, hop):
            seg = x[s:s + frame]
            rms_list.append(np.sqrt(np.mean(seg * seg) + 1e-12))
            idxs.append((s, s + frame))
        rms_arr = np.asarray(rms_list, dtype=np.float64)

        # Use low-energy frames as a proxy of background noise-dominant regions.
        q = np.percentile(rms_arr, 20.0)
        keep = np.where(rms_arr <= q)[0]
        if keep.size == 0:
            keep = np.arange(len(idxs))

        chunks = [x[idxs[i][0]:idxs[i][1]] for i in keep]
        noise_like = np.concatenate(chunks) if len(chunks) > 0 else x
        noise_rms = float(np.sqrt(np.mean(noise_like * noise_like) + 1e-12))

        if noise_like.size < 256:
            return {"rms": noise_rms, "tilt_db_per_oct": 0.0}

        w = np.hanning(noise_like.size)
        X = np.fft.rfft(noise_like * w)
        P = np.abs(X) ** 2 + 1e-18
        f = np.fft.rfftfreq(noise_like.size, d=1.0 / fs_use)
        m = (f >= 125.0) & (f <= min(6000.0, 0.48 * fs_use))
        if np.count_nonzero(m) < 8:
            return {"rms": noise_rms, "tilt_db_per_oct": 0.0}

        # Linear fit of PSD(dB) against log2(f): slope is dB per octave.
        xfit = np.log2(f[m] / 1000.0)
        yfit = 10.0 * np.log10(P[m])
        slope, _ = np.polyfit(xfit, yfit, 1)
        slope = float(np.clip(slope, -20.0, 20.0))
        return {"rms": noise_rms, "tilt_db_per_oct": slope}

    @staticmethod
    def _jitter_range(base_range, jitter_db, rng, min_width=1.0):
        a, b = float(base_range[0]), float(base_range[1])
        if b < a:
            a, b = b, a
        c = 0.5 * (a + b)
        h = 0.5 * (b - a)
        j = float(max(0.0, jitter_db))
        c = c + float(rng.uniform(-j, j))
        h = h * float(rng.uniform(0.95, 1.05))
        h = max(0.5 * float(min_width), h)
        return (float(c - h), float(c + h))

    def fit_from_recordings(
        self,
        recordings,
        room_size_hint=None,
        room_jitter_ratio=0.03,
        rt60_min_max=(0.12, 1.4),
        drr_prior_range_db=(-3.0, 8.0),
        c50_prior_range_db=(0.0, 14.0),
        drr_c50_jitter_db=0.6,
        drr_c50_mode="fixed",
        drr_c50_from_recording_jitter_db=0.2,
        fit_seed=0,
        update_generator=True,
    ):
        """
        Infer room/acoustic priors from real recordings and optionally write them back.

        Core blocks:
        1) RT60 estimation logic: `_estimate_rt60_from_recording`
        2) Band RT60 estimation: `_estimate_band_rt60_from_recording`
        3) Noise statistics: `_estimate_noise_stats`
        4) DRR/C50 mode:
           - `fixed`: always use prior + jitter,
           - `from_recording`: estimate from impulse-like recording segments,
           - `auto`: estimate when impulse-like, otherwise fallback to fixed prior.
        5) Sampling-rate policy for inversion:
           - use each recording's native wav fs (no pre-resampling),
           - if target fs (self.fs) is lower, estimate only bands within target Nyquist,
           - if target fs is higher than recording fs, raise error.
        6) Parameter write-back: `if update_generator`
        """
        items = self._resolve_audio_items(recordings, "recordings")

        rt60_vals = []
        band_rt60_vals = []
        drr_vals = []
        c50_vals = []
        noise_rms_vals = []
        noise_tilt_vals = []
        per_item = []
        fs_recordings = []

        drr_c50_mode = str(drr_c50_mode).lower()
        if drr_c50_mode not in ("fixed", "from_recording", "auto"):
            raise ValueError(f"Unsupported drr_c50_mode: {drr_c50_mode}")

        ny_target = 0.48 * float(self.fs)
        fit_band_mask = self.band_centers_ref <= ny_target
        fit_band_centers = self.band_centers_ref[fit_band_mask]
        if fit_band_centers.size == 0:
            raise RuntimeError(
                f"No usable fit bands for target fs={self.fs}. "
                f"Increase fs or lower band centers."
            )

        for item in items:
            x, fs_item, item_id = self._load_audio_mono_keep_fs(item)
            fs_recordings.append(int(fs_item))

            if self.fs > int(fs_item):
                raise ValueError(
                    f"Target fs ({self.fs}) is higher than recording fs ({fs_item}) for item: {item_id}. "
                    "Please lower target fs or provide higher-fs recordings."
                )

            if x.size < max(128, int(0.2 * fs_item)):
                per_item.append({
                    "item": item_id,
                    "fs_recording": int(fs_item),
                    "used": False,
                    "reason": "too_short",
                })
                continue

            ir_seg, _ = self._extract_impulse_segment(x, fs_hz=fs_item, pre_ms=3.0, tail_s=1.2)
            impulse_like = self._is_impulse_like(ir_seg, fs_hz=fs_item)
            if impulse_like:
                rt = self._estimate_rt60_from_impulse(ir_seg, fs_hz=fs_item)
                if rt is None:
                    rt = self._estimate_rt60_from_recording(ir_seg, fs_hz=fs_item)
            else:
                rt = self._estimate_rt60_from_recording(ir_seg, fs_hz=fs_item)
            band_rt_partial = self._estimate_band_rt60_from_recording(
                ir_seg,
                band_centers=fit_band_centers,
                fs_hz=fs_item,
            )

            if drr_c50_mode == "from_recording":
                # Force estimation from recording; require impulse-like capture.
                if not impulse_like:
                    raise ValueError(
                        f"drr_c50_mode='from_recording' requires impulse-like recordings. "
                        f"Item not impulse-like: {item_id}"
                    )
                drr, c50 = self._compute_drr_c50(ir_seg, fs_hz=fs_item)
                drr_from_recording = True
            elif drr_c50_mode == "auto":
                if impulse_like:
                    drr, c50 = self._compute_drr_c50(ir_seg, fs_hz=fs_item)
                    drr_from_recording = True
                else:
                    drr, c50 = np.nan, np.nan
                    drr_from_recording = False
            else:
                drr, c50 = np.nan, np.nan
                drr_from_recording = False

            noise_stats = self._estimate_noise_stats(x, fs_hz=fs_item)
            rms = float(noise_stats["rms"])
            tilt = float(noise_stats["tilt_db_per_oct"])

            used = rt is not None
            if used:
                rt60_vals.append(float(rt))
            if band_rt_partial is not None:
                band_full = np.full_like(self.band_centers_ref, np.nan, dtype=np.float64)
                band_full[fit_band_mask] = np.asarray(band_rt_partial, dtype=np.float64)
                band_rt60_vals.append(band_full)
            if np.isfinite(drr):
                drr_vals.append(float(drr))
            if np.isfinite(c50):
                c50_vals.append(float(c50))
            noise_rms_vals.append(rms)
            if np.isfinite(tilt):
                noise_tilt_vals.append(tilt)
            per_item.append({
                "item": item_id,
                "fs_recording": int(fs_item),
                "used": bool(used),
                "rt60": None if rt is None else float(rt),
                "drr_db": None if not np.isfinite(drr) else float(drr),
                "c50_db": None if not np.isfinite(c50) else float(c50),
                "drr_c50_from_recording": bool(drr_from_recording),
                "impulse_like": bool(impulse_like),
                "fit_band_centers_used": fit_band_centers.tolist(),
                "noise_rms": float(rms),
                "noise_tilt_db_per_oct": float(tilt),
            })

        if len(rt60_vals) == 0:
            raise RuntimeError("No valid RT60 estimates from recordings.")

        rt60_arr = np.asarray(rt60_vals, dtype=np.float64)
        if rt60_arr.size >= 5:
            # IQR filtering suppresses occasional gross over-estimates from noisy tails.
            q1, q3 = np.percentile(rt60_arr, [25.0, 75.0])
            iqr = max(float(q3 - q1), 1e-3)
            lo_o = float(q1 - 1.5 * iqr)
            hi_o = float(q3 + 1.5 * iqr)
            keep = (rt60_arr >= lo_o) & (rt60_arr <= hi_o)
            if int(np.count_nonzero(keep)) >= max(3, int(0.6 * rt60_arr.size)):
                rt60_arr = rt60_arr[keep]

        lo = float(min(rt60_min_max))
        hi = float(max(rt60_min_max))
        if room_size_hint is not None:
            rs = np.asarray(room_size_hint, dtype=np.float64).reshape(3)
            V_hint = float(max(1.0, np.prod(rs)))
            # Volume-aware upper cap to prevent non-physical RT60 for small rooms.
            if V_hint <= 25.0:
                hi = min(hi, 0.60)
            elif V_hint <= 40.0:
                hi = min(hi, 0.75)
            elif V_hint <= 70.0:
                hi = min(hi, 0.95)
        rt50 = float(np.clip(np.percentile(rt60_arr, 50), lo, hi))
        rt20 = float(np.clip(np.percentile(rt60_arr, 20), lo, hi))
        rt80 = float(np.clip(np.percentile(rt60_arr, 80), lo, hi))
        if rt80 < rt20 + 0.02:
            rt80 = min(hi, rt20 + 0.02)

        band_prior = None
        if len(band_rt60_vals) > 0:
            band_mat = np.asarray(band_rt60_vals, dtype=np.float64)
            if np.any(np.isfinite(band_mat)):
                med = np.full(band_mat.shape[1], np.nan, dtype=np.float64)
                for i in range(band_mat.shape[1]):
                    col = band_mat[:, i]
                    col = col[np.isfinite(col)]
                    if col.size > 0:
                        med[i] = float(np.median(col))
                band_prior = med
            valid_idx = np.where(np.isfinite(band_prior))[0] if band_prior is not None else np.array([], dtype=int)
            if valid_idx.size > 0:
                # Fill missing bands by nearest valid estimate (typically high bands
                # above target Nyquist) to keep vector length compatible downstream.
                all_idx = np.arange(band_prior.size)
                band_prior = np.interp(all_idx, valid_idx, band_prior[valid_idx])

                # Smooth in log-domain and clip.
                band_prior = self._smooth_curve(np.log(np.clip(band_prior, lo, hi)), passes=2)
                band_prior = np.exp(band_prior)

                # Enforce non-increasing high-frequency RT60 to avoid non-physical
                # "high-freq tail longer than low-freq" artifacts.
                if band_prior.size > 1:
                    start_i = int(np.searchsorted(self.band_centers_ref, 1000.0))
                    start_i = int(np.clip(start_i, 1, band_prior.size - 1))
                    for i in range(start_i, band_prior.size):
                        band_prior[i] = min(band_prior[i], band_prior[i - 1])
            else:
                band_prior = None

        rng_fit = np.random.default_rng(int(fit_seed) if fit_seed is not None else 0)
        strategy = "fixed_smallroom_prior_jitter"
        use_recording_drr_c50 = (drr_c50_mode in ("from_recording", "auto")) and (len(drr_vals) > 0) and (len(c50_vals) > 0)
        if use_recording_drr_c50:
            drr_q = np.percentile(np.asarray(drr_vals, dtype=np.float64), [20, 80])
            c50_q = np.percentile(np.asarray(c50_vals, dtype=np.float64), [20, 80])
            drr_range = (float(drr_q[0]), float(drr_q[1]))
            c50_range = (float(c50_q[0]), float(c50_q[1]))

            # Optional small jitter on estimated range to keep dataset diversity.
            j2 = float(max(0.0, drr_c50_from_recording_jitter_db))
            if j2 > 0.0:
                drr_range = self._jitter_range(drr_range, j2, rng_fit, min_width=1.0)
                c50_range = self._jitter_range(c50_range, j2, rng_fit, min_width=1.0)

            strategy = "from_recordings" if drr_c50_mode == "from_recording" else "auto_from_recordings"
        else:
            drr_range = self._jitter_range(drr_prior_range_db, drr_c50_jitter_db, rng_fit, min_width=1.0)
            c50_range = self._jitter_range(c50_prior_range_db, drr_c50_jitter_db, rng_fit, min_width=1.0)
            if drr_c50_mode == "auto":
                strategy = "auto_fallback_fixed_smallroom_prior_jitter"

        drr_range = (
            float(np.clip(min(drr_range), -8.0, 14.0)),
            float(np.clip(max(drr_range), -8.0, 14.0)),
        )
        c50_range = (
            float(np.clip(min(c50_range), -2.0, 20.0)),
            float(np.clip(max(c50_range), -2.0, 20.0)),
        )

        n_used_items = int(np.count_nonzero([bool(p.get("used", False)) for p in per_item]))
        n_from_recording = int(np.count_nonzero([bool(p.get("drr_c50_from_recording", False)) for p in per_item]))
        if drr_c50_mode == "fixed":
            fallback_n = 0
        else:
            fallback_n = max(0, n_used_items - n_from_recording)
        warnings = []
        if drr_c50_mode == "auto" and n_from_recording == 0:
            warnings.append(
                "No impulse-like recordings detected for DRR/C50 inversion; auto mode fell back to fixed prior."
            )

        fit = {
            "n_input": len(items),
            "n_used_rt60": int(rt60_arr.size),
            "n_used_rt60_before_filter": int(len(rt60_vals)),
            "target_fs": int(self.fs),
            "recording_fs_min_max": [int(min(fs_recordings)), int(max(fs_recordings))] if len(fs_recordings) > 0 else None,
            "rt60_median": rt50,
            "rt60_p20": rt20,
            "rt60_p80": rt80,
            "rt60_fit_bounds_used": [float(lo), float(hi)],
            "rt60_band_median": None if band_prior is None else band_prior.tolist(),
            "band_centers_ref": self.band_centers_ref.tolist(),
            "band_centers_used_for_fit": fit_band_centers.tolist(),
            "drr_db_p20_p80": [float(drr_range[0]), float(drr_range[1])],
            "c50_db_p20_p80": [float(c50_range[0]), float(c50_range[1])],
            "drr_c50_mode": drr_c50_mode,
            "drr_c50_strategy": strategy,
            "drr_c50_n_from_recording": {
                "drr": int(len(drr_vals)),
                "c50": int(len(c50_vals)),
            },
            "drr_c50_used_from_recording": bool(use_recording_drr_c50),
            "drr_c50_n_items_from_recording": int(n_from_recording),
            "drr_c50_n_items_fallback_prior": int(fallback_n),
            "drr_c50_from_recording_ratio": float(n_from_recording / max(1, n_used_items)),
            "drr_c50_prior_base": {
                "drr_range_db": [float(drr_prior_range_db[0]), float(drr_prior_range_db[1])],
                "c50_range_db": [float(c50_prior_range_db[0]), float(c50_prior_range_db[1])],
                "jitter_db": float(drr_c50_jitter_db),
                "from_recording_jitter_db": float(drr_c50_from_recording_jitter_db),
                "fit_seed": None if fit_seed is None else int(fit_seed),
            },
            "warnings": warnings,
            "noise_rms_median": float(np.median(np.asarray(noise_rms_vals, dtype=np.float64))),
            "noise_tilt_db_per_oct_median": float(np.median(np.asarray(noise_tilt_vals, dtype=np.float64))) if len(noise_tilt_vals) > 0 else 0.0,
            "per_item": per_item,
        }

        # Write inferred priors back to generator so subsequent generate
        # uses room-specific distributions instead of generic defaults.
        if update_generator:
            self.custom_rt60_range = (rt20, rt80)
            self.custom_rt60_center = rt50
            self.custom_band_rt60_prior = band_prior
            self.drr_range_db = drr_range
            self.c50_range_db = c50_range
            self.custom_noise_rms = fit["noise_rms_median"]
            self.custom_noise_tilt_db_oct = fit["noise_tilt_db_per_oct_median"]
            self.fitted = fit
        return fit

    def _sample_branch(self, rng):
        return "generic" if float(rng.uniform()) < self.generic_mix_prob else "custom"

    def _sample_scalar(self, rng, val_range):
        a, b = float(val_range[0]), float(val_range[1])
        if b < a:
            a, b = b, a
        return float(rng.uniform(a, b))

    def generate(
        self,
        clean,
        seed=0,
        return_ref=True,
        ref_direct=True,
        branch=None,
        normalize_output=False,
        apply_drr_c50=True,
    ):
        clean = np.asarray(clean, dtype=np.float64)
        if clean.ndim == 2:
            clean = clean[:, 0]
        if clean.ndim != 1:
            raise ValueError("clean must be 1-D waveform or 2-D [n,c]")

        rng = np.random.default_rng(int(seed))
        # Branch mixing strategy:
        # custom branch keeps target-room alignment,
        # generic branch keeps domain diversity and reduces overfitting.
        mode = branch if branch in ("custom", "generic") else self._sample_branch(rng)

        if mode == "custom":
            room_range = self.custom_room_range
            rt60_range = self.custom_rt60_range
            band_prior = self.custom_band_rt60_prior
        else:
            room_range = self.generic_room_range
            rt60_range = self.generic_rt60_range
            band_prior = None

        room_size = self._sample_room_size(room_range, rng)
        doa = self._sample_scalar(rng, self.doa_range)
        rt60_tgt = self._sample_scalar(rng, rt60_range)
        if apply_drr_c50:
            drr_tgt = self._sample_scalar(rng, self.drr_range_db)
            c50_tgt = self._sample_scalar(rng, self.c50_range_db)
        else:
            drr_tgt, c50_tgt = None, None

        mic_loc = self._get_mic_array_loc(room_size, rng, min_dis_to_wall=0.6)
        src_loc, src_dist = self._get_source_loc(room_size, mic_loc, doa, rng, min_dis_to_wall=0.5)

        params = sample_room_params(
            float(room_size[0]),
            float(room_size[1]),
            float(room_size[2]),
            fs=int(self.fs),
            rng=rng,
            rt60_target=float(rt60_tgt),
            material_library=getattr(self, "material_library", None),
            face_category_groups=getattr(self, "material_face_category_groups", None),
            sound_speed_m_s=float(getattr(self, "sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S)),
        )
        # Per-sample band profile randomization for SE robustness.
        fc = self._jitter_band_centers(rng)
        band_rt60 = self._sample_band_rt60(rt60_tgt, rng, band_prior=band_prior)
        params = self._apply_band_profile_to_params(params, room_size, fc, band_rt60, rng)
        # Modal tail controls are runtime-configurable via cfg and injected into
        # engine params to avoid hard-coded core behavior.
        if isinstance(params, dict):
            n_min = int(getattr(self, "mode_n_min", 3))
            n_max = int(getattr(self, "mode_n_max", 8))
            if n_max < n_min:
                n_min, n_max = n_max, n_min
            rel_lo = float(getattr(self, "mode_rel_db_min", -38.0))
            rel_hi = float(getattr(self, "mode_rel_db_max", -30.0))
            if rel_hi < rel_lo:
                rel_lo, rel_hi = rel_hi, rel_lo
            params["sound_speed_m_s"] = float(getattr(self, "sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S))
            params["late_tail_highpass_hz"] = float(getattr(self, "late_tail_highpass_hz", DEFAULT_LATE_TAIL_HIGHPASS_HZ))
            params["late_reverb_bandwidth_oct"] = float(getattr(self, "late_reverb_bandwidth_oct", DEFAULT_LATE_REVERB_BANDWIDTH_OCT))
            params["late_reverb_break_fractions"] = [float(v) for v in getattr(self, "late_reverb_break_fractions", DEFAULT_LATE_REVERB_BREAK_FRACTIONS)]
            params["late_reverb_density_scale"] = float(getattr(self, "late_reverb_density_scale", DEFAULT_LATE_REVERB_DENSITY_SCALE))
            params["late_reverb_slope_scales"] = [float(v) for v in getattr(self, "late_reverb_slope_scales", DEFAULT_LATE_REVERB_SLOPE_SCALES)]
            params["source_directivity_strength"] = float(getattr(self, "source_directivity_strength", DEFAULT_SOURCE_DIRECTIVITY_STRENGTH))
            params["source_head_shadow_strength"] = float(getattr(self, "source_head_shadow_strength", DEFAULT_SOURCE_HEAD_SHADOW_STRENGTH))
            params["source_torso_scattering_strength"] = float(getattr(self, "source_torso_scattering_strength", DEFAULT_SOURCE_TORSO_SCATTERING_STRENGTH))
            params["source_head_radius_m"] = float(getattr(self, "source_head_radius_m", DEFAULT_SOURCE_HEAD_RADIUS_M))
            params["source_torso_radius_m"] = float(getattr(self, "source_torso_radius_m", DEFAULT_SOURCE_TORSO_RADIUS_M))
            params["source_directivity_bandwidth_oct"] = float(getattr(self, "source_directivity_bandwidth_oct", DEFAULT_SOURCE_DIRECTIVITY_BANDWIDTH_OCT))
            params["mode_fmin_hz"] = float(getattr(self, "mode_fmin_hz", DEFAULT_MODE_FMIN_HZ))
            params["mode_fmax_hz"] = float(getattr(self, "mode_fmax_hz", DEFAULT_MODE_FMAX_HZ))
            params["mode_n_range"] = [int(n_min), int(n_max)]
            params["mode_rel_db_range"] = [float(rel_lo), float(rel_hi)]

        n = clean.shape[0]
        n_ch = mic_loc.shape[1]
        y = np.zeros((n_ch, n), dtype=np.float64)
        rirs = []

        rt60_real = None
        drr_real = None
        c50_real = None
        physical_scales = []
        physical_target_direct_peaks = []
        drr_c50_shape_trace = {"enabled": False}
        raw_rirs = []
        rt60_out_ref = None

        for ch in range(n_ch):
            out = simulate_rir_with_params(
                mic_xyz=mic_loc[:, ch],
                src_xyz=src_loc,
                angle_offset=float(doa),
                lx=float(room_size[0]),
                ly=float(room_size[1]),
                lz=float(room_size[2]),
                fs=int(self.fs),
                params=params,
                rng=rng,
                sound_speed_m_s=float(getattr(self, "sound_speed_m_s", DEFAULT_SOUND_SPEED_M_S)),
            )
            if isinstance(out, tuple):
                rir = out[0]
                rt60_out = out[1] if len(out) > 1 else None
            else:
                rir, rt60_out = out, None
            rir = np.asarray(rir, dtype=np.float64).reshape(-1)
            raw_rirs.append(np.asarray(rir, dtype=np.float64))
            if ch == 0:
                rt60_out_ref = rt60_out

        if apply_drr_c50:
            shaped_rirs, drr_c50_shape_trace = self._apply_drr_c50_target_multich(
                raw_rirs,
                drr_tgt_db=drr_tgt,
                c50_tgt_db=c50_tgt,
                ref_ch=0,
            )
        else:
            shaped_rirs = [np.asarray(r, dtype=np.float64).copy() for r in raw_rirs]

        for ch in range(n_ch):
            rir = shaped_rirs[ch]
            if self.enable_physical_calibration:
                rir, g_phy, tgt_dp = self._apply_physical_calibration(rir, src_xyz=src_loc, mic_xyz=mic_loc[:, ch])
            else:
                g_phy, tgt_dp = 1.0, 0.0
            physical_scales.append(float(g_phy))
            physical_target_direct_peaks.append(float(tgt_dp))
            rirs.append(rir)
            y[ch] = fftconvolve(clean, rir)[:n]
            if ch == 0:
                drr_real, c50_real = self._compute_drr_c50(rir)
                rt60_real = float(rt60_out_ref) if rt60_out_ref is not None else None

        ref = None
        if return_ref:
            if ref_direct:
                # Multi-channel direct reference: one direct-path signal per mic.
                ref = np.zeros((n_ch, n), dtype=np.float64)
                for ch in range(n_ch):
                    ref_rir = self._direct_ref(src_loc, mic_loc[:, ch], n)
                    ref[ch] = fftconvolve(clean, ref_rir)[:n]
            else:
                early_n = int(0.03 * self.fs)
                # Multi-channel early reference (<=30ms): keep mic-dependent early part.
                ref = np.zeros((n_ch, n), dtype=np.float64)
                for ch in range(n_ch):
                    r0 = rirs[ch]
                    early = np.zeros_like(r0)
                    L = min(early_n, len(r0))
                    early[:L] = r0[:L]
                    ref[ch] = fftconvolve(clean, early)[:n]

        norm_gain = 1.0
        mix_peak_before_norm = float(np.max(np.abs(y))) if y.size > 0 else 0.0
        if normalize_output:
            y, _, ref, norm_gain, mix_peak_before_norm = self._final_peak_normalize_triplet(y, clean=None, ref=ref)

        params_trace = {}
        if isinstance(params, dict):
            if "max_order" in params:
                params_trace["max_order"] = int(params["max_order"])
            if "room_dim" in params:
                try:
                    params_trace["room_dim"] = [float(v) for v in params["room_dim"]]
                except Exception:
                    pass
            if "center_freqs" in params:
                try:
                    params_trace["center_freqs"] = np.asarray(params["center_freqs"], dtype=np.float64).tolist()
                except Exception:
                    pass
            if "RT60_target" in params:
                try:
                    params_trace["rt60_param_target"] = float(params["RT60_target"])
                except Exception:
                    pass
            if "_trace_last" in params:
                try:
                    params_trace["engine_trace"] = params["_trace_last"]
                except Exception:
                    pass
            if "material_trace" in params:
                try:
                    params_trace["material_trace"] = params["material_trace"]
                except Exception:
                    pass
            if "face_categories" in params:
                try:
                    params_trace["face_categories"] = params["face_categories"]
                except Exception:
                    pass

        meta = {
            "sample_seed": int(seed),
            "mode": mode,
            "fs": int(self.fs),
            "room_size": [float(v) for v in room_size],
            "doa_deg": float(doa),
            "src_dist": float(src_dist),
            "rt60_target": float(rt60_tgt),
            "rt60_real": None if rt60_real is None else float(rt60_real),
            "drr_target_db": None if drr_tgt is None else float(drr_tgt),
            "drr_real_db": None if drr_real is None else float(drr_real),
            "c50_target_db": None if c50_tgt is None else float(c50_tgt),
            "c50_real_db": None if c50_real is None else float(c50_real),
            "drr_c50_applied": bool(apply_drr_c50),
            "band_centers": fc.tolist(),
            "band_rt60": band_rt60.tolist(),
            "mic_loc": mic_loc,
            "src_loc": src_loc,
            "ref_channels": int(n_ch) if (return_ref and ref is not None and ref.ndim == 2) else 1,
            "physical_calibration_enabled": bool(self.enable_physical_calibration),
            "physical_scales": physical_scales,
            "target_direct_peaks": physical_target_direct_peaks,
            "final_norm_applied": bool(normalize_output),
            "final_norm_gain": float(norm_gain),
            "mix_peak_before_norm": float(mix_peak_before_norm),
            "params_trace": params_trace,
            "drr_c50_shape_trace": drr_c50_shape_trace,
        }
        return y, ref, meta


def _room_range_from_hint(room_size_hint, jitter_ratio):
    room_size_hint = np.asarray(room_size_hint, dtype=np.float64).reshape(3)
    j = max(0.0, float(jitter_ratio))
    lx, ly, lz = room_size_hint.tolist()
    return {
        "lx": (max(1.5, lx * (1.0 - j)), max(1.55, lx * (1.0 + j))),
        "ly": (max(1.5, ly * (1.0 - j)), max(1.55, ly * (1.0 + j))),
        "lz": (max(2.0, lz * (1.0 - j)), max(2.05, lz * (1.0 + j))),
    }

__all__ = [
    "BaseEngine",
    "_room_range_from_hint",
    "sample_room_params",
    "simulate_rir_with_params",
]

