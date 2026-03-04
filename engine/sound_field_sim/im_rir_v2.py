import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import Cardioid, DirectionVector
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt


C = 343

# Low-frequency modal augmentation

# Low-frequency mode synthesis for late tail shaping.

def add_low_freq_modes(
    tail,
    fs,
    room_dim,
    rt60,
    fmin=40,
    fmax=200,
    n_modes_range=(3, 8),
    rel_db_range=(-25, -15),
    c=343.0,
    rng=None,
    return_meta=False,
):
    """
    Add low-frequency modes in [fmin, fmax] to improve late-tail realism.
    room_dim: (lx, ly, lz)
    rt60: RT60 target
    """
    lx, ly, lz = room_dim
    N = len(tail)
    t = np.arange(N) / fs

    # Candidate axial modal frequencies.
    cand = []
    for L in (lx, ly, lz):
        n_max = int(np.floor(2 * fmax * L / c))
        for n in range(1, max(2, n_max + 1)):
            f = (c / 2.0) * (n / L)
            if fmin <= f <= fmax:
                cand.append(f)

    cand = np.array(sorted(set(cand)))
    if len(cand) == 0:
        meta = {
            "mode_freqs_hz": [],
            "mode_taus_s": [],
            "mode_rel_db": None,
        }
        return (tail, meta) if return_meta else tail

    # Sample a subset of modes and apply small frequency jitter.
    rng = np.random.default_rng(0) if rng is None else rng

    K = rng.integers(n_modes_range[0], n_modes_range[1] + 1)
    K = min(K, len(cand))
    fk = rng.choice(cand, size=K, replace=False)
    fk = fk * rng.uniform(0.98, 1.02, size=K)

    # Synthesize damped modal sinusoids.
    modes = np.zeros_like(tail, dtype=np.float64)
    mode_taus = []
    for f in fk:
        phi = rng.uniform(0, 2*np.pi)
        # Damping time constant.
        tau = rt60 * rng.uniform(0.4, 1.2)
        # Slightly slower decay for lower frequencies.
        tau *= (120.0 / max(f, 60.0))**0.2
        mode_taus.append(float(tau))
        a = np.exp(-t / max(tau, 1e-3))
        modes += a * np.sin(2*np.pi*f*t + phi)

    # Use tail RMS as reference to set modal strength.
    rms_tail = np.sqrt(np.mean(tail**2) + 1e-12)

    # normalize
    rms_modes = np.sqrt(np.mean(modes**2) + 1e-12)
    modes /= rms_modes

    rel_db = rng.uniform(rel_db_range[0], rel_db_range[1])
    target_rms_modes = rms_tail * (10.0 ** (rel_db / 20.0))
    modes *= target_rms_modes

    out = tail + modes
    meta = {
        "mode_freqs_hz": [float(v) for v in np.asarray(fk, dtype=np.float64).tolist()],
        "mode_taus_s": mode_taus,
        "mode_rel_db": float(rel_db),
    }
    return (out, meta) if return_meta else out

def generate_velvet_noise(length, fs, density=2000, rng=None):
    # Sparse pulse noise for low-coherence late tail synthesis.
    rng = np.random.default_rng(0) if rng is None else rng
    length = int(max(0, length))
    fs = float(fs)
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
    # Remove very low-frequency rumble.
    sos = butter(4, cutoff, 'hp', fs=fs, output='sos')
    return sosfilt(sos, sig)


def _tail_decay_shape_from_alpha(alpha_f):
    """
    Build late-tail spectral shape from wall absorption curve.

    We use sqrt(1 - alpha(f)) and normalize its mean:
    - higher absorption -> weaker tail energy in that band,
    - sqrt keeps suppression stable without over-damping.
    """
    a = np.clip(np.asarray(alpha_f, dtype=np.float64), 0.02, 0.99)
    shape = np.sqrt(np.clip(1.0 - a, 1e-4, 1.0))
    shape = shape / max(float(np.mean(shape)), 1e-8)
    return shape


class SoftCardioid(Cardioid):
    # pyroom source directivity wrapper
    def __init__(self, orientation, alpha=0.3, gain=1.0):
        super().__init__(orientation, gain=gain)
        self.alpha = alpha

    def evaluate(self, direction):
        base = super().evaluate(direction)
        return self.alpha + (1 - self.alpha) * base


MATERIAL_LIBRARY = {
    "painted_wall": {
        "absorption": [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
        "scattering": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
    },
    "gypsum_board": {
        "absorption": [0.10, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05],
        "scattering": [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
    },
    "concrete": {
        "absorption": [0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02],
        "scattering": [0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
    },
    "glass": {
        "absorption": [0.03, 0.03, 0.03, 0.04, 0.05, 0.06, 0.06],
        "scattering": [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
    },
    "curtain_heavy": {
        "absorption": [0.15, 0.25, 0.40, 0.55, 0.65, 0.70, 0.72],
        "scattering": [0.12, 0.14, 0.16, 0.20, 0.24, 0.30, 0.36],
    },
    "carpet_floor": {
        "absorption": [0.08, 0.12, 0.20, 0.30, 0.40, 0.50, 0.55],
        "scattering": [0.10, 0.14, 0.18, 0.22, 0.26, 0.32, 0.36],
    },
    "wood_floor": {
        "absorption": [0.05, 0.06, 0.08, 0.10, 0.11, 0.12, 0.12],
        "scattering": [0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28],
    },
    "acoustic_tile_ceiling": {
        "absorption": [0.30, 0.45, 0.65, 0.75, 0.75, 0.70, 0.65],
        "scattering": [0.18, 0.22, 0.28, 0.34, 0.38, 0.40, 0.42],
    },
    "plaster_ceiling": {
        "absorption": [0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        "scattering": [0.10, 0.12, 0.14, 0.18, 0.22, 0.26, 0.30],
    },
}


def _weighted_scattering_scalar(scattering_curve, center_freqs):
    s = np.asarray(scattering_curve, dtype=np.float64)
    f = np.asarray(center_freqs, dtype=np.float64)
    w = np.sqrt(np.maximum(f, 1.0) / np.maximum(np.min(f), 1.0))
    return float(np.clip(np.average(s, weights=w), 0.05, 0.95))


def _sample_face_categories(rng):
    wall_candidates = ["painted_wall", "gypsum_board", "concrete", "glass", "curtain_heavy"]
    floor_candidates = ["carpet_floor", "wood_floor"]
    ceil_candidates = ["acoustic_tile_ceiling", "plaster_ceiling"]
    return {
        "west": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "east": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "south": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "north": wall_candidates[int(rng.integers(0, len(wall_candidates)))],
        "floor": floor_candidates[int(rng.integers(0, len(floor_candidates)))],
        "ceiling": ceil_candidates[int(rng.integers(0, len(ceil_candidates)))],
    }


def _build_materials_from_library(center_freqs, alpha_mean, face_categories, rng):
    materials = {}
    trace = {}
    coeff_stack = []
    for face, cat in face_categories.items():
        base = MATERIAL_LIBRARY[cat]
        abs_base = np.asarray(base["absorption"], dtype=np.float64)
        scat_base = np.asarray(base["scattering"], dtype=np.float64)

        shape = abs_base / max(float(np.mean(abs_base)), 1e-6)
        spectral_jitter = rng.uniform(0.92, 1.08, size=abs_base.shape[0])
        face_scale = float(rng.uniform(0.90, 1.10))
        coeffs = np.clip(alpha_mean * shape * spectral_jitter * face_scale, 0.01, 0.98)
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

    alpha_bar = np.mean(np.stack(coeff_stack, axis=0), axis=0)
    return materials, trace, alpha_bar


def _sample_common_room_params(lx, ly, lz, fs, rng, rt60_target):
    center_freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float64)
    room_dim = [float(lx), float(ly), float(lz)]
    rt60_value = float(rng.uniform(0.1, 1.0) if rt60_target is None else rt60_target)

    V = float(lx * ly * lz)
    S_total = float(2.0 * (lx * ly + lx * lz + ly * lz))
    alpha_mean = float(np.clip(0.161 * V / max(S_total * rt60_value, 1e-6), 0.03, 0.75))

    face_categories = _sample_face_categories(rng)
    materials, material_trace, alpha_bar = _build_materials_from_library(
        center_freqs=center_freqs,
        alpha_mean=alpha_mean,
        face_categories=face_categories,
        rng=rng,
    )

    log_fc = np.log(center_freqs)
    alpha_continuous = interp1d(log_fc, alpha_bar, kind="linear", fill_value="extrapolate")

    L_min = min(room_dim)
    t_er = float(rng.uniform(0.06, 0.12))
    max_order = int(np.ceil(C * t_er / max(L_min, 1e-6)))
    max_order = int(np.clip(max_order, 5, 40))

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


def sample_room_params(lx, ly, lz, fs=32000, rng=None, rt60_target=None):
    """Sample (or set) room-level parameters once, then reuse across multiple mics.

    Returns a dict that can be fed into ``simulate_rir_with_params``.
    """
    rng = np.random.default_rng(0) if rng is None else rng

    p = _sample_common_room_params(
        lx=lx,
        ly=ly,
        lz=lz,
        fs=fs,
        rng=rng,
        rt60_target=rt60_target,
    )
    return {
        "room_dim": p["room_dim"],
        "RT60_target": p["RT60_target"],
        "center_freqs": p["center_freqs"],
        "alpha_continuous": p["alpha_continuous"],
        "materials": p["materials"],
        "material_trace": p["material_trace"],
        "face_categories": p["face_categories"],
        "max_order": p["max_order"],
    }


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
):
    """Generate a single-channel RIR using shared room params + per-mic randomness."""
    rng = np.random.default_rng(0) if rng is None else rng

    room_dim = [lx, ly, lz]
    RT60_target = params['RT60_target']
    alpha_continuous = params['alpha_continuous']

    # early (pyroom ISM)
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=params['materials'],
        max_order=int(params['max_order']),
        use_rand_ism=True,
        air_absorption=True,
    )

    azimuth = np.deg2rad(angle_offset)
    orientation = DirectionVector(azimuth, np.pi / 2, degrees=False)
    directivity = SoftCardioid(orientation, alpha=0.7)

    room.add_source(list(src_xyz), signal=None, directivity=directivity)
    room.add_microphone_array(np.array(mic_xyz, dtype=np.float64).reshape(3, 1))
    room.compute_rir()
    rir_ism = np.asarray(room.rir[0][0], dtype=np.float64)

    # split point (Schroeder frequency heuristic)
    V = lx * ly * lz
    f_sch = 2000.0 * np.sqrt(max(RT60_target, 1e-3) / max(V, 1e-3))
    t_split = np.clip(3.0 / max(f_sch, 50.0), 0.05, 0.12)
    split_idx = int(t_split * fs)

    fade_len = int(0.02 * fs)
    split_idx = int(np.clip(split_idx, fade_len + 1, len(rir_ism)))
    early = rir_ism[:split_idx]

    # direct / early jitter
    jitter = int(rng.uniform(0.2e-3, 0.8e-3) * fs)
    if jitter >= 2:
        early = np.convolve(early, rng.standard_normal(jitter) * 0.05, mode='same')
    E = np.fft.rfft(early)
    phase_jitter = np.exp(1j * rng.uniform(-0.1, 0.1, size=E.shape))
    early = np.fft.irfft(E * phase_jitter, n=len(early))

    # tail
    tail_len = max(int(RT60_target * fs * 1.1), len(rir_ism) - split_idx)
    noise_density = float(rng.uniform(8000, max(8001.0, 0.98 * fs)))
    noise = generate_velvet_noise(
        tail_len,
        fs,
        density=noise_density,
        rng=rng,
    )

    Noise_f = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / fs)
    log_f = np.log(np.clip(freqs, 50, fs / 2))

    alpha_f = np.clip(alpha_continuous(log_f), 0.02, 0.99)
    decay_shape = _tail_decay_shape_from_alpha(alpha_f)

    time_decay = np.exp(-6.9 * np.arange(len(noise)) / (RT60_target * fs))
    Noise_f *= decay_shape
    tail = np.fft.irfft(Noise_f, n=len(noise))
    tail *= time_decay
    tail = apply_highpass(tail, fs, cutoff=40)

    mode_fmin = float(params.get("mode_fmin_hz", 40.0)) if isinstance(params, dict) else 40.0
    mode_fmax = float(params.get("mode_fmax_hz", 800.0)) if isinstance(params, dict) else 800.0
    mode_n_range = params.get("mode_n_range", [3, 8]) if isinstance(params, dict) else [3, 8]
    mode_rel_db_range = params.get("mode_rel_db_range", [-38.0, -30.0]) if isinstance(params, dict) else [-38.0, -30.0]
    if not isinstance(mode_n_range, (list, tuple)) or len(mode_n_range) < 2:
        mode_n_range = [3, 8]
    if not isinstance(mode_rel_db_range, (list, tuple)) or len(mode_rel_db_range) < 2:
        mode_rel_db_range = [-38.0, -30.0]
    n0, n1 = int(mode_n_range[0]), int(mode_n_range[1])
    if n1 < n0:
        n0, n1 = n1, n0
    r0, r1 = float(mode_rel_db_range[0]), float(mode_rel_db_range[1])
    if r1 < r0:
        r0, r1 = r1, r0

    tail, mode_meta = add_low_freq_modes(
        tail,
        fs,
        room_dim=(lx, ly, lz),
        rt60=RT60_target,
        fmin=float(mode_fmin),
        fmax=float(mode_fmax),
        n_modes_range=(int(n0), int(n1)),
        rel_db_range=(float(r0), float(r1)),
        rng=rng,
        return_meta=True,
    )

    # cross-fade
    w = np.linspace(0, 1, fade_len, endpoint=False)
    a0 = split_idx - fade_len

    rms_e = np.sqrt(np.mean(early[a0:split_idx]**2) + 1e-12)
    rms_t = np.sqrt(np.mean(tail[:fade_len]**2) + 1e-12)
    tail *= (rms_e / rms_t)

    head = early[:a0]
    xfade = early[a0:split_idx] * (1 - w) + tail[:fade_len] * w
    rest = tail[fade_len:]
    rir = np.concatenate([head, xfade, rest])

    # No per-RIR peak normalization here.
    # Keep raw physical amplitude so upstream physical calibration / DRR / C50
    # semantics are not polluted by hidden local rescaling.

    if isinstance(params, dict):
        params["_trace_last"] = {
            "engine_variant": "vector",
            "split_idx": int(split_idx),
            "split_time_ms": float(1000.0 * split_idx / fs),
            "fade_len": int(fade_len),
            "tail_len": int(tail_len),
            "noise_density": float(noise_density),
            "max_order": int(params.get("max_order", -1)),
            "mode_meta": mode_meta,
        }

    return rir, RT60_target


