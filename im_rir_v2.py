import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, iirfilter
from scipy.io import wavfile
import soundfile as sf
import pyroomacoustics as pra
import time
from pyroomacoustics.directivities import Cardioid, DirectionVector
import pandas as pd
from scipy.fft import fft, ifft, next_fast_len
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample, fftconvolve
import os
import random
import gc
import re
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt


C = 343

# 低频模态的叠加行为

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
):
    """
    40Hz以下部分切除
    room_dim: (lx, ly, lz)
    rt60: RT60_target
    """
    lx, ly, lz = room_dim
    N = len(tail)
    t = np.arange(N) / fs

    # 模态候选频率
    cand = []
    for L in (lx, ly, lz):
        n_max = int(np.floor(2 * fmax * L / c))
        for n in range(1, max(2, n_max + 1)):
            f = (c / 2.0) * (n / L)
            if fmin <= f <= fmax:
                cand.append(f)

    cand = np.array(sorted(set(cand)))
    if len(cand) == 0:
        return tail

    # 随机抽几个模态   加频率jitter
    rng = np.random.default_rng() if rng is None else rng

    K = rng.integers(n_modes_range[0], n_modes_range[1] + 1)
    K = min(K, len(cand))
    fk = rng.choice(cand, size=K, replace=False)
    fk = fk * rng.uniform(0.98, 1.02, size=K)  # 加2% jitter

    # 合成阻尼正弦 modes
    modes = np.zeros_like(tail, dtype=np.float64)
    for f in fk:
        phi = rng.uniform(0, 2*np.pi)
        # 阻尼常数
        tau = rt60 * rng.uniform(0.4, 1.2)
        # 低频更慢
        tau *= (120.0 / max(f, 60.0))**0.2
        a = np.exp(-t / max(tau, 1e-3))
        modes += a * np.sin(2*np.pi*f*t + phi)

    # 模态强度：相对tail 的低频能量
    # 为简单稳健：用全段RMS做参考（也可改成 bandpass 40-200 的RMS）
    rms_tail = np.sqrt(np.mean(tail**2) + 1e-12)

    # normalize
    rms_modes = np.sqrt(np.mean(modes**2) + 1e-12)
    modes /= rms_modes

    rel_db = rng.uniform(rel_db_range[0], rel_db_range[1])
    target_rms_modes = rms_tail * (10.0 ** (rel_db / 20.0))
    modes *= target_rms_modes

    return tail + modes




def nextpow2(M):
    return int(2**np.ceil(np.log2(M)))


def wall_alpha(alpha_bar, scale=0.15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return np.clip(alpha_bar * rng.uniform(1-scale, 1+scale, size=alpha_bar.shape), 0.01, 0.99)

def generate_velvet_noise(length, fs, density=2000, rng=None):
    # 脉冲噪声代替白噪声去相干 
    rng = np.random.default_rng() if rng is None else rng
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
    # 高通
    sos = butter(4, cutoff, 'hp', fs=fs, output='sos')
    return sosfilt(sos, sig)


class SoftCardioid(Cardioid):
    # pyroom src指向性
    def __init__(self, orientation, alpha=0.3, gain=1.0):
        super().__init__(orientation, gain=gain)
        self.alpha = alpha

    def evaluate(self, direction):
        base = super().evaluate(direction)
        return self.alpha + (1 - self.alpha) * base


def sample_room_params(lx, ly, lz, fs=32000, rng=None, rt60_target=None):
    """Sample (or set) room-level parameters once, then reuse across multiple mics.

    Returns a dict that can be fed into ``simulate_rir_with_params``.
    """
    rng = np.random.default_rng() if rng is None else rng

    center_freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    room_dim = [lx, ly, lz]

    # RT60
    RT60_target = float(rng.uniform(0.1, 1.0) if rt60_target is None else rt60_target)

    # rt60 => mean absorption
    V = lx * ly * lz
    S_total = 2 * (lx * ly + lx * lz + ly * lz)
    alpha_mean = np.clip(0.161 * V / (S_total * RT60_target), 0.05, 0.6)

    # freq. prior (shape)
    wall_base = {
        "glass":   np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.10]),
        "wall":    np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.35]),
        "curtain": np.array([0.08, 0.15, 0.30, 0.45, 0.55, 0.65, 0.70]),
        "ceiling": np.array([0.10, 0.20, 0.35, 0.50, 0.60, 0.70, 0.70]),
        "floor":   np.array([0.04, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30]),
    }
    shape_prior = np.mean(np.stack(list(wall_base.values())), axis=0)
    shape_prior /= np.mean(shape_prior)

    shape_jitter = shape_prior * rng.uniform(0.85, 1.15, size=shape_prior.shape)
    alpha_bar = np.clip(alpha_mean * shape_jitter, 0.02, 0.9)

    # continuous interp in log-freq
    log_fc = np.log(center_freqs)
    alpha_continuous = interp1d(log_fc, alpha_bar, kind='cubic', fill_value='extrapolate')

    # geo ER
    material = {
        k: pra.Material({
            'coeffs': wall_alpha(alpha_bar, rng=rng),
            'scattering': float(rng.uniform(0.2, 0.6)),
            'center_freqs': center_freqs
        }) for k in ['west', 'east', 'south', 'north', 'floor', 'ceiling']
    }

    # max order
    L_min = min(room_dim)
    t_er = float(rng.uniform(0.06, 0.12))
    max_order = int(np.ceil(C * t_er / L_min))
    max_order = int(np.clip(max_order, 5, 40))

    return {
        'room_dim': room_dim,
        'RT60_target': RT60_target,
        'center_freqs': center_freqs,
        'alpha_continuous': alpha_continuous,
        'materials': material,
        'max_order': max_order,
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
    rng = np.random.default_rng() if rng is None else rng

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
    noise = generate_velvet_noise(
        tail_len,
        fs,
        density=float(rng.uniform(8000, max(8001.0, 0.98 * fs))),
        rng=rng,
    )

    Noise_f = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / fs)
    log_f = np.log(np.clip(freqs, 50, fs / 2))

    alpha_f = np.clip(alpha_continuous(log_f), 0.02, 0.99)
    decay_shape = alpha_f / np.mean(alpha_f)

    time_decay = np.exp(-6.9 * np.arange(len(noise)) / (RT60_target * fs))
    Noise_f *= decay_shape
    tail = np.fft.irfft(Noise_f, n=len(noise))
    tail *= time_decay
    tail = apply_highpass(tail, fs, cutoff=40)

    tail = add_low_freq_modes(
        tail,
        fs,
        room_dim=(lx, ly, lz),
        rt60=RT60_target,
        fmin=40,
        fmax=200,
        n_modes_range=(3, 8),
        rel_db_range=(-30, -20),
        rng=rng,
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

    peak = np.max(np.abs(rir))
    if peak > 1e-12:
        rir = rir / peak * 0.9

    return rir, RT60_target


def pyroom_simulation_rt60_driven( 
        p1x, p1y, p1z,
        srcx, srcy, srcz,
        angle_offset,
        lx, ly, lz,
        fs=32000):
    # rt60约束下 声场仿真 ==> 再去拼接早期反射和晚期的
    center_freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    room_dim = [lx, ly, lz]

    # 随机rt60约束
    RT60_target = np.random.uniform(0.1, 1.0)

    

    # rt60 => 
    V = lx * ly * lz
    S_total = 2 * (lx*ly + lx*lz + ly*lz)

    alpha_mean = np.clip(
        0.161 * V / (S_total * RT60_target),
        0.05, 0.6)

    # freq. prior (shape)
    wall_base = {
        "glass":   np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.10]),
        "wall":    np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.35]),
        "curtain": np.array([0.08, 0.15, 0.30, 0.45, 0.55, 0.65, 0.70]),
        "ceiling": np.array([0.10, 0.20, 0.35, 0.50, 0.60, 0.70, 0.70]),
        "floor":   np.array([0.04, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30]),
    }

    shape_prior = np.mean(np.stack(list(wall_base.values())), axis=0)
    shape_prior /= np.mean(shape_prior)

    shape_jitter = shape_prior * np.random.uniform(0.85, 1.15, size=shape_prior.shape)
    alpha_bar = np.clip(alpha_mean * shape_jitter, 0.02, 0.9)

    # 频率连续不切分，
    log_fc = np.log(center_freqs)
    alpha_continuous = interp1d(
        log_fc,
        alpha_bar,
        kind='cubic',
        fill_value='extrapolate'
    )

    # geo ER
    material = {
        k: pra.Material({
            'coeffs': wall_alpha(alpha_bar),
            'scattering': np.random.uniform(0.2, 0.6),
            'center_freqs': center_freqs
        }) for k in ['west', 'east', 'south', 'north', 'floor', 'ceiling']
    }

    # 最大反射阶次计算
    C = 343
    L_min = min(room_dim)
    t_er = np.random.uniform(0.06, 0.12)  # ER
    MAX_ORDER = int(np.ceil(C * t_er / L_min))
    MAX_ORDER = int(np.clip(MAX_ORDER, 5, 40))

    # 早期（pyroom直接生成），后面用随机噪声去做lr
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=material,
        max_order=MAX_ORDER,
        use_rand_ism=True,
        air_absorption=True
    )

    azimuth = np.deg2rad(angle_offset)
    orientation = DirectionVector(azimuth, np.pi/2, degrees=False)
    directivity = SoftCardioid(orientation, alpha=0.7)

    room.add_source([srcx, srcy, srcz], signal=None, directivity=directivity)
    room.add_microphone_array(np.array([p1x, p1y, p1z]).reshape(3, 1))
    room.compute_rir()


    # ========= > 早期
    rir_ism = np.asarray(room.rir[0][0], dtype=np.float64)
    # 拼接参数预计算
    V = lx * ly * lz  # room volume
    f_sch = 2000.0 * np.sqrt(max(RT60_target, 1e-3) / max(V, 1e-3))
    t_split = np.clip(3.0 / max(f_sch, 50.0), 0.05, 0.12)   # clamp
    split_idx = int(t_split * fs)

    fade_len = int(0.02 * fs)  # 20ms cross
    split_idx = int(np.clip(split_idx, fade_len + 1, len(rir_ism)))
    early = rir_ism[:split_idx]
    # 直达/早期加一个jitter去产生抖动
    jitter = int(np.random.uniform(0.2e-3, 0.8e-3) * fs)
    early = np.convolve(
        early,
        np.random.randn(jitter) * 0.05,
        mode='same'
    )
    # 早期的反射加一个相位抖动去相关
    E = np.fft.rfft(early)
    phase_jitter = np.exp(1j * np.random.uniform(-0.1, 0.1, size=E.shape))
    early = np.fft.irfft(E * phase_jitter, n=len(early))


    # ========= > 尾巴
    tail_len = max(int(RT60_target * fs * 1.1), len(rir_ism) - split_idx)
    noise = generate_velvet_noise(tail_len, fs, density=np.random.uniform(8000, 20000))   #脉冲噪声

    Noise_f = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1/fs)
    log_f = np.log(np.clip(freqs, 50, fs/2))

    alpha_f = np.clip(alpha_continuous(log_f), 0.02, 0.99)
    decay_shape = alpha_f / np.mean(alpha_f)

    time_decay = np.exp(-6.9 * np.arange(len(noise)) / (RT60_target * fs))
    Noise_f *= decay_shape
    tail = np.fft.irfft(Noise_f, n=len(noise))
    tail *= time_decay
    # 切掉40Hz以下的直流轰鸣
    tail = apply_highpass(tail, fs, cutoff=40)
    

    # -------低频部分模态
    tail = add_low_freq_modes(
        tail, fs,
        room_dim=(lx, ly, lz),
        rt60=RT60_target,
        fmin=40, fmax=200,
        n_modes_range=(3, 8),
        rel_db_range=(-30, -20)
    )

    # choice.1   直接拼
    # e_early = np.mean(early[-100:]**2)
    # e_tail = np.mean(tail[:100]**2)
    # tail *= np.sqrt(e_early / (e_tail + 1e-8))
    # rir = np.concatenate([early, tail])

    # choice.2 拼接过渡
    w = np.linspace(0, 1, fade_len, endpoint=False)
    a0 = split_idx - fade_len

    rms_e = np.sqrt(np.mean(early[a0:split_idx]**2) + 1e-12)
    rms_t = np.sqrt(np.mean(tail[:fade_len]**2) + 1e-12)
    tail *= (rms_e / rms_t)

    head = early[:a0]
    xfade = early[a0:split_idx] * (1 - w) + tail[:fade_len] * w
    rest = tail[fade_len:]

    rir = np.concatenate([head, xfade, rest])

    peak = np.max(np.abs(rir))
    if peak > 1e-12:
        rir = rir / peak * 0.9

    return rir, RT60_target


def sample_room_params_legacy(lx, ly, lz, fs, rt60_override=None, rng=None):
    """Sample room-level parameters once and reuse across multiple mics.

    Returns a dict that can be passed to :func:`simulate_rir_with_params`.
    """
    rng = np.random.default_rng() if rng is None else rng

    center_freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    room_dim = [lx, ly, lz]

    RT60_target = float(rng.uniform(0.1, 1.0) if rt60_override is None else rt60_override)

    V = lx * ly * lz
    S_total = 2 * (lx*ly + lx*lz + ly*lz)
    alpha_mean = np.clip(0.161 * V / (S_total * max(RT60_target, 1e-3)), 0.05, 0.6)

    wall_base = {
        "glass":   np.array([0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.10]),
        "wall":    np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.35]),
        "curtain": np.array([0.08, 0.15, 0.30, 0.45, 0.55, 0.65, 0.70]),
        "ceiling": np.array([0.10, 0.20, 0.35, 0.50, 0.60, 0.70, 0.70]),
        "floor":   np.array([0.04, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30]),
    }
    shape_prior = np.mean(np.stack(list(wall_base.values())), axis=0)
    shape_prior /= np.mean(shape_prior)

    shape_jitter = shape_prior * rng.uniform(0.85, 1.15, size=shape_prior.shape)
    alpha_bar = np.clip(alpha_mean * shape_jitter, 0.02, 0.9)

    log_fc = np.log(center_freqs)
    alpha_continuous = interp1d(
        log_fc,
        alpha_bar,
        kind='cubic',
        fill_value='extrapolate'
    )

    material = {
        k: pra.Material({
            'coeffs': wall_alpha(alpha_bar, rng=rng),
            'scattering': float(rng.uniform(0.2, 0.6)),
            'center_freqs': center_freqs
        }) for k in ['west', 'east', 'south', 'north', 'floor', 'ceiling']
    }

    L_min = min(room_dim)
    t_er = float(rng.uniform(0.06, 0.12))
    max_order = int(np.ceil(C * t_er / L_min))
    max_order = int(np.clip(max_order, 5, 40))

    return {
        'RT60_target': RT60_target,
        'center_freqs': center_freqs,
        'alpha_continuous': alpha_continuous,
        'material': material,
        'max_order': max_order,
        'room_dim': room_dim,
    }


def simulate_rir_with_params(
    p1x, p1y, p1z,
    srcx, srcy, srcz,
    angle_offset,
    lx, ly, lz,
    fs,
    params,
    rng=None,
):
    """Generate a single-channel RIR using shared room params and per-mic randomness."""
    rng = np.random.default_rng() if rng is None else rng

    RT60_target = float(params['RT60_target'])
    alpha_continuous = params['alpha_continuous']
    material = params['material']
    MAX_ORDER = int(params['max_order'])
    room_dim = [lx, ly, lz]

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=material,
        max_order=MAX_ORDER,
        use_rand_ism=True,
        air_absorption=True
    )

    azimuth = np.deg2rad(angle_offset)
    orientation = DirectionVector(azimuth, np.pi/2, degrees=False)
    directivity = SoftCardioid(orientation, alpha=0.7)

    room.add_source([srcx, srcy, srcz], signal=None, directivity=directivity)
    room.add_microphone_array(np.array([p1x, p1y, p1z]).reshape(3, 1))
    room.compute_rir()

    rir_ism = np.asarray(room.rir[0][0], dtype=np.float64)

    V = lx * ly * lz
    f_sch = 2000.0 * np.sqrt(max(RT60_target, 1e-3) / max(V, 1e-3))
    t_split = np.clip(3.0 / max(f_sch, 50.0), 0.05, 0.12)
    split_idx = int(t_split * fs)

    fade_len = int(0.02 * fs)
    split_idx = int(np.clip(split_idx, fade_len + 1, len(rir_ism)))
    early = rir_ism[:split_idx]

    jitter = int(rng.uniform(0.2e-3, 0.8e-3) * fs)
    jitter = max(jitter, 1)
    early = np.convolve(early, rng.standard_normal(jitter) * 0.05, mode='same')

    E = np.fft.rfft(early)
    phase_jitter = np.exp(1j * rng.uniform(-0.1, 0.1, size=E.shape))
    early = np.fft.irfft(E * phase_jitter, n=len(early))

    tail_len = max(int(RT60_target * fs * 1.1), len(rir_ism) - split_idx)
    noise = generate_velvet_noise(
        tail_len,
        fs,
        density=float(rng.uniform(8000, max(8001.0, 0.98 * fs))),
        rng=rng,
    )

    Noise_f = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1/fs)
    log_f = np.log(np.clip(freqs, 50, fs/2))

    alpha_f = np.clip(alpha_continuous(log_f), 0.02, 0.99)
    decay_shape = alpha_f / np.mean(alpha_f)

    time_decay = np.exp(-6.9 * np.arange(len(noise)) / (RT60_target * fs))
    Noise_f *= decay_shape
    tail = np.fft.irfft(Noise_f, n=len(noise))
    tail *= time_decay
    tail = apply_highpass(tail, fs, cutoff=40)

    tail = add_low_freq_modes(
        tail,
        fs,
        room_dim=(lx, ly, lz),
        rt60=RT60_target,
        fmin=40,
        fmax=200,
        n_modes_range=(3, 8),
        rel_db_range=(-30, -20),
        rng=rng,
    )

    w = np.linspace(0, 1, fade_len, endpoint=False)
    a0 = split_idx - fade_len

    rms_e = np.sqrt(np.mean(early[a0:split_idx]**2) + 1e-12)
    rms_t = np.sqrt(np.mean(tail[:fade_len]**2) + 1e-12)
    tail *= (rms_e / rms_t)

    head = early[:a0]
    xfade = early[a0:split_idx] * (1 - w) + tail[:fade_len] * w
    rest = tail[fade_len:]

    rir = np.concatenate([head, xfade, rest])

    peak = np.max(np.abs(rir))
    if peak > 1e-12:
        rir = rir / peak * 0.9

    return rir, RT60_target


def get_existing_max_idx(folder):
    pattern = re.compile(r"rir_(\d+)\.wav")
    max_idx = -1
    if not os.path.exists(folder):
        return max_idx

    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
    return max_idx


if __name__ == "__main__":
    save_folder = '/home/xukj/dataset_comsolTest/data_py/pyroom_rir_3/'
    os.makedirs(save_folder, exist_ok=True)

    fs = 32000
    num_samples = 300000  
    # num_samples = 100  

    existing_max_idx = get_existing_max_idx(save_folder)
    start_idx = existing_max_idx + 1

    print(f"Found existing RIRs up to idx = {existing_max_idx}")
    print(f"Start simulation from idx = {start_idx}")

    # metadata
    metadata = []

    for idx in range(start_idx, num_samples):
        print(f"\n=== Simulation {idx+1}/{num_samples} ===")


        angle_offset = random.uniform(0, 360)

        lx = random.uniform(4.0, 8.0)
        ly = random.uniform(4.0, 8.0)
        lz = random.uniform(3.0, 4.5)


        margin = 0.25
        p1x = random.uniform(margin, lx - margin)
        p1y = random.uniform(margin, ly - margin)
        p1z = random.uniform(margin, lz - margin)

        max_dist = np.sqrt(lx**2 + ly**2 + lz**2)
        target_dist = np.random.uniform(0.3, max_dist * 0.8) 
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        srcx = p1x + target_dist * np.sin(phi) * np.cos(theta)
        srcy = p1y + target_dist * np.sin(phi) * np.sin(theta)
        srcz = p1z + target_dist * np.cos(phi)

        srcx = np.clip(srcx, margin, lx - margin)
        srcy = np.clip(srcy, margin, ly - margin)
        srcz = np.clip(srcz, margin, lz - margin)

        rir, rt60 = pyroom_simulation_rt60_driven(
            p1x, p1y, p1z,
            srcx, srcy, srcz,
            angle_offset,
            lx, ly, lz,
            fs=fs
        )

        output_path = os.path.join(save_folder, f"rir_{idx:04d}.wav")
        sf.write(output_path, rir, fs)

        metadata.append({
            'idx': idx,
            'lx': lx, 'ly': ly, 'lz': lz,
            'srcx': srcx, 'srcy': srcy, 'srcz': srcz,
            'micx': p1x, 'micy': p1y, 'micz': p1z,
            'angle': angle_offset,
            'rt60': rt60
        })
        rt60_val = rt60
        print(f"RT60 total: {rt60_val:.2f} | ")

        print(f"✅ Saved: {output_path}")

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(save_folder, 'metadata.csv'), index=False)

    print(f"\n✅ All simulations completed!")
    print(f"Metadata saved to {save_folder}/metadata.csv")
