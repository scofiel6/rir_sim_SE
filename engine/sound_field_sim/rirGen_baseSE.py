import inspect
import json
from pathlib import Path
from math import gcd

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import butter, fftconvolve, sosfilt, resample_poly

import im_rir_v2 as imv2


def _resample_poly_1d(x, fs_in, fs_out, allow_upsample=False):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs_in = int(round(float(fs_in)))
    fs_out = int(round(float(fs_out)))
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"Invalid sample rates: fs_in={fs_in}, fs_out={fs_out}")
    if fs_in == fs_out:
        return x
    if (fs_out > fs_in) and (not allow_upsample):
        raise ValueError(
            f"Upsampling is disabled by default: {fs_in} -> {fs_out}. "
            "Use a source with fs >= target fs or explicitly allow upsampling."
        )
    g = gcd(fs_out, fs_in)
    up = fs_out // g
    down = fs_in // g
    y = resample_poly(x, up, down).astype(np.float64)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

def _sig_params(func):
    try:
        return list(inspect.signature(func).parameters.keys())
    except Exception:
        return []


def _normalize_params_for_sim(params, need_key):
    if not isinstance(params, dict):
        return params
    if need_key in params:
        return params
    out = dict(params)
    if need_key == "material" and "materials" in params:
        out["material"] = params["materials"]
    if need_key == "materials" and "material" in params:
        out["materials"] = params["material"]
    return out


def _call_sample_room_params(imv2_mod, lx, ly, lz, fs, rng, rt60_tgt):
    """
    Compat wrapper for room-param sampling across im_rir_v2 variants.

    Why:
    im_rir_v2 may evolve over time (parameter names or optional args differ).
    This wrapper isolates API differences so the generator logic stays stable.
    """
    if hasattr(imv2_mod, "sample_room_params"):
        fn = imv2_mod.sample_room_params
        p = _sig_params(fn)
        kw = {}
        if "rng" in p:
            kw["rng"] = rng
        if "rt60_target" in p:
            kw["rt60_target"] = float(rt60_tgt)
        elif "rt60_override" in p:
            kw["rt60_override"] = float(rt60_tgt)
        elif "rt60" in p:
            kw["rt60"] = float(rt60_tgt)

        if "fs" in p:
            try:
                fs_required = (inspect.signature(fn).parameters["fs"].default is inspect._empty)
            except Exception:
                fs_required = False
            if fs_required:
                return fn(lx, ly, lz, fs, **kw)
            kw["fs"] = fs
        return fn(lx, ly, lz, **kw)

    if hasattr(imv2_mod, "sample_room_params_legacy"):
        fn = imv2_mod.sample_room_params_legacy
        p = _sig_params(fn)
        kw = {}
        if "rng" in p:
            kw["rng"] = rng
        if "rt60_override" in p:
            kw["rt60_override"] = float(rt60_tgt)
        return fn(lx, ly, lz, fs, **kw)

    raise AttributeError("im_rir_v2 has no sample_room_params / sample_room_params_legacy")


def _call_simulate_rir(imv2_mod, *, mic_xyz, src_xyz, doa_deg, lx, ly, lz, fs, params, rng):
    """
    Compat wrapper for simulate_rir_with_params across signatures.

    Some im_rir_v2 branches use vector-style args, others still use legacy
    scalar signatures. We normalize this difference here and always return
    `(rir, rt60)` in a consistent format for the SE pipeline.
    """
    if not hasattr(imv2_mod, "simulate_rir_with_params"):
        raise AttributeError("im_rir_v2 has no simulate_rir_with_params")

    fn = imv2_mod.simulate_rir_with_params
    p = _sig_params(fn)
    is_legacy = ("p1x" in p) or ("srcx" in p)

    if not is_legacy:
        kw = {}
        kw["mic_xyz"] = np.asarray(mic_xyz, dtype=np.float64)
        kw["src_xyz"] = np.asarray(src_xyz, dtype=np.float64)
        if "angle_offset" in p:
            kw["angle_offset"] = float(doa_deg)
        elif "doa_deg" in p:
            kw["doa_deg"] = float(doa_deg)
        else:
            kw["angle_offset"] = float(doa_deg)
        if "lx" in p:
            kw["lx"] = float(lx)
        if "ly" in p:
            kw["ly"] = float(ly)
        if "lz" in p:
            kw["lz"] = float(lz)
        if "fs" in p:
            kw["fs"] = int(fs)
        if "params" in p:
            kw["params"] = _normalize_params_for_sim(params, "materials")
        if "rng" in p:
            kw["rng"] = rng
        out = fn(**kw)
    else:
        p1x, p1y, p1z = [float(v) for v in mic_xyz]
        srcx, srcy, srcz = [float(v) for v in src_xyz]
        params2 = _normalize_params_for_sim(params, "material")
        out = fn(p1x, p1y, p1z, srcx, srcy, srcz, float(doa_deg), float(lx), float(ly), float(lz), int(fs), params2, rng=rng)

    if isinstance(out, tuple):
        rir = out[0]
        rt60 = out[1] if len(out) > 1 else None
    else:
        rir, rt60 = out, None
    return np.asarray(rir, dtype=np.float64).reshape(-1), rt60


class BaseSERIRGenerator:
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

        self.band_centers_ref = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float64)

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

    def _to_mono(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            return np.mean(x, axis=1)
        raise ValueError(f"Unsupported audio shape: {x.shape}")

    def _resample_to_fs(self, x, fs_in, allow_upsample=False):
        fs_in = int(round(float(fs_in)))
        if fs_in == self.fs:
            return np.asarray(x, dtype=np.float64)
        if fs_in <= 0:
            raise ValueError(f"Invalid fs_in: {fs_in}")
        return _resample_poly_1d(
            np.asarray(x, dtype=np.float64),
            fs_in=fs_in,
            fs_out=self.fs,
            allow_upsample=allow_upsample,
        )

    def _load_audio_mono(self, item):
        if isinstance(item, np.ndarray):
            x = self._to_mono(item)
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), "<ndarray>"
        if isinstance(item, dict):
            x = self._to_mono(item["audio"])
            fs_in = int(item.get("fs", self.fs))
            x = self._resample_to_fs(x, fs_in)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, str(item.get("id", item.get("path", "<dict-audio>")))
        if isinstance(item, (str, Path)):
            path = str(item)
            # Read audio from file path. This is the actual disk I/O entry point.
            x, fs_in = sf.read(path, dtype="float64")
            x = self._to_mono(x)
            x = self._resample_to_fs(x, fs_in)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, path
        raise TypeError(f"Unsupported audio item type: {type(item)}")

    def _load_audio_mono_keep_fs(self, item):
        """
        Load mono audio but keep original sampling rate.

        Used by acoustic-parameter inversion so we do not distort decay/band
        characteristics by resampling before estimation.
        """
        if isinstance(item, np.ndarray):
            x = self._to_mono(item)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, int(self.fs), "<ndarray>"
        if isinstance(item, dict):
            x = self._to_mono(item["audio"])
            fs_in = int(item.get("fs", self.fs))
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, fs_in, str(item.get("id", item.get("path", "<dict-audio>")))
        if isinstance(item, (str, Path)):
            path = str(item)
            x, fs_in = sf.read(path, dtype="float64")
            x = self._to_mono(x)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x, int(round(float(fs_in))), path
        raise TypeError(f"Unsupported audio item type: {type(item)}")

    def _crop_or_pad(self, x, target_len, rng, tile_short=False):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        if target_len <= 0:
            raise ValueError(f"target_len must be positive, got {target_len}")
        if n == target_len:
            return x.copy()
        if n > target_len:
            s = int(rng.integers(0, n - target_len + 1))
            return x[s:s + target_len].copy()
        if n == 0:
            return np.zeros(target_len, dtype=np.float64)
        if tile_short:
            reps = int(np.ceil(target_len / n))
            return np.tile(x, reps)[:target_len].astype(np.float64)
        out = np.zeros(target_len, dtype=np.float64)
        out[:n] = x
        return out

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
    def _to_material_dict(keys, coeffs, center_freqs, rng):
        out = {}
        for k in keys:
            c = np.clip(coeffs * rng.uniform(0.92, 1.08, size=len(coeffs)), 0.01, 0.99)
            out[k] = pra.Material({
                "coeffs": c,
                "scattering": float(rng.uniform(0.2, 0.6)),
                "center_freqs": center_freqs,
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

        out = dict(params) if isinstance(params, dict) else params
        if not isinstance(out, dict):
            return out

        out["center_freqs"] = np.asarray(band_centers, dtype=np.float64)
        log_fc = np.log(np.asarray(band_centers, dtype=np.float64))
        out["alpha_continuous"] = interp1d(log_fc, np.asarray(alpha, dtype=np.float64), kind="linear", fill_value="extrapolate")

        if "materials" in out:
            keys = list(out["materials"].keys())
            out["materials"] = self._to_material_dict(keys, alpha, out["center_freqs"], rng)
        if "material" in out:
            keys = list(out["material"].keys())
            out["material"] = self._to_material_dict(keys, alpha, out["center_freqs"], rng)
        return out

    def _direct_ref(self, src_xyz, mic_xyz, n_samples):
        dist = float(np.linalg.norm(np.asarray(src_xyz) - np.asarray(mic_xyz)))
        delay_samp = int(round(dist / imv2.C * self.fs))
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
                band_prior = np.nanmedian(band_mat, axis=0)
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

        if room_size_hint is not None:
            room_size_hint = np.asarray(room_size_hint, dtype=np.float64).reshape(3)
            j = max(0.0, float(room_jitter_ratio))
            lx, ly, lz = room_size_hint.tolist()
            fitted_room = {
                "lx": (max(1.5, lx * (1.0 - j)), max(1.55, lx * (1.0 + j))),
                "ly": (max(1.5, ly * (1.0 - j)), max(1.55, ly * (1.0 + j))),
                "lz": (max(2.0, lz * (1.0 - j)), max(2.05, lz * (1.0 + j))),
            }
        else:
            fitted_room = dict(self.custom_room_range)

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
            "fitted_custom_room_range": fitted_room,
            "per_item": per_item,
        }

        # Write inferred priors back to generator so subsequent generate/generate_dataset
        # use room-specific distributions instead of generic defaults.
        if update_generator:
            self.custom_room_range = fitted_room
            self.custom_rt60_range = (rt20, rt80)
            self.custom_rt60_center = rt50
            self.custom_band_rt60_prior = band_prior
            self.drr_range_db = drr_range
            self.c50_range_db = c50_range
            self.custom_noise_rms = fit["noise_rms_median"]
            self.custom_noise_tilt_db_oct = fit["noise_tilt_db_per_oct_median"]
            self.fitted = fit
        return fit

    def _sample_noise_matrix(self, n_ch, n_samples, noise_items, rng):
        noise = np.zeros((n_ch, n_samples), dtype=np.float64)
        if len(noise_items) == 0:
            for ch in range(n_ch):
                noise[ch] = rng.standard_normal(n_samples)
            return noise
        for ch in range(n_ch):
            item = noise_items[int(rng.integers(0, len(noise_items)))]
            x, _ = self._load_audio_mono(item)
            noise[ch] = self._crop_or_pad(x, n_samples, rng, tile_short=True)
        return noise

    @staticmethod
    def _mix_with_snr(y, noise, snr_db):
        y = np.asarray(y, dtype=np.float64)
        n = np.asarray(noise, dtype=np.float64)
        out = np.zeros_like(y)
        eps = 1e-12
        lin = 10.0 ** (float(snr_db) / 10.0)
        for ch in range(y.shape[0]):
            ps = np.mean(y[ch] ** 2) + eps
            pn = np.mean(n[ch] ** 2) + eps
            g = np.sqrt((ps / lin) / pn)
            out[ch] = y[ch] + g * n[ch]
        return out

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

        params = _call_sample_room_params(imv2, room_size[0], room_size[1], room_size[2], self.fs, rng, rt60_tgt)
        # Per-sample band profile randomization for SE robustness.
        fc = self._jitter_band_centers(rng)
        band_rt60 = self._sample_band_rt60(rt60_tgt, rng, band_prior=band_prior)
        params = self._apply_band_profile_to_params(params, room_size, fc, band_rt60, rng)

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
            rir, rt60_out = _call_simulate_rir(
                imv2,
                mic_xyz=mic_loc[:, ch],
                src_xyz=src_loc,
                doa_deg=doa,
                lx=room_size[0],
                ly=room_size[1],
                lz=room_size[2],
                fs=self.fs,
                params=params,
                rng=rng,
            )
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

    def generate_dataset(
        self,
        clean_sources,
        out_dir,
        n_items,
        noise_sources=None,
        clip_seconds=4.0,
        seed=0,
        return_ref=True,
        ref_direct=True,
        write_float32=True,
    ):
        clean_items = self._resolve_audio_items(clean_sources, "clean_sources")
        noise_items = self._resolve_audio_items(noise_sources, "noise_sources") if noise_sources is not None else []
        out = Path(out_dir)
        mix_dir = out / "mix"
        clean_dir = out / "clean"
        ref_dir = out / "ref"
        out.mkdir(parents=True, exist_ok=True)
        mix_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        ref_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out / "metadata.jsonl"

        rng = np.random.default_rng(int(seed))
        clip_n = int(max(1, round(float(clip_seconds) * self.fs)))
        recs = []

        # Batch generation entry: repeatedly sample clean/noise/room params and write
        # waveform pairs + metadata so SE training can reproduce each sample by seed.
        for i in range(int(n_items)):
            clean_item = clean_items[int(rng.integers(0, len(clean_items)))]
            clean, clean_id = self._load_audio_mono(clean_item)
            clean = self._crop_or_pad(clean, clip_n, rng, tile_short=False)
            sample_seed = int(rng.integers(1, 2**31 - 1))

            y_rev, ref, meta = self.generate(
                clean,
                seed=sample_seed,
                return_ref=return_ref,
                ref_direct=ref_direct,
                normalize_output=False,
            )
            snr_db = self._sample_scalar(rng, self.snr_range_db)
            noise = self._sample_noise_matrix(y_rev.shape[0], y_rev.shape[1], noise_items, rng)
            y_mix = self._mix_with_snr(y_rev, noise, snr_db)
            y_mix, clean_out, ref_out_sig, norm_gain, mix_peak_before_norm = self._final_peak_normalize_triplet(
                y_mix,
                clean=clean,
                ref=ref,
            )

            mix_path = mix_dir / f"mix_{i:06d}.wav"
            clean_path = clean_dir / f"clean_{i:06d}.wav"
            ref_path = ref_dir / f"ref_{i:06d}.wav"
            dt = np.float32 if write_float32 else np.float64
            sf.write(str(mix_path), y_mix.T.astype(dt), self.fs)
            sf.write(str(clean_path), clean_out.astype(dt), self.fs)
            if return_ref and ref_out_sig is not None:
                if ref_out_sig.ndim == 2:
                    sf.write(str(ref_path), ref_out_sig.T.astype(dt), self.fs)
                else:
                    sf.write(str(ref_path), ref_out_sig.astype(dt), self.fs)
                ref_out = str(ref_path)
            else:
                ref_out = None

            rec = {
                "idx": i,
                "clean_item": clean_id,
                "mix_path": str(mix_path),
                "clean_path": str(clean_path),
                "ref_path": ref_out,
                "snr_db": float(snr_db),
                "final_norm_gain": float(norm_gain),
                "mix_peak_before_norm": float(mix_peak_before_norm),
                "seed": int(sample_seed),
                "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in meta.items()},
            }
            recs.append(rec)

        with open(meta_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return {
            "out_dir": str(out),
            "metadata": str(meta_path),
            "n_generated": len(recs),
            "generic_mix_prob": float(self.generic_mix_prob),
        }


def _mono_resample_to_fs(x, fs_in, fs_out, allow_upsample=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if x.ndim != 1:
        raise ValueError(f"audio must be 1-D or 2-D, got shape={x.shape}")
    fs_in = int(round(float(fs_in)))
    fs_out = int(round(float(fs_out)))
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"invalid sample rate: fs_in={fs_in}, fs_out={fs_out}")
    if fs_in == fs_out:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return _resample_poly_1d(
        x,
        fs_in=fs_in,
        fs_out=fs_out,
        allow_upsample=allow_upsample,
    )


def _room_range_from_hint(room_size_hint, jitter_ratio):
    room_size_hint = np.asarray(room_size_hint, dtype=np.float64).reshape(3)
    j = max(0.0, float(jitter_ratio))
    lx, ly, lz = room_size_hint.tolist()
    return {
        "lx": (max(1.5, lx * (1.0 - j)), max(1.55, lx * (1.0 + j))),
        "ly": (max(1.5, ly * (1.0 - j)), max(1.55, ly * (1.0 + j))),
        "lz": (max(2.0, lz * (1.0 - j)), max(2.05, lz * (1.0 + j))),
    }


def generate_rir_from_recorded_pulse(
    fs,
    pulse_recording,
    use_drr_c50=True,
    seed=1234,
    room_size_hint=(3.6, 3.8, 2.7),
    room_jitter_ratio=0.04,
    custom_room_range=None,
    generic_room_range=None,
):
    """
    Minimal callable API for single-RIR generation.

    Inputs:
    - fs: target sample rate
    - pulse_recording: path/list for real recorded impulse-like signal(s)
    - use_drr_c50: whether to apply DRR/C50 target shaping

    Returns:
    - rir: np.ndarray [n]
    - meta: dict
    """
    fs = int(fs)
    seed = int(seed)

    # Single-mic setup: this API returns one RIR by design.
    mic_info = {
        "device_id": "single_mic_api",
        "device_height": 1.2,
        "array_type": "linear",
        "mic_pos": [0.0],
    }

    if custom_room_range is None:
        custom_room_range = _room_range_from_hint(room_size_hint, room_jitter_ratio)
    if generic_room_range is None:
        generic_room_range = {"lx": (2.8, 6.5), "ly": (2.8, 6.5), "lz": (2.4, 3.6)}

    # Preset aligned with previous version (before "lighter reverb" tuning).
    gen = BaseSERIRGenerator(
        fs=fs,
        mic_info=mic_info,
        custom_room_range=custom_room_range,
        generic_room_range=generic_room_range,
        custom_rt60_range=(0.2, 1.2),
        generic_rt60_range=(0.12, 1.3),
        generic_mix_prob=0.0,  # API path focuses on fitted room
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

    fit = gen.fit_from_recordings(
        recordings=pulse_recording,
        room_size_hint=room_size_hint,
        room_jitter_ratio=room_jitter_ratio,
        rt60_min_max=(0.12, 1.4),
        drr_prior_range_db=(-3.0, 8.0),
        c50_prior_range_db=(0.0, 14.0),
        drr_c50_jitter_db=0.6,
        drr_c50_mode=("auto" if bool(use_drr_c50) else "fixed"),
        drr_c50_from_recording_jitter_db=0.2,
        fit_seed=seed,
        update_generator=True,
    )

    # Delta excitation -> generated channel output equals RIR.
    rir_len = int(max(512, round(2.5 * fs)))
    dry_delta = np.zeros(rir_len, dtype=np.float64)
    dry_delta[0] = 1.0
    y, _, meta = gen.generate(
        dry_delta,
        seed=seed + 1,
        return_ref=False,
        ref_direct=True,
        branch="custom",
        normalize_output=False,
        apply_drr_c50=bool(use_drr_c50),
    )
    rir = np.asarray(y[0], dtype=np.float64)

    meta["api_use_drr_c50"] = bool(use_drr_c50)
    meta["api_fit_rt60_median"] = float(fit["rt60_median"])
    meta["api_fit_drr_range"] = fit["drr_db_p20_p80"]
    meta["api_fit_c50_range"] = fit["c50_db_p20_p80"]
    return rir, meta


if __name__ == "__main__":
    # Minimal module-style demo:
    # Input only: fs + recorded pulse signal + whether to apply DRR/C50.
    fs = 32000
    pulse_recording = "/home/xukj/dataset_comsolTest/room_test"
    use_drr_c50 = True
    dry_wav = "/home/xukj/dataset_rir/sound_field_sim/test.wav"

    out_dir = Path("./_demo_module")
    out_dir.mkdir(parents=True, exist_ok=True)

    rir, meta = generate_rir_from_recorded_pulse(
        fs=fs,
        pulse_recording=pulse_recording,
        use_drr_c50=use_drr_c50,
        seed=2026,
    )
    rir_path = out_dir / "rir_from_pulse.wav"
    sf.write(str(rir_path), rir.astype(np.float32), fs)

    if Path(dry_wav).exists():
        dry_raw, dry_fs = sf.read(dry_wav, dtype="float64")
        dry = _mono_resample_to_fs(dry_raw, dry_fs, fs)
        dry_id = dry_wav
    else:
        dry_id = "<synthetic>"
        t = np.arange(int(4.0 * fs), dtype=np.float64) / fs
        dry = 0.15 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)

    wet = fftconvolve(dry, rir)[:len(dry)]
    peak_wet = float(np.max(np.abs(wet))) if wet.size > 0 else 0.0
    if peak_wet > 0.99:
        wet = wet / peak_wet * 0.99

    dry_path = out_dir / "dry.wav"
    wet_path = out_dir / "wet.wav"
    sf.write(str(dry_path), dry.astype(np.float32), fs)
    sf.write(str(wet_path), wet.astype(np.float32), fs)

    print("\n=== [MODULE DEMO] pulse -> rir -> wet ===")
    print("fs:", fs)
    print("pulse_recording:", pulse_recording)
    print("use_drr_c50:", use_drr_c50)
    print("dry source:", dry_id)
    print("saved rir:", str(rir_path))
    print("saved dry:", str(dry_path))
    print("saved wet:", str(wet_path))
    print("rt60 target/real:", meta.get("rt60_target"), meta.get("rt60_real"))
    print("drr target/real:", meta.get("drr_target_db"), meta.get("drr_real_db"))
    print("c50 target/real:", meta.get("c50_target_db"), meta.get("c50_real_db"))
    print("drr_c50_applied:", meta.get("drr_c50_applied"))
