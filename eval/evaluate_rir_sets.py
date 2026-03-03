from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

try:
    from eval.rir_metrics import compute_drr_c50_c80, estimate_rt60_bundle, ks_distance
except Exception:
    from rir_metrics import compute_drr_c50_c80, estimate_rt60_bundle, ks_distance


def _iter_audio_files(root: Path) -> Iterable[Path]:
    exts = (".wav", ".flac", ".ogg", ".m4a", ".mp3")
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                yield p


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(path), dtype="float64")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.reshape(-1), int(fs)


def summarize_folder(folder: Path, limit: int | None = None) -> dict[str, list[float]]:
    rows = {"t20": [], "t30": [], "edt": [], "drr": [], "c50": [], "c80": []}
    files = sorted(_iter_audio_files(folder))
    if limit is not None:
        files = files[: int(limit)]

    for p in files:
        r, fs = _load_mono(p)
        rt = estimate_rt60_bundle(r, fs)
        dc = compute_drr_c50_c80(r, fs)
        for k in ("t20", "t30", "edt"):
            v = rt[k]
            if v is not None and np.isfinite(v):
                rows[k].append(float(v))
        for k in ("drr", "c50", "c80"):
            rows[k].append(float(dc[k]))
    return rows


def compare_distributions(ref_stats: dict[str, list[float]], sim_stats: dict[str, list[float]]) -> dict:
    out = {"metrics": {}}
    for k in ("t20", "t30", "edt", "drr", "c50", "c80"):
        a = np.asarray(ref_stats.get(k, []), dtype=np.float64)
        b = np.asarray(sim_stats.get(k, []), dtype=np.float64)
        out["metrics"][k] = {
            "n_ref": int(a.size),
            "n_sim": int(b.size),
            "mean_ref": None if a.size == 0 else float(np.mean(a)),
            "mean_sim": None if b.size == 0 else float(np.mean(b)),
            "std_ref": None if a.size == 0 else float(np.std(a)),
            "std_sim": None if b.size == 0 else float(np.std(b)),
            "ks_distance": float(ks_distance(a, b)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="Folder with measured RIR files")
    ap.add_argument("--sim", required=True, help="Folder with simulated RIR files")
    ap.add_argument("--limit", type=int, default=None, help="Optional max files per set")
    ap.add_argument("--out", default="eval_report.json", help="Output report path")
    args = ap.parse_args()

    real_dir = Path(args.real)
    sim_dir = Path(args.sim)
    report_path = Path(args.out)

    real_stats = summarize_folder(real_dir, limit=args.limit)
    sim_stats = summarize_folder(sim_dir, limit=args.limit)
    report = {
        "real_dir": str(real_dir),
        "sim_dir": str(sim_dir),
        "limit": args.limit,
        "comparison": compare_distributions(real_stats, sim_stats),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
