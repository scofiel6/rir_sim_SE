from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engine" / "sound_field_sim"
OUT_DIR = ROOT / "reproducibility"
OUT_PATH = OUT_DIR / "engine_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in ENGINE_DIR.glob("*.py") if p.is_file()])
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_git_head": git_head(ROOT),
        "engine_dir": str(ENGINE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "files": [
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(p),
            }
            for p in files
        ],
    }
    OUT_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
