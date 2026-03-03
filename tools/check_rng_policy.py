from __future__ import annotations

from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATTERN_IMPORT_RANDOM = re.compile(r"^\s*import\s+random\b")
PATTERN_FROM_RANDOM = re.compile(r"^\s*from\s+random\s+import\b")
PATTERN_RANDOM_DOT = re.compile(r"(?<!np\.)\brandom\.")
PATTERN_NP_RANDOM_BAD = re.compile(r"\bnp\.random\.(?!default_rng\b)")
PATTERN_DEFAULT_RNG_NO_SEED = re.compile(r"\bdefault_rng\(\s*\)")


def scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if PATTERN_IMPORT_RANDOM.search(line):
            issues.append(f"{path}:{i}: use numpy Generator instead of 'import random'")
        if PATTERN_FROM_RANDOM.search(line):
            issues.append(f"{path}:{i}: use numpy Generator instead of 'from random import ...'")
        if PATTERN_RANDOM_DOT.search(line):
            issues.append(f"{path}:{i}: random.* is not allowed")
        if PATTERN_NP_RANDOM_BAD.search(line):
            issues.append(f"{path}:{i}: only np.random.default_rng(seed) is allowed")
        if PATTERN_DEFAULT_RNG_NO_SEED.search(line):
            issues.append(f"{path}:{i}: default_rng() without seed is not allowed")
    return issues


def main() -> int:
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    issues: list[str] = []
    for f in py_files:
        if ".git" in f.parts or "__pycache__" in f.parts:
            continue
        if f == Path(__file__).resolve():
            continue
        issues.extend(scan_file(f))

    if issues:
        print("RNG policy violations found:")
        for it in issues:
            print(it)
        return 1
    print("RNG policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
