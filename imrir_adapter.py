from pathlib import Path
import importlib.util
import sys


_THIS_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _THIS_DIR / "engine" / "sound_field_sim"


def _load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_base_module():
    # Prefer vendored engine for reproducibility.
    vendored_base = _VENDOR_DIR / "rirGen_baseSE.py"
    if vendored_base.exists():
        if str(_VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(_VENDOR_DIR))
        return _load_module_from_file("rirGen_baseSE_vendor", vendored_base)

    # Fallback to external workspace copy when vendored files are unavailable.
    root = _THIS_DIR.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sound_field_sim import rirGen_baseSE as base  # type: ignore

    return base


base = _load_base_module()

# Re-export compatibility wrappers used by the physical engine layer.
call_sample_room_params = base._call_sample_room_params
call_simulate_rir = base._call_simulate_rir
