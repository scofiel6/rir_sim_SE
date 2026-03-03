from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sound_field_sim import rirGen_baseSE as base


# Re-export compatibility wrappers used by the physical engine layer.
call_sample_room_params = base._call_sample_room_params
call_simulate_rir = base._call_simulate_rir
