from pathlib import Path
import unittest

import numpy as np

from acoustic_inversion import create_generator
from config import load_rir_sim_se_config


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "rir_sim_se_config.json"


class ConfigAndEngineSmokeTest(unittest.TestCase):
    def test_config_loads_with_explicit_linear_array(self):
        cfg = load_rir_sim_se_config(CFG_PATH)
        self.assertEqual(cfg.mic_array_type, "linear")
        self.assertEqual(cfg.mic_positions_m, (0.0, 0.04, 0.36, 0.40))
        self.assertEqual(Path(cfg.acoustic_state_json).name, "acoustic_state.json")

    def test_base_engine_smoke_generate(self):
        cfg = load_rir_sim_se_config(CFG_PATH)
        gen = create_generator(cfg)

        clean = np.zeros(4096, dtype=np.float64)
        clean[0] = 1.0
        y, ref, meta = gen.generate(clean=clean, branch="custom", return_ref=True, normalize_output=False)

        self.assertEqual(y.shape, (4, 4096))
        self.assertEqual(ref.shape, (4, 4096))
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(
            meta["params_trace"]["engine_trace"]["late_reverb"]["variant"],
            "multiband_multi_slope_diffuse",
        )
        self.assertEqual(
            meta["params_trace"]["engine_trace"]["source_radiation"]["variant"],
            "voice_radiation_head_torso",
        )
