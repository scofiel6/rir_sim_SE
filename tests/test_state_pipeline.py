from pathlib import Path
import unittest

import numpy as np

from config import load_rir_sim_se_config
from rir_sim_se import generate_rir_from_state, load_acoustic_state_json


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "rir_sim_se_config.json"


class StatePipelineSmokeTest(unittest.TestCase):
    def test_state_pipeline_generates_rir_refs(self):
        cfg = load_rir_sim_se_config(CFG_PATH)
        state = load_acoustic_state_json(cfg, cfg.acoustic_state_json)
        n_ch = len(cfg.mic_positions_m) if cfg.mic_positions_m is not None else int(cfg.mic_num)
        out = generate_rir_from_state(cfg, state=state, seed=int(cfg.seed) + 7)

        self.assertEqual(np.asarray(out["rir"]).shape[0], n_ch)
        self.assertEqual(np.asarray(out["ref1"]).shape, np.asarray(out["rir"]).shape)
        self.assertEqual(np.asarray(out["ref2"]).shape, np.asarray(out["rir"]).shape)
        self.assertEqual(
            out["meta"]["params_trace"]["engine_trace"]["late_reverb"]["variant"],
            "multiband_multi_slope_diffuse",
        )
        self.assertEqual(
            out["meta"]["params_trace"]["engine_trace"]["source_radiation"]["variant"],
            "voice_radiation_head_torso",
        )
