import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class ControllerPresetContractTest(unittest.TestCase):
    def test_current_real10kg_presets_construct_and_compute_finite_actions(self):
        from evaluation.baseline_presets import (
            build_real10kg_mpc_event,
            build_real10kg_smc,
            build_real10kg_smc_steady,
        )

        target = np.zeros(6, dtype=float)
        eta = np.zeros(6, dtype=float)
        nu = np.zeros(6, dtype=float)
        current = np.array([0.2, -0.1, 0.05], dtype=float)

        for builder in [build_real10kg_smc, build_real10kg_smc_steady, build_real10kg_mpc_event]:
            controller = builder()
            controller.set_trajectory3d(True)
            action = controller.compute(
                target,
                eta,
                nu,
                t=0.0,
                dt=0.01,
                current_prediction=current,
            )
            self.assertEqual(action.shape, (6,))
            self.assertTrue(np.isfinite(action).all(), builder.__name__)


if __name__ == "__main__":
    unittest.main()
