import unittest

import numpy as np


class ControllerPresetContractTest(unittest.TestCase):
    def test_current_real10kg_presets_construct_and_compute_finite_actions(self):
        from pac.controllers.presets import (
            build_real10kg_predictive_event,
            build_real10kg_smc_steady,
        )

        target = np.zeros(6, dtype=float)
        eta = np.zeros(6, dtype=float)
        nu = np.zeros(6, dtype=float)
        current = np.array([0.2, -0.1, 0.05], dtype=float)

        for builder in [build_real10kg_smc_steady, build_real10kg_predictive_event]:
            controller = builder()
            controller.set_trajectory3d(True)
            action = controller.compute(
                target,
                eta,
                nu,
                t=0.0,
                current_prediction=current,
            )
            self.assertEqual(action.shape, (6,))
            self.assertTrue(np.isfinite(action).all(), builder.__name__)


if __name__ == "__main__":
    unittest.main()
