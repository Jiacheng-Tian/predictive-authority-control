import unittest

import numpy as np


class ControllerPresetContractTest(unittest.TestCase):
    def test_current_real10kg_presets_construct_and_compute_finite_actions(self):
        from pac.controllers.presets import (
            build_real10kg_mpc_event,
            build_real10kg_smc_steady,
        )

        target = np.zeros(6, dtype=float)
        eta = np.zeros(6, dtype=float)
        nu = np.zeros(6, dtype=float)
        current = np.array([0.2, -0.1, 0.05], dtype=float)

        for builder in [build_real10kg_smc_steady, build_real10kg_mpc_event]:
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

    def test_legacy_alias_and_true_mpc_preset_are_distinct(self):
        from pac.controllers.legacy_predictive import LegacyOneStepPredictiveController
        from pac.controllers.mpc import MPCController
        from pac.controllers.presets import (
            build_legacy_one_step_predictive_v2,
            build_real10kg_mpc_ltv_v3,
        )

        self.assertIsInstance(build_legacy_one_step_predictive_v2(), LegacyOneStepPredictiveController)
        true_mpc = build_real10kg_mpc_ltv_v3()
        self.assertIsInstance(true_mpc, MPCController)
        self.assertEqual(true_mpc.settings.N, 20)

    def test_controller_registry_accepts_legacy_and_true_mpc_ids(self):
        from pac.evaluation.episodes import build_controller

        old_name, old_controller = build_controller("real10kg_mpc_event")
        new_name, new_controller = build_controller("real10kg_mpc_ltv_v3")
        self.assertEqual(old_name, "real10kg_mpc_event")
        self.assertEqual(new_name, "real10kg_mpc_ltv_v3")
        self.assertNotEqual(type(old_controller), type(new_controller))


if __name__ == "__main__":
    unittest.main()
