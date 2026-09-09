from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class FormalV2NumericalCompatibilityTest(unittest.TestCase):
    def test_dynamics_step_matches_archived_semantics(self):
        from pac.simulation.dynamics import AUVDynamics

        dynamics = AUVDynamics()
        dynamics.eta = np.array([0.3, -0.2, 0.1, 0.05, -0.03, 0.4])
        dynamics.nu = np.array([0.7, -0.4, 0.2, 0.1, -0.3, 0.5])

        eta, nu = dynamics.step(
            np.array([1.0, -2.0, 3.0, 0.4, -0.5, 0.6]),
            np.array([0.2, -0.1, 0.05]),
        )

        np.testing.assert_allclose(eta, [
            0.3079593166018652,
            -0.20104629433182067,
            0.10202640300605163,
            0.05080754038731561,
            -0.03335770600327681,
            0.4048258710074988,
        ], rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(nu, [
            0.6948896163127599,
            -0.396748969804717,
            0.20198647738038886,
            0.09206166020835055,
            -0.32176821911815834,
            0.49733413048118236,
        ], rtol=0.0, atol=1.0e-12)

    def test_formal_controller_actions_match_archived_semantics(self):
        from pac.controllers.presets import (
            build_real10kg_mpc_event,
            build_real10kg_smc_steady,
        )

        target = np.array([1.1, -0.7, 0.35, 0.0, 0.0, 0.25])
        eta = np.array([0.2, -0.1, 0.05, 0.02, -0.01, 0.4])
        nu = np.array([0.3, -0.2, 0.1, 0.01, -0.02, 0.03])
        current = np.array([0.4, -0.15, 0.75])

        smc = build_real10kg_smc_steady()
        mpc = build_real10kg_mpc_event()
        smc.set_trajectory3d(True)
        mpc.set_trajectory3d(True)

        np.testing.assert_allclose(
            smc.compute(target, eta, nu, t=4.0, current_prediction=current),
            [-1.0, 1.0, 1.0, -1.0, 0.3895433815091396, 0.3895433815091397],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            mpc.compute(target, eta, nu, t=4.0, current_prediction=current),
            [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_alpha_filter_order_is_unchanged(self):
        from pac.authority.evaluation import filter_authority_alpha

        filtered = filter_authority_alpha(
            0.8 * 1.2,
            0.2,
            smoothing=0.5,
            rate_limit=0.0125,
            deadband=0.0,
        )
        self.assertAlmostEqual(filtered, 0.2125, places=15)

    def test_all_formal_checkpoints_load_and_infer(self):
        from pac.authority.model import load_alpha_model_checkpoint

        expected = {
            20: 0.012675686739385128,
            21: 0.003647681325674057,
            22: 0.007793508004397154,
            23: 0.0072799017652869225,
            24: 0.00868904497474432,
        }
        for seed, expected_output in expected.items():
            path = (
                ROOT
                / "results"
                / "formal_seeded_v2"
                / f"pac_train_seed_{seed}"
                / "predictive_alpha_model.pt"
            )
            model, metadata = load_alpha_model_checkpoint(
                path,
                expected_feature_mode="state_phase",
            )
            self.assertEqual(metadata["policy_architecture"], "transformer")
            self.assertEqual(metadata["history_len"], 16)
            self.assertEqual(metadata["input_dim"], 24)
            self.assertEqual(sum(param.numel() for param in model.parameters()), 14113)
            actual = float(model(torch.zeros(1, 16, 24)).item())
            np.testing.assert_allclose(
                actual,
                expected_output,
                rtol=1.0e-5,
                atol=1.0e-7,
            )

    def test_seed20_scenario1_episode_matches_archived_metrics(self):
        from pac.authority.evaluation import run_predictive_alpha_episode
        from pac.authority.model import load_alpha_model_checkpoint

        checkpoint = (
            ROOT
            / "results"
            / "formal_seeded_v2"
            / "pac_train_seed_20"
            / "predictive_alpha_model.pt"
        )
        model, metadata = load_alpha_model_checkpoint(
            checkpoint,
            expected_feature_mode="state_phase",
        )
        metrics = run_predictive_alpha_episode(
            model=model,
            scenario=1,
            seed=20000,
            steps=2100,
            mass_scale_xy=1.0,
            damping_scale_xy=1.0,
            current_amplitude_scale=2.5,
            current_frequency_scale=1.0,
            vertical_current=0.75,
            primary_controller="real10kg_smc_steady",
            authority_controller="real10kg_mpc_event",
            feature_mode="state_phase",
            initial_position_std=0.03,
            initial_velocity_std=0.01,
            alpha_gain=1.2,
            alpha_threshold=0.0,
            vehicle_profile="real_10kg_v1",
            thruster_layout="real_10kg_x",
            alpha_smoothing=0.5,
            alpha_rate_limit=0.0125,
            alpha_deadband=0.0,
            policy_architecture=metadata["policy_architecture"],
            history_len=metadata["history_len"],
            save_ts=False,
        )

        np.testing.assert_allclose(
            [
                metrics["rmse_3d"],
                metrics["max_error"],
                metrics["authority_alpha_mean"],
                metrics["action_jerk_mean"],
            ],
            [
                0.10089587780115189,
                0.2313652674301271,
                0.2534144150222329,
                0.0026697128616254644,
            ],
            rtol=1.0e-4,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
