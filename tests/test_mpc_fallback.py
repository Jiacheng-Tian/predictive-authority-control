from __future__ import annotations

import unittest

import numpy as np


class _FailingSolver:
    def __init__(self):
        self.calls = 0

    def solve(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise RuntimeError("forced solver failure")

    def reset(self):
        pass


class _Fallback:
    def __init__(self):
        self.calls = 0

    def compute(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return np.full(6, 0.25)

    def reset(self):
        self.calls = 0


class MPCFallbackTest(unittest.TestCase):
    def test_acceptance_recomputes_status_and_all_solution_residuals(self):
        from pac.controllers.mpc import MPCController
        from pac.controllers.mpc_qp import MPCQPSettings, MPCSolution
        from pac.simulation.dynamics import AUVDynamics
        from pac.simulation.thrusters import build_real_10kg_x_layout

        controller = MPCController(
            dynamics=AUVDynamics(dt=0.01),
            thruster_layout=build_real_10kg_x_layout(),
            settings=MPCQPSettings(N=1),
            fallback_controller=_Fallback(),
        )
        valid_plan = np.zeros((1, 6))
        valid_states = np.zeros((2, 12))
        valid = MPCSolution(
            plan=valid_plan,
            state_plan=valid_states,
            status="solved",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=0.0,
            warm_started=False,
            constraint_violation=0.0,
            accepted=False,
            primal_residual=0.0,
            dual_residual=0.0,
        )
        self.assertTrue(controller._solution_is_accepted(valid))
        valid_bad_status = MPCSolution(
            plan=valid_plan,
            state_plan=valid_states,
            status="maximum iterations reached",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=0.0,
            warm_started=False,
            constraint_violation=0.0,
            accepted=True,
            primal_residual=0.0,
            dual_residual=0.0,
        )
        self.assertFalse(controller._solution_is_accepted(valid_bad_status))
        inaccurate_missing_dual = MPCSolution(
            plan=valid_plan,
            state_plan=valid_states,
            status="solved inaccurate",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=0.0,
            warm_started=False,
            constraint_violation=0.0,
            primal_residual=0.0,
            dual_residual=float("inf"),
        )
        self.assertFalse(controller._solution_is_accepted(inaccurate_missing_dual))
        aggregate_only = MPCSolution(
            plan=valid_plan,
            state_plan=valid_states,
            status="solved inaccurate",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=5.0e-4,
            warm_started=False,
            constraint_violation=0.0,
        )
        self.assertTrue(controller._solution_is_accepted(aggregate_only))
        invalid_state = MPCSolution(
            plan=valid_plan,
            state_plan=np.full((2, 12), np.nan),
            status="solved",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=0.0,
            warm_started=False,
            constraint_violation=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
        )
        self.assertFalse(controller._solution_is_accepted(invalid_state))
        invalid_plan = MPCSolution(
            plan=np.full((1, 6), np.nan),
            state_plan=valid_states,
            status="solved",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=0.0,
            warm_started=False,
            constraint_violation=0.0,
            primal_residual=0.0,
            dual_residual=0.0,
        )
        self.assertFalse(controller._solution_is_accepted(invalid_plan))
        invalid_constraint = MPCSolution(
            plan=valid_plan,
            state_plan=valid_states,
            status="solved",
            iter=1,
            run_time=0.0,
            wall_time=0.0,
            residual=0.0,
            warm_started=False,
            constraint_violation=1.0e-2,
            primal_residual=0.0,
            dual_residual=0.0,
        )
        self.assertFalse(controller._solution_is_accepted(invalid_constraint))

    def test_cross_angle_linearization_keeps_affine_offset_continuous(self):
        from pac.controllers.mpc import MPCController
        from pac.controllers.mpc_qp import MPCQPSettings
        from pac.simulation.dynamics import AUVDynamics
        from pac.simulation.thrusters import build_real_10kg_x_layout

        controller = MPCController(
            dynamics=AUVDynamics(dt=0.01),
            thruster_layout=build_real_10kg_x_layout(),
            settings=MPCQPSettings(N=1),
            fallback_controller=_Fallback(),
        )
        eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.14])
        nu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        action = np.zeros(6)
        A, B, affine = controller.linearize(eta, nu, action, np.zeros(3))
        self.assertLess(abs(float(affine[5])), 1.0)
        perturbation = np.zeros(12)
        perturbation[5] = 1.0e-6
        state = np.r_[eta, nu]
        actual = controller._transition(state + perturbation, action, np.zeros(3))
        actual[5] = eta[5] + controller._wrap_angle(actual[5] - eta[5])
        predicted = A @ (state + perturbation) + B @ action + affine
        self.assertAlmostEqual(float(predicted[5]), float(actual[5]), places=5)

    def test_reference_and_linearization_are_finite_and_use_direct_wrench_map(self):
        from pac.controllers.mpc import MPCController
        from pac.controllers.mpc_qp import MPCQPSettings
        from pac.simulation.dynamics import AUVDynamics
        from pac.simulation.thrusters import build_real_10kg_x_layout

        layout = build_real_10kg_x_layout()
        controller = MPCController(
            dynamics=AUVDynamics(dt=0.01),
            thruster_layout=layout,
            settings=MPCQPSettings(N=2),
            fallback_controller=_Fallback(),
        )
        eta = np.array([0.2, -0.1, 0.05, 0.02, -0.01, 3.12])
        nu = np.array([0.3, -0.2, 0.1, 0.01, -0.02, 0.03])
        action = np.array([0.1, -0.1, 0.05, 0.0, 0.02, -0.03])
        reference = controller._reference_sequence(4.0, eta)
        self.assertEqual(reference.shape, (3, 12))
        self.assertTrue(np.isfinite(reference).all())
        self.assertTrue(np.all(np.abs(np.diff(reference[:, 5])) < np.pi))

        A, B, c = controller._linearize(eta, nu, action, np.array([0.4, -0.15, 0.75]))
        self.assertEqual(A.shape, (12, 12))
        self.assertEqual(B.shape, (12, 6))
        self.assertTrue(np.isfinite(np.concatenate([A.ravel(), B.ravel(), c])).all())
        expected_wrench = layout.allocation_matrix @ (layout.max_force * action)
        np.testing.assert_allclose(
            controller._transition(np.concatenate([eta, nu]), action, np.array([0.4, -0.15, 0.75])),
            np.concatenate(controller.dynamics.predict_step(eta, nu, expected_wrench, np.array([0.4, -0.15, 0.75]))),
        )

    def test_reuses_plan_three_times_then_uses_smc_and_reset_clears_state(self):
        from pac.controllers.mpc import MPCController
        from pac.controllers.mpc_qp import MPCSolution, MPCQPSettings
        from pac.simulation.dynamics import AUVDynamics
        from pac.simulation.thrusters import build_real_10kg_x_layout

        class _OneGoodThenFail:
            def __init__(self):
                self.calls = 0

            def solve(self, *args, **kwargs):
                del args, kwargs
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("forced solver failure")
                return MPCSolution(
                    plan=np.full((20, 6), 0.08),
                    state_plan=np.zeros((21, 12)),
                    status="solved",
                    iter=1,
                    run_time=0.0,
                    wall_time=0.0,
                    residual=0.0,
                    warm_started=False,
                    constraint_violation=0.0,
                    accepted=True,
                )

            def reset(self):
                pass

        fallback = _Fallback()
        controller = MPCController(
            dynamics=AUVDynamics(dt=0.01),
            thruster_layout=build_real_10kg_x_layout(),
            solver=_OneGoodThenFail(),
            settings=MPCQPSettings(N=20),
            fallback_controller=fallback,
        )
        zeros = np.zeros(6)
        controller.compute(zeros, zeros, zeros, t=0.0)
        self.assertIn("primal_residual", controller.last_telemetry)
        self.assertIn("dual_residual", controller.last_telemetry)
        for _ in range(3):
            action = controller.compute(zeros, zeros, zeros, t=0.0)
            self.assertFalse(controller.last_telemetry["force_primary_authority"])
            self.assertIn("primal_residual", controller.last_telemetry)
            self.assertIn("dual_residual", controller.last_telemetry)
            self.assertTrue(np.isfinite(action).all())
        controller.compute(zeros, zeros, zeros, t=0.0)
        self.assertEqual(controller.last_telemetry["fallback_mode"], "smc")
        self.assertTrue(controller.last_telemetry["force_primary_authority"])
        self.assertGreater(fallback.calls, 0)

        controller.reset()
        self.assertEqual(controller.last_telemetry["consecutive_failures"], 0)
        self.assertFalse(controller.force_primary_authority)


if __name__ == "__main__":
    unittest.main()
