from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class LinearMPCQPTest(unittest.TestCase):
    def test_settings_have_only_frozen_canonical_fields_and_bound_horizon(self):
        from pac.controllers.mpc_qp import MPCQPSettings

        settings = MPCQPSettings(N=10)
        self.assertEqual(settings.N, 10)
        self.assertEqual(settings.horizon, 10)
        self.assertEqual(settings.max_consecutive_plan_reuse, 3)
        with self.assertRaises(TypeError):
            MPCQPSettings(horizon=10)
        with self.assertRaises(TypeError):
            MPCQPSettings(q_terminal_diag=[1.0] * 12)
        with self.assertRaises(ValueError):
            MPCQPSettings(N=201)

    def test_large_horizon_uses_sparse_cost_without_dense_nvar_square_allocation(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        original_zeros = np.zeros

        def guarded_zeros(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2 and shape[0] > 1000 and shape[1] > 1000:
                raise AssertionError("dense nvar-square allocation")
            return original_zeros(shape, *args, **kwargs)

        with patch("pac.controllers.mpc_qp.np.zeros", side_effect=guarded_zeros):
            solver = LinearMPCQP(MPCQPSettings(N=200))
        self.assertTrue(solver._P.format == "csc")
        self.assertLess(solver._P.nnz, solver.nvar * 20)

    def test_qp_wall_time_includes_update_and_warm_start_work(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        solver = LinearMPCQP(MPCQPSettings(N=1))
        A = np.eye(12)
        B = np.zeros((12, 6))
        args = (A, B, np.zeros(12), np.zeros(12), np.zeros((2, 12)), np.zeros(6))
        original_update = solver._solver.update

        def slow_update(*args, **kwargs):
            import time
            time.sleep(0.002)
            return original_update(*args, **kwargs)

        solver._solver.update = slow_update
        solution = solver.solve(*args)
        self.assertGreaterEqual(solution.wall_time, 0.002)

    def test_solved_inaccurate_requires_both_residuals_but_solved_uses_plan_checks(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        def run(status, primal_residual, dual_residual):
            solver = LinearMPCQP(MPCQPSettings(N=1))
            A = np.eye(12)
            B = np.zeros((12, 6))
            args = (A, B, np.zeros(12), np.zeros(12), np.zeros((2, 12)), np.zeros(6))
            baseline = solver.solve(*args)
            raw = solver._previous_primal.copy()
            solver._solver.solve = lambda: SimpleNamespace(
                info=SimpleNamespace(
                    status=status,
                    iter=1,
                    run_time=0.0,
                    prim_res=primal_residual,
                    dual_res=dual_residual,
                ),
                x=raw,
            )
            return solver.solve(*args)

        self.assertTrue(run("solved inaccurate", 5.0e-4, 5.0e-4).accepted)
        self.assertFalse(run("solved inaccurate", 2.0e-3, 5.0e-4).accepted)
        self.assertFalse(run("solved inaccurate", 5.0e-4, 2.0e-3).accepted)
        self.assertTrue(run("solved", 2.0, 2.0).accepted)

    def test_rejected_raw_primal_does_not_replace_previous_warm_start(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        solver = LinearMPCQP(MPCQPSettings(N=1))
        A = np.eye(12)
        B = np.zeros((12, 6))
        args = (A, B, np.zeros(12), np.zeros(12), np.zeros((2, 12)), np.zeros(6))
        solver.solve(*args)
        previous = solver._previous_primal.copy()
        solver._solver.solve = lambda: SimpleNamespace(
            info=SimpleNamespace(
                status="solved",
                iter=1,
                run_time=0.0,
                prim_res=0.0,
                dual_res=0.0,
            ),
            x=np.full(solver.nvar, np.nan),
        )

        result = solver.solve(*args)

        self.assertFalse(result.accepted)
        np.testing.assert_array_equal(solver._previous_primal, previous)

    def test_finite_but_infeasible_raw_primal_is_rejected_and_not_warmed(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        solver = LinearMPCQP(MPCQPSettings(N=1))
        A = np.eye(12)
        B = np.zeros((12, 6))
        args = (A, B, np.zeros(12), np.ones(12), np.zeros((2, 12)), np.zeros(6))
        solver.solve(A, B, np.zeros(12), np.zeros(12), np.zeros((2, 12)), np.zeros(6))
        previous = solver._previous_primal.copy()
        solver._solver.solve = lambda: SimpleNamespace(
            info=SimpleNamespace(
                status="solved",
                iter=1,
                run_time=0.0,
                prim_res=0.0,
                dual_res=0.0,
            ),
            x=np.zeros(solver.nvar),
        )

        result = solver.solve(*args)

        self.assertFalse(result.accepted)
        self.assertGreater(result.constraint_violation, 1.0e-3)
        np.testing.assert_array_equal(solver._previous_primal, previous)

    def test_solution_respects_dynamics_amplitude_and_slew(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        settings = MPCQPSettings(N=3)
        solver = LinearMPCQP(settings)
        A = np.eye(12)
        B = np.zeros((12, 6))
        B[:6] = 0.1
        c = np.zeros(12)
        x0 = np.zeros(12)
        xref = np.zeros((4, 12))
        xref[:, 0] = 1.0
        u_prev = np.zeros(6)

        solution = solver.solve(A, B, c, x0, xref, u_prev)

        self.assertTrue(solution.accepted)
        self.assertEqual(solution.plan.shape, (3, 6))
        self.assertEqual(solution.state_plan.shape, (4, 12))
        np.testing.assert_allclose(
            solution.state_plan[1:],
            solution.state_plan[:-1] @ A.T + solution.plan @ B.T + c,
            atol=2e-4,
        )
        self.assertLessEqual(float(np.max(np.abs(solution.plan))), 1.0 + 1e-6)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(np.vstack([u_prev, solution.plan]), axis=0)))),
            0.1 + 1e-6,
        )

    def test_repeated_solve_reuses_setup_and_warm_starts_shifted_plan(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        solver = LinearMPCQP(MPCQPSettings(N=2))
        A = np.eye(12)
        B = np.zeros((12, 6))
        c = np.zeros(12)
        args = (A, B, c, np.zeros(12), np.zeros((3, 12)), np.zeros(6))

        first = solver.solve(*args)
        second = solver.solve(*args)

        self.assertEqual(solver.setup_count, 1)
        self.assertFalse(first.warm_started)
        self.assertTrue(second.warm_started)

    def test_invalid_dimensions_and_nonfinite_inputs_are_rejected(self):
        from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings

        solver = LinearMPCQP(MPCQPSettings(N=1))
        with self.assertRaises(ValueError):
            solver.solve(
                np.eye(11),
                np.zeros((12, 6)),
                np.zeros(12),
                np.zeros(12),
                np.zeros((2, 12)),
                np.zeros(6),
            )
        with self.assertRaises(ValueError):
            solver.solve(
                np.eye(12),
                np.zeros((12, 6)),
                np.zeros(12),
                np.full(12, np.nan),
                np.zeros((2, 12)),
                np.zeros(6),
            )


if __name__ == "__main__":
    unittest.main()
