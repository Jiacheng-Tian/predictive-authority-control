"""True finite-horizon linearized MPC controller."""

from __future__ import annotations

import numpy as np

from pac.controllers.mpc_qp import LinearMPCQP, MPCQPSettings, MPCSolution
from pac.controllers.smc import SMCController
from pac.simulation.dynamics import AUVDynamics
from pac.simulation.thrusters import ThrusterLayout, build_real_10kg_x_layout
from pac.simulation.vehicle_profiles import get_vehicle_profile


class _RejectedSolution(RuntimeError):
    def __init__(self, solution: MPCSolution):
        super().__init__(f"MPC solver rejected status {solution.status!r}")
        self.solution = solution


class MPCController:
    """Finite-horizon MPC with bounded normalized thruster commands."""

    state_eps = 1.0e-5
    control_eps = 1.0e-4

    def __init__(
            self,
            *,
            dynamics: AUVDynamics | None = None,
            thruster_layout: ThrusterLayout | None = None,
            settings: MPCQPSettings | None = None,
            solver: LinearMPCQP | None = None,
            fallback_controller=None,
            smc_fallback=None,
            vehicle_profile: str = "real_10kg_v1",
            dt: float = 0.01,
            M=None,
            D=None,
            D_quad=None,
            q_pos: float | None = None,
            q_att: float | None = None,
            q_nu: float | None = None,
            z_pos_gain: float | None = None,
            z_current_feedforward: float | None = None,
            r: float | None = None):
        del M, D, D_quad, q_pos, q_att, q_nu, z_pos_gain, z_current_feedforward, r
        self.dynamics = dynamics if dynamics is not None else AUVDynamics(
            vehicle_profile=vehicle_profile,
            dt=dt,
        )
        self.thruster_layout = (
            thruster_layout if thruster_layout is not None else build_real_10kg_x_layout()
        )
        self.settings = settings if settings is not None else MPCQPSettings()
        self.solver = solver if solver is not None else LinearMPCQP(self.settings)
        if fallback_controller is not None and smc_fallback is not None:
            raise TypeError("pass fallback_controller or smc_fallback, not both")
        self.fallback_controller = (
            fallback_controller if fallback_controller is not None else smc_fallback
        )
        if self.fallback_controller is None:
            self.fallback_controller = self._build_default_smc_fallback()
        self._trajectory3d = True
        self._last_action = np.zeros(6, dtype=float)
        self._last_plan: np.ndarray | None = None
        self._consecutive_failures = 0
        self._force_primary_authority = False
        self.last_telemetry: dict[str, object] = {}
        self._set_telemetry(
            solver_status="idle",
            accepted=False,
            iterations=0,
            wall_time=0.0,
            run_time=0.0,
            residual=float("inf"),
            warm_started=False,
            fallback_mode="none",
            consecutive_failures=0,
            deadline_missed=False,
            constraint_violation=0.0,
        )

    def _build_default_smc_fallback(self):
        profile = get_vehicle_profile("real_10kg_v1")
        fallback = SMCController(
            lambda_gain=[1.5, 1.5, 1.4, 1.0, 1.0, 1.4],
            reaching_gain=[2.0, 2.0, 4.0, 0.0, 0.0, 4.0],
            robust_gain=[0.0] * 6,
            equivalent_mass=profile.effective_mass,
            equivalent_damping=profile.linear_damping,
            equivalent_quadratic_damping=profile.quadratic_damping,
            current_feedforward=1.0,
            thruster_layout=self.thruster_layout,
            control_yaw=True,
        )
        fallback.set_trajectory3d(True)
        return fallback

    @property
    def force_primary_authority(self) -> bool:
        return bool(self._force_primary_authority)

    @property
    def force_primary(self) -> bool:
        """Compatibility alias used by authority mixers."""
        return self.force_primary_authority

    @property
    def last_plan(self) -> np.ndarray | None:
        return None if self._last_plan is None else self._last_plan.copy()

    def _set_telemetry(self, **values) -> None:
        values["force_primary_authority"] = self.force_primary_authority
        self.last_telemetry = dict(values)

    def reset(self) -> None:
        self._last_action.fill(0.0)
        self._last_plan = None
        self._consecutive_failures = 0
        self._force_primary_authority = False
        reset_solver = getattr(self.solver, "reset", None)
        if callable(reset_solver):
            reset_solver()
        reset_fallback = getattr(self.fallback_controller, "reset", None)
        if callable(reset_fallback):
            reset_fallback()
        self._set_telemetry(
            solver_status="idle",
            accepted=False,
            iterations=0,
            wall_time=0.0,
            run_time=0.0,
            residual=float("inf"),
            warm_started=False,
            fallback_mode="none",
            consecutive_failures=0,
            deadline_missed=False,
            constraint_violation=0.0,
        )

    def set_trajectory3d(self, enabled: bool = True) -> None:
        self._trajectory3d = bool(enabled)
        setter = getattr(self.fallback_controller, "set_trajectory3d", None)
        if callable(setter):
            setter(enabled)

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _wrapped_state_difference(cls, positive, negative):
        delta = np.asarray(positive, dtype=float) - np.asarray(negative, dtype=float)
        delta = delta.copy()
        delta[3:6] = [cls._wrap_angle(value) for value in delta[3:6]]
        return delta

    @staticmethod
    def _world_xy_to_body(xy, psi):
        cp, sp = np.cos(psi), np.sin(psi)
        return np.array([cp * xy[0] + sp * xy[1], -sp * xy[0] + cp * xy[1]])

    @staticmethod
    def _target_kinematics(t):
        vx = 0.9 * np.cos(0.3 * t)
        vy = 0.9 * np.cos(0.6 * t)
        ax = -0.27 * np.sin(0.3 * t)
        ay = -0.54 * np.sin(0.6 * t)
        psi = np.arctan2(vy, vx)
        denom = max(vx * vx + vy * vy, 1.0e-9)
        yaw_rate = (vx * ay - vy * ax) / denom
        return vx, vy, psi, yaw_rate

    def _reference_sequence(self, t: float, current_eta: np.ndarray) -> np.ndarray:
        dt = float(getattr(self.dynamics, "dt", 0.01))
        times = float(t) + np.arange(self.settings.N + 1, dtype=float) * dt
        refs = np.zeros((self.settings.N + 1, 12), dtype=float)
        raw_yaw = np.empty(self.settings.N + 1, dtype=float)
        for index, stamp in enumerate(times):
            x = 3.0 * np.sin(0.3 * stamp)
            y = 1.5 * np.sin(0.6 * stamp)
            z = 0.8 * np.sin(0.2 * stamp)
            vx, vy, yaw, yaw_rate = self._target_kinematics(stamp)
            raw_yaw[index] = yaw
            refs[index, :6] = [x, y, z, 0.0, 0.0, yaw]
            body_velocity = self._world_xy_to_body([vx, vy], yaw)
            refs[index, 6:9] = [body_velocity[0], body_velocity[1], 0.16 * np.cos(0.2 * stamp)]
            refs[index, 9:] = [0.0, 0.0, yaw_rate]

        unwrapped = np.unwrap(raw_yaw)
        yaw_delta = unwrapped - unwrapped[0]
        yaw_offset = self._wrap_angle(unwrapped[0] - float(current_eta[5]))
        refs[:, 5] = float(current_eta[5]) + yaw_offset + yaw_delta
        return refs

    def reference_sequence(self, t: float, current_eta: np.ndarray) -> np.ndarray:
        """Return the finite-horizon trajectory reference used by the QP."""
        return self._reference_sequence(t, current_eta).copy()

    def _current_vector(self, current_prediction) -> np.ndarray:
        if current_prediction is None:
            return np.zeros(3, dtype=float)
        current = np.asarray(current_prediction, dtype=float).reshape(-1)
        if current.size < 2 or not np.isfinite(current).all():
            raise ValueError("current_prediction must contain finite surge/sway current")
        if current.size >= 3:
            return current[:3].copy()
        return np.array([current[0], current[1], 0.0], dtype=float)

    def _transition(self, state, action, current) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(12)
        action = np.asarray(action, dtype=float).reshape(6)
        max_force = float(getattr(self.thruster_layout, "max_force", 35.0))
        wrench = self.thruster_layout.allocation_matrix @ (max_force * action)
        next_eta, next_nu = self.dynamics.predict_step(
            state[:6], state[6:], wrench, current
        )
        return np.concatenate([np.asarray(next_eta, dtype=float), np.asarray(next_nu, dtype=float)])

    def _linearize(self, eta, nu, action, current_prediction=None):
        eta = np.asarray(eta, dtype=float).reshape(6)
        nu = np.asarray(nu, dtype=float).reshape(6)
        if not np.isfinite(eta).all() or not np.isfinite(nu).all():
            raise ValueError("current state must contain only finite values")
        action = np.asarray(action, dtype=float).reshape(6)
        if not np.isfinite(action).all():
            raise ValueError("previous authority command must contain only finite values")
        current = self._current_vector(current_prediction)
        state = np.concatenate([eta, nu])
        f0 = self._transition(state, action, current)
        A = np.empty((12, 12), dtype=float)
        for index in range(12):
            perturbation = np.zeros(12, dtype=float)
            perturbation[index] = self.state_eps
            fp = self._transition(state + perturbation, action, current)
            fm = self._transition(state - perturbation, action, current)
            A[:, index] = self._wrapped_state_difference(fp, fm) / (2.0 * self.state_eps)
        B = np.empty((12, 6), dtype=float)
        for index in range(6):
            perturbation = np.zeros(6, dtype=float)
            perturbation[index] = self.control_eps
            fp = self._transition(state, action + perturbation, current)
            fm = self._transition(state, action - perturbation, current)
            B[:, index] = self._wrapped_state_difference(fp, fm) / (2.0 * self.control_eps)
        c = f0 - A @ state - B @ action
        return A, B, c

    def linearize(self, eta, nu, previous_action, current_prediction=None):
        """Return the frozen discrete transition ``A, B, c`` for one horizon."""
        return self._linearize(eta, nu, previous_action, current_prediction)

    def _project_sequence(self, plan, previous=None) -> np.ndarray:
        values = np.asarray(plan, dtype=float).reshape(self.settings.N, 6).copy()
        previous = self._last_action.copy() if previous is None else np.asarray(previous, dtype=float).reshape(6)
        projected = np.empty_like(values)
        for index, row in enumerate(values):
            clipped = np.clip(row, self.settings.u_min, self.settings.u_max)
            clipped = np.clip(
                clipped,
                previous - self.settings.slew_limit,
                previous + self.settings.slew_limit,
            )
            projected[index] = clipped
            previous = clipped
        return projected

    def _reused_plan(self) -> np.ndarray:
        old = self._last_plan
        if old is None:
            raise RuntimeError("no plan available for reuse")
        shifted = np.empty_like(old)
        shifted[:-1] = old[1:]
        shifted[-1] = old[-1]
        return self._project_sequence(shifted, self._last_action)

    @staticmethod
    def _solution_is_accepted(solution) -> bool:
        accepted = getattr(solution, "accepted", None)
        if accepted is not None:
            return bool(accepted)
        status = str(getattr(solution, "status", "")).strip().lower()
        if status == "solved":
            return True
        if status == "solved inaccurate":
            residual = float(getattr(solution, "residual", np.inf))
            return residual <= 1.0e-3
        return False

    def _record_solution_telemetry(self, solution, fallback_mode: str) -> None:
        wall_time = float(getattr(solution, "wall_time", 0.0))
        run_time = float(getattr(solution, "run_time", 0.0))
        self._set_telemetry(
            solver_status=str(getattr(solution, "status", "unknown")),
            accepted=self._solution_is_accepted(solution),
            iterations=int(getattr(solution, "iter", getattr(solution, "iterations", 0))),
            wall_time=wall_time,
            run_time=run_time,
            residual=float(getattr(solution, "residual", np.inf)),
            warm_started=bool(getattr(solution, "warm_started", False)),
            fallback_mode=fallback_mode,
            consecutive_failures=self._consecutive_failures,
            deadline_missed=max(wall_time, run_time) > float(self.settings.time_limit_s),
            constraint_violation=float(getattr(solution, "constraint_violation", np.inf)),
        )

    def compute(
            self,
            target_eta,
            current_eta,
            current_nu,
            *,
            t=None,
            current_prediction=None,
            current_estimate=None):
        """Return the first normalized action from the finite-horizon plan."""
        target_eta = np.asarray(target_eta, dtype=float).reshape(-1)
        eta = np.asarray(current_eta, dtype=float).reshape(-1)
        nu = np.asarray(current_nu, dtype=float).reshape(-1)
        if target_eta.shape != (6,) or eta.shape != (6,) or nu.shape != (6,):
            raise ValueError("target_eta, current_eta, and current_nu must have shape (6,)")
        if not np.isfinite(target_eta).all() or not np.isfinite(eta).all() or not np.isfinite(nu).all():
            raise ValueError("controller inputs must contain only finite values")
        stamp = float(t or 0.0)
        if not np.isfinite(stamp):
            raise ValueError("t must be finite")
        if current_prediction is not None and current_estimate is not None:
            raise TypeError("pass current_prediction or current_estimate, not both")
        if current_prediction is None:
            current_prediction = current_estimate
        current = self._current_vector(current_prediction)
        solution = None
        try:
            A, B, c = self._linearize(eta, nu, self._last_action, current)
            xref = self._reference_sequence(stamp, eta)
            solution = self.solver.solve(
                A,
                B,
                c,
                np.concatenate([eta, nu]),
                xref,
                self._last_action,
            )
            if not self._solution_is_accepted(solution):
                raise _RejectedSolution(solution)
            plan = np.asarray(solution.plan, dtype=float)
            if plan.shape != (self.settings.N, 6) or not np.isfinite(plan).all():
                raise RuntimeError("MPC solver returned an invalid control plan")
            plan = self._project_sequence(plan, self._last_action)
            action = plan[0].copy()
            self._last_plan = plan.copy()
            self._consecutive_failures = 0
            self._force_primary_authority = False
            self._record_solution_telemetry(solution, "none")
        except Exception as error:
            if isinstance(error, _RejectedSolution):
                solution = error.solution
                self._record_solution_telemetry(solution, "solver_failure")
            self._consecutive_failures += 1
            if self._last_plan is not None and self._consecutive_failures <= 3:
                plan = self._reused_plan()
                self._last_plan = plan.copy()
                action = plan[0].copy()
                self._force_primary_authority = False
                if solution is not None:
                    self._record_solution_telemetry(solution, "reuse_plan")
                else:
                    self._set_telemetry(
                        solver_status="error",
                        accepted=False,
                        iterations=0,
                        wall_time=0.0,
                        run_time=0.0,
                        residual=float("inf"),
                        warm_started=False,
                        fallback_mode="reuse_plan",
                        consecutive_failures=self._consecutive_failures,
                        deadline_missed=False,
                        constraint_violation=float("inf"),
                    )
            else:
                action = np.asarray(
                    self.fallback_controller.compute(
                        target_eta,
                        eta,
                        nu,
                        t=stamp,
                        current_prediction=current,
                    ),
                    dtype=float,
                ).reshape(6)
                if not np.isfinite(action).all():
                    raise ValueError("SMC fallback returned non-finite action") from error
                action = np.clip(action, self.settings.u_min, self.settings.u_max)
                self._force_primary_authority = True
                if solution is not None:
                    self._record_solution_telemetry(solution, "smc")
                else:
                    self._set_telemetry(
                        solver_status="error",
                        accepted=False,
                        iterations=0,
                        wall_time=0.0,
                        run_time=0.0,
                        residual=float("inf"),
                        warm_started=False,
                        fallback_mode="smc",
                        consecutive_failures=self._consecutive_failures,
                        deadline_missed=False,
                        constraint_violation=float("inf"),
                    )

        self._last_action = np.asarray(action, dtype=float).copy()
        self.last_telemetry["force_primary_authority"] = self.force_primary_authority
        return self._last_action.copy()
