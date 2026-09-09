"""True finite-horizon linearized MPC controller."""

from __future__ import annotations

import time

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
            dt: float = 0.01):
        self.dynamics = dynamics if dynamics is not None else AUVDynamics(
            vehicle_profile=vehicle_profile,
            dt=dt,
        )
        self.thruster_layout = (
            thruster_layout if thruster_layout is not None else build_real_10kg_x_layout()
        )
        self.settings = settings if settings is not None else MPCQPSettings()
        self.solver = solver if solver is not None else LinearMPCQP(self.settings)
        self._validate_solver_compatibility(self.solver)
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
            primal_residual=float("inf"),
            dual_residual=float("inf"),
            warm_started=False,
            fallback_mode="none",
            consecutive_failures=0,
            constraint_violation=0.0,
            qp_deadline_missed=False,
            compute_wall_time_s=0.0,
            deadline_missed=False,
        )

    def _validate_solver_compatibility(self, solver) -> None:
        solver_n = getattr(solver, "N", None)
        if solver_n is not None and int(solver_n) != self.settings.N:
            raise ValueError("solver horizon does not match controller settings")
        solver_settings = getattr(solver, "settings", None)
        if solver_settings is None:
            return
        for name in ("N", "u_min", "u_max", "slew_limit", "accept_inaccurate_residual"):
            solver_value = getattr(solver_settings, name, None)
            if solver_value is None:
                continue
            controller_value = getattr(self.settings, name)
            if not np.isclose(float(solver_value), float(controller_value), rtol=0.0, atol=1.0e-12):
                raise ValueError(f"solver setting {name} does not match controller settings")

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
            primal_residual=float("inf"),
            dual_residual=float("inf"),
            warm_started=False,
            fallback_mode="none",
            consecutive_failures=0,
            constraint_violation=0.0,
            qp_deadline_missed=False,
            compute_wall_time_s=0.0,
            deadline_missed=False,
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

    def _reference_sequence(
            self,
            t: float,
            current_eta: np.ndarray,
            target_eta: np.ndarray | None = None) -> np.ndarray:
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
        if not self._trajectory3d:
            refs[:, 2] = float(current_eta[2])
            refs[:, 8] = 0.0
        if target_eta is not None:
            target = np.asarray(target_eta, dtype=float).reshape(6)
            if not np.isfinite(target).all():
                raise ValueError("target_eta must contain only finite values")
            nominal = refs[0, :6].copy()
            offset = target - nominal
            offset[3:6] = [self._wrap_angle(value) for value in offset[3:6]]
            if not self._trajectory3d:
                offset[2] = 0.0
            refs[:, :3] += offset[:3]
            refs[:, 3:6] += offset[3:6]
            if self._trajectory3d:
                for index in range(self.settings.N + 1):
                    body_velocity = self._world_xy_to_body(
                        [0.9 * np.cos(0.3 * times[index]), 0.9 * np.cos(0.6 * times[index])],
                        refs[index, 5],
                    )
                    refs[index, 6:8] = body_velocity
            else:
                refs[:, 8] = 0.0
        return refs

    def reference_sequence(
            self,
            t: float,
            current_eta: np.ndarray,
            target_eta: np.ndarray | None = None) -> np.ndarray:
        """Return the finite-horizon trajectory reference used by the QP."""
        return self._reference_sequence(t, current_eta, target_eta).copy()

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
        continuous_f0 = f0.copy()
        continuous_f0[3:6] = state[3:6] + [
            self._wrap_angle(value)
            for value in (f0[3:6] - state[3:6])
        ]
        c = continuous_f0 - A @ state - B @ action
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
    def _solution_constraint_violation(
            plan,
            state_plan,
            A=None,
            B=None,
            c=None,
            x0=None,
            u_prev=None,
            settings=None):
        if A is None or B is None or c is None or x0 is None or u_prev is None:
            try:
                return float(getattr(settings, "constraint_violation"))
            except (TypeError, ValueError, AttributeError):
                return float("inf")
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if settings is None:
            settings = MPCQPSettings()
        c = np.asarray(c, dtype=float).reshape(12)
        x0 = np.asarray(x0, dtype=float).reshape(12)
        u_prev = np.asarray(u_prev, dtype=float).reshape(6)
        violation = [
            float(np.max(np.abs(state_plan[0] - x0))),
            float(np.max(np.abs(state_plan[1:] - state_plan[:-1] @ A.T - plan @ B.T - c))),
            float(np.max(np.maximum(settings.u_min - plan, plan - settings.u_max))),
        ]
        deltas = np.diff(np.vstack([u_prev, plan]), axis=0)
        violation.append(float(np.max(np.abs(deltas) - settings.slew_limit)))
        return max(max(violation), 0.0)

    def _solution_is_accepted(
            self,
            solution,
            expected_N: int | None = None,
            A=None,
            B=None,
            c=None,
            x0=None,
            u_prev=None,
            settings=None) -> bool:
        status = str(getattr(solution, "status", "")).strip().lower()
        try:
            plan = np.asarray(getattr(solution, "plan"), dtype=float)
            state_plan = np.asarray(getattr(solution, "state_plan"), dtype=float)
        except (TypeError, ValueError, AttributeError):
            return False
        horizon = plan.shape[0] if expected_N is None and plan.ndim == 2 else expected_N
        if horizon is None or plan.shape != (int(horizon), 6) or state_plan.shape != (int(horizon) + 1, 12):
            return False
        if not np.isfinite(plan).all() or not np.isfinite(state_plan).all():
            return False
        if A is None or B is None or c is None or x0 is None or u_prev is None:
            try:
                violation = float(getattr(solution, "constraint_violation"))
            except (TypeError, ValueError, AttributeError):
                violation = float("inf")
        else:
            violation = MPCController._solution_constraint_violation(
                plan, state_plan, A, B, c, x0, u_prev, settings
            )
        settings = self.settings if settings is None else settings
        threshold = float(getattr(settings, "accept_inaccurate_residual", 1.0e-3))
        if not np.isfinite(violation) or violation > threshold:
            return False
        if status == "solved":
            return True
        if status != "solved inaccurate":
            return False
        aggregate = getattr(solution, "residual", None)
        try:
            aggregate = float(aggregate) if aggregate is not None else float("inf")
            primal_value = getattr(solution, "primal_residual", None)
            dual_value = getattr(solution, "dual_residual", None)
            primal = aggregate if primal_value is None else float(primal_value)
            dual = aggregate if dual_value is None else float(dual_value)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            np.isfinite(primal)
            and np.isfinite(dual)
            and primal <= threshold
            and dual <= threshold
        )

    def _record_solution_telemetry(
            self,
            solution,
            fallback_mode: str,
            accepted: bool | None = None,
            qp_wall_time: float | None = None,
            qp_run_time: float | None = None,
            constraint_violation: float | None = None) -> None:
        wall_time = float(getattr(solution, "wall_time", 0.0))
        run_time = float(getattr(solution, "run_time", 0.0))
        aggregate = float(getattr(solution, "residual", np.inf))
        primal_value = getattr(solution, "primal_residual", None)
        dual_value = getattr(solution, "dual_residual", None)
        primal_residual = aggregate if primal_value is None else float(primal_value)
        dual_residual = aggregate if dual_value is None else float(dual_value)
        self._set_telemetry(
            solver_status=str(getattr(solution, "status", "unknown")),
            accepted=(
                self._solution_is_accepted(solution, self.settings.N)
                if accepted is None else bool(accepted)
            ),
            iterations=int(getattr(solution, "iter", getattr(solution, "iterations", 0))),
            wall_time=wall_time,
            run_time=run_time,
            residual=aggregate,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            warm_started=bool(getattr(solution, "warm_started", False)),
            fallback_mode=fallback_mode,
            consecutive_failures=self._consecutive_failures,
            qp_deadline_missed=max(
                wall_time if qp_wall_time is None else qp_wall_time,
                run_time if qp_run_time is None else qp_run_time,
            ) > float(self.settings.time_limit_s),
            constraint_violation=(
                float(getattr(solution, "constraint_violation", np.inf))
                if constraint_violation is None else float(constraint_violation)
            ),
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
        compute_started = time.perf_counter()
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
        validation_violation = float("inf")
        qp_started = None
        qp_elapsed = None
        try:
            A, B, c = self._linearize(eta, nu, self._last_action, current)
            x0 = np.concatenate([eta, nu])
            previous_action = self._last_action.copy()
            xref = self._reference_sequence(stamp, eta, target_eta)
            qp_started = time.perf_counter()
            solution = self.solver.solve(
                A,
                B,
                c,
                x0,
                xref,
                previous_action,
            )
            qp_elapsed = time.perf_counter() - qp_started
            validation_violation = self._solution_constraint_violation(
                np.asarray(solution.plan, dtype=float),
                np.asarray(solution.state_plan, dtype=float),
                A,
                B,
                c,
                x0,
                previous_action,
                self.settings,
            )
            accepted = self._solution_is_accepted(
                solution,
                self.settings.N,
                A,
                B,
                c,
                x0,
                previous_action,
                self.settings,
            )
            if not accepted:
                raise _RejectedSolution(solution)
            plan = np.asarray(solution.plan, dtype=float)
            if plan.shape != (self.settings.N, 6) or not np.isfinite(plan).all():
                raise RuntimeError("MPC solver returned an invalid control plan")
            plan = self._project_sequence(plan, self._last_action)
            action = plan[0].copy()
            self._last_plan = plan.copy()
            self._consecutive_failures = 0
            self._force_primary_authority = False
            self._record_solution_telemetry(
                solution,
                "none",
                accepted=True,
                qp_wall_time=qp_elapsed,
                constraint_violation=validation_violation,
            )
        except Exception as error:
            if qp_started is not None and qp_elapsed is None:
                qp_elapsed = time.perf_counter() - qp_started
            reset_solver = getattr(self.solver, "reset", None)
            if callable(reset_solver):
                try:
                    reset_solver()
                except Exception:
                    pass
            if isinstance(error, _RejectedSolution):
                solution = error.solution
                self._record_solution_telemetry(
                    solution,
                    "solver_failure",
                    accepted=False,
                    qp_wall_time=qp_elapsed,
                    constraint_violation=validation_violation,
                )
            self._consecutive_failures += 1
            if (
                self._last_plan is not None
                and self._consecutive_failures <= self.settings.max_consecutive_plan_reuse
            ):
                plan = self._reused_plan()
                self._last_plan = plan.copy()
                action = plan[0].copy()
                self._force_primary_authority = False
                if solution is not None:
                    self._record_solution_telemetry(
                        solution,
                        "reuse_plan",
                        accepted=False,
                        qp_wall_time=qp_elapsed,
                        constraint_violation=validation_violation,
                    )
                else:
                    self._set_telemetry(
                        solver_status="error",
                        accepted=False,
                        iterations=0,
                        wall_time=0.0,
                        run_time=0.0,
                        residual=float("inf"),
                        primal_residual=float("inf"),
                        dual_residual=float("inf"),
                        warm_started=False,
                        fallback_mode="reuse_plan",
                        consecutive_failures=self._consecutive_failures,
                        deadline_missed=False,
                        constraint_violation=float("inf"),
                        qp_deadline_missed=bool(
                            qp_elapsed is not None
                            and qp_elapsed > float(self.settings.time_limit_s)
                        ),
                        compute_wall_time_s=0.0,
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
                    self._record_solution_telemetry(
                        solution,
                        "smc",
                        accepted=False,
                        qp_wall_time=qp_elapsed,
                        constraint_violation=validation_violation,
                    )
                else:
                    self._set_telemetry(
                        solver_status="error",
                        accepted=False,
                        iterations=0,
                        wall_time=0.0,
                        run_time=0.0,
                        residual=float("inf"),
                        primal_residual=float("inf"),
                        dual_residual=float("inf"),
                        warm_started=False,
                        fallback_mode="smc",
                        consecutive_failures=self._consecutive_failures,
                        deadline_missed=False,
                        constraint_violation=float("inf"),
                        qp_deadline_missed=bool(
                            qp_elapsed is not None
                            and qp_elapsed > float(self.settings.time_limit_s)
                        ),
                        compute_wall_time_s=0.0,
                    )

        self._last_action = np.asarray(action, dtype=float).copy()
        self.last_telemetry["force_primary_authority"] = self.force_primary_authority
        compute_wall_time_s = time.perf_counter() - compute_started
        self.last_telemetry["compute_wall_time_s"] = float(compute_wall_time_s)
        self.last_telemetry["deadline_missed"] = bool(
            compute_wall_time_s > float(getattr(self.dynamics, "dt", 0.01))
        )
        if "qp_deadline_missed" not in self.last_telemetry:
            self.last_telemetry["qp_deadline_missed"] = False
        return self._last_action.copy()
