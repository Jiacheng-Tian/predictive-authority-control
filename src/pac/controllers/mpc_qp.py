"""Sparse linear MPC quadratic program used by the true predictive controller."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import osqp
from scipy import sparse


NX = 12
NU = 6


def _default_q() -> tuple[float, ...]:
    return (80.0, 80.0, 120.0, 4.0, 2.0, 12.0, 8.0, 8.0, 12.0, 0.5, 0.2, 3.0)


def _default_r() -> tuple[float, ...]:
    return (0.05,) * NU


def _default_s() -> tuple[float, ...]:
    return (0.5,) * NU


@dataclass(frozen=True)
class MPCQPSettings:
    """Fixed dimensions, costs, limits, and OSQP settings for the MPC QP."""

    N: int = 20
    q_diag: tuple[float, ...] = field(default_factory=_default_q)
    terminal_scale: float = 5.0
    r_diag: tuple[float, ...] = field(default_factory=_default_r)
    s_diag: tuple[float, ...] = field(default_factory=_default_s)
    u_min: float = -1.0
    u_max: float = 1.0
    slew_limit: float = 0.10
    eps_abs: float = 1.0e-4
    eps_rel: float = 1.0e-4
    max_iter: int = 1000
    time_limit_s: float = 0.0075
    accept_inaccurate_residual: float = 1.0e-3
    horizon: int | None = None
    control_min: float | None = None
    control_max: float | None = None
    max_delta_per_step: float | None = None
    q_terminal_diag: tuple[float, ...] | None = None
    u_lower: float | None = None
    u_upper: float | None = None
    du_max: float | None = None
    command_min: float | None = None
    command_max: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or not isinstance(self.N, (int, np.integer)):
            raise ValueError("N must be an integer")
        n = int(self.N)
        if self.horizon is not None:
            if isinstance(self.horizon, bool) or not isinstance(self.horizon, (int, np.integer)):
                raise ValueError("horizon must be an integer")
            horizon = int(self.horizon)
            if n != 20 and n != horizon:
                raise ValueError("N and horizon disagree")
            n = horizon
            object.__setattr__(self, "N", n)
        if n <= 0:
            raise ValueError("N must be positive")
        object.__setattr__(self, "N", n)

        try:
            q = tuple(float(value) for value in self.q_diag)
            r = tuple(float(value) for value in self.r_diag)
            s = tuple(float(value) for value in self.s_diag)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("diagonal weights must be numeric vectors") from exc
        if len(q) != NX or len(r) != NU or len(s) != NU:
            raise ValueError("q_diag, r_diag, and s_diag have invalid dimensions")
        if not all(np.isfinite(value) and value >= 0.0 for value in (*q, *r, *s)):
            raise ValueError("diagonal weights must be finite and positive semidefinite")
        object.__setattr__(self, "q_diag", q)
        object.__setattr__(self, "r_diag", r)
        object.__setattr__(self, "s_diag", s)

        lower = self.u_min if self.control_min is None else self.control_min
        upper = self.u_max if self.control_max is None else self.control_max
        slew = self.slew_limit if self.max_delta_per_step is None else self.max_delta_per_step
        if self.u_lower is not None:
            lower = self.u_lower
        if self.u_upper is not None:
            upper = self.u_upper
        if self.du_max is not None:
            slew = self.du_max
        if self.command_min is not None:
            lower = self.command_min
        if self.command_max is not None:
            upper = self.command_max
        lower = float(lower)
        upper = float(upper)
        slew = float(slew)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("control bounds must be finite and ordered")
        if not np.isfinite(slew) or slew <= 0.0:
            raise ValueError("slew_limit must be finite and positive")
        object.__setattr__(self, "u_min", lower)
        object.__setattr__(self, "u_max", upper)
        object.__setattr__(self, "slew_limit", slew)
        if not np.isfinite(float(self.terminal_scale)) or float(self.terminal_scale) < 0.0:
            raise ValueError("terminal_scale must be finite and non-negative")
        object.__setattr__(self, "terminal_scale", float(self.terminal_scale))
        if self.q_terminal_diag is not None:
            terminal_diag = tuple(float(value) for value in self.q_terminal_diag)
            if len(terminal_diag) != NX or not all(
                np.isfinite(value) and value >= 0.0 for value in terminal_diag
            ):
                raise ValueError("q_terminal_diag must be finite and positive semidefinite")
        else:
            terminal_diag = tuple(value * self.terminal_scale for value in q)
        object.__setattr__(self, "q_terminal_diag", terminal_diag)
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)):
            raise ValueError("max_iter must be an integer")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", int(self.max_iter))
        for name in (
            "eps_abs",
            "eps_rel",
            "time_limit_s",
            "accept_inaccurate_residual",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def q(self) -> np.ndarray:
        return np.diag(self.q_diag)

    @property
    def q_terminal(self) -> np.ndarray:
        return np.diag(self.q_terminal_diag)

    @property
    def Q(self) -> np.ndarray:
        return self.q

    @property
    def QN(self) -> np.ndarray:
        return self.q_terminal

    @property
    def R(self) -> np.ndarray:
        return self.r

    @property
    def S(self) -> np.ndarray:
        return self.s

    @property
    def r(self) -> np.ndarray:
        return np.diag(self.r_diag)

    @property
    def s(self) -> np.ndarray:
        return np.diag(self.s_diag)

    @property
    def nx(self) -> int:
        return NX

    @property
    def nu(self) -> int:
        return NU


@dataclass(frozen=True)
class MPCSolution:
    """Result and diagnostics returned by :class:`LinearMPCQP`."""

    plan: np.ndarray
    state_plan: np.ndarray
    status: str
    iter: int
    run_time: float
    wall_time: float
    residual: float
    warm_started: bool
    constraint_violation: float
    accepted: bool | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None

    def __post_init__(self) -> None:
        plan = np.array(self.plan, dtype=float, copy=True)
        state_plan = np.array(self.state_plan, dtype=float, copy=True)
        plan.setflags(write=False)
        state_plan.setflags(write=False)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "state_plan", state_plan)
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "iter", int(self.iter))
        for name in ("run_time", "wall_time", "residual", "constraint_violation"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("primal_residual", "dual_residual"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else float(value))
        object.__setattr__(self, "warm_started", bool(self.warm_started))
        if self.accepted is not None:
            object.__setattr__(self, "accepted", bool(self.accepted))

    @property
    def iterations(self) -> int:
        return self.iter

    @property
    def control_plan(self) -> np.ndarray:
        return self.plan


class LinearMPCQP:
    """Reusable sparse OSQP model for a linear time-invariant MPC problem."""

    nx = NX
    nu = NU

    def __init__(self, settings: MPCQPSettings | None = None, **kwargs: Any) -> None:
        self.settings = settings if settings is not None else MPCQPSettings(**kwargs)
        if kwargs and settings is not None:
            raise TypeError("pass settings or setting keywords, not both")
        self.N = self.settings.N
        self.n_state_vars = (self.N + 1) * NX
        self.n_control_vars = self.N * NU
        self.nvar = self.n_state_vars + self.n_control_vars
        self._n_x0_rows = NX
        self._n_dyn_rows = self.N * NX
        self._n_bound_rows = self.N * NU
        self._n_slew_rows = self.N * NU
        self.ncon = self._n_x0_rows + self._n_dyn_rows + self._n_bound_rows + self._n_slew_rows
        self.setup_count = 0
        self._previous_primal: np.ndarray | None = None
        self._last_l = np.zeros(self.ncon, dtype=float)
        self._last_u = np.zeros(self.ncon, dtype=float)
        self._build_cost_matrix()
        self._build_constraint_pattern()
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=self._P,
            q=np.zeros(self.nvar),
            A=self._A_template,
            l=self._last_l,
            u=self._last_u,
            eps_abs=self.settings.eps_abs,
            eps_rel=self.settings.eps_rel,
            max_iter=self.settings.max_iter,
            polishing=False,
            warm_starting=True,
            time_limit=self.settings.time_limit_s,
            verbose=False,
        )
        self.setup_count += 1

    def _x_index(self, k: int, i: int) -> int:
        return k * NX + i

    def _u_index(self, k: int, i: int) -> int:
        return self.n_state_vars + k * NU + i

    def _build_cost_matrix(self) -> None:
        p = np.zeros((self.nvar, self.nvar), dtype=float)
        q_diag = np.asarray(self.settings.q_diag, dtype=float)
        for k in range(self.N):
            indices = [self._x_index(k, i) for i in range(NX)]
            p[np.ix_(indices, indices)] += 2.0 * np.diag(q_diag)
        terminal = [self._x_index(self.N, i) for i in range(NX)]
        p[np.ix_(terminal, terminal)] += 2.0 * np.diag(self.settings.q_terminal_diag)

        r_diag = np.asarray(self.settings.r_diag, dtype=float)
        s_diag = np.asarray(self.settings.s_diag, dtype=float)
        for k in range(self.N):
            for i in range(NU):
                index = self._u_index(k, i)
                p[index, index] += 2.0 * (r_diag[i] + s_diag[i])
                if k:
                    previous = self._u_index(k - 1, i)
                    p[previous, previous] += 2.0 * s_diag[i]
                    p[index, previous] -= 2.0 * s_diag[i]
                    p[previous, index] -= 2.0 * s_diag[i]
        self._P = sparse.csc_matrix(np.triu(p))

    def _build_constraint_pattern(self) -> None:
        rows: list[int] = []
        cols: list[int] = []
        row = 0
        for i in range(NX):
            rows.append(row)
            cols.append(self._x_index(0, i))
            row += 1

        for k in range(self.N):
            for i in range(NX):
                for j in range(NX):
                    rows.append(row)
                    cols.append(self._x_index(k, j))
                rows.append(row)
                cols.append(self._x_index(k + 1, i))
                for j in range(NU):
                    rows.append(row)
                    cols.append(self._u_index(k, j))
                row += 1

        for k in range(self.N):
            for i in range(NU):
                rows.append(row)
                cols.append(self._u_index(k, i))
                row += 1

        for k in range(self.N):
            for i in range(NU):
                rows.append(row)
                cols.append(self._u_index(k, i))
                if k:
                    rows.append(row)
                    cols.append(self._u_index(k - 1, i))
                row += 1

        data = np.ones(len(rows), dtype=float)
        self._A_template = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.ncon, self.nvar)
        ).tocsc()
        self._entry_keys: list[tuple[int, int]] = []
        for col in range(self.nvar):
            start, stop = self._A_template.indptr[col], self._A_template.indptr[col + 1]
            self._entry_keys.extend((int(self._A_template.indices[pos]), col) for pos in range(start, stop))

    def _validate_vector(self, value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _validate_inputs(self, A, B, c, x0, xref, u_prev):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.shape != (NX, NX):
            raise ValueError("A must have shape (12, 12)")
        if B.shape != (NX, NU):
            raise ValueError("B must have shape (12, 6)")
        if not np.isfinite(A).all() or not np.isfinite(B).all():
            raise ValueError("A and B must contain only finite values")
        c = self._validate_vector(c, (NX,), "c")
        x0 = self._validate_vector(x0, (NX,), "x0")
        xref = self._validate_vector(xref, (self.N + 1, NX), "xref")
        u_prev = self._validate_vector(u_prev, (NU,), "u_prev")
        if np.any(u_prev < self.settings.u_min) or np.any(u_prev > self.settings.u_max):
            raise ValueError("u_prev must satisfy control bounds")
        return A, B, c, x0, xref, u_prev

    def _constraint_values(self, A, B, c, x0, u_prev):
        values: dict[tuple[int, int], float] = {}
        row = 0
        for i in range(NX):
            values[(row, self._x_index(0, i))] = 1.0
            row += 1
        for k in range(self.N):
            for i in range(NX):
                for j in range(NX):
                    values[(row, self._x_index(k, j))] = -float(A[i, j])
                values[(row, self._x_index(k + 1, i))] = 1.0
                for j in range(NU):
                    values[(row, self._u_index(k, j))] = -float(B[i, j])
                row += 1
        for k in range(self.N):
            for i in range(NU):
                values[(row, self._u_index(k, i))] = 1.0
                row += 1
        for k in range(self.N):
            for i in range(NU):
                values[(row, self._u_index(k, i))] = 1.0
                if k:
                    values[(row, self._u_index(k - 1, i))] = -1.0
                row += 1
        ax = np.array([values.get(key, 0.0) for key in self._entry_keys], dtype=float)

        lower = np.empty(self.ncon, dtype=float)
        upper = np.empty(self.ncon, dtype=float)
        row = 0
        lower[:NX] = x0
        upper[:NX] = x0
        row += NX
        for _ in range(self.N):
            lower[row:row + NX] = c
            upper[row:row + NX] = c
            row += NX
        for _ in range(self.N):
            lower[row:row + NU] = self.settings.u_min
            upper[row:row + NU] = self.settings.u_max
            row += NU
        for k in range(self.N):
            if k == 0:
                lower[row:row + NU] = u_prev - self.settings.slew_limit
                upper[row:row + NU] = u_prev + self.settings.slew_limit
            else:
                lower[row:row + NU] = -self.settings.slew_limit
                upper[row:row + NU] = self.settings.slew_limit
            row += NU
        return ax, lower, upper

    def _linear_term(self, xref: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        linear = np.zeros(self.nvar, dtype=float)
        q = np.asarray(self.settings.q_diag, dtype=float)
        for k in range(self.N):
            linear[k * NX:(k + 1) * NX] = -2.0 * q * xref[k]
        start = self._x_index(self.N, 0)
        linear[start:start + NX] = -2.0 * np.asarray(self.settings.q_terminal_diag) * xref[self.N]
        s = np.asarray(self.settings.s_diag, dtype=float)
        linear[self.n_state_vars:self.n_state_vars + NU] = -2.0 * s * u_prev
        return linear

    @staticmethod
    def _constraint_violation(z: np.ndarray, matrix: sparse.csc_matrix, lower, upper) -> float:
        if z.size == 0 or not np.isfinite(z).all():
            return float("inf")
        az = matrix @ z
        return float(max(np.max(lower - az), np.max(az - upper), 0.0))

    def _shift_primal(self) -> np.ndarray | None:
        if self._previous_primal is None:
            return None
        old = self._previous_primal
        shifted = np.zeros_like(old)
        old_x = old[:self.n_state_vars].reshape(self.N + 1, NX)
        new_x = shifted[:self.n_state_vars].reshape(self.N + 1, NX)
        new_x[:-1] = old_x[1:]
        new_x[-1] = old_x[-1]
        old_u = old[self.n_state_vars:].reshape(self.N, NU)
        new_u = shifted[self.n_state_vars:].reshape(self.N, NU)
        new_u[:-1] = old_u[1:]
        new_u[-1] = old_u[-1]
        return shifted

    def solve(self, A, B, c, x0, xref=None, u_prev=None, *, x_ref=None) -> MPCSolution:
        """Solve one linearized MPC problem and return its complete diagnostics."""
        if xref is None:
            xref = x_ref
        if xref is None or u_prev is None:
            raise ValueError("xref and u_prev are required")
        A, B, c, x0, xref, u_prev = self._validate_inputs(A, B, c, x0, xref, u_prev)
        ax, lower, upper = self._constraint_values(A, B, c, x0, u_prev)
        linear = self._linear_term(xref, u_prev)
        warm_vector = self._shift_primal()
        warm_started = warm_vector is not None
        if warm_vector is not None:
            self._solver.warm_start(x=warm_vector)
        self._solver.update(q=linear, l=lower, u=upper, Ax=ax)
        self._last_l = lower
        self._last_u = upper
        started = time.perf_counter()
        result = self._solver.solve()
        wall_time = time.perf_counter() - started
        info = result.info
        status = str(getattr(info, "status", "unknown")).strip().lower()
        run_time_value = getattr(info, "run_time", wall_time)
        run_time = float(wall_time if run_time_value is None else run_time_value)
        iterations = int(getattr(info, "iter", 0))
        primal_value = getattr(info, "prim_res", float("inf"))
        dual_value = getattr(info, "dual_res", float("inf"))
        primal_residual = float("inf") if primal_value is None else float(primal_value)
        dual_residual = float("inf") if dual_value is None else float(dual_value)
        residual = max(primal_residual, dual_residual)
        primal = getattr(result, "x", None)
        raw_primal_valid = primal is not None
        if raw_primal_valid:
            try:
                primal = np.asarray(primal, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                raw_primal_valid = False
        if raw_primal_valid:
            raw_primal_valid = primal.size == self.nvar and np.isfinite(primal).all()
        if not raw_primal_valid:
            primal = np.zeros(self.nvar, dtype=float)
        state_plan = primal[:self.n_state_vars].reshape(self.N + 1, NX).copy()
        plan = primal[self.n_state_vars:].reshape(self.N, NU).copy()
        constraint_matrix = self._A_template.copy()
        constraint_matrix.data = ax
        violation = self._constraint_violation(primal, constraint_matrix, lower, upper)
        status_accepted = status == "solved" or (
            status == "solved inaccurate"
            and primal_residual <= self.settings.accept_inaccurate_residual
            and dual_residual <= self.settings.accept_inaccurate_residual
        )
        accepted = bool(
            raw_primal_valid
            and np.isfinite(plan).all()
            and np.isfinite(state_plan).all()
            and np.isfinite(violation)
            and violation <= self.settings.accept_inaccurate_residual
            and status_accepted
        )
        if accepted:
            self._previous_primal = primal.copy()
        return MPCSolution(
            plan=plan,
            state_plan=state_plan,
            status=status,
            iter=iterations,
            run_time=run_time,
            wall_time=wall_time,
            residual=residual,
            warm_started=warm_started,
            constraint_violation=violation,
            accepted=accepted,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
        )

    def reset(self) -> None:
        """Clear the explicit warm-start state without rebuilding the QP."""
        self._previous_primal = None
        try:
            self._solver.warm_start(x=np.zeros(self.nvar), y=np.zeros(self.ncon))
        except Exception:
            pass
