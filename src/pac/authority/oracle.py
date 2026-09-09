"""Finite-horizon teacher oracle for authority blending.

The rollout in this module is deliberately offline.  It evaluates a candidate
alpha with copied state and a private actuator, while reusing the MPC plan that
was already computed for the real controller step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from pac.authority.evaluation import blend_actions_with_predicted_alpha
from pac.authority.features import wrap_angle
from pac.simulation.actuators import ActuatorLimits, SharedActuator


FORMAL_ORACLE_ALPHA_GRID = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have shape {shape} and be finite") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and be finite")
    return result.copy()


def _current_vector(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("future current must contain two or three finite values") from exc
    if result.size == 2:
        result = np.concatenate([result, [0.0]])
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("future current must contain two or three finite values")
    return result.copy()


@dataclass(frozen=True, slots=True)
class OracleSettings:
    """Immutable rollout objective and candidate-grid settings."""

    horizon: int = 20
    alpha_grid: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    xy_weight: float = 1.0
    z_weight: float = 0.7
    heading_weight: float = 0.08
    control_delta_weight: float = 0.01
    saturation_weight: float = 0.02
    terminal_scale: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.horizon, (bool, np.bool_)) or not isinstance(
            self.horizon, (int, np.integer)
        ):
            raise ValueError("oracle horizon must be an integer")
        horizon = int(self.horizon)
        if not 1 <= horizon <= 200:
            raise ValueError("oracle horizon must be in [1, 200]")
        try:
            grid = tuple(float(value) for value in self.alpha_grid)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("oracle alpha_grid must be finite numeric values") from exc
        if len(grid) < 2 or not all(math.isfinite(value) for value in grid):
            raise ValueError("oracle alpha_grid must have at least two finite values")
        if grid[0] != 0.0 or grid[-1] != 1.0:
            raise ValueError("oracle alpha_grid must include 0 and 1 as endpoints")
        if any(right <= left for left, right in zip(grid, grid[1:])):
            raise ValueError("oracle alpha_grid must be strictly increasing")
        weights = (
            self.xy_weight,
            self.z_weight,
            self.heading_weight,
            self.control_delta_weight,
            self.saturation_weight,
            self.terminal_scale,
        )
        converted = tuple(float(value) for value in weights)
        if not all(math.isfinite(value) and value >= 0.0 for value in converted):
            raise ValueError("oracle weights must be finite and non-negative")
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "alpha_grid", grid)
        for name, value in zip(
            (
                "xy_weight",
                "z_weight",
                "heading_weight",
                "control_delta_weight",
                "saturation_weight",
                "terminal_scale",
            ),
            converted,
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class OracleDecision:
    """Read-only oracle choice and per-candidate objective values."""

    alpha: float
    costs: tuple[float, ...]
    candidate_alphas: tuple[float, ...]
    best_cost: float
    worst_cost: float

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        costs = tuple(float(value) for value in self.costs)
        candidate_alphas = tuple(float(value) for value in self.candidate_alphas)
        if len(costs) != len(candidate_alphas) or not costs:
            raise ValueError("oracle costs and candidate_alphas must have equal non-zero length")
        if not np.isfinite(np.asarray(costs, dtype=float)).all():
            raise ValueError("oracle costs must be finite")
        if not np.isfinite(alpha) or not np.isfinite(np.asarray(candidate_alphas)).all():
            raise ValueError("oracle decision values must be finite")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "candidate_alphas", candidate_alphas)
        object.__setattr__(self, "best_cost", float(self.best_cost))
        object.__setattr__(self, "worst_cost", float(self.worst_cost))

    @property
    def candidate_count(self) -> int:
        return len(self.costs)

    @property
    def candidate_costs(self) -> tuple[float, ...]:
        return self.costs

    @property
    def cost_by_alpha(self) -> dict[float, float]:
        return dict(zip(self.candidate_alphas, self.costs))

    @property
    def best_alpha(self) -> float:
        return self.alpha

    @property
    def cost(self) -> float:
        return self.best_cost


def _coerce_settings(settings: OracleSettings | Any | None) -> OracleSettings:
    if settings is None:
        return OracleSettings()
    if isinstance(settings, OracleSettings):
        return settings
    getter = settings.get if isinstance(settings, dict) else lambda name: getattr(settings, name)
    values = {
        name: getter(name)
        for name in (
            "horizon",
            "alpha_grid",
            "xy_weight",
            "z_weight",
            "heading_weight",
            "control_delta_weight",
            "saturation_weight",
            "terminal_scale",
        )
    }
    return OracleSettings(**values)


def validate_formal_oracle_contract(settings: OracleSettings | Any) -> OracleSettings:
    """Validate the fixed H20/grid contract used by formal and short runs."""
    resolved = _coerce_settings(settings)
    if resolved.horizon != 20:
        raise ValueError("formal oracle contract requires horizon=20")
    if resolved.alpha_grid != FORMAL_ORACLE_ALPHA_GRID:
        raise ValueError(
            "formal oracle contract requires alpha_grid=(0.0, 0.1, ..., 1.0) with 11 points"
        )
    return resolved


def _controller_action(controller: Any, target, eta, nu, stamp: float, current) -> np.ndarray:
    if controller is None or not callable(getattr(controller, "compute", None)):
        raise TypeError("primary_controller must provide compute()")
    try:
        action = controller.compute(
            target,
            eta,
            nu,
            t=float(stamp),
            current_prediction=current,
        )
    except TypeError:
        action = controller.compute(target, eta, nu)
    return _finite_vector(action, (6,), "controller action")


def _resolve_plan(mpc_plan: Any, mpc_controller: Any) -> np.ndarray:
    plan = mpc_plan
    if plan is None and mpc_controller is not None:
        plan = getattr(mpc_controller, "last_plan", None)
        if callable(plan):
            plan = plan()
    if plan is not None and hasattr(plan, "plan"):
        plan = plan.plan
    if plan is None:
        raise ValueError("mpc_plan is required; the oracle never recomputes MPC")
    try:
        result = np.asarray(plan, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mpc_plan must be a finite (N, 6) array") from exc
    if result.ndim == 1 and result.size == 6:
        result = result.reshape(1, 6)
    if result.ndim != 2 or result.shape[1] != 6 or result.shape[0] < 1:
        raise ValueError("mpc_plan must be a finite (N, 6) array")
    if not np.isfinite(result).all():
        raise ValueError("mpc_plan must be finite")
    return result.copy()


def _resolve_current_provider(simulator: Any, provider: Callable[[float], Any] | None):
    if provider is not None:
        return provider
    if simulator is None:
        return lambda _stamp: np.zeros(3, dtype=float)
    if callable(getattr(simulator, "current_provider", None)):
        return simulator.current_provider
    if callable(getattr(simulator, "current_at", None)):
        return simulator.current_at
    if callable(getattr(simulator, "get_current", None)):
        return simulator.get_current
    if callable(getattr(simulator, "_current_provider", None)):
        return simulator._current_provider
    generate = getattr(simulator, "_generate_current", None)
    convert = getattr(simulator, "_current_for_dynamics", None)
    if callable(generate):
        if callable(convert):
            return lambda stamp: convert(generate(float(stamp)))
        return lambda stamp: generate(float(stamp))
    return lambda _stamp: np.zeros(3, dtype=float)


def _resolve_target_provider(simulator: Any, provider: Callable[[float], Any] | None):
    if provider is not None:
        return provider
    if simulator is not None and callable(getattr(simulator, "_get_target", None)):
        return simulator._get_target
    return lambda _stamp: np.zeros(6, dtype=float)


def _resolve_wrench_converter(simulator: Any, converter: Callable[[Any], Any] | None):
    if converter is not None:
        return converter
    if simulator is not None and callable(getattr(simulator, "_action_to_wrench", None)):
        converter = simulator._action_to_wrench
    elif simulator is not None:
        layout = getattr(simulator, "thruster_layout", None)
        converter = getattr(layout, "normalized_action_to_wrench", None)
    if not callable(converter):
        return lambda action: np.asarray(action, dtype=float).copy()

    def convert(action):
        result = converter(action)
        if isinstance(result, tuple):
            return result[0]
        return result

    return convert


def _resolve_dynamics(simulator: Any, dynamics: Any | None):
    resolved = dynamics
    if resolved is None and simulator is not None:
        resolved = getattr(simulator, "dynamics", None)
    if resolved is None or not callable(getattr(resolved, "predict_step", None)):
        raise TypeError("simulator/dynamics must provide predict_step()")
    return resolved


def _resolve_limits(simulator: Any, limits: ActuatorLimits | None) -> ActuatorLimits:
    if limits is None and simulator is not None:
        actuator = getattr(simulator, "actuator", None)
        limits = getattr(actuator, "limits", None)
    if limits is None:
        limits = ActuatorLimits(-1.0, 1.0, None)
    if isinstance(limits, dict):
        limits = ActuatorLimits(
            command_min=limits["command_min"],
            command_max=limits["command_max"],
            max_delta_per_step=limits.get("max_delta_per_step"),
        )
    if not isinstance(limits, ActuatorLimits):
        try:
            limits = ActuatorLimits(
                command_min=limits.command_min,
                command_max=limits.command_max,
                max_delta_per_step=getattr(limits, "max_delta_per_step", None),
            )
        except AttributeError as exc:
            raise TypeError("actuator_limits must be ActuatorLimits-like") from exc
    return limits


def _tracking_cost(settings: OracleSettings, target, eta) -> float:
    error = _finite_vector(target, (6,), "target") - _finite_vector(eta, (6,), "predicted eta")
    error[3:6] = [wrap_angle(value) for value in error[3:6]]
    return (
        settings.xy_weight * float(np.dot(error[:2], error[:2]))
        + settings.z_weight * float(error[2] ** 2)
        + settings.heading_weight * float(error[5] ** 2)
    )


def choose_rollout_oracle_alpha(
    current_eta=None,
    current_nu=None,
    t: float = 0.0,
    previous_applied=None,
    primary_controller=None,
    mpc_controller=None,
    mpc_plan=None,
    simulator=None,
    current_provider: Callable[[float], Any] | None = None,
    actuator_limits: ActuatorLimits | None = None,
    settings: OracleSettings | Any | None = None,
    *,
    dynamics=None,
    target_provider: Callable[[float], Any] | None = None,
    action_to_wrench: Callable[[Any], Any] | None = None,
    **aliases,
) -> OracleDecision:
    """Choose the lowest-cost constant alpha over a private H-step rollout.

    The aliases keep this API convenient for callers using the older ``eta``,
    ``nu``, ``previous_action``, ``plan``, or ``environment`` names.
    """
    if current_eta is None:
        current_eta = aliases.pop("eta", None)
    if current_nu is None:
        current_nu = aliases.pop("nu", None)
    if previous_applied is None:
        previous_applied = aliases.pop("previous_action", None)
    if mpc_plan is None:
        mpc_plan = aliases.pop("plan", None)
    if mpc_controller is None:
        mpc_controller = aliases.pop("authority_controller", None)
    if mpc_plan is None:
        mpc_plan = aliases.pop("mpc_solution", None)
    if simulator is None:
        simulator = aliases.pop("environment", None)
    if current_provider is None:
        current_provider = aliases.pop("future_current_provider", None)
    if actuator_limits is None:
        actuator_limits = aliases.pop("limits", None)
    if aliases:
        unknown = ", ".join(sorted(aliases))
        raise TypeError(f"unexpected oracle argument(s): {unknown}")

    oracle = _coerce_settings(settings)
    eta0 = _finite_vector(current_eta, (6,), "current_eta")
    nu0 = _finite_vector(current_nu, (6,), "current_nu")
    previous0 = _finite_vector(previous_applied, (6,), "previous_applied")
    stamp0 = float(t)
    if not math.isfinite(stamp0):
        raise ValueError("t must be finite")
    plan = _resolve_plan(mpc_plan, mpc_controller)
    resolved_dynamics = _resolve_dynamics(simulator, dynamics)
    limits = _resolve_limits(simulator, actuator_limits)
    current_at = _resolve_current_provider(simulator, current_provider)
    target_at = _resolve_target_provider(simulator, target_provider)
    to_wrench = _resolve_wrench_converter(simulator, action_to_wrench)
    dt = float(getattr(resolved_dynamics, "dt", 0.01))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dynamics dt must be finite and positive")

    # Keep snapshots only for simulator dynamics objects that expose mutable
    # state.  Real AUVDynamics.predict_step is pure, but this also protects
    # lightweight test doubles and custom providers.
    state_snapshot = {}
    for name in ("eta", "nu"):
        value = getattr(resolved_dynamics, name, None)
        if isinstance(value, np.ndarray):
            state_snapshot[name] = value.copy()

    costs: list[float] = []
    candidate_alphas = tuple(oracle.alpha_grid)
    try:
        for alpha in candidate_alphas:
            eta = eta0.copy()
            nu = nu0.copy()
            private_actuator = SharedActuator(limits)
            private_actuator._previous_applied = previous0.copy()
            total = 0.0
            for step in range(oracle.horizon):
                current_time = stamp0 + step * dt
                current = _current_vector(current_at(current_time))
                target = _finite_vector(target_at(current_time), (6,), "rollout target")
                primary_action = _controller_action(
                    primary_controller, target, eta, nu, current_time, current
                )
                authority_action = plan[min(step, plan.shape[0] - 1)]
                requested = (1.0 - float(alpha)) * primary_action + float(alpha) * authority_action
                requested = _finite_vector(requested, (6,), "blended action")
                # Evaluate the same command that reaches the dynamics after one
                # shared actuator operation; the real actuator is never touched.
                previous_for_delta = private_actuator._previous_applied.copy()
                actuator_step = private_actuator.apply(requested)
                applied = np.asarray(actuator_step.applied, dtype=float)
                delta = applied - previous_for_delta
                saturation = float(np.mean((requested - actuator_step.amplitude_clipped) ** 2))
                if limits.max_delta_per_step is not None:
                    saturation += float(
                        np.mean(
                            np.maximum(
                                np.abs(actuator_step.amplitude_clipped - previous_for_delta)
                                - limits.max_delta_per_step,
                                0.0,
                            )
                            ** 2
                        )
                    )
                tau = _finite_vector(to_wrench(applied), (6,), "actuator wrench")
                eta, nu = resolved_dynamics.predict_step(eta, nu, tau, current)
                eta = _finite_vector(eta, (6,), "predicted eta")
                nu = _finite_vector(nu, (6,), "predicted nu")
                next_target = _finite_vector(
                    target_at(stamp0 + (step + 1) * dt), (6,), "rollout target"
                )
                total += _tracking_cost(oracle, next_target, eta)
                total += oracle.control_delta_weight * float(np.mean(delta ** 2))
                total += oracle.saturation_weight * saturation
            total += oracle.terminal_scale * _tracking_cost(oracle, next_target, eta)
            costs.append(float(total))
    finally:
        for name, value in state_snapshot.items():
            try:
                setattr(resolved_dynamics, name, value)
            except Exception:
                pass

    cost_array = np.asarray(costs, dtype=float)
    if not np.isfinite(cost_array).all():
        raise ValueError("oracle rollout produced non-finite costs")
    # Python's tuple key makes the lower alpha win exact ties and near-ties
    # without relying on sort stability or floating point ordering quirks.
    best_index = min(range(len(costs)), key=lambda index: (costs[index], candidate_alphas[index]))
    return OracleDecision(
        alpha=float(candidate_alphas[best_index]),
        costs=tuple(costs),
        candidate_alphas=candidate_alphas,
        best_cost=float(costs[best_index]),
        worst_cost=float(max(costs)),
    )


__all__ = [
    "FORMAL_ORACLE_ALPHA_GRID",
    "OracleDecision",
    "OracleSettings",
    "choose_rollout_oracle_alpha",
    "validate_formal_oracle_contract",
]
