"""Typed loading and validation for the formal true-MPC v3 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import yaml

from pac.evaluation.seeds import validate_disjoint_seed_partitions


_MAX_SEED = 2**64 - 1


@dataclass(frozen=True)
class ProtocolConfig:
    version: str


@dataclass(frozen=True)
class EnvironmentConfig:
    dt: float
    steps: int
    scenarios: tuple[int, ...]
    mass_scale_xy: float
    damping_scale_xy: float
    current_amplitude_scale: float
    current_frequency_scale: float
    vertical_current: float
    eval_initial_position_std: float
    eval_initial_velocity_std: float
    vehicle_profile: str
    thruster_layout: str


@dataclass(frozen=True)
class ControllerConfig:
    primary: str
    authority: str


@dataclass(frozen=True)
class ActuatorConfig:
    command_min: float
    command_max: float
    max_delta_per_step: float
    max_force_n: float


@dataclass(frozen=True)
class MPCConfig:
    horizon: int
    q_diag: tuple[float, ...]
    terminal_scale: float
    r_diag: tuple[float, ...]
    s_diag: tuple[float, ...]
    eps_abs: float
    eps_rel: float
    max_iter: int
    time_limit_s: float
    accept_inaccurate_residual: float
    max_consecutive_plan_reuse: int


@dataclass(frozen=True)
class TrainingConfig:
    oracle_train_seeds: tuple[int, ...]
    oracle_val_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    episode_seeds: tuple[int, ...]


@dataclass(frozen=True)
class SSPOConfig:
    search_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]


@dataclass(frozen=True)
class RobustnessConfig:
    eval_seeds: tuple[int, ...]


@dataclass(frozen=True)
class OutputConfig:
    run_root: str
    evidence_root: str


@dataclass(frozen=True)
class V3ExperimentConfig:
    protocol: ProtocolConfig
    environment: EnvironmentConfig
    controller: ControllerConfig
    actuator: ActuatorConfig
    mpc: MPCConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    sspo: SSPOConfig
    robustness: RobustnessConfig
    outputs: OutputConfig

    @property
    def controllers(self) -> ControllerConfig:
        """Compatibility alias for callers that use the v2 section name."""
        return self.controller


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid config section: {name}")
    return value


def _required(section: dict[str, Any], name: str) -> Any:
    if name not in section:
        raise ValueError(f"missing config value: {name}")
    return section[name]


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_empty_string(value: Any, name: str, *, relative_path: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if relative_path:
        path = Path(result)
        windows_path = PureWindowsPath(result)
        if path.anchor or windows_path.anchor:
            raise ValueError(f"{name} must be a relative path")
        if ".." in path.parts or ".." in windows_path.parts:
            raise ValueError(f"{name} must not contain parent path '..'")
    return result


def _int_tuple(value: Any, role: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{role} must be a non-empty list")
    result: list[int] = []
    seen: set[int] = set()
    for raw_seed in value:
        seed = _int_value(raw_seed, role)
        if seed in seen:
            raise ValueError(f"duplicate seed in role {role}: {seed}")
        seen.add(seed)
        result.append(seed)
    return tuple(result)


def _seed_tuple(value: Any, role: str) -> tuple[int, ...]:
    result = _int_tuple(value, role)
    for seed in result:
        if seed < 0:
            raise ValueError(f"negative seed in role {role}: {seed}")
        if seed > _MAX_SEED:
            raise ValueError(f"seed in role {role} exceeds uint64 maximum: {seed}")
    return result


def _float_vector(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_float_value(item, name) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def load_v3_config(path: str | Path) -> V3ExperimentConfig:
    """Load and validate a formal true-MPC v3 YAML configuration."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("v3 config root must be a mapping")

    protocol_data = _section(data, "protocol")
    protocol = ProtocolConfig(
        version=_non_empty_string(_required(protocol_data, "version"), "protocol.version")
    )
    if protocol.version != "formal_true_mpc_v3":
        raise ValueError("protocol.version must be formal_true_mpc_v3")

    env = _section(data, "environment")
    environment = EnvironmentConfig(
        dt=_float_value(_required(env, "dt"), "environment.dt"),
        steps=_int_value(_required(env, "steps"), "environment.steps"),
        scenarios=_int_tuple(_required(env, "scenarios"), "scenarios"),
        mass_scale_xy=_float_value(_required(env, "mass_scale_xy"), "environment.mass_scale_xy"),
        damping_scale_xy=_float_value(_required(env, "damping_scale_xy"), "environment.damping_scale_xy"),
        current_amplitude_scale=_float_value(_required(env, "current_amplitude_scale"), "environment.current_amplitude_scale"),
        current_frequency_scale=_float_value(_required(env, "current_frequency_scale"), "environment.current_frequency_scale"),
        vertical_current=_float_value(_required(env, "vertical_current"), "environment.vertical_current"),
        eval_initial_position_std=_float_value(_required(env, "eval_initial_position_std"), "environment.eval_initial_position_std"),
        eval_initial_velocity_std=_float_value(_required(env, "eval_initial_velocity_std"), "environment.eval_initial_velocity_std"),
        vehicle_profile=_non_empty_string(
            _required(env, "vehicle_profile"), "environment.vehicle_profile"
        ),
        thruster_layout=_non_empty_string(
            _required(env, "thruster_layout"), "environment.thruster_layout"
        ),
    )
    if environment.scenarios != (1, 2, 3):
        raise ValueError("environment scenarios must be exactly (1, 2, 3)")
    if environment.dt <= 0.0 or environment.steps <= 0:
        raise ValueError("environment dt and steps must be positive")
    if (
        environment.mass_scale_xy <= 0.0
        or environment.damping_scale_xy <= 0.0
        or environment.current_frequency_scale <= 0.0
    ):
        raise ValueError("environment mass, damping, and current frequency scales must be positive")
    if (
        environment.current_amplitude_scale < 0.0
        or environment.eval_initial_position_std < 0.0
        or environment.eval_initial_velocity_std < 0.0
    ):
        raise ValueError("environment current amplitude and evaluation std values must be non-negative")

    controller_data = _section(data, "controller")
    controller = ControllerConfig(
        primary=_non_empty_string(_required(controller_data, "primary"), "controller.primary"),
        authority=_non_empty_string(_required(controller_data, "authority"), "controller.authority"),
    )

    actuator_data = _section(data, "actuator")
    actuator = ActuatorConfig(
        command_min=_float_value(_required(actuator_data, "command_min"), "actuator.command_min"),
        command_max=_float_value(_required(actuator_data, "command_max"), "actuator.command_max"),
        max_delta_per_step=_float_value(_required(actuator_data, "max_delta_per_step"), "actuator.max_delta_per_step"),
        max_force_n=_float_value(_required(actuator_data, "max_force_n"), "actuator.max_force_n"),
    )
    if actuator.command_min >= actuator.command_max:
        raise ValueError("actuator command_min must be less than command_max")
    if actuator.max_delta_per_step <= 0.0 or actuator.max_force_n <= 0.0:
        raise ValueError("actuator limits must be positive")

    mpc_data = _section(data, "mpc")
    mpc = MPCConfig(
        horizon=_int_value(_required(mpc_data, "horizon"), "mpc.horizon"),
        q_diag=_float_vector(_required(mpc_data, "q_diag"), "q_diag", 12),
        terminal_scale=_float_value(_required(mpc_data, "terminal_scale"), "mpc.terminal_scale"),
        r_diag=_float_vector(_required(mpc_data, "r_diag"), "r_diag", 6),
        s_diag=_float_vector(_required(mpc_data, "s_diag"), "s_diag", 6),
        eps_abs=_float_value(_required(mpc_data, "eps_abs"), "mpc.eps_abs"),
        eps_rel=_float_value(_required(mpc_data, "eps_rel"), "mpc.eps_rel"),
        max_iter=_int_value(_required(mpc_data, "max_iter"), "mpc.max_iter"),
        time_limit_s=_float_value(_required(mpc_data, "time_limit_s"), "mpc.time_limit_s"),
        accept_inaccurate_residual=_float_value(
            _required(mpc_data, "accept_inaccurate_residual"),
            "mpc.accept_inaccurate_residual",
        ),
        max_consecutive_plan_reuse=_int_value(
            _required(mpc_data, "max_consecutive_plan_reuse"),
            "mpc.max_consecutive_plan_reuse",
        ),
    )
    if mpc.horizon <= 0:
        raise ValueError("mpc horizon must be positive")
    if (
        mpc.terminal_scale <= 0.0
        or mpc.eps_abs <= 0.0
        or mpc.eps_rel <= 0.0
        or mpc.max_iter <= 0
        or mpc.time_limit_s <= 0.0
        or mpc.accept_inaccurate_residual <= 0.0
        or mpc.max_consecutive_plan_reuse <= 0
    ):
        raise ValueError("mpc limits must be positive")
    if any(value <= 0.0 for value in (*mpc.q_diag, *mpc.r_diag, *mpc.s_diag)):
        raise ValueError("mpc diagonal weights must be positive")

    training_data = _section(data, "training")
    training = TrainingConfig(
        oracle_train_seeds=_seed_tuple(_required(training_data, "oracle_train_seeds"), "oracle_train"),
        oracle_val_seeds=_seed_tuple(_required(training_data, "oracle_val_seeds"), "oracle_val"),
        model_seeds=_seed_tuple(_required(training_data, "model_seeds"), "model"),
        max_epochs=_int_value(_required(training_data, "max_epochs"), "training.max_epochs"),
        patience=_int_value(_required(training_data, "patience"), "training.patience"),
    )
    if training.max_epochs <= 0 or training.patience <= 0:
        raise ValueError("training limits must be positive")

    evaluation_data = _section(data, "evaluation")
    evaluation = EvaluationConfig(
        episode_seeds=_seed_tuple(_required(evaluation_data, "episode_seeds"), "evaluation"),
    )

    sspo_data = _section(data, "sspo")
    sspo = SSPOConfig(
        search_seeds=_seed_tuple(_required(sspo_data, "search_seeds"), "sspo_search"),
        eval_seeds=_seed_tuple(_required(sspo_data, "eval_seeds"), "sspo_eval"),
    )

    robustness_data = _section(data, "robustness")
    robustness = RobustnessConfig(
        eval_seeds=_seed_tuple(_required(robustness_data, "eval_seeds"), "robustness_eval"),
    )

    outputs_data = _section(data, "outputs")
    outputs = OutputConfig(
        run_root=_non_empty_string(
            _required(outputs_data, "run_root"), "outputs.run_root", relative_path=True
        ),
        evidence_root=_non_empty_string(
            _required(outputs_data, "evidence_root"), "outputs.evidence_root", relative_path=True
        ),
    )

    config = V3ExperimentConfig(
        protocol=protocol,
        environment=environment,
        controller=controller,
        actuator=actuator,
        mpc=mpc,
        training=training,
        evaluation=evaluation,
        sspo=sspo,
        robustness=robustness,
        outputs=outputs,
    )
    validate_disjoint_seed_partitions(seed_partitions(config))
    return config


def seed_partitions(config: V3ExperimentConfig) -> dict[str, tuple[int, ...]]:
    """Return the independent seed roles used by the v3 experiment."""
    return {
        "oracle_train": config.training.oracle_train_seeds,
        "oracle_val": config.training.oracle_val_seeds,
        "model": config.training.model_seeds,
        "evaluation": config.evaluation.episode_seeds,
        "sspo_search": config.sspo.search_seeds,
        "sspo_eval": config.sspo.eval_seeds,
        "robustness_eval": config.robustness.eval_seeds,
    }
