"""Typed loading and validation for the PAC runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    steps: int
    dt: float
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
    archived_authority_label: str


@dataclass(frozen=True)
class AuthorityConfig:
    architecture: str
    feature_mode: str
    history_len: int
    embed_dim: int
    heads: int
    layers: int
    dropout: float
    alpha_gain: float
    alpha_smoothing: float
    alpha_rate_limit: float
    alpha_deadband: float
    oracle_horizon_steps: int
    oracle_alpha_grid: tuple[float, ...]
    oracle_action_saturation_weight: float
    oracle_action_delta_weight: float
    oracle_alpha_delta_weight: float


@dataclass(frozen=True)
class TrainingConfig:
    data_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]
    fixed_initial_state: bool
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class EvaluationConfig:
    episode_seeds: tuple[int, ...]


@dataclass(frozen=True)
class SSPOConfig:
    search_scenarios: tuple[int, ...]
    eval_scenarios: tuple[int, ...]
    search_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    search_windows: tuple[str, ...]
    bias_grid: tuple[float, ...]
    iterations: int
    window_regret_weight: float
    rmse_guard_weight: float
    saturation_weight: float
    jerk_weight: float


@dataclass(frozen=True)
class OutputConfig:
    formal_run_dir: str
    sspo_run_dir: str
    formal_evidence_dir: str
    sspo_evidence_dir: str


@dataclass(frozen=True)
class PACConfig:
    environment: EnvironmentConfig
    controllers: ControllerConfig
    authority: AuthorityConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    sspo: SSPOConfig
    outputs: OutputConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid config section: {name}")
    return value


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(int(item) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values")
    return result


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(float(item) for item in value)


def load_config(path: str | Path) -> PACConfig:
    """Load and validate a PAC YAML configuration."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("PAC config root must be a mapping")

    env = _section(data, "environment")
    controllers = _section(data, "controllers")
    authority = _section(data, "authority")
    training = _section(data, "training")
    evaluation = _section(data, "evaluation")
    sspo = _section(data, "sspo")
    outputs = _section(data, "outputs")

    authority_config = AuthorityConfig(
        architecture=str(authority["architecture"]),
        feature_mode=str(authority["feature_mode"]),
        history_len=int(authority["history_len"]),
        embed_dim=int(authority["embed_dim"]),
        heads=int(authority["heads"]),
        layers=int(authority["layers"]),
        dropout=float(authority["dropout"]),
        alpha_gain=float(authority["alpha_gain"]),
        alpha_smoothing=float(authority["alpha_smoothing"]),
        alpha_rate_limit=float(authority["alpha_rate_limit"]),
        alpha_deadband=float(authority["alpha_deadband"]),
        oracle_horizon_steps=int(authority["oracle_horizon_steps"]),
        oracle_alpha_grid=_float_tuple(authority["oracle_alpha_grid"], "oracle_alpha_grid"),
        oracle_action_saturation_weight=float(authority["oracle_action_saturation_weight"]),
        oracle_action_delta_weight=float(authority["oracle_action_delta_weight"]),
        oracle_alpha_delta_weight=float(authority["oracle_alpha_delta_weight"]),
    )
    if authority_config.architecture != "transformer":
        raise ValueError("authority architecture must be transformer")
    if authority_config.history_len <= 0:
        raise ValueError("history_len must be positive")
    if authority_config.heads <= 0 or authority_config.embed_dim % authority_config.heads:
        raise ValueError("embed_dim must be divisible by heads")
    if not 0.0 <= authority_config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    environment_config = EnvironmentConfig(
        steps=int(env["steps"]),
        dt=float(env["dt"]),
        scenarios=_int_tuple(env["scenarios"], "scenarios"),
        mass_scale_xy=float(env["mass_scale_xy"]),
        damping_scale_xy=float(env["damping_scale_xy"]),
        current_amplitude_scale=float(env["current_amplitude_scale"]),
        current_frequency_scale=float(env["current_frequency_scale"]),
        vertical_current=float(env["vertical_current"]),
        eval_initial_position_std=float(env["eval_initial_position_std"]),
        eval_initial_velocity_std=float(env["eval_initial_velocity_std"]),
        vehicle_profile=str(env["vehicle_profile"]),
        thruster_layout=str(env["thruster_layout"]),
    )
    if environment_config.steps <= 0 or environment_config.dt <= 0.0:
        raise ValueError("steps and dt must be positive")
    if environment_config.scenarios != (1, 2, 3):
        raise ValueError("formal scenarios must be [1, 2, 3]")

    controller_config = ControllerConfig(
        primary=str(controllers["primary"]),
        authority=str(controllers["authority"]),
        archived_authority_label=str(controllers["archived_authority_label"]),
    )
    if controller_config.primary != "real10kg_smc_steady":
        raise ValueError("unsupported primary controller")
    if controller_config.authority != "real10kg_predictive_event":
        raise ValueError("unsupported authority controller")

    training_config = TrainingConfig(
        data_seeds=_int_tuple(training["data_seeds"], "data_seeds"),
        model_seeds=_int_tuple(training["model_seeds"], "model_seeds"),
        fixed_initial_state=bool(training["fixed_initial_state"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
    )
    if training_config.epochs <= 0 or training_config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    evaluation_config = EvaluationConfig(
        episode_seeds=_int_tuple(evaluation["episode_seeds"], "episode_seeds"),
    )
    sspo_config = SSPOConfig(
        search_scenarios=_int_tuple(sspo["search_scenarios"], "search_scenarios"),
        eval_scenarios=_int_tuple(sspo["eval_scenarios"], "eval_scenarios"),
        search_seeds=_int_tuple(sspo["search_seeds"], "search_seeds"),
        eval_seeds=_int_tuple(sspo["eval_seeds"], "eval_seeds"),
        search_windows=tuple(str(item) for item in sspo["search_windows"]),
        bias_grid=_float_tuple(sspo["bias_grid"], "bias_grid"),
        iterations=int(sspo["iterations"]),
        window_regret_weight=float(sspo["window_regret_weight"]),
        rmse_guard_weight=float(sspo["rmse_guard_weight"]),
        saturation_weight=float(sspo["saturation_weight"]),
        jerk_weight=float(sspo["jerk_weight"]),
    )
    if sspo_config.iterations <= 0:
        raise ValueError("SSPO iterations must be positive")

    return PACConfig(
        environment=environment_config,
        controllers=controller_config,
        authority=authority_config,
        training=training_config,
        evaluation=evaluation_config,
        sspo=sspo_config,
        outputs=OutputConfig(
            formal_run_dir=str(outputs["formal_run_dir"]),
            sspo_run_dir=str(outputs["sspo_run_dir"]),
            formal_evidence_dir=str(outputs["formal_evidence_dir"]),
            sspo_evidence_dir=str(outputs["sspo_evidence_dir"]),
        ),
    )
