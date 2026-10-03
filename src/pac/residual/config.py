"""Typed loading and validation for the predictive-authority residual experiment.

The residual configuration is fully independent of the frozen supervised experiment: it
reuses the v3 environment/controller/oracle contracts verbatim and adds the
world-model, disturbance, residual-RL, and reward sections.  Nothing in this
module mutates v3 state or reads v3 outputs except the explicitly declared
backbone checkpoint directory used to initialize v4 policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from pac.evaluation.seeds import validate_disjoint_seed_partitions
from pac.experiment_config import (
    ActuatorConfig,
    AuthorityModelConfig,
    ControllerConfig,
    EnvironmentConfig,
    MPCConfig,
    OracleConfig,
    ProtocolConfig,
)
from pac.experiment_config import (
    _float_sequence,
    _float_value,
    _int_tuple,
    _int_value,
    _non_empty_string,
    _required,
    _section,
)

_V4_PROFILE_RANGES = ("train", "test")
_STOCHASTIC_FAMILIES = ("ou_current", "colored_noise")
_PARAMETER_FAMILIES = (
    "ou_current",
    "colored_noise",
    "random_freq_amp",
    "mass_damping_mismatch",
    "actuator_delay_noise",
    "fast_ou",
    "estimation_delay",
)
_DISTURBANCE_PARAMS = {
    "ou_current": ("theta", "sigma", "mean_scale"),
    "colored_noise": ("rho", "sigma"),
    "random_freq_amp": ("amplitude_scale", "frequency_scale"),
    "mass_damping_mismatch": ("mass_scale_xy", "damping_scale_xy"),
    "actuator_delay_noise": ("delay_steps", "action_noise_std"),
    "fast_ou": ("theta", "sigma", "mean_scale"),
    "estimation_delay": ("delay_steps", "noise_std"),
}
_FLOAT_PARAMS = {
    "theta",
    "sigma",
    "mean_scale",
    "rho",
    "amplitude_scale",
    "frequency_scale",
    "mass_scale_xy",
    "damping_scale_xy",
    "action_noise_std",
    "noise_std",
}


@dataclass(frozen=True)
class WorldModelConfig:
    history_len: int
    hidden_dim: int
    num_layers: int
    residual_hidden_dim: int
    members: int
    horizons: tuple[int, ...]
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    current_aux_weight: float
    ranking_stride: int
    ranking_windows_per_episode: int
    ranking_spearman_threshold: float
    ranking_top1_threshold: float


@dataclass(frozen=True)
class DisturbanceRange:
    low: float
    high: float


@dataclass(frozen=True)
class DisturbanceFamilyConfig:
    name: str
    base_scenario: int
    ranges: dict[str, dict[str, DisturbanceRange]]


@dataclass(frozen=True)
class DisturbancesConfig:
    families: tuple[str, ...]
    base_scenario: dict[str, int]
    families_config: dict[str, DisturbanceFamilyConfig]
    eval_only: dict[str, DisturbanceFamilyConfig]

    def profile_ranges(self, family: str, profile: str) -> dict[str, DisturbanceRange]:
        if family in self.eval_only:
            return self.eval_only[family].ranges[profile]
        return self.families_config[family].ranges[profile]


@dataclass(frozen=True)
class SeedsConfig:
    wm_train: tuple[int, ...]
    wm_val: tuple[int, ...]
    wm_test: tuple[int, ...]
    eval_seen: tuple[int, ...]
    eval_unseen: tuple[int, ...]
    model: tuple[int, ...]
    sspo_search: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationPairedConfig:
    family_seed_count: int


@dataclass(frozen=True)
class SSPOCalibrationConfig:
    search_seeds: tuple[int, ...]
    bias_grid: tuple[float, ...]
    iterations: int


@dataclass(frozen=True)
class WMAuthorityConfig:
    uncertainty_gate: float
    refresh_steps: int


@dataclass(frozen=True)
class CollectionConfig:
    behavior_constant_alphas: tuple[float, ...]
    include_transformer_policy: bool
    transformer_model_seed: int


@dataclass(frozen=True)
class RewardConfig:
    version: str
    w_position: float
    w_heading: float
    w_control: float
    w_delta_control: float
    w_delta_alpha: float
    w_saturation: float
    w_deadline: float
    w_constraint: float
    position_scale_m: float
    heading_scale_rad: float
    # Optional conservative-centering prior for the RL-residual path
    # (E1/S2 experiments): penalizes the deployed alpha away from
    # alpha_center while the position error exceeds
    # center_gate_error_m (0.0 = always active).  Zero w_center (the
    # frozen protocol value) disables the term.
    w_center: float = 0.0
    alpha_center: float = 0.5
    center_gate_error_m: float = 0.0


@dataclass(frozen=True)
class RLConfig:
    algorithm: str
    delta_max: float
    lambda_blend: float
    behavior_reg: float
    freeze_backbone_round1: bool
    unfreeze_blocks_round2: tuple[str, ...]
    batch_size: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    policy_delay: int
    target_noise: float
    noise_clip: float
    expl_noise: float
    warmup_steps: int
    updates_per_step: int
    max_steps_per_round: int
    warmstart_transitions: int


@dataclass(frozen=True)
class BackboneConfig:
    checkpoint_dir: str


@dataclass(frozen=True)
class PairedOutputConfig:
    run_root: str


@dataclass(frozen=True)
class ResidualExperimentConfig:
    protocol: ProtocolConfig
    environment: EnvironmentConfig
    controller: ControllerConfig
    authority_model: AuthorityModelConfig
    actuator: ActuatorConfig
    mpc: MPCConfig
    oracle: OracleConfig
    world_model: WorldModelConfig
    disturbances: DisturbancesConfig
    seeds: SeedsConfig
    evaluation: EvaluationPairedConfig
    sspo: SSPOCalibrationConfig
    wm_authority: WMAuthorityConfig
    collection: CollectionConfig
    reward: RewardConfig
    rl: RLConfig
    backbone: BackboneConfig
    outputs: PairedOutputConfig


def _float_range(section: dict[str, Any], name: str) -> DisturbanceRange:
    values = _float_sequence(_required(section, name), name)
    if len(values) != 2:
        raise ValueError(f"disturbance range {name} must contain exactly two values")
    if not values[0] < values[1]:
        raise ValueError(f"disturbance range {name} must be strictly increasing")
    return DisturbanceRange(low=values[0], high=values[1])


def _load_family(
        name: str,
        data: Any,
        default_scenario: int,
        *,
        profiles: tuple[str, ...] = _V4_PROFILE_RANGES) -> DisturbanceFamilyConfig:
    if not isinstance(data, dict):
        raise ValueError(f"disturbance family {name} must be a mapping")
    params = _DISTURBANCE_PARAMS[name]
    ranges: dict[str, dict[str, DisturbanceRange]] = {}
    for profile in profiles:
        profile_section = _section(data, profile)
        profile_ranges: dict[str, DisturbanceRange] = {}
        for parameter in params:
            if parameter in _FLOAT_PARAMS:
                profile_ranges[parameter] = _float_range(profile_section, parameter)
            else:
                values = _float_sequence(
                    _required(profile_section, parameter), parameter
                )
                if len(values) != 2:
                    raise ValueError(
                        f"disturbance range {parameter} must contain exactly two values"
                    )
                for value in values:
                    if value < 0 or value != int(value):
                        raise ValueError(
                            f"disturbance range {parameter} must contain non-negative integers"
                        )
                profile_ranges[parameter] = DisturbanceRange(
                    low=int(values[0]), high=int(values[1])
                )
        ranges[profile] = profile_ranges
    return DisturbanceFamilyConfig(
        name=name,
        base_scenario=default_scenario,
        ranges=ranges,
    )


def _load_disturbances(data: dict[str, Any]) -> DisturbancesConfig:
    section = _section(data, "disturbances")
    families_raw = _required(section, "families")
    if not isinstance(families_raw, (list, tuple)) or not families_raw:
        raise ValueError("disturbances.families must be a non-empty list")
    families = tuple(str(name) for name in families_raw)
    if any(not name or not isinstance(name, str) for name in families):
        raise ValueError("disturbances.families entries must be non-empty strings")
    unknown = sorted(set(families) - set(_PARAMETER_FAMILIES))
    if unknown:
        raise ValueError(f"unknown disturbance families: {', '.join(unknown)}")
    base_section = section.get("base_scenario", {})
    if not isinstance(base_section, dict):
        raise ValueError("disturbances.base_scenario must be a mapping")
    base_scenario: dict[str, int] = {}
    for family in families:
        raw = base_section.get(family, 2)
        scenario = _int_value(raw, f"base_scenario.{family}")
        if scenario not in (1, 2, 3):
            raise ValueError(f"base_scenario.{family} must be one of 1, 2, 3")
        base_scenario[family] = scenario
    families_config = {
        family: _load_family(family, _section(section, family), base_scenario[family])
        for family in families
    }
    eval_only_section = section.get("eval_only", {})
    if not isinstance(eval_only_section, dict):
        raise ValueError("disturbances.eval_only must be a mapping")
    eval_only: dict[str, DisturbanceFamilyConfig] = {}
    for family, family_data in eval_only_section.items():
        if family not in _PARAMETER_FAMILIES:
            raise ValueError(f"unknown eval_only disturbance family: {family}")
        if family in families:
            raise ValueError(f"eval_only family {family} duplicates a collected family")
        raw_scenario = eval_only_section.get(family, {}).get("base_scenario", 2)
        scenario = _int_value(raw_scenario, f"eval_only.{family}.base_scenario")
        if scenario not in (1, 2, 3):
            raise ValueError(f"eval_only.{family}.base_scenario must be one of 1, 2, 3")
        eval_only[family] = _load_family(
            str(family), family_data, scenario, profiles=("test",)
        )
    return DisturbancesConfig(
        families=families,
        base_scenario=base_scenario,
        families_config=families_config,
        eval_only=eval_only,
    )


def _load_world_model(data: dict[str, Any]) -> WorldModelConfig:
    section = _section(data, "world_model")
    horizons = _int_tuple(_required(section, "horizons"), "world_model.horizons")
    config = WorldModelConfig(
        history_len=_int_value(_required(section, "history_len"), "world_model.history_len"),
        hidden_dim=_int_value(_required(section, "hidden_dim"), "world_model.hidden_dim"),
        num_layers=_int_value(_required(section, "num_layers"), "world_model.num_layers"),
        residual_hidden_dim=_int_value(
            _required(section, "residual_hidden_dim"), "world_model.residual_hidden_dim"
        ),
        members=_int_value(_required(section, "members"), "world_model.members"),
        horizons=horizons,
        max_epochs=_int_value(_required(section, "max_epochs"), "world_model.max_epochs"),
        patience=_int_value(_required(section, "patience"), "world_model.patience"),
        batch_size=_int_value(_required(section, "batch_size"), "world_model.batch_size"),
        learning_rate=_float_value(_required(section, "learning_rate"), "world_model.learning_rate"),
        weight_decay=_float_value(_required(section, "weight_decay"), "world_model.weight_decay"),
        current_aux_weight=_float_value(
            _required(section, "current_aux_weight"), "world_model.current_aux_weight"
        ),
        ranking_stride=_int_value(_required(section, "ranking_stride"), "world_model.ranking_stride"),
        ranking_windows_per_episode=_int_value(
            _required(section, "ranking_windows_per_episode"),
            "world_model.ranking_windows_per_episode",
        ),
        ranking_spearman_threshold=_float_value(
            _required(section, "ranking_spearman_threshold"),
            "world_model.ranking_spearman_threshold",
        ),
        ranking_top1_threshold=_float_value(
            _required(section, "ranking_top1_threshold"),
            "world_model.ranking_top1_threshold",
        ),
    )
    if not 1 <= config.history_len <= 128:
        raise ValueError("world_model.history_len must be in [1, 128]")
    if min(config.hidden_dim, config.residual_hidden_dim) < 1:
        raise ValueError("world_model dimensions must be positive")
    if not 1 <= config.num_layers <= 8:
        raise ValueError("world_model.num_layers must be in [1, 8]")
    if not 2 <= config.members <= 32:
        raise ValueError("world_model.members must be in [2, 32]")
    if any(horizon <= 0 for horizon in config.horizons):
        raise ValueError("world_model.horizons must be positive")
    if config.max_epochs <= 0 or config.patience <= 0:
        raise ValueError("world_model.max_epochs and patience must be positive")
    if config.batch_size <= 0 or config.learning_rate <= 0.0:
        raise ValueError("world_model batch_size and learning_rate must be positive")
    if config.weight_decay < 0.0 or config.current_aux_weight < 0.0:
        raise ValueError("world_model weight_decay and current_aux_weight must be non-negative")
    if config.ranking_stride <= 0 or config.ranking_windows_per_episode <= 0:
        raise ValueError("world_model ranking window settings must be positive")
    if not 0.0 <= config.ranking_spearman_threshold <= 1.0:
        raise ValueError("world_model.ranking_spearman_threshold must be in [0, 1]")
    if not 0.0 <= config.ranking_top1_threshold <= 1.0:
        raise ValueError("world_model.ranking_top1_threshold must be in [0, 1]")
    return config


def _load_seeds(data: dict[str, Any]) -> SeedsConfig:
    section = _section(data, "seeds")
    config = SeedsConfig(
        wm_train=_int_tuple(_required(section, "wm_train"), "wm_train"),
        wm_val=_int_tuple(_required(section, "wm_val"), "wm_val"),
        wm_test=_int_tuple(_required(section, "wm_test"), "wm_test"),
        eval_seen=_int_tuple(_required(section, "eval_seen"), "eval_seen"),
        eval_unseen=_int_tuple(_required(section, "eval_unseen"), "eval_unseen"),
        model=_int_tuple(_required(section, "model"), "model"),
        sspo_search=_int_tuple(_required(section, "sspo_search"), "sspo_search"),
    )
    for seeds in (
        config.wm_train,
        config.wm_val,
        config.wm_test,
        config.eval_seen,
        config.eval_unseen,
        config.model,
        config.sspo_search,
    ):
        for seed in seeds:
            if seed < 0 or seed > 2**64 - 1:
                raise ValueError("seeds must be non-negative uint64 values")
    return config


def _load_collection(data: dict[str, Any]) -> CollectionConfig:
    section = _section(data, "collection")
    alphas = _float_sequence(
        _required(section, "behavior_constant_alphas"), "behavior_constant_alphas"
    )
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("behavior_constant_alphas must be in [0, 1]")
    include = _required(section, "include_transformer_policy")
    if not isinstance(include, bool):
        raise ValueError("include_transformer_policy must be a boolean")
    return CollectionConfig(
        behavior_constant_alphas=tuple(sorted(set(alphas))),
        include_transformer_policy=include,
        transformer_model_seed=_int_value(
            _required(section, "transformer_model_seed"), "transformer_model_seed"
        ),
    )


def _load_reward(data: dict[str, Any]) -> RewardConfig:
    section = _section(data, "reward")
    weights = {
        name: _float_value(_required(section, name), f"reward.{name}")
        for name in (
            "w_position",
            "w_heading",
            "w_control",
            "w_delta_control",
            "w_delta_alpha",
            "w_saturation",
            "w_deadline",
            "w_constraint",
        )
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("reward weights must be non-negative")
    if weights["w_position"] <= 0.0:
        raise ValueError("reward.w_position must be positive")
    if weights["w_constraint"] <= 0.0:
        raise ValueError("reward.w_constraint must be positive")
    position_scale = _float_value(
        _required(section, "position_scale_m"), "reward.position_scale_m"
    )
    heading_scale = _float_value(
        _required(section, "heading_scale_rad"), "reward.heading_scale_rad"
    )
    if position_scale <= 0.0 or heading_scale <= 0.0:
        raise ValueError("reward normalization scales must be positive")
    w_center = _float_value(section.get("w_center", 0.0), "reward.w_center")
    alpha_center = _float_value(
        section.get("alpha_center", 0.5), "reward.alpha_center"
    )
    center_gate = _float_value(
        section.get("center_gate_error_m", 0.0), "reward.center_gate_error_m"
    )
    if w_center < 0.0:
        raise ValueError("reward.w_center must be non-negative")
    if not 0.0 <= alpha_center <= 1.0:
        raise ValueError("reward.alpha_center must be in [0, 1]")
    if center_gate < 0.0:
        raise ValueError("reward.center_gate_error_m must be non-negative")
    return RewardConfig(
        version=_non_empty_string(_required(section, "version"), "reward.version"),
        **weights,
        position_scale_m=position_scale,
        heading_scale_rad=heading_scale,
        w_center=w_center,
        alpha_center=alpha_center,
        center_gate_error_m=center_gate,
    )


def _load_rl(data: dict[str, Any]) -> RLConfig:
    section = _section(data, "rl")
    algorithm = _non_empty_string(_required(section, "algorithm"), "rl.algorithm")
    if algorithm != "td3":
        raise ValueError("rl.algorithm must be td3")
    freeze = _required(section, "freeze_backbone_round1")
    if not isinstance(freeze, bool):
        raise ValueError("rl.freeze_backbone_round1 must be a boolean")
    blocks = _required(section, "unfreeze_blocks_round2")
    if not isinstance(blocks, (list, tuple)) or not blocks:
        raise ValueError("rl.unfreeze_blocks_round2 must be a non-empty list")
    config = RLConfig(
        algorithm=algorithm,
        delta_max=_float_value(_required(section, "delta_max"), "rl.delta_max"),
        lambda_blend=_float_value(_required(section, "lambda_blend"), "rl.lambda_blend"),
        behavior_reg=_float_value(_required(section, "behavior_reg"), "rl.behavior_reg"),
        freeze_backbone_round1=freeze,
        unfreeze_blocks_round2=tuple(str(name) for name in blocks),
        batch_size=_int_value(_required(section, "batch_size"), "rl.batch_size"),
        actor_lr=_float_value(_required(section, "actor_lr"), "rl.actor_lr"),
        critic_lr=_float_value(_required(section, "critic_lr"), "rl.critic_lr"),
        gamma=_float_value(_required(section, "gamma"), "rl.gamma"),
        tau=_float_value(_required(section, "tau"), "rl.tau"),
        policy_delay=_int_value(_required(section, "policy_delay"), "rl.policy_delay"),
        target_noise=_float_value(_required(section, "target_noise"), "rl.target_noise"),
        noise_clip=_float_value(_required(section, "noise_clip"), "rl.noise_clip"),
        expl_noise=_float_value(_required(section, "expl_noise"), "rl.expl_noise"),
        warmup_steps=_int_value(_required(section, "warmup_steps"), "rl.warmup_steps"),
        updates_per_step=_int_value(_required(section, "updates_per_step"), "rl.updates_per_step"),
        max_steps_per_round=_int_value(
            _required(section, "max_steps_per_round"), "rl.max_steps_per_round"
        ),
        warmstart_transitions=_int_value(
            _required(section, "warmstart_transitions"), "rl.warmstart_transitions"
        ),
    )
    if not 0.0 < config.delta_max <= 1.0:
        raise ValueError("rl.delta_max must be in (0, 1]")
    if config.lambda_blend < 0.0 or config.behavior_reg < 0.0:
        raise ValueError("rl.lambda_blend and behavior_reg must be non-negative")
    if not 0.0 < config.gamma < 1.0:
        raise ValueError("rl.gamma must be in (0, 1)")
    if not 0.0 < config.tau <= 1.0:
        raise ValueError("rl.tau must be in (0, 1]")
    if config.policy_delay <= 0 or config.batch_size <= 0 or config.warmup_steps < 0:
        raise ValueError("rl policy_delay/batch_size/warmup_steps are invalid")
    if min(config.actor_lr, config.critic_lr) <= 0.0:
        raise ValueError("rl learning rates must be positive")
    if config.target_noise < 0.0 or config.noise_clip < 0.0 or config.expl_noise < 0.0:
        raise ValueError("rl noise values must be non-negative")
    if config.updates_per_step <= 0 or config.max_steps_per_round <= 0:
        raise ValueError("rl update budgets must be positive")
    if config.warmstart_transitions < 0:
        raise ValueError("rl.warmstart_transitions must be non-negative")
    return config


def load_residual_config(path: str | Path) -> ResidualExperimentConfig:
    """Load and validate a predictive-authority v4 YAML configuration."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("residual config root must be a mapping")

    protocol_data = _section(data, "protocol")
    protocol = ProtocolConfig(
        version=_non_empty_string(_required(protocol_data, "version"), "protocol.version")
    )
    if protocol.version != "pac_residual_runs":
        raise ValueError("protocol.version must be pac_residual_runs")

    environment = _load_supervised_compatible_environment(data)
    controller_data = _section(data, "controller")
    controller = ControllerConfig(
        primary=_non_empty_string(_required(controller_data, "primary"), "controller.primary"),
        authority=_non_empty_string(_required(controller_data, "authority"), "controller.authority"),
    )
    authority_model = _load_supervised_compatible_authority_model(data)
    actuator = _load_supervised_compatible_actuator(data)
    mpc = _load_supervised_compatible_mpc(data)
    oracle = _load_supervised_compatible_oracle(data, mpc)

    world_model = _load_world_model(data)
    disturbances = _load_disturbances(data)
    seeds = _load_seeds(data)
    collection = _load_collection(data)
    reward = _load_reward(data)
    rl = _load_rl(data)

    evaluation_data = _section(data, "evaluation")
    family_seed_count = _int_value(
        _required(evaluation_data, "family_seed_count"),
        "evaluation.family_seed_count",
    )
    if family_seed_count <= 0:
        raise ValueError("evaluation.family_seed_count must be positive")
    if (family_seed_count > len(seeds.eval_seen)
            or family_seed_count > len(seeds.eval_unseen)):
        raise ValueError(
            "evaluation.family_seed_count exceeds the eval seed partitions"
        )
    evaluation = EvaluationPairedConfig(family_seed_count=family_seed_count)

    sspo_data = _section(data, "sspo")
    bias_grid = _float_sequence(
        _required(sspo_data, "bias_grid"), "sspo.bias_grid"
    )
    if len(bias_grid) < 2 or any(
        right <= left for left, right in zip(bias_grid, bias_grid[1:])
    ):
        raise ValueError("sspo.bias_grid must be strictly increasing")
    sspo_iterations = _int_value(
        _required(sspo_data, "iterations"), "sspo.iterations"
    )
    if sspo_iterations <= 0:
        raise ValueError("sspo.iterations must be positive")
    sspo = SSPOCalibrationConfig(
        search_seeds=seeds.sspo_search,
        bias_grid=bias_grid,
        iterations=sspo_iterations,
    )

    wm_authority_data = _section(data, "wm_authority")
    uncertainty_gate = _float_value(
        _required(wm_authority_data, "uncertainty_gate"),
        "wm_authority.uncertainty_gate",
    )
    refresh_steps = _int_value(
        _required(wm_authority_data, "refresh_steps"),
        "wm_authority.refresh_steps",
    )
    if uncertainty_gate <= 0.0:
        raise ValueError("wm_authority.uncertainty_gate must be positive")
    if refresh_steps <= 0:
        raise ValueError("wm_authority.refresh_steps must be positive")
    wm_authority = WMAuthorityConfig(
        uncertainty_gate=uncertainty_gate,
        refresh_steps=refresh_steps,
    )

    backbone_data = _section(data, "backbone")
    outputs_data = _section(data, "outputs")
    config = ResidualExperimentConfig(
        protocol=protocol,
        environment=environment,
        controller=controller,
        authority_model=authority_model,
        actuator=actuator,
        mpc=mpc,
        oracle=oracle,
        world_model=world_model,
        disturbances=disturbances,
        seeds=seeds,
        evaluation=evaluation,
        sspo=sspo,
        wm_authority=wm_authority,
        collection=collection,
        reward=reward,
        rl=rl,
        backbone=BackboneConfig(
            checkpoint_dir=_non_empty_string(
                _required(backbone_data, "checkpoint_dir"), "backbone.checkpoint_dir"
            )
        ),
        outputs=PairedOutputConfig(
            run_root=_non_empty_string(_required(outputs_data, "run_root"), "outputs.run_root")
        ),
    )
    validate_disjoint_seed_partitions(seed_partitions(config))
    _validate_supervised_partition_disjointness(config)
    return config


def seed_partitions(config: ResidualExperimentConfig) -> dict[str, tuple[int, ...]]:
    """Return the independent seed roles used by the residual experiment."""
    return {
        "wm_train": config.seeds.wm_train,
        "wm_val": config.seeds.wm_val,
        "wm_test": config.seeds.wm_test,
        "eval_seen": config.seeds.eval_seen,
        "eval_unseen": config.seeds.eval_unseen,
        "model": config.seeds.model,
        "sspo_search": config.seeds.sspo_search,
    }


_V3_SEED_PARTITIONS = (
    tuple(range(11000, 11008)),
    (12000, 12001),
    tuple(range(31000, 31005)),
    tuple(range(41000, 41020)),
    tuple(range(51000, 51005)),
    tuple(range(61000, 61010)),
    tuple(range(71000, 71010)),
)


def _validate_supervised_partition_disjointness(config: ResidualExperimentConfig) -> None:
    """Environment-seed roles must not collide with any frozen supervised partition."""
    v3_environment_seeds: set[int] = set()
    for partition in _V3_SEED_PARTITIONS:
        v3_environment_seeds.update(partition)
    for role in ("wm_train", "wm_val", "wm_test", "eval_seen", "eval_unseen",
                 "sspo_search"):
        for seed in getattr(config.seeds, role):
            if seed in v3_environment_seeds:
                raise ValueError(
                    f"residual {role} seed {seed} collides with a frozen supervised seed partition"
                )


def _load_supervised_compatible_environment(data: dict[str, Any]) -> EnvironmentConfig:
    """Validate the environment section with v3 semantics (mirrors load_supervised_config)."""
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
        vehicle_profile=_non_empty_string(_required(env, "vehicle_profile"), "environment.vehicle_profile"),
        thruster_layout=_non_empty_string(_required(env, "thruster_layout"), "environment.thruster_layout"),
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
    return environment


def _load_supervised_compatible_authority_model(data: dict[str, Any]) -> AuthorityModelConfig:
    section = _section(data, "authority_model")
    config = AuthorityModelConfig(
        architecture=_non_empty_string(_required(section, "architecture"), "authority_model.architecture"),
        feature_mode=_non_empty_string(_required(section, "feature_mode"), "authority_model.feature_mode"),
        history_len=_int_value(_required(section, "history_len"), "authority_model.history_len"),
        embed_dim=_int_value(_required(section, "embed_dim"), "authority_model.embed_dim"),
        heads=_int_value(_required(section, "heads"), "authority_model.heads"),
        layers=_int_value(_required(section, "layers"), "authority_model.layers"),
        dropout=_float_value(_required(section, "dropout"), "authority_model.dropout"),
        alpha_gain=_float_value(_required(section, "alpha_gain"), "authority_model.alpha_gain"),
        alpha_smoothing=_float_value(_required(section, "alpha_smoothing"), "authority_model.alpha_smoothing"),
        alpha_rate_limit=_float_value(_required(section, "alpha_rate_limit"), "authority_model.alpha_rate_limit"),
        alpha_deadband=_float_value(_required(section, "alpha_deadband"), "authority_model.alpha_deadband"),
    )
    if config.architecture != "transformer":
        raise ValueError("authority_model.architecture must be transformer")
    if config.feature_mode != "state_phase":
        raise ValueError("authority_model.feature_mode must be state_phase")
    if not 1 <= config.history_len <= 512:
        raise ValueError("authority_model.history_len must be in [1, 512]")
    if not 1 <= config.embed_dim <= 2048:
        raise ValueError("authority_model.embed_dim must be in [1, 2048]")
    if not 1 <= config.heads <= config.embed_dim or config.embed_dim % config.heads:
        raise ValueError("authority_model heads must divide embed_dim")
    if not 1 <= config.layers <= 64:
        raise ValueError("authority_model.layers must be in [1, 64]")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("authority_model.dropout must be in [0, 1)")
    if config.alpha_gain < 0.0:
        raise ValueError("authority_model.alpha_gain must be non-negative")
    for name, value in (
        ("alpha_smoothing", config.alpha_smoothing),
        ("alpha_rate_limit", config.alpha_rate_limit),
        ("alpha_deadband", config.alpha_deadband),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"authority_model.{name} must be in [0, 1]")
    return config


def _load_supervised_compatible_actuator(data: dict[str, Any]) -> ActuatorConfig:
    section = _section(data, "actuator")
    config = ActuatorConfig(
        command_min=_float_value(_required(section, "command_min"), "actuator.command_min"),
        command_max=_float_value(_required(section, "command_max"), "actuator.command_max"),
        max_delta_per_step=_float_value(_required(section, "max_delta_per_step"), "actuator.max_delta_per_step"),
        max_force_n=_float_value(_required(section, "max_force_n"), "actuator.max_force_n"),
    )
    if config.command_min >= config.command_max:
        raise ValueError("actuator command_min must be less than command_max")
    if config.max_delta_per_step <= 0.0 or config.max_force_n <= 0.0:
        raise ValueError("actuator limits must be positive")
    return config


def _load_supervised_compatible_mpc(data: dict[str, Any]) -> MPCConfig:
    section = _section(data, "mpc")
    config = MPCConfig(
        horizon=_int_value(_required(section, "horizon"), "mpc.horizon"),
        q_diag=_float_sequence(_required(section, "q_diag"), "q_diag"),
        terminal_scale=_float_value(_required(section, "terminal_scale"), "mpc.terminal_scale"),
        r_diag=_float_sequence(_required(section, "r_diag"), "r_diag"),
        s_diag=_float_sequence(_required(section, "s_diag"), "s_diag"),
        eps_abs=_float_value(_required(section, "eps_abs"), "mpc.eps_abs"),
        eps_rel=_float_value(_required(section, "eps_rel"), "mpc.eps_rel"),
        max_iter=_int_value(_required(section, "max_iter"), "mpc.max_iter"),
        time_limit_s=_float_value(_required(section, "time_limit_s"), "mpc.time_limit_s"),
        accept_inaccurate_residual=_float_value(
            _required(section, "accept_inaccurate_residual"), "mpc.accept_inaccurate_residual"
        ),
        max_consecutive_plan_reuse=_int_value(
            _required(section, "max_consecutive_plan_reuse"), "mpc.max_consecutive_plan_reuse"
        ),
    )
    if config.horizon <= 0:
        raise ValueError("mpc horizon must be positive")
    if len(config.q_diag) != 12 or len(config.r_diag) != 6 or len(config.s_diag) != 6:
        raise ValueError("mpc diagonal vectors have invalid lengths")
    if (
        config.terminal_scale <= 0.0
        or config.eps_abs <= 0.0
        or config.eps_rel <= 0.0
        or config.max_iter <= 0
        or config.time_limit_s <= 0.0
        or config.accept_inaccurate_residual <= 0.0
        or config.max_consecutive_plan_reuse <= 0
    ):
        raise ValueError("mpc limits must be positive")
    if any(value <= 0.0 for value in (*config.q_diag, *config.r_diag, *config.s_diag)):
        raise ValueError("mpc diagonal weights must be positive")
    return config


def _load_supervised_compatible_oracle(data: dict[str, Any], mpc: MPCConfig) -> OracleConfig:
    section = _section(data, "oracle")
    alpha_grid = _float_sequence(_required(section, "alpha_grid"), "oracle.alpha_grid")
    if len(alpha_grid) < 2:
        raise ValueError("oracle.alpha_grid must contain at least two values")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alpha_grid):
        raise ValueError("oracle.alpha_grid values must be in [0, 1]")
    if alpha_grid[0] != 0.0 or alpha_grid[-1] != 1.0:
        raise ValueError("oracle.alpha_grid must include 0 and 1 as endpoints")
    if any(right <= left for left, right in zip(alpha_grid, alpha_grid[1:])):
        raise ValueError("oracle.alpha_grid must be strictly increasing")
    config = OracleConfig(
        horizon=_int_value(_required(section, "horizon"), "oracle.horizon"),
        mpc_solver_time_limit_s=_float_value(
            _required(section, "mpc_solver_time_limit_s"), "oracle.mpc_solver_time_limit_s"
        ),
        alpha_grid=alpha_grid,
        xy_weight=_float_value(_required(section, "xy_weight"), "oracle.xy_weight"),
        z_weight=_float_value(_required(section, "z_weight"), "oracle.z_weight"),
        heading_weight=_float_value(_required(section, "heading_weight"), "oracle.heading_weight"),
        control_delta_weight=_float_value(
            _required(section, "control_delta_weight"), "oracle.control_delta_weight"
        ),
        saturation_weight=_float_value(_required(section, "saturation_weight"), "oracle.saturation_weight"),
        terminal_scale=_float_value(_required(section, "terminal_scale"), "oracle.terminal_scale"),
    )
    if not 1 <= config.horizon <= 200:
        raise ValueError("oracle.horizon must be in [1, 200]")
    if config.mpc_solver_time_limit_s < mpc.time_limit_s:
        raise ValueError("oracle.mpc_solver_time_limit_s must be >= mpc.time_limit_s")
    if any(
        weight < 0.0
        for weight in (
            config.xy_weight,
            config.z_weight,
            config.heading_weight,
            config.control_delta_weight,
            config.saturation_weight,
            config.terminal_scale,
        )
    ):
        raise ValueError("oracle weights must be non-negative")
    return config


__all__ = [
    "ResidualExperimentConfig",
    "load_residual_config",
    "seed_partitions",
    "DisturbanceFamilyConfig",
    "WorldModelConfig",
]
