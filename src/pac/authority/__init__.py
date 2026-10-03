"""PAC authority-model training, rollout oracles, and datasets."""

from .dataset import OracleDataset, dataset_content_hash
from .model import load_supervised_checkpoint, save_supervised_checkpoint, train_alpha_model_supervised
from .oracle import (
    OracleDecision,
    OracleSettings,
    choose_rollout_oracle_alpha,
    validate_formal_oracle_contract,
)

__all__ = [
    "OracleDataset",
    "dataset_content_hash",
    "train_alpha_model_supervised",
    "save_supervised_checkpoint",
    "load_supervised_checkpoint",
    "OracleDecision",
    "OracleSettings",
    "choose_rollout_oracle_alpha",
    "validate_formal_oracle_contract",
]
