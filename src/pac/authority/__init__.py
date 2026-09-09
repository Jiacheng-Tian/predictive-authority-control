"""PAC authority-model training, rollout oracles, and datasets."""

from .dataset import OracleDataset, dataset_content_hash
from .model import load_v3_checkpoint, save_v3_checkpoint, train_alpha_model_v3
from .oracle import (
    OracleDecision,
    OracleSettings,
    choose_rollout_oracle_alpha,
    validate_formal_oracle_contract,
)

__all__ = [
    "OracleDataset",
    "dataset_content_hash",
    "train_alpha_model_v3",
    "save_v3_checkpoint",
    "load_v3_checkpoint",
    "OracleDecision",
    "OracleSettings",
    "choose_rollout_oracle_alpha",
    "validate_formal_oracle_contract",
]
