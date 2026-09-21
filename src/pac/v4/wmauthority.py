"""Uncertainty-gated world-model authority for v4.

The v4-WM deployment policy: the frozen Transformer alpha is the default;
every ``wm_authority.refresh_steps`` control steps the conservative
ensemble rolls the 11-point candidate grid out, and while the ensemble
disagreement stays below ``wm_authority.uncertainty_gate`` the grid argmin
alpha takes over.  Above the gate (parameter mismatch, out-of-distribution
states) the policy falls back to the Transformer — making the pure
Transformer and the pure WM-grid both degenerate cases of this one policy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from pac.v4.eval.runner import WMAdvice
from pac.v4.worldmodel.model import EnsembleDynamicsModel, load_world_model
from pac.v4.worldmodel.ranking import CandidateRolloutEvaluator, RankingWindow
from pac.v4.worldmodel.train import dataset_content_hash
from pac.v4.worldmodel.validate import ConservativeEnsemble, train_residual_sigma


class WMAuthorityComputer:
    """Computes :class:`WMAdvice` at decision steps (deployable inputs only)."""

    def __init__(
            self,
            *,
            config,
            ensemble: EnsembleDynamicsModel,
            conservative: ConservativeEnsemble,
            evaluator: CandidateRolloutEvaluator,
            uncertainty_gate: float,
            device: torch.device):
        self.config = config
        self.ensemble = ensemble
        self.conservative = conservative
        self.evaluator = evaluator
        self.uncertainty_gate = float(uncertainty_gate)
        self.device = device

    def advise(
            self,
            *,
            state,
            est_current,
            context,
            scenario_id: int,
            wm_window,
            previous_applied,
            plan,
            stamp: float) -> WMAdvice:
        started = time.perf_counter()
        window_tensor = torch.from_numpy(
            np.asarray(wm_window, dtype=np.float32)[None]
        ).to(self.device)
        deltas = []
        with torch.no_grad():
            for member in self.ensemble.members:
                delta_state, _delta_current = member.predict_delta(window_tensor)
                deltas.append(delta_state[0].cpu().numpy())
        stacked = np.stack(deltas, axis=0)
        position_std = float(np.mean(stacked.std(axis=0, ddof=0)[:3]))
        gate_open = bool(position_std < self.uncertainty_gate)
        best_alpha = float("nan")
        best_cost = float("nan")
        spread = float("nan")
        if gate_open:
            window = RankingWindow(
                episode_uid="live",
                family="live",
                split="live",
                behavior="wm_authority",
                scenario_id=int(scenario_id),
                environment_seed=0,
                row=0,
                step=0,
                initial_state=np.asarray(state, dtype=np.float32),
                window=np.asarray(wm_window, dtype=np.float32),
                previous_applied=np.asarray(previous_applied, dtype=np.float32),
                est_current=np.asarray(est_current, dtype=np.float32),
                context=np.asarray(context, dtype=np.float32),
                plan=np.asarray(plan, dtype=np.float32),
                realized_currents=np.zeros(
                    (int(self.config.oracle.horizon), 3), dtype=np.float32
                ),
                sample_time=float(stamp),
            )
            costs = self.evaluator.single_model_costs(self.conservative, window)
            best_index = int(np.argmin(costs))
            grid = np.asarray(self.config.oracle.alpha_grid, dtype=float)
            best_alpha = float(grid[best_index])
            best_cost = float(np.min(costs))
            spread = float(np.max(costs) - np.min(costs))
        return WMAdvice(
            best_alpha=best_alpha,
            best_cost=best_cost,
            cost_spread=spread,
            uncertainty=position_std,
            gate_open=gate_open,
            compute_seconds=time.perf_counter() - started,
        )


class WMHybridPolicy:
    """v4-WM: Transformer alpha by default, grid alpha under the gate."""

    uses_wm = False
    direct_action = False

    def __init__(self, backbone, *, alpha_gain: float):
        self.backbone = backbone
        self.alpha_gain = float(alpha_gain)
        self.backbone.eval()

    def reset(self, spec, episode_end: float) -> None:
        del spec, episode_end

    def select_alpha(self, context) -> float:
        import torch as _torch

        with _torch.no_grad():
            model_input = _torch.from_numpy(
                np.asarray(context.feature_window, dtype=np.float32)[None]
            )
            base = float(self.backbone(model_input).item())
        if context.wm is not None and context.wm.gate_open:
            # Pre-divide by the gain so the grid alpha survives the shared
            # gain/clip stage unchanged; smoothing/rate limits still apply.
            return float(context.wm.best_alpha) / self.alpha_gain
        return base


def build_wm_authority_computer(
        config,
        wm_checkpoint,
        dataset_dir,
        *,
        device: str | torch.device = "cpu") -> tuple[WMAuthorityComputer, str]:
    """Assemble the computer from a frozen stage-one checkpoint."""
    resolved = torch.device(device)
    dataset_hash = dataset_content_hash(dataset_dir)
    ensemble, metadata = load_world_model(
        wm_checkpoint, expected_dataset_hash=dataset_hash
    )
    ensemble.eval()
    from pac.v4.collector import load_transition_dataset

    dataset = load_transition_dataset(dataset_dir)
    sigma = train_residual_sigma(dataset)
    conservative = ConservativeEnsemble(ensemble, sigma["state"], sigma["current"])
    evaluator = CandidateRolloutEvaluator(config, device=str(resolved))
    computer = WMAuthorityComputer(
        config=config,
        ensemble=ensemble,
        conservative=conservative,
        evaluator=evaluator,
        uncertainty_gate=float(config.wm_authority.uncertainty_gate),
        device=resolved,
    )
    version = f"{metadata.get('dataset_hash', dataset_hash)[:12]}:{metadata.get('members', '?')}m"
    return computer, version


__all__ = [
    "WMAuthorityComputer",
    "WMHybridPolicy",
    "build_wm_authority_computer",
]
