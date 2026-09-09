from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch


class _Controller:
    def __init__(self, forced: bool):
        self.force_primary_authority = forced
        self._action = np.full(6, 0.1 if forced else 0.2)

    def reset(self):
        pass

    def set_trajectory3d(self, enabled=True):
        del enabled

    def compute(self, *args, **kwargs):
        del args, kwargs
        return self._action.copy()


class _Model:
    def eval(self):
        return self

    def __call__(self, value):
        return torch.ones(value.shape[0], 1)


class AuthorityForcePrimaryTest(unittest.TestCase):
    def test_forced_authority_overrides_filtered_alpha_and_is_recorded(self):
        from pac.authority.evaluation import run_predictive_alpha_episode

        primary = _Controller(forced=False)
        authority = _Controller(forced=True)
        with patch(
            "pac.authority.evaluation.build_controller",
            side_effect=[("primary", primary), ("authority", authority)],
        ):
            metrics = run_predictive_alpha_episode(
                model=_Model(),
                scenario=1,
                seed=0,
                steps=1,
                mass_scale_xy=1.0,
                damping_scale_xy=1.0,
                current_amplitude_scale=1.0,
                current_frequency_scale=1.0,
                vertical_current=0.0,
                primary_controller="primary",
                authority_controller="authority",
                feature_mode="state_phase",
                alpha_gain=1.0,
                alpha_smoothing=0.0,
                alpha_rate_limit=1.0,
                save_ts=True,
            )

        self.assertTrue(metrics["authority_forced_primary"])
        self.assertEqual(metrics["authority_forced_primary_fraction"], 1.0)
        self.assertEqual(metrics["ts"]["authority_forced_primary"].tolist(), [True])
        self.assertEqual(metrics["ts"]["authority_alpha"].tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
