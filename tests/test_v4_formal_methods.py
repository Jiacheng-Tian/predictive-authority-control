"""Tests for the formal method registry extensions (alpha sweep, rl_round)."""

from __future__ import annotations

from pathlib import Path
import unittest

from pac.v4.config import load_v4_config
from pac.v4.eval.formal import method_grid

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pac_v4.yaml"


class MethodGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_v4_config(_CONFIG_PATH)

    def test_default_grid_unchanged(self):
        combos = method_grid(self.config, include_wm_rl=False)
        methods = [combo["method"] for combo in combos]
        self.assertEqual(methods.count("smc"), 1)
        self.assertEqual(methods.count("mpc"), 1)
        self.assertEqual(methods.count("constant_alpha"), 1)
        self.assertNotIn("constant_alpha_0.25", methods)
        self.assertNotIn("constant_alpha_0.75", methods)
        self.assertNotIn("direct_rl", methods)

    def test_alpha_sweep_adds_two_probes(self):
        base = method_grid(self.config, include_wm_rl=False)
        combos = method_grid(self.config, include_wm_rl=False, alpha_sweep=True)
        methods = [combo["method"] for combo in combos]
        self.assertEqual(methods.count("constant_alpha_0.25"), 1)
        self.assertEqual(methods.count("constant_alpha_0.75"), 1)
        self.assertEqual(len(combos), len(base) + 2)

    def test_direct_rl_adds_model_seeds(self):
        combos = method_grid(
            self.config, include_wm_rl=False, include_direct_rl=True
        )
        direct = [combo for combo in combos if combo["method"] == "direct_rl"]
        self.assertEqual(
            sorted(combo["model_seed"] for combo in direct),
            sorted(int(seed) for seed in self.config.seeds.model),
        )

    def test_include_wm_rl_grid_size(self):
        combos = method_grid(self.config, include_wm_rl=True)
        self.assertEqual(len(combos), 24)

    def test_methods_override_restricts_grid(self):
        combos = method_grid(
            self.config, include_wm_rl=False,
            methods=["constant_alpha_0.25", "constant_alpha_0.75"],
        )
        self.assertEqual(
            [combo["method"] for combo in combos],
            ["constant_alpha_0.25", "constant_alpha_0.75"],
        )
        self.assertTrue(all(combo["model_seed"] is None for combo in combos))
        seeded = method_grid(
            self.config, include_wm_rl=False, methods=["direct_rl"]
        )
        self.assertEqual(len(seeded), len(self.config.seeds.model))
        with self.assertRaises(ValueError):
            method_grid(self.config, include_wm_rl=False, methods=["nope"])


class ConstantAlphaPolicyResolutionTests(unittest.TestCase):
    def test_probes_resolve_to_matching_alpha(self):
        from pac.v4.eval.formal import MethodFactory
        from pac.v4.eval.runner import ConstantAlphaPolicy

        config = load_v4_config(_CONFIG_PATH)
        factory = MethodFactory(
            config, rl_dir=Path("."), wm_rl_dir=Path("."),
            wm_checkpoint=Path("."), dataset_dir=Path("."),
            sspo_schedule=None, include_wm_rl=False,
        )
        for name, alpha in (("constant_alpha_0.25", 0.25),
                            ("constant_alpha_0.75", 0.75)):
            policy = factory.policy_for(name, None)
            self.assertIsInstance(policy, ConstantAlphaPolicy)
            self.assertEqual(policy.alpha, alpha)
            self.assertFalse(policy.direct_action)
        with self.assertRaises(ValueError):
            factory.policy_for("constant_alpha_1.5", None)

    def test_direct_rl_resolves_to_live_policy(self):
        """Regression: the checkpoint loader returns (policy, metadata)."""
        import tempfile

        import torch

        from pac.v4.eval.formal import MethodFactory
        from pac.v4.rl.direct import (
            DirectLivePolicy,
            DirectRLPolicy,
            save_direct_rl_checkpoint,
        )

        config = load_v4_config(_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            seed = int(config.seeds.model[0])
            policy = DirectRLPolicy(int(config.authority_model.history_len))
            save_direct_rl_checkpoint(
                Path(temp_dir) / f"direct_rl_seed_{seed}.pt",
                policy,
                model_seed=seed,
                reward_version=str(config.reward.version),
                training_summary={"env_steps": 1},
            )
            factory = MethodFactory(
                config, rl_dir=Path("."), wm_rl_dir=Path("."),
                wm_checkpoint=Path("."), dataset_dir=Path("."),
                sspo_schedule=None, include_wm_rl=False,
                direct_rl_dir=Path(temp_dir),
            )
            live = factory.policy_for("direct_rl", seed)
            self.assertIsInstance(live, DirectLivePolicy)
            self.assertIsInstance(live.policy, DirectRLPolicy)
            action = live.policy(torch.randn(2, 16, 24))
            self.assertEqual(tuple(action.shape), (2, 6))


    def test_alpha_squash_wraps_residual_policy(self):
        """Stage 3: the factory applies the deployment temperature."""
        import tempfile

        import torch

        from pac.authority.model import load_alpha_model_checkpoint
        from pac.v4.eval.formal import MethodFactory
        from pac.v4.eval.runner import SquashAlphaPolicy
        from pac.v4.rl.policy import (
            ResidualAlphaPolicy,
            save_rl_checkpoint,
        )

        config = load_v4_config(_CONFIG_PATH)
        seed = int(config.seeds.model[0])
        backbone, _meta = load_alpha_model_checkpoint(
            Path(config.backbone.checkpoint_dir)
            / f"pac_train_seed_{seed}" / "checkpoint.pt"
        )
        policy = ResidualAlphaPolicy(
            backbone, delta_max=0.25, lambda_blend=1.0
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            save_rl_checkpoint(
                Path(temp_dir) / f"residual_rl_seed_{seed}_round1.pt",
                policy,
                model_seed=seed,
                round_index=1,
                backbone_checkpoint="x",
                reward_version="r",
                wm_version="none",
                training_summary={},
            )
            factory = MethodFactory(
                config, rl_dir=Path(temp_dir), wm_rl_dir=Path(temp_dir),
                wm_checkpoint=Path("."), dataset_dir=Path("."),
                sspo_schedule=None, include_wm_rl=False,
                alpha_squash=0.4,
            )
            live = factory.policy_for("residual_rl", seed)
            self.assertIsInstance(live, SquashAlphaPolicy)
            self.assertEqual(live.temperature, 0.4)

    def test_squash_alpha_math(self):
        from pac.v4.eval.runner import ConstantAlphaPolicy, SquashAlphaPolicy

        inner = ConstantAlphaPolicy(0.8)
        squash = SquashAlphaPolicy(inner, temperature=0.4, center=0.5)

        class _Ctx:
            pass

        raw = squash.select_alpha(_Ctx())
        self.assertAlmostEqual(raw, 0.5 + 0.4 * (0.8 - 0.5), places=12)
        identity = SquashAlphaPolicy(ConstantAlphaPolicy(0.3), 1.0)
        self.assertAlmostEqual(identity.select_alpha(_Ctx()), 0.3, places=12)
        with self.assertRaises(ValueError):
            SquashAlphaPolicy(inner, 1.5)
        with self.assertRaises(ValueError):
            SquashAlphaPolicy(inner, 0.0)


if __name__ == "__main__":
    unittest.main()
