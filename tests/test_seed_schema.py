from __future__ import annotations

import unittest


class SeedSchemaTest(unittest.TestCase):
    def test_episode_uid_uses_only_scenario_and_episode_seed(self):
        from pac.evaluation.seeds import episode_uid

        self.assertEqual(episode_uid(3, 41007), "scn3_env41007")
        self.assertEqual(episode_uid(3, 41007), episode_uid(3, 41007))
        self.assertNotEqual(episode_uid(4, 41007), episode_uid(3, 41007))
        self.assertNotEqual(episode_uid(3, 41008), episode_uid(3, 41007))

    def test_episode_uid_has_no_model_seed_parameter(self):
        from inspect import signature

        from pac.evaluation.seeds import episode_uid

        self.assertEqual(tuple(signature(episode_uid).parameters), ("scenario_id", "episode_seed"))

    def test_validate_disjoint_seed_partitions_rejects_duplicate_and_overlap(self):
        from pac.evaluation.seeds import validate_disjoint_seed_partitions

        with self.assertRaisesRegex(ValueError, r"model.*31000"):
            validate_disjoint_seed_partitions({"model": (31000, 31000)})
        with self.assertRaisesRegex(ValueError, r"search.*eval.*51000|eval.*search.*51000"):
            validate_disjoint_seed_partitions({"search": (51000,), "eval": (51000,)})
        with self.assertRaisesRegex(ValueError, r"model.*-1"):
            validate_disjoint_seed_partitions({"model": (-1,)})
        with self.assertRaisesRegex(ValueError, r"model.*18446744073709551616"):
            validate_disjoint_seed_partitions({"model": (2**64,)})


if __name__ == "__main__":
    unittest.main()
