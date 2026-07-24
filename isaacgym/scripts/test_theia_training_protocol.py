#!/usr/bin/env python3
"""Static regression tests for the formal Theia training protocol."""

import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CONFIGS = {
    name: REPO_ROOT
    / "isaacgym"
    / "src"
    / "intermimic"
    / "data"
    / "cfg"
    / filename
    for name, filename in {
        "bootstrap": "theia_full_train.yaml",
        "finetune": "theia_full_finetune.yaml",
        "evaluation": "theia_policy_eval.yaml",
    }.items()
}


def load_env(path):
    with path.open() as source:
        return yaml.safe_load(source)["env"]


class FormalTrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {
            name: load_env(path) for name, path in ENV_CONFIGS.items()
        }

    def test_contact_objective_is_identical_across_stages(self):
        keys = (
            "contactRewardMode",
            "enableContactFailureTermination",
            "enableWristFailureTermination",
            "enableObjectContactPhaseTermination",
            "contactFailureGraceFrames",
            "fingerBonusWeight",
            "wristBonusWeight",
            "graspBonusWeight",
            "wrongContactPenalty",
        )
        expected = {
            key: self.envs["bootstrap"][key] for key in keys
        }
        for name, env in self.envs.items():
            self.assertEqual(
                {key: env[key] for key in keys},
                expected,
                msg=f"{name} changed the shared Raw/Full contact objective",
            )

    def test_full_scale_diagnostics_are_disabled(self):
        for name, env in self.envs.items():
            self.assertFalse(env["enableTrainingDiagnostics"], name)
            self.assertFalse(env["enableStepDiagnostics"], name)

    def test_rollout_is_not_dataset_outlier_dependent(self):
        for name, env in self.envs.items():
            self.assertFalse(
                env["adaptiveRolloutFromLatestContact"], name
            )
        self.assertEqual(self.envs["bootstrap"]["rolloutLength"], 100)

    def test_resource_sampling_is_nonblocking(self):
        source = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "learning"
            / "intermimic_agent.py"
        ).read_text()
        self.assertNotIn("cpu_percent(interval=1)", source)
        self.assertNotIn("subprocess.run(['nvidia-smi'", source)

    def test_checkpoint_io_is_bounded(self):
        train = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "data"
            / "cfg"
            / "train"
            / "rlg"
            / "theia.yaml"
        )
        config = yaml.safe_load(train.read_text())["params"]["config"]
        self.assertFalse(config["save_reward_best"])
        self.assertGreaterEqual(config["save_frequency"], 250)
        self.assertEqual(
            config["checkpoint_milestones"],
            [2000, 5000, 10000, 15000, 20000, 22000],
        )


if __name__ == "__main__":
    unittest.main()
