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

    def test_reference_fk_is_diagnostic_not_a_training_gate(self):
        for name, env in self.envs.items():
            self.assertFalse(env["validateReferenceFK"], name)

    def test_controller_keeps_legacy_residual_authority(self):
        source = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "env"
            / "tasks"
            / "intermimic.py"
        ).read_text()
        start = source.index("    def _action_to_pd_targets")
        end = source.index("    def _compute_reward", start)
        controller = source[start:end]
        self.assertIn("self.progress_buf + 1", controller)
        self.assertIn("action.clamp(-1.0, 1.0)", controller)
        self.assertNotIn("_residual_limit_per_dof", controller)
        self.assertNotIn("dof_limits_upper", controller)

    def test_rsi_reset_restores_reference_velocities(self):
        source = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "env"
            / "tasks"
            / "intermimic.py"
        ).read_text()
        target_start = source.index("    def _reset_target")
        target_end = source.index("    def _reset_env_tensors", target_start)
        target_reset = source[target_start:target_end]
        state_start = source.index("    def _set_env_state")
        state_end = source.index("    def _compute_task_obs", state_start)
        state_reset = source[state_start:state_end]

        self.assertNotIn("self.init_vel", target_reset)
        self.assertIn("f'{prefix}_pos_vel'", target_reset)
        self.assertIn("f'{prefix}_rot_vel'", target_reset)
        self.assertNotIn("self.init_vel", state_reset)
        self.assertIn("= root_vel", state_reset)
        self.assertIn("= root_ang_vel", state_reset)
        self.assertIn("= dof_vel", state_reset)

    def test_psi_buffer_uses_final_rollout_length(self):
        source = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "env"
            / "tasks"
            / "intermimic.py"
        ).read_text()
        self.assertIn(
            "buf_len = max(self.rollout_length, "
            "cfg['env']['episodeLength'])",
            source,
        )

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

    def test_checkpoint_history_is_retained_every_100_epochs(self):
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
        self.assertEqual(config["save_frequency"], 100)
        self.assertFalse(config["save_intermediate"])
        self.assertEqual(
            config["checkpoint_milestones"],
            list(range(100, 2001, 100)),
        )
        agent = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "learning"
            / "intermimic_agent.py"
        ).read_text()
        runner = (
            REPO_ROOT
            / "isaacgym"
            / "scripts"
            / "run_theia_policy_seed.sh"
        ).read_text()
        self.assertIn("+ '_epoch_'", agent)
        self.assertIn('"$nn_dir"/mimic_epoch_*.pth', runner)


if __name__ == "__main__":
    unittest.main()
