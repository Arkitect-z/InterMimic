#!/usr/bin/env python3
"""Static regression tests for the formal Theia training protocol."""

import ast
import os
import unittest
from pathlib import Path

import torch
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


def load_psi_checkpoint_test_class():
    """Load only InterMimic's pure checkpoint methods without Isaac Gym."""
    path = (
        REPO_ROOT
        / "isaacgym"
        / "src"
        / "intermimic"
        / "env"
        / "tasks"
        / "intermimic.py"
    )
    module = ast.parse(path.read_text())
    intermimic = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "InterMimic"
    )
    selected_names = {
        "_psi_checkpoint_metadata",
        "get_env_state",
        "set_env_state",
    }
    selected = [
        node
        for node in intermimic.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name)
                and target.id == "_ENV_STATE_SCHEMA_VERSION"
                for target in (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in selected_names
        )
    ]
    test_module = ast.parse("class CheckpointableTask:\n    pass\n")
    test_module.body[0].body = selected
    ast.fix_missing_locations(test_module)
    namespace = {"os": os, "torch": torch}
    exec(compile(test_module, str(path), "exec"), namespace)
    return namespace["CheckpointableTask"]


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
            "buf_len = max(2, int(self.rollout_length))",
            source,
        )
        allocation = source[
            source.index("        if self.psi > 1:")
            :source.index("        self._build_target_tensors()")
        ]
        self.assertNotIn("episodeLength", allocation)

    def test_psi_curriculum_checkpoint_round_trip_and_validation(self):
        task_class = load_psi_checkpoint_test_class()

        def build_task(fill):
            task = task_class()
            task.psi = 3
            task.num_motions = 1
            task.rollout_length = 4
            task.motion_file = ["/frozen/sub1_A+B_ref.pt"]
            task.max_episode_length = torch.tensor([5])
            task.hoi_refs = torch.full((1, 3, 5, 4), fill)
            task.ref_reward = torch.full((1, 3, 5), fill)
            return task

        source = build_task(2.0)
        source.hoi_refs[:, 0] = 10.0
        source.ref_reward[:, 0] = 1.0
        state = source.get_env_state()
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(state["physical_buffer_size"], 3)
        self.assertEqual(state["motion_files"], ["sub1_A+B_ref.pt"])
        self.assertEqual(state["synthetic_hoi_refs"].device.type, "cpu")
        self.assertEqual(state["synthetic_ref_reward"].device.type, "cpu")

        restored = build_task(-1.0)
        restored.hoi_refs[:, 0] = 20.0
        restored.ref_reward[:, 0] = 1.0
        restored.set_env_state(state)
        self.assertTrue(
            torch.equal(restored.hoi_refs[:, 1:], source.hoi_refs[:, 1:])
        )
        self.assertTrue(
            torch.equal(
                restored.ref_reward[:, 1:], source.ref_reward[:, 1:]
            )
        )
        self.assertTrue(
            torch.equal(
                restored.hoi_refs[:, 0],
                torch.full_like(restored.hoi_refs[:, 0], 20.0),
            )
        )

        with self.assertRaisesRegex(ValueError, "no PSI curriculum state"):
            restored.set_env_state(None)
        wrong_schema = dict(state, schema_version=2)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            restored.set_env_state(wrong_schema)
        wrong_shape = dict(
            state,
            synthetic_hoi_refs=state["synthetic_hoi_refs"][:, :, :-1],
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            restored.set_env_state(wrong_shape)
        nonfinite = state["synthetic_ref_reward"].clone()
        nonfinite[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            restored.set_env_state(
                dict(state, synthetic_ref_reward=nonfinite)
            )

    def test_psi_checkpoint_state_is_forwarded_and_required_for_resume(self):
        vec_task = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "env"
            / "tasks"
            / "vec_task.py"
        ).read_text()
        runner = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "run.py"
        ).read_text()
        seed_runner = (
            REPO_ROOT
            / "isaacgym"
            / "scripts"
            / "run_theia_policy_seed.sh"
        ).read_text()
        for source in (vec_task, runner):
            self.assertIn("    def get_env_state(self):", source)
            self.assertIn("    def set_env_state(self, state):", source)
        self.assertIn(
            '"checkpoint has no restorable PSI curriculum state', seed_runner
        )
        self.assertIn('"synthetic_hoi_refs"', seed_runner)
        self.assertIn('"synthetic_ref_reward"', seed_runner)
        self.assertIn("CHECKPOINT_CANDIDATE_COUNT", seed_runner)
        self.assertIn(
            "Refusing to restart from scratch in the same output directory",
            seed_runner,
        )

    def test_resume_budget_is_relative_to_the_restored_epoch(self):
        common_agent = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "learning"
            / "common_agent.py"
        ).read_text()
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
        self.assertIn("self.epoch_num_start = weights['epoch']", common_agent)
        self.assertIn(
            "self.epoch_num - self.epoch_num_start >= self.max_epochs",
            agent,
        )
        self.assertIn(
            "local remaining=$((target_epoch - current_epoch))", runner
        )
        self.assertIn('MAX_ITERATIONS="$remaining"', runner)

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

    def test_checkpoint_history_is_retained_every_200_epochs(self):
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
        self.assertEqual(config["save_frequency"], 200)
        self.assertFalse(config["save_intermediate"])
        self.assertEqual(
            config["checkpoint_milestones"],
            list(range(200, 2001, 200)),
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
