# Theia x InterMimic: Dual-Object Human-Object Interaction Policy

This fork extends [InterMimic](https://github.com/JudyYe/haptic) to support **dual-object** SMPLX human-object interaction with detailed finger grasping, trained from motion capture data.

## Key Changes from Original InterMimic

### Architecture
- **Dual-object support**: Each environment contains 2 objects (e.g., CupBlue + KettleGreen) with independent physics, rewards, and tracking.
- **Residual PD control**: Policy outputs corrections around reference motion (`action=0` = follow reference). Per-DOF scaling: body=0.3, wrist=0.5, fingers=0.6.
- **Data-driven joint ranges**: Joint limits computed from actual motion data (2x margin), replacing the overly wide default ranges.
- **Adaptive rollout length**: Automatically extends to cover all contact transition windows in the data.
- **Transition-weighted RSI**: Reference State Initialization samples approach-to-grasp transitions 4x more frequently.

### Reward Structure
```
rew = rb * ro_safe * rig * rcg        (multiplicative: body + object + interaction + contact)
    + 0.05 * r_finger                  (finger DOF tracking)
    + 0.3  * wrist_bonus               (wrist precision during contact phases)
    + 0.05 * grasp_bonus               (physical contact success)
```

### Reset Conditions
- Wrist tracking failure: either wrist > 15cm from reference for 20 consecutive frames
- Object trajectory failure: object deviates > 30% of its size during contact phase for 20 frames
- Standard humanoid/object/IG resets from InterMimic

### Physics
- `objectDensity: 5000`, `substeps: 4`, `num_position_iterations: 8`
- Object friction: 1.0, hand friction: 1.0, restitution: 0.0

## Quick Start

### Prerequisites
```bash
# Must activate the environment before running ANY script
conda activate intermimic
```

### Data Preparation
Training data (`theia_data/`) is included in this repo. To regenerate from raw motion capture:
```bash
python scripts/theia2intermimic.py \
    --data_dir data/testset/S1L33P01T0508V01 \
    --objects_dir data/objects \
    --output_dir thirdparty/InterMimic/theia_data
```

### Three Essential Scripts

> **`bash isaacgym/scripts/train_all.sh`** — Train the dual-object policy (headless, auto-resume from checkpoint)

> **`bash isaacgym/scripts/test_theia.sh`** — Visualize the trained policy with GUI

> **`bash isaacgym/scripts/replay_theia.sh`** — Replay reference motion data (no policy, pure kinematic)

### Other Scripts

| Script | Description |
|--------|-------------|
| `isaacgym/scripts/train_theia.sh` | Simple headless training (no auto-resume) |
| `isaacgym/scripts/train_theia_vis.sh` | Training with GUI preview (4 envs) |
| `isaacgym/scripts/diag_test.sh` | Run diagnostic test with per-step logging |

### Pre-trained Checkpoint
A pre-trained checkpoint is included at `checkpoints/theia_dual/theia_smplx/nn/mimic.pth` (via Git LFS). To test:
```bash
bash isaacgym/scripts/test_theia.sh
```
To resume training from this checkpoint:
```bash
bash isaacgym/scripts/train_all.sh
```

### TensorBoard
```bash
tensorboard --logdir checkpoints/ --port 6006
```

Key metrics to monitor:
- `human_sub/rp`, `human_sub/rr` — body position/rotation tracking
- `sub_rewards/wrist_bonus`, `sub_rewards/grasp_bonus` — interaction quality
- `object_sub/ro1`, `object_sub/ro2` — per-object tracking
- `reset_rates/contact` — contact matching (lower = better)

## File Structure

```
isaacgym/
  scripts/
    train_all.sh          # Main training script
    test_theia.sh         # Visualization
    replay_theia.sh       # Data replay
  src/intermimic/
    env/tasks/
      intermimic.py       # Core environment (dual-object, residual control, rewards)
      humanoid.py          # Base humanoid (aggregate size fix)
    data/
      cfg/
        theia_train.yaml   # Environment config
        train/rlg/theia.yaml  # RL hyperparameters
      assets/
        smplx/theia.xml    # SMPLX humanoid skeleton
        objects/            # Object URDFs and meshes
checkpoints/
  theia_dual/              # Pre-trained dual-object checkpoint
```
