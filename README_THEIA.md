# Theia x InterMimic: Dual-Object Human-Object Interaction Policy

This fork extends [InterMimic](https://github.com/JudyYe/haptic) to support **dual-object** SMPLX human-object interaction with detailed finger grasping, trained from motion capture data.

## Key Changes from Original InterMimic

### Architecture
- **Dual-object support**: Each environment contains 2 objects (e.g., CupBlue + KettleGreen) with independent physics, rewards, and tracking.
- **Residual PD control**: Policy outputs bounded corrections around the next reference frame (`action=0` = follow reference). Limits are body=0.30 rad, wrist=0.40 rad, fingers=0.45 rad, followed by XML DOF-limit clamping.
- **Data-driven joint ranges**: Joint limits computed from actual motion data (2x margin), replacing the overly wide default ranges.
- **Adaptive rollout length**: Automatically extends to cover all contact transition windows in the data.
- **Transition-weighted RSI**: Reference State Initialization samples approach-to-grasp transitions 4x more frequently.

### Reward Structure
```
rew = rb * ro * rig * rcg             (body + object + interaction + contact)
    + 0.05 * r_finger                  (finger DOF tracking)
    + 0.30 * wrist_bonus               (wrist precision during contact phases)
    + 0.05 * grasp_bonus               (correct hand-object pair)
```

The training reward uses a fast GPU proxy: a hand link must have PhysX net
contact force and be near the intended object's surface. It is not an
actor-pair contact signal, so it is never used for a wrong-contact penalty.
It is also never used for a hard contact-miss termination. Reference contact
remains the original InterMimic shaping signal. Release evaluation can
additionally query exact PhysX actor-pair records on CPU PhysX.

### Reset Conditions
- Standard humanoid/object/IG resets from InterMimic
- Sustained wrist or contact-phase object-trajectory failures
- GT contact misses remain diagnostics and do not terminate rollouts

### Physics
- Per-object density: CupBlue=1200 kg/m³, KettleGreen=1850 kg/m³
- Verified Isaac masses: CupBlue=0.309619 kg, KettleGreen=1.105104 kg
- `substeps: 4`, `num_position_iterations: 8`
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
cd ../..
python toolkit/scripts/theia2intermimic.py \
    --data_dir data/testset/S1L33P01T0508V01 \
    --objects_dir data/objects \
    --output_dir thirdparty/InterMimic/theia_data
```

The converter writes 30 Hz data, uses collision geometry for ground alignment, and marks only hand contact as explicitly required/forbidden. Other bodies use the neutral contact label.

### Local Validation and Full-data Preparation

Fast fixed-seed 128-episode regression evaluation (GPU PhysX is not bitwise
deterministic):

```bash
NUM_ENVS=128 bash isaacgym/scripts/eval_theia.sh \
  checkpoints/theia_dual/theia_smplx/nn/mimic.pth
```

This mode labels its contact metrics as proxies. For exact hand/object/table
actor-pair contact, run CPU PhysX explicitly:

```bash
STRICT_CONTACTS=1 NUM_ENVS=128 bash isaacgym/scripts/eval_theia.sh \
  checkpoints/theia_dual/theia_smplx/nn/mimic.pth
```

The exact-contact run uses a different PhysX backend and is slower; report it
separately from the nominal GPU result. Every run writes `manifest.txt`,
`eval.log`, `summary.json`, and per-episode `episodes.csv` under `evaluation/`.
Evaluation exits nonzero unless it records exactly `NUM_ENVS` initial episodes.

The rebuttal Raw-vs-Refined experiment trains each unseen S1 reference once
per condition from random policy weights. A GPU worker consumes a server-side
reference list:

```bash
python isaacgym/scripts/check_theia_server_versions.py

CUDA_VISIBLE_DEVICES=0 \
NUM_ENVS=2048 \
MINIBATCH_SIZE=16384 \
bash isaacgym/scripts/run_theia_policy_reference_list.sh \
  /server/manifests/cluster_a_gpu0.txt \
  /server/experiments/theia_policy_ab_data \
  /server/experiments/theia_policy_ab_runs
```

For every listed reference, the worker trains one Raw and one
measured-tactile-refined policy for 3000 Hybrid/RSI epochs using the same
fixed seed, then performs K=10 parallel evaluation rollouts. The seed fixes
the initialization; it is not a request for repeated training. Different
cluster/GPU workers receive disjoint reference lists. Runs are resumable and
training-time diagnostics are disabled. New objects use an approximate
default density of 1000 kg/m³, so exact density entries are optional.

See [HANDOFF_SERVER_POLICY_AB.md](HANDOFF_SERVER_POLICY_AB.md) for frozen data
preparation, the pinned Theia/InterMimic/ProtoMotions release, list sharding,
smoke tests and the final paired result aggregator. The exact repository
contract is machine-readable in
[`THEIA_POLICY_SERVER_VERSION.json`](THEIA_POLICY_SERVER_VERSION.json).
`single_reference_raw_vs_refined_v2` at 3000 epochs is the only supported
formal method. `run_theia_server.sh`, v1/1100, and the 20k+2k full-dataset
workflow are historical tools and must not be used for rebuttal results.

Do not promote a candidate by PPO reward alone. Report completion and final
object pose per sequence. Actor-pair contact metrics are separate diagnostics;
only require simultaneous dual grasp for tasks whose reference actually
contains a continuous simultaneous dual-contact phase. Wrong-contact counts
are also diagnostic by default, because an unseen task may admit a successful
interaction strategy that differs from the reference.

### Current Regression Baseline

After repairing cohort accounting and episode-level aggregation, a fixed-seed
GPU proxy-contact evaluation on 2026-07-24 produced:

- 128 evaluation episodes
- 98.44% full-sequence completion (126/128)
- 96.88% semantic success under the strengthened final pose criteria (124/128)
- 100% reach, proxy dual contact, and simultaneous stable dual grasp
- 0 wrong-contact steps
- 2.76 cm episode-level mean human pose error
- 1.40 cm episode-level mean object surface error

The separate 128-episode CPU-PhysX actor-pair run produced 127/128
completion, 128/128 intended dual contact, 128/128 simultaneous stable grasp,
124/128 semantic success, and zero wrong-contact steps. Its four failures were
one early object-trajectory failure and three final object-1 rotation failures.
A terminal world-frame object-pose bonus was tested, but is disabled in the
production from-scratch configuration because it has not been shown to improve
unseen-sequence success.

Checkpoint:

`checkpoints/theia_10h_verified/theia_smplx/nn/mimic_semantic_98_44_seed7.pth`

SHA-256:

`c9cb7d7c64e258efdc35396cac3afe572c32b50381e3e13f9f5c9b28d9a370fe`

This is a local historical artifact and is intentionally excluded from the
source commit.

These numbers are a nominal single-sequence baseline, not evidence of
cross-sequence, cross-object, or randomized-physics generalization. A formal
success-rate claim still needs repeated 128-episode runs and perturbation
evaluation.

### Local Contact A/B Result

Starting from the seed-7 verified checkpoint, the soft-contact and legacy
contact configurations were each fine-tuned for 80 epochs with 256
environments. All candidates were then compared with the same CPU-only PhysX
actor-pair evaluator:

| Candidate | Completion | Semantic success | True dual contact |
|---|---:|---:|---:|
| Original baseline | 126/128 | 118/128 | 128/128 |
| Soft final, epoch 80 | 127/128 | 123/128 | 128/128 |
| Soft reward-best, epoch 74 | 125/128 | 123/128 | 128/128 |
| Legacy final, epoch 80 | 120/128 | 117/128 | 128/128 |
| Legacy reward-best, epoch 76 | 127/128 | 125/128 | 128/128 |

The top two candidates were each evaluated three times and reproduced the
same counts in every run. The selected single-sequence warm-start checkpoint
is:

`checkpoints/theia_local_verified/theia_smplx/nn/mimic_semantic_97_66_cpu_exact.pth`

SHA-256:

`b8ab8d47195dabb4996735db27eee12d938acacf2a69ff0424bc60f415ec7cd3`

This 128 MB local A/B artifact is also intentionally excluded from the source
commit.

This A/B used a strong single-sequence checkpoint and changed several contact
variables together. It therefore does not establish that either reward is
better for unseen server sequences. The server default keeps the original
InterMimic multiplicative contact shaping, with the confirmed fixes: correct
hand-object pairing, neutral non-hand labels, no object-reward floor, no GT
contact hard termination, and no GPU wrong-contact penalty. Soft contact
remains experimental rather than the production default.

### Historical Universal-policy Workflow (Not for Rebuttal)

The following older workflow trains one universal policy over a directory.
It is retained only for engineering history and must not be used for the
formal Raw-vs-Refined results. The server agent should follow the list-sharded
v2/3000 handoff above.

1. Convert all sequences into `theia_data/`; keep the `sub<number>_<left>+<right>_<sequence>.pt` naming convention.
2. Ensure every object mesh and URDF exists. Exact density is optional; unknown
   objects use the 1000 kg/m³ default.
3. Set `NUM_ENVS` to at least `max(512, number_of_sequences)`. The 512
   minimum comes from `horizon_length=32` and `minibatch_size=16384`. Each
   environment is bound to one sequence so its object assets and fixed support
   tables cannot be mixed with another motion.
4. Run the CPU preflight before allocating the simulator:

   ```bash
   python isaacgym/scripts/validate_theia_dataset.py \
     --config isaacgym/src/intermimic/data/cfg/theia_full_train.yaml \
     --num-envs "$NUM_ENVS"
   ```

5. For a non-formal diagnostic only, run `run_theia_server.sh`. It performs
   both stages, resumes safely after interruption, and saves
   `data_manifest.json`, logs, checkpoints, and final CSV/JSON evaluation.
6. Evaluate per sequence and promote only when lower-tail per-sequence
   completion and object-pose criteria pass; an aggregate mean must not hide a
   failed sequence.

### Pre-trained Checkpoint
A pre-trained checkpoint is included at `checkpoints/theia_dual/theia_smplx/nn/mimic.pth` (via Git LFS). To test:
```bash
bash isaacgym/scripts/test_theia.sh
```
This checkpoint is for local visualization/regression, not the default
initialization for unseen server sequences. A different local checkpoint can
be supplied explicitly:
```bash
bash isaacgym/scripts/test_theia.sh \
  /path/to/checkpoint.pth
```

### TensorBoard
```bash
tensorboard --logdir checkpoints/ --port 6006
```

Key metrics to monitor:
- `human_sub/rp`, `human_sub/rr` — body position/rotation tracking
- `sub_rewards/wrist_bonus`, `sub_rewards/grasp_bonus` — interaction quality
- `sub_rewards/ro1`, `sub_rewards/ro2` — per-object tracking
- `reset_rates/contact` — GT contact-miss diagnostic
- `reset_rates/contact_termination` — must remain zero in production training

Production training uses the original InterMimic contact shaping with
pair-corrected contacts. A differently timed GT contact can reduce shaping but
cannot end the episode. The distance-based GPU wrong-contact proxy is not
penalized. Strict PhysX actor-pair contacts remain evaluation-only diagnostics.

## File Structure

```
isaacgym/
  scripts/
    train_all.sh          # Legacy single-sequence launcher
    run_theia_policy_reference_list.sh # Formal v2/3000 GPU worker
    check_theia_server_versions.py # Formal three-repository gate
    run_theia_server.sh   # Blocked-by-default legacy universal workflow
    train_theia_full.sh   # Internal training/resume stage
    train_theia_10h.sh    # Legacy-named local validation helper
    eval_theia.sh         # Fixed-seed semantic evaluation
    test_theia.sh         # Visualization
    replay_theia.sh       # Data replay
  src/intermimic/
    env/tasks/
      intermimic.py       # Core environment (dual-object, residual control, rewards)
      humanoid.py          # Base humanoid (aggregate size fix)
    data/
      cfg/
        theia_train.yaml   # Environment config
        theia_full_train.yaml # Hybrid all-data bootstrap
        theia_full_finetune.yaml # Start/full-length fine-tune
        theia_eval.yaml    # Fixed-seed semantic evaluation config
        train/rlg/theia.yaml  # RL hyperparameters
      assets/
        smplx/theia.xml    # SMPLX humanoid skeleton
        objects/            # Object URDFs and meshes
checkpoints/
  theia_dual/              # Pre-trained dual-object checkpoint
```
