#!/bin/bash
# Run one formal Theia policy condition and one training seed.
#
# GPU assignment and process scheduling are intentionally external:
#   CUDA_VISIBLE_DEVICES=0 bash isaacgym/scripts/run_theia_policy_seed.sh \
#     raw 0 /data/theia_policy/raw /data/theia_policy/policy_ab_manifest.json \
#     /results/theia_policy_ab
#
# Usage:
#   bash run_theia_policy_seed.sh \
#     <raw|full> <seed> <data-dir> <pair-manifest> [experiment-root]
#
# Output:
#   <experiment-root>/<condition>/seed_<seed>/
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "$0")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

usage() {
    echo "Usage: bash $0 <raw|full> <seed> <data-dir> <pair-manifest> [experiment-root]"
}

CONDITION="${1:-}"
TRAINING_SEED="${2:-}"
DATA_DIR_INPUT="${3:-}"
PAIR_MANIFEST_INPUT="${4:-}"
EXPERIMENT_ROOT_INPUT="${5:-${EXPERIMENT_ROOT:-$REPO_ROOT/checkpoints/theia_policy_ab}}"

if [ "$CONDITION" != "raw" ] && [ "$CONDITION" != "full" ]; then
    usage
    echo "condition must be exactly 'raw' or 'full'"
    exit 1
fi
if ! [[ "$TRAINING_SEED" =~ ^[0-9]+$ ]]; then
    usage
    echo "seed must be a non-negative integer"
    exit 1
fi
if [ ! -d "$DATA_DIR_INPUT" ]; then
    usage
    echo "data directory not found: $DATA_DIR_INPUT"
    exit 1
fi
if [ ! -f "$PAIR_MANIFEST_INPUT" ]; then
    usage
    echo "pair manifest not found: $PAIR_MANIFEST_INPUT"
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-intermimic}"
if ! python -c "from isaacgym import gymapi" >/dev/null 2>&1; then
    if [ "${THEIA_POLICY_CONDA_READY:-0}" = "1" ]; then
        echo "Isaac Gym is unavailable in Conda environment '$CONDA_ENV'."
        exit 1
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is unavailable; activate '$CONDA_ENV' and run again."
        exit 1
    fi
    exec env THEIA_POLICY_CONDA_READY=1 \
        conda run --no-capture-output -n "$CONDA_ENV" \
        bash "$SCRIPT_PATH" "$@"
fi
if [ ! -f "$SCRIPT_DIR/eval_theia_policy.sh" ]; then
    echo "Formal evaluator is missing: $SCRIPT_DIR/eval_theia_policy.sh"
    exit 1
fi

DATA_DIR="$(CDPATH= cd -- "$DATA_DIR_INPUT" && pwd)"
PAIR_MANIFEST="$(
    python - "$PAIR_MANIFEST_INPUT" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)"
mkdir -p "$EXPERIMENT_ROOT_INPUT"
EXPERIMENT_ROOT="$(CDPATH= cd -- "$EXPERIMENT_ROOT_INPUT" && pwd)"

TARGET_ENVS="${TARGET_ENVS:-2048}"
REQUESTED_NUM_ENVS="${NUM_ENVS:-}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-20000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2000}"
K="${K:-10}"
EVAL_SEED="${EVAL_SEED:-$((10000 + TRAINING_SEED))}"
ALLOW_NONFORMAL_PROTOCOL="${ALLOW_NONFORMAL_PROTOCOL:-0}"
TRAIN_ENV_CONFIG="${TRAIN_ENV_CONFIG:-isaacgym/src/intermimic/data/cfg/theia_full_train.yaml}"
PROTOCOL_MODE="${PROTOCOL_MODE:-legacy_universal}"

if ! [[ "$TARGET_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGET_ENVS must be a positive integer, got '$TARGET_ENVS'"
    exit 1
fi
if ! [[ "$BOOTSTRAP_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOOTSTRAP_EPOCHS must be a positive integer"
    exit 1
fi
if ! [[ "$FINETUNE_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "FINETUNE_EPOCHS must be a non-negative integer"
    exit 1
fi
if [ ! -f "$TRAIN_ENV_CONFIG" ]; then
    echo "Training environment config not found: $TRAIN_ENV_CONFIG"
    exit 1
fi
if ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
    echo "K must be a positive integer"
    exit 1
fi
if ! [[ "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "EVAL_SEED must be a non-negative integer"
    exit 1
fi
case "$ALLOW_NONFORMAL_PROTOCOL" in
    0|1) ;;
    *)
        echo "ALLOW_NONFORMAL_PROTOCOL must be 0 or 1"
        exit 1
        ;;
esac
case "$PROTOCOL_MODE" in
    legacy_universal|single_reference_rebuttal) ;;
    *)
        echo "Unknown PROTOCOL_MODE '$PROTOCOL_MODE'"
        exit 1
        ;;
esac
if [ "$ALLOW_NONFORMAL_PROTOCOL" = "0" ]; then
    if [ "$PROTOCOL_MODE" = "single_reference_rebuttal" ]; then
        if [ "$TRAINING_SEED" -ne 0 ] \
            || [ "$BOOTSTRAP_EPOCHS" -ne 1100 ] \
            || [ "$FINETUNE_EPOCHS" -ne 0 ] \
            || [ "$K" -ne 10 ] \
            || [ "$EVAL_SEED" -ne 10000 ] \
            || [ "$TRAIN_ENV_CONFIG" != \
                "isaacgym/src/intermimic/data/cfg/theia_reference_train.yaml" ]; then
            echo "Single-reference rebuttal protocol requires seed=0,"
            echo "epochs=1100+0, K=10, eval seed=10000, and its frozen config."
            exit 1
        fi
    else
        EXPECTED_EVAL_SEED=$((10000 + TRAINING_SEED))
        if [ "$TRAINING_SEED" -gt 3 ] \
            || [ "$BOOTSTRAP_EPOCHS" -ne 20000 ] \
            || [ "$FINETUNE_EPOCHS" -ne 2000 ] \
            || [ "$K" -ne 10 ] \
            || [ "$EVAL_SEED" -ne "$EXPECTED_EVAL_SEED" ]; then
            echo "Legacy universal protocol requires seeds 0..3,"
            echo "epochs 20000+2000, K=10, and eval seed=10000+seed."
            echo "Use ALLOW_NONFORMAL_PROTOCOL=1 only for an isolated code test."
            exit 1
        fi
    fi
fi
REFERENCE_COUNT="$(
    python - "$DATA_DIR" <<'PY'
from pathlib import Path
import sys

print(sum(path.is_file() for path in Path(sys.argv[1]).glob("*.pt")))
PY
)"
if [ "$REFERENCE_COUNT" -lt 1 ]; then
    echo "No .pt references found in $DATA_DIR"
    exit 1
fi
if [ "$ALLOW_NONFORMAL_PROTOCOL" = "0" ] \
    && [ "$PROTOCOL_MODE" = "single_reference_rebuttal" ] \
    && [ "$REFERENCE_COUNT" -ne 1 ]; then
    echo "Single-reference rebuttal run requires exactly one reference."
    exit 1
fi
if [ -n "$REQUESTED_NUM_ENVS" ]; then
    if ! [[ "$REQUESTED_NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_ENVS must be a positive integer, got '$REQUESTED_NUM_ENVS'"
        exit 1
    fi
    NUM_ENVS="$REQUESTED_NUM_ENVS"
else
    REPLICAS=$((TARGET_ENVS / REFERENCE_COUNT))
    if [ "$REPLICAS" -lt 1 ]; then
        REPLICAS=1
    fi
    NUM_ENVS=$((REFERENCE_COUNT * REPLICAS))
fi
if [ $((NUM_ENVS % REFERENCE_COUNT)) -ne 0 ]; then
    echo "NUM_ENVS=$NUM_ENVS must be divisible by N=$REFERENCE_COUNT"
    echo "Each reference must receive exactly the same number of environments."
    exit 1
fi
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((NUM_ENVS * 8))}"
if ! [[ "$MINIBATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "MINIBATCH_SIZE must be a positive integer, got '$MINIBATCH_SIZE'"
    exit 1
fi
if [ $(((NUM_ENVS * 32) % MINIBATCH_SIZE)) -ne 0 ]; then
    echo "NUM_ENVS*32 must be divisible by MINIBATCH_SIZE"
    exit 1
fi
if [ $((MINIBATCH_SIZE % 4)) -ne 0 ]; then
    echo "MINIBATCH_SIZE must be divisible by seq_len=4"
    exit 1
fi

RUN_ROOT="$EXPERIMENT_ROOT/$CONDITION/seed_$TRAINING_SEED"
BOOTSTRAP_DIR="$RUN_ROOT/bootstrap"
FINETUNE_DIR="$RUN_ROOT/finetune"
EVALUATION_DIR="$RUN_ROOT/evaluation/final"
BOOTSTRAP_CKPT="$BOOTSTRAP_DIR/theia_smplx/nn/mimic.pth"
FINETUNE_CKPT="$FINETUNE_DIR/theia_smplx/nn/mimic.pth"
mkdir -p "$RUN_ROOT"

exec 9>"$RUN_ROOT/.run.lock"
if ! flock -n 9; then
    echo "Another run is active in $RUN_ROOT"
    exit 1
fi
exec > >(tee -a "$RUN_ROOT/policy_seed.log") 2>&1

DATA_MANIFEST_TMP="$(mktemp "$RUN_ROOT/.data_manifest.XXXXXX")"
RUN_SPEC_TMP="$(mktemp "$RUN_ROOT/.run_spec.XXXXXX")"
cleanup() {
    rm -f "$DATA_MANIFEST_TMP" "$RUN_SPEC_TMP"
}
trap cleanup EXIT

CONDITION_DATA_SHA256="$(
    python - "$DATA_DIR" "$PAIR_MANIFEST" "$CONDITION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


data_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
condition = sys.argv[3]
with manifest_path.open() as source:
    manifest = json.load(source)
references = manifest.get("references")
if not isinstance(references, list):
    raise SystemExit("pair manifest must contain a references list")

filename_key = f"{condition}_filename"
hash_key = f"{condition}_sha256"
expected = {}
for entry in references:
    if entry.get("included", True) is not True:
        continue
    filename = entry.get(filename_key)
    expected_hash = entry.get(hash_key)
    if not filename or not expected_hash:
        raise SystemExit(
            f"pair manifest reference lacks {filename_key}/{hash_key}"
        )
    if filename in expected:
        raise SystemExit(f"duplicate {condition} filename in manifest: {filename}")
    expected[filename] = expected_hash

actual_names = sorted(
    path.name for path in data_dir.glob("*.pt") if path.is_file()
)
if actual_names != sorted(expected):
    missing = sorted(set(expected) - set(actual_names))
    extra = sorted(set(actual_names) - set(expected))
    raise SystemExit(
        f"{condition} data differs from frozen manifest: "
        f"missing={missing[:10]}, extra={extra[:10]}"
    )

records = []
for filename in actual_names:
    actual_hash = sha256(data_dir / filename)
    if actual_hash != expected[filename]:
        raise SystemExit(
            f"{condition} SHA-256 mismatch for {filename}: "
            f"manifest={expected[filename]} actual={actual_hash}"
        )
    records.append(f"{actual_hash}  {filename}\n")
aggregate = hashlib.sha256("".join(records).encode()).hexdigest()
print(aggregate)
PY
)"
echo "Validated frozen $CONDITION data: sha256=$CONDITION_DATA_SHA256"

python "$SCRIPT_DIR/validate_theia_dataset.py" \
    --config "$TRAIN_ENV_CONFIG" \
    --motion-file "$DATA_DIR" \
    --num-envs "$NUM_ENVS" \
    --manifest "$DATA_MANIFEST_TMP"

{
    echo "condition=$CONDITION"
    echo "training_seed=$TRAINING_SEED"
    echo "data_dir=$DATA_DIR"
    echo "condition_data_sha256=$CONDITION_DATA_SHA256"
    echo "pair_manifest=$PAIR_MANIFEST"
    echo "reference_count=$REFERENCE_COUNT"
    echo "target_envs=$TARGET_ENVS"
    echo "num_envs=$NUM_ENVS"
    echo "minibatch_size=$MINIBATCH_SIZE"
    echo "bootstrap_epochs=$BOOTSTRAP_EPOCHS"
    echo "finetune_epochs=$FINETUNE_EPOCHS"
    echo "train_env_config=$TRAIN_ENV_CONFIG"
    echo "eval_k=$K"
    echo "eval_seed=$EVAL_SEED"
    echo "evaluation_fps=30"
    echo "protocol_mode=$PROTOCOL_MODE"
    echo "allow_nonformal_protocol=$ALLOW_NONFORMAL_PROTOCOL"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_diff_sha256=$(git diff HEAD --binary | sha256sum | awk '{print $1}')"
    sha256sum \
        "$PAIR_MANIFEST" \
        "$SCRIPT_PATH" \
        isaacgym/scripts/eval_theia_policy.sh \
        isaacgym/scripts/eval_theia.sh \
        isaacgym/scripts/summarize_theia_eval.py \
        isaacgym/scripts/train_theia_full.sh \
        isaacgym/scripts/validate_theia_dataset.py \
        isaacgym/src/intermimic/data/cfg/theia_policy_eval.yaml \
        "$TRAIN_ENV_CONFIG" \
        isaacgym/src/intermimic/data/cfg/theia_full_finetune.yaml \
        isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml \
        isaacgym/src/intermimic/env/tasks/intermimic.py \
        isaacgym/src/intermimic/learning/intermimic_players.py \
        isaacgym/src/intermimic/learning/intermimic_agent.py
} > "$RUN_SPEC_TMP"

if [ -f "$RUN_ROOT/data_manifest.json" ]; then
    if ! cmp -s "$DATA_MANIFEST_TMP" "$RUN_ROOT/data_manifest.json"; then
        echo "Dataset manifest changed for existing run: $RUN_ROOT"
        echo "Use a new experiment root instead of resuming incompatible data."
        exit 1
    fi
else
    mv "$DATA_MANIFEST_TMP" "$RUN_ROOT/data_manifest.json"
fi

if [ -f "$RUN_ROOT/run_spec.txt" ]; then
    if ! cmp -s "$RUN_SPEC_TMP" "$RUN_ROOT/run_spec.txt"; then
        echo "Run specification changed for existing run: $RUN_ROOT"
        echo "Use a new experiment root instead of resuming an incompatible run."
        exit 1
    fi
else
    mv "$RUN_SPEC_TMP" "$RUN_ROOT/run_spec.txt"
fi

checkpoint_epoch() {
    python - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False
)
required = {
    "model",
    "optimizer",
    "epoch",
    "frame",
    "last_mean_rewards",
    "env_state",
    "running_mean_std",
    "amp_input_mean_std",
}
missing = sorted(required - set(checkpoint))
if missing:
    raise SystemExit(
        f"checkpoint is missing full-state fields {missing}: {sys.argv[1]}"
    )
print(int(checkpoint["epoch"]))
PY
}

SELECTED_CHECKPOINT=""
SELECTED_EPOCH=0

consider_checkpoint() {
    local candidate="$1"
    local target_epoch="$2"
    local candidate_epoch
    if [ ! -f "$candidate" ]; then
        return
    fi
    if ! candidate_epoch="$(checkpoint_epoch "$candidate" 2>/dev/null)"; then
        echo "[WARN] Ignoring unreadable/incomplete checkpoint: $candidate"
        return
    fi
    if [ "$candidate_epoch" -gt "$target_epoch" ]; then
        echo "[WARN] Ignoring checkpoint newer than target: $candidate"
        return
    fi
    if [ "$candidate_epoch" -gt "$SELECTED_EPOCH" ]; then
        SELECTED_CHECKPOINT="$candidate"
        SELECTED_EPOCH="$candidate_epoch"
    fi
}

select_resume_checkpoint() {
    local output_dir="$1"
    local target_epoch="$2"
    local fallback_checkpoint="${3:-}"
    local candidate
    local nn_dir="$output_dir/theia_smplx/nn"

    SELECTED_CHECKPOINT=""
    SELECTED_EPOCH=0
    consider_checkpoint "$nn_dir/mimic.pth" "$target_epoch"
    for candidate in "$nn_dir"/mimic_epoch_*.pth; do
        [ -e "$candidate" ] || continue
        consider_checkpoint "$candidate" "$target_epoch"
    done
    if [ -n "$fallback_checkpoint" ]; then
        consider_checkpoint "$fallback_checkpoint" "$target_epoch"
        local fallback_dir
        fallback_dir="$(dirname -- "$fallback_checkpoint")"
        for candidate in "$fallback_dir"/mimic_epoch_*.pth; do
            [ -e "$candidate" ] || continue
            consider_checkpoint "$candidate" "$target_epoch"
        done
    fi
}

run_to_epoch() {
    local stage="$1"
    local output_dir="$2"
    local target_epoch="$3"
    local fallback_checkpoint="${4:-}"
    local stage_checkpoint="$output_dir/theia_smplx/nn/mimic.pth"
    local restore_checkpoint=""
    local current_epoch=0
    local mode="fresh"
    local stage_checkpoint_epoch

    if [ -f "$stage_checkpoint" ] \
        && stage_checkpoint_epoch="$(checkpoint_epoch "$stage_checkpoint" 2>/dev/null)" \
        && [ "$stage_checkpoint_epoch" -gt "$target_epoch" ]; then
        echo "$stage checkpoint epoch $stage_checkpoint_epoch exceeds target $target_epoch"
        exit 1
    fi
    select_resume_checkpoint "$output_dir" "$target_epoch" "$fallback_checkpoint"
    restore_checkpoint="$SELECTED_CHECKPOINT"
    current_epoch="$SELECTED_EPOCH"
    if [ -n "$restore_checkpoint" ]; then
        mode="resume"
        echo "[RESUME] selected epoch=$current_epoch checkpoint=$restore_checkpoint"
    fi

    local remaining=$((target_epoch - current_epoch))
    if [ "$remaining" -eq 0 ]; then
        if [ -n "$restore_checkpoint" ] \
            && [ "$restore_checkpoint" != "$stage_checkpoint" ]; then
            echo "[RECOVER] Restoring final checkpoint from milestone"
            mkdir -p "$(dirname -- "$stage_checkpoint")"
            cp -- "$restore_checkpoint" "$stage_checkpoint"
        fi
        echo "[SKIP] $stage already reached epoch $target_epoch"
        return
    fi

    echo "[RUN] condition=$CONDITION seed=$TRAINING_SEED stage=$stage"
    echo "[RUN] current_epoch=$current_epoch target_epoch=$target_epoch"
    if [ "$mode" = "resume" ]; then
        CHECKPOINT_MODE=resume \
        CHECKPOINT="$restore_checkpoint" \
        MOTION_FILE="$DATA_DIR" \
        NUM_ENVS="$NUM_ENVS" \
        MINIBATCH_SIZE="$MINIBATCH_SIZE" \
        MAX_ITERATIONS="$remaining" \
        OUTPUT_PATH="$output_dir" \
        SEED="$TRAINING_SEED" \
        CFG_ENV_OVERRIDE="$TRAIN_ENV_CONFIG" \
            bash "$SCRIPT_DIR/train_theia_full.sh" "$stage"
    else
        CHECKPOINT_MODE=fresh \
        MOTION_FILE="$DATA_DIR" \
        NUM_ENVS="$NUM_ENVS" \
        MINIBATCH_SIZE="$MINIBATCH_SIZE" \
        MAX_ITERATIONS="$remaining" \
        OUTPUT_PATH="$output_dir" \
        SEED="$TRAINING_SEED" \
        CFG_ENV_OVERRIDE="$TRAIN_ENV_CONFIG" \
            bash "$SCRIPT_DIR/train_theia_full.sh" "$stage"
    fi

    local completed_epoch
    if ! completed_epoch="$(checkpoint_epoch "$stage_checkpoint" 2>/dev/null)"; then
        echo "[WARN] Final $stage checkpoint is unreadable; checking milestones"
        select_resume_checkpoint "$output_dir" "$target_epoch"
        if [ "$SELECTED_EPOCH" -ne "$target_epoch" ]; then
            echo "$stage did not produce a valid epoch-$target_epoch checkpoint"
            exit 1
        fi
        cp -- "$SELECTED_CHECKPOINT" "$stage_checkpoint"
        completed_epoch="$(checkpoint_epoch "$stage_checkpoint")"
        echo "[RECOVER] Replaced final checkpoint from $SELECTED_CHECKPOINT"
    fi
    if [ "$completed_epoch" -ne "$target_epoch" ]; then
        echo "$stage ended at epoch $completed_epoch, expected $target_epoch"
        exit 1
    fi
}

echo "================================================================"
echo "Theia formal policy run"
echo "condition=$CONDITION training_seed=$TRAINING_SEED"
echo "data=$DATA_DIR references=$REFERENCE_COUNT"
echo "num_envs=$NUM_ENVS minibatch_size=$MINIBATCH_SIZE"
echo "bootstrap=$BOOTSTRAP_EPOCHS finetune=$FINETUNE_EPOCHS"
echo "output=$RUN_ROOT"
echo "================================================================"

run_to_epoch bootstrap "$BOOTSTRAP_DIR" "$BOOTSTRAP_EPOCHS"
if [ ! -f "$BOOTSTRAP_CKPT" ]; then
    echo "Bootstrap checkpoint not found: $BOOTSTRAP_CKPT"
    exit 1
fi

FINETUNE_TARGET=$((BOOTSTRAP_EPOCHS + FINETUNE_EPOCHS))
if [ "$FINETUNE_EPOCHS" -gt 0 ]; then
    run_to_epoch finetune "$FINETUNE_DIR" "$FINETUNE_TARGET" "$BOOTSTRAP_CKPT"
    if [ ! -f "$FINETUNE_CKPT" ]; then
        echo "Fine-tune checkpoint not found: $FINETUNE_CKPT"
        exit 1
    fi
    FINAL_CKPT="$FINETUNE_CKPT"
else
    echo "[SKIP] Single-stage protocol: no full-sequence fine-tune"
    FINAL_CKPT="$BOOTSTRAP_CKPT"
fi

env -u NUM_ENVS \
CONDITION="$CONDITION" \
TRAINING_SEED="$TRAINING_SEED" \
K="$K" \
EVAL_SEED="$EVAL_SEED" \
FPS=30 \
    bash "$SCRIPT_DIR/eval_theia_policy.sh" \
    "$FINAL_CKPT" \
    "$DATA_DIR" \
    "$PAIR_MANIFEST" \
    "$EVALUATION_DIR"

echo "Completed condition=$CONDITION seed=$TRAINING_SEED"
echo "checkpoint=$FINAL_CKPT"
echo "evaluation=$EVALUATION_DIR"
