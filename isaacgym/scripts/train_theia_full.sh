#!/bin/bash
# Server full-dataset training. There is intentionally no 10-hour wall-clock
# limit: the earlier 10-hour requirement applied only to local repair/testing.
#
# Fresh Hybrid/RSI bootstrap:
#   NUM_ENVS=2048 MAX_ITERATIONS=20000 \
#     bash isaacgym/scripts/train_theia_full.sh bootstrap
#
# True resume after interruption (optimizer/epoch/best state are restored):
#   CHECKPOINT_MODE=resume CHECKPOINT=/path/to/mimic.pth \
#   OUTPUT_PATH=/same/output/path \
#     bash isaacgym/scripts/train_theia_full.sh bootstrap
#
# Full frame-0-to-end fine-tune after bootstrap:
#   CHECKPOINT_MODE=resume CHECKPOINT=/path/to/bootstrap/mimic.pth \
#   OUTPUT_PATH=checkpoints/theia_full_finetune \
#     bash isaacgym/scripts/train_theia_full.sh finetune
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

if ! python - <<'PY'
from isaacgym import gymapi
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "CUDA is unavailable. Training requires a GPU-visible process with "
        "/dev/nvidiactl and /dev/nvidia0 mounted."
    )
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(
    "[CUDA] "
    f"torch={torch.__version__} runtime={torch.version.cuda} "
    f"device={torch.cuda.get_device_name(0)!r} "
    f"free={free_bytes / 2**30:.1f}GiB total={total_bytes / 2**30:.1f}GiB"
)
PY
then
    echo "Isaac Gym/CUDA preflight failed."
    echo "Activate the intermimic conda environment and expose the NVIDIA"
    echo "device nodes to this process before starting training."
    exit 1
fi

STAGE="${1:-bootstrap}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-fresh}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
NUM_ENVS="${NUM_ENVS:-2048}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-}"
SEED="${SEED:-42}"
MOTION_FILE="${MOTION_FILE:-}"
CFG_ENV_OVERRIDE="${CFG_ENV_OVERRIDE:-}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-1}"
HORIZON_LENGTH=32
SEQ_LEN=4

case "$STAGE" in
    bootstrap)
        CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_full_train.yaml"
        ;;
    finetune)
        CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_full_finetune.yaml"
        ;;
    *)
        echo "Unknown stage '$STAGE' (expected bootstrap or finetune)"
        exit 1
        ;;
esac
if [ -n "$CFG_ENV_OVERRIDE" ]; then
    CFG_ENV="$CFG_ENV_OVERRIDE"
fi
if [ ! -f "$CFG_ENV" ]; then
    echo "Environment config not found: $CFG_ENV"
    exit 1
fi
case "$TORCH_DETERMINISTIC" in
    0|1) ;;
    *)
        echo "TORCH_DETERMINISTIC must be 0 or 1"
        exit 1
        ;;
esac

case "$CHECKPOINT_MODE" in
    fresh)
        if [ "$STAGE" = "finetune" ]; then
            echo "finetune requires CHECKPOINT_MODE=resume and a bootstrap checkpoint"
            exit 1
        fi
        ;;
    resume)
        if [ -z "${CHECKPOINT:-}" ] || [ ! -f "$CHECKPOINT" ]; then
            echo "resume requires CHECKPOINT=/path/to/mimic.pth"
            exit 1
        fi
        if [ -z "${OUTPUT_PATH:-}" ]; then
            echo "resume requires an explicit OUTPUT_PATH"
            exit 1
        fi
        ;;
    *)
        echo "Unknown CHECKPOINT_MODE '$CHECKPOINT_MODE' (expected fresh or resume)"
        exit 1
        ;;
esac

if ! [[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be a positive integer, got '$NUM_ENVS'"
    exit 1
fi
if [ -z "$MINIBATCH_SIZE" ]; then
    # Keep four PPO minibatches for any exactly balanced environment count.
    MINIBATCH_SIZE=$((NUM_ENVS * HORIZON_LENGTH / 4))
fi
if ! [[ "$MINIBATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "MINIBATCH_SIZE must be a positive integer, got '$MINIBATCH_SIZE'"
    exit 1
fi
TOTAL_BATCH_SIZE=$((NUM_ENVS * HORIZON_LENGTH))
if [ $((TOTAL_BATCH_SIZE % MINIBATCH_SIZE)) -ne 0 ]; then
    echo "NUM_ENVS * horizon_length must be divisible by MINIBATCH_SIZE:"
    echo "$NUM_ENVS * $HORIZON_LENGTH = $TOTAL_BATCH_SIZE,"
    echo "MINIBATCH_SIZE=$MINIBATCH_SIZE."
    exit 1
fi
if [ $((MINIBATCH_SIZE % SEQ_LEN)) -ne 0 ]; then
    echo "MINIBATCH_SIZE must be divisible by seq_len=$SEQ_LEN"
    exit 1
fi

MOTION_DIR="${MOTION_FILE:-theia_data}"
if [ ! -d "$MOTION_DIR" ]; then
    echo "Motion directory not found: $MOTION_DIR"
    exit 1
fi
SEQUENCE_COUNT="$(
    python - "$MOTION_DIR" <<'PY'
from pathlib import Path
import sys

print(sum(path.is_file() for path in Path(sys.argv[1]).glob("*.pt")))
PY
)"
if [ "$SEQUENCE_COUNT" -lt 1 ]; then
    echo "No .pt motion sequences found in $MOTION_DIR"
    exit 1
fi
if [ "$NUM_ENVS" -lt "$SEQUENCE_COUNT" ]; then
    echo "NUM_ENVS=$NUM_ENVS cannot cover $SEQUENCE_COUNT motion sequences"
    exit 1
fi

RUN_ID="theia_full_${STAGE}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_PATH:-checkpoints/${RUN_ID}}"
mkdir -p "$OUTPUT_PATH"
INVOCATION_ID="$(date +%Y%m%d_%H%M%S)_$$"
INVOCATION_DIR="$OUTPUT_PATH/invocations"
INVOCATION_MANIFEST="$INVOCATION_DIR/run_manifest_${INVOCATION_ID}.txt"
mkdir -p "$INVOCATION_DIR"

VALIDATOR_ARGS=(
    --config "$CFG_ENV"
    --num-envs "$NUM_ENVS"
    --manifest "$OUTPUT_PATH/data_manifest.json"
)
if [ -n "$MOTION_FILE" ]; then
    VALIDATOR_ARGS+=(--motion-file "$MOTION_FILE")
fi
python isaacgym/scripts/validate_theia_dataset.py "${VALIDATOR_ARGS[@]}"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "invocation_id=$INVOCATION_ID"
    echo "stage=$STAGE"
    echo "checkpoint_mode=$CHECKPOINT_MODE"
    echo "num_envs=$NUM_ENVS"
    echo "minibatch_size=$MINIBATCH_SIZE"
    echo "max_iterations=$MAX_ITERATIONS"
    echo "seed=$SEED"
    echo "torch_deterministic=$TORCH_DETERMINISTIC"
    echo "motion_file=${MOTION_FILE:-theia_data}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_diff_sha256=$(git diff --binary | sha256sum | awk '{print $1}')"
    sha256sum \
        "$CFG_ENV" \
        isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml \
        isaacgym/src/intermimic/env/tasks/intermimic.py
    if [ "$CHECKPOINT_MODE" = "resume" ]; then
        sha256sum "$CHECKPOINT"
    fi
} | tee "$INVOCATION_MANIFEST" >> "$OUTPUT_PATH/run_manifest.txt"

ARGS=(
    --task InterMimic
    --cfg_env "$CFG_ENV"
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml
    --headless
    --seed "$SEED"
    --num_envs "$NUM_ENVS"
    --minibatch_size "$MINIBATCH_SIZE"
    --max_iterations "$MAX_ITERATIONS"
    --output_path "$OUTPUT_PATH"
)
if [ "$TORCH_DETERMINISTIC" = "1" ]; then
    ARGS+=(--torch_deterministic)
fi
if [ "$CHECKPOINT_MODE" = "resume" ]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "$MOTION_FILE" ]; then
    ARGS+=(--motion_file "$MOTION_FILE")
fi

{
    echo
    echo "================================================================"
    echo "invocation=$INVOCATION_ID"
    echo "stage=$STAGE mode=$CHECKPOINT_MODE output=$OUTPUT_PATH"
    echo "envs=$NUM_ENVS minibatch=$MINIBATCH_SIZE iterations=$MAX_ITERATIONS seed=$SEED"
    echo "================================================================"
} | tee -a "$OUTPUT_PATH/train.log"
python -m intermimic.run "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT_PATH/train.log"
