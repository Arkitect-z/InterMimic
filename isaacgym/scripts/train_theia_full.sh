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
SEED="${SEED:-42}"
MOTION_FILE="${MOTION_FILE:-}"
MIN_TRAIN_ENVS=512

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

if [ "$NUM_ENVS" -lt "$MIN_TRAIN_ENVS" ]; then
    echo "NUM_ENVS must be >= ${MIN_TRAIN_ENVS}: horizon=32, minibatch=16384"
    exit 1
fi

RUN_ID="theia_full_${STAGE}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_PATH:-checkpoints/${RUN_ID}}"
mkdir -p "$OUTPUT_PATH"

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
    echo "stage=$STAGE"
    echo "checkpoint_mode=$CHECKPOINT_MODE"
    echo "num_envs=$NUM_ENVS"
    echo "max_iterations=$MAX_ITERATIONS"
    echo "seed=$SEED"
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
} > "$OUTPUT_PATH/run_manifest.txt"

ARGS=(
    --task InterMimic
    --cfg_env "$CFG_ENV"
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml
    --headless
    --torch_deterministic
    --seed "$SEED"
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITERATIONS"
    --output_path "$OUTPUT_PATH"
)
if [ "$CHECKPOINT_MODE" = "resume" ]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "$MOTION_FILE" ]; then
    ARGS+=(--motion_file "$MOTION_FILE")
fi

echo "stage=$STAGE mode=$CHECKPOINT_MODE output=$OUTPUT_PATH"
echo "envs=$NUM_ENVS iterations=$MAX_ITERATIONS seed=$SEED"
python -m intermimic.run "${ARGS[@]}" 2>&1 | tee "$OUTPUT_PATH/train.log"
