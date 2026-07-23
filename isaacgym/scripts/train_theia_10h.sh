#!/bin/bash
# Local single-sequence repair/validation helper.
# The filename is retained for compatibility, but this script has never
# enforced a 10-hour wall-clock budget and must not be used for server runs.
# Pilot:
#   MAX_ITERATIONS=80 bash isaacgym/scripts/train_theia_10h.sh pilot
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

STAGE="${1:-pilot}"
MAX_ITERATIONS="${MAX_ITERATIONS:-80}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-theia_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-checkpoints/${RUN_ID}}"
MIN_TRAIN_ENVS=512

if [ "$NUM_ENVS" -lt "$MIN_TRAIN_ENVS" ]; then
    echo "NUM_ENVS must be >= ${MIN_TRAIN_ENVS}: horizon_length=32 and minibatch_size=16384"
    exit 1
fi

case "$STAGE" in
    pilot)
        CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_train.yaml"
        ;;
    full)
        echo "The server full stage moved to train_theia_full.sh."
        echo "Use: bash isaacgym/scripts/train_theia_full.sh bootstrap"
        exit 1
        ;;
    *)
        echo "Unknown stage '$STAGE' (expected pilot or full)"
        exit 1
        ;;
esac

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
echo "stage=$STAGE output=$OUTPUT_PATH envs=$NUM_ENVS iterations=$MAX_ITERATIONS seed=$SEED"
python -m intermimic.run "${ARGS[@]}"
