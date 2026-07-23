#!/bin/bash
# Fixed-seed, full-sequence Theia evaluation.
# Usage: bash isaacgym/scripts/eval_theia.sh /path/to/checkpoint.pth
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

CKPT="${1:-}"
NUM_ENVS="${NUM_ENVS:-128}"
STRICT_CONTACTS="${STRICT_CONTACTS:-0}"
MOTION_FILE="${MOTION_FILE:-}"
EVAL_MODE="proxy_gpu"
EXTRA_ARGS=()
if [ "$STRICT_CONTACTS" = "1" ]; then
    EVAL_MODE="actor_pair_cpu_physx"
    EXTRA_ARGS+=(--sim_device cpu --pipeline cpu --exact_contact_evaluation)
fi
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-evaluation/theia_${EVAL_MODE}_$(date +%Y%m%d_%H%M%S)}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "Usage: bash isaacgym/scripts/eval_theia.sh /path/to/checkpoint.pth"
    exit 1
fi

mkdir -p "$EVAL_OUTPUT_DIR"
export THEIA_EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR"
{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "checkpoint=$CKPT"
    echo "num_envs=$NUM_ENVS"
    echo "evaluation_mode=$EVAL_MODE"
    echo "motion_file=${MOTION_FILE:-theia_data}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_diff_sha256=$(git diff --binary | sha256sum | awk '{print $1}')"
    echo "python=$(python --version 2>&1)"
    sha256sum "$CKPT"
    sha256sum \
        isaacgym/src/intermimic/env/tasks/intermimic.py \
        isaacgym/src/intermimic/learning/intermimic_players.py \
        isaacgym/src/intermimic/data/cfg/theia_eval.yaml \
        isaacgym/scripts/eval_theia.sh
    git status --short
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader 2>/dev/null || true
} > "$EVAL_OUTPUT_DIR/manifest.txt"

if [ -n "$MOTION_FILE" ]; then
    EXTRA_ARGS+=(--motion_file "$MOTION_FILE")
fi

python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/theia_eval.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml \
    --test \
    --headless \
    --torch_deterministic \
    --seed 42 \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CKPT" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$EVAL_OUTPUT_DIR/eval.log"
