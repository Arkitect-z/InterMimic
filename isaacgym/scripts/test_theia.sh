#!/bin/bash
# Visualize a trained dual-object policy with GUI.
# Usage: bash isaacgym/scripts/test_theia.sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_train.yaml"
CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml"

# Find checkpoint: prefer theia_dual, fallback to any theia_*
CKPT="checkpoints/theia_dual/theia_smplx/nn/mimic.pth"
if [ ! -f "$CKPT" ]; then
    CKPT=""
    for dir in checkpoints/theia_*/theia_smplx/nn; do
        [ -f "$dir/mimic.pth" ] && CKPT="$dir/mimic.pth"
    done
fi

if [ -z "$CKPT" ]; then
    echo "No checkpoint found"
    exit 1
fi

echo "============================================================"
echo "  Testing dual-object policy"
echo "  Checkpoint: ${CKPT}"
echo "============================================================"

python -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --test \
    --num_envs 4 \
    --checkpoint "$CKPT"
