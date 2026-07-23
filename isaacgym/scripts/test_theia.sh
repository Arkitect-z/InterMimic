#!/bin/bash
# Visualize a trained dual-object policy with GUI.
# Usage: bash isaacgym/scripts/test_theia.sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_eval.yaml"
CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml"

CKPT="${1:-checkpoints/theia_dual/theia_smplx/nn/mimic.pth}"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    echo "Usage: bash isaacgym/scripts/test_theia.sh /path/to/checkpoint.pth"
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
