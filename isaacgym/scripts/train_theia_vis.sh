#!/bin/sh
# Train with GUI visualization (small num_envs for real-time preview)
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

conda run -n intermimic python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/theia_train.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml \
    --num_envs 4 \
    --minibatch_size 128 \
    --output checkpoints/theia
