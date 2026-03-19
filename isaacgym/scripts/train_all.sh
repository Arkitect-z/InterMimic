#!/bin/bash
# Train a single dual-object policy.
# Usage: bash isaacgym/scripts/train_all.sh
# Resume: re-run the same command; it auto-detects existing checkpoints.
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_train.yaml"
CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml"
OUTPUT_DIR="checkpoints/theia_dual"
MAX_EPOCHS=20000

echo "============================================================"
echo "  Training dual-object policy"
echo "  Output: ${OUTPUT_DIR}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "============================================================"

# Archive incomplete previous run (no checkpoint saved yet)
CKPT="${OUTPUT_DIR}/theia_smplx/nn/mimic.pth"
if [ -d "$OUTPUT_DIR" ] && [ ! -f "$CKPT" ]; then
    ARCHIVE="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
    echo "  Archiving incomplete run -> ${ARCHIVE}"
    mv "$OUTPUT_DIR" "$ARCHIVE"
fi

# Backup & patch configs
cp "$CFG_TRAIN" "${CFG_TRAIN}.bak"
sed -i "s/max_epochs:.*/max_epochs: ${MAX_EPOCHS}/" "$CFG_TRAIN"

# Auto-resume
if [ -f "$CKPT" ]; then
    echo "  Found checkpoint: ${CKPT} — resuming"
    sed -i "s|resume_from:.*|resume_from: ${CKPT}|" "$CFG_TRAIN"
else
    echo "  No checkpoint found — starting fresh"
    sed -i "s|resume_from:.*|resume_from: None|" "$CFG_TRAIN"
fi

python -m intermimic.run \
    --task InterMimic \
    --cfg_env "$CFG_ENV" \
    --cfg_train "$CFG_TRAIN" \
    --headless \
    --output "$OUTPUT_DIR"

# Restore config
mv "${CFG_TRAIN}.bak" "$CFG_TRAIN"

echo "============================================================"
echo "  Training complete"
echo "============================================================"
