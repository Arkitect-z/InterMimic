#!/bin/bash
# One-command server training for a completely unseen Theia dataset.
#
# Default:
#   bash isaacgym/scripts/run_theia_server.sh
#
# External data directory:
#   THEIA_DATA_DIR=/data/theia_pt bash isaacgym/scripts/run_theia_server.sh
#
# Safe to run again after interruption: the latest full checkpoint, optimizer,
# epoch, frame count, and reward-best state are restored automatically.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "$0")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

CONDA_ENV="${CONDA_ENV:-intermimic}"
if ! python -c "from isaacgym import gymapi" >/dev/null 2>&1; then
    if [ "${THEIA_SERVER_CONDA_READY:-0}" = "1" ]; then
        echo "Isaac Gym is unavailable in Conda environment '$CONDA_ENV'."
        exit 1
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is unavailable; activate '$CONDA_ENV' and run again."
        exit 1
    fi
    exec env THEIA_SERVER_CONDA_READY=1 \
        conda run --no-capture-output -n "$CONDA_ENV" \
        bash "$SCRIPT_PATH" "$@"
fi

DATA_DIR_INPUT="${THEIA_DATA_DIR:-$REPO_ROOT/theia_data}"
if [ ! -d "$DATA_DIR_INPUT" ]; then
    echo "Theia data directory not found: $DATA_DIR_INPUT"
    echo "Set THEIA_DATA_DIR=/absolute/path/to/converted_pt_files"
    exit 1
fi
DATA_DIR="$(CDPATH= cd -- "$DATA_DIR_INPUT" && pwd)"

SEQUENCE_COUNT="$(
    python - "$DATA_DIR" <<'PY'
from pathlib import Path
import sys

print(sum(path.is_file() for path in Path(sys.argv[1]).glob("*.pt")))
PY
)"
if [ "$SEQUENCE_COUNT" -lt 1 ]; then
    echo "No .pt sequences found in $DATA_DIR"
    exit 1
fi

TARGET_ENVS="${TARGET_ENVS:-2048}"
MIN_TRAIN_ENVS=512
if [ -n "${NUM_ENVS:-}" ]; then
    TRAIN_ENVS="$NUM_ENVS"
else
    ENV_TARGET="$TARGET_ENVS"
    if [ "$ENV_TARGET" -lt "$MIN_TRAIN_ENVS" ]; then
        ENV_TARGET="$MIN_TRAIN_ENVS"
    fi
    if [ "$SEQUENCE_COUNT" -ge "$ENV_TARGET" ]; then
        TRAIN_ENVS="$SEQUENCE_COUNT"
    else
        REPLICAS=$((ENV_TARGET / SEQUENCE_COUNT))
        if [ "$REPLICAS" -lt 1 ]; then
            REPLICAS=1
        fi
        TRAIN_ENVS=$((REPLICAS * SEQUENCE_COUNT))
        if [ "$TRAIN_ENVS" -lt "$MIN_TRAIN_ENVS" ]; then
            REPLICAS=$(((MIN_TRAIN_ENVS + SEQUENCE_COUNT - 1) / SEQUENCE_COUNT))
            TRAIN_ENVS=$((REPLICAS * SEQUENCE_COUNT))
        fi
    fi
fi
if [ "$TRAIN_ENVS" -lt "$MIN_TRAIN_ENVS" ]; then
    echo "NUM_ENVS=$TRAIN_ENVS is below the PPO minimum $MIN_TRAIN_ENVS"
    exit 1
fi
if [ "$TRAIN_ENVS" -lt "$SEQUENCE_COUNT" ]; then
    echo "NUM_ENVS=$TRAIN_ENVS cannot cover $SEQUENCE_COUNT sequences"
    exit 1
fi

BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-20000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2000}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/checkpoints/theia_server_full}"
BOOTSTRAP_DIR="$OUTPUT_ROOT/bootstrap"
FINETUNE_DIR="$OUTPUT_ROOT/finetune"
EVALUATION_ROOT="$OUTPUT_ROOT/evaluation"
BOOTSTRAP_CKPT="$BOOTSTRAP_DIR/theia_smplx/nn/mimic.pth"
FINETUNE_CKPT="$FINETUNE_DIR/theia_smplx/nn/mimic.pth"
mkdir -p "$OUTPUT_ROOT"

exec 9>"$OUTPUT_ROOT/.run.lock"
if ! flock -n 9; then
    echo "Another one-click Theia server run is active in $OUTPUT_ROOT"
    exit 1
fi
exec > >(tee -a "$OUTPUT_ROOT/server_run.log") 2>&1

checkpoint_epoch() {
    python - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False
)
print(int(checkpoint.get("epoch", 0)))
PY
}

run_to_epoch() {
    local stage="$1"
    local output_dir="$2"
    local target_epoch="$3"
    local fallback_checkpoint="${4:-}"
    local stage_checkpoint="$output_dir/theia_smplx/nn/mimic.pth"
    local restore_checkpoint=""
    local current_epoch=0
    local candidate_epoch=0
    local mode="fresh"

    if [ -f "$stage_checkpoint" ]; then
        restore_checkpoint="$stage_checkpoint"
        current_epoch="$(checkpoint_epoch "$restore_checkpoint")"
    fi
    if [ -n "$fallback_checkpoint" ] && [ -f "$fallback_checkpoint" ]; then
        candidate_epoch="$(checkpoint_epoch "$fallback_checkpoint")"
        if [ "$candidate_epoch" -gt "$current_epoch" ]; then
            restore_checkpoint="$fallback_checkpoint"
            current_epoch="$candidate_epoch"
        fi
    fi
    if [ -n "$restore_checkpoint" ]; then
        mode="resume"
    fi

    local remaining=$((target_epoch - current_epoch))
    if [ "$remaining" -le 0 ]; then
        echo "[SKIP] $stage already reached epoch $current_epoch/$target_epoch"
        return
    fi

    echo "[RUN] stage=$stage current_epoch=$current_epoch target_epoch=$target_epoch"
    if [ "$mode" = "resume" ]; then
        CHECKPOINT_MODE=resume \
        CHECKPOINT="$restore_checkpoint" \
        MOTION_FILE="$DATA_DIR" \
        NUM_ENVS="$TRAIN_ENVS" \
        MAX_ITERATIONS="$remaining" \
        OUTPUT_PATH="$output_dir" \
        SEED="$SEED" \
            bash "$SCRIPT_DIR/train_theia_full.sh" "$stage"
    else
        CHECKPOINT_MODE=fresh \
        MOTION_FILE="$DATA_DIR" \
        NUM_ENVS="$TRAIN_ENVS" \
        MAX_ITERATIONS="$remaining" \
        OUTPUT_PATH="$output_dir" \
        SEED="$SEED" \
            bash "$SCRIPT_DIR/train_theia_full.sh" "$stage"
    fi
}

echo "================================================================"
echo "Theia server one-click training"
echo "data=$DATA_DIR"
echo "sequences=$SEQUENCE_COUNT train_envs=$TRAIN_ENVS"
echo "bootstrap_epochs=$BOOTSTRAP_EPOCHS finetune_epochs=$FINETUNE_EPOCHS"
echo "output=$OUTPUT_ROOT seed=$SEED"
echo "================================================================"

python "$SCRIPT_DIR/validate_theia_dataset.py" \
    --config isaacgym/src/intermimic/data/cfg/theia_full_train.yaml \
    --motion-file "$DATA_DIR" \
    --num-envs "$TRAIN_ENVS" \
    --manifest "$OUTPUT_ROOT/data_manifest.json"

run_to_epoch bootstrap "$BOOTSTRAP_DIR" "$BOOTSTRAP_EPOCHS"
if [ ! -f "$BOOTSTRAP_CKPT" ]; then
    echo "Bootstrap checkpoint was not produced: $BOOTSTRAP_CKPT"
    exit 1
fi

FINETUNE_TARGET=$((BOOTSTRAP_EPOCHS + FINETUNE_EPOCHS))
run_to_epoch finetune "$FINETUNE_DIR" "$FINETUNE_TARGET" "$BOOTSTRAP_CKPT"
if [ ! -f "$FINETUNE_CKPT" ]; then
    echo "Fine-tune checkpoint was not produced: $FINETUNE_CKPT"
    exit 1
fi

EVAL_REPEATS="${EVAL_REPEATS:-4}"
EVAL_CAP="${EVAL_TARGET_ENVS:-$TARGET_ENVS}"
if [ -n "${EVAL_ENVS:-}" ]; then
    FINAL_EVAL_ENVS="$EVAL_ENVS"
else
    MAX_REPLICAS=$((EVAL_CAP / SEQUENCE_COUNT))
    if [ "$MAX_REPLICAS" -lt 1 ]; then
        MAX_REPLICAS=1
    fi
    if [ "$MAX_REPLICAS" -gt "$EVAL_REPEATS" ]; then
        MAX_REPLICAS="$EVAL_REPEATS"
    fi
    FINAL_EVAL_ENVS=$((MAX_REPLICAS * SEQUENCE_COUNT))
fi
if [ "$FINAL_EVAL_ENVS" -lt "$SEQUENCE_COUNT" ]; then
    echo "EVAL_ENVS=$FINAL_EVAL_ENVS cannot cover $SEQUENCE_COUNT sequences"
    exit 1
fi

FINAL_EPOCH="$(checkpoint_epoch "$FINETUNE_CKPT")"
EVALUATION_DIR="$EVALUATION_ROOT/epoch_${FINAL_EPOCH}_$(date +%Y%m%d_%H%M%S)"
MOTION_FILE="$DATA_DIR" \
NUM_ENVS="$FINAL_EVAL_ENVS" \
EVAL_OUTPUT_DIR="$EVALUATION_DIR" \
    bash "$SCRIPT_DIR/eval_theia.sh" "$FINETUNE_CKPT"

MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.95}"
MIN_SEQUENCE_SUCCESS_RATE="${MIN_SEQUENCE_SUCCESS_RATE:-0.50}"
python - \
    "$EVALUATION_DIR/summary.json" \
    "$MIN_SUCCESS_RATE" \
    "$MIN_SEQUENCE_SUCCESS_RATE" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    summary = json.load(source)
success = float(summary["metrics"]["semantic_success"]["rate"])
completion = float(summary["metrics"]["completion"]["rate"])
threshold = float(sys.argv[2])
sequence_threshold = float(sys.argv[3])
sequence_rates = [
    (entry["sequence"], float(entry["semantic_rate"]))
    for entry in summary["sequences"]
    if entry["semantic_rate"] is not None
]
worst_sequence, worst_rate = min(
    sequence_rates, key=lambda item: item[1]
)
print("================================================================")
print(f"FINAL completion={completion:.2%} semantic_success={success:.2%}")
print(
    f"WORST sequence={worst_sequence} semantic_success={worst_rate:.2%}"
)
print(f"results={sys.argv[1]}")
print("================================================================")
if success < threshold:
    raise SystemExit(
        f"Semantic success {success:.2%} is below required {threshold:.2%}. "
        "Training artifacts are preserved; increase FINETUNE_EPOCHS and rerun."
    )
if worst_rate < sequence_threshold:
    raise SystemExit(
        f"Worst-sequence success {worst_rate:.2%} is below required "
        f"{sequence_threshold:.2%}. Training artifacts are preserved; "
        "increase FINETUNE_EPOCHS and rerun."
    )
PY

echo "One-click Theia server training and evaluation completed successfully."
