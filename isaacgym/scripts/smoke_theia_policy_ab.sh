#!/bin/bash
# Sequential, topology-agnostic GPU smoke gate for the paired experiment.
#
# It runs Raw and Full for 50 + 50 bootstrap epochs (the second half is a
# true full-state resume), one full-sequence fine-tune epoch, and the same
# K=10 strict cohort shape used formally. It does not consume or alter the
# formal 22k run dirs.
#
# Usage:
#   NUM_ENVS=512 bash isaacgym/scripts/smoke_theia_policy_ab.sh \
#     PREPARED_DATA_ROOT [SMOKE_OUTPUT_ROOT]
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "$0")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT_INPUT="${1:-}"
SMOKE_ROOT_INPUT="${2:-$REPO_ROOT/checkpoints/theia_policy_ab_smoke}"
NUM_ENVS="${NUM_ENVS:-512}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((NUM_ENVS * 8))}"
BOOTSTRAP_TARGET="${SMOKE_BOOTSTRAP_EPOCHS:-100}"
FIRST_CHUNK="${SMOKE_FIRST_CHUNK_EPOCHS:-50}"
TRAINING_SEED="${SMOKE_TRAINING_SEED:-0}"
EVAL_SEED="${SMOKE_EVAL_SEED:-9090}"
EVAL_K="${SMOKE_EVAL_K:-10}"

if [ -z "$DATA_ROOT_INPUT" ] || [ ! -d "$DATA_ROOT_INPUT" ]; then
    echo "Usage: NUM_ENVS=512 bash $0 PREPARED_DATA_ROOT [SMOKE_OUTPUT_ROOT]"
    exit 1
fi
if [ "$FIRST_CHUNK" -le 0 ] || [ "$FIRST_CHUNK" -ge "$BOOTSTRAP_TARGET" ]; then
    echo "Require 0 < SMOKE_FIRST_CHUNK_EPOCHS < SMOKE_BOOTSTRAP_EPOCHS"
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-intermimic}"
if ! python -c "from isaacgym import gymapi" >/dev/null 2>&1; then
    if [ "${THEIA_SMOKE_CONDA_READY:-0}" = "1" ]; then
        echo "Isaac Gym is unavailable in Conda environment '$CONDA_ENV'."
        exit 1
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "Activate the '$CONDA_ENV' environment and rerun."
        exit 1
    fi
    exec env THEIA_SMOKE_CONDA_READY=1 \
        conda run --no-capture-output -n "$CONDA_ENV" \
        bash "$SCRIPT_PATH" "$@"
fi

DATA_ROOT="$(CDPATH= cd -- "$DATA_ROOT_INPUT" && pwd)"
PAIR_MANIFEST="$DATA_ROOT/policy_ab_manifest.json"
RAW_DIR="$DATA_ROOT/eligible/raw"
FULL_DIR="$DATA_ROOT/eligible/full"
mkdir -p "$SMOKE_ROOT_INPUT"
SMOKE_ROOT="$(CDPATH= cd -- "$SMOKE_ROOT_INPUT" && pwd)"

NUM_ENVS="$NUM_ENVS" \
MINIBATCH_SIZE="$MINIBATCH_SIZE" \
ACCEPT_EXCLUSIONS="${ACCEPT_EXCLUSIONS:-0}" \
    bash "$SCRIPT_DIR/preflight_theia_policy_ab.sh" "$DATA_ROOT"

SMOKE_SPEC="$SMOKE_ROOT/smoke_spec.txt"
SMOKE_SPEC_TMP="$(mktemp "$SMOKE_ROOT/.smoke_spec.XXXXXX")"
cleanup() {
    rm -f "$SMOKE_SPEC_TMP"
}
trap cleanup EXIT
{
    echo "data_root=$DATA_ROOT"
    echo "pair_manifest=$PAIR_MANIFEST"
    echo "pair_manifest_sha256=$(sha256sum "$PAIR_MANIFEST" | awk '{print $1}')"
    echo "num_envs=$NUM_ENVS"
    echo "minibatch_size=$MINIBATCH_SIZE"
    echo "bootstrap_epochs=$BOOTSTRAP_TARGET"
    echo "first_chunk_epochs=$FIRST_CHUNK"
    echo "finetune_epochs=1"
    echo "training_seed=$TRAINING_SEED"
    echo "evaluation_seed=$EVAL_SEED"
    echo "evaluation_k=$EVAL_K"
    echo "evaluation_fps=30"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_diff_sha256=$(git diff HEAD --binary | sha256sum | awk '{print $1}')"
    sha256sum \
        "$SCRIPT_PATH" \
        isaacgym/scripts/preflight_theia_policy_ab.sh \
        isaacgym/scripts/train_theia_full.sh \
        isaacgym/scripts/eval_theia_policy.sh \
        isaacgym/src/intermimic/data/cfg/theia_full_train.yaml \
        isaacgym/src/intermimic/data/cfg/theia_full_finetune.yaml \
        isaacgym/src/intermimic/data/cfg/theia_policy_eval.yaml
} > "$SMOKE_SPEC_TMP"
if [ -f "$SMOKE_SPEC" ]; then
    if ! cmp -s "$SMOKE_SPEC_TMP" "$SMOKE_SPEC"; then
        echo "Smoke specification changed for existing output: $SMOKE_ROOT"
        echo "Use a new smoke output directory; never mix quick and formal smoke."
        exit 1
    fi
else
    mv "$SMOKE_SPEC_TMP" "$SMOKE_SPEC"
fi

checkpoint_epoch() {
    python - "$1" <<'PY'
import sys
import torch

value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(value["epoch"]))
PY
}

run_condition() {
    local condition="$1"
    local data_dir="$2"
    local condition_root="$SMOKE_ROOT/$condition"
    local bootstrap_dir="$condition_root/bootstrap"
    local finetune_dir="$condition_root/finetune"
    local bootstrap_ckpt="$bootstrap_dir/theia_smplx/nn/mimic.pth"
    local finetune_ckpt="$finetune_dir/theia_smplx/nn/mimic.pth"
    local current_epoch=0

    mkdir -p "$condition_root"
    if [ -f "$bootstrap_ckpt" ]; then
        current_epoch="$(checkpoint_epoch "$bootstrap_ckpt")"
    fi
    if [ "$current_epoch" -gt "$BOOTSTRAP_TARGET" ]; then
        echo "$condition smoke checkpoint exceeds target: $current_epoch"
        exit 1
    fi
    if [ "$current_epoch" -eq 0 ]; then
        MOTION_FILE="$data_dir" \
        NUM_ENVS="$NUM_ENVS" \
        MINIBATCH_SIZE="$MINIBATCH_SIZE" \
        MAX_ITERATIONS="$FIRST_CHUNK" \
        OUTPUT_PATH="$bootstrap_dir" \
        SEED="$TRAINING_SEED" \
        CHECKPOINT_MODE=fresh \
            bash "$SCRIPT_DIR/train_theia_full.sh" bootstrap
        current_epoch="$(checkpoint_epoch "$bootstrap_ckpt")"
    fi
    if [ "$current_epoch" -lt "$BOOTSTRAP_TARGET" ]; then
        MOTION_FILE="$data_dir" \
        NUM_ENVS="$NUM_ENVS" \
        MINIBATCH_SIZE="$MINIBATCH_SIZE" \
        MAX_ITERATIONS="$((BOOTSTRAP_TARGET - current_epoch))" \
        OUTPUT_PATH="$bootstrap_dir" \
        SEED="$TRAINING_SEED" \
        CHECKPOINT_MODE=resume \
        CHECKPOINT="$bootstrap_ckpt" \
            bash "$SCRIPT_DIR/train_theia_full.sh" bootstrap
    fi
    current_epoch="$(checkpoint_epoch "$bootstrap_ckpt")"
    if [ "$current_epoch" -ne "$BOOTSTRAP_TARGET" ]; then
        echo "$condition bootstrap ended at $current_epoch, expected $BOOTSTRAP_TARGET"
        exit 1
    fi

    if [ ! -f "$finetune_ckpt" ]; then
        MOTION_FILE="$data_dir" \
        NUM_ENVS="$NUM_ENVS" \
        MINIBATCH_SIZE="$MINIBATCH_SIZE" \
        MAX_ITERATIONS=1 \
        OUTPUT_PATH="$finetune_dir" \
        SEED="$TRAINING_SEED" \
        CHECKPOINT_MODE=resume \
        CHECKPOINT="$bootstrap_ckpt" \
            bash "$SCRIPT_DIR/train_theia_full.sh" finetune
    fi
    current_epoch="$(checkpoint_epoch "$finetune_ckpt")"
    if [ "$current_epoch" -ne $((BOOTSTRAP_TARGET + 1)) ]; then
        echo "$condition fine-tune ended at $current_epoch, expected $((BOOTSTRAP_TARGET + 1))"
        exit 1
    fi

    env -u NUM_ENVS \
    CONDITION="$condition" \
    TRAINING_SEED="$TRAINING_SEED" \
    EVAL_SEED="$EVAL_SEED" \
    K="$EVAL_K" \
    FPS=30 \
        bash "$SCRIPT_DIR/eval_theia_policy.sh" \
        "$finetune_ckpt" \
        "$data_dir" \
        "$PAIR_MANIFEST" \
        "$condition_root/evaluation"
}

run_condition raw "$RAW_DIR"
run_condition full "$FULL_DIR"

python - \
    "$SMOKE_ROOT" \
    "$BOOTSTRAP_TARGET" \
    "$NUM_ENVS" \
    "$MINIBATCH_SIZE" \
    "$EVAL_K" <<'PY'
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
target = int(sys.argv[2])
num_envs = int(sys.argv[3])
report = {
    "schema_version": 1,
    "passed": True,
    "validated_at": datetime.now(timezone.utc).isoformat(),
    "bootstrap_epochs": target,
    "finetune_epochs": 1,
    "num_envs": num_envs,
    "minibatch_size": int(sys.argv[4]),
    "evaluation_k": int(sys.argv[5]),
    "conditions": {},
}
for condition in ("raw", "full"):
    log_paths = [
        root / condition / "bootstrap" / "train.log",
        root / condition / "finetune" / "train.log",
    ]
    text = "\n".join(path.read_text() for path in log_paths)
    invalid = [
        line
        for line in text.splitlines()
        if "epoch_num:" in line
        and re.search(
            r"(?i)(?<![A-Za-z])(nan|[-+]?inf)(?![A-Za-z])", line
        )
    ]
    fps = [
        float(value)
        for value in re.findall(r"fps total:\s*([0-9]+(?:\.[0-9]+)?)", text)
    ]
    validation_path = root / condition / "evaluation" / "validation.json"
    validation = json.loads(validation_path.read_text())
    if invalid or not fps or not all(math.isfinite(value) and value > 0 for value in fps):
        raise SystemExit(
            f"{condition} smoke log has invalid values or no positive FPS"
        )
    if not validation.get("valid"):
        raise SystemExit(f"{condition} formal cohort validation failed")
    if int(validation.get("k_trials", -1)) != int(sys.argv[5]):
        raise SystemExit(f"{condition} evaluation K differs from smoke spec")
    report["conditions"][condition] = {
        "mean_total_fps": sum(fps) / len(fps),
        "min_total_fps": min(fps),
        "logged_epochs": len(fps),
        "evaluation_episodes": validation["actual_episodes"],
        "evaluation_sha256": validation["episodes_sha256"],
    }

raw_fps = report["conditions"]["raw"]["mean_total_fps"]
full_fps = report["conditions"]["full"]["mean_total_fps"]
ratio = min(raw_fps, full_fps) / max(raw_fps, full_fps)
report["raw_full_fps_ratio"] = ratio
if ratio < 0.70:
    raise SystemExit(
        f"Raw/Full FPS ratio {ratio:.3f} is below the preregistered 0.70 gate"
    )
(root / "SMOKE_READY.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
PY

echo "GPU smoke gate passed: $SMOKE_ROOT/SMOKE_READY.json"
echo "Formal 8-run launcher may now invoke run_theia_policy_seed.sh."
