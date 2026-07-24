#!/bin/bash
# Train exactly one Raw policy and one Refined policy for one reference.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#   bash isaacgym/scripts/run_theia_policy_reference.sh \
#     S1L33P01T0508V01 PREPARED_DATA_ROOT EXPERIMENT_ROOT
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

REFERENCE_ID="${1:-}"
PREPARED_ROOT_INPUT="${2:-}"
EXPERIMENT_ROOT_INPUT="${3:-}"
if [ "${THEIA_PROTOMOTIONS_CHECKED:-0}" != "1" ]; then
    python "$SCRIPT_DIR/check_theia_protomotions.py"
fi
if ! [[ "$REFERENCE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "reference_id must contain only letters, digits, dot, underscore, hyphen"
    exit 1
fi
if [ ! -f "$PREPARED_ROOT_INPUT/policy_ab_manifest.json" ]; then
    echo "Prepared policy manifest not found: $PREPARED_ROOT_INPUT"
    exit 1
fi
if [ -z "$EXPERIMENT_ROOT_INPUT" ]; then
    echo "Usage: bash $0 REFERENCE_ID PREPARED_DATA_ROOT EXPERIMENT_ROOT"
    exit 1
fi

TRAIN_EPOCHS="${TRAIN_EPOCHS:-1100}"
TRAINING_SEED="${TRAINING_SEED:-0}"
EVAL_SEED="${EVAL_SEED:-10000}"
K="${K:-10}"
TARGET_ENVS="${TARGET_ENVS:-2048}"
ALLOW_PROTOCOL_OVERRIDE="${ALLOW_PROTOCOL_OVERRIDE:-0}"
TRAIN_ENV_CONFIG="isaacgym/src/intermimic/data/cfg/theia_reference_train.yaml"

case "$ALLOW_PROTOCOL_OVERRIDE" in
    0|1) ;;
    *)
        echo "ALLOW_PROTOCOL_OVERRIDE must be 0 or 1"
        exit 1
        ;;
esac
if ! [[ "$TRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_EPOCHS must be a positive integer"
    exit 1
fi
if ! [[ "$TRAINING_SEED" =~ ^[0-9]+$ ]] \
    || ! [[ "$EVAL_SEED" =~ ^[0-9]+$ ]] \
    || ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAINING_SEED/EVAL_SEED/K are invalid"
    exit 1
fi
if [ "$ALLOW_PROTOCOL_OVERRIDE" = "0" ] && {
    [ "$TRAIN_EPOCHS" -ne 1100 ] \
    || [ "$TRAINING_SEED" -ne 0 ] \
    || [ "$EVAL_SEED" -ne 10000 ] \
    || [ "$K" -ne 10 ];
}; then
    echo "Rebuttal protocol is fixed at 1100 epochs, one seed=0, eval seed=10000, K=10."
    echo "ALLOW_PROTOCOL_OVERRIDE=1 is only for isolated smoke/pilot directories."
    exit 1
fi

PREPARED_ROOT="$(CDPATH= cd -- "$PREPARED_ROOT_INPUT" && pwd)"
mkdir -p "$EXPERIMENT_ROOT_INPUT"
EXPERIMENT_ROOT="$(CDPATH= cd -- "$EXPERIMENT_ROOT_INPUT" && pwd)"
PAIR_ROOT="$EXPERIMENT_ROOT/references/$REFERENCE_ID"
VIEW_ROOT="$PAIR_ROOT/view"
RUNS_ROOT="$PAIR_ROOT/runs"
mkdir -p "$PAIR_ROOT"

exec 9>"$PAIR_ROOT/.pair.lock"
if ! flock -n 9; then
    echo "Another paired run is active for $REFERENCE_ID"
    exit 1
fi

python "$SCRIPT_DIR/prepare_theia_reference_view.py" \
    --prepared-root "$PREPARED_ROOT" \
    --reference-id "$REFERENCE_ID" \
    --output-root "$VIEW_ROOT"

PAIR_SPEC_TMP="$(mktemp "$PAIR_ROOT/.pair_spec.XXXXXX")"
trap 'rm -f "$PAIR_SPEC_TMP"' EXIT
{
    echo "protocol=single_reference_raw_vs_refined_v1"
    echo "reference_id=$REFERENCE_ID"
    echo "train_epochs=$TRAIN_EPOCHS"
    echo "training_seed=$TRAINING_SEED"
    echo "evaluation_seed=$EVAL_SEED"
    echo "k_trials=$K"
    echo "target_envs=$TARGET_ENVS"
    echo "num_envs=${NUM_ENVS:-auto}"
    echo "minibatch_size=${MINIBATCH_SIZE:-auto}"
    echo "train_env_config=$TRAIN_ENV_CONFIG"
    echo "torch_deterministic=0"
    echo "parent_manifest_sha256=$(sha256sum "$PREPARED_ROOT/policy_ab_manifest.json" | awk '{print $1}')"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_diff_sha256=$(git diff HEAD --binary | sha256sum | awk '{print $1}')"
} > "$PAIR_SPEC_TMP"
if [ -f "$PAIR_ROOT/pair_spec.txt" ]; then
    if ! cmp -s "$PAIR_SPEC_TMP" "$PAIR_ROOT/pair_spec.txt"; then
        echo "Pair protocol changed for existing output: $PAIR_ROOT"
        echo "Use a new experiment root."
        exit 1
    fi
else
    mv "$PAIR_SPEC_TMP" "$PAIR_ROOT/pair_spec.txt"
fi

run_condition() {
    local condition="$1"
    local data_dir="$VIEW_ROOT/data/$condition"
    echo
    echo "================================================================"
    echo "reference=$REFERENCE_ID condition=$condition"
    echo "one training run: seed=$TRAINING_SEED epochs=$TRAIN_EPOCHS"
    echo "================================================================"
    ALLOW_NONFORMAL_PROTOCOL="$ALLOW_PROTOCOL_OVERRIDE" \
    PROTOCOL_MODE=single_reference_rebuttal \
    BOOTSTRAP_EPOCHS="$TRAIN_EPOCHS" \
    FINETUNE_EPOCHS=0 \
    TRAIN_ENV_CONFIG="$TRAIN_ENV_CONFIG" \
    TORCH_DETERMINISTIC=0 \
    TARGET_ENVS="$TARGET_ENVS" \
    K="$K" \
    EVAL_SEED="$EVAL_SEED" \
        bash "$SCRIPT_DIR/run_theia_policy_seed.sh" \
        "$condition" "$TRAINING_SEED" "$data_dir" \
        "$VIEW_ROOT/pair_manifest.json" "$RUNS_ROOT"
}

run_condition raw
run_condition full

python - "$PAIR_ROOT" "$REFERENCE_ID" "$TRAINING_SEED" "$K" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
reference_id = sys.argv[2]
seed = int(sys.argv[3])
k = int(sys.argv[4])
conditions = {}
for condition in ("raw", "full"):
    evaluation = (
        root / "runs" / condition / f"seed_{seed}"
        / "evaluation" / "final"
    )
    validation = evaluation / "validation.json"
    summary = evaluation / "summary.json"
    if not validation.is_file() or not summary.is_file():
        raise SystemExit(f"Missing validated {condition} evaluation")
    document = json.loads(validation.read_text())
    if document.get("valid") is not True:
        raise SystemExit(f"Invalid {condition} evaluation")
    conditions[condition] = {
        "evaluation_dir": str(evaluation.resolve()),
        "validation_sha256": hashlib.sha256(
            validation.read_bytes()
        ).hexdigest(),
        "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }
receipt = {
    "schema_version": 1,
    "protocol": "single_reference_raw_vs_refined_v1",
    "reference_id": reference_id,
    "training_runs_per_condition": 1,
    "training_seed": seed,
    "k_trials": k,
    "conditions": conditions,
}
rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
output = root / "PAIR_READY.json"
if output.exists() and output.read_text() != rendered:
    raise SystemExit(f"Existing paired receipt differs: {output}")
temporary = root / f".PAIR_READY.json.tmp.{os.getpid()}"
temporary.write_text(rendered)
temporary.replace(output)
PY

echo "Completed one Raw + one Refined training run: $REFERENCE_ID"
echo "receipt=$PAIR_ROOT/PAIR_READY.json"
