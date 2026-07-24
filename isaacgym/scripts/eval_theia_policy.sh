#!/bin/bash
# Formal InterMimic-style policy evaluation for one condition/training seed.
#
# This script is intentionally GPU-topology agnostic.  A server launcher may
# select a GPU before invoking it, but must not change K, the paired manifest,
# or the reference set between Raw and Full.
#
# Usage:
#   CONDITION=raw TRAINING_SEED=0 K=10 EVAL_SEED=10000 \
#     bash isaacgym/scripts/eval_theia_policy.sh \
#       CHECKPOINT DATA_DIR PAIR_MANIFEST OUTPUT_DIR
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

CHECKPOINT="${1:-}"
DATA_DIR_INPUT="${2:-}"
PAIR_MANIFEST="${3:-}"
OUTPUT_DIR="${4:-}"
CONDITION="${CONDITION:-}"
TRAINING_SEED="${TRAINING_SEED:-}"
K="${K:-10}"
FPS="${FPS:-30}"
EVAL_SEED="${EVAL_SEED:-}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: ${CHECKPOINT:-<missing>}"
    exit 1
fi
if [ -z "$DATA_DIR_INPUT" ] || [ ! -d "$DATA_DIR_INPUT" ]; then
    echo "Converted data directory not found: ${DATA_DIR_INPUT:-<missing>}"
    exit 1
fi
if [ -z "$PAIR_MANIFEST" ] || [ ! -f "$PAIR_MANIFEST" ]; then
    echo "Paired Raw/Full manifest not found: ${PAIR_MANIFEST:-<missing>}"
    exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "OUTPUT_DIR is required as the fourth argument"
    exit 1
fi
case "$CONDITION" in
    raw|full) ;;
    *)
        echo "CONDITION must be raw or full"
        exit 1
        ;;
esac
if ! [[ "$TRAINING_SEED" =~ ^[0-9]+$ ]]; then
    echo "TRAINING_SEED must be a non-negative integer"
    exit 1
fi
if ! [[ "$K" =~ ^[1-9][0-9]*$ ]]; then
    echo "K must be a positive integer"
    exit 1
fi
if [ -z "$EVAL_SEED" ]; then
    EVAL_SEED=$((10000 + TRAINING_SEED))
fi
if ! [[ "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "EVAL_SEED must be a non-negative integer"
    exit 1
fi
if ! python - "$FPS" <<'PY'
import math
import sys

try:
    fps = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isclose(fps, 30.0, rel_tol=0.0, abs_tol=1e-12) else 1)
PY
then
    echo "Formal Theia evaluation requires FPS=30, got '$FPS'"
    exit 1
fi

DATA_DIR="$(CDPATH= cd -- "$DATA_DIR_INPUT" && pwd)"
PAIR_MANIFEST="$(cd -- "$(dirname -- "$PAIR_MANIFEST")" && pwd)/$(basename -- "$PAIR_MANIFEST")"
OUTPUT_DIR="$(mkdir -p -- "$OUTPUT_DIR" && cd -- "$OUTPUT_DIR" && pwd)"
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
PAIR_MANIFEST_SHA256="$(sha256sum "$PAIR_MANIFEST" | awk '{print $1}')"
PIPELINE_FILES=(
    "$SCRIPT_DIR/eval_theia_policy.sh"
    "$SCRIPT_DIR/eval_theia.sh"
    "$SCRIPT_DIR/summarize_theia_eval.py"
    "$REPO_ROOT/isaacgym/src/intermimic/learning/intermimic_players.py"
    "$REPO_ROOT/isaacgym/src/intermimic/env/tasks/intermimic.py"
    "$REPO_ROOT/isaacgym/src/intermimic/data/cfg/theia_policy_eval.yaml"
)
EVALUATION_PIPELINE_SHA256="$(
    sha256sum "${PIPELINE_FILES[@]}" | sha256sum | awk '{print $1}'
)"

DATA_FINGERPRINT="$(
    python - "$DATA_DIR" "$PAIR_MANIFEST" "$CONDITION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
condition = sys.argv[3]
with open(sys.argv[2]) as source:
    manifest = json.load(source)
refs = [entry for entry in manifest["references"] if entry.get("included", True)]
expected = {
    entry[f"{condition}_filename"]: entry[f"{condition}_sha256"]
    for entry in refs
}
actual = {
    path.name: path
    for path in data_dir.glob("*.pt")
    if path.is_file()
}
if sorted(actual) != sorted(expected):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    raise SystemExit(
        f"{condition} data does not match frozen paired manifest; "
        f"missing={missing[:5]}, extra={extra[:5]}"
    )

records = []
for filename in sorted(expected):
    digest = hashlib.sha256(actual[filename].read_bytes()).hexdigest()
    if digest != expected[filename]:
        raise SystemExit(
            f"{condition} data hash mismatch for {filename}: "
            f"expected {expected[filename]}, got {digest}"
        )
    records.append(f"{digest}  {filename}\n")
aggregate = hashlib.sha256("".join(records).encode("utf-8")).hexdigest()
print(len(expected), aggregate)
PY
)"
read -r REFERENCE_COUNT CONDITION_DATA_SHA256 <<< "$DATA_FINGERPRINT"
if [ "$REFERENCE_COUNT" -lt 1 ]; then
    echo "Paired manifest contains no included references"
    exit 1
fi

if [ -f "$OUTPUT_DIR/validation.json" ] && [ "${FORCE_EVAL:-0}" != "1" ]; then
    if python - \
        "$OUTPUT_DIR/validation.json" \
        "$OUTPUT_DIR/manifest.txt" \
        "$CONDITION" \
        "$TRAINING_SEED" \
        "$K" \
        "$EVAL_SEED" \
        "$FPS" \
        "$CHECKPOINT_SHA256" \
        "$PAIR_MANIFEST_SHA256" \
        "$CONDITION_DATA_SHA256" \
        "$EVALUATION_PIPELINE_SHA256" \
        "$OUTPUT_DIR/episodes.csv" \
        "$OUTPUT_DIR/per_reference.csv" \
        "$OUTPUT_DIR/summary.json" <<'PY'
import hashlib
import json
import math
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


with open(sys.argv[1]) as source:
    result = json.load(source)
metadata = {}
try:
    with open(sys.argv[2]) as source:
        for line in source:
            key, separator, value = line.rstrip("\n").partition("=")
            if separator:
                metadata[key] = value
except OSError:
    pass
valid = result.get("valid") is True
valid &= str(result.get("condition")) == sys.argv[3]
valid &= int(result.get("training_seed", -1)) == int(sys.argv[4])
valid &= int(result.get("k_trials", -1)) == int(sys.argv[5])
valid &= int(result.get("evaluation_seed", -1)) == int(sys.argv[6])
valid &= math.isclose(
    float(result.get("fps", -1.0)), float(sys.argv[7]),
    rel_tol=0.0, abs_tol=1e-12,
)
valid &= result.get("paired_manifest_sha256") == sys.argv[9]
valid &= int(result.get("actual_episodes", -1)) == int(
    result.get("expected_episodes", -2)
)
valid &= metadata.get("checkpoint_sha256") == sys.argv[8]
valid &= (
    metadata.get("evaluation_config")
    == "isaacgym/src/intermimic/data/cfg/theia_policy_eval.yaml"
)
valid &= metadata.get("condition_data_sha256") == sys.argv[10]
valid &= metadata.get("evaluation_pipeline_sha256") == sys.argv[11]
valid &= result.get("episodes_sha256") == sha256(sys.argv[12])
valid &= result.get("per_reference_sha256") == sha256(sys.argv[13])
valid &= result.get("summary_sha256") == sha256(sys.argv[14])
valid &= result.get("evaluation_manifest_sha256") == sha256(sys.argv[2])
raise SystemExit(0 if valid else 1)
PY
    then
        echo "[SKIP] Existing validated evaluation: $OUTPUT_DIR"
        exit 0
    fi
    echo "Existing validation does not match this run: $OUTPUT_DIR"
    echo "Set FORCE_EVAL=1 to replace the known evaluation outputs."
    exit 1
fi

EXPECTED_ENVS=$((REFERENCE_COUNT * K))
if [ -n "${NUM_ENVS:-}" ] && [ "$NUM_ENVS" -ne "$EXPECTED_ENVS" ]; then
    echo "Formal evaluation requires NUM_ENVS=N*K=$EXPECTED_ENVS; got $NUM_ENVS"
    exit 1
fi

echo "Formal evaluation: condition=$CONDITION training_seed=$TRAINING_SEED"
echo "references=$REFERENCE_COUNT K=$K envs=$EXPECTED_ENVS eval_seed=$EVAL_SEED"
ATTEMPT_DIR="$(mktemp -d "$OUTPUT_DIR/.attempt.XXXXXX")"
echo "Transactional evaluation staging: $ATTEMPT_DIR"
MOTION_FILE="$DATA_DIR" \
NUM_ENVS="$EXPECTED_ENVS" \
EVAL_OUTPUT_DIR="$ATTEMPT_DIR" \
EVAL_SEED="$EVAL_SEED" \
TRAINING_SEED="$TRAINING_SEED" \
CONDITION="$CONDITION" \
CONDITION_DATA_SHA256="$CONDITION_DATA_SHA256" \
EVALUATION_PIPELINE_SHA256="$EVALUATION_PIPELINE_SHA256" \
EVAL_RUN_ID="$(basename -- "$OUTPUT_DIR")" \
EVAL_CONFIG="isaacgym/src/intermimic/data/cfg/theia_policy_eval.yaml" \
    bash "$SCRIPT_DIR/eval_theia.sh" "$CHECKPOINT"

# Preserve the simulator's episode/diagnostic summary.  The next command
# intentionally writes the paper's reference-level metrics to summary.json.
cp -- "$ATTEMPT_DIR/summary.json" "$ATTEMPT_DIR/episode_summary.json"

python "$SCRIPT_DIR/summarize_theia_eval.py" \
    --episodes "$ATTEMPT_DIR/episodes.csv" \
    --pair-manifest "$PAIR_MANIFEST" \
    --condition "$CONDITION" \
    --training-seed "$TRAINING_SEED" \
    --eval-seed "$EVAL_SEED" \
    --expected-k "$K" \
    --fps "$FPS" \
    --output-dir "$ATTEMPT_DIR" \
    --published-output-dir "$OUTPUT_DIR" \
    --evaluation-manifest "$ATTEMPT_DIR/manifest.txt"

# Publish validation.json last.  A failed rollout or interrupted publication
# therefore cannot masquerade as a complete result on the next invocation.
for artifact in "$ATTEMPT_DIR"/*; do
    [ -e "$artifact" ] || continue
    if [ "$(basename -- "$artifact")" = "validation.json" ]; then
        continue
    fi
    mv -f -- "$artifact" "$OUTPUT_DIR/"
done
mv -f -- "$ATTEMPT_DIR/validation.json" "$OUTPUT_DIR/validation.json"
rmdir -- "$ATTEMPT_DIR"
echo "Validated paper metrics: $OUTPUT_DIR/per_reference.csv"
