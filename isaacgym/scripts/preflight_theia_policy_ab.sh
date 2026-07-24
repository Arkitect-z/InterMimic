#!/bin/bash
# CPU-only, fail-fast preflight for the frozen Raw/Full policy dataset.
#
# Usage:
#   NUM_ENVS=<N-times-replicas> bash isaacgym/scripts/preflight_theia_policy_ab.sh \
#     /data/theia_policy_ab
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT_INPUT="${1:-}"
NUM_ENVS="${NUM_ENVS:-}"
if [ -z "$DATA_ROOT_INPUT" ] || [ ! -d "$DATA_ROOT_INPUT" ]; then
    echo "Usage: NUM_ENVS=<balanced> bash $0 /path/to/prepared-policy-data"
    exit 1
fi
if ! [[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS must be set to a positive, reference-balanced integer"
    exit 1
fi
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((NUM_ENVS * 8))}"
if ! [[ "$MINIBATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "MINIBATCH_SIZE must be a positive integer"
    exit 1
fi
if [ $(((NUM_ENVS * 32) % MINIBATCH_SIZE)) -ne 0 ]; then
    echo "NUM_ENVS*32 must be divisible by MINIBATCH_SIZE."
    echo "Got NUM_ENVS=$NUM_ENVS MINIBATCH_SIZE=$MINIBATCH_SIZE."
    exit 1
fi
if [ $((MINIBATCH_SIZE % 4)) -ne 0 ]; then
    echo "MINIBATCH_SIZE must be divisible by seq_len=4."
    exit 1
fi

DATA_ROOT="$(CDPATH= cd -- "$DATA_ROOT_INPUT" && pwd)"
RAW_DIR="$DATA_ROOT/eligible/raw"
FULL_DIR="$DATA_ROOT/eligible/full"
PAIR_MANIFEST="$DATA_ROOT/policy_ab_manifest.json"
for required in "$RAW_DIR" "$FULL_DIR" "$PAIR_MANIFEST"; do
    if [ ! -e "$required" ]; then
        echo "Prepared-data artifact not found: $required"
        exit 1
    fi
done

read -r REFERENCE_COUNT EXCLUDED_COUNT < <(
    python - "$PAIR_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    manifest = json.load(source)
references = manifest.get("references")
if not isinstance(references, list) or not references:
    raise SystemExit("manifest has no non-empty references list")
if any(entry.get("included") is not True for entry in references):
    raise SystemExit("every formal manifest reference must have included=true")
if len({entry["reference_id"] for entry in references}) != len(references):
    raise SystemExit("manifest contains duplicate reference_id values")
if int(manifest.get("eligible_count", -1)) != len(references):
    raise SystemExit("eligible_count disagrees with references")
print(len(references), int(manifest.get("excluded_count", 0)))
PY
)
if [ "$NUM_ENVS" -lt "$REFERENCE_COUNT" ]; then
    echo "NUM_ENVS=$NUM_ENVS cannot cover $REFERENCE_COUNT references"
    exit 1
fi
if [ $((NUM_ENVS % REFERENCE_COUNT)) -ne 0 ]; then
    echo "NUM_ENVS=$NUM_ENVS must be divisible by N=$REFERENCE_COUNT."
    echo "Formal training requires exactly balanced reference replicas."
    exit 1
fi
if [ "$EXCLUDED_COUNT" -gt 0 ] && [ "${ACCEPT_EXCLUSIONS:-0}" != "1" ]; then
    echo "$EXCLUDED_COUNT candidate references were technically excluded."
    echo "Review $DATA_ROOT/excluded_pairs.csv, then rerun with"
    echo "ACCEPT_EXCLUSIONS=1 only if those reasons follow the frozen protocol."
    exit 1
fi

python "$SCRIPT_DIR/test_theia_training_protocol.py"

python "$SCRIPT_DIR/validate_theia_policy_ab.py" \
    --raw-dir "$RAW_DIR" \
    --full-dir "$FULL_DIR" \
    --manifest "$PAIR_MANIFEST" \
    --expected-count "$REFERENCE_COUNT" \
    --output "$DATA_ROOT/policy_ab_validation.json"

python "$SCRIPT_DIR/validate_theia_dataset.py" \
    --config isaacgym/src/intermimic/data/cfg/theia_full_train.yaml \
    --motion-file "$RAW_DIR" \
    --num-envs "$NUM_ENVS" \
    --manifest "$DATA_ROOT/raw_dataset_validation.json"

python "$SCRIPT_DIR/validate_theia_dataset.py" \
    --config isaacgym/src/intermimic/data/cfg/theia_full_train.yaml \
    --motion-file "$FULL_DIR" \
    --num-envs "$NUM_ENVS" \
    --manifest "$DATA_ROOT/full_dataset_validation.json"

python - \
    "$DATA_ROOT" \
    "$PAIR_MANIFEST" \
    "$REFERENCE_COUNT" \
    "$NUM_ENVS" \
    "$MINIBATCH_SIZE" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

result = {
    "schema_version": 1,
    "ready_for_gpu_smoke_test": True,
    "validated_at": datetime.now(timezone.utc).isoformat(),
    "reference_count": int(sys.argv[3]),
    "num_envs": int(sys.argv[4]),
    "minibatch_size": int(sys.argv[5]),
    "pair_manifest": str(manifest.resolve()),
    "pair_manifest_sha256": digest(manifest),
    "reports": {
        name: {
            "path": str((root / name).resolve()),
            "sha256": digest(root / name),
        }
        for name in (
            "policy_ab_validation.json",
            "raw_dataset_validation.json",
            "full_dataset_validation.json",
        )
    },
}
(root / "PRECHECK_READY.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
PY

echo "CPU preflight passed: references=$REFERENCE_COUNT num_envs=$NUM_ENVS"
echo "PPO minibatch=$MINIBATCH_SIZE (four minibatches per epoch by default)"
echo "Next gate: paired 100-epoch GPU smoke test."
echo "Receipt: $DATA_ROOT/PRECHECK_READY.json"
