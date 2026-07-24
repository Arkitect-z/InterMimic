#!/bin/bash
# Sequential worker for one cluster/GPU shard. The server scheduler may launch
# one copy per GPU with a different frozen reference list.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LIST_PATH="${1:-}"
PREPARED_ROOT="${2:-}"
EXPERIMENT_ROOT="${3:-}"
if [ ! -f "$LIST_PATH" ] || [ ! -d "$PREPARED_ROOT" ] \
    || [ -z "$EXPERIMENT_ROOT" ]; then
    echo "Usage: bash $0 REFERENCE_LIST PREPARED_DATA_ROOT EXPERIMENT_ROOT"
    exit 1
fi

mkdir -p "$EXPERIMENT_ROOT"
python "$SCRIPT_DIR/check_theia_protomotions.py" \
    --output-json "$EXPERIMENT_ROOT/protomotions_version.json"
export THEIA_PROTOMOTIONS_CHECKED=1

mapfile -t REFERENCES < <(
    python - "$LIST_PATH" "$PREPARED_ROOT/policy_ab_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[2]).open() as source:
    manifest = json.load(source)
eligible = {
    str(entry["reference_id"])
    for entry in manifest.get("references", [])
    if entry.get("included") is True
}
if not eligible:
    raise SystemExit("Prepared manifest has no eligible references")
seen = set()
for raw_line in Path(sys.argv[1]).read_text().splitlines():
    value = raw_line.split("#", 1)[0].strip()
    if not value:
        continue
    reference_id = Path(value.rstrip("/")).name
    if reference_id in seen:
        raise SystemExit(f"Duplicate reference in list: {reference_id}")
    if reference_id not in eligible:
        raise SystemExit(
            f"Reference is absent from the frozen eligible set: {reference_id}"
        )
    seen.add(reference_id)
    print(reference_id)
PY
)
if [ "${#REFERENCES[@]}" -eq 0 ]; then
    echo "Reference list is empty: $LIST_PATH"
    exit 1
fi

mkdir -p "$EXPERIMENT_ROOT/shards"
LIST_SHA="$(sha256sum "$LIST_PATH" | awk '{print $1}')"
SHARD_NAME="${SHARD_NAME:-$(basename -- "$LIST_PATH")}"
if ! [[ "$SHARD_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "SHARD_NAME must contain only letters, digits, dot, underscore, hyphen"
    exit 1
fi
SHARD_SPEC="$EXPERIMENT_ROOT/shards/${SHARD_NAME}.spec.txt"
SHARD_RECEIPT="$EXPERIMENT_ROOT/shards/${SHARD_NAME}.ready.json"
exec 8>"$EXPERIMENT_ROOT/shards/${SHARD_NAME}.lock"
if ! flock -n 8; then
    echo "Another worker is already running shard $SHARD_NAME"
    exit 1
fi
SHARD_SPEC_TMP="$(mktemp "$EXPERIMENT_ROOT/shards/.${SHARD_NAME}.XXXXXX")"
trap 'rm -f "$SHARD_SPEC_TMP"' EXIT
{
    echo "list_sha256=$LIST_SHA"
    echo "references=${#REFERENCES[@]}"
    printf 'reference_ids=%s\n' "$(IFS=,; echo "${REFERENCES[*]}")"
} > "$SHARD_SPEC_TMP"
if [ -f "$SHARD_SPEC" ]; then
    if ! cmp -s "$SHARD_SPEC_TMP" "$SHARD_SPEC"; then
        echo "Shard list changed for existing output: $SHARD_SPEC"
        echo "Use a new SHARD_NAME or experiment root."
        exit 1
    fi
else
    mv "$SHARD_SPEC_TMP" "$SHARD_SPEC"
fi

for reference_id in "${REFERENCES[@]}"; do
    bash "$SCRIPT_DIR/run_theia_policy_reference.sh" \
        "$reference_id" "$PREPARED_ROOT" "$EXPERIMENT_ROOT"
done

python - \
    "$SHARD_RECEIPT" "$SHARD_NAME" "$LIST_PATH" "$LIST_SHA" \
    "${CUDA_VISIBLE_DEVICES:-unset}" "${REFERENCES[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
receipt = {
    "schema_version": 1,
    "complete": True,
    "shard_name": sys.argv[2],
    "list": str(Path(sys.argv[3]).resolve()),
    "list_sha256": sys.argv[4],
    "cuda_visible_devices": sys.argv[5],
    "reference_ids": sys.argv[6:],
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
echo "Completed shard $SHARD_NAME (${#REFERENCES[@]} references)"
