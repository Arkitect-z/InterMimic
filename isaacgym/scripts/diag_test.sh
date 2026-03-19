#!/bin/bash
# Diagnostic test: run policy headless with detailed per-step logging
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

OBJ="CupBlue"
CKPT="checkpoints/theia_cup/theia_smplx/nn/mimic.pth"
CFG_ENV="isaacgym/src/intermimic/data/cfg/theia_train.yaml"
CFG_TRAIN="isaacgym/src/intermimic/data/cfg/train/rlg/theia.yaml"
DIAG="diag_output.txt"

echo "============================================"
echo "  Diagnostic test: $OBJ"
echo "  Checkpoint: $CKPT"
echo "============================================"

rm -f "$DIAG"

timeout 60 python -m intermimic.run \
  --task InterMimic \
  --cfg_env "$CFG_ENV" \
  --cfg_train "$CFG_TRAIN" \
  --headless \
  --test \
  --num_envs 4 \
  --checkpoint "$CKPT" || true

echo ""
echo "========== DIAGNOSTIC ANALYSIS =========="
if [ -f "$DIAG" ]; then
  TOTAL=$(wc -l < "$DIAG")
  echo "Total log lines: $TOTAL"
  echo ""
  echo "--- Phase & Contact Statistics ---"
  CONTACT_STEPS=$(grep 'phase=CONTACT' "$DIAG" | grep -v RESET | wc -l)
  FREE_STEPS=$(grep 'phase=free' "$DIAG" | grep -v RESET | wc -l)
  TOUCH_IN_CONTACT=$(grep 'phase=CONTACT' "$DIAG" | grep 'touch=TOUCH' | wc -l)
  NO_TOUCH_IN_CONTACT=$(grep 'phase=CONTACT' "$DIAG" | grep 'touch=no' | wc -l)
  echo "Contact phase steps: $CONTACT_STEPS"
  echo "Free phase steps:    $FREE_STEPS"
  echo "CONTACT+TOUCH:       $TOUCH_IN_CONTACT"
  echo "CONTACT+no_touch:    $NO_TOUCH_IN_CONTACT"
  if [ "$CONTACT_STEPS" -gt 0 ]; then
    PCT=$(python3 -c "print(f'{100*$TOUCH_IN_CONTACT/$CONTACT_STEPS:.1f}%')")
    echo "Touch rate in contact phase: $PCT"
  fi
  echo ""
  echo "--- RESET events ---"
  grep 'RESET' "$DIAG" | head -15
  echo ""
  echo "--- Sample: Contact phase (first 15) ---"
  grep 'phase=CONTACT' "$DIAG" | grep -v RESET | head -15
  echo ""
  echo "--- Sample: Free phase (first 10) ---"
  grep 'phase=free' "$DIAG" | grep -v RESET | head -10
  echo ""
  echo "--- Wrist error & hand-obj distance in CONTACT phase ---"
  grep 'phase=CONTACT' "$DIAG" | grep -v RESET | \
    python3 -c "
import sys
wrist_errs, hand_objs, fin_errs, rcgs, rfins = [], [], [], [], []
for line in sys.stdin:
    parts = dict(p.split('=') for p in line.strip().split() if '=' in p)
    wrist_errs.append(float(parts.get('wrist_err','0')))
    hand_objs.append(float(parts.get('hand_obj','0')))
    fin_errs.append(float(parts.get('fin_err','0')))
    rcgs.append(float(parts.get('rcg','0')))
    rfins.append(float(parts.get('r_fin','0')))
n = len(wrist_errs)
if n == 0:
    print('No data')
else:
    print(f'  N={n} steps in CONTACT phase')
    print(f'  wrist_err:  mean={sum(wrist_errs)/n:.4f}  max={max(wrist_errs):.4f}')
    print(f'  hand_obj:   mean={sum(hand_objs)/n:.4f}  max={max(hand_objs):.4f}')
    print(f'  fin_err:    mean={sum(fin_errs)/n:.4f}  max={max(fin_errs):.4f}')
    print(f'  rcg:        mean={sum(rcgs)/n:.4f}')
    print(f'  r_finger:   mean={sum(rfins)/n:.4f}')
"
else
  echo "No diagnostic output found!"
fi
