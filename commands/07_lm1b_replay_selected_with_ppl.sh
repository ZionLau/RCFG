#!/usr/bin/env bash
# LM1B step 5: replay selected configurations with GPT-2-large PPL enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$(dirname "$ROOT"):${ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export OUT_ROOT="${OUT_ROOT:-results/lm1b_full_512_3seed_uncompiled}"
export NUM_GPUS="${NUM_GPUS:-4}"
REPLAY_JOBS="${OUT_ROOT}/replay_selected_jobs.txt"
REPLAY_RAW="${OUT_ROOT}/replay_raw"
mkdir -p "$REPLAY_RAW"

test -f "$REPLAY_JOBS" || { echo "[ERROR] Missing replay jobs: $REPLAY_JOBS"; exit 1; }

echo "[INFO] replay jobs:"; wc -l "$REPLAY_JOBS"

run_worker () {
  local GPU_ID="$1"
  local WORKER_ID="$2"
  local NUM_WORKERS="$3"
  local LINE_NO=0
  while IFS= read -r CMD; do
    LINE_NO=$((LINE_NO + 1))
    MOD=$(( (LINE_NO - 1) % NUM_WORKERS ))
    if [ "$MOD" -eq "$WORKER_ID" ]; then
      echo ""
      echo "============================================================"
      echo "[GPU ${GPU_ID}] replay job ${LINE_NO}: ${CMD}"
      echo "============================================================"
      CUDA_VISIBLE_DEVICES="$GPU_ID" bash -lc "$CMD"
    fi
  done < "$REPLAY_JOBS"
}

PIDS=""
for ((i=0; i<NUM_GPUS; i++)); do
  run_worker "$i" "$i" "$NUM_GPUS" &
  PIDS="$PIDS $!"
done
wait $PIDS

echo "[DONE] selected replay with PPL finished."
echo "[NEXT] Run: bash commands/08_lm1b_aggregate_results.sh"
