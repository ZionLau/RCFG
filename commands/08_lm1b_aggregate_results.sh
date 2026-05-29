#!/usr/bin/env bash
# LM1B step 6: aggregate selected replay outputs into paper-facing CSV files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(dirname "$ROOT"):${ROOT}:${PYTHONPATH:-}"
export OUT_ROOT="${OUT_ROOT:-results/lm1b_full_512_3seed_uncompiled}"

REPLAY_RAW="${OUT_ROOT}/replay_raw"
FINAL="${OUT_ROOT}/final_summary"
mkdir -p "$FINAL"

"$PYTHON_BIN" scripts/aggregate_lm1b_replay_no_pandas.py   --replay_dir "$REPLAY_RAW"   --out_dir "$FINAL"

echo ""
echo "[FINAL] Flow table:"
cat "$FINAL/lm1b_paper_flow_table.csv"
echo ""
echo "[FINAL] Non-flow NFE=8 table:"
cat "$FINAL/lm1b_paper_nonflow_nfe8_table.csv"
