#!/usr/bin/env bash
# Run the Text8 Exact-1 main evaluation. This writes raw CSV/JSONL results; no precomputed results are included.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(dirname "$ROOT"):${ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

TEXT8_RUN_DIR="${ROOT}/logs/train/text8_100k"
TEXT8_CKPT_COMPILED="${TEXT8_RUN_DIR}/checkpoints/last.ckpt"
TEXT8_CKPT="${TEXT8_RUN_DIR}/checkpoints/last_uncompiled.ckpt"
if [ ! -f "$TEXT8_CKPT" ]; then
  if [ -f "$TEXT8_CKPT_COMPILED" ]; then
    "$PYTHON_BIN" scripts/uncompile_ckpt.py --src "$TEXT8_CKPT_COMPILED" --dst "$TEXT8_CKPT"
  else
    echo "[ERROR] Missing Text8 checkpoint."
    echo "Place the reproduced Text8 100k checkpoint at: $TEXT8_CKPT_COMPILED"
    exit 1
  fi
fi
OUT="${OUT:-results/text8_combo_guidance_main.jsonl}"

"$PYTHON_BIN" scripts/text8_combo_guidance_multitask_eval.py \
  --run_dir "$TEXT8_RUN_DIR" \
  --ckpt "$TEXT8_CKPT" \
  --out "$OUT" \
  --n_samples 512 \
  --batch_size 16 \
  --nfes 8 \
  --methods fmrg,fmrg_sat,sat_pareto_quality,sfmrg_conflict_adapt,gap_aware,sat_semigroup_orth \
  --target_sets "award:award;city:city|cities|town;game:game|team|player|match;music:music|song|album|band;science:science|research|computer|system" \
  --task lexical_or \
  --step_sizes 1.0,1.25,1.5 \
  --mirror_etas 0.1,0.3,0.5 \
  --mix_lambdas 0.2,0.4,0.6 \
  --gap_betas 0.25,0.5,1.0 \
  --quality_lambdas 0.1,0.2,0.4 \
  --semi_betas 0.05,0.1,0.2 \
  --tau_slot 0.20 \
  --alpha_event 1.00 \
  --alpha_slot 0.25 \
  --rho_dup 0.50 \
  --dup_gamma 1.00 \
  --boundary_alpha 0.05 \
  --reward_mix 0.02 \
  --sat_tau 0.30 \
  --sat_kappa 0.10 \
  --sat_power 1.0 \
  --sat_floor 0.25 \
  --quality_entropy 0.02 \
  --verify_temp 0.05 \
  --verify_floor 0.25 \
  --schedule paper \
  --early_stops 1.0 \
  --p_mode auto \
  --seed 3407 \
  --paired_eval \
  --device cuda
