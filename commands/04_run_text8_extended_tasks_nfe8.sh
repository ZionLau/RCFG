#!/usr/bin/env bash
# Run the four extended Text8 evaluations used in the appendix.
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
METHODS="fmrg,fmrg_sat,sat_pareto_quality,sfmrg_conflict_adapt,gap_aware,sat_semigroup_orth"
COMMON="--run_dir $TEXT8_RUN_DIR --ckpt $TEXT8_CKPT --n_samples 512 --batch_size 16 --nfes 8 --methods $METHODS --step_sizes 1.0,1.25,1.5 --mirror_etas 0.1,0.3,0.5 --mix_lambdas 0.2,0.4,0.6 --gap_betas 0.25,0.5,1.0 --quality_lambdas 0.1,0.2,0.4 --semi_betas 0.05,0.1,0.2 --tau_slot 0.20 --alpha_event 1.00 --alpha_slot 0.25 --rho_dup 0.50 --dup_gamma 1.00 --boundary_alpha 0.05 --reward_mix 0.02 --sat_tau 0.30 --sat_kappa 0.10 --sat_power 1.0 --sat_floor 0.25 --quality_entropy 0.02 --verify_temp 0.05 --verify_floor 0.25 --schedule paper --early_stops 1.0 --p_mode auto --seed 3407 --paired_eval --device cuda"

"$PYTHON_BIN" scripts/text8_combo_guidance_multitask_eval.py $COMMON \
  --out results/text8_multi_all_nfe8_seed3407.jsonl \
  --target_sets "award_city:award|city;game_music:game|music;science_system:science|system;city_music:city|music;game_science:game|science" \
  --task multi_all --multi_reduce sum

"$PYTHON_BIN" scripts/text8_combo_guidance_multitask_eval.py $COMMON \
  --out results/text8_position_nfe8_seed3407.jsonl \
  --target_sets "award:award;city:city;game:game;music:music;science:science" \
  --position_specs "award:award@20;city:city@20;game:game@20;music:music@20;science:science@20" \
  --task position

"$PYTHON_BIN" scripts/text8_combo_guidance_multitask_eval.py $COMMON \
  --out results/text8_forbidden_nfe8_seed3407.jsonl \
  --target_sets "award:award;city:city;game:game;music:music;science:science" \
  --task forbidden --forbid_weight 1.0

"$PYTHON_BIN" scripts/text8_combo_guidance_multitask_eval.py $COMMON \
  --out results/text8_exact_rare_nfe8_seed3407.jsonl \
  --target_sets "rare1:quantum;rare2:algorithm;rare3:electron;rare4:protein;rare5:astronomy" \
  --task exact_count --exact_count 1 --count_weight 2.0
