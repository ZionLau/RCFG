#!/usr/bin/env bash
# LM1B step 3: run reward-selected grid search without PPL.
# This script assumes public assets and guidance reward models have already been prepared.
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

export LM1B_RUN_DIR="${ROOT}/logs/train/lm1b_50k"
export LM1B_CKPT_COMPILED="${LM1B_RUN_DIR}/checkpoints/last.ckpt"
export LM1B_CKPT="${LM1B_RUN_DIR}/checkpoints/last_uncompiled.ckpt"
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export HF_CACHE="${HF_CACHE:-${HF_HOME}/hub}"
export OUT_ROOT="${OUT_ROOT:-results/lm1b_full_512_3seed_uncompiled}"
export SCRIPT="${SCRIPT:-scripts/lm1b_attr_guidance_vf_baselines_treeg_nonumpy_cost.py}"
export TASKS="${TASKS:-ag_news,cola,imdb,tweet_offensive}"
export SEEDS="${SEEDS:-3407 3408 3409}"
export BATCH="${BATCH:-8}"
export EVAL_BATCH="${EVAL_BATCH:-16}"
export NUM_GPUS="${NUM_GPUS:-4}"
export N_SAMPLES="${N_SAMPLES:-512}"

if [ ! -f "$LM1B_CKPT_COMPILED" ] && [ ! -f "$LM1B_CKPT" ]; then
  echo "[ERROR] Missing LM1B checkpoint."
  echo "Expected either:"
  echo "  $LM1B_CKPT_COMPILED"
  echo "or:"
  echo "  $LM1B_CKPT"
  echo "Please follow the original Semicat/Categorical Flow-Map instructions to train the LM1B 50k checkpoint,"
  echo "then place it at logs/train/lm1b_50k/checkpoints/last.ckpt."
  exit 1
fi

# Build reward/verifier/PPL local links. This writes local_model_paths.env.
"$PYTHON_BIN" scripts/prepare_lm1b_local_models.py --root "$ROOT" --hf_cache "$HF_CACHE"
source "$ROOT/local_model_paths.env"

# Convert torch.compile checkpoint to ordinary keys if needed.
if [ ! -f "$LM1B_CKPT" ]; then
  "$PYTHON_BIN" scripts/uncompile_ckpt.py --src "$LM1B_CKPT_COMPILED" --dst "$LM1B_CKPT"
fi

test -f "$LM1B_CKPT" || { echo "Missing LM1B_CKPT=$LM1B_CKPT"; exit 1; }
test -f "$PPL_MODEL/config.json" || { echo "Missing local GPT2-large model: $PPL_MODEL"; exit 1; }

RAW="${OUT_ROOT}/grid_raw"
REPLAY_RAW="${OUT_ROOT}/replay_raw"
FINAL="${OUT_ROOT}/final_summary"
mkdir -p "$RAW" "$REPLAY_RAW" "$FINAL"

GRID_JOBS="${OUT_ROOT}/grid_jobs.txt"
: > "$GRID_JOBS"
add_job () { echo "$*" >> "$GRID_JOBS"; }

COMMON="--run_dir $LM1B_RUN_DIR \
  --ckpt $LM1B_CKPT \
  --tasks $TASKS \
  --n_samples $N_SAMPLES \
  --batch_size $BATCH \
  --eval_batch_size $EVAL_BATCH \
  --schedule constant \
  --early_stop 1.0 \
  --p_mode softmax \
  --reward_model_root $REWARD_ROOT \
  --verifier_model_root $VERIFIER_ROOT \
  --ppl_model $PPL_MODEL \
  --skip_ppl \
  --local_files_only \
  --device cuda"

# Base
for SEED in $SEEDS; do
  add_job "$PYTHON_BIN $SCRIPT $COMMON \
    --out $RAW/base_seed${SEED}.jsonl \
    --methods base \
    --nfes 2,4,8 \
    --step_sizes 0.5 \
    --seed $SEED"
done

# ATG/FMRG
for SEED in $SEEDS; do
  add_job "$PYTHON_BIN $SCRIPT $COMMON \
    --out $RAW/atg_fmrg_seed${SEED}.jsonl \
    --methods fmtg,fmrg \
    --nfes 2,4,8 \
    --step_sizes 0.5,1.0 \
    --seed $SEED"
done

# RCFG
for SEED in $SEEDS; do
  add_job "$PYTHON_BIN $SCRIPT $COMMON \
    --out $RAW/rcfg_seed${SEED}.jsonl \
    --methods gap_aware \
    --nfes 2,4,8 \
    --step_sizes 0.5,1.0 \
    --mirror_etas 0.1,0.3 \
    --mix_lambdas 0.2,0.4 \
    --quality_lambdas 0.05,0.1,0.2 \
    --sat_tau 0.30 \
    --sat_kappa 0.10 \
    --sat_power 1.0 \
    --sat_floor 0.25 \
    --seed $SEED"
done

# D-Flow, NFE=8
for SEED in $SEEDS; do
  for SRC_STEPS in 4 8; do
    for SRC_LR in 0.01 0.03 0.05 0.1; do
      for SRC_REG in 1e-5 1e-4 1e-3; do
        TAG="dflow_seed${SEED}_ss${SRC_STEPS}_lr${SRC_LR}_reg${SRC_REG}"
        add_job "$PYTHON_BIN $SCRIPT $COMMON \
          --out $RAW/${TAG}.jsonl \
          --methods dflow \
          --nfes 8 \
          --step_sizes 0.25,0.5,0.75,1.0 \
          --source_steps $SRC_STEPS \
          --source_lr $SRC_LR \
          --source_reg $SRC_REG \
          --seed $SEED"
      done
    done
  done
done

# SGFM, NFE=8
for SEED in $SEEDS; do
  for SRC_STEPS in 4 8; do
    for SRC_LR in 0.01 0.03 0.05 0.1; do
      for SRC_REG in 1e-5 1e-4 1e-3; do
        for BETA in 0.5 1.0 2.0; do
          TAG="sgfm_seed${SEED}_ss${SRC_STEPS}_lr${SRC_LR}_reg${SRC_REG}_beta${BETA}"
          add_job "$PYTHON_BIN $SCRIPT $COMMON \
            --out $RAW/${TAG}.jsonl \
            --methods sgfm \
            --nfes 8 \
            --step_sizes 0.25,0.5,0.75,1.0 \
            --source_steps $SRC_STEPS \
            --source_lr $SRC_LR \
            --source_reg $SRC_REG \
            --sgfm_beta $BETA \
            --seed $SEED"
        done
      done
    done
  done
done

# Tree-G, NFE=8
for SEED in $SEEDS; do
  for ACTIVE in 2 4; do
    for BRANCH in 4 8; do
      for NOISE in 0.02 0.05 0.1; do
        for ROLLOUT in 1 2; do
          TAG="treeg_seed${SEED}_a${ACTIVE}_b${BRANCH}_noise${NOISE}_roll${ROLLOUT}"
          add_job "$PYTHON_BIN $SCRIPT $COMMON \
            --out $RAW/${TAG}.jsonl \
            --methods treeg \
            --nfes 8 \
            --step_sizes 0.5 \
            --tree_active $ACTIVE \
            --tree_branch $BRANCH \
            --tree_noise $NOISE \
            --tree_value_rollout $ROLLOUT \
            --seed $SEED"
        done
      done
    done
  done
done

echo "[INFO] grid jobs:"; wc -l "$GRID_JOBS"

run_jobs () {
  local JOBS_FILE="$1"
  local LOG_PREFIX="$2"
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
        echo "[GPU ${GPU_ID}] ${LOG_PREFIX} job ${LINE_NO}: ${CMD}"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES="$GPU_ID" bash -lc "$CMD"
      fi
    done < "$JOBS_FILE"
  }
  PIDS=""
  for ((i=0; i<NUM_GPUS; i++)); do
    run_worker "$i" "$i" "$NUM_GPUS" &
    PIDS="$PIDS $!"
  done
  wait $PIDS
}

if [ "$(find "$RAW" -name '*.jsonl' | wc -l)" -lt "$(wc -l < "$GRID_JOBS")" ]; then
  run_jobs "$GRID_JOBS" "grid"
else
  echo "[SKIP] grid appears complete: $(find "$RAW" -name '*.jsonl' | wc -l) jsonl files."
fi

echo "[DONE] grid search finished."

echo "[NEXT] Run: bash commands/06_lm1b_select_configs.sh"
