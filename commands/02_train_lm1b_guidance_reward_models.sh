#!/usr/bin/env bash
# Train the four differentiable guidance reward BERTs from public HF datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE:-${HF_HOME}/hub}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

REWARD_ROOT="${REWARD_ROOT:-${ROOT}/reward_models}"
BASE_MODEL="${REWARD_BASE_MODEL:-bert-base-uncased}"
SEED="${REWARD_SEED:-3407}"
EPOCHS="${REWARD_EPOCHS:-3}"
MAX_STEPS="${REWARD_MAX_STEPS:-0}"
BATCH_SIZE="${REWARD_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${REWARD_EVAL_BATCH_SIZE:-64}"
LR="${REWARD_LR:-2e-5}"
MAX_LENGTH="${REWARD_MAX_LENGTH:-128}"
EVAL_EVERY="${REWARD_EVAL_EVERY:-500}"
NUM_WORKERS="${REWARD_NUM_WORKERS:-4}"
FP16="${REWARD_FP16:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

mkdir -p "$REWARD_ROOT"

echo "[INFO] Training guidance reward models into $REWARD_ROOT"
echo "[INFO] BASE_MODEL=$BASE_MODEL SEED=$SEED EPOCHS=$EPOCHS MAX_STEPS=$MAX_STEPS"

train_one () {
  local TASK="$1"
  local OUT_DIR="$2"
  if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$OUT_DIR/config.json" ]; then
    echo "[SKIP] Existing reward model: $OUT_DIR"
    return
  fi
  echo "[TRAIN] $TASK -> $OUT_DIR"
  CMD=("$PYTHON_BIN" scripts/reward/train_bert_reward.py
    --task "$TASK"
    --out_dir "$OUT_DIR"
    --base_model "$BASE_MODEL"
    --max_length "$MAX_LENGTH"
    --batch_size "$BATCH_SIZE"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --epochs "$EPOCHS"
    --max_steps "$MAX_STEPS"
    --lr "$LR"
    --eval_every "$EVAL_EVERY"
    --seed "$SEED"
    --num_workers "$NUM_WORKERS")
  if [ "$FP16" = "1" ]; then
    CMD+=(--fp16)
  fi
  if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    CMD+=(--local_files_only)
  fi
  "${CMD[@]}"
}

train_one ag_news "$REWARD_ROOT/ag_news_bert"
train_one cola "$REWARD_ROOT/cola_bert"
train_one imdb "$REWARD_ROOT/imdb_bert"
train_one tweet_offensive "$REWARD_ROOT/tweet_offensive_bert"

echo "[DONE] Guidance reward models are ready under $REWARD_ROOT"
