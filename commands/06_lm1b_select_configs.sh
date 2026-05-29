#!/usr/bin/env bash
# LM1B step 4: select the best hyperparameter configuration and write replay jobs.
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
export LM1B_CKPT="${LM1B_RUN_DIR}/checkpoints/last_uncompiled.ckpt"
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export HF_CACHE="${HF_CACHE:-${HF_HOME}/hub}"
export OUT_ROOT="${OUT_ROOT:-results/lm1b_full_512_3seed_uncompiled}"
export SCRIPT="${SCRIPT:-scripts/lm1b_attr_guidance_vf_baselines_treeg_nonumpy_cost.py}"
export SEEDS="${SEEDS:-3407 3408 3409}"
export BATCH="${BATCH:-8}"
export EVAL_BATCH="${EVAL_BATCH:-16}"
export N_SAMPLES="${N_SAMPLES:-512}"

"$PYTHON_BIN" scripts/prepare_lm1b_local_models.py --root "$ROOT" --hf_cache "$HF_CACHE"
source "$ROOT/local_model_paths.env"

test -d "${OUT_ROOT}/grid_raw" || { echo "[ERROR] Missing grid outputs: ${OUT_ROOT}/grid_raw"; exit 1; }
test -f "$LM1B_CKPT" || { echo "[ERROR] Missing LM1B_CKPT=$LM1B_CKPT"; exit 1; }

REPLAY_JOBS="${OUT_ROOT}/replay_selected_jobs.txt"
"$PYTHON_BIN" scripts/make_lm1b_selected_replay_no_pandas.py   --grid_dir "${OUT_ROOT}/grid_raw"   --out_root "$OUT_ROOT"   --jobs_out "$REPLAY_JOBS"   --script "$SCRIPT"   --python_bin "$PYTHON_BIN"   --run_dir "$LM1B_RUN_DIR"   --ckpt "$LM1B_CKPT"   --reward_root "$REWARD_ROOT"   --verifier_root "$VERIFIER_ROOT"   --ppl_model "$PPL_MODEL"   --seeds "$(echo $SEEDS | tr ' ' ',')"   --batch_size "$BATCH"   --eval_batch_size "$EVAL_BATCH"   --n_samples "$N_SAMPLES"

echo "[DONE] replay jobs written to $REPLAY_JOBS"
echo "[NEXT] Run: bash commands/07_lm1b_replay_selected_with_ppl.sh"
