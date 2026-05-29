#!/usr/bin/env bash
# Download public HuggingFace assets used by the LM1B reproduction.
# This script does NOT download Semicat flow-map checkpoints or generated results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_HOME="${HF_HOME:-${ROOT}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE:-${HF_HOME}/hub}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[INFO] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

cache_dir = os.environ["HF_HUB_CACHE"]

models = [
    "bert-base-uncased",                       # base model for guidance reward training
    "gpt2-large",                              # NLL/PPL evaluator
    "textattack/bert-base-uncased-ag-news",    # hard verifier
    "textattack/bert-base-uncased-CoLA",       # hard verifier
    "textattack/bert-base-uncased-imdb",       # hard verifier
    "cardiffnlp/twitter-roberta-base-offensive", # hard verifier
]

for repo_id in models:
    print(f"[download model] {repo_id}")
    snapshot_download(repo_id=repo_id, cache_dir=cache_dir)

# Public datasets for training the four guidance reward BERTs.
# Text8/LM1B flow-map data are prepared by the original Semicat repository.
datasets = [
    ("ag_news", None),
    ("glue", "cola"),
    ("imdb", None),
    ("tweet_eval", "offensive"),
]
for name, config in datasets:
    print(f"[download dataset] {name}" + (f"/{config}" if config else ""))
    if config is None:
        load_dataset(name)
    else:
        load_dataset(name, config)

print("[DONE] Public HuggingFace assets are available in the local cache.")
PY
