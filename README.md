# RCFG reproduction code

This repository contains anonymized reproduction code and command templates for the Text8 and LM1B experiments in **Reliability-Calibrated Guidance for Few-Step Categorical Flow-Map Text Generation**.

This repository does **not** include checkpoints, raw CSV/JSONL outputs, generated tables, HuggingFace caches, or generated result files. The `.gitignore` excludes these files.

## Overview

This package is designed to be copied into the root of the original Categorical Flow-Map / Semicat repository.

The base Text8/LM1B data preparation and flow-map training should be done by following the original Semicat project. This repository does not provide or modify the original Semicat training commands. It only provides the RCFG guidance/evaluation code used after the base checkpoints are available.

Required base checkpoints:

```text
Text8: 100k-step Categorical Flow-Map checkpoint
LM1B:  50k-step Categorical Flow-Map checkpoint
```

Required directory layout after copying this package into the Semicat root:

```text
semicat/
  commands/
  scripts/
  configs/
  semicat/
  logs/train/text8_100k/checkpoints/last.ckpt
  logs/train/lm1b_50k/checkpoints/last.ckpt
  reward_models/
```

The expected reproduction order is:

1. clone the original Semicat/Categorical Flow-Map repository;
2. follow the original Semicat instructions to prepare Text8/LM1B data and train the base checkpoints;
3. place the Text8 100k checkpoint at `logs/train/text8_100k/checkpoints/last.ckpt`;
4. place the LM1B 50k checkpoint at `logs/train/lm1b_50k/checkpoints/last.ckpt`;
5. copy this RCFG reproduction package into the Semicat root;
6. prepare public HuggingFace assets;
7. train the four LM1B guidance reward BERT models;
8. run Text8 evaluation;
9. run staged LM1B grid search, replay, and aggregation.

## Step 0: Clone and prepare the original Semicat project

Clone the original Categorical Flow-Map / Semicat repository:

```bash
git clone https://github.com/olsdavis/semicat semicat
cd semicat
```

Follow the original repository instructions to prepare datasets and train the base Text8 and LM1B Categorical Flow-Map checkpoints.

This package includes the Text8 helper files needed by our evaluator, including:

```text
scripts/text8_fmrg_semicat_eval.py
scripts/text8_fmrg_semicat_fixed_eval.py
scripts/text8_all_simplex_guidance_eval.py
scripts/text8_combo_guidance_multitask_eval.py
```

## Step 1: Place the reproduced flow-map checkpoints

Place the reproduced Text8 100k checkpoint here:

```text
logs/train/text8_100k/checkpoints/last.ckpt
```

Place the reproduced LM1B 50k checkpoint here:

```text
logs/train/lm1b_50k/checkpoints/last.ckpt
```

The evaluation scripts automatically create the corresponding uncompiled checkpoints:

```text
logs/train/text8_100k/checkpoints/last_uncompiled.ckpt
logs/train/lm1b_50k/checkpoints/last_uncompiled.ckpt
```

This uncompile step removes `._orig_mod.` prefixes. It is required for correct flow-map Jacobian pullback behavior in FMRG and RCFG.

## Step 2: Prepare public HuggingFace assets

Run:

```bash
bash commands/01_prepare_public_hf_assets.sh
```

This prepares the local HuggingFace cache under:

```text
hf_cache/
```

The script downloads or reuses the following public models:

```text
gpt2-large
bert-base-uncased
textattack/bert-base-uncased-ag-news
textattack/bert-base-uncased-CoLA
textattack/bert-base-uncased-imdb
cardiffnlp/twitter-roberta-base-offensive
```

It also downloads the public datasets used for training the four guidance reward BERTs:

```text
ag_news
glue/cola
imdb
tweet_eval/offensive
```

LM1B NLL/PPL is evaluated with **GPT-2-large**.

## Step 3: Train the four LM1B guidance reward BERT models

Run:

```bash
bash commands/02_train_lm1b_guidance_reward_models.sh
```

This creates:

```text
reward_models/ag_news_bert
reward_models/cola_bert
reward_models/imdb_bert
reward_models/tweet_offensive_bert
```

The training script uses:

```text
ag_news              -> AGNews-Sports reward
glue/cola            -> CoLA-Acceptable reward
imdb                 -> IMDb-Positive reward
tweet_eval/offensive -> TweetEval-NonOffensive reward
```

These guidance reward models are used during guidance. They are intentionally separate from the external hard verifier models, which are used for evaluation.

## Step 4: Run Text8 evaluation

Run the Text8 Exact-1 main task:

```bash
bash commands/03_run_text8_main_nfe8.sh
```

Run the extended Text8 tasks:

```bash
bash commands/04_run_text8_extended_tasks_nfe8.sh
```

These commands use:

```text
logs/train/text8_100k/checkpoints/last.ckpt
```

and write raw result files under `results/`. No precomputed results are included in this repository.

## Step 5: Run LM1B evaluation in staged commands

Run the commands below in order.

### 5.1 Grid search without PPL

```bash
bash commands/05_lm1b_run_grid_search_512.sh
```

This runs the 512-sample / 3-seed grid search with `--skip_ppl`.

### 5.2 Select hyperparameters and write replay jobs

```bash
bash commands/06_lm1b_select_configs.sh
```

This selects the best hyperparameter setting for each task-method-NFE group by mean reward.

### 5.3 Replay selected configurations with GPT-2-large PPL

```bash
bash commands/07_lm1b_replay_selected_with_ppl.sh
```

This replays only the selected configurations and computes NLL/PPL with GPT-2-large.

### 5.4 Aggregate final LM1B results

```bash
bash commands/08_lm1b_aggregate_results.sh
```

Final CSV files are written under:

```text
results/lm1b_full_512_3seed_uncompiled/final_summary/
```

The two paper-facing output files are:

```text
lm1b_paper_flow_table.csv
lm1b_paper_nonflow_nfe8_table.csv
```

## Main scripts

```text
scripts/reward/train_bert_reward.py
scripts/text8_fmrg_semicat_eval.py
scripts/text8_fmrg_semicat_fixed_eval.py
scripts/text8_all_simplex_guidance_eval.py
scripts/text8_combo_guidance_multitask_eval.py
scripts/lm1b_attr_guidance_vf_baselines_treeg_nonumpy_cost.py
scripts/uncompile_ckpt.py
scripts/prepare_lm1b_local_models.py
scripts/make_lm1b_selected_replay_no_pandas.py
scripts/aggregate_lm1b_replay_no_pandas.py
```

## Notes

- This repository does not provide or modify the original Semicat training commands.
- The flow-map checkpoints must be placed at the fixed paths listed above.
- The LM1B guidance reward models and external verifier models are intentionally separated.
- The LM1B selection and aggregation pipeline is written without pandas/numpy.
- Checkpoints, model caches, and generated result files are not included in this repository.
