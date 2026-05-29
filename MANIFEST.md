# Manifest

This file lists the code files included in the anonymized reproduction package.

## Commands

- `commands/01_prepare_public_hf_assets.sh`  
  Downloads/reuses GPT-2-large, BERT, external verifier models, and the public datasets used to train the four guidance reward BERTs.

- `commands/02_train_lm1b_guidance_reward_models.sh`  
  Trains the four LM1B guidance reward BERT models into `reward_models/`.

- `commands/03_run_text8_main_nfe8.sh`  
  Runs the Text8 Exact-1 main evaluation using `logs/train/text8_100k/checkpoints/last.ckpt`.

- `commands/04_run_text8_extended_tasks_nfe8.sh`  
  Runs the Text8 extended tasks.

- `commands/05_lm1b_run_grid_search_512.sh`  
  Runs LM1B grid search without PPL using `logs/train/lm1b_50k/checkpoints/last.ckpt`.

- `commands/06_lm1b_select_configs.sh`  
  Selects reward-best configurations and writes replay jobs.

- `commands/07_lm1b_replay_selected_with_ppl.sh`  
  Replays selected configurations with GPT-2-large PPL enabled.

- `commands/08_lm1b_aggregate_results.sh`  
  Aggregates replay outputs into paper-facing CSV summaries.

## Scripts

- `scripts/reward/train_bert_reward.py`  
  Fine-tunes BERT guidance reward models from public HuggingFace datasets.

- `scripts/text8_fmrg_semicat_eval.py`  
  Base Text8/Semicat helper utilities included with this package.

- `scripts/text8_fmrg_semicat_fixed_eval.py`  
  Fixed FMRG/Semicat helper used by the Text8 evaluator.

- `scripts/text8_all_simplex_guidance_eval.py`  
  Shared Text8 simplex guidance utilities and baseline implementations.

- `scripts/text8_combo_guidance_multitask_eval.py`  
  Main Text8 evaluator supporting Exact-1 and the extended tasks.

- `scripts/lm1b_attr_guidance_vf_baselines_treeg_nonumpy_cost.py`  
  Main LM1B evaluator for Base, ATG, FMRG, RCFG, D-Flow, SGFM, and Tree-G.

- `scripts/uncompile_ckpt.py`  
  Converts compiled checkpoints by removing `._orig_mod.` prefixes.

- `scripts/prepare_lm1b_local_models.py`  
  Creates local links for guidance reward models, hard verifier models, and GPT-2-large.

- `scripts/make_lm1b_selected_replay_no_pandas.py`  
  Selects reward-best configurations and writes replay jobs.

- `scripts/aggregate_lm1b_replay_no_pandas.py`  
  Aggregates LM1B replay outputs into final CSV summaries.

## Excluded files

The package intentionally excludes checkpoints, HuggingFace caches, raw results, generated tables, and large model files.
