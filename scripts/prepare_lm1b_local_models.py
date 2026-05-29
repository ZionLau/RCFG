#!/usr/bin/env python3
"""Create local reward/verifier/PPL model links for the LM1B experiments.

Expected guidance reward models:
  $ROOT/reward_models/{ag_news_bert,cola_bert,imdb_bert,tweet_offensive_bert}

Expected public external models in a HuggingFace hub cache:
  gpt2-large
  textattack/bert-base-uncased-ag-news
  textattack/bert-base-uncased-CoLA
  textattack/bert-base-uncased-imdb
  cardiffnlp/twitter-roberta-base-offensive

This script writes $ROOT/local_model_paths.env for the reproduction shell.
"""
import argparse
import os
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', default='.')
p.add_argument('--hf_cache', default=None, help='HuggingFace hub cache, e.g. /path/to/hf_cache/hub')
p.add_argument('--out_env', default='local_model_paths.env')
args = p.parse_args()

root = Path(args.root).resolve()
hf_cache = Path(args.hf_cache or os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE') or (root / 'hf_cache' / 'hub')).resolve()
link_root = root / 'local_model_links'
reward_root = link_root / 'reward'
verifier_root = link_root / 'verifier'
reward_root.mkdir(parents=True, exist_ok=True)
verifier_root.mkdir(parents=True, exist_ok=True)

reward_sources = {
    'ag_news_bert': root / 'reward_models' / 'ag_news_bert',
    'cola_bert': root / 'reward_models' / 'cola_bert',
    'imdb_bert': root / 'reward_models' / 'imdb_bert',
    'tweet_offensive_bert': root / 'reward_models' / 'tweet_offensive_bert',
}

# Cache directory names use the HuggingFace hub convention: org/repo -> models--org--repo.
verifier_repos = {
    'ag_news_verifier': 'models--textattack--bert-base-uncased-ag-news',
    'cola_verifier': 'models--textattack--bert-base-uncased-CoLA',
    'imdb_verifier': 'models--textattack--bert-base-uncased-imdb',
    'tweet_offensive_verifier': 'models--cardiffnlp--twitter-roberta-base-offensive',
}
ppl_repo_candidates = [
    'models--gpt2-large',
    'models--openai-community--gpt2-large',
]

missing = []

def latest_snapshot(repo_cache: Path):
    snaps = repo_cache / 'snapshots'
    if not snaps.exists():
        return None
    candidates = [p for p in snaps.iterdir() if p.is_dir() and (p / 'config.json').exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def link(src: Path, dst: Path, label: str):
    if src is None or not (src / 'config.json').exists():
        missing.append(f'{label}: {src}')
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f'[link] {dst} -> {src}')

for name, src in reward_sources.items():
    link(src, reward_root / name, name)

for name, repo_dir in verifier_repos.items():
    link(latest_snapshot(hf_cache / repo_dir), verifier_root / name, name)

ppl_model = None
for repo_dir in ppl_repo_candidates:
    cand = latest_snapshot(hf_cache / repo_dir)
    if cand is not None:
        ppl_model = cand
        break
if ppl_model is None or not (ppl_model / 'config.json').exists():
    missing.append(f'gpt2-large: {hf_cache}/models--gpt2-large or {hf_cache}/models--openai-community--gpt2-large')

out_env = root / args.out_env
out_env.write_text(
    f'export REWARD_ROOT="{reward_root}"\n'
    f'export VERIFIER_ROOT="{verifier_root}"\n'
    f'export PPL_MODEL="{ppl_model if ppl_model is not None else ""}"\n',
    encoding='utf-8',
)
print('[write]', out_env)
print(out_env.read_text())

if missing:
    print('\n[ERROR] Missing required local model directories:')
    for x in missing:
        print('  ', x)
    print('\nRun commands/prepare_public_hf_assets.sh for public models, and run')
    print('commands/train_lm1b_guidance_reward_models.sh for the four guidance reward models.')
    raise SystemExit(1)
