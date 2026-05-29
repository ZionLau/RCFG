#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base Text8/Semicat helper utilities used by the RCFG Text8 evaluators.

This helper is included so that the anonymized reproduction package is
self-contained after being copied into the original Semicat repository root.
It loads a reproduced Semicat flow-map checkpoint, reads the Text8 metadata/cache
prepared by the original repository, and provides the basic sampling/evaluation
utilities reused by the Text8 guidance scripts.
"""
import json
import os
import pickle
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    import hydra
    from omegaconf import OmegaConf
except Exception as e:
    raise RuntimeError("Need hydra-core and omegaconf installed in the Semicat environment.") from e


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_known_prefixes(sd):
    out = {}
    for k, v in sd.items():
        kk = str(k).replace('._orig_mod.', '.')
        for pref in ('_orig_mod.', 'model.', 'module.'):
            while kk.startswith(pref):
                kk = kk[len(pref):]
        out[kk] = v
    return out


def load_model(run_dir: str, ckpt: str, device: str):
    run_dir = Path(run_dir)
    cfg_path = run_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Hydra config not found: {cfg_path}')
    cfg = OmegaConf.load(str(cfg_path))
    model = hydra.utils.instantiate(cfg.model)
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = obj.get('state_dict', obj)
    model_keys = set(model.state_dict().keys())
    raw_overlap = len(model_keys & set(sd.keys()))
    stripped_sd = _strip_known_prefixes(sd)
    stripped_overlap = len(model_keys & set(stripped_sd.keys()))
    chosen = stripped_sd if stripped_overlap > raw_overlap else sd
    incompatible = model.load_state_dict(chosen, strict=False)
    print(f'[ckpt-load] raw_overlap={raw_overlap} stripped_overlap={stripped_overlap} missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}')
    model.to(device)
    model.eval()

    flow = model
    if not (hasattr(flow, 'prior') and hasattr(flow, 'xst')):
        for name in ['model', 'net', 'flow', 'fm', 'module']:
            if hasattr(model, name):
                cand = getattr(model, name)
                if hasattr(cand, 'prior') and hasattr(cand, 'xst'):
                    flow = cand
                    flow.to(device)
                    flow.eval()
                    break
    if not (hasattr(flow, 'prior') and hasattr(flow, 'xst')):
        raise AttributeError('Loaded model does not expose prior(...) and xst(...).')
    if not hasattr(flow, 'in_shape'):
        if hasattr(model, 'in_shape'):
            flow.in_shape = model.in_shape
        elif 'model' in cfg and 'in_shape' in cfg.model:
            flow.in_shape = tuple(cfg.model.in_shape)
        else:
            raise AttributeError('Flow model has no in_shape.')
    return flow, cfg


def get_data_dir(cfg):
    candidates = []
    try:
        candidates.append(Path(str(cfg.paths.data_dir)))
    except Exception:
        pass
    try:
        candidates.append(Path(str(cfg.data.data_dir)))
    except Exception:
        pass
    try:
        root = Path(str(cfg.paths.root_dir))
        candidates.append(root / 'data')
        candidates.append(root / 'data' / 'text8')
    except Exception:
        pass
    candidates += [Path('data/text8'), Path('data')]
    for p in candidates:
        p = p.expanduser()
        if p.exists():
            return p
    return candidates[0]


def _load_pickle_or_json(path: Path):
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_meta(data_dir):
    data_dir = Path(data_dir)
    candidates = []
    for root in [data_dir, data_dir / 'text8']:
        candidates += [root / 'meta.pkl', root / 'meta.json']
    candidates += list(data_dir.rglob('meta.pkl'))[:20] + list(data_dir.rglob('meta.json'))[:20]
    for p in candidates:
        if p.exists():
            meta = _load_pickle_or_json(p)
            if 'stoi' in meta and 'itos' in meta:
                if 'vocab_size' not in meta:
                    meta['vocab_size'] = len(meta['stoi'])
                return meta
    raise FileNotFoundError(f'Could not find Text8 meta.pkl/meta.json under {data_dir}. Put the original Semicat Text8 cache under data/text8 or use the original Semicat data preparation.')


def _read_token_file(path: Path):
    arr = None
    if path.suffix == '.npy':
        arr = np.load(path)
    else:
        # Text8 caches are commonly uint16/int16/int64 token arrays. Try the common choices.
        for dtype in [np.uint16, np.int16, np.int32, np.int64, np.uint8]:
            try:
                a = np.fromfile(path, dtype=dtype)
                if a.size > 1024:
                    arr = a.astype(np.int64)
                    break
            except Exception:
                pass
    if arr is None:
        raise ValueError(f'Could not read token file {path}')
    return arr.reshape(-1).astype(np.int64)


def build_bigram_scorer(data_dir, vocab_size: int, device: str):
    data_dir = Path(data_dir)
    for p in list(data_dir.rglob('bigram*.pt')) + list(data_dir.rglob('*bigram*.npy')):
        try:
            if p.suffix == '.pt':
                t = torch.load(p, map_location='cpu')
                if isinstance(t, dict):
                    t = next(v for v in t.values() if torch.is_tensor(v))
                return t.float().to(device)
            a = np.load(p)
            return torch.tensor(a, dtype=torch.float32, device=device)
        except Exception:
            continue

    token_files = []
    for name in ['train.bin', 'train.npy', 'text8.train.bin', 'train_ids.bin']:
        token_files += list(data_dir.rglob(name))
    if not token_files:
        raise FileNotFoundError(f'Could not find train token file under {data_dir} to build bigram scorer.')
    tokens = _read_token_file(token_files[0])
    tokens = tokens[(tokens >= 0) & (tokens < int(vocab_size))]
    counts = np.ones((vocab_size, vocab_size), dtype=np.float64)  # add-one smoothing
    if tokens.size >= 2:
        np.add.at(counts, (tokens[:-1], tokens[1:]), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return torch.tensor(np.log(probs), dtype=torch.float32, device=device)


def target_ids_from_word(word: str, stoi):
    ids = []
    for ch in word:
        if ch not in stoi:
            raise KeyError(f'Character {ch!r} not found in Text8 vocabulary.')
        ids.append(int(stoi[ch]))
    return ids


def norm_per_sample(x: torch.Tensor, eps: float = 1e-8):
    shape = [x.shape[0]] + [1] * (x.ndim - 1)
    return x.reshape(x.shape[0], -1).norm(dim=1).view(*shape).clamp_min(eps)


def endpoint_reward(endpoint, target_ids: Sequence[int], log_bigram, reward_mix: float = 0.05):
    p = torch.softmax(endpoint, dim=-1)
    logp = torch.log(p.clamp_min(1e-8))
    B, L, V = p.shape
    m = len(target_ids)
    vals = []
    for r in range(L - m + 1):
        s = 0.0
        for j, idx in enumerate(target_ids):
            s = s + logp[:, r + j, int(idx)]
        vals.append(s / m)
    logq = torch.stack(vals, dim=1)
    q = torch.exp(logq).clamp(0.0, 1.0 - 1e-6)
    P_event = (1.0 - torch.exp(torch.log1p(-q).sum(dim=1))).clamp(1e-8, 1.0 - 1e-6)
    R_event = torch.log(P_event)
    bigram = torch.einsum('blv,vw,blw->bl', p[:, :-1, :], log_bigram, p[:, 1:, :]).mean(dim=1)
    return R_event + float(reward_mix) * bigram


def sample_base_flowmap(model, n_samples: int, batch_size: int, nfe: int):
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'
    outs = []
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x = model.prior((b, *model.in_shape), device=device)
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        with torch.no_grad():
            for s, t in zip(ts[:-1], ts[1:]):
                x = model.xst(x, s.expand((b,)), t.expand((b,)))
        outs.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    return torch.cat(outs, dim=0)


def sample_best_of_k(model, n_samples, batch_size, nfe, k, target, itos, log_bigram, reward_mix):
    # Simple reproducible best-of-k using the hard occurrence count as the selector.
    outs = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        cand = sample_base_flowmap(model, b * k, batch_size, nfe).view(k, b, -1)
        scores = []
        for i in range(k):
            metrics = evaluate_tokens(cand[i], log_bigram, log_bigram.shape[0], target, itos)
            # Same score for the batch is sufficient for diagnostic best-of-k baseline here.
            scores.append(metrics.get('hit_rate', 0.0))
        best_i = int(np.argmax(scores))
        outs.append(cand[best_i])
        remaining -= b
    return torch.cat(outs, dim=0)


def decode_one(tok, itos):
    chars = []
    for x in tok:
        i = int(x)
        chars.append(str(itos[i] if not isinstance(itos, dict) else itos.get(i, itos.get(str(i), ''))))
    return ''.join(chars)


def count_occurrences(text: str, word: str):
    if not word:
        return 0
    count, start = 0, 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def hard_bigram_score(tokens: torch.Tensor, log_bigram: torch.Tensor):
    t = tokens.to(log_bigram.device).long()
    return float(log_bigram[t[:, :-1], t[:, 1:]].mean().item())


def distinct_n_for_text(text: str, n: int):
    grams = [text[i:i+n] for i in range(max(0, len(text)-n+1))]
    return len(set(grams)) / max(len(grams), 1)


def rep_ngram_rate_for_text(text: str, n: int):
    grams = [text[i:i+n] for i in range(max(0, len(text)-n+1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def evaluate_tokens(tokens, log_bigram, vocab_size, target, itos):
    texts = [decode_one(row.tolist(), itos) for row in tokens.cpu()]
    counts = [count_occurrences(t, target) for t in texts]
    hit = [c > 0 for c in counts]
    exact = [c == 1 for c in counts]
    oversat = [c > 1 for c in counts]
    return {
        'hit_rate': float(np.mean(hit)),
        'exact_once_rate': float(np.mean(exact)),
        'oversat_rate': float(np.mean(oversat)),
        'target_count_mean': float(np.mean(counts)),
        'hard_bigram': hard_bigram_score(tokens, log_bigram),
        'distinct2': float(np.mean([distinct_n_for_text(t, 2) for t in texts])),
        'distinct3': float(np.mean([distinct_n_for_text(t, 3) for t in texts])),
        'rep2': float(np.mean([rep_ngram_rate_for_text(t, 2) for t in texts])),
        'rep3': float(np.mean([rep_ngram_rate_for_text(t, 3) for t in texts])),
    }


def emit_row(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    f.flush()


def write_samples(path, tokens, itos, max_samples=64):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in tokens[:max_samples].cpu():
            f.write(decode_one(row.tolist(), itos) + '\n')
