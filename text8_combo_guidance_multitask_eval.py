#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined gap-aware few-step guidance evaluation on text8 for categorical/simplex flow maps.

This script is a follow-up to text8_all_simplex_guidance_eval.py.  It keeps the
same differentiable text8 lexical-event reward, but adds several FMRG-pain-point
methods and diagnostics:

Diagnostics recorded for every guided method:
  1) soft-hard reward gap: reward/event probability on endpoint simplex vs
     straight-through hard endpoint.
  2) semigroup/composition gap: Phi_{t->1}(Phi_{s->t}(x)) vs Phi_{s->1}(x).
  3) linearization gap: predicted first-order improvement <g,Delta> vs actual
     terminal reward improvement after the tentative update.

New methods:
  - fmrg:              original FMRG from the base script.
  - fmrg_sat:          saturation-aware FMRG scale from the base script.
  - sfmrg_conflict_adapt / orth variants from the base script.
  - hard_st:           hard-aware ST endpoint reward gradient.
  - hard_gap:          hard-aware objective penalizing soft-hard reward gap.
  - semigroup:         FMRG reward minus semigroup consistency penalty.
  - semigroup_orth:    FMRG + orthogonal semigroup correction.
  - self_verify:       FMRG update with predicted-vs-actual verification scale.
  - pareto_quality:    PCGrad-style reward/quality conflict correction.
  - gap_aware:         combines saturation scale + conflict simplex correction
                       + self-verification scale.  This is an exploratory
                       integrated method, not a claim of final optimality.

Put this file under:
  /root/autodl-tmp/semicat/scripts/text8_combo_guidance_eval.py
and keep this dependency in the same scripts directory:
  text8_all_simplex_guidance_eval.py
"""

import argparse
import csv
import importlib.util
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "text8_all_simplex_guidance_eval.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing dependency: {BASE_SCRIPT}\n"
        "Copy text8_all_simplex_guidance_eval.py into the scripts/ directory first."
    )

spec = importlib.util.spec_from_file_location("all_simplex", str(BASE_SCRIPT))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Reuse base utilities.
set_seed = base.set_seed
load_model = base.load_model
get_data_dir = base.get_data_dir
load_meta = base.load_meta
build_bigram_scorer = base.build_bigram_scorer
target_ids_from_word = base.target_ids_from_word
norm_per_sample = base.norm_per_sample
fmrg_weight = base.fmrg_weight

to_simplex = base.to_simplex
token_entropy = base.token_entropy
expected_bigram_score = base.expected_bigram_score
ORIGINAL_LEXICAL_EVENT_REWARD_P = base.lexical_event_reward_p
unit_per_sample = base.unit_per_sample
cosine_per_sample = base.cosine_per_sample
state_diagnostics = base.state_diagnostics
per_position_norm = base.per_position_norm
project_orthogonal_per_sample = base.project_orthogonal_per_sample

# Existing directions.
compute_fmrg_direction = base.compute_fmrg_direction
compute_fmtg_direction = base.compute_fmtg_direction
compute_smfg_direction = base.compute_smfg_direction
compute_smfg_eps_direction = base.compute_smfg_eps_direction
compute_fmrg_saturation_direction = base.compute_fmrg_saturation_direction
compute_sfmrg_orth_direction = base.compute_sfmrg_orth_direction
compute_sfmrg_conflict_direction = base.compute_sfmrg_conflict_direction
compute_tptr_direction = base.compute_tptr_direction

# Existing evaluation helpers.
decode_one = base.decode_one
count_occurrences = base.count_occurrences
distinct_n_for_text = base.distinct_n_for_text
rep_ngram_rate_for_text = base.rep_ngram_rate_for_text
hard_bigram_score = base.hard_bigram_score
evaluate_lexical_tokens = base.evaluate_lexical_tokens
parse_target_sets = base.parse_target_sets
ids_for_words = base.ids_for_words
append_jsonl = base.append_jsonl
write_csv_from_jsonl = base.write_csv_from_jsonl
write_sample_texts = base.write_sample_texts



# -----------------------------
# task-general reward/evaluation extensions
# -----------------------------
#
# Backward-compatible default:
#   --task lexical_or
# keeps the original lexical-event reward used by the existing Text8 table.
#
# New tasks:
#   --task multi_all     all listed target words should appear.
#   --task position      listed words should appear at specified start positions.
#   --task forbidden     listed words should not appear.
#   --task exact_count   each listed target word should appear exactly --exact_count times.
#
# Important: the imported base methods (FMRG, FMRG+Sat., simplex conflict, etc.)
# call base.lexical_event_reward_p internally.  We monkey-patch that symbol below
# so every method is guided by the same task reward, not only by post-hoc metrics.

CURRENT_TASK_CONFIG = {
    "task": "lexical_or",
    "position_specs_ids": [],
    "position_specs_words": [],
    "exact_count": 1,
    "multi_reduce": "sum",
    "count_weight": 2.0,
    "forbid_weight": 1.0,
}


def _soft_word_logq_and_q(p: torch.Tensor, target_ids: Sequence[int], eps: float = 1e-8):
    """Soft slot occurrence probabilities for one character-level word."""
    logp = torch.log(p.clamp_min(eps))
    B, L, V = p.shape
    m = len(target_ids)
    if m <= 0:
        raise ValueError("Empty target word.")
    if m > L:
        raise ValueError(f"Target length {m} exceeds sequence length {L}.")
    vals = []
    for r in range(L - m + 1):
        s = 0.0
        for j, idx in enumerate(target_ids):
            s = s + logp[:, r + j, int(idx)]
        vals.append(s / m)
    logq = torch.stack(vals, dim=1)
    q = torch.exp(logq).clamp(0.0, 1.0 - 1e-6)
    return logq, q


def _merge_diag_from_single_rewards(R: torch.Tensor, diags: List[Dict[str, torch.Tensor]], p: torch.Tensor, log_bigram: torch.Tensor):
    """Aggregate original single-word diagnostics into the keys expected by the sampler."""
    if not diags:
        z = torch.zeros((p.shape[0],), device=p.device, dtype=p.dtype)
        return {"reward": R.detach(), "P_event": z, "C": z, "winner_q": z, "nonwinner_q": z,
                "dup_gate": z, "pi_entropy": z, "p_entropy": token_entropy(p).mean(dim=1).detach(),
                "bigram_soft": expected_bigram_score(p, log_bigram).detach()}
    keys = ["C", "winner_q", "nonwinner_q", "dup_gate", "pi_entropy", "p_entropy", "bigram_soft"]
    out = {"reward": R.detach()}
    ps = torch.stack([d["P_event"] for d in diags], dim=0)
    # Treat P_event as soft task-success probability; for all-target tasks, success
    # requires all single-word events.
    out["P_event"] = ps.prod(dim=0).detach()
    for k in keys:
        vals = [d[k] for d in diags if k in d]
        if not vals:
            continue
        vals = torch.stack(vals, dim=0)
        if k in {"C", "nonwinner_q"}:
            out[k] = vals.sum(dim=0).detach()
        else:
            out[k] = vals.mean(dim=0).detach()
    return out


def lexical_event_reward_p(
    p: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    **reward_kwargs,
):
    """Task-general differentiable reward with the same signature as the original reward."""
    task = str(CURRENT_TASK_CONFIG.get("task", "lexical_or")).lower()
    eps = float(reward_kwargs.get("eps", 1e-8))
    reward_mix = float(reward_kwargs.get("reward_mix", 0.02))
    alpha_event = float(reward_kwargs.get("alpha_event", 1.0))
    rho_dup = float(reward_kwargs.get("rho_dup", 0.5))
    dup_gamma = float(reward_kwargs.get("dup_gamma", 1.0))

    if task in {"lexical", "lexical_or", "single_or", "or"}:
        return ORIGINAL_LEXICAL_EVENT_REWARD_P(p, target_id_lists, log_bigram, **reward_kwargs)

    if task in {"multi", "multi_all", "all"}:
        Rs, diags = [], []
        for ids in target_id_lists:
            Ri, di = ORIGINAL_LEXICAL_EVENT_REWARD_P(p, [ids], log_bigram, **reward_kwargs)
            Rs.append(Ri)
            diags.append(di)
        R_stack = torch.stack(Rs, dim=0)
        if str(CURRENT_TASK_CONFIG.get("multi_reduce", "sum")).lower() == "mean":
            R = R_stack.mean(dim=0)
        else:
            R = R_stack.sum(dim=0)
        return R, _merge_diag_from_single_rewards(R, diags, p, log_bigram)

    if task in {"forbid", "forbidden", "negative"}:
        all_q = []
        for ids in target_id_lists:
            _logq, q = _soft_word_logq_and_q(p, ids, eps=eps)
            all_q.append(q)
        q_cat = torch.cat(all_q, dim=1) if all_q else torch.zeros((p.shape[0], 1), device=p.device, dtype=p.dtype)
        log_no_forbid = torch.log1p(-q_cat.clamp(0.0, 1.0 - 1e-6)).sum(dim=1)
        p_no_forbid = torch.exp(log_no_forbid).clamp(eps, 1.0)
        C = q_cat.sum(dim=1)
        bigram = expected_bigram_score(p, log_bigram)
        R = float(CURRENT_TASK_CONFIG.get("forbid_weight", 1.0)) * log_no_forbid + reward_mix * bigram
        diag = {
            "reward": R.detach(),
            "P_event": p_no_forbid.detach(),  # here: task success = no forbidden word
            "C": C.detach(),
            "winner_q": p_no_forbid.detach(),
            "nonwinner_q": C.detach(),
            "dup_gate": (1.0 - p_no_forbid).detach(),
            "pi_entropy": torch.zeros_like(C).detach(),
            "p_entropy": token_entropy(p).mean(dim=1).detach(),
            "bigram_soft": bigram.detach(),
        }
        return R, diag

    if task in {"exact", "exact_count", "count"}:
        k = float(CURRENT_TASK_CONFIG.get("exact_count", 1))
        count_weight = float(CURRENT_TASK_CONFIG.get("count_weight", 2.0))
        Rs, Cs, Ps, bigrams = [], [], [], []
        bigram = expected_bigram_score(p, log_bigram)
        for ids in target_id_lists:
            _logq, q = _soft_word_logq_and_q(p, ids, eps=eps)
            C_i = q.sum(dim=1)
            log_no = torch.log1p(-q.clamp(0.0, 1.0 - 1e-6)).sum(dim=1)
            P_i = (1.0 - torch.exp(log_no)).clamp(eps, 1.0 - 1e-6)
            if k <= 0:
                event_term = alpha_event * log_no
            else:
                event_term = alpha_event * torch.log(P_i.clamp_min(eps))
            count_term = -count_weight * (C_i - k).pow(2)
            Rs.append(event_term + count_term)
            Cs.append(C_i)
            # Smooth success proxy used only for diagnostics/saturation.
            Ps.append(torch.exp(-(C_i - k).pow(2)).clamp(eps, 1.0))
            bigrams.append(bigram)
        R = torch.stack(Rs, dim=0).sum(dim=0) + reward_mix * bigram
        C = torch.stack(Cs, dim=0).sum(dim=0)
        P_success = torch.stack(Ps, dim=0).prod(dim=0)
        diag = {
            "reward": R.detach(),
            "P_event": P_success.detach(),
            "C": C.detach(),
            "winner_q": P_success.detach(),
            "nonwinner_q": C.detach(),
            "dup_gate": torch.relu(C - k).detach(),
            "pi_entropy": torch.zeros_like(C).detach(),
            "p_entropy": token_entropy(p).mean(dim=1).detach(),
            "bigram_soft": bigram.detach(),
        }
        return R, diag

    if task in {"position", "pos", "specified_position"}:
        specs = CURRENT_TASK_CONFIG.get("position_specs_ids", [])
        if not specs:
            specs = [(list(ids), 0) for ids in target_id_lists]
        R_parts, q_pos_list, total_C, nonpos_total = [], [], [], []
        for ids, pos in specs:
            logq, q = _soft_word_logq_and_q(p, ids, eps=eps)
            max_pos = q.shape[1] - 1
            if int(pos) < 0 or int(pos) > max_pos:
                raise ValueError(f"Position {pos} is outside valid range [0,{max_pos}] for target length {len(ids)}.")
            q_pos = q[:, int(pos)].clamp(eps, 1.0 - 1e-6)
            logq_pos = logq[:, int(pos)]
            C_i = q.sum(dim=1)
            nonpos = (C_i - q_pos).clamp_min(0.0)
            R_parts.append(alpha_event * logq_pos - rho_dup * q_pos.detach().pow(dup_gamma) * nonpos)
            q_pos_list.append(q_pos)
            total_C.append(C_i)
            nonpos_total.append(nonpos)
        bigram = expected_bigram_score(p, log_bigram)
        R = torch.stack(R_parts, dim=0).sum(dim=0) + reward_mix * bigram
        P_success = torch.stack(q_pos_list, dim=0).prod(dim=0)
        C = torch.stack(total_C, dim=0).sum(dim=0)
        nonpos = torch.stack(nonpos_total, dim=0).sum(dim=0)
        diag = {
            "reward": R.detach(),
            "P_event": P_success.detach(),
            "C": C.detach(),
            "winner_q": P_success.detach(),
            "nonwinner_q": nonpos.detach(),
            "dup_gate": P_success.detach().pow(dup_gamma),
            "pi_entropy": torch.zeros_like(C).detach(),
            "p_entropy": token_entropy(p).mean(dim=1).detach(),
            "bigram_soft": bigram.detach(),
        }
        return R, diag

    raise ValueError(f"Unknown task: {task}")


# Monkey-patch the imported base module so existing method implementations use
# the active task reward.
base.lexical_event_reward_p = lexical_event_reward_p


def _parse_position_specs(spec: str) -> Dict[str, List[Tuple[str, int]]]:
    """Parse 'set_name:word@pos|other@pos;set2:word@pos'."""
    out: Dict[str, List[Tuple[str, int]]] = {}
    spec = (spec or "").strip()
    if not spec:
        return out
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Bad position_specs block: {block}")
        name, rest = block.split(":", 1)
        pairs = []
        for item in rest.split("|"):
            item = item.strip()
            if not item:
                continue
            if "@" not in item:
                raise ValueError(f"Bad position item '{item}', expected word@pos.")
            w, pos = item.rsplit("@", 1)
            pairs.append((w.strip().lower(), int(pos)))
        out[name.strip()] = pairs
    return out


def _strip_position_word(w: str) -> str:
    w = w.strip().lower()
    return w.rsplit("@", 1)[0] if "@" in w else w


def _active_words_and_positions(task: str, target_name: str, target_words: Sequence[str], position_specs_raw: str):
    task = task.lower()
    clean_words = [_strip_position_word(w) for w in target_words]
    if task not in {"position", "pos", "specified_position"}:
        return clean_words, []
    spec_map = _parse_position_specs(position_specs_raw)
    if target_name in spec_map:
        pairs = spec_map[target_name]
    else:
        # Fallback: allow --target_sets "pos0:award@0|city@8".
        pairs = []
        for w in target_words:
            ww = w.strip().lower()
            if "@" in ww:
                name, pos = ww.rsplit("@", 1)
                pairs.append((name, int(pos)))
        if not pairs:
            # Last-resort smoke-test default.
            pairs = [(w, 0) for w in clean_words]
    return [w for w, _ in pairs], pairs


def _word_counts_for_text(text: str, words: Sequence[str]) -> Dict[str, int]:
    return {w: text.count(w) for w in words}


def evaluate_task_tokens(
    tokens: torch.Tensor,
    target_name: str,
    target_words: Sequence[str],
    itos,
    log_bigram: torch.Tensor,
) -> Tuple[Dict[str, float], List[str]]:
    """Hard decoded metrics aligned with the active task."""
    task = str(CURRENT_TASK_CONFIG.get("task", "lexical_or")).lower()
    texts = [decode_one(row.tolist(), itos) for row in tokens.cpu()]
    words = list(target_words)
    per_word_counts = [_word_counts_for_text(t, words) for t in texts]
    total_counts = np.array([sum(d.values()) for d in per_word_counts], dtype=np.float64)
    lens = np.array([len(t) for t in texts], dtype=np.float64)

    metrics = {
        "hard_bigram": hard_bigram_score(tokens, log_bigram),
        "distinct2": float(np.mean([distinct_n_for_text(t, 2) for t in texts])),
        "distinct3": float(np.mean([distinct_n_for_text(t, 3) for t in texts])),
        "rep2": float(np.mean([rep_ngram_rate_for_text(t, 2) for t in texts])),
        "rep3": float(np.mean([rep_ngram_rate_for_text(t, 3) for t in texts])),
        "uniq_text_rate": float(len(set(texts)) / max(1, len(texts))),
        "avg_len": float(lens.mean()),
    }

    if task in {"lexical", "lexical_or", "single_or", "or"}:
        base_metrics, _ = base.evaluate_lexical_tokens(tokens, words, itos, log_bigram)
        base_metrics.update({"task_success_rate": base_metrics["hit_rate"], "task_type": task})
        return base_metrics, texts

    if task in {"multi", "multi_all", "all"}:
        all_hit = np.array([all(d[w] >= 1 for w in words) for d in per_word_counts], dtype=np.float64)
        all_exact1 = np.array([all(d[w] == 1 for w in words) for d in per_word_counts], dtype=np.float64)
        any_oversat = np.array([any(d[w] >= 2 for w in words) for d in per_word_counts], dtype=np.float64)
        metrics.update({
            "task_type": task,
            "task_success_rate": float(all_hit.mean()),
            "all_hit_rate": float(all_hit.mean()),
            "all_exact_once_rate": float(all_exact1.mean()),
            "any_oversat_rate": float(any_oversat.mean()),
            "target_count_mean": float(total_counts.mean()),
            "target_count_std": float(total_counts.std()),
        })
        for w in words:
            metrics[f"hit_{w}"] = float(np.mean([d[w] >= 1 for d in per_word_counts]))
            metrics[f"count_mean_{w}"] = float(np.mean([d[w] for d in per_word_counts]))
        return metrics, texts

    if task in {"forbid", "forbidden", "negative"}:
        violate = np.array([any(d[w] >= 1 for w in words) for d in per_word_counts], dtype=np.float64)
        metrics.update({
            "task_type": task,
            "task_success_rate": float((1.0 - violate).mean()),
            "forbidden_violation_rate": float(violate.mean()),
            "forbidden_count_mean": float(total_counts.mean()),
            "forbidden_count_std": float(total_counts.std()),
        })
        for w in words:
            metrics[f"violate_{w}"] = float(np.mean([d[w] >= 1 for d in per_word_counts]))
        return metrics, texts

    if task in {"exact", "exact_count", "count"}:
        k = int(CURRENT_TASK_CONFIG.get("exact_count", 1))
        exact_all = np.array([all(d[w] == k for w in words) for d in per_word_counts], dtype=np.float64)
        hit_all = np.array([all(d[w] >= 1 for w in words) for d in per_word_counts], dtype=np.float64)
        metrics.update({
            "task_type": task,
            "task_success_rate": float(exact_all.mean()),
            "exact_count": k,
            "exact_count_success_rate": float(exact_all.mean()),
            "all_hit_rate": float(hit_all.mean()),
            "target_count_mean": float(total_counts.mean()),
            "target_count_std": float(total_counts.std()),
        })
        for w in words:
            metrics[f"exact_count_{w}"] = float(np.mean([d[w] == k for d in per_word_counts]))
            metrics[f"count_mean_{w}"] = float(np.mean([d[w] for d in per_word_counts]))
        return metrics, texts

    if task in {"position", "pos", "specified_position"}:
        pairs = CURRENT_TASK_CONFIG.get("position_specs_words", [])
        if not pairs:
            pairs = [(w, 0) for w in words]
        ok = []
        for t in texts:
            cur = []
            for w, pos in pairs:
                pos = int(pos)
                cur.append(pos >= 0 and pos + len(w) <= len(t) and t[pos:pos + len(w)] == w)
            ok.append(all(cur))
        ok = np.array(ok, dtype=np.float64)
        metrics.update({
            "task_type": task,
            "task_success_rate": float(ok.mean()),
            "position_success_rate": float(ok.mean()),
            "target_count_mean": float(total_counts.mean()),
            "target_count_std": float(total_counts.std()),
        })
        for w, pos in pairs:
            key = f"pos_hit_{w}_at_{int(pos)}"
            metrics[key] = float(np.mean([t[int(pos):int(pos) + len(w)] == w if int(pos) + len(w) <= len(t) else False for t in texts]))
        return metrics, texts

    raise ValueError(f"Unknown task for evaluation: {task}")


# -----------------------------
# gap-aware helper utilities
# -----------------------------


def hard_st_onehot_from_p(p: torch.Tensor) -> torch.Tensor:
    """Straight-through argmax one-hot with gradients through p."""
    idx = p.argmax(dim=-1)
    oh = torch.nn.functional.one_hot(idx, num_classes=p.shape[-1]).to(dtype=p.dtype, device=p.device)
    return oh.detach() - p.detach() + p


def kl_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample mean-position KL(p || q)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean(dim=1)


def sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 0.5 * (kl_pq(p, q, eps=eps) + kl_pq(q, p, eps=eps))


def l1_pq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p - q).abs().sum(dim=-1).mean(dim=1)


def reward_endpoint(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
):
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    endpoint = model.xst(x, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)
    R, diag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)
    return endpoint, p, R, diag


def soft_hard_gap_diag(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
) -> Dict[str, float]:
    """Endpoint soft reward vs ST-hard reward at the same state."""
    with torch.enable_grad():
        x_det = x.detach()
        endpoint, p, R_soft, diag_soft = reward_endpoint(
            model, x_det, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
        )
        p_hard = hard_st_onehot_from_p(p)
        R_hard, diag_hard = lexical_event_reward_p(p_hard, target_id_lists, log_bigram, **reward_kwargs)
    return {
        "soft_reward": float(R_soft.detach().mean().item()),
        "hardst_reward": float(R_hard.detach().mean().item()),
        "soft_hard_reward_gap": float((R_soft.detach() - R_hard.detach()).mean().item()),
        "soft_P_event": float(diag_soft["P_event"].detach().mean().item()),
        "hardst_P_event": float(diag_hard["P_event"].detach().mean().item()),
        "soft_hard_P_event_gap": float((diag_soft["P_event"].detach() - diag_hard["P_event"].detach()).mean().item()),
        "soft_C": float(diag_soft["C"].detach().mean().item()),
        "hardst_C": float(diag_hard["C"].detach().mean().item()),
    }


def semigroup_gap_diag(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    t_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
) -> Dict[str, float]:
    """Compare direct endpoint Phi_s1(x) with composed endpoint Phi_t1(Phi_st(x))."""
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    with torch.no_grad():
        direct = model.xst(x.detach(), s_vec, one_vec)
        mid = model.xst(x.detach(), s_vec, t_vec)
        comp = model.xst(mid, t_vec, one_vec)
        p_d = to_simplex(direct, mode=p_mode)
        p_c = to_simplex(comp, mode=p_mode)
        R_d, dg_d = lexical_event_reward_p(p_d, target_id_lists, log_bigram, **reward_kwargs)
        R_c, dg_c = lexical_event_reward_p(p_c, target_id_lists, log_bigram, **reward_kwargs)
        return {
            "semigroup_symkl": float(sym_kl(p_d, p_c).mean().item()),
            "semigroup_l1": float(l1_pq(p_d, p_c).mean().item()),
            "semigroup_reward_gap": float((R_d - R_c).abs().mean().item()),
            "semigroup_P_event_gap": float((dg_d["P_event"] - dg_c["P_event"]).abs().mean().item()),
        }


def linearization_gap_diag(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    direction: torch.Tensor,
    update: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
) -> Dict[str, float]:
    """FMRG-style local linear prediction gap at the current step."""
    with torch.no_grad():
        _ep0, _p0, R0, _d0 = reward_endpoint(
            model, x.detach(), s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
        )
        _ep1, _p1, R1, _d1 = reward_endpoint(
            model, (x.detach() + update.detach()), s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
        )
        pred = (direction.detach().reshape(direction.shape[0], -1) * update.detach().reshape(update.shape[0], -1)).sum(dim=1)
        actual = R1 - R0
        gap = actual - pred
        denom = pred.abs().clamp_min(1e-8)
        return {
            "lin_pred_improve": float(pred.mean().item()),
            "lin_actual_improve": float(actual.mean().item()),
            "lin_gap_signed": float(gap.mean().item()),
            "lin_gap_abs": float(gap.abs().mean().item()),
            "lin_actual_over_pred": float((actual / denom).mean().item()),
            "lin_fail_rate": float((actual < 0).float().mean().item()),
        }


def quality_objective_p(p: torch.Tensor, log_bigram: torch.Tensor, entropy_weight: float = 0.02):
    """A generic differentiable text-quality surrogate: bigram fluency + entropy."""
    bigram = expected_bigram_score(p, log_bigram)
    ent = token_entropy(p).mean(dim=1)
    return bigram + float(entropy_weight) * ent, {"quality_bigram": bigram.detach(), "quality_entropy": ent.detach()}


# -----------------------------
# new gap-aware guidance directions
# -----------------------------


def compute_hard_st_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hardness-aware ST endpoint reward gradient."""
    x_req = x.detach().requires_grad_(True)
    endpoint, p, R_soft, diag_soft = reward_endpoint(
        model, x_req, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
    )
    p_hard = hard_st_onehot_from_p(p)
    R_hard, diag_hard = lexical_event_reward_p(p_hard, target_id_lists, log_bigram, **reward_kwargs)
    direction = torch.autograd.grad(R_hard.mean(), x_req, retain_graph=False, create_graph=False)[0]
    diag = {k: float(v.detach().mean().item()) for k, v in diag_hard.items()}
    diag["guidance_reward"] = float(R_hard.detach().mean().item())
    diag["soft_reward"] = float(R_soft.detach().mean().item())
    diag["hardst_reward"] = float(R_hard.detach().mean().item())
    diag["soft_hard_reward_gap"] = float((R_soft.detach() - R_hard.detach()).mean().item())
    diag["soft_P_event"] = float(diag_soft["P_event"].detach().mean().item())
    diag["hardst_P_event"] = float(diag_hard["P_event"].detach().mean().item())
    return direction.detach(), diag


def compute_hard_gap_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    gap_beta: float = 0.5,
    gap_mode: str = "soft_minus_hard",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Hard-aware objective with soft-hard gap penalty.

    Default objective:
      J = R_hard_ST - beta * relu(R_soft - R_hard_ST)
    This discourages soft surrogate reward that cannot survive hard decoding.
    """
    x_req = x.detach().requires_grad_(True)
    endpoint, p, R_soft, diag_soft = reward_endpoint(
        model, x_req, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
    )
    p_hard = hard_st_onehot_from_p(p)
    R_hard, diag_hard = lexical_event_reward_p(p_hard, target_id_lists, log_bigram, **reward_kwargs)
    if gap_mode == "abs":
        penalty = (R_soft - R_hard).abs()
    else:
        penalty = torch.relu(R_soft - R_hard)
    J = R_hard - float(gap_beta) * penalty
    direction = torch.autograd.grad(J.mean(), x_req, retain_graph=False, create_graph=False)[0]
    diag = {k: float(v.detach().mean().item()) for k, v in diag_hard.items()}
    diag["guidance_reward"] = float(J.detach().mean().item())
    diag["soft_reward"] = float(R_soft.detach().mean().item())
    diag["hardst_reward"] = float(R_hard.detach().mean().item())
    diag["hard_gap_penalty"] = float(penalty.detach().mean().item())
    diag["soft_hard_reward_gap"] = float((R_soft.detach() - R_hard.detach()).mean().item())
    diag["soft_P_event"] = float(diag_soft["P_event"].detach().mean().item())
    diag["hardst_P_event"] = float(diag_hard["P_event"].detach().mean().item())
    return direction.detach(), diag


def compute_semigroup_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    t_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    semi_beta: float = 0.1,
    semi_metric: str = "symkl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Semigroup-consistent guidance:
      J = U(Pi(Phi_s1(x))) - beta * D(Phi_s1(x), Phi_t1(Phi_st(x))).
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    x_req = x.detach().requires_grad_(True)
    direct = model.xst(x_req, s_vec, one_vec)
    mid = model.xst(x_req, s_vec, t_vec)
    comp = model.xst(mid, t_vec, one_vec)
    p_d = to_simplex(direct, mode=p_mode)
    p_c = to_simplex(comp, mode=p_mode)
    R, rdiag = lexical_event_reward_p(p_d, target_id_lists, log_bigram, **reward_kwargs)
    if semi_metric == "l1":
        gap = l1_pq(p_d, p_c)
    else:
        gap = sym_kl(p_d, p_c)
    J = R - float(semi_beta) * gap
    direction = torch.autograd.grad(J.mean(), x_req, retain_graph=False, create_graph=False)[0]
    diag = {k: float(v.detach().mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R.detach().mean().item())
    diag["semigroup_penalty"] = float(gap.detach().mean().item())
    return direction.detach(), diag


def compute_semigroup_orth_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    t_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    semi_beta: float = 0.2,
    semi_metric: str = "symkl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """FMRG reward direction plus orthogonal semigroup-consistency correction."""
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    x_req = x.detach().requires_grad_(True)
    direct = model.xst(x_req, s_vec, one_vec)
    mid = model.xst(x_req, s_vec, t_vec)
    comp = model.xst(mid, t_vec, one_vec)
    p_d = to_simplex(direct, mode=p_mode)
    p_c = to_simplex(comp, mode=p_mode)
    R, rdiag = lexical_event_reward_p(p_d, target_id_lists, log_bigram, **reward_kwargs)
    if semi_metric == "l1":
        gap = l1_pq(p_d, p_c)
    else:
        gap = sym_kl(p_d, p_c)
    g_R = torch.autograd.grad(R.mean(), x_req, retain_graph=True, create_graph=False)[0]
    g_semi = -torch.autograd.grad(gap.mean(), x_req, retain_graph=False, create_graph=False)[0]
    g_semi_perp = project_orthogonal_per_sample(g_semi, g_R)
    direction = unit_per_sample(g_R) + float(semi_beta) * unit_per_sample(g_semi_perp)
    diag = {k: float(v.detach().mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R.detach().mean().item())
    diag["semigroup_penalty"] = float(gap.detach().mean().item())
    diag["semigroup_reward_cos"] = float(cosine_per_sample(g_R.detach(), g_semi.detach()).mean().item())
    return direction.detach(), diag


def compute_self_verify_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    gamma: float,
    vel_norm: torch.Tensor,
    verify_temp: float = 0.05,
    verify_floor: float = 0.20,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    FMRG direction with an update scale based on actual-vs-predicted reward gain.

    The returned direction is still FMRG.  The sampler uses diag['_sample_scale']
    to shrink updates whose tentative actual reward improvement is poor.
    """
    direction, diag = compute_fmrg_direction(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
    )
    with torch.no_grad():
        update0 = float(gamma) * unit_per_sample(direction.detach()) * vel_norm
        lin = linearization_gap_diag(
            model, x, s_vec, direction, update0,
            target_id_lists, log_bigram, p_mode, reward_kwargs,
        )
        # scale increases when actual improvement is positive and close to prediction.
        actual = torch.tensor(lin["lin_actual_improve"], device=x.device)
        # Per-batch scalar is enough for stability and cheapness.
        scale_scalar = float(verify_floor) + (1.0 - float(verify_floor)) * float(torch.sigmoid(actual / max(verify_temp, 1e-6)).item())
        scale = torch.full((x.shape[0],), scale_scalar, device=x.device)
    diag.update(lin)
    diag["self_verify_scale"] = float(scale.mean().item())
    diag["_sample_scale"] = scale
    return direction.detach(), diag


def compute_pareto_quality_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    quality_lambda: float = 0.2,
    quality_entropy: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PCGrad-style reward/quality conflict correction."""
    x_req = x.detach().requires_grad_(True)
    endpoint, p, R, rdiag = reward_endpoint(
        model, x_req, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
    )
    Q, qdiag = quality_objective_p(p, log_bigram, entropy_weight=quality_entropy)
    g_R = torch.autograd.grad(R.mean(), x_req, retain_graph=True, create_graph=False)[0]
    g_Q = torch.autograd.grad(Q.mean(), x_req, retain_graph=False, create_graph=False)[0]
    cos = cosine_per_sample(g_R.detach(), g_Q.detach())
    B = x.shape[0]
    # If reward and quality conflict, remove the component of reward gradient that goes against quality.
    g_R_corr = g_R.clone()
    mask = (cos < 0).view(B, *([1] * (x.ndim - 1)))
    g_R_perp = project_orthogonal_per_sample(g_R, g_Q)
    g_R_corr = torch.where(mask, g_R_perp, g_R)
    direction = unit_per_sample(g_R_corr) + float(quality_lambda) * unit_per_sample(g_Q)
    diag = {k: float(v.detach().mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R.detach().mean().item())
    diag["quality_obj"] = float(Q.detach().mean().item())
    diag["quality_bigram"] = float(qdiag["quality_bigram"].mean().item())
    diag["quality_entropy"] = float(qdiag["quality_entropy"].mean().item())
    diag["reward_quality_cos"] = float(cos.mean().item())
    diag["reward_quality_conflict_rate"] = float((cos < 0).float().mean().item())
    return direction.detach(), diag


def compute_gap_aware_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    gamma: float,
    vel_norm: torch.Tensor,
    mirror_eta: float = 0.3,
    mix_lambda: float = 0.4,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
    verify_temp: float = 0.05,
    verify_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Integrated exploratory method:
      conflict-adaptive simplex correction + self-verification scale.
    It uses the same task reward as FMRG.
    """
    direction, diag = compute_sfmrg_conflict_direction(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        mirror_eta=mirror_eta, mix_lambda=mix_lambda, mirror_eps=0.0,
        adaptive=True, sat_tau=sat_tau, sat_kappa=sat_kappa,
    )
    with torch.no_grad():
        update0 = float(gamma) * unit_per_sample(direction.detach()) * vel_norm
        lin = linearization_gap_diag(
            model, x, s_vec, direction, update0,
            target_id_lists, log_bigram, p_mode, reward_kwargs,
        )
        actual = torch.tensor(lin["lin_actual_improve"], device=x.device)
        scale_scalar = float(verify_floor) + (1.0 - float(verify_floor)) * float(torch.sigmoid(actual / max(verify_temp, 1e-6)).item())
        scale = torch.full((x.shape[0],), scale_scalar, device=x.device)
    diag.update(lin)
    diag["gap_aware_verify_scale"] = float(scale.mean().item())
    diag["_sample_scale"] = scale
    return direction.detach(), diag



def saturation_scale_from_diag(diag: Dict[str, float], sat_power: float = 1.0, sat_floor: float = 0.25) -> float:
    """Scalar fallback saturation scale from diagnostic P_event."""
    sat = float(diag.get("P_event", 0.0))
    sat = max(0.0, min(1.0, sat))
    return float(sat_floor) + (1.0 - float(sat_floor)) * ((1.0 - sat) ** float(sat_power))


def compute_sat_orth_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.3,
    mix_lambda: float = 0.4,
    mirror_grad_clip: float = 10.0,
    mirror_eps: float = 0.0,
    adaptive: bool = False,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Saturation-aware orthogonal simplex correction.
    Direction: FMRG reward ascent + orthogonal mirror correction.
    Update is scaled by (1 - satisfaction)^alpha, with a floor.
    """
    direction, diag = compute_sfmrg_orth_direction(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        mirror_eta=mirror_eta, mix_lambda=mix_lambda,
        mirror_grad_clip=mirror_grad_clip, mirror_eps=mirror_eps,
        adaptive=adaptive, sat_tau=sat_tau, sat_kappa=sat_kappa,
    )
    # Use the same event satisfaction used by the reward as the saturation signal.
    sat = float(diag.get("P_event", 0.0))
    scale_scalar = float(sat_floor) + (1.0 - float(sat_floor)) * ((1.0 - max(0.0, min(1.0, sat))) ** float(sat_power))
    scale = torch.full((x.shape[0],), scale_scalar, device=x.device)
    diag["sat_step_scale"] = float(scale_scalar)
    diag["_sample_scale"] = scale
    return direction.detach(), diag


def compute_sat_conflict_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.3,
    mix_lambda: float = 0.4,
    mirror_grad_clip: float = 10.0,
    mirror_eps: float = 0.0,
    adaptive: bool = False,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Saturation-aware conflict-gated simplex correction.
    This combines the previously strong conflict/orthogonal operator with the
    strongest gap-aware signal from the smoke: saturation-aware update scaling.
    """
    direction, diag = compute_sfmrg_conflict_direction(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        mirror_eta=mirror_eta, mix_lambda=mix_lambda,
        mirror_grad_clip=mirror_grad_clip, mirror_eps=mirror_eps,
        adaptive=adaptive, sat_tau=sat_tau, sat_kappa=sat_kappa,
    )
    sat = float(diag.get("P_event", 0.0))
    scale_scalar = float(sat_floor) + (1.0 - float(sat_floor)) * ((1.0 - max(0.0, min(1.0, sat))) ** float(sat_power))
    scale = torch.full((x.shape[0],), scale_scalar, device=x.device)
    diag["sat_step_scale"] = float(scale_scalar)
    diag["_sample_scale"] = scale
    return direction.detach(), diag


def compute_sat_semigroup_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    t_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    semi_beta: float = 0.1,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
    orth: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Semigroup-consistent guidance plus saturation-aware update scaling."""
    if orth:
        direction, diag = compute_semigroup_orth_direction(
            model, x, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
            semi_beta=semi_beta,
        )
    else:
        direction, diag = compute_semigroup_direction(
            model, x, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
            semi_beta=semi_beta,
        )
    sat = float(diag.get("P_event", 0.0))
    scale_scalar = float(sat_floor) + (1.0 - float(sat_floor)) * ((1.0 - max(0.0, min(1.0, sat))) ** float(sat_power))
    scale = torch.full((x.shape[0],), scale_scalar, device=x.device)
    diag["sat_step_scale"] = float(scale_scalar)
    diag["_sample_scale"] = scale
    return direction.detach(), diag


def compute_sat_pareto_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    quality_lambda: float = 0.2,
    quality_entropy: float = 0.02,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reward-quality Pareto correction plus saturation-aware update scaling."""
    direction, diag = compute_pareto_quality_direction(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        quality_lambda=quality_lambda, quality_entropy=quality_entropy,
    )
    sat = float(diag.get("P_event", 0.0))
    scale_scalar = float(sat_floor) + (1.0 - float(sat_floor)) * ((1.0 - max(0.0, min(1.0, sat))) ** float(sat_power))
    scale = torch.full((x.shape[0],), scale_scalar, device=x.device)
    diag["sat_step_scale"] = float(scale_scalar)
    diag["_sample_scale"] = scale
    return direction.detach(), diag


# -----------------------------
# sampler with diagnostics
# -----------------------------


def reduce_diag(diag_lists: Dict[str, List[float]], prefix: str) -> Dict[str, float]:
    out = {}
    for k, vals in diag_lists.items():
        if len(vals) == 0:
            continue
        out[f"{prefix}_{k}_mean"] = float(np.mean(vals))
    return out


def sample_guided_gap(
    model,
    n_samples: int,
    batch_size: int,
    nfe: int,
    method: str,
    step_size: float,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    device: str,
    schedule: str,
    early_stop: float,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mirror_grad_clip: float = 10.0,
    mix_lambda: float = 0.2,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
    trust_beta: float = 0.1,
    trust_entropy: float = 0.0,
    mirror_eps: float = 0.02,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
    gap_beta: float = 0.5,
    semi_beta: float = 0.1,
    quality_lambda: float = 0.2,
    quality_entropy: float = 0.02,
    verify_temp: float = 0.05,
    verify_floor: float = 0.25,
    diagnose: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    method = method.lower()
    valid_methods = {
        "base", "fmtg", "fmrg", "fmrg_sat", "smfg", "sfmrg_conflict_adapt", "sfmrg_orth", "sfmrg_orth_eps",
        "sat_orth", "sat_orth_eps", "sat_orth_adapt", "sat_conflict", "sat_conflict_eps",
        "sat_conflict_adapt", "sat_conflict_eps_adapt", "sat_semigroup", "sat_semigroup_orth", "sat_pareto_quality",
        "hard_st", "hard_gap", "semigroup", "semigroup_orth", "self_verify", "pareto_quality", "gap_aware",
    }
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")

    outs = []
    left = n_samples
    diag_keys = [
        "guidance_reward", "P_event", "C", "winner_q", "nonwinner_q", "dup_gate", "pi_entropy", "p_entropy", "bigram_soft",
        "direction_norm", "update_norm", "vel_norm", "update_over_vel", "gamma", "state_sum_error", "state_negative_mass",
        "state_above_one_mass", "state_min", "state_max", "soft_reward", "hardst_reward", "soft_hard_reward_gap",
        "soft_P_event", "hardst_P_event", "soft_hard_P_event_gap", "soft_C", "hardst_C", "semigroup_symkl",
        "semigroup_l1", "semigroup_reward_gap", "semigroup_P_event_gap", "lin_pred_improve", "lin_actual_improve",
        "lin_gap_signed", "lin_gap_abs", "lin_actual_over_pred", "lin_fail_rate", "hard_gap_penalty", "semigroup_penalty",
        "semigroup_reward_cos", "self_verify_scale", "quality_obj", "quality_bigram", "quality_entropy", "reward_quality_cos",
        "reward_quality_conflict_rate", "mix_lambda", "conflict_rate", "reward_mirror_cos", "mirror_kl", "mirror_delta_l1",
        "mirror_target_entropy", "orth_cos_after", "orth_perp_norm_ratio", "gap_aware_verify_scale", "sat_step_scale",
    ]
    diag_lists: Dict[str, List[float]] = {k: [] for k in diag_keys}

    while left > 0:
        b = min(batch_size, left)
        x = model.prior((b, *model.in_shape), device=device)
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)

        for s, t in zip(ts[:-1], ts[1:]):
            s_float = float(s.item())
            t_float = float(t.item())
            dt = max(1e-6, t_float - s_float)
            s_vec = s.expand((b,))
            t_vec = t.expand((b,))

            with torch.no_grad():
                base_next = model.xst(x.detach(), s_vec, t_vec)
                vel = (base_next - x.detach()) / dt
                vel_norm = norm_per_sample(vel).detach().clamp_min(1e-8)

            if method != "base" and s_float < early_stop:
                gamma = fmrg_weight(step_size, dt, s_float, schedule)

                if method == "fmtg":
                    direction, d = compute_fmtg_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs)
                elif method == "fmrg":
                    direction, d = compute_fmrg_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs)
                elif method == "fmrg_sat":
                    direction, d = compute_fmrg_saturation_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, sat_power=sat_power, sat_floor=sat_floor)
                elif method == "smfg":
                    direction, d = compute_smfg_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, mirror_eta=mirror_eta, mirror_grad_clip=mirror_grad_clip)
                elif method == "sfmrg_conflict_adapt":
                    direction, d = compute_sfmrg_conflict_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, mirror_eta=mirror_eta, mix_lambda=mix_lambda, mirror_grad_clip=mirror_grad_clip, mirror_eps=0.0, adaptive=True, sat_tau=sat_tau, sat_kappa=sat_kappa)
                elif method in {"sfmrg_orth", "sfmrg_orth_eps"}:
                    direction, d = compute_sfmrg_orth_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, mirror_eta=mirror_eta, mix_lambda=mix_lambda, mirror_grad_clip=mirror_grad_clip, mirror_eps=(mirror_eps if method.endswith("eps") else 0.0), adaptive=False, sat_tau=sat_tau, sat_kappa=sat_kappa)
                elif method in {"sat_orth", "sat_orth_eps", "sat_orth_adapt"}:
                    direction, d = compute_sat_orth_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, mirror_eta=mirror_eta, mix_lambda=mix_lambda, mirror_grad_clip=mirror_grad_clip, mirror_eps=(mirror_eps if method == "sat_orth_eps" else 0.0), adaptive=(method == "sat_orth_adapt"), sat_tau=sat_tau, sat_kappa=sat_kappa, sat_power=sat_power, sat_floor=sat_floor)
                elif method in {"sat_conflict", "sat_conflict_eps", "sat_conflict_adapt", "sat_conflict_eps_adapt"}:
                    direction, d = compute_sat_conflict_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, mirror_eta=mirror_eta, mix_lambda=mix_lambda, mirror_grad_clip=mirror_grad_clip, mirror_eps=(mirror_eps if "eps" in method else 0.0), adaptive=("adapt" in method), sat_tau=sat_tau, sat_kappa=sat_kappa, sat_power=sat_power, sat_floor=sat_floor)
                elif method == "sat_semigroup":
                    direction, d = compute_sat_semigroup_direction(model, x, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, semi_beta=semi_beta, sat_power=sat_power, sat_floor=sat_floor, orth=False)
                elif method == "sat_semigroup_orth":
                    direction, d = compute_sat_semigroup_direction(model, x, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, semi_beta=semi_beta, sat_power=sat_power, sat_floor=sat_floor, orth=True)
                elif method == "sat_pareto_quality":
                    direction, d = compute_sat_pareto_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, quality_lambda=quality_lambda, quality_entropy=quality_entropy, sat_power=sat_power, sat_floor=sat_floor)
                elif method == "hard_st":
                    direction, d = compute_hard_st_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs)
                elif method == "hard_gap":
                    direction, d = compute_hard_gap_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, gap_beta=gap_beta)
                elif method == "semigroup":
                    direction, d = compute_semigroup_direction(model, x, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, semi_beta=semi_beta)
                elif method == "semigroup_orth":
                    direction, d = compute_semigroup_orth_direction(model, x, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, semi_beta=semi_beta)
                elif method == "self_verify":
                    direction, d = compute_self_verify_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, gamma=gamma, vel_norm=vel_norm, verify_temp=verify_temp, verify_floor=verify_floor)
                elif method in {"pareto_quality", "sat_pareto_quality"}:
                    direction, d = compute_pareto_quality_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, quality_lambda=quality_lambda, quality_entropy=quality_entropy)
                elif method == "gap_aware":
                    direction, d = compute_gap_aware_direction(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs, gamma=gamma, vel_norm=vel_norm, mirror_eta=mirror_eta, mix_lambda=mix_lambda, sat_tau=sat_tau, sat_kappa=sat_kappa, verify_temp=verify_temp, verify_floor=verify_floor)
                else:
                    raise ValueError(method)

                sample_scale = d.pop("_sample_scale", None) if isinstance(d, dict) else None

                with torch.no_grad():
                    d_norm = norm_per_sample(direction).detach().clamp_min(1e-8)
                    update = gamma * direction.detach() / d_norm * vel_norm
                    if sample_scale is not None:
                        scale_view = sample_scale.to(update.device).view(b, *([1] * (update.ndim - 1)))
                        update = update * scale_view

                    # Gap diagnostics are computed for the update that will actually be applied.
                    if diagnose:
                        try:
                            sh = soft_hard_gap_diag(model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs)
                            for k, v in sh.items():
                                if k in diag_lists:
                                    diag_lists[k].append(float(v))
                        except Exception:
                            pass
                        try:
                            lin = linearization_gap_diag(model, x, s_vec, direction, update, target_id_lists, log_bigram, p_mode, reward_kwargs)
                            for k, v in lin.items():
                                if k in diag_lists:
                                    diag_lists[k].append(float(v))
                        except Exception:
                            pass
                        try:
                            # Diagnose semigroup after the guided perturbation, before the sampler advances.
                            sg = semigroup_gap_diag(model, x.detach() + update, s_vec, t_vec, target_id_lists, log_bigram, p_mode, reward_kwargs)
                            for k, v in sg.items():
                                if k in diag_lists:
                                    diag_lists[k].append(float(v))
                        except Exception:
                            pass

                    x = x.detach() + update
                    diag_lists["direction_norm"].append(float(d_norm.mean().item()))
                    diag_lists["update_norm"].append(float(norm_per_sample(update).mean().item()))
                    diag_lists["vel_norm"].append(float(vel_norm.mean().item()))
                    diag_lists["update_over_vel"].append(float((norm_per_sample(update) / vel_norm).mean().item()))
                    diag_lists["gamma"].append(float(gamma))
                    for k, v in d.items():
                        if k in diag_lists:
                            diag_lists[k].append(float(v))
                    sd = state_diagnostics(x)
                    for k, v in sd.items():
                        if k in diag_lists:
                            diag_lists[k].append(float(v))
            else:
                diag_lists["gamma"].append(0.0)

            with torch.no_grad():
                x = model.xst(x.detach(), s_vec, t_vec)

        # final endpoint soft diagnostics before argmax
        if diagnose:
            try:
                with torch.no_grad():
                    p_final = to_simplex(x.detach(), mode=p_mode)
                    Rf, dgf = lexical_event_reward_p(p_final, target_id_lists, log_bigram, **reward_kwargs)
                    pf_hard = hard_st_onehot_from_p(p_final)
                    Rh, dgh = lexical_event_reward_p(pf_hard, target_id_lists, log_bigram, **reward_kwargs)
                    diag_lists.setdefault("final_soft_reward", []).append(float(Rf.mean().item()))
                    diag_lists.setdefault("final_hardst_reward", []).append(float(Rh.mean().item()))
                    diag_lists.setdefault("final_soft_hard_reward_gap", []).append(float((Rf - Rh).mean().item()))
                    diag_lists.setdefault("final_soft_P_event", []).append(float(dgf["P_event"].mean().item()))
                    diag_lists.setdefault("final_hardst_P_event", []).append(float(dgh["P_event"].mean().item()))
                    diag_lists.setdefault("final_soft_C", []).append(float(dgf["C"].mean().item()))
            except Exception:
                pass
        outs.append(x.argmax(dim=-1).detach().cpu())
        left -= b

    return torch.cat(outs, dim=0), reduce_diag(diag_lists, prefix=method)


# -----------------------------
# IO and main
# -----------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--nfes", default="1,2,4,8")
    p.add_argument("--methods", default="fmrg,fmrg_sat,sfmrg_orth,sfmrg_conflict_adapt,sat_orth,sat_conflict,sat_conflict_adapt,sat_semigroup,sat_pareto_quality,semigroup,pareto_quality,gap_aware")
    p.add_argument("--target_sets", default="award:award;city:city|cities|town;game:game|team|player|match;music:music|song|album|band;science:science|research|computer|system")
    p.add_argument("--task", default="lexical_or",
                   choices=["lexical_or", "multi_all", "position", "forbidden", "exact_count"],
                   help="Task reward/evaluation type. lexical_or reproduces the original table.")
    p.add_argument("--position_specs", default="",
                   help="For --task position: 'set_name:word@pos|other@pos;set2:word@pos'.")
    p.add_argument("--exact_count", type=int, default=1,
                   help="For --task exact_count: required hard count for each target word.")
    p.add_argument("--multi_reduce", default="sum", choices=["sum", "mean"],
                   help="For --task multi_all: combine per-word rewards by sum or mean.")
    p.add_argument("--count_weight", type=float, default=2.0,
                   help="For --task exact_count: soft count penalty weight.")
    p.add_argument("--forbid_weight", type=float, default=1.0,
                   help="For --task forbidden: no-forbidden reward weight.")
    p.add_argument("--step_sizes", default="1.0,1.25,1.5")
    p.add_argument("--mirror_etas", default="0.1,0.3,0.5")
    p.add_argument("--mix_lambdas", default="0.2,0.4,0.6")
    p.add_argument("--gap_betas", default="0.25,0.5,1.0")
    p.add_argument("--semi_betas", default="0.05,0.1,0.2")
    p.add_argument("--quality_lambdas", default="0.1,0.2,0.4")
    p.add_argument("--mirror_grad_clip", type=float, default=10.0)
    p.add_argument("--mirror_eps", type=float, default=0.02)
    p.add_argument("--sat_tau", type=float, default=0.30)
    p.add_argument("--sat_kappa", type=float, default=0.10)
    p.add_argument("--sat_power", type=float, default=1.0)
    p.add_argument("--sat_floor", type=float, default=0.25)
    p.add_argument("--quality_entropy", type=float, default=0.02)
    p.add_argument("--verify_temp", type=float, default=0.05)
    p.add_argument("--verify_floor", type=float, default=0.25)
    p.add_argument("--early_stops", default="1.0")
    p.add_argument("--schedule", default="paper", choices=["paper", "dt", "constant"])
    p.add_argument("--p_mode", default="auto", choices=["auto", "softmax", "renorm"])
    p.add_argument("--disable_gap_diagnostics", action="store_true")

    # Same differentiable text8 reward hyperparameters for all methods.
    p.add_argument("--tau_slot", type=float, default=0.2)
    p.add_argument("--alpha_event", type=float, default=1.0)
    p.add_argument("--alpha_slot", type=float, default=0.25)
    p.add_argument("--rho_dup", type=float, default=0.5)
    p.add_argument("--dup_gamma", type=float, default=1.0)
    p.add_argument("--boundary_alpha", type=float, default=0.05)
    p.add_argument("--reward_mix", type=float, default=0.02)

    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--paired_eval", action="store_true", help="Reset RNG per target/NFE/step so all methods share the same initial prior samples.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_samples", action="store_true")
    p.add_argument("--max_saved_samples", type=int, default=64)
    args = p.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, cfg = load_model(args.run_dir, args.ckpt, device)
    model.eval()

    data_dir = get_data_dir(cfg)
    meta = load_meta(data_dir)
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = int(meta["vocab_size"])
    log_bigram = build_bigram_scorer(data_dir, vocab_size, device)

    nfes = [int(x) for x in args.nfes.split(",") if x.strip()]
    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    step_sizes = [float(x) for x in args.step_sizes.split(",") if x.strip()]
    mirror_etas = [float(x) for x in args.mirror_etas.split(",") if x.strip()]
    mix_lambdas = [float(x) for x in args.mix_lambdas.split(",") if x.strip()]
    gap_betas = [float(x) for x in args.gap_betas.split(",") if x.strip()]
    semi_betas = [float(x) for x in args.semi_betas.split(",") if x.strip()]
    quality_lambdas = [float(x) for x in args.quality_lambdas.split(",") if x.strip()]
    early_stops = [float(x) for x in args.early_stops.split(",") if x.strip()]
    target_sets = parse_target_sets(args.target_sets)
    CURRENT_TASK_CONFIG.update({
        "task": args.task,
        "exact_count": int(args.exact_count),
        "multi_reduce": args.multi_reduce,
        "count_weight": float(args.count_weight),
        "forbid_weight": float(args.forbid_weight),
    })

    valid = {"base", "fmtg", "fmrg", "fmrg_sat", "smfg", "sfmrg_conflict_adapt", "sfmrg_orth", "sfmrg_orth_eps",
             "sat_orth", "sat_orth_eps", "sat_orth_adapt", "sat_conflict", "sat_conflict_eps",
             "sat_conflict_adapt", "sat_conflict_eps_adapt", "sat_semigroup", "sat_semigroup_orth", "sat_pareto_quality",
             "hard_st", "hard_gap", "semigroup", "semigroup_orth", "self_verify", "pareto_quality", "gap_aware"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unknown method: {m}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    reward_kwargs = dict(
        tau_slot=args.tau_slot,
        alpha_event=args.alpha_event,
        alpha_slot=args.alpha_slot,
        rho_dup=args.rho_dup,
        dup_gamma=args.dup_gamma,
        boundary_alpha=args.boundary_alpha,
        reward_mix=args.reward_mix,
    )

    print("Gap-aware guidance eval. Same task reward for all methods; diagnostics enabled:", not args.disable_gap_diagnostics)
    print("Methods:", methods)
    print("Targets:", target_sets)
    print("NFEs:", nfes)

    for target_name, raw_target_words in target_sets:
        target_words, position_pairs = _active_words_and_positions(args.task, target_name, raw_target_words, args.position_specs)
        target_id_lists = ids_for_words(target_words, stoi)
        if args.task == "position":
            CURRENT_TASK_CONFIG["position_specs_words"] = position_pairs
            CURRENT_TASK_CONFIG["position_specs_ids"] = [
                (target_ids_from_word(w, stoi), int(pos)) for w, pos in position_pairs
            ]
        else:
            CURRENT_TASK_CONFIG["position_specs_words"] = []
            CURRENT_TASK_CONFIG["position_specs_ids"] = []
        print(f"\n===== target_set={target_name} task={args.task} words={target_words} =====")
        for nfe in nfes:
            print(f"\n--- NFE={nfe} ---")
            for early_stop in early_stops:
                for step_size in step_sizes:
                    for method in methods:
                        # Hyperparameter grid.
                        grid = []
                        if method in {"base", "fmtg", "fmrg", "fmrg_sat", "hard_st", "self_verify"}:
                            grid.append({})
                        elif method in {"smfg"}:
                            for eta in mirror_etas:
                                grid.append({"mirror_eta": eta})
                        elif method in {"sfmrg_conflict_adapt", "sfmrg_orth", "sfmrg_orth_eps", "gap_aware", "sat_orth", "sat_orth_eps", "sat_orth_adapt", "sat_conflict", "sat_conflict_eps", "sat_conflict_adapt", "sat_conflict_eps_adapt"}:
                            for eta in mirror_etas:
                                for lam in mix_lambdas:
                                    grid.append({"mirror_eta": eta, "mix_lambda": lam})
                        elif method == "hard_gap":
                            for gb in gap_betas:
                                grid.append({"gap_beta": gb})
                        elif method in {"semigroup", "semigroup_orth", "sat_semigroup", "sat_semigroup_orth"}:
                            for sb in semi_betas:
                                grid.append({"semi_beta": sb})
                        elif method in {"pareto_quality", "sat_pareto_quality"}:
                            for ql in quality_lambdas:
                                grid.append({"quality_lambda": ql})
                        else:
                            grid.append({})

                        for hp in grid:
                            mirror_eta = float(hp.get("mirror_eta", 0.0))
                            mix_lambda = float(hp.get("mix_lambda", 0.0))
                            gap_beta = float(hp.get("gap_beta", 0.0))
                            semi_beta = float(hp.get("semi_beta", 0.0))
                            quality_lambda = float(hp.get("quality_lambda", 0.0))
                            print(f"### {method} target={target_name} nfe={nfe} step={step_size} hp={hp}")
                            if args.paired_eval:
                                key = f"{args.seed}|{target_name}|{nfe}|{step_size}|{early_stop}"
                                row_seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)
                                set_seed(row_seed)
                            tok, diag = sample_guided_gap(
                                model=model,
                                n_samples=args.n_samples,
                                batch_size=args.batch_size,
                                nfe=nfe,
                                method=method,
                                step_size=step_size,
                                target_id_lists=target_id_lists,
                                log_bigram=log_bigram,
                                device=device,
                                schedule=args.schedule,
                                early_stop=early_stop,
                                p_mode=args.p_mode,
                                reward_kwargs=reward_kwargs,
                                mirror_eta=mirror_eta,
                                mirror_grad_clip=args.mirror_grad_clip,
                                mix_lambda=mix_lambda,
                                sat_tau=args.sat_tau,
                                sat_kappa=args.sat_kappa,
                                mirror_eps=args.mirror_eps,
                                sat_power=args.sat_power,
                                sat_floor=args.sat_floor,
                                gap_beta=gap_beta,
                                semi_beta=semi_beta,
                                quality_lambda=quality_lambda,
                                quality_entropy=args.quality_entropy,
                                verify_temp=args.verify_temp,
                                verify_floor=args.verify_floor,
                                diagnose=(not args.disable_gap_diagnostics),
                            )
                            metrics, texts = evaluate_task_tokens(tok, target_name, target_words, itos, log_bigram)
                            row = {
                                "target_set": target_name,
                                "target_words": "|".join(target_words),
                                "task": args.task,
                                "position_specs": args.position_specs,
                                "exact_count_arg": args.exact_count,
                                "multi_reduce": args.multi_reduce,
                                "count_weight": args.count_weight,
                                "forbid_weight": args.forbid_weight,
                                "method": method,
                                "nfe": nfe,
                                "step_size": step_size,
                                "early_stop": early_stop,
                                "mirror_eta": hp.get("mirror_eta"),
                                "mix_lambda": hp.get("mix_lambda"),
                                "gap_beta": hp.get("gap_beta"),
                                "semi_beta": hp.get("semi_beta"),
                                "quality_lambda": hp.get("quality_lambda"),
                                "sat_tau": args.sat_tau,
                                "sat_kappa": args.sat_kappa,
                                "mirror_eps_arg": args.mirror_eps,
                                "sat_power": args.sat_power,
                                "sat_floor": args.sat_floor,
                                "quality_entropy": args.quality_entropy,
                                "verify_temp": args.verify_temp,
                                "verify_floor": args.verify_floor,
                                "p_mode": args.p_mode,
                                "schedule": args.schedule,
                                "tau_slot": args.tau_slot,
                                "alpha_event": args.alpha_event,
                                "alpha_slot": args.alpha_slot,
                                "rho_dup": args.rho_dup,
                                "dup_gamma": args.dup_gamma,
                                "boundary_alpha": args.boundary_alpha,
                                "reward_mix": args.reward_mix,
                                "n_samples": args.n_samples,
                                "seed": args.seed,
                            }
                            row.update(metrics)
                            # Final hard-soft gap relative to hard hit is very interpretable.
                            if f"{method}_final_soft_P_event_mean" in diag:
                                row["final_soft_minus_hard_task_success"] = float(diag[f"{method}_final_soft_P_event_mean"] - metrics.get("task_success_rate", metrics.get("hit_rate", 0.0)))
                            row.update(diag)
                            append_jsonl(args.out, row)
                            if args.save_samples:
                                extra = ""
                                for k in ["mirror_eta", "mix_lambda", "gap_beta", "semi_beta", "quality_lambda"]:
                                    if hp.get(k) is not None:
                                        extra += f"_{k}{hp[k]:g}"
                                sp = args.out.replace(".jsonl", f"_{target_name}_{method}_nfe{nfe}_s{step_size:g}{extra}.txt")
                                write_sample_texts(sp, texts, args.max_saved_samples)

    csv_path = args.out.replace(".jsonl", ".csv")
    write_csv_from_jsonl(args.out, csv_path)
    print("\nSaved JSONL:", args.out)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
