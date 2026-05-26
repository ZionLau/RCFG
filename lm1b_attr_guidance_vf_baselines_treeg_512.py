#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LM1B few-step attribute-guidance smoke evaluation for Semicat / Discrete Flow Maps.

Guidance reward:
  Differentiable target-class log-probability from the user's four fine-tuned
  BERT reward models.  We feed the endpoint simplex p through the classifier via
  soft embeddings: inputs_embeds = p @ reward_model.embeddings.weight.

Final table metrics:
  - Gen.PPL / Gen.NLL from a downloaded causal LM, e.g. gpt2-large.
  - Reward from downloaded hard verifier models, evaluated on decoded text.

Compared methods:
  flow-map methods: base, fmtg, fmrg, fmrg_sat, sfmrg_conflict_adapt, sat_pareto_quality, gap_aware
  vector-field baselines: base_vf, universal_guidance, dflow, sgfm, treeg, ocflow

This script does not use black-box guidance, rejection sampling, or reranking.
All non-base guidance directions are differentiable gradients of the same
reward model, with optional saturation / Pareto / conflict corrections.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

try:
    from omegaconf import OmegaConf
    import hydra
except Exception as e:
    raise RuntimeError("Need hydra-core and omegaconf installed in the semicat env.") from e

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


TASK_DEFAULTS = {
    "ag_news": {
        "reward_dir": "ag_news_bert",
        "verifier_dir": "ag_news_verifier",
        "target_label": 1,  # Sports in AG News
        "display": "AGNews-Sports",
    },
    "cola": {
        "reward_dir": "cola_bert",
        "verifier_dir": "cola_verifier",
        "target_label": 1,  # acceptable
        "display": "CoLA-Acceptable",
    },
    "imdb": {
        "reward_dir": "imdb_bert",
        "verifier_dir": "imdb_verifier",
        "target_label": 1,  # positive
        "display": "IMDb-Positive",
    },
    "tweet_offensive": {
        "reward_dir": "tweet_offensive_bert",
        "verifier_dir": "tweet_offensive_verifier",
        "target_label": 0,  # non-offensive for TweetEval offensive
        "display": "TweetOff-NonOffensive",
    },
}


def stable_int_seed(*parts, mod: int = 2**31 - 1) -> int:
    s = "||".join(map(str, parts))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_float(vals) -> float:
    vals = list(vals)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def sync_if_cuda(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def start_timer(device: str) -> float:
    sync_if_cuda(device)
    return time.perf_counter()


def stop_timer(device: str, start: float) -> float:
    sync_if_cuda(device)
    return time.perf_counter() - start


def reset_peak_vram(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb(device: str) -> float:
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    return 0.0


def parse_mapping(s: str) -> Dict[str, str]:
    out = {}
    if not s:
        return out
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad mapping item {item!r}; expected task=path")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_label_mapping(s: str) -> Dict[str, int]:
    return {k: int(v) for k, v in parse_mapping(s).items()}


# -------------------------
# Semicat model loading
# -------------------------


def _strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["_orig_mod.", "model.", "module."]
    out = {}
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        out[kk] = v
    return out


def load_flow_model(run_dir: str, ckpt_path: str, device: str):
    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Try strict first, then prefix-stripped non-strict.
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        model.load_state_dict(_strip_known_prefixes(sd), strict=False)

    model.to(device)
    model.eval()

    # Some Lightning modules expose the actual flow model under common attrs.
    obj = model
    if not (hasattr(obj, "prior") and hasattr(obj, "xst")):
        for name in ["model", "net", "flow", "fm", "module"]:
            if hasattr(model, name):
                cand = getattr(model, name)
                if hasattr(cand, "prior") and hasattr(cand, "xst"):
                    obj = cand
                    obj.to(device)
                    obj.eval()
                    break
    if not (hasattr(obj, "prior") and hasattr(obj, "xst")):
        raise AttributeError("Loaded model does not expose prior(...) and xst(...).")
    if not hasattr(obj, "in_shape"):
        # Try config fallback.
        if hasattr(model, "in_shape"):
            obj.in_shape = model.in_shape
        elif "model" in cfg and "in_shape" in cfg.model:
            obj.in_shape = tuple(cfg.model.in_shape)
        else:
            raise AttributeError("Flow model has no in_shape; cannot sample prior.")
    return obj, cfg


# -------------------------
# Reward / verifier wrappers
# -------------------------


class SoftRewardModel:
    def __init__(self, path: str, target_label: int, device: str, local_files_only: bool = True):
        self.path = path
        self.target_label = int(target_label)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=local_files_only)
        self.model.to(device).eval()
        self.device = device
        emb = self.model.get_input_embeddings()
        if emb is None:
            raise RuntimeError(f"Reward model {path} has no input embeddings.")
        self.emb_weight = emb.weight
        self.vocab_size = self.emb_weight.shape[0]

    def soft_logprob(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-sample target logprob, prob and logits from endpoint simplex p [B,L,V]."""
        if p.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Endpoint vocab {p.shape[-1]} != reward model vocab {self.vocab_size}. "
                f"Use the BERT-vocab reward models for LM1B guidance."
            )
        inputs_embeds = torch.matmul(p, self.emb_weight)  # [B,L,H]
        attention_mask = torch.ones(p.shape[:2], dtype=torch.long, device=p.device)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = out.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return log_probs[:, self.target_label], probs[:, self.target_label], logits


class HardVerifier:
    def __init__(self, path: str, target_label: int, device: str, local_files_only: bool = True):
        self.path = path
        self.target_label = int(target_label)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=local_files_only)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 16, max_length: int = 256) -> Dict[str, float]:
        vals = []
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            logits = self.model(**tok).logits
            probs = torch.softmax(logits, dim=-1)
            vals.append(probs[:, self.target_label].detach().cpu())
            preds.append(logits.argmax(dim=-1).detach().cpu())
        if not vals:
            return {"reward": float("nan"), "target_rate": float("nan")}
        vals_t = torch.cat(vals).float()
        preds_t = torch.cat(preds)
        return {
            "reward": float(vals_t.mean().item()),
            "target_rate": float((preds_t == self.target_label).float().mean().item()),
        }




def hard_text_reward_scores(rewarder: SoftRewardModel, texts: List[str], batch_size: int = 16, max_length: int = 256) -> List[float]:
    """Score decoded texts with the guidance reward model. Used for Best-of-N reranking.

    This deliberately uses the guidance reward model, not the external verifier,
    so Best-of-N is not selected by the final evaluation model.
    """
    vals = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i+batch_size]]
            tok = rewarder.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tok = {k: v.to(rewarder.device) for k, v in tok.items()}
            logits = rewarder.model(**tok).logits
            probs = torch.softmax(logits, dim=-1)
            vals.append(probs[:, rewarder.target_label].detach().cpu())
    if not vals:
        return []
    return [float(x) for x in torch.cat(vals).detach().cpu().tolist()]

class GPT2PPL:
    def __init__(self, path: str, device: str, local_files_only: bool = True):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 4, max_length: int = 256) -> Dict[str, float]:
        total_nll = 0.0
        total_tok = 0
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i+batch_size]]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = tok["input_ids"].to(self.device)
            attn = tok["attention_mask"].to(self.device)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # loss is averaged over non-ignored tokens.
            n_tok = int((labels != -100).sum().item())
            if n_tok > 0:
                total_nll += float(out.loss.item()) * n_tok
                total_tok += n_tok
        if total_tok == 0:
            return {"gen_nll": float("nan"), "gen_ppl": float("nan")}
        nll = total_nll / total_tok
        ppl = math.exp(min(50.0, nll))
        return {"gen_nll": float(nll), "gen_ppl": float(ppl)}


# -------------------------
# Guidance utilities
# -------------------------


def to_simplex(x: torch.Tensor, mode: str = "auto", eps: float = 1e-8) -> torch.Tensor:
    if mode == "softmax":
        return torch.softmax(x, dim=-1)
    if mode == "renorm":
        y = x.clamp_min(eps)
        return y / y.sum(dim=-1, keepdim=True).clamp_min(eps)
    if mode == "auto":
        with torch.no_grad():
            mn = float(x.detach().min().item())
            mx = float(x.detach().max().item())
            sum_err = float((x.detach().sum(dim=-1) - 1.0).abs().mean().item())
        if mn >= -1e-4 and mx <= 1.5 and sum_err < 0.5:
            y = x.clamp_min(eps)
            return y / y.sum(dim=-1, keepdim=True).clamp_min(eps)
        return torch.softmax(x, dim=-1)
    raise ValueError(f"Unknown p_mode: {mode}")


def token_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(p.clamp_min(eps) * torch.log(p.clamp_min(eps))).sum(dim=-1).mean(dim=1)


def kl_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean(dim=1)


def norm_per_sample(g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return g.reshape(g.shape[0], -1).norm(dim=1).view(g.shape[0], *([1]*(g.ndim-1))).clamp_min(eps)


def unit_per_sample(g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return g / norm_per_sample(g, eps=eps)


def flat_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).sum(dim=1)


def cosine_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    aa = a.reshape(a.shape[0], -1); bb = b.reshape(b.shape[0], -1)
    return (aa * bb).sum(dim=1) / (aa.norm(dim=1).clamp_min(eps) * bb.norm(dim=1).clamp_min(eps))


def project_orthogonal(g: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dot = flat_dot(g, ref)
    denom = flat_dot(ref, ref).clamp_min(eps)
    view = (g.shape[0],) + (1,) * (g.ndim - 1)
    return g - (dot / denom).view(view) * ref


def guidance_weight(step_size: float, dt: float, s: float, schedule: str = "constant") -> float:
    if schedule == "constant":
        return float(step_size) * float(dt)
    if schedule == "sqrt":
        return float(step_size) * math.sqrt(max(1e-8, dt))
    if schedule == "late":
        return float(step_size) * float(dt) * (0.5 + 0.5 * float(s))
    raise ValueError(f"Unknown schedule: {schedule}")


def endpoint_reward(model, x, s_vec, rewarder: SoftRewardModel, p_mode: str):
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    endpoint = model.xst(x, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)
    logp, prob, logits = rewarder.soft_logprob(p)
    return endpoint, p, logp, prob, logits


def compute_fmtg_direction(model, x, s_vec, rewarder, p_mode):
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    with torch.no_grad():
        endpoint = model.xst(x.detach(), s_vec, one_vec)
    y_req = endpoint.detach().requires_grad_(True)
    p = to_simplex(y_req, mode=p_mode)
    logp, prob, _ = rewarder.soft_logprob(p)
    g = torch.autograd.grad(logp.mean(), y_req, retain_graph=False, create_graph=False)[0]
    return g.detach(), {"guidance_logprob": float(logp.detach().mean()), "guidance_prob": float(prob.detach().mean())}


def compute_fmrg_direction(model, x, s_vec, rewarder, p_mode):
    x_req = x.detach().requires_grad_(True)
    _, p, logp, prob, _ = endpoint_reward(model, x_req, s_vec, rewarder, p_mode)
    g = torch.autograd.grad(logp.mean(), x_req, retain_graph=False, create_graph=False)[0]
    return g.detach(), {"guidance_logprob": float(logp.detach().mean()), "guidance_prob": float(prob.detach().mean()), "endpoint_entropy": float(token_entropy(p).detach().mean())}


def compute_fmrg_sat_direction(model, x, s_vec, rewarder, p_mode, sat_power=1.0, sat_floor=0.25):
    g, d = compute_fmrg_direction(model, x, s_vec, rewarder, p_mode)
    # Recompute only prob cheaply enough; keep scale per sample for update magnitude.
    with torch.no_grad():
        _, p, _, prob, _ = endpoint_reward(model, x.detach(), s_vec, rewarder, p_mode)
        scale = sat_floor + (1.0 - sat_floor) * torch.clamp(1.0 - prob, min=0.0).pow(sat_power)
    d["sat_step_scale"] = float(scale.mean())
    d["_sample_scale"] = scale.detach()
    return g, d


def compute_smfg_direction(model, x, s_vec, rewarder, p_mode, mirror_eta=0.3, mirror_eps=0.0, grad_clip=10.0):
    eps = 1e-8
    x_req = x.detach().requires_grad_(True)
    _, p, logp, prob, _ = endpoint_reward(model, x_req, s_vec, rewarder, p_mode)
    r = torch.autograd.grad(logp.sum(), p, retain_graph=True, create_graph=False)[0]
    with torch.no_grad():
        p_det = p.detach().clamp_min(eps)
        if mirror_eps and mirror_eps > 0:
            p_det = (1.0 - mirror_eps) * p_det + mirror_eps / p_det.shape[-1]
            p_det = p_det / p_det.sum(dim=-1, keepdim=True).clamp_min(eps)
        r_det = r.detach() - r.detach().mean(dim=-1, keepdim=True)
        if grad_clip and grad_clip > 0:
            r_det = r_det.clamp(-grad_clip, grad_clip)
        q = torch.softmax(torch.log(p_det) + mirror_eta * r_det, dim=-1)
    kl = (q * (torch.log(q.clamp_min(eps)) - torch.log(p.clamp_min(eps)))).sum(dim=-1).mean(dim=1)
    g = -torch.autograd.grad(kl.mean(), x_req, retain_graph=False, create_graph=False)[0]
    return g.detach(), {"guidance_logprob": float(logp.detach().mean()), "guidance_prob": float(prob.detach().mean()), "mirror_kl": float(kl.detach().mean()), "reward_mirror_cos_dummy": 0.0}


def compute_sfmrg_conflict_adapt(model, x, s_vec, rewarder, p_mode, mirror_eta=0.3, mix_lambda=0.2, sat_tau=0.30, sat_kappa=0.10, mirror_eps=0.0):
    gR, dR = compute_fmrg_direction(model, x, s_vec, rewarder, p_mode)
    gM, dM = compute_smfg_direction(model, x, s_vec, rewarder, p_mode, mirror_eta=mirror_eta, mirror_eps=mirror_eps)
    cos = cosine_per_sample(gM, gR)
    gM_orth = project_orthogonal(gM, gR)
    view = (x.shape[0],) + (1,) * (x.ndim - 1)
    use_orth = (cos < 0).view(view)
    gC = torch.where(use_orth, gM_orth, gM)
    with torch.no_grad():
        _, _, _, prob, _ = endpoint_reward(model, x.detach(), s_vec, rewarder, p_mode)
        gate = mix_lambda * torch.sigmoid((prob - sat_tau) / max(1e-6, sat_kappa))
    direction = unit_per_sample(gR) + gate.view(view) * unit_per_sample(gC)
    d = dict(dR)
    d.update({
        "mix_lambda": float(gate.mean()),
        "reward_mirror_cos": float(cos.mean()),
        "conflict_rate": float((cos < 0).float().mean()),
    })
    return direction.detach(), d


def compute_quality_direction(model, x, s_vec, rewarder, p_mode, entropy_weight=1.0, basekl_weight=0.2):
    x_req = x.detach().requires_grad_(True)
    _, p, logp, prob, _ = endpoint_reward(model, x_req, s_vec, rewarder, p_mode)
    with torch.no_grad():
        _, p_base, _, _, _ = endpoint_reward(model, x.detach(), s_vec, rewarder, p_mode)
    ent = token_entropy(p)
    kl_base = kl_pq(p, p_base.detach())
    Q = entropy_weight * ent - basekl_weight * kl_base
    gQ = torch.autograd.grad(Q.mean(), x_req, retain_graph=False, create_graph=False)[0]
    return gQ.detach(), {"quality_entropy": float(ent.detach().mean()), "quality_kl_base": float(kl_base.detach().mean()), "quality_obj": float(Q.detach().mean())}


def compute_sat_pareto_quality(model, x, s_vec, rewarder, p_mode, quality_lambda=0.1, sat_power=1.0, sat_floor=0.25, entropy_weight=1.0, basekl_weight=0.2):
    gR, dR = compute_fmrg_direction(model, x, s_vec, rewarder, p_mode)
    gQ, dQ = compute_quality_direction(model, x, s_vec, rewarder, p_mode, entropy_weight=entropy_weight, basekl_weight=basekl_weight)
    dot = flat_dot(gR, gQ)
    denom = flat_dot(gQ, gQ).clamp_min(1e-8)
    view = (x.shape[0],) + (1,) * (x.ndim - 1)
    # If reward gradient conflicts with quality gradient, remove reward component that decreases quality.
    gR_pc = torch.where((dot < 0).view(view), gR - (dot / denom).view(view) * gQ, gR)
    direction = unit_per_sample(gR_pc) + quality_lambda * unit_per_sample(gQ)
    with torch.no_grad():
        _, _, _, prob, _ = endpoint_reward(model, x.detach(), s_vec, rewarder, p_mode)
        scale = sat_floor + (1.0 - sat_floor) * torch.clamp(1.0 - prob, min=0.0).pow(sat_power)
    d = dict(dR); d.update(dQ)
    d.update({
        "quality_lambda": float(quality_lambda),
        "reward_quality_cos": float(cosine_per_sample(gR, gQ).mean()),
        "quality_conflict_rate": float((dot < 0).float().mean()),
        "sat_step_scale": float(scale.mean()),
        "_sample_scale": scale.detach(),
    })
    return direction.detach(), d


def compute_gap_aware(model, x, s_vec, rewarder, p_mode, mirror_eta=0.3, mix_lambda=0.2, quality_lambda=0.1, sat_tau=0.30, sat_kappa=0.10, sat_power=1.0, sat_floor=0.25):
    gP, dP = compute_sat_pareto_quality(model, x, s_vec, rewarder, p_mode, quality_lambda=quality_lambda, sat_power=sat_power, sat_floor=sat_floor)
    gC, dC = compute_sfmrg_conflict_adapt(model, x, s_vec, rewarder, p_mode, mirror_eta=mirror_eta, mix_lambda=mix_lambda, sat_tau=sat_tau, sat_kappa=sat_kappa)
    direction = unit_per_sample(gP) + mix_lambda * unit_per_sample(project_orthogonal(gC, gP))
    d = dict(dP)
    d.update({f"conflict_{k}": v for k, v in dC.items() if not k.startswith("_")})
    return direction.detach(), d


# -------------------------
# Sampling / decoding / eval
# -------------------------


def decode_tokens(tokenizer, ids: torch.Tensor) -> List[str]:
    arr = ids.detach().cpu().tolist()
    return [tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for x in arr]


def ngram_rep_rate(text: str, n: int = 3) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(text: str, n: int = 2) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(grams)) / max(1, len(grams))


def sample_guided(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    method: str,
    n_samples: int,
    batch_size: int,
    nfe: int,
    step_size: float,
    device: str,
    p_mode: str = "softmax",
    schedule: str = "constant",
    early_stop: float = 1.0,
    mirror_eta: float = 0.3,
    mix_lambda: float = 0.2,
    quality_lambda: float = 0.1,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
    entropy_weight: float = 1.0,
    basekl_weight: float = 0.2,
    delay_start: float = 0.25,
    proj_perp_scale: float = 0.25,
):
    method = method.lower()
    valid = {"base", "fmtg", "fmrg", "fmrg_sat", "sfmrg_conflict_adapt", "sat_pareto_quality", "gap_aware", "delayed_fmrg", "proj_fmrg"}
    if method not in valid:
        raise ValueError(f"Unknown method {method}. Valid: {sorted(valid)}")
    all_ids = []
    diag: Dict[str, List[float]] = {}
    cost = {
        "guidance_time_sec": 0.0,
        "flowmap_time_sec": 0.0,
        "guidance_batch_calls": 0.0,
        "flowmap_batch_calls": 0.0,
        "guided_batch_steps": 0.0,
    }

    def add_diag(d: Dict):
        for k, v in d.items():
            if k.startswith("_"):
                continue
            try:
                diag.setdefault(k, []).append(float(v))
            except Exception:
                pass

    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x = model.prior((b, *tuple(model.in_shape)), device=device)
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for s, t in zip(ts[:-1], ts[1:]):
            s_float = float(s.item()); t_float = float(t.item()); dt = max(1e-6, t_float - s_float)
            s_vec = s.expand((b,)); t_vec = t.expand((b,))
            t_flow = start_timer(device)
            with torch.no_grad():
                base_next = model.xst(x.detach(), s_vec, t_vec)
                vel = (base_next - x.detach()) / dt
                vel_norm = norm_per_sample(vel).detach().clamp_min(1e-8)
            cost["flowmap_time_sec"] += stop_timer(device, t_flow)
            cost["flowmap_batch_calls"] += 1.0
            if method != "base" and s_float < early_stop:
                t_guidance = start_timer(device)
                guidance_count_this_step = 1.0
                if method == "fmtg":
                    direction, d = compute_fmtg_direction(model, x, s_vec, rewarder, p_mode)
                elif method == "fmrg":
                    direction, d = compute_fmrg_direction(model, x, s_vec, rewarder, p_mode)
                elif method == "delayed_fmrg":
                    if s_float < delay_start:
                        direction, d = torch.zeros_like(x), {"delayed_skip": 1.0}
                        guidance_count_this_step = 0.0
                    else:
                        direction, d = compute_fmrg_direction(model, x, s_vec, rewarder, p_mode)
                        d["delayed_skip"] = 0.0
                elif method == "proj_fmrg":
                    g_raw, d = compute_fmrg_direction(model, x, s_vec, rewarder, p_mode)
                    dot = flat_dot(g_raw, vel)
                    denom = flat_dot(vel, vel).clamp_min(1e-8)
                    view = (x.shape[0],) + (1,) * (x.ndim - 1)
                    g_para = (dot / denom).view(view) * vel
                    g_perp = g_raw - g_para
                    direction = g_para + float(proj_perp_scale) * g_perp
                    d["proj_perp_scale"] = float(proj_perp_scale)
                    d["proj_parallel_frac"] = float((norm_per_sample(g_para) / norm_per_sample(g_raw)).mean())
                elif method == "fmrg_sat":
                    direction, d = compute_fmrg_sat_direction(model, x, s_vec, rewarder, p_mode, sat_power=sat_power, sat_floor=sat_floor)
                elif method == "sfmrg_conflict_adapt":
                    direction, d = compute_sfmrg_conflict_adapt(model, x, s_vec, rewarder, p_mode, mirror_eta=mirror_eta, mix_lambda=mix_lambda, sat_tau=sat_tau, sat_kappa=sat_kappa)
                elif method == "sat_pareto_quality":
                    direction, d = compute_sat_pareto_quality(model, x, s_vec, rewarder, p_mode, quality_lambda=quality_lambda, sat_power=sat_power, sat_floor=sat_floor, entropy_weight=entropy_weight, basekl_weight=basekl_weight)
                elif method == "gap_aware":
                    direction, d = compute_gap_aware(model, x, s_vec, rewarder, p_mode, mirror_eta=mirror_eta, mix_lambda=mix_lambda, quality_lambda=quality_lambda, sat_tau=sat_tau, sat_kappa=sat_kappa, sat_power=sat_power, sat_floor=sat_floor)
                else:
                    raise AssertionError(method)
                gamma = guidance_weight(step_size, dt, s_float, schedule)
                sample_scale = d.pop("_sample_scale", None)
                with torch.no_grad():
                    dnorm = norm_per_sample(direction).detach().clamp_min(1e-8)
                    update = gamma * direction.detach() / dnorm * vel_norm
                    if sample_scale is not None:
                        update = update * sample_scale.to(device).view(b, *([1]*(update.ndim-1)))
                    x = x.detach() + update
                    d["direction_norm"] = float(dnorm.mean())
                    d["update_norm"] = float(norm_per_sample(update).mean())
                    d["update_over_vel"] = float((norm_per_sample(update) / vel_norm).mean())
                    add_diag(d)
                cost["guidance_time_sec"] += stop_timer(device, t_guidance)
                cost["guided_batch_steps"] += 1.0
                cost["guidance_batch_calls"] += guidance_count_this_step
            t_flow = start_timer(device)
            with torch.no_grad():
                x = model.xst(x.detach(), s_vec, t_vec)
            cost["flowmap_time_sec"] += stop_timer(device, t_flow)
            cost["flowmap_batch_calls"] += 1.0
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b

    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    diag_mean = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    diag_mean.update({
        "cost_guidance_time_sec": float(cost["guidance_time_sec"]),
        "cost_flowmap_time_sec": float(cost["flowmap_time_sec"]),
        "cost_guidance_batch_calls": float(cost["guidance_batch_calls"]),
        "cost_guided_batch_steps": float(cost["guided_batch_steps"]),
        "cost_flowmap_batch_calls": float(cost["flowmap_batch_calls"]),
        "cost_guidance_time_per_call_ms": float(1000.0 * cost["guidance_time_sec"] / max(1.0, cost["guidance_batch_calls"])),
        "cost_flowmap_time_per_call_ms": float(1000.0 * cost["flowmap_time_sec"] / max(1.0, cost["flowmap_batch_calls"])),
    })
    return ids, texts, diag_mean




def sample_best_of_n(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    best_of_n: int,
    batch_size: int,
    nfe: int,
    device: str,
    p_mode: str = "softmax",
    max_eval_length: int = 256,
):
    """Best-of-N over base samples, reranked by the guidance reward model on decoded text."""
    if best_of_n < 1:
        raise ValueError("best_of_n must be >= 1")
    ids_all, texts_all, diag = sample_guided(
        model, tokenizer, rewarder, "base", n_samples * best_of_n, batch_size,
        nfe, step_size=0.0, device=device, p_mode=p_mode,
    )
    t_rank = start_timer(device)
    scores = hard_text_reward_scores(rewarder, texts_all, batch_size=batch_size, max_length=max_eval_length)
    rank_time = stop_timer(device, t_rank)
    if len(scores) != n_samples * best_of_n:
        raise RuntimeError("Best-of-N scoring produced wrong number of scores")
    scores_t = torch.tensor(scores, dtype=torch.float32)
    scores_2d = scores_t.view(n_samples, best_of_n)
    pick = scores_2d.argmax(dim=1)
    idx = torch.arange(n_samples, dtype=torch.long) * best_of_n + pick.cpu()
    ids = ids_all[idx]
    texts = [texts_all[int(i)] for i in idx.tolist()]
    diag = dict(diag)
    diag.update({
        "best_of_n": float(best_of_n),
        "bon_select_reward_mean": float(scores_2d.max(dim=1).values.mean().item()),
        "bon_pool_reward_mean": float(scores_t.mean().item()),
        "cost_bon_rerank_time_sec": float(rank_time),
        "cost_effective_candidate_nfe": float(best_of_n * nfe),
    })
    return ids, texts, diag


def sample_reno_init(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    batch_size: int,
    nfe: int,
    reno_steps: int,
    reno_lr: float,
    device: str,
    p_mode: str = "softmax",
):
    """ReNO-style initial-state reward optimization adapted to categorical flow maps.

    This optimizes the initial prior state through the one-step endpoint reward,
    then runs the base flow-map sampler. It is an adapted baseline, not the
    original image-domain ReNO implementation.
    """
    all_ids = []
    diag = {"reno_logprob": [], "reno_prob": [], "reno_update_norm": []}
    cost = {"reno_guidance_time_sec": 0.0, "flowmap_time_sec": 0.0, "reno_guidance_calls": 0.0, "flowmap_batch_calls": 0.0}
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x = model.prior((b, *tuple(model.in_shape)), device=device)
        s0 = torch.zeros((b,), device=device)
        one = torch.ones((b,), device=device)
        for _ in range(max(0, reno_steps)):
            t_guidance = start_timer(device)
            t_flow = start_timer(device)
            with torch.no_grad():
                base_end = model.xst(x.detach(), s0, one)
                base_norm = norm_per_sample(base_end - x.detach()).detach().clamp_min(1e-8)
            cost["flowmap_time_sec"] += stop_timer(device, t_flow)
            cost["flowmap_batch_calls"] += 1.0
            x_req = x.detach().requires_grad_(True)
            _, p, logp, prob, _ = endpoint_reward(model, x_req, s0, rewarder, p_mode)
            # endpoint_reward contains one flow-map call; count it for cost accounting.
            cost["flowmap_batch_calls"] += 1.0
            g = torch.autograd.grad(logp.mean(), x_req, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                update = float(reno_lr) * unit_per_sample(g).detach() * base_norm
                x = x.detach() + update
                diag["reno_logprob"].append(float(logp.detach().mean()))
                diag["reno_prob"].append(float(prob.detach().mean()))
                diag["reno_update_norm"].append(float(norm_per_sample(update).mean()))
            cost["reno_guidance_time_sec"] += stop_timer(device, t_guidance)
            cost["reno_guidance_calls"] += 1.0
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for s, t in zip(ts[:-1], ts[1:]):
            s_vec = s.expand((b,)); t_vec = t.expand((b,))
            t_flow = start_timer(device)
            with torch.no_grad():
                x = model.xst(x.detach(), s_vec, t_vec)
            cost["flowmap_time_sec"] += stop_timer(device, t_flow)
            cost["flowmap_batch_calls"] += 1.0
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    diag_mean = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    diag_mean["guidance_reno_steps"] = float(reno_steps)
    diag_mean["guidance_reno_lr"] = float(reno_lr)
    diag_mean.update({
        "cost_guidance_time_sec": float(cost["reno_guidance_time_sec"]),
        "cost_flowmap_time_sec": float(cost["flowmap_time_sec"]),
        "cost_guidance_batch_calls": float(cost["reno_guidance_calls"]),
        "cost_flowmap_batch_calls": float(cost["flowmap_batch_calls"]),
        "cost_guidance_time_per_call_ms": float(1000.0 * cost["reno_guidance_time_sec"] / max(1.0, cost["reno_guidance_calls"])),
        "cost_flowmap_time_per_call_ms": float(1000.0 * cost["flowmap_time_sec"] / max(1.0, cost["flowmap_batch_calls"])),
        "cost_effective_candidate_nfe": float(nfe + 2 * reno_steps),
    })
    return ids, texts, diag_mean


# -------------------------
# Ordinary continuous vector-field guidance baselines
# These methods use the underlying Semicat vector field model.vf rather than
# the flow-map jump model.xst. They are intended as continuous-flow baselines:
#   - base_vf: ordinary Euler sampler on the vector field
#   - universal_guidance: Universal Guidance / FreeDoM-style energy guidance
#   - dflow: D-Flow source/noise optimization through the vector-field sampler
#   - sgfm: Source-Guided Flow Matching via approximate Langevin source sampling
#   - treeg: TreeG path steering with branch-out, value evaluation, and active-path selection
#   - ocflow: OC-Flow-style trajectory optimal-control variables
# -------------------------


def require_vf(model):
    if not hasattr(model, "vf"):
        raise RuntimeError("This baseline requires model.vf(x,t). The loaded model does not expose a vector-field sampler.")


def vf_eval(model, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
    require_vf(model)
    return model.vf(x, t_vec)


def soft_reward_from_state(x: torch.Tensor, rewarder: SoftRewardModel, p_mode: str):
    p = to_simplex(x, mode=p_mode)
    logp, prob, logits = rewarder.soft_logprob(p)
    return p, logp, prob, logits


def integrate_vf_euler(model, x0: torch.Tensor, nfe: int, device: str, controls: Optional[List[torch.Tensor]] = None):
    """Differentiable Euler integration of the underlying vector field."""
    x = x0
    ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    for k, (s, t) in enumerate(zip(ts[:-1], ts[1:])):
        dt = t - s
        s_vec = s.expand((x.shape[0],))
        v = vf_eval(model, x, s_vec)
        if controls is not None:
            v = v + controls[k]
        x = x + dt * v
    return x


def sample_base_vf(model, tokenizer, n_samples: int, batch_size: int, nfe: int, device: str):
    require_vf(model)
    all_ids = []
    cost = {"vf_time_sec": 0.0, "vf_batch_calls": 0.0}
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x = model.prior((b, *tuple(model.in_shape)), device=device)
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for s, t in zip(ts[:-1], ts[1:]):
            dt = t - s
            s_vec = s.expand((b,))
            t_vf = start_timer(device)
            with torch.no_grad():
                x = x + dt * vf_eval(model, x.detach(), s_vec)
            cost["vf_time_sec"] += stop_timer(device, t_vf)
            cost["vf_batch_calls"] += 1.0
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    return ids, texts, {
        "cost_vf_time_sec": float(cost["vf_time_sec"]),
        "cost_vf_batch_calls": float(cost["vf_batch_calls"]),
        "cost_vf_time_per_call_ms": float(1000.0 * cost["vf_time_sec"] / max(1.0, cost["vf_batch_calls"])),
        "cost_effective_candidate_nfe": float(nfe),
    }


def sample_universal_guidance_vf(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    batch_size: int,
    nfe: int,
    step_size: float,
    device: str,
    p_mode: str = "softmax",
    schedule: str = "constant",
):
    """Universal Guidance / FreeDoM-style energy guidance on a vector-field sampler.

    At each step we form a one-step clean estimate xhat_1 = x_t + (1-t) v_t(x_t),
    compute the reward gradient through this estimate, and add the energy gradient
    to the Euler update.
    """
    require_vf(model)
    all_ids = []
    diag: Dict[str, List[float]] = {"ug_logprob": [], "ug_prob": [], "ug_update_norm": [], "ug_update_over_vel": []}
    cost = {"vf_time_sec": 0.0, "guidance_time_sec": 0.0, "vf_batch_calls": 0.0, "guidance_calls": 0.0}
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x = model.prior((b, *tuple(model.in_shape)), device=device)
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for s, t in zip(ts[:-1], ts[1:]):
            s_float = float(s.item()); dt = float((t - s).item())
            s_vec = s.expand((b,))
            # Base vector-field step.
            t_vf = start_timer(device)
            with torch.no_grad():
                v_base = vf_eval(model, x.detach(), s_vec)
                vel_norm = norm_per_sample(v_base).detach().clamp_min(1e-8)
            cost["vf_time_sec"] += stop_timer(device, t_vf); cost["vf_batch_calls"] += 1.0

            # Energy gradient via a one-step clean estimate.
            t_guidance = start_timer(device)
            x_req = x.detach().requires_grad_(True)
            v_req = vf_eval(model, x_req, s_vec)
            # Count the guidance-side vector-field call as a vf call too.
            cost["vf_batch_calls"] += 1.0
            xhat = x_req + (1.0 - s) * v_req
            p, logp, prob, _ = soft_reward_from_state(xhat, rewarder, p_mode)
            g = torch.autograd.grad(logp.mean(), x_req, retain_graph=False, create_graph=False)[0]
            gamma = guidance_weight(step_size, dt, s_float, schedule)
            with torch.no_grad():
                update = gamma * unit_per_sample(g).detach() * vel_norm
                x = x.detach() + dt * v_base + update
                diag["ug_logprob"].append(float(logp.detach().mean()))
                diag["ug_prob"].append(float(prob.detach().mean()))
                diag["ug_update_norm"].append(float(norm_per_sample(update).mean()))
                diag["ug_update_over_vel"].append(float((norm_per_sample(update) / vel_norm).mean()))
            cost["guidance_time_sec"] += stop_timer(device, t_guidance); cost["guidance_calls"] += 1.0
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    d = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    d.update({
        "cost_guidance_time_sec": float(cost["guidance_time_sec"]),
        "cost_vf_time_sec": float(cost["vf_time_sec"]),
        "cost_guidance_batch_calls": float(cost["guidance_calls"]),
        "cost_vf_batch_calls": float(cost["vf_batch_calls"]),
        "cost_guidance_time_per_call_ms": float(1000.0 * cost["guidance_time_sec"] / max(1.0, cost["guidance_calls"])),
        "cost_vf_time_per_call_ms": float(1000.0 * cost["vf_time_sec"] / max(1.0, cost["vf_batch_calls"])),
        "cost_effective_candidate_nfe": float(2 * nfe),
    })
    return ids, texts, d


def sample_dflow_source(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    batch_size: int,
    nfe: int,
    source_steps: int,
    source_lr: float,
    source_reg: float,
    step_size: float,
    device: str,
    p_mode: str = "softmax",
):
    """D-Flow: optimize the source/noise point through the vector-field sampler."""
    require_vf(model)
    all_ids = []
    diag: Dict[str, List[float]] = {"dflow_logprob": [], "dflow_prob": [], "dflow_update_norm": []}
    cost = {"guidance_time_sec": 0.0, "vf_batch_calls": 0.0, "guidance_calls": 0.0}
    lr = float(source_lr) * float(step_size)
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x0 = model.prior((b, *tuple(model.in_shape)), device=device)
        m = torch.zeros_like(x0); vv = torch.zeros_like(x0)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for it in range(max(0, source_steps)):
            t_guidance = start_timer(device)
            x_req = x0.detach().requires_grad_(True)
            x_end = integrate_vf_euler(model, x_req, nfe, device)
            cost["vf_batch_calls"] += float(nfe)
            _, logp, prob, _ = soft_reward_from_state(x_end, rewarder, p_mode)
            prior_penalty = 0.5 * x_req.pow(2).reshape(b, -1).mean(dim=1)
            obj = logp.mean() - float(source_reg) * prior_penalty.mean()
            g = torch.autograd.grad(obj, x_req, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                vv.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                mh = m / (1.0 - beta1 ** (it + 1))
                vh = vv / (1.0 - beta2 ** (it + 1))
                update = lr * mh / (vh.sqrt() + eps)
                x0 = x0.detach() + update
                diag["dflow_logprob"].append(float(logp.detach().mean()))
                diag["dflow_prob"].append(float(prob.detach().mean()))
                diag["dflow_update_norm"].append(float(norm_per_sample(update).mean()))
            cost["guidance_time_sec"] += stop_timer(device, t_guidance); cost["guidance_calls"] += 1.0
        with torch.no_grad():
            x = x0.detach()
            ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
            for s, t in zip(ts[:-1], ts[1:]):
                dt = t - s; s_vec = s.expand((b,))
                x = x + dt * vf_eval(model, x, s_vec)
                cost["vf_batch_calls"] += 1.0
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    d = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    d.update({
        "guidance_source_steps": float(source_steps),
        "guidance_source_lr": float(lr),
        "guidance_source_reg": float(source_reg),
        "cost_guidance_time_sec": float(cost["guidance_time_sec"]),
        "cost_guidance_batch_calls": float(cost["guidance_calls"]),
        "cost_vf_batch_calls": float(cost["vf_batch_calls"]),
        "cost_guidance_time_per_call_ms": float(1000.0 * cost["guidance_time_sec"] / max(1.0, cost["guidance_calls"])),
        "cost_effective_candidate_nfe": float((source_steps + 1) * nfe),
    })
    return ids, texts, d


def sample_sgfm_langevin(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    batch_size: int,
    nfe: int,
    source_steps: int,
    source_lr: float,
    source_reg: float,
    sgfm_beta: float,
    step_size: float,
    device: str,
    p_mode: str = "softmax",
):
    """SGFM: approximate sampling from a reward-tilted source distribution.

    Uses unadjusted Langevin updates on log q_0(x0) = beta R(T(x0)) - source_reg ||x0||^2/2.
    """
    require_vf(model)
    all_ids = []
    diag: Dict[str, List[float]] = {"sgfm_logprob": [], "sgfm_prob": [], "sgfm_update_norm": []}
    cost = {"guidance_time_sec": 0.0, "vf_batch_calls": 0.0, "guidance_calls": 0.0}
    lr = float(source_lr) * float(step_size)
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x0 = model.prior((b, *tuple(model.in_shape)), device=device)
        for _ in range(max(0, source_steps)):
            t_guidance = start_timer(device)
            x_req = x0.detach().requires_grad_(True)
            x_end = integrate_vf_euler(model, x_req, nfe, device)
            cost["vf_batch_calls"] += float(nfe)
            _, logp, prob, _ = soft_reward_from_state(x_end, rewarder, p_mode)
            prior_penalty = 0.5 * x_req.pow(2).reshape(b, -1).mean(dim=1)
            obj = float(sgfm_beta) * logp.mean() - float(source_reg) * prior_penalty.mean()
            g = torch.autograd.grad(obj, x_req, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                noise = torch.randn_like(x0)
                update = 0.5 * lr * g + math.sqrt(max(lr, 0.0)) * noise
                x0 = x0.detach() + update
                diag["sgfm_logprob"].append(float(logp.detach().mean()))
                diag["sgfm_prob"].append(float(prob.detach().mean()))
                diag["sgfm_update_norm"].append(float(norm_per_sample(update).mean()))
            cost["guidance_time_sec"] += stop_timer(device, t_guidance); cost["guidance_calls"] += 1.0
        with torch.no_grad():
            x = x0.detach()
            ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
            for s, t in zip(ts[:-1], ts[1:]):
                dt = t - s; s_vec = s.expand((b,))
                x = x + dt * vf_eval(model, x, s_vec)
                cost["vf_batch_calls"] += 1.0
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    d = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    d.update({
        "guidance_source_steps": float(source_steps),
        "guidance_source_lr": float(lr),
        "guidance_source_reg": float(source_reg),
        "guidance_sgfm_beta": float(sgfm_beta),
        "cost_guidance_time_sec": float(cost["guidance_time_sec"]),
        "cost_guidance_batch_calls": float(cost["guidance_calls"]),
        "cost_vf_batch_calls": float(cost["vf_batch_calls"]),
        "cost_guidance_time_per_call_ms": float(1000.0 * cost["guidance_time_sec"] / max(1.0, cost["guidance_calls"])),
        "cost_effective_candidate_nfe": float((source_steps + 1) * nfe),
    })
    return ids, texts, d



def sample_treeg_vf(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    batch_size: int,
    nfe: int,
    tree_active: int,
    tree_branch: int,
    tree_noise: float,
    tree_value_rollout: int,
    device: str,
    p_mode: str = "softmax",
):
    """TreeG: tree-search path steering on the vector-field sampler.

    This is a direct TreeG instantiation for Semicat's continuous vector-field
    sampler.  At each ODE step, each active path branches into multiple next-step
    candidates, the candidates are evaluated by a terminal value estimate, and
    the best active paths are kept.  The value function uses the same soft reward
    model as other baselines.  No gradients are used.

    BranchOut: base Euler step plus optional Gaussian perturbations.
    Value: reward of a look-ahead endpoint.  If tree_value_rollout <= 1, we use
    a one-step endpoint estimate x + (1-t) v(x,t); otherwise we Euler-rollout
    tree_value_rollout steps from the candidate to t=1.
    """
    require_vf(model)
    A = max(1, int(tree_active))
    B = max(1, int(tree_branch))
    R = max(1, int(tree_value_rollout))
    all_ids = []
    diag: Dict[str, List[float]] = {
        "treeg_value": [],
        "treeg_selected_value": [],
        "treeg_branch_std": [],
        "treeg_active": [],
        "treeg_branch": [],
    }
    cost = {"vf_time_sec": 0.0, "value_time_sec": 0.0, "vf_batch_calls": 0.0, "value_calls": 0.0}

    def value_fn(xcand: torch.Tensor, t_scalar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return value per candidate and endpoint state estimate."""
        if R <= 1:
            t_vec = t_scalar.expand((xcand.shape[0],))
            tv = start_timer(device)
            with torch.no_grad():
                v = vf_eval(model, xcand.detach(), t_vec)
                xhat = xcand.detach() + (1.0 - t_scalar) * v
            cost["vf_time_sec"] += stop_timer(device, tv)
            cost["vf_batch_calls"] += 1.0
        else:
            with torch.no_grad():
                xhat = xcand.detach()
                ts2 = torch.linspace(float(t_scalar.item()), 1.0, R + 1, device=device)
                for ss, tt in zip(ts2[:-1], ts2[1:]):
                    dt2 = tt - ss
                    ss_vec = ss.expand((xhat.shape[0],))
                    tv = start_timer(device)
                    v = vf_eval(model, xhat, ss_vec)
                    cost["vf_time_sec"] += stop_timer(device, tv)
                    cost["vf_batch_calls"] += 1.0
                    xhat = xhat + dt2 * v
        tv = start_timer(device)
        with torch.no_grad():
            _, logp, prob, _ = soft_reward_from_state(xhat, rewarder, p_mode)
        cost["value_time_sec"] += stop_timer(device, tv)
        cost["value_calls"] += 1.0
        return logp.detach(), xhat.detach()

    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        # Maintain A active paths for each requested sample.
        x = model.prior((b * A, *tuple(model.in_shape)), device=device)
        # group id of each active path, shape [b, A]
        ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for s, t in zip(ts[:-1], ts[1:]):
            dt = t - s
            # Base step from active paths.
            s_vec = s.expand((x.shape[0],))
            tv = start_timer(device)
            with torch.no_grad():
                v = vf_eval(model, x.detach(), s_vec)
                base_next = x.detach() + dt * v
            cost["vf_time_sec"] += stop_timer(device, tv)
            cost["vf_batch_calls"] += 1.0

            # BranchOut: repeat each active path B times. Include the deterministic
            # base candidate as branch 0; noisy branches explore nearby paths.
            cand = base_next[:, None].expand(-1, B, -1, -1).reshape(x.shape[0] * B, *x.shape[1:]).contiguous()
            if B > 1 and float(tree_noise) > 0.0:
                with torch.no_grad():
                    noise = torch.randn_like(cand)
                    noise = float(tree_noise) * math.sqrt(max(float(dt.item()), 1e-8)) * noise
                    # Keep branch 0 deterministic for each active path.
                    noise = noise.reshape(x.shape[0], B, *x.shape[1:])
                    noise[:, 0].zero_()
                    cand = cand + noise.reshape_as(cand)
                    diag["treeg_branch_std"].append(float(noise.reshape(noise.shape[0], noise.shape[1], -1).std()))
            else:
                diag["treeg_branch_std"].append(0.0)

            values, _ = value_fn(cand, t)
            diag["treeg_value"].append(float(values.mean()))
            # Reshape values by original requested sample: [b, A*B]
            cand_by_group = cand.reshape(b, A * B, *x.shape[1:])
            val_by_group = values.reshape(b, A * B)
            keep = min(A, A * B)
            topv, topidx = torch.topk(val_by_group, k=keep, dim=1)
            gather_idx = topidx.view(b, keep, *([1] * (cand_by_group.dim() - 2))).expand(-1, -1, *cand_by_group.shape[2:])
            x = torch.gather(cand_by_group, dim=1, index=gather_idx).reshape(b * keep, *x.shape[1:]).contiguous()
            A = keep
            diag["treeg_selected_value"].append(float(topv.mean()))
            diag["treeg_active"].append(float(A))
            diag["treeg_branch"].append(float(B))
        # Pick best final path for each requested sample.
        final_values, _ = value_fn(x.detach(), torch.tensor(1.0, device=device))
        x_by_group = x.reshape(b, A, *x.shape[1:])
        val_by_group = final_values.reshape(b, A)
        best = torch.argmax(val_by_group, dim=1)
        gather_idx = best.view(b, 1, *([1] * (x_by_group.dim() - 2))).expand(-1, 1, *x_by_group.shape[2:])
        x_best = torch.gather(x_by_group, dim=1, index=gather_idx).squeeze(1)
        all_ids.append(x_best.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    d = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    # TreeG's cost is best understood as number of candidate transitions / value calls.
    # Branch candidates use the base transition plus look-ahead value evaluation.
    d.update({
        "guidance_tree_active": float(tree_active),
        "guidance_tree_branch": float(tree_branch),
        "guidance_tree_noise": float(tree_noise),
        "guidance_tree_value_rollout": float(tree_value_rollout),
        "cost_vf_time_sec": float(cost["vf_time_sec"]),
        "cost_value_time_sec": float(cost["value_time_sec"]),
        "cost_vf_batch_calls": float(cost["vf_batch_calls"]),
        "cost_value_batch_calls": float(cost["value_calls"]),
        "cost_vf_time_per_call_ms": float(1000.0 * cost["vf_time_sec"] / max(1.0, cost["vf_batch_calls"])),
        "cost_value_time_per_call_ms": float(1000.0 * cost["value_time_sec"] / max(1.0, cost["value_calls"])),
        "cost_effective_candidate_nfe": float(nfe * max(1, int(tree_active)) * max(1, int(tree_branch)) + nfe * max(1, int(tree_active)) * max(1, int(tree_branch)) * max(1, int(tree_value_rollout))),
    })
    return ids, texts, d


def sample_ocflow_controls(
    model,
    tokenizer,
    rewarder: SoftRewardModel,
    n_samples: int,
    batch_size: int,
    nfe: int,
    oc_steps: int,
    oc_lr: float,
    control_reg: float,
    step_size: float,
    device: str,
    p_mode: str = "softmax",
):
    """OC-Flow: optimize additive controls along the vector-field trajectory."""
    require_vf(model)
    all_ids = []
    diag: Dict[str, List[float]] = {"ocflow_logprob": [], "ocflow_prob": [], "ocflow_control_norm": []}
    cost = {"guidance_time_sec": 0.0, "vf_batch_calls": 0.0, "guidance_calls": 0.0}
    lr = float(oc_lr) * float(step_size)
    left = n_samples
    while left > 0:
        b = min(batch_size, left)
        x0 = model.prior((b, *tuple(model.in_shape)), device=device)
        controls = [torch.zeros_like(x0, requires_grad=True) for _ in range(nfe)]
        ms = [torch.zeros_like(x0) for _ in range(nfe)]
        vs = [torch.zeros_like(x0) for _ in range(nfe)]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for it in range(max(0, oc_steps)):
            t_guidance = start_timer(device)
            x_end = integrate_vf_euler(model, x0.detach(), nfe, device, controls=controls)
            cost["vf_batch_calls"] += float(nfe)
            _, logp, prob, _ = soft_reward_from_state(x_end, rewarder, p_mode)
            reg = 0.0
            for c in controls:
                reg = reg + c.pow(2).reshape(b, -1).mean()
            obj = logp.mean() - float(control_reg) * reg / max(1, nfe)
            grads = torch.autograd.grad(obj, controls, retain_graph=False, create_graph=False)
            with torch.no_grad():
                ctrl_norms = []
                for i, (c, g) in enumerate(zip(controls, grads)):
                    ms[i].mul_(beta1).add_(g, alpha=1.0 - beta1)
                    vs[i].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    mh = ms[i] / (1.0 - beta1 ** (it + 1))
                    vh = vs[i] / (1.0 - beta2 ** (it + 1))
                    c.add_(lr * mh / (vh.sqrt() + eps))
                    ctrl_norms.append(float(norm_per_sample(c).mean()))
                diag["ocflow_logprob"].append(float(logp.detach().mean()))
                diag["ocflow_prob"].append(float(prob.detach().mean()))
                diag["ocflow_control_norm"].append(mean_float(ctrl_norms))
            cost["guidance_time_sec"] += stop_timer(device, t_guidance); cost["guidance_calls"] += 1.0
        with torch.no_grad():
            x = integrate_vf_euler(model, x0.detach(), nfe, device, controls=[c.detach() for c in controls])
            cost["vf_batch_calls"] += float(nfe)
        all_ids.append(x.argmax(dim=-1).detach().cpu())
        left -= b
    ids = torch.cat(all_ids, dim=0)[:n_samples]
    texts = decode_tokens(tokenizer, ids)
    d = {f"guidance_{k}": mean_float(v) for k, v in diag.items() if v}
    d.update({
        "guidance_oc_steps": float(oc_steps),
        "guidance_oc_lr": float(lr),
        "guidance_control_reg": float(control_reg),
        "cost_guidance_time_sec": float(cost["guidance_time_sec"]),
        "cost_guidance_batch_calls": float(cost["guidance_calls"]),
        "cost_vf_batch_calls": float(cost["vf_batch_calls"]),
        "cost_guidance_time_per_call_ms": float(1000.0 * cost["guidance_time_sec"] / max(1.0, cost["guidance_calls"])),
        "cost_effective_candidate_nfe": float((oc_steps + 1) * nfe),
    })
    return ids, texts, d

def text_metrics(texts: List[str]) -> Dict[str, float]:
    lens = [len(t.split()) for t in texts]
    return {
        "avg_words": mean_float(lens) if lens else 0.0,
        "empty_rate": mean_float([1.0 if not t.strip() else 0.0 for t in texts]) if texts else 0.0,
        "uniq_text_rate": float(len(set(texts)) / max(1, len(texts))),
        "distinct2": mean_float([distinct_n(t, 2) for t in texts]) if texts else 0.0,
        "distinct3": mean_float([distinct_n(t, 3) for t in texts]) if texts else 0.0,
        "rep2": mean_float([ngram_rep_rate(t, 2) for t in texts]) if texts else 0.0,
        "rep3": mean_float([ngram_rep_rate(t, 3) for t in texts]) if texts else 0.0,
    }


def write_csv_from_jsonl(jsonl_path: str, csv_path: Optional[str] = None):
    if csv_path is None:
        csv_path = str(Path(jsonl_path).with_suffix(".csv"))
    rows = []
    keys = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def resolve_task_path(task: str, explicit: Dict[str, str], root: str, kind: str) -> str:
    if task in explicit:
        return explicit[task]
    d = TASK_DEFAULTS[task]["reward_dir" if kind == "reward" else "verifier_dir"]
    p = str(Path(root) / d)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="ag_news,cola,imdb,tweet_offensive")
    ap.add_argument("--methods", default="base,fmtg,fmrg,fmrg_sat,sfmrg_conflict_adapt,sat_pareto_quality")
    ap.add_argument("--nfes", default="1,2")
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--step_sizes", default="0.5,1.0")
    ap.add_argument("--mirror_etas", default="0.1,0.3")
    ap.add_argument("--mix_lambdas", default="0.2,0.4")
    ap.add_argument("--quality_lambdas", default="0.05,0.1,0.2")
    ap.add_argument("--sat_tau", type=float, default=0.30)
    ap.add_argument("--sat_kappa", type=float, default=0.10)
    ap.add_argument("--sat_power", type=float, default=1.0)
    ap.add_argument("--sat_floor", type=float, default=0.25)
    ap.add_argument("--entropy_weight", type=float, default=1.0)
    ap.add_argument("--basekl_weight", type=float, default=0.2)
    ap.add_argument("--schedule", default="constant", choices=["constant", "sqrt", "late"])
    ap.add_argument("--early_stop", type=float, default=1.0)
    ap.add_argument("--p_mode", default="softmax", choices=["auto", "softmax", "renorm"])
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--paired_eval", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reward_model_root", default="reward_models")
    ap.add_argument("--verifier_model_root", default="verifier_models")
    ap.add_argument("--reward_model_paths", default="")
    ap.add_argument("--verifier_model_paths", default="")
    ap.add_argument("--target_labels", default="")
    ap.add_argument("--ppl_model", default="gpt2-large")
    ap.add_argument("--skip_ppl", action="store_true")
    ap.add_argument("--skip_verifier", action="store_true")
    ap.add_argument("--local_files_only", action="store_true", default=True)
    ap.add_argument("--max_eval_length", type=int, default=256)
    ap.add_argument("--save_samples", action="store_true")
    ap.add_argument("--best_of_n", type=int, default=8)
    ap.add_argument("--reno_steps", type=int, default=8)
    ap.add_argument("--reno_lr", type=float, default=0.5)
    ap.add_argument("--delay_start", type=float, default=0.25)
    ap.add_argument("--proj_perp_scale", type=float, default=0.25)
    # Ordinary vector-field guidance baselines
    ap.add_argument("--source_steps", type=int, default=4)
    ap.add_argument("--source_lr", type=float, default=0.05)
    ap.add_argument("--source_reg", type=float, default=1e-4)
    ap.add_argument("--sgfm_beta", type=float, default=1.0)
    ap.add_argument("--oc_steps", type=int, default=4)
    ap.add_argument("--oc_lr", type=float, default=0.05)
    ap.add_argument("--control_reg", type=float, default=1e-4)
    ap.add_argument("--tree_active", type=int, default=2)
    ap.add_argument("--tree_branch", type=int, default=4)
    ap.add_argument("--tree_noise", type=float, default=0.05)
    ap.add_argument("--tree_value_rollout", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    print(f"[load] flow model: {args.ckpt}")
    flow, cfg = load_flow_model(args.run_dir, args.ckpt, device)
    print(f"[model] in_shape={tuple(flow.in_shape)}")

    # LM1B uses bert-base-uncased tokenizer in semicat.
    bert_tok_path = str(Path(args.reward_model_root) / "bert-base-uncased")
    try:
        gen_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    except Exception:
        # Fall back: any reward model tokenizer is also BERT-vocab.
        first_task = args.tasks.split(",")[0].strip()
        first_reward = resolve_task_path(first_task, parse_mapping(args.reward_model_paths), args.reward_model_root, "reward")
        gen_tokenizer = AutoTokenizer.from_pretrained(first_reward, local_files_only=args.local_files_only)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    nfes = [int(x) for x in args.nfes.split(",") if x.strip()]
    step_sizes = [float(x) for x in args.step_sizes.split(",") if x.strip()]
    mirror_etas = [float(x) for x in args.mirror_etas.split(",") if x.strip()]
    mix_lambdas = [float(x) for x in args.mix_lambdas.split(",") if x.strip()]
    quality_lambdas = [float(x) for x in args.quality_lambdas.split(",") if x.strip()]

    reward_paths = parse_mapping(args.reward_model_paths)
    verifier_paths = parse_mapping(args.verifier_model_paths)
    label_overrides = parse_label_mapping(args.target_labels)

    rewarders: Dict[str, SoftRewardModel] = {}
    verifiers: Dict[str, HardVerifier] = {}
    for task in tasks:
        if task not in TASK_DEFAULTS:
            raise ValueError(f"Unknown task {task}. Known: {sorted(TASK_DEFAULTS)}")
        label = label_overrides.get(task, TASK_DEFAULTS[task]["target_label"])
        rpath = resolve_task_path(task, reward_paths, args.reward_model_root, "reward")
        print(f"[load] guidance reward {task}: {rpath}, target_label={label}")
        rewarders[task] = SoftRewardModel(rpath, label, device, local_files_only=args.local_files_only)
        if not args.skip_verifier:
            vpath = resolve_task_path(task, verifier_paths, args.verifier_model_root, "verifier")
            if not Path(vpath).exists() and task not in verifier_paths:
                print(f"[warn] verifier path not found for {task}: {vpath}; falling back to guidance reward model")
                vpath = rpath
            print(f"[load] hard verifier {task}: {vpath}, target_label={label}")
            verifiers[task] = HardVerifier(vpath, label, device, local_files_only=args.local_files_only)

    ppl = None
    if not args.skip_ppl:
        print(f"[load] PPL model: {args.ppl_model}")
        ppl = GPT2PPL(args.ppl_model, device, local_files_only=args.local_files_only)

    total = 0
    for task in tasks:
        rewarder = rewarders[task]
        for nfe in nfes:
            for step_size in step_sizes:
                for method in methods:
                    etas = mirror_etas if method in {"sfmrg_conflict_adapt", "gap_aware"} else [0.0]
                    lambdas = mix_lambdas if method in {"sfmrg_conflict_adapt", "gap_aware"} else [0.0]
                    qlams = quality_lambdas if method in {"sat_pareto_quality", "gap_aware"} else [0.0]
                    for eta in etas:
                        for ml in lambdas:
                            for qlam in qlams:
                                if args.paired_eval:
                                    set_seed(stable_int_seed(args.seed, task, nfe, step_size, "paired"))
                                else:
                                    set_seed(stable_int_seed(args.seed, task, nfe, step_size, method, eta, ml, qlam))
                                print(f"[run] task={task} nfe={nfe} step={step_size} method={method} eta={eta} mix={ml} qlam={qlam}", flush=True)
                                reset_peak_vram(device)
                                row_start = start_timer(device)
                                gen_start = start_timer(device)
                                if method == "best_of_n":
                                    ids, texts, gdiag = sample_best_of_n(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.best_of_n,
                                        args.batch_size, nfe, device, p_mode=args.p_mode,
                                        max_eval_length=args.max_eval_length,
                                    )
                                elif method == "reno_init":
                                    ids, texts, gdiag = sample_reno_init(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.batch_size,
                                        nfe, args.reno_steps, args.reno_lr, device, p_mode=args.p_mode,
                                    )
                                elif method == "base_vf":
                                    ids, texts, gdiag = sample_base_vf(
                                        flow, gen_tokenizer, args.n_samples, args.batch_size, nfe, device,
                                    )
                                elif method == "universal_guidance":
                                    ids, texts, gdiag = sample_universal_guidance_vf(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.batch_size,
                                        nfe, step_size, device, p_mode=args.p_mode, schedule=args.schedule,
                                    )
                                elif method == "dflow":
                                    ids, texts, gdiag = sample_dflow_source(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.batch_size,
                                        nfe, args.source_steps, args.source_lr, args.source_reg,
                                        step_size, device, p_mode=args.p_mode,
                                    )
                                elif method == "sgfm":
                                    ids, texts, gdiag = sample_sgfm_langevin(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.batch_size,
                                        nfe, args.source_steps, args.source_lr, args.source_reg, args.sgfm_beta,
                                        step_size, device, p_mode=args.p_mode,
                                    )
                                elif method == "treeg":
                                    ids, texts, gdiag = sample_treeg_vf(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.batch_size,
                                        nfe, args.tree_active, args.tree_branch, args.tree_noise, args.tree_value_rollout,
                                        device, p_mode=args.p_mode,
                                    )
                                elif method == "ocflow":
                                    ids, texts, gdiag = sample_ocflow_controls(
                                        flow, gen_tokenizer, rewarder, args.n_samples, args.batch_size,
                                        nfe, args.oc_steps, args.oc_lr, args.control_reg,
                                        step_size, device, p_mode=args.p_mode,
                                    )
                                else:
                                    ids, texts, gdiag = sample_guided(
                                        flow, gen_tokenizer, rewarder, method, args.n_samples, args.batch_size,
                                        nfe, step_size, device, p_mode=args.p_mode, schedule=args.schedule,
                                        early_stop=args.early_stop, mirror_eta=eta, mix_lambda=ml,
                                        quality_lambda=qlam, sat_tau=args.sat_tau, sat_kappa=args.sat_kappa,
                                        sat_power=args.sat_power, sat_floor=args.sat_floor,
                                        entropy_weight=args.entropy_weight, basekl_weight=args.basekl_weight,
                                        delay_start=args.delay_start, proj_perp_scale=args.proj_perp_scale,
                                    )
                                generation_time_sec = stop_timer(device, gen_start)
                                row = {
                                    "seed": args.seed,
                                    "task": task,
                                    "task_display": TASK_DEFAULTS[task]["display"],
                                    "method": method,
                                    "nfe": nfe,
                                    "step_size": step_size,
                                    "mirror_eta": eta,
                                    "mix_lambda": ml,
                                    "quality_lambda": qlam,
                                    "n_samples": args.n_samples,
                                    "best_of_n": args.best_of_n if method == "best_of_n" else 0,
                                    "reno_steps": args.reno_steps if method == "reno_init" else 0,
                                    "reno_lr": args.reno_lr if method == "reno_init" else 0.0,
                                    "delay_start": args.delay_start if method == "delayed_fmrg" else 0.0,
                                    "proj_perp_scale": args.proj_perp_scale if method == "proj_fmrg" else 0.0,
                                    "source_steps": args.source_steps if method in {"dflow", "sgfm"} else 0,
                                    "source_lr": args.source_lr if method in {"dflow", "sgfm"} else 0.0,
                                    "source_reg": args.source_reg if method in {"dflow", "sgfm"} else 0.0,
                                    "sgfm_beta": args.sgfm_beta if method == "sgfm" else 0.0,
                                    "oc_steps": args.oc_steps if method == "ocflow" else 0,
                                    "oc_lr": args.oc_lr if method == "ocflow" else 0.0,
                                    "control_reg": args.control_reg if method == "ocflow" else 0.0,
                                    "tree_active": args.tree_active if method == "treeg" else 0,
                                    "tree_branch": args.tree_branch if method == "treeg" else 0,
                                    "tree_noise": args.tree_noise if method == "treeg" else 0.0,
                                    "tree_value_rollout": args.tree_value_rollout if method == "treeg" else 0,
                                    "cost_generation_time_sec": float(generation_time_sec),
                                    "cost_generation_time_per_sample_sec": float(generation_time_sec / max(1, args.n_samples)),
                                    "cost_peak_vram_gb_after_generation": peak_vram_gb(device),
                                }
                                row.update(gdiag)
                                row.update(text_metrics(texts))
                                verifier_time_sec = 0.0
                                ppl_time_sec = 0.0
                                if not args.skip_verifier:
                                    eval_start = start_timer(device)
                                    row.update(verifiers[task].score(texts, batch_size=args.eval_batch_size, max_length=args.max_eval_length))
                                    verifier_time_sec = stop_timer(device, eval_start)
                                if ppl is not None:
                                    eval_start = start_timer(device)
                                    row.update(ppl.score(texts, batch_size=max(1, min(args.eval_batch_size, 4)), max_length=args.max_eval_length))
                                    ppl_time_sec = stop_timer(device, eval_start)
                                row["cost_verifier_time_sec"] = float(verifier_time_sec)
                                row["cost_ppl_time_sec"] = float(ppl_time_sec)
                                row["cost_row_total_time_sec"] = float(stop_timer(device, row_start))
                                row["cost_row_total_time_per_sample_sec"] = float(row["cost_row_total_time_sec"] / max(1, args.n_samples))
                                row["cost_peak_vram_gb"] = peak_vram_gb(device)
                                with open(args.out, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                if args.save_samples:
                                    sample_path = str(Path(args.out).with_suffix("")) + f"_seed{args.seed}_{task}_{method}_nfe{nfe}_step{step_size}_eta{eta}_mix{ml}_q{qlam}.txt"
                                    with open(sample_path, "w", encoding="utf-8") as sf:
                                        for t in texts[: min(32, len(texts))]:
                                            sf.write(t.replace("\n", " ") + "\n")
                                total += 1
    csv_path = write_csv_from_jsonl(args.out)
    print(f"[done] rows={total}")
    print(f"[done] jsonl={args.out}")
    print(f"[done] csv={csv_path}")


if __name__ == "__main__":
    main()
