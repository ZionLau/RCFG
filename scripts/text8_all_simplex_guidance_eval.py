#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differentiable few-step guidance evaluation on text8 for flow-map models.

Methods compared:
  - fmtg: FMLM/Table-16-style ambient terminal reward gradient, no flow-map Jacobian pullback.
  - fmrg: terminal reward with flow-map Jacobian pullback.
  - smfg: pure Simplex Mirror Flow Guidance, using the same differentiable terminal reward.
  - sfmrg_mix: FMRG + simplex-mirror correction with a fixed mixing coefficient.
  - sfmrg_adapt: satisfaction-aware FMRG + simplex-mirror correction.
  - sfmrg_trust: FMRG reward ascent with a simplex trust-region to the unguided endpoint.

This script does NOT use black-box guidance, best-of-k, reranking, rejection sampling,
or hard non-differentiable rewards during guidance. The task reward signal is always
computed from the same differentiable reward on the endpoint simplex.  The sfmrg_*
methods keep the same task reward as FMRG and only change the geometry/control of
the guidance direction.

Expected location:
  Put this file under: /root/autodl-tmp/semicat/scripts/text8_diff_guidance_eval.py
It reuses the model-loading utilities from:
  scripts/text8_fmrg_semicat_fixed_eval.py
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "text8_fmrg_semicat_fixed_eval.py"

spec = importlib.util.spec_from_file_location("fmrg_fixed", str(BASE_SCRIPT))
fmrg_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmrg_fixed)

set_seed = fmrg_fixed.set_seed
load_model = fmrg_fixed.load_model
get_data_dir = fmrg_fixed.get_data_dir
load_meta = fmrg_fixed.load_meta
build_bigram_scorer = fmrg_fixed.build_bigram_scorer
target_ids_from_word = fmrg_fixed.target_ids_from_word
norm_per_sample = fmrg_fixed.norm_per_sample
fmrg_weight = fmrg_fixed.fmrg_weight


# -----------------------------
# simplex utilities
# -----------------------------


def to_simplex(x: torch.Tensor, mode: str = "auto", eps: float = 1e-8) -> torch.Tensor:
    """Convert relaxed model state/logits to a per-position simplex."""
    if mode == "softmax":
        return torch.softmax(x, dim=-1)

    if mode == "renorm":
        y = x.clamp_min(eps)
        return y / y.sum(dim=-1, keepdim=True).clamp_min(eps)

    if mode == "auto":
        with torch.no_grad():
            x_det = x.detach()
            mn = float(x_det.min().item())
            mx = float(x_det.max().item())
            sum_err = float((x_det.sum(dim=-1) - 1.0).abs().mean().item())
        if mn >= -1e-4 and mx <= 1.5 and sum_err < 0.5:
            y = x.clamp_min(eps)
            return y / y.sum(dim=-1, keepdim=True).clamp_min(eps)
        return torch.softmax(x, dim=-1)

    raise ValueError(f"Unknown p_mode: {mode}")


def token_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(p.clamp_min(eps) * torch.log(p.clamp_min(eps))).sum(dim=-1)


def expected_bigram_score(p: torch.Tensor, log_bigram: torch.Tensor) -> torch.Tensor:
    """Differentiable expected bigram log-score under adjacent soft token distributions."""
    return torch.einsum("blv,vw,blw->bl", p[:, :-1, :], log_bigram, p[:, 1:, :]).mean(dim=1)


def state_diagnostics(x: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        s = x.sum(dim=-1)
        return {
            "state_sum_error": float((s - 1.0).abs().mean().item()),
            "state_negative_mass": float(torch.relu(-x).mean().item()),
            "state_above_one_mass": float(torch.relu(x - 1.0).mean().item()),
            "state_min": float(x.min().item()),
            "state_max": float(x.max().item()),
        }


def unit_per_sample(g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to unit norm independently for each batch element."""
    shape = [g.shape[0]] + [1] * (g.ndim - 1)
    n = g.reshape(g.shape[0], -1).norm(dim=1).view(*shape).clamp_min(eps)
    return g / n


def cosine_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    aa = a.reshape(a.shape[0], -1)
    bb = b.reshape(b.shape[0], -1)
    return (aa * bb).sum(dim=1) / (aa.norm(dim=1).clamp_min(eps) * bb.norm(dim=1).clamp_min(eps))


# -----------------------------
# differentiable lexical-event reward
# -----------------------------


def word_logq_and_q(p: torch.Tensor, target_ids: Sequence[int], eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft occurrence of one target word over all possible slots.

    For word w=(w_1,...,w_m), at slot r:
      log q_r = (1/m) sum_j log p_{r+j-1,w_j}
      q_r = exp(log q_r)

    Shapes:
      p:    [B, L, V]
      logq: [B, L-m+1]
      q:    [B, L-m+1]
    """
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


def boundary_score(p: torch.Tensor, target_ids: Sequence[int], log_bigram: torch.Tensor) -> torch.Tensor:
    """
    Differentiable local boundary naturalness for each possible insertion slot.

    It estimates the expected text8 bigram compatibility of:
      left-context -> first target char
      last target char -> right-context

    Shape: [B, L-m+1]
    """
    B, L, V = p.shape
    m = len(target_ids)
    first = int(target_ids[0])
    last = int(target_ids[-1])

    scores = []
    for r in range(L - m + 1):
        parts = []
        if r > 0:
            left = torch.matmul(p[:, r - 1, :], log_bigram[:, first])
            parts.append(left)
        if r + m < L:
            right = torch.matmul(p[:, r + m, :], log_bigram[last, :])
            parts.append(right)
        if parts:
            scores.append(sum(parts) / len(parts))
        else:
            scores.append(torch.zeros(B, device=p.device, dtype=p.dtype))
    return torch.stack(scores, dim=1)


def lexical_event_reward_p(
    p: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    tau_slot: float = 0.2,
    alpha_event: float = 1.0,
    alpha_slot: float = 0.25,
    rho_dup: float = 0.5,
    dup_gamma: float = 1.0,
    boundary_alpha: float = 0.05,
    reward_mix: float = 0.02,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Differentiable text8 lexical-event reward.

    Given a target set W={w_1,...,w_M}, define soft slot probabilities q_{r,w}.
    The event probability is:
      P_evt = 1 - prod_{w in W} prod_r (1 - q_{r,w})

    The final utility is:
      U = alpha_event * log(P_evt)
          + alpha_slot * selected-slot log-prob
          - rho_dup * P_evt^gamma * non-selected soft occurrence mass
          + reward_mix * expected_bigram_score

    This reward is fully differentiable w.r.t. p.  It encourages the lexical
    event to be satisfied at least once, discourages repeated target insertion,
    and adds a small local-fluency term.
    """
    all_logq = []
    all_q = []
    all_b = []
    for ids in target_id_lists:
        logq, q = word_logq_and_q(p, ids, eps=eps)
        all_logq.append(logq)
        all_q.append(q)
        if boundary_alpha != 0.0:
            all_b.append(boundary_score(p, ids, log_bigram))
        else:
            all_b.append(torch.zeros_like(logq))

    logq_cat = torch.cat(all_logq, dim=1)  # [B, total_slots]
    q_cat = torch.cat(all_q, dim=1)        # [B, total_slots]
    b_cat = torch.cat(all_b, dim=1)        # [B, total_slots]

    # Probability that at least one target word occurs somewhere.
    log_no_event = torch.log1p(-q_cat.clamp(0.0, 1.0 - 1e-6)).sum(dim=1)
    P_event = (1.0 - torch.exp(log_no_event)).clamp(eps, 1.0 - 1e-6)

    # Total soft occurrence mass.  C≈1 is desirable; C>>1 means duplicated target stuffing.
    C = q_cat.sum(dim=1)

    # Soft winner slot/word assignment.  Detaching pi prevents the reward from
    # gaming the assignment distribution itself; gradients still flow through logq and q.
    slot_score = logq_cat + boundary_alpha * b_cat
    pi = torch.softmax(slot_score / max(tau_slot, 1e-6), dim=1)
    pi_sg = pi.detach()

    selected_logq = (pi_sg * logq_cat).sum(dim=1)
    winner_q = (pi_sg * q_cat).sum(dim=1)
    nonwinner_q = ((1.0 - pi_sg) * q_cat).sum(dim=1)

    R_event = alpha_event * torch.log(P_event.clamp_min(eps))
    R_slot = alpha_slot * selected_logq
    dup_gate = P_event.detach().pow(dup_gamma)
    R_dup = -rho_dup * dup_gate * nonwinner_q
    R_bi = reward_mix * expected_bigram_score(p, log_bigram)

    R = R_event + R_slot + R_dup + R_bi

    pi_entropy = -(pi_sg.clamp_min(eps) * torch.log(pi_sg.clamp_min(eps))).sum(dim=1)
    diag = {
        "reward": R.detach(),
        "P_event": P_event.detach(),
        "C": C.detach(),
        "winner_q": winner_q.detach(),
        "nonwinner_q": nonwinner_q.detach(),
        "dup_gate": dup_gate.detach(),
        "pi_entropy": pi_entropy.detach(),
        "p_entropy": token_entropy(p).mean(dim=1).detach(),
        "bigram_soft": R_bi.detach() / max(reward_mix, eps) if reward_mix != 0 else expected_bigram_score(p, log_bigram).detach(),
    }
    return R, diag


# -----------------------------
# guidance directions
# -----------------------------


def compute_fmtg_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    FMTG/FMLM-style ambient terminal reward gradient.

    It computes the reward gradient at the predicted terminal state y_s, but does
    not backpropagate through y_s = Phi_{s->1}(x_s).  The terminal ambient gradient
    is directly used as a perturbation direction for x_s.
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)

    with torch.no_grad():
        endpoint = model.xst(x.detach(), s_vec, one_vec)

    y_req = endpoint.detach().requires_grad_(True)
    p = to_simplex(y_req, mode=p_mode)
    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)
    direction = torch.autograd.grad(R_vec.mean(), y_req, retain_graph=False, create_graph=False)[0]

    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    return direction.detach(), diag


def compute_fmrg_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """FMRG: reward gradient pulled back through the remaining flow map."""
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)

    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)
    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)
    direction = torch.autograd.grad(R_vec.mean(), x_req, retain_graph=False, create_graph=False)[0]

    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    return direction.detach(), diag


def compute_smfg_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mirror_grad_clip: float = 10.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SMFG: Simplex Mirror Flow Guidance.

    1. Look ahead to the terminal simplex p_s = Pi(Phi_{s->1}(x_s)).
    2. Compute differentiable reward gradient r_s = dU/dp_s.
    3. Build an entropic mirror target:
         q* = Normalize(p_s * exp(eta * centered(r_s))).
    4. Pull back KL(sg(q*) || p_s) through the flow map.
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)

    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)

    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)

    # Gradient wrt endpoint simplex.  This is NOT a black-box signal; it is the
    # exact differentiable reward gradient on p.
    r = torch.autograd.grad(R_vec.sum(), p, retain_graph=True, create_graph=False)[0]

    with torch.no_grad():
        p_det = p.detach().clamp_min(eps)
        r_det = r.detach()
        # Centering makes the update invariant to reward-gradient offsets over vocab.
        r_det = r_det - r_det.mean(dim=-1, keepdim=True)
        if mirror_grad_clip is not None and mirror_grad_clip > 0:
            r_det = r_det.clamp(-mirror_grad_clip, mirror_grad_clip)
        logits = torch.log(p_det) + mirror_eta * r_det
        q_star = torch.softmax(logits, dim=-1)

    logp = torch.log(p.clamp_min(eps))
    logq = torch.log(q_star.clamp_min(eps))
    kl_pos = (q_star * (logq - logp)).sum(dim=-1)  # [B,L]
    kl = kl_pos.mean(dim=1)                       # [B]

    direction = -torch.autograd.grad(kl.mean(), x_req, retain_graph=False, create_graph=False)[0]

    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    diag["mirror_kl"] = float(kl.detach().mean().item())
    diag["mirror_delta_l1"] = float((q_star - p.detach()).abs().sum(dim=-1).mean().item())
    diag["mirror_target_entropy"] = float(token_entropy(q_star).mean().item())
    return direction.detach(), diag


# -----------------------------
# sampling and evaluation
# -----------------------------




def compute_sfmrg_mix_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mix_lambda: float = 0.2,
    adaptive: bool = False,
    sat_tau: float = 0.7,
    sat_kappa: float = 0.08,
    mirror_grad_clip: float = 10.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Simplex-corrected FMRG.

    This method uses the SAME terminal reward as FMRG.  It does not add a
    keyword-specific reward and it does not rerank samples.  It keeps the strong
    direct reward-ascent direction of FMRG and adds only a simplex-mirror
    geometric correction:

      g_R = d/dx_s U(Pi(Phi_{s->1}(x_s)))
      g_M = -d/dx_s KL(sg(q*) || Pi(Phi_{s->1}(x_s)))
      q*  = Normalize(p * exp(eta * dU/dp))

    Fixed mix:
      g = (1-lambda) unit(g_R) + lambda unit(g_M)

    Adaptive mix:
      lambda_i = lambda_max * sigmoid((P_event_i - sat_tau) / sat_kappa)

    The adaptive version leaves low-satisfaction states close to FMRG, and only
    increases simplex correction after the terminal event is likely satisfied.
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)

    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)

    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)

    g_reward = torch.autograd.grad(R_vec.mean(), x_req, retain_graph=True, create_graph=False)[0]

    r = torch.autograd.grad(R_vec.sum(), p, retain_graph=True, create_graph=False)[0]
    with torch.no_grad():
        p_det = p.detach().clamp_min(eps)
        r_det = r.detach()
        r_det = r_det - r_det.mean(dim=-1, keepdim=True)
        if mirror_grad_clip is not None and mirror_grad_clip > 0:
            r_det = r_det.clamp(-mirror_grad_clip, mirror_grad_clip)
        q_star = torch.softmax(torch.log(p_det) + mirror_eta * r_det, dim=-1)

    kl_pos = (q_star * (torch.log(q_star.clamp_min(eps)) - torch.log(p.clamp_min(eps)))).sum(dim=-1)
    kl = kl_pos.mean(dim=1)
    g_mirror = -torch.autograd.grad(kl.mean(), x_req, retain_graph=False, create_graph=False)[0]

    if adaptive:
        # text8 satisfaction score.  In LM1B this becomes target-class probability.
        sat = rdiag.get("P_event", torch.zeros((b,), device=x.device)).detach()
        lam = mix_lambda * torch.sigmoid((sat - sat_tau) / max(sat_kappa, 1e-6))
    else:
        lam = torch.full((b,), float(mix_lambda), device=x.device)
    lam_view = lam.view(b, *([1] * (x.ndim - 1)))

    direction = (1.0 - lam_view) * unit_per_sample(g_reward) + lam_view * unit_per_sample(g_mirror)

    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    diag["mirror_kl"] = float(kl.detach().mean().item())
    diag["mirror_delta_l1"] = float((q_star - p.detach()).abs().sum(dim=-1).mean().item())
    diag["mirror_target_entropy"] = float(token_entropy(q_star).mean().item())
    diag["mix_lambda"] = float(lam.detach().mean().item())
    diag["mix_lambda_min"] = float(lam.detach().min().item())
    diag["mix_lambda_max"] = float(lam.detach().max().item())
    diag["reward_mirror_cos"] = float(cosine_per_sample(g_reward.detach(), g_mirror.detach()).mean().item())
    return direction.detach(), diag


def compute_sfmrg_trust_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    trust_beta: float = 0.1,
    trust_entropy: float = 0.0,
    adaptive: bool = True,
    sat_tau: float = 0.7,
    sat_kappa: float = 0.08,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    FMRG with simplex trust region to the unguided endpoint.

    Same terminal task reward U as FMRG, but the update direction is computed from

      J = U(p_guided) - beta_s KL(p_guided || sg(p_base)) + xi H(p_guided),

    where p_base is the endpoint predicted from the current state without an
    extra guidance perturbation.  beta_s is optionally satisfaction-gated, so the
    trust region is weak before the target is satisfied and stronger after it.

    This is NOT event-specific: for LM1B, U is the classifier target reward and
    the same trust-region term can be used unchanged.
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)

    with torch.no_grad():
        endpoint_base = model.xst(x.detach(), s_vec, one_vec)
        p_base = to_simplex(endpoint_base, mode=p_mode).detach().clamp_min(eps)

    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)

    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)

    kl_base = (p.clamp_min(eps) * (torch.log(p.clamp_min(eps)) - torch.log(p_base))).sum(dim=-1).mean(dim=1)
    ent = token_entropy(p).mean(dim=1)

    if adaptive:
        sat = rdiag.get("P_event", torch.zeros((b,), device=x.device)).detach()
        beta = trust_beta * torch.sigmoid((sat - sat_tau) / max(sat_kappa, 1e-6))
    else:
        beta = torch.full((b,), float(trust_beta), device=x.device)

    J = R_vec - beta * kl_base + float(trust_entropy) * ent
    direction = torch.autograd.grad(J.mean(), x_req, retain_graph=False, create_graph=False)[0]

    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    diag["trust_kl_base"] = float(kl_base.detach().mean().item())
    diag["trust_beta"] = float(beta.detach().mean().item())
    diag["trust_beta_min"] = float(beta.detach().min().item())
    diag["trust_beta_max"] = float(beta.detach().max().item())
    diag["trust_entropy"] = float(ent.detach().mean().item())
    return direction.detach(), diag



# -----------------------------
# additional simplex-consistent guidance variants
# -----------------------------


def per_position_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-token-position norm over the vocabulary dimension. Shape [B,L,1]."""
    return x.norm(dim=-1, keepdim=True).clamp_min(eps)


def project_orthogonal_per_sample(a: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Remove from a the per-sample component parallel to ref."""
    B = a.shape[0]
    af = a.reshape(B, -1)
    rf = ref.reshape(B, -1)
    coef = (af * rf).sum(dim=1, keepdim=True) / (rf.pow(2).sum(dim=1, keepdim=True).clamp_min(eps))
    out = af - coef * rf
    return out.reshape_as(a)


def simplex_mean_center(g: torch.Tensor) -> torch.Tensor:
    """Project an ambient vocabulary-direction to the per-position simplex tangent subspace."""
    return g - g.mean(dim=-1, keepdim=True)


def compute_reward_and_mirror_core(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mirror_grad_clip: float = 10.0,
    mirror_eps: float = 0.0,
    eps: float = 1e-8,
):
    """
    Compute the two reusable directions under the SAME terminal reward:
      g_reward: FMRG reward pullback
      g_mirror: simplex mirror correction pullback

    mirror_eps smooths the multiplicative mirror base:
      p_bar = (1-eps) p + eps / V
    This avoids the low-probability barrier of naive mirror descent.
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)
    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)

    g_reward = torch.autograd.grad(R_vec.mean(), x_req, retain_graph=True, create_graph=False)[0]
    r = torch.autograd.grad(R_vec.sum(), p, retain_graph=True, create_graph=False)[0]

    with torch.no_grad():
        p_det = p.detach().clamp_min(eps)
        if mirror_eps and mirror_eps > 0:
            V = p_det.shape[-1]
            p_base = (1.0 - float(mirror_eps)) * p_det + float(mirror_eps) / float(V)
            p_base = p_base / p_base.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            p_base = p_det
        r_det = r.detach()
        r_det = r_det - r_det.mean(dim=-1, keepdim=True)
        if mirror_grad_clip is not None and mirror_grad_clip > 0:
            r_det = r_det.clamp(-mirror_grad_clip, mirror_grad_clip)
        q_star = torch.softmax(torch.log(p_base.clamp_min(eps)) + mirror_eta * r_det, dim=-1)

    kl_pos = (q_star * (torch.log(q_star.clamp_min(eps)) - torch.log(p.clamp_min(eps)))).sum(dim=-1)
    kl = kl_pos.mean(dim=1)
    g_mirror = -torch.autograd.grad(kl.mean(), x_req, retain_graph=False, create_graph=False)[0]

    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    diag["mirror_kl"] = float(kl.detach().mean().item())
    diag["mirror_delta_l1"] = float((q_star - p.detach()).abs().sum(dim=-1).mean().item())
    diag["mirror_target_entropy"] = float(token_entropy(q_star).mean().item())
    diag["mirror_eps"] = float(mirror_eps)
    diag["reward_mirror_cos"] = float(cosine_per_sample(g_reward.detach(), g_mirror.detach()).mean().item())
    return g_reward, g_mirror, R_vec, rdiag, diag



def compute_smfg_eps_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mirror_grad_clip: float = 10.0,
    mirror_eps: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pure SMFG with epsilon-smoothed mirror base."""
    _gR, gM, _R, _rdiag, diag = compute_reward_and_mirror_core(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        mirror_eta=mirror_eta, mirror_grad_clip=mirror_grad_clip, mirror_eps=mirror_eps,
    )
    return gM.detach(), diag

def compute_sfmrg_orth_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mix_lambda: float = 0.2,
    mirror_grad_clip: float = 10.0,
    mirror_eps: float = 0.0,
    adaptive: bool = False,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Orthogonal Simplex Correction.

    Instead of replacing FMRG, keep the reward-ascent direction g_R and only add
    the component of the mirror correction that is orthogonal to g_R:
        g = unit(g_R) + lambda * unit(g_M - proj_{g_R} g_M)

    This targets the simplex geometry without reducing first-order reward ascent.
    """
    g_R, g_M, R_vec, rdiag, diag = compute_reward_and_mirror_core(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        mirror_eta=mirror_eta, mirror_grad_clip=mirror_grad_clip, mirror_eps=mirror_eps,
    )
    g_M_perp = project_orthogonal_per_sample(g_M, g_R)
    b = x.shape[0]
    if adaptive:
        sat = rdiag.get("P_event", torch.zeros((b,), device=x.device)).detach()
        lam = mix_lambda * torch.sigmoid((sat - sat_tau) / max(sat_kappa, 1e-6))
    else:
        lam = torch.full((b,), float(mix_lambda), device=x.device)
    lam_view = lam.view(b, *([1] * (x.ndim - 1)))
    direction = unit_per_sample(g_R) + lam_view * unit_per_sample(g_M_perp)
    diag["mix_lambda"] = float(lam.detach().mean().item())
    diag["orth_cos_after"] = float(cosine_per_sample(g_R.detach(), g_M_perp.detach()).mean().item())
    diag["orth_perp_norm_ratio"] = float((norm_per_sample(g_M_perp) / norm_per_sample(g_M).clamp_min(1e-8)).mean().item())
    return direction.detach(), diag


def compute_sfmrg_conflict_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    mirror_eta: float = 0.5,
    mix_lambda: float = 0.2,
    mirror_grad_clip: float = 10.0,
    mirror_eps: float = 0.0,
    adaptive: bool = False,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Conflict-Gated Simplex Correction.

    If mirror correction agrees with reward ascent, add it directly. If it
    conflicts, use only its component orthogonal to g_R. This preserves FMRG's
    reward-seeking ability while still correcting simplex geometry.
    """
    g_R, g_M, R_vec, rdiag, diag = compute_reward_and_mirror_core(
        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
        mirror_eta=mirror_eta, mirror_grad_clip=mirror_grad_clip, mirror_eps=mirror_eps,
    )
    b = x.shape[0]
    cos = cosine_per_sample(g_R.detach(), g_M.detach())
    g_M_perp = project_orthogonal_per_sample(g_M, g_R)
    use_perp = (cos < 0).view(b, *([1] * (x.ndim - 1)))
    g_corr = torch.where(use_perp, g_M_perp, g_M)

    if adaptive:
        sat = rdiag.get("P_event", torch.zeros((b,), device=x.device)).detach()
        lam = mix_lambda * torch.sigmoid((sat - sat_tau) / max(sat_kappa, 1e-6))
    else:
        lam = torch.full((b,), float(mix_lambda), device=x.device)
    lam_view = lam.view(b, *([1] * (x.ndim - 1)))
    direction = unit_per_sample(g_R) + lam_view * unit_per_sample(g_corr)
    diag["mix_lambda"] = float(lam.detach().mean().item())
    diag["conflict_rate"] = float((cos < 0).float().mean().item())
    diag["conflict_cos_min"] = float(cos.min().item())
    diag["conflict_cos_max"] = float(cos.max().item())
    return direction.detach(), diag


def compute_fmrg_saturation_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """FMRG direction plus a per-sample step-scale (1 - satisfaction)^alpha."""
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)
    R_vec, rdiag = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)
    direction = torch.autograd.grad(R_vec.mean(), x_req, retain_graph=False, create_graph=False)[0]
    sat = rdiag.get("P_event", torch.zeros((b,), device=x.device)).detach().clamp(0, 1)
    scale = float(sat_floor) + (1.0 - float(sat_floor)) * (1.0 - sat).pow(float(sat_power))
    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_vec.detach().mean().item())
    diag["sat_step_scale"] = float(scale.mean().item())
    diag["_sample_scale"] = scale
    return direction.detach(), diag


def compute_tptr_direction(
    model,
    x: torch.Tensor,
    s_vec: torch.Tensor,
    target_id_lists: Sequence[Sequence[int]],
    log_bigram: torch.Tensor,
    p_mode: str,
    reward_kwargs: Dict,
    gamma: float,
    vel_norm: torch.Tensor,
    trust_beta: float = 0.1,
    trust_entropy: float = 0.0,
    sat_tau: float = 0.30,
    sat_kappa: float = 0.10,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Two-Pass Terminal Trust Region.

    First form a tentative FMRG update, then compute a terminal objective at the
    tentative point:
      U(p_tent) - beta KL(p_tent || sg(p_base)) + entropy_bonus H(p_tent)
    The returned direction is a correction direction evaluated at the tentative point.
    """
    b = x.shape[0]
    one_vec = torch.ones((b,), device=x.device)
    # Base endpoint and reward direction at current x.
    with torch.no_grad():
        endpoint_base = model.xst(x.detach(), s_vec, one_vec)
        p_base = to_simplex(endpoint_base, mode=p_mode).detach().clamp_min(eps)

    x_req = x.detach().requires_grad_(True)
    endpoint = model.xst(x_req, s_vec, one_vec)
    p = to_simplex(endpoint, mode=p_mode)
    R_vec, rdiag0 = lexical_event_reward_p(p, target_id_lists, log_bigram, **reward_kwargs)
    g_R = torch.autograd.grad(R_vec.mean(), x_req, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        x_tent = x.detach() + float(gamma) * unit_per_sample(g_R.detach()) * vel_norm

    x_tent_req = x_tent.detach().requires_grad_(True)
    endpoint_t = model.xst(x_tent_req, s_vec, one_vec)
    p_t = to_simplex(endpoint_t, mode=p_mode)
    R_t, rdiag = lexical_event_reward_p(p_t, target_id_lists, log_bigram, **reward_kwargs)
    kl_base = (p_t.clamp_min(eps) * (torch.log(p_t.clamp_min(eps)) - torch.log(p_base))).sum(dim=-1).mean(dim=1)
    ent = token_entropy(p_t).mean(dim=1)
    sat = rdiag.get("P_event", torch.zeros((b,), device=x.device)).detach()
    beta = float(trust_beta) * torch.sigmoid((sat - sat_tau) / max(sat_kappa, 1e-6))
    J = R_t - beta * kl_base + float(trust_entropy) * ent
    direction = torch.autograd.grad(J.mean(), x_tent_req, retain_graph=False, create_graph=False)[0]
    diag = {k: float(v.mean().item()) for k, v in rdiag.items()}
    diag["guidance_reward"] = float(R_t.detach().mean().item())
    diag["tptr_kl_base"] = float(kl_base.detach().mean().item())
    diag["tptr_beta"] = float(beta.detach().mean().item())
    diag["tptr_entropy"] = float(ent.detach().mean().item())
    return direction.detach(), diag

def decode_one(tok: Sequence[int], itos) -> str:
    chars = []
    for i in tok:
        ii = int(i)
        if isinstance(itos, dict):
            chars.append(str(itos[ii]))
        else:
            chars.append(str(itos[ii]))
    return "".join(chars)


def count_occurrences(text: str, words: Sequence[str]) -> int:
    return sum(text.count(w) for w in words)


def distinct_n_for_text(text: str, n: int) -> float:
    if len(text) < n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def rep_ngram_rate_for_text(text: str, n: int) -> float:
    return 1.0 - distinct_n_for_text(text, n)


def hard_bigram_score(tokens: torch.Tensor, log_bigram: torch.Tensor) -> float:
    """Mean hard argmax bigram log-score. tokens: [N,L] on CPU or GPU."""
    if tokens.numel() == 0 or tokens.shape[1] < 2:
        return float("nan")
    t = tokens.to(log_bigram.device).long()
    vals = log_bigram[t[:, :-1], t[:, 1:]]
    return float(vals.mean().detach().cpu().item())


def evaluate_lexical_tokens(
    tokens: torch.Tensor,
    target_words: Sequence[str],
    itos,
    log_bigram: torch.Tensor,
) -> Tuple[Dict[str, float], List[str]]:
    texts = [decode_one(row.tolist(), itos) for row in tokens.cpu()]
    counts = np.array([count_occurrences(t, target_words) for t in texts], dtype=np.float64)
    lens = np.array([len(t) for t in texts], dtype=np.float64)

    metrics = {
        "hit_rate": float((counts >= 1).mean()),
        "exact_once_rate": float((counts == 1).mean()),
        "oversat_rate": float((counts >= 2).mean()),
        "target_count_mean": float(counts.mean()),
        "target_count_std": float(counts.std()),
        "hard_bigram": hard_bigram_score(tokens, log_bigram),
        "distinct2": float(np.mean([distinct_n_for_text(t, 2) for t in texts])),
        "distinct3": float(np.mean([distinct_n_for_text(t, 3) for t in texts])),
        "rep2": float(np.mean([rep_ngram_rate_for_text(t, 2) for t in texts])),
        "rep3": float(np.mean([rep_ngram_rate_for_text(t, 3) for t in texts])),
        "uniq_text_rate": float(len(set(texts)) / max(1, len(texts))),
        "avg_len": float(lens.mean()),
    }
    return metrics, texts


def reduce_diag(diag_lists: Dict[str, List[float]], prefix: str) -> Dict[str, float]:
    out = {}
    for k, vals in diag_lists.items():
        if len(vals) == 0:
            continue
        out[f"{prefix}_{k}_mean"] = float(np.mean(vals))
    return out


def sample_guided(
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
    sat_tau: float = 0.7,
    sat_kappa: float = 0.08,
    trust_beta: float = 0.1,
    trust_entropy: float = 0.0,
    mirror_eps: float = 0.02,
    sat_power: float = 1.0,
    sat_floor: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run base / fmtg / fmrg / smfg / simplex-corrected FMRG samplers."""
    method = method.lower()
    valid_methods = {"base", "fmtg", "fmrg", "fmrg_posnorm", "fmrg_sat", "smfg", "smfg_eps", "sfmrg_mix", "sfmrg_adapt", "sfmrg_trust", "sfmrg_tptr", "sfmrg_orth", "sfmrg_orth_adapt", "sfmrg_orth_eps", "sfmrg_orth_eps_adapt", "sfmrg_conflict", "sfmrg_conflict_adapt", "sfmrg_conflict_eps", "sfmrg_conflict_eps_adapt", "sfmrg_conflict_posnorm", "sfmrg_conflict_eps_posnorm"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")

    outs = []
    left = n_samples

    diag_lists: Dict[str, List[float]] = {
        "guidance_reward": [],
        "P_event": [],
        "C": [],
        "winner_q": [],
        "nonwinner_q": [],
        "dup_gate": [],
        "pi_entropy": [],
        "p_entropy": [],
        "bigram_soft": [],
        "direction_norm": [],
        "update_norm": [],
        "vel_norm": [],
        "update_over_vel": [],
        "gamma": [],
        "state_sum_error": [],
        "state_negative_mass": [],
        "state_above_one_mass": [],
        "state_min": [],
        "state_max": [],
        "mirror_kl": [],
        "mirror_delta_l1": [],
        "mirror_target_entropy": [],
        "mix_lambda": [],
        "mix_lambda_min": [],
        "mix_lambda_max": [],
        "reward_mirror_cos": [],
        "trust_kl_base": [],
        "trust_beta": [],
        "trust_beta_min": [],
        "trust_beta_max": [],
        "trust_entropy": [],
        "mirror_eps": [],
        "orth_cos_after": [],
        "orth_perp_norm_ratio": [],
        "conflict_rate": [],
        "conflict_cos_min": [],
        "conflict_cos_max": [],
        "sat_step_scale": [],
        "tptr_kl_base": [],
        "tptr_beta": [],
        "tptr_entropy": [],
    }

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
                    direction, d = compute_fmtg_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
                    )
                elif method in {"fmrg", "fmrg_posnorm"}:
                    direction, d = compute_fmrg_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs
                    )
                elif method == "fmrg_sat":
                    direction, d = compute_fmrg_saturation_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                        sat_power=sat_power, sat_floor=sat_floor,
                    )
                elif method in {"smfg", "smfg_eps"}:
                    if method == "smfg_eps":
                        direction, d = compute_smfg_eps_direction(
                            model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                            mirror_eta=mirror_eta, mirror_grad_clip=mirror_grad_clip, mirror_eps=mirror_eps,
                        )
                    else:
                        direction, d = compute_smfg_direction(
                            model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                            mirror_eta=mirror_eta, mirror_grad_clip=mirror_grad_clip,
                        )
                elif method in {"sfmrg_mix", "sfmrg_adapt"}:
                    direction, d = compute_sfmrg_mix_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                        mirror_eta=mirror_eta, mix_lambda=mix_lambda,
                        adaptive=(method == "sfmrg_adapt"),
                        sat_tau=sat_tau, sat_kappa=sat_kappa,
                        mirror_grad_clip=mirror_grad_clip,
                    )
                elif method == "sfmrg_trust":
                    direction, d = compute_sfmrg_trust_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                        trust_beta=trust_beta, trust_entropy=trust_entropy,
                        adaptive=True, sat_tau=sat_tau, sat_kappa=sat_kappa,
                    )
                elif method == "sfmrg_tptr":
                    direction, d = compute_tptr_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                        gamma=gamma, vel_norm=vel_norm, trust_beta=trust_beta,
                        trust_entropy=trust_entropy, sat_tau=sat_tau, sat_kappa=sat_kappa,
                    )
                elif method in {"sfmrg_orth", "sfmrg_orth_adapt", "sfmrg_orth_eps", "sfmrg_orth_eps_adapt"}:
                    direction, d = compute_sfmrg_orth_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                        mirror_eta=mirror_eta, mix_lambda=mix_lambda,
                        mirror_grad_clip=mirror_grad_clip,
                        mirror_eps=(mirror_eps if "eps" in method else 0.0),
                        adaptive=method.endswith("adapt"),
                        sat_tau=sat_tau, sat_kappa=sat_kappa,
                    )
                elif method in {"sfmrg_conflict", "sfmrg_conflict_adapt", "sfmrg_conflict_eps", "sfmrg_conflict_eps_adapt", "sfmrg_conflict_posnorm", "sfmrg_conflict_eps_posnorm"}:
                    direction, d = compute_sfmrg_conflict_direction(
                        model, x, s_vec, target_id_lists, log_bigram, p_mode, reward_kwargs,
                        mirror_eta=mirror_eta, mix_lambda=mix_lambda,
                        mirror_grad_clip=mirror_grad_clip,
                        mirror_eps=(mirror_eps if "eps" in method else 0.0),
                        adaptive=("adapt" in method),
                        sat_tau=sat_tau, sat_kappa=sat_kappa,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                sample_scale = d.pop("_sample_scale", None) if isinstance(d, dict) else None
                with torch.no_grad():
                    use_pos_norm = method.endswith("posnorm") or method in {"fmrg_posnorm"}
                    if use_pos_norm:
                        d_norm = per_position_norm(direction).detach().clamp_min(1e-8)
                        v_norm_local = per_position_norm(vel).detach().clamp_min(1e-8)
                        update = gamma * direction.detach() / d_norm * v_norm_local
                        d_norm_for_log = norm_per_sample(direction).detach().clamp_min(1e-8)
                    else:
                        d_norm = norm_per_sample(direction).detach().clamp_min(1e-8)
                        update = gamma * direction.detach() / d_norm * vel_norm
                        d_norm_for_log = d_norm
                    if sample_scale is not None:
                        scale_view = sample_scale.to(update.device).view(b, *([1] * (update.ndim - 1)))
                        update = update * scale_view
                    x = x.detach() + update

                    diag_lists["direction_norm"].append(float(d_norm_for_log.mean().item()))
                    diag_lists["update_norm"].append(float(norm_per_sample(update).mean().item()))
                    diag_lists["vel_norm"].append(float(vel_norm.mean().item()))
                    diag_lists["update_over_vel"].append(float((norm_per_sample(update) / vel_norm).mean().item()))
                    diag_lists["gamma"].append(float(gamma))
                    for k, v in d.items():
                        if k in diag_lists:
                            diag_lists[k].append(float(v))
                    sd = state_diagnostics(x)
                    for k, v in sd.items():
                        diag_lists[k].append(float(v))
            else:
                diag_lists["gamma"].append(0.0)

            with torch.no_grad():
                x = model.xst(x.detach(), s_vec, t_vec)

        outs.append(x.argmax(dim=-1).detach().cpu())
        left -= b

    tokens = torch.cat(outs, dim=0)
    diagnostics = reduce_diag(diag_lists, prefix=method)
    return tokens, diagnostics


# -----------------------------
# task parsing and IO
# -----------------------------


def parse_target_sets(s: str) -> List[Tuple[str, List[str]]]:
    """
    Parse target sets.

    Format:
      name:word1|word2;name2:word3|word4

    If no colon is given, the name is the word itself:
      award,city,game
    """
    out: List[Tuple[str, List[str]]] = []
    if not s.strip():
        return out

    # Allow comma-separated single words as a convenience.
    raw_items = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block and "|" not in block and "," in block:
            raw_items.extend([x.strip() for x in block.split(",") if x.strip()])
        else:
            raw_items.append(block)

    for item in raw_items:
        item = item.strip().lower()
        if not item:
            continue
        if ":" in item:
            name, words = item.split(":", 1)
            name = name.strip()
            word_list = [w.strip().lower() for w in words.replace(",", "|").split("|") if w.strip()]
        else:
            name = item
            word_list = [w.strip().lower() for w in item.replace(",", "|").split("|") if w.strip()]
        if not word_list:
            raise ValueError(f"No words found for target set: {item}")
        out.append((name, word_list))
    return out


def ids_for_words(words: Sequence[str], stoi) -> List[List[int]]:
    return [target_ids_from_word(w, stoi) for w in words]


def append_jsonl(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_csv_from_jsonl(jsonl_path: str, csv_path: str) -> None:
    rows = []
    keys = []
    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_sample_texts(path: str, texts: Sequence[str], max_samples: int = 64) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts[:max_samples]):
            f.write(f"[{i}] {t}\n")


# -----------------------------
# main
# -----------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Hydra run dir containing .hydra/config.yaml")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--out", required=True, help="Output jsonl path")

    p.add_argument("--n_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--nfes", default="1,2,4,8,16")
    p.add_argument("--methods", default="fmrg,smfg,smfg_eps,sfmrg_adapt,sfmrg_orth,sfmrg_conflict,sfmrg_conflict_eps,sfmrg_tptr,fmrg_posnorm,fmrg_sat", help="Comma list. Recommended search: fmrg,smfg,smfg_eps,sfmrg_adapt,sfmrg_orth,sfmrg_orth_eps,sfmrg_conflict,sfmrg_conflict_eps,sfmrg_conflict_adapt,sfmrg_conflict_eps_adapt,sfmrg_tptr,fmrg_posnorm,sfmrg_conflict_posnorm,fmrg_sat")
    p.add_argument(
        "--target_sets",
        default="award:award;city:city|cities|town;game:game|team|player|match;music:music|song|album|band;science:science|research|computer|system",
        help="Format: name:word1|word2;name2:word3|word4 . Single words also allowed.",
    )

    p.add_argument("--step_sizes", default="0.25,0.5,1.0,2.0")
    p.add_argument("--mirror_etas", default="0.1,0.3,0.5,1.0")
    p.add_argument("--mirror_grad_clip", type=float, default=10.0)
    p.add_argument("--mirror_eps", type=float, default=0.02, help="epsilon smoothing for eps-mirror variants")
    p.add_argument("--mix_lambdas", default="0.1,0.2,0.3", help="lambda values for sfmrg_mix/sfmrg_adapt")
    p.add_argument("--sat_tau", type=float, default=0.7, help="satisfaction threshold for adaptive simplex correction")
    p.add_argument("--sat_kappa", type=float, default=0.08, help="temperature for adaptive simplex correction")
    p.add_argument("--trust_betas", default="0.05,0.1,0.2", help="trust-region beta values for sfmrg_trust")
    p.add_argument("--trust_entropy", type=float, default=0.0, help="optional endpoint entropy bonus for sfmrg_trust/tptr")
    p.add_argument("--sat_power", type=float, default=1.0, help="power for fmrg_sat step scale")
    p.add_argument("--sat_floor", type=float, default=0.25, help="minimum step scale for fmrg_sat")
    p.add_argument("--early_stops", default="1.0")
    p.add_argument("--schedule", default="paper", choices=["paper", "dt", "constant"])
    p.add_argument("--p_mode", default="auto", choices=["auto", "softmax", "renorm"])

    # Differentiable reward hyperparameters. Same reward is used by all methods.
    p.add_argument("--tau_slot", type=float, default=0.2)
    p.add_argument("--alpha_event", type=float, default=1.0)
    p.add_argument("--alpha_slot", type=float, default=0.25)
    p.add_argument("--rho_dup", type=float, default=0.5)
    p.add_argument("--dup_gamma", type=float, default=1.0)
    p.add_argument("--boundary_alpha", type=float, default=0.05)
    p.add_argument("--reward_mix", type=float, default=0.02)

    p.add_argument("--seed", type=int, default=3407)
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
    trust_betas = [float(x) for x in args.trust_betas.split(",") if x.strip()]
    early_stops = [float(x) for x in args.early_stops.split(",") if x.strip()]
    target_sets = parse_target_sets(args.target_sets)

    if not target_sets:
        raise ValueError("No target sets specified.")
    for m in methods:
        if m not in {"base", "fmtg", "fmrg", "fmrg_posnorm", "fmrg_sat", "smfg", "smfg_eps", "sfmrg_mix", "sfmrg_adapt", "sfmrg_trust", "sfmrg_tptr", "sfmrg_orth", "sfmrg_orth_adapt", "sfmrg_orth_eps", "sfmrg_orth_eps_adapt", "sfmrg_conflict", "sfmrg_conflict_adapt", "sfmrg_conflict_eps", "sfmrg_conflict_eps_adapt", "sfmrg_conflict_posnorm", "sfmrg_conflict_eps_posnorm"}:
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

    print("Differentiable reward: lexical event probability + duplicate penalty + soft bigram fluency")
    print("Methods:", methods)
    print("Target sets:", target_sets)
    print("NFE:", nfes)

    for target_name, target_words in target_sets:
        target_id_lists = ids_for_words(target_words, stoi)
        print(f"\n===== target_set={target_name} words={target_words} =====")

        for nfe in nfes:
            print(f"\n--- NFE={nfe} ---")

            # Optional base reference. It is not a guidance method.
            if "base" in methods:
                print(f"### base target={target_name} nfe={nfe}")
                tok, diag = sample_guided(
                    model=model,
                    n_samples=args.n_samples,
                    batch_size=args.batch_size,
                    nfe=nfe,
                    method="base",
                    step_size=0.0,
                    target_id_lists=target_id_lists,
                    log_bigram=log_bigram,
                    device=device,
                    schedule=args.schedule,
                    early_stop=0.0,
                    p_mode=args.p_mode,
                    reward_kwargs=reward_kwargs,
                )
                metrics, texts = evaluate_lexical_tokens(tok, target_words, itos, log_bigram)
                row = {
                    "target_set": target_name,
                    "target_words": "|".join(target_words),
                    "method": "base",
                    "nfe": nfe,
                    "n_samples": args.n_samples,
                    "seed": args.seed,
                }
                row.update(metrics)
                row.update(diag)
                append_jsonl(args.out, row)
                if args.save_samples:
                    sp = args.out.replace(".jsonl", f"_{target_name}_base_nfe{nfe}.txt")
                    write_sample_texts(sp, texts, args.max_saved_samples)

            for early_stop in early_stops:
                for step_size in step_sizes:
                    for method in methods:
                        if method == "base":
                            continue

                        # Hyperparameter grid.  FMRG/FMTG have no extra geometry parameter.
                        # smfg uses eta only. sfmrg_mix/adapt use eta x lambda. sfmrg_trust uses beta.
                        grid = []
                        if method in {"smfg", "smfg_eps"}:
                            for eta in mirror_etas:
                                grid.append(dict(mirror_eta=eta, mix_lambda=None, trust_beta=None))
                        elif method in {"sfmrg_mix", "sfmrg_adapt", "sfmrg_orth", "sfmrg_orth_adapt", "sfmrg_orth_eps", "sfmrg_orth_eps_adapt", "sfmrg_conflict", "sfmrg_conflict_adapt", "sfmrg_conflict_eps", "sfmrg_conflict_eps_adapt", "sfmrg_conflict_posnorm", "sfmrg_conflict_eps_posnorm"}:
                            for eta in mirror_etas:
                                for lam in mix_lambdas:
                                    grid.append(dict(mirror_eta=eta, mix_lambda=lam, trust_beta=None))
                        elif method in {"sfmrg_trust", "sfmrg_tptr"}:
                            for beta in trust_betas:
                                grid.append(dict(mirror_eta=None, mix_lambda=None, trust_beta=beta))
                        else:
                            grid.append(dict(mirror_eta=None, mix_lambda=None, trust_beta=None))

                        for hp in grid:
                            mirror_eta = hp["mirror_eta"]
                            mix_lambda = hp["mix_lambda"]
                            trust_beta = hp["trust_beta"]
                            if method in {"smfg", "smfg_eps"}:
                                label = f"{method}_eta{mirror_eta:g}"
                            elif method in {"sfmrg_mix", "sfmrg_adapt", "sfmrg_orth", "sfmrg_orth_adapt", "sfmrg_orth_eps", "sfmrg_orth_eps_adapt", "sfmrg_conflict", "sfmrg_conflict_adapt", "sfmrg_conflict_eps", "sfmrg_conflict_eps_adapt", "sfmrg_conflict_posnorm", "sfmrg_conflict_eps_posnorm"}:
                                label = f"{method}_eta{mirror_eta:g}_lam{mix_lambda:g}"
                            elif method in {"sfmrg_trust", "sfmrg_tptr"}:
                                label = f"{method}_beta{trust_beta:g}"
                            else:
                                label = method
                            print(
                                f"### {label} target={target_name} nfe={nfe} "
                                f"step={step_size} early={early_stop}"
                            )

                            tok, diag = sample_guided(
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
                                mirror_eta=float(mirror_eta) if mirror_eta is not None else 0.0,
                                mirror_grad_clip=args.mirror_grad_clip,
                                mix_lambda=float(mix_lambda) if mix_lambda is not None else 0.0,
                                sat_tau=args.sat_tau,
                                sat_kappa=args.sat_kappa,
                                trust_beta=float(trust_beta) if trust_beta is not None else 0.0,
                                trust_entropy=args.trust_entropy,
                                mirror_eps=args.mirror_eps,
                                sat_power=args.sat_power,
                                sat_floor=args.sat_floor,
                            )
                            metrics, texts = evaluate_lexical_tokens(tok, target_words, itos, log_bigram)
                            row = {
                                "target_set": target_name,
                                "target_words": "|".join(target_words),
                                "method": method,
                                "nfe": nfe,
                                "step_size": step_size,
                                "early_stop": early_stop,
                                "mirror_eta": mirror_eta,
                                "mix_lambda": mix_lambda,
                                "trust_beta": trust_beta,
                                "sat_tau": args.sat_tau,
                                "sat_kappa": args.sat_kappa,
                                "mirror_eps_arg": args.mirror_eps,
                                "trust_entropy_weight": args.trust_entropy,
                                "sat_power": args.sat_power,
                                "sat_floor": args.sat_floor,
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
                            row.update(diag)
                            append_jsonl(args.out, row)

                            if args.save_samples:
                                extra = ""
                                if mirror_eta is not None:
                                    extra += f"_eta{mirror_eta:g}"
                                if mix_lambda is not None:
                                    extra += f"_lam{mix_lambda:g}"
                                if trust_beta is not None:
                                    extra += f"_beta{trust_beta:g}"
                                sp = args.out.replace(
                                    ".jsonl",
                                    f"_{target_name}_{method}_nfe{nfe}_s{step_size:g}{extra}.txt",
                                )
                                write_sample_texts(sp, texts, args.max_saved_samples)

    csv_path = args.out.replace(".jsonl", ".csv")
    write_csv_from_jsonl(args.out, csv_path)
    print("\nSaved JSONL:", args.out)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
