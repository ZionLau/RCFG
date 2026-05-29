import argparse
import json
import os
import random
import importlib.util
from pathlib import Path

import numpy as np
import torch


# ------------------------------------------------------------
# Import shared utilities from the previous FMRG script.
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "text8_fmrg_semicat_eval.py"

spec = importlib.util.spec_from_file_location("fmrg_base", str(BASE_SCRIPT))
fmrg_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmrg_base)


set_seed = fmrg_base.set_seed
load_model = fmrg_base.load_model
get_data_dir = fmrg_base.get_data_dir
load_meta = fmrg_base.load_meta
build_bigram_scorer = fmrg_base.build_bigram_scorer

target_ids_from_word = fmrg_base.target_ids_from_word
sample_base_flowmap = fmrg_base.sample_base_flowmap
sample_best_of_k = fmrg_base.sample_best_of_k
evaluate_tokens = fmrg_base.evaluate_tokens
write_samples = fmrg_base.write_samples
emit_row = fmrg_base.emit_row

endpoint_reward = fmrg_base.endpoint_reward
norm_per_sample = fmrg_base.norm_per_sample


def fmrg_weight(step_size, dt, current_t, schedule):
    if schedule == "paper":
        # Practical FMRG-style scheduling: larger early guidance, weaker near t=1.
        return step_size * dt * max(0.0, 1.0 - current_t)
    if schedule == "dt":
        return step_size * dt
    if schedule == "constant":
        return step_size
    raise ValueError(f"Unknown schedule: {schedule}")


def sample_fmrg_pre(
    model,
    n_samples,
    batch_size,
    nfe,
    variant,
    step_size,
    inner_steps,
    final_steps,
    early_stop,
    target_ids,
    log_bigram,
    device,
    reward_mix,
    schedule,
    e_normalize,
    j_velocity_rescale,
):
    """
    Fixed Semicat FMRG.

    Key fix:
      Previous implementation did:
        x_t = Phi_{s->t}(x_s)
        then guide at t.
      Therefore NFE=1 had no guidance.

      This version does:
        guide x_s using endpoint lookahead Phi_{s->1}(x_s)
        then x_t = Phi_{s->t}(x_s_guided)

    This makes NFE=1 a real one-step guided flow-map sample.
    """
    outs = []
    left = n_samples

    diag_reward = []
    diag_update_norm = []
    diag_signal_norm = []
    diag_vel_norm = []
    diag_gamma = []

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
            one_vec = torch.ones((b,), device=device)

            if s_float >= early_stop and early_stop < 1.0:
                with torch.no_grad():
                    x = model.xst(x, s_vec, one_vec)
                break

            # Base velocity magnitude for FMRG-J rescaling.
            with torch.no_grad():
                base_next = model.xst(x.detach(), s_vec, t_vec)
                vel_norm = norm_per_sample((base_next - x.detach()) / dt).detach()

            gamma = fmrg_weight(step_size, dt, s_float, schedule)
            diag_gamma.append(float(gamma))

            # ------------------------------------------------------------
            # 1) Pre-flow-map FMRG guidance at x_s.
            # ------------------------------------------------------------
            if gamma > 0:
                for _ in range(inner_steps):
                    if variant == "fmrg_e":
                        # FMRG-E-style: endpoint Euclidean reward gradient.
                        # We compute dR/dx_1 and apply it in the same categorical state space.
                        with torch.no_grad():
                            endpoint0 = model.xst(x.detach(), s_vec, one_vec)

                        endpoint = endpoint0.detach().requires_grad_(True)
                        reward_vec = endpoint_reward(endpoint, target_ids, log_bigram, reward_mix)
                        signal = torch.autograd.grad(
                            reward_vec.mean(),
                            endpoint,
                            retain_graph=False,
                            create_graph=False,
                        )[0]

                        if e_normalize:
                            signal = signal / norm_per_sample(signal)

                    elif variant == "fmrg_j":
                        # FMRG-J-style: backprop through flow map:
                        # d/dx_s R(Phi_{s->1}(x_s)).
                        x_req = x.detach().requires_grad_(True)
                        endpoint = model.xst(x_req, s_vec, one_vec)
                        reward_vec = endpoint_reward(endpoint, target_ids, log_bigram, reward_mix)
                        signal = torch.autograd.grad(
                            reward_vec.mean(),
                            x_req,
                            retain_graph=False,
                            create_graph=False,
                        )[0]

                        if j_velocity_rescale:
                            signal = signal / norm_per_sample(signal) * vel_norm

                    else:
                        raise ValueError(f"Unknown variant: {variant}")

                    with torch.no_grad():
                        update = gamma * signal.detach()
                        x = x.detach() + update

                        diag_reward.append(float(reward_vec.detach().mean().item()))
                        diag_signal_norm.append(float(norm_per_sample(signal.detach()).mean().item()))
                        diag_update_norm.append(float(norm_per_sample(update).mean().item()))
                        diag_vel_norm.append(float(vel_norm.mean().item()))

            # ------------------------------------------------------------
            # 2) Flow-map step.
            # ------------------------------------------------------------
            with torch.no_grad():
                x = model.xst(x.detach(), s_vec, t_vec)

        # Optional endpoint cleanup. Default = 0.
        # This directly optimizes final endpoint and is useful only as a diagnostic.
        if final_steps > 0:
            one_vec = torch.ones((b,), device=device)
            for _ in range(final_steps):
                endpoint = x.detach().requires_grad_(True)
                reward_vec = endpoint_reward(endpoint, target_ids, log_bigram, reward_mix)
                signal = torch.autograd.grad(
                    reward_vec.mean(),
                    endpoint,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                if e_normalize:
                    signal = signal / norm_per_sample(signal)
                with torch.no_grad():
                    gamma = step_size * 0.1
                    update = gamma * signal.detach()
                    x = x.detach() + update
                    diag_reward.append(float(reward_vec.detach().mean().item()))
                    diag_signal_norm.append(float(norm_per_sample(signal.detach()).mean().item()))
                    diag_update_norm.append(float(norm_per_sample(update).mean().item()))

        outs.append(x.argmax(dim=-1).detach().cpu())
        left -= b

    diagnostics = {
        "diag_reward_mean": float(np.mean(diag_reward)) if diag_reward else None,
        "diag_signal_norm_mean": float(np.mean(diag_signal_norm)) if diag_signal_norm else None,
        "diag_update_norm_mean": float(np.mean(diag_update_norm)) if diag_update_norm else None,
        "diag_vel_norm_mean": float(np.mean(diag_vel_norm)) if diag_vel_norm else None,
        "diag_gamma_mean": float(np.mean(diag_gamma)) if diag_gamma else None,
    }
    return torch.cat(outs, dim=0), diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--n_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--nfes", default="1,2,4,8,16")
    p.add_argument("--targets", default="award")
    p.add_argument("--methods", default="base,bestof,fmrg_e,fmrg_j")

    p.add_argument("--best_of_k", type=int, default=8)
    p.add_argument("--step_sizes", default="0.03,0.1,0.3,1.0,3.0")
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--final_steps", type=int, default=0)
    p.add_argument("--early_stop", type=float, default=1.0)
    p.add_argument("--schedule", default="paper", choices=["paper", "dt", "constant"])
    p.add_argument("--reward_mix", type=float, default=0.05)

    p.add_argument("--e_normalize", action="store_true")
    p.add_argument("--j_no_velocity_rescale", action="store_true")

    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_samples", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.run_dir, args.ckpt, device)

    data_dir = get_data_dir(cfg)
    meta = load_meta(data_dir)
    vocab_size = int(meta["vocab_size"])
    stoi = meta["stoi"]
    itos = meta["itos"]
    log_bigram = build_bigram_scorer(data_dir, vocab_size, device)

    nfes = [int(x) for x in args.nfes.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    step_sizes = [float(x) for x in args.step_sizes.split(",") if x.strip()]

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        for target in targets:
            target_ids = target_ids_from_word(target, stoi)

            for nfe in nfes:
                if "base" in methods:
                    print(f"\n### target={target} nfe={nfe} base_flowmap")
                    tok = sample_base_flowmap(model, args.n_samples, args.batch_size, nfe)
                    row = {
                        "target": target,
                        "method": "base_flowmap",
                        "nfe": nfe,
                        "n_samples": args.n_samples,
                    }
                    row.update(evaluate_tokens(tok, log_bigram, vocab_size, target, itos))
                    emit_row(f, row)

                if "bestof" in methods:
                    print(f"\n### target={target} nfe={nfe} best_of_{args.best_of_k}")
                    tok = sample_best_of_k(
                        model, args.n_samples, args.batch_size, nfe,
                        args.best_of_k, target, itos, log_bigram, args.reward_mix
                    )
                    row = {
                        "target": target,
                        "method": f"best_of_{args.best_of_k}",
                        "nfe": nfe,
                        "k": args.best_of_k,
                        "n_samples": args.n_samples,
                    }
                    row.update(evaluate_tokens(tok, log_bigram, vocab_size, target, itos))
                    emit_row(f, row)

                for method in ["fmrg_e", "fmrg_j"]:
                    if method not in methods:
                        continue

                    for step_size in step_sizes:
                        print(f"\n### target={target} nfe={nfe} {method} step_size={step_size}")
                        tok, diag = sample_fmrg_pre(
                            model=model,
                            n_samples=args.n_samples,
                            batch_size=args.batch_size,
                            nfe=nfe,
                            variant=method,
                            step_size=step_size,
                            inner_steps=args.inner_steps,
                            final_steps=args.final_steps,
                            early_stop=args.early_stop,
                            target_ids=target_ids,
                            log_bigram=log_bigram,
                            device=device,
                            reward_mix=args.reward_mix,
                            schedule=args.schedule,
                            e_normalize=bool(args.e_normalize),
                            j_velocity_rescale=(not args.j_no_velocity_rescale),
                        )

                        row = {
                            "target": target,
                            "method": method + "_pre",
                            "nfe": nfe,
                            "step_size": step_size,
                            "inner_steps": args.inner_steps,
                            "final_steps": args.final_steps,
                            "early_stop": args.early_stop,
                            "schedule": args.schedule,
                            "e_normalize": bool(args.e_normalize),
                            "j_velocity_rescale": bool(not args.j_no_velocity_rescale),
                            "n_samples": args.n_samples,
                        }
                        row.update(evaluate_tokens(tok, log_bigram, vocab_size, target, itos))
                        row.update(diag)
                        emit_row(f, row)

                        if args.save_samples and nfe in [1, 4, 16]:
                            write_samples(
                                args.out.replace(".jsonl", f"_{target}_{method}_pre_nfe{nfe}_s{step_size}.txt"),
                                tok,
                                itos,
                            )

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
