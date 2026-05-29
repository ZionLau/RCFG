#!/usr/bin/env python3
"""Select LM1B hyperparameters from grid JSONL files and write replay jobs.

Selection protocol: for each task-method-NFE, average reward over the three seeds
for each hyperparameter setting; use target_rate as a tie-breaker. The generated
replay jobs remove --skip_ppl and compute Reward/Target/NLL/PPL for the selected
configuration under each seed.
"""
import argparse, glob, json, shlex
from collections import defaultdict
from pathlib import Path

HYP_FIELDS = [
    'step_size',
    'mirror_eta', 'mix_lambda', 'quality_lambda',
    'source_steps', 'source_lr', 'source_reg', 'sgfm_beta',
    'tree_active', 'tree_branch', 'tree_noise', 'tree_value_rollout',
    'sat_tau', 'sat_kappa', 'sat_power', 'sat_floor',
]

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(','.join(fields) + '\n')
        for r in rows:
            vals = []
            for k in fields:
                v = str(r.get(k, ''))
                v = v.replace('"', '""')
                if ',' in v or '\n' in v:
                    v = f'"{v}"'
                vals.append(v)
            fp.write(','.join(vals) + '\n')

def add_arg(parts, name, value):
    if value is not None and str(value) != '':
        parts += [name, str(value)]

def method_args(r):
    m = r['method']
    parts = []
    add_arg(parts, '--step_sizes', r.get('step_size', '0.5'))
    if m == 'gap_aware':
        add_arg(parts, '--mirror_etas', r.get('mirror_eta'))
        add_arg(parts, '--mix_lambdas', r.get('mix_lambda'))
        add_arg(parts, '--quality_lambdas', r.get('quality_lambda'))
        add_arg(parts, '--sat_tau', r.get('sat_tau', '0.30'))
        add_arg(parts, '--sat_kappa', r.get('sat_kappa', '0.10'))
        add_arg(parts, '--sat_power', r.get('sat_power', '1.0'))
        add_arg(parts, '--sat_floor', r.get('sat_floor', '0.25'))
    if m in {'dflow', 'sgfm'}:
        add_arg(parts, '--source_steps', r.get('source_steps'))
        add_arg(parts, '--source_lr', r.get('source_lr'))
        add_arg(parts, '--source_reg', r.get('source_reg'))
    if m == 'sgfm':
        add_arg(parts, '--sgfm_beta', r.get('sgfm_beta'))
    if m == 'treeg':
        add_arg(parts, '--tree_active', r.get('tree_active'))
        add_arg(parts, '--tree_branch', r.get('tree_branch'))
        add_arg(parts, '--tree_noise', r.get('tree_noise'))
        add_arg(parts, '--tree_value_rollout', r.get('tree_value_rollout'))
    return parts

p = argparse.ArgumentParser()
p.add_argument('--grid_dir', required=True)
p.add_argument('--out_root', required=True)
p.add_argument('--jobs_out', required=True)
p.add_argument('--script', default='scripts/lm1b_attr_guidance_vf_baselines_treeg_nonumpy_cost.py')
p.add_argument('--python_bin', default='/root/miniconda3/bin/python')
p.add_argument('--run_dir', required=True)
p.add_argument('--ckpt', required=True)
p.add_argument('--reward_root', required=True)
p.add_argument('--verifier_root', required=True)
p.add_argument('--ppl_model', required=True)
p.add_argument('--seeds', default='3407,3408,3409')
p.add_argument('--batch_size', default='8')
p.add_argument('--eval_batch_size', default='16')
p.add_argument('--n_samples', default='512')
args = p.parse_args()

grid = Path(args.grid_dir)
out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)
selected_jsonl = out_root / 'selected_configs_reward.jsonl'
selected_csv = out_root / 'selected_configs_reward.csv'
replay_raw = out_root / 'replay_raw'
replay_raw.mkdir(parents=True, exist_ok=True)

rows = []
for pth in sorted(grid.glob('*.jsonl')):
    with open(pth, encoding='utf-8') as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
print('[load rows]', len(rows))
if not rows:
    raise SystemExit('No grid jsonl rows found.')

groups = defaultdict(list)
for r in rows:
    key = (str(r.get('task')), str(r.get('method')), int(float(r.get('nfe'))))
    hyp = tuple((k, str(r.get(k, ''))) for k in HYP_FIELDS if k in r and str(r.get(k, '')) != '')
    groups[(key, hyp)].append(r)

best = {}
for (key, hyp), rs in groups.items():
    reward = sum(f(r.get('reward')) for r in rs) / max(1, len(rs))
    target = sum(f(r.get('target_rate')) for r in rs) / max(1, len(rs))
    cand = (reward, target, len(rs), hyp, rs[0])
    if key not in best or cand[:3] > best[key][:3]:
        best[key] = cand

selected = []
for key in sorted(best, key=lambda x: (x[1], x[0], x[2])):
    reward, target, n, hyp, ex = best[key]
    r = dict(ex)
    r['selected_mean_reward'] = reward
    r['selected_mean_target_rate'] = target
    r['selected_n_rows'] = n
    selected.append(r)

with open(selected_jsonl, 'w', encoding='utf-8') as fp:
    for r in selected:
        fp.write(json.dumps(r, ensure_ascii=False) + '\n')
write_csv(selected_csv, selected)
print('[selected]', len(selected))
print('[write]', selected_jsonl)
print('[write]', selected_csv)

seeds = [x.strip() for x in args.seeds.split(',') if x.strip()]
jobs = []
for r in selected:
    task = str(r['task'])
    method = str(r['method'])
    nfe = str(int(float(r['nfe'])))
    for seed in seeds:
        out = replay_raw / f'{method}_{task}_nfe{nfe}_seed{seed}.jsonl'
        parts = [
            args.python_bin, args.script,
            '--run_dir', args.run_dir,
            '--ckpt', args.ckpt,
            '--out', str(out),
            '--tasks', task,
            '--methods', method,
            '--nfes', nfe,
            '--n_samples', args.n_samples,
            '--batch_size', args.batch_size,
            '--eval_batch_size', args.eval_batch_size,
            '--schedule', 'constant',
            '--early_stop', '1.0',
            '--p_mode', 'softmax',
            '--seed', seed,
            '--reward_model_root', args.reward_root,
            '--verifier_model_root', args.verifier_root,
            '--ppl_model', args.ppl_model,
            '--local_files_only',
            '--device', 'cuda',
        ]
        parts += method_args(r)
        jobs.append(' '.join(shlex.quote(x) for x in parts))

jobs_out = Path(args.jobs_out)
jobs_out.parent.mkdir(parents=True, exist_ok=True)
jobs_out.write_text('\n'.join(jobs) + '\n', encoding='utf-8')
print('[write]', jobs_out)
print('[replay jobs]', len(jobs))
