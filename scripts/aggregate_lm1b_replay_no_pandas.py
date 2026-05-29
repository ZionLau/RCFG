#!/usr/bin/env python3
"""Aggregate selected LM1B replay JSONL files without pandas/numpy."""
import argparse, json, math, csv, re
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--replay_dir', required=True)
p.add_argument('--out_dir', required=True)
args = p.parse_args()
replay = Path(args.replay_dir)
outdir = Path(args.out_dir)
outdir.mkdir(parents=True, exist_ok=True)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) <= 1:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def pick(row, names):
    for n in names:
        if n in row and row[n] not in ('', None):
            return to_float(row[n])
    return None

def fmt_pm(m, s, nd=3):
    if m is None:
        return ''
    if s is None:
        s = 0.0
    return f'{m:.{nd}f}±{s:.{nd}f}'

rows = []
for pth in sorted(replay.glob('*.jsonl')):
    with open(pth, encoding='utf-8') as fp:
        for line in fp:
            if line.strip():
                r = json.loads(line)
                r['_file'] = pth.name
                rows.append(r)
print('[loaded rows]', len(rows))
if not rows:
    raise SystemExit('No replay rows found.')
print('[example keys]', sorted(rows[0].keys()))

norm = []
for r in rows:
    seed = str(r.get('seed', ''))
    if not seed:
        m = re.search(r'seed(\d+)', str(r.get('_file', '')))
        seed = m.group(1) if m else ''
    rr = dict(r)
    rr['seed'] = seed
    rr['task'] = str(r.get('task', ''))
    rr['method'] = str(r.get('method', ''))
    rr['nfe'] = str(int(float(r.get('nfe', 0)))) if str(r.get('nfe', '')) else ''
    rr['_reward'] = pick(r, ['reward', 'verifier_reward', 'target_reward'])
    rr['_target'] = pick(r, ['target_rate', 'target', 'success_rate'])
    rr['_nll'] = pick(r, ['nll', 'gen_nll', 'lm_nll', 'ppl_nll'])
    rr['_ppl'] = pick(r, ['ppl', 'gen_ppl', 'lm_ppl', 'perplexity'])
    norm.append(rr)

dedup = {}
for r in norm:
    key = (r['seed'], r['task'], r['method'], r['nfe'])
    if key not in dedup or (r['_reward'] or -1e9) > (dedup[key]['_reward'] or -1e9):
        dedup[key] = r
norm = list(dedup.values())
print('[dedup rows]', len(norm))

group = defaultdict(list)
for r in norm:
    group[(r['task'], r['method'], r['nfe'])].append(r)

taskwise = []
for (task, method, nfe), rs in sorted(group.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
    taskwise.append({
        'task': task, 'method': method, 'nfe': nfe, 'n_seeds': len(rs),
        'reward_mean': mean([r['_reward'] for r in rs]), 'reward_std': std([r['_reward'] for r in rs]),
        'target_mean': mean([r['_target'] for r in rs]), 'target_std': std([r['_target'] for r in rs]),
        'nll_mean': mean([r['_nll'] for r in rs]), 'nll_std': std([r['_nll'] for r in rs]),
        'ppl_mean': mean([r['_ppl'] for r in rs]), 'ppl_std': std([r['_ppl'] for r in rs]),
    })

group2 = defaultdict(list)
for r in taskwise:
    group2[(r['method'], r['nfe'])].append(r)
methodwise = []
for (method, nfe), rs in sorted(group2.items(), key=lambda x: (x[0][1], x[0][0])):
    methodwise.append({
        'method': method, 'nfe': nfe, 'n_tasks': len(rs),
        'reward_mean': mean([r['reward_mean'] for r in rs]), 'reward_std': mean([r['reward_std'] for r in rs]),
        'target_mean': mean([r['target_mean'] for r in rs]), 'target_std': mean([r['target_std'] for r in rs]),
        'nll_mean': mean([r['nll_mean'] for r in rs]), 'nll_std': mean([r['nll_std'] for r in rs]),
        'ppl_mean': mean([r['ppl_mean'] for r in rs]), 'ppl_std': mean([r['ppl_std'] for r in rs]),
    })

def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(out)
    print('[write]', path)

task_fields = ['task','method','nfe','n_seeds','reward_mean','reward_std','target_mean','target_std','nll_mean','nll_std','ppl_mean','ppl_std']
method_fields = ['method','nfe','n_tasks','reward_mean','reward_std','target_mean','target_std','nll_mean','nll_std','ppl_mean','ppl_std']
write_csv(outdir / 'lm1b_replay_taskwise_summary.csv', taskwise, task_fields)
write_csv(outdir / 'lm1b_replay_method_nfe_summary.csv', methodwise, method_fields)

name_map = {'base':'Base','fmtg':'ATG','fmrg':'FMRG','gap_aware':'RCFG','dflow':'D-Flow','sgfm':'SGFM','treeg':'Tree-G'}
flow_order = ['base','fmtg','fmrg','gap_aware']
nonflow_order = ['dflow','sgfm','treeg','gap_aware']

def paper_rows(order, nfe_filter=None):
    rs = [r for r in methodwise if r['method'] in order and (nfe_filter is None or r['nfe'] == nfe_filter)]
    rs.sort(key=lambda r: (int(r['nfe']), order.index(r['method'])))
    return [{
        'NFE': r['nfe'], 'Method': name_map.get(r['method'], r['method']),
        'Reward': fmt_pm(r['reward_mean'], r['reward_std'], 3),
        'Target': fmt_pm(r['target_mean'], r['target_std'], 3),
        'NLL': fmt_pm(r['nll_mean'], r['nll_std'], 3),
        'PPL': fmt_pm(r['ppl_mean'], r['ppl_std'], 1),
    } for r in rs]

paper_fields = ['NFE','Method','Reward','Target','NLL','PPL']
flow_rows = paper_rows(flow_order)
nonflow_rows = paper_rows(nonflow_order, '8')
write_csv(outdir / 'lm1b_paper_flow_table.csv', flow_rows, paper_fields)
write_csv(outdir / 'lm1b_paper_nonflow_nfe8_table.csv', nonflow_rows, paper_fields)
print('\n=== Flow table preview ===')
for r in flow_rows: print(r)
print('\n=== Non-flow table preview ===')
for r in nonflow_rows: print(r)
