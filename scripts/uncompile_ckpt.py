#!/usr/bin/env python3
"""Remove torch.compile '_orig_mod' prefixes from a PyTorch Lightning checkpoint.

Example:
  python scripts/uncompile_ckpt.py \
    --src logs/train/.../checkpoints/last.ckpt \
    --dst logs/train/.../checkpoints/last_uncompiled.ckpt
"""
import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--src', required=True)
p.add_argument('--dst', required=True)
args = p.parse_args()

src = Path(args.src)
dst = Path(args.dst)
ckpt = torch.load(src, map_location='cpu', weights_only=False)
sd = ckpt.get('state_dict', ckpt)
new_sd = {}
changed = 0
for k, v in sd.items():
    nk = k.replace('._orig_mod.', '.')
    if nk.startswith('_orig_mod.'):
        nk = nk[len('_orig_mod.'):]
    if nk != k:
        changed += 1
    new_sd[nk] = v

if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt['state_dict'] = new_sd
else:
    ckpt = new_sd

dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, dst)
print(f'[uncompile] src={src}')
print(f'[uncompile] dst={dst}')
print(f'[uncompile] changed_keys={changed}')
print('[uncompile] remaining _orig_mod keys:', sum('_orig_mod' in k for k in new_sd))
if isinstance(ckpt, dict):
    print('[uncompile] global_step:', ckpt.get('global_step'))
    print('[uncompile] epoch:', ckpt.get('epoch'))
