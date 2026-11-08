"""coupler-queue 0012 step 4: reload a snapshot into a fresh model, rerun the probe batch, and
compare with the probe log written during training.

  python3 py/study0012_checks.py --run_dir pt/study0012/consist/bpe --tokenizer_from pt/study0012/bpe8192.pkl \
      --snapshot pt/study0012/smoke2/bpe/snap_step000483_*.pt --compare_step 500
"""
import argparse
import glob
import json
import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402
import study_probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='dir with model_resume.pt/model_final.pt (for model_args) and probe_log.jsonl')
    ap.add_argument('--snapshot', required=True, help='glob')
    ap.add_argument('--compare_step', type=int, required=True)
    ap.add_argument('--tokenizer_from', required=True)
    ap.add_argument('--probe_batch', default='doc/0013_probe_batch_v2.json')
    args = ap.parse_args()
    snap = sorted(glob.glob(args.snapshot))[0]
    ck = torch.load(os.path.join(args.run_dir, 'model_final.pt'), map_location='cpu', weights_only=False)
    cfg = GPTConfig(**ck['model_args'])
    model = GPT(cfg)
    model.load_state_dict(torch.load(snap, map_location='cpu'))
    model.to('cuda')
    tok = load_tokenizer(args.tokenizer_from)
    runner = study_probe.ProbeRunner(args.probe_batch, tok, 'cuda', max_input=cfg.block_size)
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    out = runner.run(model, ctx)
    logged = None
    for l in open(os.path.join(args.run_dir, 'probe_log.jsonl')):
        r = json.loads(l)
        if r['step'] == args.compare_step:
            logged = r
    print(f"snapshot {os.path.basename(snap)} vs probe_log step {args.compare_step}")
    mx = 0.0
    for k, v in out['bpc'].items():
        d = abs(v - logged['bpc'][k])
        mx = max(mx, d)
        print(f"  bpc[{k}]: reloaded {v:.5f}  logged {logged['bpc'][k]:.5f}  |diff| {d:.2e}")
    for k in ('rep_first_bpc', 'rep_second_bpc'):
        print(f"  {k}: reloaded {out[k]:.5f}  logged {logged[k]:.5f}")
    di = max(abs(a - b) for ra, rb in zip(out['induction'], logged['induction']) for a, b in zip(ra, rb))
    de = max(abs(a - b) for kk in out['attn_entropy'] for ra, rb in zip(out['attn_entropy'][kk], logged['attn_entropy'][kk]) for a, b in zip(ra, rb))
    print(f"  max |diff| induction score {di:.2e}; max |diff| attention entropy {de:.2e}; max |diff| bpc {mx:.2e}")


if __name__ == '__main__':
    main()
