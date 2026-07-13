#!/usr/bin/env python
"""
recite_accuracy.py — how much of a text has a model actually memorized?

For a given model + text, run a teacher-forced pass over the whole text (feed
the true characters, ask for the next one at every position) and report:
  - bits/char : mean cross-entropy in bits (capacity/compression measure)
  - top1 %    : fraction of positions whose ARGMAX next-char is correct
                (a clean "how much of the text it has nailed")
  - free-run recite %: greedy free-running from a short seed, then char-match
                against the true continuation (errors compound — the strict test)

Runs on CPU (tiny models). Prints one line per model so several can be compared.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT
from tokenizer import load_tokenizer


def load(base):
    for cand in (base + '_final.pt', base + '.pt', base + '_rolling.pt'):
        if os.path.exists(cand):
            ckpt_path = cand
            break
    else:
        raise FileNotFoundError(base)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    conf = GPTConfig(**ckpt['model_args'])
    _o = sys.stdout; sys.stdout = open(os.devnull, 'w')
    model = GPT(conf); sys.stdout = _o
    model.load_state_dict(ckpt['model']); model.eval()
    tok = load_tokenizer(base + '_meta.pkl')
    return model, tok, conf


@torch.no_grad()
def teacher_forced(model, ids, block):
    """Mean bits/char and top-1 next-char accuracy over the text, using
    non-overlapping windows of length `block`."""
    tot_nll = 0.0; tot_tok = 0; correct = 0
    i = 0
    while i + 2 <= len(ids):
        w = ids[i:i + block]
        x = torch.tensor(w, dtype=torch.long)[None, :]
        logits = model.lm_head(model.transformer.ln_f(_hidden(model, x)))[0]
        tgt = torch.tensor(w[1:], dtype=torch.long)
        lp = F.log_softmax(logits[:-1], dim=-1)
        nll = -lp.gather(1, tgt[:, None]).squeeze(1)
        tot_nll += nll.sum().item(); tot_tok += tgt.numel()
        correct += (logits[:-1].argmax(-1) == tgt).sum().item()
        i += block
    bits = (tot_nll / tot_tok) / math.log(2)
    return bits, 100.0 * correct / tot_tok


def _hidden(model, idx):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    for blk in model.transformer.h:
        x = blk(x)
    return x


@torch.no_grad()
def free_recite(model, tok, ids, block, seed_len=64, run_len=1000):
    """Greedy free-run from the first seed_len chars; compare the next run_len
    generated chars to the truth. Returns % matched (strict, errors compound)."""
    run_len = min(run_len, len(ids) - seed_len - 1)
    if run_len <= 0:
        return float('nan')
    cur = ids[:seed_len]
    matched = 0
    for k in range(run_len):
        ctx = cur[-block:]
        x = torch.tensor(ctx, dtype=torch.long)[None, :]
        logits = model.lm_head(model.transformer.ln_f(_hidden(model, x)))[0, -1]
        nxt = int(logits.argmax())
        truth = ids[seed_len + k]
        if nxt == truth:
            matched += 1
        cur.append(truth)  # teacher-forced context so errors don't cascade off-poem
    return 100.0 * matched / run_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='txt_local/Poe_The_Raven.txt')
    ap.add_argument('bases', nargs='+', help="checkpoint stems, e.g. pt/raven_char_L2_d16")
    args = ap.parse_args()
    with open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"text: {args.text} ({len(text)} chars)")
    print(f"{'model':<26} {'n_layer':>7} {'bits/char':>10} {'top1%':>7}")
    rows = []
    for base in args.bases:
        model, tok, conf = load(base)
        ids = tok.encode(text)
        bits, top1 = teacher_forced(model, ids, conf.block_size)
        rows.append((os.path.basename(base), conf.n_layer, bits, top1))
        print(f"{os.path.basename(base):<26} {conf.n_layer:>7} {bits:>10.3f} {top1:>7.1f}")
    return rows


if __name__ == '__main__':
    main()
