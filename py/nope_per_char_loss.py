#!/usr/bin/env python3
"""
nope_per_char_loss.py — coupler-queue 0004: WHERE does a no-positional-embedding
(NoPE) character model lose bits, compared with its matched PE twin?

Teacher-forced (no sampling): both models read exactly the same held-out
windows (the last 10% of the token cache = train.py's val split) and we record
the loss, in bits, on every next-character prediction. Each target character is
then put into buckets:

  * position in the 256-character context window (does NoPE suffer most at the
    start of the window, or everywhere?),
  * letter position inside its word (0 = first letter after a non-letter),
  * the second letter of a DOUBLE letter (target equals the previous letter,
    e.g. the second T of LETTER) versus all other within-word letters,
  * the character right after a word ends (space/punctuation).

Usage (repo root):
  python py/nope_per_char_loss.py \
      --pe pt/char_nope_pilot_6L_768_pe_mps_final.pt \
      --nope pt/char_nope_pilot_6L_768_nope_mps_final.pt \
      --tokens pt/char_nope_pilot_6L_768_pe_mps_tokens.pt --windows 400
"""
import os
import sys
import math
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memorization_probe import load_model_and_tokenizer

LN2 = math.log(2.0)


@torch.no_grad()
def per_char_bits(model, x, y, bs):
    out = []
    for i in range(0, len(x), bs):
        logits, _ = model(x[i:i + bs], y[i:i + bs])
        lp = F.cross_entropy(logits.transpose(1, 2), y[i:i + bs], reduction="none")
        out.append(lp.float().cpu() / LN2)
    return torch.cat(out)  # (windows, T) bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", required=True)
    ap.add_argument("--nope", required=True)
    ap.add_argument("--tokens", required=True, help="train.py _tokens.pt cache (whole corpus)")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    toks = torch.load(a.tokens, weights_only=False)
    n = len(toks)
    val_start = n - int(n * a.val_split)
    # evenly spaced, fixed window starts across the val region (same for both arms)
    span = n - val_start - a.block - 2
    starts = [val_start + (span * k) // a.windows for k in range(a.windows)]
    x = torch.stack([toks[s:s + a.block] for s in starts]).long()
    y = torch.stack([toks[s + 1:s + 1 + a.block] for s in starts]).long()

    bits = {}
    tok = None
    for arm, path in (("pe", a.pe), ("nope", a.nope)):
        model, tok, _, _ = load_model_and_tokenizer(path, device)
        print(f"{arm}: {path} | wpe present: {'wpe' in model.transformer}", flush=True)
        bits[arm] = per_char_bits(model, x.to(device), y.to(device), a.batch)
        del model

    # character classes for the target and its history (decode the window once)
    buckets = defaultdict(lambda: [0.0, 0.0, 0])  # name -> [pe_sum, nope_sum, count]

    def add(name, w, t):
        b = buckets[name]
        b[0] += bits["pe"][w, t].item()
        b[1] += bits["nope"][w, t].item()
        b[2] += 1

    for w in range(a.windows):
        text = tok.decode(toks[starts[w]:starts[w] + a.block + 1].tolist())
        assert len(text) == a.block + 1  # char tokenizer: one token per character
        # target at position t is text[t+1]; its history is text[:t+1]
        run = 0  # letters since the last non-letter, counted over the history
        for t in range(a.block):
            prev, tgt = text[t], text[t + 1]
            run = run + 1 if prev.isalpha() else 0
            add("all", w, t)
            pb = "000-007" if t < 8 else "008-031" if t < 32 else "032-127" if t < 128 else "128-255"
            add(f"window pos {pb}", w, t)
            if tgt.isalpha():
                add(f"letter #{min(run, 8) if run < 8 else '8+'} in word", w, t)
                if run >= 1:
                    if tgt.lower() == prev.lower():
                        add("in-word: 2nd of double letter", w, t)
                    else:
                        add("in-word: other letters (pos>=1)", w, t)
            elif prev.isalpha():
                add("char right after word ends", w, t)

    print(f"\n{a.windows} windows x {a.block} chars = {a.windows * a.block:,} predictions "
          f"from the held-out (val) region; loss in bits per character")
    print(f"{'bucket':38} {'n':>7} {'PE':>7} {'NoPE':>7} {'NoPE-PE':>8} {'ratio':>6}")
    order = (["all"] + sorted(k for k in buckets if k.startswith("window")) +
             sorted(k for k in buckets if k.startswith("letter")) +
             sorted(k for k in buckets if k.startswith("in-word")) +
             ["char right after word ends"])
    for k in order:
        p, q, c = buckets[k]
        if c:
            print(f"{k:38} {c:7d} {p / c:7.3f} {q / c:7.3f} {(q - p) / c:+8.3f} {q / max(p, 1e-9):6.2f}")


if __name__ == "__main__":
    main()
