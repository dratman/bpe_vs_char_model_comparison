#!/usr/bin/env python3
"""
bpe_model_by_hand.py — the 1-layer BPE Green-Eggs model, rebuilt by hand.

Twin of char_model_by_hand.py. Same three-step recipe, same plain-numpy
arithmetic from the model's own weights — the ONLY differences are that this
model thinks in word-CHUNKS (182 of them) instead of single letters, and it
sees a shorter window (85 chunks). The forward pass code is identical to the
char version; that identity is the whole point (diary 111: same mechanism).

    Step 0  EMBED   : turn each recent chunk into a vector, add a position vector
    Step 1  HELPER-1: one attention layer makes a blurry, content-based summary
    Step 2  HELPER-2: one MLP with 512 "switches"; ~10 turn on per spot (a bigger
                      chord than char's ~5, because there are more chunks to pick
                      from), and their COMBINATION decides the next chunk
    Step 3  READ OUT: compare the result to every chunk, pick the best one

Usage:  python py/bpe_model_by_hand.py
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tokenizer import load_tokenizer

# ---- which model this program imitates ----
CKPT = 'pt/green_eggs_bpe_L1_val08_final.pt'
META = 'pt/green_eggs_bpe_L1_val08_meta.pkl'
STD_PROMPT = "I do not like"

try:
    from scipy.special import erf            # exact GELU (matches torch nn.GELU)
except Exception:
    _erf = np.vectorize(math.erf); erf = lambda x: _erf(x)


# ============================================================================
# Load the trained weights into plain numpy arrays
# ============================================================================
def load_weights(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    W = {k: v.float().numpy() for k, v in sd['model'].items()}
    return W, sd['model_args']


# ============================================================================
# The building blocks (all plain numpy) — identical to the char version
# ============================================================================
def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))

def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis, keepdims=True)

def linear(x, w, b):
    return x @ w.T + b                       # nn.Linear weight is (out, in)


# ============================================================================
# The model, by hand (same forward as char_model_by_hand.py)
# ============================================================================
class BpeModelByHand:
    def __init__(self, W, cfg):
        self.W, self.cfg = W, cfg
        self.nh = cfg['n_head']; self.C = cfg['n_embd']; self.hd = self.C // self.nh
        self.block = cfg['block_size']

    def forward(self, ids):
        W = self.W
        ids = ids[-self.block:]
        T = len(ids)

        # Step 0 — EMBED
        x = W['transformer.wte.weight'][ids] + W['transformer.wpe.weight'][:T]

        # Step 1 — HELPER-1: attention
        h = layernorm(x, W['transformer.h.0.ln_1.weight'], W['transformer.h.0.ln_1.bias'])
        qkv = linear(h, W['transformer.h.0.attn.c_attn.weight'], W['transformer.h.0.attn.c_attn.bias'])
        q, k, v = qkv[:, :self.C], qkv[:, self.C:2*self.C], qkv[:, 2*self.C:]
        q = q.reshape(T, self.nh, self.hd).transpose(1, 0, 2)
        k = k.reshape(T, self.nh, self.hd).transpose(1, 0, 2)
        v = v.reshape(T, self.nh, self.hd).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.hd)
        scores[:, np.triu(np.ones((T, T)), 1).astype(bool)] = -np.inf
        att = softmax(scores, -1)
        y = (att @ v).transpose(1, 0, 2).reshape(T, self.C)
        y = linear(y, W['transformer.h.0.attn.c_proj.weight'], W['transformer.h.0.attn.c_proj.bias'])
        x = x + y

        # Step 2 — HELPER-2: the 512 switches
        h = layernorm(x, W['transformer.h.0.ln_2.weight'], W['transformer.h.0.ln_2.bias'])
        pre = linear(h, W['transformer.h.0.mlp.c_fc.weight'], W['transformer.h.0.mlp.c_fc.bias'])
        switches = gelu(pre)
        out = linear(switches, W['transformer.h.0.mlp.c_proj.weight'], W['transformer.h.0.mlp.c_proj.bias'])
        x = x + out

        # Step 3 — READ OUT
        x = layernorm(x, W['transformer.ln_f.weight'], W['transformer.ln_f.bias'])
        logits = x @ W['transformer.wte.weight'].T
        return logits[-1], switches[-1]

    def greedy(self, ids, n):
        ids = list(ids)
        for _ in range(n):
            logits, _ = self.forward(ids)
            ids.append(int(logits.argmax()))
        return ids


# ============================================================================
# Demo + verification
# ============================================================================
def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    W, cfg = load_weights(CKPT)
    tok = load_tokenizer(META)
    hand = BpeModelByHand(W, cfg)
    chunk = lambda i: tok.decode([int(i)])            # BPE: a token id -> its chunk of text

    # --- (a) recite from the standard prompt ---
    ids = tok.encode(STD_PROMPT)
    out = hand.greedy(ids, 60)                         # 60 CHUNKS ~= 180 characters
    print("=" * 70)
    print(f"BY-HAND BPE model, greedy from prompt {STD_PROMPT!r}  ({len(ids)} chunks in):")
    print("  " + tok.decode(out).replace("\n", " "))

    # --- (b) show the mechanism for ONE step: the switches that fire ---
    logits, switches = hand.forward(ids)
    on = np.where(switches > 1.0)[0]
    votes = W['transformer.wte.weight'] @ W['transformer.h.0.mlp.c_proj.weight']   # (vocab, 512)
    print("-" * 70)
    print(f"For the step after {STD_PROMPT!r}, {len(on)} of 512 switches are ON "
          f"(a bigger chord than char, because there are more chunks to choose from):")
    for n in on:
        print(f"    switch {n:3d}  (strength {switches[n]:4.1f})  votes for {chunk(votes[:, n].argmax())!r}")
    print(f"  -> model writes chunk: {chunk(logits.argmax())!r}")

    # --- (c) verify EXACT agreement with the real model ---
    from model import GPTConfig, GPT
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    real = GPT(GPTConfig(**sd['model_args'])); real.load_state_dict(sd['model']); real.eval()
    print("-" * 70)
    print("Verification vs the real model (greedy, 60 chunks):")
    for p in [STD_PROMPT, "Would you", "Sam-I-Am", "green eggs", "I will not"]:
        ids = tok.encode(p)
        hand_txt = tok.decode(hand.greedy(ids, 60))
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            real_txt = tok.decode(real.generate(x, max_new_tokens=60, temperature=0.0)[0].tolist())
        tag = "IDENTICAL" if hand_txt == real_txt else f"match {sum(a==b for a,b in zip(hand_txt,real_txt))}/{len(real_txt)} chars"
        print(f"  {p!r:<16} -> {tag}")


if __name__ == '__main__':
    main()
