#!/usr/bin/env python3
"""
char_model_by_hand.py — the 1-layer CHARACTER Green-Eggs model, rebuilt by hand.

This reproduces exactly what the trained model does when it recites the book,
written in plain numpy from the model's own weights. It is organized around the
mechanism we discovered (diaries 110-111):

    Step 0  EMBED   : turn each recent character into a vector, add a position vector
    Step 1  HELPER-1: one attention layer makes a blurry, content-based summary
                      ("what kind of spot is this?")  -- ~58% of the answer
    Step 2  HELPER-2: one MLP with 512 "switches"; only ~5 turn on per spot, and
                      their COMBINATION decides the next letter  -- ~94%
    Step 3  READ OUT: compare the result to every letter, pick the best one

Because it uses the real weights and the real arithmetic, it matches the model
letter-for-letter. Run it and it (a) recites from the standard prompt, (b) shows
the ~5 switches that fire for one step, and (c) verifies exact agreement with the
actual model on the standard prompt AND several new prompts.

Usage:  python py/char_model_by_hand.py
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tokenizer import load_tokenizer

# ---- which model this program imitates ----
CKPT = 'pt/green_eggs_char_L1_val08_final.pt'
META = 'pt/green_eggs_char_L1_val08_meta.pkl'
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
    cfg = sd['model_args']
    return W, cfg


# ============================================================================
# The building blocks (all plain numpy)
# ============================================================================
def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))

def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)

def linear(x, w, b):
    # torch nn.Linear stores weight as (out, in); y = x @ w.T + b
    return x @ w.T + b


# ============================================================================
# The model, by hand
# ============================================================================
class CharModelByHand:
    def __init__(self, W, cfg):
        self.W, self.cfg = W, cfg
        self.nh = cfg['n_head']; self.C = cfg['n_embd']; self.hd = self.C // self.nh
        self.block = cfg['block_size']

    def forward(self, ids):
        """ids: list of token ids (the context). Returns logits for the NEXT token,
        plus the MLP switch activations at the last position (for inspection)."""
        W = self.W
        ids = ids[-self.block:]                       # model only sees the last `block` chars
        T = len(ids)

        # Step 0 — EMBED: character vector + position vector
        x = W['transformer.wte.weight'][ids] + W['transformer.wpe.weight'][:T]   # (T, C)

        # Step 1 — HELPER-1: attention (the blurry "where am I" summary)
        h = layernorm(x, W['transformer.h.0.ln_1.weight'], W['transformer.h.0.ln_1.bias'])
        qkv = linear(h, W['transformer.h.0.attn.c_attn.weight'], W['transformer.h.0.attn.c_attn.bias'])
        q, k, v = qkv[:, :self.C], qkv[:, self.C:2*self.C], qkv[:, 2*self.C:]
        q = q.reshape(T, self.nh, self.hd).transpose(1, 0, 2)   # (nh, T, hd)
        k = k.reshape(T, self.nh, self.hd).transpose(1, 0, 2)
        v = v.reshape(T, self.nh, self.hd).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.hd)   # (nh, T, T)
        mask = np.triu(np.ones((T, T)), 1).astype(bool)          # causal: no peeking ahead
        scores[:, mask] = -np.inf
        att = softmax(scores, -1)
        y = (att @ v).transpose(1, 0, 2).reshape(T, self.C)      # (T, C)
        y = linear(y, W['transformer.h.0.attn.c_proj.weight'], W['transformer.h.0.attn.c_proj.bias'])
        x = x + y                                                # residual add

        # Step 2 — HELPER-2: the MLP, i.e. the 512 switches
        h = layernorm(x, W['transformer.h.0.ln_2.weight'], W['transformer.h.0.ln_2.bias'])
        pre = linear(h, W['transformer.h.0.mlp.c_fc.weight'], W['transformer.h.0.mlp.c_fc.bias'])
        switches = gelu(pre)                                     # (T, 512): the "switch" values
        out = linear(switches, W['transformer.h.0.mlp.c_proj.weight'], W['transformer.h.0.mlp.c_proj.bias'])
        x = x + out

        # Step 3 — READ OUT: layernorm, then compare to every letter (tied unembedding)
        x = layernorm(x, W['transformer.ln_f.weight'], W['transformer.ln_f.bias'])
        logits = x @ W['transformer.wte.weight'].T              # (T, vocab)
        return logits[-1], switches[-1]                         # last position

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
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
    W, cfg = load_weights(CKPT)
    tok = load_tokenizer(META)
    hand = CharModelByHand(W, cfg)

    # --- (a) recite from the standard prompt ---
    ids = tok.encode(STD_PROMPT)
    out = hand.greedy(ids, 160)
    print("=" * 70)
    print(f"BY-HAND char model, greedy from prompt {STD_PROMPT!r}:")
    print("  " + tok.decode(out).replace("\n", " "))

    # --- (b) show the mechanism for ONE step: the switches that fire ---
    logits, switches = hand.forward(ids)
    on = np.where(switches > 1.0)[0]
    wte = W['transformer.wte.weight']; cproj = W['transformer.h.0.mlp.c_proj.weight']
    votes = wte @ cproj                                    # (vocab, 512): each switch's letter vote
    print("-" * 70)
    print(f"For the step after {STD_PROMPT!r}, {len(on)} of 512 switches are ON:")
    for n in on:
        c = tok.itos[int(votes[:, n].argmax())]
        print(f"    switch {n:3d}  (strength {switches[n]:4.1f})  votes for {c!r}")
    print(f"  -> model writes: {tok.itos[int(logits.argmax())]!r}")

    # --- (c) verify EXACT agreement with the real model, several prompts ---
    from model import GPTConfig, GPT
    dev = 'cpu'
    sd = torch.load(CKPT, map_location=dev, weights_only=False)
    real = GPT(GPTConfig(**sd['model_args'])); real.load_state_dict(sd['model']); real.eval()
    print("-" * 70)
    print("Verification vs the real model (greedy, 120 chars):")
    prompts = [STD_PROMPT, "Would you", "Sam-I-Am", "green eggs", "I will not"]
    for p in prompts:
        ids = tok.encode(p)
        hand_txt = tok.decode(hand.greedy(ids, 120))
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            real_ids = real.generate(x, max_new_tokens=120, temperature=0.0)[0].tolist()
        real_txt = tok.decode(real_ids)
        match = sum(a == b for a, b in zip(hand_txt, real_txt))
        tag = "IDENTICAL" if hand_txt == real_txt else f"match {match}/{len(real_txt)}"
        print(f"  {p!r:<16} -> {tag}")


if __name__ == '__main__':
    main()
