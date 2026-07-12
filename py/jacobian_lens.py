#!/usr/bin/env python
"""
jacobian_lens.py — a tiny Jacobian-lens / logit-lens readout for our models.

The question this tool answers, in plain words:
    As the model reads the poem, it holds a bundle of numbers at each word (its
    "residual stream"). We want to translate that inner bundle into WORDS at
    every depth of the model: "at layer L, what token is the model already
    disposed to say next?"  Then we watch how that readout changes from the
    bottom layer to the top.

Two readouts are computed at every layer L (L = 0 is the raw embedding, L =
n_layer is the final residual just before the model's own output head):

  LOGIT LENS   — decode the layer-L residual directly with the model's own
                 output head:  readout(h) = lm_head(ln_f(h)).
                 (Ignores that h is not yet in the final-layer basis.)

  JACOBIAN LENS — first transport h to the final-layer basis with an *averaged
                 Jacobian*, then decode:  readout(J_L @ h), where
                 J_L = E[ d h_final / d h_L ] averaged over positions and
                 windows of the text (same-position transport). This is the
                 Anthropic "Jacobian lens" idea (transformer-circuits.pub/2026/
                 workspace), specialized to our tiny model and computed exactly
                 by autograd rather than fit on a big web corpus.

For each layer and each lens we measure how the *correct next token* of the
poem ranks in the readout. If the correct token is already rank-1 at a low
layer and merely sharpens, the answer is "present early, carried" (lookup). If
it climbs from deep-in-the-list up to rank-1 only near the top, the answer is
"built up through the layers" (construction).

Runs on CPU (the model is tiny) for clean, reproducible autograd.
"""

import argparse
import os
import sys
import pickle

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT
from tokenizer import load_tokenizer


def load_model_and_tokenizer(base):
    """base is a checkpoint stem like 'pt/raven_bpe_L6_d16'. Prefer the
    _final.pt (most-trained) checkpoint; fall back to base.pt (best-val)."""
    for cand in (base + '_final.pt', base + '.pt', base + '_rolling.pt'):
        if os.path.exists(cand):
            ckpt_path = cand
            break
    else:
        raise FileNotFoundError(f"No checkpoint found for base '{base}'")
    meta_path = base + '_meta.pkl'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    gptconf = GPTConfig(**ckpt['model_args'])
    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = GPT(gptconf)
    sys.stdout = _stdout
    model.load_state_dict(ckpt['model'])
    model.eval()
    tok = load_tokenizer(meta_path)
    return model, tok, gptconf, ckpt_path


def residual_stream(model, idx):
    """Manual forward that mirrors model.py, capturing the residual stream at
    every depth. Returns a list hs[0..L] of tensors (1, T, C):
        hs[0] = token+position embedding (before any block)
        hs[l] = residual after block l
        hs[L] = final residual (just before ln_f)."""
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    hs = [x]
    for block in model.transformer.h:
        x = block(x)
        hs.append(x)
    return hs


def readout(model, h):
    """Decode a residual-stream tensor into vocab logits with the model's own
    output head: lm_head(ln_f(h)). Shape (..., vocab)."""
    return model.lm_head(model.transformer.ln_f(h))


def build_windows(token_ids, block_size, stride=None):
    """Chop the token stream into windows of length block_size. Default:
    non-overlapping consecutive windows (continuous-mode style)."""
    if stride is None:
        stride = block_size
    windows = []
    i = 0
    n = len(token_ids)
    while i + 2 <= n:  # need at least 2 tokens (one target)
        w = token_ids[i:i + block_size]
        windows.append(w)
        i += stride
    return windows


@torch.no_grad()
def collect_logit_lens_stats(model, windows, n_layer, device):
    """For every layer, accumulate rank/top-1 stats of the correct next token
    under the plain logit lens. Returns per-layer dicts."""
    L = n_layer
    stats = [{'ranks': [], 'top1': 0, 'n': 0} for _ in range(L + 1)]
    for w in windows:
        idx = torch.tensor(w, dtype=torch.long, device=device)[None, :]
        T = idx.size(1)
        hs = residual_stream(model, idx)
        targets = idx[0, 1:]  # next-token targets for positions 0..T-2
        for l in range(L + 1):
            logits = readout(model, hs[l])[0]  # (T, vocab)
            logits = logits[:T - 1]            # positions with a target
            # rank of the target token (1 = best)
            tgt_logit = logits.gather(1, targets[:, None]).squeeze(1)  # (T-1,)
            ranks = (logits > tgt_logit[:, None]).sum(dim=1) + 1       # (T-1,)
            stats[l]['ranks'].append(ranks.float())
            stats[l]['top1'] += (ranks == 1).sum().item()
            stats[l]['n'] += ranks.numel()
    for l in range(L + 1):
        stats[l]['ranks'] = torch.cat(stats[l]['ranks'])
    return stats


def averaged_jacobians(model, windows, n_layer, device, max_windows):
    """Compute J_l = E[ d h_final / d h_l ] (same-position transport), a (C,C)
    matrix per layer l = 0..L, averaged over positions and a sample of windows.
    J_L (final layer) is the identity by definition."""
    C = model.config.n_embd
    L = n_layer
    blocks = list(model.transformer.h)
    acc = [torch.zeros(C, C) for _ in range(L + 1)]
    count = [0 for _ in range(L + 1)]
    acc[L] = torch.eye(C)          # transport from final layer is identity
    count[L] = 1
    sample = windows[:max_windows]
    for wi, w in enumerate(sample):
        idx = torch.tensor(w, dtype=torch.long, device=device)[None, :]
        T = idx.size(1)
        hs = residual_stream(model, idx)  # no_grad not set here; fine, we detach
        for l in range(L):
            h_l = hs[l].detach().clone()  # (1, T, C)

            def transport(h, _l=l):
                x = h
                for block in blocks[_l:]:
                    x = block(x)
                return x  # (1, T, C) = h_final

            try:
                jac = torch.autograd.functional.jacobian(transport, h_l,
                                                          vectorize=True)
            except Exception:
                # vmap can fail to compose with flash-attention (SDPA); fall
                # back to the slower, always-correct per-output loop.
                jac = torch.autograd.functional.jacobian(transport, h_l,
                                                          vectorize=False)
            # jac shape: (1, T, C, 1, T, C). Diagonal-position blocks:
            jac = jac[0, :, :, 0, :, :]     # (T, C, T, C)
            for t in range(T):
                acc[l] += jac[t, :, t, :]   # (C, C)
                count[l] += 1
        if (wi + 1) % 2 == 0:
            print(f"    jacobian: {wi + 1}/{len(sample)} windows", flush=True)
    Js = [acc[l] / max(count[l], 1) for l in range(L + 1)]
    return Js


@torch.no_grad()
def collect_jacobian_lens_stats(model, windows, Js, n_layer, device):
    """Same rank/top-1 stats, but each layer's residual is first transported to
    the final basis by its averaged Jacobian J_l before decoding."""
    L = n_layer
    stats = [{'ranks': [], 'top1': 0, 'n': 0} for _ in range(L + 1)]
    for w in windows:
        idx = torch.tensor(w, dtype=torch.long, device=device)[None, :]
        T = idx.size(1)
        hs = residual_stream(model, idx)
        targets = idx[0, 1:]
        for l in range(L + 1):
            h = hs[l][0]                      # (T, C)
            h_t = h @ Js[l].t()               # transport: (T,C) @ (C,C)^T
            logits = readout(model, h_t)      # (T, vocab)
            logits = logits[:T - 1]
            tgt_logit = logits.gather(1, targets[:, None]).squeeze(1)
            ranks = (logits > tgt_logit[:, None]).sum(dim=1) + 1
            stats[l]['ranks'].append(ranks.float())
            stats[l]['top1'] += (ranks == 1).sum().item()
            stats[l]['n'] += ranks.numel()
    for l in range(L + 1):
        stats[l]['ranks'] = torch.cat(stats[l]['ranks'])
    return stats


def summarize(name, stats, L):
    print(f"\n  {name}")
    print(f"    layer   top1%   median-rank   mean-rank   MRR")
    for l in range(L + 1):
        r = stats[l]['ranks']
        top1 = 100.0 * stats[l]['top1'] / stats[l]['n']
        med = r.median().item()
        mean = r.mean().item()
        mrr = (1.0 / r).mean().item()
        tag = 'emb' if l == 0 else ('final' if l == L else f'blk{l}')
        print(f"    {l:>2} {tag:>6} {top1:6.1f}   {med:9.1f}   {mean:9.1f}   {mrr:5.3f}")


def example_readout(model, tok, token_ids, block_size, Js, n_layer, device,
                    target_text):
    """Show, for one memorable spot in the poem, the top-1 token each lens reads
    out at every layer, at the step that PREDICTS the target word.

    We find the first token j whose decoded text contains target_text, then feed
    the context ending just before it (token_ids[start:j]); the last fed
    position is the one predicting token j."""
    L = n_layer
    key = target_text.strip().lower()
    j = None
    for i in range(1, len(token_ids)):
        if key in tok.decode([token_ids[i]]).lower():
            j = i
            break
    if j is None:
        print(f"\n  Example: token containing {target_text!r} not found.")
        return
    start = max(0, j - block_size)
    ctx_ids = token_ids[start:j]                 # ends at token j-1
    idx = torch.tensor(ctx_ids, dtype=torch.long, device=device)[None, :]
    hs = residual_stream(model, idx)
    t = idx.size(1) - 1                          # last position predicts token j
    true_next = tok.decode([token_ids[j]])
    ctx_show = tok.decode(ctx_ids[-7:])
    print(f"\n  Example: the step that should predict {target_text!r}")
    print(f"    context: ...{ctx_show!r}  ->  true next token: {true_next!r}")
    print(f"    layer    logit-lens top1     jacobian-lens top1")
    for l in range(L + 1):
        h = hs[l][0][t:t + 1]                    # (1, C)
        ll_tok = tok.decode([int(readout(model, h)[0].argmax())])
        jl_tok = tok.decode([int(readout(model, h @ Js[l].t())[0].argmax())])
        tag = 'emb' if l == 0 else ('final' if l == L else f'blk{l}')
        print(f"    {l:>2} {tag:>6}   {ll_tok!r:>18}   {jl_tok!r:>18}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True,
                    help="checkpoint stem, e.g. pt/raven_bpe_L6_d16")
    ap.add_argument('--text', default='txt_local/Poe_The_Raven.txt')
    ap.add_argument('--max_jac_windows', type=int, default=10,
                    help="windows used to average the Jacobian (autograd cost)")
    ap.add_argument('--example', default='Nevermore',
                    help="a word to trace through the layers")
    ap.add_argument('--out', default='doc/figures/jacobian_lens.png')
    ap.add_argument('--dump', default=None,
                    help="if set, pickle per-layer stats to this path for "
                         "cross-model comparison plots")
    args = ap.parse_args()

    device = 'cpu'
    model, tok, cfg, ckpt_path = load_model_and_tokenizer(args.base)
    L = cfg.n_layer
    print(f"model: {ckpt_path}  | n_layer={L} n_embd={cfg.n_embd} "
          f"vocab={cfg.vocab_size} block={cfg.block_size} tok={tok.tokenizer_type}")

    with open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()
    token_ids = tok.encode(text)
    print(f"text: {args.text}  | {len(text)} chars -> {len(token_ids)} tokens")

    windows = build_windows(token_ids, cfg.block_size)
    print(f"windows: {len(windows)} of length {cfg.block_size}")

    print("\n[1/3] logit lens ...")
    ll_stats = collect_logit_lens_stats(model, windows, L, device)
    summarize("LOGIT LENS  (rank of correct next token by layer)", ll_stats, L)

    print(f"\n[2/3] averaged Jacobians over {args.max_jac_windows} windows ...")
    Js = averaged_jacobians(model, windows, L, device, args.max_jac_windows)

    print("\n[3/3] jacobian lens ...")
    jl_stats = collect_jacobian_lens_stats(model, windows, Js, L, device)
    summarize("JACOBIAN LENS  (rank of correct next token by layer)", jl_stats, L)

    example_readout(model, tok, token_ids, cfg.block_size, Js, L, device,
                    args.example)

    if args.dump:
        layers = list(range(L + 1))
        dump = {
            'base': args.base,
            'n_layer': L, 'n_embd': cfg.n_embd, 'vocab': cfg.vocab_size,
            'tokenizer': tok.tokenizer_type,
            'logit': {
                'top1': [100.0 * ll_stats[l]['top1'] / ll_stats[l]['n'] for l in layers],
                'mrr': [(1.0 / ll_stats[l]['ranks']).mean().item() for l in layers],
                'median': [ll_stats[l]['ranks'].median().item() for l in layers],
            },
            'jacobian': {
                'top1': [100.0 * jl_stats[l]['top1'] / jl_stats[l]['n'] for l in layers],
                'mrr': [(1.0 / jl_stats[l]['ranks']).mean().item() for l in layers],
                'median': [jl_stats[l]['ranks'].median().item() for l in layers],
            },
        }
        with open(args.dump, 'wb') as f:
            pickle.dump(dump, f)
        print(f"stats dumped: {args.dump}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        layers = list(range(L + 1))
        ll_top1 = [100.0 * ll_stats[l]['top1'] / ll_stats[l]['n'] for l in layers]
        jl_top1 = [100.0 * jl_stats[l]['top1'] / jl_stats[l]['n'] for l in layers]
        ll_mrr = [(1.0 / ll_stats[l]['ranks']).mean().item() for l in layers]
        jl_mrr = [(1.0 / jl_stats[l]['ranks']).mean().item() for l in layers]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.plot(layers, ll_top1, 'o-', label='logit lens')
        ax1.plot(layers, jl_top1, 's--', label='jacobian lens')
        ax1.set_xlabel('layer depth (0 = embedding, top = final)')
        ax1.set_ylabel('% correct next token is TOP-1')
        ax1.set_title('When does the answer become the top guess?')
        ax1.set_ylim(-2, 102)
        ax1.grid(alpha=0.3)
        ax1.legend()
        ax2.plot(layers, ll_mrr, 'o-', label='logit lens')
        ax2.plot(layers, jl_mrr, 's--', label='jacobian lens')
        ax2.set_xlabel('layer depth (0 = embedding, top = final)')
        ax2.set_ylabel('mean reciprocal rank of correct next token')
        ax2.set_title('How high the answer sits, by layer')
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(alpha=0.3)
        ax2.legend()
        base_name = os.path.basename(args.base)
        fig.suptitle(f'Reading the residual stream layer by layer — {base_name}')
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"\nfigure saved: {args.out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == '__main__':
    main()
