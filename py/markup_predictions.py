#!/usr/bin/env python
"""
markup_predictions.py — Per-token confidence and rank markup of a passage.

For each token of a passage, run one forward pass under teacher forcing
and report, against the *actual* next token:
    (a) its rank in the model's predicted distribution (1, 2, 3, ...)
    (b) its probability under softmax(logits)
    (c) the top-K predicted alternatives with their probabilities.

Output is a colored terminal view plus a per-position detail table.

Designed for the project's char model — each "token" is one character,
so the markup reads as inline rank labels over the passage. The script
also works for BPE; tokens are then subwords.

Usage (run from repo root, on a machine that has the checkpoint):
    python py/markup_predictions.py \\
        --model pt/char_uppercase_16L_1280.pt \\
        --corpus txt_local/corpus_high_quality_uppercase_2026_05_08.txt \\
        --context_chars 512 --mark_chars 200 --seed 42

Or pass text directly:
    python py/markup_predictions.py \\
        --model pt/char_uppercase_16L_1280.pt \\
        --text "Some passage to mark up."
"""

import os
import sys
import argparse
import random

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT
from tokenizer import load_tokenizer
from sample import detect_case_preservation


ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_MAGENTA = "\x1b[35m"
ANSI_RED = "\x1b[31m"
ANSI_CYAN = "\x1b[36m"


def color_for_rank(rank):
    """Map a 1-based rank to an ANSI color."""
    if rank == 1:
        return ANSI_GREEN
    if rank == 2:
        return ANSI_YELLOW
    if rank == 3:
        return ANSI_MAGENTA
    return ANSI_RED


def visible(ch):
    """Replace whitespace/control chars with a visible glyph for the inline view."""
    if ch == '\n':
        return '\\n'
    if ch == '\t':
        return '\\t'
    if ch == '\r':
        return '\\r'
    if ch == ' ':
        return '_'
    return ch


def derive_meta_path(model_path):
    base = model_path.replace('.pt', '')
    if '_iter' in base:
        base = base.rsplit('_iter', 1)[0]
    elif base.endswith('_final'):
        base = base[:-len('_final')]
    elif base.endswith('_rolling'):
        base = base[:-len('_rolling')]
    return base + '_meta.pkl'


def load_model_and_tokenizer(model_path, device):
    meta_path = derive_meta_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])

    _stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        model = GPT(gptconf)
    finally:
        sys.stdout = _stdout

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(meta_path)
    iter_num = checkpoint.get('iter_num', None)
    best_val = checkpoint.get('best_val_loss', None)
    return model, tokenizer, iter_num, best_val


@torch.no_grad()
def score_passage(model, tokenizer, context_text, mark_text, device, top_k=5):
    """Score each token of mark_text against the model under teacher forcing.

    Returns a list of dicts, one per marked token, in mark_text order:
        {
            'token_str': decoded token (string),
            'rank': 1-based rank of actual token in predicted distribution,
            'prob': softmax probability of the actual token,
            'topk': [(token_str, prob), ...] of top-K predictions,
        }
    Plus a leading dict if any tokens were dropped due to block_size limit.
    """
    block_size = model.config.block_size
    ctx_ids = tokenizer.encode(context_text)
    mark_ids = tokenizer.encode(mark_text)

    # Total sequence is ctx + mark. Each mark token at sequence position p
    # is predicted from the logits at sequence position p-1. We need at
    # least one token of context for position 0 of mark.
    full = ctx_ids + mark_ids
    dropped_from_front = 0
    if len(full) > block_size:
        dropped_from_front = len(full) - block_size
        full = full[-block_size:]

    # How many mark tokens survived the crop?
    n_mark_scored = min(len(mark_ids), len(full) - 1)
    if n_mark_scored <= 0:
        return [], dropped_from_front, len(mark_ids)

    idx = torch.tensor(full, dtype=torch.long, device=device)[None, ...]
    targets = idx.clone()  # forces forward() to return per-position logits
    logits, _ = model(idx, targets)
    logits = logits[0]  # (T, vocab)
    probs = F.softmax(logits.float(), dim=-1)

    T = idx.size(1)
    # Scored mark tokens are the LAST n_mark_scored positions of the sequence.
    # Token at sequence position p is predicted from logits at position p-1.
    mark_start = T - n_mark_scored

    rows = []
    # Build a top-K-once-per-position structure
    for p in range(mark_start, T):
        actual_id = idx[0, p].item()
        row_logits = logits[p - 1]
        row_probs = probs[p - 1]

        # Rank of the actual token under this distribution
        # (1-based; higher logit = better rank)
        sorted_ids = torch.argsort(row_logits, descending=True)
        # rank = position of actual_id in sorted_ids, +1
        rank = int((sorted_ids == actual_id).nonzero(as_tuple=True)[0].item()) + 1

        topk_ids = sorted_ids[:top_k].tolist()
        topk = [(tokenizer.decode([tid]), float(row_probs[tid].item()))
                for tid in topk_ids]

        rows.append({
            'token_str': tokenizer.decode([actual_id]),
            'rank': rank,
            'prob': float(row_probs[actual_id].item()),
            'topk': topk,
        })

    n_skipped_from_start = len(mark_ids) - n_mark_scored
    return rows, dropped_from_front, n_skipped_from_start


def print_inline_markup(rows, mark_text_displayed):
    """One line of color per token; legend at the end."""
    parts = []
    for r in rows:
        col = color_for_rank(r['rank'])
        parts.append(f"{col}{visible(r['token_str'])}{ANSI_RESET}")
    print("".join(parts))
    print(f"{ANSI_DIM}Legend: "
          f"{ANSI_RESET}{ANSI_GREEN}rank 1{ANSI_RESET}  "
          f"{ANSI_YELLOW}rank 2{ANSI_RESET}  "
          f"{ANSI_MAGENTA}rank 3{ANSI_RESET}  "
          f"{ANSI_RED}rank 4+{ANSI_RESET}    "
          f"{ANSI_DIM}(spaces shown as _, newlines as \\n){ANSI_RESET}")


def print_detail_table(rows, top_k=5, show_all=False):
    """Per-position table: position, actual, prob, rank, top-K alternates."""
    header = (
        f"{'pos':>4}  {'actual':<8}  {'rank':>4}  {'p(actual)':>9}  "
        f"top-{top_k} predictions (token : prob)"
    )
    print(header)
    print('-' * len(header))
    for i, r in enumerate(rows):
        if not show_all and r['rank'] == 1 and r['prob'] >= 0.5:
            continue  # by default, hide easy positions
        topk_str = "  ".join(
            f"{repr(visible(t))[1:-1]}:{p:.2f}" for t, p in r['topk']
        )
        col = color_for_rank(r['rank'])
        print(f"{i:>4}  "
              f"{col}{repr(visible(r['token_str']))[1:-1]:<8}{ANSI_RESET}  "
              f"{col}{r['rank']:>4}{ANSI_RESET}  "
              f"{r['prob']:>9.3f}  {topk_str}")


def print_summary(rows):
    n = len(rows)
    if n == 0:
        print("No tokens scored.")
        return
    rank_counts = {}
    for r in rows:
        rank_counts[r['rank']] = rank_counts.get(r['rank'], 0) + 1
    mean_prob = sum(r['prob'] for r in rows) / n
    geo_mean_logprob = sum(
        (math_log(r['prob']) if r['prob'] > 0 else -1e9) for r in rows
    ) / n
    print(f"Tokens scored: {n}")
    print(f"Mean p(actual): {mean_prob:.3f}    "
          f"Geo-mean p(actual) (= exp(mean log-p)): "
          f"{(math_exp(geo_mean_logprob)) if geo_mean_logprob > -1e8 else 0.0:.3f}")
    rank1 = rank_counts.get(1, 0)
    rank_le3 = sum(c for r, c in rank_counts.items() if r <= 3)
    print(f"Rank 1: {rank1}/{n} ({100*rank1/n:.1f}%)    "
          f"Rank ≤ 3: {rank_le3}/{n} ({100*rank_le3/n:.1f}%)")
    # Show top-N rank counts
    bins = sorted(rank_counts.items())
    bin_str = "  ".join(f"r{r}:{c}" for r, c in bins[:8])
    rest = sum(c for r, c in bins if r > 8)
    if rest:
        bin_str += f"  r9+:{rest}"
    print(f"Rank histogram: {bin_str}")


def math_log(x):
    import math
    return math.log(x)


def math_exp(x):
    import math
    return math.exp(x)


def main():
    parser = argparse.ArgumentParser(
        description="Mark up a passage with per-token model confidence and rank.")
    parser.add_argument('--model', required=True, help='Checkpoint .pt path')

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--corpus', type=str,
                     help='Pick a passage from this corpus file')
    src.add_argument('--text', type=str,
                     help='Mark this literal text (full passage, no separate context)')

    parser.add_argument('--context_chars', type=int, default=512,
                        help='Chars of preceding context fed to the model (corpus mode)')
    parser.add_argument('--mark_chars', type=int, default=200,
                        help='Chars to mark up (corpus mode)')
    parser.add_argument('--text_context_chars', type=int, default=0,
                        help='With --text, mark only the tail; treat the leading '
                             'N chars as unscored context. Default 0 = mark all.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for corpus passage selection')
    parser.add_argument('--top_k', type=int, default=5,
                        help='How many top predictions to record per position')
    parser.add_argument('--show_all', action='store_true',
                        help='Show every position in the detail table (default: only '
                             'positions where the actual token was not the high-confidence rank 1)')
    parser.add_argument('--no_color', action='store_true',
                        help='Disable ANSI colors (for piping to a file)')
    args = parser.parse_args()

    if args.no_color:
        # Wipe the ANSI codes so output is plain text
        for name in ('ANSI_RESET', 'ANSI_BOLD', 'ANSI_DIM', 'ANSI_GREEN',
                     'ANSI_YELLOW', 'ANSI_MAGENTA', 'ANSI_RED', 'ANSI_CYAN'):
            globals()[name] = ''

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')
    print(f"Device: {device}", file=sys.stderr, flush=True)

    print(f"Loading model: {args.model}", file=sys.stderr, flush=True)
    model, tokenizer, iter_num, best_val = load_model_and_tokenizer(args.model, device)
    case_preserved = detect_case_preservation(tokenizer)
    print(f"  iter={iter_num}, best_val={best_val}, tokenizer={tokenizer.tokenizer_type}, "
          f"vocab={tokenizer.vocab_size}, case_preserved={case_preserved}",
          file=sys.stderr, flush=True)

    if args.text is not None:
        if args.text_context_chars > 0:
            context_text = args.text[:args.text_context_chars]
            mark_text = args.text[args.text_context_chars:]
        else:
            context_text = ''
            mark_text = args.text
    else:
        if not os.path.exists(args.corpus):
            print(f"Error: corpus not found: {args.corpus}", file=sys.stderr)
            sys.exit(1)
        with open(args.corpus, 'r', encoding='utf-8') as f:
            corpus = f.read()
        span = args.context_chars + args.mark_chars
        if len(corpus) <= span:
            print(f"Error: corpus too short ({len(corpus)} chars) for span {span}",
                  file=sys.stderr)
            sys.exit(1)
        rng = random.Random(args.seed)
        off = rng.randint(0, len(corpus) - span)
        context_text = corpus[off:off + args.context_chars]
        mark_text = corpus[off + args.context_chars:off + args.context_chars + args.mark_chars]
        print(f"Corpus offset: {off:,}", file=sys.stderr, flush=True)

    print(f"Context: {len(context_text)} chars   Marked: {len(mark_text)} chars",
          file=sys.stderr, flush=True)

    rows, dropped, skipped_start = score_passage(
        model, tokenizer, context_text, mark_text, device, top_k=args.top_k)

    if dropped:
        print(f"NOTE: {dropped} leading tokens dropped to fit block_size "
              f"({model.config.block_size}).", file=sys.stderr)
    if skipped_start:
        print(f"NOTE: {skipped_start} mark-text tokens at the start could not be "
              f"scored (block_size limit).", file=sys.stderr)

    print()
    print(f"{ANSI_BOLD}=== Context (unmarked) ==={ANSI_RESET}")
    if context_text:
        print(f"{ANSI_DIM}{context_text}{ANSI_RESET}")
    else:
        print(f"{ANSI_DIM}(none){ANSI_RESET}")
    print()
    print(f"{ANSI_BOLD}=== Marked passage (color = rank of actual token) ==={ANSI_RESET}")
    print_inline_markup(rows, mark_text)
    print()
    print(f"{ANSI_BOLD}=== Per-position detail "
          f"({'all positions' if args.show_all else 'positions where rank>1 or prob<0.5'}) ==={ANSI_RESET}")
    print_detail_table(rows, top_k=args.top_k, show_all=args.show_all)
    print()
    print(f"{ANSI_BOLD}=== Summary ==={ANSI_RESET}")
    print_summary(rows)


if __name__ == '__main__':
    main()
