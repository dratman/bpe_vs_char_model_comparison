#!/usr/bin/env python
"""
memorization_probe.py - Quantify verbatim memorization across checkpoints.

Motivation (see HANDOFF.md, BPE-resumed section, and diary 094):
    The BPE model was resumed past its early-stop point to test the
    overtraining hypothesis — "samples grow more memorized as training
    proceeds past the val-loss minimum." The BPE-resumed run found its
    true val minimum at iter 168K (epoch 5.34) and held flat through
    iter 220K. The meaningful contrast points for the probe are therefore
    {132K (original early best), 168K (true min), 220K (final)} for BPE,
    and the analogous late-training checkpoints for char.

    This script measures, for each checkpoint, how strongly the model
    reproduces the training corpus verbatim, using two cheap and
    complementary metrics drawn from real corpus passages:

      1. EXTRACTABLE MEMORIZATION (autoregressive, the primary signal).
         Take a real prefix from the corpus, greedy-decode a continuation
         from the model alone, and measure the length (in CHARACTERS) of
         the exact-match prefix between the model's free continuation and
         the true continuation. This is the Carlini et al. notion of
         extractable memorization: a model that has memorized a passage
         will regenerate it verbatim from its own outputs. Reported as
         mean/median matched-prefix chars and the fraction of prompts that
         "extract" >= --extract_threshold chars verbatim.

      2. TEACHER-FORCED GREEDY ACCURACY (one forward pass, secondary).
         Feed prompt+true-continuation and check, at each continuation
         position, whether the model's argmax equals the true next token.
         This is "predictability" rather than memorization, but its trend
         across checkpoints of the SAME tokenizer is a useful companion.

    Characters are the comparable unit across tokenizations (the project's
    standing rule — BPE per-token and char per-token losses are not
    directly comparable). Metric 1 is fully char-based and therefore
    comparable across char vs BPE. Metric 2's token-accuracy is only
    meaningful as a WITHIN-tokenizer trend (char-vs-BPE token accuracy is
    not comparable), which is exactly what the overtraining experiment
    needs: monotonic growth across checkpoints of one model.

Usage (run from repo root, after rsyncing the checkpoints + meta files):
    python py/memorization_probe.py \
        --models pt/bpe_uppercase_16L_1280_b2_iter132000.pt \
                 pt/bpe_uppercase_16L_1280_b2_resumed_iter168000.pt \
                 pt/bpe_uppercase_16L_1280_b2_resumed_final.pt \
        --corpus txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
        --num_prompts 200 --prompt_chars 512 --continue_chars 256 --seed 42

Pass several checkpoints at once to get a single comparison table sorted
by iteration. If extractable-memorization mean/median and the >=K fraction
rise monotonically from 132K -> 168K -> 220K, the overtraining hypothesis
is quantitatively confirmed for BPE.
"""

import os
import sys
import argparse
import random

import torch

from model import GPTConfig, GPT
from tokenizer import load_tokenizer
# Reuse sample.py's greedy decoder and case detection so behavior matches
# the established sampling path exactly (importing is side-effect free:
# sample.py guards its main() under __name__ == '__main__').
from sample import generate_local, detect_case_preservation


def derive_meta_path(model_path):
    """Map a checkpoint path to its tokenizer meta .pkl, matching the
    logic in sample.py (strip _iter{N} or _final to reach the base name).

    Note (see HANDOFF.md, BPE-resumed section): the M3 keeps the meta
    files under the pre-resume base name, but the Studio sample scripts
    rsync them to the _resumed base name, so deriving the meta path from
    the model path is the established convention on the machine that runs
    this probe.
    """
    base = model_path.replace('.pt', '')
    if '_iter' in base:
        base = base.rsplit('_iter', 1)[0]
    elif base.endswith('_final'):
        base = base[:-len('_final')]
    return base + '_meta.pkl'


def load_model_and_tokenizer(model_path, device):
    """Load a checkpoint + its tokenizer the same way sample.py does."""
    meta_path = derive_meta_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])

    # Suppress model.py's constructor chatter, as sample.py does.
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


def longest_common_prefix_chars(a, b):
    """Length of the longest common prefix of two strings, in characters."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def sample_prompt_offsets(corpus_len, prompt_chars, continue_chars, num_prompts, rng):
    """Pick random start offsets such that prompt+continuation fits."""
    span = prompt_chars + continue_chars
    if corpus_len <= span:
        raise ValueError(
            f"Corpus ({corpus_len} chars) too short for prompt_chars + "
            f"continue_chars = {span}")
    high = corpus_len - span
    return [rng.randint(0, high) for _ in range(num_prompts)]


@torch.no_grad()
def teacher_forced_accuracy(model, tokenizer, prompt_text, true_cont_text, device):
    """Fraction of continuation positions where the model's argmax matches
    the true next token, under teacher forcing (single forward pass).

    Returns (correct, total) token counts, or (0, 0) if the continuation
    region is empty after cropping to block_size.
    """
    block_size = model.config.block_size
    prompt_ids = tokenizer.encode(prompt_text)
    cont_ids = tokenizer.encode(true_cont_text)
    if len(cont_ids) == 0:
        return 0, 0

    full = prompt_ids + cont_ids
    # Keep the tail that fits in the context window; ensure at least one
    # continuation token remains to be scored.
    if len(full) > block_size:
        full = full[-block_size:]
    # Recompute how many of the trailing tokens are continuation tokens
    # after the crop.
    n_cont = min(len(cont_ids), len(full) - 1)
    if n_cont <= 0:
        return 0, 0

    idx = torch.tensor(full, dtype=torch.long, device=device)[None, ...]
    # Passing targets makes GPT.forward return logits for ALL positions
    # (without targets it returns only the last position's logits).
    targets = idx.clone()
    logits, _ = model(idx, targets)
    preds = logits[0].argmax(dim=-1)  # (T,) prediction of token at t+1

    # Position t predicts token t+1. The continuation occupies the last
    # n_cont tokens (indices T-n_cont .. T-1); they are predicted from
    # positions T-n_cont-1 .. T-2.
    T = idx.size(1)
    correct = 0
    for t in range(T - n_cont - 1, T - 1):
        if preds[t].item() == idx[0, t + 1].item():
            correct += 1
    return correct, n_cont


@torch.no_grad()
def extractable_match_chars(model, tokenizer, prompt_text, true_cont_text,
                            continue_chars, device):
    """Greedy free-generate a continuation and return the length (chars) of
    its exact-match prefix against the true continuation."""
    prompt_ids = tokenizer.encode(prompt_text)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    # How many tokens to generate so the decoded text covers continue_chars.
    # Char tokenizer: 1 token ~ 1 char. BPE: ~4.5 chars/token, so fewer
    # tokens are needed; overshoot a little, then trim the decoded string.
    if tokenizer.tokenizer_type == 'char':
        max_new = continue_chars
    else:
        max_new = continue_chars // 3 + 8

    y = generate_local(
        model, x, max_new,
        temperature=0.0,          # greedy = deterministic, the memorization test
        top_k=None,
        rep_penalty=1.0,          # off
        device=device,
        stop_token_id=None,
    )
    gen_ids = y[0].tolist()[len(prompt_ids):]
    gen_text = tokenizer.decode(gen_ids)[:continue_chars]
    truth = true_cont_text[:continue_chars]
    return longest_common_prefix_chars(gen_text, truth)


def percentile(sorted_vals, q):
    """Simple nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def probe_model(model_path, corpus, offsets, prompt_chars, continue_chars,
                extract_threshold, device, progress_every):
    model, tokenizer, iter_num, best_val = load_model_and_tokenizer(model_path, device)
    case_preserved = detect_case_preservation(tokenizer)

    match_lengths = []   # extractable-memorization: matched-prefix chars
    tf_correct = 0
    tf_total = 0

    for i, off in enumerate(offsets):
        prompt_text = corpus[off:off + prompt_chars]
        true_cont = corpus[off + prompt_chars:off + prompt_chars + continue_chars]

        match_lengths.append(
            extractable_match_chars(model, tokenizer, prompt_text, true_cont,
                                    continue_chars, device))
        c, t = teacher_forced_accuracy(model, tokenizer, prompt_text, true_cont, device)
        tf_correct += c
        tf_total += t

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [{os.path.basename(model_path)}] {i + 1}/{len(offsets)} prompts",
                  file=sys.stderr, flush=True)

    match_lengths.sort()
    n = len(match_lengths)
    mean_match = sum(match_lengths) / n if n else 0.0
    median_match = percentile(match_lengths, 0.5)
    p95_match = percentile(match_lengths, 0.95)
    frac_extracted = (sum(1 for m in match_lengths if m >= extract_threshold) / n
                      if n else 0.0)
    tf_acc = (tf_correct / tf_total) if tf_total else 0.0

    # Free model memory before the next checkpoint loads.
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'path': model_path,
        'iter': iter_num,
        'best_val': best_val,
        'tokenizer': tokenizer.tokenizer_type,
        'case_preserved': case_preserved,
        'mean_match': mean_match,
        'median_match': median_match,
        'p95_match': p95_match,
        'frac_extracted': frac_extracted,
        'tf_acc': tf_acc,
    }


def fmt_iter(it):
    if it is None:
        return '?'
    return str(it)


def main():
    parser = argparse.ArgumentParser(
        description='Probe verbatim/extractable memorization across checkpoints')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='One or more checkpoint .pt files (compared in one table)')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Training corpus text file (source of prompts + true continuations)')
    parser.add_argument('--num_prompts', type=int, default=200,
                        help='Number of corpus passages to probe')
    parser.add_argument('--prompt_chars', type=int, default=512,
                        help='Context length fed to the model, in characters')
    parser.add_argument('--continue_chars', type=int, default=256,
                        help='Continuation length compared, in characters')
    parser.add_argument('--extract_threshold', type=int, default=50,
                        help='Matched-prefix chars at/above which a prompt counts as "extracted"')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for prompt-offset selection (same offsets across all models)')
    parser.add_argument('--progress_every', type=int, default=50,
                        help='Print progress to stderr every N prompts (0=silent)')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')

    if not os.path.exists(args.corpus):
        print(f"Error: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus: {args.corpus}", file=sys.stderr, flush=True)
    with open(args.corpus, 'r', encoding='utf-8') as f:
        corpus = f.read()
    print(f"Corpus length: {len(corpus):,} chars | device: {device}", file=sys.stderr, flush=True)

    # SAME prompt offsets for every model, so differences are due to the
    # model and not the prompt sample.
    rng = random.Random(args.seed)
    offsets = sample_prompt_offsets(len(corpus), args.prompt_chars,
                                    args.continue_chars, args.num_prompts, rng)

    results = []
    for mp in args.models:
        print(f"Probing: {mp}", file=sys.stderr, flush=True)
        try:
            results.append(probe_model(
                mp, corpus, offsets, args.prompt_chars, args.continue_chars,
                args.extract_threshold, device, args.progress_every))
        except FileNotFoundError as e:
            print(f"  SKIP ({e})", file=sys.stderr, flush=True)

    # Sort by iteration so the overtraining trend reads top-to-bottom.
    results.sort(key=lambda r: (r['iter'] is None, r['iter'] if r['iter'] is not None else 0))

    print()
    print(f"Memorization probe — {args.num_prompts} prompts, "
          f"prompt={args.prompt_chars} chars, continuation={args.continue_chars} chars, "
          f"extract threshold={args.extract_threshold} chars, seed={args.seed}")
    print("Metric 1 (extractable memorization): exact-match prefix of GREEDY free continuation vs truth, in chars.")
    print("Metric 2 (teacher-forced acc): per-token argmax==truth under teacher forcing (within-tokenizer trend only).")
    print()
    header = (f"{'iter':>9}  {'tok':>4}  {'best_val':>9}  "
              f"{'mean_chr':>9}  {'med_chr':>8}  {'p95_chr':>8}  "
              f"{'extract%':>9}  {'tf_acc%':>8}  checkpoint")
    print(header)
    print('-' * len(header))
    for r in results:
        bv = f"{r['best_val']:.4f}" if isinstance(r['best_val'], float) else '?'
        print(f"{fmt_iter(r['iter']):>9}  {r['tokenizer']:>4}  {bv:>9}  "
              f"{r['mean_match']:>9.1f}  {r['median_match']:>8.0f}  {r['p95_match']:>8.0f}  "
              f"{100 * r['frac_extracted']:>8.1f}%  {100 * r['tf_acc']:>7.1f}%  "
              f"{os.path.basename(r['path'])}")
    print()
    print("Reading the table: if mean_chr / extract% rise monotonically with iter")
    print("for a single tokenizer, memorization is growing past the val minimum —")
    print("the overtraining hypothesis is quantitatively confirmed.")


if __name__ == '__main__':
    main()
