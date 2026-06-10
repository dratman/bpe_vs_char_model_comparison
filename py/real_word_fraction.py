#!/usr/bin/env python
"""
real_word_fraction.py - Fraction of generated words that are real words.

Motivation (see diary 098 and diary 099, step 1):
    Two models can reach similar loss while one produces real English
    words and the other produces plausibly-shaped neologisms (*plake*,
    *culty*, *capacific* — the no-GELU run at iter 20K). Diary 098
    proposed the natural metric that separates the two behaviors:
    "fraction of generated tokens that are real English words" — a
    measure that probes the lexical inventory specifically, separately
    from the statistical envelope. This script computes that metric
    across checkpoints, turning a qualitative sample observation into
    a curve over training time.

What counts as a "real word":
    A word that occurs at least --min_count times in the TRAINING
    CORPUS (lowercased). The corpus is the right reference — the model
    can only have learned words it actually saw, and a 19th-century
    Gutenberg corpus contains proper nouns and period vocabulary that
    /usr/share/dict/words lacks. The corpus word-count table is built
    once and cached beside the corpus as <corpus>.wordvocab.pkl
    (invalidated automatically if the corpus is newer than the cache).

Method:
    For each checkpoint: generate --num_samples continuations of
    --sample_chars tokens each (default prompt is a single space,
    temperature 0.8, top_k 40 — matching the training-time samples
    that diary 098 compared). Words are extracted with the same
    regex used to build the corpus vocabulary; the first and last
    word of every sample are discarded because they may be truncated
    at the sample boundary. torch.manual_seed(seed + sample_index) is
    set before each sample, so every checkpoint sees the same RNG
    stream and differences are due to the model alone.

Usage (run from repo root, on the machine that has pt/ and txt_local/):
    python py/real_word_fraction.py \
        --models pt/char_uppercase_16L_1280_iter20000.pt \
                 pt/char_uppercase_16L_1280_no_gelu_matched_lr_iter20000.pt \
        --corpus txt_local/corpus_high_quality_uppercase_2026_05_08.txt \
        --num_samples 16 --sample_chars 512 --seed 42 \
        --tsv plots/real_word_fraction.tsv

Pass many checkpoints at once to get a single iter-sorted table.
The --tsv output has one row per checkpoint for later plotting.
"""

import os
import re
import sys
import time
import pickle
import argparse
from collections import Counter

import torch

# Reuse the established loader (meta-path derivation + sample.py-matching
# model/tokenizer load) and the established sampling path.
from memorization_probe import load_model_and_tokenizer
from sample import generate_local

# Letters plus internal apostrophes: "don't" is one word, "end-of-line"
# splits into three (its components are usually real words themselves).
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


def build_corpus_vocab(corpus_path, cache_path=None):
    """Return Counter of lowercased word -> count over the corpus.

    Cached to <corpus>.wordvocab.pkl; rebuilt if the corpus file is
    newer than the cache (same invalidation rule train.py uses for its
    token cache).
    """
    if cache_path is None:
        cache_path = corpus_path + '.wordvocab.pkl'

    if (os.path.exists(cache_path)
            and os.path.getmtime(cache_path) >= os.path.getmtime(corpus_path)):
        print(f"Loading corpus word vocab from cache: {cache_path}",
              file=sys.stderr, flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Building corpus word vocab (one pass over {corpus_path})...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    counts = Counter()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            counts.update(m.group(0).lower() for m in WORD_RE.finditer(line))
    print(f"  {len(counts):,} distinct words, {sum(counts.values()):,} total, "
          f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(counts, f)
    print(f"  cached to {cache_path}", file=sys.stderr, flush=True)
    return counts


def extract_scorable_words(text):
    """Words from a generated sample, dropping the first and last (they
    may be truncated at the sample boundaries)."""
    words = [m.group(0).lower() for m in WORD_RE.finditer(text)]
    return words[1:-1] if len(words) > 2 else []


@torch.no_grad()
def sample_checkpoint(model_path, vocab, min_count, args, device):
    """Generate samples from one checkpoint and score real-word fraction."""
    model, tokenizer, iter_num, best_val = load_model_and_tokenizer(model_path, device)
    try:
        best_val = float(best_val)  # may be stored as a 0-dim tensor
    except (TypeError, ValueError):
        best_val = None

    prompt_ids = tokenizer.encode(args.prompt)
    nonwords = Counter()
    n_words = 0
    n_real = 0

    for i in range(args.num_samples):
        # Same RNG stream per sample index across every checkpoint.
        torch.manual_seed(args.seed + i)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

        # Char tokenizer: 1 token = 1 char. BPE: ~4.5 chars/token, so
        # fewer tokens cover the same character budget.
        if tokenizer.tokenizer_type == 'char':
            max_new = args.sample_chars
        else:
            max_new = args.sample_chars // 3 + 8

        y = generate_local(
            model, x, max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            rep_penalty=1.0,  # off, matching training-time samples
            device=device,
            stop_token_id=None,
        )
        gen_text = tokenizer.decode(y[0].tolist()[len(prompt_ids):])

        for w in extract_scorable_words(gen_text):
            n_words += 1
            if vocab.get(w, 0) >= min_count:
                n_real += 1
            else:
                nonwords[w] += 1

        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"    [{os.path.basename(model_path)}] "
                  f"{i + 1}/{args.num_samples} samples",
                  file=sys.stderr, flush=True)

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'path': model_path,
        'iter': iter_num,
        'best_val': best_val,
        'n_words': n_words,
        'n_real': n_real,
        'frac': (n_real / n_words) if n_words else 0.0,
        'nonwords': nonwords,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fraction of generated words present in the training corpus vocabulary')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='One or more checkpoint .pt files (compared in one table)')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Training corpus text file (source of the word vocabulary)')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Corpus occurrences at/above which a word counts as real '
                             '(filters OCR junk and one-off typos in the corpus itself)')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Generated samples per checkpoint')
    parser.add_argument('--sample_chars', type=int, default=512,
                        help='Characters generated per sample')
    parser.add_argument('--prompt', type=str, default=' ',
                        help='Generation prompt (default: single space, matching '
                             'the training-time samples diary 098 compared)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default matches training-time samples)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling cutoff (default matches training-time samples)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base RNG seed; sample i uses seed+i on every checkpoint')
    parser.add_argument('--show_nonwords', type=int, default=15,
                        help='How many most-frequent non-words to print per checkpoint')
    parser.add_argument('--tsv', type=str, default=None,
                        help='Optional path to write one TSV row per checkpoint for plotting')
    parser.add_argument('--progress_every', type=int, default=0,
                        help='Print progress to stderr every N samples (0=silent)')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')

    if not os.path.exists(args.corpus):
        print(f"Error: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    vocab = build_corpus_vocab(args.corpus)
    print(f"device: {device} | real word = corpus count >= {args.min_count}",
          file=sys.stderr, flush=True)

    results = []
    for mp in args.models:
        print(f"Sampling: {mp}", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            r = sample_checkpoint(mp, vocab, args.min_count, args, device)
        except FileNotFoundError as e:
            print(f"  SKIP ({e})", file=sys.stderr, flush=True)
            continue
        print(f"  real-word fraction {100 * r['frac']:.1f}% "
              f"({r['n_real']}/{r['n_words']} words, {time.time() - t0:.0f}s)",
              file=sys.stderr, flush=True)
        results.append(r)

    results.sort(key=lambda r: (r['iter'] is None, r['iter'] if r['iter'] is not None else 0))

    print()
    print(f"Real-word fraction — {args.num_samples} samples x {args.sample_chars} chars "
          f"per checkpoint, temp={args.temperature}, top_k={args.top_k}, "
          f"prompt={args.prompt!r}, seed={args.seed}")
    print(f"Real word = appears >= {args.min_count}x in {os.path.basename(args.corpus)} "
          f"(first/last word of each sample excluded as boundary-truncated)")
    print()
    # 'val' is best_val_loss for best/final checkpoints, or the at-save
    # eval val_loss for periodic _iter{N} checkpoints.
    header = (f"{'iter':>9}  {'val':>9}  {'words':>6}  {'real%':>7}  "
              f"top non-words (count)  |  checkpoint")
    print(header)
    print('-' * 100)
    for r in results:
        bv = f"{r['best_val']:.4f}" if isinstance(r['best_val'], float) else '?'
        top_nw = ', '.join(f"{w}({c})" for w, c in
                           r['nonwords'].most_common(args.show_nonwords))
        it = str(r['iter']) if r['iter'] is not None else '?'
        print(f"{it:>9}  {bv:>9}  {r['n_words']:>6}  {100 * r['frac']:>6.1f}%  "
              f"{top_nw}  |  {os.path.basename(r['path'])}")

    if args.tsv:
        os.makedirs(os.path.dirname(args.tsv) or '.', exist_ok=True)
        with open(args.tsv, 'w', encoding='utf-8') as f:
            f.write("checkpoint\titer\tval_loss\tn_words\tn_real\treal_frac\tnonwords\n")
            for r in results:
                bv = f"{r['best_val']:.6f}" if isinstance(r['best_val'], float) else ''
                nw = ','.join(f"{w}:{c}" for w, c in r['nonwords'].most_common(50))
                it = str(r['iter']) if r['iter'] is not None else ''
                f.write(f"{os.path.basename(r['path'])}\t{it}\t{bv}\t"
                        f"{r['n_words']}\t{r['n_real']}\t{r['frac']:.6f}\t{nw}\n")
        print(f"\nTSV written: {args.tsv}")


if __name__ == '__main__':
    main()
