#!/usr/bin/env python
"""
plot_real_word_fraction.py - Plot the diary-100 real-word-fraction curves.

Reads the TSVs written by py/real_word_fraction.py (one row per
checkpoint) for the baseline char run and the no-GELU matched-LR run,
and renders the centerpiece figure of the diary-099 write-up plan:
fraction of generated words that are real corpus words, vs training
iteration, for both architectures. Error bars are binomial standard
error (sqrt(p(1-p)/n) on ~1,500 words per point).

Usage (from repo root):
    python py/plot_real_word_fraction.py \
        --baseline_tsv terminal_logs/real_word_fraction_baseline_2026_06_10_1507.tsv \
        --no_gelu_tsv terminal_logs/real_word_fraction_no_gelu_2026_06_10_1507.tsv \
        --out plots/real_word_fraction_2026_06_10.png
"""

import os
import csv
import math
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_tsv(path):
    iters, fracs, errs = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            n = int(row['n_words'])
            p = float(row['real_frac'])
            iters.append(int(row['iter']))
            fracs.append(100.0 * p)
            errs.append(100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / n))
    return iters, fracs, errs


def main():
    parser = argparse.ArgumentParser(description='Plot real-word-fraction curves')
    parser.add_argument('--baseline_tsv', type=str, required=True)
    parser.add_argument('--no_gelu_tsv', type=str, required=True)
    parser.add_argument('--out', type=str, default='plots/real_word_fraction.png')
    args = parser.parse_args()

    b_it, b_fr, b_er = read_tsv(args.baseline_tsv)
    n_it, n_fr, n_er = read_tsv(args.no_gelu_tsv)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(b_it, b_fr, yerr=b_er, marker='o', ms=4, lw=1.5, capsize=2,
                color='tab:blue', label='baseline (with GELU)')
    ax.errorbar(n_it, n_fr, yerr=n_er, marker='s', ms=4, lw=1.5, capsize=2,
                color='tab:red', label='no-GELU (matched LR)')

    # The 4x-corpus-exposure correspondence: no-GELU@80K matches baseline@20K
    ax.annotate('no-GELU @ 80K ≈ baseline @ 20K\n(~4× corpus exposure)',
                xy=(n_it[-1], n_fr[-1]), xytext=(120000, 92),
                fontsize=9, color='dimgray',
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=1))

    ax.set_xscale('log')
    ax.set_xlabel('training iteration (log scale)')
    ax.set_ylabel('real-word fraction of generated text (%)')
    ax.set_ylim(78, 101)
    ax.set_title('Lexical inventory acquisition: real-word fraction vs training\n'
                 '(16 samples × 512 chars/checkpoint, temp 0.8, top_k 40; '
                 'real = ≥5 occurrences in training corpus)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
