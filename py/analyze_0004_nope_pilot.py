#!/usr/bin/env python3
"""
analyze_0004_nope_pilot.py — coupler-queue 0004 analysis (PE vs NoPE char pilot).

Inputs:
  * the two training logs (val loss every 200 iters),
  * the real-word-fraction TSV and the per-checkpoint sample files written by
    py/real_word_fraction.py --tsv ... --samples_out ... (same seeds for both arms).

Outputs:
  * <out>/0004_val_bpc.csv           val loss in bits per character, both arms
  * <out>/0004_nope_pilot.png        val bpc + real-word fraction vs iteration
  * printed tables: bpc headline, real-word fraction, and the spelling ERROR
    SIGNATURE for each checkpoint:
      - real-word rate by word length,
      - where in the word the error starts (letters correct before the first
        letter that leaves every real word's spelling),
      - one-edit classification of each non-word: would undoubling a doubled
        letter, doubling a letter, or swapping two adjacent letters turn it
        into a real word (versus some other one-letter edit, or none)?

A "real word" is the same as in real_word_fraction.py: corpus count >= min_count.
"""
import os
import re
import csv
import math
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_word_fraction import build_corpus_vocab, extract_scorable_words

LN2 = math.log(2.0)
PAT = re.compile(r"Step\s+(\d+)\s+\|.*\|\s*val loss\s+([0-9.]+)\s*\|")
LETTERS = "abcdefghijklmnopqrstuvwxyz'"


def parse_log(path):
    series = {}
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                series[int(m.group(1))] = float(m.group(2)) / LN2
    return series


def read_tsv(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["checkpoint"]] = r
    return rows


def one_edit_class(w, real):
    """Classify how a non-word w relates to the real-word set by one edit."""
    # undouble: remove one letter of a doubled pair (LETTTER -> LETTER)
    for i in range(len(w) - 1):
        if w[i] == w[i + 1] and (w[:i] + w[i + 1:]) in real:
            return "undouble"
    # double: duplicate one letter (LETER -> LETTER)
    for i in range(len(w)):
        if (w[:i + 1] + w[i] + w[i + 1:]) in real:
            return "double"
    # adjacent transposition (FORM <-> FROM)
    for i in range(len(w) - 1):
        if w[i] != w[i + 1] and (w[:i] + w[i + 1] + w[i] + w[i + 2:]) in real:
            return "transpose"
    # any other single edit: substitution, insertion, deletion
    for i in range(len(w) + 1):
        if i < len(w) and (w[:i] + w[i + 1:]) in real:
            return "other1"
        for c in LETTERS:
            if (w[:i] + c + w[i:]) in real:
                return "other1"
            if i < len(w) and c != w[i] and (w[:i] + c + w[i + 1:]) in real:
                return "other1"
    return "none"


def divergence_pos(w, prefixes):
    """Number of leading letters of w that are still a prefix of some real word."""
    k = 0
    while k < len(w) and w[:k + 1] in prefixes:
        k += 1
    return k


def signature(sample_path, real, prefixes):
    with open(sample_path, encoding="utf-8") as f:
        samples = f.read().split("\n" + "=" * 40 + "\n")
    by_len = defaultdict(lambda: [0, 0])          # len bucket -> [n, n_real]
    div = Counter()
    edit = Counter()
    n = n_real = 0
    for s in samples:
        for w in extract_scorable_words(s):
            n += 1
            b = min(len(w), 10)
            by_len[b][0] += 1
            if w in real:
                n_real += 1
                by_len[b][1] += 1
            else:
                edit[one_edit_class(w, real)] += 1
                div[min(divergence_pos(w, prefixes), 8)] += 1
    return n, n_real, by_len, div, edit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_pe", required=True)
    ap.add_argument("--log_nope", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- loss curves (bits per character)
    pe, nope = parse_log(a.log_pe), parse_log(a.log_nope)
    steps = sorted(set(pe) | set(nope))
    csv_path = os.path.join(a.out, "0004_val_bpc.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iter", "pe_val_bpc", "nope_val_bpc", "nope_minus_pe"])
        for s in steps:
            p, q = pe.get(s), nope.get(s)
            wr.writerow([s, f"{p:.4f}" if p else "", f"{q:.4f}" if q else "",
                         f"{q - p:+.4f}" if p and q else ""])
    print(f"CSV: {csv_path}")
    print("\nval bpc (selected iters):")
    print(f"{'iter':>6} {'PE':>7} {'NoPE':>7} {'gap':>7}")
    for s in steps:
        if s % 1000 == 0 or s in (200, 400, 600):
            p, q = pe.get(s), nope.get(s)
            if p and q:
                print(f"{s:>6} {p:7.4f} {q:7.4f} {q - p:+7.4f}")
    bp, bq = min(pe.values()), min(nope.values())
    print(f"best  PE {bp:.4f} (iter {min(pe, key=pe.get)})  NoPE {bq:.4f} "
          f"(iter {min(nope, key=nope.get)})  gap {bq - bp:+.4f} bpc ({100 * (bq - bp) / bp:+.2f}%)")
    print(f"final PE {pe[max(pe)]:.4f}  NoPE {nope[max(nope)]:.4f}  (iter {max(steps)})")

    # ---- real-word fraction
    rows = read_tsv(a.tsv)
    rwf = {"pe": {}, "nope": {}}
    for name, r in rows.items():
        arm = "nope" if "_nope_mps" in name else "pe"  # every name has "char_nope_pilot"
        if "_iter" in name and r["iter"]:
            rwf[arm][int(r["iter"])] = float(r["real_frac"])
    print("\nreal-word fraction (same sampling seeds both arms):")
    print(f"{'iter':>6} {'PE':>7} {'NoPE':>7}")
    for it in sorted(set(rwf["pe"]) | set(rwf["nope"])):
        p, q = rwf["pe"].get(it), rwf["nope"].get(it)
        print(f"{it:>6} {100 * p if p is not None else float('nan'):6.1f}% "
              f"{100 * q if q is not None else float('nan'):6.1f}%")

    # ---- plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm, ser, c in (("PE (positional embedding)", pe, "#2a6fdb"),
                        ("NoPE (no positional embedding)", nope, "#d9534f")):
        xs = sorted(ser)
        ax1.plot(xs, [ser[x] for x in xs], label=arm, color=c)
    ax1.set_xlabel("iteration"); ax1.set_ylabel("validation loss (bits per character)")
    ax1.set_ylim(top=min(3.0, ax1.get_ylim()[1])); ax1.grid(alpha=0.3); ax1.legend()
    ax1.set_title("Loss")
    for arm, key, c in (("PE", "pe", "#2a6fdb"), ("NoPE", "nope", "#d9534f")):
        xs = sorted(rwf[key])
        ax2.plot(xs, [100 * rwf[key][x] for x in xs], "o-", label=arm, color=c)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("generated words that are real words (%)")
    ax2.grid(alpha=0.3); ax2.legend(); ax2.set_title("Spelling (real-word fraction)")
    fig.suptitle("0004: 6-layer character model, with vs without positional embedding (Mac Studio)")
    fig.tight_layout()
    png = os.path.join(a.out, "0004_nope_pilot.png")
    fig.savefig(png, dpi=130)
    print(f"\nPNG: {png}")

    # ---- error signature
    vocab = build_corpus_vocab(a.corpus)
    real = {w for w, c in vocab.items() if c >= a.min_count}
    prefixes = set()
    for w in real:
        for k in range(1, len(w) + 1):
            prefixes.add(w[:k])
    for fn in sorted(os.listdir(a.samples_dir)):
        if not fn.endswith(".samples.txt") or "_iter" not in fn:
            continue
        n, nr, by_len, div, edit = signature(os.path.join(a.samples_dir, fn), real, prefixes)
        nw = n - nr
        print(f"\n== {fn}: {n} words, {nr} real ({100 * nr / max(n, 1):.1f}%), {nw} non-words")
        print("   real% by word length: " + "  ".join(
            f"{L if L < 10 else '10+'}:{100 * by_len[L][1] / by_len[L][0]:.0f}%(n={by_len[L][0]})"
            for L in sorted(by_len) if by_len[L][0]))
        print("   non-words: letters still on a real spelling before the first wrong letter: " +
              "  ".join(f"{k if k < 8 else '8+'}:{div[k]}" for k in sorted(div)))
        print("   non-words fixable by one edit: " + "  ".join(
            f"{k}:{edit[k]}({100 * edit[k] / max(nw, 1):.0f}%)"
            for k in ("undouble", "double", "transpose", "other1", "none")))


if __name__ == "__main__":
    main()
