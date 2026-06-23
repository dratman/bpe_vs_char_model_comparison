#!/usr/bin/env python
"""
category_geometry_compare.py - Run the diary-105 category-geometry probe on
the char model and the BPE model on the IDENTICAL words, and compare.

The point of contrast (diary 105): in a CHARACTER model the category "animal"
has no embedding row -- it must be ASSEMBLED in the residual stream over the
first several layers, and below that the representation is dominated by
spelling (so the minimal pairs cat/hat, fox/box, ... pull each animal toward
its look-alike object, giving NEGATIVE separation early). In a BPE model every
one of these 18 words is a SINGLE token (verified: Gcat, Ghat, ... are distinct
vocab rows), so the category can be looked up directly from the embedding and
there is no letter-sharing between cat and hat at the token level.

Prediction: the BPE model shows category separation EARLY (already at the
embedding / layer 0) and never dips negative; the char model starts near zero
or negative and only separates after a mid-network assembly band (~L5-L9).
"Assembled vs looked-up" should appear as a difference in DEPTH-OF-ONSET.

Same words, frames, mean-centering, permutation null, and minimal-pair test as
category_geometry_probe.py -- it imports that module's internals unchanged, and
the per-model metrics come from category_geometry_sweep.metrics_for_checkpoint,
so the char numbers match diary 105 exactly. Read-only; CPU by default.

Usage:
    python py/category_geometry_compare.py \
        --char pt/char_uppercase_16L_1280.pt \
        --bpe  pt/bpe_uppercase_16L_1280_b2_resumed.pt \
        --out_tsv terminal_logs/category_geometry_compare_<date>.tsv \
        --out_plot plots/category_geometry_compare_<date>.png
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import category_geometry_probe as cg
from category_geometry_sweep import metrics_for_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", default="pt/char_uppercase_16L_1280.pt")
    ap.add_argument("--bpe", default="pt/bpe_uppercase_16L_1280_b2_resumed.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", default=None)
    ap.add_argument("--out_plot", default=None)
    args = ap.parse_args()

    models = [("char", args.char), ("bpe", args.bpe)]
    results = {}
    for name, path in models:
        if not os.path.exists(path):
            sys.exit(f"Error: {name} model '{path}' not found")
        print(f"Probing {name}: {path} ...", flush=True)
        rng = np.random.default_rng(args.seed)  # same seed per model -> comparable null
        _iter, rows = metrics_for_checkpoint(path, args.device, args.n_perm, rng)
        results[name] = rows

    # layer axis (shared: both models are 16 layers -> embed + L00..L15)
    layers = [r["layer"] for r in results["char"] if r["readout"] == "after"]
    layer_idx = list(range(len(layers)))

    def grid(name, readout, field):
        sub = [r for r in results[name] if r["readout"] == readout]
        sub.sort(key=lambda r: r["layer_idx"])
        return [r[field] for r in sub]

    # ---- report --------------------------------------------------------
    for readout in ("final", "after"):
        label = ("word's own token / final letter" if readout == "final"
                 else "position after the word")
        print("\n" + "=" * 72)
        print(f"SEPARATION by layer  [readout: {label}]")
        print("  (separation = within-category cosine minus between-category)")
        print("=" * 72)
        print(f"{'layer':>6} {'char_sep':>9} {'bpe_sep':>9}   "
              f"{'char_mp':>7} {'bpe_mp':>7}")
        cs = grid("char", readout, "separation")
        bs = grid("bpe", readout, "separation")
        cm = grid("char", readout, "minpair")
        bm = grid("bpe", readout, "minpair")
        for i, ly in enumerate(layers):
            print(f"{ly:>6} {cs[i]:>9.3f} {bs[i]:>9.3f}   "
                  f"{cm[i]:>5}/9 {bm[i]:>5}/9")
        print(f"  embed: char {cs[0]:+.3f} vs bpe {bs[0]:+.3f}   |   "
              f"L00: char {cs[1]:+.3f} vs bpe {bs[1]:+.3f}")

    # headline contrast at the embedding (the looked-up vs assembled tell)
    cs_f = grid("char", "final", "separation")
    bs_f = grid("bpe", "final", "separation")
    print("\n" + "-" * 72)
    print("HEADLINE (word's-own-token readout):")
    print(f"  At the embedding layer the BPE category separation is "
          f"{bs_f[0]:+.3f} (looked up),")
    print(f"  while the char separation is {cs_f[0]:+.3f} "
          f"(no word unit yet -- only a letter).")
    # where does char first cross positive at the after readout?
    cs_a = grid("char", "after", "separation")
    bs_a = grid("bpe", "after", "separation")
    first_pos = next((layers[i] for i in range(len(layers)) if cs_a[i] > 0.05), "never")
    print(f"  After-word readout: char separation first exceeds 0.05 at "
          f"{first_pos}; bpe is already {bs_a[0]:+.3f} at embed.")

    # ---- TSV -----------------------------------------------------------
    if args.out_tsv:
        with open(args.out_tsv, "w") as f:
            f.write("model\treadout\tlayer_idx\tlayer\tseparation\tz\tp\tminpair\n")
            for name in ("char", "bpe"):
                for r in sorted(results[name], key=lambda r: (r["readout"], r["layer_idx"])):
                    f.write(f"{name}\t{r['readout']}\t{r['layer_idx']}\t{r['layer']}\t"
                            f"{r['separation']:.4f}\t{r['z']:.3f}\t{r['p']:.4f}\t"
                            f"{r['minpair']}\n")
        print(f"\nWrote {args.out_tsv}")

    # ---- plot ----------------------------------------------------------
    if args.out_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            for ax, readout, title in [
                (axes[0], "final", "Readout: word's own token (final letter for char)"),
                (axes[1], "after", "Readout: position after the word")]:
                ax.axhline(0, color="grey", lw=0.8, ls="--")
                ax.plot(layer_idx, grid("char", readout, "separation"),
                        "o-", color="C0", label="char (assembled)")
                ax.plot(layer_idx, grid("bpe", readout, "separation"),
                        "s-", color="C3", label="bpe (looked up)")
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("layer")
                ax.set_xticks(layer_idx)
                ax.set_xticklabels(layers, rotation=90, fontsize=7)
                ax.legend()
            axes[0].set_ylabel("category separation\n(within minus between cosine)")
            fig.suptitle("Category geometry: char vs BPE on the same 18 words "
                         "(animals vs surface-matched objects)", fontsize=12)
            fig.tight_layout()
            fig.savefig(args.out_plot, dpi=120)
            print(f"Wrote {args.out_plot}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
