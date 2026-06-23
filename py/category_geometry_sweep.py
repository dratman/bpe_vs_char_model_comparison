#!/usr/bin/env python
"""
category_geometry_sweep.py - Run category_geometry_probe across training
checkpoints to watch the category region FORM over training.

Diary 105 showed that in the final char model the "animal" region is assembled
in a mid-network band (separation goes positive around layers 5-9 at the
after-word readout; the minimal-pair decision locks to 9/9 by layer 7). This
driver runs the identical probe on every saved iter-checkpoint of the
char_uppercase_16L_1280 run and asks: does that band appear gradually as
training proceeds, or snap into place? (The browser conversation that started
diary 105 raised exactly this -- generalization sometimes appears suddenly.)

It reuses the validated internals of category_geometry_probe.py unchanged, so
the per-checkpoint numbers are directly comparable to diary 105's single-model
result. Read-only; defaults to CPU so it never contends with a live MPS run.

Outputs:
  - long-form TSV: one row per (iter, readout, layer)
  - iter x layer heatmaps (separation and minimal-pair count) for the
    after-word readout -- the picture of the band forming over training.

Usage:
  python py/category_geometry_sweep.py \
      --glob 'pt/char_uppercase_16L_1280_iter*.pt' --include_final \
      --out_tsv terminal_logs/category_geometry_sweep_2026_06_23.tsv \
      --out_heatmap plots/category_geometry_sweep_2026_06_23.png
"""

import os
import sys
import gc
import glob as globmod
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import category_geometry_probe as cg


def metrics_for_checkpoint(model_path, device, n_perm, rng):
    """Load one checkpoint, run the probe, return (iter_num, rows) where each
    row is dict(readout, layer_idx, layer, within, between, separation, z, p,
    minpair). Frees the model before returning."""
    words = cg.ANIMALS + cg.OBJECTS
    n_animals = len(cg.ANIMALS)
    labels = np.array([0] * len(cg.ANIMALS) + [1] * len(cg.OBJECTS))

    model, tokenizer, ckpt = cg.load_model_and_tokenizer(model_path, device)
    iter_num = ckpt.get("iter_num", -1)
    store, handles, layer_names = cg.register_residual_hooks(model)
    vectors = cg.collect_word_vectors(model, tokenizer, words, device,
                                      layer_names, store)
    for h in handles:
        h.remove()

    rows = []
    for readout in ("final", "after"):
        for li, nm in enumerate(layer_names):
            cos, Vn = cg.cosine_matrix(vectors[readout][nm])
            sep, within, between = cg.separation_from_labels(cos, labels)
            null = cg.permutation_null(cos, n_animals, n_perm, rng)
            z = (sep - null.mean()) / (null.std() or 1.0)
            p = float((null >= sep).mean())
            wins, _ = cg.minimal_pair_test(Vn, n_animals)
            rows.append(dict(readout=readout, layer_idx=li, layer=nm,
                             within=within, between=between, separation=sep,
                             z=z, p=p, minpair=wins))

    # free the 3.6 GB checkpoint + model before the next iteration
    del model, ckpt, vectors, store
    gc.collect()
    return iter_num, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="pt/char_uppercase_16L_1280_iter*.pt")
    ap.add_argument("--include_final", action="store_true",
                    help="also include the _final checkpoint (iter 500K)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--out_heatmap", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(globmod.glob(args.glob))
    if args.include_final:
        final = args.glob.split("_iter")[0] + "_final.pt"
        if os.path.exists(final):
            paths.append(final)
    if not paths:
        sys.exit(f"No checkpoints matched {args.glob!r}")

    # Probe each checkpoint; collect (iter_num, rows).
    results = []
    for i, path in enumerate(paths):
        iter_num, rows = metrics_for_checkpoint(path, args.device,
                                                args.n_perm, rng)
        results.append((iter_num, rows))
        # progress: after-word peak separation + earliest 9/9 minpair layer
        after = [r for r in rows if r["readout"] == "after"]
        peak = max(after, key=lambda r: r["separation"])
        nine = [r for r in after if r["minpair"] == 9]
        onset = min(nine, key=lambda r: r["layer_idx"])["layer"] if nine else "--"
        print(f"[{i+1}/{len(paths)}] iter {iter_num:>7}: "
              f"after-word peak sep {peak['separation']:.3f} at {peak['layer']}, "
              f"first 9/9 minpair at {onset}  ({os.path.basename(path)})",
              flush=True)

    results.sort(key=lambda r: r[0])  # by iter_num

    # ---- write long-form TSV ------------------------------------------
    with open(args.out_tsv, "w") as f:
        f.write("iter\treadout\tlayer_idx\tlayer\twithin\tbetween\t"
                "separation\tz\tp\tminpair\n")
        for iter_num, rows in results:
            for r in rows:
                f.write(f"{iter_num}\t{r['readout']}\t{r['layer_idx']}\t"
                        f"{r['layer']}\t{r['within']:.4f}\t{r['between']:.4f}\t"
                        f"{r['separation']:.4f}\t{r['z']:.3f}\t{r['p']:.4f}\t"
                        f"{r['minpair']}\n")
    print(f"\nWrote {args.out_tsv}")

    # ---- iter x layer heatmaps (after-word readout) -------------------
    if args.out_heatmap:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            iters = [r[0] for r in results]
            layer_names = [d["layer"] for d in results[0][1]
                           if d["readout"] == "after"]
            n_it, n_ly = len(iters), len(layer_names)
            sep_grid = np.zeros((n_it, n_ly))
            min_grid = np.zeros((n_it, n_ly))
            for ii, (_, rows) in enumerate(results):
                after = [r for r in rows if r["readout"] == "after"]
                after.sort(key=lambda r: r["layer_idx"])
                for li, r in enumerate(after):
                    sep_grid[ii, li] = r["separation"]
                    min_grid[ii, li] = r["minpair"]

            fig, axes = plt.subplots(1, 2, figsize=(15, 8))
            ylabels = [f"{it//1000}K" for it in iters]

            ax = axes[0]
            vmax = np.abs(sep_grid).max()
            im = ax.imshow(sep_grid, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title("Category separation (after-word)\nwithin minus between cosine")
            ax.set_xlabel("layer"); ax.set_ylabel("training iter")
            ax.set_xticks(range(n_ly)); ax.set_xticklabels(layer_names, rotation=90, fontsize=7)
            ax.set_yticks(range(n_it)); ax.set_yticklabels(ylabels, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)

            ax = axes[1]
            im = ax.imshow(min_grid, aspect="auto", cmap="viridis",
                           vmin=0, vmax=9, origin="lower")
            ax.set_title("Minimal-pair count (after-word)\n# of 9 animals beating their spelling-twin")
            ax.set_xlabel("layer"); ax.set_ylabel("training iter")
            ax.set_xticks(range(n_ly)); ax.set_xticklabels(layer_names, rotation=90, fontsize=7)
            ax.set_yticks(range(n_it)); ax.set_yticklabels(ylabels, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)

            fig.suptitle("Category geometry forming over training — "
                         "char_uppercase_16L_1280", fontsize=12)
            fig.tight_layout()
            fig.savefig(args.out_heatmap, dpi=120)
            print(f"Wrote {args.out_heatmap}")
        except Exception as e:
            print(f"(heatmap skipped: {e})")


if __name__ == "__main__":
    main()
