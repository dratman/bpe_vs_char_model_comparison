#!/usr/bin/env python
"""
plot_lens_comparison.py — overlay the layer-by-layer lens readouts of several
models on one figure, NORMALIZED so the *shape* is comparable across models
that have very different absolute accuracy (a memorizer scores far higher than
a generalizer, so raw curves would mislead).

For each model and each lens we plot, per layer L:
    fraction of final performance = MRR(L) / MRR(final layer)
So every model ends at 1.0 by construction; what differs is HOW it gets there.
    - "snaps at the end"  -> stays near 0 through the middle, jumps to 1 at top.
    - "builds up early"   -> climbs steadily from the bottom layers.

Usage:
    plot_lens_comparison.py out.png  label1=stats1.pkl  label2=stats2.pkl ...
"""

import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = sys.argv[1]
    specs = sys.argv[2:]
    models = []
    for spec in specs:
        label, path = spec.split('=', 1)
        with open(path, 'rb') as f:
            models.append((label, pickle.load(f)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = ['C3', 'C0', 'C2', 'C4', 'C1']

    for ax, lens, title in ((ax1, 'logit', 'Logit lens'),
                            (ax2, 'jacobian', 'Jacobian lens')):
        for i, (label, d) in enumerate(models):
            mrr = d[lens]['mrr']
            L = d['n_layer']
            xs = [l / L for l in range(L + 1)]        # normalized depth 0..1
            final = mrr[-1] if mrr[-1] != 0 else 1e-9
            ys = [m / final for m in mrr]             # fraction of final perf
            ax.plot(xs, ys, 'o-', color=colors[i % len(colors)], label=label)
        ax.set_xlabel('layer depth (0 = embedding, 1.0 = final)')
        ax.set_ylabel('fraction of final-layer performance (MRR)')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Where the next-token answer forms across depth — '
                 'memorizers vs generalizer')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")

    # also print a compact table
    for label, d in models:
        L = d['n_layer']
        jm = d['jacobian']['mrr']
        frac = [round(m / (jm[-1] or 1e-9), 2) for m in jm]
        print(f"{label:>16} (jacobian) fraction-of-final by layer: {frac}")


if __name__ == '__main__':
    main()
