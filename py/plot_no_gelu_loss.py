"""
Plot train and val loss over iterations for the no-GELU matched-LR
char model run (pt/char_uppercase_16L_1280_no_gelu_matched_lr.pt).

Train loss comes from per-iteration "iter N: loss X" lines (noisy, so we
also draw a running-mean smoothing). Val loss comes from the periodic
"Step N | Epoch E | train loss T | val loss V" eval lines.

Usage:
    python py/plot_no_gelu_loss.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "terminal_logs/terminal_log_for_char_uppercase_16L_1280_no_gelu_matched_lr_2026_06_06_1344.txt"
OUT = "plots/no_gelu_matched_lr_loss_2026_06_25.png"

ITER_RE = re.compile(r"^iter\s+(\d+):\s+loss\s+([\d.]+)")
STEP_RE = re.compile(
    r"Step\s+(\d+)\s*\|\s*Epoch\s+([\d.]+)\s*\|.*train loss\s+([\d.]+)\s*\|.*val loss\s+([\d.]+)"
)


def running_mean(y, w):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def main():
    it_x, it_y = [], []
    val_x, val_y, vtrain_y = [], [], []
    with open(LOG) as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                it_x.append(int(m.group(1)))
                it_y.append(float(m.group(2)))
                continue
            m = STEP_RE.search(line)
            if m:
                val_x.append(int(m.group(1)))
                vtrain_y.append(float(m.group(3)))
                val_y.append(float(m.group(4)))

    it_x = np.array(it_x); it_y = np.array(it_y)
    val_x = np.array(val_x); val_y = np.array(val_y); vtrain_y = np.array(vtrain_y)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Raw per-iter train loss, faint
    ax.plot(it_x, it_y, color="lightsteelblue", lw=0.5, alpha=0.5,
            label="train loss (per iter, raw)")
    # Smoothed train loss
    w = 50
    sm = running_mean(it_y, w)
    sm_x = it_x[w - 1:]
    ax.plot(sm_x, sm, color="steelblue", lw=1.6,
            label=f"train loss (running mean, w={w})")
    # Val loss
    ax.plot(val_x, val_y, color="darkred", lw=1.8, marker="o", ms=3,
            label="val loss (eval)")

    best_i = int(np.argmin(val_y))
    ax.axhline(val_y[best_i], color="darkred", ls=":", lw=0.8, alpha=0.6)
    ax.annotate(f"best val {val_y[best_i]:.4f} @ iter {val_x[best_i]:,}",
                xy=(val_x[best_i], val_y[best_i]),
                xytext=(0.45, 0.85), textcoords="axes fraction",
                color="darkred", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (per-character, char tokenizer)")
    ax.set_title("char 16L/8H n_embd=1280, no-GELU, matched LR 1.5e-4 — "
                 f"train/val loss (through iter {it_x[-1]:,})")
    ax.set_ylim(0.5, 1.6)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print(f"train iters parsed: {len(it_x)}  (last iter {it_x[-1]:,}, last loss {it_y[-1]:.4f})")
    print(f"val points parsed:  {len(val_x)}  (last val {val_y[-1]:.4f} @ iter {val_x[-1]:,})")
    print(f"best val {val_y[best_i]:.4f} @ iter {val_x[best_i]:,}; "
          f"current val {val_y[-1]:.4f} ({'still improving' if best_i==len(val_y)-1 else f'{len(val_y)-1-best_i} evals past best'})")


if __name__ == "__main__":
    main()
