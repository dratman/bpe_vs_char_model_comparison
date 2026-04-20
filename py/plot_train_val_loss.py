"""
Plot train and val loss for all BPE training runs on a single plot.
Uses per-iteration train loss (from "iter N: loss X" lines)
and per-eval val loss (from "Step N | ... val loss X" lines).

Usage:
    python py/plot_train_val_loss.py
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt

RUNS = [
    {
        'log': 'terminal_logs/terminal_log_for_bpe_16L16H_2026_04_15_1256.txt',
        'label': '2.5GB unshuffled (block=1024, batch=16)',
        'train_color': 'coral',
        'val_color': 'darkred',
    },
    {
        'log': 'terminal_logs/terminal_log_for_bpe_16L16H_2026_04_19_1124.txt',
        'label': '2.1GB doc-shuf 32K bf16 (block=2048, batch=4)',
        'train_color': 'limegreen',
        'val_color': 'darkgreen',
    },
]

# "iter 30: loss 6.8562, time 4992.84ms, mfu 5.60%"
ITER_RE = re.compile(r'^iter\s+(\d+):\s+loss\s+([\d.]+)')

# "Step 163000 | Epoch 1.46 | train loss 2.9903 | val loss 3.0544"
STEP_RE = re.compile(
    r'Step\s+(\d+)\s*\|\s*Epoch\s+([\d.]+)\s*\|.*train loss\s+([\d.]+)\s*\|.*val loss\s+([\d.]+)'
)


def parse_log(path):
    """Parse a terminal log for per-iter train loss and per-eval val loss."""
    iter_steps, iter_losses = [], []
    val_steps, val_epochs, val_train, val_losses = [], [], [], []
    if not os.path.exists(path):
        return (np.array(iter_steps), np.array(iter_losses),
                np.array(val_steps), np.array(val_epochs),
                np.array(val_train), np.array(val_losses))
    with open(path) as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                iter_steps.append(int(m.group(1)))
                iter_losses.append(float(m.group(2)))
                continue
            m = STEP_RE.search(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_epochs.append(float(m.group(2)))
                val_train.append(float(m.group(3)))
                val_losses.append(float(m.group(4)))
    return (np.array(iter_steps), np.array(iter_losses),
            np.array(val_steps), np.array(val_epochs),
            np.array(val_train), np.array(val_losses))


# Parse all runs
parsed = []
for run in RUNS:
    iters, iter_loss, val_steps, val_epochs, val_train, val_loss = parse_log(run['log'])
    parsed.append((iters, iter_loss, val_steps, val_epochs, val_train, val_loss))
    n_train = len(iters)
    n_val = len(val_steps)
    print(f"{run['label']}: {n_train} train points, {n_val} eval points", end="")
    if n_train:
        print(f", iters {iters[0]}–{iters[-1]}, loss {iter_loss[0]:.4f} -> {iter_loss[-1]:.4f}")
    else:
        print()

# Find the latest (currently training) run to set x-axis range
latest = None
for i, (iters, _, _, _, _, _) in enumerate(parsed):
    if len(iters):
        if latest is None or iters[-1] > parsed[latest][0][-1]:
            latest = i

# Clip all runs to the range of the latest run
if latest is not None:
    max_iter = parsed[latest][0][-1]
    clipped = []
    for iters, iter_loss, val_steps, val_epochs, val_train, val_loss in parsed:
        mask = iters <= max_iter
        vmask = val_steps <= max_iter if len(val_steps) else np.array([], dtype=bool)
        clipped.append((
            iters[mask], iter_loss[mask],
            val_steps[vmask], val_epochs[vmask],
            val_train[vmask], val_loss[vmask]
        ))
    parsed = clipped

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

for i, run in enumerate(RUNS):
    iters, iter_loss, val_steps, val_epochs, val_train, val_loss = parsed[i]
    if len(iters):
        ax.plot(iters, iter_loss, linewidth=0.5, alpha=0.6,
                color=run['train_color'], label=f"{run['label']} — train")
    if len(val_steps):
        ax.plot(val_steps, val_loss, linewidth=1.5, alpha=0.9,
                color=run['val_color'], label=f"{run['label']} — val",
                linestyle='--')

# Per-character equivalent on right y-axis
ax2 = ax.twinx()
ymin, ymax = ax.get_ylim()
ax2.set_ylim(ymin / 4.09, ymax / 4.09)
ax2.set_ylabel('Per-character equivalent loss (÷ 4.09)', fontsize=11, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Loss (per BPE token)', fontsize=13)
ax.set_title('BPE 16L16H — Training Loss Comparison', fontsize=15)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Show latest epoch for the current run
latest_iters, _, _, latest_epochs, _, _ = parsed[-1]
if len(latest_epochs):
    epoch_text = f'Current run epoch: {latest_epochs[-1]:.2f}'
    ax.text(0.02, 0.02, epoch_text, transform=ax.transAxes,
            fontsize=12, color='darkgreen', verticalalignment='bottom')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
out_path = 'plots/train_val_loss_old_vs_new.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
