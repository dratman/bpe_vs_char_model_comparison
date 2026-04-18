"""
Plot train and val loss for both old-corpus and new-corpus BPE training runs
on a single plot. Uses per-iteration train loss (from "iter N: loss X" lines)
and per-eval val loss (from "Step N | ... val loss X" lines).

Usage:
    python py/plot_train_val_loss.py
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt

OLD_LOG = "terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt"
NEW_LOG = "terminal_logs/terminal_log_for_bpe_16L16H_2026_04_15_1256.txt"

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
        print(f"Warning: {path} not found, skipping")
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


old_iters, old_iter_loss, old_val_steps, old_val_epochs, old_val_train, old_val_loss = parse_log(OLD_LOG)
new_iters, new_iter_loss, new_val_steps, new_val_epochs, new_val_train, new_val_loss = parse_log(NEW_LOG)

# Clip old data to match the range of the new data so both are comparable
if len(new_iters):
    max_new_iter = new_iters[-1]
    old_iter_mask = old_iters <= max_new_iter
    old_iters = old_iters[old_iter_mask]
    old_iter_loss = old_iter_loss[old_iter_mask]
    old_val_mask = old_val_steps <= max_new_iter
    old_val_steps = old_val_steps[old_val_mask]
    old_val_train = old_val_train[old_val_mask]
    old_val_loss = old_val_loss[old_val_mask]

print(f"Old corpus (8 GB): {len(old_iters)} train points, {len(old_val_steps)} eval points")
if len(old_iters):
    print(f"  Train loss: {old_iter_loss[0]:.4f} -> {old_iter_loss[-1]:.4f} (iters {old_iters[0]}–{old_iters[-1]})")
if len(old_val_steps):
    print(f"  Val loss:   {old_val_loss[0]:.4f} -> {old_val_loss[-1]:.4f}")

print(f"New corpus (2.5 GB): {len(new_iters)} train points, {len(new_val_steps)} eval points")
if len(new_iters):
    print(f"  Train loss: {new_iter_loss[0]:.4f} -> {new_iter_loss[-1]:.4f} (iters {new_iters[0]}–{new_iters[-1]})")
if len(new_val_steps):
    print(f"  Val loss:   {new_val_loss[0]:.4f} -> {new_val_loss[-1]:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

if len(old_iters):
    ax.plot(old_iters, old_iter_loss, linewidth=0.5, alpha=0.6,
            color='steelblue', label='Old corpus — train')
if len(old_val_steps):
    ax.plot(old_val_steps, old_val_loss, linewidth=1.5, alpha=0.9,
            color='navy', label='Old corpus — val', linestyle='--')

if len(new_iters):
    ax.plot(new_iters, new_iter_loss, linewidth=0.5, alpha=0.6,
            color='coral', label='New corpus — train')
if len(new_val_steps):
    ax.plot(new_val_steps, new_val_loss, linewidth=1.5, alpha=0.9,
            color='darkred', label='New corpus — val', linestyle='--')

# Per-character equivalent on right y-axis
ax2 = ax.twinx()
ymin, ymax = ax.get_ylim()
ax2.set_ylim(ymin / 4.09, ymax / 4.09)
ax2.set_ylabel('Per-character equivalent loss (÷ 4.09)', fontsize=11, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Loss (per BPE token)', fontsize=13)
ax.set_title('BPE 16L16H — Train & Val Loss: Old vs New Corpus', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Show latest new-corpus epoch as text annotation
if len(new_val_epochs):
    epoch_text = f'New corpus epoch: {new_val_epochs[-1]:.2f}'
    ax.text(0.02, 0.02, epoch_text, transform=ax.transAxes,
            fontsize=12, color='darkred', verticalalignment='bottom')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
out_path = 'plots/train_val_loss_old_vs_new.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
