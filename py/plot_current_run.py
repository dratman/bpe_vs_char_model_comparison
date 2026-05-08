"""
Plot smoothed training loss for the current run only.

Usage:
    python py/plot_current_run.py
"""

import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "terminal_logs"

# Find the most recent log file
logs = glob.glob(os.path.join(LOG_DIR, "terminal_log_for_bpe_*.txt"))
if not logs:
    print("No training logs found")
    exit(1)
LOG = max(logs, key=os.path.getmtime)

iter_re = re.compile(r'^iter\s+(\d+):\s+loss\s+([\d.]+)')
step_re = re.compile(
    r'Step\s+(\d+)\s*\|\s*Epoch\s+([\d.]+)\s*\|.*train loss\s+([\d.]+)\s*\|.*val loss\s+([\d.]+)'
)

iters, losses = [], []
val_steps, val_epochs, val_train, val_loss = [], [], [], []

with open(LOG) as f:
    for line in f:
        m = iter_re.match(line)
        if m:
            iters.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            continue
        m = step_re.search(line)
        if m:
            val_steps.append(int(m.group(1)))
            val_epochs.append(float(m.group(2)))
            val_train.append(float(m.group(3)))
            val_loss.append(float(m.group(4)))

if not iters:
    print("No iter lines found in log")
    exit(1)

iters = np.array(iters)
losses = np.array(losses)

# Smooth with rolling average (window=50 points)
window = min(50, len(losses))
smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
smoothed_iters = iters[window - 1:]

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(iters, losses, linewidth=0.3, alpha=0.3, color='limegreen', label='Train loss (raw)')
ax.plot(smoothed_iters, smoothed, linewidth=1.5, color='darkgreen', label='Train loss (smoothed)')
if val_steps:
    ax.plot(val_steps, val_loss, linewidth=1.5, color='darkred', linestyle='--', label='Val loss')

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Loss (per BPE token)', fontsize=13)
ax.set_title('Current Training Run', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

if val_epochs:
    ax.text(0.02, 0.02, f'Epoch: {val_epochs[-1]:.2f}', transform=ax.transAxes,
            fontsize=12, color='darkred', verticalalignment='bottom')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
out_path = 'plots/current_run_smoothed.png'
plt.savefig(out_path, dpi=150)
print(f'Saved to {out_path}')
print(f'Log: {os.path.basename(LOG)}')
print(f'Iters: {iters[0]}-{iters[-1]}, smoothed loss: {smoothed[-1]:.4f}')
if val_loss:
    print(f'Val loss: {val_loss[-1]:.4f} at epoch {val_epochs[-1]:.2f}')
