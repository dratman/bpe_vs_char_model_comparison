"""Plot training loss vs wall-clock time from a terminal log file."""

import re
import matplotlib.pyplot as plt
from datetime import datetime

LOG_PATH = "terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt"

# Parse iter lines: "iter N: loss X.XXXX, time Yms"
iter_loss = []
# Parse timestamp+iter lines to map iterations to wall-clock time
# Two formats:
#   Speed lines: "[2026-04-05 19:01:15] [...] Speed at iter 100 ..."
#   Step lines:  "[2026-04-05 18:52:48] [...] Step     0 | ..."
timestamp_iter = []

with open(LOG_PATH) as f:
    for line in f:
        # Extract iter/loss
        m = re.match(r'^iter\s+(\d+):\s+loss\s+([\d.]+)', line)
        if m:
            iter_loss.append((int(m.group(1)), float(m.group(2))))

        # Extract timestamp at speed reports
        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Speed at iter (\d+)', line)
        if m:
            dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            timestamp_iter.append((int(m.group(2)), dt))

        # Extract timestamp at step/eval reports
        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Step\s+(\d+)', line)
        if m:
            dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            timestamp_iter.append((int(m.group(2)), dt))

# Sort timestamp_iter by iteration
timestamp_iter.sort(key=lambda x: x[0])

# Build a mapping from iteration -> hours elapsed since training start
# Training started at iter 0
t0 = timestamp_iter[0][1]
iter_to_hours = {}
for it, dt in timestamp_iter:
    iter_to_hours[it] = (dt - t0).total_seconds() / 3600.0

# Interpolate wall-clock time for each iter in iter_loss
# Use linear interpolation between known timestamp points
known_iters = [x[0] for x in timestamp_iter]
known_hours = [iter_to_hours[x[0]] for x in timestamp_iter]

def interp_hours(it):
    if it <= known_iters[0]:
        return known_hours[0]
    if it >= known_iters[-1]:
        return known_hours[-1]
    for i in range(len(known_iters) - 1):
        if known_iters[i] <= it <= known_iters[i + 1]:
            frac = (it - known_iters[i]) / (known_iters[i + 1] - known_iters[i])
            return known_hours[i] + frac * (known_hours[i + 1] - known_hours[i])
    return known_hours[-1]

iters = [x[0] for x in iter_loss]
losses = [x[1] for x in iter_loss]
hours = [interp_hours(it) for it in iters]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hours, losses, linewidth=0.6, alpha=0.8, color='steelblue')
ax.set_xlabel('Hours since training start', fontsize=13)
ax.set_ylabel('Training loss', fontsize=13)
ax.set_title('BPE 16L16H — Training Loss vs Time', fontsize=15)
ax.grid(True, alpha=0.3)

# Add iteration markers on a secondary x-axis
ax2 = ax.twiny()
# Place tick marks at round iteration numbers
tick_iters = list(range(0, max(iters) + 1, 5000))
tick_hours = [interp_hours(it) for it in tick_iters]
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(tick_hours)
ax2.set_xticklabels([f'{it//1000}k' for it in tick_iters], fontsize=9)
ax2.set_xlabel('Iteration', fontsize=11)

plt.tight_layout()
out_path = 'plots/loss_vs_time_bpe_16L16H.png'
import os
os.makedirs('plots', exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
print(f"Data: {len(iter_loss)} loss samples, {len(timestamp_iter)} timestamp anchors")
print(f"Training range: iter 0–{iters[-1]}, {hours[-1]:.1f} hours")
print(f"Loss range: {max(losses):.4f} → {min(losses):.4f}")
