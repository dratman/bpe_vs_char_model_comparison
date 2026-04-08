"""Plot training loss vs time with extrapolation 24 hours into the future.

Fits a power-law model: loss(t) = a * (t + c)^b + d
This captures the rapid initial drop and the slow asymptotic decay.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit

LOG_PATH = "terminal_logs/terminal_log_for_bpe_16L16H_2026_04_05_1754.txt"

# ---------------------------------------------------------------------------
# 1. Parse the log (same as plot_loss.py)
# ---------------------------------------------------------------------------
iter_loss = []
timestamp_iter = []

with open(LOG_PATH) as f:
    for line in f:
        m = re.match(r'^iter\s+(\d+):\s+loss\s+([\d.]+)', line)
        if m:
            iter_loss.append((int(m.group(1)), float(m.group(2))))

        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Speed at iter (\d+)', line)
        if m:
            dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            timestamp_iter.append((int(m.group(2)), dt))

        m = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Step\s+(\d+)', line)
        if m:
            dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            timestamp_iter.append((int(m.group(2)), dt))

timestamp_iter.sort(key=lambda x: x[0])
t0 = timestamp_iter[0][1]
iter_to_hours = {it: (dt - t0).total_seconds() / 3600.0 for it, dt in timestamp_iter}

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
hours = np.array([interp_hours(it) for it in iters])
losses_arr = np.array(losses)

# ---------------------------------------------------------------------------
# 2. Fit a power-law model: loss(t) = a * (t + c)^b + d
#    - a, b control the shape of the decay
#    - c is a time offset (avoids singularity at t=0)
#    - d is the asymptotic floor the loss approaches
# ---------------------------------------------------------------------------
def power_law(t, a, b, c, d):
    return a * (t + c) ** b + d

# Use data from hour 1 onward (skip the chaotic first few minutes)
mask = hours >= 1.0
h_fit = hours[mask]
l_fit = losses_arr[mask]

# Smooth the data for fitting (reduce noise)
# Bin into 0.5-hour windows
bin_width = 0.5
bins = np.arange(h_fit.min(), h_fit.max() + bin_width, bin_width)
h_binned = []
l_binned = []
for i in range(len(bins) - 1):
    in_bin = (h_fit >= bins[i]) & (h_fit < bins[i + 1])
    if in_bin.sum() > 0:
        h_binned.append((bins[i] + bins[i + 1]) / 2)
        l_binned.append(l_fit[in_bin].mean())

h_binned = np.array(h_binned)
l_binned = np.array(l_binned)

# Initial guesses: loss starts ~5.3 at 1h, ends ~3.4 at 33h
# a*(1+c)^b + d ~ 5.3,  a*(33+c)^b + d ~ 3.4
# Guess: a=5, b=-0.3, c=0.5, d=2.5
p0 = [5.0, -0.3, 0.5, 2.5]
bounds = ([0.1, -2.0, 0.01, 0.0], [50.0, 0.0, 10.0, 4.0])

popt, pcov = curve_fit(power_law, h_binned, l_binned, p0=p0, bounds=bounds, maxfev=10000)
a, b, c, d = popt
perr = np.sqrt(np.diag(pcov))

print("Fitted parameters:  loss(t) = a * (t + c)^b + d")
print(f"  a = {a:.4f} ± {perr[0]:.4f}")
print(f"  b = {b:.4f} ± {perr[1]:.4f}")
print(f"  c = {c:.4f} ± {perr[2]:.4f}")
print(f"  d = {d:.4f} ± {perr[3]:.4f}")

# Compute R² on the binned data
residuals = l_binned - power_law(h_binned, *popt)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((l_binned - l_binned.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"  R² = {r_squared:.6f}")

# ---------------------------------------------------------------------------
# 3. Extrapolate and plot
# ---------------------------------------------------------------------------
current_hours = hours[-1]
future_hours = current_hours + 24
print(f"\nCurrent training time: {current_hours:.1f} hours")
print(f"Extrapolating to:     {future_hours:.1f} hours (+24h)")
print(f"Current loss (last):  {losses[-1]:.4f}")
print(f"Predicted loss at +24h: {power_law(future_hours, *popt):.4f}")

# Also show predictions at intermediate points
for dt in [6, 12, 18, 24]:
    t = current_hours + dt
    print(f"  +{dt:2d}h (t={t:.1f}h): predicted loss = {power_law(t, *popt):.4f}")

# Extrapolation curve
t_extrap = np.linspace(0.5, future_hours, 500)
l_extrap = power_law(t_extrap, *popt)

fig, ax = plt.subplots(figsize=(14, 6))

# Actual data
ax.plot(hours, losses, linewidth=0.5, alpha=0.6, color='steelblue', label='Actual training loss')

# Fitted curve over observed range
t_fit_plot = np.linspace(1.0, current_hours, 300)
ax.plot(t_fit_plot, power_law(t_fit_plot, *popt), linewidth=2, color='darkorange',
        label=f'Power-law fit (R²={r_squared:.4f})')

# Extrapolation
t_future = np.linspace(current_hours, future_hours, 200)
l_future = power_law(t_future, *popt)
ax.plot(t_future, l_future, linewidth=2, linestyle='--', color='red',
        label='Extrapolation (+24h)')

# Mark the predicted endpoint
pred_loss = power_law(future_hours, *popt)
ax.plot(future_hours, pred_loss, 'o', color='red', markersize=8, zorder=5)
ax.annotate(f'Predicted: {pred_loss:.3f}',
            xy=(future_hours, pred_loss),
            xytext=(future_hours - 8, pred_loss + 0.3),
            fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Mark the current endpoint
ax.axvline(x=current_hours, color='gray', linestyle=':', alpha=0.7)
ax.text(current_hours + 0.3, max(losses) * 0.55, f'Now\n({current_hours:.1f}h)',
        fontsize=10, color='gray')

# Confidence band (rough estimate using parameter uncertainty)
# Propagate parameter uncertainty for a simple band
n_samples = 200
samples = np.random.multivariate_normal(popt, pcov, size=n_samples)
future_preds = np.array([power_law(t_future, *s) for s in samples
                         if s[1] < 0 and s[2] > 0 and s[3] > 0])
if len(future_preds) > 10:
    lo = np.percentile(future_preds, 10, axis=0)
    hi = np.percentile(future_preds, 90, axis=0)
    ax.fill_between(t_future, lo, hi, color='red', alpha=0.1, label='80% confidence band')

ax.set_xlabel('Hours since training start', fontsize=13)
ax.set_ylabel('Training loss', fontsize=13)
ax.set_title('BPE 16L16H — Training Loss with 24h Extrapolation', fontsize=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, future_hours + 1)

# Secondary x-axis for iterations (extrapolated assuming ~0.2 iter/sec steady state)
# From the log: ~700 iters/hour in the later part of training
iters_per_hour = (iters[-1] - iters[len(iters)//2]) / (hours[-1] - hours[len(hours)//2])
ax2 = ax.twiny()
max_iter_extrap = int(iters[-1] + iters_per_hour * 24)
tick_step = 5000
tick_iters_list = list(range(0, max_iter_extrap + 1, tick_step))
tick_hours_list = []
for ti in tick_iters_list:
    if ti <= iters[-1]:
        tick_hours_list.append(interp_hours(ti))
    else:
        tick_hours_list.append(current_hours + (ti - iters[-1]) / iters_per_hour)
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(tick_hours_list)
ax2.set_xticklabels([f'{ti//1000}k' for ti in tick_iters_list], fontsize=9)
ax2.set_xlabel('Iteration (extrapolated beyond dashed line)', fontsize=11)

plt.tight_layout()
import os
os.makedirs('plots', exist_ok=True)
out_path = 'plots/loss_vs_time_bpe_16L16H_extrapolated.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
