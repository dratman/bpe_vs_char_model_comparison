"""
Plot smoothed training loss for one training run.

Usage:
    # default: pick the most recent terminal_logs/terminal_log_for_bpe_*.txt
    python py/plot_current_run.py

    # specify a log explicitly
    python py/plot_current_run.py --log terminal_logs/terminal_log_for_char_uppercase_16L_1280_2026_05_09_0038.txt

    # specify an output path explicitly
    python py/plot_current_run.py --log <log> --out plots/some_name.png

If --out is not given, the output filename is derived from the log
basename: plots/<log-stem>_smoothed.png. The y-axis label adapts to
the tokenization (BPE token vs character) inferred from the log name.
"""

import argparse
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log', default=None,
                   help='Path to a training log. Defaults to the most '
                        'recent terminal_logs/terminal_log_for_bpe_*.txt.')
    p.add_argument('--out', default=None,
                   help='Output PNG path. Defaults to plots/<log-stem>_smoothed.png.')
    return p.parse_args()


def find_default_log():
    logs = glob.glob('terminal_logs/terminal_log_for_bpe_*.txt')
    if not logs:
        raise SystemExit('No training logs found and no --log given')
    return max(logs, key=os.path.getmtime)


def parse_log(path):
    """Return (iters, losses, val_steps, val_epochs, val_train, val_loss)."""
    iter_re = re.compile(r'^iter\s+(\d+):\s+loss\s+([\d.]+)')
    step_re = re.compile(
        r'Step\s+(\d+)\s*\|\s*Epoch\s+([\d.]+)\s*\|.*train loss\s+([\d.]+)\s*\|.*val loss\s+([\d.]+)'
    )
    iters, losses = [], []
    val_steps, val_epochs, val_train, val_loss = [], [], [], []
    with open(path) as f:
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
    return iters, losses, val_steps, val_epochs, val_train, val_loss


def infer_token_label(log_path):
    """BPE log names contain '_for_bpe_'; char log names contain '_for_char_'."""
    name = os.path.basename(log_path).lower()
    if '_for_bpe_' in name or 'bpe' in name:
        return 'BPE token'
    return 'character'


def main():
    args = parse_args()
    log_path = args.log or find_default_log()

    iters, losses, val_steps, val_epochs, val_train, val_loss = parse_log(log_path)
    if not iters:
        raise SystemExit(f'No iter lines found in {log_path}')

    iters = np.array(iters)
    losses = np.array(losses)

    # Smooth with rolling average. Window scales with data so smoothing
    # is visible even very early in training.
    window = max(3, min(50, len(losses) // 5))
    smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
    smoothed_iters = iters[window - 1:]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(iters, losses, linewidth=0.6, alpha=0.6, color='limegreen', label='Train loss (raw)')
    ax.plot(smoothed_iters, smoothed, linewidth=2.0, color='darkgreen', label='Train loss (smoothed)')
    if val_steps:
        ax.plot(val_steps, val_loss, linewidth=2.0, color='darkred', linestyle='--', label='Val loss')

    token_label = infer_token_label(log_path)
    log_stem = os.path.splitext(os.path.basename(log_path))[0]

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel(f'Loss (per {token_label})', fontsize=13)
    ax.set_title(log_stem, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if val_epochs:
        ax.text(0.02, 0.02, f'Epoch: {val_epochs[-1]:.2f}', transform=ax.transAxes,
                fontsize=12, color='darkred', verticalalignment='bottom')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = args.out or f'plots/{log_stem}_smoothed.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')
    print(f'Log: {os.path.basename(log_path)}')
    print(f'Iters: {iters[0]}-{iters[-1]}, smoothed loss: {smoothed[-1]:.4f}')
    if val_loss:
        print(f'Val loss: {val_loss[-1]:.4f} at epoch {val_epochs[-1]:.2f}')


if __name__ == '__main__':
    main()
